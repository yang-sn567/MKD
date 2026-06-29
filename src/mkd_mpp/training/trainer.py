from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mkd_mpp.data import MolecularPropertyDataset, collate_molecules
from mkd_mpp.models.mkd import MKDStudent, build_teachers
from mkd_mpp.training.losses import MKDLoss, supervised_loss


def load_state_dict_safely(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def infer_active_modalities(config: Any) -> list[str]:
    requested = list(config.model.modalities)
    available = {"smiles"}
    if config.data.graph_path:
        available.add("graph")
    if config.data.geometry_path:
        available.add("geometry")
    if config.data.llm_text_path:
        available.add("llm_text")
    return [modality for modality in requested if modality in available]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("smiles_tokens", "labels"):
        moved[key] = moved[key].to(device)
    return moved


class BaseTrainer:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.output_dir = Path(config.training.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dataset(self, split: str) -> MolecularPropertyDataset:
        return MolecularPropertyDataset(
            self.config.data.data_path,
            one_d_path=self.config.data.one_d_path,
            graph_path=self.config.data.graph_path,
            geometry_path=self.config.data.geometry_path,
            llm_text_path=self.config.data.llm_text_path,
            smiles_column=self.config.data.smiles_column,
            label_column=self.config.data.label_column,
            split_column=self.config.data.split_column,
            text_column=self.config.data.text_column,
            split=split,
            split_method=self.config.data.split_method,
            split_root=self.config.data.split_root,
            skip_missing_llm_text=self.config.data.skip_missing_llm_text,
        )

    def loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.dataset(split),
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            collate_fn=collate_molecules,
        )

    def append_log(self, filename: str, record: dict[str, Any]) -> None:
        with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def evaluate_loss(self, model: torch.nn.Module, split: str) -> float:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in self.loader(split, shuffle=False):
                batch = move_batch(batch, self.device)
                output = model(batch)
                loss = supervised_loss(output["prediction"], batch["labels"], self.config.data.task_type)
                batch_size = batch["labels"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        if total_samples == 0:
            raise ValueError(f"No samples available for split: {split}")
        return total_loss / total_samples


class TeacherTrainer(BaseTrainer):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.teachers = build_teachers(config)

    def train_all(self) -> dict[str, dict[str, float]]:
        available = self.dataset("train").available_modalities
        results = {}
        for name, teacher in self.teachers.items():
            if name not in available:
                continue
            results[name] = self.train_one(name, teacher)
        return results

    def train_one(self, name: str, teacher: torch.nn.Module) -> dict[str, float]:
        teacher.to(self.device)
        optimizer = torch.optim.AdamW(
            teacher.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        best_loss = float("inf")
        best_record = {}
        patience = 0

        for epoch in tqdm(range(self.config.training.epochs), desc=f"teacher:{name}"):
            teacher.train()
            train_loss = 0.0
            train_samples = 0
            for batch in self.loader("train", shuffle=True):
                batch = move_batch(batch, self.device)
                output = teacher(batch)
                loss = supervised_loss(output["prediction"], batch["labels"], self.config.data.task_type)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = batch["labels"].size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size

            train_loss = train_loss / max(train_samples, 1)
            valid_loss = self.evaluate_loss(teacher, "valid")
            record = {
                "stage": "teacher",
                "modality": name,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
            self.append_log("train_log.jsonl", record)
            torch.save(teacher.state_dict(), self.checkpoint_dir / f"{name}_teacher_last.pt")

            monitor = valid_loss
            if monitor < best_loss:
                best_loss = monitor
                best_record = record | {"checkpoint": str(self.checkpoint_dir / f"{name}_teacher.pt")}
                patience = 0
                torch.save(teacher.state_dict(), self.checkpoint_dir / f"{name}_teacher.pt")
            else:
                patience += 1
                if patience >= self.config.training.early_stopping_patience:
                    break
        return best_record


class MKDTrainer(BaseTrainer):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.active_modalities = infer_active_modalities(config)
        self.student = MKDStudent(
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            modalities=self.active_modalities,
            transformer_config=config.model.transformer,
            gin_config=config.model.gin,
            schnet_config=config.model.schnet,
            text_config=config.model.text,
            dropout=config.model.dropout,
        ).to(self.device)
        self.teachers = build_teachers(config)
        self.loss_fn = MKDLoss(
            task_type=config.data.task_type,
            lambda_kd=config.training.lambda_kd,
            lambda_cross=config.training.lambda_cross,
            lambda_cf=config.training.lambda_cf,
            temperature=config.training.temperature,
        )

    def load_teacher_checkpoints(self) -> None:
        for name, teacher in self.teachers.items():
            if name not in self.active_modalities:
                continue
            path = self.checkpoint_dir / f"{name}_teacher.pt"
            if not path.exists():
                if self.config.training.require_teacher_checkpoints:
                    raise FileNotFoundError(f"Required teacher checkpoint is missing: {path}")
                continue
            teacher.load_state_dict(load_state_dict_safely(path))
            teacher.to(self.device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

    def train(self) -> dict[str, float]:
        self.load_teacher_checkpoints()
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        best_loss = float("inf")
        best_record = {}
        patience = 0

        for epoch in tqdm(range(self.config.training.epochs), desc="mkd-student"):
            self.student.train()
            train_loss = 0.0
            train_task_loss = 0.0
            train_kd_loss = 0.0
            train_cross_loss = 0.0
            train_samples = 0
            for batch in self.loader("train", shuffle=True):
                batch = move_batch(batch, self.device)
                with torch.no_grad():
                    teacher_outputs = {
                        name: teacher(batch)
                        for name, teacher in self.teachers.items()
                        if name in self.active_modalities
                    }
                student_output = self.student(batch)
                losses = self.loss_fn(student_output, teacher_outputs, batch["labels"])
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
                batch_size = batch["labels"].size(0)
                train_loss += losses["total"].item() * batch_size
                train_task_loss += losses["task"].item() * batch_size
                train_kd_loss += losses["kd"].item() * batch_size
                train_cross_loss += losses["cross"].item() * batch_size
                train_samples += batch_size

            train_samples = max(train_samples, 1)
            valid_loss = self.evaluate_loss(self.student, "valid")
            record = {
                "stage": "student",
                "epoch": epoch + 1,
                "train_loss": train_loss / train_samples,
                "task_loss": train_task_loss / train_samples,
                "kd_loss": train_kd_loss / train_samples,
                "cross_loss": train_cross_loss / train_samples,
                "valid_loss": valid_loss,
                "modalities": self.active_modalities,
            }
            self.append_log("train_log.jsonl", record)
            torch.save(self.student.state_dict(), self.checkpoint_dir / "mkd_student_last.pt")

            monitor = valid_loss
            if monitor < best_loss:
                best_loss = monitor
                best_record = record | {"checkpoint": str(self.checkpoint_dir / "mkd_student.pt")}
                patience = 0
                torch.save(self.student.state_dict(), self.checkpoint_dir / "mkd_student.pt")
            else:
                patience += 1
                if patience >= self.config.training.early_stopping_patience:
                    break

        with (self.output_dir / "student_valid_loss.json").open("w", encoding="utf-8") as handle:
            json.dump(best_record, handle, indent=2)
        return best_record
