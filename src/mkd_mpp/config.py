from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_path: str = "data/example"
    one_d_path: str | None = None
    graph_path: str | None = None
    geometry_path: str | None = None
    llm_text_path: str | None = None
    smiles_column: str = "smiles"
    label_column: str = "homolumogap"
    split_column: str = "split"
    text_column: str = "llm_text"
    task_type: str = "regression"
    num_tasks: int = 1
    split_method: str = "official_pcqm4mv2"
    split_root: str = "dataset"
    skip_missing_llm_text: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    lambda_kd: float = 1.0
    lambda_cross: float = 0.1
    lambda_cf: float = 0.0
    temperature: float = 1.0
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    early_stopping_patience: int = 20
    train_teachers: bool = True
    train_student: bool = True
    require_teacher_checkpoints: bool = True


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    output_dim: int = 1
    dropout: float = 0.1
    modalities: list[str] = field(default_factory=lambda: ["smiles", "graph", "geometry"])
    transformer: dict[str, Any] = field(default_factory=dict)
    gin: dict[str, Any] = field(default_factory=dict)
    schnet: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    values = values or {}
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ProjectConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
    )
