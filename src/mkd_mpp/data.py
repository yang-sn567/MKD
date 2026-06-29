from __future__ import annotations

import csv
import io
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mkd_mpp.pcqm4mv2 import iter_sdf_molecules, load_official_split, smiles2graph


class MolecularPropertyDataset(Dataset):
    """Receive available modality files and expose only provided modalities."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        one_d_path: str | Path | None = None,
        graph_path: str | Path | None = None,
        geometry_path: str | Path | None = None,
        llm_text_path: str | Path | None = None,
        smiles_column: str = "smiles",
        label_column: str = "label",
        split_column: str = "split",
        text_column: str = "llm_text",
        split: str | None = None,
        split_method: str = "official_pcqm4mv2",
        split_root: str | Path = "dataset",
        skip_missing_llm_text: bool = True,
    ) -> None:
        self.one_d_path = Path(one_d_path or data_path) if one_d_path or data_path else None
        if self.one_d_path is None:
            raise ValueError("A 1D CSV path must be provided through data_path or one_d_path.")

        self.csv_path = self._resolve_csv(self.one_d_path)
        self.rows = self._read_rows(self.csv_path, split_column, split, split_method, Path(split_root))
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.split_column = split_column
        self.text_column = text_column
        self.available_modalities = ["smiles"]

        if not self.rows:
            raise ValueError(f"No rows loaded from {self.csv_path}.")
        self._validate_required_columns(self.rows[0], [smiles_column, label_column])

        self.graph_path = graph_path
        if graph_path:
            self.available_modalities.append("graph")

        self.geometry_index = None
        if geometry_path:
            self.geometry_index = SDFMoleculeIndex(Path(geometry_path))
            self.available_modalities.append("geometry")

        self.llm_text_by_idx = None
        if llm_text_path:
            self.llm_text_by_idx = self._read_llm_text(Path(llm_text_path), text_column)
            self.rows = self._align_llm_text_rows(
                self.rows,
                self.llm_text_by_idx,
                skip_missing=skip_missing_llm_text,
            )
            if not self.rows:
                raise ValueError("No samples remain after skipping rows with missing llm_text.")
            self.available_modalities.append("llm_text")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        idx = row.get("idx", str(index))
        sample = {
            "idx": idx,
            "smiles": str(row[self.smiles_column]),
            "label": torch.as_tensor(float(row[self.label_column]), dtype=torch.float32),
            "split": str(row[self.split_column]),
            "graph": None,
            "geometry": None,
            "llm_text": None,
        }

        if "graph" in self.available_modalities:
            sample["graph"] = smiles2graph(sample["smiles"])
        if "geometry" in self.available_modalities and self.geometry_index is not None:
            sample["geometry"] = self.geometry_index.get(idx)
        if "llm_text" in self.available_modalities and self.llm_text_by_idx is not None:
            sample["llm_text"] = self.llm_text_by_idx.get(idx)
        return sample

    @staticmethod
    def _resolve_csv(data_path: Path) -> Path:
        if data_path.is_file():
            return data_path
        for name in ("data.csv", "dataset.csv", "molecules.csv"):
            candidate = data_path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No CSV dataset found at {data_path}. Expected data.csv, dataset.csv, or molecules.csv."
        )

    @staticmethod
    def _read_rows(
        csv_path: Path,
        split_column: str,
        split: str | None,
        split_method: str,
        split_root: Path,
    ) -> list[dict[str, str]]:
        official_split = None
        if split is not None and split_method == "official_pcqm4mv2":
            official_split = load_official_split(split_root).get(split)
            if official_split is None:
                raise ValueError(f"Unsupported official PCQM4Mv2 split: {split}")

        rows = []
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")
            for row in reader:
                if split_column not in row or not row[split_column]:
                    row[split_column] = "train"
                if split is not None:
                    if official_split is not None:
                        if row.get("idx") not in official_split:
                            continue
                    elif row[split_column] != split:
                        continue
                rows.append(row)
        return rows

    @staticmethod
    def _validate_required_columns(row: dict[str, str], columns: list[str]) -> None:
        missing = [column for column in columns if column not in row]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

    @staticmethod
    def _read_llm_text(llm_text_path: Path, text_column: str) -> dict[str, str]:
        csv_path = MolecularPropertyDataset._resolve_csv(llm_text_path)
        text_by_idx = {}
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")
            if "idx" not in reader.fieldnames:
                raise ValueError(f"LLM text CSV must contain an idx column: {csv_path}")
            if text_column not in reader.fieldnames:
                raise ValueError(f"LLM text CSV must contain {text_column}: {csv_path}")
            for row in reader:
                text_by_idx[row["idx"]] = row[text_column]
        return text_by_idx

    @staticmethod
    def _align_llm_text_rows(
        rows: list[dict[str, str]],
        text_by_idx: dict[str, str],
        skip_missing: bool,
    ) -> list[dict[str, str]]:
        row_indices = {row.get("idx", "") for row in rows}
        extra_text_indices = set(text_by_idx) - row_indices
        if extra_text_indices and not row_indices.issubset(set(text_by_idx)):
            raise ValueError("LLM text file is not aligned with the main CSV idx column.")

        missing = [row.get("idx", "") for row in rows if row.get("idx", "") not in text_by_idx]
        if missing and not skip_missing:
            raise ValueError(f"Missing llm_text for {len(missing)} selected molecules.")
        if missing:
            missing_set = set(missing)
            return [row for row in rows if row.get("idx", "") not in missing_set]
        return rows


class SDFMoleculeIndex:
    """Lazy SDF access with a lightweight idx-to-offset cache."""

    def __init__(self, sdf_path: Path) -> None:
        self.sdf_path = sdf_path
        self.cache_path = sdf_path.with_suffix(sdf_path.suffix + ".idx.pkl")
        self.offsets = self._load_or_build_offsets()

    def get(self, idx: str) -> Any | None:
        offset = self.offsets.get(str(idx))
        if offset is None:
            raise KeyError(f"SDF molecule index {idx} is missing; CSV and SDF are not fully matched.")
        with self.sdf_path.open("rb") as handle:
            handle.seek(offset)
            block = bytearray()
            while True:
                line = handle.readline()
                if not line:
                    break
                block.extend(line)
                if line.strip() == b"$$$$":
                    break
        return next(iter_sdf_molecules_from_block(bytes(block)))

    def _load_or_build_offsets(self) -> dict[str, int]:
        if self.cache_path.exists():
            with self.cache_path.open("rb") as handle:
                return pickle.load(handle)

        offsets = {}
        idx = 0
        with self.sdf_path.open("rb") as handle:
            offset = handle.tell()
            for line in handle:
                if line.strip() == b"$$$$":
                    offsets[str(idx)] = offset
                    idx += 1
                    offset = handle.tell()
        with self.cache_path.open("wb") as handle:
            pickle.dump(offsets, handle)
        return offsets


def iter_sdf_molecules_from_block(block: bytes):
    from rdkit import Chem

    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(block))
    for mol in supplier:
        yield mol


def smiles_to_ascii_tokens(smiles: list[str], max_length: int = 256) -> torch.Tensor:
    tokens = torch.zeros((len(smiles), max_length), dtype=torch.long)
    for row, item in enumerate(smiles):
        encoded = [min(ord(char), 127) for char in item[:max_length]]
        if encoded:
            tokens[row, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    return tokens


def collate_molecules(batch: list[dict[str, Any]]) -> dict[str, Any]:
    smiles = [sample["smiles"] for sample in batch]
    llm_text = [sample["llm_text"] for sample in batch]
    labels = torch.stack([sample["label"].reshape(-1) for sample in batch]).float()
    return {
        "idx": [sample["idx"] for sample in batch],
        "smiles": smiles,
        "smiles_tokens": smiles_to_ascii_tokens(smiles),
        "labels": labels,
        "graphs": [sample["graph"] for sample in batch],
        "geometries": [sample["geometry"] for sample in batch],
        "llm_text": llm_text,
    }
