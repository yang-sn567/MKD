from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .encoders import GINEncoder, SchNetEncoder, SmilesTransformerEncoder, build_text_encoder


class UnimodalTeacher(nn.Module):
    def __init__(self, modality: str, encoder: nn.Module, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.modality = modality
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        representation = encode_modality(self.modality, self.encoder, batch)
        prediction = self.head(representation)
        return {"representation": representation, "prediction": prediction}


class CrossModalityInformationGain(nn.Module):
    def __init__(self, hidden_dim: int, modalities: list[str]) -> None:
        super().__init__()
        self.modalities = modalities
        self.scorers = nn.ModuleDict(
            {
                f"{source}_to_{target}": nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for source in modalities
                for target in modalities
                if source != target
            }
        )

    def forward(self, reps: dict[str, torch.Tensor]) -> torch.Tensor:
        scores = []
        for source in self.modalities:
            source_score = 0.0
            for target in self.modalities:
                if source == target:
                    continue
                pair = torch.cat([reps[source], reps[target]], dim=-1)
                source_score = source_score + self.scorers[f"{source}_to_{target}"](pair)
            scores.append(source_score)
        return torch.softmax(torch.cat(scores, dim=-1), dim=-1)


class MKDStudent(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 1,
        modalities: list[str] | None = None,
        transformer_config: dict[str, Any] | None = None,
        gin_config: dict[str, Any] | None = None,
        schnet_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = modalities or ["smiles", "graph", "geometry", "llm_text"]
        self.encoders = nn.ModuleDict()
        if "smiles" in self.modalities:
            self.encoders["smiles"] = SmilesTransformerEncoder(
                hidden_dim=hidden_dim, dropout=dropout, **(transformer_config or {})
            )
        if "graph" in self.modalities:
            self.encoders["graph"] = GINEncoder(
                hidden_dim=hidden_dim, dropout=dropout, **(gin_config or {})
            )
        if "geometry" in self.modalities:
            self.encoders["geometry"] = SchNetEncoder(hidden_dim=hidden_dim, **(schnet_config or {}))
        if "llm_text" in self.modalities:
            self.encoders["llm_text"] = build_text_encoder(hidden_dim, dropout, text_config)

        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
                for name in self.modalities
            }
        )
        self.cmig = CrossModalityInformationGain(hidden_dim, self.modalities)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        reps = {
            name: self.projectors[name](encode_modality(name, self.encoders[name], batch))
            for name in self.modalities
        }
        weights = self.cmig(reps)
        stacked = torch.stack([reps[name] for name in self.modalities], dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.fusion_norm(fused)
        prediction = self.head(fused)
        return {
            "prediction": prediction,
            "fused": fused,
            "modal_representations": reps,
            "modality_weights": weights,
        }


def encode_modality(modality: str, encoder: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    if modality == "smiles":
        return encoder(batch["smiles_tokens"])
    if modality == "graph":
        return encoder(batch.get("graphs"), batch_size=batch["labels"].size(0))
    if modality == "geometry":
        return encoder(batch.get("geometries"), batch_size=batch["labels"].size(0))
    if modality == "llm_text":
        return encoder(batch["llm_text"])
    raise KeyError(f"Unsupported modality: {modality}")


def build_teachers(config: Any) -> dict[str, UnimodalTeacher]:
    hidden_dim = config.model.hidden_dim
    output_dim = config.model.output_dim
    return {
        "smiles": UnimodalTeacher(
            "smiles",
            SmilesTransformerEncoder(hidden_dim=hidden_dim, **config.model.transformer),
            hidden_dim,
            output_dim,
        ),
        "graph": UnimodalTeacher(
            "graph",
            GINEncoder(hidden_dim=hidden_dim, **config.model.gin),
            hidden_dim,
            output_dim,
        ),
        "geometry": UnimodalTeacher(
            "geometry",
            SchNetEncoder(hidden_dim=hidden_dim, **config.model.schnet),
            hidden_dim,
            output_dim,
        ),
        "llm_text": UnimodalTeacher(
            "llm_text",
            build_text_encoder(hidden_dim, config.model.dropout, config.model.text),
            hidden_dim,
            output_dim,
        ),
    }
