from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SmilesTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 256,
        max_length: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        ) -> None:
        super().__init__()
        self.uses_raw_text = False
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.max_length = max_length

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens[:, : self.max_length]
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        padding_mask = tokens.eq(0)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class HuggingFaceTextEncoder(nn.Module):
    """HuggingFace text encoder for ChemBERTa or compatible backbones."""

    def __init__(
        self,
        pretrained_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
        hidden_dim: int = 256,
        max_length: int = 256,
        freeze_encoder: bool = True,
        pooling: str = "cls",
        use_safetensors: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.uses_raw_text = True
        try:
            import transformers.utils.import_utils as transformers_import_utils

            transformers_import_utils._torchvision_available = False
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ChemBERTa text encoding requires transformers and tokenizers. "
                "Install dependencies from requirements.txt or environment.yml."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name,
            use_safetensors=use_safetensors,
        )
        self.max_length = max_length
        self.pooling = pooling
        self.proj = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, texts: list[str | None]) -> torch.Tensor:
        normalized = [text or "" for text in texts]
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            normalized,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        output = self.encoder(**tokens)

        if self.pooling == "mean":
            mask = tokens["attention_mask"].float().unsqueeze(-1)
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = output.last_hidden_state[:, 0]
        return self.norm(self.proj(pooled))


def build_text_encoder(
    hidden_dim: int,
    dropout: float = 0.1,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    config = dict(config or {})
    encoder_type = config.pop("encoder_type", "chemberta")
    if encoder_type == "chemberta":
        return HuggingFaceTextEncoder(hidden_dim=hidden_dim, **config)
    raise ValueError(f"Unsupported text encoder_type: {encoder_type}")


class GINEncoder(nn.Module):
    """GIN encoder wrapper.

    Real experiments should pass PyG ``Batch`` objects. If graph data is not yet
    available, the encoder returns a zero representation so the project can keep
    a stable interface while dataset preprocessing is being added.
    """

    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.LazyLinear(hidden_dim)
        self.edge_proj = nn.LazyLinear(hidden_dim)
        self.convs = nn.ModuleList()
        try:
            from torch_geometric.nn import GINEConv
        except ImportError:
            GINEConv = None
        if GINEConv is not None:
            for _ in range(num_layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph_batch: Any, batch_size: int | None = None) -> torch.Tensor:
        if isinstance(graph_batch, list) and any(graph is not None for graph in graph_batch):
            return self._forward_graph_dicts(graph_batch)

        if graph_batch is None or isinstance(graph_batch, list):
            if batch_size is None:
                batch_size = len(graph_batch) if isinstance(graph_batch, list) else 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        x = self.input_proj(graph_batch.x.float())
        for layer in self.layers:
            x = x + self.dropout(layer(x))

        if hasattr(graph_batch, "batch"):
            try:
                from torch_geometric.nn import global_mean_pool
            except ImportError as exc:
                raise ImportError("torch-geometric is required for graph pooling.") from exc
            x = global_mean_pool(x, graph_batch.batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.norm(x)

    def _forward_graph_dicts(self, graph_batch: list[dict[str, Any] | None]) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.convs:
            try:
                from torch_geometric.data import Batch, Data
                from torch_geometric.nn import global_mean_pool
            except ImportError as exc:
                raise ImportError("torch-geometric is required for PCQM4Mv2 graph batches.") from exc

            data_list = []
            empty_rows = []
            for row, graph in enumerate(graph_batch):
                if graph is None:
                    empty_rows.append(row)
                    continue
                data_list.append(
                    Data(
                        x=torch.as_tensor(graph["node_feat"], dtype=torch.float32),
                        edge_index=torch.as_tensor(graph["edge_index"], dtype=torch.long),
                        edge_attr=torch.as_tensor(graph["edge_feat"], dtype=torch.float32),
                        original_row=row,
                    )
                )
            if not data_list:
                return torch.zeros(len(graph_batch), self.hidden_dim, device=device)
            batch = Batch.from_data_list(data_list).to(device)
            x = self.input_proj(batch.x)
            edge_attr = self.edge_proj(batch.edge_attr.float())
            for conv in self.convs:
                x = x + self.dropout(conv(x, batch.edge_index, edge_attr))
            pooled = global_mean_pool(x, batch.batch)
            output = torch.zeros(len(graph_batch), self.hidden_dim, device=device)
            for compact_row, data in enumerate(data_list):
                output[data.original_row] = pooled[compact_row]
            return self.norm(output)

        outputs = []
        for graph in graph_batch:
            if graph is None:
                outputs.append(torch.zeros(self.hidden_dim, device=device))
                continue
            node_feat = torch.as_tensor(graph["node_feat"], dtype=torch.float32, device=device)
            x = self.input_proj(node_feat)
            for layer in self.layers:
                x = x + self.dropout(layer(x))
            outputs.append(x.mean(dim=0))
        return self.norm(torch.stack(outputs, dim=0))


class SchNetEncoder(nn.Module):
    """SchNet-style geometry encoder.

    Uses PyG SchNet when geometric tensors are available. Without geometry data,
    it returns zeros to preserve the multimodal API until preprocessing is added.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_atomic_number: int = 100,
        num_interactions: int = 6,
        cutoff: float = 10.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_atomic_number = max_atomic_number
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self._schnet: nn.Module | None = None

    def _build_schnet(self) -> nn.Module:
        try:
            from torch_geometric.nn import SchNet
        except ImportError as exc:
            raise ImportError("torch-geometric is required for SchNet geometry encoding.") from exc
        return SchNet(
            hidden_channels=self.hidden_dim,
            num_filters=self.hidden_dim,
            num_interactions=self.num_interactions,
            cutoff=self.cutoff,
            max_num_neighbors=32,
        )

    def forward(self, geometry_batch: Any, batch_size: int | None = None) -> torch.Tensor:
        if isinstance(geometry_batch, list) and any(mol is not None for mol in geometry_batch):
            geometry_batch = self._rdkit_mols_to_batch(geometry_batch)

        if geometry_batch is None or isinstance(geometry_batch, list):
            if batch_size is None:
                batch_size = len(geometry_batch) if isinstance(geometry_batch, list) else 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        if self._schnet is None:
            self._schnet = self._build_schnet().to(next(self.parameters()).device)
        output = self._schnet(geometry_batch.z, geometry_batch.pos, batch=geometry_batch.batch)
        if hasattr(geometry_batch, "output_rows"):
            padded = torch.zeros(
                geometry_batch.original_size,
                output.size(-1),
                dtype=output.dtype,
                device=output.device,
            )
            for original_row, compact_row in enumerate(geometry_batch.output_rows):
                if compact_row is not None:
                    padded[original_row] = output[compact_row]
            output = padded
        return self.proj(output)

    def _rdkit_mols_to_batch(self, mols: list[Any | None]):
        device = next(self.parameters()).device
        z_values = []
        pos_values = []
        batch_values = []
        valid_graph_index = 0
        output_rows = []

        for row_index, mol in enumerate(mols):
            if mol is None or mol.GetNumConformers() == 0:
                output_rows.append(None)
                continue
            conformer = mol.GetConformer()
            for atom_index, atom in enumerate(mol.GetAtoms()):
                position = conformer.GetAtomPosition(atom_index)
                z_values.append(atom.GetAtomicNum())
                pos_values.append([position.x, position.y, position.z])
                batch_values.append(valid_graph_index)
            output_rows.append(valid_graph_index)
            valid_graph_index += 1

        if not z_values:
            return None

        class GeometryBatch:
            pass

        compact = GeometryBatch()
        compact.z = torch.tensor(z_values, dtype=torch.long, device=device)
        compact.pos = torch.tensor(pos_values, dtype=torch.float32, device=device)
        compact.batch = torch.tensor(batch_values, dtype=torch.long, device=device)
        compact.output_rows = output_rows
        compact.original_size = len(mols)
        return compact
