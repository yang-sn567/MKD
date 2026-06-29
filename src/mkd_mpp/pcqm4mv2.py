from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from rdkit import Chem

try:
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
except ImportError as exc:  # pragma: no cover - dependency guard
    atom_to_feature_vector = None
    bond_to_feature_vector = None
    OGB_IMPORT_ERROR = exc
else:
    OGB_IMPORT_ERROR = None


def reorder_canonical_rank_atoms(mol: Chem.Mol) -> tuple[Chem.Mol, tuple[int, ...]]:
    order = tuple(zip(*sorted((rank, idx) for idx, rank in enumerate(Chem.CanonicalRankAtoms(mol)))))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


def smiles2graph(
    smiles_string: str,
    remove_hs: bool = True,
    reorder_atoms: bool = False,
) -> dict[str, np.ndarray | int]:
    """Official PCQM4Mv2/OGB-style SMILES to 2D graph conversion."""

    if atom_to_feature_vector is None or bond_to_feature_vector is None:
        raise ImportError("ogb is required for official atom/bond features.") from OGB_IMPORT_ERROR

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_string}")
    mol = mol if remove_hs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = reorder_canonical_rank_atoms(mol)

    atom_features_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    node_feat = np.array(atom_features_list, dtype=np.int64)

    num_bond_features = 3
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_feat = np.array(edge_features_list, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_feat = np.empty((0, num_bond_features), dtype=np.int64)

    return {
        "edge_index": edge_index,
        "edge_feat": edge_feat,
        "node_feat": node_feat,
        "num_nodes": len(node_feat),
    }


def iter_sdf_molecules(sdf_path: str | Path) -> Iterator[tuple[int, Chem.Mol | None]]:
    """Yield RDKit molecules from a PCQM4Mv2 SDF file without writing outputs."""

    supplier = Chem.SDMolSupplier(str(sdf_path))
    for idx, mol in enumerate(supplier):
        yield idx, mol


def load_official_split(split_root: str | Path) -> dict[str, set[str]]:
    """Load the official PCQM4Mv2 split from OGB and return string indices."""

    try:
        from ogb.lsc import PygPCQM4Mv2Dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("ogb is required to load the official PCQM4Mv2 split.") from exc

    dataset = PygPCQM4Mv2Dataset(root=str(split_root))
    split_dict = dataset.get_idx_split()
    return {
        "train": {str(int(idx)) for idx in split_dict["train"]},
        "valid": {str(int(idx)) for idx in split_dict["valid"]},
        "test-dev": {str(int(idx)) for idx in split_dict["test-dev"]},
        "test-challenge": {str(int(idx)) for idx in split_dict["test-challenge"]},
    }
