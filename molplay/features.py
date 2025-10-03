from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch_geometric.data import Data

ALLOWED_ATOMS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRALITY_TYPES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
    Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
    Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
    Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
}


def _atom_feature_vector(atom: Chem.Atom) -> np.ndarray:
    atomic_num = atom.GetAtomicNum()
    atom_type = np.array([1 if atomic_num == z else 0 for z in ALLOWED_ATOMS], dtype=np.float32)
    degree = np.array([atom.GetTotalDegree()], dtype=np.float32)
    valence = np.array([atom.GetTotalValence()], dtype=np.float32)
    formal_charge = np.array([atom.GetFormalCharge()], dtype=np.float32)
    is_aromatic = np.array([float(atom.GetIsAromatic())], dtype=np.float32)
    hybridization = np.array(
        [1 if atom.GetHybridization() == h else 0 for h in HYBRIDIZATION_TYPES], dtype=np.float32
    )
    chirality = np.array(
        [1 if atom.GetChiralTag() == c else 0 for c in CHIRALITY_TYPES], dtype=np.float32
    )
    explicit_hs = np.array([atom.GetTotalNumHs()], dtype=np.float32)
    return np.concatenate((atom_type, degree, valence, formal_charge, is_aromatic, explicit_hs, hybridization, chirality))


def _bond_feature_vector(bond: Optional[Chem.Bond]) -> np.ndarray:
    if bond is None:
        core = np.array([1, 0, 0, 0], dtype=np.float32)
        conjugated = np.array([0], dtype=np.float32)
        ring = np.array([0], dtype=np.float32)
    else:
        core = np.array(BOND_TYPES.get(bond.GetBondType(), [1, 0, 0, 0]), dtype=np.float32)
        conjugated = np.array([float(bond.GetIsConjugated())], dtype=np.float32)
        ring = np.array([float(bond.IsInRing())], dtype=np.float32)
    return np.concatenate((core, conjugated, ring))


def mol_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [_atom_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.stack(atom_features), dtype=torch.float)

    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_feature_vector(bond)
        edges.append([start, end])
        edges.append([end, start])
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    if edges:
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.stack(edge_attrs), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(_bond_feature_vector(None))), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    return data


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@dataclass
class TargetEncoder:
    method: str = "one_hot"

    def fit(self, targets: Sequence[str], sequence_lengths: Optional[Sequence[int]] = None) -> "TargetEncoder":
        unique_targets = sorted(set(targets))
        self._mapping = {tid: idx for idx, tid in enumerate(unique_targets)}
        self._lengths: Dict[str, int] = {}
        if sequence_lengths is not None:
            for tid, length in zip(targets, sequence_lengths):
                self._lengths.setdefault(tid, length)
        return self

    def transform(self, targets: Sequence[str]) -> np.ndarray:
        if self.method == "one_hot":
            size = len(getattr(self, "_mapping", {}))
            encoded = np.zeros((len(targets), size), dtype=np.float32)
            for i, tid in enumerate(targets):
                idx = self._mapping.get(tid)
                if idx is not None:
                    encoded[i, idx] = 1.0
            return encoded
        if self.method == "length_bucket":
            buckets = np.zeros((len(targets), 3), dtype=np.float32)
            for i, tid in enumerate(targets):
                length = self._lengths.get(tid, 0)
                if length < 800:
                    buckets[i, 0] = 1.0
                elif length < 1200:
                    buckets[i, 1] = 1.0
                else:
                    buckets[i, 2] = 1.0
            return buckets
        raise ValueError(f"Unknown encoding method {self.method}")

    def get_feature_size(self) -> int:
        if self.method == "one_hot":
            return len(getattr(self, "_mapping", {}))
        if self.method == "length_bucket":
            return 3
        raise ValueError(f"Unknown encoding method {self.method}")


__all__ = ["TargetEncoder", "mol_to_graph", "morgan_fingerprint"]
