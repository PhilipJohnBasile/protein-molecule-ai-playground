from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


LOGGER = logging.getLogger("molplay")


def setup_logging(level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def murcko_scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


@dataclass
class ScaffoldSplit:
    train: List[int]
    valid: List[int]
    test: List[int]


def scaffold_split(
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> ScaffoldSplit:
    if not math.isclose(frac_train + frac_valid + frac_test, 1.0, rel_tol=1e-6):
        raise ValueError("Split fractions must sum to 1.0")

    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaff = murcko_scaffold_from_smiles(smi)
        scaffolds[scaff].append(idx)

    rng = random.Random(seed)
    scaffold_groups = list(scaffolds.items())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=lambda kv: len(kv[1]), reverse=True)

    total_count = len(smiles_list)
    train_cutoff = int(round(frac_train * total_count))
    valid_cutoff = int(round((frac_train + frac_valid) * total_count))

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    for _, indices in scaffold_groups:
        if len(train_idx) + len(indices) <= train_cutoff:
            train_idx.extend(indices)
        elif len(train_idx) + len(valid_idx) + len(indices) <= valid_cutoff:
            valid_idx.extend(indices)
        else:
            test_idx.extend(indices)

    assigned = set(train_idx) | set(valid_idx) | set(test_idx)
    remaining = [idx for idx in range(total_count) if idx not in assigned]
    if remaining:
        test_idx.extend(remaining)

    return ScaffoldSplit(train=sorted(train_idx), valid=sorted(valid_idx), test=sorted(test_idx))


# Synthetic accessibility implementation adapted from RDKit's contrib scripts.
_atom_penalties = {
    1: 1.0,
    5: 1.7,
    6: 0.0,
    7: 0.0,
    8: 0.0,
    9: 1.5,
    15: 1.5,
    16: 0.0,
    17: 1.5,
    35: 1.5,
    53: 1.5,
}


def _ring_complexity(mol: Chem.Mol) -> float:
    ri = mol.GetRingInfo()
    ring_counts = len(ri.AtomRings())
    if ring_counts == 0:
        return 0.0
    fused_rings = sum(1 for ring in ri.BondRings() if len(ring) > 6)
    return math.log(2 + fused_rings)


def _fingerprint_density(mol: Chem.Mol) -> float:
    from rdkit.Chem import rdMolDescriptors

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = fp.GetNumOnBits()
    if mol.GetNumHeavyAtoms() == 0:
        return 0.0
    return float(onbits) / mol.GetNumHeavyAtoms()


def calculate_sa_score(mol: Chem.Mol) -> float:
    score = 0.0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        score += _atom_penalties.get(atomic_num, 2.0)
        if atom.GetDegree() > 4:
            score += (atom.GetDegree() - 4) * 0.5
    score += _ring_complexity(mol)
    density = _fingerprint_density(mol)
    score += max(0.0, density - 2.0)
    size_penalty = 0.0
    if mol.GetNumHeavyAtoms() > 50:
        size_penalty = math.log(mol.GetNumHeavyAtoms() - 49)
    score += size_penalty
    return max(1.0, min(10.0, 1.5 + score))


def save_json(data: Dict, path: Path) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(data, indent=2))


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


__all__ = [
    "ScaffoldSplit",
    "calculate_sa_score",
    "ensure_directory",
    "load_json",
    "murcko_scaffold_from_smiles",
    "save_json",
    "scaffold_split",
    "set_global_seed",
    "setup_logging",
]
