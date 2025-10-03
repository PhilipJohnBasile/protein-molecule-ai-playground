from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import typer
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, QED

from molplay.features import morgan_fingerprint
from molplay.utils import calculate_sa_score, ensure_directory, set_global_seed, setup_logging

OUTPUT_PATH = Path("outputs/generated.csv")
MODEL_DIR = Path("outputs/models")


app = typer.Typer(help="Generate simple analogues and score them with the trained model.")


def _load_baseline_bundle(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {path}. Run `python -m molplay.train --model baseline` first"
        )
    return joblib.load(path)


def _physchem_descriptors(mol: Chem.Mol) -> Tuple[float, float, float]:
    qed = float(QED.qed(mol))
    logp = float(Descriptors.MolLogP(mol))
    sa = float(calculate_sa_score(mol))
    return qed, logp, sa


def _assemble_baseline_feature(smiles: str, target_id: str, bundle: Dict) -> np.ndarray:
    fp_cfg = bundle.get("fingerprint", {"radius": 2, "n_bits": 1024})
    fp = morgan_fingerprint(smiles, radius=fp_cfg["radius"], n_bits=fp_cfg["n_bits"])
    if fp is None:
        raise ValueError(f"Could not featurize SMILES {smiles}")

    mapping: Dict[str, int] = bundle.get("target_mapping", {})
    one_hot = np.zeros((len(mapping),), dtype=np.float32)
    if target_id in mapping:
        one_hot[mapping[target_id]] = 1.0
    lengths: Dict[str, float] = bundle.get("target_lengths", {})
    seq_len = np.array([lengths.get(target_id, 0.0) / 2000.0], dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES {smiles}")
    qed, logp, sa = _physchem_descriptors(mol)
    physchem = np.array([qed, logp, sa], dtype=np.float32)
    return np.hstack([fp, one_hot, seq_len, physchem])


def _predict_baseline(smiles: str, target_id: str, bundle: Dict) -> float:
    features = _assemble_baseline_feature(smiles, target_id, bundle)
    model = bundle["model"]
    pred = model.predict(features.reshape(1, -1))[0]
    return float(pred)


def _enumerate_brics(mol: Chem.Mol, max_samples: int, seed: int) -> List[str]:
    fragments = BRICS.BRICSDecompose(mol)
    if not fragments:
        return []
    fragment_mols = [Chem.MolFromSmiles(frag) for frag in fragments]
    fragment_mols = [frag for frag in fragment_mols if frag is not None]
    if not fragment_mols:
        return []
    rng = np.random.default_rng(seed)
    candidates: Set[str] = set()

    def _build_from(frags):
        for candidate in BRICS.BRICSBuild(frags):
            smi = Chem.MolToSmiles(candidate)
            if smi:
                candidates.add(Chem.CanonSmiles(smi))
            if len(candidates) >= max_samples * 5:
                break

    _build_from(fragment_mols)
    if len(candidates) < max_samples:
        shuffled = list(fragment_mols)
        rng.shuffle(shuffled)
        _build_from(shuffled)

    all_candidates = list(candidates)
    rng.shuffle(all_candidates)
    return all_candidates[: max_samples * 2]


@app.command()
def main(
    smiles: str = typer.Option(..., help="Seed molecule SMILES."),
    target_id: Optional[str] = typer.Option(None, help="Target identifier to condition predictions."),
    model_path: Optional[Path] = typer.Option(None, help="Path to trained baseline model bundle."),
    max_samples: int = typer.Option(40, help="Maximum number of analog candidates to evaluate."),
    top_k: int = typer.Option(10, help="Number of suggestions to keep."),
    min_qed: float = typer.Option(0.4, help="Lower bound on QED."),
    max_logp: float = typer.Option(5.0, help="Upper bound on logP."),
    seed: int = typer.Option(42, help="Random seed for enumeration."),
) -> None:
    setup_logging()
    set_global_seed(seed)
    bundle_path = model_path or (MODEL_DIR / "baseline_latest.joblib")
    bundle = _load_baseline_bundle(bundle_path)

    mapping = bundle.get("target_mapping", {})
    if target_id is None:
        if mapping:
            target_id = next(iter(mapping.keys()))
            typer.echo(f"No target provided. Using default target {target_id}.")
        else:
            raise typer.BadParameter("Model bundle missing target mapping and no target_id supplied.")

    seed_mol = Chem.MolFromSmiles(smiles)
    if seed_mol is None:
        raise typer.BadParameter(f"Invalid seed SMILES: {smiles}")

    candidates = _enumerate_brics(seed_mol, max_samples=max_samples, seed=seed)
    results: List[Dict[str, object]] = []
    seen: Set[str] = {Chem.CanonSmiles(smiles)}
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        mol = Chem.MolFromSmiles(cand)
        if mol is None:
            continue
        qed, logp, sa = _physchem_descriptors(mol)
        if qed < min_qed or logp > max_logp:
            continue
        try:
            pred = _predict_baseline(cand, target_id, bundle)
        except Exception:
            continue
        results.append(
            {
                "seed_smiles": smiles,
                "candidate_smiles": cand,
                "target_id": target_id,
                "qed": round(float(qed), 4),
                "logp": round(float(logp), 4),
                "sa_score": round(float(sa), 4),
                "predicted_pChEMBL": round(float(pred), 4),
            }
        )
        if len(results) >= max_samples:
            break

    if not results:
        typer.echo("No candidates met the filtering constraints.")
        return

    results.sort(key=lambda item: item["predicted_pChEMBL"], reverse=True)
    results = results[:top_k]

    ensure_directory(OUTPUT_PATH.parent)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    typer.echo(f"Wrote {len(results)} candidates to {OUTPUT_PATH}")


if __name__ == "__main__":  # pragma: no cover
    app()
