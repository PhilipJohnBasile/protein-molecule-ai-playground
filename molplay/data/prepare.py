from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import typer
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from molplay.utils import calculate_sa_score, ensure_directory, setup_logging

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")


app = typer.Typer(help="Prepare cleaned dataset with descriptors for modelling.")


def _find_latest_raw(target_family: str) -> Path:
    pattern = f"chembl_{target_family.lower()}_activities.parquet"
    candidate = RAW_DIR / pattern
    if not candidate.exists():
        raise FileNotFoundError(
            f"Did not find raw dataset {candidate}. Run `python -m molplay.data.fetch` first."
        )
    return candidate


def _compute_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    qed = float(QED.qed(mol))
    logp = float(Descriptors.MolLogP(mol))
    sa = float(calculate_sa_score(mol))
    return {"qed": qed, "logp": logp, "sa_score": sa}


@app.command()
def main(
    target_family: str = typer.Option("kinase", help="Target family keyword used during fetch."),
    min_pchembl: float = typer.Option(5.0, help="Drop rows with pChEMBL below this threshold."),
) -> None:
    setup_logging()
    raw_path = _find_latest_raw(target_family)
    df = pd.read_parquet(raw_path)

    required_cols = {
        "chembl_id",
        "smiles",
        "target_id",
        "pchembl_value",
        "target_sequence_length",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in raw dataset")

    df = df.dropna(subset=["smiles", "pchembl_value"])
    df = df[df["pchembl_value"] >= min_pchembl]

    descriptors = df["smiles"].apply(_compute_descriptors)
    mask_valid = descriptors.notnull()
    df = df[mask_valid].reset_index(drop=True)
    descriptor_df = pd.DataFrame(list(descriptors[mask_valid]))
    df = pd.concat([df, descriptor_df], axis=1)

    df = df.rename(columns={"pchembl_value": "pChEMBL", "target_sequence_length": "target_seq_len"})
    df = df[
        [
            "chembl_id",
            "smiles",
            "target_id",
            "target_seq_len",
            "pChEMBL",
            "qed",
            "logp",
            "sa_score",
        ]
    ]

    output_path = CLEAN_DIR / "train.parquet"
    ensure_directory(output_path.parent)
    df.to_parquet(output_path, index=False)

    meta = {
        "records": len(df),
        "targets": sorted(df["target_id"].unique().tolist()),
        "target_seq_len_mean": float(df["target_seq_len"].mean()),
        "pChEMBL_range": [float(df["pChEMBL"].min()), float(df["pChEMBL"].max())],
    }
    meta_path = CLEAN_DIR / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    typer.echo(f"Prepared dataset with {len(df)} rows at {output_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
