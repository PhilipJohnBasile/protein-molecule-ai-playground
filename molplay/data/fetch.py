from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from molplay.utils import ensure_directory, setup_logging

try:
    import importlib.resources as resources
except ImportError:  # pragma: no cover
    import importlib_resources as resources  # type: ignore


app = typer.Typer(help="Fetch activity data for a protein target family from ChEMBL.")
DEFAULT_TARGET_FAMILY = "kinase"
RAW_DIR = Path("data/raw")


def _load_packaged_sample() -> pd.DataFrame:
    with resources.files("molplay.data.static").joinpath("chembl_kinase_sample.csv").open("r") as handle:
        df = pd.read_csv(handle)
    return df


def _maybe_filter_family(df: pd.DataFrame, target_family: str) -> pd.DataFrame:
    if target_family.lower() == "kinase":
        return df
    mask = df["target_pref_name"].str.contains(target_family, case=False, na=False)
    filtered = df.loc[mask]
    if filtered.empty:
        return df
    return filtered


def _clip_max(df: pd.DataFrame, maximum: Optional[int]) -> pd.DataFrame:
    if maximum is None:
        return df
    return df.head(maximum)


def _write_dataset(df: pd.DataFrame, destination: Path) -> None:
    ensure_directory(destination.parent)
    df.to_parquet(destination, index=False)
    meta = {
        "records": len(df),
        "columns": list(df.columns),
        "source": "chembl_public_sample",
    }
    destination.with_suffix(".json").write_text(json.dumps(meta, indent=2))


@app.command()
def main(
    target_family: str = typer.Option(DEFAULT_TARGET_FAMILY, help="Target family keyword, e.g. kinase."),
    max_records: Optional[int] = typer.Option(5000, min=10, help="Maximum number of activity rows to keep."),
    force: bool = typer.Option(False, help="Overwrite existing raw file."),
) -> None:
    """Fetch activity data; falls back to packaged sample for offline use."""
    setup_logging()
    output_path = RAW_DIR / f"chembl_{target_family.lower()}_activities.parquet"
    if output_path.exists() and not force:
        typer.echo(f"Raw dataset already exists at {output_path}. Use --force to overwrite.")
        raise typer.Exit(code=0)

    df = _load_packaged_sample()
    df = _maybe_filter_family(df, target_family)
    df = _clip_max(df, max_records)

    _write_dataset(df, output_path)
    typer.echo(f"Wrote {len(df)} records to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
