from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import typer
from torch_geometric.loader import DataLoader

from molplay.features import TargetEncoder, mol_to_graph, morgan_fingerprint
from molplay.models import (
    BaselineRegressor,
    GINAffinityNet,
    compute_metrics,
    load_gnn,
    save_gnn,
)
from molplay.utils import ScaffoldSplit, ensure_directory, scaffold_split, set_global_seed, setup_logging

app = typer.Typer(help="Train affinity models on the prepared dataset.")

DATA_PATH = Path("data/clean/train.parquet")
MODEL_DIR = Path("outputs/models")
METRICS_PATH = Path("outputs/metrics.json")


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {path} not found. Run `python -m molplay.data.fetch` and `python -m molplay.data.prepare` first."
        )
    return pd.read_parquet(path)


def _split_indices(df: pd.DataFrame, seed: int) -> ScaffoldSplit:
    return scaffold_split(df["smiles"].tolist(), frac_train=0.7, frac_valid=0.15, frac_test=0.15, seed=seed)


def _build_target_encoder(df: pd.DataFrame) -> TargetEncoder:
    encoder = TargetEncoder(method="one_hot")
    encoder.fit(df["target_id"].tolist(), df["target_seq_len"].tolist())
    return encoder


def _assemble_baseline_features(
    df: pd.DataFrame,
    encoder: TargetEncoder,
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    fps: List[np.ndarray] = []
    target_feats = encoder.transform(df["target_id"].tolist())
    seq_lengths = df["target_seq_len"].astype(np.float32).to_numpy()[:, None] / 2000.0
    physchem = df[["qed", "logp", "sa_score"]].to_numpy(dtype=np.float32)
    for smiles in df["smiles"]:
        fp = morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            raise ValueError(f"Failed to featurize SMILES {smiles}")
        fps.append(fp)
    fps_arr = np.stack(fps)
    return np.hstack([fps_arr, target_feats, seq_lengths, physchem])


def _train_baseline(df: pd.DataFrame, split: ScaffoldSplit, seed: int) -> Dict[str, Dict[str, float]]:
    encoder = _build_target_encoder(df)
    features = _assemble_baseline_features(df, encoder)
    labels = df["pChEMBL"].to_numpy(dtype=np.float32)

    idx_train, idx_valid, idx_test = split.train, split.valid, split.test

    X_train, y_train = features[idx_train], labels[idx_train]
    X_valid, y_valid = features[idx_valid], labels[idx_valid]
    X_test, y_test = features[idx_test], labels[idx_test]

    model = BaselineRegressor(random_state=seed)
    model.fit(X_train, y_train)

    metrics = {
        "train": compute_metrics(y_train, model.predict(X_train)).as_dict(),
        "valid": compute_metrics(y_valid, model.predict(X_valid)).as_dict(),
        "test": compute_metrics(y_test, model.predict(X_test)).as_dict(),
    }

    ensure_directory(MODEL_DIR)
    bundle = {
        "model": model.model,
        "target_mapping": getattr(encoder, "_mapping", {}),
        "target_lengths": getattr(encoder, "_lengths", {}),
        "method": encoder.method,
        "fingerprint": {"radius": 2, "n_bits": 1024},
        "feature_order": ["fingerprint", "target_one_hot", "target_seq_len", "physchem"],
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    baseline_path = MODEL_DIR / "baseline_latest.joblib"
    import joblib

    joblib.dump(bundle, baseline_path)
    typer.echo(f"Saved baseline model to {baseline_path}")
    return metrics


def _prepare_graph_dataset(df: pd.DataFrame, encoder: TargetEncoder) -> List:
    dataset = []
    for row in df.itertuples():
        graph = mol_to_graph(row.smiles)
        if graph is None:
            continue
        target_vec = encoder.transform([row.target_id])[0]
        aux = np.array(
            [row.target_seq_len / 2000.0, row.qed, row.logp, row.sa_score],
            dtype=np.float32,
        )
        target_tensor = torch.tensor(np.concatenate([target_vec, aux]), dtype=torch.float).unsqueeze(0)
        graph.target_feat = target_tensor
        graph.y = torch.tensor([row.pChEMBL], dtype=torch.float)
        dataset.append(graph)
    if len(dataset) != len(df):
        raise ValueError("Graph dataset size mismatch after featurization")
    return dataset


def _evaluate_gnn(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[float] = []
    trues: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            preds.extend(output.detach().cpu().numpy().tolist())
            trues.extend(batch.y.detach().cpu().numpy().flatten().tolist())
    metrics = compute_metrics(np.array(trues), np.array(preds)).as_dict()
    return metrics


def _train_gnn(
    df: pd.DataFrame,
    split: ScaffoldSplit,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, Dict[str, float]]:
    set_global_seed(seed)
    encoder = _build_target_encoder(df)
    dataset = _prepare_graph_dataset(df, encoder)

    idx_train, idx_valid, idx_test = split.train, split.valid, split.test
    train_dataset = [dataset[i] for i in idx_train]
    valid_dataset = [dataset[i] for i in idx_valid]
    test_dataset = [dataset[i] for i in idx_test]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = dataset[0].num_node_features
    target_dim = int(dataset[0].target_feat.view(-1).shape[0])
    model = GINAffinityNet(in_channels=in_channels, target_dim=target_dim, hidden_dim=hidden_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    best_valid = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch)
            loss = loss_fn(preds, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch % 5 == 0 or epoch == epochs:
            valid_metrics = _evaluate_gnn(model, valid_loader, device)
            if valid_metrics["mae"] < best_valid:
                best_valid = valid_metrics["mae"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            typer.echo(
                f"Epoch {epoch:03d} | train_loss={np.mean(epoch_losses):.4f} | valid_mae={valid_metrics['mae']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train": _evaluate_gnn(model, train_loader, device),
        "valid": _evaluate_gnn(model, valid_loader, device),
        "test": _evaluate_gnn(model, test_loader, device),
    }

    model_bundle_path = MODEL_DIR / "gnn_latest.pt"
    ensure_directory(MODEL_DIR)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "target_dim": target_dim,
            "hidden_dim": hidden_dim,
            "encoder": {
                "method": encoder.method,
                "mapping": getattr(encoder, "_mapping", {}),
                "lengths": getattr(encoder, "_lengths", {}),
            },
        },
        model_bundle_path,
    )
    typer.echo(f"Saved GNN model to {model_bundle_path}")
    return metrics


@app.command()
def main(
    model: str = typer.Option("baseline", help="Model to train: baseline or gnn"),
    seed: int = typer.Option(42, help="Random seed."),
    epochs: int = typer.Option(60, help="Epochs for GNN training."),
    batch_size: int = typer.Option(8, help="Batch size for GNN training."),
    hidden_dim: int = typer.Option(128, help="Hidden dimension for GNN."),
    lr: float = typer.Option(1e-3, help="Learning rate for GNN."),
    weight_decay: float = typer.Option(1e-5, help="Weight decay for GNN."),
) -> None:
    setup_logging()
    set_global_seed(seed)
    df = _load_dataset(DATA_PATH)

    split = _split_indices(df, seed)
    all_metrics: Dict[str, Dict[str, float]] = {}

    if model in {"baseline", "both"}:
        typer.echo("Training baseline random forest...")
        baseline_metrics = _train_baseline(df, split, seed)
        all_metrics["baseline"] = baseline_metrics
        typer.echo(
            f"Baseline MAE={baseline_metrics['test']['mae']:.3f} | R2={baseline_metrics['test']['r2']:.3f} (test set)"
        )

    if model in {"gnn", "both"}:
        typer.echo("Training GNN model...")
        gnn_metrics = _train_gnn(df, split, seed, epochs, batch_size, hidden_dim, lr, weight_decay)
        all_metrics["gnn"] = gnn_metrics
        typer.echo(f"GNN MAE={gnn_metrics['test']['mae']:.3f} | R2={gnn_metrics['test']['r2']:.3f} (test set)")

    ensure_directory(METRICS_PATH.parent)
    snapshot = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "metrics": all_metrics,
    }
    METRICS_PATH.write_text(json.dumps(snapshot, indent=2))
    typer.echo("Saved metrics to outputs/metrics.json")


if __name__ == "__main__":  # pragma: no cover
    app()
