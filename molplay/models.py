from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool


@dataclass
class RegressionMetrics:
    mae: float
    r2: float

    def as_dict(self) -> Dict[str, float]:
        return {"mae": self.mae, "r2": self.r2}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return RegressionMetrics(mae=mae, r2=r2)


class BaselineRegressor:
    """Random forest baseline on concatenated Morgan fingerprint and target features."""

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineRegressor":
        model = cls()
        model.model = joblib.load(path)
        return model


class GINAffinityNet(nn.Module):
    """Simple 3-layer GIN network for molecular graph regression."""

    def __init__(self, in_channels: int, target_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        mlp3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gnn_layers = nn.ModuleList([
            GINConv(mlp1),
            GINConv(mlp2),
            GINConv(mlp3),
        ])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.target_proj = nn.Linear(target_dim, hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.gnn_layers, self.bn_layers):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        graph_emb = global_add_pool(x, batch)
        target_feat = data.target_feat
        if target_feat.dim() > 2:
            target_feat = target_feat.view(data.num_graphs, -1)
        elif target_feat.dim() == 1:
            target_feat = target_feat.view(data.num_graphs, -1)
        target_emb = torch.relu(self.target_proj(target_feat))
        combined = torch.cat([graph_emb, target_emb], dim=1)
        out = self.regressor(combined)
        return out.squeeze(-1)


def save_gnn(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_gnn(model: nn.Module, path: Path, map_location: Optional[str] = None) -> nn.Module:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


__all__ = [
    "BaselineRegressor",
    "GINAffinityNet",
    "RegressionMetrics",
    "compute_metrics",
    "load_gnn",
    "save_gnn",
]
