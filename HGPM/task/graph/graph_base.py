"""Shared building blocks for graph-side HGPM task drivers.

Contains the dataset wrapper, data container, attention pool, and evaluation
helpers reused by ``semantic_pretrain_hgpm`` and ``semantic_finetune_hgpm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from HGPM.utils.training import move_batch_to_device


@dataclass
class HyperDAGData:
    rows_by_center: dict[str, list[dict]] | None
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    num_classes: int
    drug_vocab: dict[str, int]
    available_centers: set[str] | None = None
    row_builder: Callable[[str, int], list[dict]] | None = None

    def has_center(self, center_id: str) -> bool:
        if self.rows_by_center is not None:
            return center_id in self.rows_by_center
        if self.available_centers is not None:
            return center_id in self.available_centers
        return False

    def center_keys(self) -> list[str]:
        if self.rows_by_center is not None:
            return list(self.rows_by_center.keys())
        if self.available_centers is not None:
            return list(self.available_centers)
        return []

    def get_rows(self, center_id: str, *, k_views: int) -> list[dict]:
        if self.rows_by_center is not None:
            rows = self.rows_by_center[center_id]
            return sorted(rows, key=lambda row: int(row.get("view_id", 0)))[: int(k_views)]
        if self.row_builder is None:
            raise KeyError(f"No rows or row builder available for center {center_id}.")
        rows = self.row_builder(center_id, int(k_views))
        return sorted(rows, key=lambda row: int(row.get("view_id", 0)))[: int(k_views)]


class HyperDAGDataset(Dataset):
    def __init__(self, data: HyperDAGData, node_indices: np.ndarray, *, k_views: int) -> None:
        self.data = data
        self.k_views = int(k_views)
        self.node_ids = [int(node_id) for node_id in node_indices if data.has_center(str(int(node_id)))]

    def __len__(self) -> int:
        return len(self.node_ids)

    def __getitem__(self, index: int) -> dict:
        node_id = self.node_ids[index]
        selected = self.data.get_rows(str(node_id), k_views=self.k_views)
        return {
            "node_id": str(node_id),
            "label": int(self.data.labels[node_id]),
            "views": selected,
        }


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        valid = seq_mask > 0
        scores = self.scorer(hidden).squeeze(-1)
        scores = scores.masked_fill(~valid, float("-inf"))
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            scores = scores.masked_fill(no_valid.unsqueeze(-1), 0.0)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, *, max_steps: int | None = None) -> dict:
    model.eval()
    losses = []
    all_pred = []
    all_label = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["labels"])
            losses.append(float(loss.item()))
            all_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_label.extend(batch["labels"].cpu().numpy().tolist())
            if max_steps is not None and step + 1 >= int(max_steps):
                break
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(all_label, all_pred)) if all_label else 0.0,
        "macro_f1": float(f1_score(all_label, all_pred, average="macro", zero_division=0)) if all_label else 0.0,
    }


def load_hyperdag_data(config: dict) -> tuple[HyperDAGData, torch.Tensor]:
    """Load benchmark protocol + centered DAG sequences into a HyperDAGData bundle.

    Active configs always set ``data.protocol_dir`` and route through the
    benchmark loader; older free-form configs are no longer supported.
    """

    data_cfg = config["data"]
    if not data_cfg.get("protocol_dir"):
        raise ValueError("config['data']['protocol_dir'] is required.")
    from HGPM.data.graph.benchmark_hyperdag_data import load_benchmark_hyperdag_data

    return load_benchmark_hyperdag_data(config)
