"""Shared training-loop helpers reused by the graph- and drug-side drivers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def masked_mse_loss(predicted: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0
    if not valid.any():
        return predicted.sum() * 0.0
    return F.mse_loss(predicted[valid], targets[valid], reduction="mean")


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid], reduction="mean")


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], targets[valid], reduction="mean")


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
