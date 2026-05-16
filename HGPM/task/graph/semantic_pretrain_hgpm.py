from __future__ import annotations

import argparse
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from HGPM.model.graph.hgpm import (
    HGPMGraphCollator,
    RelationAwareTransformerBlock,
)
from HGPM.task.graph.graph_base import HyperDAGDataset, load_hyperdag_data
from HGPM.utils.io import PACKAGE_ROOT, resolve_repo_path, save_json
from HGPM.utils.seed import set_seed
from HGPM.utils.training import masked_bce_loss, masked_ce_loss, masked_mse_loss, move_batch_to_device

REPO_ROOT = PACKAGE_ROOT


class HyperDAGPretrainDataset(Dataset):
    def __init__(self, base_dataset: HyperDAGDataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        return self.base_dataset[index]


class MaskedSemanticHGPMCollator:
    """Reuses the HGPM tokenizer and adds masked semantic reconstruction labels."""

    def __init__(self, *, drug_vocab: dict[str, int], node_feature_table: torch.Tensor, mask_rate: float, seed: int) -> None:
        self.base_collator = HGPMGraphCollator(drug_vocab=drug_vocab)
        self.node_feature_table = node_feature_table
        self.mask_rate = float(mask_rate)
        self.seed = int(seed)
        self.call_index = 0

    def __call__(self, batch: list[dict]) -> dict:
        base = self.base_collator(batch, self.node_feature_table)
        token_features = base["token_features"]
        seq_mask = base["seq_mask"]
        center_mask = base["center_token_mask"]
        rng = random.Random(self.seed * 1000003 + self.call_index)
        self.call_index += 1

        masked_positions = torch.zeros_like(seq_mask)
        teacher_inputs = token_features.clone()
        for batch_index in range(token_features.size(0)):
            valid_positions = [
                pos
                for pos in range(token_features.size(1))
                if seq_mask[batch_index, pos].item() > 0 and center_mask[batch_index, pos].item() == 0
            ]
            if not valid_positions:
                continue
            num_to_mask = max(1, int(round(len(valid_positions) * self.mask_rate)))
            chosen = rng.sample(valid_positions, k=min(num_to_mask, len(valid_positions)))
            for pos in chosen:
                masked_positions[batch_index, pos] = 1.0

        base["teacher_inputs"] = teacher_inputs
        base["masked_positions"] = masked_positions
        return base


class HGPMSemanticPretrainModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        order_vocab_size: int,
        exist_vocab_size: int,
        exist_source_vocab_size: int,
        view_vocab_size: int,
        teacher_ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.teacher_ema_decay = float(teacher_ema_decay)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.order_embedding = nn.Embedding(order_vocab_size, hidden_dim, padding_idx=0)
        self.exist_embedding = nn.Embedding(exist_vocab_size, hidden_dim, padding_idx=0)
        self.exist_source_embedding = nn.Embedding(exist_source_vocab_size, hidden_dim, padding_idx=0)
        self.view_embedding = nn.Embedding(view_vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(1024, hidden_dim)
        self.edge_direction_bias = nn.Embedding(3, num_heads)
        self.order_distance_bias = nn.Embedding(8, num_heads)
        self.overlap_bucket_bias = nn.Embedding(5, num_heads)
        self.exist_transition_bias = nn.Embedding(7, num_heads)
        self.sibling_bias = nn.Parameter(torch.zeros(num_heads))
        nn.init.zeros_(self.edge_direction_bias.weight)
        nn.init.zeros_(self.order_distance_bias.weight)
        nn.init.zeros_(self.overlap_bucket_bias.weight)
        nn.init.zeros_(self.exist_transition_bias.weight)
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [RelationAwareTransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.exist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.exist_source_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, exist_source_vocab_size),
        )
        self.teacher_feature_projection = copy.deepcopy(self.feature_projection)
        self.teacher_order_embedding = copy.deepcopy(self.order_embedding)
        self.teacher_exist_embedding = copy.deepcopy(self.exist_embedding)
        self.teacher_exist_source_embedding = copy.deepcopy(self.exist_source_embedding)
        self.teacher_view_embedding = copy.deepcopy(self.view_embedding)
        self.teacher_position_embedding = copy.deepcopy(self.position_embedding)
        self.teacher_blocks = copy.deepcopy(self.blocks)
        self.teacher_norm = copy.deepcopy(self.norm)
        self.teacher_target_head = copy.deepcopy(self.prediction_head)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        teacher_modules = [
            self.teacher_feature_projection,
            self.teacher_order_embedding,
            self.teacher_exist_embedding,
            self.teacher_exist_source_embedding,
            self.teacher_view_embedding,
            self.teacher_position_embedding,
            self.teacher_blocks,
            self.teacher_norm,
            self.teacher_target_head,
        ]
        for module in teacher_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, ema_decay: float | None = None) -> None:
        decay = float(self.teacher_ema_decay if ema_decay is None else ema_decay)
        student_teacher_pairs = [
            (self.feature_projection, self.teacher_feature_projection),
            (self.order_embedding, self.teacher_order_embedding),
            (self.exist_embedding, self.teacher_exist_embedding),
            (self.exist_source_embedding, self.teacher_exist_source_embedding),
            (self.view_embedding, self.teacher_view_embedding),
            (self.position_embedding, self.teacher_position_embedding),
            (self.blocks, self.teacher_blocks),
            (self.norm, self.teacher_norm),
            (self.prediction_head, self.teacher_target_head),
        ]
        for student, teacher in student_teacher_pairs:
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
            for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
                teacher_buffer.copy_(student_buffer)

    def build_attention_bias(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        edge = self.edge_direction_bias(batch["edge_direction_ids"]).permute(0, 3, 1, 2)
        order = self.order_distance_bias(batch["order_distance_ids"]).permute(0, 3, 1, 2)
        overlap = self.overlap_bucket_bias(batch["overlap_bucket_ids"]).permute(0, 3, 1, 2)
        exist = self.exist_transition_bias(batch["exist_transition_ids"]).permute(0, 3, 1, 2)
        sibling = batch["sibling_bias_mask"].unsqueeze(1) * self.sibling_bias.view(1, -1, 1, 1)
        return edge + order + overlap + exist + sibling

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token_features = batch["token_features"]
        batch_size, seq_len, _ = token_features.shape
        positions = torch.arange(seq_len, device=token_features.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.feature_projection(token_features)
        hidden = hidden + self.order_embedding(batch["order_ids"].clamp_max(self.order_embedding.num_embeddings - 1))
        hidden = hidden + self.exist_embedding(batch["exist_ids"].clamp_max(self.exist_embedding.num_embeddings - 1))
        hidden = hidden + self.exist_source_embedding(batch["exist_source_ids"].clamp_max(self.exist_source_embedding.num_embeddings - 1))
        hidden = hidden + self.view_embedding(batch["view_ids"].clamp_max(self.view_embedding.num_embeddings - 1))
        hidden = hidden + self.position_embedding(positions.clamp_max(self.position_embedding.num_embeddings - 1))
        mask_positions = batch["masked_positions"] > 0
        if mask_positions.any():
            hidden = hidden.clone()
            hidden[mask_positions] = self.mask_embedding
        attn_bias = self.build_attention_bias(batch)
        for block in self.blocks:
            hidden = block(hidden, batch["seq_mask"], attn_bias)
        return self.norm(hidden)

    def encode_teacher(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token_features = batch["teacher_inputs"]
        batch_size, seq_len, _ = token_features.shape
        positions = torch.arange(seq_len, device=token_features.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.teacher_feature_projection(token_features)
        hidden = hidden + self.teacher_order_embedding(batch["order_ids"].clamp_max(self.teacher_order_embedding.num_embeddings - 1))
        hidden = hidden + self.teacher_exist_embedding(batch["exist_ids"].clamp_max(self.teacher_exist_embedding.num_embeddings - 1))
        hidden = hidden + self.teacher_exist_source_embedding(batch["exist_source_ids"].clamp_max(self.teacher_exist_source_embedding.num_embeddings - 1))
        hidden = hidden + self.teacher_view_embedding(batch["view_ids"].clamp_max(self.teacher_view_embedding.num_embeddings - 1))
        hidden = hidden + self.teacher_position_embedding(positions.clamp_max(self.teacher_position_embedding.num_embeddings - 1))
        attn_bias = self.build_attention_bias(batch)
        for block in self.teacher_blocks:
            hidden = block(hidden, batch["seq_mask"], attn_bias)
        return self.teacher_norm(hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encode(batch)
        predicted = self.prediction_head(encoded)
        exist_logits = self.exist_head(encoded).squeeze(-1)
        exist_source_logits = self.exist_source_head(encoded)
        with torch.no_grad():
            teacher_target = self.teacher_target_head(self.encode_teacher(batch)).detach()
        return {
            "predicted_embeddings": predicted,
            "teacher_targets": teacher_target,
            "exist_logits": exist_logits,
            "exist_source_logits": exist_source_logits,
        }


def compute_pretrain_losses(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, lambda_exist: float, lambda_source: float) -> dict[str, torch.Tensor]:
    semantic_loss = masked_mse_loss(output["predicted_embeddings"], output["teacher_targets"], batch["masked_positions"])
    valid_mask = batch["seq_mask"]
    exist_targets = (batch["exist_ids"] - 1).clamp(min=0, max=1).to(dtype=output["exist_logits"].dtype)
    exist_loss = masked_bce_loss(output["exist_logits"], exist_targets, valid_mask)
    exist_source_loss = masked_ce_loss(output["exist_source_logits"], batch["exist_source_ids"], valid_mask)
    total_loss = semantic_loss + (lambda_exist * exist_loss) + (lambda_source * exist_source_loss)
    return {
        "total": total_loss,
        "semantic": semantic_loss,
        "exist": exist_loss,
        "exist_source": exist_source_loss,
    }


@torch.no_grad()
def evaluate_pretrain(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_exist: float,
    lambda_source: float,
    max_steps: int | None = None,
) -> dict:
    model.eval()
    total_losses = []
    semantic_losses = []
    exist_losses = []
    source_losses = []
    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        losses = compute_pretrain_losses(output, batch, lambda_exist=lambda_exist, lambda_source=lambda_source)
        total_losses.append(float(losses["total"].item()))
        semantic_losses.append(float(losses["semantic"].item()))
        exist_losses.append(float(losses["exist"].item()))
        source_losses.append(float(losses["exist_source"].item()))
        if max_steps is not None and step + 1 >= int(max_steps):
            break
    return {
        "pretrain_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "masked_semantic_mse": float(np.mean(semantic_losses)) if semantic_losses else 0.0,
        "exist_aux_loss": float(np.mean(exist_losses)) if exist_losses else 0.0,
        "exist_source_aux_loss": float(np.mean(source_losses)) if source_losses else 0.0,
    }


def build_pretrain_loaders(config: dict, hyperdag, node_feature_table: torch.Tensor, *, smoke: bool) -> tuple[DataLoader, DataLoader]:
    k_views = int(config["data"].get("k_views", 2))
    full_dataset = HyperDAGDataset(hyperdag, np.asarray(sorted(hyperdag.center_keys(), key=int), dtype=np.int64), k_views=k_views)
    indices = list(range(len(full_dataset)))
    rng = random.Random(int(config["seed"]))
    rng.shuffle(indices)
    train_ratio = float(config["data"].get("train_ratio", 0.9))
    train_cutoff = max(1, int(round(len(indices) * train_ratio)))
    train_subset = torch.utils.data.Subset(HyperDAGPretrainDataset(full_dataset), indices[:train_cutoff])
    val_subset = torch.utils.data.Subset(HyperDAGPretrainDataset(full_dataset), indices[train_cutoff:])
    batch_size = int(config["training"]["smoke_batch_size"] if smoke else config["training"]["batch_size"])
    collator = MaskedSemanticHGPMCollator(
        drug_vocab=hyperdag.drug_vocab,
        node_feature_table=node_feature_table,
        mask_rate=float(config["pretrain"].get("mask_rate", 0.2)),
        seed=int(config["seed"]),
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    return train_loader, val_loader


def run_pretraining(config: dict, *, smoke: bool = False) -> dict:
    set_seed(int(config["seed"]))
    hyperdag, node_feature_table = load_hyperdag_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() and bool(config["training"].get("use_gpu", True)) else "cpu")
    model = HGPMSemanticPretrainModel(
        input_dim=int(node_feature_table.size(1)),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        dropout=float(config["model"]["dropout"]),
        order_vocab_size=int(config["model"].get("order_vocab_size", 16)),
        exist_vocab_size=int(config["model"].get("exist_vocab_size", 4)),
        exist_source_vocab_size=int(config["model"].get("exist_source_vocab_size", 8)),
        view_vocab_size=int(config["model"].get("view_vocab_size", 8)),
        teacher_ema_decay=float(config.get("pretrain", {}).get("teacher_ema_decay", 0.996)),
    ).to(device)

    train_loader, val_loader = build_pretrain_loaders(config, hyperdag, node_feature_table, smoke=smoke)
    lambda_exist = float(config["pretrain"].get("lambda_exist", 0.0))
    lambda_source = float(config["pretrain"].get("lambda_source", 0.0))
    eval_max_steps = int(config["training"].get("smoke_max_eval_steps", config["training"].get("smoke_max_steps", 4))) if smoke else None
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    output_dir = REPO_ROOT / config["training"]["output_dir"]
    checkpoint_dir = REPO_ROOT / config["training"]["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config_used.json", {**config, "device": str(device)})

    best = {"epoch": 0, "val_loss": float("inf")}
    best_path = checkpoint_dir / "hgpm_graph_semantic_pretrain_best.pt"
    last_path = checkpoint_dir / "hgpm_graph_semantic_pretrain_last.pt"
    patience = int(config["training"].get("patience", 5))
    epochs = int(config["training"]["smoke_epochs"] if smoke else config["training"]["epochs"])
    stale = 0
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_losses = []
        semantic_losses = []
        exist_losses = []
        source_losses = []
        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_pretrain_losses(output, batch, lambda_exist=lambda_exist, lambda_source=lambda_source)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("gradient_clip_norm", 1.0)))
            optimizer.step()
            model.update_teacher()
            total_losses.append(float(losses["total"].item()))
            semantic_losses.append(float(losses["semantic"].item()))
            exist_losses.append(float(losses["exist"].item()))
            source_losses.append(float(losses["exist_source"].item()))
            if smoke and step + 1 >= int(config["training"].get("smoke_max_steps", 4)):
                break

        val_metrics = evaluate_pretrain(
            model,
            val_loader,
            device,
            lambda_exist=lambda_exist,
            lambda_source=lambda_source,
            max_steps=eval_max_steps,
        )
        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "train_masked_semantic_mse": float(np.mean(semantic_losses)) if semantic_losses else 0.0,
            "train_exist_aux_loss": float(np.mean(exist_losses)) if exist_losses else 0.0,
            "train_exist_source_aux_loss": float(np.mean(source_losses)) if source_losses else 0.0,
            "val_total_loss": float(val_metrics["pretrain_total_loss"]),
            "val_masked_semantic_mse": float(val_metrics["masked_semantic_mse"]),
            "val_exist_aux_loss": float(val_metrics["exist_aux_loss"]),
            "val_exist_source_aux_loss": float(val_metrics["exist_source_aux_loss"]),
        }
        history.append(row)
        print(
            f"[hgpm_graph_semantic_pretrain] epoch={epoch} train_total={row['train_total_loss']:.6f} "
            f"train_sem={row['train_masked_semantic_mse']:.6f} train_exist={row['train_exist_aux_loss']:.6f} "
            f"train_source={row['train_exist_source_aux_loss']:.6f} val_total={row['val_total_loss']:.6f} "
            f"val_sem={row['val_masked_semantic_mse']:.6f}"
        )
        torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, last_path)
        if row["val_total_loss"] < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": row["val_total_loss"]}
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, best_path)
            stale = 0
        else:
            stale += 1
            if not smoke and stale >= patience:
                break

    result = {
        "best": best,
        "checkpoint": str(best_path),
        "output_dir": str(output_dir),
        "train_time_sec": float(time.time() - start),
    }
    save_json(output_dir / "history.json", {"history": history})
    save_json(output_dir / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/graph/citeseer_pretrain.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_repo_path(args.config).read_text(encoding="utf-8"))
    result = run_pretraining(config, smoke=args.smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
