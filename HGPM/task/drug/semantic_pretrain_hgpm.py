from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from HGPM.model.drug.hgpm import (
    HGPMDrugEncoder,
    HGPMSemanticPretrainCollator,
    HGPMSemanticPretrainModel,
)
from HGPM.data.drug.pretrain_collator import build_pretrain_artifacts
from HGPM.data.drug.pretrain_dataset import (
    HoddiPretrainDataset,
    build_or_load_sample_splits,
    load_drug_smiles_feature_table,
)
from HGPM.data.drug.regime_dataset import discover_pretrain_sequence_file, load_pretrain_sequence_rows
from HGPM.utils.io import PACKAGE_ROOT, resolve_repo_path, save_json
from HGPM.utils.seed import set_seed
from HGPM.utils.training import (
    build_scheduler,
    masked_ce_loss,
    masked_mse_loss,
    move_batch_to_device,
)

REPO_ROOT = PACKAGE_ROOT


def compute_pretrain_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    stage: str,
    lambda_exist: float,
    lambda_comp: float,
) -> tuple[torch.Tensor, dict]:
    semantic = masked_mse_loss(output["predicted_embeddings"], output["teacher_targets"], batch["masked_positions"])
    exist = masked_ce_loss(output["exist_logits"], batch["exist_target_ids"].clamp(min=0, max=1), batch["masked_positions"])
    if stage == "stage2":
        comp = masked_ce_loss(output["comp_logits"], batch["primary_comp_ids"], batch["comp_masked_positions"])
    else:
        comp = output["comp_logits"].sum() * 0.0
    total = semantic + float(lambda_exist) * exist + float(lambda_comp) * comp
    return total, {
        "semantic_loss": float(semantic.item()),
        "exist_loss": float(exist.item()),
        "comp_loss": float(comp.item()),
        "total_loss": float(total.item()),
    }


@torch.no_grad()
def evaluate_pretrain(model: HGPMSemanticPretrainModel, loader: DataLoader, device: torch.device, *, mixed_precision: bool) -> dict:
    model.eval()
    losses = []
    semantic_losses = []
    exist_losses = []
    comp_losses = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(device_type=autocast_device, enabled=mixed_precision and device.type == "cuda"):
            output = model(batch)
            loss, parts = compute_pretrain_losses(
                output,
                batch,
                stage=str(getattr(model, "_pretrain_stage", "stage1")),
                lambda_exist=float(getattr(model, "_lambda_exist", 0.05)),
                lambda_comp=float(getattr(model, "_lambda_comp", 0.0)),
            )
        losses.append(float(loss.item()))
        semantic_losses.append(parts["semantic_loss"])
        exist_losses.append(parts["exist_loss"])
        comp_losses.append(parts["comp_loss"])
    return {
        "pretrain_loss": float(np.mean(losses)) if losses else 0.0,
        "semantic_loss": float(np.mean(semantic_losses)) if semantic_losses else 0.0,
        "exist_loss": float(np.mean(exist_losses)) if exist_losses else 0.0,
        "comp_loss": float(np.mean(comp_losses)) if comp_losses else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HGPM drug semantic pretrain.")
    parser.add_argument("--config", default="config/drug/hoddi_pretrain.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(resolve_repo_path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]["smoke" if args.smoke else "formal"]
    root_train_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() and root_train_cfg.get("use_gpu", True) else "cpu")

    output_dir = REPO_ROOT / root_train_cfg["output_dir"]
    checkpoint_dir = REPO_ROOT / root_train_cfg["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hoddi_output_dir = resolve_repo_path(data_cfg["hoddi_output_dir"])
    sequence_path = discover_pretrain_sequence_file(hoddi_output_dir, data_cfg.get("sequence_glob"))
    rows = load_pretrain_sequence_rows(sequence_path)
    base_drug_vocab = json.loads((hoddi_output_dir / "drug2id.json").read_text(encoding="utf-8"))
    artifacts = build_pretrain_artifacts(rows, base_drug_vocab)
    drug_feature_table, smiles_meta = load_drug_smiles_feature_table(resolve_repo_path(data_cfg["drug_smiles_feature_path"]), artifacts)

    train_idx, val_idx, train_split_path, val_split_path = build_or_load_sample_splits(
        rows,
        resolve_repo_path(data_cfg.get("split_dir", "outputs/hoddi_hgpm_splits")),
        seed=int(config["seed"]),
        train_ratio=float(data_cfg.get("train_ratio", 0.9)),
        split_prefix=str(data_cfg.get("split_prefix", "hgpm_semantic")),
    )
    train_ds = HoddiPretrainDataset(rows, train_idx)
    val_ds = HoddiPretrainDataset(rows, val_idx)
    pretrain_stage = str(config.get("pretrain", {}).get("stage", "stage1"))
    token_mask_rate = float(config.get("pretrain", {}).get("token_mask_rate", config.get("pretrain", {}).get("mask_rate", 0.2)))
    comp_mask_rate = float(config.get("pretrain", {}).get("comp_mask_rate", 0.2))
    lambda_exist = float(config.get("pretrain", {}).get("lambda_exist", 0.05))
    lambda_comp = float(config.get("pretrain", {}).get("lambda_comp", 0.05 if pretrain_stage == "stage2" else 0.0))
    collator = HGPMSemanticPretrainCollator(
        artifacts,
        stage=pretrain_stage,
        token_mask_rate=token_mask_rate,
        comp_mask_rate=comp_mask_rate,
        seed=int(config["seed"]),
    )
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    encoder = HGPMDrugEncoder(
        num_drugs=len(artifacts.drug_vocab),
        num_side_effects=len(artifacts.side_effect_vocab),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        drug_feature_table=drug_feature_table,
        use_drug_id_residual=bool(model_cfg.get("use_drug_id_residual", True)),
        use_side_effect_in_encoder=bool(model_cfg.get("use_side_effect_in_encoder", False)),
    )
    model = HGPMSemanticPretrainModel(
        encoder,
        hidden_dim=int(model_cfg["hidden_dim"]),
        raw_feature_dim=int(smiles_meta["feature_dim"]),
        dropout=float(model_cfg["dropout"]),
        teacher_ema_decay=float(config.get("pretrain", {}).get("teacher_ema_decay", 0.996)),
    ).to(device)
    model._pretrain_stage = pretrain_stage
    model._lambda_exist = lambda_exist
    model._lambda_comp = lambda_comp
    if pretrain_stage == "stage2":
        resume_stage1_checkpoint = model_cfg.get("resume_stage1_checkpoint")
        if resume_stage1_checkpoint:
            checkpoint = torch.load(resolve_repo_path(resume_stage1_checkpoint), map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    total_steps = max(1, len(train_loader) * int(train_cfg["epochs"]))
    scheduler = build_scheduler(optimizer, total_steps, float(train_cfg["warmup_ratio"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda")

    save_json(
        output_dir / "config_used.json",
        {
            **config,
            "device": str(device),
            "resolved_sequence_path": str(sequence_path),
            "resolved_drug_smiles_feature_path": str(resolve_repo_path(data_cfg["drug_smiles_feature_path"]).resolve()),
            "smiles_feature_meta": smiles_meta,
            "train_split_path": str(train_split_path),
            "val_split_path": str(val_split_path),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
    )

    best = {"epoch": 0, "val_loss": float("inf")}
    history = []
    best_path = checkpoint_dir / "hgpm_drug_semantic_pretrain_best.pt"
    last_path = checkpoint_dir / "hgpm_drug_semantic_pretrain_last.pt"
    stale = 0
    start = time.time()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    max_steps = int(train_cfg.get("max_train_steps_per_epoch", 0)) or None

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        for step, raw_batch in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"):
                output = model(batch)
                loss, _ = compute_pretrain_losses(
                    output,
                    batch,
                    stage=pretrain_stage,
                    lambda_exist=lambda_exist,
                    lambda_comp=lambda_comp,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            model.update_teacher()
            scheduler.step()
            train_losses.append(float(loss.item()))
        val_metrics = evaluate_pretrain(model, val_loader, device, mixed_precision=bool(train_cfg.get("mixed_precision", True)))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": float(val_metrics["pretrain_loss"]),
            "val_semantic_loss": float(val_metrics["semantic_loss"]),
            "val_exist_loss": float(val_metrics["exist_loss"]),
            "val_comp_loss": float(val_metrics["comp_loss"]),
        }
        history.append(row)
        print(f"[hgpm_drug_semantic_pretrain] epoch={epoch} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f}")
        torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, last_path)
        if row["val_loss"] < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": row["val_loss"]}
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, best_path)
            stale = 0
        else:
            stale += 1
            if stale >= int(train_cfg["early_stop_patience"]):
                break

    result = {
        "best": best,
        "checkpoint": str(best_path),
        "output_dir": str(output_dir),
        "train_time_sec": float(time.time() - start),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    save_json(output_dir / "history.json", {"history": history})
    save_json(output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
