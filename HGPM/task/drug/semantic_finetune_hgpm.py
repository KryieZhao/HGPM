from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from HGPM.model.drug.hgpm import (
    HGPMRegimeCollator,
    HGPMDrugEncoder,
    HGPMRegimeModel,
)
from HGPM.data.drug.pretrain_dataset import load_drug_smiles_feature_table
from HGPM.data.drug.regime_dataset import (
    HoddiRecordTask1Dataset,
    build_record_task1_artifacts,
    build_record_views,
    build_train_context_db,
    discover_pretrain_sequence_file,
    load_pretrain_sequence_rows,
    load_record_frame,
    split_record_frame,
)
from HGPM.utils.io import PACKAGE_ROOT, resolve_repo_path, save_json
from HGPM.utils.seed import set_seed
from HGPM.utils.training import build_scheduler, move_batch_to_device

REPO_ROOT = PACKAGE_ROOT


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def resolve_pos_weight(records: list[dict], device: torch.device) -> torch.Tensor | None:
    positives = sum(int(record["label"]) for record in records)
    negatives = len(records) - positives
    if positives == 0:
        return None
    return torch.tensor(float(negatives / max(positives, 1)), dtype=torch.float32, device=device)


def run_epoch(
    model: HGPMRegimeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    mixed_precision: bool,
    gradient_clip_norm: float,
    pos_weight: torch.Tensor | None,
    max_steps: int | None = None,
) -> dict:
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    total_loss = 0.0
    total_steps = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for step, raw_batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=mixed_precision and device.type == "cuda"):
            logits = model(batch)
            loss = criterion(logits, batch["labels"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.item())
        total_steps += 1
    return {"train_loss": total_loss / max(total_steps, 1), "train_steps": total_steps}


def _safe_auroc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    return {
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_metrics_by_order(predictions: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in predictions:
        grouped.setdefault(str(row["drug_count"]), []).append(row)
    payload = {}
    for order, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        labels = [int(row["label"]) for row in rows]
        scores = [float(row["pred_prob"]) for row in rows]
        metrics = compute_binary_metrics(labels, scores)
        metrics["num_records"] = len(rows)
        payload[order] = metrics
    return payload


def evaluate_record_task1(
    model: HGPMRegimeModel,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool,
    max_steps: int | None = None,
) -> tuple[dict, dict, list[dict], list[dict]]:
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    losses = []
    predictions = []
    debug_examples = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for step, raw_batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = move_batch_to_device(raw_batch, device)
            with torch.autocast(device_type=autocast_device, enabled=mixed_precision and device.type == "cuda"):
                logits = model(batch)
                loss = criterion(logits, batch["labels"])
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            labels = batch["labels"].detach().cpu().tolist()
            for idx, record_id in enumerate(raw_batch["record_ids"]):
                row = {
                    "record_id": record_id,
                    "drug_set": raw_batch["drug_sets"][idx],
                    "drug_count": len(raw_batch["drug_sets"][idx]),
                    "side_effect_id": raw_batch["record_side_effect_ids"][idx].item() if torch.is_tensor(raw_batch["record_side_effect_ids"]) else raw_batch["record_side_effect_ids"][idx],
                    "label": int(labels[idx]),
                    "pred_prob": float(probs[idx]),
                    "pred_label": int(probs[idx] >= 0.5),
                }
                predictions.append(row)
                if len(debug_examples) < 5:
                    debug_examples.append(row)
    labels = [int(row["label"]) for row in predictions]
    scores = [float(row["pred_prob"]) for row in predictions]
    metrics = compute_binary_metrics(labels, scores)
    metrics["val_loss"] = float(sum(losses) / max(len(losses), 1))
    metrics["num_records"] = len(predictions)
    return metrics, compute_metrics_by_order(predictions), predictions, debug_examples


def load_pretrained_hgpm_encoder(model: HGPMRegimeModel, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = {}
    for key, value in state_dict.items():
        if not key.startswith("encoder."):
            continue
        plain_key = key[len("encoder.") :]
        current = model.encoder.state_dict().get(plain_key)
        if current is None:
            continue
        if current.shape == value.shape:
            encoder_state[plain_key] = value
        elif plain_key == "side_effect_embedding.weight" and value.dim() == 2 and current.dim() == 2:
            expanded = current.clone()
            rows = min(value.size(0), expanded.size(0))
            cols = min(value.size(1), expanded.size(1))
            expanded[:rows, :cols] = value[:rows, :cols]
            encoder_state[plain_key] = expanded
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(
        f"[hgpm_drug_semantic_finetune] loaded encoder from {checkpoint_path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HGPM drug semantic finetune.")
    parser.add_argument("--config", default="config/drug/hoddi_finetune.yaml")
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
    pretrain_rows = load_pretrain_sequence_rows(sequence_path)
    record_frame = load_record_frame(resolve_repo_path(data_cfg["record_csv"]))
    train_df, val_df, test_df = split_record_frame(record_frame, data_cfg.get("split_column", "split"))
    artifacts = build_record_task1_artifacts(hoddi_output_dir, pretrain_rows, record_frame)
    drug_feature_table, smiles_meta = load_drug_smiles_feature_table(resolve_repo_path(data_cfg["drug_smiles_feature_path"]), artifacts)
    context_db = build_train_context_db(train_df)
    build_kwargs = {
        "context_db": context_db,
        "max_context_order": int(data_cfg.get("max_context_order", 4)),
        "keep_unknown_context": bool(data_cfg.get("keep_unknown_context", True)),
        "mask_target_token": bool(data_cfg.get("mask_target_token", True)),
        "order_clip": int(data_cfg.get("order_clip", 7)),
    }
    train_records = build_record_views(train_df, **build_kwargs)
    val_records = build_record_views(val_df, **build_kwargs)
    test_records = build_record_views(test_df, **build_kwargs)

    train_single_view = bool(data_cfg.get("train_single_view", False))
    train_single_view_seed = int(data_cfg.get("train_single_view_seed", config["seed"]))

    train_loader = DataLoader(
        HoddiRecordTask1Dataset(
            train_records,
            single_view_mode=train_single_view,
            single_view_seed=train_single_view_seed,
        ),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=HGPMRegimeCollator(
            artifacts,
            use_exist_input_in_formal=bool(model_cfg.get("use_exist_input_in_formal", True)),
            use_comp_input_in_formal=bool(model_cfg.get("use_comp_input_in_formal", True)),
        ),
    )
    val_loader = DataLoader(
        HoddiRecordTask1Dataset(val_records),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=HGPMRegimeCollator(
            artifacts,
            use_exist_input_in_formal=bool(model_cfg.get("use_exist_input_in_formal", True)),
            use_comp_input_in_formal=bool(model_cfg.get("use_comp_input_in_formal", True)),
        ),
    )
    test_loader = DataLoader(
        HoddiRecordTask1Dataset(test_records),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=HGPMRegimeCollator(
            artifacts,
            use_exist_input_in_formal=bool(model_cfg.get("use_exist_input_in_formal", True)),
            use_comp_input_in_formal=bool(model_cfg.get("use_comp_input_in_formal", True)),
        ),
    )

    encoder = HGPMDrugEncoder(
        num_drugs=len(artifacts.drug_vocab),
        num_side_effects=len(artifacts.encoder_side_effect_vocab),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        drug_feature_table=drug_feature_table,
        use_drug_id_residual=bool(model_cfg.get("use_drug_id_residual", True)),
        use_side_effect_in_encoder=bool(model_cfg.get("use_side_effect_in_encoder", False)),
    )
    model = HGPMRegimeModel(
        encoder=encoder,
        num_record_side_effects=len(artifacts.record_side_effect_vocab),
        hidden_dim=int(model_cfg["hidden_dim"]),
        record_side_effect_dim=int(model_cfg["record_side_effect_dim"]),
        dropout=float(model_cfg["dropout"]),
        pooling=str(model_cfg.get("pooling", "mean")),
        center_representation=str(model_cfg.get("center_representation", "center_only")),
    ).to(device)
    load_pretrained_hgpm_encoder(model, resolve_repo_path(model_cfg["resume_semantic_checkpoint"]))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    total_steps = max(1, len(train_loader) * int(train_cfg["epochs"]))
    scheduler = build_scheduler(optimizer, total_steps, float(train_cfg["warmup_ratio"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda")
    pos_weight = resolve_pos_weight(train_records, device)

    save_json(
        output_dir / "config_used.yaml",
        {
            **config,
            "device": str(device),
            "resolved_sequence_path": str(sequence_path),
            "resolved_pretrain_checkpoint": str(resolve_repo_path(model_cfg["resume_semantic_checkpoint"]).resolve()),
            "resolved_drug_smiles_feature_path": str(resolve_repo_path(data_cfg["drug_smiles_feature_path"]).resolve()),
            "smiles_feature_meta": smiles_meta,
            "train_single_view": train_single_view,
            "train_single_view_seed": train_single_view_seed,
            "record_split_sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        },
    )

    best_metric = None
    best_epoch = -1
    patience = 0
    selection_metric = str(model_cfg.get("selection_metric", "auroc"))
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            mixed_precision=bool(train_cfg.get("mixed_precision", True)),
            gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
            pos_weight=pos_weight,
            max_steps=int(train_cfg.get("max_train_steps_per_epoch", 0)) or None,
        )
        val_metrics, val_by_order, val_predictions, val_examples = evaluate_record_task1(
            model,
            val_loader,
            device,
            mixed_precision=bool(train_cfg.get("mixed_precision", True)),
            max_steps=int(train_cfg.get("max_val_steps", 0)) or None,
        )
        log_row = {"epoch": epoch, **train_stats, **val_metrics}
        append_jsonl(output_dir / root_train_cfg.get("train_log_name", "train_log.jsonl"), log_row)
        metric_value = val_metrics.get(selection_metric)
        current_metric = float(metric_value) if metric_value is not None else float("-inf")
        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
        }
        save_checkpoint(checkpoint_dir / root_train_cfg.get("last_checkpoint_name", "formal_last.pt"), checkpoint_payload)
        if best_metric is None or current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience = 0
            save_checkpoint(checkpoint_dir / root_train_cfg.get("best_checkpoint_name", "formal_best.pt"), checkpoint_payload)
            save_json(output_dir / root_train_cfg.get("val_metrics_name", "formal_val_metrics.json"), {**val_metrics, "best_epoch": best_epoch, "selection_metric": selection_metric})
            save_json(output_dir / root_train_cfg.get("metrics_by_order_name", "formal_metrics_by_order.json"), {"val": val_by_order})
            write_jsonl(output_dir / root_train_cfg.get("val_predictions_name", "formal_val_predictions.jsonl"), val_predictions)
            save_json(output_dir / root_train_cfg.get("val_examples_name", "formal_val_examples.json"), val_examples)
        else:
            patience += 1
            if patience >= int(train_cfg["early_stop_patience"]):
                break

    best_checkpoint = torch.load(checkpoint_dir / root_train_cfg.get("best_checkpoint_name", "formal_best.pt"), map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state"])
    model.to(device)
    final_eval_max_steps = (int(train_cfg.get("max_val_steps", 0)) or None) if args.smoke else None
    val_metrics, val_by_order, val_predictions, val_examples = evaluate_record_task1(
        model,
        val_loader,
        device,
        mixed_precision=bool(train_cfg.get("mixed_precision", True)),
        max_steps=final_eval_max_steps,
    )
    test_metrics, test_by_order, test_predictions, test_examples = evaluate_record_task1(
        model,
        test_loader,
        device,
        mixed_precision=bool(train_cfg.get("mixed_precision", True)),
        max_steps=final_eval_max_steps,
    )
    save_json(output_dir / root_train_cfg.get("val_metrics_name", "formal_val_metrics.json"), {**val_metrics, "best_epoch": best_epoch, "selection_metric": selection_metric})
    save_json(output_dir / root_train_cfg.get("test_metrics_name", "formal_test_metrics.json"), {**test_metrics, "best_epoch": best_epoch, "selection_metric": selection_metric})
    save_json(output_dir / root_train_cfg.get("metrics_by_order_name", "formal_metrics_by_order.json"), {"val": val_by_order, "test": test_by_order})
    write_jsonl(output_dir / root_train_cfg.get("val_predictions_name", "formal_val_predictions.jsonl"), val_predictions)
    write_jsonl(output_dir / root_train_cfg.get("test_predictions_name", "formal_test_predictions.jsonl"), test_predictions)
    save_json(output_dir / root_train_cfg.get("val_examples_name", "formal_val_examples.json"), val_examples)
    save_json(output_dir / root_train_cfg.get("test_examples_name", "formal_test_examples.json"), test_examples)

    result = {
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection_metric": selection_metric,
    }
    baseline_metrics_path = model_cfg.get("baseline_metrics_path")
    if baseline_metrics_path:
        save_json(
            output_dir / "comparison_with_hoddi_target_centered_hyperedge.json",
            {
                "baseline_metrics_path": str(resolve_repo_path(baseline_metrics_path)),
                "baseline": json.loads(resolve_repo_path(baseline_metrics_path).read_text(encoding="utf-8")),
                "new_result": result,
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
