from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from HGPM.model.graph.hgpm import HGPMGraphCollator, HGPMGraphModel
from HGPM.task.graph.graph_base import HyperDAGDataset, evaluate, load_hyperdag_data
from HGPM.utils.io import PACKAGE_ROOT, resolve_repo_path, save_json
from HGPM.utils.seed import set_seed
from HGPM.utils.training import move_batch_to_device

REPO_ROOT = PACKAGE_ROOT


def load_pretrained_hgpm_encoder(model: HGPMGraphModel, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state.get("model_state_dict", state)
    transferable_prefixes = (
        "feature_projection.",
        "order_embedding.",
        "exist_embedding.",
        "exist_source_embedding.",
        "view_embedding.",
        "position_embedding.",
        "edge_direction_bias.",
        "order_distance_bias.",
        "overlap_bucket_bias.",
        "exist_transition_bias.",
        "sibling_bias",
        "blocks.",
        "norm.",
    )
    encoder_state = {key: value for key, value in model_state.items() if key.startswith(transferable_prefixes)}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(
        f"[benchmark_hgpm_semantic_finetune] loaded encoder from {checkpoint_path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def run_training(config: dict, *, smoke: bool = False) -> dict:
    set_seed(int(config["seed"]))
    hyperdag, node_feature_table = load_hyperdag_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() and bool(config["training"].get("use_gpu", True)) else "cpu")
    model = HGPMGraphModel(
        input_dim=int(node_feature_table.size(1)),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_classes=hyperdag.num_classes,
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        dropout=float(config["model"]["dropout"]),
        order_vocab_size=int(config["model"].get("order_vocab_size", 16)),
        exist_vocab_size=int(config["model"].get("exist_vocab_size", 4)),
        exist_source_vocab_size=int(config["model"].get("exist_source_vocab_size", 8)),
        view_vocab_size=int(config["model"].get("view_vocab_size", 8)),
    ).to(device)
    load_pretrained_hgpm_encoder(model, resolve_repo_path(config["training"]["init_checkpoint"]))

    train_ds = HyperDAGDataset(hyperdag, hyperdag.train_idx, k_views=int(config["data"].get("k_views", 2)))
    val_ds = HyperDAGDataset(hyperdag, hyperdag.val_idx, k_views=int(config["data"].get("k_views", 2)))
    test_ds = HyperDAGDataset(hyperdag, hyperdag.test_idx, k_views=int(config["data"].get("k_views", 2)))
    batch_size = int(config["training"]["smoke_batch_size"] if smoke else config["training"]["batch_size"])
    collator = HGPMGraphCollator(drug_vocab=hyperdag.drug_vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collator(batch, node_feature_table), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collator(batch, node_feature_table), num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collator(batch, node_feature_table), num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    output_dir = REPO_ROOT / config["training"]["output_dir"]
    checkpoint_dir = REPO_ROOT / config["training"]["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config_used.json", {**config, "device": str(device)})

    best = {"epoch": 0, "val_accuracy": -1.0}
    best_path = checkpoint_dir / "hgpm_graph_semantic_finetune_best.pt"
    last_path = checkpoint_dir / "hgpm_graph_semantic_finetune_last.pt"
    patience = int(config["training"].get("patience", 5))
    epochs = int(config["training"]["smoke_epochs"] if smoke else config["training"]["epochs"])
    eval_max_steps = int(config["training"].get("smoke_max_eval_steps", config["training"].get("smoke_max_steps", 4))) if smoke else None
    stale = 0
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("gradient_clip_norm", 1.0)))
            optimizer.step()
            losses.append(float(loss.item()))
            if smoke and step + 1 >= int(config["training"].get("smoke_max_steps", 4)):
                break

        val_metrics = evaluate(model, val_loader, device, max_steps=eval_max_steps)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
        }
        history.append(row)
        print(
            f"[benchmark_hgpm_semantic_finetune] epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_macro_f1={row['val_macro_f1']:.4f}"
        )
        torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, last_path)
        if row["val_accuracy"] > best["val_accuracy"]:
            best = {"epoch": epoch, **row}
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, best_path)
            stale = 0
        else:
            stale += 1
            if not smoke and stale >= patience:
                break

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, max_steps=eval_max_steps)
    result = {
        "best": best,
        "test": test_metrics,
        "checkpoint": str(best_path),
        "output_dir": str(output_dir),
        "train_time_sec": float(time.time() - start),
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    save_json(output_dir / "history.json", {"history": history})
    save_json(output_dir / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/graph/citeseer_finetune.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_repo_path(args.config).read_text(encoding="utf-8"))
    result = run_training(config, smoke=args.smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
