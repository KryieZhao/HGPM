from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def load_sequence_rows(sequence_path: Path) -> list[dict]:
    rows: list[dict] = []
    with sequence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_drug_smiles_feature_table(feature_path: Path, artifacts) -> tuple[torch.Tensor, dict]:
    """Load a precomputed SMILES feature table and align it to ``artifacts.drug_vocab``.

    Returns the (vocab_size, feature_dim) tensor plus a small metadata dict
    describing how many rows were mapped vs. missing.
    """

    payload = torch.load(feature_path, map_location="cpu")
    feature_dim = int(payload["feature_dim"])
    source_drug_ids = list(payload["drug_ids"])
    source_features = payload["features"].float()
    feature_table = torch.zeros((len(artifacts.drug_vocab), feature_dim), dtype=torch.float32)
    source_index = {drug_id: idx for idx, drug_id in enumerate(source_drug_ids)}
    mapped = 0
    missing: list[str] = []
    for drug_id, vocab_index in artifacts.drug_vocab.items():
        if drug_id in {"[PAD]", "[UNK]"}:
            continue
        feature_idx = source_index.get(drug_id)
        if feature_idx is None:
            missing.append(drug_id)
            continue
        feature_table[vocab_index] = source_features[feature_idx]
        mapped += 1
    meta = {
        "model_name": payload.get("model_name", "unknown"),
        "feature_dim": feature_dim,
        "mapped_drugs": mapped,
        "missing_drugs": len(missing),
        "missing_drugs_preview": missing[:50],
    }
    return feature_table, meta


def build_or_load_sample_splits(
    rows: list[dict],
    split_dir: Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.9,
    split_prefix: str = "default",
) -> tuple[list[int], list[int], Path, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / f"train_keys.{split_prefix}.seed{seed}.json"
    val_path = split_dir / f"val_keys.{split_prefix}.seed{seed}.json"

    if train_path.exists() and val_path.exists():
        train_keys = {
            (item["center_drug_id"], item["side_effect_id"], int(item.get("view_id", 0)))
            for item in json.loads(train_path.read_text(encoding="utf-8"))
        }
        val_keys = {
            (item["center_drug_id"], item["side_effect_id"], int(item.get("view_id", 0)))
            for item in json.loads(val_path.read_text(encoding="utf-8"))
        }
    else:
        sample_keys = [(row["center_drug_id"], row["side_effect_id"], int(row.get("view_id", 0))) for row in rows]
        shuffled = list(sample_keys)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        train_count = int(len(shuffled) * train_ratio)
        train_keys = set(shuffled[:train_count])
        val_keys = set(shuffled[train_count:])
        train_payload = [{"center_drug_id": key[0], "side_effect_id": key[1], "view_id": int(key[2])} for key in sorted(train_keys)]
        val_payload = [{"center_drug_id": key[0], "side_effect_id": key[1], "view_id": int(key[2])} for key in sorted(val_keys)]
        train_path.write_text(json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        val_path.write_text(json.dumps(val_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    train_indices = [
        index
        for index, row in enumerate(rows)
        if (row["center_drug_id"], row["side_effect_id"], int(row.get("view_id", 0))) in train_keys
    ]
    val_indices = [
        index
        for index, row in enumerate(rows)
        if (row["center_drug_id"], row["side_effect_id"], int(row.get("view_id", 0))) in val_keys
    ]
    return train_indices, val_indices, train_path, val_path


class HoddiPretrainDataset:
    def __init__(self, rows: list[dict], indices: list[int]) -> None:
        self.rows = rows
        self.indices = indices
        self.lengths = [int(rows[index]["sequence_length"]) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        row = dict(self.rows[self.indices[item]])
        row["sample_idx"] = int(self.indices[item])
        return row
