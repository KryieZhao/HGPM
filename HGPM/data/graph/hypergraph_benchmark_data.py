from __future__ import annotations

import gzip
import json
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from HGPM.utils.io import ensure_dir, save_json

EXIST_ABSENT = 0
EXIST_PRESENT = 1
EXIST_UNKNOWN = 2

COMP_PAD = 0
COMP_ROOT = 1
COMP_COMP = 2
COMP_EMER = 3
COMP_INHIB = 4
COMP_UNKNOWN = 5

TOKEN_PAD = 0
TOKEN_CANDIDATE = 1
TOKEN_SUBSET = 2


@dataclass
class BenchmarkArtifacts:
    dataset_name: str
    node_vocab: dict[str, int]
    label_vocab: dict[str, int]
    side_effect_vocab: dict[str, int]
    split_nodes: dict[str, list[str]]
    num_hyperedges: int
    max_hyperedge_size: int
    feature_dim: int = 0
    data_source: str = "raw_text"


RAW_DATASET_PATHS = {
    "congress": Path("processed/hypergraph_benchmarks/raw/congress-bills.txt"),
    "house": Path("processed/hypergraph_benchmarks/raw/house-committees.txt"),
    "senate": Path("processed/hypergraph_benchmarks/raw/senate-committees.txt"),
    "cora": Path("processed/hypergraph_benchmarks/raw/cora-coauthorship.txt.gz"),
    "dblp": Path("processed/hypergraph_benchmarks/raw/dblp-coauthorship.txt.gz"),
    "walmart": Path("processed/hypergraph_benchmarks/raw/walmart-trips.txt"),
}

STANDARD_DATASET_DIRS = {
    "cora_ca": Path("processed/hypergraph_benchmark_standard/cora"),
    "dblp_ca": Path("processed/hypergraph_benchmark_standard/dblp"),
    "citeseer": Path("processed/hypergraph_benchmark_standard/citeseer"),
    "pubmed": Path("processed/hypergraph_benchmark_standard/pubmed"),
}


def _opener(path: Path):
    return gzip.open if path.suffix == ".gz" else open


def parse_raw_benchmark(path: Path) -> tuple[dict[str, dict], list[tuple[str, ...]]]:
    opener = _opener(path)
    node_meta: dict[str, dict] = {}
    hyperedges: list[tuple[str, ...]] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        next(handle)
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if " " in text:
                left, right = text.split(" ", 1)
            else:
                left, right = text, "{}"
            payload = json.loads(right) if right.startswith("{") else {}
            if payload and "," not in left:
                node_meta[str(int(left))] = payload
                continue
            members = tuple(sorted({part.strip() for part in left.split(",") if part.strip()}))
            if members:
                hyperedges.append(members)
    return node_meta, hyperedges


def _label_from_meta(meta: dict) -> str:
    for key in ("category", "party", "label", "class"):
        value = meta.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _build_vocab(values: list[str]) -> dict[str, int]:
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for value in values:
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def _edge_key(nodes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(nodes)))


def _comp_from_relation(child_exists: bool, current_exists: bool) -> int:
    if child_exists and current_exists:
        return COMP_COMP
    if (not child_exists) and current_exists:
        return COMP_EMER
    if child_exists and (not current_exists):
        return COMP_INHIB
    return COMP_UNKNOWN


def _candidate_token(edge_nodes: tuple[str, ...]) -> dict:
    return {
        "subset_drug_ids": list(edge_nodes),
        "order_id": len(edge_nodes),
        "exist_id": EXIST_UNKNOWN,
        "comp_ids": [COMP_ROOT],
        "token_type_id": TOKEN_CANDIDATE,
    }


def _truncate_edge_nodes(edge_nodes: tuple[str, ...], anchor: str, max_members_per_view: int | None) -> tuple[str, ...]:
    ordered_nodes = tuple(sorted(edge_nodes))
    if max_members_per_view is None or max_members_per_view <= 0 or len(ordered_nodes) <= max_members_per_view:
        return ordered_nodes
    others = [node for node in ordered_nodes if node != anchor]
    keep_count = max(max_members_per_view - 1, 0)
    truncated = tuple(sorted((anchor, *others[:keep_count])))
    return truncated if truncated else (anchor,)


def _enumerate_anchor_tokens(
    *,
    edge_nodes: tuple[str, ...],
    anchor: str,
    existence_index: set[tuple[str, ...]],
    max_token_order: int,
) -> list[dict]:
    edge_nodes = tuple(sorted(edge_nodes))
    tokens: list[dict] = [_candidate_token(edge_nodes)]
    others = [node for node in edge_nodes if node != anchor]

    if len(edge_nodes) >= 2:
        for other in sorted(others):
            subset = tuple(sorted((anchor, other)))
            exist_id = EXIST_PRESENT if _edge_key(subset) in existence_index else EXIST_ABSENT
            tokens.append(
                {
                    "subset_drug_ids": list(subset),
                    "order_id": 2,
                    "exist_id": exist_id,
                    "comp_ids": [COMP_ROOT],
                    "token_type_id": TOKEN_SUBSET,
                }
            )

    for order in range(3, min(max_token_order, len(edge_nodes)) + 1):
        choose_k = order - 1
        for combo in combinations(sorted(others), choose_k):
            subset = tuple(sorted((anchor, *combo)))
            current_exists = _edge_key(subset) in existence_index
            child_sets = [tuple(sorted((anchor, *child))) for child in combinations(combo, choose_k - 1)]
            comp_ids = [_comp_from_relation(_edge_key(child) in existence_index, current_exists) for child in child_sets]
            tokens.append(
                {
                    "subset_drug_ids": list(subset),
                    "order_id": order,
                    "exist_id": EXIST_PRESENT if current_exists else EXIST_ABSENT,
                    "comp_ids": comp_ids,
                    "token_type_id": TOKEN_SUBSET,
                }
            )

    ordered_subsets = sorted(tokens[1:], key=lambda item: (item["order_id"], item["subset_drug_ids"]))
    return [tokens[0], *ordered_subsets]


def _serialize_pretrain_sample(
    *,
    edge_nodes: tuple[str, ...],
    anchor: str,
    existence_index: set[tuple[str, ...]],
    max_token_order: int,
    max_members_per_view: int | None = None,
) -> dict:
    truncated_edge_nodes = _truncate_edge_nodes(edge_nodes, anchor, max_members_per_view)
    return {
        "anchor_node": anchor,
        "edge_nodes": list(truncated_edge_nodes),
        "tokens": _enumerate_anchor_tokens(
            edge_nodes=truncated_edge_nodes,
            anchor=anchor,
            existence_index=existence_index,
            max_token_order=max_token_order,
        ),
    }


def build_anchor_pretrain_sample(
    *,
    edge_nodes: tuple[str, ...],
    anchor: str,
    existence_index: set[tuple[str, ...]],
    max_token_order: int,
    max_members_per_view: int | None = None,
) -> dict:
    return _serialize_pretrain_sample(
        edge_nodes=edge_nodes,
        anchor=anchor,
        existence_index=existence_index,
        max_token_order=max_token_order,
        max_members_per_view=max_members_per_view,
    )


def _serialize_node_classification_sample(
    *,
    target_node: str,
    incident_edges: list[tuple[str, ...]],
    label_id: int,
    existence_index: set[tuple[str, ...]],
    max_token_order: int,
    max_views_per_node: int,
    max_members_per_view: int | None = None,
) -> dict:
    ordered_edges = sorted(incident_edges, key=lambda edge: (-len(edge), edge))
    views = []
    selected_edges = ordered_edges[:max_views_per_node] if ordered_edges else [(target_node,)]
    for edge_nodes in selected_edges:
        views.append(
            _serialize_pretrain_sample(
                edge_nodes=edge_nodes,
                anchor=target_node,
                existence_index=existence_index,
                max_token_order=max_token_order,
                max_members_per_view=max_members_per_view,
            )
        )
    return {
        "target_node": target_node,
        "label_id": int(label_id),
        "views": views,
        "degree": len(incident_edges),
    }


def build_node_classification_sample(
    *,
    target_node: str,
    incident_edges: list[tuple[str, ...]],
    label_id: int,
    existence_index: set[tuple[str, ...]],
    max_token_order: int,
    max_views_per_node: int,
    max_members_per_view: int | None = None,
) -> dict:
    return _serialize_node_classification_sample(
        target_node=target_node,
        incident_edges=incident_edges,
        label_id=label_id,
        existence_index=existence_index,
        max_token_order=max_token_order,
        max_views_per_node=max_views_per_node,
        max_members_per_view=max_members_per_view,
    )


def _split_nodes_by_label(
    *,
    node_meta: dict[str, dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node_id, meta in node_meta.items():
        grouped[_label_from_meta(meta)].append(node_id)

    rng = random.Random(seed)
    split_nodes = {"train": [], "val": [], "test": []}
    label_vocab = _build_vocab(sorted(grouped.keys()))
    for _, nodes in grouped.items():
        nodes = sorted(nodes)
        rng.shuffle(nodes)
        n_total = len(nodes)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else max(0, n_total - n_train - 1)
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1) if n_total >= 3 else 0
        n_test = n_total - n_train - n_val
        if n_test <= 0 and n_total >= 3:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val = max(0, n_val - 1)
        split_nodes["train"].extend(nodes[:n_train])
        split_nodes["val"].extend(nodes[n_train : n_train + n_val])
        split_nodes["test"].extend(nodes[n_train + n_val :])

    for split in split_nodes:
        split_nodes[split] = sorted(split_nodes[split], key=lambda value: int(value))
    return split_nodes, label_vocab


def _prepare_manifest(
    *,
    dataset_name: str,
    data_source: str,
    max_token_order: int,
    max_views_per_node: int,
    max_members_per_view: int | None,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    standard_split_id: int | None,
    standard_split_strategy: str,
) -> dict[str, int | float | str]:
    return {
        "dataset_name": dataset_name,
        "data_source": data_source,
        "max_token_order": int(max_token_order),
        "max_views_per_node": int(max_views_per_node),
        "max_members_per_view": None if max_members_per_view is None else int(max_members_per_view),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "standard_split_id": None if standard_split_id is None else int(standard_split_id),
        "standard_split_strategy": standard_split_strategy,
    }


def _split_train_val_from_indices(
    train_nodes: list[str],
    labels_by_node: dict[str, str],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node_id in train_nodes:
        grouped[labels_by_node[node_id]].append(node_id)

    rng = random.Random(seed)
    final_train: list[str] = []
    final_val: list[str] = []
    for _, nodes in grouped.items():
        nodes = sorted(nodes, key=int)
        rng.shuffle(nodes)
        if len(nodes) <= 1:
            final_train.extend(nodes)
            continue
        n_val = max(1, int(round(len(nodes) * val_ratio)))
        n_val = min(n_val, len(nodes) - 1)
        final_val.extend(nodes[:n_val])
        final_train.extend(nodes[n_val:])
    return sorted(final_train, key=int), sorted(final_val, key=int)


def _load_standard_hypergcn_dataset(
    repo_root: Path,
    *,
    dataset_key: str,
    standard_split_id: int,
    split_strategy: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, dict], list[tuple[str, ...]], dict[str, list[str]], dict[str, int], torch.Tensor]:
    dataset_dir = repo_root / STANDARD_DATASET_DIRS[dataset_key]
    features_path = dataset_dir / "features.pickle"
    hypergraph_path = dataset_dir / "hypergraph.pickle"
    labels_path = dataset_dir / "labels.pickle"
    split_path = dataset_dir / f"split_{standard_split_id}.pickle"
    required_paths = [features_path, hypergraph_path, labels_path]
    if split_strategy == "official_train_test":
        required_paths.append(split_path)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing standard benchmark files: {missing_text}")

    with features_path.open("rb") as handle:
        feature_matrix = pickle.load(handle)
    if hasattr(feature_matrix, "toarray"):
        feature_array = feature_matrix.toarray()
    else:
        feature_array = np.asarray(feature_matrix)
    node_feature_table = torch.tensor(feature_array, dtype=torch.float32)

    with hypergraph_path.open("rb") as handle:
        hypergraph = pickle.load(handle)
    with labels_path.open("rb") as handle:
        labels = pickle.load(handle)
    labels = np.asarray(labels, dtype=np.int64)
    split_payload = None
    if split_path.exists():
        with split_path.open("rb") as handle:
            split_payload = pickle.load(handle)

    num_nodes = int(node_feature_table.size(0))
    if len(labels) != num_nodes:
        raise ValueError(f"Label count {len(labels)} does not match node feature rows {num_nodes}.")

    node_meta = {str(node_idx): {"label": str(int(labels[node_idx]))} for node_idx in range(num_nodes)}
    hyperedges: list[tuple[str, ...]] = []
    for members in hypergraph.values():
        edge = tuple(sorted({str(int(member)) for member in members}, key=int))
        if len(edge) >= 2:
            hyperedges.append(edge)

    label_vocab = _build_vocab(sorted({str(int(label)) for label in labels}, key=int))
    labels_by_node = {node_id: meta["label"] for node_id, meta in node_meta.items()}
    if split_strategy == "official_train_test":
        if split_payload is None:
            raise FileNotFoundError(f"Missing official split file: {split_path}")
        raw_train_nodes = sorted({str(int(node_idx)) for node_idx in split_payload["train"]}, key=int)
        raw_test_nodes = sorted({str(int(node_idx)) for node_idx in split_payload["test"]}, key=int)
        train_nodes, val_nodes = _split_train_val_from_indices(
            raw_train_nodes,
            labels_by_node,
            val_ratio=val_ratio,
            seed=seed,
        )
        split_nodes = {"train": train_nodes, "val": val_nodes, "test": raw_test_nodes}
    elif split_strategy == "random_prop":
        all_nodes = np.arange(num_nodes, dtype=np.int64)
        train_nodes, remaining_nodes = train_test_split(
            all_nodes,
            train_size=train_ratio,
            random_state=seed,
            stratify=labels,
        )
        remaining_ratio = max(1e-8, 1.0 - train_ratio)
        val_share = val_ratio / remaining_ratio
        val_nodes, test_nodes = train_test_split(
            remaining_nodes,
            train_size=val_share,
            random_state=seed,
            stratify=labels[remaining_nodes],
        )
        split_nodes = {
            "train": sorted([str(int(node)) for node in train_nodes], key=int),
            "val": sorted([str(int(node)) for node in val_nodes], key=int),
            "test": sorted([str(int(node)) for node in test_nodes], key=int),
        }
    else:
        raise ValueError(f"Unsupported split_strategy={split_strategy}.")
    return node_meta, hyperedges, split_nodes, label_vocab, node_feature_table


def prepare_benchmark_data(
    repo_root: Path,
    *,
    dataset_name: str,
    data_source: str = "raw_text",
    max_token_order: int,
    max_views_per_node: int,
    max_members_per_view: int | None,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    standard_split_id: int | None = None,
    standard_split_strategy: str = "official_train_test",
    force_rebuild: bool = False,
) -> BenchmarkArtifacts:
    dataset_key = dataset_name.lower()
    processed_dir = repo_root / "processed" / "hypergraph_benchmarks" / dataset_key
    ensure_dir(processed_dir)
    artifact_path = processed_dir / "artifacts.pkl"
    storage_path = processed_dir / "storage.pkl"
    manifest_path = processed_dir / "prepare_manifest.json"
    expected_manifest = _prepare_manifest(
        dataset_name=dataset_key,
        data_source=data_source,
        max_token_order=max_token_order,
        max_views_per_node=max_views_per_node,
        max_members_per_view=max_members_per_view,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        standard_split_id=standard_split_id,
        standard_split_strategy=standard_split_strategy,
    )
    if artifact_path.exists() and storage_path.exists() and manifest_path.exists() and not force_rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest == expected_manifest:
            with artifact_path.open("rb") as handle:
                return pickle.load(handle)

    node_feature_table = None
    if data_source == "raw_text":
        raw_relative = RAW_DATASET_PATHS.get(dataset_key)
        if raw_relative is None:
            raise ValueError(
                f"Unsupported raw_text dataset_name={dataset_name}. Expected one of {sorted(RAW_DATASET_PATHS)}."
            )
        raw_path = repo_root / raw_relative
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset file not found: {raw_path}")
        node_meta, hyperedges = parse_raw_benchmark(raw_path)
        split_nodes, label_vocab = _split_nodes_by_label(
            node_meta=node_meta,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    elif data_source == "hypergcn_standard":
        if dataset_key not in STANDARD_DATASET_DIRS:
            raise ValueError(
                f"Unsupported hypergcn_standard dataset_name={dataset_name}. "
                f"Expected one of {sorted(STANDARD_DATASET_DIRS)}."
            )
        split_id = 1 if standard_split_id is None else int(standard_split_id)
        node_meta, hyperedges, split_nodes, label_vocab, node_feature_table = _load_standard_hypergcn_dataset(
            repo_root,
            dataset_key=dataset_key,
            standard_split_id=split_id,
            split_strategy=standard_split_strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported data_source={data_source}.")

    node_vocab = _build_vocab(sorted(node_meta.keys(), key=lambda value: int(value)))
    side_effect_vocab = {"[PAD]": 0, "[NONE]": 1}
    existence_index = {_edge_key(edge) for edge in hyperedges}
    node_to_edge_indices: dict[str, list[int]] = defaultdict(list)
    for edge_idx, edge in enumerate(hyperedges):
        for node_id in edge:
            node_to_edge_indices[node_id].append(edge_idx)

    storage = {
        "hyperedges": hyperedges,
        "existence_index": existence_index,
        "node_to_edge_indices": dict(node_to_edge_indices),
        "max_token_order": int(max_token_order),
        "max_views_per_node": int(max_views_per_node),
        "max_members_per_view": None if max_members_per_view is None else int(max_members_per_view),
        "feature_dim": 0 if node_feature_table is None else int(node_feature_table.size(1)),
    }
    with storage_path.open("wb") as handle:
        pickle.dump(storage, handle)
    if node_feature_table is not None:
        aligned_feature_table = torch.zeros((len(node_vocab), int(node_feature_table.size(1))), dtype=torch.float32)
        if node_feature_table.numel() > 0:
            aligned_feature_table[1] = node_feature_table.mean(dim=0)
        for node_id, vocab_index in node_vocab.items():
            if node_id in {"[PAD]", "[UNK]"}:
                continue
            aligned_feature_table[vocab_index] = node_feature_table[int(node_id)]
        torch.save(aligned_feature_table, processed_dir / "node_features.pt")

    for split, node_ids in split_nodes.items():
        records = []
        for node_id in node_ids:
            records.append({"target_node": node_id, "label_id": label_vocab[_label_from_meta(node_meta[node_id])]})
        with (processed_dir / f"node_classification_{split}.pkl").open("wb") as handle:
            pickle.dump(records, handle)

    stats = {
        "dataset_name": dataset_key,
        "num_nodes": len(node_meta),
        "num_hyperedges": len(hyperedges),
        "num_pretrain_samples": sum(len(edge) for edge in hyperedges),
        "num_classes": len(label_vocab) - 2,
        "max_hyperedge_size": max(len(edge) for edge in hyperedges) if hyperedges else 0,
        "mean_hyperedge_size": sum(len(edge) for edge in hyperedges) / max(len(hyperedges), 1),
        "feature_dim": 0 if node_feature_table is None else int(node_feature_table.size(1)),
        "data_source": data_source,
        "label_counts": dict(Counter(_label_from_meta(meta) for meta in node_meta.values())),
        "split_counts": {split: len(node_ids) for split, node_ids in split_nodes.items()},
    }
    save_json(processed_dir / "stats.json", stats)

    artifacts = BenchmarkArtifacts(
        dataset_name=dataset_key,
        node_vocab=node_vocab,
        label_vocab=label_vocab,
        side_effect_vocab=side_effect_vocab,
        split_nodes=split_nodes,
        num_hyperedges=len(hyperedges),
        max_hyperedge_size=stats["max_hyperedge_size"],
        feature_dim=stats["feature_dim"],
        data_source=data_source,
    )
    with artifact_path.open("wb") as handle:
        pickle.dump(artifacts, handle)
    save_json(manifest_path, expected_manifest)
    return artifacts
