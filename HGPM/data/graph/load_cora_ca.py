from __future__ import annotations

import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HYPERGCN_COAUTHORSHIP_BASE = "https://raw.githubusercontent.com/malllabiisc/HyperGCN/master/data/coauthorship"
REQUIRED_FILES = ["features.pickle", "hypergraph.pickle", "labels.pickle"]


@dataclass
class CoraCA:
    features: np.ndarray
    labels: np.ndarray
    hyperedges: list[list[int]]
    hyperedge_names: list[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def infer_dataset_name(root_dir: str | Path, dataset_name: str | None = None) -> str:
    if dataset_name is not None:
        normalized = str(dataset_name).strip().lower()
    else:
        normalized = Path(root_dir).name.strip().lower()
    alias_map = {
        "cora": "cora",
        "cora_ca": "cora",
        "dblp": "dblp",
        "dblp_ca": "dblp",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported coauthorship dataset: {dataset_name or root_dir}")
    return alias_map[normalized]


def ensure_cora_files(root_dir: str | Path, dataset_name: str | None = None) -> Path:
    dataset = infer_dataset_name(root_dir, dataset_name)
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        target = root / filename
        if not target.exists():
            _download_file(f"{HYPERGCN_COAUTHORSHIP_BASE}/{dataset}/{filename}", target)
    return root


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def make_per_class_split(
    labels: np.ndarray,
    *,
    train_per_class: int = 20,
    val_per_class: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    for class_id in sorted(np.unique(labels).tolist()):
        class_nodes = np.where(labels == class_id)[0]
        rng.shuffle(class_nodes)
        n_train = min(train_per_class, len(class_nodes))
        n_val = min(val_per_class, max(0, len(class_nodes) - n_train))
        train_ids.extend(class_nodes[:n_train].tolist())
        val_ids.extend(class_nodes[n_train : n_train + n_val].tolist())
        test_ids.extend(class_nodes[n_train + n_val :].tolist())
    return (
        np.asarray(sorted(train_ids), dtype=np.int64),
        np.asarray(sorted(val_ids), dtype=np.int64),
        np.asarray(sorted(test_ids), dtype=np.int64),
    )


def load_cora_ca(
    root_dir: str | Path,
    *,
    dataset_name: str | None = None,
    split_strategy: str = "random_per_class",
    split_id: int = 1,
    train_per_class: int = 20,
    val_per_class: int = 30,
    seed: int = 42,
) -> CoraCA:
    dataset = infer_dataset_name(root_dir, dataset_name)
    root = ensure_cora_files(root_dir, dataset_name=dataset)
    features = _load_pickle(root / "features.pickle")
    labels = np.asarray(_load_pickle(root / "labels.pickle"), dtype=np.int64)
    hypergraph = _load_pickle(root / "hypergraph.pickle")

    feature_array = features.toarray() if hasattr(features, "toarray") else np.asarray(features)
    feature_array = np.asarray(feature_array, dtype=np.float32)
    hyperedge_names = [str(name) for name in hypergraph.keys()]
    hyperedges = [sorted({int(member) for member in members}) for members in hypergraph.values()]

    if split_strategy == "random_per_class":
        split_path = root / f"split_per_class_seed{seed}_tr{train_per_class}_val{val_per_class}.npz"
        if split_path.exists():
            payload = np.load(split_path)
            train_idx, val_idx, test_idx = payload["train"], payload["val"], payload["test"]
        else:
            train_idx, val_idx, test_idx = make_per_class_split(
                labels,
                train_per_class=train_per_class,
                val_per_class=val_per_class,
                seed=seed,
            )
            np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)
    elif split_strategy == "hypergcn_split":
        split_path = root / f"split_{split_id}.pickle"
        if not split_path.exists():
            _download_file(f"{HYPERGCN_COAUTHORSHIP_BASE}/{dataset}/splits/{split_id}.pickle", split_path)
        split_payload = _load_pickle(split_path)
        train_idx = np.asarray([int(idx) for idx in split_payload["train"]], dtype=np.int64)
        test_idx = np.asarray([int(idx) for idx in split_payload["test"]], dtype=np.int64)
        train_idx, val_idx = make_per_class_split(
            labels[train_idx],
            train_per_class=max(1, train_per_class),
            val_per_class=max(1, val_per_class),
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported split_strategy={split_strategy}")

    return CoraCA(
        features=feature_array,
        labels=labels,
        hyperedges=hyperedges,
        hyperedge_names=hyperedge_names,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def dataset_stats(data: CoraCA) -> dict[str, int]:
    return {
        "num_nodes": int(data.features.shape[0]),
        "feature_dim": int(data.features.shape[1]),
        "num_hyperedges": int(len(data.hyperedges)),
        "num_classes": int(len(np.unique(data.labels))),
        "train_size": int(len(data.train_idx)),
        "val_size": int(len(data.val_idx)),
        "test_size": int(len(data.test_idx)),
        "max_hyperedge_size": int(max(len(edge) for edge in data.hyperedges)),
    }
