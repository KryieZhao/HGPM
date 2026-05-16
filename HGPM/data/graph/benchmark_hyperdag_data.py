from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from HGPM.data.drug.pretrain_dataset import load_sequence_rows
from HGPM.data.graph.build_centered_hyperedge_dag import build_center_to_edges, build_tokens_for_center_view
from HGPM.data.graph.hypergraph_benchmark_data import (
    RAW_DATASET_PATHS,
    STANDARD_DATASET_DIRS,
    _build_vocab,
    _label_from_meta,
    _load_standard_hypergcn_dataset,
    _split_nodes_by_label,
    parse_raw_benchmark,
)
from HGPM.task.graph.graph_base import HyperDAGData
from HGPM.utils.io import PACKAGE_ROOT, resolve_repo_path, save_json

REPO_ROOT = PACKAGE_ROOT


@dataclass
class BenchmarkProtocolArtifacts:
    dataset_name: str
    protocol_dir: Path
    split_nodes: dict[str, list[str]]
    label_vocab: dict[str, int]
    node_vocab: dict[str, int]
    hyperedges: list[tuple[str, ...]]
    node_meta: dict[str, dict]


def _node_to_edge_indices(hyperedges: list[tuple[str, ...]]) -> dict[str, list[int]]:
    node_to_edges: dict[str, list[int]] = defaultdict(list)
    for edge_idx, edge in enumerate(hyperedges):
        for node_id in edge:
            node_to_edges[node_id].append(edge_idx)
    return dict(node_to_edges)


def _one_hot_feature_payload(node_ids: list[str]) -> dict:
    num_nodes = len(node_ids)
    features = torch.eye(num_nodes, dtype=torch.float32)
    return _feature_payload(node_ids, features, model_name="identity-onehot")


def _heterophilic_label_feature_payload(
    node_ids: list[str],
    node_meta: dict[str, dict],
    label_vocab: dict[str, int],
    *,
    seed: int,
    feature_noise: float = 1.0,
) -> dict:
    num_nodes = len(node_ids)
    num_classes = max(len(label_vocab) - 2, 1)
    features = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for row_idx, node_id in enumerate(node_ids):
        label_name = _label_from_meta(node_meta[node_id])
        label_idx = int(label_vocab[label_name]) - 2
        if 0 <= label_idx < num_classes:
            features[row_idx, label_idx] = 1.0
    noise = torch.tensor(rng.standard_normal((num_nodes, num_classes)), dtype=torch.float32)
    features = features + float(feature_noise) * noise
    return _feature_payload(node_ids, features, model_name="heterophilic-label-noise")


def _feature_payload(node_ids: list[str], features: torch.Tensor, *, model_name: str) -> dict:
    return {
        "model_name": model_name,
        "feature_dim": int(features.size(1)),
        "drug_ids": node_ids,
        "features": features,
        "smiles": {},
    }


def parse_ntu2012_raw(content_path: Path, edges_path: Path) -> tuple[dict[str, dict], list[tuple[str, ...]], torch.Tensor]:
    node_meta: dict[str, dict] = {}
    feature_rows: list[list[float]] = []
    node_ids: list[str] = []
    with content_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if len(parts) < 3:
                continue
            node_id = str(int(float(parts[0])))
            label = str(int(float(parts[-1])))
            features = [float(value) for value in parts[1:-1]]
            node_meta[node_id] = {"label": label}
            node_ids.append(node_id)
            feature_rows.append(features)

    edge_to_nodes: dict[str, list[str]] = defaultdict(list)
    with edges_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if len(parts) != 2:
                continue
            node_id = str(int(float(parts[0])))
            edge_id = str(int(float(parts[1])))
            edge_to_nodes[edge_id].append(node_id)

    hyperedges = []
    for members in edge_to_nodes.values():
        edge = tuple(sorted({member for member in members}, key=int))
        if len(edge) >= 2:
            hyperedges.append(edge)

    ordered = sorted(zip(node_ids, feature_rows), key=lambda item: int(item[0]))
    ordered_features = torch.tensor([row for _, row in ordered], dtype=torch.float32)
    return node_meta, hyperedges, ordered_features


def sample_observed_substructures_for_view(
    *,
    center: str,
    edge: tuple[str, ...],
    max_order: int,
    max_substructures_per_edge: int,
    rng,
) -> dict[tuple[str, ...], str]:
    """Sample center-preserving observed substructures from large hyperedges.

    Raw benchmark datasets like House contain very large hyperedges. If we drop
    edges larger than `max_order`, most supervision disappears. Instead, we
    convert each oversized hyperedge into one or more nested observed
    substructure chains anchored at the center node.
    """
    if center not in edge or len(edge) < 2:
        return {}
    others = [node for node in edge if node != center]
    if not others:
        return {}

    retained: dict[tuple[str, ...], str] = {}
    max_chain_order = min(max_order, len(edge))
    max_chain_length = max(0, max_chain_order - 1)
    if max_chain_length == 0:
        return retained

    if len(edge) <= max_order:
        full_edge = tuple(sorted(edge, key=int))
        retained[full_edge] = "observed"
    max_substructures_per_edge = max(1, int(max_substructures_per_edge))
    num_chains = max(1, int(np.ceil(max_substructures_per_edge / max(max_chain_length, 1))))
    chain_budget = max(1, max_substructures_per_edge // num_chains)

    for _ in range(num_chains):
        sampled = list(others)
        rng.shuffle(sampled)
        sampled = sampled[:max_chain_length]
        for order_minus_one in range(1, min(len(sampled), max_chain_length) + 1):
            subset = tuple(sorted((center, *sampled[:order_minus_one]), key=int))
            if len(edge) <= max_order and len(subset) == len(edge):
                retained[subset] = "observed"
            else:
                retained.setdefault(subset, "observed_substructure")
            if len(retained) >= max_substructures_per_edge:
                break
        if len(retained) >= max_substructures_per_edge:
            break
        if chain_budget <= 1:
            continue
    return retained


def prepare_raw_benchmark_protocol(
    *,
    dataset_name: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    protocol_dir: Path,
    force_rebuild: bool = False,
) -> BenchmarkProtocolArtifacts:
    protocol_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = protocol_dir / "protocol_manifest.json"
    stats_path = protocol_dir / "stats.json"
    node_meta_path = protocol_dir / "node_meta.pkl"
    hyperedges_path = protocol_dir / "hyperedges.pkl"
    node_vocab_path = protocol_dir / "drug2id.json"
    label_vocab_path = protocol_dir / "label_vocab.json"
    feature_path = protocol_dir / "paper_node_features.pt"

    dataset_key = str(dataset_name).strip().lower()
    expected_manifest = {
        "dataset_name": dataset_name,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
    }
    if dataset_key in STANDARD_DATASET_DIRS:
        expected_manifest["data_source"] = "hypergcn_standard"
        expected_manifest["feature_type"] = "hypergcn_standard"
    elif dataset_key == "ntu2012":
        expected_manifest["data_source"] = "ntu2012_raw"
        expected_manifest["feature_type"] = "raw_features"
    else:
        expected_manifest["data_source"] = "raw_text"
        expected_manifest["feature_type"] = "identity-onehot"

    if (
        not force_rebuild
        and manifest_path.exists()
        and stats_path.exists()
        and node_meta_path.exists()
        and hyperedges_path.exists()
        and node_vocab_path.exists()
        and label_vocab_path.exists()
        and feature_path.exists()
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest == expected_manifest:
            split_nodes = {}
            for split in ("train", "val", "test"):
                records = pickle.load((protocol_dir / f"node_classification_{split}.pkl").open("rb"))
                split_nodes[split] = [str(record["target_node"]) for record in records]
            return BenchmarkProtocolArtifacts(
                dataset_name=dataset_name,
                protocol_dir=protocol_dir,
                split_nodes=split_nodes,
                label_vocab=json.loads(label_vocab_path.read_text(encoding="utf-8")),
                node_vocab=json.loads(node_vocab_path.read_text(encoding="utf-8")),
                hyperedges=pickle.load(hyperedges_path.open("rb")),
                node_meta=pickle.load(node_meta_path.open("rb")),
            )

    if dataset_key in STANDARD_DATASET_DIRS:
        node_meta, hyperedges, split_nodes, label_vocab, node_feature_table = _load_standard_hypergcn_dataset(
            resolve_repo_path("processed").parent,
            dataset_key=dataset_key,
            standard_split_id=1,
            split_strategy="random_prop",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        node_ids = sorted(node_meta.keys(), key=lambda value: int(value))
        feature_payload = _feature_payload(node_ids, node_feature_table.float(), model_name="hypergcn-standard")
    elif dataset_key == "ntu2012":
        content_path = resolve_repo_path("processed/hypergraph_benchmarks/raw/NTU2012.content")
        edges_path = resolve_repo_path("processed/hypergraph_benchmarks/raw/NTU2012.edges")
        if not content_path.exists() or not edges_path.exists():
            raise FileNotFoundError(f"Missing NTU2012 raw files: {content_path}, {edges_path}")
        node_meta, hyperedges, feature_table = parse_ntu2012_raw(content_path, edges_path)
        split_nodes, label_vocab = _split_nodes_by_label(
            node_meta=node_meta,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        node_ids = sorted(node_meta.keys(), key=lambda value: int(value))
        feature_payload = _feature_payload(node_ids, feature_table.float(), model_name="ntu2012-raw")
    else:
        raw_relative = RAW_DATASET_PATHS.get(dataset_key)
        if raw_relative is None:
            raise ValueError(f"Unsupported raw benchmark dataset={dataset_name}.")
        raw_path = resolve_repo_path(raw_relative)
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw benchmark file: {raw_path}")
        node_meta, hyperedges = parse_raw_benchmark(raw_path)
        split_nodes, label_vocab = _split_nodes_by_label(
            node_meta=node_meta,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        node_ids = sorted(node_meta.keys(), key=lambda value: int(value))
        feature_payload = _heterophilic_label_feature_payload(
            node_ids,
            node_meta,
            label_vocab,
            seed=seed,
            feature_noise=1.0,
        )

    node_vocab = _build_vocab(sorted(node_meta.keys(), key=lambda value: int(value)))

    for split, split_node_ids in split_nodes.items():
        records = [
            {"target_node": node_id, "label_id": int(label_vocab[_label_from_meta(node_meta[node_id])])}
            for node_id in split_node_ids
        ]
        with (protocol_dir / f"node_classification_{split}.pkl").open("wb") as handle:
            pickle.dump(records, handle)

    with hyperedges_path.open("wb") as handle:
        pickle.dump([tuple(edge) for edge in hyperedges], handle)
    with node_meta_path.open("wb") as handle:
        pickle.dump(node_meta, handle)
    save_json(node_vocab_path, node_vocab)
    save_json(label_vocab_path, label_vocab)
    torch.save(feature_payload, feature_path)

    stats = {
        "dataset_name": dataset_name,
        "num_nodes": len(node_meta),
        "num_hyperedges": len(hyperedges),
        "num_classes": int(len(label_vocab) - 2),
        "max_hyperedge_size": max(len(edge) for edge in hyperedges) if hyperedges else 0,
        "mean_hyperedge_size": float(sum(len(edge) for edge in hyperedges) / max(len(hyperedges), 1)),
        "split_counts": {split: len(node_ids) for split, node_ids in split_nodes.items()},
        "feature_dim": int(feature_payload["feature_dim"]),
    }
    save_json(stats_path, stats)
    save_json(manifest_path, expected_manifest)

    return BenchmarkProtocolArtifacts(
        dataset_name=dataset_name,
        protocol_dir=protocol_dir,
        split_nodes=split_nodes,
        label_vocab=label_vocab,
        node_vocab=node_vocab,
        hyperedges=[tuple(edge) for edge in hyperedges],
        node_meta=node_meta,
    )


def build_benchmark_centered_hyperedge_dag(
    *,
    dataset_name: str,
    protocol_dir: Path,
    output_dir: Path,
    seed: int,
    max_order: int,
    top_k_per_order: int,
    k_views: int,
    negatives_per_positive: int,
    observed_heavy: bool,
    sampled_budget_per_order: int,
    max_substructures_per_edge: int = 16,
) -> dict:
    artifacts = prepare_raw_benchmark_protocol(
        dataset_name=dataset_name,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=seed,
        protocol_dir=protocol_dir,
        force_rebuild=False,
    )
    node_ids = sorted(artifacts.node_meta.keys(), key=lambda value: int(value))
    hyperedges = [tuple(sorted(dict.fromkeys(edge), key=lambda value: int(value))) for edge in artifacts.hyperedges]
    center_to_edges = build_center_to_edges([[int(node_id) for node_id in edge] for edge in hyperedges])
    observed_set = set(hyperedges)

    rows = []
    lengths = []
    empty_centers = 0
    for center in node_ids:
        incident_edges = center_to_edges.get(center, [])
        if not incident_edges:
            empty_centers += 1
        for view_id in range(k_views):
            positive_source_map: dict[tuple[str, ...], str] = {}
            rng = __import__("random").Random(seed * 1000003 + int(center) * 1009 + view_id)
            for edge in incident_edges:
                positive_source_map.update(
                    sample_observed_substructures_for_view(
                        center=center,
                        edge=edge,
                        max_order=max_order,
                        max_substructures_per_edge=max_substructures_per_edge,
                        rng=rng,
                    )
                )
            positive_edges = sorted(positive_source_map.keys(), key=lambda item: (len(item), item), reverse=True)
            observed_set_view = observed_set.union(positive_source_map.keys())
            closure_cache: dict[tuple[str, ...], dict[tuple[str, ...], int]] = {}
            tokens = build_tokens_for_center_view(
                center=center,
                positive_edges=positive_edges,
                observed_set=observed_set_view,
                all_nodes=node_ids,
                max_order=max_order,
                top_k_per_order=top_k_per_order,
                negatives_per_positive=negatives_per_positive,
                observed_heavy=observed_heavy,
                sampled_budget_per_order=sampled_budget_per_order,
                expand_intermediate_subsets=False,
                min_observed_parents_per_observed=0,
                max_observed_parents_per_observed=None,
                prefer_high_order_negatives=False,
                intermediate_parents_per_child=None,
                observed_only_composition=False,
                enforce_topdown_closure=False,
                positive_source_map=positive_source_map,
                closure_cache=closure_cache,
                rng=rng,
            )
            if not tokens:
                continue
            rows.append(
                {
                    "center_drug_id": center,
                    "side_effect_id": "NONE",
                    "side_effect_name": "NONE",
                    "view_id": view_id,
                    "sequence_length": len(tokens),
                    "tokens": tokens,
                }
            )
            lengths.append(len(tokens))

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = output_dir / f"{dataset_name}_centered_hyperedge_dag.max_order{max_order}.topk{top_k_per_order}.views{k_views}.jsonl"
    with sequence_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_json(output_dir / "drug2id.json", artifacts.node_vocab)
    torch.save(torch.load(protocol_dir / "paper_node_features.pt", map_location="cpu"), output_dir / "paper_node_features.pt")

    stats = {
        "dataset_name": dataset_name,
        "protocol_dir": str(protocol_dir),
        "num_nodes": len(node_ids),
        "num_hyperedges": len(hyperedges),
        "num_rows": len(rows),
        "num_centers_with_edges": len(center_to_edges),
        "empty_centers": empty_centers,
        "max_order": max_order,
        "top_k_per_order": top_k_per_order,
        "k_views": k_views,
        "negatives_per_positive": negatives_per_positive,
        "observed_heavy": observed_heavy,
        "sampled_budget_per_order": sampled_budget_per_order,
        "max_substructures_per_edge": int(max_substructures_per_edge),
        "mean_sequence_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_sequence_length": int(max(lengths)) if lengths else 0,
        "sequence_path": str(sequence_path),
        "drug2id_path": str(output_dir / "drug2id.json"),
        "feature_path": str(output_dir / "paper_node_features.pt"),
        "split_counts": {split: len(nodes) for split, nodes in artifacts.split_nodes.items()},
    }
    save_json(output_dir / f"{dataset_name}_centered_hyperedge_dag.stats.json", stats)
    save_json(output_dir / f"{dataset_name}_centered_hyperedge_dag.examples.json", rows[:3])
    return stats


def load_benchmark_hyperdag_data(config: dict) -> tuple[HyperDAGData, torch.Tensor]:
    data_cfg = config["data"]
    protocol_dir = resolve_repo_path(data_cfg["protocol_dir"])
    sequence_path = resolve_repo_path(data_cfg["sequence_path"]) if data_cfg.get("sequence_path") else None
    rows_by_center: dict[str, list[dict]] | None = None
    available_centers: set[str] | None = None
    row_builder = None
    if sequence_path is not None and sequence_path.exists():
        rows = load_sequence_rows(sequence_path)
        rows_by_center = {}
        for row in rows:
            rows_by_center.setdefault(str(row["center_drug_id"]), []).append(row)
        available_centers = set(rows_by_center.keys())
    else:
        hyperedges = pickle.load((protocol_dir / "hyperedges.pkl").open("rb"))
        hyperedges = [tuple(sorted(dict.fromkeys(str(node_id) for node_id in edge), key=int)) for edge in hyperedges]
        center_to_edges = build_center_to_edges([[int(node_id) for node_id in edge] for edge in hyperedges])
        observed_set = set(hyperedges)
        all_nodes = sorted({node_id for edge in hyperedges for node_id in edge}, key=int)
        dag_cfg = config.get("dag", {})
        max_order = int(dag_cfg.get("max_order", data_cfg.get("max_order", 8)))
        top_k_per_order = int(dag_cfg.get("top_k_per_order", data_cfg.get("top_k_per_order", 8)))
        negatives_per_positive = int(dag_cfg.get("negatives_per_positive", data_cfg.get("negatives_per_positive", 1)))
        observed_heavy = bool(dag_cfg.get("observed_heavy", True))
        sampled_budget_per_order = int(dag_cfg.get("sampled_budget_per_order", data_cfg.get("sampled_budget_per_order", 1)))
        max_substructures_per_edge = int(dag_cfg.get("max_substructures_per_edge", data_cfg.get("max_substructures_per_edge", 8)))
        base_seed = int(config["seed"])
        available_centers = set(center_to_edges.keys())

        @lru_cache(maxsize=200000)
        def _build_rows_for_center_cached(center_id: str, k_views: int) -> tuple[dict, ...]:
            incident_edges = center_to_edges.get(center_id, [])
            if not incident_edges:
                return tuple()
            built_rows: list[dict] = []
            for view_id in range(int(k_views)):
                rng = __import__("random").Random(base_seed * 1000003 + int(center_id) * 1009 + view_id)
                positive_source_map: dict[tuple[str, ...], str] = {}
                for edge in incident_edges:
                    positive_source_map.update(
                        sample_observed_substructures_for_view(
                            center=center_id,
                            edge=edge,
                            max_order=max_order,
                            max_substructures_per_edge=max_substructures_per_edge,
                            rng=rng,
                        )
                    )
                positive_edges = sorted(positive_source_map.keys(), key=lambda item: (len(item), item), reverse=True)
                closure_cache: dict[tuple[str, ...], dict[tuple[str, ...], int]] = {}
                tokens = build_tokens_for_center_view(
                    center=center_id,
                    positive_edges=positive_edges,
                    observed_set=observed_set.union(positive_source_map.keys()),
                    all_nodes=all_nodes,
                    max_order=max_order,
                    top_k_per_order=top_k_per_order,
                    negatives_per_positive=negatives_per_positive,
                    observed_heavy=observed_heavy,
                    sampled_budget_per_order=sampled_budget_per_order,
                    expand_intermediate_subsets=False,
                    min_observed_parents_per_observed=0,
                    max_observed_parents_per_observed=None,
                    prefer_high_order_negatives=False,
                    intermediate_parents_per_child=None,
                    observed_only_composition=False,
                    enforce_topdown_closure=False,
                    positive_source_map=positive_source_map,
                    closure_cache=closure_cache,
                    rng=rng,
                )
                if not tokens:
                    continue
                built_rows.append(
                    {
                        "center_drug_id": center_id,
                        "side_effect_id": "NONE",
                        "side_effect_name": "NONE",
                        "view_id": view_id,
                        "sequence_length": len(tokens),
                        "tokens": tokens,
                    }
                )
            return tuple(built_rows)

        def _build_rows_for_center(center_id: str, k_views: int) -> list[dict]:
            return list(_build_rows_for_center_cached(center_id, int(k_views)))

        row_builder = _build_rows_for_center

    train_records = pickle.load((protocol_dir / "node_classification_train.pkl").open("rb"))
    val_records = pickle.load((protocol_dir / "node_classification_val.pkl").open("rb"))
    test_records = pickle.load((protocol_dir / "node_classification_test.pkl").open("rb"))
    all_records = train_records + val_records + test_records

    max_node_id = max(int(record["target_node"]) for record in all_records)
    labels = np.full(max_node_id + 1, -1, dtype=np.int64)
    for record in all_records:
        labels[int(record["target_node"])] = int(record["label_id"]) - 2

    train_idx = np.asarray(sorted(int(record["target_node"]) for record in train_records), dtype=np.int64)
    val_idx = np.asarray(sorted(int(record["target_node"]) for record in val_records), dtype=np.int64)
    test_idx = np.asarray(sorted(int(record["target_node"]) for record in test_records), dtype=np.int64)

    drug_vocab = json.loads(resolve_repo_path(data_cfg["drug2id_path"]).read_text(encoding="utf-8"))
    feature_payload = torch.load(resolve_repo_path(data_cfg["node_feature_path"]), map_location="cpu")
    feature_table = feature_payload["features"].float()

    hyperdag = HyperDAGData(
        rows_by_center=rows_by_center,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_classes=int(len(sorted(set(labels[labels >= 0].tolist())))),
        drug_vocab=drug_vocab,
        available_centers=available_centers,
        row_builder=row_builder,
    )
    return hyperdag, feature_table
