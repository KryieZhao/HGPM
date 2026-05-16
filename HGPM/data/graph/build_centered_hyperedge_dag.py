from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from HGPM.data.graph.load_cora_ca import load_cora_ca
from HGPM.utils.io import ensure_dir, save_json
from HGPM.utils.seed import set_seed


COMP_LABELS = ("COMP", "EMER", "INHIB", "UNKNOWN")


def subset_to_str(subset: tuple[str, ...]) -> str:
    return "|".join(subset)


def comp_label(parent_exist: int, child_exist: int) -> str:
    if parent_exist == 1 and child_exist == 1:
        return "COMP"
    if parent_exist == 1 and child_exist == 0:
        return "EMER"
    if parent_exist == 0 and child_exist == 1:
        return "INHIB"
    return "UNKNOWN"


def build_center_to_edges(hyperedges: list[list[int]]) -> dict[str, list[tuple[str, ...]]]:
    center_to_edges: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    seen = set()
    for members in hyperedges:
        subset = tuple(sorted({str(int(node_id)) for node_id in members}, key=int))
        if len(subset) < 2 or subset in seen:
            continue
        seen.add(subset)
        for center in subset:
            center_to_edges[center].append(subset)
    return center_to_edges


def sample_negative_subsets(
    *,
    center: str,
    positive_edges: list[tuple[str, ...]],
    observed_set: set[tuple[str, ...]],
    all_nodes: list[str],
    max_order: int,
    negatives_per_positive: int,
    rng: random.Random,
) -> set[tuple[str, ...]]:
    negatives: set[tuple[str, ...]] = set()
    all_node_set = set(all_nodes)
    for edge in positive_edges:
        edge_set = set(edge)
        if len(edge) > 2:
            removable = [node for node in edge if node != center]
            removed = rng.choice(removable)
            candidate = tuple(sorted(edge_set - {removed}, key=int))
            if center in candidate and candidate not in observed_set:
                negatives.add(candidate)
        if len(edge) < max_order:
            pool = sorted(all_node_set - edge_set, key=int)
            if pool:
                added = rng.choice(pool)
                candidate = tuple(sorted((*edge, added), key=int))
                if center in candidate and candidate not in observed_set:
                    negatives.add(candidate)
        replaceable = [node for node in edge if node != center]
        pool = sorted(all_node_set - edge_set, key=int)
        for _ in range(max(0, negatives_per_positive)):
            if not replaceable or not pool:
                break
            removed = rng.choice(replaceable)
            added = rng.choice(pool)
            candidate = tuple(sorted((edge_set - {removed}) | {added}, key=int))
            if len(candidate) >= 2 and len(candidate) <= max_order and center in candidate and candidate not in observed_set:
                negatives.add(candidate)
    return negatives


def observed_immediate_subsets(
    subset: tuple[str, ...],
    center: str,
    observed_set: set[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    return [child for child in immediate_subsets(subset, center) if child == (center,) or child in observed_set]


def immediate_subsets(subset: tuple[str, ...], center: str) -> list[tuple[str, ...]]:
    if len(subset) == 2:
        return [(center,)]
    if len(subset) < 3:
        return []
    children = []
    for removed in subset:
        if removed == center:
            continue
        child = tuple(sorted((node for node in subset if node != removed), key=int))
        if center in child:
            children.append(child)
    return children


def immediate_supersets(
    subset: tuple[str, ...],
    center: str,
    retained_subsets: set[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    if len(subset) < 1:
        return []
    supersets = []
    subset_set = set(subset)
    target_size = len(subset) + 1
    for candidate in retained_subsets:
        if len(candidate) != target_size or center not in candidate:
            continue
        if subset_set.issubset(candidate):
            supersets.append(candidate)
    return sorted(supersets, key=lambda item: (len(item), item), reverse=True)


def add_sampled_intermediates(
    *,
    center: str,
    candidates: dict[tuple[str, ...], int],
    max_order: int,
    max_parents_per_child: int | None = None,
) -> None:
    for subset, exist in list(candidates.items()):
        if len(subset) < 2 or len(subset) > max_order:
            continue
        children = immediate_subsets(subset, center)
        if max_parents_per_child is not None:
            children = children[: max(0, max_parents_per_child)]
        for child in children:
            if child not in candidates:
                candidates[child] = 0


def collect_topdown_closure(
    *,
    center: str,
    subset: tuple[str, ...],
    observed_set: set[tuple[str, ...]],
    closure_cache: dict[tuple[str, ...], dict[tuple[str, ...], int]],
) -> dict[tuple[str, ...], int]:
    cached = closure_cache.get(subset)
    if cached is not None:
        return cached

    closure: dict[tuple[str, ...], int] = {}
    if subset == (center,):
        closure[(center,)] = 1
        closure_cache[subset] = closure
        return closure

    for child in immediate_subsets(subset, center):
        child_exist = 1 if child == (center,) or child in observed_set else 0
        previous = closure.get(child)
        if previous is None or child_exist > previous:
            closure[child] = child_exist
        child_closure = collect_topdown_closure(
            center=center,
            subset=child,
            observed_set=observed_set,
            closure_cache=closure_cache,
        )
        for nested_subset, nested_exist in child_closure.items():
            previous = closure.get(nested_subset)
            if previous is None or nested_exist > previous:
                closure[nested_subset] = nested_exist

    closure_cache[subset] = closure
    return closure


def token_source(
    subset: tuple[str, ...],
    center: str,
    retained: dict[tuple[str, ...], int],
    positive_source_map: dict[tuple[str, ...], str] | None = None,
) -> str:
    if subset == (center,):
        return "center_node"
    if positive_source_map is not None and subset in positive_source_map and int(retained[subset]) == 1:
        return str(positive_source_map[subset])
    return "observed" if int(retained[subset]) == 1 else "sampled_negative"


def build_tokens_for_center_view(
    *,
    center: str,
    positive_edges: list[tuple[str, ...]],
    observed_set: set[tuple[str, ...]],
    all_nodes: list[str],
    max_order: int,
    top_k_per_order: int,
    negatives_per_positive: int,
    observed_heavy: bool = False,
    sampled_budget_per_order: int = 2,
    expand_intermediate_subsets: bool = False,
    min_observed_parents_per_observed: int = 0,
    max_observed_parents_per_observed: int | None = None,
    prefer_high_order_negatives: bool = False,
    intermediate_parents_per_child: int | None = None,
    observed_only_composition: bool = False,
    enforce_topdown_closure: bool = False,
    positive_source_map: dict[tuple[str, ...], str] | None = None,
    closure_cache: dict[tuple[str, ...], dict[tuple[str, ...], int]] | None = None,
    rng: random.Random,
) -> list[dict]:
    positives = [edge for edge in positive_edges if 2 <= len(edge) <= max_order]
    negatives = sample_negative_subsets(
        center=center,
        positive_edges=positives,
        observed_set=observed_set,
        all_nodes=all_nodes,
        max_order=max_order,
        negatives_per_positive=negatives_per_positive,
        rng=rng,
    )
    candidates: dict[tuple[str, ...], int] = {(center,): 1}
    candidates.update({edge: 1 for edge in positives})
    for subset in negatives:
        candidates.setdefault(subset, 0)
    by_order: dict[int, list[tuple[str, ...]]] = defaultdict(list)
    for subset in candidates:
        if center in subset and 1 <= len(subset) <= max_order:
            by_order[len(subset)].append(subset)

    seed_retained: dict[tuple[str, ...], int] = {}
    for order, subsets in by_order.items():
        positives_order = [subset for subset in subsets if candidates[subset] == 1]
        negatives_order = [subset for subset in subsets if candidates[subset] == 0]
        rng.shuffle(positives_order)
        rng.shuffle(negatives_order)
        effective_negative_budget = sampled_budget_per_order
        if prefer_high_order_negatives:
            if order <= 3:
                effective_negative_budget = max(0, sampled_budget_per_order // 2)
            elif order >= 6:
                effective_negative_budget = sampled_budget_per_order + 1
        if observed_heavy:
            positive_slots = min(len(positives_order), top_k_per_order)
            negative_slots = min(max(0, effective_negative_budget), max(0, top_k_per_order - positive_slots))
            selected = positives_order[:positive_slots] + negatives_order[:negative_slots]
        else:
            half = max(1, top_k_per_order // 2)
            selected = positives_order[:half] + negatives_order[: max(0, top_k_per_order - min(len(positives_order), half))]
        for subset in selected[:top_k_per_order]:
            seed_retained[subset] = candidates[subset]

    if min_observed_parents_per_observed > 0:
        retained = dict(seed_retained)
        observed_subsets = sorted(
            [subset for subset, exist in retained.items() if exist == 1 and len(subset) >= 3],
            key=lambda item: (len(item), item),
            reverse=True,
        )
        for subset in observed_subsets:
            observed_parents = observed_immediate_subsets(subset, center, observed_set)
            if not observed_parents:
                continue
            rng.shuffle(observed_parents)
            current_observed_parents = [
                parent for parent in observed_parents if retained.get(parent) == 1
            ]
            needed = max(0, min_observed_parents_per_observed - len(current_observed_parents))
            if needed <= 0:
                continue
            add_candidates = [parent for parent in observed_parents if parent not in retained]
            if max_observed_parents_per_observed is not None:
                allowed_total = max(0, max_observed_parents_per_observed - len(current_observed_parents))
                needed = min(needed, allowed_total)
            for parent in add_candidates[:needed]:
                retained[parent] = 1
        seed_retained = retained

    if enforce_topdown_closure:
        retained: dict[tuple[str, ...], int] = {}
        if closure_cache is None:
            closure_cache = {}
        for subset in sorted(seed_retained, key=lambda item: (len(item), item), reverse=True):
            retained[subset] = seed_retained[subset]
            for closure_subset, closure_exist in collect_topdown_closure(
                center=center,
                subset=subset,
                observed_set=observed_set,
                closure_cache=closure_cache,
            ).items():
                previous = retained.get(closure_subset)
                if previous is None or closure_exist > previous:
                    retained[closure_subset] = closure_exist
    elif expand_intermediate_subsets:
        retained = dict(seed_retained)
        add_sampled_intermediates(
            center=center,
            candidates=retained,
            max_order=max_order,
            max_parents_per_child=intermediate_parents_per_child,
        )
    else:
        retained = dict(seed_retained)

    tokens = []
    ordered_subsets = sorted(retained, key=lambda item: (len(item), item), reverse=True)
    retained_set = set(ordered_subsets)
    for position, subset in enumerate(ordered_subsets):
        exist = int(retained[subset])
        parents = immediate_supersets(subset, center, retained_set)
        source = token_source(subset, center, retained, positive_source_map=positive_source_map)
        if expand_intermediate_subsets and exist == 0 and any(child in retained for child in immediate_subsets(subset, center)):
            source = "sampled_negative_or_intermediate"
        supervised_parents = parents
        if observed_only_composition:
            supervised_parents = [
                parent
                for parent in parents
                if source == "observed" and token_source(parent, center, retained) == "observed"
            ]
        parent_subset_strs = [subset_to_str(parent) for parent in supervised_parents]
        composition_types = [comp_label(int(retained[parent]), exist) for parent in supervised_parents]
        tokens.append(
            {
                "token_id": f"{center}::NONE::{position:06d}",
                "center_drug_id": center,
                "side_effect_id": "NONE",
                "side_effect_name": "NONE",
                "subset_drug_ids": list(subset),
                "subset_str": subset_to_str(subset),
                "order": len(subset),
                "exist": exist,
                "exist_source": source,
                "parent_subsets": parent_subset_strs,
                "parent_token_keys": parent_subset_strs,
                "parent_exist": [int(retained[parent]) for parent in supervised_parents],
                "parent_exist_source": [
                    token_source(parent, center, retained, positive_source_map=positive_source_map)
                    for parent in supervised_parents
                ],
                "composition_types": composition_types,
                "position_in_sequence": position,
            }
        )
    return tokens


def build_centered_hyperedge_dag_rows(
    *,
    root_dir: Path,
    output_dir: Path,
    seed: int = 42,
    max_order: int = 8,
    top_k_per_order: int = 12,
    k_views: int = 4,
    negatives_per_positive: int = 2,
    observed_heavy: bool = False,
    sampled_budget_per_order: int = 2,
    expand_intermediate_subsets: bool = False,
    min_observed_parents_per_observed: int = 0,
    max_observed_parents_per_observed: int | None = None,
    prefer_high_order_negatives: bool = False,
    intermediate_parents_per_child: int | None = None,
    observed_only_composition: bool = False,
    enforce_topdown_closure: bool = False,
) -> tuple[list[dict], dict]:
    data = load_cora_ca(root_dir, seed=seed)
    center_to_edges = build_center_to_edges(data.hyperedges)
    observed_set = {tuple(sorted({str(int(node_id)) for node_id in edge}, key=int)) for edge in data.hyperedges if len(edge) >= 2}
    all_nodes = [str(node_id) for node_id in range(data.features.shape[0])]

    rows = []
    length_values = []
    empty_centers = 0
    for center in all_nodes:
        positive_edges = center_to_edges.get(center, [])
        if not positive_edges:
            empty_centers += 1
        closure_cache: dict[tuple[str, ...], dict[tuple[str, ...], int]] = {}
        for view_id in range(k_views):
            rng = random.Random(seed * 1000003 + int(center) * 1009 + view_id)
            tokens = build_tokens_for_center_view(
                center=center,
                positive_edges=positive_edges,
                observed_set=observed_set,
                all_nodes=all_nodes,
                max_order=max_order,
                top_k_per_order=top_k_per_order,
                negatives_per_positive=negatives_per_positive,
                observed_heavy=observed_heavy,
                sampled_budget_per_order=sampled_budget_per_order,
                expand_intermediate_subsets=expand_intermediate_subsets,
                min_observed_parents_per_observed=min_observed_parents_per_observed,
                max_observed_parents_per_observed=max_observed_parents_per_observed,
                prefer_high_order_negatives=prefer_high_order_negatives,
                intermediate_parents_per_child=intermediate_parents_per_child,
                observed_only_composition=observed_only_composition,
                enforce_topdown_closure=enforce_topdown_closure,
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
            length_values.append(len(tokens))

    output_dir = ensure_dir(output_dir)
    sequence_path = output_dir / f"cora_centered_hyperedge_dag.max_order{max_order}.topk{top_k_per_order}.views{k_views}.jsonl"
    with sequence_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "num_nodes": int(data.features.shape[0]),
        "num_hyperedges": int(len(data.hyperedges)),
        "num_centers_with_edges": int(len(center_to_edges)),
        "num_rows": int(len(rows)),
        "empty_centers": int(empty_centers),
        "max_order": int(max_order),
        "top_k_per_order": int(top_k_per_order),
        "k_views": int(k_views),
        "negatives_per_positive": int(negatives_per_positive),
        "observed_heavy": bool(observed_heavy),
        "sampled_budget_per_order": int(sampled_budget_per_order),
        "expand_intermediate_subsets": bool(expand_intermediate_subsets),
        "min_observed_parents_per_observed": int(min_observed_parents_per_observed),
        "max_observed_parents_per_observed": max_observed_parents_per_observed,
        "prefer_high_order_negatives": bool(prefer_high_order_negatives),
        "intermediate_parents_per_child": intermediate_parents_per_child,
        "observed_only_composition": bool(observed_only_composition),
        "enforce_topdown_closure": bool(enforce_topdown_closure),
        "mean_sequence_length": float(np.mean(length_values)) if length_values else 0.0,
        "max_sequence_length": int(max(length_values)) if length_values else 0,
        "sequence_path": str(sequence_path),
    }
    save_json(output_dir / "cora_centered_hyperedge_dag.stats.json", stats)
    save_json(output_dir / "cora_centered_hyperedge_dag.examples.json", rows[:3])

    drug2id_path = output_dir / "drug2id.json"
    save_json(drug2id_path, {str(node_id): int(node_id) for node_id in range(data.features.shape[0])})

    feature_path = output_dir / "paper_node_features.pt"
    try:
        import torch

        features = torch.as_tensor(data.features, dtype=torch.float32)
        torch.save(
            {
                "model_name": "cora-ca-raw-bag-of-words",
                "feature_dim": int(features.size(1)),
                "drug_ids": [str(node_id) for node_id in range(data.features.shape[0])],
                "features": features,
                "smiles": {},
            },
            feature_path,
        )
    except Exception as exc:  # pragma: no cover - metadata export should not block DAG building.
        print(f"[centered_dag] Warning: failed to save paper feature table: {exc}")

    stats["drug2id_path"] = str(drug2id_path)
    stats["feature_path"] = str(feature_path)
    return rows, stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="../processed/hypergraph_benchmark_standard/cora")
    parser.add_argument("--output-dir", default="outputs/centered_dag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument("--top-k-per-order", type=int, default=12)
    parser.add_argument("--k-views", type=int, default=4)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--observed-heavy", action="store_true")
    parser.add_argument("--sampled-budget-per-order", type=int, default=2)
    parser.add_argument("--expand-intermediate-subsets", action="store_true")
    parser.add_argument("--min-observed-parents-per-observed", type=int, default=0)
    parser.add_argument("--max-observed-parents-per-observed", type=int, default=None)
    parser.add_argument("--prefer-high-order-negatives", action="store_true")
    parser.add_argument("--intermediate-parents-per-child", type=int, default=None)
    parser.add_argument("--observed-only-composition", action="store_true")
    parser.add_argument("--enforce-topdown-closure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows, stats = build_centered_hyperedge_dag_rows(
        root_dir=Path(args.root_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        max_order=args.max_order,
        top_k_per_order=args.top_k_per_order,
        k_views=args.k_views,
        negatives_per_positive=args.negatives_per_positive,
        observed_heavy=args.observed_heavy,
        sampled_budget_per_order=args.sampled_budget_per_order,
        expand_intermediate_subsets=args.expand_intermediate_subsets,
        min_observed_parents_per_observed=args.min_observed_parents_per_observed,
        max_observed_parents_per_observed=args.max_observed_parents_per_observed,
        prefer_high_order_negatives=args.prefer_high_order_negatives,
        intermediate_parents_per_child=args.intermediate_parents_per_child,
        observed_only_composition=args.observed_only_composition,
        enforce_topdown_closure=args.enforce_topdown_closure,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for row in rows[:3]:
        print(json.dumps(row, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
