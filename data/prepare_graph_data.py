from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from HGPM.data.graph.benchmark_hyperdag_data import (
    build_benchmark_centered_hyperedge_dag,
    prepare_raw_benchmark_protocol,
)
from HGPM.utils.io import PACKAGE_ROOT as HGPM_PACKAGE_ROOT, resolve_repo_path, save_json


PAPER_GRAPH_DATASETS = (
    "citeseer",
    "pubmed",
    "cora_ca",
    "dblp_ca",
    "congress",
    "senate",
    "walmart",
    "house",
)

GRAPH_PARAMS = {
    "citeseer": dict(max_order=8, top_k_per_order=24, k_views=2, negatives_per_positive=1, observed_heavy=True, sampled_budget_per_order=2, max_substructures_per_edge=16),
    "pubmed": dict(max_order=8, top_k_per_order=24, k_views=2, negatives_per_positive=1, observed_heavy=True, sampled_budget_per_order=2, max_substructures_per_edge=16),
    "cora_ca": dict(max_order=8, top_k_per_order=16, k_views=2, negatives_per_positive=1, observed_heavy=True, sampled_budget_per_order=2, max_substructures_per_edge=16),
    "dblp_ca": dict(max_order=8, top_k_per_order=24, k_views=2, negatives_per_positive=1, observed_heavy=True, sampled_budget_per_order=2, max_substructures_per_edge=16),
    "congress": dict(max_order=16, top_k_per_order=4, k_views=2, negatives_per_positive=1, observed_heavy=True, sampled_budget_per_order=2, max_substructures_per_edge=16),
    "senate": dict(max_order=24, top_k_per_order=16, k_views=2, negatives_per_positive=3, observed_heavy=True, sampled_budget_per_order=2, max_substructures_per_edge=16),
    "house": dict(max_order=8, top_k_per_order=24, k_views=2, negatives_per_positive=1, observed_heavy=True, sampled_budget_per_order=2, max_substructures_per_edge=16),
    "walmart": dict(max_order=4, top_k_per_order=2, k_views=1, negatives_per_positive=1, observed_heavy=True, sampled_budget_per_order=1, max_substructures_per_edge=4),
}

PRECOMPUTED_DAG_DIRS = {
    "citeseer": "citeseer_centered_hyperedge_dag_adjacent_k24_midneg_v2",
    "pubmed": "pubmed_centered_hyperedge_dag_adjacent_k24_midneg_v2",
    "cora_ca": "cora_ca_centered_hyperedge_dag_paper_v1",
    "dblp_ca": "dblp_ca_centered_hyperedge_dag_paper_v1",
    "congress": "congress_centered_hyperedge_dag_paper_v1",
    "senate": "senate_centered_hyperedge_dag_paper_v1",
    "house": "house_centered_hyperedge_dag_adjacent_k24_midneg_v2",
}


def _copytree_if_missing(src: Path, dst: Path, *, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        if not overwrite:
            return True
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def _package_graph_root() -> Path:
    return HGPM_PACKAGE_ROOT / "data" / "graph"


def prepare_graph_dataset(*, dataset: str, seed: int, rebuild_protocol: bool, rebuild_dag: bool, overwrite: bool) -> dict:
    graph_root = _package_graph_root()
    raw_dir = graph_root / "raw"
    protocols_dir = graph_root / "protocols"
    dags_dir = graph_root / "dags"
    raw_dir.mkdir(parents=True, exist_ok=True)
    protocols_dir.mkdir(parents=True, exist_ok=True)
    dags_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = protocols_dir / f"{dataset}_protocol_502525"
    workspace_protocol_dir = resolve_repo_path(f"processed/hypergraph_benchmarks/{dataset}_protocol_502525")

    if rebuild_protocol or not protocol_dir.exists():
        if not rebuild_protocol and workspace_protocol_dir.exists():
            _copytree_if_missing(workspace_protocol_dir, protocol_dir, overwrite=overwrite)
        else:
            prepare_raw_benchmark_protocol(
                dataset_name=dataset,
                train_ratio=0.5,
                val_ratio=0.25,
                seed=seed,
                protocol_dir=protocol_dir,
                force_rebuild=rebuild_protocol,
            )

    dag_dir_name = PRECOMPUTED_DAG_DIRS.get(dataset, f"{dataset}_centered_hyperedge_dag_paper_v1")
    dag_output_dir = dags_dir / dag_dir_name
    workspace_dag_dir = resolve_repo_path(f"cora_ca_dag_pretrain/outputs/{dag_dir_name}")

    if dataset != "walmart":
        if rebuild_dag or not dag_output_dir.exists():
            if not rebuild_dag and workspace_dag_dir.exists():
                _copytree_if_missing(workspace_dag_dir, dag_output_dir, overwrite=overwrite)
            else:
                build_benchmark_centered_hyperedge_dag(
                    dataset_name=dataset,
                    protocol_dir=protocol_dir,
                    output_dir=dag_output_dir,
                    seed=seed,
                    **GRAPH_PARAMS[dataset],
                )

    sequence_path = None
    if dataset != "walmart":
        params = GRAPH_PARAMS[dataset]
        sequence_path = dag_output_dir / f"{dataset}_centered_hyperedge_dag.max_order{params['max_order']}.topk{params['top_k_per_order']}.views{params['k_views']}.jsonl"

    return {
        "dataset": dataset,
        "protocol_dir": str(protocol_dir),
        "dag_output_dir": str(dag_output_dir) if dataset != "walmart" else None,
        "sequence_path": str(sequence_path) if sequence_path is not None else None,
        "copied_workspace_protocol": bool(workspace_protocol_dir.exists() and not rebuild_protocol),
        "copied_workspace_dag": bool(dataset != "walmart" and workspace_dag_dir.exists() and not rebuild_dag),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare package-local graph benchmark data and centered DAGs.")
    parser.add_argument("--datasets", nargs="+", default=list(PAPER_GRAPH_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-protocol", action="store_true")
    parser.add_argument("--rebuild-dag", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepared = [
        prepare_graph_dataset(
            dataset=dataset,
            seed=args.seed,
            rebuild_protocol=bool(args.rebuild_protocol),
            rebuild_dag=bool(args.rebuild_dag),
            overwrite=bool(args.overwrite),
        )
        for dataset in args.datasets
    ]

    manifest = {
        "datasets": prepared,
        "package_graph_root": str(_package_graph_root()),
    }
    save_json(_package_graph_root() / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
