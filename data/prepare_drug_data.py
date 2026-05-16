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

from HGPM.utils.io import PACKAGE_ROOT as HGPM_PACKAGE_ROOT, resolve_repo_path, save_json


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _package_drug_root() -> Path:
    return HGPM_PACKAGE_ROOT / "data" / "drug"


def prepare_drug_data(*, overwrite: bool) -> dict:
    drug_root = _package_drug_root()
    hoddi_root = drug_root / "hoddi"
    jader_root = drug_root / "jader"
    for path in (
        hoddi_root / "raw",
        hoddi_root / "features",
        hoddi_root / "splits",
        hoddi_root / "baselines",
        jader_root / "raw",
        jader_root / "features",
    ):
        path.mkdir(parents=True, exist_ok=True)

    source_hoddi_root = resolve_repo_path("outputs/hoddi")
    source_jader_feature_root = resolve_repo_path("outputs/jader_hybrid")

    hoddi_sequence = source_hoddi_root / "node_center_sequences.global_observed.max_order4.topk32.views8.jsonl"
    hoddi_drug2id = source_hoddi_root / "drug2id.json"
    hoddi_smiles = source_hoddi_root / "drug_smiles_features.pt"
    hoddi_smiles_stats = source_hoddi_root / "drug_smiles_features.stats.json"
    hoddi_eval_csv = resolve_repo_path("results/data/hoddi_eval_subset1_hyperedges.csv")
    hoddi_eval_csv_502525 = resolve_repo_path("results/data/hoddi_eval_subset1_hyperedges_502525.csv")
    hoddi_split_dir = source_hoddi_root / "splits"
    hoddi_baseline_metrics = resolve_repo_path("outputs/hoddi_record_task1_target_centered_hyperedge/formal_test_metrics.json")

    jader_eval_csv = resolve_repo_path("results/data/jader_eval_subset1_3plus.csv")
    jader_smiles = source_jader_feature_root / "drug_smiles_features.pt"
    jader_smiles_stats = source_jader_feature_root / "drug_smiles_features.stats.json"
    jader_feature_csv = source_jader_feature_root / "jader_hybrid_drug_features.csv"

    _copy_file(hoddi_sequence, hoddi_root / hoddi_sequence.name, overwrite=overwrite)
    _copy_file(hoddi_drug2id, hoddi_root / "drug2id.json", overwrite=overwrite)
    _copy_file(hoddi_smiles, hoddi_root / "features" / "drug_smiles_features.pt", overwrite=overwrite)
    if hoddi_smiles_stats.exists():
        _copy_file(hoddi_smiles_stats, hoddi_root / "features" / hoddi_smiles_stats.name, overwrite=overwrite)
    _copy_file(hoddi_eval_csv, hoddi_root / "raw" / hoddi_eval_csv.name, overwrite=overwrite)
    if hoddi_eval_csv_502525.exists():
        _copy_file(hoddi_eval_csv_502525, hoddi_root / "raw" / hoddi_eval_csv_502525.name, overwrite=overwrite)
    if hoddi_split_dir.exists():
        target_split_dir = hoddi_root / "splits" / "pretrain"
        if target_split_dir.exists() and overwrite:
            shutil.rmtree(target_split_dir)
        if not target_split_dir.exists():
            shutil.copytree(hoddi_split_dir, target_split_dir)
    if hoddi_baseline_metrics.exists():
        _copy_file(hoddi_baseline_metrics, hoddi_root / "baselines" / "formal_test_metrics.json", overwrite=overwrite)

    _copy_file(jader_eval_csv, jader_root / "raw" / jader_eval_csv.name, overwrite=overwrite)
    _copy_file(jader_smiles, jader_root / "features" / "drug_smiles_features.pt", overwrite=overwrite)
    if jader_smiles_stats.exists():
        _copy_file(jader_smiles_stats, jader_root / "features" / jader_smiles_stats.name, overwrite=overwrite)
    if jader_feature_csv.exists():
        _copy_file(jader_feature_csv, jader_root / "features" / jader_feature_csv.name, overwrite=overwrite)

    manifest = {
        "hoddi": {
            "root": str(hoddi_root),
            "sequence_path": str(hoddi_root / hoddi_sequence.name),
            "drug2id_path": str(hoddi_root / "drug2id.json"),
            "feature_path": str(hoddi_root / "features" / "drug_smiles_features.pt"),
            "record_csv": str(hoddi_root / "raw" / hoddi_eval_csv.name),
            "baseline_metrics_path": str(hoddi_root / "baselines" / "formal_test_metrics.json"),
        },
        "jader": {
            "root": str(jader_root),
            "record_csv": str(jader_root / "raw" / jader_eval_csv.name),
            "feature_path": str(jader_root / "features" / "drug_smiles_features.pt"),
        },
    }
    save_json(drug_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare package-local HODDI/JADER data assets.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare_drug_data(overwrite=bool(args.overwrite))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
