from __future__ import annotations

import ast
import copy
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from HGPM.data.drug.pretrain_dataset import load_sequence_rows
from HGPM.data.drug.target_center_builder import build_target_center_sequence


MASK_EXIST = "MASK_EXIST"
MASK_COMP = "MASK_COMP"
NO_COMP = "NO_COMP"
COMP_LABELS = ("COMP", "EMER", "INHIB", "UNKNOWN")


def parse_drug_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        drugs = [str(item) for item in value]
    elif value is None:
        drugs = []
    else:
        text = str(value).strip()
        if not text:
            drugs = []
        else:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, str):
                drugs = [parsed]
            else:
                drugs = [str(item) for item in parsed]
    return tuple(sorted(drugs))


def load_record_frame(record_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(record_path)
    canonical_col = "drug_ids_canonical" if "drug_ids_canonical" in frame.columns else "drug_ids"
    frame = frame.copy()
    frame["drug_tuple"] = frame[canonical_col].map(parse_drug_tuple)
    frame["drug_set"] = frame["drug_tuple"].map(list)
    frame["drug_count"] = frame["drug_tuple"].map(len)
    frame["side_effect_id"] = frame["se_label"].astype(str)
    frame["label"] = frame["hyperedge_label"].astype(int).replace({-1: 0})
    frame["record_id"] = frame["report_id"].astype(str)
    if "split" not in frame.columns:
        raise ValueError("Benchmark CSV must contain a 'split' column.")
    return frame


def split_record_frame(frame: pd.DataFrame, split_column: str = "split") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = frame[frame[split_column] == "train"].reset_index(drop=True)
    val_df = frame[frame[split_column] == "val"].reset_index(drop=True)
    test_df = frame[frame[split_column] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def build_train_context_db(train_df: pd.DataFrame) -> dict[tuple[tuple[str, ...], str], int]:
    context_db: dict[tuple[tuple[str, ...], str], int] = {}
    for row in train_df.itertuples(index=False):
        key = (tuple(row.drug_tuple), str(row.side_effect_id))
        label = int(row.label)
        existing = context_db.get(key)
        if existing is None:
            context_db[key] = label
        elif existing != label:
            context_db[key] = max(existing, label)
    return context_db


def discover_pretrain_sequence_file(output_dir: Path, sequence_glob: str | None = None) -> Path:
    patterns = [sequence_glob] if sequence_glob else []
    patterns.extend(
        [
            "*global_observed*max_order4*topk32*views8*.jsonl",
            "*global_observed*max_order4*topk32*.jsonl",
            "*global_all_samples*max_order4*topk32*views8*.jsonl",
            "*global_all_samples*max_order4*topk32*.jsonl",
        ]
    )
    for pattern in patterns:
        if not pattern:
            continue
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find a pretrain sequence file under {output_dir}.")


def load_pretrain_sequence_rows(sequence_path: Path) -> list[dict]:
    return load_sequence_rows(sequence_path)


@dataclass
class HoddiRecordArtifacts:
    drug_vocab: dict[str, int]
    encoder_side_effect_vocab: dict[str, int]
    encoder_side_effect_id: int
    record_side_effect_vocab: dict[str, int]


def _load_base_drug_vocab(drug_vocab_path: Path) -> dict[str, int]:
    raw = json.loads(drug_vocab_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected drug vocab payload in {drug_vocab_path}.")
    return {str(key): int(value) for key, value in raw.items()}


def build_record_task1_artifacts(
    hoddi_output_dir: Path,
    pretrain_sequence_rows: list[dict],
    benchmark_frame: pd.DataFrame,
) -> HoddiRecordArtifacts:
    base_drug_vocab = _load_base_drug_vocab(hoddi_output_dir / "drug2id.json")
    drug_vocab = {"[PAD]": 0, "[UNK]": 1}
    for drug in sorted(base_drug_vocab):
        if drug not in drug_vocab:
            drug_vocab[drug] = len(drug_vocab)
    # Allow benchmark-specific datasets to extend the original HODDI drug vocab
    # instead of collapsing unseen drugs to [UNK].
    for drug_tuple in benchmark_frame["drug_tuple"].tolist():
        for drug in drug_tuple:
            if drug not in drug_vocab:
                drug_vocab[drug] = len(drug_vocab)

    record_side_effect_vocab = {"[PAD]": 0, "[UNK]": 1}
    for side_effect_id in sorted({str(value) for value in benchmark_frame["side_effect_id"].tolist()}):
        record_side_effect_vocab[side_effect_id] = len(record_side_effect_vocab)

    encoder_side_effect_vocab = {"[PAD]": 0, "[UNK]": 1}
    for side_effect_id in sorted({str(value) for value in benchmark_frame["side_effect_id"].tolist()}):
        if side_effect_id not in encoder_side_effect_vocab:
            encoder_side_effect_vocab[side_effect_id] = len(encoder_side_effect_vocab)
    for row in pretrain_sequence_rows:
        side_effect_id = str(row.get("side_effect_id", "__GLOBAL__"))
        if side_effect_id not in encoder_side_effect_vocab:
            encoder_side_effect_vocab[side_effect_id] = len(encoder_side_effect_vocab)

    encoder_side_effect_id = encoder_side_effect_vocab["[UNK]"]

    return HoddiRecordArtifacts(
        drug_vocab=drug_vocab,
        encoder_side_effect_vocab=encoder_side_effect_vocab,
        encoder_side_effect_id=encoder_side_effect_id,
        record_side_effect_vocab=record_side_effect_vocab,
    )


def _enumerate_center_subsets(drug_tuple: tuple[str, ...], center: str, max_context_order: int) -> list[tuple[str, ...]]:
    others = [drug for drug in drug_tuple if drug != center]
    candidates: set[tuple[str, ...]] = set()
    max_order = min(max_context_order, len(drug_tuple))
    candidates.add((center,))
    for order in range(2, max_order + 1):
        choose = order - 1
        for combo in itertools.combinations(others, choose):
            subset = tuple(sorted((center, *combo)))
            candidates.add(subset)
    candidates.add(tuple(sorted(drug_tuple)))
    return sorted(candidates, key=lambda subset: (len(subset), subset))


def _comp_label(parent_exist: int, child_exist: int) -> str:
    if parent_exist == 1 and child_exist == 1:
        return "COMP"
    if parent_exist == 0 and child_exist == 1:
        return "EMER"
    if parent_exist == 1 and child_exist == 0:
        return "INHIB"
    return "UNKNOWN"


def _build_token(
    subset: tuple[str, ...],
    *,
    center: str,
    full_target: tuple[str, ...],
    side_effect_id: str,
    context_db: dict[tuple[tuple[str, ...], str], int],
    keep_unknown_context: bool,
    mask_target_token: bool,
    order_clip: int,
) -> dict | None:
    subset_key = (subset, side_effect_id)
    label = context_db.get(subset_key)
    is_target = subset == full_target
    if label is None and not is_target and not keep_unknown_context:
        return None

    if label is None:
        exist_label = None
        exist_input = MASK_EXIST
    else:
        exist_label = int(label)
        exist_input = int(label)

    parent_subsets: list[str] = []
    comp_targets: list[str] = []
    comp_inputs: list[str] = []
    comp_observed_mask: list[int] = []

    if len(subset) <= 2:
        comp_targets = [NO_COMP]
        comp_inputs = [NO_COMP]
        comp_observed_mask = [1]
    else:
        child_label = context_db.get(subset_key)
        for drug in subset:
            if drug == center:
                continue
            parent_subset = tuple(sorted(item for item in subset if item != drug))
            parent_subsets.append("|".join(parent_subset))
            parent_label = context_db.get((parent_subset, side_effect_id))
            if parent_label is None or child_label is None:
                comp_targets.append(MASK_COMP)
                comp_inputs.append(MASK_COMP)
                comp_observed_mask.append(0)
            else:
                label_name = _comp_label(int(parent_label), int(child_label))
                comp_targets.append(label_name)
                comp_inputs.append(label_name)
                comp_observed_mask.append(1)

    if is_target and mask_target_token:
        exist_input = MASK_EXIST
        if len(subset) > 2:
            comp_inputs = [MASK_COMP] * len(comp_inputs)

    return {
        "subset_drug_ids": list(subset),
        "subset_str": "|".join(subset),
        "order": min(len(subset), order_clip),
        "order_raw": len(subset),
        "center_drug_id": center,
        "side_effect_id": side_effect_id,
        "is_target": is_target,
        "exist_label": exist_label,
        "exist_input": exist_input,
        "parent_subsets": parent_subsets,
        "composition_types_target": comp_targets,
        "composition_types_input": comp_inputs,
        "composition_observed_mask": comp_observed_mask,
    }


def build_center_sequence_for_record(
    drug_tuple: tuple[str, ...],
    *,
    center: str,
    side_effect_id: str,
    context_db: dict[tuple[tuple[str, ...], str], int],
    max_context_order: int,
    keep_unknown_context: bool,
    mask_target_token: bool,
    order_clip: int = 7,
) -> dict:
    tokens: list[dict] = []
    for subset in _enumerate_center_subsets(drug_tuple, center, max_context_order):
        token = _build_token(
            subset,
            center=center,
            full_target=drug_tuple,
            side_effect_id=side_effect_id,
            context_db=context_db,
            keep_unknown_context=keep_unknown_context,
            mask_target_token=mask_target_token,
            order_clip=order_clip,
        )
        if token is None:
            continue
        tokens.append(token)

    tokens = sorted(tokens, key=lambda token: (int(token["order_raw"]), tuple(token["subset_drug_ids"])))
    target_token_index = -1
    for index, token in enumerate(tokens):
        token["position_in_sequence"] = index
        if token["is_target"]:
            target_token_index = index
    if target_token_index < 0:
        raise ValueError(f"Target token {drug_tuple} was not preserved for center {center}.")

    return {
        "center_drug_id": center,
        "side_effect_id": side_effect_id,
        "tokens": tokens,
        "target_token_index": target_token_index,
        "sequence_length": len(tokens),
    }


def build_record_views(
    frame: pd.DataFrame,
    *,
    context_db: dict[tuple[tuple[str, ...], str], int],
    max_context_order: int,
    keep_unknown_context: bool,
    mask_target_token: bool,
    use_target_centered_branch: bool = False,
    target_center_max_depth: int | None = None,
    order_clip: int = 7,
) -> list[dict]:
    records: list[dict] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        drug_tuple = tuple(row.drug_tuple)
        center_sequences = [
            build_center_sequence_for_record(
                drug_tuple,
                center=center,
                side_effect_id=str(row.side_effect_id),
                context_db=context_db,
                max_context_order=max_context_order,
                keep_unknown_context=keep_unknown_context,
                mask_target_token=mask_target_token,
                order_clip=order_clip,
            )
            for center in drug_tuple
        ]
        target_center_sequence = None
        if use_target_centered_branch:
            target_center_sequence = build_target_center_sequence(
                drug_tuple,
                side_effect_id=str(row.side_effect_id),
                context_db=context_db,
                max_depth=target_center_max_depth,
                keep_unknown_context=keep_unknown_context,
                mask_target_token=mask_target_token,
                order_clip=order_clip,
            )
        records.append(
            {
                "record_id": str(row.record_id),
                "row_index": int(row_index),
                "drug_set": list(drug_tuple),
                "drug_count": int(row.drug_count),
                "side_effect_id": str(row.side_effect_id),
                "label": int(row.label),
                "split": str(row.split),
                "quarter": str(row.quarter) if hasattr(row, "quarter") else None,
                "center_sequences": center_sequences,
                "target_center_sequence": target_center_sequence,
            }
        )
    return records


class HoddiRecordTask1Dataset:
    def __init__(
        self,
        records: list[dict],
        *,
        single_view_mode: bool = False,
        single_view_seed: int = 42,
    ) -> None:
        self.records = records
        self.single_view_mode = bool(single_view_mode)
        self._rng = random.Random(int(single_view_seed))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        item = copy.deepcopy(self.records[index])
        # Single-view training ablation: keep exactly one node-centered view
        # during training, while validation/test can still use all views.
        if self.single_view_mode and len(item.get("center_sequences", [])) > 1:
            item["center_sequences"] = [self._rng.choice(item["center_sequences"])]
        return item
