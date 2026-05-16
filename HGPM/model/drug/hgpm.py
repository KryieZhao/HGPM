from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from HGPM.task.graph.graph_base import AttentionPool
from HGPM.model.graph.hgpm import RelationAwareTransformerBlock
from HGPM.data.drug.pretrain_collator import (
    COMP_COMP,
    COMP_EMER,
    COMP_INHIB,
    COMP_MASK,
    COMP_NO_COMP,
    COMP_PAD,
    COMP_UNKNOWN,
    EXIST_ABSENT,
    EXIST_MASK,
    EXIST_PRESENT,
    HoddiPretrainArtifacts,
)
from HGPM.data.drug.regime_collator import COMP_INPUT_MAP, EXIST_INPUT_MAP
from HGPM.data.drug.regime_dataset import MASK_COMP, MASK_EXIST, NO_COMP


def _overlap_bucket(overlap_ratio: float) -> int:
    if overlap_ratio <= 0.0:
        return 0
    if overlap_ratio <= 0.25:
        return 1
    if overlap_ratio <= 0.5:
        return 2
    if overlap_ratio <= 0.75:
        return 3
    return 4


def _raw_exist_role(token: dict) -> str:
    exist_source = str(token.get("exist_source", "")).upper()
    if exist_source in {"POS", "OBSERVED"}:
        return "observed"
    if exist_source in {"NEG", "SAMPLED_NEGATIVE"}:
        return "negative"
    exist_label = token.get("exist_label", token.get("exist"))
    if exist_label == 1:
        return "observed"
    if exist_label == 0:
        return "negative"
    return "other"


PRIMARY_COMP_LABEL_TO_ID = {
    "NO_COMP": 0,
    "COMP": 1,
    "EMER": 2,
    "INHIB": 3,
    "UNKNOWN": 4,
}
PRIMARY_COMP_MASK_ID = len(PRIMARY_COMP_LABEL_TO_ID)


def _primary_comp_label(token: dict) -> str:
    values = list(
        token.get("composition_types_input")
        or token.get("composition_types_target")
        or token.get("composition_types")
        or []
    )
    filtered = [str(value) for value in values if str(value) not in {MASK_COMP, NO_COMP, "PAD", "None"}]
    if not filtered:
        return "NO_COMP"
    counts: dict[str, int] = {}
    for value in filtered:
        key = value if value in {"COMP", "EMER", "INHIB", "UNKNOWN"} else "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _primary_comp_id(token: dict) -> int:
    return int(PRIMARY_COMP_LABEL_TO_ID[_primary_comp_label(token)])


def _comp_pair_bucket(src_comp_id: int, dst_comp_id: int) -> int:
    vocab = len(PRIMARY_COMP_LABEL_TO_ID)
    return int(src_comp_id * vocab + dst_comp_id)


def _is_comp_eligible(token: dict) -> bool:
    return _primary_comp_label(token) != "NO_COMP"


def _exist_source_id(token: dict) -> int:
    exist_source = str(token.get("exist_source", "")).upper()
    if exist_source in {"POS", "OBSERVED"}:
        return 1
    if exist_source in {"NEG", "SAMPLED_NEGATIVE"}:
        return 2
    return 3


def _encode_comp_ids(
    token: dict,
    *,
    use_comp_input: bool,
    max_comp_slots: int,
    force_mask: bool = False,
) -> tuple[list[int], list[float]]:
    raw_values = list(token.get("composition_types_input", token.get("composition_types", [])))
    if not raw_values:
        return [COMP_NO_COMP] + [COMP_PAD] * (max_comp_slots - 1), [1.0] + [0.0] * (max_comp_slots - 1)
    if force_mask:
        comp_ids = [COMP_MASK] * min(len(raw_values), max_comp_slots)
    elif use_comp_input:
        comp_ids = [COMP_INPUT_MAP.get(value, COMP_MASK) for value in raw_values[:max_comp_slots]]
    else:
        comp_ids = [COMP_MASK] * min(len(raw_values), max_comp_slots)
    comp_mask = [1.0] * len(comp_ids)
    while len(comp_ids) < max_comp_slots:
        comp_ids.append(COMP_PAD)
        comp_mask.append(0.0)
    return comp_ids, comp_mask


def _encode_pretrain_token(token: dict, artifacts: HoddiPretrainArtifacts) -> dict:
    subset_ids = [artifacts.drug_vocab.get(drug, artifacts.drug_vocab["[UNK]"]) for drug in token["subset_drug_ids"][: artifacts.max_subset_size]]
    subset_mask = [1.0] * len(subset_ids)
    while len(subset_ids) < artifacts.max_subset_size:
        subset_ids.append(artifacts.drug_vocab["[PAD]"])
        subset_mask.append(0.0)
    comp_ids = [COMP_INPUT_MAP.get(value, COMP_UNKNOWN) for value in token.get("composition_types", [])[: artifacts.max_comp_slots]]
    comp_mask = [1.0] * len(comp_ids)
    if not comp_ids:
        comp_ids = [COMP_NO_COMP]
        comp_mask = [1.0]
    while len(comp_ids) < artifacts.max_comp_slots:
        comp_ids.append(COMP_PAD)
        comp_mask.append(0.0)
    return {
        "subset_drug_ids": subset_ids,
        "subset_mask": subset_mask,
        "exist_ids": int(token.get("exist", 0)),
        "order_ids": int(token["order"]),
        "comp_ids": comp_ids,
        "comp_mask": comp_mask,
        "token_type_ids": 1,
        "raw_token": token,
    }


def _encode_record_token(
    token: dict,
    drug_vocab: dict[str, int],
    *,
    max_subset_size: int,
    max_comp_slots: int,
    use_exist_input_in_formal: bool,
    use_comp_input_in_formal: bool,
    token_type_id: int,
    force_mask_comp: bool = False,
) -> dict:
    subset_ids = [drug_vocab.get(drug, drug_vocab["[UNK]"]) for drug in token["subset_drug_ids"][:max_subset_size]]
    subset_mask = [1.0] * len(subset_ids)
    while len(subset_ids) < max_subset_size:
        subset_ids.append(drug_vocab["[PAD]"])
        subset_mask.append(0.0)
    exist_id = EXIST_INPUT_MAP[token["exist_input"]] if use_exist_input_in_formal else EXIST_MASK
    comp_ids, comp_mask = _encode_comp_ids(
        token,
        use_comp_input=use_comp_input_in_formal,
        max_comp_slots=max_comp_slots,
        force_mask=force_mask_comp,
    )
    return {
        "subset_drug_ids": subset_ids,
        "subset_mask": subset_mask,
        "exist_ids": exist_id,
        "order_ids": int(token["order"]),
        "comp_ids": comp_ids,
        "comp_mask": comp_mask,
        "token_type_ids": token_type_id,
        "raw_token": token,
    }


def _build_relation_tensors(
    tokens: list[dict],
    *,
    masked_comp_token_positions: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = len(tokens)
    edge_direction = torch.zeros((seq_len, seq_len), dtype=torch.long)
    overlap_bucket = torch.zeros((seq_len, seq_len), dtype=torch.long)
    comp_pair_bucket = torch.zeros((seq_len, seq_len), dtype=torch.long)
    sibling_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)
    token_keys = [str(token["raw_token"].get("subset_str", "|".join(token["raw_token"]["subset_drug_ids"]))) for token in tokens]
    token_subsets = [set(token["raw_token"]["subset_drug_ids"]) for token in tokens]
    token_comp_ids = [_primary_comp_id(token["raw_token"]) for token in tokens]
    token_parent_sets: list[set[str]] = [set(str(parent) for parent in token["raw_token"].get("parent_subsets", [])) for token in tokens]
    token_child_sets: list[set[str]] = [set() for _ in tokens]
    masked_comp_token_positions = masked_comp_token_positions or set()
    for child_idx, parents in enumerate(token_parent_sets):
        for parent_idx, parent_key in enumerate(token_keys):
            if parent_key in parents:
                token_child_sets[parent_idx].add(token_keys[child_idx])

    for i in range(seq_len):
        for j in range(seq_len):
            if token_keys[i] in token_parent_sets[j]:
                edge_direction[i, j] = 1
            elif token_keys[j] in token_parent_sets[i]:
                edge_direction[i, j] = 2
            inter = len(token_subsets[i].intersection(token_subsets[j]))
            union = max(len(token_subsets[i].union(token_subsets[j])), 1)
            overlap_bucket[i, j] = _overlap_bucket(float(inter / union))
            if i in masked_comp_token_positions or j in masked_comp_token_positions:
                comp_pair_bucket[i, j] = PRIMARY_COMP_MASK_ID * len(PRIMARY_COMP_LABEL_TO_ID) + PRIMARY_COMP_MASK_ID
            else:
                comp_pair_bucket[i, j] = _comp_pair_bucket(token_comp_ids[i], token_comp_ids[j])
            if i != j:
                if token_parent_sets[i].intersection(token_parent_sets[j]) or token_child_sets[i].intersection(token_child_sets[j]):
                    sibling_mask[i, j] = 1.0
    return edge_direction, overlap_bucket, comp_pair_bucket, sibling_mask


class HGPMSemanticPretrainCollator:
    """Two-stage masked pretrain collator for HGPM drug semantic pretraining.

    Stage 1 masks token content only.
    Stage 2 masks both token content and composition inputs/bias for selected tokens.
    """

    def __init__(
        self,
        artifacts: HoddiPretrainArtifacts,
        *,
        stage: str,
        token_mask_rate: float,
        comp_mask_rate: float,
        seed: int,
    ) -> None:
        self.artifacts = artifacts
        self.stage = str(stage)
        self.token_mask_rate = float(token_mask_rate)
        self.comp_mask_rate = float(comp_mask_rate)
        self.seed = int(seed)
        self.call_index = 0

    def __call__(self, batch: list[dict]) -> dict:
        rng = random.Random(self.seed * 1000003 + self.call_index)
        self.call_index += 1
        sequence_items = []
        masked_positions = []
        comp_masked_positions = []
        for item in batch:
            tokens = [_encode_pretrain_token(token, self.artifacts) for token in item["tokens"]]
            valid_positions = list(range(len(tokens)))
            chosen: list[int] = []
            chosen_comp: list[int] = []
            if self.stage == "stage2":
                comp_positions = [pos for pos, token in enumerate(tokens) if _is_comp_eligible(token["raw_token"])]
                if comp_positions:
                    num_to_mask = max(1, int(round(len(comp_positions) * self.comp_mask_rate)))
                    chosen_comp = rng.sample(comp_positions, k=min(num_to_mask, len(comp_positions)))
                    chosen = list(chosen_comp)
            if not chosen and valid_positions:
                num_to_mask = max(1, int(round(len(valid_positions) * self.token_mask_rate)))
                chosen = rng.sample(valid_positions, k=min(num_to_mask, len(valid_positions)))
            sequence_items.append(
                {
                    "anchor_id": self.artifacts.drug_vocab.get(item["center_drug_id"], self.artifacts.drug_vocab["[UNK]"]),
                    "se_id": self.artifacts.side_effect_vocab.get(item.get("side_effect_id", "__GLOBAL__"), self.artifacts.side_effect_vocab["[UNK]"]),
                    "tokens": tokens,
                }
            )
            masked_positions.append(set(chosen))
            comp_masked_positions.append(set(chosen_comp))
        packed = _pack_sequence_items(
            sequence_items,
            pad_drug_id=self.artifacts.drug_vocab["[PAD]"],
            pad_side_effect_id=self.artifacts.side_effect_vocab["[PAD]"],
            comp_masked_positions=comp_masked_positions,
        )
        masked = []
        masked_comp = []
        for row_index, token_rows in enumerate(sequence_items):
            current = [1.0 if pos in masked_positions[row_index] else 0.0 for pos in range(len(token_rows["tokens"]))]
            current_comp = [1.0 if pos in comp_masked_positions[row_index] else 0.0 for pos in range(len(token_rows["tokens"]))]
            while len(current) < packed["subset_drug_ids"].size(1):
                current.append(0.0)
                current_comp.append(0.0)
            masked.append(current)
            masked_comp.append(current_comp)
        packed["masked_positions"] = torch.tensor(masked, dtype=torch.float32)
        packed["comp_masked_positions"] = torch.tensor(masked_comp, dtype=torch.float32)
        return packed


class HGPMRegimeCollator:
    """Observed drug-regime collator with HGPM pairwise metadata."""

    def __init__(
        self,
        artifacts,
        *,
        use_exist_input_in_formal: bool = True,
        use_comp_input_in_formal: bool = True,
    ) -> None:
        self.artifacts = artifacts
        self.use_exist_input_in_formal = bool(use_exist_input_in_formal)
        self.use_comp_input_in_formal = bool(use_comp_input_in_formal)
        self.max_subset_size = 4
        self.max_comp_slots = 3

    def __call__(self, batch: list[dict]) -> dict:
        flat_sequences = []
        record_index = []
        target_positions = []
        for batch_record_index, record in enumerate(batch):
            for sequence in record["center_sequences"]:
                tokens = [
                    _encode_record_token(
                        token,
                        self.artifacts.drug_vocab,
                        max_subset_size=self.max_subset_size,
                        max_comp_slots=self.max_comp_slots,
                        use_exist_input_in_formal=self.use_exist_input_in_formal,
                        use_comp_input_in_formal=self.use_comp_input_in_formal,
                        token_type_id=1,
                    )
                    for token in sequence["tokens"]
                ]
                flat_sequences.append(
                    {
                        "anchor_id": self.artifacts.drug_vocab.get(sequence["center_drug_id"], self.artifacts.drug_vocab["[UNK]"]),
                        "se_id": self.artifacts.encoder_side_effect_vocab.get(sequence["side_effect_id"], self.artifacts.encoder_side_effect_vocab["[UNK]"]),
                        "tokens": tokens,
                    }
                )
                record_index.append(batch_record_index)
                target_positions.append(int(sequence["target_token_index"]))
        packed = _pack_sequence_items(
            flat_sequences,
            pad_drug_id=self.artifacts.drug_vocab["[PAD]"],
            pad_side_effect_id=self.artifacts.encoder_side_effect_vocab["[PAD]"],
        )
        batch_side_effect_ids = [
            self.artifacts.record_side_effect_vocab.get(record["side_effect_id"], self.artifacts.record_side_effect_vocab["[UNK]"])
            for record in batch
        ]
        batch_labels = [float(record["label"]) for record in batch]
        batch_record_ids = [record["record_id"] for record in batch]
        batch_drug_sets = [list(record["drug_set"]) for record in batch]
        max_record_drug_count = max(len(drug_set) for drug_set in batch_drug_sets) if batch_drug_sets else 1
        batch_drug_set_ids = []
        batch_drug_set_mask = []
        for drug_set in batch_drug_sets:
            encoded = [self.artifacts.drug_vocab.get(drug_id, self.artifacts.drug_vocab["[UNK]"]) for drug_id in drug_set]
            mask = [1.0] * len(encoded)
            while len(encoded) < max_record_drug_count:
                encoded.append(self.artifacts.drug_vocab["[PAD]"])
                mask.append(0.0)
            batch_drug_set_ids.append(encoded)
            batch_drug_set_mask.append(mask)
        packed.update(
            {
                "target_positions": torch.tensor(target_positions, dtype=torch.long),
                "record_index": torch.tensor(record_index, dtype=torch.long),
                "record_side_effect_ids": torch.tensor(batch_side_effect_ids, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.float32),
                "record_ids": batch_record_ids,
                "drug_sets": batch_drug_sets,
                "record_drug_set_ids": torch.tensor(batch_drug_set_ids, dtype=torch.long),
                "record_drug_set_mask": torch.tensor(batch_drug_set_mask, dtype=torch.float32),
            }
        )
        return packed


def _pack_sequence_items(
    sequence_items: list[dict],
    *,
    pad_drug_id: int,
    pad_side_effect_id: int,
    comp_masked_positions: list[set[int]] | None = None,
) -> dict:
    max_seq_len = max(len(item["tokens"]) for item in sequence_items)
    max_subset_size = max(len(token["subset_drug_ids"]) for item in sequence_items for token in item["tokens"])
    max_comp_slots = max(len(token["comp_ids"]) for item in sequence_items for token in item["tokens"])

    subset_ids = []
    subset_mask = []
    exist_ids = []
    exist_source_ids = []
    order_ids = []
    comp_ids = []
    comp_mask = []
    token_type_ids = []
    seq_mask = []
    center_token_mask = []
    anchor_ids = []
    se_ids = []
    edge_direction_ids = []
    overlap_bucket_ids = []
    comp_pair_bucket_ids = []
    sibling_bias_masks = []
    primary_comp_ids = []
    exist_target_ids = []

    def _get_comp_mask_set(item_index: int, comp_masked_positions: list[set[int]] | None) -> set[int]:
        if comp_masked_positions is None:
            return set()
        return comp_masked_positions[item_index]

    for item_index, item in enumerate(sequence_items):
        token_rows = []
        token_masks = []
        token_exist = []
        token_order = []
        token_comp = []
        token_comp_mask = []
        token_type = []
        token_primary_comp = []
        token_exist_targets = []
        token_center_mask = []
        comp_mask_set = _get_comp_mask_set(item_index, comp_masked_positions)
        for token in item["tokens"]:
            current_subset_ids = list(token["subset_drug_ids"])
            current_subset_mask = list(token["subset_mask"])
            while len(current_subset_ids) < max_subset_size:
                current_subset_ids.append(pad_drug_id)
                current_subset_mask.append(0.0)
            current_comp_ids = list(token["comp_ids"])
            current_comp_mask = list(token["comp_mask"])
            current_pos = len(token_rows)
            if current_pos in comp_mask_set:
                current_comp_ids = [COMP_MASK] + [COMP_PAD] * (max_comp_slots - 1)
                current_comp_mask = [1.0] + [0.0] * (max_comp_slots - 1)
            while len(current_comp_ids) < max_comp_slots:
                current_comp_ids.append(COMP_PAD)
                current_comp_mask.append(0.0)
            token_rows.append(current_subset_ids)
            token_masks.append(current_subset_mask)
            token_exist.append(int(token["exist_ids"]))
            token_order.append(int(token["order_ids"]))
            token_comp.append(current_comp_ids)
            token_comp_mask.append(current_comp_mask)
            token_type.append(int(token["token_type_ids"]))
            token_primary_comp.append(_primary_comp_id(token["raw_token"]))
            token_exist_targets.append(int(token["raw_token"].get("exist", token["raw_token"].get("exist_label", 0)) or 0))
            current_valid_subset_ids = [
                int(drug_id)
                for drug_id, keep in zip(token["subset_drug_ids"], token["subset_mask"])
                if float(keep) > 0
            ]
            token_center_mask.append(
                1.0 if len(current_valid_subset_ids) == 1 and current_valid_subset_ids[0] == int(item["anchor_id"]) else 0.0
            )

        current_len = len(item["tokens"])
        while len(token_rows) < max_seq_len:
            token_rows.append([pad_drug_id] * max_subset_size)
            token_masks.append([0.0] * max_subset_size)
            token_exist.append(EXIST_MASK)
            token_order.append(0)
            token_comp.append([COMP_PAD] * max_comp_slots)
            token_comp_mask.append([0.0] * max_comp_slots)
            token_type.append(0)
            token_primary_comp.append(PRIMARY_COMP_LABEL_TO_ID["NO_COMP"])
            token_exist_targets.append(0)
            token_center_mask.append(0.0)

        edge_direction, overlap_bucket, comp_pair_bucket, sibling_mask = _build_relation_tensors(
            item["tokens"],
            masked_comp_token_positions=comp_mask_set,
        )
        padded_square = torch.zeros((max_seq_len, max_seq_len), dtype=torch.long)
        padded_square[:current_len, :current_len] = edge_direction
        edge_direction_ids.append(padded_square)
        padded_square = torch.zeros((max_seq_len, max_seq_len), dtype=torch.long)
        padded_square[:current_len, :current_len] = overlap_bucket
        overlap_bucket_ids.append(padded_square)
        padded_square = torch.zeros((max_seq_len, max_seq_len), dtype=torch.long)
        padded_square[:current_len, :current_len] = comp_pair_bucket
        comp_pair_bucket_ids.append(padded_square)
        padded_float_square = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float32)
        padded_float_square[:current_len, :current_len] = sibling_mask
        sibling_bias_masks.append(padded_float_square)

        subset_ids.append(token_rows)
        subset_mask.append(token_masks)
        exist_ids.append(token_exist)
        exist_source_ids.append([_exist_source_id(token["raw_token"]) for token in item["tokens"]] + [3] * (max_seq_len - current_len))
        order_ids.append(token_order)
        comp_ids.append(token_comp)
        comp_mask.append(token_comp_mask)
        token_type_ids.append(token_type)
        primary_comp_ids.append(token_primary_comp)
        exist_target_ids.append(token_exist_targets)
        center_token_mask.append(token_center_mask)
        seq_mask.append([1.0] * current_len + [0.0] * (max_seq_len - current_len))
        anchor_ids.append(item["anchor_id"])
        se_ids.append(item.get("se_id", pad_side_effect_id))

    return {
        "subset_drug_ids": torch.tensor(subset_ids, dtype=torch.long),
        "subset_mask": torch.tensor(subset_mask, dtype=torch.float32),
        "exist_ids": torch.tensor(exist_ids, dtype=torch.long),
        "exist_source_ids": torch.tensor(exist_source_ids, dtype=torch.long),
        "order_ids": torch.tensor(order_ids, dtype=torch.long),
        "comp_ids": torch.tensor(comp_ids, dtype=torch.long),
        "comp_mask": torch.tensor(comp_mask, dtype=torch.float32),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "primary_comp_ids": torch.tensor(primary_comp_ids, dtype=torch.long),
        "exist_target_ids": torch.tensor(exist_target_ids, dtype=torch.long),
        "center_token_mask": torch.tensor(center_token_mask, dtype=torch.float32),
        "seq_mask": torch.tensor(seq_mask, dtype=torch.float32),
        "anchor_ids": torch.tensor(anchor_ids, dtype=torch.long),
        "se_ids": torch.tensor(se_ids, dtype=torch.long),
        "edge_direction_ids": torch.stack(edge_direction_ids, dim=0),
        "overlap_bucket_ids": torch.stack(overlap_bucket_ids, dim=0),
        "comp_pair_bucket_ids": torch.stack(comp_pair_bucket_ids, dim=0),
        "sibling_bias_mask": torch.stack(sibling_bias_masks, dim=0),
    }


class HGPMDrugEncoder(nn.Module):
    """Drug-side HGPM encoder over centered local regime DAGs."""

    def __init__(
        self,
        *,
        num_drugs: int,
        num_side_effects: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        drug_feature_table: torch.Tensor,
        use_drug_id_residual: bool = True,
        comp_vocab_size: int = 7,
        exist_vocab_size: int = 3,
        exist_source_vocab_size: int = 4,
        order_vocab_size: int = 10,
        token_type_vocab_size: int = 4,
        use_side_effect_in_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_drug_id_residual = bool(use_drug_id_residual)
        self.use_side_effect_in_encoder = bool(use_side_effect_in_encoder)
        self.register_buffer("drug_feature_table", drug_feature_table.detach().clone(), persistent=True)
        self.drug_embedding = nn.Embedding(num_drugs, hidden_dim, padding_idx=0)
        self.anchor_embedding = nn.Embedding(num_drugs, hidden_dim, padding_idx=0)
        self.side_effect_embedding = nn.Embedding(num_side_effects, hidden_dim, padding_idx=0)
        self.exist_embedding = nn.Embedding(exist_vocab_size, hidden_dim)
        self.exist_source_embedding = nn.Embedding(exist_source_vocab_size, hidden_dim)
        self.order_embedding = nn.Embedding(order_vocab_size, hidden_dim, padding_idx=0)
        self.comp_embedding = nn.Embedding(comp_vocab_size, hidden_dim, padding_idx=0)
        self.token_type_embedding = nn.Embedding(token_type_vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(256, hidden_dim)
        self.smiles_projection = nn.Linear(int(drug_feature_table.size(1)), hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_direction_bias = nn.Embedding(3, num_heads)
        self.overlap_bucket_bias = nn.Embedding(5, num_heads)
        self.comp_pair_bias = nn.Embedding((PRIMARY_COMP_MASK_ID + 1) * len(PRIMARY_COMP_LABEL_TO_ID) + 1, num_heads)
        self.sibling_bias = nn.Parameter(torch.zeros(num_heads))
        nn.init.zeros_(self.edge_direction_bias.weight)
        nn.init.zeros_(self.overlap_bucket_bias.weight)
        nn.init.zeros_(self.comp_pair_bias.weight)
        self.blocks = nn.ModuleList(
            [RelationAwareTransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
        return (values * mask.unsqueeze(-1)).sum(dim=dim) / denom

    def token_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raw_features = self.drug_feature_table[batch["subset_drug_ids"]]
        subset_emb = self._masked_mean(self.smiles_projection(raw_features), batch["subset_mask"], dim=2)
        if self.use_drug_id_residual:
            subset_emb = subset_emb + self._masked_mean(self.drug_embedding(batch["subset_drug_ids"]), batch["subset_mask"], dim=2)
        exist_emb = self.exist_embedding(batch["exist_ids"].clamp(min=0, max=self.exist_embedding.num_embeddings - 1))
        exist_source_emb = self.exist_source_embedding(batch["exist_source_ids"].clamp(min=0, max=self.exist_source_embedding.num_embeddings - 1))
        order_emb = self.order_embedding(batch["order_ids"].clamp(min=0, max=self.order_embedding.num_embeddings - 1))
        comp_emb = self._masked_mean(self.comp_embedding(batch["comp_ids"]), batch["comp_mask"], dim=2)
        token_type_emb = self.token_type_embedding(batch["token_type_ids"])
        anchor_emb = self.anchor_embedding(batch["anchor_ids"]).unsqueeze(1).expand_as(subset_emb)
        seq_len = batch["subset_drug_ids"].size(1)
        position_ids = torch.arange(seq_len, device=batch["subset_drug_ids"].device).unsqueeze(0).expand(batch["subset_drug_ids"].size(0), seq_len)
        pos_emb = self.position_embedding(position_ids)
        hidden = self.token_mlp(torch.cat([subset_emb, anchor_emb, exist_emb, exist_source_emb, order_emb, comp_emb, token_type_emb], dim=-1)) + pos_emb
        if self.use_side_effect_in_encoder:
            se_emb = self.side_effect_embedding(batch["se_ids"]).unsqueeze(1).expand_as(subset_emb)
            hidden = hidden + se_emb
        return hidden

    def build_attention_bias(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        edge = self.edge_direction_bias(batch["edge_direction_ids"]).permute(0, 3, 1, 2)
        overlap = self.overlap_bucket_bias(batch["overlap_bucket_ids"]).permute(0, 3, 1, 2)
        comp = self.comp_pair_bias(batch["comp_pair_bucket_ids"]).permute(0, 3, 1, 2)
        sibling = batch["sibling_bias_mask"].unsqueeze(1) * self.sibling_bias.view(1, -1, 1, 1)
        return edge + overlap + comp + sibling

    def encode_from_hidden(self, hidden: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        attn_bias = self.build_attention_bias(batch)
        for block in self.blocks:
            hidden = block(hidden, batch["seq_mask"], attn_bias)
        hidden = self.norm(hidden)
        pooled = self._masked_mean(hidden, batch["seq_mask"], dim=1)
        return hidden, pooled

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.token_embeddings(batch)
        return self.encode_from_hidden(hidden, batch)

    def raw_subset_means(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raw_features = self.drug_feature_table[batch["subset_drug_ids"]]
        return self._masked_mean(raw_features, batch["subset_mask"], dim=2)


class HGPMSemanticPretrainModel(nn.Module):
    def __init__(
        self,
        encoder: HGPMDrugEncoder,
        *,
        hidden_dim: int,
        raw_feature_dim: int,
        dropout: float,
        teacher_ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.teacher_ema_decay = float(teacher_ema_decay)
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.exist_head = nn.Linear(hidden_dim, 2)
        self.comp_head = nn.Linear(hidden_dim, len(PRIMARY_COMP_LABEL_TO_ID))
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_prediction_head = copy.deepcopy(self.prediction_head)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher_encoder.eval()
        self.teacher_prediction_head.eval()
        for module in (self.teacher_encoder, self.teacher_prediction_head):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, ema_decay: float | None = None) -> None:
        decay = float(self.teacher_ema_decay if ema_decay is None else ema_decay)
        for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
            teacher_buffer.copy_(student_buffer)
        for teacher_param, student_param in zip(self.teacher_prediction_head.parameters(), self.prediction_head.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(self.teacher_prediction_head.buffers(), self.prediction_head.buffers()):
            teacher_buffer.copy_(student_buffer)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.encoder.token_embeddings(batch)
        mask_positions = batch["masked_positions"] > 0
        if mask_positions.any():
            hidden = hidden.clone()
            hidden[mask_positions] = self.mask_embedding
        encoded, _ = self.encoder.encode_from_hidden(hidden, batch)
        predicted = self.prediction_head(encoded)
        with torch.no_grad():
            teacher_encoded, _ = self.teacher_encoder.encode(batch)
            teacher_targets = self.teacher_prediction_head(teacher_encoded).detach()
        return {
            "predicted_embeddings": predicted,
            "teacher_targets": teacher_targets,
            "exist_logits": self.exist_head(encoded),
            "comp_logits": self.comp_head(encoded),
        }


class HGPMRegimeModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: HGPMDrugEncoder,
        num_record_side_effects: int,
        hidden_dim: int,
        record_side_effect_dim: int,
        dropout: float,
        pooling: str = "mean",
        center_representation: str = "center_only",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unsupported record pooling: {pooling}")
        if center_representation not in {"center_only", "target_only", "target_plus_summary"}:
            raise ValueError(f"Unsupported center representation mode: {center_representation}")
        self.encoder = encoder
        self.pooling = pooling
        self.center_representation = center_representation
        self.record_side_effect_embedding = nn.Embedding(num_record_side_effects, record_side_effect_dim, padding_idx=0)
        self.pool_query = nn.Linear(record_side_effect_dim, hidden_dim) if pooling == "attention" else None
        self.center_repr_mlp = (
            nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if center_representation == "target_plus_summary"
            else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + record_side_effect_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool_target_states(
        self,
        target_states: torch.Tensor,
        record_index: torch.Tensor,
        num_records: int,
        side_effect_repr: torch.Tensor,
    ) -> torch.Tensor:
        hidden_dim = target_states.size(-1)
        pooled = target_states.new_zeros((num_records, hidden_dim))
        if self.pooling == "mean":
            pooled.index_add_(0, record_index, target_states)
            counts = target_states.new_zeros((num_records, 1))
            counts.index_add_(0, record_index, torch.ones((target_states.size(0), 1), device=target_states.device, dtype=target_states.dtype))
            return pooled / counts.clamp_min(1.0)
        if self.pool_query is None:
            raise RuntimeError("Attention pooling query is missing.")
        for record_id in range(num_records):
            record_mask = record_index == record_id
            record_states = target_states[record_mask]
            if record_states.size(0) == 0:
                continue
            query = self.pool_query(side_effect_repr[record_id]).unsqueeze(0)
            scores = (record_states * query).sum(dim=-1)
            weights = torch.softmax(scores, dim=0).unsqueeze(-1)
            pooled[record_id] = (record_states * weights).sum(dim=0)
        return pooled

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token_states, sequence_pooled = self.encoder.encode(batch)
        if self.center_representation == "center_only":
            center_mask = batch["center_token_mask"] > 0
            if not center_mask.any():
                raise RuntimeError("center_only representation requested but no center singleton token was found.")
            center_states = (token_states * center_mask.unsqueeze(-1)).sum(dim=1)
            center_counts = center_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            target_states = center_states / center_counts
        else:
            seq_indices = torch.arange(token_states.size(0), device=token_states.device)
            target_states = token_states[seq_indices, batch["target_positions"]]
        if self.center_representation == "target_plus_summary":
            if self.center_repr_mlp is None:
                raise RuntimeError("center_repr_mlp is not initialized.")
            target_states = self.center_repr_mlp(
                torch.cat(
                    [target_states, sequence_pooled, target_states * sequence_pooled, target_states - sequence_pooled],
                    dim=-1,
                )
            )
        side_effect_repr = self.record_side_effect_embedding(batch["record_side_effect_ids"])
        pooled = self._pool_target_states(target_states, batch["record_index"], int(batch["labels"].size(0)), side_effect_repr)
        logits = self.classifier(torch.cat([pooled, side_effect_repr], dim=-1)).squeeze(-1)
        return logits


