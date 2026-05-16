from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch


EXIST_ABSENT = 0
EXIST_PRESENT = 1
EXIST_MASK = 2

COMP_PAD = 0
COMP_COMP = 1
COMP_EMER = 2
COMP_INHIB = 3
COMP_UNKNOWN = 4
COMP_MASK = 5
COMP_NO_COMP = 6

COMP_LABEL_TO_ID = {
    "COMP": COMP_COMP,
    "EMER": COMP_EMER,
    "INHIB": COMP_INHIB,
    "UNKNOWN": COMP_UNKNOWN,
}
COMP_ID_TO_LABEL = {value: key for key, value in COMP_LABEL_TO_ID.items()}


@dataclass
class HoddiPretrainArtifacts:
    drug_vocab: dict[str, int]
    side_effect_vocab: dict[str, int]
    max_subset_size: int = 4
    max_comp_slots: int = 3


def build_pretrain_artifacts(rows: list[dict], base_drug_vocab: dict[str, int]) -> HoddiPretrainArtifacts:
    drug_vocab = {"[PAD]": 0, "[UNK]": 1}
    for drug in sorted(base_drug_vocab):
        if drug not in drug_vocab:
            drug_vocab[drug] = len(drug_vocab)

    side_effect_vocab = {"[PAD]": 0, "[UNK]": 1}
    for side_effect_id in sorted({str(row["side_effect_id"]) for row in rows}):
        side_effect_vocab[side_effect_id] = len(side_effect_vocab)
    max_subset_size = max(
        4,
        max((len(token.get("subset_drug_ids", [])) for row in rows for token in row.get("tokens", [])), default=4),
    )
    return HoddiPretrainArtifacts(
        drug_vocab=drug_vocab,
        side_effect_vocab=side_effect_vocab,
        max_subset_size=max_subset_size,
    )


def _sample_seed(base_seed: int, sample_idx: int, stage_name: str) -> int:
    return (base_seed * 1000003 + sample_idx * 1009 + sum(ord(ch) for ch in stage_name)) & 0xFFFFFFFF


def _encode_original_comp(token: dict, artifacts: HoddiPretrainArtifacts) -> list[int]:
    encoded = [COMP_LABEL_TO_ID[label] for label in token["composition_types"][: artifacts.max_comp_slots]]
    while len(encoded) < artifacts.max_comp_slots:
        encoded.append(COMP_PAD)
    return encoded


def _sample_stage1_mask_indices(tokens: list[dict], exist_mask_rate: float, rng: random.Random | random.Random, strategy: str) -> list[int]:
    eligible_indices = list(range(len(tokens)))
    if not eligible_indices:
        return []

    target_count = max(1, int(round(len(tokens) * exist_mask_rate)))
    target_count = min(target_count, len(eligible_indices))

    if strategy == "balanced_binary":
        positive_indices = [index for index, token in enumerate(tokens) if int(token.get("exist", token.get("existence", 0))) == 1]
        negative_indices = [index for index, token in enumerate(tokens) if int(token.get("exist", token.get("existence", 0))) == 0]

        if positive_indices and negative_indices:
            per_class_target = max(1, target_count // 2)
            positive_target = min(len(positive_indices), per_class_target)
            negative_target = min(len(negative_indices), per_class_target)

            masked_indices = []
            masked_indices.extend(rng.sample(positive_indices, k=positive_target))
            masked_indices.extend(rng.sample(negative_indices, k=negative_target))

            if target_count % 2 == 1:
                remaining_positive = [index for index in positive_indices if index not in set(masked_indices)]
                remaining_negative = [index for index in negative_indices if index not in set(masked_indices)]
                if len(remaining_positive) >= len(remaining_negative) and remaining_positive:
                    masked_indices.append(rng.choice(remaining_positive))
                elif remaining_negative:
                    masked_indices.append(rng.choice(remaining_negative))
            return sorted(masked_indices)

        if positive_indices:
            return sorted(rng.sample(positive_indices, k=min(len(positive_indices), target_count)))

        if negative_indices:
            return sorted(rng.sample(negative_indices, k=1))

    masked_indices = [index for index in eligible_indices if rng.random() < exist_mask_rate]
    if not masked_indices:
        masked_indices = [rng.choice(eligible_indices)]
    if len(masked_indices) > target_count:
        masked_indices = rng.sample(masked_indices, k=target_count)
    return sorted(masked_indices)


def _tensorize_token(
    token: dict,
    artifacts: HoddiPretrainArtifacts,
    input_existence: int,
    input_comp_ids: list[int],
) -> dict:
    subset_ids = [artifacts.drug_vocab.get(drug, artifacts.drug_vocab["[UNK]"]) for drug in token["subset_drug_ids"][: artifacts.max_subset_size]]
    subset_mask = [1.0] * len(subset_ids)
    while len(subset_ids) < artifacts.max_subset_size:
        subset_ids.append(artifacts.drug_vocab["[PAD]"])
        subset_mask.append(0.0)

    valid_comp_slots = len(token["composition_types"])
    comp_ids = list(input_comp_ids[: artifacts.max_comp_slots])
    comp_slot_mask = [1.0] * min(valid_comp_slots, artifacts.max_comp_slots)
    while len(comp_ids) < artifacts.max_comp_slots:
        comp_ids.append(COMP_PAD)
    while len(comp_slot_mask) < artifacts.max_comp_slots:
        comp_slot_mask.append(0.0)

    if valid_comp_slots == 0:
        comp_ids = [COMP_NO_COMP, COMP_PAD, COMP_PAD]
        comp_slot_mask = [1.0, 0.0, 0.0]

    return {
        "subset_drug_ids": subset_ids,
        "subset_mask": subset_mask,
        "existence_input": input_existence,
        "order_id": int(token["order"]),
        "comp_input_ids": comp_ids,
        "comp_slot_mask": comp_slot_mask,
    }


def _build_stage1_sample(
    sample: dict,
    artifacts: HoddiPretrainArtifacts,
    *,
    exist_mask_rate: float,
    mask_sampling_strategy: str,
    seed: int,
    deterministic: bool,
) -> dict:
    rng = random.Random(_sample_seed(seed, int(sample["sample_idx"]), "stage1")) if deterministic else random
    tokens = sample["tokens"]
    masked_indices = _sample_stage1_mask_indices(tokens, exist_mask_rate, rng, mask_sampling_strategy)
    masked_index_set = set(masked_indices)
    masked_subset_strs = {tokens[index]["subset_str"] for index in masked_indices}

    masked_comp_slots: dict[int, set[int]] = {index: set() for index in range(len(tokens))}
    for index in masked_indices:
        for slot_id, _ in enumerate(tokens[index]["composition_types"]):
            masked_comp_slots[index].add(slot_id)
    for child_index, token in enumerate(tokens):
        for slot_id, parent_subset in enumerate(token["parent_subsets"]):
            if parent_subset in masked_subset_strs:
                masked_comp_slots[child_index].add(slot_id)

    encoded_tokens = []
    exist_targets = []
    exist_loss_mask = []
    debug_original = []
    debug_masked = []
    for index, token in enumerate(tokens):
        raw_comp_targets = _encode_original_comp(token, artifacts)
        input_comp_ids = list(raw_comp_targets)
        for slot_id in masked_comp_slots[index]:
            if slot_id < len(input_comp_ids) and input_comp_ids[slot_id] != COMP_PAD:
                input_comp_ids[slot_id] = COMP_MASK

        token_exist = int(token.get("exist", token.get("existence", 0)))
        input_existence = EXIST_MASK if index in masked_index_set else token_exist
        encoded_tokens.append(_tensorize_token(token, artifacts, input_existence=input_existence, input_comp_ids=input_comp_ids))
        exist_targets.append(float(token_exist))
        exist_loss_mask.append(1.0 if index in masked_index_set else 0.0)
        debug_original.append(
            {
                "subset_str": token["subset_str"],
                "order": token["order"],
                "existence": token_exist,
                "composition_types": list(token["composition_types"]),
            }
        )
        debug_masked.append(
            {
                "subset_str": token["subset_str"],
                "order": token["order"],
                "existence_input": "MASK_EXIST" if input_existence == EXIST_MASK else int(input_existence),
                "composition_input": [
                    "MASK_COMP" if value == COMP_MASK else
                    "NO_COMP" if value == COMP_NO_COMP else
                    COMP_ID_TO_LABEL.get(value, "PAD")
                    for value in input_comp_ids
                    if value != COMP_PAD
                ],
            }
        )

    return {
        "center_id": artifacts.drug_vocab.get(sample["center_drug_id"], artifacts.drug_vocab["[UNK]"]),
        "side_effect_id": artifacts.side_effect_vocab.get(sample["side_effect_id"], artifacts.side_effect_vocab["[UNK]"]),
        "tokens": encoded_tokens,
        "exist_targets": exist_targets,
        "exist_loss_mask": exist_loss_mask,
        "debug": {
            "sample_key": {
                "center_drug_id": sample["center_drug_id"],
                "side_effect_id": sample["side_effect_id"],
                "view_id": int(sample.get("view_id", 0)),
            },
            "original_tokens": debug_original,
            "masked_tokens": debug_masked,
            "masked_indices": sorted(masked_indices),
        },
    }


def _build_stage2_sample(
    sample: dict,
    artifacts: HoddiPretrainArtifacts,
    *,
    comp_mask_rate: float,
    seed: int,
    deterministic: bool,
) -> dict:
    rng = random.Random(_sample_seed(seed, int(sample["sample_idx"]), "stage2")) if deterministic else random
    tokens = sample["tokens"]
    eligible_indices = [index for index, token in enumerate(tokens) if int(token["order"]) >= 3 and len(token["composition_types"]) > 0]
    masked_indices = [index for index in eligible_indices if rng.random() < comp_mask_rate]
    if not masked_indices and eligible_indices:
        masked_indices = [rng.choice(eligible_indices)]
    masked_index_set = set(masked_indices)

    encoded_tokens = []
    comp_targets = []
    comp_loss_mask = []
    debug_original = []
    debug_masked = []
    for index, token in enumerate(tokens):
        raw_comp_targets = _encode_original_comp(token, artifacts)
        input_comp_ids = list(raw_comp_targets)
        valid_slots = min(len(token["composition_types"]), artifacts.max_comp_slots)
        if index in masked_index_set:
            for slot_id in range(valid_slots):
                input_comp_ids[slot_id] = COMP_MASK

        encoded_tokens.append(
            _tensorize_token(
                token,
                artifacts,
                input_existence=int(token.get("exist", token.get("existence", 0))),
                input_comp_ids=input_comp_ids,
            )
        )
        comp_targets.append(raw_comp_targets)
        comp_loss_mask.append(
            [1.0 if index in masked_index_set and slot_id < valid_slots else 0.0 for slot_id in range(artifacts.max_comp_slots)]
        )
        debug_original.append(
            {
                "subset_str": token["subset_str"],
                "order": token["order"],
                "existence": int(token.get("exist", token.get("existence", 0))),
                "composition_types": list(token["composition_types"]),
            }
        )
        debug_masked.append(
            {
                "subset_str": token["subset_str"],
                "order": token["order"],
                "existence_input": int(token.get("exist", token.get("existence", 0))),
                "composition_input": [
                    "MASK_COMP" if value == COMP_MASK else
                    "NO_COMP" if value == COMP_NO_COMP else
                    COMP_ID_TO_LABEL.get(value, "PAD")
                    for value in input_comp_ids
                    if value != COMP_PAD
                ],
            }
        )

    return {
        "center_id": artifacts.drug_vocab.get(sample["center_drug_id"], artifacts.drug_vocab["[UNK]"]),
        "side_effect_id": artifacts.side_effect_vocab.get(sample["side_effect_id"], artifacts.side_effect_vocab["[UNK]"]),
        "tokens": encoded_tokens,
        "comp_targets": comp_targets,
        "comp_loss_mask": comp_loss_mask,
        "debug": {
            "sample_key": {
                "center_drug_id": sample["center_drug_id"],
                "side_effect_id": sample["side_effect_id"],
                "view_id": int(sample.get("view_id", 0)),
            },
            "original_tokens": debug_original,
            "masked_tokens": debug_masked,
            "masked_indices": sorted(masked_indices),
        },
    }


def _pad_and_stack(
    processed_batch: list[dict],
    artifacts: HoddiPretrainArtifacts,
    *,
    include_exist_targets: bool = False,
    include_comp_targets: bool = False,
    disable_side_effect_conditioning: bool = False,
) -> dict:
    max_seq_len = max(len(item["tokens"]) for item in processed_batch)

    subset_ids = []
    subset_mask = []
    exist_ids = []
    order_ids = []
    comp_ids = []
    comp_mask = []
    token_type_ids = []
    seq_mask = []
    center_ids = []
    side_effect_ids = []
    debug_items = []

    exist_targets = []
    exist_loss_mask = []
    comp_targets = []
    comp_loss_mask = []

    pad_token = {
        "subset_drug_ids": [artifacts.drug_vocab["[PAD]"]] * artifacts.max_subset_size,
        "subset_mask": [0.0] * artifacts.max_subset_size,
        "existence_input": EXIST_ABSENT,
        "order_id": 0,
        "comp_input_ids": [COMP_PAD] * artifacts.max_comp_slots,
        "comp_slot_mask": [0.0] * artifacts.max_comp_slots,
    }

    for item in processed_batch:
        current_tokens = list(item["tokens"])
        current_len = len(current_tokens)
        while len(current_tokens) < max_seq_len:
            current_tokens.append(copy.deepcopy(pad_token))

        subset_ids.append([token["subset_drug_ids"] for token in current_tokens])
        subset_mask.append([token["subset_mask"] for token in current_tokens])
        exist_ids.append([token["existence_input"] for token in current_tokens])
        order_ids.append([token["order_id"] for token in current_tokens])
        comp_ids.append([token["comp_input_ids"] for token in current_tokens])
        comp_mask.append([token["comp_slot_mask"] for token in current_tokens])
        token_type_ids.append([1] * max_seq_len)
        seq_mask.append([1.0] * current_len + [0.0] * (max_seq_len - current_len))
        center_ids.append(item["center_id"])
        side_effect_ids.append(artifacts.side_effect_vocab["[PAD]"] if disable_side_effect_conditioning else item["side_effect_id"])
        debug_items.append(item["debug"])

        if include_exist_targets:
            exist_targets.append(item["exist_targets"] + [0.0] * (max_seq_len - current_len))
            exist_loss_mask.append(item["exist_loss_mask"] + [0.0] * (max_seq_len - current_len))

        if include_comp_targets:
            padded_targets = list(item["comp_targets"])
            padded_loss_mask = list(item["comp_loss_mask"])
            while len(padded_targets) < max_seq_len:
                padded_targets.append([COMP_PAD] * artifacts.max_comp_slots)
                padded_loss_mask.append([0.0] * artifacts.max_comp_slots)
            comp_targets.append(padded_targets)
            comp_loss_mask.append(padded_loss_mask)

    batch = {
        "subset_drug_ids": torch.tensor(subset_ids, dtype=torch.long),
        "subset_mask": torch.tensor(subset_mask, dtype=torch.float32),
        "exist_ids": torch.tensor(exist_ids, dtype=torch.long),
        "order_ids": torch.tensor(order_ids, dtype=torch.long),
        "comp_ids": torch.tensor(comp_ids, dtype=torch.long),
        "comp_mask": torch.tensor(comp_mask, dtype=torch.float32),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "seq_mask": torch.tensor(seq_mask, dtype=torch.float32),
        "anchor_ids": torch.tensor(center_ids, dtype=torch.long),
        "se_ids": torch.tensor(side_effect_ids, dtype=torch.long),
        "debug_items": debug_items,
    }
    if include_exist_targets:
        batch["exist_targets"] = torch.tensor(exist_targets, dtype=torch.float32)
        batch["exist_loss_mask"] = torch.tensor(exist_loss_mask, dtype=torch.float32)
    if include_comp_targets:
        batch["comp_targets"] = torch.tensor(comp_targets, dtype=torch.long)
        batch["comp_loss_mask"] = torch.tensor(comp_loss_mask, dtype=torch.float32)
    return batch


class Stage1ExistenceMaskCollator:
    def __init__(
        self,
        artifacts: HoddiPretrainArtifacts,
        *,
        exist_mask_rate: float,
        mask_sampling_strategy: str,
        seed: int,
        deterministic: bool,
        disable_side_effect_conditioning: bool = False,
    ) -> None:
        self.artifacts = artifacts
        self.exist_mask_rate = exist_mask_rate
        self.mask_sampling_strategy = mask_sampling_strategy
        self.seed = seed
        self.deterministic = deterministic
        self.disable_side_effect_conditioning = bool(disable_side_effect_conditioning)

    def __call__(self, batch: list[dict]) -> dict:
        processed = [
            _build_stage1_sample(
                sample,
                self.artifacts,
                exist_mask_rate=self.exist_mask_rate,
                mask_sampling_strategy=self.mask_sampling_strategy,
                seed=self.seed,
                deterministic=self.deterministic,
            )
            for sample in batch
        ]
        return _pad_and_stack(
            processed,
            self.artifacts,
            include_exist_targets=True,
            disable_side_effect_conditioning=self.disable_side_effect_conditioning,
        )


class Stage2CompositionMaskCollator:
    def __init__(
        self,
        artifacts: HoddiPretrainArtifacts,
        *,
        comp_mask_rate: float,
        seed: int,
        deterministic: bool,
        disable_side_effect_conditioning: bool = False,
    ) -> None:
        self.artifacts = artifacts
        self.comp_mask_rate = comp_mask_rate
        self.seed = seed
        self.deterministic = deterministic
        self.disable_side_effect_conditioning = bool(disable_side_effect_conditioning)

    def __call__(self, batch: list[dict]) -> dict:
        processed = [
            _build_stage2_sample(
                sample,
                self.artifacts,
                comp_mask_rate=self.comp_mask_rate,
                seed=self.seed,
                deterministic=self.deterministic,
            )
            for sample in batch
        ]
        return _pad_and_stack(
            processed,
            self.artifacts,
            include_comp_targets=True,
            disable_side_effect_conditioning=self.disable_side_effect_conditioning,
        )
