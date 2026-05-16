from __future__ import annotations

import torch

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
)
from HGPM.data.drug.regime_dataset import HoddiRecordArtifacts, MASK_COMP, MASK_EXIST, NO_COMP


COMP_INPUT_MAP = {
    "COMP": COMP_COMP,
    "EMER": COMP_EMER,
    "INHIB": COMP_INHIB,
    "UNKNOWN": COMP_UNKNOWN,
    MASK_COMP: COMP_MASK,
    NO_COMP: COMP_NO_COMP,
}
EXIST_INPUT_MAP = {
    0: EXIST_ABSENT,
    1: EXIST_PRESENT,
    MASK_EXIST: EXIST_MASK,
}


def _encode_token(
    token: dict,
    artifacts: HoddiRecordArtifacts,
    *,
    token_type_id: int,
    use_exist_input_in_formal: bool = True,
    use_comp_input_in_formal: bool = True,
) -> dict:
    subset_ids = [artifacts.drug_vocab.get(drug, artifacts.drug_vocab["[UNK]"]) for drug in token["subset_drug_ids"]]
    subset_mask = [1.0] * len(subset_ids)
    comp_input_ids = [COMP_INPUT_MAP.get(value, COMP_MASK) for value in token["composition_types_input"]]
    comp_slot_mask = [1.0 if int(value) == 1 else 0.0 for value in token["composition_observed_mask"]]

    if not comp_input_ids:
        comp_input_ids = [COMP_NO_COMP]
        comp_slot_mask = [1.0]

    exist_id = EXIST_INPUT_MAP[token["exist_input"]] if use_exist_input_in_formal else EXIST_MASK
    if not use_comp_input_in_formal:
        comp_input_ids = [COMP_PAD] * len(comp_input_ids)
        comp_slot_mask = [0.0] * len(comp_slot_mask)

    return {
        "subset_drug_ids": subset_ids,
        "subset_mask": subset_mask,
        "exist_ids": exist_id,
        "order_ids": int(token["order"]),
        "comp_ids": comp_input_ids,
        "comp_mask": comp_slot_mask,
        "token_type_ids": token_type_id,
    }


class HoddiRecordTask1Collator:
    def __init__(
        self,
        artifacts: HoddiRecordArtifacts,
        *,
        use_target_centered_branch: bool = False,
        use_se_token: bool = False,
        disable_side_effect_conditioning: bool = False,
        use_exist_input_in_formal: bool = True,
        use_comp_input_in_formal: bool = True,
    ) -> None:
        self.artifacts = artifacts
        self.use_target_centered_branch = bool(use_target_centered_branch)
        self.use_se_token = bool(use_se_token)
        self.disable_side_effect_conditioning = bool(disable_side_effect_conditioning)
        self.use_exist_input_in_formal = bool(use_exist_input_in_formal)
        self.use_comp_input_in_formal = bool(use_comp_input_in_formal)

    def _make_se_token(self) -> dict:
        return {
            "subset_drug_ids": [self.artifacts.drug_vocab["[PAD]"]],
            "subset_mask": [0.0],
            "exist_ids": EXIST_MASK,
            "order_ids": 0,
            "comp_ids": [COMP_PAD],
            "comp_mask": [0.0],
            "token_type_ids": 3,
        }

    def _encode_sequence(
        self,
        sequence: dict,
        *,
        branch_type_id: int,
    ) -> tuple[list[dict], int]:
        encoded_tokens = [
            _encode_token(
                token,
                self.artifacts,
                token_type_id=branch_type_id,
                use_exist_input_in_formal=self.use_exist_input_in_formal,
                use_comp_input_in_formal=self.use_comp_input_in_formal,
            )
            for token in sequence["tokens"]
        ]
        target_position = int(sequence["target_token_index"])
        if self.use_se_token:
            encoded_tokens = [self._make_se_token(), *encoded_tokens]
            target_position += 1
        return encoded_tokens, target_position

    def __call__(self, batch: list[dict]) -> dict:
        flat_sequences: list[dict] = []
        record_index: list[int] = []
        target_positions: list[int] = []
        target_sequences: list[dict] = []
        target_target_positions: list[int] = []

        for batch_record_index, record in enumerate(batch):
            for sequence in record["center_sequences"]:
                encoded_tokens, encoded_target_position = self._encode_sequence(sequence, branch_type_id=1)
                flat_sequences.append(
                    {
                        "center_id": self.artifacts.drug_vocab.get(sequence["center_drug_id"], self.artifacts.drug_vocab["[UNK]"]),
                        "encoder_side_effect_id": self.artifacts.encoder_side_effect_vocab.get(
                            sequence["side_effect_id"],
                            self.artifacts.encoder_side_effect_vocab["[UNK]"],
                        ),
                        "tokens": encoded_tokens,
                        "target_position": encoded_target_position,
                        "debug": {
                            "center_drug_id": sequence["center_drug_id"],
                            "side_effect_id": sequence["side_effect_id"],
                            "target_position": encoded_target_position,
                            "tokens": sequence["tokens"][:12],
                        },
                    }
                )
                record_index.append(batch_record_index)
                target_positions.append(encoded_target_position)

            if self.use_target_centered_branch and record.get("target_center_sequence") is not None:
                encoded_tokens, encoded_target_position = self._encode_sequence(
                    record["target_center_sequence"],
                    branch_type_id=2,
                )
                target_sequences.append(
                    {
                        "anchor_id": self.artifacts.drug_vocab["[PAD]"],
                        "encoder_side_effect_id": self.artifacts.encoder_side_effect_vocab.get(
                            record["side_effect_id"],
                            self.artifacts.encoder_side_effect_vocab["[UNK]"],
                        ),
                        "tokens": encoded_tokens,
                        "target_position": encoded_target_position,
                        "debug": {
                            "side_effect_id": record["side_effect_id"],
                            "target_position": encoded_target_position,
                            "tokens": record["target_center_sequence"]["tokens"][:12],
                        },
                    }
                )
                target_target_positions.append(encoded_target_position)

        all_sequence_groups = [flat_sequences]
        if target_sequences:
            all_sequence_groups.append(target_sequences)
        max_seq_len = max(len(item["tokens"]) for group in all_sequence_groups for item in group)
        max_subset_size = max(len(token["subset_drug_ids"]) for group in all_sequence_groups for item in group for token in item["tokens"])
        max_comp_slots = max(len(token["comp_ids"]) for group in all_sequence_groups for item in group for token in item["tokens"])

        def _pack_sequences(sequence_items: list[dict]) -> dict:
            subset_ids = []
            subset_mask = []
            exist_ids = []
            order_ids = []
            comp_ids = []
            comp_mask = []
            token_type_ids = []
            seq_mask = []
            anchor_ids = []
            se_ids = []
            debug_items = []

            for item in sequence_items:
                padded_subset_ids = []
                padded_subset_mask = []
                padded_exist_ids = []
                padded_order_ids = []
                padded_comp_ids = []
                padded_comp_mask = []
                padded_token_type_ids = []
                current_len = len(item["tokens"])

                for token in item["tokens"]:
                    current_subset_ids = list(token["subset_drug_ids"])
                    current_subset_mask = list(token["subset_mask"])
                    while len(current_subset_ids) < max_subset_size:
                        current_subset_ids.append(self.artifacts.drug_vocab["[PAD]"])
                        current_subset_mask.append(0.0)

                    current_comp_ids = list(token["comp_ids"])
                    current_comp_mask = list(token["comp_mask"])
                    while len(current_comp_ids) < max_comp_slots:
                        current_comp_ids.append(COMP_PAD)
                        current_comp_mask.append(0.0)

                    padded_subset_ids.append(current_subset_ids)
                    padded_subset_mask.append(current_subset_mask)
                    padded_exist_ids.append(token["exist_ids"])
                    padded_order_ids.append(token["order_ids"])
                    padded_comp_ids.append(current_comp_ids)
                    padded_comp_mask.append(current_comp_mask)
                    padded_token_type_ids.append(token["token_type_ids"])

                while len(padded_subset_ids) < max_seq_len:
                    padded_subset_ids.append([self.artifacts.drug_vocab["[PAD]"]] * max_subset_size)
                    padded_subset_mask.append([0.0] * max_subset_size)
                    padded_exist_ids.append(EXIST_MASK)
                    padded_order_ids.append(0)
                    padded_comp_ids.append([COMP_PAD] * max_comp_slots)
                    padded_comp_mask.append([0.0] * max_comp_slots)
                    padded_token_type_ids.append(0)

                subset_ids.append(padded_subset_ids)
                subset_mask.append(padded_subset_mask)
                exist_ids.append(padded_exist_ids)
                order_ids.append(padded_order_ids)
                comp_ids.append(padded_comp_ids)
                comp_mask.append(padded_comp_mask)
                token_type_ids.append(padded_token_type_ids)
                seq_mask.append([1.0] * current_len + [0.0] * (max_seq_len - current_len))
                anchor_ids.append(item["center_id"] if "center_id" in item else item["anchor_id"])
                se_ids.append(self.artifacts.encoder_side_effect_vocab["[PAD]"] if self.disable_side_effect_conditioning else item["encoder_side_effect_id"])
                debug_items.append(item["debug"])

            return {
                "subset_drug_ids": torch.tensor(subset_ids, dtype=torch.long),
                "subset_mask": torch.tensor(subset_mask, dtype=torch.float32),
                "exist_ids": torch.tensor(exist_ids, dtype=torch.long),
                "order_ids": torch.tensor(order_ids, dtype=torch.long),
                "comp_ids": torch.tensor(comp_ids, dtype=torch.long),
                "comp_mask": torch.tensor(comp_mask, dtype=torch.float32),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "seq_mask": torch.tensor(seq_mask, dtype=torch.float32),
                "anchor_ids": torch.tensor(anchor_ids, dtype=torch.long),
                "se_ids": torch.tensor(se_ids, dtype=torch.long),
                "debug_items": debug_items,
            }

        node_packed = _pack_sequences(flat_sequences)
        target_packed = _pack_sequences(target_sequences) if target_sequences else None

        batch_side_effect_ids = [
            self.artifacts.record_side_effect_vocab["[PAD]"]
            if self.disable_side_effect_conditioning
            else self.artifacts.record_side_effect_vocab.get(record["side_effect_id"], self.artifacts.record_side_effect_vocab["[UNK]"])
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
        batch_side_effect_texts = [record["side_effect_id"] for record in batch]

        payload = {
            "node_subset_drug_ids": node_packed["subset_drug_ids"],
            "node_subset_mask": node_packed["subset_mask"],
            "node_exist_ids": node_packed["exist_ids"],
            "node_order_ids": node_packed["order_ids"],
            "node_comp_ids": node_packed["comp_ids"],
            "node_comp_mask": node_packed["comp_mask"],
            "node_token_type_ids": node_packed["token_type_ids"],
            "node_seq_mask": node_packed["seq_mask"],
            "node_anchor_ids": node_packed["anchor_ids"],
            "node_se_ids": node_packed["se_ids"],
            "target_positions": torch.tensor(target_positions, dtype=torch.long),
            "record_index": torch.tensor(record_index, dtype=torch.long),
            "record_side_effect_ids": torch.tensor(batch_side_effect_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.float32),
            "record_ids": batch_record_ids,
            "drug_sets": batch_drug_sets,
            "record_drug_set_ids": torch.tensor(batch_drug_set_ids, dtype=torch.long),
            "record_drug_set_mask": torch.tensor(batch_drug_set_mask, dtype=torch.float32),
            "record_side_effect_texts": batch_side_effect_texts,
            "debug_items": node_packed["debug_items"],
        }
        if target_packed is not None:
            payload.update(
                {
                    "target_subset_drug_ids": target_packed["subset_drug_ids"],
                    "target_subset_mask": target_packed["subset_mask"],
                    "target_exist_ids": target_packed["exist_ids"],
                    "target_order_ids": target_packed["order_ids"],
                    "target_comp_ids": target_packed["comp_ids"],
                    "target_comp_mask": target_packed["comp_mask"],
                    "target_token_type_ids": target_packed["token_type_ids"],
                    "target_seq_mask": target_packed["seq_mask"],
                    "target_anchor_ids": target_packed["anchor_ids"],
                    "target_se_ids": target_packed["se_ids"],
                    "target_target_positions": torch.tensor(target_target_positions, dtype=torch.long),
                    "target_debug_items": target_packed["debug_items"],
                }
            )
        return payload
