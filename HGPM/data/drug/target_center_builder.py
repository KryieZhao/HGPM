from __future__ import annotations

import itertools


MASK_EXIST = "MASK_EXIST"
MASK_COMP = "MASK_COMP"
NO_COMP = "NO_COMP"


def _comp_label(parent_exist: int, child_exist: int) -> str:
    if parent_exist == 1 and child_exist == 1:
        return "COMP"
    if parent_exist == 0 and child_exist == 1:
        return "EMER"
    if parent_exist == 1 and child_exist == 0:
        return "INHIB"
    return "UNKNOWN"


def _enumerate_downward_subsets(
    drug_tuple: tuple[str, ...],
    *,
    max_depth: int | None,
) -> list[tuple[str, ...]]:
    max_order = len(drug_tuple)
    if max_depth is None:
        min_order = 2
    else:
        min_order = max(2, max_order - int(max_depth))
    subsets: list[tuple[str, ...]] = []
    for order in range(max_order, min_order - 1, -1):
        for combo in itertools.combinations(drug_tuple, order):
            subsets.append(tuple(sorted(combo)))
    return subsets


def build_target_center_sequence(
    drug_tuple: tuple[str, ...],
    *,
    side_effect_id: str,
    context_db: dict[tuple[tuple[str, ...], str], int],
    max_depth: int | None = None,
    keep_unknown_context: bool = True,
    mask_target_token: bool = True,
    order_clip: int = 7,
) -> dict:
    tokens: list[dict] = []
    full_target = tuple(sorted(drug_tuple))
    subsets = _enumerate_downward_subsets(full_target, max_depth=max_depth)
    for subset in subsets:
        subset_key = (subset, side_effect_id)
        label = context_db.get(subset_key)
        is_target = subset == full_target
        if label is None and not is_target and not keep_unknown_context:
            continue

        if label is None:
            exist_label = None
            exist_input = MASK_EXIST
        else:
            exist_label = int(label)
            exist_input = int(label)

        if len(subset) == 2:
            parent_subsets: list[str] = []
            comp_targets = [NO_COMP]
            comp_inputs = [NO_COMP]
            comp_observed_mask = [1]
        else:
            parent_subsets = []
            comp_targets = []
            comp_inputs = []
            comp_observed_mask = []
            child_label = context_db.get(subset_key)
            for drop_drug in subset:
                parent_subset = tuple(sorted(item for item in subset if item != drop_drug))
                parent_subsets.append("|".join(parent_subset))
                parent_label = context_db.get((parent_subset, side_effect_id))
                if parent_label is None or child_label is None:
                    comp_targets.append(MASK_COMP)
                    comp_inputs.append(MASK_COMP)
                    comp_observed_mask.append(0)
                else:
                    comp_name = _comp_label(int(parent_label), int(child_label))
                    comp_targets.append(comp_name)
                    comp_inputs.append(comp_name)
                    comp_observed_mask.append(1)

        if is_target and mask_target_token:
            exist_input = MASK_EXIST
            if len(subset) > 2:
                comp_inputs = [MASK_COMP] * len(comp_inputs)

        tokens.append(
            {
                "subset_drug_ids": list(subset),
                "subset_str": "|".join(subset),
                "order": min(len(subset), order_clip),
                "order_raw": len(subset),
                "side_effect_id": side_effect_id,
                "is_target": is_target,
                "exist_label": exist_label,
                "exist_input": exist_input,
                "parent_subsets": parent_subsets,
                "composition_types_target": comp_targets,
                "composition_types_input": comp_inputs,
                "composition_observed_mask": comp_observed_mask,
            }
        )

    tokens = sorted(tokens, key=lambda token: (-int(token["order_raw"]), tuple(token["subset_drug_ids"])))
    target_token_index = -1
    for index, token in enumerate(tokens):
        token["position_in_sequence"] = index
        if token["is_target"]:
            target_token_index = index
    if target_token_index < 0:
        raise ValueError(f"Target token {full_target} missing in target-centered sequence.")

    return {
        "side_effect_id": side_effect_id,
        "tokens": tokens,
        "target_token_index": target_token_index,
        "sequence_length": len(tokens),
    }
