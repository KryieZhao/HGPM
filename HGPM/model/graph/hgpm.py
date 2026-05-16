from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from HGPM.task.graph.graph_base import AttentionPool



def _order_distance_bucket(delta: int) -> int:
    if delta <= -3:
        return 0
    if delta == -2:
        return 1
    if delta == -1:
        return 2
    if delta == 0:
        return 3
    if delta == 1:
        return 4
    if delta == 2:
        return 5
    if delta >= 3:
        return 6
    return 7


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


def _exist_role(token: dict) -> str:
    exist_source = str(token.get("exist_source", ""))
    exist = int(token.get("exist", token.get("existence", 0)))
    if exist_source == "center_node":
        return "center"
    if exist == 1 and exist_source in {"observed", "observed_substructure"}:
        return "observed"
    if exist == 0 or exist_source == "sampled_negative":
        return "negative"
    return "other"


def _exist_transition_bucket(src_role: str, dst_role: str) -> int:
    mapping = {
        ("observed", "observed"): 1,
        ("observed", "negative"): 2,
        ("negative", "observed"): 3,
        ("negative", "negative"): 4,
        ("center", "observed"): 5,
        ("center", "negative"): 6,
    }
    return mapping.get((src_role, dst_role), 0)


class HGPMGraphCollator:
    """Centered hyperedge DAG tokenizer with HGPM relation-aware attention bias.

    Token flattening, token features, and readout behavior stay aligned with the
    existing hyperdag_gpm path. The only extension is finer-grained pairwise DAG
    relation metadata for attention logits.
    """

    def __init__(self, *, drug_vocab: dict[str, int]) -> None:
        self.drug_vocab = drug_vocab
        self.exist_source_vocab = {
            "unknown": 0,
            "observed": 1,
            "sampled_negative": 2,
            "center_node": 3,
            "observed_substructure": 4,
        }

    @staticmethod
    def _subset_mean_feature(subset_drug_ids: list[str], node_feature_table: torch.Tensor, drug_vocab: dict[str, int]) -> torch.Tensor:
        raw_ids = [drug_vocab.get(drug_id, drug_vocab.get("[UNK]", 1)) for drug_id in subset_drug_ids]
        if node_feature_table.size(0) + 2 == max(drug_vocab.values()) + 1:
            vocab_ids = [max(0, min(node_feature_table.size(0) - 1, idx - 2)) for idx in raw_ids]
        else:
            vocab_ids = [max(0, min(node_feature_table.size(0) - 1, idx)) for idx in raw_ids]
        features = node_feature_table[torch.tensor(vocab_ids, dtype=torch.long)]
        return features.mean(dim=0)

    def __call__(self, batch: list[dict], node_feature_table: torch.Tensor) -> dict:
        feature_dim = int(node_feature_table.size(1))
        seq_features = []
        seq_orders = []
        seq_exists = []
        seq_exist_sources = []
        seq_view_ids = []
        seq_masks = []
        center_token_masks = []
        edge_direction_ids = []
        order_distance_ids = []
        overlap_bucket_ids = []
        exist_transition_ids = []
        sibling_bias_masks = []
        labels = []
        node_ids = []

        for sample in batch:
            flat_tokens: list[dict] = []
            for view in sample["views"]:
                view_id = int(view.get("view_id", 0))
                for token in view["tokens"]:
                    copied = dict(token)
                    copied["_view_id"] = view_id
                    flat_tokens.append(copied)

            center = sample["node_id"]
            token_keys = [str(token["subset_str"]) + f"::v{int(token['_view_id'])}" for token in flat_tokens]
            token_orders = [int(token["order"]) for token in flat_tokens]
            token_subsets = [set(token["subset_drug_ids"]) for token in flat_tokens]
            token_parent_sets = []
            token_child_sets = []
            token_roles = [_exist_role(token) for token in flat_tokens]

            for token in flat_tokens:
                parent_keys = {
                    str(parent_subset) + f"::v{int(token['_view_id'])}"
                    for parent_subset in token.get("parent_subsets", [])
                }
                token_parent_sets.append(parent_keys)
                token_child_sets.append(set())

            for child_idx, parents in enumerate(token_parent_sets):
                for parent_idx, parent_key in enumerate(token_keys):
                    if parent_key in parents:
                        token_child_sets[parent_idx].add(token_keys[child_idx])

            token_features = []
            token_order_ids = []
            token_exist_ids = []
            token_exist_source_ids = []
            token_view_ids = []
            token_center_mask = []
            seq_len = len(flat_tokens)
            edge_direction = torch.zeros((seq_len, seq_len), dtype=torch.long)
            order_distance = torch.zeros((seq_len, seq_len), dtype=torch.long)
            overlap_bucket = torch.zeros((seq_len, seq_len), dtype=torch.long)
            exist_transition = torch.zeros((seq_len, seq_len), dtype=torch.long)
            sibling_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)

            for idx, token in enumerate(flat_tokens):
                token_features.append(self._subset_mean_feature(token["subset_drug_ids"], node_feature_table, self.drug_vocab))
                token_order_ids.append(int(token["order"]))
                token_exist_ids.append(int(token.get("exist", token.get("existence", 0))) + 1)
                exist_source = str(token.get("exist_source", "unknown"))
                token_exist_source_ids.append(self.exist_source_vocab.get(exist_source, 0))
                token_view_ids.append(int(token["_view_id"]) + 1)
                is_center_token = tuple(sorted(token["subset_drug_ids"], key=int)) == (center,)
                token_center_mask.append(1.0 if is_center_token else 0.0)

            for i in range(seq_len):
                for j in range(seq_len):
                    if token_keys[i] in token_parent_sets[j]:
                        edge_direction[i, j] = 1  # parent -> child
                    elif token_keys[j] in token_parent_sets[i]:
                        edge_direction[i, j] = 2  # child -> parent
                    else:
                        edge_direction[i, j] = 0

                    order_distance[i, j] = _order_distance_bucket(token_orders[i] - token_orders[j])

                    intersection = len(token_subsets[i].intersection(token_subsets[j]))
                    union = max(len(token_subsets[i].union(token_subsets[j])), 1)
                    overlap_ratio = float(intersection / union)
                    overlap_bucket[i, j] = _overlap_bucket(overlap_ratio)

                    exist_transition[i, j] = _exist_transition_bucket(token_roles[i], token_roles[j])

                    if i != j:
                        shares_parent = len(token_parent_sets[i].intersection(token_parent_sets[j])) > 0
                        shares_child = len(token_child_sets[i].intersection(token_child_sets[j])) > 0
                        if shares_parent or shares_child:
                            sibling_mask[i, j] = 1.0

            seq_features.append(torch.stack(token_features, dim=0) if token_features else torch.zeros((0, feature_dim)))
            seq_orders.append(torch.tensor(token_order_ids, dtype=torch.long))
            seq_exists.append(torch.tensor(token_exist_ids, dtype=torch.long))
            seq_exist_sources.append(torch.tensor(token_exist_source_ids, dtype=torch.long))
            seq_view_ids.append(torch.tensor(token_view_ids, dtype=torch.long))
            seq_masks.append(torch.ones(seq_len, dtype=torch.float32))
            center_token_masks.append(torch.tensor(token_center_mask, dtype=torch.float32))
            edge_direction_ids.append(edge_direction)
            order_distance_ids.append(order_distance)
            overlap_bucket_ids.append(overlap_bucket)
            exist_transition_ids.append(exist_transition)
            sibling_bias_masks.append(sibling_mask)
            labels.append(int(sample["label"]))
            node_ids.append(center)

        max_len = max((tensor.size(0) for tensor in seq_features), default=1)

        def pad_feature_rows(rows: list[torch.Tensor]) -> torch.Tensor:
            padded = []
            for row in rows:
                if row.size(0) < max_len:
                    pad = torch.zeros((max_len - row.size(0), row.size(1)), dtype=row.dtype)
                    row = torch.cat([row, pad], dim=0)
                padded.append(row)
            return torch.stack(padded, dim=0)

        def pad_long_rows(rows: list[torch.Tensor]) -> torch.Tensor:
            padded = []
            for row in rows:
                if row.size(0) < max_len:
                    row = torch.cat([row, torch.zeros(max_len - row.size(0), dtype=row.dtype)], dim=0)
                padded.append(row)
            return torch.stack(padded, dim=0)

        def pad_square(rows: list[torch.Tensor]) -> torch.Tensor:
            padded = []
            for row in rows:
                current = row.size(0)
                padded_row = torch.zeros((max_len, max_len), dtype=row.dtype)
                padded_row[:current, :current] = row
                padded.append(padded_row)
            return torch.stack(padded, dim=0)

        return {
            "token_features": pad_feature_rows(seq_features),
            "order_ids": pad_long_rows(seq_orders),
            "exist_ids": pad_long_rows(seq_exists),
            "exist_source_ids": pad_long_rows(seq_exist_sources),
            "view_ids": pad_long_rows(seq_view_ids),
            "seq_mask": pad_long_rows(seq_masks),
            "center_token_mask": pad_long_rows(center_token_masks),
            "edge_direction_ids": pad_square(edge_direction_ids),
            "order_distance_ids": pad_square(order_distance_ids),
            "overlap_bucket_ids": pad_square(overlap_bucket_ids),
            "exist_transition_ids": pad_square(exist_transition_ids),
            "sibling_bias_mask": pad_square(sibling_bias_masks),
            "labels": torch.tensor(labels, dtype=torch.long),
            "node_ids": node_ids,
        }


class RelationAwareBiasedSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, seq_mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + attn_bias
        valid = seq_mask > 0
        key_mask = ~valid.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, float("-inf"))
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            scores[no_valid] = 0.0
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(context)


class RelationAwareTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RelationAwareBiasedSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, seq_mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.drop1(self.attn(self.norm1(hidden), seq_mask, attn_bias))
        hidden = hidden + self.drop2(self.ffn(self.norm2(hidden)))
        return hidden * seq_mask.unsqueeze(-1).to(hidden.dtype)


class HGPMGraphModel(nn.Module):
    """DAG-only relation-aware attention bias v2, without any global propagation branch."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        order_vocab_size: int,
        exist_vocab_size: int,
        exist_source_vocab_size: int,
        view_vocab_size: int,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.order_embedding = nn.Embedding(order_vocab_size, hidden_dim, padding_idx=0)
        self.exist_embedding = nn.Embedding(exist_vocab_size, hidden_dim, padding_idx=0)
        self.exist_source_embedding = nn.Embedding(exist_source_vocab_size, hidden_dim, padding_idx=0)
        self.view_embedding = nn.Embedding(view_vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(1024, hidden_dim)

        self.edge_direction_bias = nn.Embedding(3, num_heads)
        self.order_distance_bias = nn.Embedding(8, num_heads)
        self.overlap_bucket_bias = nn.Embedding(5, num_heads)
        self.exist_transition_bias = nn.Embedding(7, num_heads)
        self.sibling_bias = nn.Parameter(torch.zeros(num_heads))
        nn.init.zeros_(self.edge_direction_bias.weight)
        nn.init.zeros_(self.order_distance_bias.weight)
        nn.init.zeros_(self.overlap_bucket_bias.weight)
        nn.init.zeros_(self.exist_transition_bias.weight)

        self.blocks = nn.ModuleList(
            [RelationAwareTransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_pool = AttentionPool(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def build_attention_bias(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        edge = self.edge_direction_bias(batch["edge_direction_ids"]).permute(0, 3, 1, 2)
        order = self.order_distance_bias(batch["order_distance_ids"]).permute(0, 3, 1, 2)
        overlap = self.overlap_bucket_bias(batch["overlap_bucket_ids"]).permute(0, 3, 1, 2)
        exist = self.exist_transition_bias(batch["exist_transition_ids"]).permute(0, 3, 1, 2)
        sibling = batch["sibling_bias_mask"].unsqueeze(1) * self.sibling_bias.view(1, -1, 1, 1)
        return edge + order + overlap + exist + sibling

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        token_features = batch["token_features"]
        batch_size, seq_len, _ = token_features.shape
        positions = torch.arange(seq_len, device=token_features.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.feature_projection(token_features)
        hidden = hidden + self.order_embedding(batch["order_ids"].clamp_max(self.order_embedding.num_embeddings - 1))
        hidden = hidden + self.exist_embedding(batch["exist_ids"].clamp_max(self.exist_embedding.num_embeddings - 1))
        hidden = hidden + self.exist_source_embedding(batch["exist_source_ids"].clamp_max(self.exist_source_embedding.num_embeddings - 1))
        hidden = hidden + self.view_embedding(batch["view_ids"].clamp_max(self.view_embedding.num_embeddings - 1))
        hidden = hidden + self.position_embedding(positions.clamp_max(self.position_embedding.num_embeddings - 1))

        attn_bias = self.build_attention_bias(batch)
        for block in self.blocks:
            hidden = block(hidden, batch["seq_mask"], attn_bias)
        hidden = self.norm(hidden)

        center_mask = batch["center_token_mask"]
        valid_centers = center_mask > 0
        center_den = valid_centers.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        center_repr = (hidden * valid_centers.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / center_den
        no_center = ~valid_centers.any(dim=1)
        if no_center.any():
            center_repr = torch.where(no_center.unsqueeze(-1), hidden[:, 0, :], center_repr)
        return hidden, center_repr

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded, center_repr = self.encode(batch)
        pooled_all = self.attention_pool(encoded, batch["seq_mask"])
        return self.classifier(torch.cat([center_repr, pooled_all], dim=-1))
