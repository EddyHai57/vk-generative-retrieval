"""Local two-tower primitives. Only train-derived lookup tables enter the model.

The loss adapts vkgr.retrieval_negatives at c4e4eb8c. No scheduler or run gates.
DCN-V2's full-matrix recurrence is independently implemented from its equation.
"""

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TowerConfig:
    dim: int = 64
    max_history: int = 200
    id_dropout: float = 0.3
    use_rank: bool = False
    cross_layers: int = 0
    transformer_layers: int = 1
    heads: int = 4
    dropout: float = 0.1
    temperature: float = 0.10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    batch_size: int = 128
    epochs: int = 5
    patience: int = 3
    seed: int = 42
    user_hash_seed: int = 2026111801
    id_min_count: int = 5
    embedding_init_std: float = 0.02
    packed_window: bool = False
    sequence_chunk: int = 128
    max_user_targets: int = 64
    causal_readout: str = "last"
    balance_user_batches: bool = False
    shuffle_balanced_batches: bool = False
    shuffle_all_targets: bool = False


def user_hash(user_id, seed=2026111801):
    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{int(user_id)}".encode(), digest_size=8).digest(),
        "little",
    )


def rank_partition(rank, train_count):
    """0=local seen, 1=only locally unseen, 2=global zero-train proxy."""
    rank = np.asarray(rank)
    seen = np.asarray(train_count) > 0
    if not seen.any():
        raise ValueError("rank boundary needs at least one training-seen item")
    boundary = int(rank[seen].max()) + 1
    return boundary, np.where(seen, 0, np.where(rank < boundary, 1, 2)).astype(np.uint8)


def rank_buckets(rank, boundary):
    # Ranks are unique 0..N-1. Equal-frequency video bins need no quantile fit.
    rank = np.asarray(rank)
    return np.where(
        rank < boundary, 1 + rank.astype(np.int64) * 20 // boundary, 0
    ).astype(np.int64)


def initialize_embeddings(module, std):
    # Adapted from retrieval_hp.init_embeddings at c4e4eb8c.
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Embedding):
                child.weight.normal_(0.0, std)
                if child.padding_idx is not None:
                    child.weight[child.padding_idx].zero_()


class ItemTower(nn.Module):
    def __init__(self, config, id_count, author_count):
        super().__init__()
        self.config = config
        d = config.dim
        self.content = nn.Sequential(nn.Linear(64, d), nn.ReLU(), nn.Linear(d, d))
        self.ids = nn.Embedding(id_count, d, padding_idx=0)
        self.authors = nn.Embedding(author_count, d, padding_idx=0)
        self.durations = nn.Embedding(13, d, padding_idx=0)
        self.ranks = nn.Embedding(21, d) if config.use_rank else None
        self.output = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        initialize_embeddings(self, config.embedding_init_std)
        self.cross = nn.ModuleList(nn.Linear(d, d) for _ in range(config.cross_layers))

    def forward(self, features, *, cold=False):
        indices = features["id_index"]
        drop = (
            torch.ones_like(indices, dtype=torch.bool)
            if cold
            else torch.zeros_like(indices, dtype=torch.bool)
        )
        if self.training and not cold:
            drop = (
                torch.rand(indices.shape, device=indices.device)
                < self.config.id_dropout
            )
        x = (
            self.content(features["content"])
            + self.authors(features["author_index"])
            + self.durations(features["duration_bucket"])
        )
        x = x + self.ids(indices) * (~drop).unsqueeze(-1)
        if self.ranks is not None:
            rank = torch.where(drop, torch.zeros_like(indices), features["rank_bucket"])
            x = x + self.ranks(rank)
        origin = x
        for layer in self.cross:
            x = origin * layer(x) + x
        return F.normalize(self.output(x).float(), dim=-1)


class UserTower(nn.Module):
    def __init__(self, config, structure="mean"):
        super().__init__()
        self.config = config
        self.structure = structure
        d = config.dim
        self.feedback = nn.Linear(9, d, bias=False)
        self.ratio = nn.Embedding(6, d, padding_idx=0)
        self.place = nn.Embedding(257, d, padding_idx=0)
        self.position = nn.Embedding(config.max_history, d)
        self.gap = nn.Embedding(28, d, padding_idx=0)
        initialize_embeddings(self, config.embedding_init_std)
        if structure == "causal":
            layer = nn.TransformerEncoderLayer(
                d,
                config.heads,
                d * 4,
                config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, config.transformer_layers, enable_nested_tensor=False
            )
            self.gate = nn.Parameter(torch.tensor(0.0))
            self.residual_norm = nn.LayerNorm(d)
            nn.init.constant_(self.residual_norm.weight, 0.1)

    def events(self, items, event, valid):
        positions = torch.arange(items.shape[1], device=items.device)
        x = (
            items
            + self.feedback(event["feedback"])
            + self.ratio(event["ratio"])
            + self.place(event["place"])
        )
        return (
            x + self.position(positions) + self.gap(event["gap"])
        ) * valid.unsqueeze(-1)

    def encode(self, items, event, valid):
        x = self.events(items, event, valid)
        if self.structure != "causal":
            return x
        # Right padding plus a dummy visible zero slot avoids empty attention rows.
        safe = valid.clone()
        safe[:, 0] = True
        width = x.shape[1]
        causal = torch.ones(width, width, dtype=torch.bool, device=x.device).triu(1)
        return self.transformer(
            x, mask=causal, src_key_padding_mask=~safe
        ) * valid.unsqueeze(-1)

    def forward(self, items, event, valid):
        if "group_offsets" in event:
            return self.forward_packed(items, event, valid)
        x = self.events(items, event, valid)
        mean = x.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        if self.structure == "causal":
            hidden = self.encode(items, event, valid)
            normalized = self.residual_norm(hidden) * valid.unsqueeze(-1)
            residual = normalized.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            mean = mean + self.gate * residual
        out = F.normalize(mean.float(), dim=-1)
        return out * valid.any(1, keepdim=True)

    @staticmethod
    def relative_distances(readouts, keys):
        return readouts[:, None] - 1 - keys[None, :]

    def forward_packed(self, items, event, valid):
        """One shared event projection per user/week segment, bounded attention.

        A readout is an exclusive target position. Relative positions enter
        attention keys and the window mean. The last-state path has support
        200. The shared-state mean is diagnostic and has support up to 399.
        """
        x = (items + self.feedback(event["feedback"]) + self.ratio(event["ratio"])
             + self.place(event["place"]) + self.gap(event["gap"])) * valid[:, None]
        ends = event["readouts"]
        group_starts = torch.as_tensor(event["group_offsets"][:-1], device=x.device)
        starts = torch.maximum(group_starts[event["target_group"]], ends - self.config.max_history)
        counts = ends - starts
        prefix = F.pad(x.cumsum(0), (0, 0, 1, 0))
        position_prefix = F.pad(self.position.weight.cumsum(0), (0, 0, 1, 0))
        mean = (prefix[ends] - prefix[starts] + position_prefix[counts]) / counts.clamp_min(1)[:, None]
        if self.structure == "causal":
            if len(self.transformer.layers) != 1:
                raise ValueError("packed 200-event support requires the single-layer smoke model")
            layer = self.transformer.layers[0]
            normalized = layer.norm1(x)
            q, k, v = F.linear(normalized, layer.self_attn.in_proj_weight,
                               layer.self_attn.in_proj_bias).chunk(3, dim=-1)
            heads = self.config.heads
            width = self.config.dim // heads
            relative_keys = self.position.weight.view(-1, heads, width).permute(1, 2, 0)
            if self.config.causal_readout == "window_mean":
                residual = self.shared_window_residual(layer, x, q, k, v, starts, ends,
                                                       relative_keys, event, valid)
                return F.normalize(mean + self.gate * residual, dim=-1) * (counts > 0)[:, None]
            hidden = torch.zeros_like(mean)
            for indices, key_start, key_end in event["attention_blocks"]:
                indices = torch.as_tensor(indices, device=x.device)
                readouts = ends[indices]
                state = self.local_attention_state(layer, x, q, k, v, relative_keys,
                                                   readouts - 1, key_start, key_end, valid)
                hidden = hidden.index_copy(0, indices, state)
            mean = mean + self.gate * self.residual_norm(hidden)
        return F.normalize(mean, dim=-1) * (counts > 0)[:, None]

    def local_attention_state(self, layer, x, q, k, v, relative_keys, positions,
                              key_start, key_end, valid):
        heads = self.config.heads
        width = self.config.dim // heads
        queries = q[positions].view(-1, heads, width).transpose(0, 1)
        keys = k[key_start:key_end].view(-1, heads, width).transpose(0, 1)
        values = v[key_start:key_end].view(-1, heads, width).transpose(0, 1)
        distances = self.relative_distances(positions + 1, torch.arange(key_start, key_end, device=x.device))
        visible = (distances >= 0) & (distances < self.config.max_history)
        visible &= valid[key_start:key_end][None, :]
        relative_scores = queries @ relative_keys / width**0.5
        bias = relative_scores.gather(2, distances.clamp(0, self.config.max_history - 1)[None].expand(heads, -1, -1))
        bias = bias.masked_fill(~visible[None], -torch.inf)
        attended = F.scaled_dot_product_attention(
            queries[None], keys[None], values[None], attn_mask=bias[None],
            dropout_p=layer.self_attn.dropout if self.training else 0.0,
        )[0].transpose(0, 1).reshape(-1, self.config.dim)
        state = x[positions] + layer.dropout1(layer.self_attn.out_proj(attended))
        return state + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(state))))))

    def shared_window_residual(self, layer, x, q, k, v, starts, ends, relative_keys, event, valid):
        """Diagnostic shared-state mean, with at most 399 earlier exposures."""
        positions_parts = []
        states_parts = []
        groups = event["target_group"].detach().cpu().numpy()
        for group, offset in enumerate(event["group_offsets"][:-1]):
            indices = np.flatnonzero(groups == group)
            first = int(starts[indices].min())
            last = int(ends[indices].max())
            for begin in range(first, last, self.config.sequence_chunk):
                stop = min(last, begin + self.config.sequence_chunk)
                positions = torch.arange(begin, stop, device=x.device)
                key_start = max(int(offset), begin - self.config.max_history + 1)
                state = self.local_attention_state(layer, x, q, k, v, relative_keys,
                                                   positions, key_start, stop, valid)
                positions_parts.append(positions)
                states_parts.append(self.residual_norm(state))
        normalized = torch.zeros_like(x)
        if positions_parts:
            normalized = normalized.index_copy(0, torch.cat(positions_parts), torch.cat(states_parts))
        prefix = F.pad(normalized.cumsum(0), (0, 0, 1, 0))
        return (prefix[ends] - prefix[starts]) / (ends - starts).clamp_min(1)[:, None]


def corrected_loss(
    users, items, item_ids, positive_sets, probabilities, weights, temperature
):
    """Natural target-frequency Q, row weights, and full train-positive masking."""
    from .two_tower_tensors import PositiveBatch, PositiveIndex
    from .two_tower_runtime import precision_mode

    with precision_mode(users.device, 'fp32'):
        scores = (
            F.normalize(users.float(), dim=-1)
            @ F.normalize(items.float(), dim=-1).T
            / temperature
        )
        scores = scores - probabilities.to(scores.device, dtype=torch.float32).log()[None, :]
        if isinstance(positive_sets, torch.Tensor) and positive_sets.dtype == torch.bool:
            mask = positive_sets.to(scores.device)
        else:
            if not isinstance(positive_sets, PositiveBatch):
                positive_sets = PositiveIndex.from_sets(positive_sets, scores.device).batch(
                    torch.arange(len(users), device=scores.device)
                )
            mask = positive_sets.allowed(item_ids).to(scores.device)
        labels = torch.arange(len(users), device=scores.device)
        valid = mask.sum(1) > 1
        if not valid.any():
            return scores.sum() * 0, int((~valid).sum())
        losses = F.cross_entropy(
            scores.masked_fill(~mask, -torch.inf)[valid], labels[valid], reduction="none"
        )
        selected = weights.to(scores.device, dtype=torch.float32)[valid]
        return (losses * selected).sum() / selected.sum(), int((~valid).sum())


class Catalog:
    """CPU column arrays or memmaps; gather only requested rows for the device."""

    def __init__(self, columns, config):
        self.columns = columns
        self.ids = np.asarray(columns["item_id"])
        count = np.asarray(columns["train_count"])
        supported = count >= config.id_min_count
        self.id_index = np.zeros(len(count), dtype=np.int64)
        self.id_index[supported] = np.arange(1, supported.sum() + 1)
        known = np.unique(np.asarray(columns["author_id"])[count > 0])
        authors = np.asarray(columns["author_id"])
        pos = np.searchsorted(known, authors)
        matched = (pos < len(known)) & (
            known[np.minimum(pos, len(known) - 1)] == authors
        )
        self.author_index = np.where(matched, pos + 1, 0)
        self.author_count = len(known) + 1
        self.id_count = int(supported.sum()) + 1
        self.boundary, self.groups = rank_partition(columns["rank"], count)
        self.rank_index = rank_buckets(columns["rank"], self.boundary)
        self.duration_index = (
            np.searchsorted(
                [10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 181],
                columns["duration"],
                side="right",
            )
            + 1
        )
        self.config = config
        self.cached_content = None
        self.cached_rows = None

    def rows(self, ids):
        ids = np.asarray(ids)
        rows = np.searchsorted(self.ids, ids)
        if np.any(rows >= len(self.ids)) or np.any(self.ids[rows] != ids):
            raise ValueError("unknown catalog item")
        return rows

    def cache_training_content(self, item_ids, output):
        from pathlib import Path

        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        rows = self.rows(np.unique(item_ids))
        np.save(output / "catalog_rows.npy", rows)
        source = self.columns["content"]
        values = np.lib.format.open_memmap(
            output / "values.npy", mode="w+", dtype=source.dtype, shape=(len(rows), 64)
        )
        for start in range(0, len(rows), 65536):
            values[start : start + 65536] = source[rows[start : start + 65536]]
        values.flush()
        self.cached_content = np.load(output / "values.npy", mmap_mode="r")
        self.cached_rows = np.full(len(self.ids), -1, dtype=np.int32)
        self.cached_rows[rows] = np.arange(len(rows), dtype=np.int32)

    def features(self, ids, device="cpu"):
        rows = self.rows(np.asarray(ids))
        cols = self.columns
        source = cols["content"]
        content_rows = rows
        if self.cached_rows is not None:
            cached_rows = self.cached_rows[rows]
            if np.all(cached_rows >= 0):
                source = self.cached_content
                content_rows = cached_rows
        values = dict(
            content=np.asarray(source[content_rows], dtype=np.float32),
            id_index=self.id_index[rows],
            author_index=self.author_index[rows],
            duration_bucket=self.duration_index[rows],
            rank_bucket=self.rank_index[rows],
        )
        return {
            name: torch.as_tensor(value, device=device)
            for name, value in values.items()
        }

    def tensor_features(self, ids, valid, device):
        from .two_tower_tensors import TensorCatalog

        if not hasattr(self, '_tensor_catalog'):
            self._tensor_catalog = TensorCatalog.from_catalog(self)
        return self._tensor_catalog.tensor_features(ids, valid, device)


class TwoTower(nn.Module):
    def __init__(self, catalog, config, user_structure="mean"):
        super().__init__()
        self.config = config
        self.item = ItemTower(config, catalog.id_count, catalog.author_count)
        # Keep the shared event initialization independent of item-tower upgrades.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed + 1)
            self.user = UserTower(config, user_structure)
