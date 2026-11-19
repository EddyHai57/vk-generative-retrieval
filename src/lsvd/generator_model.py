"""A1 encoder-decoder with LSVD event histories and optional dense retrieval."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from vkgr.tiger_arch import _Core

from .two_tower import corrected_loss


SID_LAYERS = 3
PAD = 0
BOS = 1


@dataclass
class GeneratorConfig:
    history_mode: str = "event"
    shared_head: bool = True
    size: int = 1024
    vector_dim: int = 64
    d_model: int = 128
    heads: int = 6
    head_dim: int = 64
    encoder_layers: int = 4
    decoder_layers: int = 4
    ff: int = 1024
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 1024
    seed: int = 42
    epochs: int = 3
    patience: int = 3
    max_history: int = 200
    user_hash_seed: int = 2026111801
    temperature: float = 0.1


def _attention_padding(padding):
    """Expose one internal dummy only when there are no real tokens."""
    safe = padding.clone()
    safe[:, 0] &= ~padding.all(dim=1)
    return safe


class GeneratorModel(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        if config.history_mode not in {"event", "tiger", "event_no_feedback"}:
            raise ValueError(f"unknown generator history mode {config.history_mode!r}")
        self.config = config
        dim = config.d_model
        self.sid_embedding = nn.Embedding(SID_LAYERS * config.size + 2, dim, padding_idx=PAD)
        self.decoder_position = nn.Embedding(SID_LAYERS, dim)
        if config.history_mode == "tiger":
            self.source_position = nn.Embedding(SID_LAYERS * config.max_history, dim)
        else:
            self.source_position = nn.Embedding(config.max_history, dim)
            self.ratio = nn.Embedding(6, dim, padding_idx=0)
            self.place = nn.Embedding(257, dim, padding_idx=0)
            self.gap = nn.Embedding(28, dim, padding_idx=0)
            event_dim = (SID_LAYERS + 4) * dim + config.vector_dim + 9
            self.event_projection = nn.Sequential(
                nn.Linear(event_dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
        self.transformer = _Core(
            dim, config.heads, config.head_dim, config.encoder_layers,
            config.decoder_layers, config.ff, config.dropout,
        )
        self.sid_heads = nn.ModuleList(nn.Linear(dim, config.size) for _ in range(SID_LAYERS))
        self.dense_head = nn.Linear(dim, config.vector_dim) if config.shared_head else None

    def _sid_tokens(self, codes):
        layers = torch.arange(codes.shape[-1], device=codes.device)
        return codes.long() + layers * self.config.size + 2

    def _embed_events(self, history, valid, tokens):
        def index(name):
            return history[name].long().masked_fill(~valid, 0)

        codes = self.sid_embedding(tokens).flatten(start_dim=-2)
        cold = history["cold"].to(codes.dtype).masked_fill(~valid[..., None], 0)
        if self.config.history_mode == "event":
            feedback = history["feedback"].to(codes.dtype).masked_fill(~valid[..., None], 0)
            ratio = history["ratio"]
            bucket = 1 + (ratio >= 0.1).long() + (ratio >= 0.5) + (ratio >= 1.0) + (ratio > 1.0)
            bucket = bucket.masked_fill(~valid, 0)
        else:
            feedback = codes.new_zeros((*valid.shape, 9))
            bucket = torch.zeros_like(valid, dtype=torch.long)
        features = torch.cat((
            codes, cold, feedback, self.ratio(bucket), self.place(index("place")),
            self.gap(index("gap")), self.source_position(index("position")),
        ), dim=-1)
        return self.event_projection(features)

    def encode(self, history):
        """Encode strictly earlier events. Returned padding also marks empty rows."""
        valid = history["valid"]
        batch, width = valid.shape
        codes = history["codes"]
        if codes.shape != (batch, width, SID_LAYERS) or width > self.config.max_history:
            raise ValueError("invalid generator history shape")
        padding = ~valid
        if width == 0:
            padding = valid.new_ones((batch, 1))
            x = self.sid_embedding.weight.new_zeros((batch, 1, self.config.d_model))
        else:
            tokens = self._sid_tokens(codes).masked_fill(padding[..., None], PAD)
            if self.config.history_mode == "tiger":
                padding = padding.repeat_interleave(SID_LAYERS, dim=1)
                tokens = tokens.flatten(start_dim=1)
                positions = torch.arange(tokens.shape[1], device=tokens.device)
                x = (self.sid_embedding(tokens) * math.sqrt(self.config.d_model)
                     + self.source_position(positions)[None])
            else:
                x = self._embed_events(history, valid, tokens)
            x = x.masked_fill(padding[..., None], 0)

        safe_padding = _attention_padding(padding)
        for layer in self.transformer.encoder.layers:
            x = layer(x, safe_padding)
        return self.transformer.encoder.norm(x), padding

    def user_vector(self, memory, padding):
        """Read the last real encoder state, with the dummy for empty training rows."""
        if self.dense_head is None:
            return None
        positions = torch.arange(memory.shape[1], device=memory.device)
        last = positions.masked_fill(padding, 0).amax(dim=1)
        readout = memory[torch.arange(memory.shape[0], device=memory.device), last]
        return F.normalize(self.dense_head(readout).float(), dim=-1)

    def _decode_prefix(self, memory, padding, prefixes):
        bos = torch.full((prefixes.shape[0], 1), BOS, device=prefixes.device, dtype=torch.long)
        tokens = torch.cat((bos, self._sid_tokens(prefixes)), dim=1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = (self.sid_embedding(tokens) * math.sqrt(self.config.d_model)
             + self.decoder_position(positions)[None])
        safe_padding = _attention_padding(padding)
        for layer in self.transformer.decoder.layers:
            x = layer(x, memory, safe_padding)
        return self.transformer.decoder.norm(x)

    def decode_next(self, memory, padding, prefixes):
        """Raw SID prefix [N, 0..2] to raw logits in the next layer's vocabulary."""
        if prefixes.ndim != 2 or prefixes.shape[1] >= SID_LAYERS:
            raise ValueError("invalid generator prefix shape")
        states = self._decode_prefix(memory, padding, prefixes)
        return self.sid_heads[prefixes.shape[1]](states[:, -1])

    @torch.no_grad()
    def make_beam_scorer(self, memory, padding, *, hypothesis_batch_size: int = 128):
        """Cache encoder K/V for one eval batch and score raw [B, W, step] prefixes.

        Use the scorer before changing model weights, device, or precision.
        """
        if self.training:
            raise ValueError("beam scoring requires eval mode")
        if hypothesis_batch_size < 1:
            raise ValueError("hypothesis batch size must be positive")
        batch_size = memory.shape[0]
        valid_keys = ~_attention_padding(padding)[:, None, None, :]
        decoder = self.transformer.decoder
        cross_memory = []
        for layer in decoder.layers:
            attention = layer.multihead_attn
            kv = F.linear(memory, attention.in_proj_weight[attention.inner:],
                          attention.in_proj_bias[attention.inner:])
            kv = kv.unflatten(-1, (2, attention.heads, attention.head_dim))
            cross_memory.append(kv.permute(2, 0, 3, 1, 4).unbind(0))

        def score_batch(prefixes, owners):
            batch, width, depth = prefixes.shape
            steps = depth + 1
            bos = prefixes.new_full((batch * width, 1), BOS, dtype=torch.long)
            tokens = torch.cat((bos, self._sid_tokens(prefixes.reshape(batch * width, depth))), dim=1)
            positions = torch.arange(steps, device=prefixes.device)
            x = (self.sid_embedding(tokens) * math.sqrt(self.config.d_model)
                 + self.decoder_position(positions)[None])
            for layer, (keys, values) in zip(decoder.layers, cross_memory):
                x = layer.norm1(x + layer.dropout1(layer.self_attn(x, x, None, True)))
                attention = layer.multihead_attn
                q = F.linear(x, attention.in_proj_weight[:attention.inner],
                             attention.in_proj_bias[:attention.inner])
                # Cross-attention can share a query's encoder K/V across all its beam tokens.
                q = q.reshape(batch, width * steps, attention.heads, attention.head_dim).transpose(1, 2)
                attended = F.scaled_dot_product_attention(q, keys[owners], values[owners], valid_keys[owners])
                attended = attended.transpose(1, 2).reshape(batch * width, steps, attention.inner)
                x = layer.norm2(x + layer.dropout2(attention.out_proj(attended)))
                x = layer.norm3(x + layer.dropout3(layer.linear2(layer.dropout(F.relu(layer.linear1(x))))))
            return self.sid_heads[depth](decoder.norm(x[:, -1])).reshape(batch, width, self.config.size)

        @torch.no_grad()
        def score(prefixes):
            if self.training:
                raise ValueError("beam scoring requires eval mode")
            if (prefixes.ndim != 3 or prefixes.shape[0] != batch_size
                    or not min(prefixes.shape[:2]) or prefixes.shape[2] >= SID_LAYERS):
                raise ValueError("invalid generator beam prefix shape")
            query_logits = []
            for start in range(0, batch_size, hypothesis_batch_size):
                stop = min(start + hypothesis_batch_size, batch_size)
                owners = slice(start, stop)
                width_per_batch = hypothesis_batch_size // (stop - start)
                query_logits.append(torch.cat([
                    score_batch(chunk, owners) for chunk in prefixes[owners].split(width_per_batch, dim=1)
                ], dim=1))
            return torch.cat(query_logits, dim=0)

        return score

    def sid_logits(self, memory, padding, target_codes):
        """Teacher forcing with BOS and the first two raw target codes."""
        if target_codes.ndim != 2 or target_codes.shape[1] != SID_LAYERS:
            raise ValueError("invalid generator target shape")
        states = self._decode_prefix(memory, padding, target_codes[:, :-1])
        return torch.stack([head(states[:, depth]) for depth, head in enumerate(self.sid_heads)], dim=1)

    @staticmethod
    def _sid_cross_entropy(logits, target_codes, weights):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            terms = F.cross_entropy(
                logits.float().transpose(1, 2), target_codes.long(), reduction="none"
            )
            return (terms.mean(dim=1) * weights).sum() / weights.sum()

    def sid_loss(self, history, target_codes, weights=None):
        """Compute the SID-only objective for training or validation."""
        memory, padding = self.encode(history)
        weights = self._loss_weights(target_codes, weights)
        return self._sid_cross_entropy(self.sid_logits(memory, padding, target_codes), target_codes, weights)

    @staticmethod
    def _loss_weights(target_codes, weights):
        if weights is None:
            return target_codes.new_ones(target_codes.shape[0], dtype=torch.float32)
        return weights.to(device=target_codes.device, dtype=torch.float32)

    def loss(self, history, target_codes, target_vectors, item_ids, positives, probabilities, weights):
        """Compute the SID and optional corrected dense objectives, weighted 1:1."""
        memory, padding = self.encode(history)
        weights = self._loss_weights(target_codes, weights)
        sid = self._sid_cross_entropy(self.sid_logits(memory, padding, target_codes), target_codes, weights)
        actor = self.user_vector(memory, padding)
        if actor is None:
            return sid, sid, sid.new_zeros(())
        dense, _ = corrected_loss(
            actor, target_vectors.detach(), item_ids, positives, probabilities, weights, self.config.temperature
        )
        return sid + dense, sid, dense
