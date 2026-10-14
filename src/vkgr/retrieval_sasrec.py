"""P4H-02 fixed ID-only causal SASRec-style two-tower model."""
import torch

ARCHITECTURE_SPEC = {
    "item_embedding_dim": 64,
    "position_count": 50,
    "layers": 1,
    "heads": 4,
    "feedforward_dim": 256,
    "dropout": 0.1,
    "activation": "relu",
    "bias": True,
    "norm_first": False,
    "layer_norm_eps": 1e-5,
    "batch_first": True,
    "attention": "upper_triangular_causal",
    "pooling": "last_valid_hidden",
    "content_used": False,
}


class SASRecIDTwoTower(torch.nn.Module):
    """One frozen architecture; unsupported IDs contribute position only."""

    def __init__(self, content, supported_indices):
        super().__init__()
        if (
            not isinstance(content, torch.Tensor)
            or content.dtype != torch.float32
            or content.ndim != 2
            or content.shape[1] != 64
            or len(content) < 2
            or not torch.isfinite(content).all()
            or torch.any(content[0])
        ):
            raise ValueError("invalid content compatibility input")
        if (
            not isinstance(supported_indices, (list, tuple))
            or len(set(supported_indices)) != len(supported_indices)
            or any(type(index) is not int or not 1 <= index < len(content) for index in supported_indices)
        ):
            raise ValueError("invalid supported ID")
        self.embedding = torch.nn.Embedding(len(content), 64, padding_idx=0)
        self.positions = torch.nn.Embedding(50, 64)
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        supported = torch.zeros(len(content), dtype=torch.bool)
        supported[list(supported_indices)] = True
        self.register_buffer("supported", supported)

    def items(self, indices):
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.long
            or torch.any(indices < 0)
            or torch.any(indices >= len(self.supported))
        ):
            raise ValueError("invalid item indices")
        return self.embedding(indices) * self.supported[indices].unsqueeze(-1)

    def _compact(self, histories):
        if (
            not isinstance(histories, torch.Tensor)
            or histories.dtype != torch.long
            or histories.ndim != 2
            or not 1 <= histories.shape[1] <= 50
            or torch.any(histories < 0)
            or torch.any(histories >= len(self.supported))
        ):
            raise ValueError("invalid histories")
        counts = (histories != 0).sum(1)
        if torch.any(counts == 0):
            raise ValueError("empty history")
        compacted = torch.zeros_like(histories)
        for row in range(len(histories)):
            values = histories[row][histories[row] != 0]
            compacted[row, : len(values)] = values
        return compacted, compacted != 0

    def encode_sequence(self, histories):
        compacted, valid = self._compact(histories)
        width = compacted.shape[1]
        ordinals = torch.arange(width, device=compacted.device).expand(len(compacted), width)
        encoded = self.items(compacted) + self.positions(ordinals) * valid.unsqueeze(-1)
        causal = torch.triu(torch.ones(width, width, dtype=torch.bool, device=compacted.device), diagonal=1)
        hidden = self.transformer(encoded, src_mask=causal, src_key_padding_mask=~valid)
        if not torch.isfinite(hidden[valid]).all():
            raise ValueError("nonfinite user encoding")
        return hidden, valid

    def users(self, histories):
        hidden, valid = self.encode_sequence(histories)
        last = valid.sum(1) - 1
        result = hidden[torch.arange(len(hidden), device=hidden.device), last]
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite user vectors")
        return result

    def history_diagnostics(self, histories):
        compacted, valid = self._compact(histories)
        supported = self.supported[compacted] & valid
        valid_count = int(valid.sum().item())
        supported_count = int(supported.sum().item())
        unsupported_count = valid_count - supported_count
        return {
            "valid_token_count": valid_count,
            "supported_token_count": supported_count,
            "unsupported_token_count": unsupported_count,
            "unsupported_token_fraction": unsupported_count / valid_count,
            "all_zero_identity_history_count": int((supported.sum(1) == 0).sum().item()),
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = self.supported
        actual = state_dict.get("supported")
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError("checkpoint immutable support mismatch")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
