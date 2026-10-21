"""G6-PREP: depth-generic semantic-ID codec (new file; the frozen depth-3 code is imported, never edited).

The frozen codec contract (``generative_data._check_codec``, ``tiger.SIDTokens``, ``generative_data._ValidatedCodec``,
``sid_rqvae.assign_fourth_code``) hard-codes four digits (three semantic digits plus one collision suffix). This module
repeats exactly the same rules for ``depth`` semantic digits plus one suffix. With ``depth == 3`` every function here
gives the same values, accepts the same inputs and raises the same messages as its frozen twin (pinned by tests).
"""
import numpy as np

from . import generative_data as gd
from .generative_data_v2 import SIDTokensL
from .tiger import SIDTokens

MIN_DEPTH = 2
MAX_DEPTH = 8  # ``sid.fit`` accepts depth <= 8
WINDOW = 50  # the only registered item window of the A1 line


def _depth(depth):
    if type(depth) is not int or not MIN_DEPTH <= depth <= MAX_DEPTH:
        raise ValueError('depth must be an int between 2 and 8')
    return depth


def check_codec_v2(codec, depth, prepared=None):
    """``generative_data._check_codec`` for ``depth`` semantic digits; returns the catalog."""
    depth = _depth(depth)
    if not isinstance(codec, SIDTokens):
        raise ValueError('SIDTokens codec required')
    model = codec.catalog
    if model.ids.dtype != np.uint32 or model.ids.ndim != 1 or not len(model.ids) or np.any(model.ids[1:] <= model.ids[:-1]) \
            or model.sids.dtype != np.uint32 or model.sids.shape != (len(model.ids), depth + 1) or model.depth != depth \
            or type(model.branching) is not int or model.branching < 2 or np.any(model.sids[:, :depth] >= model.branching):
        raise ValueError('invalid codec catalog/SID schema')
    if model.bases != [model.branching] * depth + [int(model.sids[:, depth].max()) + 1] \
            or codec.offsets != list(np.cumsum([2] + model.bases[:-1])) or codec.vocab_size != 2 + sum(model.bases):
        raise ValueError('codec namespace identity mismatch')
    order = np.lexsort((model.ids,) + tuple(model.sids[:, j] for j in reversed(range(depth))))
    codes = model.sids[order]
    starts = np.r_[True, np.any(codes[1:, :depth] != codes[:-1, :depth], axis=1)]
    row_indices = np.arange(len(codes))
    suffix = row_indices - np.maximum.accumulate(np.where(starts, row_indices, 0))
    if not np.array_equal(codes[:, depth], suffix):
        raise ValueError('noncanonical suffix or duplicate SID')
    if prepared is not None and not np.array_equal(model.ids, prepared['catalog']):
        raise ValueError('codec/prepared catalog mismatch')
    return model


class SIDTokensD(SIDTokens):
    """``tiger.SIDTokens`` for ``catalog.depth`` semantic digits; only the digit-count rule differs."""

    def __init__(self, catalog):
        if len(catalog.bases) != catalog.depth + 1:
            raise ValueError('depth+1 SID digits required')
        self.catalog = catalog
        self.offsets = []
        total = 2
        for base in catalog.bases:
            self.offsets.append(total)
            total += base
        self.vocab_size = total


class ValidatedCodecD(SIDTokensD, gd._ValidatedCodec):
    """Once-validated immutable depth-``depth`` mapping; identity checks reuse the frozen ``check_identity``."""

    def __init__(self, model, depth):
        depth = _depth(depth)
        for name in self._arrays:
            value = getattr(model, name)
            setattr(model, name, np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape))
        SIDTokensD.__init__(self, model)
        self.depth = depth
        check_codec_v2(self, depth)
        self._model = model
        self._references = [getattr(model, name) for name in self._arrays]
        self._array_metadata = [(v.shape, v.dtype.str, v.nbytes, v.strides, v.ctypes.data) for v in self._references]
        self._namespace = self._namespace_bytes()


class SIDTokensLD(SIDTokensL):
    """``SIDTokensL`` restricted to the registered window 50 and to a depth-validated codec."""

    def __init__(self, codec, max_len):
        if not isinstance(codec, ValidatedCodecD) or type(max_len) is not int or max_len != WINDOW:
            raise ValueError('validated depth codec and the registered window 50 required')
        super().__init__(codec, max_len)


def assign_suffix(ids, codes):
    """``sid_rqvae.assign_fourth_code`` for codes [N, depth]: rank inside the group sharing all digits (ascending id)."""
    ids = np.asarray(ids)
    codes = np.asarray(codes)
    if ids.ndim != 1 or codes.ndim != 2 or not MIN_DEPTH <= codes.shape[1] <= MAX_DEPTH or codes.shape[0] != len(ids) \
            or len(np.unique(ids)) != len(ids) or (codes < 0).any():
        raise ValueError('invalid ids/codes')
    depth = codes.shape[1]
    order = np.lexsort((ids,) + tuple(codes[:, j] for j in reversed(range(depth))))
    sorted_codes = codes[order]
    starts = np.r_[True, np.any(sorted_codes[1:] != sorted_codes[:-1], axis=1)]
    rows = np.arange(len(ids))
    rank = rows - np.maximum.accumulate(np.where(starts, rows, 0))
    out = np.zeros((len(ids), depth + 1), dtype=np.uint32)
    out[order, :depth] = sorted_codes
    out[order, depth] = rank
    return out
