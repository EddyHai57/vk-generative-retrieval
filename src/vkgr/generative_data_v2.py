"""Batch1 G data path for HIST_V2_OBS_L{20,50,100}.

Reads the shared history-v2 sidecar (``history_v2.py`` is byte-identical on the P and G
branches) by SHA and overlays it on an already-verified G3 context: examples and queries
keep their v1 identity; only ``history``/``history_positions`` change. This path does not
call ``generative_data.load_prepared``'s like-chain check (v2 histories are not
"previous history + previous target"); the sidecar checks in ``history_v2`` replace it.
"""
import copy

import torch

from . import generative_experiment as ge
from . import history_v2

SID_VERSIONS = ('SID_BHKM_B16D3_S42', 'SID_RANDPERM42_OF_BHKM_B16D3')


class SIDTokensL:
    """SID tokenisation with a configurable item window; L<=50 equals ``SIDTokens.history``."""

    def __init__(self, codec, max_len):
        if max_len not in (20, 50, 100):
            raise ValueError('registered history length required')
        self.codec = codec; self.max_len = max_len
        self.catalog = codec.catalog; self.offsets = codec.offsets; self.vocab_size = codec.vocab_size

    def item(self, item):
        return self.codec.item(item)

    def history(self, history):
        if not isinstance(history, list) or not history:
            raise ValueError('nonempty history required')
        return [token for item in history[-self.max_len:] for token in self.item(item)]

    def batch(self, histories, targets, device='cpu'):
        if len(histories) != len(targets) or not targets:
            raise ValueError('invalid batch')
        rows = [self.history(h) for h in histories]; width = max(map(len, rows))
        source = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long, device=device)
        labels = torch.tensor([self.item(t) for t in targets], dtype=torch.long, device=device)
        decoder = torch.cat([torch.ones((len(rows), 1), dtype=torch.long, device=device), labels[:, :-1]], dim=1)
        return source, decoder, labels


def sid_targets_l(rows, tokens):
    if not rows:
        raise ValueError('nonempty training rows required')
    for row in rows:
        if type(row.get('history')) is not list or not 1 <= len(row['history']) <= tokens.max_len or type(row.get('target')) is not int:
            raise ValueError('invalid training history/target schema')
    return tokens.batch([r['history'] for r in rows], [r['target'] for r in rows])


class ListExamples:
    """In-memory v2 examples with the ExampleStore read interface."""

    def __init__(self, rows):
        self._rows = tuple(copy.deepcopy(rows))

    def __len__(self):
        return len(self._rows)

    def rows(self, indices):
        if any(type(i) is not int or not 0 <= i < len(self._rows) for i in indices):
            raise ValueError('example indices out of range')
        return [copy.deepcopy(self._rows[i]) for i in indices]


def build_v2_context(context, loaded, length, sid, codec=None):
    """RunContext whose queries carry v2 histories; binding records the history/SID versions."""
    if sid not in SID_VERSIONS or loaded.get('length') != length:
        raise ValueError('registered SID/history version required')
    context.check()
    catalog = set(context.catalog)
    originals = context.examples.rows(list(range(len(context.examples))))
    rows = history_v2.overlay_examples(originals, loaded['train'], length, catalog)
    queries = history_v2.overlay_queries(context.queries, loaded['validation'], length, catalog)
    base_codec = context.codec
    codec = base_codec if codec is None else codec
    if (sid == SID_VERSIONS[0]) != (codec is base_codec):
        raise ValueError('SID version/codec mismatch')
    sids_sha = lambda c: ge._digest(c.catalog.sids.tolist())
    binding = {**context.binding, 'queries_sha256': ge._digest(queries), 'codec_sha256': sids_sha(codec),
               'b1': {'history_version': f'HIST_V2_OBS_L{length}', 'history_manifest_sha256': loaded['manifest_sha256'],
                      'train_history_sha256': ge._digest(rows), 'base_queries_sha256': context.binding['queries_sha256'],
                      'sid_version': sid, 'base_sid_sha256': sids_sha(base_codec), 'sid_sha256': sids_sha(codec)}}
    return ge.RunContext(ge._CONTEXT_KEY, examples=ListExamples(rows), codec=codec, queries=queries, binding=binding, stats=context.slice_definition).check()
