"""GFEAT-01: FEAT-01 F2/F3 history-event side features in G's encoder (opt-in entry only; default path untouched).

The feature DEFINITIONS are not written here. Feedback bit order, bucket edges, bucket counts and the sidecar format are
imported from the modules shared byte-for-byte with the P line (feat01/p d70a993): ``history_side``, ``retrieval_features``.
This module only decides where the per-event side vector enters G's encoder:

  x = (E[token] + side_vector(item of that token)) * sqrt(d_model) + position        (encoder input only)

``side_vector`` is broadcast onto all four SID tokens of a history item.
  f2_feedback       sum of the item's feedback-bit embeddings (6 x d_model table, multi-hot)
  f3_watch_buckets  timespent-bucket embedding + watch-ratio-bucket embedding (8 x d, 6 x d tables, id 0 = PAD)
Side tables are created after the anchor's initialisation (same random stream), N(0, 0.02), PAD rows 0.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn

from . import generative_experiment as ge
from . import generative_training as gt
from .generative_data import sid_targets
from .generative_data_v2 import SIDTokensL
from .generative_hp import core_config, embedding_report, init_embeddings, make_model_b1
from .history_side import FEEDBACK_BITS, SIDE_VERSION, load_side, side_digest
from .retrieval_features import FEEDBACK_DIM, RATIO_BUCKETS, TIMESPENT_BUCKETS, ratio_bucket, timespent_bucket
from .tiger import TigerModel

FEATURES = ('none', 'f2_feedback', 'f3_watch_buckets')
SIDE_KEYS = ('feedback_mask', 'timespent', 'duration')


@dataclass(frozen=True)
class SidePack:
    """``[feedback_mask, timespent, duration]`` per history event, aligned 1:1 with the context's histories.

    ``train`` is indexed by example_id, ``validation`` by query ordinal. It lives OUTSIDE ``RunContext.queries/examples``:
    putting side into the queries would change every ``query_sha256`` and break the feature=none == anchor guarantee."""
    train: tuple
    validation: tuple
    binding: dict


def _check_event(event):
    if type(event) not in (list, tuple) or len(event) != 3 or any(type(v) is not int for v in event):
        raise ValueError('side event must be three ints')
    mask, timespent, duration = event
    if not 0 <= mask < 1 << FEEDBACK_DIM or not 1 <= timespent <= 255 or not 1 <= duration <= 255:
        raise ValueError('side value outside registered range')


def _example_rows(context, chunk=4096):
    """Training rows chunk by chunk: the example store copies every row it returns, so the full set is never copied at once."""
    for low in range(0, len(context.examples), chunk):
        yield from context.examples.rows(list(range(low, min(low + chunk, len(context.examples)))))


def build_side_pack(side, loaded, context, history_manifest_sha, side_manifest_sha):
    """Validate a loaded sidecar against the context it will be used with and freeze it as a ``SidePack``."""
    count = len(context.examples); queries = context.queries
    if len(side['train']) != count or len(side['validation']) != len(queries) \
            or len(loaded['train']) != count or len(loaded['validation']) != len(queries):
        raise ValueError('side denominator drift')
    duration_of = {}
    for split, rows, reference, events in (('train', _example_rows(context), loaded['train'], side['train']), ('validation', queries, loaded['validation'], side['validation'])):
        for ordinal, (row, ref, line) in enumerate(zip(rows, reference, events)):
            if split == 'train' and row.get('example_id') != ordinal:
                raise ValueError('side/example identity')
            history = row['history']
            if history != ref['history'] or row['history_positions'] != ref['history_positions'] or type(line) is not list or len(line) != len(history):
                raise ValueError('side/history alignment')
            for item, event in zip(history, line):
                _check_event(event)
                # duration is an item property: a row/position mix-up shows up as one item with two durations
                if duration_of.setdefault(item, event[2]) != event[2]:
                    raise ValueError('item duration not single-valued: side rows misaligned with history')
    binding = {'side_version': SIDE_VERSION, 'side_manifest_sha256': side_manifest_sha, 'history_manifest_sha256': history_manifest_sha,
               'train_side_sha256': side_digest(side['train']), 'validation_side_sha256': side_digest(side['validation'])}
    return SidePack(tuple(tuple(tuple(e) for e in line) for line in side['train']),
                    tuple(tuple(tuple(e) for e in line) for line in side['validation']), binding)


def load_side_pack(side_manifest, side_sha, history_manifest_sha, loaded, context, length):
    """sha-bound read of HIST_V2_SIDE_V1 (shared P reader) + alignment against ``context``; fail-closed."""
    side = load_side(side_manifest, side_sha, history_manifest_sha, length, loaded)
    return build_side_pack(side, loaded, context, history_manifest_sha, side_sha)


def attach_side(context, pack):
    """Same examples / codec / queries / slice statistics; the side digests are recorded in the context binding."""
    binding = {**context.binding, 'feat01': dict(pack.binding)}
    return ge.RunContext(ge._CONTEXT_KEY, examples=context.examples, codec=context.codec, queries=context.queries,
                         binding=binding, stats=context.slice_definition).check()


def side_tensor(sides, device='cpu'):
    """Right-padded ``(B, items, 3)`` long tensor, the same layout as the SID source tokens; PAD slots are all zero."""
    width = max((len(s) for s in sides), default=0)
    if width < 1:
        raise ValueError('nonempty side rows required')
    out = torch.zeros(len(sides), width, 3, dtype=torch.long)
    for b, events in enumerate(sides):
        if len(events):
            out[b, :len(events)] = torch.tensor(events, dtype=torch.long)
    return out.to(device)


class TigerModelSide(TigerModel):
    """TigerModel whose ENCODER input also carries a per-history-item side vector; the decoder path is untouched."""
    feature = None

    def add_side_tables(self, feature):
        """Run after ``init_embeddings(self)``: every anchor parameter then keeps the anchor's random stream bit for bit."""
        d = self.d_model
        if feature == 'f2_feedback':
            self.feedback = nn.Embedding(FEEDBACK_DIM, d)
        elif feature == 'f3_watch_buckets':
            self.timespent_bucket = nn.Embedding(TIMESPENT_BUCKETS + 1, d, padding_idx=0)
            self.ratio_bucket = nn.Embedding(RATIO_BUCKETS + 1, d, padding_idx=0)
        else:
            raise ValueError('unregistered GFEAT-01 feature')
        self.feature = feature
        with torch.no_grad():
            for name in ('feedback', 'timespent_bucket', 'ratio_bucket'):
                table = getattr(self, name, None)
                if table is not None:
                    table.weight.normal_(0.0, 0.02)
                    if table.padding_idx is not None:
                        table.weight[table.padding_idx].zero_()

    def _valid(self, source, side):
        """Mirror of the P-side ``_check``: validates source/side and returns the (B, items) validity mask."""
        if source.ndim != 2 or source.shape[1] % 4 or not 4 <= source.shape[1] <= 200:
            raise ValueError('source width must be 4 x items (1..50)')
        batch, width = source.shape
        present = source.reshape(batch, width // 4, 4).ne(0)
        if bool((present.any(-1) != present.all(-1)).any()):
            raise ValueError('partial PAD item')
        valid = present[..., 0]
        if side is None or side.dtype != torch.long or tuple(side.shape) != (batch, width // 4, 3):
            raise ValueError('history side required: (batch, items, 3) long')
        if bool((side[~valid] != 0).any()):
            raise ValueError('PAD slot carries side values')
        mask, timespent, duration = side[..., 0][valid], side[..., 1][valid], side[..., 2][valid]
        if bool(((mask < 0) | (mask >= 1 << FEEDBACK_DIM)).any() | ((timespent < 1) | (timespent > 255)).any() | ((duration < 1) | (duration > 255)).any()):
            raise ValueError('side value outside registered range')
        return valid

    def side_vector(self, side, valid):
        """(B, items, d_model): F2 = sum of the item's feedback-bit embeddings; F3 = timespent bucket + watch-ratio bucket."""
        if self.feature == 'f2_feedback':
            bits = torch.stack([(side[..., 0] >> b) & 1 for b in range(FEEDBACK_DIM)], -1).to(self.embedding.weight.dtype) * valid.unsqueeze(-1)
            return bits @ self.feedback.weight
        if self.feature == 'f3_watch_buckets':
            return self.timespent_bucket(timespent_bucket(side[..., 1], valid)) + self.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
        raise ValueError('side tables not initialised')

    def embed_source(self, source, side):
        """Side is added in TOKEN space (before x sqrt(d_model)) and broadcast onto all four SID tokens of its item."""
        valid = self._valid(source, side)
        token = self.embedding(source) + self.side_vector(side, valid).repeat_interleave(4, dim=1)
        return token * math.sqrt(self.d_model) + self.position(torch.arange(source.shape[1], device=source.device))[None]

    def encode(self, source, side=None):
        padding = source.eq(0)
        if padding.all(dim=1).any():
            raise ValueError('empty source')
        return self.transformer.encoder(self.embed_source(source, side), src_key_padding_mask=padding), padding

    def forward(self, source, decoder, side=None):
        memory, padding = self.encode(source, side)
        return self.decode(memory, padding, decoder)


def make_model_feat(codec, config, *, device):
    """feature=none -> the anchor's ``make_model_b1`` (plain TigerModel); f2/f3 -> anchor init, then the side tables."""
    feature = config['feature']
    if feature not in FEATURES:
        raise ValueError('unregistered GFEAT-01 feature')
    if feature == 'none':
        return make_model_b1(codec, config, device=device)
    if config['init'] != 'small' or config['history_len'] != 50:
        raise ValueError('GFEAT-01 side features: init=small, history_len=50 only')
    core = gt._core(core_config(config))
    model = TigerModelSide(codec.vocab_size, core['d_model'], core['heads'], 1, 1, core['ff']).to(device=device, dtype=torch.float32)
    init_embeddings(model)  # same draw sequence as make_model_b1: embedding, then position
    model.add_side_tables(feature)
    model.to(device=device, dtype=torch.float32)
    model.init_report = {'init': 'small', 'std': 0.02, 'embeddings': embedding_report(model)}
    return model


class SIDTokensSide(SIDTokensL):
    """``SIDTokensL`` (token layout unchanged) plus the side rows of the training examples, looked up by ``example_id``."""

    def __init__(self, codec, max_len, pack):
        super().__init__(codec, max_len)
        self.pack = pack

    def train_side(self, rows, device='cpu'):
        """``(B, items, 3)`` side tensor right-padded exactly like the ``(B, 4 x items)`` source tokens of the same rows."""
        sides = []
        for row in rows:
            example_id = row.get('example_id')
            if type(example_id) is not int or not 0 <= example_id < len(self.pack.train):
                raise ValueError('training row without a sidecar row')
            side = self.pack.train[example_id]
            if len(side) != len(row['history']):
                raise ValueError('side/history length mismatch')
            sides.append(side[-self.max_len:])
        return side_tensor(sides, device)


def train_batch_feat(model, optimizer, rows, context, tokens, config, device):
    """``generative_training.train_effective_batch`` line for line, with the per-item side tensor passed to the model."""
    microbatch = config['microbatch']
    if type(microbatch) is not int or microbatch < 1 or not rows or len(rows) > 512:
        raise ValueError('nonempty bounded effective batch required')
    model.train(); optimizer.zero_grad(set_to_none=True)
    n = len(rows); total = 0.; sizes = []
    try:
        for start in range(0, n, microbatch):
            batch = rows[start:start + microbatch]
            source, decoder, labels = (x.to(device) for x in sid_targets(batch, context.codec))
            side = tokens.train_side(batch, device)
            loss = gt.full_vocab_ce(model(source, decoder, side), labels)
            weighted = loss * (len(batch) / n)
            if not torch.isfinite(weighted):
                raise ValueError('nonfinite loss')
            weighted.backward(); total += float(loss.detach()) * len(batch) / n; sizes.append(len(batch))
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        except RuntimeError as exc:
            raise ValueError('nonfinite gradient') from exc
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite updated parameters')
        return gt.StepResult(total, n, tuple(sizes))
    finally:
        optimizer.zero_grad(set_to_none=True)
