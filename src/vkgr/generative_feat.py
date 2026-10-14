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
from .history_side import FEEDBACK_BITS, SIDE_VERSION, load_side, side_digest
from .retrieval_features import FEEDBACK_DIM, RATIO_BUCKETS, TIMESPENT_BUCKETS, ratio_bucket, timespent_bucket

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


def build_side_pack(side, loaded, context, history_manifest_sha, side_manifest_sha):
    """Validate a loaded sidecar against the context it will be used with and freeze it as a ``SidePack``."""
    examples = context.examples.rows(list(range(len(context.examples)))); queries = context.queries
    if len(side['train']) != len(examples) or len(side['validation']) != len(queries) \
            or len(loaded['train']) != len(examples) or len(loaded['validation']) != len(queries):
        raise ValueError('side denominator drift')
    duration_of = {}
    for split, rows, reference, events in (('train', examples, loaded['train'], side['train']), ('validation', queries, loaded['validation'], side['validation'])):
        for ordinal, (row, ref, line) in enumerate(zip(rows, reference, events)):
            if split == 'train' and row.get('example_id') != ordinal:
                raise ValueError('side/example identity')
            history = row['history']
            if history != ref['history'] or type(line) is not list or len(line) != len(history):
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
