"""Bounded CPU sample tensors for the two pinned deterministic constructors.

No random draws, persistent data, global patches, or changes to training steps.
Namespace hashes describe the immutable input context; effective map values are
also part of every row key. This is engineering code pending real-anchor parity.
"""
from collections import OrderedDict
import hashlib
import json
import re
from types import FunctionType


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def namespace(*, input, sid, config, source, runtime, constructor):
    values = dict(input=input, sid=sid, config=config, source=source,
                  runtime=runtime, constructor=constructor)
    if any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in values.values()):
        raise ValueError('all six namespace SHA256 identities required')
    return _digest(values)


def bind_step(original, builder_name, replacement):
    """Same bytecode/defaults/closure; only one builder in a private globals copy."""
    if builder_name not in original.__globals__ or not callable(replacement):
        raise ValueError('existing constructor and callable replacement required')
    globals_copy = dict(original.__globals__)
    globals_copy[builder_name] = replacement
    result = FunctionType(original.__code__, globals_copy, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


class TensorCache:
    def __init__(self, identity, *, max_bytes=64 * 1024**2):
        if not isinstance(identity, str) or re.fullmatch('[0-9a-f]{64}', identity) is None:
            raise ValueError('namespace identity required')
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('positive CPU tensor byte limit required')
        self.identity, self.max_bytes = identity, max_bytes
        self._items = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def _row(self, kind, row, dependency, build):
        key = _digest([self.identity, kind, row, dependency])
        if key in self._items:
            self.hits += 1
            value, size = self._items.pop(key)
            self._items[key] = value, size
            return tuple(x.clone() for x in value)
        self.misses += 1
        value = tuple(build())
        if any(x.device.type != 'cpu' or x.requires_grad for x in value):
            raise ValueError('deterministic CPU constructor required')
        size = sum(x.numel() * x.element_size() for x in value)
        if size <= self.max_bytes:
            while self.bytes + size > self.max_bytes:
                _, (_, old_size) = self._items.popitem(last=False)
                self.bytes -= old_size
            self._items[key] = tuple(x.clone() for x in value), size
            self.bytes += size
        return value

    def g(self, rows, tokens, constructor):
        import torch
        if not rows:
            return constructor(rows, tokens)  # preserve original failure
        values = []
        for row in rows:
            # The pinned constructor validates list/nonempty/max_len and target.
            if type(row.get('history')) is not list or not 1 <= len(row['history']) <= tokens.max_len or type(row.get('target')) is not int:
                return constructor(rows, tokens)
            effective = [tokens.max_len, [tokens.item(i) for i in row['history']], tokens.item(row['target'])]
            values.append(self._row('g', row, effective, lambda: constructor([row], tokens)))
        width = max(v[0].shape[1] for v in values)
        source = torch.zeros((len(rows), width), dtype=values[0][0].dtype)
        for j, value in enumerate(values):
            source[j, :value[0].shape[1]] = value[0][0]
        return source, torch.cat([v[1] for v in values]), torch.cat([v[2] for v in values])

    def p(self, rows, idmap, width, feature, constructor):
        import torch
        if not rows:
            return constructor(rows, idmap, width, feature)
        values = []
        for row in rows:
            n = len(row['history'])
            if n > width or not n:
                return constructor(rows, idmap, width, feature)
            dep = [feature, [idmap[i] for i in row['history']]]
            values.append(self._row('p', row, dep, lambda: constructor([row], idmap, n, feature)))
        histories = torch.zeros((len(rows), width), dtype=values[0][0].dtype)
        side = torch.zeros((len(rows), width, values[0][1].shape[2]), dtype=values[0][1].dtype)
        for j, (h, s) in enumerate(values):
            histories[j, :h.shape[1]] = h[0]
            side[j, :s.shape[1]] = s[0]
        return histories, side
