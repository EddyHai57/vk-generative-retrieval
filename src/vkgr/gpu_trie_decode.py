"""Exact batch-one SID beam search with device-resident layer CSR constraints.

Independent implementation inspired by STATIC's published CSR representation;
no third-party source is copied. Model calls and full-vocabulary log_softmax
are the frozen serial path. IEEE float64 addition and stable lexicographic
sorting reproduce Python's ``(-score, path)`` and ``(-score, item)`` ordering.
"""
import math

import numpy as np
import torch


class CSRTrie:
    """One-time sorted-key -> per-layer CSR; never scan catalog rows per query.

    State is the mixed-radix prefix key. Each row stores sorted legal digits.
    Layer storage is O(number of distinct prefix edges), not dense N x vocab.
    The caller must retain the immutable catalog used to build this index.
    """
    def __init__(self, catalog, *, device):
        if catalog.depth != 3 or len(catalog.bases) != 4:
            raise ValueError('registered depth3/four steps required')
        self.catalog = catalog
        self.device = torch.empty(0, device=device).device
        self.bases = tuple(catalog.bases)
        if math.prod(self.bases) > np.iinfo(np.int64).max:
            raise ValueError('prefix key exceeds int64')
        self.layers = []
        for position, base in enumerate(self.bases):
            stride = math.prod(self.bases[position + 1:])
            # catalog.keys is already sorted. All duplicates are adjacent.
            raw = catalog.keys // stride
            children = raw[np.r_[True, raw[1:] != raw[:-1]]]
            parents = children // base
            starts = np.flatnonzero(np.r_[True, parents[1:] != parents[:-1]])
            row_ptr = np.r_[starts, len(children)]
            arrays = (parents[starts], row_ptr, children % base)
            self.layers.append(tuple(torch.tensor(a.astype(np.int64), dtype=torch.long,
                                                  device=self.device) for a in arrays))
        self.storage_bytes = sum(x.numel() * x.element_size() for layer in self.layers for x in layer)

    def prefix_key(self, prefix):
        key = 0
        for digit, base in zip(prefix, self.bases):
            key = key * base + digit
        return key

    def allowed(self, states, position):
        """Gather padded CSR rows on device; validity excludes padding edges."""
        keys, pointers, digits = self.layers[position]
        rows = torch.searchsorted(keys, states)
        safe_rows = rows.clamp(max=len(keys) - 1)
        if torch.any(rows >= len(keys)) or torch.any(keys[safe_rows] != states):
            raise ValueError('unreachable prefix')
        first, last = pointers[rows], pointers[rows + 1]
        offsets = torch.arange(self.bases[position], device=self.device)
        indices = first[:, None] + offsets[None, :]
        valid = indices < last[:, None]
        return digits[indices.clamp(max=len(digits) - 1)], valid


def constrained_beam_csr(catalog, codec, score, trie, beam=100, *, trace=None):
    """Serial model callback, device constraints; optional all-step exact audit.

    ``trace`` receives the selected (path, Python float) pairs after every step.
    Diagnostic copies are excluded from production when trace is None.
    """
    if type(beam) is not int or beam < 1:
        raise ValueError('invalid beam')
    if trie.catalog is not catalog or tuple(catalog.bases) != trie.bases:
        raise ValueError('CSR/catalog binding differs')
    device = trie.device
    paths = [()]
    states = torch.zeros(1, dtype=torch.long, device=device)
    totals = torch.zeros(1, dtype=torch.float64, device=device)
    audit = {'invalid': 0, 'deadend': 0, 'collision': 0, 'max_active': 1}
    for position in range(4):
        logits = score(paths)
        if logits.shape != (len(paths), codec.vocab_size) or not torch.isfinite(logits).all():
            raise ValueError('nonfinite/invalid beam logits')
        if logits.device != device:
            raise ValueError('logits/CSR device differs')
        logprob = torch.log_softmax(logits, dim=-1).detach()
        digits, valid = trie.allowed(states, position)
        row = torch.arange(len(paths), device=device)[:, None].expand_as(digits)
        parent = row[valid]
        digit = digits[valid]
        if not digit.numel():
            raise ValueError('beam deadend')
        # Convert the exact same FP32 log probability before the binary64 add.
        expanded_scores = totals[parent] + logprob[parent, codec.offsets[position] + digit].to(torch.float64)
        expanded_keys = states[parent] * trie.bases[position] + digit
        # Stable secondary path ordering, then stable primary score ordering.
        # torch.topk has no stable tie guarantee and must not be used here.
        lexical = torch.argsort(expanded_keys, stable=True)
        order = lexical[torch.argsort(expanded_scores[lexical], descending=True, stable=True)][:beam]
        selected_parent = parent[order].tolist()
        selected_digit = digit[order].tolist()
        paths = [paths[p] + (d,) for p, d in zip(selected_parent, selected_digit)]
        states, totals = expanded_keys[order], expanded_scores[order]
        audit['max_active'] = max(audit['max_active'], len(paths))
        if trace is not None:
            trace.append(list(zip(paths, totals.tolist())))
    # Match key dtype explicitly: scalar signed/unsigned searchsorted can copy
    # the entire catalog in NumPy. One vector lookup preserves decode semantics.
    final_keys = np.asarray(states.tolist(), dtype=catalog.keys.dtype)
    first = np.searchsorted(catalog.keys, final_keys, side='left')
    last = np.searchsorted(catalog.keys, final_keys, side='right')
    if np.any(last - first != 1):
        raise ValueError('SID collision')
    ranked = [(int(item), total) for item, total in zip(catalog.key_items[first], totals.tolist())]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    audit['collision'] = len(ranked) - len(set(x[0] for x in ranked))
    if audit['collision']:
        raise ValueError('beam item collision')
    return [x[0] for x in ranked], audit


def decode_history_csr(model, codec, history, beam=100, *, trie, trace=None):
    """Same batch-one model calls and shapes as frozen decode_history_d."""
    device = next(model.parameters()).device
    source = torch.tensor([codec.history(history)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        memory, padding = model.encode(source)
        def score(prefixes):
            rows = [[1] + [codec.offsets[j] + digit for j, digit in enumerate(path)] for path in prefixes]
            decoder = torch.tensor(rows, dtype=torch.long, device=device)
            return model.decode(memory.expand(len(rows), -1, -1), padding.expand(len(rows), -1), decoder)[:, -1]
        return constrained_beam_csr(codec.catalog, codec, score, trie, beam, trace=trace)
