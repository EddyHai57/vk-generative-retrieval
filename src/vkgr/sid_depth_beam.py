"""G6-PREP: constrained beam decode for ``catalog.depth + 1`` positions (new file; frozen decoders stay untouched).

``tiger.constrained_beam`` / ``tiger.decode_history`` and ``generative_batched._decode_group`` /
``decode_histories_batched`` loop over a literal ``range(4)``. The functions here are line-for-line copies whose only
change is the position count ``catalog.depth + 1``. With ``depth == 3`` they return exactly what the frozen twins return
(ranked items, audit dict, float64 accumulation, tie-breaks); tests pin that.
"""
import numpy as np
import torch

from . import generative_batched as gb


def constrained_beam_d(catalog, codec, score, beam=20):
    if type(beam) is not int or beam < 1:
        raise ValueError('invalid beam')
    active = [((), 0.)]
    audit = {'invalid': 0, 'deadend': 0, 'collision': 0, 'max_active': 1}
    for position in range(catalog.depth + 1):
        logits = score([path for path, _ in active])
        if logits.shape != (len(active), codec.vocab_size) or not torch.isfinite(logits).all():
            raise ValueError('nonfinite/invalid beam logits')
        logprob = torch.log_softmax(logits, dim=-1).detach().cpu()
        expanded = []
        for row, (path, total) in enumerate(active):
            allowed = catalog.allowed_next(path)
            if not allowed:
                audit['deadend'] += 1
            for digit in allowed:
                expanded.append((path + (digit,), total + float(logprob[row, codec.offsets[position] + digit])))
        if not expanded:
            raise ValueError('beam deadend')
        expanded.sort(key=lambda x: (-x[1], x[0]))
        active = expanded[:beam]
        audit['max_active'] = max(audit['max_active'], len(active))
    ranked = []
    for path, total in active:
        item = catalog.decode(path)
        ranked.append((item, total))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    audit['collision'] = len(ranked) - len(set(x[0] for x in ranked))
    if audit['collision']:
        raise ValueError('beam item collision')
    return [x[0] for x in ranked], audit


def decode_history_d(model, codec, history, beam=20):
    device = next(model.parameters()).device
    source = torch.tensor([codec.history(history)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        memory, padding = model.encode(source)

        def score(prefixes):
            rows = [[1] + [codec.offsets[j] + digit for j, digit in enumerate(path)] for path in prefixes]
            decoder = torch.tensor(rows, dtype=torch.long, device=device)
            return model.decode(memory.expand(len(rows), -1, -1), padding.expand(len(rows), -1), decoder)[:, -1]
        return constrained_beam_d(codec.catalog, codec, score, beam)


def decode_group_d(model, tokens, sources, beam):
    """Same-width group; returns [(items, audit)] in input order (copy of ``generative_batched._decode_group``)."""
    device = next(model.parameters()).device
    catalog = tokens.catalog
    offsets = tokens.offsets
    n = len(sources)
    memory, padding = model.encode(torch.tensor(sources, dtype=torch.long, device=device))
    paths = [np.zeros((1, 0), dtype=np.int64) for _ in range(n)]
    totals = [np.zeros(1, dtype=np.float64) for _ in range(n)]
    audits = [{'invalid': 0, 'deadend': 0, 'collision': 0, 'max_active': 1} for _ in range(n)]
    for position in range(catalog.depth + 1):
        counts = [len(t) for t in totals]
        owner = np.repeat(np.arange(n), counts)
        prefix = np.concatenate(paths)
        rows = len(prefix)
        decoder = np.empty((rows, position + 1), dtype=np.int64)
        decoder[:, 0] = 1
        for j in range(position):
            decoder[:, j + 1] = offsets[j] + prefix[:, j]
        index = torch.from_numpy(owner).to(device)
        logits = model.decode(memory.index_select(0, index), padding.index_select(0, index), torch.from_numpy(decoder).to(device))[:, -1]
        if logits.shape != (rows, tokens.vocab_size) or not torch.isfinite(logits).all():
            raise ValueError('nonfinite/invalid beam logits')
        logprob = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
        first = 0
        for q in range(n):
            parents = []
            digits = []
            for i, path in enumerate(paths[q]):
                allowed = gb._allowed(catalog, tuple(int(x) for x in path))
                if not len(allowed):
                    audits[q]['deadend'] += 1
                parents.append(np.full(len(allowed), i, dtype=np.int64))
                digits.append(allowed)
            parent = np.concatenate(parents)
            digit = np.concatenate(digits)
            if not len(digit):
                raise ValueError('beam deadend')
            # float32 -> float64 is exact; the add is the serial Python-float add.
            score = totals[q][parent] + logprob[first + parent, offsets[position] + digit].astype(np.float64)
            expanded = np.concatenate([paths[q][parent], digit[:, None]], axis=1)
            order = np.lexsort(tuple(expanded[:, j] for j in reversed(range(position + 1))) + (-score,))[:beam]
            paths[q] = expanded[order]
            totals[q] = score[order]
            audits[q]['max_active'] = max(audits[q]['max_active'], len(order))
            first += counts[q]
    results = []
    for q in range(n):
        ranked = [(catalog.decode(tuple(int(x) for x in path)), float(total)) for path, total in zip(paths[q], totals[q])]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        audits[q]['collision'] = len(ranked) - len(set(x[0] for x in ranked))
        if audits[q]['collision']:
            raise ValueError('beam item collision')
        results.append(([x[0] for x in ranked], audits[q]))
    return results


def decode_histories_batched_d(model, tokens, histories, beam, batch_size):
    """Batched twin of ``decode_history_d`` for many histories (copy of ``generative_batched.decode_histories_batched``)."""
    if type(beam) is not int or beam < 1:
        raise ValueError('invalid beam')
    gb._check_batch(batch_size)
    sources = [tokens.history(h) for h in histories]
    groups = {}
    for index, source in enumerate(sources):
        groups.setdefault(len(source), []).append(index)
    results = [None] * len(sources)
    model.eval()
    with torch.no_grad():
        for width in sorted(groups):
            members = groups[width]
            for start in range(0, len(members), batch_size):
                chunk = members[start:start + batch_size]
                for index, value in zip(chunk, decode_group_d(model, tokens, [sources[i] for i in chunk], beam)):
                    results[index] = value
    return results
