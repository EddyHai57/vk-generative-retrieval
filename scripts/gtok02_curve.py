"""GTOK-02 curve helpers: collision rates from a mapping audit, the pre-registered state rules, the Q1 reading table, CSV / markdown output.

Everything is descriptive. Q2 never claims causality (the points with a non-zero collision rate differ from the others in branching and in
the number of semantic levels too). Only validation hits and audit.json numbers are read; matplotlib is used only if it is installed.

Definitions
  CER = items_needing_suffix / items (= 1 - leaves.count / items): the share of items that need the last position to be told apart.
  LCR = semantic_collision_items / items: the share of items that sit in a leaf with at least two items.
States are judged against T3 on seeds 2024 and 2025 only (seed 42 took part in the selection, so it can only give a tentative state):
  UP / DOWN: both seeds differ from T3 by at least 6 hits in the same direction; FLAT: everything else.
"""
import csv
import importlib.util
import math
from pathlib import Path

THRESHOLD_HITS = 6
T3_HITS = {42: 254, 2024: 274, 2025: 275}
ORDER = ('B16', 'B32', 'B64', 'B128', 'B256D3', 'B512', 'B1024', 'B256D2')
SEED_COLUMNS = (42, 2024, 2025)
FIELDS = ('label', 'branching', 'depth', 'status', 'CER', 'LCR', 'mean_leaf_size', 'leaves_max', 'positions_distinct',
          'hits_s42', 'hits_s2024', 'hits_s2025')
DEPTH = {label: 2 if label == 'B256D2' else 3 for label in ORDER}


def collision_rates(audit):
    """CER, LCR and the leaf statistics from an ``audit_mapping`` dict. CER is computed in two ways that must agree."""
    items = audit['items']
    needing = audit['items_needing_suffix']
    cer = needing / items
    if not math.isclose(cer, 1 - audit['leaves']['count'] / items, rel_tol=0, abs_tol=1e-12):
        raise ValueError('items_needing_suffix and leaves.count disagree')
    return {'CER': cer, 'LCR': audit['semantic_collision_items'] / items, 'mean_leaf_size': items / audit['leaves']['count'],
            'leaves_max': audit['leaves']['max'], 'positions_distinct': [p['distinct'] for p in audit['positions']]}


def state(delta_2024, delta_2025):
    if delta_2024 >= THRESHOLD_HITS and delta_2025 >= THRESHOLD_HITS:
        return 'UP'
    if delta_2024 <= -THRESHOLD_HITS and delta_2025 <= -THRESHOLD_HITS:
        return 'DOWN'
    return 'FLAT'


def tentative_state(delta_42):
    if delta_42 >= THRESHOLD_HITS:
        return 'T_UP'
    if delta_42 <= -THRESHOLD_HITS:
        return 'T_DOWN'
    return 'T_FLAT'


def classify(hits, reference=T3_HITS):
    """State of one config against the reference. Without both later seeds the state is tentative (seed 42), or MISSING."""
    if 2024 in hits and 2025 in hits:
        return state(hits[2024] - reference[2024], hits[2025] - reference[2025])
    if 42 in hits:
        return tentative_state(hits[42] - reference[42])
    return 'MISSING'


Q1_TEXT = {
    'RISING': 'Resolution still rises beyond 256 (both later seeds, at least one of B512 / B1024).',
    'UNDECIDED': 'Undecided: seed 42 suggests a rise but the later seeds are not in. No conclusion for Q1.',
    'PEAK_AT_256': '256 is the peak: B512 and B1024 are both lower on both later seeds.',
    'NOT_RISING': 'No further rise after 256 (flat or lower on the later seeds).',
    'TENTATIVE_NOT_RISING': 'Tentative: no further rise after 256, based on a single seed for at least one point.',
}


def judge_q1(state_b512, state_b1024, hits_b512=None, hits_b1024=None):
    """First matching row of the pre-registered table wins. ``hits_*`` ({2024: h, 2025: h}) only serve the peak sub-rule of row 1."""
    pair = (state_b512, state_b1024)
    if 'UP' in pair:
        row, verdict = 1, 'RISING'
    elif 'T_UP' in pair:
        row, verdict = 2, 'UNDECIDED'
    elif pair == ('DOWN', 'DOWN'):
        row, verdict = 3, 'PEAK_AT_256'
    elif not any(s.startswith('T_') for s in pair):
        row, verdict = 4, 'NOT_RISING'
    else:
        row, verdict = 5, 'TENTATIVE_NOT_RISING'
    peak = None
    if row == 1:
        peak = 'UNDECIDED'
        if hits_b512 and hits_b1024 and all(seed in hits_b512 and seed in hits_b1024 for seed in (2024, 2025)):
            between = state(hits_b1024[2024] - hits_b512[2024], hits_b1024[2025] - hits_b512[2025])
            peak = {'UP': 'B1024', 'DOWN': 'B512', 'FLAT': 'B512_B1024_INDISTINGUISHABLE'}[between]
    return {'row': row, 'verdict': verdict, 'text': Q1_TEXT[verdict], 'peak': peak}


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation (average ranks for ties); None when one side is constant."""
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def rows(points):
    """points: {label: {'audit': mapping audit or None, 'hits': {seed: hits}, 'status': str}}. Missing values stay None (no interpolation)."""
    out = []
    for label in ORDER:
        if label not in points:
            continue
        point = points[label]
        row = {'label': label, 'branching': None, 'depth': DEPTH[label], 'status': point['status'], 'CER': None, 'LCR': None,
               'mean_leaf_size': None, 'leaves_max': None, 'positions_distinct': None}
        if point['audit'] is not None:
            row.update(collision_rates(point['audit']))
            row['branching'] = point['audit']['branching']
        for seed in SEED_COLUMNS:
            row[f'hits_s{seed}'] = point['hits'].get(seed)
        out.append(row)
    return out


def has_matplotlib():
    return importlib.util.find_spec('matplotlib') is not None


def _cell(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return format(value, '.6g')
    if isinstance(value, list):
        return '/'.join(str(x) for x in value)
    return str(value)


def _markdown(table):
    lines = ['| ' + ' | '.join(FIELDS) + ' |', '|' + '---|' * len(FIELDS)]
    lines += ['| ' + ' | '.join(_cell(row[f]) for f in FIELDS) + ' |' for row in table]
    note = ('CER = items_needing_suffix / items; LCR = semantic_collision_items / items. The collision rate is confounded with the branching '
            'and the number of semantic levels; B256D2 shares the (c0, c1) partition with B256D3 and differs only in numbering position, '
            'rule and the constant position, so it is not a collision control. Balance of B256D2 is judged on the three unpadded columns.')
    return '\n'.join(lines) + '\n\n' + note + '\n'


def write_outputs(points, directory):
    """Write curve.csv and curve.md (and curve.png when matplotlib is installed) into a new directory; nothing is overwritten."""
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    table = rows(points)
    written = [directory / 'curve.csv', directory / 'curve.md']
    with open(written[0], 'x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for row in table:
            writer.writerow([_cell(row[f]) for f in FIELDS])
    with open(written[1], 'x') as handle:
        handle.write(_markdown(table))
    if has_matplotlib():  # pragma: no cover - optional dependency, not installed in the project environment
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, key in zip(axes, ('branching', 'CER')):
            xs = [r[key] for r in table if r[key] is not None and r['hits_s42'] is not None]
            ys = [r['hits_s42'] for r in table if r[key] is not None and r['hits_s42'] is not None]
            labels = [r['label'] for r in table if r[key] is not None and r['hits_s42'] is not None]
            axis.scatter(xs, ys)
            for x, y, name in zip(xs, ys, labels):
                axis.annotate(name, (x, y))
            axis.set_xlabel(key)
            axis.set_ylabel('hits (seed 42)')
        figure.savefig(directory / 'curve.png')
        written.append(directory / 'curve.png')
    return written
