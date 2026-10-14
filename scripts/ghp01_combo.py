"""G-HP-01 combination trigger (G7) and card 1.5 selection (pure standard library; prints ONE JSON line).

Pre-registered rule: a run wins when it succeeded, was verified and beats the anchor A1 by at least 6 of 2520 validation hits
(= dR@50 >= 0.002; 5 hits = 0.001984 is not enough). The combination is triggered when at least 2 DISTINCT knobs have a winner;
knob groups: width {G1}, learning rate {G2, G3}, layers {G4}, dropout {G5}, SID depth {G6 - deferred}. G2 and G3 winning
together count as one knob. The combination config = A1 with each winning knob set to its best level (R@50 -> N@50 ->
enumeration order). This batch judges G1-G5 only (status 'provisional', G6 pending); nothing is enqueued here.
Selection (card 1.5): candidates = anchor + verified G runs, R@50 -> N@50 -> enumeration order anchor < G1 < ... < G7.
The registry constants are copied from ``vkgr.generative_hp_arch`` (no import); a test pins them equal.
"""
import argparse
import hashlib
import json
from pathlib import Path

QUERIES = 2520
MIN_DELTA_HITS = 6
MIN_DISTINCT_KNOBS = 2
SHAPE_KEYS = ('d_model', 'heads', 'head_dim', 'encoder_layers', 'decoder_layers', 'ff', 'dropout')
HP_SHAPES = tuple((d, h, hd, n, n, ff, p) for d, h, hd, ff in ((128, 6, 64, 1024), (256, 6, 128, 2048)) for n in (4, 2) for p in (0.1, 0.0))
LEARNING_RATES = (.0001, .0003, .0006)
KNOBS = {'width': ('d_model', 'ff', 'head_dim'), 'lr': ('learning_rate',), 'layers': ('encoder_layers', 'decoder_layers'), 'dropout': ('dropout',)}
KNOB_OF = {'G1': 'width', 'G2': 'lr', 'G3': 'lr', 'G4': 'layers', 'G5': 'dropout'}
ORDER = ('anchor', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7')
TASK_ID = 'G-HP-01'


def hits(r50, queries=QUERIES):
    count = round(r50 * queries)
    if abs(count / queries - r50) >= 1e-12:
        raise ValueError(f'R@50 {r50} is not an integer hit count over {queries} queries')
    return count


def config_bytes(config):
    return (json.dumps(config, indent=1, sort_keys=True) + '\n').encode()


def _rank(entry):
    """Sort key: higher R@50, then higher N@50, then earlier enumeration order."""
    return (-entry['r50'], -entry['n50'], ORDER.index(entry['id']))


def _eligible(run):
    return run.get('status') == 'succeeded' and run.get('verified') is True and run.get('r50') is not None


def decide(anchor, runs):
    if anchor.get('verified') is not True:
        raise ValueError('anchor must be verified')
    anchor_hits = hits(anchor['r50'])
    deltas, winners, excluded = {}, [], []
    for run in runs:
        if run['id'] not in KNOB_OF:
            raise ValueError(f"unknown run id {run['id']}")
        if not _eligible(run):
            excluded.append(run['id'])
            continue
        deltas[run['id']] = hits(run['r50']) - anchor_hits
        if deltas[run['id']] >= MIN_DELTA_HITS:
            winners.append(run)
    knobs_won = [k for k in KNOBS if any(KNOB_OF[r['id']] == k for r in winners)]
    triggered = len(knobs_won) >= MIN_DISTINCT_KNOBS
    levels, combo, combo_sha = {}, None, None
    if triggered:
        combo = {**anchor['config'], 'task_id': TASK_ID, 'seed': 42}
        for knob in knobs_won:
            best = sorted((r for r in winners if KNOB_OF[r['id']] == knob), key=_rank)[0]
            levels[knob] = best['id']
            combo.update({k: best['config'][k] for k in KNOBS[knob]})
        if tuple(combo[k] for k in SHAPE_KEYS) not in HP_SHAPES or combo['learning_rate'] not in LEARNING_RATES:
            raise ValueError('combination config outside the G-HP registry')
        combo_sha = hashlib.sha256(config_bytes(combo)).hexdigest()
    candidates = [{'id': 'anchor', 'r50': anchor['r50'], 'n50': anchor['n50']}] + [{'id': r['id'], 'r50': r['r50'], 'n50': r['n50']} for r in runs if _eligible(r)]
    return {'triggered': triggered, 'knobs_won': knobs_won, 'winners': [r['id'] for r in winners], 'levels': levels, 'excluded': excluded,
            'anchor_hits': anchor_hits, 'delta_hits': deltas, 'combo_config': combo, 'combo_config_sha256': combo_sha,
            'selection': sorted(candidates, key=_rank)[0]['id'], 'judged_on': ['G1', 'G2', 'G3', 'G4', 'G5'], 'pending': ['G6'], 'status': 'provisional',
            'rule': {'min_delta_hits': MIN_DELTA_HITS, 'queries': QUERIES, 'min_distinct_knobs': MIN_DISTINCT_KNOBS, 'knob_of': KNOB_OF,
                     'text': '>= 2 distinct knobs each with a verified run beating the anchor by >= 6 of 2520 hits (dR@50 >= 0.002)'},
            'enqueued': False}


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _verified(path):
    path = Path(path)
    if not path.is_file():
        return False
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    try:
        return bool(lines) and json.loads(lines[-1]).get('status') == 'verified'
    except (json.JSONDecodeError, AttributeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description='G-HP-01 combination trigger + card 1.5 selection (one JSON line)')
    parser.add_argument('--anchor-manifest', required=True)
    parser.add_argument('--anchor-verify-stdout', required=True)
    parser.add_argument('--run', action='append', default=[], help='G1=<manifest>,<verify stdout>,<terminal.json>')
    args = parser.parse_args(argv)
    manifest = _read_json(args.anchor_manifest)
    if manifest is None:
        raise SystemExit('anchor manifest missing')
    metric = manifest['selection']['best_metric']
    anchor = {'id': 'anchor', 'r50': metric[0], 'n50': metric[1], 'verified': _verified(args.anchor_verify_stdout), 'config': manifest['config']}
    runs = []
    for spec in args.run:
        gid, paths = spec.split('=', 1)
        manifest_path, stdout_path, terminal_path = paths.split(',')
        value, terminal = _read_json(manifest_path), _read_json(terminal_path)
        metric = (value or {}).get('selection', {}).get('best_metric') or (None, None)
        runs.append({'id': gid, 'status': (terminal or {}).get('status', 'missing'), 'verified': _verified(stdout_path),
                     'r50': metric[0], 'n50': metric[1], 'config': (value or {}).get('config')})
    print(json.dumps(decide(anchor, runs), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
