"""P-HP-01 H7 trigger + selection (pre-registered; new file). Pure functions plus a ``python -m`` CLI.

Rule (machine-readable copy in configs/php01/p-php01-h7-combo-rule.json, checked against these constants at start):
  winner      a completed, verified H1-H6 seed42 run with delta R@50 >= 0.002 over the F4 anchor; on n=2520 queries that is
              >= 6 more hits (5 hits = 0.001984 is not enough), so the comparison is done on integer hits
  trigger     winners cover >= 2 DIFFERENT knobs (temperature H1/H2, dim H3/H4, learning rate H5/H6); two levels of one knob count once
  H7          anchor values with every winning knob set to its best winning level (R@50 -> N@50 -> enumeration order); else H7 = void
  selection   anchor + verified H1-H6 (+ H7 once run) ranked by R@50 -> N@50 -> enumeration order (anchor first)
Failed runs are reported, never winners, and their slot is not refunded. The generated H7 config is committed to Git
(configs/php01/p-php01-h7-combo-s42.json) and frozen before it is enqueued.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from .retrieval_ablation_run import read_pinned, strict_json
from .retrieval_php01_run import ANCHOR_VALUES, COORDINATES, KNOB_FIELD, KNOBS, default_config, validate_config

THRESHOLD = 0.002
N_QUERIES = 2520
MIN_DELTA_HITS = 6
MIN_GROUPS = 2
GROUPS = {'temperature': ('H1', 'H2'), 'dim': ('H3', 'H4'), 'learning_rate': ('H5', 'H6')}
SINGLE = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6')
ORDER = ('anchor', *SINGLE, 'H7')
ANCHOR_CONFIG = 'configs/feat02/p-feat02-f4-recency-s42.json'
ANCHOR_CONFIG_SHA256 = 'bead30cb6d635e7b2a204f0006f67ef1c4dd9c61d70193bedf2ec2061decc7ea'
H7_REPO_PATH = 'configs/php01/p-php01-h7-combo-s42.json'
REPO = Path(__file__).resolve().parents[2]
RULE_PATH = REPO / 'configs' / 'php01' / 'p-php01-h7-combo-rule.json'
COMPARABLE = ('inputs', 'rows_sha256', 'queries_sha256', 'catalog_sha256', 'support_sha256', 'side_sha256', 'weeks_sha256', 'bucket_spec_sha256')


def rule_value():
    return {'schema': 'p-php01-h7-rule-v1', 'task': 'P-HP-01', 'anchor_config': ANCHOR_CONFIG, 'anchor_config_sha256': ANCHOR_CONFIG_SHA256,
            'threshold_recall50': THRESHOLD, 'n_queries': N_QUERIES, 'min_delta_hits': MIN_DELTA_HITS, 'min_distinct_winning_knobs': MIN_GROUPS,
            'groups': {g: list(k) for g, k in GROUPS.items()},
            'knobs': {k: {'field': KNOB_FIELD[k], 'value': KNOBS[k][KNOB_FIELD[k]]} for k in SINGLE},
            'enumeration_order': list(ORDER), 'selection': ['recall@50', 'ndcg@50', 'enumeration_order'],
            'winner_requires': ['status_complete', 'verify_verified', 'seed_42'],
            'h7_config_path': H7_REPO_PATH, 'h7_config_committed_before_enqueue': True}


def load_rule(path=RULE_PATH):
    value = strict_json(Path(path).read_text())
    if value != rule_value() or MIN_DELTA_HITS != math.ceil(THRESHOLD * N_QUERIES - 1e-9):
        raise ValueError('H7 rule file differs from the pre-registered constants')
    return value


# ---------------------------------------------------------------- seams (mutation tests patch these)

def knob_group(knob):
    return KNOB_FIELD[knob]


def is_winner(delta_hits):
    return delta_hits >= MIN_DELTA_HITS


def selection_key(entry):
    return (-entry['r50'], -entry['n50'], ORDER.index(entry['name']))


def best_level(group_entries):
    return min(group_entries, key=selection_key)


# ---------------------------------------------------------------- pure decision

def _entry(result):
    return {'name': result['name'], 'r50': result['r50'], 'n50': result['n50'], 'hits': result['hits']}


def _ranking(entries):
    ranking = sorted(entries, key=selection_key)
    return {'ranking': ranking, 'winner': ranking[0]['name']}


def decide(anchor, results, failed):
    """``results``: knob -> loaded result of a completed H1-H6 run; ``failed``: knob -> reason. Together exactly H1-H6."""
    if anchor.get('name') != 'anchor' or not anchor.get('verified'):
        raise ValueError('verified anchor result required')
    if set(results) & set(failed) or set(results) | set(failed) != set(SINGLE):
        raise ValueError('every H1-H6 knob needs exactly one of a result or a failure reason')
    rows = {}; winners = []
    for knob in SINGLE:
        if knob in failed:
            rows[knob] = {'status': 'failed', 'reason': failed[knob], 'winner': False}
            continue
        r = results[knob]
        if r.get('name') != knob:
            raise ValueError('result/knob name mismatch')
        delta_hits = r['hits'] - anchor['hits']
        win = bool(r['verified']) and is_winner(delta_hits)
        rows[knob] = {'status': 'complete', 'r50': r['r50'], 'n50': r['n50'], 'hits': r['hits'], 'delta_r50': r['r50'] - anchor['r50'],
                      'delta_hits': delta_hits, 'delta_n50': r['n50'] - anchor['n50'], 'winner': win, 'verified': bool(r['verified']),
                      'best_epoch': r.get('best_epoch'), 'stop_epoch': r.get('stop_epoch'), 'manifest_sha256': r.get('manifest_sha256')}
        if win:
            winners.append(r)
    by_group = {}
    for r in winners:
        by_group.setdefault(knob_group(r['name']), []).append(r)
    groups = [g for g in GROUPS if g in by_group] + sorted(g for g in by_group if g not in GROUPS)
    triggered = len(groups) >= MIN_GROUPS
    candidates = [_entry(anchor)] + [_entry(results[k]) for k in SINGLE if k in results and results[k]['verified']]
    h7 = 'void'
    if triggered:
        values = dict(ANCHOR_VALUES)
        for g in groups:
            level = best_level(by_group[g])
            values[KNOB_FIELD[level['name']]] = KNOBS[level['name']][KNOB_FIELD[level['name']]]
        config = validate_config({**default_config(), 'knob': 'H7', **values})
        h7 = {'values': values, 'config': config, 'levels': {g: best_level(by_group[g])['name'] for g in groups}}
    return {'rule': {'threshold_recall50': THRESHOLD, 'n_queries': N_QUERIES, 'min_delta_hits': MIN_DELTA_HITS, 'min_distinct_winning_knobs': MIN_GROUPS,
                     'groups': {g: list(k) for g, k in GROUPS.items()}, 'enumeration_order': list(ORDER)},
            'anchor': {'r50': anchor['r50'], 'n50': anchor['n50'], 'hits': anchor['hits'], 'manifest_sha256': anchor.get('manifest_sha256')},
            'runs': rows, 'winner_groups': groups, 'triggered': triggered, 'h7': h7,
            'selection': {'stage': 'interim' if triggered else 'final', **_ranking(candidates)}}


def select(decision, h7_result):
    """Final selection. H7 void: ``h7_result`` must be None. Triggered: a loaded H7 result or ``{'failed': reason}``."""
    if not decision['triggered']:
        if h7_result is not None:
            raise ValueError('H7 is void: no H7 result may be selected')
        return decision['selection']
    if h7_result is None:
        raise ValueError('H7 was triggered: its result or failure reason is required')
    entries = list(decision['selection']['ranking'])
    if 'failed' not in h7_result:
        if h7_result.get('name') != 'H7':
            raise ValueError('H7 result name mismatch')
        if h7_result['verified']:
            entries.append(_entry(h7_result))
    return {'stage': 'final', **_ranking(entries)}


# ---------------------------------------------------------------- loading pinned run evidence

def _last_json_line(path, digest):
    data = Path(path).read_bytes()
    if type(digest) is not str or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('external hash mismatch (verify log)')
    lines = [line for line in data.decode().splitlines() if line.strip()]
    try:
        value = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return {}
    return value if type(value) is dict else {}


def load_result(name, entry):
    manifest = read_pinned(entry['manifest'], entry['manifest_sha256'])
    config = manifest.get('config'); selection = manifest.get('selection') or {}
    if name == 'anchor':
        anchor_config = strict_json((REPO / ANCHOR_CONFIG).read_text())
        if manifest.get('schema') != 'p-feat02-train-manifest-v1' or manifest.get('kind') != 'p_feat02_train' or config != anchor_config:
            raise ValueError('anchor must be the committed FEAT-02 F4 seed42 run')
    else:
        if manifest.get('schema') != 'p-php01-train-manifest-v1' or manifest.get('kind') != 'p_php01_train' or type(config) is not dict or config.get('knob') != name:
            raise ValueError(f'{name}: P-HP-01 manifest of this knob required')
        validate_config(config)
    if manifest.get('status') != 'complete' or config.get('seed') != 42:
        raise ValueError(f'{name}: completed seed42 run required')
    r50, n50 = selection['best_metric']
    hits = round(r50 * N_QUERIES)
    if abs(hits / N_QUERIES - r50) > 1e-9:
        raise ValueError(f'{name}: R@50 is not a hit count over {N_QUERIES} queries')
    log = _last_json_line(entry['verify_log'], entry.get('verify_log_sha256'))
    verified = log.get('status') == 'verified' and log.get('best_epoch') == selection.get('best_epoch')
    return {'name': name, 'r50': r50, 'n50': n50, 'hits': hits, 'best_epoch': selection.get('best_epoch'), 'stop_epoch': selection.get('epoch'),
            'verified': verified, 'manifest_sha256': entry['manifest_sha256'], 'binding': manifest.get('binding') or {}, 'config': config}


def load_results(spec):
    """p-php01-results-v1: {anchor: entry, runs: {H1..H6[, H7]: entry | {failed: reason}}}; every H must be comparable to the anchor."""
    if type(spec) is not dict or spec.get('schema') != 'p-php01-results-v1' or set(spec) != {'schema', 'anchor', 'runs'}:
        raise ValueError('p-php01-results-v1 spec required')
    if not set(SINGLE) <= set(spec['runs']) <= set(SINGLE) | {'H7'}:
        raise ValueError('results spec must list H1-H6 (and optionally H7)')
    anchor = load_result('anchor', spec['anchor'])
    if not anchor['verified']:
        raise ValueError('anchor run is not verified')
    results = {}; failed = {}; h7 = None
    for name, entry in spec['runs'].items():
        if set(entry) == {'failed'}:
            if type(entry['failed']) is not str or not entry['failed']:
                raise ValueError('failure reason required')
            value = None; failed[name] = entry['failed']
        else:
            value = load_result(name, entry)
            for key in COMPARABLE:
                if value['binding'].get(key) != anchor['binding'].get(key):
                    raise ValueError(f'{name}: not comparable with the anchor ({key})')
            results[name] = value
        if name == 'H7':
            h7 = entry if value is None else value
            results.pop('H7', None); failed.pop('H7', None)
    return {'anchor': anchor, 'results': results, 'failed': failed, 'h7': h7}


# ---------------------------------------------------------------- CLI

def _canonical(value):
    return json.dumps(value, indent=1, sort_keys=True) + '\n'


def _write_new(path, text):
    """Atomic, never overwrites: write ``.partial`` then hard-link to the target (fails if it exists)."""
    path = Path(path); tmp = path.with_name(path.name + '.partial')
    with tmp.open('x') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return hashlib.sha256(text.encode()).hexdigest()


def _strip(decision):
    return {k: decision[k] for k in ('runs', 'winner_groups', 'triggered', 'selection')} | {'h7': decision['h7'] if decision['h7'] == 'void' else
                                                                                         {k: decision['h7'][k] for k in ('values', 'config', 'levels')}}


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description='P-HP-01 H7 trigger (decide) and final selection (select)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('decide'); p.add_argument('--results', required=True); p.add_argument('--results-sha', required=True)
    p.add_argument('--output', required=True); p.add_argument('--h7-config', required=True)
    p = sub.add_parser('select'); p.add_argument('--decision', required=True); p.add_argument('--decision-sha', required=True)
    p.add_argument('--results', required=True); p.add_argument('--results-sha', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        rule_sha = hashlib.sha256(RULE_PATH.read_bytes()).hexdigest(); load_rule()
        loaded = load_results(read_pinned(args.results, args.results_sha))
        if args.command == 'decide':
            if loaded['h7'] is not None:
                raise ValueError('decide takes H1-H6 only')
            for path in (args.output, args.h7_config):
                if Path(path).exists():
                    raise ValueError(f'output exists, not overwritten: {Path(path).name}')
            decision = decide(loaded['anchor'], loaded['results'], loaded['failed'])
            decision = {'schema': 'p-php01-combo-decision-v1', 'results_sha256': args.results_sha, **decision,
                        'rule': {**decision['rule'], 'rule_file_sha256': rule_sha}}
            h7_sha = None
            if decision['triggered']:
                h7_sha = _write_new(args.h7_config, _canonical(decision['h7']['config']))
                decision['h7'] = {**decision['h7'], 'config_sha256': h7_sha, 'repo_path': H7_REPO_PATH}
            sha = _write_new(args.output, _canonical(decision))
            print(json.dumps({'status': 'decided', 'triggered': decision['triggered'], 'decision_sha256': sha, 'h7_config_sha256': h7_sha})); return 0
        decision = read_pinned(args.decision, args.decision_sha)
        if decision.get('schema') != 'p-php01-combo-decision-v1':
            raise ValueError('p-php01-combo-decision-v1 required')
        if Path(args.output).exists():
            raise ValueError('output exists, not overwritten')
        if _strip(decide(loaded['anchor'], loaded['results'], loaded['failed'])) != _strip(decision):
            raise ValueError('decision not reproducible from the H1-H6 results')
        h7 = loaded['h7']
        if decision['triggered'] and h7 is not None and 'failed' not in h7 and h7['config'] != decision['h7']['config']:
            raise ValueError('H7 run config differs from the decided combo')
        selection = {'schema': 'p-php01-selection-v1', 'decision_sha256': args.decision_sha, 'results_sha256': args.results_sha, **select(decision, h7)}
        if h7 is not None and 'failed' not in h7:
            selection['h7'] = {'r50': h7['r50'], 'n50': h7['n50'], 'hits': h7['hits'], 'verified': h7['verified'], 'manifest_sha256': h7['manifest_sha256']}
        sha = _write_new(args.output, _canonical(selection))
        print(json.dumps({'status': 'selected', 'stage': selection['stage'], 'winner': selection['winner'], 'selection_sha256': sha})); return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'ERROR P-HP-01 select gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
