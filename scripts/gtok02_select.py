"""GTOK-02 seed selection: a pure function that turns the verified seed-42 validation results of the eight sweep points into a proposal.

Rules (pre-registered, gtok02-design 3.7):
  1. rank by hits (descending), then NDCG@50 (descending), then the fixed enumeration order;
  2. top2 = the first two; near = every point within 10 hits of the best (inclusive);
  3. selected = top2 + near; the new configs among them get seeds 2024 and 2025 (B16 and B256D3 already have three seeds);
  4. rule E: B256D2 gets both seeds whenever its seed-42 run is verified, whatever its rank;
  5. k = number of configs that get seeds, trainings = 2k.
The output is a proposal only; nothing here enqueues anything. Only validation numbers are read.

Input: {label: {'recall50': float, 'ndcg50': float, 'verified': bool}}. hits = round(recall50 * 2520) and must be integral.
"""
import argparse
import json
import sys
from pathlib import Path

ORDER = ('B16', 'B32', 'B64', 'B128', 'B256D3', 'B512', 'B1024', 'B256D2')
NEW = ('B32', 'B64', 'B128', 'B512', 'B1024', 'B256D2')
N_QUERIES = 2520
NEAR_HITS = 10
SEEDS = [2024, 2025]
INTEGRALITY = 1e-6


def hits_of(recall50):
    """Integer hits behind a Recall@50 value; refuses a value that is not k / 2520."""
    hits = round(recall50 * N_QUERIES)
    if abs(recall50 * N_QUERIES - hits) >= INTEGRALITY:
        raise ValueError('Recall@50 is not an integer number of hits out of 2520')
    return hits


def select(results):
    unknown = sorted(set(results) - set(ORDER))
    if unknown:
        raise ValueError(f'unknown points: {unknown}')
    excluded = {label: 'missing' for label in ORDER if label not in results}
    ranked = []
    for label in ORDER:
        if label not in results:
            continue
        value = results[label]
        if value['verified'] is not True:
            excluded[label] = 'not verified'
            continue
        ranked.append({'label': label, 'hits': hits_of(value['recall50']), 'ndcg50': value['ndcg50']})
    ranked.sort(key=lambda r: (-r['hits'], -r['ndcg50'], ORDER.index(r['label'])))
    out = {'status': 'proposal', 'ranking': ranked, 'excluded': dict(sorted(excluded.items(), key=lambda kv: ORDER.index(kv[0]))), 'seeds': SEEDS}
    if not ranked:
        return {**out, 'max_hits': None, 'top2': [], 'near': [], 'selected': [], 'supplement': [], 'rule_e': False, 'k': 0, 'trainings': 0,
                'note': 'no verified point'}
    best = ranked[0]['hits']
    top2 = [r['label'] for r in ranked[:2]]
    near = [label for label in ORDER if any(r['label'] == label and r['hits'] >= best - NEAR_HITS for r in ranked)]
    selected = [label for label in ORDER if label in top2 or label in near]
    supplement = [label for label in selected if label in NEW]
    rule_e = any(r['label'] == 'B256D2' for r in ranked)
    if rule_e and 'B256D2' not in supplement:
        supplement.append('B256D2')
    out.update({'max_hits': best, 'top2': top2, 'near': near, 'selected': selected, 'supplement': supplement, 'rule_e': rule_e,
                'k': len(supplement), 'trainings': 2 * len(supplement)})
    if len(ranked) < 2:
        out['note'] = 'fewer than two verified points: the rule ran on what exists'
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='GTOK-02 seed-selection proposal (does not enqueue anything)')
    parser.add_argument('--input', required=True, help='JSON {label: {recall50, ndcg50, verified}}')
    parser.add_argument('--output', help='new file for the proposal JSON (printed when omitted)')
    args = parser.parse_args(argv)
    try:
        proposal = select(json.loads(Path(args.input).read_text()))
        text = json.dumps(proposal, sort_keys=True, indent=1) + '\n'
        if args.output:
            with open(args.output, 'x') as handle:
                handle.write(text)
        else:
            sys.stdout.write(text)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
