"""Frozen four-route weighted RRF, with validation-only selection."""
from fractions import Fraction
from itertools import product
import math


def weight_candidates():
    seen = set()
    result = []
    for weights in product(range(3), repeat=4):
        if not any(weights):
            continue
        first = next(w for w in weights if w)
        ratio = tuple(Fraction(w, first) for w in weights)
        if ratio not in seen:
            seen.add(ratio)
            result.append(weights)
    return result


def weighted_rrf(route_rankings, weights, final_k=50, rank_constant=60):
    """Rank positions start at one; no candidate from a zero-weight route."""
    if (len(route_rankings) != 4 or len(weights) != 4 or
            any(type(w) is not int or w not in (0, 1, 2) for w in weights) or
            not any(weights) or type(final_k) is not int or final_k < 1 or
            rank_constant != 60):
        raise ValueError('invalid frozen fusion parameters')
    scores = {}
    for ranks, weight in zip(route_rankings, weights):
        if any(type(item) is not int or item < 0 for item in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('invalid or duplicate item rank')
        if not weight:
            continue
        for rank, item in enumerate(ranks, 1):
            scores[item] = scores.get(item, Fraction(0)) + Fraction(weight, rank_constant + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:final_k]


def redistribute_quota(active_routes, total=400):
    active = sorted(active_routes)
    if (not active or len(set(active)) != len(active) or
            any(type(r) is not int or r not in range(4) for r in active) or total != 400):
        raise ValueError('invalid frozen route quota')
    base, remainder = divmod(total, len(active))
    return {route: base + (i < remainder) for i, route in enumerate(active)}


def select_weights(candidate_metrics):
    """Caller supplies full eligible validation metrics with unchanged denominator."""
    rows = list(candidate_metrics)
    if not rows:
        raise ValueError('empty validation candidates')
    permitted = set(weight_candidates())
    seen = set()
    for row in rows:
        weights = tuple(row['weights'])
        if weights not in permitted or weights in seen:
            raise ValueError('duplicate or noncanonical weight candidate')
        seen.add(weights)
        for key in ('recall@50', 'ndcg@50'):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError('nonfinite or invalid validation metric')
    return tuple(min(rows, key=lambda r: (-r['recall@50'], -r['ndcg@50'],
                                         sum(w != 0 for w in r['weights']), tuple(r['weights'])))['weights'])
