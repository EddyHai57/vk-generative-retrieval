"""FEAT-04b P user-tower / candidate-side variants on the F4-T010 anchor (new file; FEAT-02/FEAT-03 modules are used through the module, never modified).

``variant`` switch (one per run; the item tower is the anchor's unless a later variant says otherwise):
  f4_control          equivalence fixture for the anchor F4 (id_mean + F3 + F4), tests only, never a real run
  f4_recency_pool     chain A = Q2-T010: F4 event vectors pooled with 13 learnable recency logits (init exactly 0); delegated to the FEAT-03 Q2 functions
  f4_item_popularity  C: candidate side only, cand(i) = items(i) + supported(i) * (EX[expo bucket] + FW[first-week bucket]) from ITEM_STATS_V1; the user tower is the anchor's
  f4_item_quality     D: candidate side only, cand(i) = items(i) + supported(i) * (RT[watch-ratio bucket] + LK[smoothed like-rate bucket])
  f4_recency_behavior E: chain A plus six behaviour-bit logits beta (init exactly 0): l_k = u[pb] + v[gb] + sum_b bit_b(mask_k) * beta_b; beta acts on the pooling logit only
Seams are resolved at call time through the modules (``f3.<name>``, ``f2.<name>``): the mutation tests patch the module attribute, so this file never
writes ``from .retrieval_feat03 import x`` (K2). ``Feat04bIDMean`` bypasses ``Feat03IDMean.__init__`` (it only accepts FEATURES3) and calls the
FEAT-02 constructor with the FEAT-03 base feature string, exactly what ``Feat03IDMean`` does after its own check.
"""
import hashlib
import json

import torch

from . import item_stats
from . import retrieval_feat02 as f2
from . import retrieval_feat03 as f3
from .history_side import FEEDBACK_BITS
from .retrieval_features import ratio_bucket, timespent_bucket
from .retrieval_hp import init_embeddings
from .training import seed_all

CONTROL = 'f4_control'
A_FEATURE = 'f4_recency_pool'
E_FEATURE = 'f4_recency_behavior'
FEATURES4B = ('f4_control', 'f4_recency_pool', 'f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')
IMPLEMENTED = (CONTROL, A_FEATURE, E_FEATURE, 'f4_item_popularity', 'f4_item_quality')
REAL_FEATURES4B = ('f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')       # the variants this runner may submit as real runs (later features)
BEHAVIOR_BITS = FEEDBACK_BITS                                          # bit b of side[..., 0] is BEHAVIOR_BITS[b] (history_side.FEEDBACK_BITS order)


# ---- C/D bucket constants: the AUDITED values (configs/feat04b/stats-bucket-constants.json = the train-only audit stdout, verbatim). The audit took the pre-set grid
# (EXPO 2..512, FIRST 1..23, RATIO 200..1250, LIKE 5..160) and applied the 2% merge rule mechanically: EXPO lost its edge 2, RATIO its edge 1250, LIKE its edge 5, FIRST
# stayed. Every bucket function reads these module globals at call time (a mutation test patches them); guard_stats_constants_match_audit compares them with the JSON.
EXPO_EDGES = (4, 8, 16, 32, 64, 128, 256, 512)           # n (train exposure rows)        -> 9 buckets
FIRST_EDGES = (1, 3, 6, 10, 15, 20, 23)                  # first_week (observed-item proxy) -> 8 buckets
RATIO_EDGES = (200, 350, 500, 650, 800, 1000)            # mean watch ratio in permille   -> 7 buckets
LIKE_EDGES = (10, 20, 40, 80, 160)                       # smoothed like rate in permille -> 6 buckets
LIKE_K = 10                                              # smoothing strength (pseudo-exposures)
LIKE_P0 = 28                                             # global like rate in permille: round(1000 * 105700 / 3720328), from the audit
STATS_SHA256 = '25f6cc4cb2ea11ab395173d136ac7b26576b43c63cf47970bc69e6e41f48b685'      # audited sidecar content digest; load_item_stats(expected_sha256=...) refuses any other
STAT_VARIANTS = ('f4_item_popularity', 'f4_item_quality')
# per C/D variant: (table attribute, bucket-id buffer, name of the edge constant); the table has len(edges) + 2 rows (PAD row 0, buckets 1..len+1)
STAT_LAYOUT = {'f4_item_popularity': (('expo_table', 'xb', 'EXPO_EDGES'), ('first_table', 'fb', 'FIRST_EDGES')),
               'f4_item_quality': (('ratio_stat_table', 'rb', 'RATIO_EDGES'), ('like_table', 'lb', 'LIKE_EDGES'))}


def expo_bucket(n):
    """1 + #{e in EXPO_EDGES : n >= e}."""
    return 1 + sum(1 for e in EXPO_EDGES if n >= e)


def first_bucket(first_week):
    """1 + #{e in FIRST_EDGES : first_week >= e}."""
    return 1 + sum(1 for e in FIRST_EDGES if first_week >= e)


def ratio_bucket4(ratio_pm_sum, n):
    """1 + #{e in RATIO_EDGES : ratio_pm_sum >= e * n}: the integer form of 'mean ratio in permille >= e'."""
    return 1 + sum(1 for e in RATIO_EDGES if ratio_pm_sum >= e * n)


def like_bucket(likes, n):
    """1 + #{e in LIKE_EDGES : 1000*likes + LIKE_K*LIKE_P0 >= e*(n + LIKE_K)}: the smoothed like rate in permille against each edge, in integers."""
    return 1 + sum(1 for e in LIKE_EDGES if 1000 * likes + LIKE_K * LIKE_P0 >= e * (n + LIKE_K))


def stat_bucket_ids(stats, catalog, idmap, size):
    """Four long vectors over the model's index space ``size`` (row 0 = PAD = 0): ``{'xb', 'fb', 'rb', 'lb'}`` = expo / first / ratio / like bucket id of each catalog item.
    ``stats`` is the ITEM_STATS_V1 table in catalog order; the row of a catalog item is ``idmap[item]`` (mirrors ``retrieval_feat02_run._author_index``)."""
    catalog = list(catalog)
    if any(name not in stats or len(stats[name]) != len(catalog) for name in item_stats.FIELDS):
        raise ValueError('stats columns must hold one entry per catalog item')
    ids = {name: torch.zeros(size, dtype=torch.long) for name in ('xb', 'fb', 'rb', 'lb')}
    for k, item in enumerate(catalog):
        row = idmap.get(item)
        if type(row) is not int or not 1 <= row < size:
            raise ValueError('catalog item without a valid model index')
        n, first, likes, ratio = (stats[name][k] for name in item_stats.FIELDS)
        ids['xb'][row], ids['fb'][row] = expo_bucket(n), first_bucket(first)
        ids['rb'][row], ids['lb'][row] = ratio_bucket4(ratio, n), like_bucket(likes, n)
    return ids


class Feat04bIDMean(f3.Feat03IDMean):
    def __init__(self, variant, content, supported_indices, max_len, stat_buckets=None):
        if variant not in FEATURES4B:
            raise ValueError('unregistered FEAT-04b variant')
        f2.Feat02IDMean.__init__(self, f3.BASE_FEATURE, content, supported_indices, max_len)      # bypass Feat03IDMean.__init__ (FEATURES3 only)
        object.__setattr__(self, 'variant', variant)
        object.__setattr__(self, 'stat_buffers', ())
        if variant not in STAT_VARIANTS:
            if stat_buckets is not None:
                raise ValueError('stat_buckets belong to the C/D variants only')
            return
        if type(stat_buckets) is not dict:
            raise ValueError('C/D need the stat bucket ids (a dict of long vectors)')
        names = []
        for _, buffer, edges in STAT_LAYOUT[variant]:                                   # only the buffers this variant reads: a constant of the other variant cannot touch this run
            ids = stat_buckets.get(buffer)
            if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.shape != (len(content),) or int(ids[0]) != 0:
                raise ValueError(f'{buffer}: a long vector over the index space with PAD row 0 is required')
            if int(ids.min()) < 0 or int(ids.max()) > len(globals()[edges]) + 1:
                raise ValueError(f'{buffer}: bucket id outside the table')
            self.register_buffer(buffer, ids.detach().clone()); names.append(buffer)
        object.__setattr__(self, 'stat_buffers', tuple(names))

    def candidate_items(self, indices):
        """Item vectors used as candidates in train_step and in the evaluation export. C/D: items + supported * (two bucket-table terms); the others: items."""
        encoded = self.items(indices)
        return encoded + candidate_stat_terms(self, indices) if self.variant in STAT_VARIANTS else encoded

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name in self.stat_buffers:                                                  # the bucket ids are immutable data, not weights
            expected = getattr(self, name); actual = state_dict.get(name)
            if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
                raise ValueError('checkpoint immutable stat bucket ids mismatch')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def history_items(self, indices):
        """Item vectors the user tower reads for history events (default = items; the C/D seam keeps this path untouched)."""
        return self.items(indices)


    def users(self, histories, side=None):
        if self.variant not in IMPLEMENTED:
            raise ValueError('variant not implemented yet')
        valid = self._check(histories, side)
        x = event_vectors4(self, histories, side, valid)
        if self.variant in (A_FEATURE, E_FEATURE):
            pos_ids = f2.position_bucket(valid)
            gap_ids = f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid)
            logits = f3.recency_logits(self.recency_logit_pos, self.recency_logit_gap, pos_ids, gap_ids, valid)
            if self.variant == E_FEATURE:
                logits = logits + behavior_logits(self.behavior_beta, side[..., 0], valid)         # beta acts on the logit path only; at beta == 0 this adds exactly 0.0
            return f3.weighted_mean(x, f3.pooled_weights(logits, self._pool_mask(histories, valid)))
        return x.sum(1) / valid.sum(1, keepdim=True)


def event_vectors4(model, histories, side, valid):
    """Per-event vectors (B, W, 64); one left-to-right addition at a time, in the Feat02IDMean.users order (f4_control must be bit-equal):
    history_items, +T, +R, +P, +G (P first, then G; never P+G first). Every variant takes x_k from here."""
    x = model.history_items(histories)
    x = x + model.timespent_bucket(timespent_bucket(side[..., 1], valid))
    x = x + model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    x = x + model.position_table(f2.position_bucket(valid))
    return x + model.weekgap_table(f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid))


def candidate_stat_terms(model, indices):
    """C/D: supported(i) * (T1[b1(i)] + T2[b2(i)]) with the variant's two tables and its two long bucket-id buffers; exactly 0 for unsupported items and PAD."""
    (first, first_buffer, _), (second, second_buffer, _) = STAT_LAYOUT[model.variant]
    terms = getattr(model, first)(getattr(model, first_buffer)[indices]) + getattr(model, second)(getattr(model, second_buffer)[indices])
    return terms * model.supported[indices].unsqueeze(-1).to(terms.dtype)


def init_stat_tables(model, variant):
    """C/D: for each table in order: construct (it consumes a default draw), N(0, 0.02), clear the PAD row. Never ``f2._small_init``: written out here so a table
    without a padding_idx can not be wiped by ``weight[None].zero_()``. Called after the P / G draws, so the anchor tables equal f4_control's."""
    for table, _, edges in STAT_LAYOUT[variant]:
        setattr(model, table, torch.nn.Embedding(len(globals()[edges]) + 2, f2.EMBEDDING_DIM, padding_idx=0))
        with torch.no_grad():
            getattr(model, table).weight.normal_(0.0, 0.02)
            getattr(model, table).weight[0].zero_()


def stat_bucket_ids_sha256(model):
    """sha256 of the bucket-id buffers THIS variant reads (C: xb, fb; D: rb, lb): the run binding names the exact ids the model was built with."""
    digest = hashlib.sha256()
    for name in model.stat_buffers:
        digest.update(name.encode()); digest.update(getattr(model, name).cpu().numpy().tobytes())
    return digest.hexdigest()


def behavior_logits(beta, mask, valid):
    """(B, W) float: sum_b bit_b(mask_k) * beta_b with bit b = (mask >> b) & 1 in the BEHAVIOR_BITS (= FEEDBACK_BITS) order. PAD slots give 0; beta == 0 gives exactly 0.0."""
    shifts = torch.arange(len(BEHAVIOR_BITS), device=mask.device)
    bits = ((mask.unsqueeze(-1) >> shifts) & 1).to(beta.dtype) * valid.unsqueeze(-1).to(beta.dtype)
    return bits @ beta


def init_behavior_beta(model):
    """E: the six behaviour-bit scalars as one nn.Parameter, exactly 0 (so the start equals chain A bit for bit). No random draw."""
    model.behavior_beta = torch.nn.Parameter(torch.zeros(len(BEHAVIOR_BITS)))


def make_feat04b_model(variant, content, supported_indices, max_len, seed, stat_buckets=None):
    """Always init=small. Draw order (the FEAT-03 one): item, T, R, small-init T,R, P, G, small-init P,G; chain A then the 13 logits (exactly 0, no draw).

    C/D then draw their two bucket tables (``init_stat_tables``); E draws nothing extra (beta is zeros). ``stat_buckets`` (``stat_bucket_ids``) is required for C/D and refused otherwise.
    """
    if variant not in FEATURES4B:
        raise ValueError('unregistered FEAT-04b variant')
    if variant not in IMPLEMENTED:
        raise ValueError('variant not implemented yet')
    if max_len not in (20, 50, 100):
        raise ValueError('unregistered history length')
    if (stat_buckets is not None) != (variant in STAT_VARIANTS):
        raise ValueError('stat_buckets are required for the C/D variants and refused for the others')
    seed_all(seed)
    model = Feat04bIDMean(variant, content, supported_indices, max_len, stat_buckets)
    init_embeddings(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    f2._small_init(model, model.add_anchor_tables())
    f3.draw_variant_tables(model)
    if variant in STAT_VARIANTS:
        init_stat_tables(model, variant)
    f3.drop_unused_tables(model, variant)                              # no table is dropped for these variants; it recomputes init_report (now including the stat tables)
    if variant in (A_FEATURE, E_FEATURE):
        f3.init_recency_logits(model)
    if variant == E_FEATURE:
        init_behavior_beta(model)
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    return model


def _with_entries(report, entries):
    """Add by_table entries to a FEAT-03 style report and recompute its totals."""
    by_table = {**report['by_table'], **entries}
    return {'total': sum(v['total'] for v in by_table.values()), 'trainable_effective': sum(v['trainable'] for v in by_table.values()), 'by_table': by_table}


def _beta_entry():
    return {'behavior_beta': {'total': len(BEHAVIOR_BITS), 'trainable': len(BEHAVIOR_BITS)}}


def parameter_report4b(model):
    """Parameters added on top of the item embedding (FEAT-03 convention; PAD rows are not trainable)."""
    report = f3.parameter_report3(model)
    return _with_entries(report, _beta_entry()) if hasattr(model, 'behavior_beta') else report


def expected_param_counts4b(variant):
    """Pure formula for ``parameter_report4b``."""
    if variant not in FEATURES4B:
        raise ValueError('unregistered FEAT-04b variant')
    if variant not in IMPLEMENTED:
        raise ValueError('parameter counts not defined for this variant yet')
    if variant == E_FEATURE:
        return _with_entries(f3.expected_param_counts3(A_FEATURE), _beta_entry())
    if variant in STAT_VARIANTS:                                       # the control's tables plus the two bucket tables (a PAD row each)
        rows = {table: len(globals()[edges]) + 2 for table, _, edges in STAT_LAYOUT[variant]}
        return _with_entries(f3.expected_param_counts3(CONTROL), {t: {'total': n * f2.EMBEDDING_DIM, 'trainable': (n - 1) * f2.EMBEDDING_DIM} for t, n in rows.items()})
    return f3.expected_param_counts3(variant)


def bucket_spec_sha256(variant):
    """Per-variant digest of every constant the variant relies on (stored in that variant's run binding only).

    f4_control / chain A: {FEAT-03 spec digest, logit sizes [7, 6, 6], FEATURES4B} plus the variant name; E adds the FEEDBACK_BITS order; C/D add their own constants in a later feature.
    """
    if variant not in FEATURES4B:
        raise ValueError('unregistered FEAT-04b variant')
    if variant not in IMPLEMENTED:
        raise ValueError('bucket spec not defined for this variant yet')
    spec = {'feat03_bucket_spec_sha256': f3.bucket_spec3_sha256(), 'features4b': list(FEATURES4B), 'variant': variant,
            'logit_sizes': [f3.RECENCY_PARAMS['recency_logit_pos'], f3.RECENCY_PARAMS['recency_logit_gap'], len(BEHAVIOR_BITS)]}
    if variant == E_FEATURE:
        spec['feedback_bits'] = list(BEHAVIOR_BITS)                      # E additionally depends on the bit order
    if variant == 'f4_item_popularity':                                  # C and D hash only their own constants (design 3.4): a drifted constant of the other one cannot break this variant's verify
        spec = {'feat03_bucket_spec_sha256': f3.bucket_spec3_sha256(), 'features4b': list(FEATURES4B), 'variant': variant, 'expo_edges': list(EXPO_EDGES),
                'first_edges': list(FIRST_EDGES), 'stat_cutoff_week': item_stats.STAT_CUTOFF_WEEK}
    if variant == 'f4_item_quality':
        spec = {'feat03_bucket_spec_sha256': f3.bucket_spec3_sha256(), 'features4b': list(FEATURES4B), 'variant': variant, 'ratio_edges': list(RATIO_EDGES),
                'like_edges': list(LIKE_EDGES), 'like_k': LIKE_K, 'like_p0': LIKE_P0, 'ratio_cap_pm': item_stats.RATIO_CAP_PM, 'stat_cutoff_week': item_stats.STAT_CUTOFF_WEEK}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
