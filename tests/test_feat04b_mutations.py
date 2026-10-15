"""FEAT-04b mutation tests (M56-M59, M83 in FEAT-002; M44-M47, M70a/b, M71-M75, M77, M78, M82 for E and M60-M69, M76, M79, M80, M84 for C/D in FEAT-004; later features append the rest of M41-M86): each deliberately wrong implementation must be
caught by its named guard. ``detected`` (single patch) comes from tests/test_feat02_mutations.py and is imported, never copied; ``detected_multi``
stacks several patches (M70a needs two at once).
Every guard is first run on the unmodified code (must pass), then under the patch (must raise AssertionError), then again after the revert (must pass).
"""
import contextlib
import inspect
import random
import sys
import unittest
from pathlib import Path
from unittest import mock
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_mutations import detected  # noqa: E402,F401
import test_feat04b_model as tmb  # noqa: E402
import test_feat04b_run as trn  # noqa: E402
import test_feat04b_stats as ts  # noqa: E402


def detected_multi(case, patches, guards):
    """``patches`` is a list of ``(module_or_class, name, replacement)`` applied together."""
    for guard in guards:
        guard()
    with contextlib.ExitStack() as stack:
        for module, name, replacement in patches:
            stack.enter_context(mock.patch.object(module, name, replacement))
        for guard in guards:
            with case.assertRaises(AssertionError, msg=f'{[p[1] for p in patches]}: {guard.__name__} did not catch the mutation'):
                guard()
    for guard in guards:
        guard()


def named(fn, *args):
    """A guard with fixed arguments that still has a ``__name__`` (``detected`` prints it)."""
    def bound():
        return fn(*args)
    bound.__name__ = f"{fn.__name__}[{','.join(str(a) for a in args)}]"
    return bound


def reference_aggregate(first='min', count='all', zero='keep'):
    """A wrong re-implementation of ``item_stats.aggregate_rows`` with one switch turned: first_week from the last week (``first='max'``),
    exposure counted on like rows only (``count='likes'``), timespent == 0 rows dropped (``zero='drop'``) or refused (``zero='raise'``)."""
    from vkgr import item_stats as s

    def aggregate(partitions, catalog, durations):
        acc = {item: [0, None, 0, 0] for item in catalog}
        for week, events in partitions:
            if not 0 <= week <= s.STAT_CUTOFF_WEEK:
                raise ValueError('non-train week')
            for event in events:
                if event['timespent'] == 0 and zero == 'drop':
                    continue
                if event['timespent'] == 0 and zero == 'raise':
                    raise ValueError('timespent must be positive')
                entry = acc[event['item_id']]
                if count == 'all' or event['like']:
                    entry[0] += 1
                if first == 'max' or entry[1] is None:
                    entry[1] = week
                entry[2] += event['like']
                entry[3] += s.ratio_pm(event['timespent'], durations[event['item_id']])
        return acc
    return aggregate


class StatsMutationTests(unittest.TestCase):
    def test_m56_cutoff_relaxed_to_week_25(self):
        """The constructor accepts the validation week (a leak): the cutoff constant is read at call time."""
        from vkgr import item_stats
        detected(self, item_stats, 'STAT_CUTOFF_WEEK', 25, [ts.guard_stats_cutoff, ts.guard_stats_formula])

    def test_m57_first_week_takes_the_last_week(self):
        from vkgr import item_stats
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(first='max'), [ts.guard_stats_formula])

    def test_m58_exposure_counts_like_rows_only(self):
        from vkgr import item_stats
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(count='likes'), [ts.guard_stats_formula])

    def test_m59_ratio_neither_capped_nor_integer(self):
        from vkgr import item_stats
        detected(self, item_stats, 'ratio_pm', lambda timespent, duration: 1000 * timespent / duration, [ts.guard_stats_ratio_integer])
        detected(self, item_stats, 'ratio_pm', lambda timespent, duration: 1000 * timespent // duration, [ts.guard_stats_ratio_integer])    # integer but uncapped

    def test_m83_timespent_zero_rows_dropped_or_refused(self):
        from vkgr import item_stats
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(zero='drop'), [ts.guard_stats_timespent_zero])
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(zero='raise'), [ts.guard_stats_timespent_zero])

    def test_reference_aggregate_with_no_switch_equals_the_real_one(self):
        """The mutant factory is honest: with every switch off it reproduces the real aggregation (so a detection is due to the switch)."""
        from vkgr import item_stats
        self.assertEqual(reference_aggregate()(ts.world(), ts.CATALOG, ts.DURATIONS), item_stats.aggregate_rows(ts.world(), ts.CATALOG, ts.DURATIONS))

    def test_detected_multi_stacks_patches(self):
        from vkgr import item_stats
        detected_multi(self, [(item_stats, 'STAT_CUTOFF_WEEK', 25), (item_stats, 'aggregate_rows', reference_aggregate(first='max'))],
                       [ts.guard_stats_cutoff, ts.guard_stats_formula])


class EMutationTests(unittest.TestCase):
    """M44-M47 and M82 patch a point shared by chain A and E, so every one runs the A guards AND the E guards."""
    A_E_POOL = [tmb.guard_a_formula, tmb.guard_a_logits_receive_gradient, tmb.guard_e_formula, tmb.guard_e_beta_receive_gradient]

    def test_m44_pooling_falls_back_to_the_uniform_mean(self):
        from vkgr import retrieval_feat03 as f3
        detected(self, f3, 'weighted_mean', lambda x, w: x.sum(1) / (w > 0).sum(1, keepdim=True), self.A_E_POOL)

    def test_m45_recency_logits_start_at_n002(self):
        from vkgr import retrieval_feat03 as f3

        def noisy(model):
            g = torch.Generator().manual_seed(5)
            model.recency_logit_pos = torch.nn.Parameter(torch.randn(7, generator=g) * 0.02); model.recency_logit_gap = torch.nn.Parameter(torch.randn(6, generator=g) * 0.02)
        detected(self, f3, 'init_recency_logits', noisy, [tmb.guard_a_start_equals_anchor, tmb.guard_e_start_equals_a])

    def test_m46_recency_logits_detached(self):
        from vkgr import retrieval_feat03 as f3
        original = f3.recency_logits
        detected(self, f3, 'recency_logits', lambda *a: original(*a).detach(), [tmb.guard_a_logits_receive_gradient, tmb.guard_e_recency_logits_receive_gradient])

    def test_m47_unsupported_events_dropped_from_the_pooling(self):
        from vkgr import retrieval_feat04b as m4
        detected(self, m4.Feat04bIDMean, '_pool_mask', lambda self, histories, valid: valid & self.supported[histories], [tmb.guard_a_start_equals_anchor, tmb.guard_e_start_equals_a])

    def test_m82_pooled_weights_falls_back_to_a_single_where(self):
        from vkgr import retrieval_feat03 as f3

        def single_where(logits, valid):
            m = logits.masked_fill(~valid, float('-inf')).amax(1, keepdim=True).detach()
            return torch.where(valid, torch.exp(logits - m), torch.zeros_like(logits))
        detected(self, f3, 'pooled_weights', single_where, [tmb.guard_a_pooled_weights_pad_cannot_overflow, tmb.guard_e_pad_cannot_overflow_with_extreme_beta])

    def _beta_on_x(self, m4, original_logits):
        original = m4.event_vectors4

        def beta_in_x(model, histories, side, valid):
            x = original(model, histories, side, valid)
            return x + original_logits(model.behavior_beta, side[..., 0], valid).unsqueeze(-1) if hasattr(model, 'behavior_beta') else x
        return beta_in_x

    def test_m70a_beta_added_to_the_event_vector_instead_of_the_logit(self):
        """Two patches at once: the logit term returns exact 0 and x_k gets sum(bit * beta)."""
        from vkgr import retrieval_feat04b as m4
        original_logits = m4.behavior_logits; beta_in_x = self._beta_on_x(m4, original_logits)
        detected_multi(self, [(m4, 'behavior_logits', lambda beta, mask, valid: torch.zeros(mask.shape)), (m4, 'event_vectors4', beta_in_x)],
                       [tmb.guard_e_logit_path_only, tmb.guard_e_formula])

    def test_m70b_beta_in_both_the_event_vector_and_the_logit(self):
        from vkgr import retrieval_feat04b as m4
        beta_in_x = self._beta_on_x(m4, m4.behavior_logits)
        detected(self, m4, 'event_vectors4', beta_in_x, [tmb.guard_e_logit_path_only])

    def test_m71_bit_order_reversed(self):
        from vkgr import retrieval_feat04b as m4
        original = m4.behavior_logits
        detected(self, m4, 'behavior_logits', lambda beta, mask, valid: original(beta.flip(0), mask, valid), [tmb.guard_e_bit_order, tmb.guard_e_formula])

    def test_m72_mask_of_the_next_event(self):
        from vkgr import retrieval_feat04b as m4
        original = m4.behavior_logits
        detected(self, m4, 'behavior_logits', lambda beta, mask, valid: original(beta, mask.roll(-1, 1), valid), [tmb.guard_e_bit_order, tmb.guard_e_formula])

    def test_m73_beta_starts_at_n002(self):
        from vkgr import retrieval_feat04b as m4

        def noisy(model):
            model.behavior_beta = torch.nn.Parameter(torch.randn(6, generator=torch.Generator().manual_seed(4)) * 0.02)
        detected(self, m4, 'init_behavior_beta', noisy, [tmb.guard_e_start_equals_a, tmb.guard_e_param_count_19])

    def test_m74_beta_detached(self):
        from vkgr import retrieval_feat04b as m4
        original = m4.behavior_logits
        detected(self, m4, 'behavior_logits', lambda beta, mask, valid: original(beta.detach(), mask, valid), [tmb.guard_e_beta_receive_gradient])

    def test_m75_popcount_instead_of_per_bit_weights(self):
        from vkgr import retrieval_feat04b as m4

        def popcount(beta, mask, valid):
            bits = ((mask.unsqueeze(-1) >> torch.arange(6)) & 1).to(beta.dtype).sum(-1)
            return bits * beta.mean() * valid.to(beta.dtype)
        detected(self, m4, 'behavior_logits', popcount, [tmb.guard_e_bit_order, tmb.guard_e_formula])

    def test_m77_week_gap_anchor_replaced_by_the_target_week(self):
        original = trn.fr2()._tensors2

        def mutated(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                side[j, len(r['history']) - 1, 3] = min(r['target_position'][0], 24)
            return histories, side
        detected(self, trn.fr2(), '_tensors2', mutated, [named(trn.guard_target_flip_leaves_feature_unchanged, v) for v in (trn.A, trn.E)])

    def test_m78_target_week_or_ed_into_the_last_feedback_mask(self):
        original = trn.fr2()._tensors2

        def mutated(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                side[j, len(r['history']) - 1, 0] |= min(r['target_position'][0], 24) & 63
            return histories, side
        detected(self, trn.fr2(), '_tensors2', mutated, [named(trn.guard_target_flip_leaves_feature_unchanged, trn.E)])


def train_step_on_items(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
    """retrieval_feat04b_run.train_step with ONE change: the candidate column is read through ``model.items`` (M65: training without the statistics)."""
    from vkgr import retrieval_b1_run as b1
    from vkgr.retrieval_negatives import masked_ce_logq
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, rng, pool)
    histories, side = trn.r4()._tensors4b(batch, data['idmap'], max(len(r['history']) for r in batch), config['feature'])
    log_q = b1._log_q(batch, negatives, pool_size, config, frequency)
    device = next(model.parameters()).device; model.train(); optimizer.zero_grad(set_to_none=True)
    loss, skips = masked_ce_logq(model.users(histories.to(device), side.to(device)), model.items(columns.to(device)), labels.to(device), mask.to(device), config['temperature'], log_q)
    stats.record(epoch, len(batch), skips)
    result = {'drawn': len(batch), 'skipped': skips, 'negatives': len(negatives), 'negatives_sha256': b1._digest(negatives), 'loss': None, 'gradient_norm': None}
    if loss is None:
        return result
    loss.backward(); optimizer.step()
    return {**result, 'loss': float(loss.detach()), 'gradient_norm': 0.0}


def export_on_items(model, catalog, queries, idmap, batch_size, width, feature, check=None):
    """retrieval_feat04b_run.export_vectors_side4b with ONE change: the catalog vectors come from ``model.items`` (M66: evaluation without the statistics)."""
    from vkgr.retrieval_experiment import normalize_export
    model.eval(); items, users = [], []
    with torch.no_grad():
        for start in range(0, len(catalog), batch_size):
            items.append(model.items(torch.tensor([idmap[i] for i in catalog[start:start + batch_size]])).numpy())
        for start in range(0, len(queries), batch_size):
            histories, side = trn.r4()._tensors4b(queries[start:start + batch_size], idmap, width, feature)
            users.append(model.users(histories, side).numpy())
    import numpy as np
    raw = np.concatenate(items)
    return normalize_export(np.concatenate(users), check), normalize_export(raw, check), np.linalg.norm(raw.astype(np.float64), axis=1)


class CDMutationTests(unittest.TestCase):
    def test_m60_like_smoothing_off_or_p0_wrong(self):
        from vkgr import retrieval_feat04b as m4
        guards = [ts.guard_stats_bucket_boundaries, ts.guard_stats_constants_match_audit]
        detected(self, m4, 'LIKE_K', 0, guards)
        detected(self, m4, 'LIKE_P0', 50, guards)
        detected(self, m4, 'LIKE_P0', 0, guards)                                      # the pre-audit placeholder value must not survive
    def test_m61_bucket_edges_drift(self):
        from vkgr import retrieval_feat04b as m4
        guards = [ts.guard_stats_bucket_boundaries, ts.guard_stats_constants_match_audit]
        for name, value in (('EXPO_EDGES', m4.EXPO_EDGES[:-1] + (m4.EXPO_EDGES[-1] + 1,)), ('FIRST_EDGES', m4.FIRST_EDGES[:-1] + (m4.FIRST_EDGES[-1] - 1,)),
                            ('RATIO_EDGES', m4.RATIO_EDGES[:-1] + (m4.RATIO_EDGES[-1] + 10,)), ('LIKE_EDGES', m4.LIKE_EDGES[:-1] + (m4.LIKE_EDGES[-1] + 1,)),
                            ('EXPO_EDGES', (2,) + m4.EXPO_EDGES), ('LIKE_EDGES', (5,) + m4.LIKE_EDGES)):                      # a drifted edge and a merged-away edge that comes back
            detected(self, m4, name, value, guards)
    def test_m62_bucket_ids_indexed_by_raw_item_id_or_catalog_position(self):
        from vkgr import retrieval_feat04b as m4
        original = m4.stat_bucket_ids

        def reindexed(pick):
            def mutated(stats, catalog, idmap, size):
                ids = {name: torch.zeros(size, dtype=torch.long) for name in ('xb', 'fb', 'rb', 'lb')}
                reference = original(stats, catalog, idmap, size)
                for k, item in enumerate(catalog):
                    row = pick(k, item, idmap)
                    if 1 <= row < size:
                        for name in ids:
                            ids[name][row] = reference[name][idmap[item]]
                return ids
            return mutated
        detected(self, m4, 'stat_bucket_ids', reindexed(lambda k, item, idmap: item), [ts.guard_stats_alignment])
        detected(self, m4, 'stat_bucket_ids', reindexed(lambda k, item, idmap: k + 1), [ts.guard_stats_alignment])

    def test_m63_bucket_tables_added_to_the_history_path(self):
        from vkgr import retrieval_feat04b as m4
        terms = m4.candidate_stat_terms
        detected(self, m4.Feat04bIDMean, 'history_items', lambda self, indices: self.items(indices) + (terms(self, indices) if self.variant in m4.STAT_VARIANTS else 0.0),
                 [tmb.guard_cd_history_path_unchanged])

    def test_m64_supported_mask_lost(self):
        from vkgr import retrieval_feat04b as m4

        def no_mask(model, indices):
            (first, first_buffer, _), (second, second_buffer, _) = m4.STAT_LAYOUT[model.variant]
            return getattr(model, first)(getattr(model, first_buffer)[indices]) + getattr(model, second)(getattr(model, second_buffer)[indices])
        detected(self, m4, 'candidate_stat_terms', no_mask, [tmb.guard_cd_unsupported_zero, tmb.guard_cd_candidate_formula])

    def test_m65_train_step_candidate_column_through_items(self):
        detected(self, trn.r4(), 'train_step', train_step_on_items, [trn.guard_cd_train_step_uses_candidate_items, trn.guard_cd_train_step_tables_receive_gradient])

    def test_m66_evaluation_export_through_items(self):
        detected(self, trn.r4(), 'export_vectors_side4b', export_on_items, [trn.guard_cd_export_uses_candidate_items])

    def test_m67_bucket_table_output_detached(self):
        from vkgr import retrieval_feat04b as m4
        original = m4.candidate_stat_terms
        detected(self, m4, 'candidate_stat_terms', lambda model, indices: original(model, indices).detach(), [tmb.guard_cd_tables_receive_gradient, trn.guard_cd_train_step_tables_receive_gradient])

    def test_m68_bucket_tables_wiped_by_small_init_with_no_padding_idx(self):
        from vkgr import retrieval_feat04b as m4

        def wiped(model, variant):
            for table, _, edges in m4.STAT_LAYOUT[variant]:
                setattr(model, table, torch.nn.Embedding(len(getattr(m4, edges)) + 2, 64))                    # padding_idx=None
                with torch.no_grad():
                    getattr(model, table).weight.normal_(0.0, 0.02); getattr(model, table).weight[None].zero_()      # the F3D K6 trap: clears the whole table
        detected(self, m4, 'init_stat_tables', wiped, [tmb.guard_cd_init_and_draw_order])

    def test_m69_load_state_dict_no_longer_checks_the_bucket_ids(self):
        from vkgr import retrieval_feat02 as f2, retrieval_feat04b as m4
        detected(self, m4.Feat04bIDMean, 'load_state_dict', lambda self, state_dict, strict=True, assign=False: f2.Feat02IDMean.load_state_dict(self, state_dict, strict=strict, assign=assign),
                 [tmb.guard_cd_checkpoint_immutable_bucket_ids])

    def test_m76_target_item_placed_in_the_history(self):
        original = trn.fr2()._tensors2

        def mutated(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                histories[j, len(r['history']) - 1] = idmap[r['target']]
            return histories, side
        detected(self, trn.fr2(), '_tensors2', mutated, [named(trn.guard_target_flip_leaves_feature_unchanged, v) for v in (trn.A, trn.C, trn.D, trn.E)])

    def test_m79_diagnostics_called_from_the_candidate_path(self):
        from vkgr import retrieval_feat04b as m4

        def reads_the_target(self, indices, queries=()):
            bias = sum(q['target'] for q in queries)                                  # a validation target reaches the candidate vectors
            return self.items(indices) + 0.0 * bias

        def calls_a_diagnostic(self, indices):
            return self.items(indices) + target_bucket_hist(None, None, None, None, None)

        def scan_candidate_items():
            trn.guard_leak_l7_source_scan(inspect.getsource(m4.Feat04bIDMean.candidate_items))
        detected(self, m4.Feat04bIDMean, 'candidate_items', reads_the_target, [scan_candidate_items])
        detected(self, m4.Feat04bIDMean, 'candidate_items', calls_a_diagnostic, [scan_candidate_items])
        detected(self, m4.Feat04bIDMean, 'candidate_items', reads_the_target, [trn.guard_l7_source_has_no_target_access])

    def test_m80_binding_without_the_cutoff_week(self):
        original = trn.r4().binding_for4b
        detected(self, trn.r4(), 'binding_for4b', lambda *a, **k: {key: value for key, value in original(*a, **k).items() if key != 'stat_cutoff_week'}, [trn.guard_cd_binding_has_cutoff])

    def test_m84_one_hash_for_every_variant(self):
        import hashlib
        import json
        from vkgr import item_stats, retrieval_feat03 as f3, retrieval_feat04b as m4
        original = m4.bucket_spec_sha256

        def everything(variant):
            original(variant)                                                         # keeps the variant check
            spec = {'feat03': f3.bucket_spec3_sha256(), 'variant_family': 'all', 'expo': m4.EXPO_EDGES, 'first': m4.FIRST_EDGES, 'ratio': m4.RATIO_EDGES, 'like': m4.LIKE_EDGES,
                    'k': m4.LIKE_K, 'p0': m4.LIKE_P0, 'cap': item_stats.RATIO_CAP_PM, 'cutoff': item_stats.STAT_CUTOFF_WEEK, 'bits': list(m4.BEHAVIOR_BITS)}
            return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        detected(self, m4, 'bucket_spec_sha256', everything, [tmb.guard_s_spec_hash_per_variant])


class RunnerMutationTests(unittest.TestCase):
    """M41-M43, M81, M85: wiring of the runner (temperature, config registry, learning rate, default device) and the module-attribute scan convention."""
    def test_m41_train_step_temperature_hard_coded_to_0_15(self):
        real = trn.r4().train_step
        def hard_coded(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
            return real(model, optimizer, batch, data, stats, epoch, {**config, 'temperature': 0.15}, rng, pool, frequency)
        detected(self, trn.r4(), 'train_step', hard_coded, [trn.guard_s_temperature_plumbing, trn.guard_s_equiv_h1])
    def test_m42_real_path_accepts_t015_and_the_fixtures(self):
        real = trn.r4().validate_config
        detected(self, trn.r4(), 'validate_config', lambda value, synthetic=False: real(value, True), [trn.guard_s_config_registry, trn.guard_real_configs_match_anchor])
    def test_m43_default_device_cuda(self):
        detected(self, trn.r4(), 'DEFAULT_DEVICE', 'cuda', [trn.guard_s_cli_default_device_cpu])
    def test_m81_module_attribute_scan_convention_broken(self):
        tmb.guard_s_module_attr_scan('from . import retrieval_feat03 as f3\nvalue = f3.weighted_mean\n')           # the convention itself is sound
        tmb.guard_s_module_attribute_calls()                                                                         # and the real sources follow it
        for bad in ('from .retrieval_feat02 import small_init\n', 'from .retrieval_feat03 import weighted_mean, pooled_weights\n', 'x = 1\nfrom .retrieval_feat03 import recency_logits\n'):
            with self.assertRaises(AssertionError, msg=bad):
                tmb.guard_s_module_attr_scan(bad)
    def test_m85_real_path_accepts_learning_rate_0_0003(self):
        real = trn.r4().validate_config
        def lenient(value, synthetic=False):
            if not synthetic and value.get('learning_rate') == 0.0003:
                real({**value, 'learning_rate': 0.001}, synthetic)                       # the other checks stay, the learning-rate check is lost
                return value
            return real(value, synthetic)
        detected(self, trn.r4(), 'validate_config', lenient, [trn.guard_s_lr_fixed])
class CoverageTests(unittest.TestCase):
    def test_every_mutation_id_has_a_test(self):
        """Design 5.3: M41-M86, M70 as M70a/M70b, M48-M55 reserved (04a's B). A test method ``test_m<id>_...`` must exist for each id, and none for the reserved ones."""
        import re
        root = Path(__file__).resolve().parent
        found = set()
        for name in ('test_feat04b_mutations.py', 'test_feat04b_freeze.py'):
            found |= {m.group(1) + m.group(2) for m in re.finditer(r'def test_m(\d+)([ab]?)_', (root / name).read_text())}
        expected = {str(i) for i in range(41, 87) if not 48 <= i <= 55 and i != 70} | {'70a', '70b'}
        self.assertEqual(sorted(expected - found), [], msg='mutations without a test')
        self.assertEqual(sorted(found - expected), [], msg='tests for reserved or unknown mutation ids')
def target_bucket_hist(*args):
    """Stand-in for the offline diagnostic of the same name (the M79 patch only needs the NAME to appear in the candidate path source)."""
    return 0.0


if __name__ == '__main__':
    unittest.main()
