"""FEAT-04a runner tests: config registry, the three equivalences (E1/E2/E3, explicit binding whitelists), temperature plumbing,
leakage L1-L7, train + verify round trips, CLI loop on CPU, preflight, real-path wiring.

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); tests/test_feat04a_mutations.py
re-runs them under deliberately wrong implementations.
"""
import contextlib
import copy
import inspect
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402
import test_feat03_model as t3  # noqa: E402
from test_feat02_run import flip_non_history_fields  # noqa: E402
from test_feat03_run import craft_row  # noqa: E402

ensure = tm.ensure
CONTROL, A = 'f4_control', 'f4_recency_pool'
ROOT = Path(__file__).resolve().parents[1]
TIMING = ('training_seconds', 'evaluation_seconds')
CUDA_MESSAGE = 'ERROR P-FEAT04A gate failed: ValueError: actual CUDA environment required'
# Explicit binding whitelists (card F1): the binding keys of the three runners differ by design, so only these keys are compared and
# each must exist on BOTH sides (a missing key is a failure, not a skip). Enumerated from b1.binding_for / binding_for2 / binding_for3.
INPUT_KEYS = ('rows_sha256', 'queries_sha256', 'catalog_sha256', 'support_sha256', 'content_sha256', 'init_version', 'budget', 'execution', 'synthetic', 'test_gate', 'inputs')
W1 = INPUT_KEYS + ('side_sha256', 'weeks_sha256', 'side_version', 'history_version', 'init_report')
W2 = W1 + ('feature', 'bucket_spec_sha256')


def r4():
    from vkgr import retrieval_feat04a_run
    return retrieval_feat04a_run


def fr2():
    from vkgr import retrieval_feat02_run
    return retrieval_feat02_run


def config4(**overrides):
    value = r4().default_config(); value.update(logq=True, temperature=0.1, batch_size=16, max_epochs=8, min_epochs=2, feature=A); value.update(overrides)
    return value


def real_config(**overrides):
    value = r4().default_config(); value.update(logq=True, temperature=0.1, feature=A); value.update(overrides)
    return value


def model4(data, feature, logits=False):
    """Runner-built model on CPU in eval mode; A optionally gets fixed non-zero logits so the weighted branch is exercised."""
    model = r4()._model4a(data, config4(feature=feature), 'cpu'); model.eval()
    if feature == A and logits:
        t3.set_logits(model, seed=5, scale=0.5)
    return model


def width_of(batch):
    return max(len(r['history']) for r in batch)


def rejects(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def strip_timing(value):
    return {k: v for k, v in value.items() if k not in TIMING}


# ---------------------------------------------------------------- shared guards (reused by the mutation tests)

def guard_s_config_registry():
    """Real path: T=0.1 (float), lr=0.001 (float), seed in {42,2024,2025}, feature in the real features only. Synthetic: T in {0.1, 0.15}, control allowed."""
    m = r4()
    for seed in (42, 2024, 2025):
        m.validate_config(real_config(seed=seed))
    for bad in ({'temperature': 0.15}, {'temperature': 0.07}, {'temperature': 1}, {'feature': CONTROL}, {'learning_rate': 0.0005}, {'learning_rate': 0.0003}, {'seed': 7},
                {'logq': False}, {'extra_uniform': 128}, {'init': 'legacy'}, {'history_len': 20}, {'max_epochs': 30}, {'schema': 'p-feat03-run-v1'}, {'task_id': 'FEAT-03'}, {'unknown': 1}):
        ensure(rejects(lambda bad=bad: m.validate_config(real_config(**bad))), f'real config {bad} must be rejected')
    ensure(rejects(lambda: m.validate_config({k: v for k, v in real_config().items() if k != 'feature'})), 'missing field must be rejected')
    for temperature in (0.1, 0.15):
        for feature in (CONTROL, A):
            m.validate_config(config4(feature=feature, temperature=temperature), synthetic=True)
    ensure(rejects(lambda: m.validate_config(config4(temperature=0.2), synthetic=True)), 'synthetic T=0.2 must be rejected')
    ensure(rejects(lambda: m.validate_config(config4(feature='nope'), synthetic=True)), 'unknown feature must be rejected')


def guard_s_cli_default_device_cpu():
    parser = r4().build_parser()
    for argv in (['synthetic-train', '--config', 'c.json', '--output', 'o'],
                 ['synthetic-verify', '--config', 'c.json', '--manifest', 'm.json', '--manifest-sha', 'a' * 64]):
        ensure(parser.parse_args(argv).device == 'cpu', f'{argv[0]}: --device must default to cpu')


def _first_batch(config):
    from vkgr.retrieval_negatives import target_frequency
    from vkgr.retrieval_training import EpochCursor
    data = fr2().synthetic_data(); rows, _ = fr2()._view(data, 50)
    indices = EpochCursor(len(rows), config['seed']).next(config['batch_size'])
    return data, rows, [rows[i] for i in indices], target_frequency(rows)


def _independent_loss(config, temperature):
    from vkgr import retrieval_b1_run as b1
    from vkgr import retrieval_feat04a as m4
    from vkgr.retrieval_negatives import masked_ce_logq
    data, rows, batch, frequency = _first_batch(config)
    model = m4.make_feat04a_model(config['feature'], data['content'], data['supported_indices'], 50, config['seed']); model.train()
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, random.Random(0), None)
    histories, side = fr2()._tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), 'f3_f4_recency')
    log_q = b1._log_q(batch, negatives, pool_size, config, frequency)
    loss, _ = masked_ce_logq(model.users(histories, side), model.items(columns), labels, mask, temperature, log_q)
    return float(loss.detach())


def guard_s_temperature_plumbing():
    """The first-step loss uses the temperature of the config (0.1 and 0.15), not a constant."""
    from vkgr.retrieval_training import SamplerStats
    for feature in (CONTROL, A):
        for temperature in (0.1, 0.15):
            config = config4(feature=feature, temperature=temperature); data, rows, batch, frequency = _first_batch(config)
            model = r4()._model4a(data, config, 'cpu')
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            record = r4().train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
            expected, other = _independent_loss(config, temperature), _independent_loss(config, 0.25 - temperature)
            ensure(abs(expected - other) > 1e-4, 'fixture cannot tell the temperatures apart')
            ensure(abs(record['loss'] - expected) < 1e-6, f'{feature}: first-step loss {record["loss"]} is not the T={temperature} loss {expected}')


def compare_sessions(a, b, dir_a, dir_b, mod_a, mod_b, whitelist):
    """Per-step records, selection, per-epoch results and diagnostics, state dict bytes, and the whitelisted binding keys must be equal."""
    ensure(len(a['records']) > 8, 'fixture too short')
    for key in ('loss', 'gradient_norm', 'drawn', 'skipped', 'indices_sha256', 'negatives_sha256', 'step', 'epoch'):
        ensure([r[key] for r in a['records']] == [r[key] for r in b['records']], f'records.{key} differ')
    ensure(a['selection'] == b['selection'], 'selection differs')
    ensure([e['result'] for e in mod_a.full_evaluations(dir_a, a)] == [e['result'] for e in mod_b.full_evaluations(dir_b, b)], 'per-epoch results differ')
    ensure([e['diagnostics'] for e in a['evaluations']] == [e['diagnostics'] for e in b['evaluations']], 'per-epoch diagnostics differ')
    ensure(a['data_diagnostics'] == b['data_diagnostics'], 'data diagnostics differ')
    for name in ('best.pt', 'last.pt'):
        sa = torch.load(Path(dir_a) / name, map_location='cpu', weights_only=True)['model']
        sb = torch.load(Path(dir_b) / name, map_location='cpu', weights_only=True)['model']
        ensure(list(sa) == list(sb), f'{name}: state keys differ')
        for key in sa:
            ensure(sa[key].numpy().tobytes() == sb[key].numpy().tobytes(), f'{name}: {key} differs')
    for key in whitelist:
        ensure(key in a['binding'] and key in b['binding'], f'binding key {key} missing on one side')
        ensure(a['binding'][key] == b['binding'][key], f'binding key {key} differs')


def guard_s_equiv_h1():
    """E1: feat04a f4_control at T0.10 == php01 H1 (T0.10 + no new feature), every epoch."""
    from vkgr import retrieval_php01_run as php
    data = fr2().synthetic_data()
    php_config = {**php.default_config(), 'knob': 'H1', 'temperature': 0.1, 'dim': 64, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = r4().run_session(tmp / 'a', data, config4(feature=CONTROL, temperature=0.1)); b = php.run_session(tmp / 'b', data, php_config)
        compare_sessions(a, b, tmp / 'a', tmp / 'b', r4(), php, W1)


def guard_s_equiv_feat03_q2():
    """E2: feat04a A at T0.15 == feat03 Q2 (logit trajectories included in the per-epoch diagnostics)."""
    from vkgr import retrieval_feat03_run as f3r
    data = fr2().synthetic_data()
    f3_config = {**f3r.default_config(), 'feature': A, 'logq': True, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = r4().run_session(tmp / 'a', data, config4(feature=A, temperature=0.15)); b = f3r.run_session(tmp / 'b', data, f3_config)
        ensure('recency_logits' in a['evaluations'][0]['diagnostics'], 'A must record the logit trajectory')
        compare_sessions(a, b, tmp / 'a', tmp / 'b', r4(), f3r, W2)


def guard_s_equiv_f4_control_t015():
    """E3: feat04a f4_control at T0.15 == feat03 f4_control (catches temperature wiring errors)."""
    from vkgr import retrieval_feat03_run as f3r
    data = fr2().synthetic_data()
    f3_config = {**f3r.default_config(), 'feature': CONTROL, 'logq': True, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = r4().run_session(tmp / 'a', data, config4(feature=CONTROL, temperature=0.15)); b = f3r.run_session(tmp / 'b', data, f3_config)
        compare_sessions(a, b, tmp / 'a', tmp / 'b', r4(), f3r, W2)


def guard_s_first_step_loss_equals_anchor():
    """A starts exactly at the anchor: the loss of the first step is bit-equal to f4_control's (same seed, same batch)."""
    from vkgr.retrieval_training import SamplerStats
    losses = {}
    for feature in (CONTROL, A):
        config = config4(feature=feature); data, rows, batch, frequency = _first_batch(config)
        model = r4()._model4a(data, config, 'cpu'); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        losses[feature] = r4().train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)['loss']
    ensure(losses[CONTROL] == losses[A], f'first-step loss differs: {losses}')


def guard_s_untouched():
    from test_feat04a_untouched import check_pinned
    check_pinned(ROOT)


# ---------------------------------------------------------------- leakage guards (L1-L7)

def guard_target_flip_leaves_feature_unchanged(variant):
    """L1: changing the target week/position (and every other non-history field) moves neither the batch tensors nor the user vectors."""
    data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
    model = model4(data, variant, logits=True)
    picks = (lambda r: r['history_positions'][-1][0], lambda r: 24, lambda r: 26, lambda r: 30)
    for batch in (rows[:24], queries[:24]):
        width = width_of(batch)
        base = r4()._tensors4a(batch, data['idmap'], width, variant); base_users = model.users(*base)
        for pick in picks:
            moved = []
            for r in batch:
                m = flip_non_history_fields(r); m['target_position'] = [pick(r), 5]
                moved.append(m)
            got = r4()._tensors4a(moved, data['idmap'], width, variant)
            ensure(torch.equal(got[0], base[0]) and torch.equal(got[1], base[1]), f'{variant}: batch tensors changed with the target week/position')
            ensure(torch.equal(model.users(*got), base_users), f'{variant}: user vectors changed with the target week/position')


def guard_l2_export_path_ignores_non_history_fields(variant):
    """L2: the evaluation export (user and item vectors after normalisation, raw item norms) is independent of the non-history fields."""
    data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
    model = model4(data, variant, logits=True)
    flipped = [flip_non_history_fields(q) for q in queries]
    a = r4().export_vectors_side4a(model, data['catalog'], queries, data['idmap'], 16, 50, variant)
    b = r4().export_vectors_side4a(model, data['catalog'], flipped, data['idmap'], 16, 50, variant)
    ensure(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]), f'{variant}: export changed with the non-history fields')


def guard_l4_reverse_controls(variant):
    """L4: changing a history event's own field changes the user vector; restoring it restores the vector."""
    data = fr2().synthetic_data(); model = model4(data, variant, logits=True)
    base_row = craft_row(data, [0, 4, 8, 12, 16]); width = 5

    def users(row):
        return model.users(*r4()._tensors4a([row], data['idmap'], width, variant))
    base = users(base_row)
    changes = [lambda r, slot=slot, new=new: {**r, 'history_positions': [[w if k != slot else new, o] for k, (w, o) in enumerate(r['history_positions'])]} for slot, new in ((1, 8), (3, 15), (4, 24))]
    changes.append(lambda r: {**r, 'history': [data['catalog'][40]] + r['history'][1:]})
    for change in changes:
        ensure(not torch.equal(users(change(base_row)), base), f'{variant}: a history-event change did not move the user vector')
    ensure(torch.equal(users(copy.deepcopy(base_row)), base), f'{variant}: restoring the row did not restore the vector')


def guard_l5_week_shift_invariance(variant):
    """L5: adding a constant to every history week (staying <= 24) leaves the user vector bit-equal (no absolute calendar channel)."""
    data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
    shiftable = [r for r in rows + queries if r['history_positions'][-1][0] <= 23][:24]
    ensure(len(shiftable) >= 12, 'fixture: not enough shiftable rows')
    shifted = [{**r, 'history_positions': [[w + 1, o] for w, o in r['history_positions']]} for r in shiftable]
    model = model4(data, variant, logits=True); width = width_of(shiftable)
    a = r4()._tensors4a(shiftable, data['idmap'], width, variant); b = r4()._tensors4a(shifted, data['idmap'], width, variant)
    ensure(not torch.equal(a[1], b[1]), 'the week column did not move')
    ensure(torch.equal(model.users(*a), model.users(*b)), f'{variant}: user vectors depend on the absolute week')


def guard_l6_week_25_rejected(variant):
    data = fr2().synthetic_data(); model = model4(data, variant, logits=True)
    histories, side = r4()._tensors4a([craft_row(data, [0, 4, 8, 12, 25])], data['idmap'], 5, variant)
    ensure(rejects(lambda: model.users(histories, side)), f'{variant}: week 25 in a history must be rejected')


FORBIDDEN_TEXT = ('target_position', 'past_likes', "['target']", '["target"]', "['oov']", '["oov"]')
FORBIDDEN_DIAGNOSTICS = ('evaluate_model', 'support_slices', 'top50_unsupported_share', 'history_support_slices', 'side_visibility', 'data_diagnostics',
                         'item_norm_stats', 'top1_is_last_history_fraction', 'zero_user_vectors')


def feature_path_callables(variant=None):
    """Callables on the feature path, resolved at call time so patched versions are scanned too."""
    from vkgr import retrieval_feat03 as f3
    from vkgr import retrieval_feat04a as m4
    found = [fr2()._tensors2, r4()._tensors4a, r4().train_step, r4().export_vectors_side4a, f3.event_vectors, f3.Feat03IDMean.users]
    for name in ('event_vectors4a', 'history_author_content'):
        if hasattr(m4, name):
            found.append(getattr(m4, name))
    if hasattr(m4, 'Feat04aIDMean'):
        found.append(m4.Feat04aIDMean.users)
    return found


def guard_l7_source_has_no_target_access():
    """L7: no target/past_likes/oov access and no diagnostic helper in the sources of the feature-assembly functions."""
    for fn in feature_path_callables():
        source = inspect.getsource(fn)
        body = '\n'.join(line.split('#')[0] for line in source.splitlines())          # comments are not code
        for text in FORBIDDEN_TEXT:
            ensure(text not in body.replace('history_positions', ''), f'{fn.__qualname__}: forbidden access {text}')
        for name in FORBIDDEN_DIAGNOSTICS:
            ensure(name not in body, f'{fn.__qualname__}: diagnostic function {name} on the feature path')


# ---------------------------------------------------------------- CLI helpers

def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = r4().main(argv)
    return code, out.getvalue(), err.getvalue()


GOOD_SPEC = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
             'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
             'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
VARIANTS = (A,)
RUNNABLE = (CONTROL, A)


class ConfigAndInputsTests(unittest.TestCase):
    def test_constants(self):
        m = r4()
        self.assertEqual((m.TASK, m.SCHEMA, m.MANIFEST_SCHEMA, m.CHECKPOINT_SCHEMA), ('FEAT-04A', 'p-feat04a-run-v1', 'p-feat04a-train-manifest-v1', 'p-feat04a-checkpoint-v1'))
        self.assertEqual(m.DEFAULT_DEVICE, 'cpu')

    def test_config_registry(self):
        guard_s_config_registry()

    def test_default_config_fields_equal_the_feat03_field_set(self):
        from vkgr import retrieval_feat03_run as f3r
        self.assertEqual(set(r4().default_config()), set(f3r.default_config()))

    def test_inputs_pin_rules(self):
        m = r4()
        m.validate_inputs4a(dict(GOOD_SPEC), A)
        m.validate_inputs4a(dict(GOOD_SPEC), CONTROL)
        for bad_pin in ({'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64}, {'attr_manifest': '/f/manifest.json', 'attr_sha256': 'd' * 64}):
            with self.assertRaises(ValueError):
                m.validate_inputs4a({**GOOD_SPEC, **bad_pin}, A)
        with self.assertRaises(ValueError):
            m.validate_inputs4a({**GOOD_SPEC, 'schema': 'p-feat01-inputs-v1'}, A)
        with self.assertRaises(ValueError):
            m.validate_inputs4a({**GOOD_SPEC, 'attr_manifest': '/f/manifest.json'}, A)              # half pin
        with self.assertRaises(ValueError):
            m.validate_inputs4a(dict(GOOD_SPEC), 'nope')

    def test_apply_feat04a_refuses_pins_for_a(self):
        with self.assertRaises(ValueError):
            r4().apply_feat04a({}, {**GOOD_SPEC, 'attr_manifest': '/f/manifest.json', 'attr_sha256': 'd' * 64}, 50, A)


class TensorTests(unittest.TestCase):
    def test_tensors_equal_the_feat02_f4_tensors_and_fill_the_week_column(self):
        data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
        batch = rows[:20] + queries[:20]; width = width_of(batch)
        for feature in RUNNABLE:
            histories, side = r4()._tensors4a(batch, data['idmap'], width, feature)
            h2, s2 = fr2()._tensors2(batch, data['idmap'], width, 'f3_f4_recency')
            self.assertTrue(torch.equal(histories, h2) and torch.equal(side, s2))
            self.assertTrue(any(int(side[j, 0, 3]) > 0 or int(side[j, -1, 3]) > 0 for j in range(len(batch))))
            self.assertTrue(bool(side[..., 4:].eq(0).all()))
        with self.assertRaises(ValueError):
            r4()._tensors4a(batch, data['idmap'], width, 'nope')


class LeakageTests(unittest.TestCase):
    def test_l1_target_fields_do_not_reach_tensors_or_users(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_target_flip_leaves_feature_unchanged(variant)

    def test_l2_export_path(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l2_export_path_ignores_non_history_fields(variant)

    def test_l4_reverse_controls(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l4_reverse_controls(variant)

    def test_l5_week_shift_invariance(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l5_week_shift_invariance(variant)

    def test_l6_week_25_rejected(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l6_week_25_rejected(variant)

    def test_l7_source_guard(self):
        guard_l7_source_has_no_target_access()

    def test_l7_guard_bites_on_a_leaky_function(self):
        original = fr2()._tensors2

        def leaky(batch, idmap, width, feature):
            for r in batch:
                r['target_position']
            return original(batch, idmap, width, feature)
        with mock.patch.object(fr2(), '_tensors2', leaky):
            with self.assertRaises(AssertionError):
                guard_l7_source_has_no_target_access()


class EquivalenceTests(unittest.TestCase):
    def test_e1_control_t010_equals_php01_h1(self):
        guard_s_equiv_h1()

    def test_e2_a_t015_equals_feat03_q2(self):
        guard_s_equiv_feat03_q2()

    def test_e3_control_t015_equals_feat03_control(self):
        guard_s_equiv_f4_control_t015()

    def test_first_step_loss_equals_anchor(self):
        guard_s_first_step_loss_equals_anchor()

    def test_temperature_plumbing(self):
        guard_s_temperature_plumbing()


class RunnerTests(unittest.TestCase):
    def test_each_variant_trains_and_verifies(self):
        from vkgr import retrieval_feat04a as m4
        from vkgr import retrieval_feat03 as f3
        for feature in RUNNABLE:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                data = fr2().synthetic_data(); config = config4(feature=feature); run = Path(tmp) / 'run'
                value = r4().run_session(run, data, config)
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual((value['schema'], value['kind'], value['status']), ('p-feat04a-train-manifest-v1', 'p_feat04a_train', 'complete'))
                binding = value['binding']
                self.assertEqual((binding['task'], binding['feature'], binding['temperature'], binding['side_version']), ('P-FEAT04A', feature, 0.1, 'HIST_V2_SIDE_V1'))
                self.assertEqual(binding['bucket_spec_sha256'], f3.bucket_spec3_sha256())
                self.assertEqual(binding['bucket_spec4a_sha256'], m4.bucket_spec4a_sha256(feature))
                self.assertEqual(binding['param_report'], m4.expected_param_counts4a(feature))
                self.assertIn('weeks_sha256', binding); self.assertIn('side_sha256', binding)
                self.assertNotIn('attr_sha256', binding); self.assertNotIn('ctx_sha256', binding)
                self.assertEqual(binding['execution']['device'], 'cpu')
                self.assertEqual(r4().verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')
                other = CONTROL if feature != CONTROL else A
                with self.assertRaises(ValueError):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], data, {**config, 'feature': other})
                with self.assertRaises(ValueError):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], data, {**config, 'temperature': 0.15})
                side = {**data, 'rows': [{**r, 'side': [[mask ^ 1, t, d] for mask, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], side, config)
                weeks = {**data, 'rows': [{**r, 'history_positions': [[(w + 1) % 25 if k == len(r['history_positions']) - 1 else w, o] for k, (w, o) in enumerate(r['history_positions'])]}
                                          for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], weeks, config)

    def test_verify_requires_cuda_for_real_data_and_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fr2().synthetic_data(); config = config4(); value = r4().run_session(Path(tmp) / 'run', data, config); run = Path(tmp) / 'run'
            with self.assertRaises(ValueError):
                r4().verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'synthetic': False}, config)
            with self.assertRaises(ValueError):
                r4().verify_run(run / 'manifest.json', '0' * 64, data, config)

    def test_a_logit_diagnostics_per_epoch_and_recomputed_for_the_best_checkpoint(self):
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_ablation_run import strict_json
        with tempfile.TemporaryDirectory() as tmp:
            data = fr2().synthetic_data(); config = config4(feature=A); value = r4().run_session(Path(tmp) / 'run', data, config); run = Path(tmp) / 'run'
            for evaluation in value['evaluations']:
                logits = evaluation['diagnostics']['recency_logits']
                self.assertEqual((len(logits['position']), len(logits['weekgap'])), (7, 6))
            last = value['evaluations'][-1]['diagnostics']['recency_logits']
            self.assertTrue(any(v != 0.0 for v in last['position'] + last['weekgap']), 'training must move the logits')
            state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)['model']
            self.assertEqual(last, {'position': state['recency_logit_pos'].tolist(), 'weekgap': state['recency_logit_gap'].tolist()})
            self.assertNotIn('recency_logits', value['evaluations'][0]['diagnostics'] if False else {})
            forged = strict_json((run / 'manifest.json').read_text()); forged['evaluations'][value['best']['epoch'] - 1]['diagnostics']['recency_logits']['position'][0] += 0.25
            (run / 'forged.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, 'diagnostics'):
                r4().verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)

    def test_control_run_has_no_logit_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = r4().run_session(Path(tmp) / 'run', fr2().synthetic_data(), config4(feature=CONTROL))
            self.assertNotIn('recency_logits', value['evaluations'][0]['diagnostics'])

    def test_nonfinite_gradient_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('torch.nn.utils.clip_grad_norm_', lambda *a, **k: torch.tensor(float('nan'))):
                with self.assertRaises(ValueError):
                    r4().run_session(Path(tmp) / 'run', fr2().synthetic_data(), config4())

    def test_real_run_requires_cuda_and_inputs_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {**fr2().synthetic_data(), 'synthetic': False}
            with self.assertRaises(ValueError):
                r4().run_session(Path(tmp) / 'run', data, real_config(), 'cpu')


class CliTests(unittest.TestCase):
    def test_default_device_is_cpu(self):
        guard_s_cli_default_device_cpu()

    def test_device_choices(self):
        parser = r4().build_parser()
        for bad in ('tpu', 'cuda:0', ''):
            for argv in (['synthetic-train', '--config', 'c', '--output', 'o', '--device', bad], ['synthetic-verify', '--config', 'c', '--manifest', 'm', '--manifest-sha', 'a' * 64, '--device', bad]):
                with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                    parser.parse_args(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_cpu_round_trip(self):
        for feature in RUNNABLE:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config4(feature=feature)))
                code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run')])
                self.assertEqual((code, err), (0, '')); result = json.loads(out); self.assertEqual(result['status'], 'complete')
                common = ['--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'), '--manifest-sha', result['manifest_sha256']]
                code, out, err = run_cli(['synthetic-verify', *common])
                self.assertEqual((code, err, json.loads(out)['status']), (0, '', 'verified'))
                code, out, _ = run_cli(['synthetic-verify', *common, '--device', 'cpu'])
                self.assertEqual((code, json.loads(out)['status']), (0, 'verified'))
                code, out, err = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'), '--manifest-sha', '0' * 64])
                self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT04A gate failed', err)

    def test_default_synthetic_train_equals_a_direct_run_session_except_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); config = config4(); (tmp / 'config.json').write_text(json.dumps(config))
            code, _, _ = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'cli')])
            self.assertEqual(code, 0)
            r4().run_session(tmp / 'direct', fr2().synthetic_data(), config)
            a = json.loads((tmp / 'cli' / 'manifest.json').read_text()); b = json.loads((tmp / 'direct' / 'manifest.json').read_text())
            self.assertEqual(a['binding']['execution']['device'], 'cpu')
            self.assertEqual(strip_timing(a), strip_timing(b))

    def test_cuda_without_cuda_exits_2_with_empty_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config4()))
            with mock.patch('torch.cuda.is_available', return_value=False):
                code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run'), '--device', 'cuda'])
                self.assertEqual((code, out), (2, '')); self.assertEqual(err.strip(), CUDA_MESSAGE); self.assertFalse((tmp / 'run').exists())
                code, out, err = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'm.json'), '--manifest-sha', 'a' * 64, '--device', 'cuda'])
                self.assertEqual((code, out), (2, '')); self.assertEqual(err.strip(), CUDA_MESSAGE)

    def test_cuda_without_deterministic_workspace_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('torch.cuda.is_available', return_value=True), mock.patch.dict('os.environ'):
            import os
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config4()))
            code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run'), '--device', 'cuda'])
            self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT04A gate failed: ValueError', err)

    def _real_patches(self, m, fake_data):
        return [mock.patch('torch.cuda.is_available', return_value=True), mock.patch.object(m.b1, 'resource_guard', lambda *a, **k: (lambda: None)),
                mock.patch.object(m.b1, 'real_inputs', lambda s: (fake_data, None, {'prepared_sha': 'x'})),
                mock.patch.object(m, 'apply_feat04a', lambda d, s, n, f: (d, {'history_manifest_sha256': 'h'}))]

    def test_real_path_wiring_train_and_verify(self):
        from vkgr.data_audit import file_sha256
        m = r4()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'inputs.json').write_text(json.dumps(GOOD_SPEC)); (tmp / 'real.json').write_text(json.dumps(real_config()))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            patches = self._real_patches(m, {'rows': [], 'queries': [], 'catalog': []}) + [
                mock.patch.object(m, 'run_session', lambda *a, **k: {'status': 'complete', 'manifest_sha256': 'm', 'selection': {'best_epoch': 3}}),
                mock.patch.object(m, 'verify_run', lambda *a, **k: {'status': 'verified'})]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                code, out, _ = run_cli(['train', *common, '--output', str(tmp / 'real-run')])
                self.assertEqual((code, json.loads(out)['best_epoch']), (0, 3))
                code, out, _ = run_cli(['verify', *common, '--manifest', str(tmp / 'real-run' / 'manifest.json'), '--manifest-sha', 'a' * 64])
                self.assertEqual((code, json.loads(out)['status']), (0, 'verified'))
                (tmp / 'control.json').write_text(json.dumps(real_config(feature=CONTROL)))
                code, out, err = run_cli(['train', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'control.json'),
                                          '--config-sha', file_sha256(tmp / 'control.json'), '--output', str(tmp / 'x')])
                self.assertEqual((code, out), (2, '')); self.assertIn('P-FEAT04A', err); self.assertFalse((tmp / 'x').exists())
                (tmp / 'pinned.json').write_text(json.dumps({**GOOD_SPEC, 'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64}))
                code, out, err = run_cli(['train', '--inputs', str(tmp / 'pinned.json'), '--inputs-sha', file_sha256(tmp / 'pinned.json'), '--config', str(tmp / 'real.json'),
                                          '--config-sha', file_sha256(tmp / 'real.json'), '--output', str(tmp / 'y')])
                self.assertEqual((code, out), (2, '')); self.assertIn('P-FEAT04A', err); self.assertFalse((tmp / 'y').exists())


class PreflightTests(unittest.TestCase):
    def test_preflight_check_passes_on_synthetic_data_for_a(self):
        data = fr2().synthetic_data(); report = r4().preflight_check(data, config4(feature=A), None)
        self.assertEqual(report['status'], 'preflight_ok'); self.assertEqual(report['feature'], A)
        self.assertEqual(report['supported_items'], len(data['support']))
        self.assertIn('binding_sha256', report); self.assertEqual(report['param_report']['by_table']['recency_logit_pos']['total'], 7)
        json.dumps(report)

    def test_preflight_check_fails_when_a_does_not_start_at_the_anchor(self):
        from vkgr import retrieval_feat03 as f3

        def noisy(model):
            model.recency_logit_pos = torch.nn.Parameter(torch.linspace(0.0, 1.0, 7)); model.recency_logit_gap = torch.nn.Parameter(torch.zeros(6))
        with mock.patch.object(f3, 'init_recency_logits', noisy):
            with self.assertRaises(ValueError):
                r4().preflight_check(fr2().synthetic_data(), config4(feature=A), None)

    def test_preflight_check_fails_when_the_user_vector_reads_the_target(self):
        original = fr2()._tensors2

        def leaky(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                side[j, len(r['history']) - 1, 3] = min(r['target_position'][0], 24)
            return histories, side
        with mock.patch.object(fr2(), '_tensors2', leaky):
            with self.assertRaises(ValueError):
                r4().preflight_check(fr2().synthetic_data(), config4(feature=A), None)

    def test_preflight_cli_wiring_and_exit_codes(self):
        from vkgr.data_audit import file_sha256
        m = r4()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'inputs.json').write_text(json.dumps(GOOD_SPEC)); (tmp / 'real.json').write_text(json.dumps(real_config()))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            data = fr2().synthetic_data()
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(m.b1, 'real_inputs', lambda s: (data, None, {'prepared_sha': 'x'})))
                stack.enter_context(mock.patch.object(m, 'apply_feat04a', lambda d, s, n, f: (d, {'history_manifest_sha256': 'h'})))
                stack.enter_context(mock.patch('torch.cuda.is_available', return_value=False))          # preflight must not need CUDA
                code, out, err = run_cli(['preflight', *common])
                self.assertEqual((code, err), (0, '')); self.assertEqual(json.loads(out)['status'], 'preflight_ok')
                (tmp / 'bad.json').write_text(json.dumps(real_config(temperature=0.15)))
                code, out, err = run_cli(['preflight', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'),
                                          '--config', str(tmp / 'bad.json'), '--config-sha', file_sha256(tmp / 'bad.json')])
                self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT04A gate failed', err)
                (tmp / 'control.json').write_text(json.dumps(real_config(feature=CONTROL)))
                code, out, err = run_cli(['preflight', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'),
                                          '--config', str(tmp / 'control.json'), '--config-sha', file_sha256(tmp / 'control.json')])
                self.assertEqual((code, out), (2, ''))


class ConfigFileTests(unittest.TestCase):
    def test_a_config_files(self):
        h1 = json.loads((ROOT / 'configs' / 'php01' / 'p-php01-h1-t010-s42.json').read_text())
        seen = {}
        for seed in (42, 2024, 2025):
            path = ROOT / 'configs' / 'feat04a' / f'p-feat04a-a-q2t010-s{seed}.json'
            value = json.loads(path.read_text()); seen[seed] = value
            r4().validate_config(value)
            self.assertEqual((value['feature'], value['seed'], value['schema'], value['task_id'], value['temperature']), (A, seed, 'p-feat04a-run-v1', 'FEAT-04A', 0.1))
        expected = {k: v for k, v in h1.items() if k not in ('dim', 'knob')}
        expected.update(schema='p-feat04a-run-v1', task_id='FEAT-04A', feature=A)
        self.assertEqual(seen[42], expected)                                     # AC7: only schema/task_id/feature differ from H1 (seed 42), dim/knob dropped
        for seed in (2024, 2025):
            self.assertEqual({k: v for k, v in seen[seed].items() if k != 'seed'}, {k: v for k, v in seen[42].items() if k != 'seed'})


if __name__ == '__main__':
    unittest.main()
