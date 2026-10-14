"""FEAT-03 runner tests (FEAT-005): runner level equivalence of f4_control (AC2), leakage L1-L5/L7/L8 for the 4 variants (AC6),
train + verify round trips, CLI loop on CPU, config registry and inputs rejection (AC9).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call them
and tests/test_feat03_mutations.py re-runs them under deliberately wrong implementations (M15, M25, M31, M37, CLI default device).
"""
import contextlib
import copy
import io
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat03_model as t3  # noqa: E402
from test_feat02_run import flip_non_history_fields  # noqa: E402
from test_feat01 import prepared_world  # noqa: E402

ensure = t3.ensure
CONTROL, Q1A, Q1B, Q2, Q3 = t3.CONTROL, t3.Q1A, t3.Q1B, t3.Q2, t3.Q3
VARIANTS4 = (Q1A, Q1B, Q2, Q3)
ALL5 = (CONTROL,) + VARIANTS4
WEEK_USERS = (CONTROL, Q1B, Q2, Q3)                       # the variants whose week column is filled by _tensors3
ROOT = Path(__file__).resolve().parents[1]
TIMING = ('training_seconds', 'evaluation_seconds')
CUDA_MESSAGE = 'ERROR P-FEAT03 gate failed: ValueError: actual CUDA environment required'


def r3():
    from vkgr import retrieval_feat03_run
    return retrieval_feat03_run


def fr2():
    from vkgr import retrieval_feat02_run
    return retrieval_feat02_run


def config3(**overrides):
    value = r3().default_config(); value.update(batch_size=16, max_epochs=8, min_epochs=2); value.update(overrides)
    return value


def real_config(**overrides):
    value = r3().default_config(); value.update(overrides)
    return value


def model3(data, feature, logits=False):
    """Runner-built model on CPU in eval mode; Q2 optionally gets fixed non-zero logits N(0, 0.5) so the weighted branch is exercised."""
    model = r3()._model3(data, config3(feature=feature), 'cpu'); model.eval()
    if feature == Q2 and logits:
        t3.set_logits(model, seed=5, scale=0.5)
    return model


def width_of(batch):
    return max(len(r['history']) for r in batch)


def crafted_q3_rows(rows):
    """Rows whose last two events share a week < 24 and whose distinct-week count a is ACT_EDGES[i] - 1: setting the last week to 24 makes a cross an edge."""
    from vkgr import retrieval_feat03 as m
    out = []
    for j, r in enumerate(rows):
        n = len(r['history'])
        if n < 2:
            continue
        e = m.ACT_EDGES[j % len(m.ACT_EDGES)]
        e = e if e <= n else m.ACT_EDGES[0]
        a = e - 1
        weeks = list(range(a - 1)) + [a - 1] * (n - (a - 1))
        out.append({**r, 'history_positions': [[w, k] for k, w in enumerate(weeks)], 'target_position': [a - 1, n]})
    return out


def craft_row(data, weeks, timespent=None, items=None):
    """One hand-made history row (its own fields only): item ids from the catalog, side [mask, timespent, duration], positions [week, ordinal]."""
    n = len(weeks); ids = items or data['catalog'][10:10 + n]
    spent = timespent or [20 + 7 * k for k in range(n)]
    return {'history': list(ids), 'side': [[1, spent[k], 120] for k in range(n)], 'history_positions': [[w, k] for k, w in enumerate(weeks)],
            'target': data['catalog'][0], 'target_position': [24, n], 'oov': False}


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_target_week_change_leaves_feature_unchanged(variant):
    """Changing the target week/position (and the other non-history fields) moves neither the batch tensors nor the user vectors."""
    data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
    model = model3(data, variant, logits=True)
    picks = (lambda r: r['history_positions'][-1][0], lambda r: 24, lambda r: 26, lambda r: 30)
    crafted = crafted_q3_rows(rows)[:30]
    ensure(len(crafted) >= 20, 'fixture: crafted rows missing')
    for batch in (rows[:24], queries[:24], crafted):
        width = width_of(batch)
        base = r3()._tensors3(batch, data['idmap'], width, variant); base_users = model.users(*base)
        for pick in picks:
            moved = []
            for r in batch:
                m = flip_non_history_fields(r); m['target_position'] = [pick(r), 5]
                moved.append(m)
            got = r3()._tensors3(moved, data['idmap'], width, variant)
            ensure(torch.equal(got[0], base[0]) and torch.equal(got[1], base[1]), f'{variant}: batch tensors changed with the target week/position')
            ensure(torch.equal(model.users(*got), base_users), f'{variant}: user vectors changed with the target week/position')


def guard_tensors3_week_column_wiring():
    """The week column is filled (with the real history weeks) for f4_control, f3_gap_only, f4_recency_pool, f3_user_activity and stays 0 for f3_pos_only."""
    data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
    batch = rows[:20] + queries[:20]; width = width_of(batch)
    ensure(any(p[0] > 0 for r in batch for p in r['history_positions']), 'fixture needs history weeks > 0')
    for feature in t3.FEATURES3_LITERAL:
        histories, side = r3()._tensors3(batch, data['idmap'], width, feature)
        ensure(tuple(side.shape) == (len(batch), width, 6), f'{feature}: wrong side shape')
        ensure(bool(side[..., 4:].eq(0).all()), f'{feature}: place/platform columns must stay 0')
        for j, r in enumerate(batch):
            n = len(r['history'])
            expected = [p[0] for p in r['history_positions']] if feature in WEEK_USERS else [0] * n
            ensure(side[j, :n, 3].tolist() == expected, f'{feature}: week column wrong in row {j}')
            ensure(side[j, :n, :3].tolist() == [list(s) for s in r['side']], f'{feature}: feedback/timespent/duration columns wrong in row {j}')
            ensure(bool(side[j, n:].eq(0).all()) and bool(histories[j, n:].eq(0).all()), f'{feature}: PAD slots must be 0')


def guard_cli_default_device_is_cpu():
    """Both synthetic commands default to cpu (a default of cuda would make every local use fail or silently differ from feat02)."""
    parser = r3().build_parser()
    for argv in (['synthetic-train', '--config', 'c.json', '--output', 'o'],
                 ['synthetic-verify', '--config', 'c.json', '--manifest', 'm.json', '--manifest-sha', 'a' * 64]):
        ensure(parser.parse_args(argv).device == 'cpu', f'{argv[0]}: --device must default to cpu')


# ---------------------------------------------------------------- helpers for CLI tests

def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = r3().main(argv)
    return code, out.getvalue(), err.getvalue()


def strip_timing(value):
    return {k: v for k, v in value.items() if k not in TIMING}


GOOD_SPEC = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
             'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
             'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}


class ConfigAndInputsTests(unittest.TestCase):
    def test_constants_and_registry(self):
        m = r3()
        self.assertEqual((m.TASK, m.SCHEMA, m.MANIFEST_SCHEMA, m.CHECKPOINT_SCHEMA), ('FEAT-03', 'p-feat03-run-v1', 'p-feat03-train-manifest-v1', 'p-feat03-checkpoint-v1'))
        self.assertIsInstance(m.NEEDS_WEEK, frozenset)
        self.assertEqual(m.NEEDS_WEEK, frozenset({CONTROL, Q1B, Q2, Q3}))
        self.assertEqual((m.default_config()['schema'], m.default_config()['task_id']), ('p-feat03-run-v1', 'FEAT-03'))
        self.assertIn(m.default_config()['feature'], t3.FEATURES3_LITERAL)

    def test_validate_config_accepts_the_registry_and_rejects_the_rest(self):
        m = r3()
        for feature in t3.FEATURES3_LITERAL:
            m.validate_config(config3(feature=feature), True)
            if feature != CONTROL:
                m.validate_config(real_config(feature=feature))
        for bad in ({'feature': 'f3'}, {'feature': 'f3_f4_recency'}, {'feature': 'none'}, {'init': 'legacy'}, {'architecture': 'sasrec'}, {'history_len': 20},
                    {'max_epochs': 30}, {'side_version': 'v0'}, {'schema': 'p-feat02-run-v1'}, {'task_id': 'FEAT-02'}, {'unknown': 1}):
            with self.assertRaises(ValueError, msg=bad):
                m.validate_config(real_config(**bad))
        with self.assertRaises(ValueError):
            m.validate_config({k: v for k, v in real_config().items() if k != 'feature'})

    def test_f4_control_is_rejected_for_real_runs(self):
        m = r3()
        with self.assertRaises(ValueError):
            m.validate_config(real_config(feature=CONTROL))
        with self.assertRaises(ValueError):
            m.validate_config(real_config(feature=CONTROL), False)
        m.validate_config(config3(feature=CONTROL), True)
        with self.assertRaises(ValueError):                                          # a real run session refuses it before touching data
            m.run_session(Path(tempfile.gettempdir()) / 'never-created-feat03', {'synthetic': False, 'rows': []}, real_config(feature=CONTROL))

    def test_inputs_pins_must_all_be_null(self):
        m = r3()
        for feature in (None,) + t3.FEATURES3_LITERAL:
            m.validate_inputs(GOOD_SPEC, feature)
        for bad in ({'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64}, {'attr_manifest': '/g/manifest.json', 'attr_sha256': 'e' * 64},
                    {'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64, 'attr_manifest': '/g/manifest.json', 'attr_sha256': 'e' * 64},
                    {'ctx_manifest': '/f/manifest.json'}, {'ctx_sha256': 'd' * 64}, {'attr_manifest': '/g/manifest.json'}, {'attr_sha256': 'e' * 64},
                    {'side_sha256': None}, {'schema': 'p-feat01-inputs-v1'}, {'items_metadata_sha256': 'x'}):
            for feature in (Q1A, Q2, Q3):
                with self.assertRaises(ValueError, msg=(bad, feature)):
                    m.validate_inputs({**GOOD_SPEC, **bad}, feature)
        with self.assertRaises(ValueError):
            m.validate_inputs({k: v for k, v in GOOD_SPEC.items() if k != 'attr_sha256'})

    def test_apply_feat03_is_apply_side(self):
        from vkgr import retrieval_feat01_run as f1
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            partitions, examples, queries, catalog, durations, h2, side = prepared_world(tmp)
            spec = {**GOOD_SPEC, 'history_manifest': str(tmp / 'h2' / 'manifest.json'), 'history_sha256': h2['manifest_sha256'],
                    'side_manifest': str(tmp / 'side' / 'manifest.json'), 'side_sha256': side['manifest_sha256']}
            r3().validate_inputs(spec)
            data = {'rows': examples, 'queries': queries, 'catalog': catalog, 'synthetic': False}
            for length in (50, 20):
                expected = f1.apply_side(data, {k: spec[k] for k in ('history_manifest', 'history_sha256', 'side_manifest', 'side_sha256')}, length)
                self.assertEqual(r3().apply_feat03(data, spec, length), expected)

    def test_committed_feat03_configs_are_registered(self):
        stems = {'p-feat03-q1a-pos-only-s42': Q1A, 'p-feat03-q1b-gap-only-s42': Q1B, 'p-feat03-q2-recency-pool-s42': Q2, 'p-feat03-q3-user-activity-s42': Q3}
        files = sorted((ROOT / 'configs' / 'feat03').glob('p-*.json'))
        self.assertEqual({f.stem for f in files}, set(stems))
        anchor = json.loads((ROOT / 'configs' / 'feat02' / 'p-feat02-f4-recency-s42.json').read_text())
        for path in files:
            value = json.loads(path.read_text())
            r3().validate_config(value)
            self.assertEqual(value['feature'], stems[path.stem], path.stem)
            self.assertEqual((value['schema'], value['task_id']), ('p-feat03-run-v1', 'FEAT-03'))
            self.assertEqual((value['seed'], value['history_len'], value['logq'], value['extra_uniform'], value['learning_rate'], value['weight_decay'], value['batch_size']),
                             (42, 50, True, 0, 0.001, 0.0001, 512), path.stem)
            self.assertEqual(path.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n', path.stem)
            self.assertEqual(set(value), set(anchor), path.stem)
            self.assertEqual({k for k in anchor if anchor[k] != value[k]}, {'schema', 'task_id', 'feature'}, path.stem)

    def test_docs_and_configs_carry_no_local_path(self):
        forbidden = ('/Us' + 'ers', '/work' + 'space', 'Hello' + 'Bike', '.code' + 'x')
        docs = (ROOT / 'docs' / 'feat03_switches.md').read_text()
        for needle in ('f4_control', 'f3_pos_only', 'f3_gap_only', 'f4_recency_pool', 'f3_user_activity',
                       'synthetic-train', 'synthetic-verify', '--device', 'train', 'verify'):
            self.assertIn(needle, docs, needle)
        for path in [ROOT / 'docs' / 'feat03_switches.md'] + sorted((ROOT / 'configs' / 'feat03').glob('*.json')):
            text = path.read_text()
            for word in forbidden:
                self.assertNotIn(word, text, (path.name, word))

    def test_k2_run_module_uses_module_attributes_only(self):
        text = (ROOT / 'src' / 'vkgr' / 'retrieval_feat03_run.py').read_text()
        self.assertNotIn('from .retrieval_feat02_run import', text)
        self.assertNotIn('from .retrieval_feat02 import', text)
        self.assertIn('from . import retrieval_feat02_run as fr', text)


class TensorTests(unittest.TestCase):
    def test_tensors3_week_column_wiring(self):
        guard_tensors3_week_column_wiring()

    def test_tensors3_reads_needs_week_at_call_time(self):
        m = r3(); data = fr2().synthetic_data(); rows, _ = fr2()._view(data, 50)
        batch = rows[:12]; width = width_of(batch)
        self.assertTrue(any(p[0] > 0 for r in batch for p in r['history_positions']))
        with mock.patch.object(m, 'NEEDS_WEEK', frozenset()):
            self.assertTrue(bool(m._tensors3(batch, data['idmap'], width, Q2)[1][..., 3].eq(0).all()))
        with mock.patch.object(m, 'NEEDS_WEEK', frozenset({Q1A})):
            self.assertFalse(bool(m._tensors3(batch, data['idmap'], width, Q1A)[1][..., 3].eq(0).all()))

    def test_tensors3_is_tensors2_with_the_f3_or_f4_feature(self):
        m = r3(); data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
        batch = rows[:16] + queries[:16]; width = width_of(batch)
        for feature in t3.FEATURES3_LITERAL:
            a = m._tensors3(batch, data['idmap'], width, feature)
            b = fr2()._tensors2(batch, data['idmap'], width, 'f3_f4_recency' if feature in WEEK_USERS else 'f3')
            self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), feature)

    def test_tensors3_fails_closed_on_misaligned_input(self):
        m = r3(); data = fr2().synthetic_data(); rows, _ = fr2()._view(data, 50)
        row = copy.deepcopy(rows[0]); width = len(row['history'])
        with self.assertRaises(ValueError):
            m._tensors3([{**row, 'side': row['side'][:-1]}], data['idmap'], width, Q1A)
        with self.assertRaises(ValueError):
            m._tensors3([{**row, 'history_positions': row['history_positions'][:-1]}], data['idmap'], width, Q1B)
        with self.assertRaises(ValueError):
            m._tensors3([{**row, 'history_positions': [[w + 0.5, o] for w, o in row['history_positions']]}], data['idmap'], width, Q2)

    def test_l5_source_guards(self):
        m = r3()
        source = inspect.getsource(m._tensors3)
        for banned in ('target', 'past_likes', 'oov'):
            self.assertNotIn(banned, source)
        self.assertIn('fr._tensors2', source)
        for name in ('retrieval_feat03.py', 'retrieval_feat03_run.py'):
            text = (ROOT / 'src' / 'vkgr' / name).read_text()
            for banned in ('target_position', 'past_likes', "['target']", "['oov']", '["target"]', '["oov"]'):
                self.assertNotIn(banned, text, (name, banned))
        self.assertNotIn('torch.median', (ROOT / 'src' / 'vkgr' / 'retrieval_feat03_run.py').read_text())


class LeakageTests(unittest.TestCase):
    """AC6: the 4 variants, training rows and validation queries, training path and evaluation path."""

    @classmethod
    def setUpClass(cls):
        cls.data = fr2().synthetic_data(); cls.rows, cls.queries = fr2()._view(cls.data, 50)

    def test_l1_target_week_and_non_history_fields_do_not_reach_tensors_or_users(self):
        for variant in VARIANTS4:
            with self.subTest(variant=variant):
                guard_target_week_change_leaves_feature_unchanged(variant)
                model = model3(self.data, variant, logits=True)
                for batch in (self.rows[:24], self.queries[:24]):
                    flipped = [flip_non_history_fields(r) for r in batch]
                    self.assertNotEqual([r['target'] for r in batch], [r['target'] for r in flipped])
                    a = r3()._tensors3(batch, self.data['idmap'], width_of(batch), variant)
                    b = r3()._tensors3(flipped, self.data['idmap'], width_of(batch), variant)
                    self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
                    self.assertTrue(torch.equal(model.users(*a), model.users(*b)))

    def test_l2_export_path_is_independent_of_non_history_fields_and_matches_the_training_path(self):
        for variant in VARIANTS4:
            with self.subTest(variant=variant):
                model = model3(self.data, variant, logits=True)
                flipped = [flip_non_history_fields(q) for q in self.queries]
                a = r3().export_vectors_side3(model, self.data['catalog'], self.queries, self.data['idmap'], 16, 50, variant)
                b = r3().export_vectors_side3(model, self.data['catalog'], flipped, self.data['idmap'], 16, 50, variant)
                self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]))
                batch = self.rows[:24]                                              # training width (longest row in the batch) vs evaluation width (50)
                narrow = r3()._tensors3(batch, self.data['idmap'], width_of(batch), variant)
                wide = r3()._tensors3(batch, self.data['idmap'], 50, variant)
                w = width_of(batch)
                self.assertTrue(torch.equal(wide[0][:, :w], narrow[0]) and torch.equal(wide[1][:, :w], narrow[1]))
                self.assertTrue(bool(wide[0][:, w:].eq(0).all()) and bool(wide[1][:, w:].eq(0).all()))
                self.assertTrue(torch.equal(model.users(*narrow), model.users(*wide)))

    def test_l3_reverse_controls_changing_a_history_event_changes_the_output(self):
        data = self.data
        cases = {
            Q1A: (craft_row(data, [0, 4, 8, 12, 16]),
                  [lambda r: {**r, 'side': [r['side'][0], r['side'][1], [1, 111, 120], *r['side'][3:]]},
                   lambda r: {**r, 'history': r['history'][1:], 'side': r['side'][1:], 'history_positions': r['history_positions'][1:]}]),
            Q1B: (craft_row(data, [0, 4, 8, 12, 16]),
                  [lambda r, slot=slot, new=new: {**r, 'history_positions': [[w if k != slot else new, o] for k, (w, o) in enumerate(r['history_positions'])]} for slot, new in ((1, 8), (3, 15), (4, 24))]),
            Q2: (craft_row(data, [0, 4, 8, 12, 16]),
                 [lambda r, slot=slot, new=new: {**r, 'history_positions': [[w if k != slot else new, o] for k, (w, o) in enumerate(r['history_positions'])]} for slot, new in ((1, 8), (3, 15), (4, 24))]),
            Q3: (craft_row(data, [0, 0, 1, 1]),
                 [lambda r: {**r, 'history_positions': [[w if k != 3 else 2, o] for k, (w, o) in enumerate(r['history_positions'])]},       # distinct weeks 2 -> 3 crosses an edge
                  lambda r: {**r, 'history': r['history'][1:], 'side': r['side'][1:], 'history_positions': r['history_positions'][1:]}]),     # n 4 -> 3 crosses an edge
        }
        for variant, (base_row, changes) in cases.items():
            with self.subTest(variant=variant):
                model = model3(data, variant, logits=True)
                width = len(base_row['history'])

                def users(row):
                    return model.users(*r3()._tensors3([row], data['idmap'], width, variant))
                base = users(base_row)
                self.assertGreaterEqual(len(changes), 2)
                for change in changes:
                    self.assertFalse(torch.equal(users(change(base_row)), base), variant)
                self.assertTrue(torch.equal(users(copy.deepcopy(base_row)), base))     # restored -> restored

    def test_l4_week_shift_invariance(self):
        shiftable = [r for r in self.rows + self.queries if r['history_positions'][-1][0] <= 23][:24]
        self.assertGreaterEqual(len(shiftable), 12)
        shifted = [{**r, 'history_positions': [[w + 1, o] for w, o in r['history_positions']]} for r in shiftable]
        for variant in (Q1B, Q2, Q3):
            with self.subTest(variant=variant):
                model = model3(self.data, variant, logits=True)
                a = r3()._tensors3(shiftable, self.data['idmap'], width_of(shiftable), variant)
                b = r3()._tensors3(shifted, self.data['idmap'], width_of(shiftable), variant)
                self.assertFalse(torch.equal(a[1], b[1]))                            # the week column did move
                self.assertTrue(torch.equal(model.users(*a), model.users(*b)))

    def test_l7_week_25_in_a_history_is_rejected(self):
        for variant in VARIANTS4:
            with self.subTest(variant=variant):
                model = model3(self.data, variant, logits=True)
                row = craft_row(self.data, [0, 4, 8, 12, 25]); width = 5
                histories, side = r3()._tensors3([row], self.data['idmap'], width, variant)
                if variant == Q1A:                                                    # the runner leaves Q1a's week column at 0; the model still validates a hand-made one
                    side = side.clone(); side[0, :, 3] = torch.tensor([0, 4, 8, 12, 25])
                with self.assertRaises(ValueError):
                    model.users(histories, side)

    def test_l8_q3_and_q2_ignore_pad_columns(self):
        t3.guard_q3_pad_width_invariance()
        t3.guard_q2_pad_width_invariance()
        for variant in (Q2, Q3):
            with self.subTest(variant=variant):
                model = model3(self.data, variant, logits=True)
                short = [r for r in self.rows + self.queries if len(r['history']) <= 30][:60]
                self.assertGreaterEqual(len(short), 20)
                a = r3().export_vectors_side3(model, self.data['catalog'], short, self.data['idmap'], 16, 50, variant)
                b = r3().export_vectors_side3(model, self.data['catalog'], short, self.data['idmap'], 16, 31, variant)
                if variant == Q3:
                    self.assertTrue(np.array_equal(a[0], b[0]))
                else:        # Q2: (x * w).sum(1) over 31 vs 50 slots may differ by one rounding step (measured 7.5e-9) when the width crosses a vector block; not a leak
                    self.assertTrue(np.allclose(a[0], b[0], atol=1e-6, rtol=0))


class RunnerTests(unittest.TestCase):
    def _run(self, tmp, **overrides):
        data = fr2().synthetic_data(); config = config3(**overrides)
        return data, config, r3().run_session(Path(tmp) / 'run', data, config)

    def test_f4_control_runner_equals_feat02_runner(self):
        """AC2 runner level: per-step loss, selection and the full evaluations equal those of the feat02 runner on F4 (state included)."""
        old, new = fr2(), r3()
        with tempfile.TemporaryDirectory() as tmp:
            data = old.synthetic_data()
            a = old.run_session(Path(tmp) / 'old', data, {**old.default_config(), 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2, 'feature': 'f3_f4_recency'})
            b = new.run_session(Path(tmp) / 'new', data, config3(feature=CONTROL))
            self.assertGreater(len(a['records']), 8)
            for key in ('loss', 'gradient_norm', 'drawn', 'skipped', 'indices_sha256', 'negatives_sha256', 'step', 'epoch'):
                self.assertEqual([r[key] for r in a['records']], [r[key] for r in b['records']], key)
            self.assertEqual(a['selection'], b['selection'])
            self.assertEqual([e['result'] for e in old.full_evaluations(Path(tmp) / 'old', a)], [e['result'] for e in new.full_evaluations(Path(tmp) / 'new', b)])
            self.assertEqual([e['diagnostics'] for e in a['evaluations']], [e['diagnostics'] for e in b['evaluations']])
            for name in ('best.pt', 'last.pt'):
                sa = torch.load(Path(tmp) / 'old' / name, map_location='cpu', weights_only=True)['model']
                sb = torch.load(Path(tmp) / 'new' / name, map_location='cpu', weights_only=True)['model']
                self.assertEqual(list(sa), list(sb))
                for key in sa:
                    self.assertTrue(torch.equal(sa[key], sb[key]), (name, key))

    def test_each_variant_trains_and_verifies(self):
        from vkgr import retrieval_feat03 as m
        for feature in ALL5:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                data, config, value = self._run(tmp, feature=feature)
                run = Path(tmp) / 'run'
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual((value['schema'], value['kind'], value['status']), ('p-feat03-train-manifest-v1', 'p_feat03_train', 'complete'))
                binding = value['binding']
                self.assertEqual((binding['task'], binding['feature'], binding['side_version']), ('P-FEAT03', feature, 'HIST_V2_SIDE_V1'))
                self.assertEqual(binding['bucket_spec_sha256'], m.bucket_spec3_sha256())
                self.assertEqual(binding['param_report'], m.parameter_report3(r3()._model3(data, config, 'cpu')))
                self.assertEqual(binding['param_report'], m.expected_param_counts3(feature))
                self.assertEqual(('weeks_sha256' in binding), feature in WEEK_USERS)
                self.assertNotIn('ctx_sha256', binding); self.assertNotIn('attr_sha256', binding)
                self.assertEqual(binding['execution']['device'], 'cpu')
                self.assertEqual(('user_activity_buckets' in value['data_diagnostics']), feature == Q3)
                self.assertEqual(r3().verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')
                with self.assertRaises(ValueError):
                    r3().verify_run(run / 'manifest.json', value['manifest_sha256'], data, {**config, 'feature': Q1A if feature != Q1A else Q1B})
                side = {**data, 'rows': [{**r, 'side': [[mask ^ 1, t, d] for mask, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaises(ValueError):
                    r3().verify_run(run / 'manifest.json', value['manifest_sha256'], side, config)
                weeks = {**data, 'rows': [{**r, 'history_positions': [[(w + 1) % 25 if k == len(r['history_positions']) - 1 else w, o] for k, (w, o) in enumerate(r['history_positions'])]}
                                          for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    r3().verify_run(run / 'manifest.json', value['manifest_sha256'], weeks, config)

    def test_verify_requires_cuda_for_real_data_and_config_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, value = self._run(tmp, feature=Q1A)
            run = Path(tmp) / 'run'
            with self.assertRaises(ValueError):
                r3().verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'synthetic': False}, config)
            with self.assertRaises(ValueError):
                r3().verify_run(run / 'manifest.json', '0' * 64, data, config)

    def test_q2_logit_diagnostics_are_recorded_per_epoch_and_recomputed_for_the_best_checkpoint(self):
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_ablation_run import strict_json
        with tempfile.TemporaryDirectory() as tmp:
            data, config, value = self._run(tmp, feature=Q2)
            run = Path(tmp) / 'run'
            for evaluation in value['evaluations']:
                logits = evaluation['diagnostics']['recency_logits']
                self.assertEqual((len(logits['position']), len(logits['weekgap'])), (7, 6))
                self.assertTrue(all(type(v) is float for v in logits['position'] + logits['weekgap']))
            for name, evaluation in (('best.pt', value['evaluations'][value['best']['epoch'] - 1]), ('last.pt', value['evaluations'][-1])):
                state = torch.load(run / name, map_location='cpu', weights_only=True)['model']
                self.assertEqual(evaluation['diagnostics']['recency_logits'], {'position': state['recency_logit_pos'].tolist(), 'weekgap': state['recency_logit_gap'].tolist()})
            last = value['evaluations'][-1]['diagnostics']['recency_logits']
            self.assertTrue(any(v != 0.0 for v in last['position'] + last['weekgap']), 'training must move the logits')
            for other in (Q1A, Q1B, Q3):
                self.assertNotIn('recency_logits', r3().evaluate_model(model3(data, other), data, fr2()._view(data, 50)[1][:20], config3(feature=other))[1])
            best = value['best']['epoch'] - 1
            forged = strict_json((run / 'manifest.json').read_text()); forged['evaluations'][best]['diagnostics']['recency_logits']['position'][0] += 0.25
            (run / 'forged.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, 'diagnostics'):
                r3().verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)

    def test_q3_bucket_and_scale_diagnostics(self):
        from vkgr import retrieval_feat03 as m
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_ablation_run import strict_json
        with tempfile.TemporaryDirectory() as tmp:
            data, config, value = self._run(tmp, feature=Q3)
            run = Path(tmp) / 'run'; rows, queries = fr2()._view(data, 50)
            buckets = value['data_diagnostics']['user_activity_buckets']
            self.assertEqual(set(buckets), {'train', 'validation'})
            for key, items in (('train', rows), ('validation', queries)):
                expected_len = [0] * (len(m.LEN_EDGES) + 1); expected_act = [0] * (len(m.ACT_EDGES) + 1)
                for r in items:
                    expected_len[sum(1 for e in m.LEN_EDGES if len(r['history']) >= e)] += 1
                    expected_act[sum(1 for e in m.ACT_EDGES if len({p[0] for p in r['history_positions']}) >= e)] += 1
                self.assertEqual(buckets[key], {'length': expected_len, 'activity': expected_act})
                self.assertEqual(sum(expected_len), len(items))
            self.assertEqual(r3().user_activity_buckets(rows, queries, data['idmap']), buckets)
            for evaluation in value['evaluations']:
                self.assertEqual(type(evaluation['diagnostics']['q3_scale_ratio']), float)
                self.assertGreater(evaluation['diagnostics']['q3_scale_ratio'], 0.0)
            model = r3()._model3(data, config, 'cpu')
            model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True)['model']); model.eval()
            ratios = []
            for start in range(0, len(queries), 16):
                ratios.append(m.q3_scale_ratio(model, *r3()._tensors3(queries[start:start + 16], data['idmap'], 50, Q3)).numpy())
            self.assertEqual(float(np.median(np.concatenate(ratios))), value['evaluations'][value['best']['epoch'] - 1]['diagnostics']['q3_scale_ratio'])
            for edit, pattern in ((lambda v: v['data_diagnostics']['user_activity_buckets']['train']['length'].__setitem__(0, 10 ** 6), 'user activity'),
                                  (lambda v: v['evaluations'][value['best']['epoch'] - 1]['diagnostics'].__setitem__('q3_scale_ratio', 123.0), 'diagnostics')):
                forged = strict_json((run / 'manifest.json').read_text()); edit(forged)
                (run / 'forged.json').write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError, pattern):
                    r3().verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)

    def test_nonfinite_gradient_or_parameter_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fr2().synthetic_data(); config = config3(feature=Q2)
            with mock.patch('torch.nn.utils.clip_grad_norm_', lambda *a, **k: torch.tensor(float('nan'))):
                with self.assertRaises(ValueError):
                    r3().run_session(Path(tmp) / 'run', data, config)


class CliTests(unittest.TestCase):
    def test_default_device_is_cpu(self):
        guard_cli_default_device_is_cpu()

    def test_device_choices_are_cpu_and_cuda_only(self):
        parser = r3().build_parser()
        for argv in (['synthetic-train', '--config', 'c', '--output', 'o', '--device', 'cuda'], ['synthetic-verify', '--config', 'c', '--manifest', 'm', '--manifest-sha', 'a' * 64, '--device', 'cpu']):
            parser.parse_args(argv)
        for bad in ('tpu', 'mps', 'cuda:0', ''):
            for argv in (['synthetic-train', '--config', 'c', '--output', 'o', '--device', bad], ['synthetic-verify', '--config', 'c', '--manifest', 'm', '--manifest-sha', 'a' * 64, '--device', bad]):
                with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                    parser.parse_args(argv)
                self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            r3().main(['synthetic-train', '--config', 'c', '--output', 'o', '--device', 'tpu'])
        self.assertEqual(cm.exception.code, 2)

    def test_cpu_round_trip_synthetic_train_then_synthetic_verify(self):
        for feature in VARIANTS4:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); config = config3(feature=feature)
                (tmp / 'config.json').write_text(json.dumps(config))
                code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run')])
                self.assertEqual((code, err), (0, ''))
                result = json.loads(out)
                self.assertEqual(result['status'], 'complete')
                code, out, err = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'),
                                          '--manifest-sha', result['manifest_sha256']])
                self.assertEqual((code, err), (0, ''))
                self.assertEqual(json.loads(out)['status'], 'verified')
                # --device cpu spelled out behaves like the default
                code, out, _ = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'),
                                        '--manifest-sha', result['manifest_sha256'], '--device', 'cpu'])
                self.assertEqual((code, json.loads(out)['status']), (0, 'verified'))
                # wrong pin -> exit 2, empty stdout
                code, out, err = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'), '--manifest-sha', '0' * 64])
                self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT03 gate failed', err)

    def test_default_synthetic_train_equals_a_direct_run_session_except_the_timing_fields(self):
        for feature in VARIANTS4:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); config = config3(feature=feature)
                (tmp / 'config.json').write_text(json.dumps(config))
                code, out, _ = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'cli')])
                self.assertEqual(code, 0)
                r3().run_session(tmp / 'direct', fr2().synthetic_data(), config)
                a = json.loads((tmp / 'cli' / 'manifest.json').read_text()); b = json.loads((tmp / 'direct' / 'manifest.json').read_text())
                self.assertEqual(a['binding']['execution']['device'], 'cpu')
                self.assertEqual(strip_timing(a), strip_timing(b))
                for key in ('binding', 'records', 'evaluations', 'selection', 'best', 'last'):
                    self.assertEqual(a[key], b[key], key)

    def test_cuda_without_cuda_exits_2_with_empty_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config3(feature=Q2)))
            with mock.patch('torch.cuda.is_available', return_value=False):
                code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run'), '--device', 'cuda'])
                self.assertEqual((code, out), (2, '')); self.assertEqual(err.strip(), CUDA_MESSAGE)
                self.assertFalse((tmp / 'run').exists())
                code, out, err = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'),
                                          '--manifest-sha', 'a' * 64, '--device', 'cuda'])
                self.assertEqual((code, out), (2, '')); self.assertEqual(err.strip(), CUDA_MESSAGE)

    def test_cuda_without_deterministic_workspace_exits_2_with_empty_stdout(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('torch.cuda.is_available', return_value=True), mock.patch.dict('os.environ'):
            import os
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config3(feature=Q3)))
            code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run'), '--device', 'cuda'])
            self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT03 gate failed: ValueError', err)

    def test_real_path_wiring_train_and_verify(self):
        from vkgr.data_audit import file_sha256
        m = r3()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'inputs.json').write_text(json.dumps(GOOD_SPEC))
            real = real_config(feature=Q2)
            (tmp / 'real.json').write_text(json.dumps(real))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            fake_data = {'rows': [], 'queries': [], 'catalog': []}
            with mock.patch('torch.cuda.is_available', return_value=True), \
                    mock.patch.object(m.b1, 'resource_guard', lambda *a, **k: (lambda: None)), \
                    mock.patch.object(m.b1, 'real_inputs', lambda s: (fake_data, None, {'prepared_sha': 'x'})), \
                    mock.patch.object(m, 'apply_feat03', lambda d, s, n: (d, {'history_manifest_sha256': 'h'})), \
                    mock.patch.object(m, 'run_session', lambda *a, **k: {'status': 'complete', 'manifest_sha256': 'm', 'selection': {'best_epoch': 3}}), \
                    mock.patch.object(m, 'verify_run', lambda *a, **k: {'status': 'verified'}):
                code, out, _ = run_cli(['train', *common, '--output', str(tmp / 'real-run')])
                self.assertEqual((code, json.loads(out)['best_epoch']), (0, 3))
                code, out, _ = run_cli(['verify', *common, '--manifest', str(tmp / 'real-run' / 'manifest.json'), '--manifest-sha', 'a' * 64])
                self.assertEqual((code, json.loads(out)['status']), (0, 'verified'))
                # f4_control never goes to a real run
                (tmp / 'control.json').write_text(json.dumps({**real, 'feature': CONTROL}))
                code, out, err = run_cli(['train', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'control.json'),
                                          '--config-sha', file_sha256(tmp / 'control.json'), '--output', str(tmp / 'x')])
                self.assertEqual((code, out), (2, '')); self.assertIn('P-FEAT03', err); self.assertFalse((tmp / 'x').exists())
                # non-null sidecar pins are refused
                (tmp / 'pinned.json').write_text(json.dumps({**GOOD_SPEC, 'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64}))
                code, out, err = run_cli(['train', '--inputs', str(tmp / 'pinned.json'), '--inputs-sha', file_sha256(tmp / 'pinned.json'), '--config', str(tmp / 'real.json'),
                                          '--config-sha', file_sha256(tmp / 'real.json'), '--output', str(tmp / 'y')])
                self.assertEqual((code, out), (2, '')); self.assertIn('P-FEAT03', err); self.assertFalse((tmp / 'y').exists())


if __name__ == '__main__':
    unittest.main()
