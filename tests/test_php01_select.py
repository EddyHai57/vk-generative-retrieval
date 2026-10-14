"""P-HP-01 H7 trigger + selection tests (pre-registered rule; synthetic manifests only, never real metrics).

Threshold: delta R@50 >= 0.002 on n=2520 validation queries == at least 6 more hits (5 hits = 0.001984 is not enough).
``guard_*`` functions are plain assertion functions; tests/test_php01_mutations.py re-runs them under wrong implementations.
"""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import ensure  # noqa: E402
from test_php01_run import F4_CONFIG, KNOB_VALUES, ROOT  # noqa: E402

N = 2520
ANCHOR_HITS = 424
KNOBS = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6')
RULE_FILE = ROOT / 'configs' / 'php01' / 'p-php01-h7-combo-rule.json'


def res(name, delta_hits=0, n50=0.05, verified=True):
    hits = ANCHOR_HITS + delta_hits
    return {'name': name, 'r50': hits / N, 'n50': n50, 'hits': hits, 'best_epoch': 10, 'stop_epoch': 13, 'verified': verified, 'manifest_sha256': name[:1].lower() * 64}


def anchor():
    return res('anchor')


def runs(**deltas):
    """H1..H6 results; ``deltas`` maps knob -> delta hits (or (delta hits, n50))."""
    out = {}
    for k in KNOBS:
        d = deltas.get(k, -3)
        out[k] = res(k, *d) if isinstance(d, tuple) else res(k, d)
    return out


# ---------------------------------------------------------------- guards

def guard_five_hits_is_not_a_winner():
    from vkgr.retrieval_php01_select import decide
    five = decide(anchor(), runs(H1=5), {})
    ensure(five['runs']['H1']['winner'] is False, '5 more hits (0.001984) must not pass the 0.002 threshold')
    six = decide(anchor(), runs(H1=6), {})
    ensure(six['runs']['H1']['winner'] is True, '6 more hits (0.002381) must pass the threshold')
    ensure(decide(anchor(), runs(H1=(5, 0.9)), {})['runs']['H1']['winner'] is False, 'a higher N@50 cannot make up for R@50')
    ensure(decide(anchor(), runs(H1=0), {})['runs']['H1']['winner'] is False and decide(anchor(), runs(H1=-6), {})['runs']['H1']['winner'] is False, 'zero/negative delta')


def guard_same_knob_twice_does_not_trigger():
    from vkgr.retrieval_php01_select import decide
    try:
        decision = decide(anchor(), runs(H1=10, H2=8), {})
    except ValueError as exc:                               # e.g. a one-knob "combo" rejected downstream by validate_config
        raise AssertionError(f'two levels of one knob must give a void H7, not an error: {exc}') from exc
    ensure(decision['winner_groups'] == ['temperature'], 'H1 and H2 are the same knob')
    ensure(decision['triggered'] is False and decision['h7'] == 'void', 'two levels of one knob must not trigger H7')
    ensure(decide(anchor(), runs(H1=10, H5=7), {})['triggered'] is True, 'control: two different knobs trigger H7')


def guard_tie_goes_to_earlier_enumeration():
    from vkgr.retrieval_php01_select import decide
    tie = decide(anchor(), runs(H1=0, H2=0), {})          # same R@50 and N@50 as the anchor
    ensure(tie['selection']['winner'] == 'anchor', 'exact tie with the anchor must go to the anchor (enumeration order 0)')
    ensure([e['name'] for e in tie['selection']['ranking']][:3] == ['anchor', 'H1', 'H2'], 'tie order must follow the enumeration order')
    pair = decide(anchor(), runs(H3=8, H4=8, H1=7), {})
    ensure(pair['h7']['values']['dim'] == 32, 'a tie inside a knob group goes to the earlier level (H3)')


def guard_h7_takes_each_knobs_best_level():
    from vkgr.retrieval_php01_select import decide
    decision = decide(anchor(), runs(H3=7, H4=12, H6=9, H5=6), {})
    ensure(decision['h7']['values'] == {'temperature': 0.15, 'dim': 128, 'learning_rate': 0.0005}, 'H7 must use the best winning level of every group')
    by_n50 = decide(anchor(), runs(H1=(8, 0.04), H2=(8, 0.06), H5=6), {})
    ensure(by_n50['h7']['values']['temperature'] == 0.25, 'equal R@50 inside a group: higher N@50 wins')


# ---------------------------------------------------------------- tests

class DecideTests(unittest.TestCase):
    def test_threshold_in_hits(self):
        guard_five_hits_is_not_a_winner()
        from vkgr import retrieval_php01_select as s
        self.assertEqual((s.THRESHOLD, s.N_QUERIES, s.MIN_DELTA_HITS), (0.002, 2520, 6))
        self.assertLess(5 / N, 0.002); self.assertGreaterEqual(6 / N, 0.002)

    def test_one_group_or_same_knob_is_void(self):
        from vkgr.retrieval_php01_select import decide
        guard_same_knob_twice_does_not_trigger()
        for deltas in ({'H3': 9}, {}, {'H5': 6, 'H6': 20}):
            decision = decide(anchor(), runs(**deltas), {})
            self.assertFalse(decision['triggered']); self.assertEqual(decision['h7'], 'void')
            self.assertEqual(decision['selection']['stage'], 'final')

    def test_trigger_builds_combo(self):
        from vkgr.retrieval_php01_run import validate_config
        from vkgr.retrieval_php01_select import decide
        decision = decide(anchor(), runs(H1=7, H5=6), {})
        self.assertTrue(decision['triggered']); self.assertEqual(decision['winner_groups'], ['temperature', 'learning_rate'])
        self.assertEqual(decision['h7']['values'], {'temperature': 0.1, 'dim': 64, 'learning_rate': 0.002})
        validate_config(decision['h7']['config']); self.assertEqual(decision['h7']['config']['knob'], 'H7')
        self.assertEqual(decision['selection']['stage'], 'interim')
        three = decide(anchor(), runs(H2=6, H3=6, H6=6), {})
        self.assertEqual(three['h7']['values'], {'temperature': 0.25, 'dim': 32, 'learning_rate': 0.0005})
        guard_h7_takes_each_knobs_best_level()

    def test_run_rows_are_reported(self):
        from vkgr.retrieval_php01_select import decide
        decision = decide(anchor(), runs(H1=7), {})
        row = decision['runs']['H1']
        self.assertEqual((row['status'], row['delta_hits'], row['hits'], row['winner'], row['verified']), ('complete', 7, ANCHOR_HITS + 7, True, True))
        self.assertAlmostEqual(row['delta_r50'], 7 / N)
        self.assertEqual(decision['anchor']['hits'], ANCHOR_HITS)
        self.assertEqual(decision['rule']['min_delta_hits'], 6)

    def test_failed_and_unverified_runs(self):
        from vkgr.retrieval_php01_select import decide
        results = runs(H1=9, H5=9); del results['H5']
        decision = decide(anchor(), results, {'H5': 'cuda oom at epoch 3'})
        self.assertEqual(decision['runs']['H5'], {'status': 'failed', 'reason': 'cuda oom at epoch 3', 'winner': False})
        self.assertFalse(decision['triggered'])
        self.assertNotIn('H5', [e['name'] for e in decision['selection']['ranking']])
        unverified = runs(H1=9, H5=9); unverified['H5'] = res('H5', 9, verified=False)
        decision = decide(anchor(), unverified, {})
        self.assertFalse(decision['runs']['H5']['winner']); self.assertFalse(decision['triggered'])
        self.assertNotIn('H5', [e['name'] for e in decision['selection']['ranking']])
        missing = runs(); del missing['H4']
        with self.assertRaises(ValueError):
            decide(anchor(), missing, {})
        with self.assertRaises(ValueError):
            decide(anchor(), runs(), {'H4': 'x'})              # both a result and a failure
        with self.assertRaises(ValueError):
            decide(anchor(), {**runs(), 'H7': res('H7')}, {})

    def test_selection(self):
        from vkgr.retrieval_php01_select import decide, select
        guard_tie_goes_to_earlier_enumeration()
        self.assertEqual(decide(anchor(), runs(), {})['selection']['winner'], 'anchor')
        self.assertEqual(decide(anchor(), runs(H4=3), {})['selection']['winner'], 'H4')       # below the threshold, still the best measured
        decision = decide(anchor(), runs(H1=7, H5=6), {})
        self.assertEqual(decision['selection']['winner'], 'H1')
        final = select(decision, res('H7', 15))
        self.assertEqual((final['stage'], final['winner']), ('final', 'H7'))
        self.assertEqual(select(decision, {'failed': 'disk'})['winner'], 'H1')
        with self.assertRaises(ValueError):
            select(decision, None)
        void = decide(anchor(), runs(H1=7), {})
        self.assertEqual(select(void, None), void['selection'])
        with self.assertRaises(ValueError):
            select(void, res('H7', 15))


class RuleFileTests(unittest.TestCase):
    def test_rule_file_matches_code_constants(self):
        from vkgr import retrieval_php01_select as s
        self.assertEqual(s.load_rule(), s.rule_value())
        value = json.loads(RULE_FILE.read_text())
        self.assertEqual(RULE_FILE.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n')
        self.assertEqual(value['anchor_config_sha256'], hashlib.sha256(F4_CONFIG.read_bytes()).hexdigest())
        self.assertEqual(value['anchor_config_sha256'], 'bead30cb6d635e7b2a204f0006f67ef1c4dd9c61d70193bedf2ec2061decc7ea')
        self.assertEqual(value['knobs'], {k: {'field': f, 'value': v[('temperature', 'dim', 'learning_rate').index(f)]}
                                          for k, v, f in ((k, KNOB_VALUES[k], f) for k, f in zip(KNOBS, ['temperature'] * 2 + ['dim'] * 2 + ['learning_rate'] * 2))})
        self.assertEqual((value['min_delta_hits'], value['threshold_recall50'], value['min_distinct_winning_knobs']), (6, 0.002, 2))
        self.assertTrue(value['h7_config_committed_before_enqueue'])

    def test_changed_rule_file_is_rejected(self):
        from vkgr import retrieval_php01_select as s
        value = json.loads(RULE_FILE.read_text())
        for key, bad in (('threshold_recall50', 0.001), ('min_delta_hits', 5), ('min_distinct_winning_knobs', 1)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'rule.json'; path.write_text(json.dumps({**value, key: bad}))
                with self.assertRaises(ValueError, msg=key):
                    s.load_rule(path)


# ---------------------------------------------------------------- file / CLI path

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


BINDING = {'inputs': {'inputs_sha256': 'i' * 64}, 'rows_sha256': 'r' * 64, 'queries_sha256': 'q' * 64, 'catalog_sha256': 'c' * 64,
           'support_sha256': 's' * 64, 'side_sha256': 'd' * 64, 'weeks_sha256': 'w' * 64, 'bucket_spec_sha256': 'b' * 64}


def fake_run(tmp, name, delta_hits, n50=0.05, verified=True, config=None, binding=None):
    from vkgr.retrieval_php01_run import default_config, knob_values
    if name == 'anchor':
        schema, kind, config = 'p-feat02-train-manifest-v1', 'p_feat02_train', config or json.loads(F4_CONFIG.read_text())
    else:
        schema, kind = 'p-php01-train-manifest-v1', 'p_php01_train'
        config = config or {**default_config(), 'knob': name, **knob_values(name)}
    hits = ANCHOR_HITS + delta_hits
    manifest = {'schema': schema, 'kind': kind, 'status': 'complete', 'config': config, 'binding': {**BINDING, **(binding or {})},
                'selection': {'best_metric': [hits / N, n50], 'best_epoch': 10, 'epoch': 13}}
    run = Path(tmp) / name
    sha = _write(run / 'manifest.json', manifest)
    log = run / 'verify.log'; log.write_text('noise\n' + json.dumps({'status': 'verified' if verified else 'partial', 'best_epoch': 10}) + '\n')
    return {'manifest': str(run / 'manifest.json'), 'manifest_sha256': sha, 'verify_log': str(log), 'verify_log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}


def results_spec(tmp, deltas, **extra):
    spec = {'schema': 'p-php01-results-v1', 'anchor': fake_run(tmp, 'anchor', 0), 'runs': {}}
    for k in KNOBS:
        spec['runs'][k] = deltas[k] if isinstance(deltas.get(k), dict) else fake_run(tmp, k, deltas.get(k, -3))
    spec['runs'].update(extra)
    return spec


class FileTests(unittest.TestCase):
    def test_load_result_and_comparability(self):
        from vkgr import retrieval_php01_select as s
        with tempfile.TemporaryDirectory() as tmp:
            spec = results_spec(tmp, {'H1': 7})
            loaded = s.load_results(spec)
            self.assertEqual(loaded['anchor']['hits'], ANCHOR_HITS); self.assertEqual(loaded['results']['H1']['hits'], ANCHOR_HITS + 7)
            self.assertTrue(loaded['results']['H1']['verified'])
            for key in ('rows_sha256', 'inputs', 'weeks_sha256', 'bucket_spec_sha256'):
                bad = results_spec(tmp + f'/{key}', {}); bad['runs']['H2'] = fake_run(tmp + f'/{key}', 'H2', 0, binding={key: 'x'})
                with self.assertRaisesRegex(ValueError, 'comparab'):
                    s.load_results(bad)
            wrong_anchor = {**json.loads(F4_CONFIG.read_text()), 'seed': 2024}
            bad = results_spec(tmp + '/a', {}); bad['anchor'] = fake_run(tmp + '/a2', 'anchor', 0, config=wrong_anchor)
            with self.assertRaisesRegex(ValueError, 'anchor'):
                s.load_results(bad)
            bad = results_spec(tmp + '/b', {}); bad['runs']['H3'] = {**bad['runs']['H3'], 'manifest_sha256': '0' * 64}
            with self.assertRaisesRegex(ValueError, 'external hash mismatch'):
                s.load_results(bad)
            bad = results_spec(tmp + '/c', {}); bad['runs']['H3'] = fake_run(tmp + '/c', 'H3', 0, config=fake_config('H4'))
            with self.assertRaisesRegex(ValueError, 'knob'):
                s.load_results(bad)
            unverified = results_spec(tmp + '/d', {}); unverified['runs']['H4'] = fake_run(tmp + '/d', 'H4', 9, verified=False)
            self.assertFalse(s.load_results(unverified)['results']['H4']['verified'])
            failed = results_spec(tmp + '/e', {'H6': {'failed': 'oom'}})
            self.assertEqual(s.load_results(failed)['failed'], {'H6': 'oom'})

    def test_cli_decide_writes_h7_config_then_select(self):
        from vkgr import retrieval_php01_select as s
        from vkgr.retrieval_php01_run import validate_config
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); spec = results_spec(tmp, {'H2': 6, 'H3': 8})
            sha = _write(tmp / 'results.json', spec)
            argv = ['decide', '--results', str(tmp / 'results.json'), '--results-sha', sha, '--output', str(tmp / 'decision.json'), '--h7-config', str(tmp / 'h7.json')]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(s.main(argv), 0)
            receipt = json.loads(out.getvalue()); self.assertTrue(receipt['triggered'])
            text = (tmp / 'h7.json').read_text(); config = json.loads(text)
            self.assertEqual(text, json.dumps(config, indent=1, sort_keys=True) + '\n'); validate_config(config)
            decision = json.loads((tmp / 'decision.json').read_text())
            self.assertEqual(decision['schema'], 'p-php01-combo-decision-v1')
            self.assertEqual(decision['h7']['config_sha256'], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(decision['h7']['repo_path'], 'configs/php01/p-php01-h7-combo-s42.json')
            self.assertEqual(decision['results_sha256'], sha)
            self.assertEqual(decision['rule']['rule_file_sha256'], hashlib.sha256(RULE_FILE.read_bytes()).hexdigest())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(s.main(argv), 2)                 # outputs exist: never overwritten
            # H7 trained; select
            spec2 = {**spec, 'runs': {**spec['runs'], 'H7': fake_run(tmp, 'H7', 20, config=config)}}
            sha2 = _write(tmp / 'results2.json', spec2)
            argv = ['select', '--decision', str(tmp / 'decision.json'), '--decision-sha', receipt['decision_sha256'], '--results', str(tmp / 'results2.json'),
                    '--results-sha', sha2, '--output', str(tmp / 'selection.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(s.main(argv), 0)
            selection = json.loads((tmp / 'selection.json').read_text())
            self.assertEqual((selection['stage'], selection['winner']), ('final', 'H7'))
            # an H7 run whose config differs from the decided combo is rejected
            other = {**config, 'learning_rate': 0.002}
            spec3 = {**spec, 'runs': {**spec['runs'], 'H7': fake_run(tmp / 'x', 'H7', 20, config=other)}}
            sha3 = _write(tmp / 'results3.json', spec3)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(s.main([*argv[:6], str(tmp / 'results3.json'), '--results-sha', sha3, '--output', str(tmp / 'sel3.json')]), 2)
            self.assertIn('H7', err.getvalue())

    def test_cli_decide_void_needs_no_h7_config(self):
        from vkgr import retrieval_php01_select as s
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); sha = _write(tmp / 'results.json', results_spec(tmp, {'H1': 9}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(s.main(['decide', '--results', str(tmp / 'results.json'), '--results-sha', sha, '--output', str(tmp / 'd.json'), '--h7-config', str(tmp / 'h7.json')]), 0)
            self.assertFalse((tmp / 'h7.json').exists())
            self.assertEqual(json.loads((tmp / 'd.json').read_text())['h7'], 'void')

    def test_decide_rejects_tampered_rule(self):
        from vkgr import retrieval_php01_select as s
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); sha = _write(tmp / 'results.json', results_spec(tmp, {}))
            with mock.patch.object(s, 'MIN_DELTA_HITS', 5), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(s.main(['decide', '--results', str(tmp / 'results.json'), '--results-sha', sha, '--output', str(tmp / 'd.json'), '--h7-config', str(tmp / 'h7.json')]), 2)


def fake_config(knob):
    from vkgr.retrieval_php01_run import default_config, knob_values
    return {**default_config(), 'knob': knob, **knob_values(knob)}


if __name__ == '__main__':
    unittest.main()
