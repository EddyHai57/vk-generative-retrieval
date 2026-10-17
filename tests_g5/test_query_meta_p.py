"""Synthetic tests for the P-side query metadata extractor (uses the P code base's synthetic fixture)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import vkgr  # noqa: E402
from vkgr import retrieval_b1_run as b1  # noqa: E402
from vkgr import retrieval_feat02_run as fr  # noqa: E402
from vkgr_g5 import query_meta_p as qm  # noqa: E402

LINE_ROOT = str(Path(vkgr.__file__).resolve().parents[2])


def manifest_for(data, **overrides):
    rec = qm.extract(data, 50)
    binding = {'queries_sha256': rec['queries_digest'], 'catalog_sha256': rec['catalog_digest'], 'support_sha256': rec['support_sha256'], **overrides}
    return {'binding': binding}


class ExtractTests(unittest.TestCase):
    def test_rows_follow_query_order_with_labels_and_digests(self):
        data = fr.synthetic_data(); out = qm.extract(data, 50)
        self.assertEqual((out['schema'], out['split'], out['query_count'], out['catalog_size']), (qm.SCHEMA, 'validation', len(data['queries']), len(data['catalog'])))
        self.assertEqual([r['ordinal'] for r in out['rows']], list(range(len(data['queries']))))
        for row, q in zip(out['rows'], data['queries']):
            self.assertEqual((row['user_id'], row['target'], row['oov']), (q['user_id'], q['target'], bool(q['oov'])))
            self.assertEqual(row['history'], list(q['history'][-50:]))
            want = 'oov' if q['oov'] else ('supported' if q['target'] in data['support'] else 'unsupported')
            self.assertEqual(row['support_p'], want)

    def test_digests_are_the_binding_formulas(self):
        data = fr.synthetic_data(); out = qm.extract(data, 50)
        self.assertEqual(out['queries_digest'], b1._digest(fr._view(data, 50)[1]))
        self.assertEqual(out['catalog_digest'], b1._digest(list(data['catalog'])))
        self.assertEqual(out['support_sha256'], b1._digest(sorted(data['support'])))
        self.assertNotEqual(out['queries_digest'], b1._digest(out['rows']))

    def test_support_digest_is_over_the_sorted_list_not_set_iteration_order(self):
        data = fr.synthetic_data(); support = {8, 1, 16, 3, 24}
        self.assertNotEqual(list(support), sorted(support))                       # the fixture really distinguishes the two
        data = {**data, 'support': support}
        self.assertEqual(qm.extract(data, 50)['support_sha256'], b1._digest(sorted(support)))

    def test_duplicate_users_and_bad_history_are_fatal(self):
        data = fr.synthetic_data()
        dup = {**data, 'queries': [dict(q) for q in data['queries']]}; dup['queries'][1]['user_id'] = dup['queries'][0]['user_id']
        with self.assertRaisesRegex(ValueError, 'duplicate user_id'):
            qm.extract(dup, 50)
        empty = {**data, 'queries': [dict(q) for q in data['queries']]}; empty['queries'][0]['history'] = []
        with self.assertRaises((ValueError, IndexError)):
            qm.extract(empty, 50)

    def test_history_length_gate_runs_even_when_view_accepts_empty_aligned_fields(self):
        data = fr.synthetic_data(); queries = [dict(q) for q in data['queries']]
        queries[0].update(history=[], side=[], history_positions=[], ctx=[])
        self.assertEqual(len(fr._view({**data, 'queries': queries}, 50)[1][0]['history']), 0)           # the upstream view alone accepts it
        with self.assertRaisesRegex(ValueError, 'history length outside the L-view'):
            qm.extract({**data, 'queries': queries}, 50)

    def test_json_serialisable_and_has_no_vectors(self):
        text = json.dumps(qm.extract(fr.synthetic_data(), 50), allow_nan=False)
        for needle in ('embedding', 'vector', 'checkpoint'):
            self.assertNotIn(needle, text)


class ManifestGateTests(unittest.TestCase):
    def test_each_binding_class_is_compared_strictly_and_a_manifest_is_required(self):
        data = fr.synthetic_data(); rec = qm.extract(data, 50)
        self.assertEqual(qm.check_against_manifests(rec, rec['support_sha256'], [manifest_for(data), manifest_for(data)]), 2)
        for key in ('queries_sha256', 'catalog_sha256', 'support_sha256'):
            with self.assertRaisesRegex(ValueError, 'differ from an accepted run binding'):
                qm.check_against_manifests(rec, rec['support_sha256'], [manifest_for(data), manifest_for(data, **{key: '0' * 64})])
        with self.assertRaisesRegex(ValueError, 'at least one'):
            qm.check_against_manifests(rec, rec['support_sha256'], [])
        with self.assertRaises(ValueError):
            qm.check_against_manifests(rec, '1' * 64, [manifest_for(data)])             # the support hash argument itself is compared

    def test_nearest_existing_and_script_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(qm.nearest_existing(Path(tmp) / 'a' / 'b' / 'c'), Path(tmp))
        import hashlib
        self.assertEqual(qm.script_sha256(), hashlib.sha256(Path(qm.__file__).read_bytes()).hexdigest())


class CliTests(unittest.TestCase):
    def args(self, out, data, extra=()):
        return ['--line-root', LINE_ROOT, '--inputs', 'i', '--inputs-sha', 'a', '--config', 'c', '--config-sha', 'b', '--output', str(out), '--split', 'validation',
                '--manifest', 'm1', '--manifest-sha', 'd1', '--manifest', 'm2', '--manifest-sha', 'd2', '--expect-queries', str(len(data['queries'])),
                '--expect-oov', str(sum(1 for q in data['queries'] if q['oov'])), *extra]

    @contextlib.contextmanager
    def patched(self, data, manifests=None):
        manifests = manifests or [manifest_for(data), manifest_for(data)]
        pins = iter(manifests)
        def read_pinned(path, digest):
            if path == 'c':
                return {'history_len': 50}
            return {} if path == 'i' else next(pins)
        with mock.patch('vkgr.retrieval_b1_run.real_inputs', lambda spec: (data, None, None)), mock.patch('vkgr.retrieval_b1_run.resource_guard', lambda out, cap: (lambda: None)), \
                mock.patch('vkgr.retrieval_ablation_run.read_pinned', read_pinned), mock.patch('vkgr.retrieval_seqm1_run.validate_config', lambda c: c), \
                mock.patch('vkgr.retrieval_feat02_run.validate_inputs', lambda v, f: {'disk_cap_bytes': 1}), mock.patch('vkgr.retrieval_feat02_run.apply_feat02', lambda d, s, h: (d, None)):
            yield

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch('sys.stdout', out), mock.patch('sys.stderr', err):
            code = qm.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_check_only_prints_counts_and_creates_nothing_at_all(self):
        data = fr.synthetic_data()
        with tempfile.TemporaryDirectory() as tmp, self.patched(data):
            out = Path(tmp) / 'sub' / 'deeper' / 'meta.json'
            code, stdout, stderr = self.run_main(self.args(out, data, ['--check-only']))
            self.assertEqual(code, 0, stderr); got = json.loads(stdout)
            self.assertEqual((got['status'], got['written'], got['manifests_checked']), ('preflight_ok', False, 2))
            self.assertEqual(got['query_count'], len(data['queries'])); self.assertEqual(sum(got['support_p'].values()), len(data['queries']))
            self.assertGreater(got['serialized_bytes'], 0); self.assertEqual(len(got['script_sha256']), 64)
            self.assertFalse((Path(tmp) / 'sub').exists())                      # not even a parent directory

    def test_real_run_writes_one_new_file_in_a_fresh_directory_only(self):
        data = fr.synthetic_data()
        with tempfile.TemporaryDirectory() as tmp, self.patched(data):
            out = Path(tmp) / 'out' / 'meta.json'
            code, stdout, stderr = self.run_main(self.args(out, data))
            self.assertEqual(code, 0, stderr); written = json.loads(out.read_text())
            self.assertEqual((written['query_count'], written['manifests_checked'], len(written['script_sha256'])), (len(data['queries']), 2, 64))
            self.assertEqual(json.loads(stdout)['status'], 'complete')
        with tempfile.TemporaryDirectory() as tmp, self.patched(data):
            (Path(tmp) / 'out').mkdir()                                      # an existing directory must be refused, never written into
            code, _, stderr = self.run_main(self.args(Path(tmp) / 'out' / 'meta.json', data))
            self.assertEqual(code, 2); self.assertFalse((Path(tmp) / 'out' / 'meta.json').exists())

    def test_binding_difference_and_wrong_counts_fail_closed_without_writing(self):
        data = fr.synthetic_data()
        with tempfile.TemporaryDirectory() as tmp:
            bad = manifest_for(data, queries_sha256='0' * 64)
            with self.patched(data, [manifest_for(data), bad]):
                code, _, stderr = self.run_main(self.args(Path(tmp) / 'o' / 'm.json', data))
            self.assertEqual(code, 2); self.assertIn('differ from an accepted run binding', stderr); self.assertFalse((Path(tmp) / 'o').exists())
            with self.patched(data):
                argv = self.args(Path(tmp) / 'o2' / 'm.json', data); argv[argv.index('--expect-queries') + 1] = '2520'
                code, _, stderr = self.run_main(argv)
            self.assertEqual(code, 2); self.assertIn('registered validation set', stderr)
            with self.patched(data):
                argv = self.args(Path(tmp) / 'o3' / 'm.json', data); argv[argv.index('--expect-oov') + 1] = '623'
                self.assertEqual(self.run_main(argv)[0], 2)
            with self.patched(data):
                argv = self.args(Path(tmp) / 'o4' / 'm.json', data)[:-0 or None]; del argv[argv.index('--manifest-sha') + 1]; argv.remove('--manifest-sha')
                self.assertEqual(self.run_main(argv)[0], 2)

    def test_resource_guard_receives_an_existing_directory_even_for_a_fresh_output_tree(self):
        data = fr.synthetic_data(); seen = []
        with tempfile.TemporaryDirectory() as tmp, self.patched(data):
            def guard(path, cap):
                seen.append(Path(path)); return lambda: None
            with mock.patch('vkgr.retrieval_b1_run.resource_guard', guard):
                self.run_main(self.args(Path(tmp) / 'a' / 'b' / 'meta.json', data, ['--check-only']))
            self.assertEqual(seen, [Path(tmp).resolve()]); self.assertTrue(seen[0].exists())

    def test_subprocess_failures_exit_two_not_a_traceback(self):
        import subprocess
        data = fr.synthetic_data()
        with tempfile.TemporaryDirectory() as tmp, self.patched(data):
            with mock.patch('vkgr.retrieval_b1_run.resource_guard', side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')):
                code, _, err = self.run_main(self.args(Path(tmp) / 'o' / 'm.json', data))
            self.assertEqual(code, 2); self.assertIn('CalledProcessError', err)

    def test_registered_counts_must_match_exactly_in_both_directions(self):
        data = fr.synthetic_data()
        actual_q = len(data['queries']); actual_oov = sum(1 for q in data['queries'] if q['oov'])
        for flag, actual in (('--expect-queries', actual_q), ('--expect-oov', actual_oov)):
            for wrong in (actual - 1, actual + 1):
                with tempfile.TemporaryDirectory() as tmp, self.patched(data):
                    argv = self.args(Path(tmp) / 'o' / 'm.json', data, ['--check-only']); argv[argv.index(flag) + 1] = str(wrong)
                    code, _, err = self.run_main(argv)
                self.assertEqual(code, 2, (flag, wrong)); self.assertIn('registered validation set', err)

    def test_split_root_and_output_gates(self):
        data = fr.synthetic_data()
        with self.assertRaises(SystemExit):
            qm.main(self.args('o', data)[:self.args('o', data).index('--split') + 1] + ['test'] + self.args('o', data)[self.args('o', data).index('--split') + 2:])
        argv = self.args('/tmp/q-meta-x.json', data); argv[1] = '/nonexistent-root'
        code, _, err = self.run_main(argv); self.assertEqual(code, 2); self.assertIn('vkgr is not imported from --line-root', err)
        argv = self.args('/tmp/q-meta-x.json', data); argv[1] = str(Path(LINE_ROOT).parent)          # an ancestor of the real root must not pass
        code, _, err = self.run_main(argv); self.assertEqual(code, 2); self.assertIn('vkgr is not imported from --line-root', err)
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'out.json'; existing.write_text('{}')
            code, _, err = self.run_main(self.args(existing, data)); self.assertEqual(code, 2); self.assertIn('new file outside the deployment', err)
            partial = Path(tmp) / 'p.json.partial'; partial.write_text('x')
            with self.patched(data):
                code, _, err = self.run_main(self.args(Path(tmp) / 'p.json', data))
            self.assertEqual(code, 2); self.assertIn('new file outside the deployment', err)
            code, _, err = self.run_main(self.args(Path(LINE_ROOT) / 'meta-in-deployment.json', data)); self.assertEqual(code, 2); self.assertIn('outside the deployment', err)


if __name__ == '__main__':
    unittest.main()
