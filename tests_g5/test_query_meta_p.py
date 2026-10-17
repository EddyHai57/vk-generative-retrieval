"""Synthetic tests for the P-side query metadata extractor (uses the P code base's synthetic fixture)."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from vkgr import retrieval_feat02_run as fr  # noqa: E402
from vkgr_g5 import query_meta_p as qm  # noqa: E402


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
        self.assertEqual(len(out['queries_digest']), 64); self.assertEqual(out['queries_digest'], qm.extract(data, 50)['queries_digest'])

    def test_duplicate_users_and_bad_history_are_fatal(self):
        data = fr.synthetic_data()
        dup = {**data, 'queries': [dict(q) for q in data['queries']]}; dup['queries'][1]['user_id'] = dup['queries'][0]['user_id']
        with self.assertRaisesRegex(ValueError, 'duplicate user_id'):
            qm.extract(dup, 50)
        empty = {**data, 'queries': [dict(q) for q in data['queries']]}; empty['queries'][0]['history'] = []
        with self.assertRaises((ValueError, IndexError)):
            qm.extract(empty, 50)

    def test_json_serialisable_and_has_no_vectors(self):
        text = json.dumps(qm.extract(fr.synthetic_data(), 50))
        for needle in ('embedding', 'vector', 'checkpoint'):
            self.assertNotIn(needle, text)


class CheckOnlyTests(unittest.TestCase):
    def test_check_only_prints_counts_and_writes_nothing(self):
        import contextlib
        data = fr.synthetic_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'sub' / 'meta.json'
            fake_b1 = mock.MagicMock(); fake_b1.real_inputs.return_value = (data, None, None); fake_b1.resource_guard.return_value = lambda: None
            fake_b1._digest = __import__('vkgr.retrieval_b1_run', fromlist=['x'])._digest
            fake_b1._write_json_atomic.side_effect = AssertionError('must not write')
            buf = io.StringIO(); err = io.StringIO()
            import vkgr
            with mock.patch('vkgr.retrieval_b1_run.real_inputs', fake_b1.real_inputs), mock.patch('vkgr.retrieval_b1_run.resource_guard', fake_b1.resource_guard), \
                    mock.patch('vkgr.retrieval_b1_run._write_json_atomic', fake_b1._write_json_atomic), \
                    mock.patch('vkgr.retrieval_ablation_run.read_pinned', lambda p, d: {'history_len': 50}), mock.patch('vkgr.retrieval_seqm1_run.validate_config', lambda c: c), \
                    mock.patch('vkgr.retrieval_feat02_run.validate_inputs', lambda v, f: {'disk_cap_bytes': 1}), \
                    mock.patch('vkgr.retrieval_feat02_run.apply_feat02', lambda d, s, h: (d, None)), mock.patch('sys.stdout', buf), mock.patch('sys.stderr', err):
                code = qm.main(['--line-root', str(Path(vkgr.__file__).resolve().parents[2]), '--inputs', 'i', '--inputs-sha', 'a', '--config', 'c', '--config-sha', 'b',
                                '--output', str(out), '--split', 'validation', '--check-only'])
            self.assertEqual(code, 0, err.getvalue())
            got = json.loads(buf.getvalue()); self.assertEqual(got['status'], 'preflight_ok'); self.assertFalse(got['written'])
            self.assertEqual(got['query_count'], len(data['queries'])); self.assertFalse(out.exists())
            self.assertEqual(sum(got['support_p'].values()), len(data['queries']))


class CliTests(unittest.TestCase):
    def run_main(self, args):
        err = io.StringIO()
        with mock.patch('sys.stderr', err):
            code = qm.main(args)
        return code, err.getvalue()

    def test_split_must_be_validation_and_output_must_be_new_outside_root(self):
        base = ['--inputs', 'i', '--inputs-sha', 'a', '--config', 'c', '--config-sha', 'b']
        with self.assertRaises(SystemExit):
            qm.main(['--line-root', str(ROOT), *base, '--output', 'o', '--split', 'test'])
        code, err = self.run_main(['--line-root', '/nonexistent-root', *base, '--output', '/tmp/q-meta-x.json', '--split', 'validation'])
        self.assertEqual(code, 2); self.assertIn('vkgr is not imported from --line-root', err)
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'out.json'; existing.write_text('{}')
            import vkgr
            code, err = self.run_main(['--line-root', str(Path(vkgr.__file__).resolve().parents[2]), *base, '--output', str(existing), '--split', 'validation'])
            self.assertEqual(code, 2); self.assertIn('new file outside the deployment', err)


if __name__ == '__main__':
    unittest.main()
