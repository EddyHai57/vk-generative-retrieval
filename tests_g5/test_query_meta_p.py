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
