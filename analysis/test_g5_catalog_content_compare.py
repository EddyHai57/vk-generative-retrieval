"""Hand-computed synthetic cases for the authorised read-only comparison."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import hashlib
from unittest.mock import patch
from types import SimpleNamespace
import json

SPEC = importlib.util.spec_from_file_location('catcompare', Path(__file__).with_name('g5_catalog_content_compare.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class CompareTests(unittest.TestCase):
    def side(self, values, newline=False):
        return m.describe(values, newline=newline)

    def test_equal_content_different_serialisation_is_branch_one(self):
        p, g = self.side([1, 3, 9]), self.side([1, 3, 9], True)
        r = m.compare(p, g)
        self.assertEqual(r['branch'], 1)
        self.assertTrue(r['same_set'] and r['same_order'])
        self.assertNotEqual(r['p']['original_digest'], r['g']['original_digest'])
        self.assertEqual(r['p']['normalised_digest'], r['g']['normalised_digest'])
        self.assertNotIn('values', r['p'])

    def test_reorder_is_branch_two_with_original_order_reported(self):
        r = m.compare(self.side([1, 3, 9]), self.side([9, 1, 3]))
        self.assertEqual(r['branch'], 2)
        self.assertTrue(r['p']['strictly_ascending'])
        self.assertFalse(r['g']['strictly_ascending'])
        self.assertEqual(r['different_positions_count'], 3)

    def test_same_size_different_set_stops(self):
        r = m.compare(self.side([1, 3, 9]), self.side([1, 3, 8]))
        self.assertEqual(r['branch'], 3)
        self.assertEqual((r['only_p_count'], r['only_g_count']), (1, 1))

    def test_duplicate_float_bool_and_empty_are_rejected(self):
        for x in ([1, 1], [1.0, 2], [True, 2], []):
            with self.assertRaises(ValueError):
                self.side(x)

    def test_report_has_bounded_indices_not_catalog(self):
        r = m.compare(self.side(list(range(200))), self.side(list(reversed(range(200)))))
        self.assertLessEqual(len(r['different_positions']), 5)
        self.assertNotIn('values', r['g'])

    def test_read_pins_exact_bytes_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'meta.json'; p.write_bytes(b'{"a":1}')
            self.assertEqual(m.read_json(p, hashlib.sha256(p.read_bytes()).hexdigest()), {'a': 1})
            with self.assertRaises(ValueError): m.read_json(p, '0' * 64)
            link = Path(d).resolve() / 'alias.json'; link.symlink_to(p)
            with self.assertRaises(ValueError): m.read_json(link, hashlib.sha256(p.read_bytes()).hexdigest())

    def test_unrecognised_native_formula_stops(self):
        r = self.side([1, 2])
        with self.assertRaises(ValueError): m.native_description([1, 2], lambda _: '0' * 64)
        self.assertEqual(m.native_description([1, 2], lambda _: r['original_digest'])['serialization'], r['serialization'])
        lf = self.side([1, 2], True)
        self.assertEqual(m.native_description([1, 2], lambda _: lf['original_digest'])['serialization'], lf['serialization'])

    def test_scratch_is_new_run_fixed_child_not_an_old_bundle_child(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); run = root / 'runs/g5-catalog-content-diag-v1'
            run.mkdir(parents=True); (run / 'run.json').write_text('{"run_id":"g5-catalog-content-diag-v1"}')
            with patch.dict(m.os.environ, VKGR_RUN_ID=run.name):
                self.assertEqual(m.check_scratch(str(run / 'catalog-scratch'), root), run / 'catalog-scratch')
                for p in (root / 'old-run/catalog-scratch', run / '../other/catalog-scratch', run / 'other'):
                    with self.assertRaises(ValueError): m.check_scratch(str(p), root)
                (run / 'catalog-scratch').mkdir()
                with self.assertRaises(ValueError): m.check_scratch(str(run / 'catalog-scratch'), root)

    def test_producer_receipt_pin_and_last_json_without_prediction_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'stdout.log'; p.write_bytes(b'progress\n{"catalog_digest":"x"}\n')
            self.assertEqual(m.read_receipt(p, hashlib.sha256(p.read_bytes()).hexdigest()), {'catalog_digest': 'x'})
            with self.assertRaises(ValueError): m.read_receipt(p, '0' * 64)

    def test_collect_routes_only_to_pinned_stdout_and_original_loader_environments(self):
        req = {'scratch': '/newrun/catalog-scratch'}
        jobs = {}
        for side, rid in [('p', 'g5-fusion-qmeta'), ('g', 'g5-fusion-asm-s42')]:
            req.update({side+'_job': '/'+side+'-job', side+'_job_sha': side,
                        side+'_metadata': '/'+side+'-stdout', side+'_metadata_sha': side})
            jobs[req[side+'_job']] = {'run_id': rid, 'cwd': '/'+side+'-original',
                'argv': ['/'+side+'-python'], 'env': ['PYTHONPATH=/'+side+'-original/src', 'VKGR_RUN_ID=old']}
        catalog = list(range(120415)); results = [m.describe(catalog), m.describe(catalog, newline=True)]
        headers = {req[s+'_metadata']: {'catalog_digest': r['original_digest']} for s, r in zip(('p', 'g'), results)}
        with patch.object(m, 'check_scratch'), patch.object(m, 'read_json', side_effect=lambda p, h: jobs[p]) as meta, \
             patch.object(m, 'read_receipt', side_effect=lambda p, h: headers[p]) as receipt, \
             patch.object(m.subprocess, 'run', side_effect=[SimpleNamespace(stdout=json.dumps(x)) for x in results]) as child, \
             patch.object(m.Path, 'read_bytes', return_value=b'identity'), \
             patch.dict(m.os.environ, VKGR_RUN_ID='g5-catalog-content-diag-v1'):
            out = m.collect(req)
        self.assertEqual(out['branch'], 1)
        self.assertNotIn('values', out['p'])
        self.assertEqual(meta.call_count, 2); self.assertEqual(receipt.call_count, 2)
        for side, call in zip(('p', 'g'), child.call_args_list):
            self.assertEqual(call.args[0][0], '/'+side+'-python')
            self.assertEqual(call.kwargs['cwd'], '/'+side+'-original')
            self.assertEqual(call.kwargs['env']['VKGR_RUN_ID'], 'g5-catalog-content-diag-v1')


if __name__ == '__main__':
    unittest.main()
