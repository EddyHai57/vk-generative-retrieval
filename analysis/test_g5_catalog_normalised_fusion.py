"""Synthetic checks for the already authorised branch-one representation fix."""
import importlib.util
from pathlib import Path
import copy
import unittest
from types import SimpleNamespace
import tempfile
import json
from unittest.mock import patch
from unittest.mock import Mock
import sys

spec = importlib.util.spec_from_file_location('normalfusion', Path(__file__).with_name('g5_catalog_normalised_fusion.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class NormalisationTests(unittest.TestCase):
    def proof(self):
        return {'status': 'catalog_content_compared', 'run_id': 'g5-catalog-content-diag-v1',
            'script_sha256': m.DIAG_SCRIPT_SHA, 'run_manifest_sha256': m.DIAG_RUN_SHA,
            'same_set': True, 'same_order': True, 'branch': 1, 'only_p_count': 0, 'only_g_count': 0,
            'different_positions_count': 0, 'test_read': False, 'checkpoint_read': False, 'prediction_read': False,
            'p': {'count': 120415, 'strictly_ascending': True, 'original_digest': m.P_DIGEST, 'normalised_digest': m.P_DIGEST},
            'g': {'count': 120415, 'strictly_ascending': True, 'original_digest': m.G_DIGEST, 'normalised_digest': m.P_DIGEST}}

    def records(self):
        return {role: {'catalog_size': 120415, 'catalog_digest': m.P_DIGEST if role == 'p' else m.G_DIGEST,
                       'rows': [{'items': [7, 2, 5], 'history': [1, 8]}], 'seed': role}
                for role in ('p', '42', '2024', '2025')}

    def test_only_representation_and_provenance_change_without_mutating_inputs(self):
        before = self.records(); original = copy.deepcopy(before)
        after = m.normalise(before, self.proof(), {k: 'a'*64 for k in before})
        self.assertEqual(before, original)
        for role, value in after.items():
            self.assertEqual(value['catalog_digest'], m.P_DIGEST)
            rest = {k:v for k,v in value.items() if k not in ('catalog_digest', 'catalog_normalisation')}
            self.assertEqual(rest, {k:v for k,v in original[role].items() if k != 'catalog_digest'})
            self.assertEqual(value['catalog_normalisation']['proof_sha256'], m.PROOF_SHA)

    def test_set_order_branch_or_diagnostic_binding_change_stops(self):
        for key, value in [('same_set', False), ('same_order', False), ('branch', 2), ('script_sha256', '0'*64),
                           ('run_manifest_sha256', '0'*64), ('only_g_count', 1), ('test_read', True)]:
            p = self.proof(); p[key] = value
            with self.assertRaises(ValueError): m.normalise(self.records(), p, {k:'a'*64 for k in self.records()})

    def test_native_digest_count_missing_seed_or_duplicate_normalisation_stops(self):
        for mode in ('digest', 'count', 'missing', 'repeat'):
            r = self.records()
            if mode == 'digest': r['2024']['catalog_digest'] = '0'*64
            if mode == 'count': r['2025']['catalog_size'] = 120414
            if mode == 'missing': del r['2025']
            if mode == 'repeat': r['42']['catalog_normalisation'] = {}
            with self.assertRaises(ValueError): m.normalise(r, self.proof(), {k:'a'*64 for k in r})

    def test_preflight_failure_stops_without_formal_call(self):
        calls = []
        fake = SimpleNamespace(main=lambda a: calls.append(a) or 2)
        with self.assertRaises(ValueError): m.run_commands(fake, ['fixed-args'])
        self.assertEqual(calls, [['fixed-args', '--check-only']])

    def test_preflight_then_formal_with_identical_arguments(self):
        calls = []
        fake = SimpleNamespace(main=lambda a: calls.append(a) or 0)
        m.run_commands(fake, ['fixed-args'])
        self.assertEqual(calls, [['fixed-args', '--check-only'], ['fixed-args']])

    def test_verify_replays_but_rejects_modified_statistics_or_input_bindings(self):
        p = {s: ('p', str(s)) for s in (42, 2024, 2025)}
        g = {s: ('g', str(s)) for s in p}; meta = ('meta', 'meta-sha')
        expected = {'rows': {'C1': {'hits': 2}}, 'paired': {'p': 0.5}}
        fake = SimpleNamespace(build_inputs=lambda *a: ('routes', 'queries', 120415, {'checked': True}),
                               fv=SimpleNamespace(run=lambda *a: expected))
        r = {'schema': 'g5-fusion-results-v1', 'checks': {'checked': True}, 'fusion': copy.deepcopy(expected),
             'inputs': {'p_export': {str(s):v[1] for s,v in p.items()}, 'g_assembly': {str(s):v[1] for s,v in g.items()}, 'p_query_meta': meta[1]},
             'provenance': {'fuse_script_sha256': m.FUSE_SHA, 'fusion_v2_script_sha256': m.CORE_SHA}}
        self.assertTrue(m.verify_primary(fake, r, p, g, meta))
        for key in ('fuse_script_sha256', 'fusion_v2_script_sha256'):
            bad = copy.deepcopy(r); bad['provenance'][key] = '0'*64
            with self.assertRaises(ValueError): m.verify_primary(fake, bad, p, g, meta)
        r['fusion']['rows']['C1']['hits'] = 3
        with self.assertRaises(ValueError): m.verify_primary(fake, r, p, g, meta)
        r['fusion'] = copy.deepcopy(expected); r['inputs']['p_query_meta'] = 'wrong'
        with self.assertRaises(ValueError): m.verify_primary(fake, r, p, g, meta)

    def test_pipeline_output_cannot_target_an_old_run_or_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); run = root / 'runs/g5-fusion-primary-v1'; run.mkdir(parents=True)
            (run/'run.json').write_text(json.dumps({'run_id':run.name}))
            with patch.dict(m.os.environ, VKGR_RUN_ID=run.name):
                self.assertEqual(m.new_output(str(run/'out'), root), run/'out')
                with self.assertRaises(ValueError): m.new_output(str(root/'runs/old/out'), root)
                (run/'out').mkdir()
                with self.assertRaises(ValueError): m.new_output(str(run/'out'), root)

    def test_exclusive_json_write_refuses_existing_path(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json'; m.write_new(p, {'x':1})
            with self.assertRaises(FileExistsError): m.write_new(p, {'x':2})
            self.assertEqual(json.loads(p.read_text()), {'x':1})

    def test_pipeline_rejects_an_unregistered_request_before_reading_inputs(self):
        with self.assertRaises(ValueError): m.pipeline({}, None)

    def request(self, out='/unused/out'):
        return {'proof': ['/proof.json', m.PROOF_SHA], 'meta': ['/meta.json', m.META_SHA],
                'g': {s:['/g'+s+'.json', sha] for s,sha in m.G_SHAS.items()},
                'p': {s:['/p'+s+'.json', sha] for s,sha in m.P_SHAS.items()}, 'output_root': out}

    def test_bad_original_pin_is_rejected_before_input_io(self):
        for role in ('proof', 'meta', 'p', 'g'):
            req = self.request()
            (req[role] if role in ('proof','meta') else req[role]['2024'])[1] = '0'*64
            with patch.object(m, 'read_pinned') as read:
                with self.assertRaisesRegex(ValueError, 'accepted original'): m.pipeline(req, None)
                read.assert_not_called()

    def test_main_rejects_wrong_original_source_before_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'wrong.py'; f.write_text('not the frozen source')
            fake = SimpleNamespace(__file__=str(f), fv=SimpleNamespace(__file__=str(f)))
            with patch.dict(sys.modules, {'vkgr_g5': SimpleNamespace(fuse=fake)}), \
                 patch.object(sys, 'argv', ['entry','--request','/unused','--request-sha','0'*64]), \
                 patch.object(m, 'read_pinned', return_value={}), patch.object(m, 'pipeline', return_value={}) as pipe:
                with self.assertRaisesRegex(ValueError, 'frozen original fusion source'): m.main()
                pipe.assert_not_called()

    def test_changed_written_candidate_is_rejected_before_fuse(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d).resolve()/'out'; req=self.request(str(out)); records=self.records()
            originals={'/proof.json':self.proof(), '/meta.json':records['p'],
                       **{'/g'+s+'.json':records[s] for s in m.G_SHAS}}
            real_read=m.read_pinned
            def read(path, pin):
                if path in originals: return copy.deepcopy(originals[path])
                value=real_read(path, pin)
                if Path(path).name=='42.json': value['rows'][0]['items'][0]=999
                return value
            fake=SimpleNamespace(refuse_held_out_paths=lambda _:None, main=Mock(side_effect=AssertionError('fuse reached')))
            with patch.object(m, 'new_output', return_value=out), patch.object(m, 'read_pinned', side_effect=read):
                with self.assertRaisesRegex(ValueError, 'normalisation full-field replay'): m.pipeline(req, fake)
            fake.main.assert_not_called()


if __name__ == '__main__': unittest.main()
