import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from analysis.g_decode_timing import balanced_sids


class GPUDecodeTimingTests(unittest.TestCase):
    def test_session_is_worker_compatible_and_csr_mutation_changes_identity(self):
        from vkgr.gpu_decode_session import SessionFactory
        from vkgr.throughput_decode import TorchSession, InitializerBudget, decode_queries
        from vkgr.throughput_config import Engineering
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            np.save(root / 'ids.npy', np.arange(1, 12, dtype=np.uint32))
            np.save(root / 'sids.npy', balanced_sids(11, 229, 3))
            factory = SessionFactory(str(root), device='cpu')
            session = factory()
            self.assertIsInstance(session, TorchSession)
            queries = [{'history': [1, 2] * 25}, {'history': [2, 1] * 25}]
            serial = decode_queries(queries, session.decode, Engineering())
            budget = InitializerBudget.for_session(session, estimated_init_bytes=2**29, budget_bytes_per_worker=2**30)
            parallel = decode_queries(queries, session.decode, Engineering(decode_workers=2), session_factory=factory, state_sha=session.identity(), initializer_budget=budget)
            self.assertEqual([x['result'] for x in serial], [x['result'] for x in parallel])
            self.assertEqual(serial[0]['vram_mib'], 0)
            from analysis.gpu_trie_decode_probe import original_decode, trace_digest
            from vkgr.sid_depth import SIDTokensLD
            trace = []
            original = original_decode(session.model, SIDTokensLD(session.codec, 50), queries[0]['history'], trace=trace)
            self.assertEqual(serial[0]['result']['ranked'], original)
            self.assertEqual(serial[0]['result']['trace_sha256'], trace_digest(trace))
            self.assertEqual(set(serial[0]['result']), {'ranked', 'trace_sha256'})
            self.assertEqual(len(serial[0]['result']['trace_sha256']), 64)
            before = session.identity()
            session.trie.layers[0][2][0] += 1
            self.assertNotEqual(before, session.identity())

    def test_measurement_code_is_frozen_original(self):
        from analysis.gpu_decode_timing import timing_function
        from analysis.g_decode_timing import run
        adapted = timing_function(lambda *a: None, lambda *a: None, lambda *a: None)
        self.assertIs(adapted.__code__, run.__code__)

    def test_predecessor_rejects_wrong_argv_inputs_and_terminal(self):
        from analysis.gpu_decode_timing import accepted_producer
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(); (root / 'logs').mkdir()
            expected = {'run_id': 'synthetic-v1', 'git': {'commit': 'a'*40, 'dirty': False}, 'kind': 'ops', 'cwd': '/synthetic', 'argv': ['python', '-B', '-m', 'analysis.g_decode_timing', 'generate'], 'config': {'path': '/config', 'sha256': 'b'*64}, 'inputs': [{'path': '/input', 'sha256': 'c'*64}]}
            payload = {'synthetic_input': True}
            terminal = {'run_id': expected['run_id'], 'status': 'succeeded', 'exit_code': 0}
            (root/'run.json').write_text(json.dumps(expected)); (root/'terminal.json').write_text(json.dumps(terminal)); (root/'logs/stdout.log').write_text(json.dumps(payload))
            self.assertEqual(len(accepted_producer(root, expected, payload)), 64)
            for field, value in [('argv', ['wrong']), ('inputs', []), ('cwd', '/wrong'), ('git', {'commit': 'z'*40, 'dirty': False}), ('config', {})]:
                bad = {**expected, field: value}; (root/'run.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError): accepted_producer(root, expected, payload)
            (root/'run.json').write_text(json.dumps(expected))
            (root/'logs/stdout.log').write_text(json.dumps({'synthetic_input': False}))
            with self.assertRaises(ValueError): accepted_producer(root, expected, payload)
            (root/'logs/stdout.log').write_text(json.dumps(payload))
            (root/'terminal.json').write_text(json.dumps({**terminal,'status':'failed'}))
            with self.assertRaises(ValueError): accepted_producer(root, expected, payload)

    def test_launcher_input_size_metadata_preserves_pin_binding(self):
        from analysis.gpu_decode_timing import check_run
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw).resolve()/'input';path.write_text('abc')
            expected={'run_id':'r','kind':'ops','git':{'commit':'a'*40,'dirty':False},'cwd':'/source','argv':['python'],'config':{},'inputs':[{'path':str(path),'sha256':hashlib.sha256(b'abc').hexdigest()}]}
            actual={**expected,'inputs':[{**expected['inputs'][0],'bytes':3}]}
            check_run(actual,expected)
            actual['inputs'][0]['bytes']=4
            with self.assertRaises(ValueError):check_run(actual,expected)

    def test_query_trace_gate_rejects_score_path_and_order_changes(self):
        from analysis.gpu_decode_timing import verify_query
        trace=[[(tuple([0]*i),-0.25*i)] for i in range(1,5)]
        self.assertEqual(len(verify_query(([1],{}),trace,([1],{}),trace)),64)
        for changed in [trace[:3], [[((1,),-0.25)]]+trace[1:], trace[:3]+[[((0,0,0,0),-1.0000000000000002)]]]:
            with self.assertRaises(ValueError):verify_query(([1],{}),trace,([1],{}),changed)
        with self.assertRaises(ValueError):verify_query(([1,2],{}),trace,([2,1],{}),trace)

    def test_preflight_peak_includes_reference_trace_phase(self):
        from analysis.gpu_decode_timing import merge_parity_peak
        pre={'per_worker_peak_mib':512,'vram_samples':2}
        merge_parity_peak(pre,{'per_worker_peak_mib':1024,'vram_samples':3})
        self.assertEqual(pre,{'per_worker_peak_mib':1024,'vram_samples':5})
        with self.assertRaises(MemoryError):
            merge_parity_peak({'per_worker_peak_mib':512,'vram_samples':2},{'per_worker_peak_mib':8193,'vram_samples':3})

    def test_engine_requires_same_deployment_tree(self):
        from analysis import g_decode_timing as original
        from analysis.gpu_decode_timing import engine
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw).resolve()/'same-bytes.py';path.write_bytes(Path(original.__file__).read_bytes())
            before=original.__file__
            try:
                original.__file__=str(path)
                with self.assertRaises(ValueError):engine()
            finally:
                original.__file__=before

    def test_real_subset_fails_before_read_without_available_authorized_bundle(self):
        from analysis.gpu_decode_timing import real_subset
        with self.assertRaisesRegex(ValueError, 'no authorized real subset bundle'):
            real_subset(Path('/nonexistent/never-read'))

    def test_real_launcher_chain(self):
        done=subprocess.run([sys.executable,'-B','tests/fixtures/gpu_decode_timing_chain.py'],capture_output=True,text=True,timeout=90)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        self.assertEqual(json.loads(done.stdout)['status'],'real_launcher_prepare_metadata_dry_verified')

    def test_cli_dry_illegal_and_request_sha_required(self):
        cmd=[sys.executable,'-B','-m','analysis.gpu_decode_timing']
        good=subprocess.run(cmd+['dry'],capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stderr)
        value=json.loads(good.stdout)
        self.assertEqual(value['workers'],[1,2,4,8]);self.assertEqual(value['queries'],512)
        for argv in [['invalid'],['preflight'],['run'],['prepare']]:
            self.assertNotEqual(subprocess.run(cmd+argv,capture_output=True).returncode,0)


if __name__ == '__main__':
    unittest.main()
