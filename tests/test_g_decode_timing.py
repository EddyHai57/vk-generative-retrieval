import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


class DecodeTimingTests(unittest.TestCase):
    def test_original_capacity_and_suffix(self):
        from analysis.g_decode_timing import balanced_sids
        actual = balanced_sids(11, 2, 3)
        expected = []
        for path, count in [((0, 0, 0), 2), ((0, 0, 1), 1),
                            ((0, 1, 0), 2), ((0, 1, 1), 1),
                            ((1, 0, 0), 2), ((1, 0, 1), 1),
                            ((1, 1, 0), 1), ((1, 1, 1), 1)]:
            expected.extend([list(path) + [suffix] for suffix in range(count)])
        self.assertTrue(np.array_equal(actual, np.array(expected, dtype=np.uint32)))
        self.assertEqual(actual.dtype, np.dtype('u4'))

    def test_capacity_when_child_count_is_bounded_by_n(self):
        from analysis.g_decode_timing import balanced_sids
        self.assertEqual(balanced_sids(2, 3, 3).tolist(), [[0, 0, 0, 0], [1, 0, 0, 0]])

    def test_comparison_ignores_only_measurements(self):
        from analysis.g_decode_timing import verify_results
        a = [{'result': [[1, 2], {'invalid': 0}], 'seconds': 1.0, 'vram_mib': 200}]
        b = [{'result': [[1, 2], {'invalid': 0}], 'seconds': 2.0, 'vram_mib': 300}]
        self.assertEqual(verify_results(a, b), verify_results(a, a))
        for bad in [[{'result': [[2, 1], {'invalid': 0}]}],
                    [{'result': [[1, 2], {'invalid': 1}]}], []]:
            with self.assertRaises(ValueError):
                verify_results(a, bad)

    def test_real_a1_codec_and_parallel_session_match_on_synthetic_input(self):
        if os.environ.get('VKGR_G_TIMING_TEST_CHILD') != '1':
            root = Path(__file__).resolve().parents[1]
            env = {**os.environ, 'VKGR_G_TIMING_TEST_CHILD': '1',
                   'PYTHONPATH': os.pathsep.join([str(root / 'src'), str(root / 'tests')])}
            result = subprocess.run([sys.executable, '-B', '-m', 'unittest',
                                     'test_g_decode_timing.DecodeTimingTests.test_real_a1_codec_and_parallel_session_match_on_synthetic_input'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        import tempfile
        from analysis.g_decode_timing import SessionFactory, balanced_sids, verify_results
        from vkgr.throughput_config import Engineering
        from vkgr.throughput_decode import decode_queries, InitializerBudget
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            np.save(root / 'ids.npy', np.arange(1, 12, dtype=np.uint32))
            np.save(root / 'sids.npy', balanced_sids(11, 2, 3))
            factory = SessionFactory(str(root), device='cpu')
            session = factory()
            self.assertEqual(session.model.spec['encoder_layers'], 4)
            self.assertEqual(session.model.spec['decoder_layers'], 4)
            self.assertGreaterEqual(session.trie_measurement['build_seconds'], 0)
            queries = [{'history': [1, 2] * 25}, {'history': [2, 1] * 25}]
            baseline = decode_queries(queries, session.decode, Engineering())
            budget = InitializerBudget.for_session(session, estimated_init_bytes=2**28,
                                                   budget_bytes_per_worker=2**30)
            other = decode_queries(queries, session.decode, Engineering(decode_workers=2),
                                   session_factory=factory, state_sha=session.identity(),
                                   initializer_budget=budget)
            self.assertEqual(verify_results(baseline, other), verify_results(baseline, baseline))

    def test_preflight_rejects_missing_context_peak(self):
        from analysis.g_decode_timing import require_gpu_peak
        require_gpu_peak(1024, 1)
        for peak, samples in [(0, 0), (0, 3), (1024, 0), (float('nan'), 1),
                              (-1, 1), (True, 1), (1024, True)]:
            with self.subTest(peak=peak, samples=samples), self.assertRaises(ValueError):
                require_gpu_peak(peak, samples)

    def test_formal_cli_negative_and_dry(self):
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        bad = subprocess.run([sys.executable, '-B', '-m', 'analysis.g_decode_timing', 'invalid'],
                             capture_output=True, text=True, env=env)
        self.assertNotEqual(bad.returncode, 0)
        good = subprocess.run([sys.executable, '-B', '-m', 'analysis.g_decode_timing', 'dry'],
                              capture_output=True, text=True, env=env)
        self.assertEqual(good.returncode, 0, good.stderr)
        doc = json.loads(good.stdout)
        self.assertEqual((doc['items'], doc['branching'], doc['queries'], doc['history']),
                         (12_012_895, 229, 512, 50))
        self.assertEqual(doc['workers'], [1, 2, 4, 8])
        self.assertFalse(doc['science_training'])


if __name__ == '__main__':
    unittest.main()
