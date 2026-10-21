import os
from pathlib import Path
import subprocess
import sys
import unittest


class SGNSThroughputTests(unittest.TestCase):
    def test_aggregate_identity_fails_before_synthetic_allocation(self):
        import tempfile
        from analysis.sgns_throughput import load_histogram
        with tempfile.TemporaryDirectory(dir='/private/tmp') as raw:
            path = Path(raw) / 'aggregate.json'
            path.write_text('{"train": {}}')
            with self.assertRaisesRegex(ValueError, 'aggregate identity'):
                load_histogram(path)

    def test_existing_output_refused_before_runtime_or_training(self):
        import tempfile
        from types import SimpleNamespace
        from analysis.sgns_throughput import execute
        with tempfile.TemporaryDirectory(dir='/private/tmp') as raw:
            with self.assertRaises(FileExistsError):
                execute(SimpleNamespace(root=Path(raw)))

    def test_histogram_sequences_are_exact_and_repeatable(self):
        from analysis.sgns_throughput import synthetic_histories
        a = synthetic_histories({'1': 2, '6': 1, '13': 2}, 17, 42)
        b = synthetic_histories({'1': 2, '6': 1, '13': 2}, 17, 42)
        self.assertEqual([len(x) for x in a.values()], [1, 1, 6, 13, 13])
        self.assertEqual([x.tolist() for x in a.values()], [x.tolist() for x in b.values()])
        self.assertTrue(all(1 <= int(v) <= 17 for rows in a.values() for v in rows))

    def test_frozen_extrapolation_includes_verify_ratio(self):
        from analysis.sgns_throughput import extrapolate
        value = extrapolate(8_192_000, 100)
        expected = 18_787_015_620 / 81_920
        self.assertEqual(value['fit_seconds'], expected)
        self.assertAlmostEqual(value['verify_seconds'], 64.40261220932007 * 18_787_015_620 / 173_851_080)
        self.assertEqual(value['within_96_hours'], value['total_seconds'] <= 96 * 3600)
        for bad in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                extrapolate(8_192_000, bad)

    def test_original_prefix_and_rng_are_unchanged_and_health_is_not_called(self):
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(root), str(root / 'src')])}
        result = subprocess.run([sys.executable, '-B', 'tests/fixtures/sgns_throughput_prefix_check.py'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('prefix_bitwise_equal', result.stdout)

    def test_formal_cli_dry_and_invalid(self):
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        base = [sys.executable, '-B', '-m', 'analysis.sgns_throughput']
        bad = subprocess.run([*base, 'invalid'], env=env, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        good = subprocess.run([*base, 'dry'], env=env, capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        import json
        value = json.loads(good.stdout)
        self.assertEqual((value['device'], value['batches']), ('cpu', 2000))
        self.assertFalse(value['science_training'])


if __name__ == '__main__':
    unittest.main()
