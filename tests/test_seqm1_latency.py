"""Latency contract guards; fixtures contain no real query identifiers."""
import contextlib
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

spec = importlib.util.spec_from_file_location('latency', Path(__file__).resolve().parents[1] / 'scripts/seqm1_latency_bench.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def users(self, h, s):
        self.seen.append((h.clone(), s.clone()))
        return h.sum(1)


class LatencyTests(unittest.TestCase):
    def source(self, root, rows):
        path = root / 'validation.json'
        path.write_text(json.dumps(rows))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_source_hash_count_and_range_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            path, sha = self.source(Path(d), [{'history': [1] * n, 'target': 'ignored'} for n in (1, 7, 50)])
            source = b.load_lengths(path, sha, expected_count=3)
            self.assertEqual(source['histogram'], {'1': 1, '7': 1, '50': 1})
            self.assertEqual(source['sha256'], sha)
            for bad_sha, count in [('0' * 64, 3), (sha, 2520)]:
                with self.assertRaises(ValueError):
                    b.load_lengths(path, bad_sha, expected_count=count)
            path, sha = self.source(Path(d), [{'history': []}])
            with self.assertRaises(ValueError):
                b.load_lengths(path, sha, expected_count=1)

    def test_empirical_schedule_has_no_uniform_fallback_and_is_deterministic(self):
        a = b.length_schedule({'1': 1, '7': 2, '50': 7}, 1000, 7)
        self.assertEqual(a, b.length_schedule({'1': 1, '7': 2, '50': 7}, 1000, 7))
        self.assertEqual(b.histogram(a), {'1': 100, '7': 200, '50': 700})

    def test_actual_calls_replay_inputs_and_report_counts(self):
        source = {'histogram': {'1': 1, '7': 2}, 'sha256': 'a' * 64, 'count': 3}
        left, right = Recorder(), Recorder()
        a = b.measure('a', left, 'cpu', 32, 2, 23, 3, source, seed=7)
        other = b.measure('b', right, 'cpu', 32, 2, 23, 3, source, seed=7)
        self.assertEqual(len(left.seen), 2 * (23 + 3))
        for x, y in zip(left.seen, right.seen):
            self.assertTrue(all(torch.equal(i, j) for i, j in zip(x, y)))
        for mode, rows in [('offline_ms_per_query', 46), ('online_ms_per_query', 23)]:
            self.assertEqual(a[mode]['n'], 23)
            self.assertEqual(a[mode]['timed_queries'], rows)
            self.assertEqual(sum(a[mode]['timed_length_histogram'].values()), rows)
            self.assertEqual(a[mode]['input_sha256'], other[mode]['input_sha256'])
            self.assertEqual(len(a[mode]['raw_ms_per_query']), 23)
        self.assertTrue(all(x[0].shape[1] in (1, 7) for x in left.seen[26:]))

    def test_formal_parameters_are_locked_and_source_is_mandatory(self):
        parser = b.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(['--lengths-json', 'validation.json', '--lengths-sha', 'a'*64, '--iters', '2'])
        with self.assertRaises(ValueError):
            b.validate_args(args)
        args.fixture_mode = True
        b.validate_args(args)

    def test_export_pass_covers_source_once_including_partial_batch(self):
        model = Recorder()
        result = b.export_pass(model, [1, 7, 7, 50, 1], 'cpu', 32, 2)
        self.assertEqual(result['batch_sizes'], [2, 2, 1])
        self.assertEqual(result['queries'], 5)
        self.assertEqual(result['calls'], 3)
        self.assertEqual(result['length_histogram'], {'1': 2, '7': 2, '50': 1})
        self.assertEqual([h.shape for h, _ in model.seen], [torch.Size([2, 50]), torch.Size([2, 50]), torch.Size([1, 50])])

    def test_tiny_cli_exercises_actual_anchor_and_mean(self):
        with tempfile.TemporaryDirectory() as d:
            path, sha = self.source(Path(d), [{'history': [1] * n} for n in (1, 7, 50)])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                b.main(['--lengths-json', str(path), '--lengths-sha', sha, '--fixture-mode',
                        '--expected-count', '3', '--catalog', '64', '--batch', '2', '--iters', '2', '--warmup', '1'])
            report = json.loads(output.getvalue())
            self.assertFalse(report['formal'])
            self.assertEqual(report['threads'], 1)
            self.assertEqual(report['length_source']['sha256'], sha)
            for mode in ('online_ms_per_query', 'offline_ms_per_query'):
                self.assertEqual(report['models']['anchor'][mode]['n'], 2)
                self.assertEqual(report['models']['anchor'][mode]['input_sha256'], report['models']['mean'][mode]['input_sha256'])

    def test_summary_nearest_rank_reproducible(self):
        result = b.summary(list(range(1, 101)))
        self.assertEqual([result[k] for k in ('mean', 'p50', 'p95', 'p99', 'n')], [50.5, 50, 95, 99, 100])


if __name__ == '__main__':
    unittest.main()
