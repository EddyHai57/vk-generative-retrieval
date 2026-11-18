import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lsvd.pre_rental_probe import case_grid, run_child, run_cases


class ProbeTests(unittest.TestCase):
    def test_cuda_grid_covers_both_towers_precision_id_threshold_and_placement(self):
        cases = case_grid('two_tower', local_smoke=False)
        self.assertEqual(len(cases), 36)
        self.assertEqual({x['structure'] for x in cases}, {'mlp_mean', 'mlp_causal'})
        self.assertEqual({x['precision'] for x in cases}, {'fp32', 'tf32', 'bf16'})
        self.assertEqual({x['id_min_count'] for x in cases}, {5, 20, 100})
        self.assertEqual({x['data_placement'] for x in cases}, {'cpu', 'cuda'})
        generator = case_grid('generator', local_smoke=False)
        self.assertEqual(len(generator), 12)
        self.assertEqual({x['positions'] for x in generator}, {200, 600})
        self.assertEqual({x['batch_size'] for x in generator}, {256, 512, 1024})

    def test_timed_out_child_is_recorded_and_next_child_can_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            timed = run_child([sys.executable, '-c', 'import time;time.sleep(30)'],
                               root / 'timeout', .1)
            self.assertEqual(timed['status'], 'timeout')
            done = run_child([sys.executable, '-c', 'print("completed")'],
                              root / 'next', 5)
            self.assertEqual(done['returncode'], 0)

    def test_cpu_smoke_cli_enters_every_stage_and_records_cuda_skip(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'probe'
            result = subprocess.run(
                [sys.executable, '-m', 'lsvd.pre_rental_probe', '--synthetic',
                 '--local-smoke', '--output', str(output), '--threads', '1',
                 '--warmup', '1', '--steps', '1', '--total-seconds', '120'],
                capture_output=True, text=True, timeout=130,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            summary = json.loads((output / 'probe.json').read_text())
            self.assertEqual([row['stage'] for row in summary['stages']],
                             ['machine', 'two_tower', 'generator', 'checkpoint', 'evaluate'])
            self.assertTrue(summary['cuda_cases_skipped'])
            self.assertTrue(all(row['status'] == 'complete' for row in summary['stages']))
            self.assertTrue(summary['stages'][3]['result']['resume_losses_exact'])
            recovery = summary['stages'][3]['result']
            self.assertTrue(recovery['best_checkpoint_restored_exact'])
            self.assertGreater(recovery['checkpoint_with_best_bytes'], recovery['checkpoint_bytes'])
            for stage in summary['stages'][1:3]:
                self.assertTrue(all(case['status'] == 'complete'
                                    for case in stage['result']['cases']))

    def test_failed_case_keeps_stage_partial_and_runs_following_case(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'cases'
            output.mkdir()
            args = SimpleNamespace(output=output, synthetic=True, dataset=None,
                device='cpu', threads=1, local_smoke=True, total_seconds=30,
                warmup=0, steps=1, generator_vocab=3074, decoder_positions=3,
                download_url='')
            valid = case_grid('two_tower', local_smoke=True)[0]
            invalid = dict(valid, precision='invalid-precision')
            with patch('lsvd.pre_rental_probe.case_grid', return_value=[invalid, valid]):
                result = run_cases(args, 'two_tower')
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['cases'][0]['status'], 'failed')
            self.assertEqual(result['cases'][1]['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
