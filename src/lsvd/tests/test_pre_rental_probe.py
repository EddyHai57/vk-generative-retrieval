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
        self.assertEqual(len({tuple(row.values()) for row in cases}), 36)
        keys = ('structure', 'precision', 'id_min_count', 'data_placement')
        self.assertEqual([tuple(row[key] for key in keys) for row in cases[:7]], [
            ('mlp_causal', 'bf16', 20, 'cuda'),
            ('mlp_causal', 'fp32', 20, 'cuda'),
            ('mlp_causal', 'bf16', 20, 'cpu'),
            ('mlp_causal', 'bf16', 5, 'cuda'),
            ('mlp_causal', 'tf32', 20, 'cuda'),
            ('mlp_mean', 'bf16', 20, 'cuda'),
            ('mlp_mean', 'fp32', 20, 'cuda'),
        ])
        self.assertEqual([(row['positions'], row['precision'], row['batch_size'])
                          for row in generator[:6]],
                         [(200, dtype, size) for dtype in ('bf16', 'fp32')
                          for size in (256, 512, 1024)])
        self.assertTrue(all(row['positions'] == 600 for row in generator[6:]))

    def test_fp32_comparison_flags_field_mask_and_vector_mismatches(self):
        import torch
        from lsvd.pre_rental_cases import compare_fp32

        inputs = {'history.items': torch.tensor([3, 7]),
                  'mask': torch.tensor([[True, False], [True, True]])}
        outputs = {name: torch.tensor([1.0]) for name in ('loss', 'users', 'items')}
        self.assertEqual(compare_fp32((inputs, outputs), (inputs, outputs))['status'], 'pass')
        below = compare_fp32((inputs, outputs), (inputs, dict(outputs, users=torch.tensor([1.00005]))))
        self.assertEqual(below['status'], 'pass')
        changed = dict(inputs, mask=~inputs['mask'])
        failed = compare_fp32((inputs, outputs), (changed, dict(outputs, items=torch.tensor([1.001]))))
        self.assertEqual(failed['status'], 'fail')
        self.assertEqual(failed['mismatched_fields'], ['mask'])
        self.assertGreater(failed['max_abs_diff']['items'], 1e-4)
        self.assertEqual(compare_fp32((inputs, outputs),
            (inputs, dict(outputs, loss=torch.tensor([float('nan')]))))['status'], 'fail')

    def test_progress_recovers_attempted_case_after_stage_timeout(self):
        from lsvd.pre_rental_probe import case_progress

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            first = output / 'positions-200-batch-256-bf16'
            second = output / 'positions-200-batch-512-bf16'
            first.mkdir()
            first.with_suffix('.launch.json').touch()
            (first / 'result.json').write_text('{"status": "complete"}')
            second.with_suffix('.launch.json').touch()
            progress = case_progress(output, 'generator', local_smoke=False)
            self.assertEqual(progress['planned'], 12)
            self.assertEqual(progress['last_attempted_position'], 2)
            self.assertEqual(progress['last_reported_position'], 1)
            self.assertEqual(progress['last_attempted_case'], case_grid('generator')[1])

    def test_explicit_cuda_unavailable_is_distinct_from_cpu_smoke(self):
        from lsvd.pre_rental_cases import fp32_cpu_cuda

        with patch('torch.cuda.is_available', return_value=False):
            result = fp32_cpu_cuda(SimpleNamespace(device='cuda'))
        self.assertEqual(result['status'], 'cuda_unavailable')

    def test_synthetic_provenance_uses_actual_input(self):
        from lsvd.pre_rental_cases import record

        args = SimpleNamespace(synthetic=True, dataset=Path('unused-fixture-path'),
                               device='cpu', local_smoke=True)
        synthetic = record(args, 'mlp_causal')['protocol']['data_sha']
        args.dataset = None
        self.assertEqual(synthetic, record(args, 'mlp_causal')['protocol']['data_sha'])

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
            self.assertEqual(summary['fp32_cpu_cuda']['status'], 'skipped_cpu')
            self.assertEqual(summary['evaluation_model_source'], 'seeded_causal_initialization')
            self.assertEqual(summary['stages'][4]['result']['protocol']['id_min_count'], 20)
            self.assertEqual(summary['auto_stop'], False)
            self.assertIn('先拉回结果，再由Eddy销毁', result.stdout)
            for stage in summary['stages'][1:3]:
                self.assertEqual(stage['case_progress']['last_reported_position'], 2)
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

    def test_diagnostic_timeout_is_retained_and_grid_still_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = SimpleNamespace(output=output, synthetic=True, dataset=None,
                device='cuda', threads=1, local_smoke=False, total_seconds=30,
                warmup=0, steps=1, generator_vocab=3074, decoder_positions=3,
                download_url='')
            case = case_grid('two_tower')[0]
            with patch('lsvd.pre_rental_probe.case_grid', return_value=[case]), \
                 patch('lsvd.pre_rental_probe.run_child', side_effect=[
                     dict(status='timeout', returncode=-9, seconds=2.0),
                     dict(status='complete', returncode=0, seconds=3.0),
                 ]):
                result = run_cases(args, 'two_tower')
            self.assertEqual(result['cases'][0]['status'], 'complete')
            from lsvd.pre_rental_probe import fp32_summary
            diagnostic = fp32_summary(output, local_smoke=False)
            self.assertEqual(diagnostic['status'], 'timeout')
            self.assertEqual(diagnostic['severity'], 'red')


    def test_only_completed_primary_causal_case_supplies_evaluation_weights(self):
        from lsvd.pre_rental_deploy import sha256_file
        from lsvd.pre_rental_cases import evaluation_reference
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder)
            (stage / 'reference-model.pt').touch()
            case = stage / 'mlp_causal-bf16-20-cuda'
            case.mkdir()
            weights = case / 'reference-model.pt'
            weights.touch()
            identity = dict(source_sha='source', data_sha='dataset')
            self.assertIsNone(evaluation_reference(stage, identity))
            source = dict(identity, precision='bf16', model='mlp_causal', id_min_count=20,
                          batch_size=1024)
            receipt = dict(status='complete', protocol=source, data_placement='cuda',
                           reference_sha256=sha256_file(weights), reference_probe_updates=1)
            (case / 'result.json').write_text(json.dumps(receipt))
            self.assertEqual(evaluation_reference(stage, identity), (weights, 1))
            with self.assertRaisesRegex(ValueError, 'identity'):
                evaluation_reference(stage, dict(identity, data_sha='other'))
            weights.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'sha256'):
                evaluation_reference(stage, identity)


if __name__ == '__main__':
    unittest.main()
