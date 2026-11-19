import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess


class QueueTests(unittest.TestCase):
    def test_resume_keeps_budget_unless_explicitly_increased(self):
        from lsvd.pre_rental_queue import Progress

        with tempfile.TemporaryDirectory() as folder:
            queue = Path(folder)
            (queue / 'status.json').write_text(json.dumps(
                dict(elapsed_seconds=3600, gpu_hours_limit=1)))
            self.assertEqual(Progress(queue, 'cuda', None).stop_reason(), 'gpu_hours')
            self.assertIsNone(Progress(queue, 'cuda', 2).stop_reason())

    def test_latest_checkpoint_follows_registry_across_attempts(self):
        from lsvd.pre_rental_queue import latest_checkpoint

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            older = run / 'attempt-000001/checkpoints'
            older.mkdir(parents=True)
            (older / 'best.pt').touch()
            (older / 'latest.pt').touch()
            (older / 'checkpoints.json').write_text(json.dumps(
                dict(latest=['latest.pt'], best='best.pt', counter=2)))
            (run / 'attempt-000002/checkpoints').mkdir(parents=True)
            self.assertEqual(latest_checkpoint(run), older / 'latest.pt')

    def test_status_cli_does_not_report_killed_driver_as_running(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = Path(folder)
            (queue / 'queue.lock').touch()
            status = dict(status='running', current_run='example-mean', epoch=1,
                          step=3, elapsed_seconds=5.0, remaining_training_seconds=2.0)
            (queue / 'status.json').write_text(json.dumps(status))
            result = subprocess.run([sys.executable, '-m', 'lsvd.pre_rental_queue',
                                     'status', '--queue', str(queue)],
                                    capture_output=True, text=True, check=True)
            recovered = json.loads(result.stdout)
            self.assertEqual(recovered['status'], 'interrupted')
            self.assertEqual(recovered['step'], 3)
