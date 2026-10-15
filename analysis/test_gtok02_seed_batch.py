import importlib.util
from pathlib import Path
import unittest


class BatchChecks(unittest.TestCase):
    def module(self):
        p = Path(__file__).with_name('gtok02_seed_batch.py')
        self.assertTrue(p.exists(), 'seed submission/driver implementation is missing')
        spec = importlib.util.spec_from_file_location('seed_batch', p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m

    def test_seed_only_contract(self):
        m = self.module()
        m.check_seed_config({'seed': 42, 'lr': .1}, {'seed': 2024, 'lr': .1}, 2024)
        with self.assertRaises(ValueError): m.check_seed_config({'seed': 42, 'lr': .1}, {'seed': 2024, 'lr': .2}, 2024)

    def test_rank_top2_and_inclusive_ten_hits(self):
        m = self.module()
        rows = [{'variant': str(i), 'hits': h, 'ndcg': .1, 'order': i} for i, h in enumerate([300, 285, 290, 289])]
        self.assertEqual([x['variant'] for x in m.pick_p1(rows)], ['0', '2'])

    def test_train_derivation_changes_only_seed_config_identity(self):
        m = self.module()
        job = {'run_id': 'g-tok2-b256d2-s42', 'config': '/workspace/code/configs/x-s42.json', 'seed': 42,
               'argv': ['python', '-m', 'runner', 'train', '--config', 'configs/x-s42.json', '--config-sha', 'old', '--output', '/runs/g-tok2-b256d2-s42/out', '--scratch', '/runs/g-tok2-b256d2-s42/scratch'], 'cpu_cores': 1, 'vram_mib': 3072}
        v = m.derive_train(job, 2024, 'new')
        self.assertEqual(v['seed'], 2024)
        self.assertEqual(v['run_id'], 'g-tok2-b256d2-s2024')
        self.assertEqual(v['vram_mib'], 3072)
        self.assertIn('configs/x-s2024.json', v['argv'])

    def test_publish_never_overwrites_or_duplicates(self):
        m = self.module()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            q = Path(td); [ (q/x).mkdir() for x in ('pending','claimed','rejected') ]
            runs=q/'runs';runs.mkdir()
            m.publish(q, runs, {'run_id': 'g-tok2-b32d3-s2024'}, '0001-g-tok2-b32d3-s2024.json')
            with self.assertRaises(FileExistsError): m.publish(q, runs, {'run_id': 'g-tok2-b32d3-s2024'}, '0002-g-tok2-b32d3-s2024.json')

    def test_exit_zero_is_not_science_verification(self):
        m=self.module()
        self.assertFalse(m.is_verified({'status':'complete'}, 'a'*64))
        self.assertFalse(m.is_verified({'status':'verified','manifest_sha256':'b'*64}, 'a'*64))
        self.assertTrue(m.is_verified({'status':'verified','manifest_sha256':'a'*64}, 'a'*64))

    def test_concurrent_publish_has_one_logical_run(self):
        m=self.module()
        import tempfile
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);[(q/x).mkdir() for x in ('pending','claimed','rejected')];runs=q/'runs';runs.mkdir()
            def attempt(i):
                try:m.publish(q,runs,{'run_id':'g-tok2-b32d3-s2024'},f'{i:04d}-g-tok2-b32d3-s2024.json');return True
                except FileExistsError:return False
            with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(attempt,range(16)))
            self.assertEqual(sum(results),1)
            self.assertEqual(len(list((q/'pending').glob('*.json'))),1)

    def test_driver_may_watch_pending_but_not_claim_success(self):
        m=self.module()
        self.assertTrue(hasattr(m,'watchable_queue'),'pending acceptance-watch support is missing')
        self.assertTrue(m.watchable_queue({'pending':['job.json'],'claimed':[],'rejected':[]}))
        self.assertFalse(m.watchable_queue({'pending':[],'claimed':[],'rejected':[]}))
        self.assertFalse(m.watchable_queue({'pending':[],'claimed':[],'rejected':['job.json']}))


if __name__ == '__main__': unittest.main()
