import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class FullEntry(unittest.TestCase):
    def test_dry_and_invalid_real_entry(self):
        run = subprocess.run([sys.executable, '-B', '-m', 'analysis.sgns_gpu_full', 'dry'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        value = json.loads(run.stdout)
        self.assertEqual(value['train_files'], 25)
        self.assertFalse(value['training_enabled'])
        bad = subprocess.run([sys.executable, '-B', '-m', 'analysis.sgns_gpu_full', 'bad'], capture_output=True)
        self.assertNotEqual(bad.returncode, 0)

    def test_inventory_train_only(self):
        from analysis import sgns_gpu_full as s
        records = s.fixed_records()
        self.assertEqual([r['path'] for r in records[:-1]], [f'subsamples/ur0.01/train/week_{i:02d}.parquet' for i in range(25)])
        self.assertEqual(records[-1]['path'], 'metadata/item_embeddings.npz')

    def test_four_gates_missing_is_fail_closed(self):
        from analysis import sgns_gpu_full as s
        with self.assertRaisesRegex(ValueError, 'four.*receipt'):
            s.require_acceptance(None)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'fake.json'
            p.write_text(json.dumps({'verified': True, 'subset_four_gate_acceptance': True}))
            with self.assertRaisesRegex(ValueError, 'four.*receipt'):
                s.require_acceptance(p)

    def test_full_verifier_only_identity_replacements(self):
        from vkgr.sgns_subset_verify import verify_c1_audit
        from vkgr.sgns_full_verify import verify_full_audit
        old = inspect.getsource(verify_c1_audit)
        expected = old.replace('verify_c1_audit(ids,histories,content,vectors,audit,*,engineering=False)',
                               'verify_full_audit(ids,histories,content,vectors,audit,*,binding,engineering=False)')
        expected = expected.replace("    if not engineering and len(ids)!=120415:raise ValueError('formal catalog count')",
                                    "    check_full_binding(ids, content, binding, engineering)")
        expected = expected.replace("    if not engineering and au.feature_digest(content)!=au.CONTENT_V1_FEATURE_SHA256:raise ValueError('CONTENT_V1 source mismatch')\n", '')
        self.assertEqual(inspect.getsource(verify_full_audit), expected)

    def test_full_verifier_cannot_enable_engineering_or_fake_binding(self):
        from vkgr.sgns_full_verify import check_full_binding
        for engineering in (False, True):
            with self.assertRaises(ValueError):
                check_full_binding([1,2,3], None, {}, engineering)

    def test_physical_order_and_count_guard(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from analysis import sgns_gpu_full as s
        from analysis.ur001_derive_cli import digest
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp).resolve()/'week_00.parquet'
            pq.write_table(pa.table({'user_id':pa.array([2,1,2,2],pa.uint32()), 'item_id':pa.array([7,3,9,7],pa.uint32())}),p,row_group_size=2)
            rec = {'size':p.stat().st_size,'sha256':digest(p),'rows':4,'row_groups':[2,2]}
            h, ids = s.load_sequences([p], [rec])
            self.assertEqual(h,{2:[7,9,7],1:[3]})
            self.assertEqual(ids,[3,7,9])
            with self.assertRaisesRegex(ValueError,'row|count'):
                s.load_sequences([p], [{**rec,'rows':5}])
            with self.assertRaisesRegex(ValueError,'row|count'):
                s.load_sequences([p], [{**rec,'row_groups':[4]}])


class FullIdentityBinding(unittest.TestCase):
    def test_public_source_baseline_pinned(self):
        from analysis import sgns_gpu_full as s
        self.assertEqual(s.fixed_dag()['base_source']['commit'],
                         'c9a60c5ce6eeb02db862455a5cafbbc861c4effb')

    def test_exact_source_delta_rejects_existing_bytes_and_unknown_paths(self):
        from analysis import sgns_gpu_full as s
        pin = s.fixed_dag()
        source = {'commit':'a'*40,'files':{**pin['base_source']['files'],**{p:'b'*64 for p in s.NEW_PATHS}}}
        s.check_source(source,'a'*40)
        for change in ('existing','unknown','missing','commit'):
            bad = {'commit':source['commit'],'files':dict(source['files'])}
            if change == 'existing': bad['files']['src/vkgr/sid_coll_sgns.py'] = 'c'*64
            elif change == 'unknown': bad['files']['src/vkgr/unfrozen.py'] = 'c'*64
            elif change == 'missing': bad['files'].pop('src/vkgr/sgns_full_verify.py')
            else: bad['commit'] = 'c'*40
            with self.assertRaisesRegex(ValueError,'source delta'):
                s.check_source(bad,'a'*40)

    def test_full_stage2_manifest_pin_before_payload(self):
        from analysis import sgns_gpu_full as s
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root/'UR001-manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'stage2 manifest'):
                s.stage2_binding(root,root/'run')

    def test_consumed_json_pin_rejects_corruption(self):
        from analysis import sgns_gpu_full as s
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp).resolve()/'manifest.json'
            p.write_text('{"training":false}')
            with self.assertRaisesRegex(ValueError,'consumed'):
                s.pinned_json(p,'0'*64)

    def test_incomplete_wrongseed_producer_is_refused(self):
        from analysis import sgns_gpu_full as s
        from analysis.ur001_derive_cli import encoded
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()/'fixture-run';(root/'logs').mkdir(parents=True)
            source={'commit':'a'*40};argv=['python','-B','-m','worker','preflight']
            run={'run_id':root.name,'kind':'ops','git':source,'argv':argv,'seed':43,'cwd':'/wrong','config':{},'inputs':[]}
            terminal={'run_id':root.name,'status':'succeeded','exit_code':0,'signal':15,'wall_cap_exceeded':True}
            receipt={'verified':True}
            for name,value in [('run.json',run),('terminal.json',terminal),('logs/stdout.log',receipt)]:
                (root/name).write_bytes(encoded(value))
            with self.assertRaises(ValueError):s.producer(root,argv,source,receipt)

    def test_job_and_stdout_receipt_binding(self):
        from analysis import sgns_gpu_full as s
        from analysis.ur001_derive_cli import encoded
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/'fixture-run'; (root/'logs').mkdir(parents=True)
            source={'commit':'a'*40}; argv=['python','-B','-m','analysis.sgns_gpu_full','preflight']
            expected={'schema':'vkgr-run-v1','run_id':root.name,'kind':'ops','argv':argv,
                      'git':{**source,'dirty':False,'dirty_files':[]},'seed':42,'cwd':str(s.ROOT),
                      'config':{'path':'/fixture/config.json','sha256':'b'*64},
                      'inputs':[{'path':'/fixture/uv.lock','sha256':'c'*64,'bytes':12}],
                      'env':{'OMP_NUM_THREADS':'1'},'resources':{'cpu_cores':1},
                      'ops_source':{'commit':'d'*40,'sha256':'e'*64},'overlay':None,
                      'pinned_files':{},'stop_mode':'signal'}
            terminal={'schema':'vkgr-terminal-v1','run_id':root.name,'status':'succeeded',
                      'exit_code':0,'signal':None,'wall_cap_exceeded':False,
                      'started_at':'fixture-start','ended_at':'fixture-end','wall_seconds':1.0}
            receipt={'verified':True,'engineering':False}
            (root/'run.json').write_bytes(encoded(expected))
            (root/'terminal.json').write_bytes(encoded(terminal))
            (root/'logs/stdout.log').write_bytes(encoded(receipt))
            proof=s.producer(root,argv,source,receipt,expected_job=expected)
            self.assertEqual(set(proof),{'run.json','terminal.json','logs/stdout.log'})
            mutations=[('run.json',{**expected,'seed':43}),('run.json',{**expected,'cwd':'/wrong'}),
                       ('run.json',{**expected,'config':None}),('run.json',{**expected,'inputs':[]}),
                       ('run.json',{**expected,'env':{}}),('run.json',{**expected,'resources':{}}),
                       ('run.json',{**expected,'ops_source':{}}),
                       ('terminal.json',{**terminal,'signal':15}),
                       ('terminal.json',{k:v for k,v in terminal.items() if k!='signal'}),
                       ('terminal.json',{**terminal,'wall_cap_exceeded':True}),
                       ('terminal.json',{**terminal,'run_id':'other'}),
                       ('logs/stdout.log',{'verified':False})]
            for target,bad in mutations:
                with self.subTest(target=target,bad=bad):
                    path=root/target; old=path.read_bytes(); path.write_bytes(encoded(bad))
                    with self.assertRaisesRegex(ValueError,'producer binding'):
                        s.producer(root,argv,source,receipt,expected_job=expected)
                    path.write_bytes(old)
            # Replace stdout during parsing; a proof may only hash its verified snapshot.
            original=s.receipt_from_bytes
            def replace_after_parse(raw):
                value=original(raw)
                (root/'logs/stdout.log').write_bytes(encoded({'verified':False}))
                return value
            with patch.object(s,'receipt_from_bytes',side_effect=replace_after_parse):
                with self.assertRaisesRegex(ValueError,'snapshot|producer binding'):
                    s.producer(root,argv,source,receipt,expected_job=expected)


class FullFitGuards(unittest.TestCase):
    def _fit_to_guard(self, *, rss=1024**3, disk_used=500):
        # Only the environment and earlier receipt/data readers are synthetic.
        # fit(), full_input_guards() and the registered guard remain real.
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import patch
        from analysis import sgns_gpu_full as s
        from vkgr import throughput_config as controls
        from vkgr import sid_coll_sgns
        original_exists = Path.exists
        original_read_text = Path.read_text
        def exists(path):
            return True if path == Path('/proc/meminfo') else original_exists(path)
        def read_text(path, *args, **kwargs):
            if path == Path('/proc/meminfo'):
                return 'MemTotal: 1073741824 kB\n'
            return original_read_text(path, *args, **kwargs)
        class BeforeTraining(RuntimeError): pass
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp).resolve()
            output = root/'never-written'
            args = SimpleNamespace(acceptance=root/'acceptance', acceptance_sha256='a'*64,
                acceptance_scratch=root/'scratch', deployment_sha='b'*40, lock=root/'lock',
                bundle=root/'bundle', bundle_sha='c'*64, predecessor_run=root/'prepare-run',
                output=output)
            for name, value in (('require_acceptance', {}), ('source_identity', {}),
                    ('check_source', {}), ('runtime', {}),
                    ('read_bundle', (([], {}, []), {'prepare_argv':[]})),
                    ('producer', {}), ('full_job_contract', {})):
                stack.enter_context(patch.object(s, name, return_value=value))
            stack.enter_context(patch.object(Path, 'exists', exists))
            stack.enter_context(patch.object(Path, 'read_text', read_text))
            stack.enter_context(patch('vkgr.throughput_cgroups.effective_limits',
                                      return_value={'memory_bytes':1024**4}))
            stack.enter_context(patch.object(controls, 'rss_bytes', return_value=rss))
            stack.enter_context(patch.object(controls.shutil, 'disk_usage',
                return_value=SimpleNamespace(total=1000, used=disk_used, free=1000-disk_used)))
            stack.enter_context(patch.object(sid_coll_sgns, 'fit_sgns',
                side_effect=BeforeTraining('guard passed; no scientific function executed')))
            try:
                s.fit(args)
            except BeforeTraining:
                self.assertFalse(output.exists())
                return 'before-training'
            finally:
                self.assertFalse(output.exists())

    def test_loaded_full_fit_has_room_for_nonzero_rss_and_96gib_projection(self):
        self.assertEqual(self._fit_to_guard(), 'before-training')

    def test_full_fit_retains_registered_rss_stop_before_training(self):
        with self.assertRaisesRegex(MemoryError, 'registered RSS limit exceeded'):
            self._fit_to_guard(rss=200*1024**3)

    def test_full_fit_retains_registered_disk_stop_before_training(self):
        with self.assertRaisesRegex(OSError, 'registered disk headroom exceeded'):
            self._fit_to_guard(disk_used=801)

    def test_full_fit_retains_192gib_engineering_cap_before_training(self):
        with self.assertRaisesRegex(MemoryError, 'engineering memory cap exceeded'):
            self._fit_to_guard(rss=100*1024**3)


if __name__ == '__main__': unittest.main()
