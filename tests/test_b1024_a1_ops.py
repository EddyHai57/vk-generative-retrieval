"""Short ops guards: no real data, server or scientific run needed."""
import copy
import hashlib
import subprocess
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gtok02a1_ops', ROOT/'scripts/gtok02a1_ops.py')


def module():
    if not (ROOT/'scripts/gtok02a1_ops.py').exists():
        return None
    m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
    return m


def template(seed=42):
    return {'run_id':f'g-gtok01b-t3-b256d3-s{seed}', 'seed':seed, 'cwd':'/old', 'config':'/old/config.json',
        'cpu_cores':1, 'ram_mib':18432, 'vram_mib':6656, 'wall_hours':24., 'disk_gib':4,
        'no_admission':False, 'kind':'train', 'stop_mode':'term', 'schema':'fixture', 'pinned_manifest':'/old/pin',
        'env':['OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src','VKGR_DECODE=batched','VKGR_DECODE_BATCH=1'],
        'inputs':['/prepared','/history','/old-sid','/content'],
        'argv':['python','-B','-m','vkgr.generative_g6_run','train', '--prepared','/prepared','--prepared-sha','p'*64,
            '--history-manifest','/history','--history-sha','h'*64,'--sid-manifest','/old-sid','--sid-sha','s'*64,
            '--content','/content','--content-sha','c'*64,'--config','old.json','--config-sha','f'*64,
            '--output','/old/out','--scratch','/old/scratch']}


class OpsTests(unittest.TestCase):
    def setUp(self):
        self.m = module()
        self.assertIsNotNone(self.m, 'versioned ops implementation is missing')

    def test_preflight_binding_model_uses_producer_cuda_initialization(self):
        from unittest.mock import Mock
        g6=Mock();context=Mock();config={'seed':42}
        self.m.binding_model(g6,context,config,{'device':'cuda:0'})
        self.assertEqual(g6.make_model_arch_d.call_args.kwargs['device'],'cuda:0')

    def test_train_derivation_changes_sid_entry_pin_but_preserves_a1_resources(self):
        m=self.m; old=template(); original=copy.deepcopy(old)
        with patch.object(m,'one',return_value=old), patch.object(m,'sha',return_value='f'*64):
            d=m.derive_job(42,9472)
        self.assertEqual(old,original)
        self.assertEqual(d['run_id'],'g-gtok02-a1-b1024d3-s42')
        self.assertEqual(d['argv'][3:5],['vkgr.generative_g6_tok2_run','train'])
        self.assertEqual(m.flag(d['argv'],'--sid-sha'),m.SID_SHA)
        self.assertEqual(d['pinned_manifest'],str(m.OPS/'pinned.json'))
        self.assertIn('VKGR_PINNED_MANIFEST='+d['pinned_manifest'],d['env'])
        self.assertEqual([d[k] for k in ('cpu_cores','ram_mib','vram_mib','wall_hours','disk_gib')],[1,18432,9472,24.,4])
        self.assertNotIn('/old-sid',d['inputs'])
        with self.assertRaises(ValueError):m.derive_job(123)

    def test_verify_derivation_has_complete_g6_argv_and_keeps_resources(self):
        m=self.m; d=template(); d['run_id']=m.RIDS[0]
        v=m.derive_verify(d,'a'*64)
        self.assertIsNone(v['config']); self.assertEqual(v['wall_hours'],2); self.assertEqual(v['disk_gib'],1)
        self.assertEqual(v['argv'][4],'verify')
        self.assertNotIn('--config',v['argv']);self.assertNotIn('--config-sha',v['argv'])
        self.assertEqual(m.flag(v['argv'],'--manifest-sha'),'a'*64)
        self.assertEqual(m.flag(v['argv'],'--output'),str(m.R/'runs'/(d['run_id']+'-verify')/'out'))
        self.assertEqual(m.flag(v['argv'],'--scratch'),str(m.R/'runs'/(d['run_id']+'-verify')/'scratch'))
        self.assertFalse(any(x.startswith('VKGR_DECODE') for x in v['env']))
        self.assertIn('VKGR_EVAL_WORKERS=1',v['env'])
        for key in ('cpu_cores','ram_mib','vram_mib','cwd','pinned_manifest'):self.assertEqual(v[key],d[key])

    def test_engineering_memory_requires_real_shape_proof_and_rounds_up(self):
        m=self.m
        proof={'status':'cuda_engineering_verified','measured_mib':7000,'declared_vram_mib':8192,
            'real_shape':m.REAL_SHAPE,'synthetic_grad_steps':2,'reload_decode_equal':True}
        self.assertEqual(m.probe_vram(proof),8192)
        for changed in ({'declared_vram_mib':6656},{'real_shape':{}},{'reload_decode_equal':False},{'synthetic_grad_steps':0},{'measured_mib':float('nan')}):
            with self.assertRaises(ValueError):m.probe_vram({**proof,**changed})

    def test_exclusive_publication_never_overwrites_or_retries(self):
        m=self.m
        with tempfile.TemporaryDirectory() as d:
            q=Path(d)/'queue';runs=Path(d)/'runs';runs.mkdir()
            for n in ('pending','claimed','rejected'):(q/n).mkdir(parents=True)
            job=template();name=m.publish(q,runs,job)
            self.assertEqual(json.loads((q/'pending'/name).read_text()),job)
            with self.assertRaises(FileExistsError):m.publish(q,runs,job)
            (q/'rejected'/('1-other.json')).write_text('{}')
            (q/'rejected'/('1-other.reason')).write_text('reason')
            self.assertIn('reason',str(m.where(q,'other')['reasons']))

    def test_binding_digest_matches_project_canonical_newline(self):
        value={'source':{'commit':'fixture'},'test_gate':'closed'}
        expected=hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
        self.assertEqual(self.m.digest(value),expected)

    def test_partial_rejected_does_not_block_other_seed_verify_publication(self):
        m=self.m; entries=[{'job':template(s),'expected':{'source':{}}} for s in (42,2024)]
        def snap(e):
            rejected=e['job']['seed']==42
            return {'run_id':e['job']['run_id'],'train_status':'succeeded','verify_status':None,
                'rejected':rejected,'verified':False,'manifest_sha256':'a'*64,'exceptions':[]}
        with patch.object(m,'snapshot',side_effect=snap),patch.object(m,'exists',return_value=False),patch.object(m,'prepare_job') as prepared,patch.object(m,'publish') as published:
            states,done=m.poll(entries)
        self.assertFalse(done)
        self.assertEqual(published.call_count,1)
        self.assertTrue(published.call_args.args[2]['run_id'].endswith('s2024-verify'))
        self.assertEqual(prepared.call_count,1)

    def test_overlay_deploy_full_diff_data_pins_and_unchanged_base_head(self):
        m=self.m
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'repo';src.mkdir();old=root/'old';old.mkdir();dep=root/'deploy';ops=root/'ops';q=root/'queue';runs=root/'runs';runs.mkdir()
            for k in ('pending','claimed','rejected'):(q/k).mkdir(parents=True)
            def local(*a,text=True):
                return subprocess.run(['git','-C',str(src),*a],capture_output=True,text=text,check=True).stdout
            local('init','-q')
            (src/'docs').mkdir();(src/'docs/p3c_validation.md').write_text('git base')
            (src/'legacy.py').write_text('old model')
            local('add','.')
            local('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','base')
            base=local('rev-parse','HEAD').strip()
            (src/'new.py').write_text('new module');(src/'legacy.py').write_text('updated')
            local('add','.');local('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','target')
            target=local('rev-parse','HEAD').strip()
            (old/'docs').mkdir();(old/'docs/p3c_validation.md').write_text('frozen channel')
            (old/'provenance/g0-v1').mkdir(parents=True);(old/'provenance/g0-v1/proof.json').write_text('{}')
            oldpin=root/'oldpin.json';oldpin.write_text(json.dumps({'files':{'provenance/g0-v1/proof.json':m.sha(old/'provenance/g0-v1/proof.json')},'overrides':{'docs/p3c_validation.md':{'sha256':m.sha(old/'docs/p3c_validation.md')}}}))
            def remote_free(*a,text=True):
                if a[0]=='fetch':return ''
                if a==('rev-parse','origin/fixture'):return target+'\n'
                return local(*a,text=text)
            with patch.multiple(m,SRC=src,DEP=dep,OPS=ops,OLD=old,OLDPIN=oldpin,Q=q,R=root,BASE=base),patch.object(m,'git',side_effect=remote_free):
                result=m.deploy({'commit':target,'branch':'fixture'})
                self.assertEqual(result['model_commit'],target)
                self.assertEqual(subprocess.check_output(['git','-C',str(dep),'rev-parse','HEAD'],text=True).strip(),base)
                pin=json.loads((ops/'pinned.json').read_text())
                self.assertEqual(pin['target_commit'],target)
                self.assertIn('legacy.py',pin['files']);self.assertIn('new.py',pin['files'])
                self.assertIn('provenance/g0-v1/proof.json',pin['files'])
                self.assertEqual((dep/'docs/p3c_validation.md').read_text(),'frozen channel')
                (dep/'new.py').write_text('tampered')
                with self.assertRaises(ValueError):m.identity()
                with self.assertRaises(FileExistsError):m.deploy({'commit':target,'branch':'fixture'})

    def test_snapshot_only_releases_metrics_after_bound_raw_verified_receipt(self):
        m=self.m
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);q=root/'queue';rid=m.RIDS[0];config=root/'config.json';config.write_text('{}')
            job=template();job.update(run_id=rid,config=str(config));binding={'synthetic':False,'test_gate':'closed'}
            expected={'source':{'deployment_ref':{'target_commit':'a'*40}},'config_sha256':m.sha(config),'data_hashes':{},'expected_binding_sha256':m.digest(binding)}
            entry={'job':job,'expected':expected}
            td=root/'runs'/rid;vd=root/'runs'/(rid+'-verify');(td/'out').mkdir(parents=True);(vd/'logs').mkdir(parents=True)
            (td/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
            manifest={'binding':binding,'config':{},'selection':{'best_epoch':1,'epoch':5},'epochs':[{'epoch':1,'metrics':{'metrics':{'all_eligible':{'recall@50':0.1,'ndcg@50':0.05}}}}]}
            mp=td/'out/manifest.json';mp.write_text(json.dumps(manifest))
            with patch.multiple(m,R=root,Q=q):
                ver=m.derive_verify(job,m.sha(mp));jobs={rid:job,rid+'-verify':ver}
                with patch.object(m,'one',side_effect=lambda r:jobs[r]):
                    before=m.snapshot(entry);self.assertFalse(before['verified']);self.assertNotIn('metrics',before)
                    (vd/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
                    (vd/'logs/stdout.log').write_text('noise\n'+json.dumps({'status':'verified'})+'\n')
                    after=m.snapshot(entry);self.assertTrue(after['verified']);self.assertEqual(after['metrics']['recall@50'],0.1)
                    jobs[rid+'-verify']={**ver,'pinned_manifest':'wrong'}
                    rejected=m.snapshot(entry);self.assertFalse(rejected['verified']);self.assertNotIn('metrics',rejected)

    def test_expired_driver_writes_final_fallback_without_poll_or_submission(self):
        m=self.m
        with tempfile.TemporaryDirectory() as d:
            ops=Path(d);entries=[{'job':template(),'expected':{'source':{'commit':'fixture'}}}]
            (ops/'plan.json').write_text(json.dumps({'entries':entries,'deadline_epoch':1}))
            with patch.object(m,'OPS',ops),patch.object(m,'configure'),patch.object(m,'identity'),patch.object(m.time,'time',return_value=2),patch.object(m,'poll') as poll,patch.object(m,'publish') as publish:
                m.driver()
            self.assertFalse(poll.called);self.assertFalse(publish.called)
            final=json.loads((ops/'summary-final.json').read_text())
            self.assertEqual(final['status'],'ALERT');self.assertFalse(final['runs'][0]['verified'])
            self.assertTrue(final['runs'][0]['unaccepted']);self.assertNotIn('metrics',final['runs'][0])

    def test_terminal_summary_and_verified_receipt_gate(self):
        m=self.m
        self.assertFalse(m.accepts_verify('succeeded',False,[{'status':'verified'}]))
        self.assertFalse(m.accepts_verify('failed',True,[{'status':'verified'}]))
        self.assertTrue(m.accepts_verify('succeeded',True,[{'status':'verified'}]))
        states=[{'verified':False,'exceptions':[],'rejected':True},{'verified':False,'exceptions':[]}]
        self.assertEqual(m.summary_status(states,False),'ALERT')
        self.assertEqual(m.summary_status([{'verified':True}],True),'verified')
        self.assertTrue(m.can_start_observation({'pending':[],'claimed':[],'rejected':['x']},False))
        self.assertEqual(len(m.deadline_fallback([{'job':template()}])),1)


if __name__=='__main__':unittest.main()
