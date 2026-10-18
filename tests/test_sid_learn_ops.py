import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile,json,time,copy
ROOT=Path(__file__).resolve().parents[1]

def module():
    spec=importlib.util.spec_from_file_location('structure_ops',ROOT/'scripts/sid_learn_ops.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class StructureOpsTests(unittest.TestCase):
    def test_context_peak_and_finite_deadline_are_unchanged(self):
        m=module();self.assertEqual(m.ENTRY,'vkgr.sid_learn_a1_run')
        memory={'pid':1,'context_process_mib':500.,'context_reserved_mib':0.,'samples':[{'own_process_mib':1100.,'reserved_mib':600.,'max_reserved_mib':1000.,'max_allocated_mib':900.}]}
        self.assertEqual(m.measured_vram(memory),1500.)
        self.assertEqual(m.declared_vram(1500),1792)
        self.assertFalse(m.valid_deadline(time.time()+49*3600))
        for value in (0,-1,True,float('nan')):
            with self.assertRaises(ValueError):m.declared_vram(value)

    def test_rejected_or_prior_run_never_republished(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);q=r/'queue'
            for name in ('pending','claimed','rejected'):(q/name).mkdir(parents=True)
            (q/'rejected/001-g-learn-a1-s42.json').write_text('{}')
            with self.assertRaises(FileExistsError):m.publish(q,r/'runs',{'run_id':'g-learn-a1-s42'})
            self.assertEqual(list((q/'pending').iterdir()),[])

    def test_unverified_fit_stops_before_model_context(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);proof=r/'verification.json';proof.write_text(json.dumps({'status':'complete'}))
            req={'fit_verify_receipt':{'path':str(proof),'sha256':m.sha(proof)}}
            with self.assertRaises(ValueError):m.check_fit_verification(req)

    def test_recipe_changes_only_sid_identity_and_binds_original_template_bytes(self):
        from vkgr import sid_learn_a1_run as new,coll_sid_a1_run as old
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d).resolve();cfg=r/'configs/sid_learn';cfg.mkdir(parents=True)
            base=old.default_config();oldcfg=r/'old.json';oldcfg.write_text(json.dumps(base))
            wanted={**new.default_config(),'sid_manifest_sha256':'d'*64};(cfg/'g-learn-a1-s42.json').write_text(json.dumps(wanted))
            t={'run_id':'g-coll-a1-c1-s42','seed':42,'cpu_cores':1,'ram_mib':18432,'vram_mib':6656,'wall_hours':24.,'disk_gib':4,'no_admission':False,
               'config':str(oldcfg),'inputs':['old_sid'],'env':['PYTHONPATH=old'],
               'argv':['python','-B','-m','vkgr.coll_sid_a1_run','train','--config','old','--config-sha','a'*64,'--output','oldout','--scratch','oldscratch',
                       '--sid-manifest','old_sid','--sid-sha','a'*64,'--input-manifest','old_input','--input-sha','b'*64,'--vectors','old_vectors','--vectors-sha','c'*64]}
            tp=r/'template.json';tp.write_text(json.dumps(t))
            req={'deployment':str(r),'runs':str(r/'runs'),'templates':{'42':str(tp)},'templates_sha256':{'42':m.sha(tp)},'python':'python',
                 'pinned_manifest':'/synthetic-pinned.json','data_root':str(r),'files':{'src/vkgr/sid_learn_a1_run.py':'a'*64},
                 'coll_refs':{k:{'path':'/'+k,'sha256':v*64} for k,v in [('sid_manifest','d'),('input_manifest','b'),('vectors','c')]}}
            job=m.derive_job(req,42,4096)
            self.assertEqual(job['run_id'],'g-learn-a1-s42')
            self.assertEqual(m.flag(job['argv'],'--sid-sha'),'d'*64)
            self.assertEqual(job['argv'].count('--vectors'),1)
            self.assertEqual(job['argv'].count('--input-manifest'),1)
            self.assertEqual(m.flag(job['argv'],'--vectors-sha'),'c'*64)
            t['seed']=2024;tp.write_text(json.dumps(t))
            with self.assertRaises(ValueError):m.derive_job(req,42,4096)

    def test_verified_fit_requires_matching_job_source_and_original_stdout(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d).resolve();runs=r/'runs';run=runs/'sid-learn-fit-verify-v1';out=run/'out';out.mkdir(parents=True);logs=run/'logs';logs.mkdir()
            config=r/'configs/sid_learn/fit-s42.json';config.parent.mkdir(parents=True);config.write_text('{}')
            (run/'run.json').write_text(json.dumps({'run_id':run.name,'git':{'commit':'a'*40},'config':{'sha256':m.sha(config)}}))
            raw={'schema':'learn-fit-verify-v1','status':'verified','mapping_replay':True,'state_replay':True,'pca_replay':True,'fallback_recomputed':True,'replay_steps':300000,'sid_version':'LEARN_LETTER_B256D3_S42','run_id':run.name,
                 'run_manifest_sha256':m.sha(run/'run.json'),'manifest_sha256':'d'*64,'vectors_file_sha256':'b'*64,'input_manifest_file_sha256':'c'*64,
                 'catalog_file_sha256':'e'*64,'fit_config_file_sha256':m.sha(config)}
            proof=out/'verification.json';log=logs/'stdout.log'
            req={'deployment':str(r),'runs':str(runs),'fit_source_commit':'a'*40,'catalog_ref':{'sha256':'e'*64},
                 'coll_refs':{k:{'sha256':v*64} for k,v in [('sid_manifest','d'),('vectors','b'),('input_manifest','c')]}}
            def save():
                proof.write_text(json.dumps(raw));log.write_text(json.dumps(raw)+'\n')
                req['fit_verify_receipt']={'path':str(proof),'sha256':m.sha(proof)};req['fit_verify_stdout']={'path':str(log),'sha256':m.sha(log)}
            from vkgr import sid_learn_a1_run as a1
            identity=a1.coll.producer_identity()
            # Exercise receipt gates with actual producer/runtime bytes. Deployment
            # Git identity is checked separately; this fixture owns only run/queue IO.
            req['queue']=str(r/'queue')
            for state in ('claimed','pending','rejected'):(r/'queue'/state).mkdir(parents=True)
            meta=json.loads((run/'run.json').read_text())
            meta.update(argv=['python','-B','-m','vkgr.sid_learn_run','verify'],cwd=str(ROOT))
            (run/'run.json').write_text(json.dumps(meta))
            raw.update(run_manifest_sha256=m.sha(run/'run.json'),producer=identity['files'],
                       producer_runtime=identity['runtime'],verifier_commit='a'*40)
            terminal=run/'terminal.json';terminal.write_text(json.dumps({'status':'succeeded','exit_code':0}))
            claimed=r/'queue/claimed/001-sid-learn-fit-verify-v1.json';claimed.write_text(json.dumps(meta))
            original=m.identity;m.identity=lambda request:a1
            self.addCleanup(setattr,m,'identity',original)
            save();m.check_fit_verification(req)
            good=copy.deepcopy(raw)
            for mode in ('producer','runtime','commit','terminal','claimed','rejected','argv','stdout','state_replay','pca_replay','fallback_recomputed','replay_steps'):
                raw.clear();raw.update(copy.deepcopy(good))
                if mode in ('state_replay','pca_replay','fallback_recomputed'):raw[mode]=False
                if mode=='replay_steps':raw[mode]=299999
                if mode=='producer':raw['producer']['unregistered.py']='f'*64
                if mode=='runtime':raw['producer_runtime']['python']='0.0'
                if mode=='commit':raw['verifier_commit']='f'*40
                if mode=='terminal':terminal.write_text(json.dumps({'status':'failed','exit_code':1}))
                if mode=='claimed':claimed.rename(claimed.with_suffix('.held'))
                reject=r/'queue/rejected/001-sid-learn-fit-verify-v1.json'
                if mode=='rejected':reject.write_text('{}')
                if mode=='argv':claimed.write_text(json.dumps({**meta,'argv':['wrong']}))
                save()
                if mode=='stdout':log.write_text('{}\n');req['fit_verify_stdout']['sha256']=m.sha(log)
                with self.subTest(mode=mode),self.assertRaises(ValueError):m.check_fit_verification(req)
                terminal.write_text(json.dumps({'status':'succeeded','exit_code':0}))
                if mode=='claimed':claimed.with_suffix('.held').rename(claimed)
                if mode=='rejected':reject.rename(reject.with_suffix('.held'))
                claimed.write_text(json.dumps(meta))
            raw.clear();raw.update(good)

            raw['manifest_sha256']='f'*64;save()
            with self.assertRaisesRegex(ValueError,'data/config'):m.check_fit_verification(req)
            raw['manifest_sha256']='d'*64;save();log.write_text('{}\n');req['fit_verify_stdout']['sha256']=m.sha(log)
            with self.assertRaisesRegex(ValueError,'absent'):m.check_fit_verification(req)

if __name__=='__main__':unittest.main()
