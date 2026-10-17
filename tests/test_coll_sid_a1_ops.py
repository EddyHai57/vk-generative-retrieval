"""Synthetic lifecycle, source/input guards and process-local VRAM tests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module():
    spec=importlib.util.spec_from_file_location('coll_a1_ops',ROOT/'scripts/coll_sid_a1_ops.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Ops(unittest.TestCase):
    def test_ops_exists(self):
        self.assertTrue((ROOT/'scripts/coll_sid_a1_ops.py').exists(),'new bounded driver missing')
    def test_context_inclusive_vram(self):
        m=module();mem={'pid':1,'context_process_mib':500.,'context_reserved_mib':0.,'samples':[
            {'own_process_mib':1100.,'reserved_mib':600.,'max_reserved_mib':1000.,'max_allocated_mib':900.}]}
        self.assertEqual(m.measured_vram(mem),1500.)
        self.assertEqual(m.declared_vram(1500),1792)
        for x in (0,-1,float('nan'),True):
            with self.assertRaises(ValueError):m.declared_vram(x)
    def test_rejected_run_never_republished(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);q=r/'queue'
            for f in ('pending','claimed','rejected'):(q/f).mkdir(parents=True)
            (q/'rejected/000-r.json').write_text('{}')
            with self.assertRaises(FileExistsError):m.publish(q,r/'runs',{'run_id':'r'})
            self.assertEqual(list((q/'pending').iterdir()),[])
    def test_verify_derivation_owns_new_manifest(self):
        m=module();j={'run_id':'g-coll-a1-c1-s42','inputs':[],'argv':['python','-m',m.ENTRY,'train','--config','c','--config-sha','a'*64,'--output','o','--scratch','s']}
        v=m.derive_verify(j,'b'*64,Path('/synthetic-runs'))
        self.assertEqual(m.flag(v['argv'],'--manifest-sha'),'b'*64)
        self.assertEqual(v['argv'][3],'verify');self.assertNotIn('--config',v['argv'])
        self.assertEqual(v['run_id'],'g-coll-a1-c1-s42-verify')
    def test_finite_deadline(self):
        import time
        m=module();self.assertTrue(m.valid_deadline(time.time()+300));self.assertFalse(m.valid_deadline(time.time()+49*3600))

if __name__=='__main__':unittest.main()

class Admission(unittest.TestCase):
    def test_fit_receipt_binds_raw_stdout_and_frozen_sid(self):
        import json
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);raw={'status':'verified','manifest_sha256':'s'*64,'mapping_replay':True,'sid_version':'COLL_C1_B256D3_S42'}
            receipt={'schema':'coll-stage-receipt-v1','stage':'sid_verify','variant':'C1','status':'verified','raw_proof':raw,
                     'binding':{'data':{'sid_manifest':'s'*64,'input_manifest':'i'*64,'vectors':'v'*64}}}
            path=r/'receipt.json';path.write_text(json.dumps(receipt));ref={'path':str(path),'sha256':m.sha(path)}
            log=r/'stdout.log';log.write_text(json.dumps(raw)+'\n'+json.dumps({'status':'verified','receipt':ref})+'\n')
            req={'fit_verify_receipt':ref,'fit_verify_stdout':{'path':str(log),'sha256':m.sha(log)},
                 'coll_refs':{key:{'sha256':value*64} for key,value in [('sid_manifest','s'),('input_manifest','i'),('vectors','v')]}}
            m.check_fit_verification(req)
            raw['manifest_sha256']='x'*64;receipt['raw_proof']=raw;path.write_text(json.dumps(receipt));req['fit_verify_receipt']['sha256']=m.sha(path)
            with self.assertRaises(ValueError):m.check_fit_verification(req)
    def test_identity_in_other_queue_subdirectory_blocks_publish(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);q=r/'queue';(q/'archive').mkdir(parents=True)
            (q/'archive/000-r.json').write_text('{}')
            self.assertTrue(m.exists(q,r/'runs','r'))

class Derivation(unittest.TestCase):
    def test_identity_only_science_diff_and_input_pins(self):
        import json
        m=module()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            r=Path(d)
            for seed in (42,2024,2025):
                old=json.loads((ROOT/f'configs/gtok01b/g-gtok-a1-t3-b256d3-s{seed}.json').read_text())
                cp=r/f'old{seed}.json';cp.write_text(json.dumps(old))
                t={'cpu_cores':1,'ram_mib':18432,'vram_mib':6656,'wall_hours':24.,'disk_gib':4,'no_admission':False,'config':str(cp),
                   'inputs':['data','old-sid'],'env':['VKGR_DECODE=serial','PYTHONPATH=old'],
                   'argv':['python','-m','vkgr.generative_g6_run','train','--config','old','--config-sha','0'*64,
                           '--prepared','same-prepared','--prepared-sha','a'*64,'--history-manifest','same-history','--history-sha','b'*64,
                           '--sid-manifest','old-sid','--sid-sha','0'*64,'--content','same-content','--content-sha','c'*64,'--output','out','--scratch','scratch']}
                tp=r/f'job{seed}.json';tp.write_text(json.dumps(t))
                req={'deployment':str(ROOT),'runs':str(r/'runs'),'python':'python','data_root':str(r),'pinned_manifest':'private-pinned',
                     'templates':{str(seed):str(tp)},'coll_refs':{role:{'path':role+'-private','sha256':letter*64} for role,letter in
                     [('sid_manifest','s'),('input_manifest','i'),('vectors','v')]}}
                job=m.derive_job(req,seed,8192)
                self.assertEqual(job['run_id'],f'g-coll-a1-c1-s{seed}')
                self.assertEqual(job['vram_mib'],8192)
                self.assertEqual(m.flag(job['argv'],'--prepared'),'same-prepared')
                self.assertEqual(m.flag(job['argv'],'--history-sha'),'b'*64)
                self.assertEqual(m.flag(job['argv'],'--content-sha'),'c'*64)
                self.assertEqual(m.flag(job['argv'],'--sid-manifest'),'sid_manifest-private')
                self.assertEqual(m.flag(job['argv'],'--input-sha'),'i'*64)
                self.assertEqual(m.flag(job['argv'],'--vectors-sha'),'v'*64)
                self.assertNotIn('old-sid',job['inputs'])
                self.assertIn('VKGR_DECODE_BATCH=1',job['env'])
                self.assertEqual(m.derive_job(req,seed)['vram_mib'],None)

class VerifyNoRetry(unittest.TestCase):
    def test_failed_verify_publication_intent_is_never_attempted_again(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        m=module()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);q=root/'queue';runs=root/'runs';ops=root/'ops';ops.mkdir()
            for name in ('pending','claimed','rejected'):(q/name).mkdir(parents=True)
            rid='g-coll-a1-c1-s42';out=runs/rid/'out';out.mkdir(parents=True)
            cp=root/'config.json';cp.write_text('{}');(runs/rid/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
            manifest={'config':{},'binding':{}};(out/'manifest.json').write_text(json.dumps(manifest))
            job={'run_id':rid,'config':str(cp),'inputs':[],'argv':['python','-m',m.ENTRY,'train','--config',str(cp),'--config-sha','a'*64,'--output','o','--scratch','s']}
            (q/'claimed'/('001-'+rid+'.json')).write_text(json.dumps(job))
            req={'queue':str(q),'runs':str(runs),'ops':str(ops)}
            entry={'job':job,'expected_binding_sha256':m.digest({})}
            with patch.object(m,'identity',return_value=SimpleNamespace(_digest=m.digest)),patch.object(m,'launcher_prepare',side_effect=OSError('transport failure')) as prepare:
                with self.assertRaises(OSError):m.observe(req,entry)
                with self.assertRaises(FileExistsError):m.observe(req,entry)
                self.assertEqual(prepare.call_count,1)
