"""Transport guards, frozen science routing and engineering receipt identity."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
def bridge():
    path=ROOT/'analysis/gtok02a1_launch_bridge.py'
    assert path.exists(), 'clean launch bridge is missing'
    spec=importlib.util.spec_from_file_location('launch_bridge_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sample():
    return dict(run_id='train42',seed=42,cwd='/old',config='/old/config.json',pinned_manifest='/old/pinned.json',pinned_files=['a=x'],inputs=['/data'],env=['OMP_NUM_THREADS=1','PYTHONPATH=/wrong','VKGR_PINNED_MANIFEST=/old/pinned.json'],argv=['python','-B','-m','vkgr.generative_g6_tok2_run','train','--config','configs/c.json'],kind='real',no_admission=False)
class BridgeTests(unittest.TestCase):
    def test_oversized_record_rejected_with_16k_headroom(self):
        b=bridge()
        for size in (66316,65536,49153):
            with self.assertRaisesRegex(ValueError,'metadata'):b.metadata_guard({}, {'dirty':'x'*size})
        assert b.metadata_guard({}, {'clean':True})=={'clean':True}
    def test_route_keeps_science_pins_env_and_absolute_config(self):
        b=bridge();old=sample();d=b.route(old)
        assert old==sample()
        assert d['cwd']=='/workspace/vkgr-gtok02-a1-launch-v2/code'
        assert d['pinned_manifest'] is None and d['pinned_files']==[]
        assert d['kind']=='real' and d['no_admission'] is False
        assert d['argv'][1:6]==['-B',str(b.CODE/'analysis/gtok02a1_job_entry.py'),'-B','-m','vkgr.generative_g6_tok2_run']
        assert d['argv'][d['argv'].index('--config')+1]=='/old/config.json'
        assert 'PYTHONPATH=/workspace/vkgr-gtok02-a1-v1/src:/workspace/vkgr-gtok02-a1-v1/scripts' in d['env']
        assert 'VKGR_PINNED_MANIFEST=/old/pinned.json' in d['env']
        assert str(b.OPS/'pinned.json') in d['inputs'] and str(b.OPS/'deployment.json') in d['inputs']
    def test_claim_checks_all_expected_fields(self):
        b=bridge();expected={'argv':['probe'],'env':['pin'],'kind':'real','seed':42}
        b.check_claim({**expected,'scheduler_extra':1},expected)
        for key in expected:
            drift=copy.deepcopy(expected);drift[key]=None
            with self.assertRaisesRegex(ValueError,'engineering job identity'):b.check_claim(drift,expected)
    def test_probe_has_fresh_identity_and_guarded_entry(self):
        b=bridge();ops=SimpleNamespace(derive_job=lambda seed:b.route(sample()),PY='python',R=Path('/runsroot'))
        d=b.probe_job(ops)
        assert d['run_id']=='g-gtok02-a1-b1024d3-cuda-v2'
        assert d['argv']==['python','-B',str(b.CODE/'analysis/gtok02a1_job_entry.py'),'--probe','/runsroot/runs/g-gtok02-a1-b1024d3-cuda-v2/out']
        assert (d['wall_hours'],d['vram_mib'],d['disk_gib'])==(1.,8192,2)
    def test_entry_guards_before_probe_and_module(self):
        b=bridge();spec=importlib.util.spec_from_file_location('entry',ROOT/'analysis/gtok02a1_job_entry.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry);events=[]
        ops=SimpleNamespace(DEP=Path('/old'),ENTRY='vkgr.generative_g6_tok2_run')
        entry.load_bridge=lambda:SimpleNamespace(guard=lambda:events.append('guard') or ops)
        self.addCleanup(patch.stopall);patch.object(entry.runpy,'run_path',lambda *a,**kw:events.append(('probe',a))).start()
        entry.main(['--probe','/out']);assert events==['guard',('probe',('/old/scripts/gtok02a1_cuda_check.py',))]
        patch.object(entry.runpy,'run_module',lambda *a,**kw:events.append(('module',a))).start()
        entry.main(['-B','-m','vkgr.generative_g6_tok2_run','verify'])
        assert events[2:]==['guard',('module',('vkgr.generative_g6_tok2_run',))]
    def test_driver_uses_clean_control_script(self):
        b=bridge();assert b.driver_argv(SimpleNamespace(PY='python'))==['python','-B',str(b.CODE/'analysis/gtok02a1_launch_bridge.py'),'--driver']
    def test_science_drift_blocks_clean_guard_and_execution(self):
        b=bridge();events=[]
        ops=SimpleNamespace(identity=lambda:events.append('science') or {'model_commit':'wrong','pinned_sha256':b.PIN_SHA})
        with patch.object(b,'load_ops',return_value=ops),patch.object(b,'clean_identity',side_effect=lambda _:events.append('clean')):
            with self.assertRaisesRegex(ValueError,'science identity'):b.guard()
        assert events==['science']
        ops.identity=lambda:events.append('science') or {'model_commit':b.MODEL_SHA,'pinned_sha256':b.PIN_SHA}
        with patch.object(b,'load_ops',return_value=ops),patch.object(b,'clean_identity',side_effect=lambda _:events.append('clean')):
            assert b.guard() is ops
        assert events==['science','science','clean']
    def test_metadata_guard_wraps_actual_prepare_result(self):
        b=bridge();ops=SimpleNamespace(derive_job=lambda *a,**kw:sample(),prepare_job=lambda job:{'dirty':'x'*66316})
        b.patch(ops)
        with self.assertRaisesRegex(ValueError,'metadata'):ops.prepare_job(ops.derive_job(42))
    def test_preflight_rejects_binding_drift_and_never_writes(self):
        b=bridge();original={'status':'preflight_ok','model_commit':'model','pinned_sha256':'pin','entries':{'train42':{'binding':'frozen'}}}
        changed=copy.deepcopy(original);changed['entries']['train42']['binding']='drift'
        ops=SimpleNamespace(load=lambda path:copy.deepcopy(original),preflight=lambda:changed)
        with self.assertRaisesRegex(ValueError,'frozen receipt'):b.readonly_preflight(ops)
        ops.preflight=lambda:{**original,'read_at_utc':'later','scratch':'new'}
        assert b.readonly_preflight(ops)['entries']==original['entries']
    def test_unchanged_source_sha_required_before_ops_exec(self):
        import tempfile
        b=bridge()
        with tempfile.TemporaryDirectory() as directory:
            dep=Path(directory);(dep/'scripts').mkdir();(dep/'scripts/gtok02a1_ops.py').write_text("raise RuntimeError('must not execute')")
            with patch.object(b,'DEP',dep):
                with self.assertRaisesRegex(ValueError,'ops source changed'):b.load_ops()
