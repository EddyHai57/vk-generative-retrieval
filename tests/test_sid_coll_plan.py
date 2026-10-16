"""Frozen metadata-only pipeline assembly; never reads target data or checkpoints."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

def module():
    spec=importlib.util.spec_from_file_location('coll_plan',ROOT/'scripts/sid_coll_plan.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class PlanTests(unittest.TestCase):
    def fixture(self):
        import hashlib
        names=['scripts/sid_coll_job_entry.py']+[f'configs/sid_coll/g-coll-{v}-s{s}.json' for v in ('c1','c2') for s in (42,2024,2025)]+[f'configs/sid_coll/fit-coll-{v}.json' for v in ('c1','c2')]
        dep={'commit':'a'*40,'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}
        common={name:{'path':'/workspace/fixture/'+name+('.npy' if name=='content' else '.json'),'sha256':'b'*64} for name in ('catalog','content','history','g_prepared','slices','s0_result')}
        plan={'raw_root':'/workspace/raw','train_audit':{'path':'/workspace/fixture/train-audit.json','sha256':'c'*64},'catalog':common['catalog'],'history':common['history']}
        meta={'created_epoch':1000,'common':common,'deployment_ref':{'path':'/workspace/vkgr-sid-coll-ops-v1/deployment.json','sha256':'d'*64},
              'support_ref':{'path':'/workspace/vkgr-sid-coll-ops-v1/supported.npy','sha256':'e'*64},
              'manifest':{'path':'/workspace/vkgr-infra01-root/runs/p-seqm1-mean-s42/out/manifest.json','sha256':'f'*64},
              'lock':{'schema':'coll-c2-source-lock-v1','train_sequence_sha256':'PENDING_TRAIN_SEQUENCE'}}
        return dep,meta,plan

    def test_complete_bounded_forward_only_plan_and_exact_job_environment(self):
        m=module();dep,meta,s0=self.fixture();docs=m.build_documents(dep,meta,s0,root=ROOT)
        plan=docs['pipeline-plan.json'];self.assertEqual(set(plan['branches']),{'C1p','C2'})
        self.assertEqual(plan['deadline_epoch'],173800)
        self.assertEqual(len(plan['common_refs']),2)
        self.assertLess(len(m.wire_bytes(plan)),2*1024*1024)
        for branch in plan['branches'].values():
            self.assertEqual(set(branch['stages']),set(m.STAGES))
            prior=set()
            for stage in m.STAGES:
                st=branch['stages'][stage];job=st['job'];spec=docs[Path(job['config']).name]
                self.assertEqual(len(spec['stages']),1)
                self.assertEqual(job['pinned_manifest'],None);self.assertEqual(job['pinned_files'],[])
                self.assertEqual(job['inputs'],[job['config'],'/workspace/vkgr-sid-coll-ops-v1/deployment.json'])
                self.assertIn('VKGR_DATA_ROOT=/workspace/vkgr-b1-g',job['env'])
                self.assertIn('PYTHONPATH=/workspace/vkgr-sid-coll-g-v1/src:/workspace/vkgr-sid-coll-g-v1',job['env'])
                self.assertEqual(st['proof']['expected']['binding.spec_sha256'],m.wire_sha(spec))
                for token in job['argv']:
                    if isinstance(token,str) and token.startswith('@'):self.assertIn(token[1:].split('.')[0],prior)
                if stage.startswith(('train','verify')):self.assertEqual(job['vram_mib'],'@cuda.raw_proof.declared_vram_mib')
                if stage=='cuda':self.assertEqual(job['vram_mib'],39312)
                prior.add(stage)
        self.assertEqual(len(docs),30)

    def test_minimal_refs_and_producer_binding(self):
        m=module();dep,meta,s0=self.fixture();docs=m.build_documents(dep,meta,s0,root=ROOT)
        early=docs['sid-coll-c1-input-v1.json'];self.assertEqual(set(early['refs']),{'catalog','content','s0_plan'})
        verify=docs['sid-coll-c2-verify2024-v1.json']
        origin=verify['refs']['train_receipt']['from']
        self.assertEqual(origin['run_id'],'sid-coll-c2-train2024-v1');self.assertEqual(origin['stage'],'train')
        self.assertEqual(origin['proof'],'__receipt__')
        self.assertEqual(verify['stages']['verify']['parameters']['seed'],2024)
        self.assertNotIn('config42',verify['refs'])
        for spec in [v for v in docs.values() if v.get('schema')=='coll-stage-spec-v1']:
            for ref in spec['refs'].values():self.assertFalse(Path(ref['path']).suffix in ('.pt','.pth','.ckpt'))

    def test_missing_metadata_and_config_tamper_fail_closed(self):
        m=module();dep,meta,s0=self.fixture()
        for role in ('content','g_prepared'):
            bad=copy.deepcopy(meta);bad['common'].pop(role)
            with self.assertRaises(ValueError):m.build_documents(dep,bad,s0,root=ROOT)
        bad=copy.deepcopy(meta);bad.pop('support_ref')
        with self.assertRaises(ValueError):m.build_documents(dep,bad,s0,root=ROOT)
        bad=copy.deepcopy(dep);bad['files']['configs/sid_coll/g-coll-c1-s42.json']='0'*64
        with self.assertRaises(ValueError):m.build_documents(bad,meta,s0,root=ROOT)

    def test_cli_exclusive_output(self):
        m=module();dep,meta,s0=self.fixture()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            tmp=Path(tmp)
            for name,value in [('deployment',dep),('metadata',meta),('s0-plan',s0)]:
                (tmp/(name+'.json')).write_text(json.dumps(value))
            args=sum(([f'--{n}',str(tmp/(n+'.json'))] for n in ('deployment','metadata','s0-plan')),[])+['--output',str(tmp/'out')]
            self.assertEqual(m.main(args),0)
            docs=json.loads((tmp/'out/documents.json').read_text());self.assertIn('pipeline-plan.json',docs)
            self.assertEqual(m.main(args),1)
