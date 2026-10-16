import importlib.util,unittest,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Retry(unittest.TestCase):
 def test_reuses_all_successes_and_changes_only_failed_preflight_identity(self):
  m=module('retry',ROOT/'scripts/sid_coll_retry.py');fixture=module('fixture',ROOT/'tests/test_sid_coll_plan.py');dep,meta,s0=fixture.PlanTests().fixture()
  base=module('base',ROOT/'scripts/sid_coll_plan.py');docs=base.build_documents(dep,meta,s0,root=ROOT);old=copy.deepcopy(docs);snapshot={'runs':{}}
  for branch in docs['pipeline-plan.json']['branches'].values():
   for name in m.REUSED:
    rid=branch['stages'][name]['job']['run_id'];out='/workspace/vkgr-infra01-root/runs/'+rid+'/out/'
    snapshot['runs'][rid]={'state':'ACCEPTED','receipt_ref':{'path':out+'receipt.json','sha256':'a'*64},'artifacts':{key:{'path':out+file,'sha256':'b'*64} for key,file in [('vectors','payload/vectors.npy'),('input_manifest','input-manifest.json'),('sid_manifest','payload/run_manifest.json')]}}
  newdep={**dep,'commit':'c'*40};result=m.build(docs,snapshot,newdep,{'path':m.OPS+'/deployment.json','sha256':'d'*64})
  plan=result['pipeline-plan.json'];self.assertEqual(plan['deadline_epoch'],old['pipeline-plan.json']['deadline_epoch']);self.assertEqual(docs,old)
  for label,branch in plan['branches'].items():
   for name in m.REUSED:self.assertEqual(branch['stages'][name],old['pipeline-plan.json']['branches'][label]['stages'][name])
   self.assertTrue(branch['stages']['preflight']['job']['run_id'].endswith('-v2'))
   self.assertTrue(branch['stages']['train42']['job']['run_id'].endswith('-v1'))
   spec=result[Path(branch['stages']['preflight']['job']['config']).name]
   self.assertEqual(spec['refs']['vectors']['sha256'],'b'*64);self.assertEqual(spec['commit'],'c'*40)
  broken=copy.deepcopy(snapshot);broken['runs'][next(iter(broken['runs']))]['state']='UNACCEPTED'
  with self.assertRaises(ValueError):m.build(docs,broken,newdep,{'path':m.OPS+'/deployment.json','sha256':'d'*64})
 def test_recovery_control_rejects_old_failure_id_and_science_change(self):
  m=module('control',ROOT/'scripts/sid_coll_retry_control.py')
  names={'src/vkgr/model.py':'a'*64,'configs/sid_coll/g-coll-c1-s42.json':'b'*64}
  self.assertEqual(m.scientific_equal(names,names),2)
  with self.assertRaises(ValueError):m.scientific_equal(names,{**names,'src/vkgr/model.py':'c'*64})
  stage=lambda rid:{'job':{'run_id':rid}}
  old={'created_epoch':1,'deadline_epoch':172801,'branches':{'C1p':{'stages':{'input':stage('input-v1'),'preflight':stage('pf-v1'),'train42':stage('train-v1')}}}}
  new=copy.deepcopy(old);new['branches']['C1p']['stages']['preflight']['job']['run_id']='pf-v2'
  m.validate_retry_ids(old,new)
  with self.assertRaises(ValueError):m.validate_retry_ids(old,old)
  new['deadline_epoch']+=1
  with self.assertRaises(ValueError):m.validate_retry_ids(old,new)
 def test_dynamic_vram_alone_is_deferred_until_measurement(self):
  m=module('control',ROOT/'scripts/sid_coll_retry_control.py')
  self.assertTrue(m.has_dynamic({'argv':['python'],'vram_mib':'@cuda.raw_proof.declared_vram_mib'}))
  self.assertTrue(m.has_dynamic({'argv':['python','@input.sha'],'vram_mib':0}))
  self.assertFalse(m.has_dynamic({'argv':['python'],'vram_mib':2048}))
if __name__=='__main__':unittest.main()
