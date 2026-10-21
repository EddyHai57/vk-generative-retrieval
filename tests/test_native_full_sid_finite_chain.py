import copy,json
from types import SimpleNamespace
from pathlib import Path
import pytest
from ops.tools.remote import native_full_sid_finite_chain as m


def template(tmp_path):
    return {'schema':'UR001-native-full-SID-request-v1','source':{'commit':m.SID_SOURCE,'files':{}},'lock':{'path':str(tmp_path/'lock'),'sha256':'a'*64},'producer':{'root':str(m.FULL_CODE),'source':m.FULL_SOURCE,'request':{'path':str(tmp_path/'full-request.json'),'sha256':'0'*64},'receipt':{'path':str(tmp_path/'full-out/fit/result.json'),'sha256':'0'*64},'verified_producer':{'run_id':'ur001-native-full-fit-v1',**{k:'0'*64 for k in m.PROOF_HASHES}}},'ops_code':str(m.OPS_CODE),'ops_root':str(m.OPS_ROOT),'output':str(tmp_path/'sid-output'),'preflight_run_id':'ur001-native-full-sid-preflight-v1','fit_run_id':'ur001-native-full-sid-fit-v1','verify_run_id':'ur001-native-full-sid-verify-v1','resources':{'cpu_cores':1,'ram_mib':368640,'vram_mib':0,'disk_gib':16,'wall_hours':48},'engineering':False,'test_read':False}


def test_materialize_only_mechanical_pins(tmp_path):
 t=template(tmp_path);full={'runner_source':m.FULL_SOURCE,'fit_run_id':t['producer']['verified_producer']['run_id'],'output':str(tmp_path/'full-out'),'ops_root':str(m.OPS_ROOT),'lock':t['lock']['path']};proof={'run_id':full['fit_run_id'],**{k:('d'*64 if k=='receipt_sha256' else 'b'*64) for k in m.PROOF_HASHES}}
 got=m.materialize(t,full,'c'*64,'d'*64,proof)
 expected=copy.deepcopy(t);expected['producer']['request']['sha256']='c'*64;expected['producer']['receipt']['sha256']='d'*64;expected['producer']['verified_producer']=proof
 assert got==expected and t['producer']['request']['sha256']=='0'*64
 for key in ('resources','source','engineering','test_read'):assert got[key]==t[key]

@pytest.mark.parametrize('key',['cpu_cores','ram_mib','disk_gib','wall_hours'])
def test_resources_explicit_no_default(tmp_path,key):
 t=template(tmp_path);del t['resources'][key]
 with pytest.raises(ValueError):m.validate_template(t)

@pytest.mark.parametrize('slot',['schema','engineering','test_read','source','producer','resources'])
def test_template_closed(tmp_path,slot):
 t=template(tmp_path);t[slot]=None
 with pytest.raises((ValueError,TypeError)):m.validate_template(t)


def test_deadline_per_block(tmp_path):
 p=tmp_path/'blob';p.write_bytes(b'a'*12);a=SimpleNamespace(deadline=1,now=iter([0,0,2]).__next__)
 with pytest.raises(TimeoutError):m.hash_file(a,p,block_size=4)


def test_overlap_rejects(tmp_path):
 a=SimpleNamespace(evidence=tmp_path/'safe',template=tmp_path/'safe/template',full_request=tmp_path/'request')
 with pytest.raises(ValueError):m.check_paths(a,template(tmp_path))


def test_cli_subprocess():
 import subprocess,sys
 root=Path(__file__).resolve().parents[1]
 for command,ok in [('dry',True),('wrong-command',False)]:
  result=subprocess.run([sys.executable,'-B','-m','ops.tools.remote.native_full_sid_finite_chain',command],cwd=root,capture_output=True,text=True)
  assert (result.returncode==0)is ok
  if ok:assert json.loads(result.stdout)['submitted']is False


def test_tree_rejects_ignored_extra_file(tmp_path):
 import subprocess,time
 root=tmp_path/'tree';root.mkdir()
 def git(*a):return subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
 git('init','-q');(root/'.gitignore').write_text('ignored\n');(root/'code').write_text('x');git('add','.');git('-c','user.name=Fixture','-c','user.email=f@example.invalid','commit','-qm','fixture')
 a=SimpleNamespace(now=time.time,deadline=time.time()+60)
 m.tree(a,root,git('rev-parse','HEAD'))
 (root/'ignored').write_text('untracked ignored payload')
 with pytest.raises(ValueError,match='actual file set'):m.tree(a,root,git('rev-parse','HEAD'))


def stage_fixture(tmp_path,monkeypatch):
 import time
 ops=tmp_path/'ops';rid='ur001-native-full-sid-preflight-v1'
 for state in ('pending','claimed','rejected'):(ops/'queue'/state).mkdir(parents=True)
 monkeypatch.setattr(m,'OPS_ROOT',ops);monkeypatch.setattr(m,'fixed',lambda a:{})
 a=SimpleNamespace(evidence=tmp_path/'evidence',sid_request=tmp_path/'request',sid_pin='a'*64,now=time.time,deadline=time.time()+60);a.evidence.mkdir()
 return a,{'preflight_run_id':rid,'output':str(tmp_path/'sid-output')},ops,rid


def test_stage_rejected_no_submit(tmp_path,monkeypatch):
 a,v,ops,rid=stage_fixture(tmp_path,monkeypatch);(ops/'queue/rejected'/('1-'+rid+'.json')).write_text('{}')
 monkeypatch.setattr(m,'execute',lambda *a,**k:pytest.fail('must not run'))
 with pytest.raises(ValueError,match='rejected'):m.sid_stage(a,v,'preflight',lambda _:None)


def test_stage_pending_then_disappeared_never_submit(tmp_path,monkeypatch):
 a,v,ops,rid=stage_fixture(tmp_path,monkeypatch);p=ops/'queue/pending'/('1-'+rid+'.json');p.write_text('{}')
 monkeypatch.setattr(m,'execute',lambda *a,**k:pytest.fail('must not run'))
 with pytest.raises(ValueError,match='disappeared'):m.sid_stage(a,v,'preflight',lambda _:p.unlink())


def test_stage_failed_terminal_no_retry(tmp_path,monkeypatch):
 a,v,ops,rid=stage_fixture(tmp_path,monkeypatch);run=ops/'runs'/rid;run.mkdir(parents=True);(run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':rid,'status':'failed','exit_code':1,'signal':None,'wall_cap_exceeded':False}))
 monkeypatch.setattr(m,'execute',lambda *a,**k:pytest.fail('must not run'))
 with pytest.raises(ValueError,match='failed'):m.sid_stage(a,v,'preflight',lambda _:None)


def test_stage_submits_once_then_claimed_accepted(tmp_path,monkeypatch):
 a,v,ops,rid=stage_fixture(tmp_path,monkeypatch);calls=[]
 def execute(args,code,argv,name,**kw):
  calls.append(name)
  if name.endswith('-plan'):return {'status':'prepared_dry','spec':{'run_id':rid},'metadata':{'git':{'commit':m.SID_SOURCE,'dirty':False,'dirty_files':[]}}}
  if name.endswith('-plan-checked'):return {'verified':True,'submitted':False}
  if name.endswith('-submit'):
   (ops/'queue/claimed'/('1-'+rid+'.json')).write_text('{}');return 'queued'
  return {'producer':{'run_id':rid}}
 monkeypatch.setattr(m,'execute',execute);monkeypatch.setattr(m,'submit_argv',lambda spec:['original-scheduler'])
 def sleep(seconds):
  assert seconds<=m.POLL_SECONDS
  run=ops/'runs'/rid;run.mkdir(parents=True);(run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}))
 assert m.sid_stage(a,v,'preflight',sleep)['producer']['run_id']==rid
 assert calls==['preflight-plan','preflight-plan-checked','preflight-submit','preflight-accepted']


def test_deadline_before_submit(tmp_path,monkeypatch):
 a,v,ops,rid=stage_fixture(tmp_path,monkeypatch);calls=[];real_execute=m.execute
 monkeypatch.setattr(m,'fixed',lambda args:{})
 def execute(args,code,argv,name,**kw):
  calls.append(name)
  if name.endswith('-plan'):return {'status':'prepared_dry','spec':{'run_id':rid},'metadata':{'git':{'commit':m.SID_SOURCE,'dirty':False,'dirty_files':[]}}}
  if name.endswith('-plan-checked'):
   a.deadline=0;return {'verified':True,'submitted':False}
  return real_execute(args,code,argv,name,**kw)
 monkeypatch.setattr(m,'execute',execute);monkeypatch.setattr(m,'submit_argv',lambda spec:['must-never-execute'])
 with pytest.raises(TimeoutError):m.sid_stage(a,v,'preflight',lambda _:None)
 assert not list((ops/'queue/claimed').iterdir())


def test_real_launcher_and_check_plan_synthetic_e2e(tmp_path,monkeypatch):
 import os,subprocess,sys,hashlib
 from analysis.a40_ops import load_ops
 repo=Path(__file__).resolve().parents[1]
 def git(root,*argv):return subprocess.check_output(['git','-C',str(root),*argv],text=True).strip()
 def clone(name,ref):
  root=tmp_path/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(root)],check=True,capture_output=True);git(root,'checkout','-q','--detach',ref);return root
 code=clone('code',m.SID_SOURCE);ops=clone('ops','5840e6b686ec9651314a9e28a1a2c0befcc05158')
 stub='''import argparse,json,hashlib,sys
from pathlib import Path
from analysis.a40_ops import load_ops
from analysis.sgns_subset_acceptance_a1 import transport_job
ROOT=Path(__file__).resolve().parents[1]
exact=lambda a,b:json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
def load(path,pin):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('fixture request pin')
 return json.loads(raw)
def identity(v):return {'source':v['source'],'runtime':{'fixture':'same-runtime'}}
def rid(v,stage):return v[stage+'_run_id']
def argv(stage,path,pin):return [sys.executable,'-B','-m','analysis.sgns_native_full_sid',stage,'--request',str(path),'--request-sha256',pin,'--execute-on-new-instance']
def resources(v):return {**{k:v['resources'][k]for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')},'wall_seconds':v['resources']['wall_hours']*3600.0}
def pins(v,path,stage):return [{'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()},v['lock']]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['dry']);p.parse_args();print(json.dumps({'status':'dry','training':False,'submitted':False}))
'''
 (code/'analysis/sgns_native_full_sid.py').write_text(stub);git(code,'add','analysis/sgns_native_full_sid.py');git(code,'-c','user.name=Fixture','-c','user.email=f@example.invalid','commit','-qm','Synthetic SID transport fixture')
 source={'commit':git(code,'rev-parse','HEAD'),'files':{n:hashlib.sha256((code/n).read_bytes()).hexdigest()for n in git(code,'ls-tree','-r','--name-only','HEAD').splitlines()}}
 lock=tmp_path/'lock';lock.write_text('synthetic lock');rid='ur001-native-full-sid-preflight-fixture';root=tmp_path/'OPS';request=tmp_path/'request.json'
 value={'source':source,'ops_root':str(root),'ops_code':str(ops),'preflight_run_id':rid,'resources':{'cpu_cores':1,'ram_mib':1024,'vram_mib':0,'disk_gib':1,'wall_hours':1},'lock':{'path':str(lock),'sha256':hashlib.sha256(lock.read_bytes()).hexdigest()}}
 request.write_text(json.dumps(value));pin=hashlib.sha256(request.read_bytes()).hexdigest()
 for key in list(os.environ):
  if key.startswith('VKGR_') or key in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG'):monkeypatch.delenv(key,raising=False)
 env={'PYTHONPATH':str(code/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 flags=['--root',str(root),'--run-id',rid,'--kind','ops','--cwd',str(code),'--config',str(request),'--seed','42','--cpu-cores','1','--ram-mib','1024','--vram-mib','0','--disk-gib','1','--wall-hours','1','--input',str(request),'--input',str(lock)]
 for k,v in env.items():flags+=['--env',k+'='+v]
 argv=[sys.executable,'-B','-m','analysis.sgns_native_full_sid','preflight','--request',str(request),'--request-sha256',pin,'--execute-on-new-instance']
 _,launcher,_,_=load_ops(ops);spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]));metadata=launcher.prepare(spec);plan=tmp_path/'plan.json';plan.write_text(json.dumps({'spec':spec,'metadata':metadata}))
 result=subprocess.run([sys.executable,'-B','-c',m.CHECK_PLAN,str(request),pin,'preflight',str(plan),source['commit']],cwd=code,env={**os.environ,**env,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 assert json.loads(result.stdout)=={'verified':True,'submitted':False}
 for key,changed in [('argv',[*spec['argv'][:-1]]),('inputs',[str(lock),str(request)]),('env',['PYTHONPATH=/wrong']),('seed',2024),('ram_mib',2048)]:
  bad=copy.deepcopy(spec);bad[key]=changed;badplan=tmp_path/('bad-'+key+'.json');badplan.write_text(json.dumps({'spec':bad,'metadata':launcher.prepare(bad)}))
  badresult=subprocess.run([sys.executable,'-B','-c',m.CHECK_PLAN,str(request),pin,'preflight',str(badplan),source['commit']],cwd=code,env={**os.environ,**env,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
  assert badresult.returncode!=0,key
 for command,expected in [('dry',0),('forbidden',2)]:
  dry=subprocess.run([sys.executable,'-B','-m','analysis.sgns_native_full_sid',command],cwd=code,env={**os.environ,**env},capture_output=True,text=True)
  assert dry.returncode==expected
  if expected==0:assert json.loads(dry.stdout)['training']is False


@pytest.mark.parametrize('slot',['runner_source','fit_run_id','ops_root','lock','output'])
def test_materialize_lineage_drift(tmp_path,slot):
 t=template(tmp_path);full={'runner_source':m.FULL_SOURCE,'fit_run_id':t['producer']['verified_producer']['run_id'],'output':str(tmp_path/'full-out'),'ops_root':str(m.OPS_ROOT),'lock':t['lock']['path']};proof={'run_id':full['fit_run_id'],**{k:'b'*64 for k in m.PROOF_HASHES}}
 full[slot]='wrong'
 with pytest.raises(ValueError,match='lineage'):m.materialize(t,full,'c'*64,'b'*64,proof)


@pytest.mark.parametrize('slot',list(m.PROOF_HASHES))
def test_materialize_zero_proof_refused(tmp_path,slot):
 t=template(tmp_path);full={'runner_source':m.FULL_SOURCE,'fit_run_id':t['producer']['verified_producer']['run_id'],'output':str(tmp_path/'full-out'),'ops_root':str(m.OPS_ROOT),'lock':t['lock']['path']};proof={'run_id':full['fit_run_id'],**{k:'b'*64 for k in m.PROOF_HASHES}};proof[slot]='0'*64
 with pytest.raises(ValueError):m.materialize(t,full,'c'*64,'b'*64,proof)


@pytest.mark.parametrize('hours',[0,-1,201,float('inf'),float('nan'),True])
def test_driver_deadline_finite(hours):
 with pytest.raises(ValueError):m.base.deadline_seconds(hours)


def test_ppid_guard(tmp_path,monkeypatch):
 monkeypatch.setattr(m.os,'getppid',lambda:2)
 with pytest.raises(ValueError,match='PPID1'):m.run(SimpleNamespace())


def test_full_chain_order_and_failure_alert(tmp_path,monkeypatch):
 import time
 monkeypatch.setattr(m.os,'getppid',lambda:1);v=template(tmp_path);events=[]
 monkeypatch.setattr(m,'fixed',lambda args:v)
 monkeypatch.setattr(m,'wait_full',lambda args,t,sleep:events.append('full-verified')or v)
 def stage(args,value,name,sleep):
  events.append(name)
  if name=='verify':
   p=Path(v['output'])/'verify';p.mkdir(parents=True);(p/'stage-receipt.json').write_text('{}')
 monkeypatch.setattr(m,'sid_stage',stage)
 def execute(args,code,argv,name,**kw):
  events.append(name);assert code==m.SID_CODE
  return {'producer':{'run_id':v['verify_run_id']}}
 monkeypatch.setattr(m,'execute',execute)
 a=SimpleNamespace(evidence=tmp_path/'evidence',deadline_hours=200)
 result=m.run(a)
 assert events==['full-verified','preflight','fit','verify','sid-consumed']
 assert result['g_training_submitted']is False and result['test_read']is False
 assert json.loads((a.evidence/'summary.json').read_text())==result
 def fail(*args):raise ValueError('original failure')
 monkeypatch.setattr(m,'wait_full',fail)
 b=SimpleNamespace(evidence=tmp_path/'failure-evidence',deadline_hours=200)
 with pytest.raises(ValueError,match='original failure'):m.run(b)
 assert json.loads((b.evidence/'summary.json').read_text())['status']=='ALERT'
 assert events==['full-verified','preflight','fit','verify','sid-consumed']


@pytest.mark.parametrize('state',['pending','claimed','rejected','run','output'])
def test_submit_final_guard_observes_late_presence(tmp_path,monkeypatch,state):
 a,v,ops,rid=stage_fixture(tmp_path,monkeypatch)
 def fixed(args):
  if state in ('pending','claimed','rejected'):(ops/'queue'/state/('1-'+rid+'.json')).write_text('{}')
  elif state=='run':(ops/'runs'/rid).mkdir(parents=True)
  else:(Path(v['output'])/'preflight').mkdir(parents=True)
 monkeypatch.setattr(m,'fixed',fixed)
 monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('must never submit after late presence'))
 if state in ('pending','claimed'):
  assert m.execute(a,tmp_path,['scheduler'],'submit',json_output=False,before_submit=lambda:m.submission_state(v,'preflight')['state']=='absent')is None
 else:
  with pytest.raises(ValueError):m.execute(a,tmp_path,['scheduler'],'submit',json_output=False,before_submit=lambda:m.submission_state(v,'preflight')['state']=='absent')
 assert not (a.evidence/'submit.stdout').exists()


def test_ops_actual_file_set_rejects_ignored_pyc(tmp_path):
 import time,subprocess
 repo=Path(__file__).resolve().parents[1];ops=tmp_path/'ops'
 subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(ops)],check=True,capture_output=True);subprocess.run(['git','-C',str(ops),'checkout','-q','--detach',m.OPS_SOURCE],check=True,capture_output=True)
 a=SimpleNamespace(now=time.time,deadline=time.time()+60);m.tree(a,ops,m.OPS_SOURCE)
 p=ops/'ops/vkgr_ops/__pycache__/ignored.pyc';p.parent.mkdir();p.write_bytes(b'ignored bytecode')
 assert subprocess.run(['git','-C',str(ops),'check-ignore',str(p)],capture_output=True).returncode==0
 with pytest.raises(ValueError,match='actual file set'):m.tree(a,ops,m.OPS_SOURCE)


def test_no_submit_if_queue_appears_during_plan_check(tmp_path,monkeypatch):
 a,v,ops,rid=stage_fixture(tmp_path,monkeypatch);calls=[]
 def execute(args,code,argv,name,**kw):
  calls.append(name)
  if name.endswith('-plan'):return {'status':'prepared_dry','spec':{'run_id':rid},'metadata':{'git':{'commit':m.SID_SOURCE,'dirty':False,'dirty_files':[]}}}
  if name.endswith('-plan-checked'):
   (ops/'queue/pending'/('1-'+rid+'.json')).write_text('{}')
   return {'verified':True,'submitted':False}
  if name.endswith('-submit'):raise AssertionError('duplicate submission attempted despite newly existing queue')
  pytest.fail(name)
 monkeypatch.setattr(m,'execute',execute);monkeypatch.setattr(m,'submit_argv',lambda spec:['scheduler'])
 def stop_sleep(_):raise RuntimeError('stop independent simulation')
 with pytest.raises(RuntimeError,match='stop independent simulation'):m.sid_stage(a,v,'preflight',stop_sleep)
 assert not any(n.endswith('-submit')for n in calls)


def test_fixed_includes_actual_ops_tree(tmp_path,monkeypatch):
 t=template(tmp_path);p=tmp_path/'template.json';p.write_text(json.dumps(t));seen=[]
 a=SimpleNamespace(driver_source='a'*40,driver_sha256='b'*64,template=p,template_sha256='c'*64,full_request=Path(t['producer']['request']['path']))
 monkeypatch.setattr(m,'tree',lambda args,root,pin:seen.append(root))
 monkeypatch.setattr(m,'hash_file',lambda args,path: a.driver_sha256 if str(path)==m.__file__ else a.template_sha256)
 monkeypatch.setattr(m,'check_paths',lambda *a:None)
 assert m.fixed(a)==t
 assert m.OPS_CODE in seen,'OPS imported implementation has no actual-file-set verification'
