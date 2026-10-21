"""Independent C reviewer controls ported verbatim except stable repository root."""
import importlib.util,json,types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module():
 s=importlib.util.spec_from_file_location('p_driver_extra',ROOT/'ops/tools/remote/p_popularity_finite_chain.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_ppid_rejects_before_inputs(tmp_path,monkeypatch):
 m=module();monkeypatch.setattr(m.os,'getppid',lambda:123)
 with pytest.raises(ValueError,match='PPID=1'):m.run(types.SimpleNamespace())

@pytest.mark.parametrize('mode',['pending-terminal','queue-drift','accept-wrong','dry-fail','submit-absent','rejected'])
def test_defensive_stage_failure_never_evaluates(tmp_path,monkeypatch,mode):
 m=module();a=types.SimpleNamespace(inputs=tmp_path/'inputs',request=tmp_path/'request',consumer_code=tmp_path/'consumer',ops_code=tmp_path/'ops-code',evidence=tmp_path/'evidence',request_sha256='a'*64,deadline_hours=1)
 ops=tmp_path/'ops';out=tmp_path/'out';v={'ops_root':str(ops),'output_root':str(out),'run_ids':{k:'p-full-popularity-'+s+'-v1'for s,k in m.STAGES.items()},'job_contract':{k:{'env':{'PYTHONPATH':str(a.consumer_code/'src')}}for k in m.STAGES.values()}}
 for state in ('pending','claimed','rejected'):(ops/'queue'/state).mkdir(parents=True)
 monkeypatch.setattr(m.os,'getppid',lambda:1);monkeypatch.setattr(m,'fixed_request',lambda a:v)
 calls=[];clock=[0];rid=v['run_ids']['preflight'];pending=ops/'queue/pending'/('1-'+rid+'.json')
 def command(a,argv,name,decode=True):
  calls.append(name)
  if name.endswith('-plan'):return {'stage':'preflight','training':False,'status':'planned_dry','expected_job':{'run_id':rid},'launcher_argv':['--run-id',rid]}
  if name.endswith('-dry'):return {'verified':mode!='dry-fail','submitted':False,'spec':{'run_id':rid}}
  if name.endswith('-submit'):
   if mode!='submit-absent':pending.write_text(json.dumps({'schema':'queue','run_id':rid,'unexpected':True}if mode=='queue-drift'else{'schema':'queue','run_id':rid}))
   return b'queue-path'
  if name.endswith('-accepted'):return {'verified':False,'stage':'preflight','training':False,'test_read':False}
  pytest.fail(name)
 monkeypatch.setattr(m,'command',command)
 def sleep(seconds):
  assert seconds==900;clock[0]+=seconds
  if mode=='rejected':pending.rename(ops/'queue/rejected'/pending.name);return
  if mode!='pending-terminal':pending.rename(ops/'queue/claimed'/pending.name)
  run=ops/'runs'/rid;run.mkdir(parents=True)
  (run/'run.json').write_text(json.dumps({'run_id':rid}))
  (run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}))
 reasons={'pending-terminal':'terminal lacks claimed confirmation','queue-drift':'original queue spec differs','accept-wrong':'original acceptance required','dry-fail':'original launcher dry required','submit-absent':'submission absent from original queue','rejected':'rejected; no retry'}
 with pytest.raises(ValueError,match=reasons[mode]):m.run(a,now=lambda:clock[0],sleep=sleep)
 assert json.loads((a.evidence/'summary.json').read_bytes())['status']=='ALERT'
 assert 'evaluate-plan'not in calls and calls.count('preflight-submit')<=1
 if mode=='dry-fail':assert 'preflight-submit'not in calls
