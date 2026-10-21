"""Bounded synthetic transport tests; never submit a real job."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]

def module():
    spec=importlib.util.spec_from_file_location('pop_chain',ROOT/'ops/tools/remote/p_popularity_finite_chain.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_cli():
    for cmd,code in [('dry',0),('invalid',2)]:
        p=subprocess.run([sys.executable,'-B',str(ROOT/'ops/tools/remote/p_popularity_finite_chain.py'),cmd],capture_output=True,text=True)
        assert p.returncode==code,p.stderr
        if cmd=='dry':assert json.loads(p.stdout)['submitted'] is False

@pytest.mark.parametrize('v',[True,0,-1,201,float('inf'),float('nan')])
def test_deadline(v):
    with pytest.raises(ValueError):module().deadline_seconds(v)

def test_remaining_and_command_timeout(tmp_path,monkeypatch):
    m=module();a=types.SimpleNamespace(deadline=10,now=lambda:4,consumer_code=tmp_path,env={},evidence=tmp_path)
    seen={}
    def run(argv,**kw):seen.update(kw);return types.SimpleNamespace(stdout=b'queue-path\n',stderr=b'',returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run)
    assert m.command(a,['x'],'submit',decode=False)==b'queue-path\n'
    assert seen['timeout']==6
    a.now=lambda:11
    with pytest.raises(TimeoutError):m.remaining(a)

def git_tree(tmp_path):
    root=tmp_path/'tree';root.mkdir();(root/'a.py').write_text('x=1\n')
    for argv in (['init','-q'],['add','.'],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture']):subprocess.run(['git','-C',str(root),*argv],check=True)
    sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    return root,sha

def test_tree_rejects_ignored_extra_and_symlink(tmp_path):
    m=module();root,sha=git_tree(tmp_path);a=types.SimpleNamespace(deadline=100,now=lambda:0)
    m.exact_tree(a,root,sha)
    (root/'__pycache__').mkdir();extra=root/'__pycache__/a.pyc';extra.write_bytes(b'ignored')
    with pytest.raises(ValueError):m.exact_tree(a,root,sha)
    extra.unlink();(root/'__pycache__').rmdir();(root/'a.py').unlink();(root/'a.py').symlink_to(tmp_path/'elsewhere')
    with pytest.raises(ValueError):m.exact_tree(a,root,sha)

def test_queue_failure_duplicate_and_terminal(tmp_path):
    m=module();rid='p-full-popularity-pf-v1'
    for s in ('pending','claimed','rejected'):(tmp_path/'queue'/s).mkdir(parents=True)
    assert m.queue_state(tmp_path,rid)['state']=='absent'
    path=tmp_path/'queue/pending'/('1-'+rid+'.json');path.write_text('{}')
    assert m.queue_state(tmp_path,rid)['state']=='pending'
    other=tmp_path/'queue/rejected'/path.name;other.write_text('{}')
    with pytest.raises(ValueError):m.queue_state(tmp_path,rid)
    path.unlink()
    with pytest.raises(ValueError):m.queue_state(tmp_path,rid)
    run=tmp_path/'runs'/rid;run.mkdir(parents=True)
    terminal={'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
    (run/'terminal.json').write_text(json.dumps(terminal));assert m.terminal(run,rid)
    terminal['exit_code']=False;(run/'terminal.json').write_text(json.dumps(terminal))
    with pytest.raises(ValueError):m.terminal(run,rid)

def test_evidence_overlap(tmp_path):
    m=module();a=types.SimpleNamespace(evidence=tmp_path/'out/evidence',consumer_code=tmp_path/'code',ops_code=tmp_path/'ops-code',inputs=tmp_path/'inputs',request=tmp_path/'request')
    with pytest.raises(ValueError):m.evidence_path(a,{'output_root':str(tmp_path/'out')})

def test_acceptance_uses_original_and_all_producers():
    m=module()
    assert 'p.accepted(' in m.CHECK and 'p.context(' in m.CHECK
    assert "('preflight','evaluate','verify')" in m.CHECK
    assert m.STAGES=={'preflight':'preflight','evaluate':'train','verify':'verify'}

def test_real_launcher_synthetic_e2e(tmp_path):
    # Reuse the independently reviewed fixed-layout synthetic admission builder,
    # then run this driver's own complete prepare/dry comparison (no scheduler).
    from test_p_full_popularity import test_popularity_cli_plan_real_launcher_prepare
    test_popularity_cli_plan_real_launcher_prepare(tmp_path)
    m=module();deployment=tmp_path/'deployed';ops=tmp_path/'ops-source';request=tmp_path/'request.json';inputs=tmp_path/'inputs.json'
    for argv in (['init','-q'],['add','.'],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','ops fixture']):subprocess.run(['git','-C',str(ops),*argv],check=True)
    env={**os.environ,'PYTHONPATH':str(ops/'ops'),'PYTHONDONTWRITEBYTECODE':'1'}
    source=subprocess.check_output([sys.executable,'-B','-c','import json;from vkgr_ops import launcher;print(json.dumps(launcher._ops_source()))'],env=env,text=True)
    value=json.loads(request.read_bytes())
    for c in value['job_contract'].values():c['ops_source']=json.loads(source)
    request.write_text(json.dumps(value,sort_keys=True));pin=m.sha(request)
    env.update(value['job_contract']['preflight']['env'])
    plan=subprocess.run([sys.executable,'-B','-m','analysis.p_full_popularity','plan','--stage','preflight','--inputs',str(inputs),'--request',str(request),'--request-sha256',pin],cwd=deployment,env=env,capture_output=True,check=True)
    path=tmp_path/'plan.json';path.write_bytes(plan.stdout)
    result=subprocess.run([sys.executable,'-B','-c',m.PREPARE,str(path),str(ops)],cwd=deployment,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['verified'] is True
    # A synchronized expected-job env change is refused by actual launcher dry.
    v=json.loads(plan.stdout);v['expected_job']['env']['OMP_NUM_THREADS']='8';path.write_text(json.dumps(v))
    result=subprocess.run([sys.executable,'-B','-c',m.PREPARE,str(path),str(ops)],cwd=deployment,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'complete original launcher' in result.stderr
    assert not Path(value['ops_root']).exists()

@pytest.mark.parametrize('mode',['success','disappear','accept-fail','terminal-fail','deadline'])
def test_finite_chain_single_submit_and_stop(tmp_path,monkeypatch,mode):
    m=module();a=types.SimpleNamespace(inputs=tmp_path/'inputs',request=tmp_path/'request',consumer_code=tmp_path/'consumer',ops_code=tmp_path/'ops-code',evidence=tmp_path/'evidence',request_sha256='a'*64,deadline_hours=1)
    ops=tmp_path/'ops';output=tmp_path/'output';v={'ops_root':str(ops),'output_root':str(output),'run_ids':{key:'p-full-popularity-'+stage+'-v1' for stage,key in m.STAGES.items()},'job_contract':{key:{'env':{'PYTHONPATH':str(a.consumer_code/'src')}} for key in m.STAGES.values()}}
    for state in ('pending','claimed','rejected'):(ops/'queue'/state).mkdir(parents=True)
    monkeypatch.setattr(m.os,'getppid',lambda:1);monkeypatch.setattr(m,'fixed_request',lambda a:v)
    clock=[0];current={};calls=[]
    def cmd(args,argv,name,decode=True):
        calls.append(name)
        if name.endswith('-plan'):
            stage=name[:-5];rid=v['run_ids'][m.STAGES[stage]]
            current.update(stage=stage,rid=rid)
            return {'stage':stage,'training':False,'status':'planned_dry','expected_job':{'run_id':rid},'launcher_argv':['--run-id',rid]}
        if name.endswith('-dry'):return {'verified':True,'submitted':False,'spec':{'run_id':current['rid']}}
        if name.endswith('-submit'):
            assert decode is False
            rid=current['rid'];(ops/'queue/pending'/('1-'+rid+'.json')).write_text(json.dumps({'schema':'queue','run_id':rid}))
            return b'queue-path\n'
        if name.endswith('-accepted'):
            if mode=='accept-fail':raise ValueError('original accepted refused')
            return {'verified':True,'stage':current['stage'],'training':False,'test_read':False}
        pytest.fail(name)
    monkeypatch.setattr(m,'command',cmd)
    def sleep(seconds):
        assert 0<seconds<=900
        clock[0]+=seconds
        rid=current['rid'];pending=ops/'queue/pending'/('1-'+rid+'.json')
        if mode=='disappear':pending.unlink();return
        if mode=='deadline':clock[0]=3601;return
        claimed=ops/'queue/claimed'/pending.name;pending.rename(claimed)
        run=ops/'runs'/rid;run.mkdir(parents=True)
        (run/'run.json').write_text(json.dumps({'run_id':rid}))
        (run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded' if mode!='terminal-fail' else 'failed','exit_code':0,'signal':None,'wall_cap_exceeded':False}))
    if mode=='success':
        result=m.run(a,now=lambda:clock[0],sleep=sleep)
        assert result['status']=='verified' and len(result['submitted_run_ids'])==3
        assert [n for n in calls if n.endswith('-accepted')]==[s+'-accepted' for s in m.STAGES]
    else:
        with pytest.raises((ValueError,TimeoutError)):m.run(a,now=lambda:clock[0],sleep=sleep)
        assert json.loads((a.evidence/'summary.json').read_bytes())['status']=='ALERT'
        assert 'evaluate-submit' not in calls
    assert len([n for n in calls if n=='preflight-submit'])==1

@pytest.mark.parametrize('scientific',[True,False])
def test_check_calls_original_acceptance_for_each_producer(tmp_path,scientific):
    m=module();package=tmp_path/'analysis';package.mkdir();(package/'__init__.py').write_text('')
    (package/'p_full_popularity.py').write_text('''import json
class Backend:
 def close_context(self,*a):pass
 def encoded(self,v):return json.dumps(v,sort_keys=True)
backend=Backend()
def transport(*a):return {'fixed':True}
def context(*a):return (None,None,None,None,None,{'DAG':'authenticated'},lambda:None)
def accepted(v,i,r,p,stage,binding):
 assert binding=={'DAG':'authenticated'}
 print(stage,file=__import__('sys').stderr)
 return {'scientific_verified':SCIENCE}
'''.replace('SCIENCE',repr(scientific)))
    env={**os.environ,'PYTHONPATH':str(tmp_path),'PYTHONDONTWRITEBYTECODE':'1'}
    p=subprocess.run([sys.executable,'-B','-c',m.CHECK,str(tmp_path/'inputs'),str(tmp_path/'request'),'a'*64,'verify'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert p.stderr.splitlines()[:3]==['preflight','evaluate','verify']
    if scientific:assert p.returncode==0 and json.loads(p.stdout)['verified'] is True
    else:assert p.returncode!=0 and 'original scientific verify required' in p.stderr
