"""Single-seed driver transport checks; no server or live submission."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from test_p_full_backend import own_job_fixture, actual_dag_module


def module():
    return importlib.import_module('ops.tools.remote.p_first_batch_chain')


def test_queue_rejected_and_duplicate_never_become_submit_eligible(tmp_path):
    m = module();root=tmp_path/'ops';run='fixed'
    assert m.queue_state(root,run)=={'state':'absent','path':None}
    path=root/'queue/rejected/000-fixed.json';path.parent.mkdir(parents=True);path.write_text('{}')
    path.with_suffix('.reason').write_text('original rejection')
    with pytest.raises(ValueError,match='original rejection'):m.queue_state(root,run)
    path.unlink();path.with_suffix('.reason').unlink()
    for state in ('pending','claimed'):
        path=root/'queue'/state/'000-fixed.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    with pytest.raises(ValueError,match='duplicate'):m.queue_state(root,run)


def test_derived_request_changes_only_predecessor(tmp_path):
    m=module();base={'schema':'fixed','config':{'seed':42},'predecessor':None}
    pred={'manifest':'receipt','run_dir':'run','pin':{'fixed':True}}
    path,digest,value=m.stage_request(base,pred,tmp_path,'train',{'state':'absent','path':None},tmp_path/'no-run')
    assert value=={**base,'predecessor':pred}
    assert json.loads(path.read_bytes())==value and digest==m.sha(path)
    with pytest.raises(FileExistsError):m.stage_request(base,pred,tmp_path,'train',{'state':'absent','path':None},tmp_path/'no-run')


def test_existing_claimed_reuses_pinned_original_request(tmp_path):
    m=module();base={'config':{'seed':42},'predecessor':None};pred={'fixed':True}
    original=tmp_path/'original.json';original.write_text(json.dumps({**base,'predecessor':pred}))
    run=tmp_path/'run';run.mkdir();(run/'run.json').write_text(json.dumps({'config':{'path':str(original),'sha256':m.sha(original)}}))
    evidence=tmp_path/'evidence';evidence.mkdir()
    path,_,_=m.stage_request(base,pred,evidence,'train',{'state':'claimed','path':None},run)
    assert path==original and not list(evidence.iterdir())
    original.write_text(json.dumps({**base,'config':{'seed':2024},'predecessor':pred}))
    with pytest.raises(ValueError):m.stage_request(base,pred,evidence,'train',{'state':'claimed','path':None},run)
    (run/'run.json').write_text(json.dumps({'config':{'path':str(original),'sha256':m.sha(original)}}))
    with pytest.raises(ValueError,match='beyond mechanical predecessor'):
        m.stage_request(base,pred,evidence,'train',{'state':'claimed','path':None},run)


def test_timing_and_formal_cli():
    m=module()
    assert m.POLL_SECONDS>=900
    for hours in (0,-1,201,float('inf')):
        with pytest.raises(ValueError):m.deadline_seconds(hours)
    assert m.deadline_seconds(200)==720000
    import subprocess
    for cmd,status in [('dry',0),('invalid',2)]:
        run=subprocess.run([sys.executable,'-B','-m','ops.tools.remote.p_first_batch_chain',cmd],capture_output=True,text=True)
        assert run.returncode==status
        if not status:assert json.loads(run.stdout)['submitted'] is False


def test_dry_metadata_requires_complete_job_and_ops_identity():
    m=module()
    metadata={'env':{'NUM':'1'},'cwd':'code','git':{'commit':'fixed'},'overlay':None,'pinned_files':{},'config':{'path':'request'},'inputs':[]}
    spec={'run_id':'run','kind':'ops','argv':['original'],'seed':42,'stop_mode':'signal','cpu_cores':1,'wall_hours':1.0,'no_admission':False,'pinned_manifest':None,'pinned_files':[]}
    expected={**metadata,'env':{'NUM':'1','VKGR_RUN_ID':'run','VKGR_PROGRESS_PATH':'progress'},'schema':'vkgr-run-v1','run_id':'run','kind':'ops','argv':['original'],'seed':42,'stop_mode':'signal','ops_source':{'fixed':True},'resources':{'cpu_cores':1,'wall_seconds':3600.0}}
    launcher=SimpleNamespace(RESOURCE_ARGS=('cpu_cores',),build_parser=lambda:SimpleNamespace(parse_args=lambda a:None),spec_from_args=lambda a:spec,prepare=lambda s:metadata,_ops_source=lambda:{'fixed':True})
    assert m.prepare_exact({'launcher_argv':[],'expected_job':expected},launcher)[0] is spec
    for field,value in [('argv',['drift']),('seed',2024),('ops_source',{'fixed':False}),('inputs',[{}])]:
        with pytest.raises(ValueError):m.prepare_exact({'launcher_argv':[],'expected_job':{**expected,field:value}},launcher)


@pytest.mark.parametrize('fault',['','rejected','timeout','disappeared'])
def test_chain_submits_once_per_stage_and_observes_claimed(tmp_path,monkeypatch,fault):
    m=module()
    import analysis.ur001_derive_cli as derive
    import analysis.new_instance_prepare as environment
    ops=tmp_path/'ops';evidence=tmp_path/'evidence';evidence.mkdir()
    config={'seed':42,'science':'unchanged'}
    base={'predecessor':None,'config':config,'native_root':str(tmp_path/'native'),'source':{'fixed':True},'runtime':{},'lock':{},
          'ops_root':str(ops),'output_root':str(tmp_path/'output'),'run_ids':{'preflight':'pf','train':'train','verify':'verify'},
          'job_contract':{s:{'ops_source':{'fixed':True},'resources':{'wall_seconds':345600}} for s in ('preflight','train','verify')}}
    backend=SimpleNamespace(read_transport=lambda *a:base,native_for=lambda p:SimpleNamespace(default_config=lambda:config),record_path=lambda a:None,
                            plan=lambda stage,*a:{'stage':stage,'expected_job':{'run_id':stage}})
    monkeypatch.setattr(derive,'source_identity',lambda:base['source']);monkeypatch.setattr(environment,'runtime',lambda p:{})
    dag=SimpleNamespace(check_consumer=lambda *a:None);launcher=SimpleNamespace(_ops_source=lambda:{'fixed':True})
    completed=[]
    def complete(*args):
        stage=args[7];completed.append(stage);return {'stage':stage},{'binding':'fixed'}
    monkeypatch.setattr(m,'complete',complete)
    monkeypatch.setattr(m,'prepare_exact',lambda plan,l:({'run_id':plan['stage']},{'dry':True}))
    submissions=[];sleeps=[];clock=[0]
    def submit(root,spec):
        submissions.append(spec['run_id']);path=root/'queue/pending'/('000-'+spec['run_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(spec));return path
    def sleep(seconds):
        sleeps.append(seconds);clock[0]+=seconds
        for pending in list((ops/'queue/pending').glob('*.json')):
            target=ops/'queue'/('rejected' if fault=='rejected' else 'claimed')/pending.name;target.parent.mkdir(parents=True,exist_ok=True);pending.rename(target)
            if fault=='rejected':target.with_suffix('.reason').write_text('synthetic rejection');continue
            run_id=pending.stem.split('-')[-1]
            if fault=='timeout':continue
            run=ops/'runs'/run_id;run.mkdir(parents=True)
            (run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':run_id,'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}))
    args=SimpleNamespace(inputs=tmp_path/'inputs',request=tmp_path/'request',request_sha256='a'*64,evidence=evidence,deadline_hours=0.5 if fault=='timeout' else 1.0)
    if fault=='disappeared':
        original=tmp_path/'original-train-request.json';original.write_text(json.dumps({**base,'predecessor':{'stage':'preflight'}}))
        pending=ops/'queue/pending/000-train.json';pending.parent.mkdir(parents=True);pending.write_text(json.dumps({'config':str(original),'run_id':'train'}))
        def prepare(plan,launcher):
            if pending.exists():pending.unlink()
            return {'run_id':plan['stage']},{'dry':True}
        monkeypatch.setattr(m,'prepare_exact',prepare)
    if fault:
        with pytest.raises(TimeoutError if fault=='timeout' else ValueError):m.run_chain(args,backend,None,dag,launcher,SimpleNamespace(submit=submit),now=lambda:clock[0],sleep=sleep)
        assert submissions==([] if fault=='disappeared' else ['train']) and completed==['preflight']
    else:
        result=m.run_chain(args,backend,None,dag,launcher,SimpleNamespace(submit=submit),now=lambda:clock[0],sleep=sleep)
        assert submissions==['train','verify'] and completed==['preflight','train','verify']
        assert result['status']=='verified'
        assert [r['claimed'] for r in result['observations']]==[False,True,False,True]
    assert (not sleeps if fault=='disappeared' else bool(sleeps)) and all(s>=900 for s in sleeps)


@pytest.mark.parametrize('fault',['','binding','input','stdout_end','job'])
def test_complete_authenticates_original_job_receipt_and_inputs(own_job_fixture,actual_dag_module,monkeypatch,fault):
    m=module()
    from analysis import p_full_engineering as helper
    backend,value,records,request,digest,_=own_job_fixture
    value['source']={'commit':value['source']['commit'],'files':{'native.py':'a'*64}}
    value['runtime']={'runtime':{'python':'fixture'},'lock_sha256':'b'*64}
    value['output_root']=str(request.parent/'output')
    stage='preflight';run=Path(value['ops_root'])/'runs'/value['run_ids'][stage];(run/'logs').mkdir(parents=True)
    job=backend.job_contract(value,records,request,digest,stage)
    (run/'run.json').write_text(json.dumps(job))
    terminal={k:None for k in actual_dag_module.TERMINAL_KEYS};terminal.update(schema='vkgr-terminal-v1',run_id=run.name,status='succeeded',exit_code=0,wall_cap_exceeded=False)
    (run/'terminal.json').write_text(json.dumps(terminal))
    binding={'fixed':True}
    receipt={'schema':'p-full-backend-receipt-v1','stage':stage,'run_id':run.name,'source':value['source'],'runtime':value['runtime'],
       'request_sha256':digest,'run_manifest_sha256':m.sha(run/'run.json'),'verified':True,'engineering':False,'test_read':False,
       'training':False,'scientific_verified':False,'status':'verified','binding':binding}
    path=backend.stage_receipt_path(Path(value['output_root']),stage);path.parent.mkdir(parents=True);path.write_text(json.dumps(receipt))
    (run/'logs/stdout.log').write_text(json.dumps(receipt)+'\n')
    monkeypatch.setattr(backend,'stage_inputs',lambda *a:records)
    def closure(*a):
        if fault=='stdout_end':(run/'logs/stdout.log').write_text('{}\n')
    monkeypatch.setattr(backend,'close_context',closure)
    if fault=='input':Path(records[-1]['path']).write_bytes(b'drift')
    if fault=='job':(run/'run.json').write_text(json.dumps({**job,'seed':2024}))
    expected_binding={'fixed':False} if fault=='binding' else binding
    if fault:
        with pytest.raises(ValueError):m.complete(backend,helper,actual_dag_module,value,Path(records[1]['path']),request,digest,stage,expected_binding)
    else:
        predecessor,found=m.complete(backend,helper,actual_dag_module,value,Path(records[1]['path']),request,digest,stage,binding)
        assert predecessor['pin']['receipt']==receipt and found==binding


def test_actual_launcher_prepare_and_original_scheduler_queue_only(tmp_path):
    import os,subprocess
    from test_p_full_backend import ROOT,test_cli_plan_inputs_feed_real_launcher_prepare
    root=tmp_path/'chain';root.mkdir();test_cli_plan_inputs_feed_real_launcher_prepare(root)
    script=r'''import importlib.util,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);driver=Path(sys.argv[2]);deployed=root/'deployed';ops=root/'ops-source'
subprocess.run(['git','init','-q',str(ops)],check=True)
subprocess.run(['git','-C',str(ops),'add','.'],check=True)
subprocess.run(['git','-C',str(ops),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixed original OPS fixture'],check=True)
sys.path.insert(0,str(ops/'ops'));from vkgr_ops import launcher,scheduler
spec=importlib.util.spec_from_file_location('p_first_driver_fixture',driver);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
request=root/'request.json';value=json.loads(request.read_bytes())
for contract in value['job_contract'].values():contract['ops_source']=launcher._ops_source()
request.write_text(json.dumps(value));digest=m.sha(request)
run=subprocess.run([sys.executable,'-B','-m','analysis.p_full_training_backend','plan','--stage','preflight','--inputs',str(root/'inputs.json'),'--request',str(request),'--request-sha256',digest],cwd=deployed,capture_output=True,text=True)
assert run.returncode==0,run.stdout+run.stderr
plan=json.loads(run.stdout);spec,metadata=m.prepare_exact(plan,launcher)
opsroot=Path(value['ops_root']);assert m.queue_state(opsroot,spec['run_id'])['state']=='absent'
path=scheduler.submit(opsroot,spec)
state=m.queue_state(opsroot,spec['run_id']);assert state['state']=='pending' and state['path']==path
assert not (opsroot/'runs'/spec['run_id']).exists()
assert json.loads(path.read_bytes())['argv']==plan['expected_job']['argv']
print(json.dumps({'verified':True,'queue':'pending','claimed':False,'launched':False}))
'''
    got=subprocess.run([sys.executable,'-B','-c',script,str(root),str(ROOT/'ops/tools/remote/p_first_batch_chain.py')],
       cwd=root/'deployed',env={**os.environ,'PYTHONPATH':str(root/'deployed')},capture_output=True,text=True)
    assert got.returncode==0,got.stdout+got.stderr
    assert json.loads(got.stdout)['launched'] is False


def test_independent_driver_identity_pins_own_git_bytes(tmp_path,monkeypatch):
    import subprocess
    m=module();root=tmp_path/'driver';path=root/'ops/tools/remote/p_first_batch_chain.py';path.parent.mkdir(parents=True);path.write_text('fixed driver\n')
    subprocess.run(['git','init','-q',str(root)],check=True);subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixed driver'],check=True)
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'__file__',str(path))
    identity=m.driver_identity(m.sha(path));assert identity['sha256']==m.sha(path)
    with pytest.raises(ValueError):m.driver_identity('0'*64)
    path.write_text('drift\n')
    with pytest.raises(ValueError,match='clean fixed'):m.driver_identity(identity['sha256'],identity)


def test_main_imports_existing_consumer_without_equating_driver_commit(tmp_path,monkeypatch,capsys):
    import types,analysis
    m=module();driver=tmp_path/'driver';driver.mkdir();consumer=tmp_path/'consumer';consumer.mkdir()
    evidence=tmp_path/'evidence';request=tmp_path/'request';request.write_text('{}');inputs=tmp_path/'inputs';inputs.write_text('{}')
    source={'commit':'consumer-source'}
    value={'source':source,'native_root':str(tmp_path/'native'),'stage2':str(tmp_path/'stage2'),'output_root':str(tmp_path/'out'),'ops_root':str(tmp_path/'ops')}
    backend=SimpleNamespace(ROOT=consumer,checked_path=lambda p:Path(p),read_transport=lambda *a:value)
    monkeypatch.setattr(analysis,'p_full_training_backend',backend,raising=False)
    monkeypatch.setattr(analysis,'p_full_engineering',SimpleNamespace(),raising=False)
    monkeypatch.setattr(analysis,'producer_dag_binding',SimpleNamespace(),raising=False)
    ops=types.ModuleType('vkgr_ops');ops.launcher=object();ops.scheduler=object();monkeypatch.setitem(sys.modules,'vkgr_ops',ops)
    identity={'commit':'different-driver-source','sha256':'f'*64}
    monkeypatch.setattr(m,'ROOT',driver);monkeypatch.setattr(m,'driver_identity',lambda *a:identity)
    monkeypatch.setattr(m.os,'getppid',lambda:1);monkeypatch.chdir(tmp_path)
    def run(args,*a,driver_check):
        driver_check();assert Path.cwd()==consumer
        assert sys.path[:2]==[str(consumer),str(consumer/'src')]
        return {'status':'verified'}
    monkeypatch.setattr(m,'run_chain',run)
    old_path=list(sys.path)
    try:
        assert m.main(['run','--inputs',str(inputs),'--request',str(request),'--request-sha256','a'*64,
            '--consumer-code',str(consumer),'--ops-code',str(tmp_path/'ops-code'),'--evidence',str(evidence),
            '--deadline-hours','200','--driver-sha256','f'*64])==0
    finally:sys.path[:]=old_path
    summary=json.loads((evidence/'summary.json').read_bytes())
    assert summary['driver_source']['commit']!=summary['consumer_source']['commit']
    assert not list(consumer.iterdir())
