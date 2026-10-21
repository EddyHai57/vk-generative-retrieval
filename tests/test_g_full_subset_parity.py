"""Isolated engineering comparison; no real data or checkpoint reads."""
import json
import subprocess
import sys
import pytest
from analysis import g_full_subset_parity as parity


def test_cli_closed_dry_and_holdout(tmp_path):
    path=tmp_path/'request.json'
    value={'schema':parity.SCHEMA,'stage':'compare','runner_root':'/runner','runner_source':{'commit':'a'*40,'files':{}},'producer_root':'/producer','a1_request':{'path':'/a1.json','sha256':parity.A1_REQUEST_SHA},'scratch_root':'/scratch','steps':2,'device':'cuda','ops_root':'/ops','run_id':'g-subset-parity','resources':{'cpu_cores':1,'ram_mib':1024,'vram_mib':1024,'disk_gib':1,'wall_seconds':3600.0},'runtime':{'fixture':True}}
    path.write_text(json.dumps(value))
    pin=parity.digest(path.read_bytes())
    run=subprocess.run([sys.executable,'-B','-m',parity.MODULE,'dry','--request',str(path),'--request-sha256',pin],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['data_opened'] is False
    value['a1_request']['path']='/week_26/holdout.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='holdout'):parity.read_request(path,parity.digest(path.read_bytes()))
    run=subprocess.run([sys.executable,'-B','-m',parity.MODULE,'illegal'],capture_output=True)
    assert run.returncode!=0


def test_exact_recursive_state_digest():
    import torch
    a={'tensor':torch.tensor([0.,1.]),'nested':[3,1.]}
    assert parity.state_digest(a)==parity.state_digest(a)
    b={'tensor':torch.tensor([-0.,1.]),'nested':[3,1.]}
    assert parity.state_digest(a)!=parity.state_digest(b)
    assert parity.state_digest(a)!=parity.state_digest({**a,'nested':[3,1]})


def test_original_vs_lazy_training_and_decode(tmp_path):
    import numpy as np
    from vkgr import sid_v2d, sid_structure_a1_run as original
    from vkgr.generative_data_v2 import ListExamples
    from analysis import g_full_training_backend as backend
    codec=sid_v2d.build_codec_d(np.array([1,2,3],dtype=np.uint32),np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0]],dtype=np.uint32),48,3)
    rows=[{'example_id':i,'user_id':7,'history':[1,3],'target':2,'history_positions':[[0,0],[0,1]],'target_position':[0,2],'past_likes':[1]} for i in range(3)]
    raw=b''.join(json.dumps(r).encode()+b'\n' for r in rows);path=tmp_path/'train.jsonl';path.write_bytes(raw)
    lazy=backend.lazy_examples(path,tmp_path/'offsets',3,parity.digest(raw),lambda:None)
    config=backend.registered_config(42,'a'*64)
    # Tiny fixture only: real entry separately requires unchanged registered config.
    config.update(d_model=8,heads=1,head_dim=8,encoder_layers=1,decoder_layers=1,ff=16,microbatch=2,effective_batch=2)
    result,_=parity.compare_training(ListExamples(rows),lazy,codec,config,steps=2,device='cpu',native=original.__dict__)
    assert result['verified'] is True
    assert len(result['steps'])==2
    assert {'tensors','loss','parameters','optimizer','cursor','rng','scores'}<=set(result['steps'][0])
    bad=ListExamples([{**r,'target':1} for r in rows])
    with pytest.raises(ValueError,match='parity'):parity.compare_training(bad,lazy,codec,config,steps=1,device='cpu',native=original.__dict__)


def test_evaluation_signature_only_excludes_runtime_and_locations(tmp_path):
    shard=tmp_path/'shard.json';row={'ordinal':0,'query_sha256':'a'*64,'items':[2,1],'audit':{'invalid':0,'deadend':0,'collision':0},'latency_seconds':1.}
    shard.write_text(json.dumps({'rows':[row]}))
    manifest={'shards':[{'path':str(shard)}],'aggregate':{'recall':1.},'complete':True,'query_count':1,'completed_queries':1,'computation_sha256':'b'*64,'observations':{'rss':100}}
    before=parity.evaluation_signature({'manifest':manifest})
    row['latency_seconds']=99.;shard.write_text(json.dumps({'rows':[row]}));manifest['observations']['rss']=200
    assert parity.evaluation_signature({'manifest':manifest})==before
    row['items']=[1,2];shard.write_text(json.dumps({'rows':[row]}))
    assert parity.evaluation_signature({'manifest':manifest})!=before
    row['items']=[2,1];shard.write_text(json.dumps({'rows':[row]}));manifest['aggregate']['recall']=0.
    assert parity.evaluation_signature({'manifest':manifest})!=before


def test_original_complete_evaluator_with_diagnostic_context(tmp_path):
    import numpy as np
    from vkgr import sid_v2d,sid_structure_a1_run as original,generative_real as gr,sid_depth
    from vkgr.generative_data_v2 import ListExamples
    from analysis import g_full_training_backend as backend
    codec=sid_v2d.build_codec_d(np.array([1,2,3],dtype=np.uint32),np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0]],dtype=np.uint32),48,3)
    query=[{'user_id':7,'target':2,'oov':False,'history':[1]}]
    stats=backend.slice_statistics({'schema_version':1,'popularity_thresholds':{'median':1,'p90':4},'item_like_counts':{'1':1,'2':4,'3':0},'user_like_counts':{'7':5},'user_buckets':[1,5,20],'definition':'train_like_counts_only_no_holdout_boundaries'},catalog=[1,2,3],queries=query,manifest_sha256='a'*64,statistics_sha256='b'*64,binding_sha256='c'*64)
    contexts=[backend.diagnostic_context(ListExamples([{'history':[1],'target':2}]),codec,query,stats,{'side':i}) for i in (0,1)]
    config=backend.registered_config(42,'a'*64);config.update(d_model=8,heads=1,head_dim=8,encoder_layers=1,decoder_layers=1,ff=16)
    signatures=[]
    for i,ctx in enumerate(contexts):
        gr._seed(42);model=original.make_model_arch_d(codec,config,device='cpu');model.eval()
        kind,result=original._evaluator_d(None)(model,ctx,sid_depth.SIDTokensLD(codec,50),config,{},tmp_path/str(i),'a'*64,lambda:None,1)
        assert kind=='query_evaluation'
        signatures.append(parity.evaluation_signature(result))
    assert signatures[0]==signatures[1]


@pytest.mark.parametrize('stage',['preflight','compare'])
def test_real_launcher_metadata_and_dry(tmp_path,stage):
    import os
    from analysis import g_full_training_backend as backend
    from analysis.a40_ops import load_ops
    def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    def clone(name,ref):
        out=tmp_path/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(backend.ROOT),str(out)],check=True,capture_output=True);git(out,'checkout','-q','--detach',ref);return out
    code=clone('CODE','1ba8043b');ops=clone('OPS','5840e6b686ec9651314a9e28a1a2c0befcc05158')
    name='analysis/g_full_subset_parity.py';(code/name).write_bytes((backend.ROOT/name).read_bytes());(code/'docs').mkdir(exist_ok=True);(code/'docs/p3c_validation.md').write_bytes(b'Synthetic public fixture.\n');git(code,'add',name,'docs/p3c_validation.md');git(code,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','Local parity fixture')
    # Same data-channel overlay shape as formal deployment; synthetic bytes only.
    relative='docs/p3c_validation.md';public=tmp_path/'OPS_PUBLIC';public.mkdir();(public/'p3c_validation.md').write_bytes((code/relative).read_bytes());overlay=b'Synthetic parity data-channel fixture.\n';(code/relative).write_bytes(overlay)
    sha=git(code,'rev-parse','HEAD');overlay_file=tmp_path/'overlay.json';overlay_file.write_text(json.dumps({'schema':'vkgr-pinned-v1','base_commit':sha,'target_commit':sha,'files':{},'overrides':{relative:{'sha256':parity.digest(overlay),'reason':'Synthetic layout only'}}}))
    request=tmp_path/'request.json';request.write_text(json.dumps({'schema':parity.SCHEMA,'stage':stage,'runner_root':str(code),'runner_source':{'commit':sha,'files':{}},'producer_root':str(tmp_path/'producer'),'a1_request':{'path':str(tmp_path/'a1.json'),'sha256':parity.A1_REQUEST_SHA},'scratch_root':str(tmp_path/'scratch'),'steps':2,'device':'cuda','ops_root':'/ops','run_id':'g-subset-parity','resources':{'cpu_cores':1,'ram_mib':1024,'vram_mib':1024,'disk_gib':1,'wall_seconds':3600.0},'runtime':{'fixture':True}}));pin=parity.digest(request.read_bytes())
    argv=[sys.executable,'-B','-m',parity.MODULE,stage,'--request',str(request),'--request-sha256',pin]
    flags=['--root',str(tmp_path/'OPS_ROOT'),'--run-id','g-full-subset-parity-dry','--kind','ops','--cwd',str(code),'--config',str(request),'--seed','42','--pinned-manifest',str(overlay_file),'--env','PYTHONPATH='+str(code/'src'),'--cpu-cores','1','--ram-mib','1024','--vram-mib','0','--disk-gib','1','--wall-hours','1']
    _,launcher,_,_=load_ops(ops);spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]));metadata=launcher.prepare(spec)
    assert metadata['git']['commit']==sha
    dry_argv=list(spec['argv']);dry_argv[4]='dry'
    run=subprocess.run(dry_argv,cwd=code,env={**os.environ,'PYTHONPATH':str(code/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['data_opened'] is False
    assert metadata['overlay']['overrides']=={relative:parity.digest(overlay)}


def own_fixture(tmp_path,monkeypatch):
    import os
    root=tmp_path/'runner';root.mkdir();(root/'entry.py').write_text('value=1\n')
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    git('init','-q');git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','Synthetic diagnostic source')
    a1=tmp_path/'a1.json';a1.write_text('{}');a1pin=parity.digest(a1.read_bytes());monkeypatch.setattr(parity,'A1_REQUEST_SHA',a1pin)
    request=tmp_path/'diagnostic.json';ops=tmp_path/'ops';runroot=ops/'runs'/'g-subset-parity-fixture';runroot.mkdir(parents=True)
    v={'schema':parity.SCHEMA,'stage':'compare','runner_root':str(root),'runner_source':{'commit':git('rev-parse','HEAD'),'files':{'entry.py':parity.digest((root/'entry.py').read_bytes())}},'producer_root':str(tmp_path/'producer'),'a1_request':{'path':str(a1),'sha256':a1pin},'scratch_root':str(tmp_path/'scratch'),'steps':2,'device':'cuda','ops_root':str(ops),'run_id':runroot.name,'resources':{'cpu_cores':1,'ram_mib':1024,'vram_mib':1024,'disk_gib':1,'wall_seconds':3600.0},'runtime':{'fixture':True}}
    request.write_bytes(parity.encoded(v));pin=parity.digest(request.read_bytes())
    env={'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1','VKGR_RUN_ID':runroot.name,'VKGR_PROGRESS_PATH':str(runroot/'progress.json')}
    for k,val in env.items():monkeypatch.setenv(k,val)
    rec={'path':str(request),'sha256':pin};inputs=[rec,v['a1_request']]
    run={'schema':'vkgr-run-v1','run_id':runroot.name,'kind':'ops','cwd':str(root),'argv':[sys.executable,'-B','-m',parity.MODULE,'compare','--request',str(request),'--request-sha256',pin],'seed':42,'git':{'commit':v['runner_source']['commit'],'dirty':False,'dirty_files':[]},'config':rec,'inputs':[{**r,'bytes':len(__import__('pathlib').Path(r['path']).read_bytes())} for r in inputs],'env':env,'resources':json.loads(parity.encoded(v['resources'])),'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},'overlay':None,'pinned_files':{},'stop_mode':'signal'}
    path=runroot/'run.json';path.write_bytes(parity.encoded(run));return v,request,pin,path,run


@pytest.mark.parametrize('mutation',['argv','input','request','decoded-request','source','run','env'])
def test_own_job_rejects_mismatched_or_changed_binding(tmp_path,monkeypatch,mutation):
    v,request,pin,path,run=own_fixture(tmp_path,monkeypatch)
    before=parity.own_binding(v,request,pin)
    assert before['request_sha256']==pin and before['run_manifest_sha256']==parity.digest(path.read_bytes())
    if mutation=='argv':run['argv'][4]='dry'
    elif mutation=='input':run['inputs'][0]['sha256']='f'*64
    elif mutation=='request':request.write_bytes(parity.encoded({**v,'steps':3}))
    elif mutation=='decoded-request':v['steps']=3
    elif mutation=='source':(__import__('pathlib').Path(v['runner_root'])/'entry.py').write_text('changed=1\n')
    elif mutation=='run':run['resources']['vram_mib']=2
    else:monkeypatch.setenv('VKGR_RUN_ID','wrong')
    if mutation in ('argv','input','run'):path.write_bytes(parity.encoded(run))
    with pytest.raises(ValueError):parity.own_binding(v,request,pin)


@pytest.mark.parametrize('mutation',['run','request','source'])
def test_end_snapshot_rejects_post_read_change(tmp_path,monkeypatch,mutation):
    v,request,pin,path,run=own_fixture(tmp_path,monkeypatch)
    before=parity.own_binding(v,request,pin)
    assert parity.close_own_binding(v,request,pin,before)==before
    if mutation=='run':run['created_at']='post-read-change';path.write_bytes(parity.encoded(run))
    elif mutation=='request':request.write_bytes(parity.encoded({**v,'steps':3}))
    else:(__import__('pathlib').Path(v['runner_root'])/'entry.py').write_text('changed=1\n')
    with pytest.raises(ValueError):parity.close_own_binding(v,request,pin,before)


def test_execute_rejects_own_job_before_original_import_or_data(tmp_path,monkeypatch):
    v,request,pin,path,run=own_fixture(tmp_path,monkeypatch)
    run['argv'][4]='dry';path.write_bytes(parity.encoded(run))
    # This narrow runner has no original native producer module. The own-job
    # rejection must happen first, without importing/opening producer data.
    with pytest.raises(ValueError,match='launcher metadata'):
        parity.execute(v,request,pin)
    assert not __import__('pathlib').Path(v['scratch_root']).exists()


def test_publish_authenticates_own_snapshot_and_separates_upstream(tmp_path,monkeypatch):
    v,request,pin,path,run=own_fixture(tmp_path,monkeypatch)
    before=parity.own_binding(v,request,pin)
    scratch=__import__('pathlib').Path(v['scratch_root']);scratch.mkdir()
    run['created_at']='changed-after-comparison';path.write_bytes(parity.encoded(run))
    with pytest.raises(ValueError):parity.publish_report(v,request,pin,before,{'verified':True})
    assert not (scratch/'parity-receipt.json').exists()
    run.pop('created_at');path.write_bytes(parity.encoded(run))
    result=parity.publish_report(v,request,pin,before,{'verified':True})
    assert result['request_sha256']==pin
    assert result['diagnostic']['run_manifest_sha256']==parity.digest(path.read_bytes())
    assert result['original_a1_request']==v['a1_request']
    assert result['request_sha256']!=v['a1_request']['sha256']


def staged_fixture(tmp_path,monkeypatch,stage):
    """Separate synthetic launcher contracts; no producer data."""
    from pathlib import Path
    v,request,pin,oldpath,run=own_fixture(tmp_path,monkeypatch)
    v['stage']=stage;v['run_id']+='-'+stage
    runroot=oldpath.parent.parent/v['run_id'];runroot.mkdir()
    v['scratch_root']=str(tmp_path/('scratch-'+stage))
    request.write_bytes(parity.encoded(v));pin=parity.digest(request.read_bytes())
    rec={'path':str(request),'sha256':pin}
    run.update(run_id=v['run_id'],config=rec,argv=[sys.executable,'-B','-m',parity.MODULE,stage,'--request',str(request),'--request-sha256',pin])
    run['inputs']=[{**r,'bytes':Path(r['path']).stat().st_size}for r in (rec,v['a1_request'])]
    run['env'].update(VKGR_RUN_ID=v['run_id'],VKGR_PROGRESS_PATH=str(runroot/'progress.json'))
    for key,value in run['env'].items():monkeypatch.setenv(key,value)
    path=runroot/'run.json';path.write_bytes(parity.encoded(run))
    return v,request,pin,path,run


@pytest.mark.parametrize('stage',['preflight','compare'])
def test_stage_specific_job_contract_cannot_be_interchanged(tmp_path,monkeypatch,stage):
    v,request,pin,path,run=staged_fixture(tmp_path,monkeypatch,stage)
    assert parity.read_request(request,pin)['stage']==stage
    assert parity.own_binding(v,request,pin)['stage']==stage
    run['argv'][4]='compare'if stage=='preflight'else'preflight'
    path.write_bytes(parity.encoded(run))
    with pytest.raises(ValueError,match='launcher metadata'):parity.own_binding(v,request,pin)


def test_preflight_cli_rejects_stage_mismatch_before_data(tmp_path,monkeypatch):
    frozen=parity.A1_REQUEST_SHA
    v,request,pin,*_=staged_fixture(tmp_path,monkeypatch,'compare')
    v['a1_request']['sha256']=frozen;request.write_bytes(parity.encoded(v));pin=parity.digest(request.read_bytes())
    result=subprocess.run([sys.executable,'-B','-m',parity.MODULE,'preflight','--request',str(request),'--request-sha256',pin],capture_output=True,text=True)
    assert result.returncode!=0
    assert json.loads(result.stdout)['error']=='diagnostic CLI/request stage differs'
    assert not __import__('pathlib').Path(v['scratch_root']).exists()


@pytest.mark.parametrize('engineering,synthetic',[(False,False),(True,False),(False,True)])
def test_preflight_stops_after_original_real_context_without_models(tmp_path,monkeypatch,engineering,synthetic):
    import types
    from pathlib import Path
    v,request,pin,path,run=staged_fixture(tmp_path,monkeypatch,'preflight')
    producer=Path(v['producer_root'])
    subprocess.run(['git','clone','-q','--shared',v['runner_root'],str(producer)],check=True)
    req={'source':v['runner_source'],**{k:str(tmp_path/k)for k in ('data_root','output_root','scratch_root','sid_root','ops_root','ops_code')}}
    context=types.SimpleNamespace(binding={'synthetic':synthetic,'inputs':'original'},examples=[{'history':[1],'target':2}],queries=[{'history':[1],'target':2}])
    calls=[]
    def contexts(*args,**kwargs):calls.append('contexts');return context,engineering,None
    def forbidden(*args,**kwargs):raise AssertionError('preflight called scientific/model path')
    original=types.SimpleNamespace(load=lambda *a:req,identity=lambda *a:(req['source'],v['runtime']),input_check=lambda *a:calls.append('inputs'),engine_for=lambda *a:({'_contexts':contexts,'make_model_arch_d':forbidden,'train_batch_d':forbidden,'_evaluator_d':forbidden},[],{},None),core=types.SimpleNamespace(registered_config=lambda *a:{'seed':42}),sid_manifest=lambda *a:{'sha256':'a'*64})
    monkeypatch.setattr(parity,'PRODUCER_COMMIT',req['source']['commit'])
    monkeypatch.setattr(parity,'BASE_COMMIT',req['source']['commit'])
    monkeypatch.setattr(parity,'NEW_FILES',())
    monkeypatch.setattr(parity,'__file__',str(Path(v['runner_root'])/'analysis/g_full_subset_parity.py'))
    monkeypatch.setitem(sys.modules,'analysis.sgns_native_subset_a1',original)
    monkeypatch.setattr(parity,'compare_training',forbidden)
    if engineering or synthetic:
        with pytest.raises(ValueError,match='original real subset context'):
            parity.execute(v,request,pin)
        assert not (Path(v['scratch_root'])/'parity-receipt.json').exists()
        return
    result=parity.execute(v,request,pin)
    assert calls==['inputs','contexts']
    assert result['schema']=='g-full-subset-parity-preflight-v1'
    assert result['stage']=='preflight'and result['verified']is True
    assert result['engineering']is False and result['training']is False
    assert result['model_created']is False and result['optimizer_created']is False and result['evaluation_executed']is False
    assert result['full_preflight_passed']is False
    assert result['context_sha256']==parity.digest(parity.encoded(context.binding))
    assert {p.name for p in Path(v['scratch_root']).iterdir()}=={'parity-receipt.json'}


def test_preflight_formal_cli_with_real_launcher_and_synthetic_producer(tmp_path):
    """Actual module argv/launcher; producer stub is a public synthetic fixture."""
    import os
    from pathlib import Path
    from analysis.a40_ops import load_ops
    def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    def clone(name,ref):
        root=tmp_path/name
        subprocess.run(['git','clone','-q','--shared','--no-checkout',str(Path(__file__).resolve().parents[1]),str(root)],check=True,capture_output=True)
        git(root,'checkout','-q','--detach',ref);return root
    producer=clone('producer',parity.BASE_COMMIT);runner=clone('runner',parity.BASE_COMMIT);ops=clone('ops-source','5840e6b686ec9651314a9e28a1a2c0befcc05158')
    stub='''import hashlib,json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def load(path,pin):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('fixture input pin')
 return json.loads(raw)
def identity(v):return v['source'],v['runtime']
def input_check(v):load(Path(v['input']['path']),v['input']['sha256'])
def contexts(v,*a,**kw):
 row=load(Path(v['input']['path']),v['input']['sha256'])
 return SimpleNamespace(binding={'synthetic':False,'fixture_sha256':v['input']['sha256']},examples=[row],queries=[row]),False,None
def forbidden(*a,**kw):raise AssertionError('fixture forbids model/optimizer/training/scoring')
def engine_for(v,*a):return {'_contexts':contexts,'make_model_arch_d':forbidden,'train_batch_d':forbidden,'_evaluator_d':forbidden},[v],{},None
core=SimpleNamespace(registered_config=lambda *a:{'seed':42})
def sid_manifest(v):return {'sha256':'a'*64}
'''
    (producer/'analysis/sgns_native_subset_a1.py').write_text(stub);git(producer,'add','analysis/sgns_native_subset_a1.py');git(producer,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','Synthetic input-only producer')
    def ledger(root):return {'commit':git(root,'rev-parse','HEAD'),'files':{n:parity.digest((root/n).read_bytes())for n in git(root,'ls-tree','-r','--name-only','HEAD').splitlines()}}
    data=tmp_path/'input.json';data.write_text('{"history":[1],"target":2}')
    runtime={'fixture':'same-interpreter'}
    req={'source':ledger(producer),'runtime':runtime,'input':{'path':str(data),'sha256':parity.digest(data.read_bytes())},**{k:str(tmp_path/('original-'+k))for k in ('data_root','output_root','scratch_root','sid_root','ops_root','ops_code')}}
    upstream=tmp_path/'upstream.json';upstream.write_bytes(parity.encoded(req));upstream_sha=parity.digest(upstream.read_bytes())
    repo=Path(__file__).resolve().parents[1]
    for name in parity.NEW_FILES:
        raw=(repo/name).read_bytes()
        if name=='analysis/g_full_subset_parity.py':
            raw=raw.replace(parity.A1_REQUEST_SHA.encode(),upstream_sha.encode()).replace(parity.PRODUCER_COMMIT.encode(),req['source']['commit'].encode())
        (runner/name).parent.mkdir(parents=True,exist_ok=True);(runner/name).write_bytes(raw)
    git(runner,'add',*parity.NEW_FILES);git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','Synthetic preflight deployment')
    opsroot=tmp_path/'OPS';rid='synthetic-g-parity-preflight';runroot=opsroot/'runs'/rid;runroot.mkdir(parents=True)
    value={'schema':parity.SCHEMA,'stage':'preflight','runner_root':str(runner),'runner_source':ledger(runner),'producer_root':str(producer),'a1_request':{'path':str(upstream),'sha256':upstream_sha},'scratch_root':str(tmp_path/'new-scratch'),'steps':2,'device':'cuda','ops_root':str(opsroot),'run_id':rid,'resources':{'cpu_cores':1,'ram_mib':1024,'vram_mib':1024,'disk_gib':1,'wall_seconds':3600.0},'runtime':runtime}
    request=tmp_path/'diagnostic.json';request.write_bytes(parity.encoded(value));pin=parity.digest(request.read_bytes());argv=[sys.executable,'-B','-m',parity.MODULE,'preflight','--request',str(request),'--request-sha256',pin]
    env={'PYTHONPATH':str(runner/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1'}
    flags=['--root',str(opsroot),'--run-id',rid,'--kind','ops','--cwd',str(runner),'--config',str(request),'--seed','42','--input',str(request),'--input',str(upstream),'--cpu-cores','1','--ram-mib','1024','--vram-mib','1024','--disk-gib','1','--wall-hours','1']
    for k,val in env.items():flags.extend(['--env',k+'='+val])
    _,launcher,_,_=load_ops(ops);spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]));metadata=launcher.prepare(spec)
    recorded={**metadata['env'],'VKGR_RUN_ID':rid,'VKGR_PROGRESS_PATH':str(runroot/'progress.json')}
    run={'schema':'vkgr-run-v1','run_id':rid,'kind':'ops','cwd':str(runner),'argv':argv,'seed':42,'stop_mode':spec['stop_mode'],'ops_source':launcher._ops_source(),'resources':value['resources'],**metadata,'env':recorded}
    (runroot/'run.json').write_bytes(parity.encoded(run))
    result=subprocess.run(argv,cwd=runner,env={**os.environ,**recorded},capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['stage']=='preflight'and receipt['verified']is True
    assert receipt['diagnostic']['run_manifest_sha256']==parity.digest((runroot/'run.json').read_bytes())
    assert receipt['request_sha256']==pin and receipt['original_a1_request']['sha256']==upstream_sha
    assert receipt['model_created']is False and receipt['evaluation_executed']is False
    assert {p.name for p in Path(value['scratch_root']).iterdir()}=={'parity-receipt.json'}


def test_source_ledger_names_are_not_data_paths(tmp_path,monkeypatch):
    v,request,pin,*_=staged_fixture(tmp_path,monkeypatch,'preflight')
    v['runner_source']['files']['tests/test_holdout_gate.py']='a'*64
    request.write_bytes(parity.encoded(v));pin=parity.digest(request.read_bytes())
    assert parity.read_request(request,pin)['stage']=='preflight'
    v['scratch_root']=str(tmp_path/'holdout-data')
    request.write_bytes(parity.encoded(v));pin=parity.digest(request.read_bytes())
    with pytest.raises(ValueError,match='holdout'):parity.read_request(request,pin)
