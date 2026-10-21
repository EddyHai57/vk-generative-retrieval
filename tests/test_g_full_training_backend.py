"""G full transport boundaries; no real data or training authorization."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis import g_full_training_backend as backend


def test_frozen_config_only_data_identity_changes():
    for seed in (42,2024,2025):
        original=json.loads((backend.ROOT/f'configs/sid_structure/g-structure-a1-s{seed}.json').read_text())
        actual=backend.registered_config(seed,'a'*64)
        assert {k:v for k,v in actual.items() if k not in ('sid','sid_manifest_sha256')}=={k:v for k,v in original.items() if k not in ('sid','sid_manifest_sha256')}
        assert actual['sid']==backend.SID_VERSION
        backend.validate_config(actual)
        actual['microbatch']=128
        with pytest.raises(ValueError):backend.validate_config(actual)


def test_final_sid_unavailable_always_rejects():
    for value in (None,{}, {'verified':True}, {'branching':229,'depth':3}):
        with pytest.raises(ValueError,match='final full SID/trie'):
            backend.require_final_sid(value)


def test_lazy_examples_original_rows_and_order(tmp_path):
    rows=[{'example_id':i,'target':i+1,'history':[1],'history_positions':[[0,0]],'target_position':[0,i+1],'past_likes':[]} for i in range(3)]
    raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
    path=tmp_path/'train.jsonl';path.write_bytes(raw)
    import hashlib
    store=backend.lazy_examples(path,tmp_path/'offsets.bin',len(rows),hashlib.sha256(raw).hexdigest(),lambda:None)
    assert len(store)==3
    assert store.rows(np.array([2,0],dtype=np.int64))==[rows[2],rows[0]]
    assert store.rows([1,1])==[rows[1],rows[1]]
    path.write_bytes(raw+b'{}\n')
    with pytest.raises(ValueError,match='changed'):store.rows([0])


def test_cli_illegal_subcommand():
    result=subprocess.run([sys.executable,'-B','-m','analysis.g_full_training_backend','illegal'],capture_output=True,text=True)
    assert result.returncode!=0


def inputs(tmp_path):
    from analysis.ur001_science_entry import A_ROOT,SCHEMA
    value={'schema':SCHEMA,'train':[str(A_ROOT/f'subsamples/ur0.01/train/week_{w:02}.parquet') for w in range(25)],'validation':str(A_ROOT/'subsamples/ur0.01/validation/week_25.parquet'),'metadata':[str(A_ROOT/'metadata'/n) for n in ('users_metadata.parquet','items_metadata.parquet','item_embeddings.npz')]}
    path=tmp_path/'inputs.json';path.write_text(json.dumps(value));return path,value


def test_cli_dry_and_preflight_are_distinct(tmp_path):
    path,_=inputs(tmp_path)
    for kind,code,status in [('dry',0,'INPUTS_ADMITTED'),('preflight',2,'FINAL_SID_CONTRACT_UNAVAILABLE')]:
        result=subprocess.run([sys.executable,'-B','-m','analysis.g_full_training_backend',kind,'--inputs',str(path)],capture_output=True,text=True)
        assert result.returncode==code,result.stderr
        receipt=json.loads(result.stdout)
        assert receipt['status']==status
        assert receipt['data_opened'] is False
        assert receipt['engineering'] is False
        assert receipt['verified'] is False


def test_holdout_rejected_before_final_sid_or_data(tmp_path,monkeypatch):
    path,value=inputs(tmp_path)
    value['validation']='/workspace/vkgr-ur001-week26-holdout-20261021-v1/secret'
    path.write_text(json.dumps(value))
    monkeypatch.setattr(backend,'require_final_sid',lambda x:pytest.fail('SID gate reached'))
    assert backend.main(['preflight','--inputs',str(path)])==2


def test_lazy_indices_reject_coercion_and_negative(tmp_path):
    path=tmp_path/'rows';path.write_bytes(b'{"history":[1],"target":2}\n')
    import hashlib
    store=backend.lazy_examples(path,tmp_path/'offsets',1,hashlib.sha256(path.read_bytes()).hexdigest(),lambda:None)
    for value in ([-1],[1],[.5],['0'],[True]):
        with pytest.raises(ValueError):store.rows(value)


def test_original_function_identity_and_formal_closure():
    from vkgr import sid_structure_a1_run as original
    namespace=backend.native_functions()
    for name in ('run','verify','_binding','default_config','validate_config'):
        assert namespace[name].__code__ is getattr(original,name).__code__
    for name in ('make_model_arch_d','train_batch_d','_evaluator_d','_gate','_guard','_runtime'):
        assert namespace[name] is getattr(original,name)
    assert namespace['ga'] is original.ga
    assert namespace['gt'] is original.gt
    with pytest.raises(ValueError,match='final full SID/trie'):namespace['_contexts']()


def s0():
    return {'schema_version':1,'popularity_thresholds':{'median':1,'p90':4},'item_like_counts':{'1':1,'2':4,'3':0},'user_like_counts':{'7':5},'user_buckets':[1,5,20],'definition':'train_like_counts_only_no_holdout_boundaries'}


def test_statistics_original_aggregator_and_strict_schema():
    from vkgr import generative_evaluation as ev
    query=[{'user_id':7,'target':2,'oov':False,'history':[1]}]
    kwargs={'catalog':[1,2,3],'queries':query,'manifest_sha256':'a'*64,'statistics_sha256':'b'*64,'binding_sha256':'c'*64}
    stats=backend.slice_statistics(s0(),**kwargs)
    assert type(stats)is ev.SliceStatistics
    expected=ev.SliceStatistics((1,4),stats.item_counts,stats.user_counts,'a'*64,'b'*64,'c'*64)
    assert ev.aggregate_items([[2,1,3]],query,[1,2,3],slice_definition=stats)==ev.aggregate_items([[2,1,3]],query,[1,2,3],slice_definition=expected)
    with pytest.raises(ValueError):backend.slice_statistics(s0(),**{**kwargs,'catalog':[1,1,2,3]})
    with pytest.raises(ValueError):backend.slice_statistics(s0(),**{**kwargs,'catalog':[3,2,1]})
    import copy
    for mutate in (lambda x:x['item_like_counts'].pop('3'),lambda x:x['popularity_thresholds'].update(median=4),lambda x:x['popularity_thresholds'].update(median=True),lambda x:x.update(schema_version=True),lambda x:x.update(user_buckets=[True,5,20])):
        value=copy.deepcopy(s0());mutate(value)
        with pytest.raises(ValueError):backend.slice_statistics(value,**kwargs)


def test_lazy_original_tensors_loss_update_and_scores_bit_equal(tmp_path):
    import hashlib
    import torch
    from vkgr import sid_v2d, sid_depth, generative_real as gr
    from vkgr.generative_data_v2 import ListExamples,sid_targets_l
    from vkgr import generative_arch as ga
    codec=sid_v2d.build_codec_d(np.array([1,2,3],dtype=np.uint32),np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0]],dtype=np.uint32),229,3)
    rows=[{'example_id':0,'user_id':7,'history':[1,3],'target':2,'history_positions':[[0,0],[0,1]],'target_position':[0,2],'past_likes':[1]}, {'example_id':1,'user_id':7,'history':[3],'target':1,'history_positions':[[0,1]],'target_position':[0,2],'past_likes':[]}]
    raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows);path=tmp_path/'rows';path.write_bytes(raw)
    lazy=backend.lazy_examples(path,tmp_path/'offsets',2,hashlib.sha256(raw).hexdigest(),lambda:None)
    tokens=sid_depth.SIDTokensLD(codec,50)
    direct=ListExamples(rows).rows([1,0]);adapted=lazy.rows([1,0])
    assert all(torch.equal(a,b) for a,b in zip(sid_targets_l(direct,tokens),sid_targets_l(adapted,tokens)))
    namespace=backend.native_functions();config=backend.registered_config(42,'a'*64)
    def step(batch):
        gr._seed(42)
        model=namespace['make_model_arch_d'](codec,config,device='cpu')
        optimizer=ga.make_optimizer_arch(model,config)
        result=namespace['train_batch_d'](model,optimizer,batch,codec,tokens,config,'cpu')
        model.eval()
        source,decoder,_=sid_targets_l(batch,tokens)
        with torch.no_grad():scores=model(source,decoder).clone()
        return result,{k:v.clone() for k,v in model.state_dict().items()},scores,torch.random.get_rng_state().clone()
    a,b=step(direct),step(adapted)
    assert a[0]==b[0]
    assert all(torch.equal(a[1][k],b[1][k]) for k in a[1])
    assert torch.equal(a[2],b[2])
    assert torch.equal(a[3],b[3])


def test_real_pinned_launcher_metadata_and_module_dry(tmp_path):
    """Actual OPS code, synthetic transport overlay; no raw data/PF claim."""
    import hashlib
    import os
    repo=backend.ROOT
    def git(root,*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
    def clone(name,ref):
        target=tmp_path/name
        subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(target)],capture_output=True,check=True)
        git(target,'checkout','-q','--detach',ref)
        return target
    code=clone('CODE',git(repo,'rev-parse','HEAD'))
    # New-only candidate bytes are committed in the disposable deployment.
    name='analysis/g_full_training_backend.py'
    (code/name).write_bytes((repo/name).read_bytes())
    (code/'docs/p3c_validation.md').write_bytes(b'Synthetic public contract layout fixture.\n')
    git(code,'add','--',name,'docs/p3c_validation.md')
    if git(code,'diff','--cached','--name-only'):
        git(code,'-c','user.name=Local engineering fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-q','-m','Synthetic G transport candidate')
    ops=clone('OPS','5840e6b686ec9651314a9e28a1a2c0befcc05158')
    from analysis.a40_ops import load_ops
    _,launcher,_,_=load_ops(ops)
    path,_=inputs(tmp_path)
    # Mirror CODE + OPS-preserved public file + external data-channel overlay.
    # Fixture bytes are deliberately not the formal p3c contract.
    public=tmp_path/'OPS_PUBLIC';public.mkdir()
    relative='docs/p3c_validation.md'
    (public/'p3c_validation.md').write_bytes((code/relative).read_bytes())
    overlay=b'Synthetic data-channel layout fixture; not a scientific contract.\n'
    (code/relative).write_bytes(overlay)
    sha=git(code,'rev-parse','HEAD')
    manifest=tmp_path/'overlay.json'
    manifest.write_text(json.dumps({'schema':'vkgr-pinned-v1','base_commit':sha,'target_commit':sha,'files':{},'overrides':{relative:{'sha256':hashlib.sha256(overlay).hexdigest(),'reason':'Synthetic data-channel layout fixture only'}}}))
    argv=[sys.executable,'-B','-m','analysis.g_full_training_backend','preflight','--inputs',str(path)]
    flags=['--root',str(tmp_path/'OPS_ROOT'),'--run-id','g-full-backend-transport-only','--kind','ops','--cwd',str(code),'--config',str(path),'--seed','42','--pinned-manifest',str(manifest),'--env','PYTHONPATH='+str(code/'src'),'--cpu-cores','1','--ram-mib','1024','--vram-mib','0','--disk-gib','1','--wall-hours','1']
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]))
    metadata=launcher.prepare(spec)
    assert metadata['git']['commit']==sha
    assert metadata['overlay']['overrides']=={relative:hashlib.sha256(overlay).hexdigest()}
    assert metadata['config']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    env={**os.environ,'PYTHONPATH':str(code/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run(argv[:4]+['dry',*argv[5:]],cwd=code,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['status']=='INPUTS_ADMITTED' and receipt['verified'] is False
    result=subprocess.run(spec['argv'],cwd=code,env=env,capture_output=True,text=True)
    assert result.returncode==2
    assert json.loads(result.stdout)['status']=='FINAL_SID_CONTRACT_UNAVAILABLE'
    assert (public/'p3c_validation.md').read_bytes()!=overlay



def test_lazy_rejects_holdout_before_source_open(tmp_path):
    from analysis.ur001_science_entry import InputRejected
    with pytest.raises(InputRejected,match='week26'):
        backend.lazy_examples('/workspace/vkgr-ur001-week26-holdout-20261021-v1/nonexistent',tmp_path/'offsets',1,'a'*64,lambda:pytest.fail('storage guard reached'))


def test_diagnostic_context_is_explicitly_synthetic():
    from vkgr import sid_v2d
    from vkgr.generative_data_v2 import ListExamples
    codec=sid_v2d.build_codec_d(np.array([1,2,3],dtype=np.uint32),np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0]],dtype=np.uint32),229,3)
    queries=[{'user_id':7,'target':2,'oov':False,'history':[1]}]
    stats=backend.slice_statistics(s0(),catalog=[1,2,3],queries=queries,manifest_sha256='a'*64,statistics_sha256='b'*64,binding_sha256='c'*64)
    examples=ListExamples([{'history':[1],'target':2}])
    context=backend.diagnostic_context(examples,codec,queries,stats,{'fixture':True})
    assert context.check() is context
    assert context.examples is examples
    assert context.slice_definition is stats
    assert context.binding['synthetic'] is True
    assert context.binding['engineering'] is True
    assert context.support==(1,2)


def test_formal_context_rejects_unbound_or_wrong_codec():
    with pytest.raises(ValueError,match='authenticated'):
        backend.native_functions(context=object(),sid=object(),binding={})


def test_full_request_admission_precedes_request_read(tmp_path):
    path,value=inputs(tmp_path)
    value['validation']='/workspace/vkgr-ur001-week26-holdout-20261021-v1/secret'
    path.write_text(json.dumps(value))
    from analysis.ur001_science_entry import InputRejected
    with pytest.raises(InputRejected):
        backend.read_transport(path,tmp_path/'missing','a'*64)


def test_full_request_requires_exact_closed_schema(tmp_path):
    path,_=inputs(tmp_path)
    request=tmp_path/'request';request.write_text('{"schema":"g-full-backend-request-v1","verified":true}')
    import hashlib
    with pytest.raises(ValueError,match='closed G'):
        backend.read_transport(path,request,hashlib.sha256(request.read_bytes()).hexdigest())


def test_formal_invocation_uses_g_module_and_explicit_approval(tmp_path):
    args=backend.invocation('train',tmp_path/'inputs',tmp_path/'request','a'*64)
    assert args[2:4]==['-m','analysis.g_full_training_backend']
    assert '--execute-approved-science' in args
    assert '--execute-approved-science' not in backend.invocation('preflight',tmp_path/'inputs',tmp_path/'request','a'*64)


def test_sid_consumer_rejects_unsealed_caller_claims():
    for value in (None,{}, {'verified':True}):
        with pytest.raises(ValueError,match='closed final'):
            backend.consume_sid(value)


def full_context_fixture(tmp_path,monkeypatch):
    import hashlib,os
    from analysis import ur001_derive_cli as derive,sgns_gpu_full
    from vkgr.sid_v2 import catalog_digest
    root=tmp_path/'stage2';root.mkdir();fit=tmp_path/'fit';fit.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    put=lambda path,value:path.write_text(json.dumps(value))
    ids=np.array([1,2,3],dtype=np.uint32)
    sids=np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0]],dtype=np.uint32)
    from vkgr.sid_v2d import build_codec_d
    codec=build_codec_d(ids,sids,229,3)
    np.savez(fit/'mapping.npz',ids=ids,sids=sids)
    np.savez(fit/'trie.npz',keys=codec.catalog.keys,key_items=codec.catalog.key_items,bases=np.array(codec.catalog.bases,dtype=np.uint64))
    for name in ('nodes.json','audit.json','config.json'):put(fit/name,{})
    sm={'schema':'UR001-native-full-SID-trie-v1','artifacts':{p.name:sha(p) for p in fit.iterdir()}}
    put(fit/'manifest.json',sm)
    train={'example_id':0,'user_id':7,'history':[1],'target':2,'history_positions':[[0,0]],'target_position':[0,1],'past_likes':[]}
    query={'user_id':7,'history':[1],'target':2,'history_positions':[[0,0]],'target_position':[25,0],'oov':False,'past_likes':[]}
    (root/'UR001-train.jsonl').write_text(json.dumps(train)+'\n')
    (root/'UR001-validation.jsonl').write_text(json.dumps(query)+'\n')
    put(root/'UR001-catalog.json',[1,2,3]);put(root/'UR001-S0.json',s0())
    artifacts={p.name:{'sha256':sha(p),'size':p.stat().st_size} for p in root.iterdir()}
    put(root/'UR001-manifest.json',{'artifacts':artifacts,'counts':{'train_examples':1,'eligible':1}})
    proof={'manifest':{'path':str(fit/'manifest.json'),'sha256':sha(fit/'manifest.json')},'codec':{'sid_version':backend.SID_VERSION,'branching':229,'depth':3,'catalog_digest':catalog_digest(ids),'manifest_sha256':sha(fit/'manifest.json'),'suffix_vocabulary':1,'vocab_size':690},'parent':{'binding':{'stage2_manifest_sha256':sha(root/'UR001-manifest.json')}},'runtime':{}}
    value={'stage2':str(root),'stage2_run':str(tmp_path/'run'),'producer_pin':{'manifest_sha256':sha(root/'UR001-manifest.json')},'sid_producer':{},'source':{},'runtime':{},'scratch_root':str(tmp_path/'scratch'),'run_ids':{'preflight':'pf'}}
    monkeypatch.setenv('VKGR_STAGE','preflight')
    monkeypatch.setattr(derive,'verify_dag',lambda *a,**kw:{'counts':{'train_examples':1}})
    monkeypatch.setattr(backend,'verify_producer',lambda *a:{'manifest':{}})
    monkeypatch.setattr(backend,'consume_sid',lambda *a:proof)
    monkeypatch.setattr(sgns_gpu_full,'fixed_dag',lambda:{'stage2_manifest_sha256':value['producer_pin']['manifest_sha256']})
    return value,proof,root,fit


def test_formal_full_context_uses_real_codec_rows_statistics_and_native_bytecode(tmp_path,monkeypatch):
    value,proof,root,fit=full_context_fixture(tmp_path,monkeypatch)
    context=backend.full_context(value,lambda:None)
    assert context.context.binding['engineering']is False
    assert context.context.examples.rows([0])[0]['history']==[1]
    namespace=backend.native_functions(context=context,sid=context.sid,binding=context.proof)
    from vkgr import sid_structure_a1_run as original
    assert namespace['run'].__code__ is original.run.__code__
    assert namespace['verify'].__code__ is original.verify.__code__
    assert namespace['train_batch_d']is original.train_batch_d
    config=backend.registered_config(42,proof['manifest']['sha256'])
    result=namespace['_contexts'](*([None]*9+[config]))
    assert result==(context.context,False,context.sid)


@pytest.mark.parametrize('mutation',['trie','history','ordinal','test_query'])
def test_formal_full_context_rejects_wrong_artifact_or_history(tmp_path,monkeypatch,mutation):
    value,proof,root,fit=full_context_fixture(tmp_path,monkeypatch)
    if mutation=='trie':
        with np.load(fit/'trie.npz')as original:arrays={k:original[k]for k in original.files}
        arrays['key_items']=arrays['key_items'][::-1];np.savez(fit/'trie.npz',**arrays)
        # Updating caller manifest cannot bypass original trie recomputation.
        import hashlib
        sm=json.loads((fit/'manifest.json').read_text());sm['artifacts']['trie.npz']=hashlib.sha256((fit/'trie.npz').read_bytes()).hexdigest()
        (fit/'manifest.json').write_text(json.dumps(sm));proof['manifest']['sha256']=hashlib.sha256((fit/'manifest.json').read_bytes()).hexdigest();proof['codec']['manifest_sha256']=proof['manifest']['sha256']
    else:
        name='UR001-validation.jsonl'if mutation=='test_query'else'UR001-train.jsonl'
        row=json.loads((root/name).read_text())
        if mutation=='history':row['history_positions']=[[25,0]]
        elif mutation=='ordinal':row['example_id']=1
        else:row['target_position']=[26,0]
        (root/name).write_text(json.dumps(row)+'\n')
        import hashlib
        m=json.loads((root/'UR001-manifest.json').read_text());m['artifacts'][name]['sha256']=hashlib.sha256((root/name).read_bytes()).hexdigest();(root/'UR001-manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):backend.full_context(value,lambda:None)


def test_formal_request_missing_p3c_is_rejected(tmp_path):
    import hashlib
    path,_=inputs(tmp_path);request=tmp_path/'request'
    request.write_text(json.dumps({'schema':backend.REQUEST_SCHEMA,'source':{},'runtime':{},'lock':{},'stage2':'/tmp/a','stage2_run':'/tmp/b','producer_pin':{},'sid_producer':{},'config':{},'ops_root':'/tmp/c','output_root':'/tmp/d','scratch_root':'/tmp/e','run_ids':{},'job_contract':{},'admission_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'predecessor':None}))
    with pytest.raises(ValueError,match='closed G'):backend.read_transport(path,request,hashlib.sha256(request.read_bytes()).hexdigest())


def test_stage_input_rejects_wrong_p3c_before_stage2(tmp_path,monkeypatch):
    import hashlib
    from analysis import profile_anchor_steps
    monkeypatch.setattr(profile_anchor_steps,'_source_pins',lambda:{'p':{'p3c_data_sha256':'a'*64}})
    files=[]
    for name in ('request','inputs','lock','p3c'):
        p=tmp_path/name;p.write_text('{}');files.append(p)
    request,inputs_file,lock,p3c=files
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    value={'admission_sha256':digest(inputs_file),'lock':{'path':str(lock),'sha256':digest(lock)},'p3c':{'path':str(p3c),'sha256':'b'*64}}
    with pytest.raises(ValueError,match='p3c data channel'):backend.stage_inputs(value,inputs_file,request,digest(request),'preflight')


def test_context_requires_fullB_same_stage2_dag(tmp_path,monkeypatch):
    value,proof,root,fit=full_context_fixture(tmp_path,monkeypatch)
    proof['parent']={'binding':{'stage2_manifest_sha256':'0'*64}}
    with pytest.raises(ValueError,match='stage2 DAG'):backend.full_context(value,lambda:None)


def test_g_job_transport_reuses_exact_original_closed_checks(tmp_path):
    from analysis import p_full_training_backend as pb
    functions=backend.transport_functions()
    for name in ('job_contract','check_job','plan'):
        assert functions[name].__code__ is getattr(pb,name).__code__
    value={'job_contract':{'preflight':{'env':{'PYTHONPATH':str(backend.ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'},'resources':{'cpu_cores':1,'ram_mib':1024,'vram_mib':0,'disk_gib':1,'wall_seconds':3600.},'ops_source':{'commit':'a'*40,'sha256':'b'*64},'overlay':None,'pinned_files':{},'stop_mode':'signal'}},'config':{'seed':42},'run_ids':{'preflight':'g-pf'},'ops_root':str(tmp_path/'ops'),'source':{'commit':'c'*40}}
    request=tmp_path/'request';inputs_file=tmp_path/'inputs'
    records=[{'path':str(request),'sha256':'d'*64,'bytes':1},{'path':str(inputs_file),'sha256':'e'*64,'bytes':1}]
    actual=functions['job_contract'](value,records,request,'d'*64,'preflight')
    assert actual['argv'][3]=='analysis.g_full_training_backend'
    assert actual['cwd']==str(backend.ROOT)
    value['job_contract']['preflight']['env']['OMP_NUM_THREADS']='2'
    with pytest.raises(ValueError,match='numerical environment'):
        functions['job_contract'](value,records,request,'d'*64,'preflight')


def test_formal_cli_requires_explicit_science_flag_before_execution(tmp_path):
    path,_=inputs(tmp_path)
    result=subprocess.run([sys.executable,'-B','-m','analysis.g_full_training_backend','train','--inputs',str(path),'--request',str(tmp_path/'absent'),'--request-sha256','a'*64],capture_output=True,text=True)
    assert result.returncode==2
    assert 'explicit approved science' in result.stderr
    assert json.loads(result.stdout)['verified']is False


def test_formal_cli_holdout_negative_precedes_missing_request(tmp_path):
    path,value=inputs(tmp_path)
    value['validation']='/workspace/vkgr-ur001-week26-holdout-20261021-v1/secret';path.write_text(json.dumps(value))
    result=subprocess.run([sys.executable,'-B','-m','analysis.g_full_training_backend','preflight','--inputs',str(path),'--request',str(tmp_path/'absent'),'--request-sha256','a'*64],capture_output=True,text=True)
    assert result.returncode==2
    assert json.loads(result.stdout)['status']=='HOLDOUT_INPUT_REJECTED'


def test_sid_consumer_rejects_nonregistered_entry_before_receipt(tmp_path,monkeypatch):
    from analysis import sgns_subset_acceptance as provenance
    monkeypatch.setattr(provenance,'source_tree',lambda *a:None)
    record={'root':str(tmp_path),'source':{'commit':'a'*40,'files':{'analysis/sgns_native_full_sid.py':'b'*64}},'entry_sha256':'b'*64,'receipt':{'path':str(tmp_path/'absent'),'sha256':'c'*64}}
    with pytest.raises(ValueError,match='registered SID producer'):
        backend.consume_sid(record)
