"""Synthetic only; existing scientific implementations and cache bytes untouched."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis import p_full_compact_inputs as compact
from analysis import p_pf_cache_reuse as reuse


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    root=tmp_path/'cache';root.mkdir()
    history={'train':[{'example_id':0,'history':[1,2],'history_positions':[[0,0],[1,1]],'target':3},
                      {'example_id':1,'history':[2],'history_positions':[[1,1]],'target':1}],
             'validation':[{'history':[1,2],'history_positions':[[0,0],[1,1]],'target':3}]}
    records={};stores={}
    for split in history:
        path=tmp_path/(split+'.jsonl');path.write_bytes(b''.join(json.dumps(row).encode()+b'\n'for row in history[split]))
        records[split]={'path':str(path),'sha256':digest(path),'count':len(history[split])}
        stores[split]=compact.JsonRows.build(path,root/(split+'-offsets'),len(history[split]),digest(path),lambda:None)
    parts=[(0,[{'item_id':1,'side':[1,2,3]}]),(1,[{'item_id':9,'side':[0,0,0]},{'item_id':2,'side':[4,5,6]}])]
    side=compact.DenseSide.build(root/'side',stores,parts,{},lambda row,duration:row['side'],lambda:None)
    return root,records,history,parts,side


def test_seal_semantics_and_readonly_original_views(tmp_path):
    root,records,history,parts,side=fixture(tmp_path)
    before=reuse.inventory(root)
    proof=reuse.verify_semantics(root,records,parts,{},lambda row,duration:row['side'],lambda:None)
    assert proof['rows']=={'train':2,'validation':1}
    assert proof['wanted_positions']==2
    assert reuse.inventory(root)==before
    views=reuse.readonly_views(root,records,before,lambda:None)
    assert type(views['train'])is compact.SideRows
    assert type(views['train'].rows)is compact.JsonRows
    assert type(views['train'].side)is compact.DenseSide
    assert views['train'][0]=={**history['train'][0],'side':[[1,2,3],[4,5,6]]}
    assert not views['train'].side.arrays[0].flags.writeable
    assert not views['train'].rows.offsets.flags.writeable


@pytest.mark.parametrize('mutation',['offset','side','state','item','shape','extra','symlink','missing','unused'])
def test_semantic_and_file_drift_rejected(tmp_path,mutation):
    root,records,history,parts,side=fixture(tmp_path)
    if mutation=='offset':
        x=np.memmap(root/'train-offsets',dtype='<u8',mode='r+');x[1]+=1;x.flush()
    elif mutation=='extra':(root/'extra').write_text('unexpected')
    elif mutation=='symlink':(root/'link').symlink_to(root/'train-offsets')
    elif mutation=='missing':
        # Fixture copy excludes a file; no deletion of user/project artifacts.
        import shutil
        other=tmp_path/'incomplete';other.mkdir();shutil.copy(root/'train-offsets',other/'train-offsets');root=other
    else:
        array=np.memmap(root/'side/week-1.bin',dtype=compact.DenseSide.DTYPE,mode='r+')
        if mutation=='side':array[1]['side']=[0,0,0]
        elif mutation=='state':array[1]['state']=1
        elif mutation=='item':array[1]['item']=3
        elif mutation=='unused':array[0]['item']=8
        elif mutation=='shape':
            with (root/'side/week-1.bin').open('ab')as stream:stream.write(b'\0'*8)
        array.flush()
    with pytest.raises((ValueError,OSError)):
        reuse.verify_semantics(root,records,parts,{},lambda row,duration:row['side'],lambda:None)


def test_reader_detects_post_seal_drift(tmp_path):
    root,records,*_=fixture(tmp_path);sealed=reuse.inventory(root)
    (root/'side/week-1.bin').write_bytes(b'changed')
    with pytest.raises(ValueError):reuse.readonly_views(root,records,sealed,lambda:None)


def test_additive_delta_is_closed_and_runtime_exact():
    base={'commit':'a'*40,'files':{'old.py':'b'*64}}
    delta={name:'c'*64 for name in reuse.ADDITIONS}
    reader={'commit':'d'*40,'files':{**base['files'],**delta}}
    runtime={'runtime':{'python':'3.11.17'},'lock_sha256':'e'*64}
    assert reuse.fixed_delta(base,reader,delta,runtime,runtime)['added']==delta
    for bad in ({**reader,'files':{**reader['files'],'old.py':'f'*64}},
                {**reader,'files':{**reader['files'],'shadow.py':'f'*64}}):
        with pytest.raises(ValueError):reuse.fixed_delta(base,bad,delta,runtime,runtime)
    with pytest.raises(ValueError):reuse.fixed_delta(base,reader,delta,runtime,{**runtime,'lock_sha256':'f'*64})


def test_cli_dry_nonempty_illegal_nonzero():
    ok=subprocess.run([sys.executable,'-B','-m','analysis.p_pf_cache_reuse','dry'],capture_output=True,text=True)
    assert ok.returncode==0 and json.loads(ok.stdout)['training']is False
    no=subprocess.run([sys.executable,'-B','-m','analysis.p_pf_cache_reuse','illegal'],capture_output=True,text=True)
    assert no.returncode!=0


def test_closed_request_rejects_before_producer(tmp_path):
    request=tmp_path/'request';request.write_text('{"schema":"p-pf-cache-reuse-request-v1","verified":true}')
    with pytest.raises(ValueError,match='closed cache'):
        reuse.read_request(request,digest(request))


def test_provenance_cannot_be_caller_verified_boolean():
    with pytest.raises(ValueError):reuse.consume({'verified':True},lambda:None)


@pytest.fixture
def seal_fixture(tmp_path,monkeypatch):
    """Synthetic complete real producer verifier; context is a unit-test boundary."""
    from analysis import producer_dag_binding as dag
    root,records,*_=fixture(tmp_path)
    run=tmp_path/'ops/runs/seal';(run/'logs').mkdir(parents=True)
    output=tmp_path/'seal.json';request=tmp_path/'request.json';request.write_text('{}')
    source={'commit':'a'*40,'files':{'old.py':'b'*64}}
    runtime={'runtime':{'python':'3.11.17'},'lock_sha256':'c'*64}
    value={'output':str(output),'ops_root':str(tmp_path/'ops'),'run_id':'seal','original':{'fixed':'old'},'reader':{'source':source,'runtime':runtime}}
    original={'value':{},'receipt':{'binding':{'old':'binding'}},'proof':{'binding':{'old':'proof'}}}
    delta={'added':'closed'}
    expected={k:None for k in dag.RUN_KEYS}
    expected.update(schema='vkgr-run-v1',run_id='seal',git={'commit':source['commit'],'dirty':False,'dirty_files':[]})
    (run/'run.json').write_bytes(reuse.backend.encoded(expected))
    receipt={'schema':reuse.SCHEMA,'verified':True,'status':'verified','training':False,'scientific_verified':False,'engineering':False,'test_read':False,
             'source':source,'runtime':runtime,'run_id':'seal','request':{'path':str(request),'sha256':digest(request)},
             'original_pf':value['original'],'original_binding':original['receipt']['binding'],'reader_delta':delta,'inputs':records,
             'cache_root':str(root),'run_manifest_sha256':digest(run/'run.json'),
             'semantic':{'scope':'current_cache_revalidated_not_historical_PF_bytes','scientific_training':False,'files':reuse.inventory(root)}}
    output.write_bytes(reuse.backend.encoded(receipt));(run/'logs/stdout.log').write_bytes(reuse.backend.encoded(receipt)+b'\n')
    terminal={k:None for k in dag.TERMINAL_KEYS}
    terminal.update(schema='vkgr-terminal-v1',run_id='seal',status='succeeded',exit_code=0,signal=None,wall_cap_exceeded=False)
    (run/'terminal.json').write_bytes(reuse.backend.encoded(terminal))
    pin={'manifest_sha256':digest(output),'source':source,'runtime':runtime,'job_sha256':{name:digest(run/name)for name in dag.JOB_FILES},'receipt':receipt}
    reference={'manifest':{'path':str(output),'sha256':digest(output)},'run_dir':str(run),'pin':pin}
    monkeypatch.setattr(reuse,'context',lambda *a:(value,original,root,records,delta))
    monkeypatch.setattr(reuse,'expected_job',lambda *a:(expected,[],{},{}))
    return reference,run,root,original


def test_complete_producer_consume_keeps_dual_identity(seal_fixture):
    ref,run,root,original=seal_fixture
    result=reuse.consume(ref,lambda:None)
    assert len(result['rows'])==2 and len(result['queries'])==1
    assert result['scientific_binding']==original['receipt']['binding']
    assert result['reader_identity']['original_pf_producer']==original['proof']['binding']
    assert result['reader_identity']['seal_producer']==ref['pin']


@pytest.mark.parametrize('fault',['stdout','terminal','job','manifest','source','runtime','ownjob','semantic','cache','ledger','reader-runtime'])
def test_real_seal_producer_closure_rejects(seal_fixture,fault):
    ref,run,root,original=seal_fixture
    if fault in ('stdout','terminal','job'):
        target=run/({'stdout':'logs/stdout.log','terminal':'terminal.json','job':'run.json'}[fault]);target.write_bytes(target.read_bytes()+b' ')
    elif fault=='manifest':Path(ref['manifest']['path']).write_bytes(b'{}')
    elif fault in ('source','runtime'):
        import copy
        ref['pin']=copy.deepcopy(ref['pin'])
        if fault=='source':ref['pin']['source']['commit']='d'*40
        else:ref['pin']['runtime']['lock_sha256']='d'*64
    elif fault in ('ownjob','semantic','ledger','reader-runtime'):
        # Re-pin malformed producer bytes so downstream own semantics must reject.
        if fault=='ownjob':
            value=json.loads((run/'run.json').read_bytes());value['argv']=['changed'];(run/'run.json').write_bytes(reuse.backend.encoded(value))
            receipt=json.loads(Path(ref['manifest']['path']).read_bytes());receipt['run_manifest_sha256']=digest(run/'run.json')
            Path(ref['manifest']['path']).write_bytes(reuse.backend.encoded(receipt));(run/'logs/stdout.log').write_bytes(reuse.backend.encoded(receipt)+b'\n')
            ref['pin']['receipt']=receipt;ref['manifest']['sha256']=digest(ref['manifest']['path']);ref['pin']['manifest_sha256']=ref['manifest']['sha256']
        else:
            value=json.loads(Path(ref['manifest']['path']).read_bytes());
            if fault=='semantic':value['semantic']['scope']='old_PF_timestamp'
            elif fault=='ledger':value['source']={**value['source'],'files':{'old.py':'d'*64}};ref['pin']['source']=value['source']
            else:value['runtime']={**value['runtime'],'lock_sha256':'d'*64};ref['pin']['runtime']=value['runtime']
            Path(ref['manifest']['path']).write_bytes(reuse.backend.encoded(value))
            (run/'logs/stdout.log').write_bytes(reuse.backend.encoded(value)+b'\n');ref['pin']['receipt']=value
            ref['manifest']['sha256']=digest(ref['manifest']['path']);ref['pin']['manifest_sha256']=ref['manifest']['sha256']
        ref['pin']['job_sha256']={name:digest(run/name)for name in ref['pin']['job_sha256']}
    else:(root/'side/week-1.bin').write_bytes(b'changed')
    with pytest.raises(ValueError):reuse.consume(ref,lambda:None)


def test_consume_rechecks_context_after_open(seal_fixture,monkeypatch):
    ref,run,root,original=seal_fixture
    prior=reuse.context;calls=[]
    def changed(*a):
        result=prior(*a);calls.append(True)
        return result if len(calls)==1 else ({**result[0],'drift':True},*result[1:])
    monkeypatch.setattr(reuse,'context',changed)
    with pytest.raises(ValueError,match='closure changed'):reuse.consume(ref,lambda:None)


def test_actual_launcher_metadata_and_module_dry(tmp_path):
    """Unmocked fixed launcher; synthetic deployment + separate p3c fixture."""
    import io,os,tarfile
    repo=Path(__file__).resolve().parents[1]
    deployment=tmp_path/'deployment';deployment.mkdir()
    ops=tmp_path/'ops-source';ops.mkdir()
    for target,args in ((deployment,['HEAD']),(ops,['5840e6b','ops'])):
        raw=subprocess.check_output(['git','-C',str(repo),'archive',*args])
        with tarfile.open(fileobj=io.BytesIO(raw))as archive:archive.extractall(target,filter='data')
    (deployment/'analysis/p_pf_cache_reuse.py').write_bytes((repo/'analysis/p_pf_cache_reuse.py').read_bytes())
    native=tmp_path/'native';(native/'docs').mkdir(parents=True)
    (native/'docs/p3c_validation.md').write_bytes(b'synthetic private data-channel fixture\n')
    subprocess.run(['git','init','-q',str(deployment)],check=True)
    subprocess.run(['git','-C',str(deployment),'add','.'],check=True)
    subprocess.run(['git','-C',str(deployment),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
    request=tmp_path/'request';request.write_text('{}')
    worker=reuse.invocation('seal','unused',request,digest(request))
    code='''import json,sys
from vkgr_ops import launcher
root,cwd,request,worker=json.loads(sys.argv[1])
a=launcher.build_parser().parse_args(['--root',root,'--run-id','cache-seal-fixture','--kind','ops','--cwd',cwd,'--config',request,'--input',request,'--cpu-cores','1','--ram-mib','1024','--vram-mib','0','--disk-gib','0','--wall-hours','1','--',*worker])
spec=launcher.spec_from_args(a);metadata=launcher.prepare(spec)
assert metadata['git']['dirty'] is False and len(metadata['inputs'])==1
print(json.dumps({'status':'prepared','argv':spec['argv']}))
'''
    result=subprocess.run([sys.executable,'-B','-c',code,json.dumps([str(tmp_path/'ops-root'),str(deployment),str(request),worker])],env={**os.environ,'PYTHONPATH':str(ops/'ops')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['argv']==worker
    env={**os.environ,'PYTHONPATH':str(deployment/'src')+os.pathsep+str(deployment)}
    dry=subprocess.run(worker[:4]+['dry'],cwd=deployment,env=env,capture_output=True,text=True)
    assert dry.returncode==0,dry.stderr
    assert json.loads(dry.stdout)['cache_opened']is False
    illegal=subprocess.run(worker[:4]+['illegal'],cwd=deployment,env=env,capture_output=True,text=True)
    assert illegal.returncode!=0
    assert not(tmp_path/'ops-root/runs').exists()


@pytest.mark.parametrize('target',['side/week-1.bin','train-offsets','validation-offsets'])
def test_opened_mapping_inode_cannot_detach_from_sealed_path(tmp_path,monkeypatch,target):
    root,records,*_=fixture(tmp_path);sealed=reuse.inventory(root);real=np.memmap;changed=[]
    def opened(path,*args,**kwargs):
        array=real(path,*args,**kwargs)
        name=Path(getattr(path,'name',path))
        if name==root/target and not changed:
            changed.append(True);raw=name.read_bytes();name.rename(tmp_path/'detached');name.write_bytes(raw)
            with (tmp_path/'detached').open('r+b')as stream:
                stream.seek(0);stream.write(b'\xff');stream.flush()
        return array
    monkeypatch.setattr(reuse.np,'memmap',opened)
    with pytest.raises(ValueError,match='identity|inode|changed'):
        reuse.readonly_views(root,records,sealed,lambda:None)


def test_jsonl_identity_is_captured_before_hash(tmp_path,monkeypatch):
    root,records,*_=fixture(tmp_path);sealed=reuse.inventory(root);real=reuse.backend.file_sha;changed=[];target=Path(records['train']['path'])
    def hashed(path):
        value=real(path)
        if Path(path)==target and not changed:
            changed.append(True);raw=target.read_bytes();target.rename(tmp_path/'detached-jsonl');target.write_bytes(raw)
        return value
    monkeypatch.setattr(reuse.backend,'file_sha',hashed)
    with pytest.raises(ValueError,match='identity|inode|changed'):
        reuse.readonly_views(root,records,sealed,lambda:None)


def test_consume_inode_closure_after_context_even_same_bytes(seal_fixture,monkeypatch,tmp_path):
    ref,run,root,original=seal_fixture;prior=reuse.context;calls=[]
    def changed(*a):
        result=prior(*a);calls.append(True)
        if len(calls)==2:
            target=root/'side/week-1.bin';raw=target.read_bytes();target.rename(tmp_path/'late-detached');target.write_bytes(raw)
        return result
    monkeypatch.setattr(reuse,'context',changed)
    with pytest.raises(ValueError,match='inode|identity|changed'):
        reuse.consume(ref,lambda:None)
