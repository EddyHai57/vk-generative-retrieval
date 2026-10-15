"""Versioned SEQ-M1 deployment, read-only real-input preflight and CUDA rehearsal.
No formal M1 training is submitted by this tool. Rehearsal uses synthetic data.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

R=Path('/workspace/vkgr-infra01-root')
DEP=Path('/workspace/vkgr-seqm1-p-v1')
OPS=Path('/workspace/vkgr-seqm1-ops-20261016')
REPO=Path('/workspace/vkgr-b1-src')
PUB='0ca61bc35ad68bd79b86103b54e470445cbab07f'
BASE_INPUT=Path('/workspace/vkgr-b1-inputs/p-feat02-inputs-base-v1.json')
BASE_SHA='a12b98003aa2fbb58df83bc70bb448e426d01c2a4cc2e6357bf1f95778f23e38'
P3C='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
CFG={42:'e88dc757ff4085af020efc1f070ebba8df54876e85adf68120dff730f983b621',
     2024:'3e1e40f200c3a09a39e15048df33657bbb54080a7da33d604d92c41518a5dde5',
     2025:'e4f8f612991ee960b496067b905cb57bb46d7d2b9e4c4af9026004a53e8713f3'}
SMOKE='p-seqm1-cuda-engineering-v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def declared_vram(measured):
    if not math.isfinite(measured) or measured<=0:raise ValueError('positive measured per-process VRAM required')
    return math.ceil(measured*1.15/256)*256
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],stderr=subprocess.STDOUT)

def verify_tree(directory,commit):
    names=git('ls-tree','-r','--name-only',commit).decode().splitlines()
    for name in names:
        if sha(directory/name)!=hashlib.sha256(git('show',commit+':'+name)).hexdigest():raise ValueError('tree mismatch: '+name)
    return len(names)

def deploy(req):
    if DEP.exists() or OPS.exists():raise FileExistsError('deployment directories must be new')
    OPS.mkdir()
    git('fetch','origin','codex/seq-m1-pub')
    git('worktree','add','--detach',str(DEP),PUB)
    files=verify_tree(DEP,PUB)
    git('fetch','origin','codex/phase2-readonly-tools')
    git('worktree','add','--detach',str(OPS/'code'),req['ops_commit'])
    ops_files=verify_tree(OPS/'code',req['ops_commit'])
    script=OPS/'code/analysis/seqm1_prepare.py'
    if sha(script)!=req['source_sha256']:raise ValueError('ops source mismatch')
    original=Path('/workspace/vkgr-b1-p/docs/p3c_validation.md')
    if sha(original)!=P3C:raise ValueError('frozen p3c data-channel pin')
    # Preserve the Git version outside the model tree before creating the
    # historical frozen input override; no existing artifact is discarded.
    (DEP/'docs/p3c_validation.md').rename(OPS/'p3c-validation-git-version.md')
    with (DEP/'docs/p3c_validation.md').open('xb') as f:f.write(original.read_bytes())
    receipt={'public_commit':PUB,'model_tree_files_checked':files,'ops_commit':req['ops_commit'],
             'ops_files_checked':ops_files,'source_sha256':sha(script),'p3c_override_sha256':sha(DEP/'docs/p3c_validation.md')}
    with (OPS/'deployment.json').open('x') as f:json.dump(receipt,f,sort_keys=True)
    return receipt

def setup():
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.chdir(DEP);sys.path.insert(0,str(DEP/'src'))

def preflight():
    setup()
    from vkgr import retrieval_b1_run as b1,retrieval_feat02_run as fr,retrieval_seqm1_run as m1
    from vkgr.retrieval_ablation_run import read_pinned
    if sha(DEP/'docs/p3c_validation.md')!=P3C or b1.ROOT!=DEP:raise ValueError('preflight root/p3c mismatch')
    configs={}
    for seed,pin in CFG.items():
        p=DEP/f'configs/seqm1/p-seqm1-mean-s{seed}.json';cfg=read_pinned(p,pin);m1.validate_config(cfg)
        if (R/'runs'/p.stem).exists():raise FileExistsError(p.stem)
        configs[seed]=cfg
    spec=fr.validate_inputs(read_pinned(BASE_INPUT,BASE_SHA),m1.FEATURE)
    data,_,base=b1.real_inputs(spec)
    data,extra=fr.apply_feat02(data,spec,configs[42]['history_len'])
    counts={'rows':len(data['rows']),'queries':len(data['queries']),'catalog':len(data['catalog'])}
    if counts!={'rows':86785,'queries':2520,'catalog':120415}:raise ValueError('real population mismatch')
    return {'status':'preflight_ok','counts':counts,'config_sha256':CFG,'inputs_sha256':BASE_SHA,
            'binding':{**base,**extra},'public_commit':PUB,'test_read':False,'training_started':False}

def own_gpu_mib():
    try:
        s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        vals=[float(l.split(',')[1].strip()) for l in s.splitlines() if int(l.split(',')[0].strip())==os.getpid()]
        return max(vals) if vals else None
    except (OSError,ValueError,subprocess.CalledProcessError):return None

def cuda_rehearsal(out):
    setup()
    import gc
    import numpy as np
    import torch
    from vkgr import retrieval_seqm1_run as m1,retrieval_feat02_run as fr
    from vkgr.retrieval_seqm1 import make_seqm1_model
    from vkgr.retrieval_negatives import masked_ce_logq
    from vkgr.retrieval_experiment import normalize_export
    from vkgr.baselines import blockwise_topk
    from vkgr.training import seed_all
    sys.path.insert(0,str(DEP/'tests'));sys.path.insert(0,str(DEP/'scripts'))
    import test_seqm1_model as guards
    from php01_vram_smoke import synthetic_batch
    if not torch.cuda.is_available():raise ValueError('actual CUDA required')
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False);result={}
    for arm in ('mean','last','attn','bi','pure'):
        data=fr.synthetic_data();cfg={**m1.default_config(),'arm':arm,'batch_size':16,'max_epochs':6,'min_epochs':2}
        v=m1.run_session(out/arm,data,cfg,device='cuda')
        verified=m1.verify_run(out/arm/'manifest.json',v['manifest_sha256'],data,cfg,device='cuda')
        if verified['status']!='verified':raise ValueError('CUDA synthetic scientific verify')
        model=guards.activate(guards.build(arm)).cuda().eval()
        h,side=guards.make_batch([2,7,20],20)
        with torch.no_grad():
            ref=model.users(h.cuda(),side.cuda())
            for width in (35,50):
                hp,sp=guards.pad_to(h,side,width);actual=model.users(hp.cuda(),sp.cuda())
                if not torch.allclose(ref,actual,atol=1e-6,rtol=0):raise ValueError('CUDA padding invariant')
        del model,h,side,ref;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        context=own_gpu_mib();reserved0=torch.cuda.memory_reserved()/2**20
        if context is None:raise ValueError('per-process CUDA context measurement unavailable')
        seed_all(42);model=make_seqm1_model(torch.zeros(120416,64),list(range(1,21497)),50,arm).cuda()
        opt=m1.make_optimizer(model,cfg);gen=torch.Generator().manual_seed(7);finite=True;observed=[]
        for _ in range(3):
            h,side=synthetic_batch(gen,512,21496,120416);targets=torch.randperm(21496,generator=gen)[:512]+1
            model.train();opt.zero_grad(set_to_none=True)
            loss,_=masked_ce_logq(model.users(h.cuda(),side.cuda()),model.items(targets.cuda()),torch.arange(512,device='cuda'),torch.ones(512,512,dtype=torch.bool,device='cuda'),.10,torch.full((512,),math.log(1/21496)))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();finite &= bool(torch.isfinite(loss))
            value=own_gpu_mib()
            if value is not None:observed.append(value)
        model.eval();items=[];users=[]
        with torch.no_grad():
            for start in range(1,120416,512):items.append(model.items(torch.arange(start,min(start+512,120416),device='cuda')).cpu().numpy())
            for start in range(0,2520,512):
                h,side=synthetic_batch(gen,min(512,2520-start),21496,120416);users.append(model.users(h.cuda(),side.cuda()).cpu().numpy())
        blockwise_topk(normalize_export(np.concatenate(users))[:32],normalize_export(np.concatenate(items)),np.arange(1,120416),50)
        torch.cuda.synchronize();allocated=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
        measured=max([reserved+max(0,context-reserved0)]+observed)
        if not finite:raise ValueError('nonfinite engineering loss')
        result[arm]={'synthetic_verify':verified,'cuda_padding_passed':True,'full_shape_steps':3,'batch':512,'history':50,'embedding_rows':120416,
                     'allocated_peak_mib':allocated,'reserved_peak_mib':reserved,'context_observed_mib':context,'own_nvml_observations_mib':observed,
                     'measured_bound_mib':measured,'suggested_vram_mib':declared_vram(measured),'finite_loss':finite}
        del model,opt,items,users,h,side,loss;gc.collect();torch.cuda.empty_cache()
    receipt={'status':'engineering_passed','arms':result,'public_commit':PUB,'ops_source_sha256':sha(__file__),
             'claim_boundary':'synthetic CUDA/full-shape engineering only; no real quality metric, no formal training, no test'}
    with (out/'engineering.json').open('x') as f:json.dump(receipt,f,sort_keys=True)
    print(json.dumps(receipt,sort_keys=True))

def smoke_job():
    source=OPS/'code/analysis/seqm1_prepare.py'
    old=list((R/'queue/claimed').glob('*-p-php02-cudasmoke.json'))
    if len(old)!=1:raise ValueError('unique previous CUDA rehearsal template required')
    d=load(old[0]);py=d['argv'][0]
    boot="import runpy,sys;sys.argv=[sys.argv[1],'--cuda',sys.argv[2]];runpy.run_path(sys.argv[0],run_name='__main__')"
    d.update(run_id=SMOKE,cwd=str(DEP),config=str(DEP/'configs/seqm1/p-seqm1-mean-s42.json'),inputs=[str(source)],disk_gib=2,wall_hours=1.0,vram_mib=3072,
             argv=[py,'-B','-c',boot,str(source),str(R/'runs'/SMOKE/'out/rehearsal')])
    if d['no_admission'] is not False or (d['kind'],d['cpu_cores'],d['ram_mib'])!=('real',1,9216):
        raise ValueError('engineering job must retain normal admission and frozen CPU/RAM budget')
    return d

def main():
    if '--cuda' in sys.argv:cuda_rehearsal(Path(sys.argv[sys.argv.index('--cuda')+1]));return
    raw=sys.stdin.read();req=json.loads(raw) if raw.strip() else {'action':'preflight'};action=req['action']
    if action=='deploy':answer=deploy(req)
    elif action=='preflight':answer=preflight()
    elif action in ('smoke-dry','smoke-submit'):
        answer={'preflight':preflight(),'job':smoke_job(),'status':'dry'}
        sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
        from vkgr_ops import launcher
        answer['launcher_prepare']=launcher.prepare(answer['job'])
        if action=='smoke-submit':
            sys.path.insert(0,str(OPS/'code/analysis'));from gtok02_seed_batch import publish
            answer['queue_file']=publish(R/'queue',R/'runs',answer['job']);answer['status']='written_not_accepted'
    else:raise ValueError('unregistered action')
    answer['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(answer,sort_keys=True))

if __name__=='__main__':main()
