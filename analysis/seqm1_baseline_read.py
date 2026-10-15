"""SEQM1-000: pinned F4 best checkpoints -> aggregate raw validation norms.
Read only; no training, vector export, queue writes, test split or threshold tuning.
"""
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time

DEP=Path('/workspace/vkgr-seqm1-p-v1')
OPS=Path('/workspace/vkgr-seqm1-baseline-ops-20261016')

def checked_source_ledger(root, ledger):
    required={f'src/vkgr/{name}.py' for name in ('retrieval_php01','retrieval_php01_run','retrieval_features','retrieval_models','retrieval_hp','training')}
    names={name for name in ledger if name.startswith('src/vkgr/') and name.endswith('.py')}
    if not required<=names:raise ValueError('incomplete baseline source ledger')
    result={}
    for name in sorted(names):
        path=(root/name).resolve()
        if not path.is_relative_to((root/'src/vkgr').resolve()):raise ValueError('source ledger escaped root')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=ledger[name]:raise ValueError('baseline/M1 source mismatch: '+name)
        result[name]=actual
    return result

def raw_norm_summary(vectors):
    if not vectors or not isinstance(vectors[0],(list,tuple)) or not vectors[0]:raise ValueError('nonempty matrix required')
    width=len(vectors[0]);norms=[]
    for row in vectors:
        if len(row)!=width or any(not math.isfinite(float(v)) for v in row):raise ValueError('bad raw vector')
        norms.append(math.sqrt(math.fsum(float(v)*float(v) for v in row)))
    if any(not math.isfinite(v) for v in norms):raise ValueError('nonfinite norm')
    norms.sort()
    def q(p):
        i=(len(norms)-1)*p;j=int(i)
        return norms[j]+(i-j)*(norms[min(j+1,len(norms)-1)]-norms[j])
    median=q(.5)
    return {'queries':len(norms),'p10':q(.1),'median':median,'p90':q(.9),'min':norms[0],'max':norms[-1],
            'scale_check_passed':.05<=median<=5.,'fixed_scale_interval':[.05,5.]}

def cpu_room():
    c=Path('/sys/fs/cgroup');quota,period=(c/'cpu.max').read_text().split()
    if quota=='max':raise ValueError('expected finite CPU quota')
    def used():return int(dict(x.split() for x in (c/'cpu.stat').read_text().splitlines())['usage_usec'])
    t=time.monotonic();a=used();time.sleep(1.);cores=(used()-a)/1e6/(time.monotonic()-t)
    return {'used_cores':cores,'quota_cores':int(quota)/int(period),'allowed_cores':.8*int(quota)/int(period),
            'one_core_fits':cores+1<=.8*int(quota)/int(period)}

def main():
    raw=sys.stdin.read();req=json.loads(raw) if raw.strip() else {}
    if req.get('action')=='deploy':
        if OPS.exists():raise FileExistsError('new ops worktree required')
        def git(*args):return subprocess.check_output(['git','-C','/workspace/vkgr-b1-src',*args],stderr=subprocess.STDOUT)
        git('fetch','origin','codex/phase2-readonly-tools');git('worktree','add','--detach',str(OPS),req['commit'])
        names=git('ls-tree','-r','--name-only',req['commit']).decode().splitlines()
        for name in names:
            if hashlib.sha256((OPS/name).read_bytes()).hexdigest()!=hashlib.sha256(git('show',req['commit']+':'+name)).hexdigest():raise ValueError('ops tree SHA mismatch')
        script_sha=hashlib.sha256((OPS/'analysis/seqm1_baseline_read.py').read_bytes()).hexdigest()
        if script_sha!=req['source_sha256']:raise ValueError('ops helper mismatch')
        print(json.dumps({'status':'ops_deployed','commit':req['commit'],'files_checked':len(names),'source_sha256':script_sha,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}));return
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    room=cpu_room()
    if not room['one_core_fits']:
        print(json.dumps({'read_at_utc':started,'status':'deferred_cpu','cpu':room,'checkpoint_read':False,'test_read':False}));return
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
    sys.path.insert(0,str(OPS/'analysis'));sys.path.insert(0,str(DEP/'src'));os.chdir(DEP)
    from seqm1_prepare import sha,BASE_INPUT,BASE_SHA,P3C
    from phase2_validation_read import read,inspect_run,load_archive,ROOT,HISTORY_PIN
    import torch
    from vkgr import retrieval_b1_run as b1,retrieval_feat02_run as fr,retrieval_php01_run as ph
    torch.set_num_threads(1);b1.execution_context('cpu')
    if b1.ROOT!=DEP or sha(DEP/'docs/p3c_validation.md')!=P3C:raise ValueError('model deployment/root mismatch')
    hist,_=read(ROOT/'runs/h2-prepare/out/manifest.json',HISTORY_PIN)
    queries,_=read(ROOT/'runs/h2-prepare/out/validation-l50.json',hist['artifacts']['validation-l50.json'])
    if len(queries)!=2520:raise ValueError('validation population mismatch')
    spec=fr.validate_inputs(read(BASE_INPUT,BASE_SHA)[0],ph.FEATURE)
    data,_,_=b1.real_inputs(spec);data,_=fr.apply_feat02(data,spec,50)
    source=None
    result={};checkpoint_pins={}
    for seed in (42,2024,2025):
        rid=f'p-php01-h1-t010-s{seed}';state=inspect_run(rid)
        if not state['verified']:raise ValueError('F4 baseline scientific verifier not accepted')
        archive=load_archive(rid,queries,hist,state)
        m,_=read(state['manifest_path'],state['manifest_sha256']);cfg=m['config'];ph.validate_config(cfg)
        if cfg['seed']!=seed or cfg['knob']!='H1' or cfg['history_len']!=50:raise ValueError('unregistered F4 baseline config')
        current_source=checked_source_ledger(DEP,m['binding']['source']['files'])
        if source is not None and source!=current_source:raise ValueError('three seed source ledger mismatch')
        source=current_source
        rows,valid=fr._view(data,50)
        for key,value in [('rows_sha256',b1._digest(rows)),('queries_sha256',b1._digest(valid)),('catalog_sha256',b1._digest(data['catalog'])),
                          ('support_sha256',b1._digest(sorted(data['support']))),('content_sha256',hashlib.sha256(data['content'].cpu().numpy().tobytes()).hexdigest())]:
            if m['binding'][key]!=value:raise ValueError('baseline live input binding mismatch: '+key)
        p=Path(state['manifest_path']).parent/'best.pt';best=m['best']
        if best['path']!='best.pt' or sha(p)!=best['sha256']:raise ValueError('best checkpoint SHA mismatch')
        saved=torch.load(p,map_location='cpu',weights_only=True)
        if saved['binding_sha256']!=b1._digest(m['binding']) or saved['epoch']!=best['epoch'] or saved['step']!=best['step']:raise ValueError('best checkpoint identity mismatch')
        model=ph._model(data,cfg,'cpu');model.load_state_dict(saved['model'],strict=True);model.eval();vectors=[]
        with torch.no_grad():
            for start in range(0,len(valid),512):
                h,s=fr._tensors2(valid[start:start+512],data['idmap'],50,cfg['feature'])
                vectors.extend(model.users(h,s).tolist())
        norms=raw_norm_summary(vectors)
        checkpoint_pins[str(p)]=best['sha256']
        result[str(seed)]={'raw_m_norms':norms,'manifest_sha256':state['manifest_sha256'],'source_commit':m['binding']['source']['git_commit'],
                           'checkpoint_sha256':best['sha256'],'best_epoch':best['epoch'],'validation_predictions':archive['summary'],'science_verify':state['verify']}
        del saved,model,vectors;gc.collect()
        if not norms['scale_check_passed']:break
    print(json.dumps({'read_at_utc':started,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'status':'baseline_check_passed' if len(result)==3 and all(x['raw_m_norms']['scale_check_passed'] for x in result.values()) else 'STOP_REPORT_KIRO',
                     'seeds':result,'matched_dependency_sha256':source,'inputs_sha256':BASE_SHA,'history_sha256':HISTORY_PIN,
                     'checkpoint_pins':checkpoint_pins,'cpu':room,'test_read':False,'checkpoint_read':True,'vector_exported':False,'threshold_changed':False},sort_keys=True))

if __name__=='__main__':main()
