"""Fixed M2 metadata and shallow timing inventory; stdout only, no model reads.

Supports python -B - < source: no __file__ dependency. Never reads predictions,
checkpoints, embeddings, training rows or test. Does not execute M2 ops code.
"""
import copy
import datetime
import hashlib
import json
import sys
import math
from pathlib import Path
import re
import types

PHASE2=None
PHASE2_SHA='548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8'
OPS=None
DEP=None
COMMIT='8688f09b9f58fc5f3b3ef3f396099403222cc206'
PINS={42:'a07faf79e00983a450750cde78c22934ea0dbb46516beeb4006b925755a144d8',2024:'31ac1df62299e4543b167a46b4596797f1b127cf05219c83ab10ebecf7d4d0d9',2025:'b41e32a451a64030eff0fdc8da0714d92c3118ad4f0aab0a489b3638eb21fea5'}
BLOCKED=('test','checkpoint','embedding','prediction','vector','optimizer','weights','model')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pinned_module(path,pin,reads):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>1024*1024:
        raise ValueError('unsafe source')
    payload=path.read_bytes();actual=hashlib.sha256(payload).hexdigest()
    if actual!=pin:raise ValueError('source SHA mismatch')
    reads[str(path)]=actual
    module=types.ModuleType('m2_metadata_pinned_phase2')
    exec(compile(payload,str(path),'exec'),module.__dict__)
    return module


def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite metadata')
    if isinstance(value,dict):
        for item in value.values():finite_tree(item)
    if isinstance(value,list):
        for item in value:finite_tree(item)


def small_json(path,reads,pin=None):
    path=Path(path)
    if path.suffix!='.json' or any(x in path.name.lower() for x in BLOCKED):
        raise ValueError('prohibited metadata path')
    if path.is_symlink() or path.stat().st_size>65536:raise ValueError('unsafe small metadata')
    payload=path.read_bytes();actual=hashlib.sha256(payload).hexdigest()
    if pin is not None and actual!=pin:raise ValueError('artifact SHA mismatch')
    value=json.loads(payload);finite_tree(value);reads[str(path)]=actual
    return value


def prediction_ref(ref,pin,expected):
    if ref!=expected or not isinstance(pin,str) or re.fullmatch('[0-9a-f]{64}',pin) is None:
        raise ValueError('prediction ref/pin mismatch')
    return {'ref':ref,'sha256':pin,'read':False}


def metadata(manifest,entry,config,deployment):
    finite_tree(manifest)
    seed=config['seed'];rid=f'p-seqm2-k4-s{seed}';binding=manifest['binding']
    if seed not in PINS or manifest['config']!=config or deployment['commit']!=COMMIT or binding['source']['git_commit']!=COMMIT:
        raise ValueError('config/source mismatch')
    if binding.get('synthetic') is not False or binding.get('test_gate')!='closed_hash_only':raise ValueError('real validation/test gate mismatch')
    if entry['job']['run_id']!=rid or entry['job']['seed']!=seed:raise ValueError('job/seed mismatch')
    if digest(binding)!=entry['expected']['expected_binding_sha256']:raise ValueError('preflight binding mismatch')
    for key,pin in entry['expected']['data_hashes'].items():
        if binding[key]!=pin:raise ValueError('preflight data mismatch')
    sel=manifest['selection'];best=sel['best_epoch']
    rows=[e for e in manifest['evaluations'] if e['epoch']==best]
    if manifest['best']['epoch']!=best or len(rows)!=1:raise ValueError('best selection mismatch')
    ev=rows[0];ablation=copy.deepcopy(manifest['ablation_g0'])
    pred=prediction_ref(ev['predictions_ref'],ev['predictions_sha256'],f'predictions-epoch-{best:03d}.json')
    prediction_ref(ablation['predictions_ref'],ablation['predictions_sha256'],'predictions-best-g0.json')
    return {'run_id':rid,'seed':seed,'best_epoch':best,'stop_epoch':sel['epoch'],'best_metric':sel['best_metric'],
            'diagnostics':ev['diagnostics'],'ablation_g0':ablation,'best_predictions':pred,
            'source':binding['source'],'data_hashes':entry['expected']['data_hashes']}


def derive_verify(job,manifest_sha,root):
    v=copy.deepcopy(job);rid=job['run_id'];mp=str(root/'runs'/rid/'out/manifest.json')
    v.update(run_id=rid+'-verify',disk_gib=1,wall_hours=3.,inputs=[*job['inputs'],mp])
    argv=v['argv'];i=argv.index('--output');del argv[i:i+2]
    # Match the original ops rule: module entry's train subcommand follows -m module.
    if '-m' in argv:argv[argv.index('-m')+2]='verify'
    elif 'train' in argv:argv[argv.index('train')]='verify'
    else:raise ValueError('missing train command')
    argv.extend(['--manifest',mp,'--manifest-sha',manifest_sha])
    return v


def claimed(r,rid):
    matches=sorted((r.ROOT/'queue/claimed').glob('*-'+rid+'.json'))
    if len(matches)!=1:raise ValueError('unique claimed receipt required')
    return r.read(matches[0])[0]


def latency_inventory(directories,reads):
    receipts=[];skipped=[]
    for directory in directories:
        directory=Path(directory)
        if directory.is_symlink():raise ValueError('symlink inventory directory')
        if not directory.exists():continue
        candidates=sorted(directory.glob('*.json'))
        if len(candidates)>256:raise ValueError('unbounded shallow metadata inventory')
        for path in candidates:
            name=path.name.lower()
            if not (any(x in name for x in ('latency','timing','benchmark')) or name.startswith('summary')):continue
            if any(x in name for x in BLOCKED) or path.is_symlink() or path.stat().st_size>65536:
                skipped.append({'path':str(path),'reason':'prohibited name, symlink or over 64KiB; not read'})
                continue
            value=small_json(path,reads)
            receipts.append({'path':str(path),'sha256':reads[str(path)],'payload':value,'completeness':'NOT_ASSESSED'})
    return {'receipts':receipts,'skipped':skipped,'recursive':False,'complete_latency_verdict':'UNKNOWN',
            'scope':'Direct OPS, train/verify run and out JSON names containing latency/timing/benchmark or starting summary; not exhaustive server inventory.'}



def configure_paths(paths):
    global PHASE2, OPS, DEP
    if not isinstance(paths, dict) or set(paths) != {'ops', 'phase2', 'deployment'}:
        raise ValueError("exact private path keys required")
    if any(not isinstance(v, str) or not Path(v).is_absolute() or ".." in Path(v).parts for v in paths.values()):
        raise ValueError("absolute private locations required")
    PHASE2 = Path(paths['phase2'])
    OPS = Path(paths['ops'])
    DEP = Path(paths['deployment'])


def main():
    configure_paths(json.loads(sys.stdin.read(65537)))
    started=now();reads={};r=pinned_module(PHASE2,PHASE2_SHA,reads);r.READS.update(reads)
    plan,_=r.read(OPS/'plan.json');deployment,_=r.read(OPS/'deployment.json')
    entries={e['job']['run_id']:e for e in plan['entries']}
    if len(plan['entries'])!=3 or set(entries)!={f'p-seqm2-k4-s{s}' for s in PINS}:raise ValueError('exact registered seeds required')
    results=[];dirs=[OPS]
    for seed,pin in PINS.items():
        rid=f'p-seqm2-k4-s{seed}';entry=entries[rid];rd=r.ROOT/'runs'/rid
        # Check before the shared inspector can read its fixed raw verifier log.
        for suffix in ('-verify','-v1-verify'):
            log=r.ROOT/'runs'/(rid+suffix)/'logs/stdout.log'
            if log.exists() and (log.is_symlink() or log.stat().st_size>1024*1024):raise ValueError('unsafe verifier log')
        manifest,actual=r.read(rd/'out/manifest.json',pin)
        state=r.inspect_run(rid)
        if state.get('manifest_sha256')!=pin or not state['verified'] or state['terminal']['exit_code']!=0:
            raise ValueError('raw verified binding required')
        if any(state['queue'][k] for k in ('pending','rejected')):raise ValueError('unexpected train queue state')
        if claimed(r,rid)!=entry['job']:raise ValueError('claimed train/plan mismatch')
        if claimed(r,rid+'-verify')!=derive_verify(entry['job'],pin,r.ROOT):raise ValueError('claimed verify full binding mismatch')
        if entry['job']['config']!=str(DEP/'configs/seqm2'/(rid+'.json')):raise ValueError('fixed config path required')
        vqueue=r.queue(rid+'-verify')
        if any(vqueue[k] for k in ('pending','rejected')):raise ValueError('unexpected verify queue state')
        config,_=r.read(entry['job']['config'],entry['expected']['config_sha256'])
        result=metadata(manifest,entry,config,deployment)
        result.update(manifest_path=str(rd/'out/manifest.json'),manifest_sha256=actual,raw_verify=state['verify'])
        for verify in state['verify']['jobs']:
            for evidence in verify['evidence']:r.READS[evidence['path']]=evidence['sha256']
        results.append(result)
        for run in (rd,r.ROOT/'runs'/(rid+'-verify')):dirs.extend((run,run/'out'))
    latency=latency_inventory(dirs,r.READS)
    print(json.dumps({'read_at_utc':started,'finished_at_utc':now(),'states':results,'latency_inventory':latency,
          'read_artifacts_sha256':r.READS,'test_read':False,'checkpoint_read':False,'embedding_read':False,
          'prediction_read':False,'new_submit':False,'anchor_verdict':'NOT_ISSUED'},sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
