"""Private engineering parity on the fixed B subset, never full-SID acceptance."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

MODULE='analysis.g_full_subset_parity'
SCHEMA='g-full-subset-parity-request-v1'
A1_REQUEST_SHA='d129b4a325ac9063ffef26ac2ca7d208fdec908fd25470ef7615024afd77742b'
PRODUCER_COMMIT='96bfb9e6fcb4081168cf4d39563f50c58ee3fbad'
BASE_COMMIT='1ba8043b9363c3426aa76524b8d5daf9cd61a3ce'
NEW_FILES=('analysis/g_full_subset_parity.py','tests/test_g_full_subset_parity.py','docs/superpowers/reviews/2026-10-22-g-full-subset-parity-implementation.md')


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()


def read_request(path,pin):
    raw=Path(path).read_bytes()
    if digest(raw)!=pin:raise ValueError('request bytes differ')
    v=json.loads(raw)
    fields={'schema','stage','runner_root','runner_source','producer_root','a1_request','scratch_root','steps','device','ops_root','run_id','resources','runtime'}
    if type(v)is not dict or set(v)!=fields or v['schema']!=SCHEMA:raise ValueError('closed diagnostic request required')
    for name in ('runner_root','producer_root','scratch_root','ops_root'):
        p=Path(v[name])
        if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('absolute nonsymlink roots required')
    if v['stage'] not in ('preflight','compare'):raise ValueError('explicit diagnostic stage required')
    rec=v['a1_request']
    if type(rec)is not dict or set(rec)!={'path','sha256'} or rec['sha256']!=A1_REQUEST_SHA:raise ValueError('fixed original A1 request required')
    input_paths=[v[k]for k in ('runner_root','producer_root','scratch_root','ops_root')]+[rec['path']]
    if any(token in str(path).lower() for path in input_paths for token in ('holdout','week_26','week26')):raise ValueError('holdout path prohibited')
    if type(v['steps'])is not int or not 1<=v['steps']<=3 or v['device']!='cuda':raise ValueError('one to three CUDA engineering steps required')
    if type(v['run_id'])is not str or re.fullmatch('[a-z0-9][a-z0-9._-]{0,127}',v['run_id']) is None:raise ValueError('explicit diagnostic run ID required')
    resources=v['resources']
    if type(resources)is not dict or set(resources)!={'cpu_cores','ram_mib','vram_mib','disk_gib','wall_seconds'} or any(type(resources[k])is not int or resources[k]<1 for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')) or type(resources['wall_seconds'])is not float or not 0<resources['wall_seconds']<=172800.:raise ValueError('explicit bounded diagnostic resources required')
    if type(v['runtime'])is not dict or not v['runtime']:raise ValueError('pinned diagnostic runtime required')
    roots=[Path(v[k]) for k in ('runner_root','producer_root','scratch_root')]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(roots) for b in roots[i+1:]):raise ValueError('separate diagnostic roots required')
    scratch=Path(v['scratch_root']);ops=Path(v['ops_root'])
    if scratch==ops or scratch in ops.parents or ops in scratch.parents:raise ValueError('scratch overlaps diagnostic OPS')
    return v


def progress(stage):
    print(json.dumps({'schema':'g-subset-parity-progress-v1','stage':stage}),file=sys.stderr,flush=True)


def original_data_root(v):
    """Read the data channel only from authenticated frozen upstream bytes."""
    from analysis.sgns_subset_acceptance import record_bytes,absolute
    request=json.loads(record_bytes(v['a1_request']))
    value=request.get('data_root')
    if type(value)is not str:raise ValueError('fixed original data root required')
    root=absolute(value)
    if any(token in value.lower() for token in ('holdout','week_26','week26')) or not root.is_dir():raise ValueError('fixed original data root directory required')
    return str(root)


def own_binding(v,path,pin):
    from analysis.sgns_subset_acceptance import source_tree,record_bytes
    from analysis.sgns_subset_acceptance_a1 import transport_job
    path=Path(path)
    if encoded(read_request(path,pin))!=encoded(v):raise ValueError('diagnostic request snapshot differs')
    source_tree(Path(v['runner_root']),v['runner_source'])
    record={'path':str(path),'sha256':pin}
    runroot=Path(v['ops_root'])/'runs'/v['run_id'];runpath=runroot/'run.json'
    raw=record_bytes({'path':str(runpath),'sha256':digest(runpath.read_bytes())});run=json.loads(raw)
    argv=[sys.executable,'-B','-m',MODULE,v['stage'],'--request',str(path),'--request-sha256',pin]
    transport_job(run,runroot=runroot,code=Path(v['runner_root']),source=v['runner_source'],argv=argv,config=record,inputs=[record,v['a1_request']],resources=v['resources'],data_root=original_data_root(v),cublas=True,extra_env={'PYTHONDONTWRITEBYTECODE':'1'})
    if any(os.environ.get(k)!=val for k,val in run['env'].items()):raise ValueError('diagnostic process environment differs')
    return {'stage':v['stage'],'request':record,'request_sha256':pin,'run_id':v['run_id'],'run_manifest_sha256':digest(raw),'source':v['runner_source'],'runtime':v['runtime']}


def close_own_binding(v,path,pin,before):
    after=own_binding(v,path,pin)
    if encoded(after)!=encoded(before):raise ValueError('diagnostic final snapshot differs')
    return after


def publish_report(v,path,pin,before,report):
    progress('final-own-begin')
    close_own_binding(v,path,pin,before)
    progress('final-own-end')
    report.update(request_sha256=pin,diagnostic=before,original_a1_request=v['a1_request'])
    with (Path(v['scratch_root'])/'parity-receipt.json').open('xb') as stream:stream.write(encoded(report))
    return report


def state_digest(value):
    """Typed canonical state hash; tensor bytes include signed zero and dtype."""
    import torch
    def freeze(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous()
            return ['tensor',str(a.dtype),list(a.shape),a.reshape(-1).view(torch.uint8).numpy().tobytes().hex()]
        if isinstance(x,dict):return ['dict',[[freeze(k),freeze(v)] for k,v in sorted(x.items(),key=lambda p:repr(p[0]))]]
        if type(x)in (list,tuple):return [type(x).__name__,[freeze(v) for v in x]]
        if x is None or type(x)in (str,int,float,bool):return [type(x).__name__,x]
        raise TypeError('unsupported parity state type: '+str(type(x)))
    return digest(encoded(freeze(value)))


def compare_training(reference,adapted,codec,config,*,steps,device,native):
    """Two independent initializations; original scientific helpers unchanged."""
    import torch
    from vkgr import generative_real as gr, sid_depth
    from vkgr.generative_data_v2 import sid_targets_l
    if len(reference)!=len(adapted):raise ValueError('parity example count differs')
    tokens=sid_depth.SIDTokensLD(codec,50)
    def run(examples):
        gr._seed(config['seed'])
        model=native['make_model_arch_d'](codec,config,device=device)
        optimizer=native['ga'].make_optimizer_arch(model,config)
        cursor=native['gt'].EpochCursor(config['seed'],len(examples))
        records=[]
        for _ in range(steps):
            indices=cursor.next_batch(config['effective_batch'])
            if not indices:raise ValueError('bounded diagnostic crosses original evaluation boundary')
            rows=examples.rows(indices)
            tensors=sid_targets_l(rows,tokens)
            result=native['train_batch_d'](model,optimizer,rows,codec,tokens,config,device)
            rng={'cpu':torch.random.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device=='cuda' else []}
            model.eval()
            with torch.no_grad():scores=model(tensors[0].to(device),tensors[1].to(device))
            records.append({key:state_digest(val) for key,val in {'indices':indices,'rows':rows,'tensors':tensors,'loss':(result.loss_mean,result.examples,result.microbatch_sizes,result.optimizer_steps),'parameters':model.state_dict(),'optimizer':optimizer.state_dict(),'cursor':cursor.state_dict(),'rng':rng,'scores':scores}.items()})
        return records,model
    a,oracle=run(reference);b,model=run(adapted)
    if a!=b:raise ValueError('training parity differs: '+','.join(f'step{i}:{k}' for i,(x,y) in enumerate(zip(a,b)) for k in x if x[k]!=y[k]))
    return {'verified':True,'steps':a},(oracle,model)


def evaluation_signature(result):
    """Keep all deterministic outputs, excluding timing/RSS/VRAM and locations."""
    manifest=result['manifest'];rows=[]
    for record in manifest['shards']:
        shard=json.loads(Path(record['path']).read_bytes())
        rows.extend({k:v for k,v in row.items() if k!='latency_seconds'} for row in shard['rows'])
    return digest(encoded({'rows':rows,'aggregate':manifest['aggregate'],'complete':manifest['complete'],'query_count':manifest['query_count'],'completed_queries':manifest['completed_queries'],'computation_sha256':manifest['computation_sha256']}))


def execute(v,request_path,request_sha):
    """Called in fresh producer-root process before introducing runner namespace."""
    progress('own-begin')
    own=own_binding(v,request_path,request_sha)
    progress('own-end')
    progress('source-begin')
    if Path(__file__).resolve()!=Path(v['runner_root'])/'analysis/g_full_subset_parity.py':raise ValueError('diagnostic entry origin differs')
    from analysis import sgns_native_subset_a1 as original
    from analysis.sgns_subset_acceptance import source_tree
    producer=Path(v['producer_root']);runner=Path(v['runner_root']);scratch=Path(v['scratch_root'])
    req=original.load(Path(v['a1_request']['path']),v['a1_request']['sha256'])
    if req['source']['commit']!=PRODUCER_COMMIT:raise ValueError('fixed original producer commit differs')
    source_tree(producer,req['source']);source_tree(runner,v['runner_source'])
    # Exact existing baseline ledger plus only the three registered additions.
    names=subprocess.check_output(['git','-C',str(runner),'ls-tree','-r','--name-only',BASE_COMMIT],text=True).splitlines()
    baseline={n:digest(subprocess.check_output(['git','-C',str(runner),'show',BASE_COMMIT+':'+n])) for n in names}
    files=v['runner_source']['files']
    if set(files)!=set(baseline)|set(NEW_FILES) or any(files[n]!=sha for n,sha in baseline.items()):raise ValueError('closed runner source pinned delta differs')
    _,actual_runtime=original.identity(req)
    if encoded(actual_runtime)!=encoded(v['runtime']):raise ValueError('diagnostic/original runtime differs')
    progress('source-end')
    progress('input-check-begin')
    original.input_check(req)
    progress('input-check-end')
    engine,common,extra,_=original.engine_for(req,v['a1_request']['sha256'],req['source'])
    config=original.core.registered_config(original.sid_manifest(req)['sha256'])
    immutable=[producer,runner,*(Path(req[k]) for k in ('data_root','output_root','scratch_root','sid_root','ops_root','ops_code')),*[Path(r['path']) for r in req.values() if type(r)is dict and 'path' in r]]
    if scratch.exists() or any(scratch==p or scratch in p.parents or p in scratch.parents for p in immutable):raise ValueError('exclusive new engineering scratch required')
    scratch.mkdir(mode=0o700)
    progress('context-begin')
    context,engineering,sid=engine['_contexts'](*common,config,scratch/'original-context',**extra)
    progress('context-end')
    if engineering is not False or context.binding.get('synthetic') is not False:raise ValueError('original real subset context required')
    if v['stage']=='preflight':
        report={'schema':'g-full-subset-parity-preflight-v1','stage':'preflight','status':'verified','verified':True,
                'engineering':False,'training':False,'science_training':False,'test_read':False,'checkpoint_written':False,
                'model_created':False,'optimizer_created':False,'evaluation_executed':False,'full_preflight_passed':False,
                'scope':'original-real-subset-input-preflight-only','training_rows':len(context.examples),
                'validation_queries':len(context.queries),'context_sha256':digest(encoded(context.binding)),
                'config_sha256':digest(encoded(config))}
        progress('final-upstream-begin')
        if encoded(original.load(Path(v['a1_request']['path']),v['a1_request']['sha256']))!=encoded(req):raise ValueError('original A1 request changed')
        source_tree(producer,req['source'])
        if encoded(original.identity(req)[1])!=encoded(actual_runtime):raise ValueError('final original runtime differs')
        progress('final-upstream-end')
        return publish_report(v,request_path,request_sha,own,report)
    # All original input/source closure checks finish before adding runner modules.
    import analysis
    analysis.__path__.append(str(runner/'analysis'))
    from analysis import g_full_training_backend as backend
    if Path(backend.__file__).resolve()!=runner/'analysis/g_full_training_backend.py':raise ValueError('backend import origin differs')
    rawpath=scratch/'train.jsonl'
    with rawpath.open('xb') as out:
        for start in range(0,len(context.examples),512):
            for row in context.examples.rows(list(range(start,min(start+512,len(context.examples))))):out.write(encoded(row)+b'\n')
    lazy=backend.lazy_examples(rawpath,scratch/'offsets',len(context.examples),digest(rawpath.read_bytes()),lambda:None)
    adapted=backend.diagnostic_context(lazy,context.codec,context.queries,context.slice_definition,context.binding)
    backend_native=backend.native_functions()
    for name in ('make_model_arch_d','train_batch_d','_evaluator_d','_gate','_runtime'):
        if backend_native[name] is not engine[name]:raise ValueError('native helper identity differs')
    engine['_gate'](None,1)
    runtime=engine['_runtime'](False)
    report,(oracle,model)=compare_training(context.examples,adapted.examples,context.codec,config,steps=v['steps'],device=v['device'],native=engine)
    from vkgr import sid_depth
    tokens=sid_depth.SIDTokensLD(context.codec,50)
    evaluate=engine['_evaluator_d'](None)
    signatures=[]
    ephemeral_state_sha=state_digest(oracle.state_dict())
    for label,network,ctx in (('original',oracle,context),('backend',model,adapted)):
        kind,result=evaluate(network,ctx,tokens,config,runtime,scratch/('evaluation-'+label),ephemeral_state_sha,lambda:None,1)
        signatures.append((kind,evaluation_signature(result)))
    if signatures[0]!=signatures[1]:raise ValueError('evaluation parity differs')
    report.update(schema='g-full-subset-parity-receipt-v1',stage='compare',engineering=True,science_training=False,full_preflight_passed=False,checkpoint_written=False,test_read=False,validation_role='fixed-authorized-engineering-reference',evaluation_sha256=signatures[0][1],excluded_fields=['latency_seconds','observations','artifact_paths','artifact_sha256_with_timing','context_binding'],exclusion_reason='runtime observations, artifact locations, and explicitly distinct engineering context provenance are not scientific outputs',scope='subset-storage-input-parity-only')
    if encoded(original.load(Path(v['a1_request']['path']),v['a1_request']['sha256']))!=encoded(req):raise ValueError('original A1 request changed')
    source_tree(producer,req['source'])
    if encoded(original.identity(req)[1])!=encoded(actual_runtime):raise ValueError('final original runtime differs')
    return publish_report(v,request_path,request_sha,own,report)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('dry','preflight','compare'));parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv)
    try:
        v=read_request(args.request,args.request_sha256)
        if args.stage=='dry':print(encoded({'schema':SCHEMA,'status':'DIAGNOSTIC_DRY','data_opened':False,'verified':False,'engineering':True,'full_preflight_passed':False}).decode());return 0
        if args.stage!=v['stage']:raise ValueError('diagnostic CLI/request stage differs')
        if Path.cwd()!=Path(v['runner_root']):raise ValueError('diagnostic launcher cwd differs')
        # A clean Python process imports original modules exclusively first.
        bootstrap=('import importlib.util,sys,json;from pathlib import Path;'
            'p=Path(sys.argv[1]);sys.path[:0]=[str(p),str(p/"src")];'
            's=importlib.util.spec_from_file_location("_g_parity",sys.argv[2]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'v=m.read_request(sys.argv[3],sys.argv[4]);print(m.encoded(m.execute(v,sys.argv[3],sys.argv[4])).decode())')
        import os
        env={k:val for k,val in os.environ.items() if k!='PYTHONHOME'};env['PYTHONDONTWRITEBYTECODE']='1'
        result=subprocess.run([sys.executable,'-B','-c',bootstrap,v['producer_root'],str(Path(__file__).resolve()),args.request,args.request_sha256],cwd=v['producer_root'],env=env)
        return result.returncode
    except (ValueError,OSError) as exc:
        print(encoded({'status':'REJECTED','verified':False,'error':str(exc)}).decode());return 2


if __name__=='__main__':raise SystemExit(main())
