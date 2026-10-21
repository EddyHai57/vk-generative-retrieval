"""Formal A40 engineering jobs: resources and synthetic P exact cost only."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
YAML=ROOT/'configs/throughput/a40-m153999.yaml'
YAML_SHA='69f8039793ace8dfc67063ab94013a0ae958ad1da75e2f6e369dbc61582c8cc5'
REFERENCE=ROOT/'tests/fixtures/throughput_frozen_baselines.py'
REFERENCE_SHA='ec92bb0b9858e9da7e192f8cc32da96d5a274ad195413220695b093361b96a0b'
N_ITEMS=12_000_000
N_QUERIES=26_827
DIM=64
TOP_K=50  # 0ca61bc retrieval_experiment.evaluate_vectors default max(10,20,50)
SEED=42


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts:raise ValueError('absolute task path required')
    if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('symlink task path rejected')
    if p==ROOT or ROOT in p.parents or p in ROOT.parents:raise ValueError('work outside source tree required')
    return p


def selected():
    if sha_file(YAML)!=YAML_SHA:raise ValueError('selected engineering YAML drift')
    from vkgr.throughput_config import Engineering
    return Engineering.load(YAML)


def reference():
    if sha_file(REFERENCE)!=REFERENCE_SHA:raise ValueError('original P baseline source drift')
    namespace={};exec(compile(REFERENCE.read_bytes(),str(REFERENCE),'exec'),namespace)
    return namespace['blockwise_topk']


def array_bytes(n,q):return (n*DIM+q*DIM+n)*4


def bhkm_bound(n,k,depth,iterations,block_rows):
    runs=sum((n+block_rows-1)//block_rows+min(n,k**level) for level in range(depth))*iterations
    fit=n*k*16*depth*iterations+256*runs
    return {'items':n,'branching_scenario':k,'depth':depth,'max_iter_scenario':iterations,
            'block_rows':block_rows,'fit_bytes':fit,'fit_plus_replay_bytes':2*fit,
            'scenario_only':True,'scientific_branching_selected':False,'measured':False}


def build_arrays(root,n,q):
    import numpy as np
    paths=[root/name for name in ('items.npy','queries.npy','ids.npy')]
    if any(p.exists() or p.is_symlink() for p in paths):raise FileExistsError('synthetic inputs already exist')
    rng=np.random.default_rng(SEED)
    for path,shape in [(paths[0],(n,DIM)),(paths[1],(q,DIM))]:
        a=np.lib.format.open_memmap(path,mode='w+',dtype='f4',shape=shape)
        for start in range(0,len(a),8192):a[start:start+8192]=rng.normal(size=a[start:start+8192].shape).astype('f4')
        a.flush();del a
    a=np.lib.format.open_memmap(paths[2],mode='w+',dtype='u4',shape=(n,))
    for start in range(0,n,8192):a[start:start+8192]=np.arange(start,min(start+8192,n),dtype='u4')
    a.flush();del a
    return [{'path':p.name,'size':p.stat().st_size,'sha256':sha_file(p)} for p in paths]


def runtime(lock):
    from analysis.new_instance_prepare import runtime as original
    return original(lock)


def source_sha():
    return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()


def worker_argv(stage,root,lock,ops_root):
    return [sys.executable,'-B','-m','analysis.a40_cost',stage,'--root',str(root),
            '--lock',str(lock),'--ops-root',str(ops_root),'--execute-on-new-instance']


def accepted_stage(ops_root,stage,source,argv,input_pins,config_pin,payload):
    """Bind immutable producer run, successful terminal and original stdout.

    OPS_ROOT is the existing task-owned root, never an imported external queue.
    Artifact JSON alone is insufficient: the producer must have run this exact
    fixed source/argv with only the declared lock/engineering inputs.
    """
    ops_root=safe(ops_root);run_id='a40-exact-'+stage+'-v1';base=ops_root/'runs'/run_id
    def read(name):
        path=safe(base/name)
        if not path.is_file() or path.stat().st_size>2*1024**2:raise ValueError('producer receipt missing/oversize')
        return path.read_bytes()
    run_record=json.loads(read('run.json'));terminal=json.loads(read('terminal.json'));stdout=read('logs/stdout.log')
    if (run_record.get('schema')!='vkgr-run-v1' or terminal.get('schema')!='vkgr-terminal-v1'
        or run_record.get('run_id')!=run_id or terminal.get('run_id')!=run_id
        or terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0
        or run_record.get('kind')!='ops' or run_record.get('git',{}).get('commit')!=source
        or run_record.get('git',{}).get('dirty') is not False or run_record.get('cwd')!=str(ROOT)
        or run_record.get('argv')!=argv or run_record.get('config')!=config_pin):
        raise ValueError('producer run/terminal/source/argv binding differs')
    pins={p['path']:p['sha256'] for p in run_record.get('inputs',[])}
    if pins!=input_pins or len(pins)!=len(run_record.get('inputs',[])):
        raise ValueError('producer input closure differs')
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'))
    if canonical(json.loads(stdout))!=canonical(payload):raise ValueError('producer original stdout differs from artifact')
    return {'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'run_id':run_id}


def proof(ops_root,stage,root,lock,payload):
    pins={str(lock.resolve()):sha_file(lock)}
    if stage=='preflight':pins[str((root/'inputs.json').resolve())]=sha_file(root/'inputs.json')
    return accepted_stage(ops_root,stage,source_sha(),worker_argv(stage,root,lock,ops_root),pins,
                          {'path':str(YAML),'sha256':YAML_SHA},payload)


def generate(root,lock,ops_root):
    env=runtime(lock);config=selected();root=safe(root)
    if root.exists():raise FileExistsError('new synthetic work root required')
    from vkgr.throughput_config import guard
    guard(config,root.parent,projected_memory=64*1024**2,projected_disk=array_bytes(N_ITEMS,N_QUERIES)+4096)
    root.mkdir()
    files=build_arrays(root,N_ITEMS,N_QUERIES)
    value={'schema':'a40-p-exact-synthetic-v1','status':'generated','data_kind':'synthetic',
           'items':N_ITEMS,'queries':N_QUERIES,'dim':DIM,'top_k':TOP_K,'synthetic_seed':SEED,
           'files':files,'generator_sha256':sha_file(__file__),'selected_yaml_sha256':YAML_SHA,
           'baseline_sha256':REFERENCE_SHA,'science_data_read':False,**env}
    with (root/'inputs.json').open('x') as f:json.dump(value,f,sort_keys=True)
    return value


def inputs(root,lock,ops_root):
    import numpy as np
    env=runtime(lock);config=selected();root=safe(root)
    doc=json.loads((root/'inputs.json').read_text())
    expected={'schema':'a40-p-exact-synthetic-v1','data_kind':'synthetic','items':N_ITEMS,
              'queries':N_QUERIES,'dim':DIM,'top_k':TOP_K,'synthetic_seed':SEED,
              'generator_sha256':sha_file(__file__),'selected_yaml_sha256':YAML_SHA,'baseline_sha256':REFERENCE_SHA}
    if any(doc.get(k)!=v for k,v in expected.items()):raise ValueError('synthetic input contract drift')
    producer=proof(ops_root,'generate',root,lock,doc)
    names=['items.npy','queries.npy','ids.npy']
    if [r['path'] for r in doc['files']]!=names:raise ValueError('fixed synthetic input file set required')
    arrays=[]
    for record,shape,dtype in zip(doc['files'],[(N_ITEMS,DIM),(N_QUERIES,DIM),(N_ITEMS,)],['float32','float32','uint32']):
        path=safe(root/record['path'])
        if path.stat().st_size!=record['size'] or sha_file(path)!=record['sha256']:raise ValueError('synthetic bytes differ')
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        if a.shape!=shape or str(a.dtype)!=dtype:raise ValueError('synthetic shape/dtype differs')
        arrays.append(a)
    from vkgr.throughput_config import guard
    guard(config,root,projected_memory=2*array_bytes(N_ITEMS,N_QUERIES))
    return arrays,{**env,'generation_stdout_sha256':producer['stdout_sha256']}


def preflight(root,lock,ops_root):
    import numpy as np
    arrays,env=inputs(root,lock,ops_root)
    from vkgr.throughput_config import Engineering
    from vkgr.throughput_exact import exact_topk
    baseline=reference();rng=np.random.default_rng(SEED)
    q=rng.normal(size=(35,DIM)).astype('f4');items=rng.normal(size=(4103,DIM)).astype('f4')
    items[:4]=0;ids=np.arange(len(items),dtype='u4')[::-1].copy()
    expected_scores=[]
    def observed_einsum(*args,**kwargs):
        result=np.einsum(*args,**kwargs)
        expected_scores.append((result.shape,hashlib.sha256(result.tobytes()).hexdigest()))
        return result
    proxy=SimpleNamespace(**{**vars(np),'einsum':observed_einsum})
    observed=FunctionType(baseline.__code__,{**baseline.__globals__,'np':proxy},baseline.__name__,baseline.__defaults__)
    expected=observed(q,items,ids,20,query_block=32,item_block=4096)
    trace=None
    for workers in (1,2,8):
        actual,current=exact_topk(q,items,ids,20,Engineering(eval_workers=workers),trace=True)
        if actual.tobytes()!=expected.tobytes() or (trace is not None and current!=trace):
            raise ValueError('P original/worker parity differs')
        if [x for block in current for x in block]!=expected_scores:
            raise ValueError('P original score bytes differ')
        trace=current
    value={'status':'ready','engineering':False,'synthetic_input':True,'science_data_read':False,
            'items':N_ITEMS,'queries':N_QUERIES,'dim':DIM,'top_k':TOP_K,
            'worker_parity':[1,2,8],'parity_scope':'35x4103 synthetic; full fixed block and tails',
            'baseline_sha256':REFERENCE_SHA,'inputs_manifest_sha256':sha_file(root/'inputs.json'),
            'script_sha256':sha_file(__file__),'selected_yaml_sha256':YAML_SHA,
            'score_bytes_equal_to_original':True,**env}
    with (root/'preflight.json').open('x') as f:json.dump(value,f,sort_keys=True)
    return value


def require_preflight(doc,inputs_sha,script_sha,yaml_sha,generation_sha):
    expected={'status':'ready','engineering':False,'inputs_manifest_sha256':inputs_sha,
              'script_sha256':script_sha,'selected_yaml_sha256':yaml_sha,'worker_parity':[1,2,8],
              'score_bytes_equal_to_original':True,'synthetic_input':True,'science_data_read':False,
              'items':N_ITEMS,'queries':N_QUERIES,'dim':DIM,'top_k':TOP_K,
              'baseline_sha256':REFERENCE_SHA,'generation_stdout_sha256':generation_sha,
              'parity_scope':'35x4103 synthetic; full fixed block and tails'}
    if any(doc.get(k)!=v or type(doc.get(k))is not type(v) for k,v in expected.items()):
        raise ValueError('formal preflight binding differs')


def run(root,lock,ops_root):
    arrays,env=inputs(root,lock,ops_root);items,queries,ids=arrays;baseline=reference()
    pre=json.loads((root/'preflight.json').read_text())
    require_preflight(pre,sha_file(root/'inputs.json'),sha_file(__file__),YAML_SHA,env['generation_stdout_sha256'])
    predecessor=proof(ops_root,'preflight',root,lock,pre)
    output=root/'cost.json'
    if output.exists():raise FileExistsError('cost receipt already exists')
    started=time.perf_counter();cpu=time.process_time()
    predicted=baseline(queries,items,ids,TOP_K,query_block=32,item_block=4096)
    elapsed=time.perf_counter()-started;cpu_seconds=time.process_time()-cpu
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    value={'status':'exact_cost_measured','engineering':False,'synthetic_input':True,'science_data_read':False,
           'items':N_ITEMS,'queries':N_QUERIES,'dim':DIM,'top_k':TOP_K,'workers':1,
           'function_wall_seconds':elapsed,'function_cpu_seconds':cpu_seconds,'peak_rss_bytes':peak,
           'peak_scope':'whole worker including load/validation; not scoring-only',
           'input_state':'mmap; byte hashing before timer may warm the OS cache',
           'prediction_sha256':hashlib.sha256(predicted.tobytes()).hexdigest(),
           'baseline_sha256':REFERENCE_SHA,'script_sha256':sha_file(__file__),'selected_yaml_sha256':YAML_SHA,
           'inputs_manifest_sha256':sha_file(root/'inputs.json'),
           'preflight_stdout_sha256':predecessor['stdout_sha256'],
           'source':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),**env}
    with output.open('x') as f:json.dump(value,f,sort_keys=True)
    return value


def probe(root,lock,ops_root):
    runtime(lock);root=safe(root)
    if root.exists():raise FileExistsError('new resource output directory required')
    root.mkdir()
    from vkgr.throughput_probe import resource_manifest
    config=selected();value=resource_manifest(root,config)
    usage=shutil.disk_usage(root);bounds=[]
    for n in (1_200_000,12_000_000):
        for k in (48,229):
            b=bhkm_bound(n,k,3,10,config.bhkm_rows)
            b['fit_plus_replay_capacity_fits']=usage.used+b['fit_plus_replay_bytes']<=min(int(usage.total*.8),config.disk_limit_bytes)
            bounds.append(b)
    value['bhkm_retention_scenarios']=bounds
    value['bhkm_scope']='conservative source max_iter=10 scenarios, not scientific K selection or measured cost'
    value['selected_yaml_sha256']=YAML_SHA
    with (root/'resources.json').open('x') as f:json.dump(value,f,sort_keys=True)
    return value


def job(args):
    from analysis.a40_ops import load_ops
    common,launcher,scheduler,status=load_ops(args.ops_code)
    stage=args.stage;run_id='a40-engineering-probe-v1' if stage=='probe' else 'a40-exact-'+stage+'-v1'
    cpu,ram,disk,hours=(1,16384,8,2) if stage=='generate' else (1,16384,0,48 if stage=='run' else 2)
    if stage=='preflight':cpu=9  # eight pure workers plus parent/receipt work
    argv=worker_argv(stage,args.root,args.lock,args.ops_root)
    raw=['--root',str(args.ops_root),'--run-id',run_id,'--kind','ops','--cwd',str(ROOT),
         '--config',str(YAML),'--input',str(args.lock),'--env','PYTHONPATH='+str(ROOT/'src'),
         '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
         '--cpu-cores',str(cpu),'--ram-mib',str(ram),'--vram-mib','0','--disk-gib',str(disk),'--wall-hours',str(hours)]
    if stage in ('preflight','run'):raw+=['--input',str(args.root/'inputs.json')]
    if stage=='run':raw+=['--input',str(args.root/'preflight.json')]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*raw,'--',*argv]));meta=launcher.prepare(spec)
    if args.command=='prepare':return {'status':'prepared_dry','spec':spec,'metadata':meta}
    root=launcher.resolve_root(str(args.ops_root))
    if (root/'runs'/run_id).exists():raise ValueError('original run exists')
    for state in ('pending','claimed','rejected'):
        if list((root/'queue'/state).glob('*-'+run_id+'.json')):raise ValueError('original run queued')
    path=scheduler.submit(root,spec)
    return {'status':'written','run_id':run_id,'queue_file':path.name}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for name in ('generate','preflight','run','probe','prepare','submit'):
        c=sub.add_parser(name);c.add_argument('--root',type=Path,required=True);c.add_argument('--lock',type=Path,required=True)
        c.add_argument('--ops-root',type=Path,required=True)
        if name in ('prepare','submit'):
            c.add_argument('--stage',choices=('probe','generate','preflight','run'),required=True)
            c.add_argument('--ops-code',required=True)
        else:c.add_argument('--execute-on-new-instance',action='store_true',required=True)
    a=p.parse_args(argv)
    if a.command=='dry':value={'status':'ready','items':N_ITEMS,'queries':N_QUERIES,'top_k':TOP_K,'science_data_read':False}
    elif a.command in ('prepare','submit'):value=job(a)
    else:value=globals()[a.command](a.root,a.lock,a.ops_root)
    print(json.dumps(value,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
