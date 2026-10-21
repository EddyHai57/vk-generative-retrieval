"""Fixed OPS adapter for synthetic P catalog diagnostics and original-cost parity."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from analysis.a40_cost import sha_file as digest
from analysis.p_query_blocks import parity, bind_original_cost, git
from vkgr.p_query_blocks import parallel_topk, cpu_capacity
from vkgr.throughput_config import Engineering, guard

ROOT=Path(__file__).resolve().parents[1]
P3C='docs/p3c_validation.md'
P3C_SHA='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
COST_SHA='a18d2523df8ae5e002d161b416e0a229c994884891b9e65e593a63df93a0a15f'
MIB=1024**2
MEASUREMENT_SOURCE='43eb5bec6d32a5d37746a033452824fd399f2dc4'
SOURCE_DELTA_PATHS={'analysis/p_query_sprint_ops.py','tests/test_p_query_sprint_capacity.py',
                    'tests/fixtures/p_query_sprint_capacity_chain.py',
                    'docs/superpowers/reviews/2026-10-21-p-query-sprint-capacity-implementation.md'}


def absolute(value):
    path=Path(value)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('absolute non-symlink path required')
    return path


def read(path):
    path=absolute(path)
    if path.stat().st_size>2*MIB:raise ValueError('receipt/request exceeds size limit')
    return json.loads(path.read_text())


def validate_request(req):
    required={'schema','stage','run_id','runner_source','original_source','workers','query_count',
              'engine','root','lock','original_ops_root','ops_root','output'}
    if set(req)-required-{'measurement_request','previous_failure_request','previous_failure_request_sha256','source_delta','source_delta_sha256'} or required-set(req) or req['schema']!='p-query-sprint-v1':
        raise ValueError('fixed sprint request contract required')
    if req['stage'] not in ('diagnostic','cost-parity') or type(req['workers'])is not int or req['workers'] not in (1,2,4,8):
        raise ValueError('fixed stage/workers required')
    count=req['query_count']
    if type(count)is not int or count not in ((32,64,256) if req['stage']=='diagnostic' else (26827,)):
        raise ValueError('fixed original query prefix required')
    for key in ('runner_source','original_source'):
        if type(req[key])is not str or len(req[key])!=40 or any(c not in '0123456789abcdef' for c in req[key]):raise ValueError('fixed source SHA required')
    expected='p-query-'+req['stage']+'-q'+str(count)+'-w'+str(req['workers'])+'-v'
    if not req['run_id'].startswith(expected) or not req['run_id'][len(expected):].isdigit():raise ValueError('descriptive fixed run identity required')
    paths={key:absolute(req[key]) for key in ('engine','root','lock','original_ops_root','ops_root','output')}
    output=paths['output']
    for parent in (ROOT,paths['root'],paths['engine'],paths['original_ops_root'],paths['ops_root']):
        if output==parent or output in parent.parents or parent in output.parents:raise ValueError('exclusive outside-source/input/ops output required')
    if 'previous_failure_request'in req or 'previous_failure_request_sha256'in req:
        if (req.get('run_id')!='p-query-diagnostic-q32-w2-v2' or req['stage']!='diagnostic' or count!=32 or req['workers']!=2
            or 'measurement_request'in req or 'previous_failure_request'not in req or type(req.get('previous_failure_request_sha256'))is not str
            or len(req['previous_failure_request_sha256'])!=64 or any(c not in '0123456789abcdef' for c in req['previous_failure_request_sha256'])):
            raise ValueError('fixed retry1 request and original request SHA required')
        absolute(req['previous_failure_request'])
    elif 'measurement_request'in req:absolute(req['measurement_request'])
    elif req['stage']!='diagnostic' or count!=32 or req['run_id']!=expected+'1':
        raise ValueError('subsequent jobs require a measured diagnostic producer')
    if 'source_delta'in req or 'source_delta_sha256'in req:
        if ('measurement_request'not in req or 'source_delta'not in req
            or type(req.get('source_delta_sha256'))is not str or len(req['source_delta_sha256'])!=64
            or any(c not in '0123456789abcdef' for c in req['source_delta_sha256'])):
            raise ValueError('measurement source delta path and SHA required')
        absolute(req['source_delta'])
    return req


def load_request(path):return validate_request(read(path))


def source(root,expected,*,overlay=False):
    root=absolute(root)
    if git(root,'rev-parse','HEAD')!=expected:raise ValueError('fixed source SHA differs')
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).splitlines()
    if dirty and (not overlay or {line[3:] for line in dirty}!={P3C}):raise ValueError('unregistered source differences')
    files={}
    for name in git(root,'ls-tree','-r','--name-only',expected).splitlines():
        wanted=hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',expected+':'+name])).hexdigest()
        actual=digest(absolute(root/name))
        if actual!=wanted and not (overlay and name==P3C and actual==P3C_SHA):raise ValueError('tracked source bytes differ: '+name)
        files[name]=actual
    if overlay and (root/P3C).exists():
        actual=digest(root/P3C)
        if dirty and actual!=P3C_SHA:raise ValueError('P3C data-channel identity differs')
        if P3C not in files:files[P3C]=actual
    return {'commit':expected,'files_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'files_count':len(files),'p3c_data_channel':bool(dirty)}


def original(req,*,arrays=False):
    root=absolute(req['engine']);ledger=source(root,req['original_source'])
    path=root/'analysis/a40_cost.py'
    if digest(path)!=COST_SHA:raise ValueError('original cost script identity differs')
    # The original module imports runtime from the active analysis package.
    for name in ('analysis/new_instance_prepare.py','src/vkgr/throughput_config.py','src/vkgr/throughput_cgroups.py'):
        if digest(root/name)!=digest(ROOT/name):raise ValueError('original runtime/guard dependency differs')
    spec=importlib.util.spec_from_file_location('_p_sprint_original_cost',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.reference();old.selected()
    data=absolute(req['root']);lock=absolute(req['lock']);ops=absolute(req['original_ops_root'])
    doc=read(data/'inputs.json')
    expected={'schema':'a40-p-exact-synthetic-v1','data_kind':'synthetic','items':old.N_ITEMS,'queries':old.N_QUERIES,
              'dim':old.DIM,'top_k':old.TOP_K,'synthetic_seed':old.SEED,'generator_sha256':COST_SHA,
              'selected_yaml_sha256':old.YAML_SHA,'baseline_sha256':old.REFERENCE_SHA}
    if any(doc.get(k)!=v or type(doc.get(k))is not type(v) for k,v in expected.items()):
        raise ValueError('original synthetic input contract differs')
    generation=old.proof(ops,'generate',data,lock,doc)
    pre=read(data/'preflight.json')
    old.require_preflight(pre,digest(data/'inputs.json'),COST_SHA,old.YAML_SHA,generation['stdout_sha256'])
    predecessor=old.proof(ops,'preflight',data,lock,pre)
    binding={'original_source':req['original_source'],'source_ledger':ledger,'inputs_manifest_sha256':digest(data/'inputs.json'),
             'lock_sha256':digest(lock),'generation_stdout_sha256':generation['stdout_sha256'],
             'preflight_stdout_sha256':predecessor['stdout_sha256']}
    opened=old.inputs(data,lock,ops)[0] if arrays else None
    return old,binding,opened


def diagnostic(req,arrays,config):
    items,queries,ids=arrays;count=req['query_count'];path=absolute(req['output'])/('queries-q'+str(count)+'.npy')
    if len(queries)<count:raise ValueError('original query prefix unavailable')
    guard(config,path.parent,projected_disk=count*queries.shape[1]*4+4096)
    with path.open('xb') as stream:np.save(stream,queries[:count],allow_pickle=False)
    copied=np.load(path,mmap_mode='r',allow_pickle=False)
    if copied.tobytes()!=queries[:count].tobytes():raise ValueError('query prefix copy bytes differ')
    value=parity(copied,items,ids,config,k=50)
    value.update({'full_query_set':False,'scope':'original synthetic catalog with aligned query prefix only',
                  'query_copy':{'file':path.name,'sha256':digest(path),'source_query_sha256':digest(queries.filename),
                                'ordinal_start':0,'ordinal_stop':count}})
    return value


def formal_cost(req,config):
    arrays,cost,producer=bind_original_cost(absolute(req['engine']),req['original_source'],absolute(req['root']),
                                          absolute(req['lock']),absolute(req['original_ops_root']))
    items,queries,ids=arrays;start=time.perf_counter()
    predicted,_=parallel_topk(queries,items,ids,50,config)
    prediction=hashlib.sha256(predicted.tobytes()).hexdigest()
    if prediction!=cost['prediction_sha256']:raise ValueError('original cost prediction bytes differ')
    return {'status':'verified','queries':len(queries),'items':len(items),'full_query_set':True,
            'original_cost_terminal_compared':True,'prediction_sha256':prediction,
            'original_cost_stdout_sha256':producer['stdout_sha256'],'parallel_wall_seconds':time.perf_counter()-start,
            'score_bytes_equal_to_original':'UNKNOWN: original cost did not retain score traces'}


def measured_resources(req,previous,binding):
    peak=previous.get('resources',{}).get('process_tree_peak_rss_upper_bound_bytes')
    if (previous.get('status')!='verified' or previous.get('stage')!='diagnostic' or previous.get('workers')!=req['workers']
        or previous.get('input_binding')!=binding or type(peak)is not int or peak<=0):
        raise ValueError('same-worker measured producer/input binding required')
    # Covers extra query/result/trace buffers without changing fixed arithmetic.
    extra=max(0,req['query_count']-previous['queries'])*4096
    return {'cpu_cores':req['workers']+1 if req['workers']>1 else 1,
            'ram_mib':max(8192,math.ceil((peak*1.25+2*1024**3+extra)/MIB)),
            'measurement':'measured_upper_bound','measurement_source_queries':previous['queries']}


def resources(req,previous=None,binding=None):
    cpu=req['workers']+1 if req['workers']>1 else 1
    if cpu>cpu_capacity():raise ValueError('CPU quota cannot admit requested workers plus parent')
    value=measured_resources(req,previous,binding) if previous is not None else {'cpu_cores':cpu,'ram_mib':98304,'measurement':'not_measured'}
    value.update({'vram_mib':0,'disk_gib':1,'wall_hours':8 if req['stage']=='diagnostic' else 48})
    return value


def unoccupied(ops_root,run_id):
    if (ops_root/'runs'/run_id).exists():raise FileExistsError('run identity already exists')
    for state in ('pending','claimed','rejected'):
        if list((ops_root/'queue'/state).glob('*-'+run_id+'.json')):raise FileExistsError('queue identity already exists: '+state)


def invocation(request_path):
    return [sys.executable,'-B','-m','analysis.p_query_sprint_ops','run','--request',str(request_path),'--execute-on-new-instance']


def input_files(req,request_path,*,runner_root=None):
    runner_root=ROOT if runner_root is None else runner_root
    paths=[absolute(req['lock']),absolute(req['root'])/'inputs.json',absolute(req['root'])/'preflight.json',
           absolute(req['engine'])/'analysis/a40_cost.py',runner_root/'analysis/p_query_sprint_ops.py']
    stages=['generate','preflight']+(['run'] if req['stage']=='cost-parity' else [])
    for stage in stages:
        base=absolute(req['original_ops_root'])/'runs'/('a40-exact-'+stage+'-v1')
        paths.extend(base/name for name in ('run.json','terminal.json','logs/stdout.log'))
    if req['stage']=='cost-parity':paths.append(absolute(req['root'])/'cost.json')
    if 'source_delta'in req:paths.append(absolute(req['source_delta']))
    if 'measurement_request'in req:
        old=load_request(req['measurement_request']);base=absolute(old['ops_root'])/'runs'/old['run_id']
        paths.extend([absolute(req['measurement_request']),absolute(old['output'])/'result.json',
                      *(base/name for name in ('run.json','terminal.json','logs/stdout.log'))])
    if 'previous_failure_request'in req:
        path=absolute(req['previous_failure_request']);old=load_request(path);base=absolute(old['ops_root'])/'runs'/old['run_id']
        paths.extend([path,*(base/name for name in ('run.json','terminal.json','logs/stdout.log','logs/stderr.log'))])
    return paths


def ops_resources(declaration):
    return {**{key:declaration[key] for key in ('cpu_cores','ram_mib','vram_mib','disk_gib')},
            'wall_seconds':declaration['wall_hours']*3600.0}


def accepted(request_path,payload,*,runner_root=None,runner_ledger=None):
    runner_root=ROOT if runner_root is None else runner_root
    req=load_request(request_path);base=absolute(req['ops_root'])/'runs'/req['run_id']
    record=read(base/'run.json');terminal=read(base/'terminal.json');stdout=read(base/'logs/stdout.log')
    expected_pins={str(path):digest(path) for path in input_files(req,request_path,runner_root=runner_root)}
    pins={pin['path']:pin['sha256'] for pin in record.get('inputs',[])}
    dirty=record.get('git',{}).get('dirty_files')
    expected_dirty=([{'path':P3C,'sha256':P3C_SHA}] if (runner_root/P3C).exists() and digest(runner_root/P3C)==P3C_SHA else [])
    if (record.get('schema')!='vkgr-run-v1' or terminal.get('schema')!='vkgr-terminal-v1'
        or record.get('run_id')!=req['run_id'] or terminal.get('run_id')!=req['run_id']
        or terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0 or record.get('kind')!='ops'
        or record.get('git')!={'commit':req['runner_source'],'dirty':bool(expected_dirty),'dirty_files':expected_dirty}
        or record.get('cwd')!=str(runner_root) or record.get('argv')!=invocation(request_path)
        or record.get('config')!={'path':str(request_path),'sha256':digest(request_path)}
        or payload.get('schema')!='p-query-sprint-result-v1' or payload.get('status')!='verified'
        or payload.get('run_id')!=req['run_id'] or payload.get('request_sha256')!=digest(request_path)
        or payload.get('argv')!=invocation(request_path) or payload.get('runner_source',{}).get('commit')!=req['runner_source']
        or (runner_ledger is not None and payload.get('runner_source')!=runner_ledger)
        or record.get('resources')!=ops_resources(payload.get('resources',{}))
        or payload.get('resources',{}).get('cpu_cores')!=(req['workers']+1 if req['workers']>1 else 1)
        or pins!=expected_pins or len(record.get('inputs',[]))!=len(pins) or stdout!=payload):
        raise ValueError('diagnostic producer terminal/stdout/source/argv/input binding differs')
    return {'run_id':req['run_id'],'stdout_sha256':digest(base/'logs/stdout.log')}


def previous_failure(req):
    validate_request(req)
    path=absolute(req['previous_failure_request'])
    if digest(path)!=req['previous_failure_request_sha256']:raise ValueError('previous failed request SHA differs')
    old=load_request(path)
    if ('previous_failure_request'in old or 'measurement_request'in old
        or old['run_id']!='p-query-diagnostic-q32-w2-v1'
        or old['runner_source']!='669297abe65c3ec699b3476c13ccf867736ab56a'):
        raise ValueError('only fixed original v1 failure can authorize retry1')
    unchanged=set(old)-{'run_id','runner_source','output'}
    if any(req.get(key)!=old[key] for key in unchanged):raise ValueError('retry original scientific/input/resource request differs')
    previous_output=absolute(old['output']);output=absolute(req['output'])
    if (not previous_output.is_dir() or output==previous_output or output in previous_output.parents or previous_output in output.parents
        or (previous_output/'result.json').exists() or (previous_output/'result.json').is_symlink()):
        raise ValueError('previous failed output must remain separate with no result')
    base=absolute(old['ops_root'])/'runs'/old['run_id']
    record=read(base/'run.json');terminal=read(base/'terminal.json')
    oldroot=absolute(record['cwd']);ledger=source(oldroot,old['runner_source'],overlay=True)
    dirty=[{'path':P3C,'sha256':P3C_SHA}] if ledger['p3c_data_channel'] else []
    expected_pins={str(oldroot/'analysis/p_query_sprint_ops.py') if p==Path(__file__).resolve() else str(p):
                   digest(oldroot/'analysis/p_query_sprint_ops.py') if p==Path(__file__).resolve() else digest(p)
                   for p in input_files(old,path)}
    pins={p['path']:p['sha256'] for p in record.get('inputs',[])}
    stdout=absolute(base/'logs/stdout.log');stderr=absolute(base/'logs/stderr.log')
    if stdout.stat().st_size!=0 or stderr.stat().st_size>2*MIB:raise ValueError('previous failure stdout/stderr invalid')
    error=stderr.read_text()
    if ('throughput_workers.py'not in error or not error.rstrip().endswith('MemoryError: one task exceeds input/output reservation cap')
        or record.get('schema')!='vkgr-run-v1' or terminal.get('schema')!='vkgr-terminal-v1'
        or record.get('run_id')!=old['run_id'] or terminal.get('run_id')!=old['run_id']
        or terminal.get('status')!='failed' or terminal.get('exit_code')!=1 or record.get('kind')!='ops'
        or record.get('git')!={'commit':old['runner_source'],'dirty':bool(dirty),'dirty_files':dirty}
        or record.get('argv')!=invocation(path) or record.get('config')!={'path':str(path),'sha256':digest(path)}
        or record.get('resources')!={'cpu_cores':3,'ram_mib':98304,'vram_mib':0,'disk_gib':1,'wall_seconds':28800.0}
        or pins!=expected_pins or len(record.get('inputs',[]))!=len(pins)):
        raise ValueError('fixed previous failure producer binding differs')
    return {'kind':'previous_failure','run_id':old['run_id'],'request_sha256':digest(path),
            'run_sha256':digest(base/'run.json'),'terminal_sha256':digest(base/'terminal.json'),
            'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr)}


def measurement_source(req,old):
    """Closed transport-only delta; both actual worktrees must match full Git ledgers."""
    if old['runner_source']!=MEASUREMENT_SOURCE:raise ValueError('fixed old measurement source required')
    if 'source_delta'not in req:raise ValueError('cross-source measurement requires a fixed delta')
    path=absolute(req['source_delta'])
    if digest(path)!=req['source_delta_sha256']:raise ValueError('measurement source delta SHA differs')
    if ROOT in path.parents:raise ValueError('source delta must be outside source tree')
    delta=read(path)
    if (set(delta)!={'schema','old_source','new_source','files'} or delta['schema']!='p-query-source-delta-v1'
        or delta['old_source']!=MEASUREMENT_SOURCE or delta['new_source']!=req['runner_source']
        or type(delta['files'])is not dict or not delta['files']
        or not set(delta['files'])<=SOURCE_DELTA_PATHS or 'analysis/p_query_sprint_ops.py'not in delta['files']):
        raise ValueError('closed fixed measurement source delta required')
    base=absolute(old['ops_root'])/'runs'/old['run_id']
    oldroot=absolute(read(base/'run.json')['cwd'])
    if oldroot in path.parents:raise ValueError('source delta must be outside producer tree')
    oldledger=source(oldroot,MEASUREMENT_SOURCE,overlay=True)
    newledger=source(ROOT,req['runner_source'],overlay=True)
    if not oldledger['p3c_data_channel'] or not newledger['p3c_data_channel'] or digest(oldroot/P3C)!=digest(ROOT/P3C):
        raise ValueError('measurement P3C data channel differs')
    names=git(ROOT,'diff','--name-only',MEASUREMENT_SOURCE,req['runner_source']).splitlines()
    if set(names)!=set(delta['files']):raise ValueError('source delta paths differ')
    for name in names:
        before=digest(oldroot/name) if (oldroot/name).exists() else None
        after=digest(ROOT/name)
        if (delta['files'][name]!={'old_sha256':before,'new_sha256':after}
            or (before is not None and name!='analysis/p_query_sprint_ops.py')):
            raise ValueError('source delta bytes differ: '+name)
    return oldroot,oldledger


def measurement(req,binding):
    if 'measurement_request'not in req:return None,None
    path=absolute(req['measurement_request']);old=load_request(path)
    previous=read(absolute(old['output'])/'result.json')
    if old['runner_source']!=req['runner_source']:
        root,ledger=measurement_source(req,old)
        proof=accepted(path,previous,runner_root=root,runner_ledger=ledger)
        proof['source_delta_sha256']=req['source_delta_sha256']
    else:
        if 'source_delta'in req:raise ValueError('same-source measurement cannot carry a delta')
        proof=accepted(path,previous)
    measured_resources(req,previous,binding)
    return previous,proof


def plan(request_path,*,open_arrays=False):
    req=load_request(request_path);runner=source(ROOT,req['runner_source'],overlay=True)
    old,binding,arrays=original(req)
    previous,proof=measurement(req,binding)
    if 'previous_failure_request'in req:proof=previous_failure(req)
    declaration=resources(req,previous,binding)
    if open_arrays and req['stage']=='diagnostic':
        config=Engineering(memory_limit_bytes=int(declaration['ram_mib']*MIB*.9))
        guard(config,absolute(req['output']).parent,projected_memory=2*old.array_bytes(old.N_ITEMS,old.N_QUERIES))
        arrays=old.inputs(absolute(req['root']),absolute(req['lock']),absolute(req['original_ops_root']))[0]
    if req['stage']=='cost-parity':
        # Reuse the reviewed successful original-terminal gate even at prepare.
        if not open_arrays:
            cost=read(absolute(req['root'])/'cost.json')
            data=absolute(req['root']);lock=absolute(req['lock']);ops=absolute(req['original_ops_root'])
            pins={str(p):digest(p) for p in (lock,data/'inputs.json',data/'preflight.json')}
            old.accepted_stage(ops,'run',req['original_source'],old.worker_argv('run',data,lock,ops),pins,
                               {'path':str(old.YAML),'sha256':old.YAML_SHA},cost)
    return req,runner,binding,arrays,declaration,proof


def job(args):
    request_path=absolute(args.request)
    req,runner,binding,_,declaration,proof=plan(request_path)
    output=absolute(req['output'])
    if output.exists():raise FileExistsError('exclusive new output directory required')
    unoccupied(absolute(req['ops_root']),req['run_id'])
    from analysis.a40_ops import load_ops
    _,launcher,scheduler,_=load_ops(args.ops_code)
    raw=['--root',req['ops_root'],'--run-id',req['run_id'],'--kind','ops','--cwd',str(ROOT),
         '--config',str(request_path),'--env','PYTHONPATH='+str(ROOT/'src'),
         '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1']
    for key in ('cpu_cores','ram_mib','vram_mib','disk_gib','wall_hours'):raw+=['--'+key.replace('_','-'),str(declaration[key])]
    for path in input_files(req,request_path):raw+=['--input',str(path)]
    if runner['p3c_data_channel']:raw+=['--pinned-file',P3C+'='+P3C_SHA]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*raw,'--',*invocation(request_path)]))
    metadata=launcher.prepare(spec)
    value={'status':'prepared_dry','spec':spec,'metadata':metadata,'input_binding':binding,
           'runner_source':runner,'resources':declaration,'measurement_producer':None if 'previous_failure_request'in req else proof,
           'previous_failure_producer':proof if 'previous_failure_request'in req else None}
    if args.command=='submit':
        path=scheduler.submit(launcher.resolve_root(req['ops_root']),spec)
        value.update(status='written',queue_file=path.name,claimed_verified=False)
    return value


@contextmanager
def phase(stage):
    def emit(event):
        print(json.dumps({'stage':stage,'event':event,'monotonic_seconds':time.monotonic()},sort_keys=True),file=sys.stderr,flush=True)
    emit('begin')
    try:
        yield
    except BaseException:
        emit('error');raise
    else:emit('end')


def execute(args):
    request_path=absolute(args.request)
    with phase('plan_and_inputs'):
        req,runner,binding,arrays,declaration,proof=plan(request_path,open_arrays=True)
    output=absolute(req['output'])
    # Each fixed prefetch=1 worker gets the prior conservative one-task cap.
    config=Engineering(eval_workers=req['workers'],use_mmap=True,prefetch=1,
                       memory_limit_bytes=int(declaration['ram_mib']*MIB*.9),max_in_flight_bytes=128*MIB*req['workers'])
    guard(config,output.parent,projected_disk=1024**2)
    output.mkdir()
    with phase(req['stage']):
        value=diagnostic(req,arrays,config) if req['stage']=='diagnostic' else formal_cost(req,config)
    # Inputs remain original immutable data; hash after use, before final receipt.
    with phase('post_read_verify'):
        manifest=read(absolute(req['root'])/'inputs.json')
        for record in manifest['files']:
            if digest(absolute(req['root'])/record['path'])!=record['sha256']:raise ValueError('original input changed during work')
        if digest(absolute(req['root'])/'inputs.json')!=binding['inputs_manifest_sha256']:raise ValueError('original input manifest changed during work')
        if req['stage']=='diagnostic' and digest(output/value['query_copy']['file'])!=value['query_copy']['sha256']:raise ValueError('query copy changed during work')
    # Conservative bound: sum parent maximum with one maximum for each worker
    # plus the multiprocessing resource tracker. This is not a sampled peak.
    unit=1 if sys.platform=='darwin' else 1024
    parent=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*unit
    child=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)*unit
    peak=parent+(req['workers']+1)*child
    value.update({'schema':'p-query-sprint-result-v1','engineering':False,'synthetic_input':True,'science_data_read':False,
                  'stage':req['stage'],'run_id':req['run_id'],'workers':req['workers'],'input_binding':binding,
                  'runner_source':runner,'request_sha256':digest(request_path),'resources':{**declaration,
                  'parent_peak_rss_bytes':parent,'max_child_peak_rss_bytes':child,
                  'process_tree_peak_rss_upper_bound_bytes':peak,'peak_scope':'parent maximum + (workers+tracker) times maximum child RSS'},
                  'measurement_producer':None if 'previous_failure_request'in req else proof,
           'previous_failure_producer':proof if 'previous_failure_request'in req else None,'argv':invocation(request_path)})
    with phase('receipt'):
        with (output/'result.json').open('x') as stream:json.dump(value,stream,sort_keys=True)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('dry').add_argument('--request',type=Path)
    verify=sub.add_parser('verify');verify.add_argument('--request',type=Path,required=True)
    for command in ('prepare','submit','run'):
        item=sub.add_parser(command);item.add_argument('--request',type=Path,required=True)
        if command=='run':item.add_argument('--execute-on-new-instance',action='store_true',required=True)
        else:item.add_argument('--ops-code',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='dry':value={'status':'dry','stages':['diagnostic','cost-parity'],'query_prefixes':[32,64,256],
                                  'workers':[1,2,4,8],'original_cost_untouched':True,'server_executed':False}
    elif args.command=='verify':
        path=absolute(args.request);req=load_request(path);payload=read(absolute(req['output'])/'result.json')
        value={'status':'verified','producer':accepted(path,payload),'result_sha256':digest(absolute(req['output'])/'result.json')}
    else:value=execute(args) if args.command=='run' else job(args)
    if args.command=='dry' and args.request is not None:
        req,runner,binding,_,declaration,proof=plan(absolute(args.request))
        value.update({'request_sha256':digest(args.request),'runner_source':runner,'input_binding':binding,'resources':declaration})
    print(json.dumps(value,sort_keys=True),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
