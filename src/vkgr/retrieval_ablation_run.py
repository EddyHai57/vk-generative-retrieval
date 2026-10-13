"""P4E independent consumer with strict JSON and external input pins."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .data_audit import file_sha256
ROOT=Path(__file__).resolve().parents[2]
CORE_SHA='bc70aae38aa53af8ff70c6e159f0a523cbf362d72b20cbc04039daf44657b6de'
PREPARED_SHA='7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2'
ABLATIONS=('uniform_negatives','user_balance','watch_history')

def strict_json(text):
    def pairs(entries):
        result={}
        for key,value in entries:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def bad(value):raise ValueError('nonfinite JSON')
    value=json.loads(text,object_pairs_hook=pairs,parse_constant=bad)
    def finite(v):
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite JSON number')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
    finite(value);return value

def read_pinned(path,digest):
    path=Path(path)
    if type(digest) is not str or len(digest)!=64 or path.is_symlink() or file_sha256(path)!=digest:raise ValueError('external hash mismatch')
    return strict_json(path.read_text())

def write_json(path,value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    with Path(path).open('xb') as stream:stream.write(payload)
    return file_sha256(path)

def default_config():
    return {'schema_version':1,'task_id':'P4E','architecture':'id_mean','learning_rate':.001,'temperature':.15,'batch_size':512,'max_epochs':30,'patience':3,'precision':'fp32','weight_decay':.0001,'gradient_clip':1.,'extra_negatives':128,'seeds':[42,2024,2025],'ablations':list(ABLATIONS),'test_gate':'closed_hash_only'}

def validate_config(value):
    if value!=default_config() or set(value)!=set(default_config()) or any(type(value[k]) is not int for k in ('schema_version','batch_size','max_epochs','patience','extra_negatives')) or any(type(value[k]) not in (int,float) for k in ('learning_rate','temperature','weight_decay','gradient_clip')) or any(type(v) is not int for v in value['seeds']):raise ValueError('unregistered ablation config')
    return value

def source_identity():
    import platform,subprocess
    from importlib.metadata import version
    paths=[ROOT/name for name in ('.python-version','pyproject.toml','uv.lock')]+sorted((ROOT/'src/vkgr').glob('*.py'))
    ledger={p.relative_to(ROOT).as_posix():file_sha256(p) for p in paths}
    source=hashlib.sha256(''.join(f'{n}\0{h}\n' for n,h in ledger.items()).encode()).hexdigest()
    try:
        commit=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        dirty=bool(subprocess.run(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],capture_output=True,text=True,check=True).stdout.strip())
    except subprocess.CalledProcessError:commit=dirty=None
    return {'source_sha':source,'files':ledger,'git_commit':commit,'git_dirty':dirty,'python':platform.python_version(),'dependencies':{n:version(n) for n in ('numpy','torch','pyarrow','PyYAML')},'lock_sha':file_sha256(ROOT/'uv.lock')}

def build_plan(config,inputs):
    validate_config(config)
    if set(inputs) not in ({'prepared_sha','core_sha','test_gate'},{'prepared_sha','core_sha','test_gate','runtime'}) or {k:v for k,v in inputs.items() if k!='runtime'}!={'prepared_sha':PREPARED_SHA,'core_sha':CORE_SHA,'test_gate':'closed_hash_only'}:raise ValueError('input/core binding mismatch')
    if 'runtime' in inputs:validate_runtime_plan(inputs['runtime'])
    slots=[{'slot':j+1,'ablation':a,'seed':s} for j,(a,s) in enumerate((a,s) for a in ABLATIONS for s in (42,2024,2025))]
    return {'schema_version':1,'task_id':'P4E','kind':'plan','config':config,'inputs':inputs,'slots':slots,'consumer':source_identity()}

def verify_plan(path,digest,core_sha):
    value=read_pinned(path,digest)
    if set(value)!={'schema_version','task_id','kind','config','inputs','slots','consumer'} or value['schema_version']!=1 or value['task_id']!='P4E' or value['kind']!='plan' or core_sha!=CORE_SHA:raise ValueError('plan identity mismatch')
    if value!=build_plan(value['config'],value['inputs']):raise ValueError('plan source/config/slot drift')
    return value


def state_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b


def synthetic_data():
    import torch
    catalog=list(range(160));idmap={i:i+1 for i in catalog}
    rows=[{'example_id':j,'user_id':j%4,'target':j+1,'target_position':[0,j+60],'history':[0],'history_positions':[[0,0]],'past_likes':[0]} for j in range(16)]
    queries=[{'user_id':j,'target':j+1,'target_position':[25,j],'history':[0],'history_positions':[[0,0]],'oov':False} for j in range(4)]
    return {'rows':rows,'queries':queries,'catalog':catalog,'idmap':idmap,'content':torch.zeros(161,64),'support':set(range(17)),'supported_indices':list(range(1,18)),'synthetic':True,'watch_rows':[{**r,'history':[0,20,20],'history_positions':[[0,0],[0,1],[0,2]]} for r in rows],'watch_queries':[{**q,'history':[0,20,20],'history_positions':[[0,0],[0,1],[0,2]]} for q in queries]}


def data_binding(data,ablation,seed,batch_size,device='cpu'):
    def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return {'task':'P4E','ablation':ablation,'seed':seed,'batch_size':batch_size,'core_sha':CORE_SHA,'prepared_sha':PREPARED_SHA,'source':source_identity(),'execution':{'device':device,'precision':'fp32','tf32':False,'deterministic':True},'rows_sha':digest(data['rows']),'queries_sha':digest(data['queries']),'catalog_sha':digest(data['catalog']),'watch_rows_sha':digest(data['watch_rows']) if ablation=='watch_history' else None,'watch_queries_sha':digest(data['watch_queries']) if ablation=='watch_history' else None,'test_gate':'closed_hash_only','synthetic':bool(data.get('synthetic',False)),'support_sha':digest(sorted(data['support'])),'content_sha':hashlib.sha256(data['content'].cpu().numpy().tobytes()).hexdigest()}


def execution_context(device):
    import os,torch
    if device not in ('cpu','cuda') or device=='cuda' and (not torch.cuda.is_available() or os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8')):raise ValueError('bound execution device unavailable or nondeterministic')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def _cursor(data,ablation,seed):
    from .retrieval_ablations import UserPrefixCursor
    from .retrieval_training import EpochCursor
    return UserPrefixCursor([r['user_id'] for r in data['rows']],seed,len(data['rows'])) if ablation=='user_balance' else EpochCursor(len(data['rows']),seed)


def _rng_state():
    import random,numpy as np,torch
    r=np.random.get_state()
    return {'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python_rng':random.getstate(),'numpy_rng':[r[0],r[1].tolist(),int(r[2]),int(r[3]),float(r[4])]}


def _restore_rng(state):
    import random,numpy as np,torch
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);r=state['numpy_rng'];np.random.set_state((r[0],np.array(r[1],dtype=np.uint32),r[2],r[3],r[4]))
    if state['cuda_rng']:
        if not torch.cuda.is_available():raise ValueError('CUDA checkpoint on CPU')
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def _stats_value(stats):
    return {'drawn':stats.drawn,'skipped':stats.skipped,'epochs':[[e,*stats.epochs[e]] for e in sorted(stats.epochs)]}


def run_session(output,data,ablation,seed,batch_size=512,until_step=None,resume=None,device='cpu',check=None,plan_binding=None):
    import random,time,torch
    from .training import seed_all
    from .retrieval_models import TwoTower
    from .retrieval_training import SamplerStats
    from .retrieval_ablations import ablation_train_step
    from .retrieval_experiment import ValidationSelection,export_vectors,evaluate_vectors
    if ablation not in ABLATIONS or seed not in (42,2024,2025) or type(batch_size) is not int or batch_size<1 or (until_step is not None and (type(until_step) is not int or until_step<1)):
        raise ValueError('invalid run config')
    if not data.get('synthetic') and (batch_size!=512 or len(data['rows'])!=86785 or device!='cuda' or plan_binding is None):raise ValueError('unapproved real runtime')
    execution_context(device)
    output=Path(output);binding=data_binding(data,ablation,seed,batch_size,device)
    binding['plan']=plan_binding
    seed_all(seed);model=TwoTower('id_mean',data['content'],data['supported_indices']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    cursor=_cursor(data,ablation,seed);stats=SamplerStats();selection=ValidationSelection();negative_rng=random.Random(f'p4e-negative-v1:{seed}')
    step=0;records=[];evaluations=[];checkpoints=[];best=None;training_seconds=0.;evaluation_seconds=0.
    if resume:
        prior=read_pinned(resume[0],resume[1])
        if Path(resume[0]).parent.resolve()!=output.resolve() or prior['binding']!=binding or prior['status']=='complete':raise ValueError('resume binding/completion mismatch')
        state=torch.load(output/prior['last']['path'],map_location='cpu',weights_only=True)
        if file_sha256(output/prior['last']['path'])!=prior['last']['sha256'] or state['binding']!=binding or state['step']!=prior['steps'] or state['records']!=prior['records']:raise ValueError('checkpoint resume mismatch')
        cursor.load_state_dict(state['cursor']);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);negative_rng.setstate(state['negative_rng']);_restore_rng(state)
        stats.drawn=state['stats']['drawn'];stats.skipped=state['stats']['skipped'];stats.epochs={e:[d,s] for e,d,s in state['stats']['epochs']};selection.__dict__.update(state['selection']);selection.best_metric=tuple(selection.best_metric) if selection.best_metric is not None else None
        step=state['step'];records=prior['records'];evaluations=prior['evaluations'];checkpoints=prior['epoch_checkpoints'];best=prior['best'];training_seconds=prior['training_seconds'];evaluation_seconds=prior['evaluation_seconds']
        if until_step is not None and until_step<=step:raise ValueError('exhausted segment')
    else:output.mkdir(parents=True,exist_ok=False)
    rows=data['watch_rows'] if ablation=='watch_history' else data['rows'];queries=data['watch_queries'] if ablation=='watch_history' else data['queries']
    clock=time.monotonic();segment_eval=0.
    while not selection.stopped:
        if check:check()
        indices=cursor.next(batch_size)
        record=ablation_train_step(model,optimizer,[rows[i] for i in indices],data['idmap'],stats,cursor.epoch,.15,ablation,negative_rng)
        step+=1;records.append({**record,'step':step,'epoch':cursor.epoch+1,'example_indices':indices})
        improved=False
        if cursor.offset==cursor.size:
            start=time.monotonic();users,items=export_vectors(model,data['catalog'],queries,data['idmap'],batch_size,check)
            result=evaluate_vectors(users,items,data['catalog'],queries,data['support'],check=check)
            primary=result['metrics']['all'];improved=selection.observe(cursor.epoch+1,primary['recall@50'],primary['ndcg@50'])
            evaluations.append({'epoch':cursor.epoch+1,'result':{k:v for k,v in result.items() if k!='exact_seconds'}});segment_eval+=time.monotonic()-start
        cutoff=until_step is not None and step>=until_step
        if cursor.offset==cursor.size or cutoff or step%50==0:
            if check:check()
            path=output/f'checkpoint-step-{step}.pt'
            selected={**selection.__dict__};selected['best_metric']=list(selected['best_metric']) if selected['best_metric'] is not None else None
            state={'schema_version':1,'task_id':'P4E','binding':binding,'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'cursor':cursor.state_dict(),'negative_rng':negative_rng.getstate(),'stats':_stats_value(stats),'selection':selected,'records':records,**_rng_state()}
            with path.open('xb') as stream:torch.save(state,stream)
            last={'path':path.name,'sha256':file_sha256(path),'step':step,'epoch':selection.epoch}
            if cursor.offset==cursor.size:checkpoints.append(last)
            if improved:best={**last,'metric':list(selection.best_metric)}
            value={'schema_version':1,'task_id':'P4E','kind':'ablation_train','status':'complete' if selection.stopped else 'segment_complete' if cutoff else 'running','binding':binding,'max_epochs':30,'patience':3,'steps':step,'records':records,'evaluations':evaluations,'epoch_checkpoints':checkpoints,'selection':selected,'stats':_stats_value(stats),'last':last,'best':best,'training_seconds':training_seconds+time.monotonic()-clock-segment_eval,'evaluation_seconds':evaluation_seconds+segment_eval,'manifest_name':f'manifest-step-{step}.json'}
            digest=write_json(output/value['manifest_name'],value)
            if cutoff or selection.stopped:return {**value,'manifest_sha':digest}
    raise ValueError('unexpected exhausted session')


def verify_run(path,digest,data,core_sha,plan_sha=None,watch_sha=None,check=None):
    if check:check()
    import random,torch
    from .retrieval_ablations import uniform_columns
    from .retrieval_training import candidate_batch,SamplerStats
    from .retrieval_experiment import ValidationSelection,export_vectors,evaluate_vectors
    from .retrieval_models import TwoTower
    path=Path(path);value=read_pinned(path,digest)
    expected_fields={'schema_version','task_id','kind','status','binding','max_epochs','patience','steps','records','evaluations','epoch_checkpoints','selection','stats','last','best','training_seconds','evaluation_seconds','manifest_name'}
    if set(value)!=expected_fields or value['task_id']!='P4E' or value['kind']!='ablation_train' or value['schema_version']!=1 or value['max_epochs']!=30 or value['patience']!=3 or core_sha!=CORE_SHA:raise ValueError('run external identity mismatch')
    binding=value['binding'];ablation=binding['ablation'];device=binding['execution']['device']
    execution_context(device)
    if not data.get('synthetic') and device!='cuda':raise ValueError('real verifier requires producer CUDA')
    expected=data_binding(data,ablation,binding['seed'],binding['batch_size'],device);expected['plan']=binding['plan']
    if binding!=expected:raise ValueError('run input/source mismatch')
    if not data.get('synthetic') and (binding['plan'] is None or plan_sha!=binding['plan']['plan_sha'] or ablation=='watch_history' and watch_sha!=binding['plan']['watch_sha']):raise ValueError('external plan/watch pins missing')
    cursor=_cursor(data,ablation,binding['seed']);rng=random.Random(f"p4e-negative-v1:{binding['seed']}");stats=SamplerStats();selection=ValidationSelection()
    rows=data['watch_rows'] if ablation=='watch_history' else data['rows'];queries=data['watch_queries'] if ablation=='watch_history' else data['queries']
    for j,record in enumerate(value['records'],1):
        if check and j%50==1:check()
        indices=cursor.next(binding['batch_size']);batch=[rows[i] for i in indices]
        if ablation=='uniform_negatives':_,_,mask,negatives=uniform_columns(batch,data['catalog'],data['idmap'],rng)
        else:_,_,mask=candidate_batch(batch,data['idmap']);negatives=[]
        skips=int((mask.sum(1)<=1).sum())
        if record['step']!=j or record['epoch']!=cursor.epoch+1 or record['example_indices']!=indices or record['negative_ids']!=negatives or record['drawn']!=len(indices) or record['skipped']!=skips or record['loss'] is not None and not math.isfinite(record['loss']) or record['gradient_norm'] is not None and not math.isfinite(record['gradient_norm']):raise ValueError('independent draw/loss replay mismatch')
        stats.record(cursor.epoch,len(indices),skips);stats.check(cursor.epoch)
    if len(value['records'])!=value['steps'] or _stats_value(stats)!=value['stats'] or len(value['epoch_checkpoints'])!=len(value['evaluations']):raise ValueError('draw/checkpoint budget mismatch')
    model=TwoTower('id_mean',data['content'],data['supported_indices']).to(device)
    for checkpoint,evaluation in zip(value['epoch_checkpoints'],value['evaluations']):
        if check:check()
        if Path(checkpoint['path']).name!=checkpoint['path'] or file_sha256(path.parent/checkpoint['path'])!=checkpoint['sha256']:raise ValueError('checkpoint hash mismatch')
        state=torch.load(path.parent/checkpoint['path'],map_location='cpu',weights_only=True)
        if state['binding']!=binding or state['selection']['epoch']!=evaluation['epoch']:raise ValueError('epoch binding mismatch')
        model.load_state_dict(state['model']);users,items=export_vectors(model,data['catalog'],queries,data['idmap'],binding['batch_size'],check)
        result=evaluate_vectors(users,items,data['catalog'],queries,data['support'],check=check);result.pop('exact_seconds')
        if result!=evaluation['result']:raise ValueError('checkpoint exact reexport mismatch')
        primary=result['metrics']['all'];selection.observe(evaluation['epoch'],primary['recall@50'],primary['ndcg@50'])
    expected_best=None
    if selection.best_epoch is not None:
        checkpoint=value['epoch_checkpoints'][selection.best_epoch-1]
        expected_best={**checkpoint,'metric':list(selection.best_metric)}
    if value['best']!=expected_best:raise ValueError('best checkpoint selection mismatch')
    if (value['status']=='complete')!=selection.stopped:raise ValueError('terminal selection mismatch')
    if Path(value['last']['path']).name!=value['last']['path']:raise ValueError('last checkpoint path escape')
    state=torch.load(path.parent/value['last']['path'],map_location='cpu',weights_only=True)
    if file_sha256(path.parent/value['last']['path'])!=value['last']['sha256'] or state['binding']!=binding or state['step']!=value['steps'] or state['records']!=value['records'] or not state_equal(state['cursor'],cursor.state_dict()) or not state_equal(state['negative_rng'],rng.getstate()) or state['stats']!=value['stats'] or not state_equal(state['selection'],selection.__dict__) or not state_equal(value['selection'],selection.__dict__):raise ValueError('last state/selection mismatch')
    return value


def verify_completion(path,digest,data,plan_sha,watch_sha,check=None):
    """Require all fixed slots, pinned stage exits, and independently verified terminals."""
    path=Path(path);value=read_pinned(path,digest)
    fields={'schema_version','task_id','test_gate','plan_sha','runs'}
    if set(value)!=fields or type(value['schema_version']) is not int or value['schema_version']!=1 or value['task_id']!='P4E' or value['test_gate']!='closed_hash_only' or value['plan_sha']!=plan_sha:raise ValueError('completion identity mismatch')
    records=value['runs']
    if not isinstance(records,list) or len(records)!=9:raise ValueError('nine actual slots required')
    expected=[(a,s) for a in ABLATIONS for s in (42,2024,2025)]
    terminals=[]
    for slot,(record,(ablation,seed)) in enumerate(zip(records,expected),1):
        if not isinstance(record,dict) or set(record)!={'slot','manifest_ref','manifest_sha','train_exit_sha','verify_exit_sha'} or type(record['slot']) is not int or record['slot']!=slot:raise ValueError('fixed completion slot mismatch')
        ref=Path(record['manifest_ref']);directory=path.parent/f'slot-{slot:02d}'
        if ref.is_absolute() or len(ref.parts)!=2 or ref.parts[0]!=directory.name or directory.is_symlink() or (path.parent/ref).resolve().parent!=directory.resolve():raise ValueError('completion path escape')
        for stage in ('train','verify'):
            exit_path=directory/f'{stage}.exit'
            if exit_path.is_symlink() or file_sha256(exit_path)!=record[f'{stage}_exit_sha'] or exit_path.read_text().strip()!='0':raise ValueError('actual zero stage exit required')
        terminal=verify_run(path.parent/ref,record['manifest_sha'],data,CORE_SHA,plan_sha,watch_sha,check)
        if terminal['status']!='complete' or terminal['binding']['ablation']!=ablation or terminal['binding']['seed']!=seed:raise ValueError('incomplete or substituted terminal')
        terminals.append(terminal)
    return terminals


def main(argv=None):
    import sys
    parser=argparse.ArgumentParser(description='P4E independent ablations; real server execution requires coordinator gates')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('synthetic-train');p.add_argument('--ablation',choices=ABLATIONS,required=True);p.add_argument('--seed',type=int,choices=(42,2024,2025),required=True);p.add_argument('--until-step',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--resume-sha')
    p=sub.add_parser('verify');p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--core-sha',required=True);p.add_argument('--synthetic',action='store_true');p.add_argument('--plan');p.add_argument('--plan-sha');p.add_argument('--watch-sha')
    p=sub.add_parser('plan');p.add_argument('--config',required=True);p.add_argument('--input-plan',required=True);p.add_argument('--input-plan-sha',required=True);p.add_argument('--output',required=True)
    for name in ('prepare-watch','verify-watch','pilot','train','summarize'):
        p=sub.add_parser(name);p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--core-sha',required=True);p.add_argument('--output');p.add_argument('--slot',type=int);p.add_argument('--watch-sha');p.add_argument('--manifest');p.add_argument('--manifest-sha');p.add_argument('--until-step',type=int);p.add_argument('--resume');p.add_argument('--resume-sha');p.add_argument('--pilot-plan');p.add_argument('--pilot-plan-sha');p.add_argument('--core-runs')
    args=parser.parse_args(argv)
    try:
        if args.command=='synthetic-train':
            if not 1<=args.until_step<=20 or bool(args.resume)!=bool(args.resume_sha):raise ValueError('invalid synthetic segment')
            value=run_session(args.output,synthetic_data(),args.ablation,args.seed,8,args.until_step,(args.resume,args.resume_sha) if args.resume else None)
            print(json.dumps({'status':value['status'],'manifest':value['manifest_name'],'manifest_sha':value['manifest_sha']}));return 0
        if args.command=='verify' and args.synthetic:
            value=verify_run(args.manifest,args.manifest_sha,synthetic_data(),args.core_sha)
            print(json.dumps({'status':'verified','steps':value['steps'],'scope':'synthetic_only'}));return 0
        if args.command=='plan':
            config=strict_json(Path(args.config).read_text());inputs=read_pinned(args.input_plan,args.input_plan_sha)
            value=build_plan(config,inputs);print(json.dumps({'plan_sha':write_json(args.output,value),'slots':9}));return 0
        plan=verify_plan(args.plan,args.plan_sha,args.core_sha)
        runtime=validate_runtime_plan(plan['inputs']['runtime'])
        if args.command not in runtime['allowed_operations']:raise ValueError('operation outside frozen runtime scope')
        if args.command=='prepare-watch':
            if args.output is None:raise ValueError('exclusive output required')
            check=operation_resource_guard(args.output,runtime,args.plan_sha);data,partitions,binding=real_data(plan)
            value=write_watch(args.output,checked_partitions(partitions(),check),data['rows'],data['queries'],data['catalog'],binding);check()
            print(json.dumps({'status':'prepared','manifest_sha':value['manifest_sha']}));return 0
        if args.command=='verify-watch':
            if args.manifest is None or args.manifest_sha is None:raise ValueError('external watch manifest pin required')
            check=operation_resource_guard(Path(args.manifest).parent,runtime,args.plan_sha);data,partitions,binding=real_data(plan)
            verify_watch(args.manifest,args.manifest_sha,checked_partitions(partitions(),check),data['rows'],data['queries'],data['catalog'],binding);check()
            print(json.dumps({'status':'watch_verified'}));return 0
        return dispatch_real_run(args,plan)
    except (ValueError,KeyError,TypeError,OSError,RuntimeError):
        print('ERROR P4E input/state/budget gate failed; preserve evidence',file=sys.stderr);return 2




def write_watch(output,partitions,examples,queries,catalog,binding):
    from .retrieval_watch import build_watch
    if binding.get('core_sha')!=CORE_SHA or binding.get('prepared_sha')!=PREPARED_SHA or binding.get('test_gate')!='closed_hash_only' or binding.get('consumer')!=source_identity():raise ValueError('watch producer binding mismatch')
    train,validation,slices=build_watch(partitions,examples,queries,catalog)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    artifacts={name:write_json(output/name,value) for name,value in [('watch-train.json',train),('watch-validation.json',validation),('train-slices.json',slices)]}
    slice_manifest={'schema_version':1,'task_id':'P4E','kind':'slice_statistics','binding':{k:binding.get(k) for k in ('raw_records','raw_producer','prepared_sha','catalog_sha','consumer')},'statistics_ref':'train-slices.json','statistics_sha':artifacts['train-slices.json'],'threshold_definition':'nearest_rank_positive_item_event_counts_median_p90','thresholds':slices['popularity_thresholds'],'item_groups':['unsupported0','positive_le_median','positive_gt_median_le_p90','positive_gt_p90'],'user_groups':['1','2-5','6-20','gt20'],'holdout_used':False}
    artifacts['slice-statistics-manifest.json']=write_json(output/'slice-statistics-manifest.json',slice_manifest)
    manifest={'schema_version':1,'task_id':'P4E','kind':'watch_prepare','binding':binding,'artifacts':artifacts,'training_examples':len(train),'validation_queries':len(validation),'test_gate':'closed_hash_only'}
    return {**manifest,'manifest_sha':write_json(output/'watch-manifest.json',manifest)}


def verify_watch(path,digest,partitions,examples,queries,catalog,binding):
    from .retrieval_watch import build_watch
    path=Path(path);value=read_pinned(path,digest)
    if set(value)!={'schema_version','task_id','kind','binding','artifacts','training_examples','validation_queries','test_gate'} or value['schema_version']!=1 or value['task_id']!='P4E' or value['kind']!='watch_prepare' or value['binding']!=binding or value['test_gate']!='closed_hash_only' or set(value['artifacts'])!={'watch-train.json','watch-validation.json','train-slices.json','slice-statistics-manifest.json'}:raise ValueError('watch external binding mismatch')
    train,validation,slices=build_watch(partitions,examples,queries,catalog)
    for name,expected in [('watch-train.json',train),('watch-validation.json',validation),('train-slices.json',slices)]:
        if read_pinned(path.parent/name,value['artifacts'][name])!=expected:raise ValueError('raw watch replay mismatch')
    slice_manifest=read_pinned(path.parent/'slice-statistics-manifest.json',value['artifacts']['slice-statistics-manifest.json'])
    if slice_manifest['statistics_sha']!=value['artifacts']['train-slices.json'] or slice_manifest['thresholds']!=slices['popularity_thresholds'] or slice_manifest['holdout_used'] is not False or slice_manifest['binding']!={k:binding.get(k) for k in ('raw_records','raw_producer','prepared_sha','catalog_sha','consumer')}:raise ValueError('shared slice lineage mismatch')
    if value['training_examples']!=len(train) or value['validation_queries']!=len(validation):raise ValueError('watch denominator mismatch')
    return value


def slice_metrics(predictions,queries,catalog,slices,ks=(10,20,50)):
    from .baselines import metrics
    groups={'all':list(range(len(queries))),'in_catalog':[],'oov':[],'supported':[],'unsupported':[],'pop_low':[],'pop_mid':[],'pop_high':[],'history_1':[],'history_2_5':[],'history_6_20':[],'history_gt20':[]}
    median=slices['popularity_thresholds']['median'];p90=slices['popularity_thresholds']['p90']
    for j,q in enumerate(queries):
        count=slices['item_like_counts'].get(str(q['target']),0);history=slices['user_like_counts'].get(str(q['user_id']),0)
        groups['oov' if q['oov'] else 'in_catalog'].append(j)
        if not q['oov']:
            groups['supported' if count else 'unsupported'].append(j)
            if count:groups['pop_low' if count<=median else 'pop_mid' if count<=p90 else 'pop_high'].append(j)
        if history<1:raise ValueError('eligible query lacks train like count')
        groups['history_1' if history==1 else 'history_2_5' if history<=5 else 'history_6_20' if history<=20 else 'history_gt20'].append(j)
    return {name:metrics([predictions[j] for j in indices],[queries[j] for j in indices],catalog,ks)['all'] for name,indices in groups.items()}


def summarize_metrics(records,cores):
    import statistics
    expected={(a,s) for a in ABLATIONS for s in (42,2024,2025)}
    identities=[(r['ablation'],r['seed']) for r in records]
    if len(records)!=9 or set(identities)!=expected or len(set(identities))!=9 or set(cores)!={'42','2024','2025'}:raise ValueError('nine unique seed/core bindings required')
    result={}
    for ablation in ABLATIONS:
        rows=sorted([r for r in records if r['ablation']==ablation],key=lambda r:(42,2024,2025).index(r['seed']));groups=rows[0]['metrics']
        result[ablation]={}
        for group in groups:
            if any(set(r['metrics'])!=set(groups) or set(r['metrics'][group])!=set(groups[group]) for r in rows):raise ValueError('metric group schema mismatch')
            for row in rows:
                baseline=cores[str(row['seed'])]
                if set(baseline)!=set(groups) or set(baseline[group])!=set(row['metrics'][group]) or baseline[group].get('count')!=row['metrics'][group].get('count'):raise ValueError('paired query denominator/schema mismatch')
            result[ablation][group]={}
            for key in groups[group]:
                if key not in ('recall@10','recall@20','recall@50','ndcg@10','ndcg@20','ndcg@50'):continue
                values=[r['metrics'][group][key] for r in rows];baseline=[cores[str(r['seed'])][group][key] for r in rows]
                if any(v is None for v in values+baseline):
                    if not all(v is None for v in values+baseline):raise ValueError('paired empty cohort mismatch')
                    result[ablation][group][key]={'values':values,'mean':None,'sample_std':None,'paired_delta':None,'paired_delta_mean':None,'paired_delta_sample_std':None};continue
                if any(not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in values+baseline):raise ValueError('invalid metric')
                deltas=[v-b for v,b in zip(values,baseline)]
                result[ablation][group][key]={'values':values,'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'paired_delta':deltas,'paired_delta_mean':statistics.mean(deltas),'paired_delta_sample_std':statistics.stdev(deltas)}
    return result




def summary_session(path,digest,data,slices,cores,plan_sha,watch_sha,output,check=None):
    if check:check()
    terminals=verify_completion(path,digest,data,plan_sha,watch_sha,check)
    records=[]
    for terminal in terminals:
        epoch=terminal['selection']['best_epoch']
        predictions=terminal['evaluations'][epoch-1]['result']['predictions']
        records.append({'ablation':terminal['binding']['ablation'],'seed':terminal['binding']['seed'],'metrics':slice_metrics(predictions,data['queries'],data['catalog'],slices)})
        if check:check()
    value={'schema_version':1,'task_id':'P4E','kind':'verified_ablation_summary','status':'awaiting_coordinator_acceptance','consumer':source_identity(),'completion_sha':digest,'plan_sha':plan_sha,'watch_sha':watch_sha,'core_sha':CORE_SHA,'test_gate':'closed_hash_only','synthetic':bool(data.get('synthetic')),'runs':records,'cores':cores,'aggregate':summarize_metrics(records,cores)}
    if check:check()
    return {**value,'manifest_sha':write_json(output,value)}


def load_core_metrics(root,data,slices):
    """Read accepted core predictions under their immutable public artifact pins."""
    root=Path(root).resolve()
    public=read_pinned(ROOT/'artifacts/p4d-core-manifest.json','6ddccee458c3c66bb185d51527c0dc32981ae83949efe420eef97ed45881ac6e')
    records=[r for r in public['runs'] if r['config']['architecture']=='id_mean' and r['config']['learning_rate']==.001 and r['config']['temperature']==.15 and r['config']['batch_size']==512]
    if len(records)!=3 or {r['config']['seed'] for r in records}!={42,2024,2025}:raise ValueError('three fixed accepted cores required')
    result={}
    for record in records:
        ref=Path(record['manifest_logical_ref']);path=root/ref
        if ref.is_absolute() or len(ref.parts)!=2 or path.parent.is_symlink() or path.resolve().parent.parent!=root:raise ValueError('core path escape')
        terminal=read_pinned(path,record['manifest_sha256']);binding=terminal['binding']
        if terminal['status']!='complete' or binding['config']!=record['config'] or binding['validation_query_sha256']!=data['query_sha256'] or binding['catalog_sha256']!=data['prepared']['artifacts']['catalog.json'] or binding['test_gate']!='closed_hash_only':raise ValueError('core query/catalog/config drift')
        epoch=terminal['selection']['best_epoch']
        if type(epoch) is not int or not 1<=epoch<=30:raise ValueError('core best epoch invalid')
        evaluations=[e for e in terminal['evaluations'] if e['epoch']==epoch]
        pred_sha=record['best_artifacts']['predictions.json']
        if len(evaluations)!=1 or evaluations[0]['result']['artifacts']['predictions.json']!=pred_sha:raise ValueError('core prediction provenance drift')
        pred_path=path.parent/f'epoch-{epoch}'/'predictions.json'
        if pred_path.parent.is_symlink():raise ValueError('core epoch symlink')
        predictions=read_pinned(pred_path,pred_sha)
        result[str(record['config']['seed'])]=slice_metrics(predictions,data['queries'],data['catalog'],slices)
    return result


def validate_runtime_plan(value):
    fields={'prepared','holdout','raw_root','core_selection','watch_manifest','watch_sha','wall_seconds','disk_bytes','allowed_operations','pilot_steps','resume_steps'}
    if set(value) not in (fields,fields|{'operation_root'}) or any(type(value[k]) is not int or value[k]<=0 for k in ('wall_seconds','disk_bytes','pilot_steps')) or value['pilot_steps']!=30 or value['resume_steps']!=[7,20,30]:raise ValueError('finite runtime budget required')
    if 'operation_root' in value and (type(value['operation_root']) is not str or not Path(value['operation_root']).is_absolute()):raise ValueError('absolute operation root required')
    if not value['allowed_operations'] or len(set(value['allowed_operations']))!=len(value['allowed_operations']) or any(x not in ('prepare-watch','verify-watch','pilot','train','verify','summarize') for x in value['allowed_operations']):raise ValueError('unknown runtime scope')
    for name in ('prepared','holdout','raw_root','core_selection'):
        if type(value[name]) is not str or not Path(value[name]).is_absolute():raise ValueError('absolute private input path required')
    if (value['watch_manifest'] is None)!=(value['watch_sha'] is None):raise ValueError('partial watch pin')
    if value['watch_manifest'] is not None and (not Path(value['watch_manifest']).is_absolute() or type(value['watch_sha']) is not str or len(value['watch_sha'])!=64):raise ValueError('invalid watch external pin')
    return value


def visible_memory_fraction(current,root,host_fraction):
    current,root=Path(current).resolve(),Path(root).resolve()
    if current!=root and root not in current.parents:raise ValueError('cgroup mapping outside mount')
    fractions=[host_fraction];finite_limit=False
    while True:
        limit=current/'memory.max';usage=current/'memory.current'
        if not limit.is_file() or not usage.is_file():raise ValueError('missing visible cgroup files')
        text=limit.read_text().strip();used=int(usage.read_text().strip())
        if used<0:raise ValueError('invalid cgroup usage')
        if text!='max':
            total=int(text)
            if total<=0:raise ValueError('invalid cgroup limit')
            finite_limit=True;fractions.append(1-used/total)
        if current==root:break
        current=current.parent
    if not finite_limit or not math.isfinite(host_fraction) or not 0<=host_fraction<=1:raise ValueError('no finite visible memory limit')
    return min(fractions)





def real_data(plan):
    from .retrieval_experiment import load_training_inputs
    from .retrieval import PREPARED_SOURCE
    from .retrieval_data import validate_contract,checked
    from .data_audit import parquet_rows
    runtime=validate_runtime_plan(plan['inputs']['runtime']);read_pinned(runtime['core_selection'],CORE_SHA)
    pins=validate_contract(ROOT);query_pin=next(p for p in pins['full_query_pins'] if p['role']=='validation-queries.json')
    data=load_training_inputs(runtime['prepared'],runtime['holdout'],PREPARED_SHA,PREPARED_SOURCE,query_pin['sha256'])
    prepared=data['prepared'];records=prepared['binding']['input_records']
    names=[f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(25)]
    train_records=[r for r in records if r['role'] in names]
    if [r['role'] for r in train_records]!=names or prepared['artifacts']['catalog.json']!=pins['catalog_sha256'] or prepared['binding']['query_pins']!=pins['full_query_pins'] or prepared['binding']['legacy_producer']!=pins['legacy_p3_producer']:raise ValueError('raw/prepared lineage conflict')
    for record in train_records:
        path=checked(Path(runtime['raw_root'])/record['role'],record['sha256'])
        if path.stat().st_size!=record['size_bytes']:raise ValueError('raw input size mismatch')
    def partitions():
        for week,record in enumerate(train_records):yield week,parquet_rows(Path(runtime['raw_root'])/record['role'])
    binding={'raw_records':train_records,'raw_producer':prepared['binding']['legacy_producer'],'prepared_sha':PREPARED_SHA,'query_sha':query_pin['sha256'],'catalog_sha':pins['catalog_sha256'],'core_sha':CORE_SHA,'consumer':source_identity(),'test_gate':'closed_hash_only'}
    data['synthetic']=False
    return data,partitions,binding




def checked_partitions(partitions,check):
    for week,rows in partitions:
        def events(rows=rows):
            for ordinal,row in enumerate(rows):
                if ordinal%65536==0:check()
                yield row
            check()
        yield week,events()


def resource_guard(output,runtime):
    import time,shutil,subprocess,sys
    if sys.platform!='linux':raise ValueError('real resources require Linux')
    runtime=validate_runtime_plan(runtime);started=time.monotonic();output=Path(output);samples=[]
    def check():
        if time.monotonic()-started>runtime['wall_seconds']:raise ValueError('wall budget exhausted')
        memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        membership=[line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
        mounts=[line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup2 ' in line]
        if len(membership)!=1 or len(mounts)!=1:raise ValueError('ambiguous cgroup2 mapping')
        mount=mounts[0];root=Path(mount[4]);member=Path(membership[0]);mountroot=Path(mount[3])
        relative=Path('.') if member==mountroot else member.relative_to(mountroot) if mountroot in member.parents else Path(str(member).lstrip('/'))
        ram=visible_memory_fraction(root/relative,root,memory['MemAvailable']/memory['MemTotal'])
        disk=shutil.disk_usage(output if output.exists() else output.parent)
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        pairs=[tuple(map(float,line.split(','))) for line in gpu.stdout.strip().splitlines()]
        fractions={'ram':ram,'vram':min(f/t for f,t in pairs),'disk':disk.free/disk.total}
        if any(not math.isfinite(f) or f<.2 for f in fractions.values()):raise ValueError('resource headroom below 20%')
        used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0
        if used>runtime['disk_bytes']:raise ValueError('artifact disk budget exceeded')
        samples.append({**fractions,'wall_seconds':time.monotonic()-started,'output_bytes':used});return fractions
    check.samples=samples;check();return check




def pilot_session(output,data,check,device='cuda',plan_binding=None):
    import torch,numpy as np,time
    from .retrieval_models import TwoTower
    from .retrieval_experiment import export_vectors,evaluate_vectors
    output=Path(output);output.mkdir(parents=True,exist_ok=False);runs=[];started=time.monotonic()
    if not data.get('synthetic') and (device!='cuda' or plan_binding is None):raise ValueError('real pilot CUDA/plan required')
    batch=8 if data.get('synthetic') else 512
    for ablation in ABLATIONS:
        check();directory=output/ablation;prior=None;value=None
        worst=worst_case_context(data,ablation,check,device,batch)
        for cutoff in (7,20,30):
            if value is not None and value['status']=='complete':break
            value=run_session(directory,data,ablation,42,batch,cutoff,prior,device,check,plan_binding)
            verify_run(directory/value['manifest_name'],value['manifest_sha'],data,CORE_SHA,plan_binding['plan_sha'] if plan_binding else None,plan_binding['watch_sha'] if plan_binding else None,check)
            prior=(directory/value['manifest_name'],value['manifest_sha'])
        state=torch.load(directory/value['last']['path'],map_location='cpu',weights_only=True)
        model=TwoTower('id_mean',data['content'],data['supported_indices']).to(device);model.load_state_dict(state['model'])
        queries=data['watch_queries'] if ablation=='watch_history' else data['queries']
        users,items=export_vectors(model,data['catalog'],queries,data['idmap'],batch,check)
        result=evaluate_vectors(users,items,data['catalog'],queries,data['support'],check=check);result.pop('exact_seconds')
        users2,items2=export_vectors(model,data['catalog'],queries,data['idmap'],batch,check)
        if not np.array_equal(users,users2) or not np.array_equal(items,items2):raise ValueError('pilot checkpoint export mismatch')
        check();runs.append({'ablation':ablation,'steps':value['steps'],'drawn':value['stats']['drawn'],'manifest_ref':f'{ablation}/{value["manifest_name"]}','manifest_sha':value['manifest_sha'],'export_verified':True,'worst_case':worst,'full_export_shape':[len(users),len(items),64],'full_validation_result':result})
        del model,state,users,items,users2,items2
    manifest={'schema_version':1,'task_id':'P4E','kind':'resource_restore_pilot','consumer':source_identity(),'binding':plan_binding,'synthetic':bool(data.get('synthetic')),'maximum_optimizer_steps':90,'runs':runs,'wall_seconds':time.monotonic()-started,'resources':getattr(check,'samples',[]),'test_gate':'closed_hash_only'}
    digest=write_json(output/'pilot-manifest.json',manifest);return {**manifest,'manifest_sha':digest}

def operation_resource_guard(output,runtime,plan_sha):
    if 'operation_root' not in runtime:raise ValueError('approved persistent operation root required')
    budget=operation_guard(output,runtime,plan_sha)
    budget()
    resource=resource_guard(output,runtime)
    def check():budget();return resource()
    check.samples=getattr(resource,'samples',[])
    check()
    return check


def operation_guard(output,runtime,plan_sha,now=None):
    import time
    now=now or time.time;root=Path(runtime['operation_root']);output=Path(output)
    if not root.is_absolute() or root.is_symlink() or root.resolve() not in output.resolve().parents:
        raise ValueError('output outside approved operation root')
    root.mkdir(parents=True,exist_ok=True);path=root/'operation-budget-ledger.json'
    if not path.exists():
        start=now()
        write_json(path,{'schema_version':1,'plan_sha':plan_sha,'started_utc':start,'deadline_utc':start+runtime['wall_seconds'],'wall_seconds':runtime['wall_seconds'],'disk_bytes':runtime['disk_bytes']})
    def check():
        value=strict_json(path.read_text())
        fields={'schema_version','plan_sha','started_utc','deadline_utc','wall_seconds','disk_bytes'}
        if set(value)!=fields or value['plan_sha']!=plan_sha or value['wall_seconds']!=runtime['wall_seconds'] or value['disk_bytes']!=runtime['disk_bytes'] or value['deadline_utc']!=value['started_utc']+value['wall_seconds'] or not value['started_utc']<=now()<=value['deadline_utc']:
            raise ValueError('persistent operation budget exhausted/mismatched')
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if size>runtime['disk_bytes']:raise ValueError('persistent disk budget exceeded')
        return value
    return check

def worst_case_context(data,ablation,check,device='cuda',batch_size=512):
    import torch,random
    from .retrieval_models import TwoTower
    from .retrieval_ablations import uniform_columns,masked_ce
    from .retrieval_training import candidate_batch
    if ablation not in ABLATIONS or len(data['support'])<batch_size:
        raise ValueError('insufficient worst-case distinct support')
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    model=TwoTower('id_mean',data['content'],data['supported_indices']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    for parameter in model.parameters():
        optimizer.state[parameter]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(parameter),'exp_avg_sq':torch.zeros_like(parameter)}
    targets=sorted(data['support'])[:batch_size]
    past=max((r['past_likes'] for r in data['rows']),key=len)
    rows=[{'target':item,'history':[targets[(j+k+1)%batch_size] for k in range(50)],'past_likes':[i for i in past if i!=targets[(j+1)%batch_size]]} for j,item in enumerate(targets)]
    if ablation=='uniform_negatives':
        columns,labels,mask,_=uniform_columns(rows,data['catalog'],data['idmap'],random.Random(42))
    else:columns,labels,mask=candidate_batch(rows,data['idmap'])
    history=torch.tensor([[data['idmap'][i] for i in r['history']] for r in rows],device=device)
    loss,skips=masked_ce(model.users(history),model.items(columns.to(device)),labels.to(device),mask.to(device),.15)
    if loss is None or skips:raise ValueError('invalid resource-only synthetic rows')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if device=='cuda':torch.cuda.synchronize()
    check()
    return {'history_width':50,'candidate_columns':len(columns),'maximum_past_like_items':len(past),'optimizer_steps':0,'backward_probes':1,'scope':'resource_only_synthetic_no_optimizer_step_no_quality_metrics','peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}

def load_watch_artifacts(data,runtime,binding,external_sha):
    from .retrieval_watch import history_overlay
    if runtime['watch_sha'] is None or external_sha!=runtime['watch_sha']:raise ValueError('external watch pin required')
    path=Path(runtime['watch_manifest']);value=read_pinned(path,external_sha)
    if value['task_id']!='P4E' or value['kind']!='watch_prepare' or value['binding']!=binding or value['test_gate']!='closed_hash_only':raise ValueError('watch lineage drift')
    artifacts=value['artifacts']
    if set(artifacts)!={'watch-train.json','watch-validation.json','train-slices.json','slice-statistics-manifest.json'}:raise ValueError('watch artifact scope mismatch')
    data['watch_rows']=history_overlay(data['rows'],read_pinned(path.parent/'watch-train.json',artifacts['watch-train.json']))
    data['watch_queries']=read_pinned(path.parent/'watch-validation.json',artifacts['watch-validation.json'])
    if len(data['watch_queries'])!=len(data['queries']):raise ValueError('watch query count drift')
    for original,watch in zip(data['queries'],data['watch_queries']):
        if {k:v for k,v in watch.items() if k not in ('history','history_positions')}!={k:v for k,v in original.items() if k not in ('history','history_positions')}:raise ValueError('watch query identity drift')
        if not 1<=len(watch['history'])<=50 or len(watch['history'])!=len(watch['history_positions']) or any(i not in data['idmap'] for i in watch['history']) or any(p[0]>24 for p in watch['history_positions']):raise ValueError('watch evaluation history mismatch')
    data['slices']=read_pinned(path.parent/'train-slices.json',artifacts['train-slices.json'])
    return data


def dispatch_real_run(args,plan):
    import torch
    runtime=validate_runtime_plan(plan['inputs']['runtime'])
    if args.command not in runtime['allowed_operations'] or 'operation_root' not in runtime:raise ValueError('approved finite operation root required')
    pilot=None
    if args.command=='train':
        if not args.pilot_plan or not args.pilot_plan_sha:raise ValueError('external accepted pilot receipt required')
        pilot=read_pinned(args.pilot_plan,args.pilot_plan_sha)
        if pilot['kind']!='resource_restore_pilot' or pilot['synthetic'] or pilot['consumer']!=source_identity() or pilot['test_gate']!='closed_hash_only' or [r['ablation'] for r in pilot['runs']]!=list(ABLATIONS) or any(r['steps']>30 or not r['export_verified'] or r['worst_case']['history_width']!=50 for r in pilot['runs']):raise ValueError('pilot scope/provenance mismatch')
    if not torch.cuda.is_available():raise ValueError('actual CUDA environment required')
    target=Path(args.manifest).parent if args.command=='verify' else Path(args.output)
    budget_check=operation_guard(target,runtime,args.plan_sha)
    resource_check=resource_guard(target,runtime)
    def check():budget_check();return resource_check()
    check.samples=resource_check.samples
    check();data,partitions,binding=real_data(plan)
    data=load_watch_artifacts(data,runtime,binding,args.watch_sha)
    pinned={'plan_sha':args.plan_sha,'watch_sha':args.watch_sha,'pilot_sha':args.pilot_plan_sha if args.command=='train' else None}
    if args.command=='pilot':
        if target!=Path(runtime['operation_root'])/'pilot':raise ValueError('exclusive pilot output path mismatch')
        value=pilot_session(target,data,check,'cuda',pinned);check()
        print(json.dumps({'status':'pilot_complete','manifest_sha':value['manifest_sha']}));return 0
    if args.command=='train':
        if type(args.slot) is not int or not 1<=args.slot<=9 or target!=Path(runtime['operation_root'])/f'slot-{args.slot:02d}':raise ValueError('fixed slot/output mismatch')
        slot=plan['slots'][args.slot-1]
        if bool(args.resume)!=bool(args.resume_sha):raise ValueError('partial resume pin')
        if args.resume:verify_run(args.resume,args.resume_sha,data,CORE_SHA,args.plan_sha,args.watch_sha,check)
        value=run_session(target,data,slot['ablation'],slot['seed'],512,args.until_step,(args.resume,args.resume_sha) if args.resume else None,'cuda',check,pinned)
        check();print(json.dumps({'status':value['status'],'manifest_sha':value['manifest_sha']}));return 0
    if args.command=='verify':
        value=verify_run(args.manifest,args.manifest_sha,data,CORE_SHA,args.plan_sha,args.watch_sha,check)
        check();print(json.dumps({'status':'verified','steps':value['steps']}));return 0
    if args.command=='summarize':
        if not args.core_runs or not args.manifest or not args.manifest_sha or target!=Path(runtime['operation_root'])/'summary-cli.json':raise ValueError('pinned completion/core runs and fixed summary output required')
        cores=load_core_metrics(args.core_runs,data,data['slices']);check()
        value=summary_session(args.manifest,args.manifest_sha,data,data['slices'],cores,args.plan_sha,args.watch_sha,target,check)
        print(json.dumps({'status':value['status'],'manifest_sha':value['manifest_sha']}));return 0
    raise ValueError('unsupported real operation')

if __name__=='__main__':raise SystemExit(main())
