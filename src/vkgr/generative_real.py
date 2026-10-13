"""G3 real-input adapter; historical G1/G2 engines remain immutable."""
import copy,hashlib,math
from pathlib import Path
from . import generative_data as gd,generative_experiment as ge
from . import generative_training as gt,generative_evaluation as ev
from .data_audit import canonical_bytes,file_sha256

G1_PINS={'40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',
         '4bd3d093d4f86f4a033eda293798b807e46dd9d8ea0af68240481972d3e000a7'}
G2_SHA='d12e7d78ef16d147010a625fb6cf1a98730e5ac613d62465a65b5b4c2970c82d'
AUDIT={'aggregate_sha256':'d03978b3e4d5346208c605866640b25d83a36e4164419364dc5bd4c0be817e47',
       'supervision_sha256':'ce39dac731f7ff411a8831bdda255338dc1b1e4de11f476bc33d1173a4097d4c'}
NEW_MEMBERS=('src/vkgr/generative_real.py','src/vkgr/generative_pilot.py',
             'tests/test_generative_real.py','tests/test_generative_pilot.py')
P4E_STATS_SHA='6e96e5bc3d21e22b71fcc64293cc2844be85ca3c100c7d28e910e2360470628d'

def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()

def _source(root,contract):
    files={**contract['consumer']['source_members'],**{p:file_sha256(gd._path(root,p)) for p in (*ge._G2_MEMBERS,*NEW_MEMBERS)}}
    # Independently preserve the accepted G2 source, not merely its manifest pin.
    legacy={p:files[p] for p in (*contract['consumer']['source_members'],*ge._G2_MEMBERS)}
    if digest(legacy)!=G2_SHA:raise ValueError('accepted G2 source changed')
    return {**gd._git(root),**gd._environment(),'source_members':files,'source_sha256':digest(files),'lock_sha256':contract['consumer']['lock_sha256']}

def _real_spec(value):
    keys={'schema_version','task_id','kind','status','contract_path','contract_sha256',
          'prepared_path','prepared_sha256','sid_path','sid_sha256','validation_path','validation_sha256',
          'statistics_path','statistics_sha256','statistics_binding','g1_audit'}
    if type(value) is not dict or set(value)!=keys or type(value['schema_version']) is not int or value['schema_version']!=1 or value['task_id']!='G3' or value['kind']!='real_input_spec' or value['status']!='inputs_accepted':raise ValueError('explicit accepted G3 real input spec required')
    if value['contract_sha256'] not in G1_PINS or value['g1_audit']!=AUDIT or value['statistics_sha256']!=P4E_STATS_SHA:raise ValueError('accepted G1 contract/audit and P4E pins required')
    roles={'prepared_path':'prepare-manifest.json','validation_path':'validation-queries.json','statistics_path':'slice-statistics-manifest.json'}
    for key,name in roles.items():
        if type(value[key]) is not str or Path(value[key]).name!=name:raise ValueError('fixed input role; test closed')
    for key in ('contract_path','sid_path'):
        if type(value[key]) is not str:raise ValueError('explicit input path required')
    for key in ('contract_sha256','prepared_sha256','sid_sha256','validation_sha256','statistics_sha256'):
        pin=value[key]
        if type(pin) is not str or len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin):raise ValueError('external input pins required')
    binding=value['statistics_binding']
    if type(binding) is not dict or set(binding)!={'raw_records','raw_producer','prepared_sha','catalog_sha','consumer'} or any(binding[k].get('synthetic') is True for k in ('raw_producer','consumer') if type(binding[k]) is dict):raise ValueError('real statistics accepted lineage required')
    return value

def _context(root,contract,prepared,codec,queries,stats,binding):
    support=sorted(i for i,n in stats.item_counts.items() if n>0)
    if support!=list(prepared['support']):raise ValueError('shared full-event statistics/support mismatch')
    examples=gt.ExampleStore(prepared['examples_path'],binding['examples_sha256'],prepared['example_count'])
    return ge.RunContext(ge._CONTEXT_KEY,examples=examples,codec=codec,queries=queries,binding=binding,stats=stats).check()

def prepare_real(root,spec_path,spec_sha,output):
    """Only accepted pinned real inputs; all old loaders/strict-past gates apply."""
    root=Path(root);spec_path=gd._checked_file(spec_path,spec_sha)
    spec=_real_spec(gd._json(spec_path))
    contract=gd.load_contract(root,Path(spec['contract_path']),spec['contract_sha256'])
    for role,key in (('prepared','manifest_sha256'),('sid','manifest_sha256'),('validation','artifact_sha256')):
        if spec[role+'_sha256']!=contract['inputs'][role][key]:raise ValueError('G1/input external pin mismatch')
    expected=spec['statistics_binding']
    train=[r for r in contract['inputs']['prepared']['binding_checks']['input_records'] if '/train/week_' in r['role']]
    if len(train)!=25 or canonical_bytes(expected['raw_records'])!=canonical_bytes(train) or expected['prepared_sha']!=spec['prepared_sha256']:raise ValueError('all 25 pinned train partitions/shared statistics lineage required')
    prepared=gd.load_prepared(Path(spec['prepared_path']),contract)
    codec=gd.load_sid(Path(spec['sid_path']),contract,prepared)
    queries=gd.load_validation(Path(spec['validation_path']),contract,prepared['catalog'])
    stats=ev.read_slice_statistics(Path(spec['statistics_path']),spec['statistics_sha256'],expected,prepared['catalog'].tolist(),queries)
    binding={'schema_version':1,'task_id':'G3','phase':'real_prepared','synthetic':False,'engineering':False,
             'g1_manifest_sha256':spec['contract_sha256'],'g1_audit':AUDIT,'g2_source_sha256':G2_SHA,
             'consumer':_source(root,contract),'core_config_sha256':gd.CONFIG_SHA,
             'input_spec_sha256':spec_sha,'prepared_sha256':spec['prepared_sha256'],'sid_sha256':spec['sid_sha256'],
             'examples_sha256':contract['inputs']['examples']['artifact_sha256'],'queries_sha256':digest(queries),
             'validation_file_sha256':spec['validation_sha256'],'catalog_sha256':digest(prepared['catalog'].tolist()),
             'statistics_manifest_sha256':spec['statistics_sha256'],'statistics_binding_sha256':stats.binding_sha256,
             'engine_schema':'G2 primitives; G3 real binding; test closed'}
    context=_context(root,contract,prepared,codec,queries,stats,binding)
    output=gd._no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    path=output/'manifest.json';pin=ge._write_json(path,{'schema_version':1,'task_id':'G3','kind':'real_prepared',
        'spec_path':str(spec_path.absolute()),'spec_sha256':spec_sha,'binding':binding})
    return context,{'manifest_path':path,'manifest_sha256':pin}

def _engineering_context(root,original):
    contract=gd.load_contract(root,root/'artifacts/generative-contract-manifest.json',
        '40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d')
    binding={**original.binding,'task_id':'G3','phase':'engineering_prepared','engineering':True,
             'synthetic':True,'consumer':_source(root,contract),'g2_source_sha256':G2_SHA,
             'engine_schema':'G2 primitives; G3 original synthetic engineering; no real/CUDA claim'}
    return ge.RunContext(ge._CONTEXT_KEY,examples=original.examples,codec=original.codec,
        queries=original.queries,binding=binding,stats=original.slice_definition).check()

def prepare_engineering(root,output):
    root=Path(root);output=gd._no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    fixture=root/'tests/fixtures/g2_synthetic.json'
    original=ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json',
        '40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,file_sha256(fixture),output/'inputs')
    context=_engineering_context(root,original);path=output/'manifest.json'
    pin=ge._write_json(path,{'schema_version':1,'task_id':'G3','kind':'engineering_prepared','binding':context.binding,
        'engine_prepared_path':str((output/'inputs/manifest.json').absolute()),'engine_prepared_sha256':file_sha256(output/'inputs/manifest.json')})
    return context,{'manifest_path':path,'manifest_sha256':pin}

def load_context(root,path,pin,temporary):
    root=Path(root);value=gd._json(gd._checked_file(path,pin))
    if value.get('task_id')!='G3' or value.get('schema_version')!=1:raise ValueError('G3 prepared manifest required')
    if value.get('kind')=='engineering_prepared':
        original=ge.load_prepared_context(root,Path(value['engine_prepared_path']),value['engine_prepared_sha256'],Path(temporary)/'examples.jsonl')
        context=_engineering_context(root,original)
    elif value.get('kind')=='real_prepared':context,_=prepare_real(root,Path(value['spec_path']),value['spec_sha256'],Path(temporary)/'actual-input-check')
    else:raise ValueError('prepared role/test gate')
    if canonical_bytes(context.binding)!=canonical_bytes(value['binding']):raise ValueError('actual G3 prepared binding drift')
    return context

def runtime_snapshot(spec):
    import torch
    if type(spec) is not dict or set(spec)!={'mode','device','deterministic_algorithms','allow_tf32','environment','cuda_version','gpu_name','gpu_uuid','threads'} or spec['mode'] not in ('engineering','real'):raise ValueError('explicit frozen runtime specification')
    if spec['environment']!=gd._environment() or type(spec['threads']) is not int or spec['threads']<1 or spec['threads']!=torch.get_num_threads():raise ValueError('runtime exact environment/threads mismatch')
    if spec['mode']=='engineering':
        if spec['device']!='cpu' or any(spec[k] is not None for k in ('cuda_version','gpu_name','gpu_uuid')):raise ValueError('CPU original engineering only')
    else:
        if spec['device']!='cuda:0' or spec['deterministic_algorithms'] is not True or spec['allow_tf32'] is not False:raise ValueError('actual deterministic FP32 CUDA required')
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no CPU fallback')
        prop=torch.cuda.get_device_properties(0)
        if torch.version.cuda!=spec['cuda_version'] or prop.name!=spec['gpu_name'] or str(getattr(prop,'uuid',None))!=spec['gpu_uuid'] or spec['gpu_uuid'] in (None,'None'):raise ValueError('actual CUDA/device identity mismatch')
    if type(spec['deterministic_algorithms']) is not bool or type(spec['allow_tf32']) is not bool or torch.are_deterministic_algorithms_enabled()!=spec['deterministic_algorithms'] or torch.backends.cuda.matmul.allow_tf32!=spec['allow_tf32'] or torch.backends.cudnn.allow_tf32!=spec['allow_tf32']:raise ValueError('actual deterministic/TF32 flags mismatch')
    return copy.deepcopy(spec)

def engineering_runtime():
    import torch
    # Preserve actual backend flags; CPU engineering never initializes CUDA.
    if torch.backends.cuda.matmul.allow_tf32!=torch.backends.cudnn.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return runtime_snapshot({'mode':'engineering','device':'cpu','deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'environment':gd._environment(),'cuda_version':None,'gpu_name':None,'gpu_uuid':None,'threads':torch.get_num_threads()})

def configure_runtime(spec):
    """Apply a pinned profile in a fresh CLI process, then verify actual flags."""
    import torch
    required={'mode','device','deterministic_algorithms','allow_tf32','environment','cuda_version','gpu_name','gpu_uuid','threads'}
    if type(spec) is not dict or set(spec)!=required or spec['environment']!=gd._environment() or type(spec['deterministic_algorithms']) is not bool or type(spec['allow_tf32']) is not bool or spec['threads']!=torch.get_num_threads():raise ValueError('frozen runtime before backend configuration')
    if spec['mode']=='real' and (spec['device']!='cuda:0' or spec['deterministic_algorithms'] is not True or spec['allow_tf32'] is not False):raise ValueError('strict CUDA runtime profile')
    if spec['mode']=='engineering' and spec['device']!='cpu':raise ValueError('engineering CPU only')
    if spec['mode'] not in ('real','engineering'):raise ValueError('runtime mode')
    torch.use_deterministic_algorithms(spec['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32=spec['allow_tf32'];torch.backends.cudnn.allow_tf32=spec['allow_tf32']
    return runtime_snapshot(spec)

def engineering_limits(context,prepared,*,max_epochs=2):
    import datetime
    steps=max_epochs*math.ceil(len(context.examples)/512)
    return {'kind':'engineering_limits','runtime':engineering_runtime(),'max_epochs':max_epochs,'microbatch':2,'max_resumes':1,
        'prepared':{k:str(v) if isinstance(v,Path) else v for k,v in prepared.items()},'permit':None,'registration':None,
        'pool_spec':{'task_id':'G3','kind':'engineering_pool','steps_by_phase':{'formal':steps,'replay':steps},
            'max_steps':2*steps,'max_decode':(2*max_epochs+1)*len(context.queries),'max_disk_bytes':64*1024**2,'max_resumes':1,
            'deadline_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=60)).isoformat(),'binding_sha256':digest(context.binding)}}

def _limits(context,limits):
    from . import generative_pilot as gp
    required={'kind','runtime','max_epochs','microbatch','max_resumes','prepared','permit','pool_spec','registration'}
    if type(limits) is not dict or set(limits)!=required or type(limits['max_epochs']) is not int or not 1<=limits['max_epochs']<=30 or type(limits['microbatch']) is not int or not 1<=limits['microbatch']<=512 or limits['max_resumes']!=1:raise ValueError('explicit finite runtime limits')
    engineering=context.binding.get('engineering') is True and context.binding.get('synthetic') is True
    if engineering:
        if limits['kind']!='engineering_limits' or limits['runtime']['mode']!='engineering' or limits['pool_spec']['kind']!='engineering_pool' or limits['max_epochs']>2 or limits['permit'] is not None or limits['registration'] is not None:raise ValueError('bounded CPU engineering profile')
        import datetime
        spec=limits['pool_spec'];steps=limits['max_epochs']*math.ceil(len(context.examples)/512)
        remaining=(datetime.datetime.fromisoformat(spec['deadline_utc'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if spec['steps_by_phase']!={'formal':steps,'replay':steps} or spec['max_steps']!=2*steps or spec['max_decode']!=(2*limits['max_epochs']+1)*len(context.queries) or spec['max_disk_bytes']!=64*1024**2 or spec['max_resumes']!=1 or not 0<remaining<=60:raise ValueError('frozen CPU engineering cumulative caps/60 second window')
    else:
        if context.binding.get('synthetic') is not False or limits['kind']!='formal_limits' or limits['runtime']['mode']!='real' or limits['microbatch'] not in (512,256,128) or limits['max_epochs']!=30 or limits['pool_spec']['kind']!='formal_pool':raise ValueError('formal real profile; test closed')
        permit=limits['permit']
        if type(permit) is not dict or set(permit)!={'path','sha256'}:raise ValueError('Astra external formal permit required')
        value=gd._json(gd._checked_file(permit['path'],permit['sha256']))
        expected={'task_id':'G3','kind':'formal_permit','status':'approved','context_binding_sha256':digest(context.binding),
                  'limits':{k:v for k,v in limits.items() if k!='permit'}}
        if canonical_bytes(value)!=canonical_bytes(expected):raise ValueError('formal permit/input/runtime identity')
    if limits['pool_spec']['binding_sha256']!=digest(context.binding):raise ValueError('pool actual input binding')
    gp._spec(limits['pool_spec']);runtime_snapshot(limits['runtime'])
    return copy.deepcopy(limits)

def check_registration(config,entry,output):
    keys={'run_id','config_index','seed','output','selection'}
    if type(entry) is not dict or set(entry)!=keys or type(entry['config_index']) is not int or not 0<=entry['config_index']<4 or type(entry['seed']) is not int or entry['seed'] not in (42,2024,2025) or type(entry['output']) is not str or Path(entry['output']).absolute()!=Path(output).absolute() or config!=ge._config(entry['config_index'],entry['seed']):raise ValueError('fixed registration/config/seed/exclusive output')
    if entry['seed']==42:
        if entry['run_id']!=f"core-{entry['config_index']:02d}-seed42" or entry['selection'] is not None:raise ValueError('four registered seed42 runs only')
    else:
        receipt=entry['selection']
        if entry['run_id']!=f"selected-seed{entry['seed']}" or type(receipt) is not dict or set(receipt)!={'path','sha256'}:raise ValueError('additional seed requires frozen core selection')
        value=gd._json(gd._checked_file(receipt['path'],receipt['sha256']))
        if value.get('status')!='core_seed42_selection_frozen' or value.get('task_id')!='G3' or value.get('test_gate')!='closed' or len(value.get('runs',[]))!=4 or value.get('winner',{}).get('config_index')!=entry['config_index']:raise ValueError('selected configuration only; no best seed or extra HPO')

def _seed(seed):
    import random,numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def _guard(context,limits,output,pool):
    pool.check();runtime_snapshot(limits['runtime']);context.check()
    if context.binding['synthetic'] is False:
        from .generative_pilot import observe_resources
        observe_resources(output,limits['runtime']['device'])

def _progress(event,phase,config,epoch):
    import sys
    row={'task_id':'G3','event':event,'phase':phase,'seed':config['seed'],'d_model':config['d_model'],'learning_rate':config['learning_rate'],'epoch':epoch}
    payload=canonical_bytes(row)
    if len(payload)>256:raise ValueError('bounded progress record')
    sys.stderr.write(payload.decode());sys.stderr.flush()

def _train_trajectory(context,config,limits,output,pool,phase,*,parent=None,stop_after_step=None,should_stop=None):
    """Orchestrate existing engine primitives; local model steps stay local."""
    import time,torch
    from .generative_pilot import accounted_ledger,write_accounted_json,save_accounted_checkpoint
    output=gd._no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    pool.reserve(phase,'disk',8192)  # maxepochs30 + start/end, <=32 records.
    _progress('stage_start',phase,config,0)
    root=Path(__file__).resolve().parents[2];binding={**ge.build_binding(root,context,config),'runtime':limits['runtime'],
        'limits':limits,'pool_path':str(pool.root.absolute()),'trajectory':phase}
    _seed(config['seed']);model=gt.make_model(context.codec,config,device=limits['runtime']['device'])
    optimizer=gt.make_optimizer(model,config);cursor=gt.EpochCursor(config['seed'],len(context.examples))
    selected={'epoch':0,'best_epoch':None,'best_metric':None,'bad_epochs':0};epochs=[];evaluations=[];losses=[];best=None;pending=None
    if parent is None:
        budget={'reserved_steps':0,'completed_steps':0,'consumed_wall_by_phase':{'train':0.,'evaluate':0.,'verify':0.},'resume_attempts':0,
            'max_optimizer_steps':limits['max_epochs']*math.ceil(len(context.examples)/512),'max_resume_attempts':1,
            'deadline_utc':pool.spec['deadline_utc'],'parent_ledger_head':None}
        ledger=accounted_ledger(pool,phase)(output/'ledger',budget,binding)
    else:
        if parent['binding']!=binding or parent['status']!='interrupted':raise ValueError('same trajectory/source/runtime safe parent required')
        begin=time.perf_counter();state=gt.load_checkpoint(Path(parent['last']['path']),parent['last']['sha256'],model,optimizer,cursor,expected_binding=binding)
        ge.BudgetLedger.read_existing(Path(parent['ledger_path']),parent['ledger_head'],binding)
        if state.budget!=parent['budget'] or ge._state_digest(model,optimizer,cursor,state.selection)!=parent['state_sha256']:raise ValueError('parent actual checkpoint/state')
        selected=state.selection;epochs=copy.deepcopy(parent['epoch_metrics']);evaluations=copy.deepcopy(parent['evaluations']);losses=copy.deepcopy(parent['losses']);best=parent['best'];pending=parent['pending_evaluation']
        ledger=accounted_ledger(pool,phase).resume(Path(parent['ledger_path']),parent['ledger_head'],binding,output/'ledger');ledger.charge_wall('verify',time.perf_counter()-begin)
    number=0;last=None;status='complete'
    def checkpoint():
        nonlocal number,last
        _guard(context,limits,output,pool)
        path=output/f'checkpoint-{number:04d}.pt';number+=1
        pin,_=save_accounted_checkpoint(pool,phase,path,model,optimizer,cursor,binding=binding,selection=selected,budget=ledger.snapshot())
        last={'path':str(path.absolute()),'sha256':pin};pool.bind_child(phase,ledger.root,ledger.head);return last
    while True:
        if cursor.phase=='epoch_done':
            if cursor.epoch>=limits['max_epochs'] or selected['bad_epochs']>=3:break
            cursor.advance_epoch()
        while cursor.phase=='train':
            _guard(context,limits,output,pool);pool.reserve(phase,'steps');ledger.reserve_step();begin=time.perf_counter()
            indices=cursor.next_batch(512);result=gt.train_effective_batch(model,optimizer,context.examples.rows(indices),context.codec,microbatch=limits['microbatch'],device=limits['runtime']['device'])
            if limits['runtime']['device'].startswith('cuda'):torch.cuda.synchronize()
            elapsed=time.perf_counter()-begin;ledger.complete_step(elapsed);pool.charge_wall(phase,elapsed);pool.bind_child(phase,ledger.root,ledger.head)
            losses.append({'epoch':cursor.epoch,'examples':result.examples,'loss_mean':result.loss_mean})
            if stop_after_step is not None and ledger.snapshot()['completed_steps']>=stop_after_step or should_stop is not None and should_stop():
                checkpoint();status='interrupted';break
        if status=='interrupted':break
        if cursor.phase=='evaluate':
            _guard(context,limits,output,pool);candidate=pending['checkpoint'] if pending else checkpoint();begin=time.perf_counter()
            eval_output=output/f'evaluation-{cursor.epoch:02d}'
            pool.reserve(phase,'disk',len(context.queries)*4096+65536*(1+math.ceil(len(context.queries)/64)))
            # Engine calls the hook immediately before each query and immediately
            # after decode. Reserve once before each actual invocation.
            hook_calls=0
            def query_guard():
                nonlocal hook_calls
                _guard(context,limits,output,pool)
                if hook_calls>0 and hook_calls%2==1:pool.reserve(phase,'decode')
                hook_calls+=1
            try:
                evaluation=ev.evaluate_queries(model,context,output=eval_output,checkpoint_sha256=candidate['sha256'],shard_size=64,
                    resume_manifest=Path(pending['manifest_path']) if pending else None,resume_sha256=pending['manifest_sha256'] if pending else None,
                    should_stop=should_stop,budget_check=query_guard)
            except (KeyboardInterrupt,InterruptedError):
                evaluation=ge._recover_evaluation(eval_output,model,context,candidate['sha256'],64)
                if evaluation is None:
                    elapsed=time.perf_counter()-begin;ledger.charge_wall('evaluate',elapsed);pool.charge_wall(phase,elapsed);checkpoint();status='interrupted';break
            elapsed=time.perf_counter()-begin;ledger.charge_wall('evaluate',elapsed);pool.charge_wall(phase,elapsed);pool.bind_child(phase,ledger.root,ledger.head)
            if not evaluation['manifest']['complete']:
                pending={'manifest_path':str(evaluation['manifest_path'].absolute()),'manifest_sha256':evaluation['manifest_sha256'],'checkpoint':candidate}
                checkpoint();status='interrupted';break
            pending=None;aggregate=evaluation['aggregate'];main=aggregate['metrics']['all_eligible'];metric=[main['recall@50'],main['ndcg@50']]
            improved=selected['best_metric'] is None or tuple(metric)>tuple(selected['best_metric'])
            selected={'epoch':cursor.epoch,'best_epoch':cursor.epoch if improved else selected['best_epoch'],'best_metric':metric if improved else selected['best_metric'],'bad_epochs':0 if improved else selected['bad_epochs']+1}
            epochs.append({'epoch':cursor.epoch,'metrics':aggregate,'model_sha256':ev.model_digest(model)})
            evaluations.append({'epoch':cursor.epoch,'manifest_path':str(evaluation['manifest_path'].absolute()),'manifest_sha256':evaluation['manifest_sha256'],'checkpoint':candidate})
            cursor.complete_evaluation();checkpoint()
            _progress('epoch_complete',phase,config,cursor.epoch)
            if improved:best={**last,'epoch':cursor.epoch,'metric':metric}
            if should_stop is not None and should_stop() and cursor.epoch<limits['max_epochs']:status='interrupted';break
    if last is None:checkpoint()
    value={'schema_version':1,'task_id':'G3','kind':'trajectory','status':status,'binding':binding,'limits':limits,
        'selection':selected,'budget':ledger.snapshot(),'ledger_path':str(ledger.root.absolute()),'ledger_head':ledger.head,
        'last':last,'best':best,'epoch_metrics':epochs,'evaluations':evaluations,'losses':losses,'pending_evaluation':pending,
        'state_sha256':ge._state_digest(model,optimizer,cursor,selected),'parent':None if parent is None else parent['receipt'],
        'pool_head':pool.head,'pool_state':copy.deepcopy(pool.state)}
    path=output/'manifest.json';pin=write_accounted_json(pool,phase,path,value,stamp_pool=True)
    _progress('stage_complete' if status=='complete' else 'stage_interrupted',phase,config,cursor.epoch)
    return {'manifest':value,'manifest_path':path,'manifest_sha256':pin,'pool_head':pool.head,'pool_state':copy.deepcopy(pool.state)}

def run(context,config,limits,output,*,stop_after_step=None,should_stop=None):
    from .generative_pilot import ExecutionPool
    context.check();limits=_limits(context,limits);gt._core(config)
    if context.binding['synthetic'] is False:check_registration(config,limits['registration'],output)
    if context.binding['synthetic'] is False and stop_after_step is not None:raise ValueError('manual step interruption is engineering-only')
    output=Path(output)
    with ExecutionPool.create(output.with_name(output.name+'-pool'),limits['pool_spec']) as pool:
        return _train_trajectory(context,config,limits,output,pool,'formal',stop_after_step=stop_after_step,should_stop=should_stop)

def _read_run(context,path,pin):
    value=gd._json(gd._checked_file(path,pin))
    if value.get('task_id')!='G3' or value.get('kind')!='trajectory' or value.get('status') not in ('complete','interrupted'):raise ValueError('G3 actual trajectory manifest')
    binding=value['binding'];config=binding['effective_config'];root=Path(__file__).resolve().parents[2]
    expected={**ge.build_binding(root,context,config),'runtime':value['limits']['runtime'],'limits':value['limits'],'pool_path':binding['pool_path'],'trajectory':binding['trajectory']}
    if canonical_bytes(binding)!=canonical_bytes(expected):raise ValueError('current actual input/source/runtime identity')
    runtime_snapshot(binding['runtime']);ledger=ge.BudgetLedger.read_existing(Path(value['ledger_path']),value['ledger_head'],binding)
    b=ledger['budget'];b['parent_ledger_head']=value['ledger_head']
    if b!=value['budget'] or ledger['pending']:raise ValueError('actual final local ledger')
    model=gt.make_model(context.codec,config,device=binding['runtime']['device']);optimizer=gt.make_optimizer(model,config);cursor=gt.EpochCursor(config['seed'],len(context.examples))
    state=gt.load_checkpoint(Path(value['last']['path']),value['last']['sha256'],model,optimizer,cursor,expected_binding=binding)
    if state.budget!=value['budget'] or state.selection!=value['selection'] or ge._state_digest(model,optimizer,cursor,state.selection)!=value['state_sha256']:raise ValueError('actual last checkpoint state')
    if value['status']=='complete' and (cursor.phase!='epoch_done' or cursor.epoch!=value['limits']['max_epochs'] and state.selection['bad_epochs']!=3):raise ValueError('complete maxepochs/patience full validation')
    selected={'epoch':0,'best_epoch':None,'best_metric':None,'bad_epochs':0}
    if len(value['epoch_metrics'])!=len(value['evaluations']) or len(value['epoch_metrics'])!=state.selection['epoch']:raise ValueError('complete epoch coverage')
    for epoch,(record,receipt) in enumerate(zip(value['epoch_metrics'],value['evaluations']),1):
        ck=receipt['checkpoint'];gt.load_checkpoint(Path(ck['path']),ck['sha256'],model,optimizer,cursor,expected_binding=binding)
        if record['epoch']!=epoch or receipt['epoch']!=epoch or cursor.epoch!=epoch or cursor.phase!='evaluate' or ev.model_digest(model)!=record['model_sha256']:raise ValueError('epoch checkpoint actual phase/model')
        eb=ev._evaluation_binding(model,context,ck['sha256'],100,64)
        evaluation,rows=ev._read_evaluation(Path(receipt['manifest_path']),receipt['manifest_sha256'],eb,context.queries,context.catalog)
        aggregate=ev.aggregate_items([r['items'] for r in rows],context.queries,context.catalog,slice_definition=context.slice_definition)
        if not evaluation['complete'] or aggregate!=record['metrics'] or aggregate!=evaluation['aggregate'] or ev._computation(rows,aggregate)!=evaluation['computation_sha256']:raise ValueError('actual full query metrics/computation')
        main=aggregate['metrics']['all_eligible'];metric=[main['recall@50'],main['ndcg@50']];improved=selected['best_metric'] is None or tuple(metric)>tuple(selected['best_metric'])
        selected={'epoch':epoch,'best_epoch':epoch if improved else selected['best_epoch'],'best_metric':metric if improved else selected['best_metric'],'bad_epochs':0 if improved else selected['bad_epochs']+1}
    if selected!=value['selection']:raise ValueError('independent strict selection/earliest tie')
    if selected['epoch']:
        best=value['best'];gt.load_checkpoint(Path(best['path']),best['sha256'],model,optimizer,cursor,expected_binding=binding)
        if best['epoch']!=selected['best_epoch'] or best['metric']!=selected['best_metric'] or cursor.phase!='epoch_done' or ev.model_digest(model)!=value['epoch_metrics'][best['epoch']-1]['model_sha256']:raise ValueError('actual best checkpoint')
    elif value['best'] is not None:raise ValueError('best without full epoch')
    value['receipt']={'manifest_path':str(Path(path).absolute()),'manifest_sha256':pin}
    return value

def resume(context,path,pin,pool_head,output,*,should_stop=None):
    from .generative_pilot import ExecutionPool
    parent=_read_run(context,path,pin)
    if parent['status']!='interrupted':raise ValueError('safe interrupted parent only')
    limits=_limits(context,parent['limits'])
    with ExecutionPool.open(Path(parent['binding']['pool_path']),limits['pool_spec'],pool_head) as pool:
        pool.resume();return _train_trajectory(context,parent['binding']['effective_config'],limits,output,pool,'formal',parent=parent,should_stop=should_stop)

def verify(context,path,pin,pool_head,output,*,replay):
    from .generative_pilot import ExecutionPool,write_accounted_json
    value=_read_run(context,path,pin)
    if value['status']!='complete':raise ValueError('standalone verify complete only')
    limits=_limits(context,value['limits'])
    with ExecutionPool.open(Path(value['binding']['pool_path']),limits['pool_spec'],pool_head) as pool:
        if replay:
            fresh=_train_trajectory(context,value['binding']['effective_config'],limits,output,pool,'replay')
            for key in ('state_sha256','selection','losses','epoch_metrics'):
                if fresh['manifest'][key]!=value[key]:raise ValueError('actual device training/decode replay mismatch: '+key)
            for a,b in zip(fresh['manifest']['evaluations'],value['evaluations']):
                actual=gd._json(gd._checked_file(a['manifest_path'],a['manifest_sha256']));original=gd._json(gd._checked_file(b['manifest_path'],b['manifest_sha256']))
                if actual['computation_sha256']!=original['computation_sha256']:raise ValueError('actual per-query prediction/audit replay mismatch')
        pool.check()
        result={'schema_version':2,'task_id':'G3','kind':'verification','status':'verified','run_path':str(Path(path).absolute()),'run_sha256':pin,
            'replay':bool(replay),'pool_path':str(pool.root.absolute()),'pool_spec':pool.spec,'pool_head':pool.head,'pool_state':copy.deepcopy(pool.state),
            'boundary':'original synthetic CPU engineering' if context.binding['synthetic'] else 'actual CUDA real inputs; validation only'}
        if replay:
            roots,inputs=verification_scope(value,fresh['manifest'],context)
            result.update(binding=value['binding'],selection=value['selection'],epoch_metrics=value['epoch_metrics'],
                          epoch_count=len(value['epoch_metrics']),query_count=len(context.queries),test_gate='closed',
                          tree_roots=roots,input_files=inputs,files=sealed_files(roots,inputs,value['limits']['pool_spec'],budget_check=pool.check))
        output=Path(output)
        if not replay:output.mkdir(mode=0o700,exist_ok=False)
        receipt_path=output/'verification-receipt.json';receipt_sha=write_accounted_json(pool,'replay' if replay else 'formal',receipt_path,result,stamp_pool=True)
        return {**result,'manifest_path':receipt_path,'manifest_sha256':receipt_sha}

def seal_bounds(spec):
    """Finite metadata allowance derived from already registered execution caps."""
    count=1024+8*spec['max_steps']+2*spec['max_decode']
    return count,count*1536+131072

def sealed_files(roots,inputs,spec,*,budget_check=lambda:None):
    count,_=seal_bounds(spec);paths=set();visited=0
    for root in roots:
        budget_check()
        root=gd._no_symlink(root)
        for p in root.rglob('*'):
            budget_check();visited+=1
            if visited>2*count:raise ValueError('bounded closure traversal limit')
            gd._no_symlink(p)
            if p.is_file():paths.add(str(p.absolute()))
            if len(paths)>count:raise ValueError('bounded closure member limit')
    for path in inputs:paths.add(str(gd._no_symlink(path).absolute()))
    if len(paths)>count:raise ValueError('bounded closure member limit')
    records=[]
    for name in sorted(paths):
        budget_check()
        if len(name.encode())>1024:raise ValueError('bounded closure path limit')
        p=gd._no_symlink(name)
        if not p.is_file():raise ValueError('closure regular file required')
        h=hashlib.sha256()
        with p.open('rb') as stream:
            while True:
                budget_check();block=stream.read(1024**2)
                if not block:break
                h.update(block)
        budget_check();records.append({'path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    return records

def verification_scope(original,replayed,context):
    """Only durable verified trajectories and their accepted input artifacts."""
    root=Path(__file__).resolve().parents[2]
    trees={str(Path(original['receipt']['manifest_path']).parent),str(Path(replayed['ledger_path']).parent),str(Path(original['binding']['pool_path']))}
    parent=original['parent']
    while parent:
        p=gd._checked_file(parent['manifest_path'],parent['manifest_sha256']);trees.add(str(p.parent));parent=gd._json(p)['parent']
        if len(trees)>8:raise ValueError('bounded resume ancestry')
    prepared=original['limits']['prepared'];p=gd._checked_file(prepared['manifest_path'],prepared['manifest_sha256']);value=gd._json(p)
    inputs={str(root/name) for name in context.binding['consumer']['source_members']};inputs.add(str(p))
    if value['kind']=='engineering_prepared':
        trees.add(str(p.parent));engine=gd._json(gd._checked_file(value['engine_prepared_path'],value['engine_prepared_sha256']))
        inputs.update((engine['fixture_path'],engine['contract_path']))
    else:
        spec=gd._json(gd._checked_file(value['spec_path'],value['spec_sha256']));inputs.add(value['spec_path'])
        for role in ('contract','prepared','sid','validation','statistics'):inputs.add(spec[role+'_path'])
        for role in ('prepared','sid'):
            m=Path(spec[role+'_path']);payload=gd._json(m)
            for name in payload['artifacts']:
                if Path(name).name!=name:raise ValueError('input artifact role path')
                inputs.add(str(m.parent/name))
        stats=Path(spec['statistics_path']);inputs.add(str(stats.parent/gd._json(stats)['statistics_ref']))
    if original['limits']['permit']:inputs.add(original['limits']['permit']['path'])
    return sorted(trees),sorted(inputs)

def read_verified_v2(path,pin,invocation_path,invocation_pin,engineering):
    """Externally pinned success + byte integrity; no model/runtime initialization."""
    receipt=gd._json(gd._checked_file(path,pin));inv=gd._json(gd._checked_file(invocation_path,invocation_pin))
    receipt_keys={'schema_version','task_id','kind','status','run_path','run_sha256','replay','pool_path','pool_spec','pool_head','pool_state','boundary','binding','selection','epoch_metrics','epoch_count','query_count','test_gate','tree_roots','input_files','files'}
    inv_keys={'schema_version','kind','status','exit','pid','pgid','started_utc','ended_utc','verification_path','verification_sha256','helper_sha256','argv','cwd','environment','logs','pool_head','pool_state','files','closure_sha256','reserved_bytes'}
    if type(receipt) is not dict or set(receipt)!=receipt_keys or type(inv) is not dict or set(inv)!=inv_keys:raise ValueError('v2 receipt/invocation schema')
    if receipt.get('schema_version')!=2 or type(receipt['schema_version']) is not int or receipt.get('kind')!='verification' or receipt.get('task_id')!='G3' or receipt.get('status')!='verified' or receipt.get('replay') is not True or receipt.get('test_gate')!='closed':raise ValueError('v2 full replay receipt required')
    if inv.get('schema_version')!=2 or type(inv['schema_version']) is not int or inv.get('kind')!='verification_invocation' or inv.get('status')!='sealed' or type(inv.get('exit')) is not int or inv['exit']!=0 or inv.get('verification_path')!=str(Path(path).absolute()) or inv.get('verification_sha256')!=pin:raise ValueError('actual successful verification invocation required')
    binding=receipt['binding'];consumer=binding['consumer'];root=Path(__file__).resolve().parents[2]
    import sys,datetime
    argv=inv['argv']
    expected=[sys.executable,'-B','-m','vkgr.generative_real','verify','--manifest',receipt['run_path'],'--sha256',receipt['run_sha256'],'--pool-head',argv[10] if type(argv) is list and len(argv)>10 else None,'--replay','--output',str(Path(path).parent)]
    if engineering:expected.append('--engineering')
    if argv!=expected or type(argv[10]) is not str or len(argv[10])!=64 or inv['cwd']!=str(root) or inv['environment']!={'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}:raise ValueError('actual verification argv/environment')
    for key in ('pid','pgid'):
        if type(inv[key]) is not int or inv[key]<1:raise ValueError('actual invocation process identity')
    if inv['pid']!=inv['pgid'] or any(type(inv[k]) not in (int,float) or not math.isfinite(inv[k]) for k in ('started_utc','ended_utc')) or not inv['started_utc']<=inv['ended_utc']<datetime.datetime.fromisoformat(receipt['pool_spec']['deadline_utc']).timestamp():raise ValueError('actual invocation finite deadline')
    _,allowance=seal_bounds(receipt['pool_spec'])
    if type(inv['reserved_bytes']) is not int or inv['reserved_bytes']!=allowance or type(inv['logs']) is not dict or set(inv['logs'])!={'stdout','stderr'}:raise ValueError('bounded invocation envelope')
    for name,entry in inv['logs'].items():
        if type(entry) is not dict or set(entry)!={'path','bytes','sha256'} or entry['path']!=str(Path(invocation_path).parent/name) or type(entry['bytes']) is not int or not 0<=entry['bytes']<=32768:raise ValueError('bounded invocation logs')
        p=gd._checked_file(entry['path'],entry['sha256'])
        if p.stat().st_size!=entry['bytes']:raise ValueError('actual invocation log bytes')
    if binding.get('engineering') is not engineering or binding.get('synthetic') is not engineering:raise ValueError('engineering/real receipt identity')
    if digest(consumer['source_members'])!=consumer['source_sha256']:raise ValueError('source declaration hash')
    for name,sha in consumer['source_members'].items():gd._checked_file(gd._path(root,name),sha)
    if gd._git(root)!={k:consumer[k] for k in ('git_base','git_dirty')} or gd._environment()!={k:consumer[k] for k in ('python','exact_dependencies')}:raise ValueError('current source/environment identity')
    if inv['helper_sha256']!=file_sha256(root/'cache/g3-formal-invoke-v2.py'):raise ValueError('invocation helper source identity')
    current=sealed_files(receipt['tree_roots'],receipt['input_files'],receipt['pool_spec'])
    if inv['files']!=current or inv['closure_sha256']!=digest(current):raise ValueError('sealed closure identity or tail drift')
    if not all(row in current for row in receipt['files']):raise ValueError('verified payload closure drift')
    if inv['pool_head']!=receipt['pool_head'] or inv['pool_state']!=receipt['pool_state'] or not any(row['sha256']==argv[10] and str(Path(row['path']).parent)==receipt['pool_path'] for row in current):raise ValueError('sealed pool identity')
    run=gd._json(gd._checked_file(receipt['run_path'],receipt['run_sha256']))
    if run['binding']!=binding or run['status']!='complete' or run['selection']!=receipt['selection'] or run['epoch_metrics']!=receipt['epoch_metrics'] or receipt['epoch_count']!=len(run['evaluations']):raise ValueError('verified selection/coverage identity')
    if type(receipt['query_count']) is not int or receipt['query_count']<1:raise ValueError('verified query coverage')
    return run

def evaluate(context,path,pin,pool_head,output,*,should_stop=None):
    import time
    from .generative_pilot import ExecutionPool
    value=_read_run(context,path,pin)
    if value['status']!='complete':raise ValueError('selected best evaluation complete run only')
    limits=_limits(context,value['limits']);cfg=value['binding']['effective_config']
    with ExecutionPool.open(Path(value['binding']['pool_path']),limits['pool_spec'],pool_head) as pool:
        model=gt.make_model(context.codec,cfg,device=limits['runtime']['device']);opt=gt.make_optimizer(model,cfg);cursor=gt.EpochCursor(cfg['seed'],len(context.examples))
        best=value['best'];gt.load_checkpoint(Path(best['path']),best['sha256'],model,opt,cursor,expected_binding=value['binding'])
        pool.reserve('formal','disk',len(context.queries)*4096+65536*(1+math.ceil(len(context.queries)/64)))
        hook_calls=0;begin=time.perf_counter()
        def guard():
            nonlocal hook_calls
            _guard(context,limits,Path(output).parent,pool)
            if hook_calls>0 and hook_calls%2==1:pool.reserve('formal','decode')
            hook_calls+=1
        try:result=ev.evaluate_queries(model,context,output=output,checkpoint_sha256=best['sha256'],shard_size=64,should_stop=should_stop,budget_check=guard)
        finally:pool.charge_wall('formal',time.perf_counter()-begin)
        return {**result,'pool_head':pool.head,'pool_state':copy.deepcopy(pool.state)}

def select(verification_paths,pins,output,*,engineering=False,invocation_paths=None,invocation_pins=None):
    if len(verification_paths)!=len(pins) or len(pins)!=(1 if engineering else 4):raise ValueError('all registered configurations verified before selection')
    if invocation_paths is None or invocation_pins is None or len(invocation_paths)!=len(pins) or len(invocation_pins)!=len(pins):raise ValueError('actual invocation pins required')
    rows=[];identities=set();common=None
    for path,pin,inv,inv_pin in zip(verification_paths,pins,invocation_paths,invocation_pins):
        actual=read_verified_v2(path,pin,inv,inv_pin,engineering)
        cfg=actual['binding']['effective_config'];identity=(cfg['learning_rate'],cfg['d_model'])
        order=[(.001,64),(.001,128),(.0003,64),(.0003,128)]
        if cfg['seed']!=42 or identity not in order or identity in identities:raise ValueError('unique registered seed42 config identity')
        identities.add(identity)
        signature=digest({'context':{k:v for k,v in actual['binding'].items() if k not in ('effective_config','runtime','limits','pool_path','trajectory')},'runtime':actual['limits']['runtime']})
        if common is not None and common!=signature:raise ValueError('common inputs/source/runtime for fair selection')
        common=signature;rows.append({'config_index':order.index(identity),'metric':actual['selection']['best_metric'],'best_epoch':actual['selection']['best_epoch'],
                                    'run_path':str(Path(actual['receipt']['manifest_path']).absolute()) if 'receipt' in actual else str(Path(gd._json(gd._checked_file(path,pin))['run_path']).absolute()),
                                    'run_sha256':gd._json(gd._checked_file(path,pin))['run_sha256'],'verification_sha256':pin,'invocation_sha256':inv_pin})
    rows.sort(key=lambda r:r['config_index']);winner=max(rows,key=lambda r:tuple(r['metric']))
    output=gd._no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    value={'schema_version':2,'task_id':'G3','kind':'selection','status':'engineering_selection_no_real_winner' if engineering else 'core_seed42_selection_frozen',
        'runs':rows,'winner':None if engineering else winner,'engineering_oracle':winner if engineering else None,'input_source_runtime_sha256':common,'test_gate':'closed'}
    path=output/'manifest.json';pin=ge._write_json(path,value)
    return {'status':value['status'],'manifest_path':path,'manifest_sha256':pin}

def main(argv=None):
    import argparse,json,signal,sys,tempfile
    parser=argparse.ArgumentParser(description='G3 pinned inputs and bounded execution; GPU requires external permit; test closed')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','pilot','train','resume','evaluate','verify','select'):
        p=sub.add_parser(command);p.add_argument('--engineering',action='store_true');p.add_argument('--output',type=Path,required=True)
        if command=='prepare':p.add_argument('--spec',type=Path);p.add_argument('--spec-sha')
        if command in ('train','pilot'):
            p.add_argument('--prepared',type=Path,required=True);p.add_argument('--prepared-sha',required=True)
            p.add_argument('--permit',type=Path);p.add_argument('--permit-sha')
        if command=='train':
            p.add_argument('--config-index',type=int,choices=range(4),default=0);p.add_argument('--seed',type=int,choices=(42,2024,2025),default=42)
            p.add_argument('--stop-after-step',type=int)
        if command=='resume':
            p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-sha',required=True);p.add_argument('--pool-head',required=True)
        if command in ('verify','evaluate'):
            p.add_argument('--manifest',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--pool-head',required=True)
            if command=='verify':p.add_argument('--replay',action='store_true')
        if command=='select':
            p.add_argument('--verification',type=Path,action='append',required=True);p.add_argument('--verification-sha',action='append',required=True)
            p.add_argument('--verification-invocation',type=Path,action='append',required=True);p.add_argument('--verification-invocation-sha',action='append',required=True)
    args=parser.parse_args(argv);root=Path(__file__).resolve().parents[2];stop=[False]
    def handle(signum,frame):stop[0]=True
    old_handlers={sig:signal.signal(sig,handle) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        if args.command=='prepare':
            if args.engineering:
                if args.spec is not None or args.spec_sha is not None:raise ValueError('engineering cannot accept real input spec')
                _,result=prepare_engineering(root,args.output)
            else:
                if args.spec is None or args.spec_sha is None:raise ValueError('real prepare explicit accepted spec pins')
                _,result=prepare_real(root,args.spec,args.spec_sha,args.output)
        elif args.command=='select':result=select(args.verification,args.verification_sha,args.output,engineering=args.engineering,invocation_paths=args.verification_invocation,invocation_pins=args.verification_invocation_sha)
        else:
            if args.command in ('train','pilot'):
                if not args.engineering and (args.permit is None or args.permit_sha is None):raise ValueError('real execution external permit required before inputs')
                prepared={'manifest_path':str(args.prepared.absolute()),'manifest_sha256':args.prepared_sha}
                value=None
            else:
                path=args.parent if args.command=='resume' else args.manifest;pin=args.parent_sha if args.command=='resume' else args.sha256
                value=gd._json(gd._checked_file(path,pin));prepared=value['limits']['prepared']
            with tempfile.TemporaryDirectory() as d:
                d=str(Path(d).resolve())
                context=load_context(root,Path(prepared['manifest_path']),prepared['manifest_sha256'],Path(d))
                if context.binding['engineering'] is not args.engineering:raise ValueError('explicit engineering versus real mode')
                if args.command=='train':
                    if args.engineering:limits=engineering_limits(context,prepared)
                    else:
                        permit=gd._json(gd._checked_file(args.permit,args.permit_sha));limits={**permit['limits'],'permit':{'path':str(args.permit.absolute()),'sha256':args.permit_sha}}
                    configure_runtime(limits['runtime'])
                    result=run(context,ge._config(args.config_index,args.seed),limits,args.output,stop_after_step=args.stop_after_step,should_stop=lambda:stop[0])
                elif args.command in ('resume','verify','evaluate'):
                    configure_runtime(value['limits']['runtime'])
                    if args.command=='resume':result=resume(context,args.parent,args.parent_sha,args.pool_head,args.output,should_stop=lambda:stop[0])
                    elif args.command=='verify':result=verify(context,args.manifest,args.sha256,args.pool_head,args.output,replay=args.replay)
                    else:result=evaluate(context,args.manifest,args.sha256,args.pool_head,args.output,should_stop=lambda:stop[0])
                else:
                    from .generative_pilot import pilot
                    result=pilot(context,args.output,engineering=args.engineering,permit_path=args.permit,permit_sha=args.permit_sha,prepared=prepared)
        receipt={k:str(v.absolute()) if isinstance(v,Path) else v for k,v in result.items() if k in ('status','manifest_path','manifest_sha256','pool_head','pool_state','replay','boundary')}
        print(json.dumps(receipt,sort_keys=True));return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        print('G3 gate failed: '+str(exc),file=sys.stderr);return 1
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)

if __name__=='__main__':
    from .generative_real import main as canonical_main
    raise SystemExit(canonical_main())
