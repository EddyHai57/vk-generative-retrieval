"""G2 stage orchestration, bounded budgets and independently replayable CLI."""
import copy
import datetime
import hashlib
import math
from contextlib import contextmanager
from pathlib import Path
from .data_audit import canonical_bytes,file_sha256
from .generative_data import _no_symlink,_json
from .generative_training import _budget


_CONTEXT_KEY=object()
_G2_MEMBERS=tuple(f'{folder}/{name}.py' for folder in ('src/vkgr','tests') for name in
                  (('generative_training','generative_evaluation','generative_experiment') if folder=='src/vkgr' else
                   ('test_generative_training','test_generative_evaluation','test_generative_experiment')))+('tests/fixtures/g2_synthetic.json',)


def _digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _write_json(path,value):
    import os
    data=canonical_bytes(value);path=_no_symlink(path)
    with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


class RunContext:
    """Factory-issued context; copies expose metadata without allowing mutation."""
    def __init__(self,key=None,*,examples=None,codec=None,queries=None,binding=None,stats=None):
        if key is not _CONTEXT_KEY:raise ValueError('prepare_context factory required')
        self._seal=key;self.examples=examples;self.codec=codec
        self._queries=canonical_bytes(queries);self._binding=canonical_bytes(binding)
        self._catalog=tuple(map(int,codec.catalog.ids));self.slice_definition=stats
        self._references=(examples,codec,stats)
        self.role='validation'
        self._sealed=True

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('RunContext metadata is immutable')
        object.__setattr__(self,name,value)

    @property
    def queries(self):
        import json
        return json.loads(self._queries)

    @property
    def binding(self):
        import json
        return json.loads(self._binding)

    @property
    def catalog(self):return self._catalog

    @property
    def support(self):return tuple(item for item in self.catalog if self.slice_definition.item_counts[item]>0)

    def check(self):
        if self._seal is not _CONTEXT_KEY or self.role!='validation' or (self.examples,self.codec,self.slice_definition)!=self._references:raise ValueError('context reference/role mismatch')
        self.codec.check_identity()
        if tuple(map(int,self.codec.catalog.ids))!=self.catalog:raise ValueError('context catalog identity')
        if _digest(self.queries)!=self.binding['queries_sha256'] or _digest(list(self.catalog))!=self.binding['catalog_sha256']:raise ValueError('context canonical metadata identity')
        return self


def prepare_context(root,manifest_path,expected_sha256,*,fixture_path=None,fixture_sha256=None,
                    prepared_path=None,sid_path=None,validation_path=None,synthetic=False,
                    stats_manifest_path=None,stats_manifest_sha256=None,expected_stats_binding=None,
                    examples_path=None):
    # Authorization/role gates precede stat/open, including contract loading.
    if synthetic is not True or any(p is not None for p in (prepared_path,sid_path,validation_path)):
        raise ValueError('G2 synthetic validation only; real/test gate closed')
    if any(p is None for p in (fixture_path,fixture_sha256,stats_manifest_path,stats_manifest_sha256,expected_stats_binding,examples_path)):
        raise ValueError('explicit fixture/statistics pins and examples output required')
    import os
    from . import generative_data as gd
    from .generative_foundation import prepare_fixture
    from .generative_training import ExampleStore
    from .generative_evaluation import read_slice_statistics
    contract=gd.load_contract(root,manifest_path,expected_sha256)
    fixture=gd._json(gd._checked_file(fixture_path,fixture_sha256));prepared=prepare_fixture(fixture)
    codec=gd._ValidatedCodec(prepared['codec'].catalog)
    stats=read_slice_statistics(stats_manifest_path,stats_manifest_sha256,expected_stats_binding,fixture['catalog'],prepared['queries'])
    # Recompute synthetic statistics from all train-like events, never examples.
    items={item:0 for item in fixture['catalog']};users={}
    for partition in fixture['train_partitions']:
        for row in partition['rows']:
            if row['like']:
                items[row['item_id']]+=1;users[row['user_id']]=users.get(row['user_id'],0)+1
    if dict(stats.item_counts)!=items or dict(stats.user_counts)!=users:raise ValueError('synthetic full-train statistics mismatch')
    data=b''.join(canonical_bytes(row).rstrip(b'\n')+b'\n' for row in prepared['examples']);path=gd._no_symlink(examples_path)
    with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    examples=ExampleStore(path,hashlib.sha256(data).hexdigest(),len(prepared['examples']))
    files={**contract['consumer']['source_members'],**{name:file_sha256(gd._path(root,name)) for name in _G2_MEMBERS}}
    import torch
    binding={'schema_version':1,'task_id':'G2','synthetic':True,'g1_manifest_sha256':expected_sha256,
             'consumer':{**gd._git(root),**gd._environment(),'source_members':files,'source_sha256':_digest(files),'lock_sha256':contract['consumer']['lock_sha256']},
             'core_config_sha256':gd.CONFIG_SHA,'fixture_sha256':fixture_sha256,
             'examples_sha256':examples.expected_sha256 if hasattr(examples,'expected_sha256') else hashlib.sha256(data).hexdigest(),
             'queries_sha256':_digest(prepared['queries']),'catalog_sha256':_digest(fixture['catalog']),
             'codec_sha256':_digest(fixture['sids']),'statistics_manifest_sha256':stats_manifest_sha256,
             'statistics_binding_sha256':stats.binding_sha256,'device':'cpu','threads':torch.get_num_threads(),
             'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
             'g1_audit':{'aggregate_sha256':'d03978b3e4d5346208c605866640b25d83a36e4164419364dc5bd4c0be817e47','supervision_sha256':'ce39dac731f7ff411a8831bdda255338dc1b1e4de11f476bc33d1173a4097d4c'}}
    return RunContext(_CONTEXT_KEY,examples=examples,codec=codec,queries=prepared['queries'],binding=binding,stats=stats)


def prepare_synthetic(root,contract_path,contract_sha,fixture_path,fixture_sha,output):
    """Produce bounded original synthetic inputs and factory-validated context."""
    from . import generative_data as gd
    from .generative_foundation import prepare_fixture
    from .generative_evaluation import ITEM_GROUPS,USER_GROUPS
    gd.load_contract(root,contract_path,contract_sha)
    fixture=gd._json(gd._checked_file(fixture_path,fixture_sha));prepared=prepare_fixture(fixture)
    items={str(item):0 for item in fixture['catalog']};users={}
    for partition in fixture['train_partitions']:
        for row in partition['rows']:
            if row['like']:
                items[str(row['item_id'])]+=1;uid=str(row['user_id']);users[uid]=users.get(uid,0)+1
    counts=sorted(c for c in items.values() if c>0)
    thresholds={'median':counts[math.ceil(len(counts)*.5)-1],'p90':counts[math.ceil(len(counts)*.9)-1]}
    output=_no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    stats={'schema_version':1,'popularity_thresholds':thresholds,'item_like_counts':items,'user_like_counts':users,'user_buckets':[1,5,20],'definition':'train_like_counts_only_no_holdout_boundaries'}
    stat_sha=_write_json(output/'train-slices.json',stats)
    train_bytes=canonical_bytes(fixture['train_partitions'])
    binding={'raw_records':[{'role':'synthetic_train','sha256':hashlib.sha256(train_bytes).hexdigest(),'size_bytes':len(train_bytes)}],
             'raw_producer':{'synthetic':True,'fixture_sha256':fixture_sha},'prepared_sha':_digest(prepared['examples']),
             'catalog_sha':_digest(fixture['catalog']),'consumer':{'synthetic':True,'statistics_definition':'full_train_like_events_v1'}}
    manifest={'schema_version':1,'task_id':'P4E','kind':'slice_statistics','binding':binding,'statistics_ref':'train-slices.json','statistics_sha':stat_sha,
              'threshold_definition':'nearest_rank_positive_item_event_counts_median_p90','thresholds':thresholds,'item_groups':list(ITEM_GROUPS),'user_groups':list(USER_GROUPS),'holdout_used':False}
    manifest_path=output/'slice-statistics-manifest.json';stat_manifest_sha=_write_json(manifest_path,manifest)
    context=prepare_context(root,contract_path,contract_sha,fixture_path=fixture_path,fixture_sha256=fixture_sha,synthetic=True,
                            stats_manifest_path=manifest_path,stats_manifest_sha256=stat_manifest_sha,expected_stats_binding=binding,examples_path=output/'examples.jsonl')
    _write_json(output/'manifest.json',{'schema_version':1,'task_id':'G2','kind':'synthetic_prepared','fixture_path':str(Path(fixture_path).absolute()),
                  'contract_path':str(Path(contract_path).absolute()),'binding':context.binding,'statistics_binding':binding,'statistics_manifest_sha256':stat_manifest_sha})
    return context


def _live(budget):
    _budget(budget)
    if datetime.datetime.fromisoformat(budget['deadline_utc'])<=datetime.datetime.now(datetime.timezone.utc):raise ValueError('absolute runtime deadline expired')


class BudgetLedger:
    """Immutable write-ahead step records, external head and read-only parent fork."""
    def __init__(self,root,budget,binding):
        _live(budget)
        if budget['reserved_steps'] or budget['completed_steps'] or budget['resume_attempts'] or budget['parent_ledger_head'] is not None:raise ValueError('new run cannot reset inherited budget')
        self._start(root,budget,binding,'initial')

    def _start(self,root,budget,binding,action):
        self.root=_no_symlink(root);self.root.mkdir(mode=0o700,exist_ok=False)
        self.binding=copy.deepcopy(binding);self._budget=copy.deepcopy(budget);self._pending=False;self.head=None;self._index=0
        self._append(action,self._budget,False)

    def _append(self,action,budget,pending,phase=None,elapsed=0.):
        import os
        _budget(budget)
        if type(pending) is not bool or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('ledger event types')
        row={'schema_version':1,'index':self._index,'action':action,'binding':self.binding,'previous_head':self.head,'budget':copy.deepcopy(budget),'pending':pending,'phase':phase,'elapsed_seconds':elapsed}
        data=canonical_bytes(row);path=self.root/f'ledger-{self._index:06d}.json';_no_symlink(path)
        with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        self.head=hashlib.sha256(data).hexdigest();self._index+=1;self._budget=copy.deepcopy(budget);self._pending=pending

    def snapshot(self):
        result=copy.deepcopy(self._budget);result['parent_ledger_head']=self.head;return result

    def reserve_step(self):
        _live(self._budget)
        if self._pending:raise ValueError('complete reserved step first')
        budget=copy.deepcopy(self._budget);budget['reserved_steps']+=1
        self._append('reserve',budget,True)

    def complete_step(self,elapsed_seconds):
        if not self._pending:raise ValueError('no reserved optimizer step')
        budget=copy.deepcopy(self._budget);budget['completed_steps']+=1;budget['consumed_wall_by_phase']['train']+=elapsed_seconds
        self._append('complete',budget,False,'train',elapsed_seconds)

    def charge_wall(self,phase,elapsed_seconds):
        if phase not in ('train','evaluate','verify') or self._pending:raise ValueError('wall phase or pending step conflict')
        budget=copy.deepcopy(self._budget);budget['consumed_wall_by_phase'][phase]+=elapsed_seconds
        self._append('wall',budget,False,phase,elapsed_seconds)

    @staticmethod
    def read_existing(root,expected_head,binding):
        root=_no_symlink(root)
        paths=sorted(root.iterdir())
        if not paths:raise ValueError('empty budget ledger')
        previous=None;prior=None
        for index,path in enumerate(paths):
            if path.name!=f'ledger-{index:06d}.json' or path.is_symlink() or not path.is_file():raise ValueError('ledger member/gap gate')
            row=_json(path);keys={'schema_version','index','action','binding','previous_head','budget','pending','phase','elapsed_seconds'}
            if type(row) is not dict or set(row)!=keys or type(row['schema_version']) is not int or row['schema_version']!=1 or type(row['index']) is not int or row['index']!=index or canonical_bytes(row['binding'])!=canonical_bytes(binding) or row['previous_head']!=previous or type(row['pending']) is not bool or type(row['elapsed_seconds']) not in (int,float) or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<0:raise ValueError('budget chain schema/identity/hash')
            _budget(row['budget'])
            if index==0:
                if row['action'] not in ('initial','resume') or row['phase'] is not None or row['elapsed_seconds']!=0 or row['pending']:raise ValueError('budget chain initial event')
                b=row['budget']
                if row['action']=='initial' and (b['reserved_steps'] or b['completed_steps'] or b['resume_attempts'] or b['parent_ledger_head'] is not None):raise ValueError('budget reset or false initial')
                if row['action']=='resume' and (b['resume_attempts']<1 or b['parent_ledger_head'] is None):raise ValueError('resume provenance missing')
            else:
                expected=copy.deepcopy(prior['budget']);pending=prior['pending'];action=row['action'];phase=row['phase'];elapsed=row['elapsed_seconds']
                if action=='reserve':
                    if pending or phase is not None or elapsed!=0:raise ValueError('reserve ordering')
                    expected['reserved_steps']+=1;pending=True
                elif action=='complete':
                    if not pending or phase!='train':raise ValueError('complete ordering')
                    expected['completed_steps']+=1;expected['consumed_wall_by_phase']['train']+=elapsed;pending=False
                elif action=='wall':
                    if pending or phase not in ('train','evaluate','verify'):raise ValueError('wall ordering')
                    expected['consumed_wall_by_phase'][phase]+=elapsed
                else:raise ValueError('unknown budget event')
                if canonical_bytes(expected)!=canonical_bytes(row['budget']) or pending!=row['pending']:raise ValueError('budget monotonic transition conflict')
            previous=file_sha256(path);prior=row
        if previous!=expected_head:raise ValueError('external ledger head mismatch')
        return copy.deepcopy(prior)

    @classmethod
    def resume(cls,parent,expected_head,binding,output):
        state=cls.read_existing(parent,expected_head,binding);budget=copy.deepcopy(state['budget']);budget['resume_attempts']+=1;budget['parent_ledger_head']=expected_head
        # An unfinished reservation remains charged; only new work starts unreserved.
        _live(budget)
        result=cls.__new__(cls);result._start(output,budget,binding,'resume');return result


def build_binding(root,context,config):
    from . import generative_data as gd,generative_training as gt
    if type(context) is not RunContext:raise ValueError('factory-issued context required')
    context.check();gt._core(config)
    consumer=context.binding['consumer']
    for name,pin in consumer['source_members'].items():
        if file_sha256(gd._path(root,name))!=pin:raise ValueError('fresh G2 source identity changed')
    if gd._git(root)!={k:consumer[k] for k in ('git_base','git_dirty')} or gd._environment()!= {k:consumer[k] for k in ('python','exact_dependencies')}:raise ValueError('runtime Git/environment identity changed')
    return {**context.binding,'effective_config':copy.deepcopy(config)}


def _state_digest(model,optimizer,cursor,selection):
    from .generative_training import _rng
    import torch
    def normalized(value):
        if isinstance(value,torch.Tensor):
            t=value.detach().cpu().contiguous()
            return {'tensor_dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest()}
        if isinstance(value,dict):return {str(k):normalized(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [normalized(v) for v in value]
        return value
    return _digest(normalized({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'cursor':cursor.state_dict(),'selection':selection,'rng':_rng(model)}))


def _recover_evaluation(output,model,context,checkpoint_sha,shard_size):
    """Recover only a complete, read-back-validated receipt from our owned dir."""
    from . import generative_evaluation as ev
    if not output.is_dir():return None
    candidates=[output/'manifest.json'] if (output/'manifest.json').is_file() else sorted(output.glob('progress-*.json'),reverse=True)
    if not candidates:return None
    path=candidates[0];pin=file_sha256(_no_symlink(path));binding=ev._evaluation_binding(model,context,checkpoint_sha,100,shard_size)
    manifest,rows=ev._read_evaluation(path,pin,binding,context.queries,context.catalog)
    aggregate=ev.aggregate_items([r['items'] for r in rows],context.queries,context.catalog,slice_definition=context.slice_definition) if manifest['complete'] else None
    if canonical_bytes(aggregate)!=canonical_bytes(manifest['aggregate']) or ev._computation(rows,aggregate)!=manifest['computation_sha256']:raise ValueError('interrupted evaluation receipt computation')
    return {'manifest':manifest,'manifest_path':path,'manifest_sha256':pin,'aggregate':aggregate,'observations':manifest['observations']}


def run_train(context,config,*,output,limits,resume_path=None,resume_sha256=None,
              parent_manifest=None,parent_sha256=None,stop_after_step=None,verification_deadline=None,
              evaluation_shard_size=64,stop_after_evaluation_shards=None,should_stop=None):
    """G2-only complete epochs, full validation, safe boundaries and exact resume."""
    import random,time
    import numpy as np
    import torch
    from . import generative_training as gt,generative_data as gd,generative_evaluation as ev
    if type(context) is not RunContext or context.binding.get('synthetic') is not True:raise ValueError('G2 synthetic gate closed for real input')
    context.check()
    required={'max_epochs','max_wall_seconds','max_resume_attempts','microbatch','max_optimizer_steps'}
    if type(limits) is not dict or set(limits)!=required or any(type(v) is not int for v in limits.values()) or limits['max_epochs']!=2 or limits['max_wall_seconds']!=60 or limits['max_resume_attempts']!=1 or not 1<=limits['microbatch']<=512 or limits['max_optimizer_steps']!=2*math.ceil(len(context.examples)/512):raise ValueError('frozen G2 synthetic budget 2 epochs/60s/one resume')
    if stop_after_step is not None and (type(stop_after_step) is not int or stop_after_step<1):raise ValueError('safe synthetic stop step')
    if type(evaluation_shard_size) is not int or not 1<=evaluation_shard_size<=64:raise ValueError('bounded evaluation shard size')
    supplied=(resume_path,resume_sha256,parent_manifest,parent_sha256)
    if any(v is not None for v in supplied) and any(v is None for v in supplied):raise ValueError('paired checkpoint and parent external pins required')
    root=Path(__file__).resolve().parents[2]
    binding={**build_binding(root,context,config),'runtime_limits':copy.deepcopy(limits),'evaluation_shard_size':evaluation_shard_size}
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    model=gt.make_model(context.codec,config,device='cpu');optimizer=gt.make_optimizer(model,config);cursor=gt.EpochCursor(config['seed'],len(context.examples))
    selected={'epoch':0,'best_epoch':None,'best_metric':None,'bad_epochs':0};best=None;epochs=[];evaluations=[];losses=[];parent=None;pending=None
    resume_verification_seconds=0.
    if resume_path is not None:
        resume_begin=time.perf_counter()
        parent=gd._json(gd._checked_file(parent_manifest,parent_sha256))
        if parent.get('task_id')!='G2' or parent.get('kind')!='synthetic_run' or canonical_bytes(parent.get('binding'))!=canonical_bytes(binding) or parent.get('status')!='interrupted' or parent.get('last')!={'path':str(Path(resume_path).absolute()),'sha256':resume_sha256}:raise ValueError('parent run/checkpoint/config identity')
        _validate_run(root,parent_manifest,parent_sha256,context)
        ledger_state=BudgetLedger.read_existing(Path(parent['ledger_path']),parent['ledger_head'],binding)
        state=gt.load_checkpoint(resume_path,resume_sha256,model,optimizer,cursor,expected_binding=binding)
        parent_budget=copy.deepcopy(ledger_state['budget']);parent_budget['parent_ledger_head']=parent['ledger_head']
        if canonical_bytes(state.budget)!=canonical_bytes(parent['budget']) or canonical_bytes(parent_budget)!=canonical_bytes(state.budget):raise ValueError('checkpoint/parent/current ledger budget divergence')
        if _state_digest(model,optimizer,cursor,state.selection)!=parent['state_sha256']:raise ValueError('parent actual model/optimizer/RNG/cursor state')
        _live(state.budget);selected=state.selection;best=parent['best'];epochs=parent['epoch_metrics'];evaluations=parent['evaluations'];losses=parent['losses'];pending=parent['pending_evaluation']
        if state.budget['resume_attempts']>=state.budget['max_resume_attempts']:raise ValueError('resume attempts exhausted')
        resume_verification_seconds=time.perf_counter()-resume_begin
    output=_no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    if parent is None:
        budget={'reserved_steps':0,'completed_steps':0,'consumed_wall_by_phase':{'train':0.,'evaluate':0.,'verify':0.},'resume_attempts':0,'max_optimizer_steps':limits['max_optimizer_steps'],'max_resume_attempts':1,'deadline_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=60)).isoformat(),'parent_ledger_head':None}
        if verification_deadline is not None:
            deadline=datetime.datetime.fromisoformat(verification_deadline)
            if deadline>datetime.datetime.fromisoformat(budget['deadline_utc']):raise ValueError('verification cannot extend original absolute deadline')
            budget['deadline_utc']=verification_deadline
        ledger=BudgetLedger(output/'ledger',budget,binding)
    else:
        ledger=BudgetLedger.resume(Path(parent['ledger_path']),parent['ledger_head'],binding,output/'ledger')
        ledger.charge_wall('verify',resume_verification_seconds)
    checkpoint_number=0;last=None;status='complete'
    def checkpoint():
        nonlocal checkpoint_number,last
        path=output/f'checkpoint-{checkpoint_number:04d}.pt';checkpoint_number+=1
        pin=gt.save_checkpoint(path,model,optimizer,cursor,binding=binding,selection=selected,budget=ledger.snapshot())
        last={'path':str(path.absolute()),'sha256':pin};return last
    while True:
        if cursor.phase=='epoch_done':
            if cursor.epoch>=limits['max_epochs'] or selected['bad_epochs']>=3:break
            cursor.advance_epoch()
        while cursor.phase=='train':
            _live(ledger.snapshot());ledger.reserve_step();begin=time.perf_counter()
            indices=cursor.next_batch(512);result=gt.train_effective_batch(model,optimizer,context.examples.rows(indices),context.codec,microbatch=limits['microbatch'],device='cpu')
            ledger.complete_step(time.perf_counter()-begin);losses.append({'epoch':cursor.epoch,'examples':result.examples,'loss_mean':result.loss_mean})
            if (stop_after_step is not None and ledger.snapshot()['completed_steps']>=stop_after_step) or (should_stop is not None and should_stop()):
                checkpoint();status='interrupted';break
        if status=='interrupted':break
        if cursor.phase=='evaluate':
            _live(ledger.snapshot());candidate=pending['checkpoint'] if pending is not None else checkpoint();begin=time.perf_counter()
            interrupted=False;eval_output=output/f'evaluation-epoch-{cursor.epoch:02d}'
            try:
                evaluation=ev.evaluate_queries(model,context,output=eval_output,checkpoint_sha256=candidate['sha256'],shard_size=evaluation_shard_size,
                    resume_manifest=Path(pending['manifest_path']) if pending is not None else None,resume_sha256=pending['manifest_sha256'] if pending is not None else None,
                    stop_after_shards=stop_after_evaluation_shards,should_stop=should_stop,budget_check=lambda:_live(ledger.snapshot()))
            except (KeyboardInterrupt,InterruptedError):
                interrupted=True;evaluation=_recover_evaluation(eval_output,model,context,candidate['sha256'],evaluation_shard_size)
                if evaluation is None:
                    ledger.charge_wall('evaluate',time.perf_counter()-begin);checkpoint();status='interrupted';break
            ledger.charge_wall('evaluate',time.perf_counter()-begin);_live(ledger.snapshot())
            if not evaluation['manifest']['complete']:
                pending={'manifest_path':str(evaluation['manifest_path'].absolute()),'manifest_sha256':evaluation['manifest_sha256'],'checkpoint':candidate}
                checkpoint();status='interrupted';break
            pending=None
            metric=evaluation['aggregate']['metrics']['all_eligible'];pair=[metric['recall@50'],metric['ndcg@50']]
            improved=selected['best_metric'] is None or tuple(pair)>tuple(selected['best_metric'])
            selected={'epoch':cursor.epoch,'best_epoch':cursor.epoch if improved else selected['best_epoch'],'best_metric':pair if improved else selected['best_metric'],'bad_epochs':0 if improved else selected['bad_epochs']+1}
            epochs.append({'epoch':cursor.epoch,'metrics':evaluation['aggregate'],'model_sha256':ev.model_digest(model)})
            evaluations.append({'epoch':cursor.epoch,'manifest_path':str(evaluation['manifest_path'].absolute()),'manifest_sha256':evaluation['manifest_sha256'],'checkpoint':candidate})
            cursor.complete_evaluation();checkpoint()
            if improved:best={**last,'epoch':cursor.epoch,'metric':pair}
            if (interrupted or (should_stop is not None and should_stop())) and cursor.epoch<limits['max_epochs']:
                status='interrupted';break
    if last is None:checkpoint()
    manifest={'schema_version':1,'task_id':'G2','kind':'synthetic_run','status':status,'binding':binding,'selection':selected,'budget':ledger.snapshot(),'ledger_path':str(ledger.root.absolute()),'ledger_head':ledger.head,
              'last':last,'best':best,'epoch_metrics':epochs,'evaluations':evaluations,'losses':losses,'state_sha256':_state_digest(model,optimizer,cursor,selected),
              'parent':None if parent is None else {'manifest_path':str(Path(parent_manifest).absolute()),'manifest_sha256':parent_sha256},'pending_evaluation':pending}
    path=output/'manifest.json';pin=_write_json(path,manifest)
    return {'manifest':manifest,'manifest_path':path,'manifest_sha256':pin}


def _synthetic_context_for_binding(root,binding,output):
    """Re-derive actual synthetic inputs, then compare the external run binding."""
    if type(binding) is not dict or binding.get('synthetic') is not True or binding.get('task_id')!='G2':raise ValueError('real/test stage gate closed')
    context=prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json',binding['g1_manifest_sha256'],
                              root/'tests/fixtures/g2_synthetic.json',binding['fixture_sha256'],output)
    expected={**context.binding,'effective_config':binding['effective_config'],'runtime_limits':binding['runtime_limits'],'evaluation_shard_size':binding['evaluation_shard_size']}
    if canonical_bytes(expected)!=canonical_bytes(binding):raise ValueError('actual synthetic inputs/source/runtime binding mismatch')
    return context


class OperationJournal:
    """Append-only phase costs anchored to one external run manifest.

    A process lock gives evaluate/verify one budget owner. A crashed begin is
    charged through the next observation before another operation may start.
    The original absolute deadline never changes; training files stay read-only.
    """
    def __init__(self,manifest_path,manifest_sha,budget,phase,*,ledger_path,expected_head=None):
        if phase not in ('evaluate','verify'):raise ValueError('operation phase')
        _live(budget)
        # Copies of a run manifest must share its producer ledger's operation
        # history instead of opening a fresh wall-cost account at the copy path.
        self.root=_no_symlink(Path(ledger_path).parent/'operations');self.pin=manifest_sha
        self.base=copy.deepcopy(budget);self.phase=phase;self._lock=None;self.head=None;self.index=0
        self.expected_head=expected_head

    def _append(self,action,budget,phase,started,elapsed=0.,status=None):
        value={'schema_version':1,'index':self.index,'run_manifest_sha256':self.pin,'previous_head':self.head,
               'action':action,'budget':copy.deepcopy(budget),'phase':phase,'started_utc':started,
               'elapsed_seconds':elapsed,'status':status}
        self.head=_write_json(self.root/f'operation-{self.index:06d}.json',value);self.index+=1;self.budget=copy.deepcopy(budget);self.last=value

    def __enter__(self):
        import fcntl,time
        self.root.mkdir(mode=0o700,exist_ok=True);lock_path=_no_symlink(self.root/'lock')
        self._lock=lock_path.open('a+b')
        try:
            fcntl.flock(self._lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            if lock_path.stat().st_size:raise ValueError('operation lock file identity')
            marker_path=_no_symlink(self.root/'started.json')
            marker={'schema_version':1,'run_manifest_sha256':self.pin,'genesis_budget_sha256':_digest(self.base)}
            started=marker_path.exists()
            if started:
                if canonical_bytes(_json(marker_path))!=canonical_bytes(marker):raise ValueError('operation genesis identity')
                if self.expected_head is None:raise ValueError('external operation head required after first operation')
            self.budget=copy.deepcopy(self.base);self.last=None
            for path in sorted(self.root.iterdir()):
                if path.name in ('lock','started.json'):continue
                if path.name!=f'operation-{self.index:06d}.json':raise ValueError('operation chain member/gap')
                row=_json(_no_symlink(path))
                keys={'schema_version','index','run_manifest_sha256','previous_head','action','budget','phase','started_utc','elapsed_seconds','status'}
                if type(row) is not dict or set(row)!=keys or row['schema_version']!=1 or type(row['schema_version']) is not int or type(row['index']) is not int or row['index']!=self.index or row['run_manifest_sha256']!=self.pin or row['previous_head']!=self.head or row['phase'] not in ('evaluate','verify') or type(row['elapsed_seconds']) not in (int,float) or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<0:raise ValueError('operation chain schema/identity')
                stamp=datetime.datetime.fromisoformat(row['started_utc'])
                if stamp.tzinfo is None:raise ValueError('aware operation start required')
                expected=copy.deepcopy(self.budget)
                if row['action']=='begin':
                    if self.last is not None and self.last['action']!='complete' or row['elapsed_seconds']!=0 or row['status'] is not None:raise ValueError('operation begin ordering')
                elif row['action']=='complete':
                    if self.last is None or self.last['action']!='begin' or row['phase']!=self.last['phase'] or row['started_utc']!=self.last['started_utc'] or row['status'] not in ('complete','failed','abandoned'):raise ValueError('operation complete ordering')
                    expected['consumed_wall_by_phase'][row['phase']]+=row['elapsed_seconds']
                else:raise ValueError('operation action')
                _budget(row['budget'])
                if canonical_bytes(expected)!=canonical_bytes(row['budget']):raise ValueError('operation cumulative budget transition')
                self.head=file_sha256(path);self.index+=1;self.budget=copy.deepcopy(row['budget']);self.last=row
            if self.head!=self.expected_head:raise ValueError('external operation head mismatch or truncated tail')
            if not started:
                if self.index:raise ValueError('operation genesis marker missing')
                _write_json(marker_path,marker)
            if self.last is not None and self.last['action']=='begin':
                elapsed=max(0.,(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(self.last['started_utc'])).total_seconds())
                b=copy.deepcopy(self.budget);b['consumed_wall_by_phase'][self.last['phase']]+=elapsed
                self._append('complete',b,self.last['phase'],self.last['started_utc'],elapsed,'abandoned')
            _live(self.budget);self._begin=time.perf_counter();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._append('begin',self.budget,self.phase,started);return self
        except BaseException:
            self._lock.close();self._lock=None;raise

    def check(self):_live(self.budget)

    def __exit__(self,kind,value,tb):
        import time
        try:
            elapsed=time.perf_counter()-self._begin;b=copy.deepcopy(self.budget);b['consumed_wall_by_phase'][self.phase]+=elapsed
            self._append('complete',b,self.phase,self.last['started_utc'],elapsed,'complete' if kind is None else 'failed')
            if value is not None:
                try:value.operation_head=self.head
                except (AttributeError,TypeError):pass
        finally:self._lock.close();self._lock=None
        return False


def load_prepared_context(root,path,pin,examples_output):
    from . import generative_data as gd
    value=gd._json(gd._checked_file(path,pin))
    keys={'schema_version','task_id','kind','fixture_path','contract_path','binding','statistics_binding','statistics_manifest_sha256'}
    if type(value) is not dict or set(value)!=keys or type(value['schema_version']) is not int or value['schema_version']!=1 or value['task_id']!='G2' or value['kind']!='synthetic_prepared' or value['binding'].get('synthetic') is not True:raise ValueError('synthetic prepared manifest schema')
    binding=value['binding']
    context=prepare_context(root,Path(value['contract_path']),binding['g1_manifest_sha256'],fixture_path=Path(value['fixture_path']),fixture_sha256=binding['fixture_sha256'],synthetic=True,
                            stats_manifest_path=Path(path).parent/'slice-statistics-manifest.json',stats_manifest_sha256=value['statistics_manifest_sha256'],expected_stats_binding=value['statistics_binding'],examples_path=examples_output)
    if canonical_bytes(context.binding)!=canonical_bytes(binding):raise ValueError('prepared actual source/input identity mismatch')
    return context


def _validate_run(root,path,pin,context):
    from . import generative_data as gd,generative_training as gt,generative_evaluation as ev
    import random,numpy as np,torch
    value=gd._json(gd._checked_file(path,pin))
    keys={'schema_version','task_id','kind','status','binding','selection','budget','ledger_path','ledger_head','last','best','epoch_metrics','evaluations','losses','state_sha256','parent','pending_evaluation'}
    if type(value) is not dict or set(value)!=keys or type(value['schema_version']) is not int or value['schema_version']!=1 or value['task_id']!='G2' or value['kind']!='synthetic_run' or value['status'] not in ('complete','interrupted'):raise ValueError('run manifest schema/status')
    binding=value['binding'];config=binding['effective_config']
    expected={**build_binding(root,context,config),'runtime_limits':binding['runtime_limits'],'evaluation_shard_size':binding['evaluation_shard_size']}
    if canonical_bytes(binding)!=canonical_bytes(expected):raise ValueError('run current input/source/config identity')
    ledger=BudgetLedger.read_existing(Path(value['ledger_path']),value['ledger_head'],binding)
    budget=copy.deepcopy(ledger['budget']);budget['parent_ledger_head']=value['ledger_head']
    if canonical_bytes(budget)!=canonical_bytes(value['budget']) or ledger['pending']:raise ValueError('run final budget ledger mismatch')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    model=gt.make_model(context.codec,config,device='cpu');optimizer=gt.make_optimizer(model,config);cursor=gt.EpochCursor(config['seed'],len(context.examples))
    last=value['last']
    if type(last) is not dict or set(last)!={'path','sha256'}:raise ValueError('last checkpoint receipt')
    state=gt.load_checkpoint(Path(last['path']),last['sha256'],model,optimizer,cursor,expected_binding=binding)
    last_model_sha=ev.model_digest(model);last_epoch=cursor.epoch;last_phase=cursor.phase
    if canonical_bytes(state.budget)!=canonical_bytes(value['budget']) or canonical_bytes(state.selection)!=canonical_bytes(value['selection']) or _state_digest(model,optimizer,cursor,state.selection)!=value['state_sha256']:raise ValueError('actual last checkpoint state/selection/budget')
    if value['status']=='complete' and (cursor.phase!='epoch_done' or cursor.epoch!=2):raise ValueError('complete epoch/full validation gate')
    epochs=value['epoch_metrics'];evaluations=value['evaluations']
    if type(epochs) is not list or type(evaluations) is not list or len(epochs)!=state.selection['epoch'] or len(evaluations)!=len(epochs):raise ValueError('epoch evaluation coverage')
    selection={'epoch':0,'best_epoch':None,'best_metric':None,'bad_epochs':0}
    for epoch,(record,receipt) in enumerate(zip(epochs,evaluations),1):
        if type(record) is not dict or set(record)!={'epoch','metrics','model_sha256'} or record['epoch']!=epoch or type(receipt) is not dict or set(receipt)!={'epoch','manifest_path','manifest_sha256','checkpoint'} or receipt['epoch']!=epoch:raise ValueError('epoch evidence schema/order')
        checkpoint=receipt['checkpoint'];ck=gt.load_checkpoint(Path(checkpoint['path']),checkpoint['sha256'],model,optimizer,cursor,expected_binding=binding)
        if cursor.epoch!=epoch or cursor.phase!='evaluate' or ck.selection['epoch']!=epoch-1 or ev.model_digest(model)!=record['model_sha256']:raise ValueError('actual evaluation checkpoint phase/model')
        eval_binding=ev._evaluation_binding(model,context,checkpoint['sha256'],100,binding['evaluation_shard_size'])
        evaluation,rows=ev._read_evaluation(Path(receipt['manifest_path']),receipt['manifest_sha256'],eval_binding,context.queries,context.catalog)
        aggregate=ev.aggregate_items([r['items'] for r in rows],context.queries,context.catalog,slice_definition=context.slice_definition)
        if not evaluation['complete'] or canonical_bytes(aggregate)!=canonical_bytes(record['metrics']) or canonical_bytes(aggregate)!=canonical_bytes(evaluation['aggregate']) or evaluation['computation_sha256']!=ev._computation(rows,aggregate):raise ValueError('actual full query item metrics/computation')
        main=aggregate['metrics']['all_eligible'];pair=[main['recall@50'],main['ndcg@50']]
        improved=selection['best_metric'] is None or tuple(pair)>tuple(selection['best_metric'])
        selection={'epoch':epoch,'best_epoch':epoch if improved else selection['best_epoch'],'best_metric':pair if improved else selection['best_metric'],'bad_epochs':0 if improved else selection['bad_epochs']+1}
    if canonical_bytes(selection)!=canonical_bytes(value['selection']):raise ValueError('independent selection replay/tie gate')
    pending=value['pending_evaluation']
    if pending is not None:
        if value['status']!='interrupted' or type(pending) is not dict or set(pending)!={'manifest_path','manifest_sha256','checkpoint'}:raise ValueError('pending evaluation receipt schema')
        ck=pending['checkpoint'];gt.load_checkpoint(Path(ck['path']),ck['sha256'],model,optimizer,cursor,expected_binding=binding)
        if cursor.phase!='evaluate' or cursor.epoch!=selection['epoch']+1 or last_phase!='evaluate' or last_epoch!=cursor.epoch or ev.model_digest(model)!=last_model_sha:raise ValueError('pending validation checkpoint phase/model')
        eb=ev._evaluation_binding(model,context,ck['sha256'],100,binding['evaluation_shard_size'])
        partial,rows=ev._read_evaluation(Path(pending['manifest_path']),pending['manifest_sha256'],eb,context.queries,context.catalog)
        if partial['complete'] or partial['aggregate'] is not None or partial['computation_sha256']!=ev._computation(rows,None):raise ValueError('pending query evidence')
    if selection['epoch']==0:
        if value['best'] is not None:raise ValueError('best without complete validation')
    else:
        best=value['best']
        if type(best) is not dict or set(best)!={'path','sha256','epoch','metric'} or best['epoch']!=selection['best_epoch'] or canonical_bytes(best['metric'])!=canonical_bytes(selection['best_metric']):raise ValueError('best epoch/metric receipt')
        best_state=gt.load_checkpoint(Path(best['path']),best['sha256'],model,optimizer,cursor,expected_binding=binding)
        if cursor.phase!='epoch_done' or cursor.epoch!=best['epoch'] or best_state.selection['best_epoch']!=best['epoch'] or ev.model_digest(model)!=epochs[best['epoch']-1]['model_sha256']:raise ValueError('actual best checkpoint identity')
    if value['parent'] is not None:
        parent=value['parent']
        if type(parent) is not dict or set(parent)!={'manifest_path','manifest_sha256'}:raise ValueError('parent receipt schema')
        original=gd._json(gd._checked_file(Path(parent['manifest_path']),parent['manifest_sha256']))
        if original['binding']!=binding or original['budget']['deadline_utc']!=value['budget']['deadline_utc'] or value['budget']['resume_attempts']!=original['budget']['resume_attempts']+1:raise ValueError('parent binding/deadline/attempt lineage')
    return value


def verify_run(root,manifest_path,expected_sha256,*,replay,expected_operation_head=None):
    """Structural verification plus actual model training/decode replay on request."""
    import tempfile,time
    from . import generative_data as gd
    root=Path(root);value=gd._json(gd._checked_file(manifest_path,expected_sha256))
    if value.get('kind')!='synthetic_run' or value.get('binding',{}).get('synthetic') is not True:raise ValueError('G2 real/test verification gate closed')
    # Standalone operations own completed runs only. An interrupted run is
    # verified inside resume, where its cost enters the child's inherited ledger.
    if value.get('status')!='complete':raise ValueError('standalone verify requires complete run; use resume for interrupted state')
    journal=OperationJournal(manifest_path,expected_sha256,value['budget'],'verify',ledger_path=value['ledger_path'],expected_head=expected_operation_head)
    with journal,tempfile.TemporaryDirectory(dir='/private/tmp') as d:
        output=Path(d);context=_synthetic_context_for_binding(root,value['binding'],output/'prepared')
        value=_validate_run(root,manifest_path,expected_sha256,context)
        if replay:
            if value['status']!='complete':raise ValueError('full replay requires complete run')
            _live(value['budget'])
            fresh=run_train(context,value['binding']['effective_config'],output=output/'replay',limits=value['binding']['runtime_limits'],verification_deadline=value['budget']['deadline_utc'],evaluation_shard_size=value['binding']['evaluation_shard_size'])['manifest']
            for key in ('state_sha256','selection','epoch_metrics','losses'):
                if canonical_bytes(fresh[key])!=canonical_bytes(value[key]):raise ValueError(f'actual training/decode replay mismatch: {key}')
            for actual,original in zip(fresh['evaluations'],value['evaluations']):
                a=gd._json(gd._checked_file(Path(actual['manifest_path']),actual['manifest_sha256']))
                b=gd._json(gd._checked_file(Path(original['manifest_path']),original['manifest_sha256']))
                if a['computation_sha256']!=b['computation_sha256']:raise ValueError('actual per-query prediction/audit replay mismatch')
            _live(value['budget'])
        journal.check()
    return {'status':'verified','replay':bool(replay),'manifest_sha256':expected_sha256,'state_sha256':value['state_sha256'],'budget':journal.budget,'operation_head':journal.head,
            'boundary':'CPU synthetic engineering replay; no real data or model quality claim'}


def select_runs(manifest_paths,expected_sha256s,*,output,expected_operation_heads=None):
    from . import generative_data as gd
    if len(manifest_paths)!=len(expected_sha256s) or not 1<=len(manifest_paths)<=4:raise ValueError('paired bounded synthetic run pins')
    root=Path(__file__).resolve().parents[2];runs=[];seen=set()
    heads=[None]*len(manifest_paths) if expected_operation_heads is None else expected_operation_heads
    if len(heads)!=len(manifest_paths):raise ValueError('paired external operation heads')
    order=[(lr,d) for lr in (.001,.0003) for d in (64,128)]
    operation_heads=[]
    for path,pin,head in zip(manifest_paths,expected_sha256s,heads):
        verification=verify_run(root,path,pin,replay=False,expected_operation_head=head);operation_heads.append(verification['operation_head'])
        value=gd._json(gd._checked_file(path,pin));config=value['binding']['effective_config'];identity=(config['learning_rate'],config['d_model'])
        if value['status']!='complete' or config['seed']!=42 or identity in seen:raise ValueError('complete seed42 unique configuration required')
        seen.add(identity);runs.append({'manifest_path':str(Path(path).absolute()),'manifest_sha256':pin,'config_index':order.index(identity),'metric':value['selection']['best_metric'],'best_epoch':value['selection']['best_epoch']})
    if len({_digest(gd._json(gd._checked_file(path,pin))['binding']['consumer']) for path,pin in zip(manifest_paths,expected_sha256s)})!=1:raise ValueError('selection common source identity')
    runs.sort(key=lambda r:r['config_index']);winner=max(runs,key=lambda r:tuple(r['metric']))
    output=_no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    manifest={'schema_version':1,'task_id':'G2','kind':'synthetic_selection','runs':runs,'selected':winner,'boundary':'synthetic selection oracle; no core winner'}
    path=output/'manifest.json';pin=_write_json(path,manifest)
    return {'manifest':manifest,'manifest_path':path,'manifest_sha256':pin,'operation_heads':operation_heads}


def _config(index,seed):
    if index not in range(4) or seed not in (42,2024,2025):raise ValueError('frozen core configuration/seed')
    lr,d=[(lr,d) for lr in (.001,.0003) for d in (64,128)][index]
    return {'d_model':d,'heads':4,'encoder_layers':1,'decoder_layers':1,'ff':4*d,'dropout':0.,'precision':'fp32','learning_rate':lr,'weight_decay':.0001,'gradient_clip_norm':1.,'effective_batch':512,'seed':seed}


@contextmanager
def _signal_stop(enabled):
    """Handlers only set a flag; writers finish the current safe boundary."""
    import signal,threading
    stopped=[False];previous={}
    if enabled:
        if threading.current_thread() is not threading.main_thread():raise ValueError('training CLI requires main-thread signal ownership')
        def handler(signum,frame):stopped[0]=True
        for signum in (signal.SIGINT,signal.SIGTERM):
            previous[signum]=signal.getsignal(signum);signal.signal(signum,handler)
    try:yield lambda:stopped[0]
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)


def main(argv=None):
    import argparse,json,tempfile,sys
    from . import generative_data as gd,generative_training as gt,generative_evaluation as ev
    parser=argparse.ArgumentParser(description='G2 bounded synthetic engineering; real/test gates closed')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','train','resume','evaluate','select'):
        sub=commands.add_parser(name);sub.add_argument('--synthetic',action='store_true',required=True);sub.add_argument('--output',type=Path,required=True)
        if name=='prepare':
            for flag in ('fixture','fixture-sha','contract','contract-sha'):sub.add_argument('--'+flag,required=True)
        elif name=='train':
            sub.add_argument('--prepared',type=Path,required=True);sub.add_argument('--prepared-sha',required=True)
            for flag,default in (('config-index',0),('seed',42),('microbatch',2),('max-epochs',2),('max-wall-seconds',60)):sub.add_argument('--'+flag,type=int,default=default)
            sub.add_argument('--stop-after-step',type=int)
            sub.add_argument('--shard-size',type=int,default=64);sub.add_argument('--stop-after-evaluation-shards',type=int)
        elif name=='resume':
            for flag in ('checkpoint','checkpoint-sha','parent-manifest','parent-sha'):sub.add_argument('--'+flag,required=True)
        else:
            sub.add_argument('--run-manifest',type=Path,required=True,action='append' if name=='select' else 'store');sub.add_argument('--run-sha',required=True,action='append' if name=='select' else 'store')
            sub.add_argument('--operation-head',action='append' if name=='select' else 'store')
            if name=='evaluate':sub.add_argument('--shard-size',type=int,default=64)
    verify=commands.add_parser('verify');verify.add_argument('--manifest',type=Path,required=True);verify.add_argument('--sha256',required=True);verify.add_argument('--replay',action='store_true');verify.add_argument('--operation-head')
    args=parser.parse_args(argv);root=Path(__file__).resolve().parents[2]
    try:
        with _signal_stop(args.command in ('train','resume')) as should_stop,tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            temporary=Path(d)
            if args.command=='prepare':
                prepare_synthetic(root,Path(args.contract),args.contract_sha,Path(args.fixture),args.fixture_sha,args.output)
                result={'manifest_path':args.output/'manifest.json','manifest_sha256':file_sha256(args.output/'manifest.json')}
            elif args.command=='train':
                context=load_prepared_context(root,args.prepared,args.prepared_sha,temporary/'examples.jsonl');limits={'max_epochs':args.max_epochs,'max_wall_seconds':args.max_wall_seconds,'max_resume_attempts':1,'microbatch':args.microbatch,'max_optimizer_steps':args.max_epochs*math.ceil(len(context.examples)/512)}
                result=run_train(context,_config(args.config_index,args.seed),output=args.output,limits=limits,stop_after_step=args.stop_after_step,evaluation_shard_size=args.shard_size,stop_after_evaluation_shards=args.stop_after_evaluation_shards,should_stop=should_stop)
            elif args.command=='resume':
                parent=gd._json(gd._checked_file(Path(args.parent_manifest),args.parent_sha));context=_synthetic_context_for_binding(root,parent['binding'],temporary/'prepared')
                result=run_train(context,parent['binding']['effective_config'],output=args.output,limits=parent['binding']['runtime_limits'],resume_path=Path(args.checkpoint),resume_sha256=args.checkpoint_sha,parent_manifest=Path(args.parent_manifest),parent_sha256=args.parent_sha,evaluation_shard_size=parent['binding']['evaluation_shard_size'],should_stop=should_stop)
            elif args.command=='evaluate':
                run=gd._json(gd._checked_file(args.run_manifest,args.run_sha));context=_synthetic_context_for_binding(root,run['binding'],temporary/'prepared');_validate_run(root,args.run_manifest,args.run_sha,context)
                if run['status']!='complete':raise ValueError('evaluation requires complete selected run')
                config=run['binding']['effective_config'];model=gt.make_model(context.codec,config,device='cpu');optimizer=gt.make_optimizer(model,config);cursor=gt.EpochCursor(config['seed'],len(context.examples));best=run['best']
                gt.load_checkpoint(Path(best['path']),best['sha256'],model,optimizer,cursor,expected_binding=run['binding'])
                with OperationJournal(args.run_manifest,args.run_sha,run['budget'],'evaluate',ledger_path=run['ledger_path'],expected_head=args.operation_head) as journal:
                    result=ev.evaluate_queries(model,context,output=args.output,checkpoint_sha256=best['sha256'],shard_size=args.shard_size,budget_check=journal.check)
                    journal.check()
                result['budget']=journal.budget;result['operation_head']=journal.head
            elif args.command=='verify':result=verify_run(root,args.manifest,args.sha256,replay=args.replay,expected_operation_head=args.operation_head)
            else:result=select_runs(args.run_manifest,args.run_sha,output=args.output,expected_operation_heads=args.operation_head)
        receipt={k:(str(v.absolute()) if isinstance(v,Path) else v) for k,v in result.items() if k in ('manifest_path','manifest_sha256','status','replay','state_sha256','budget','operation_head','operation_heads','boundary')}
        print(json.dumps(receipt,sort_keys=True));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'G2 gate failed: {exc}',file=sys.stderr)
        if hasattr(exc,'operation_head'):print(json.dumps({'status':'failed','operation_head':exc.operation_head}),file=sys.stderr)
        return 1


if __name__=='__main__':
    # -m executes this file as __main__; use the canonical module's factory so
    # evaluation sees the same RunContext class and seal as CLI orchestration.
    from .generative_experiment import main as canonical_main
    raise SystemExit(canonical_main())
