"""P6 static artifact consumer and bounded train/resume/decode runner."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,platform,resource,time
import torch
from .manifest import read_json,source_fingerprint
from .data_audit import canonical_bytes,file_sha256,validate_frozen_manifest,producer_identity
from .baseline_run import FROZEN_SHA,read_checked
from .sid_run import load_static as load_sid,P4_MANIFEST_SHA,sample_headroom,check_headroom
from .tiger import SIDTokens,TigerModel,decode_history
from .training import seed_all,ShuffledCursor,train_step,save_checkpoint,load_checkpoint,stop_reason

P5_SHA='669029504ab18df5f6a6109820d3d4a36a243f6ab0d3e8c46fc8ce17b1b6bbe9'
CONFIG={'task_id':'P6','seed':42,'d_model':128,'nhead':4,'encoder_layers':2,'decoder_layers':2,'ff':512,'dropout':0.0,
        'batch_size':64,'lr':0.001,'weight_decay':0.01,'clip':1.0,'max_steps':1000,'checkpoint_step':100,
        'max_train_seconds':2700,'beam':20,'history_items':50,'ks':[10,20],
        'train_deadline_utc':'2026-10-07T02:16:21+00:00','global_deadline_utc':'2026-10-07T02:46:21+00:00',
        'beam_normalization':'full_vocab_before_constraint','eos':False}


def validate_config(value):
    if type(value) is not dict or set(value)!=set(CONFIG) or any(type(value[k]) is not type(v) or value[k]!=v for k,v in CONFIG.items()): raise ValueError('unapproved P6 config')
    return value


def verify_inputs(records):
    for r in records:
        p=Path(r['path'])
        if p.is_symlink() or file_sha256(p)!=r['sha256']: raise ValueError('input hash mismatch')


def validate_examples(rows,catalog):
    allowed=set(map(int,catalog))
    if not isinstance(rows,list) or not rows: raise ValueError('empty examples')
    for r in rows:
        if not 1<=len(r['history'])<=50 or len(r['history_positions'])!=len(r['history']): raise ValueError('invalid history')
        positions=r['history_positions'];target=r['target_position']
        if any(type(i) is not int or i not in allowed for i in r['history']+[r['target']]): raise ValueError('training catalog/OOV mismatch')
        if len(target)!=2 or any(type(x) is not int or x<0 for x in target) or target[0]>24 or any(len(p)!=2 or any(type(x) is not int or x<0 for x in p) or tuple(p)>=tuple(target) for p in positions) or any(tuple(a)>=tuple(b) for a,b in zip(positions,positions[1:])): raise ValueError('training temporal violation')


def publish_success(path,stage,splits,extra=None):
    if stage.get('resume_from_step')!=100 or stage.get('step',0)<101 or stage.get('status') not in ('step_limit','time_limit','deadline'): raise ValueError('missing actual resume evidence')
    if stage['status']=='step_limit' and stage['step']!=1000: raise ValueError('step target mismatch')
    if set(splits)!={'validation','test'} or any(splits[s]['queries']!=512 for s in splits): raise ValueError('fixed audit incomplete')
    value={'schema_version':1,'task_id':'P6','status':'success','training_completion':'complete_1000_steps' if stage['step']==1000 else 'partial_time_bound',
           'training':stage,'splits':splits,**(extra or {})}
    with Path(path).open('xb') as stream: stream.write(canonical_bytes(value))
    return value


def build_model(codec,config,device):
    model=TigerModel(codec.vocab_size,config['d_model'],config['nhead'],config['encoder_layers'],config['decoder_layers'],config['ff']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    return model,optimizer


def train_session(output,codec,rows,config,inputs,end_step,device,resume=None,on_boundary=None):
    output=Path(output);seed_all(config['seed']);model,optimizer=build_model(codec,config,device)
    cursor=ShuffledCursor(len(rows),config['seed']);binding={'config':config,'inputs':inputs,'source_fingerprint':source_fingerprint()}
    step=0;start=datetime.now(timezone.utc).isoformat();previous=0
    if resume is not None:
        checkpoint=load_checkpoint(output/resume['checkpoint_path'],resume['checkpoint_sha256'],model,optimizer,cursor,binding)
        step=checkpoint['step'];start=checkpoint['started_at_utc'];previous=step
        if step!=resume['step'] or start!=resume['started_at_utc']: raise ValueError('resume stage mismatch')
    if end_step<=step: raise ValueError('end step must advance')
    log=output/f'train-from-{step}.jsonl'
    with log.open('x') as stream:
        while step<end_step:
            if on_boundary is not None: on_boundary()
            reason=stop_reason(start,config['train_deadline_utc'],config['max_train_seconds'])
            if reason: break
            values=train_step(model,optimizer,codec,rows,cursor,config['batch_size']);step+=1
            stream.write(json.dumps({'step':step,'epoch':cursor.epoch,'cursor':cursor.cursor,**values},allow_nan=False)+'\n');stream.flush()
        else: reason='step_limit' if step==config['max_steps'] else 'interrupted'
    checkpoint_path=f'checkpoint-{step}-from-{previous}.pt'
    sha=save_checkpoint(output/checkpoint_path,model,optimizer,cursor,step,binding,start)
    return {'step':step,'resume_from_step':previous,'status':reason,'started_at_utc':start,'ended_at_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint_path':checkpoint_path,'checkpoint_sha256':sha,'training_log':log.name,'training_log_sha256':file_sha256(log),
            'epoch':cursor.epoch,'cursor':cursor.cursor,'binding':binding,'device':device,
            'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0}


def evaluate_model(model,codec,queries,config,on_boundary=None):
    from .baselines import metrics
    results={};artifacts={};device=next(model.parameters()).device
    for split in ('validation','test'):
        selected=queries[split];predictions=[];diagnostics=[];latencies=[]
        for query in selected:
            if stop_reason(datetime.now(timezone.utc).isoformat(),config['global_deadline_utc'],1e9)=='deadline': raise ValueError('decode deadline reached')
            if on_boundary is not None: on_boundary()
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter();predicted,audit=decode_history(model,codec,query['history'],config['beam'])
            if device.type=='cuda': torch.cuda.synchronize()
            latencies.append(time.perf_counter()-start);predictions.append(predicted);diagnostics.append(audit)
        results[split]={'queries':len(selected),'metrics':metrics(predictions,selected,codec.catalog.ids,config['ks']),
                        **{key:sum(a[key] for a in diagnostics) for key in ('invalid','deadend','collision')},
                        'decode_seconds':sum(latencies),'ms_per_query':1000*sum(latencies)/len(selected) if selected else None,
                        'latency_boundary':'single query tokenize+encoder+beam+item mapping; CUDA synchronized; excludes input load'}
        artifacts[split]={'queries':selected,'predictions':predictions,'diagnostics':diagnostics,'latency_seconds':latencies}
    return results,artifacts


def load_inputs(p3_path,p4_path,p5_path):
    from .baselines import sample_queries
    p3_path=Path(p3_path).resolve();p4_path=Path(p4_path).resolve();p5_path=Path(p5_path).resolve()
    frozen=validate_frozen_manifest(p3_path,FROZEN_SHA);evidence=frozen['train_evidence']
    train_path=p3_path.parent/'train-audit.json';train=read_checked(train_path,evidence['train_audit_sha256'])
    if train['producer']!=evidence['producer'] or train['policy_sha256']!=frozen['policy_sha256']: raise ValueError('P3 producer/policy mismatch')
    catalog_path=p3_path.parent/'catalog.json';catalog=read_checked(catalog_path,evidence['catalog_sha256'])
    example_path=p3_path.parent/'training-examples.json';rows=read_checked(example_path,train['training_examples_sha256'])
    if len(rows)!=train['training_examples']: raise ValueError('example denominator mismatch')
    sid=load_sid(p5_path,P5_SHA)
    if catalog!=sid.ids.astype(int).tolist(): raise ValueError('SID train catalog mismatch')
    validate_examples(rows,catalog)
    p4=read_checked(p4_path,P4_MANIFEST_SHA)
    if p4['task_id']!='P4' or p4['status']!='success' or p4['evidence']['frozen_manifest_sha256']!=FROZEN_SHA: raise ValueError('P4 input contract mismatch')
    records=[{'role':role,'path':str(path),'sha256':digest} for role,path,digest in [('p3_frozen',p3_path,FROZEN_SHA),('p3_train',train_path,evidence['train_audit_sha256']),('catalog',catalog_path,evidence['catalog_sha256']),('training_examples',example_path,train['training_examples_sha256']),('p4_manifest',p4_path,P4_MANIFEST_SHA),('p5_manifest',p5_path,P5_SHA)]]
    queries={}
    for split in ('validation','test'):
        path=p4_path.parent/f'{split}-audit.json';digest=p4['artifacts'][path.name];audit=read_checked(path,digest)
        selected=audit['queries'];queries[split]=selected
        query_path=str(Path(p4['holdout_directory'])/f'{split}-queries.json')
        matches=[r for r in p4['evidence']['inputs'] if r['path']==query_path]
        if len(matches)!=1: raise ValueError('P4 query source evidence missing')
        all_queries=read_checked(Path(query_path),matches[0]['sha256'])
        if canonical_bytes(sample_queries(all_queries,split,42,512))!=canonical_bytes(selected): raise ValueError('fixed P4 audit mismatch')
        if len(selected)!=512 or len({q['user_id'] for q in selected})!=512: raise ValueError('fixed audit denominator mismatch')
        allowed=set(catalog)
        for q in selected:
            if type(q['oov']) is not bool or q['oov']!=(q['target'] not in allowed) or not 1<=len(q['history'])<=50 or any(i not in allowed for i in q['history']): raise ValueError('query catalog/OOV mismatch')
            if any(tuple(p)>=tuple(q['target_position']) or p[0]>24 for p in q['history_positions']): raise ValueError('query temporal violation')
        records.extend([{'role':f'p4_{split}_audit','path':str(path),'sha256':digest},{'role':f'p3_{split}_queries','path':query_path,'sha256':matches[0]['sha256']}])
    p5=read_checked(p5_path,P5_SHA)
    records.extend({'role':'p5_'+name,'path':str(p5_path.parent/name),'sha256':digest} for name,digest in p5['artifacts'].items())
    verify_inputs(records)
    return SIDTokens(sid),rows,queries,records


class ResourceMonitor:
    def __init__(self,sampler):
        import threading
        self.sampler=sampler;self.samples=[];self.error=None;self.event=threading.Event();self.thread=None

    def check(self):
        if self.error is not None: raise self.error
        sample=self.sampler();check_headroom({k:sample[k] for k in ('ram_available_fraction','disk_available_fraction')})
        if sample.get('vram_available_fraction',1)<.2: raise ValueError('VRAM headroom gate failed')
        self.samples.append(sample)

    def start(self):
        import threading
        self.check()
        def loop():
            while not self.event.wait(.25):
                try:self.check()
                except Exception as e:self.error=e;return
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()

    def finish(self):
        self.event.set()
        if self.thread is not None:self.thread.join()
        self.check()
        return {'samples':len(self.samples),'minimum':{key:min(r[key] for r in self.samples) for key in self.samples[0]},'sampling_seconds':.25}


def environment():
    from importlib.metadata import version
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'dependencies':{name:version(name) for name in ('PyYAML','numpy','pyarrow','torch')},
            'gpu_name':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'lock_sha256':file_sha256(Path(__file__).resolve().parents[2]/'uv.lock')}


def write_json(path,value):
    with Path(path).open('xb') as stream:stream.write(canonical_bytes(value))


def run(mode,p3_path,p4_path,p5_path,config_path,output):
    output=Path(output).resolve();config_path=Path(config_path).resolve();monitor=None
    if mode=='train': output.mkdir(parents=True,exist_ok=False)
    elif mode not in ('resume','decode'):raise ValueError('invalid mode')
    elif not output.is_dir() or (output/'run_manifest.json').exists():raise ValueError('invalid stage output')
    try:
        config=validate_config(read_json(config_path))
        if stop_reason(datetime.now(timezone.utc).isoformat(),config['global_deadline_utc'],1e9)=='deadline':raise ValueError('global deadline reached')
        codec,rows,queries,records=load_inputs(p3_path,p4_path,p5_path)
        if not torch.cuda.is_available():raise ValueError('real P6 run requires CUDA')
        if not platform.python_version().startswith('3.11.') or torch.__version__!='2.11.0+cu128':raise ValueError('unapproved training environment')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        def sample():
            value=sample_headroom(output);free,total=torch.cuda.mem_get_info();value['vram_available_fraction']=free/total;return value
        monitor=ResourceMonitor(sample);monitor.start();fingerprint=source_fingerprint()
        binding_inputs={r['role']:r['sha256'] for r in records};binding_inputs['config_sha256']=file_sha256(config_path)
        if mode in ('train','resume'):
            previous=read_json(output/'stage-100.json') if mode=='resume' else None
            if previous is not None and (previous['step']!=100 or previous['status']!='interrupted'):raise ValueError('actual 100-step interruption required')
            stage=train_session(output,codec,rows,config,binding_inputs,100 if mode=='train' else 1000,'cuda',previous,monitor.check)
            verify_inputs(records);stage['resource']=monitor.finish();monitor=None
            stage['environment']=environment();stage['records']=records;stage['config_path']=str(config_path)
            name='stage-100.json' if mode=='train' else 'stage-final.json';write_json(output/name,stage)
            print(json.dumps({'stage':mode,'step':stage['step'],'resume_from_step':stage['resume_from_step'],'status':stage['status'],'checkpoint_sha256':stage['checkpoint_sha256']}))
            return stage
        stage=read_json(output/'stage-final.json');initial=read_json(output/'stage-100.json')
        if initial['step']!=100 or initial['status']!='interrupted' or stage['resume_from_step']!=100:raise ValueError('actual checkpoint resume required')
        seed_all(42);model,optimizer=build_model(codec,config,'cuda');cursor=ShuffledCursor(len(rows),42)
        binding={'config':config,'inputs':binding_inputs,'source_fingerprint':fingerprint}
        checkpoint=load_checkpoint(output/stage['checkpoint_path'],stage['checkpoint_sha256'],model,optimizer,cursor,binding)
        if checkpoint['step']!=stage['step'] or checkpoint['started_at_utc']!=stage['started_at_utc']:raise ValueError('final checkpoint stage mismatch')
        splits,artifacts=evaluate_model(model,codec,queries,config,monitor.check)
        for split,value in artifacts.items():write_json(output/f'{split}-audit.json',value)
        write_json(output/'metrics.json',splits);verify_inputs(records)
        resources=monitor.finish();monitor=None
        files=['stage-100.json','stage-final.json','metrics.json','validation-audit.json','test-audit.json',initial['checkpoint_path'],stage['checkpoint_path'],initial['training_log'],stage['training_log']]
        extra={'effective_config':config,'config_path':str(config_path),'config_sha256':file_sha256(config_path),'source_fingerprint':fingerprint,
               'producer':producer_identity(),'environment':environment(),'inputs':records,'resource':resources,
               'peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_unit':'KiB_on_Linux_bytes_on_macOS',
               'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),
               'artifacts':{name:file_sha256(output/name) for name in files},'decode_completed_at_utc':datetime.now(timezone.utc).isoformat(),
               'claim_boundary':'original small TIGER-style smoke; fixed audit only; no faithful TIGER reproduction, full-test, online, causal or superiority claim'}
        result=publish_success(output/'run_manifest.json',stage,splits,extra)
        print('P6 actual train/resume/fixed-audit decode completed');return result
    except Exception as e:
        if monitor is not None:
            monitor.event.set()
            if monitor.thread is not None:monitor.thread.join()
        name=output/f'failure-{mode}.json'
        if output.is_dir() and not name.exists():write_json(name,{'task_id':'P6','status':'failed','mode':mode,'error_type':type(e).__name__,'error':str(e),'utc':datetime.now(timezone.utc).isoformat()})
        if output.is_dir() and not (output/'failure.json').exists():write_json(output/'failure.json',{'status':'failed','mode':mode,'error_type':type(e).__name__})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description='P6 bounded TIGER-style smoke')
    parser.add_argument('mode',choices=['train','resume','decode','verify']);parser.add_argument('--p3-manifest');parser.add_argument('--p4-manifest');parser.add_argument('--p5-manifest');parser.add_argument('--config',default='configs/p6.json');parser.add_argument('--output');parser.add_argument('--manifest')
    args=parser.parse_args(argv)
    if args.mode=='verify':verify_run(args.manifest)
    else:
        if not all((args.p3_manifest,args.p4_manifest,args.p5_manifest,args.output)):parser.error('all pinned input manifests and output required')
        run(args.mode,args.p3_manifest,args.p4_manifest,args.p5_manifest,args.config,args.output)




def verify_run(manifest_path):
    from .baselines import metrics
    import math
    path=Path(manifest_path).resolve();value=read_json(path)
    required={'training','splits','effective_config','config_path','config_sha256','source_fingerprint','environment','inputs','resource','artifacts','training_completion'}
    if value.get('schema_version')!=1 or value.get('task_id')!='P6' or value.get('status')!='success' or not required.issubset(value):raise ValueError('invalid P6 manifest')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');seed_all(42)
    if value['source_fingerprint']!=source_fingerprint() or value['environment']!=environment():raise ValueError('P6 source/environment mismatch')
    config=validate_config(read_checked(Path(value['config_path']),value['config_sha256']))
    if config!=value['effective_config']:raise ValueError('P6 config mismatch')
    records=value['inputs'];verify_inputs(records)
    for name,digest in value['artifacts'].items():
        if Path(name).name!=name or (path.parent/name).is_symlink() or file_sha256(path.parent/name)!=digest:raise ValueError('P6 artifact hash mismatch')
    roles={r['role']:r for r in records}
    codec,rows,queries,fresh=load_inputs(roles['p3_frozen']['path'],roles['p4_manifest']['path'],roles['p5_manifest']['path'])
    if fresh!=records:raise ValueError('P6 input records mismatch')
    initial=read_json(path.parent/'stage-100.json');stage=read_json(path.parent/'stage-final.json')
    if initial['step']!=100 or initial['resume_from_step']!=0 or initial['status']!='interrupted' or stage['resume_from_step']!=100 or stage!=value['training'] or initial['started_at_utc']!=stage['started_at_utc'] or initial['binding']!=stage['binding']:raise ValueError('P6 resume evidence mismatch')
    if stage['status'] not in ('step_limit','time_limit','deadline') or not 100<stage['step']<=1000 or (stage['status']=='step_limit' and stage['step']!=1000):raise ValueError('P6 actual step/status mismatch')
    if value['training_completion']!=('complete_1000_steps' if stage['step']==1000 else 'partial_time_bound'):raise ValueError('P6 completion claim mismatch')
    for progress,low in ((initial,1),(stage,101)):
        logs=[json.loads(line) for line in (path.parent/progress['training_log']).read_text().splitlines()]
        if [r['step'] for r in logs]!=list(range(low,progress['step']+1)) or any(not math.isfinite(r['loss']) or not math.isfinite(r['gradient_norm']) or r['gradient_norm']<0 for r in logs):raise ValueError('P6 finite training log/step mismatch')
        check_headroom({k:progress['resource']['minimum'][k] for k in ('ram_available_fraction','disk_available_fraction')})
        if progress['resource']['minimum']['vram_available_fraction']<.2:raise ValueError('P6 training VRAM gate mismatch')
    binding_inputs={r['role']:r['sha256'] for r in records};binding_inputs['config_sha256']=value['config_sha256']
    binding={'config':config,'inputs':binding_inputs,'source_fingerprint':source_fingerprint()}
    seed_all(42);model,optimizer=build_model(codec,config,'cuda');cursor=ShuffledCursor(len(rows),42)
    checkpoint=load_checkpoint(path.parent/stage['checkpoint_path'],stage['checkpoint_sha256'],model,optimizer,cursor,binding)
    if checkpoint['step']!=stage['step']:raise ValueError('P6 checkpoint actual step mismatch')
    replay,artifacts=evaluate_model(model,codec,queries,config)
    for split in ('validation','test'):
        stored=read_json(path.parent/f'{split}-audit.json')
        if stored['queries']!=queries[split] or stored['predictions']!=artifacts[split]['predictions'] or stored['diagnostics']!=artifacts[split]['diagnostics']:raise ValueError('P6 query/prediction/constraint replay mismatch')
        if len(stored['queries'])!=512 or replay[split]['metrics']!=value['splits'][split]['metrics'] or metrics(stored['predictions'],stored['queries'],codec.catalog.ids,config['ks'])!=replay[split]['metrics']:raise ValueError('P6 item metrics mismatch')
        if any(value['splits'][split][k] for k in ('invalid','deadend','collision')):raise ValueError('P6 decode gate mismatch')
    print('P6 source/input/checkpoint/actual-step/query/prediction/item-metric replay verified')
    return value


if __name__=='__main__':main()
