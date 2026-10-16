"""R2 pinned preflight/CUDA proof/dry-run/3-seed publication and finite verify driver.

All locations and existing scheduler integration are supplied in a locally private
request JSON. No SSH, deployment, deletion, retry, checkpoint reuse or model calls.
"""
import argparse
import copy
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
ENTRY = 'vkgr.ranking_r2_run'
SEEDS = (42, 2024, 2025)
READY_TIMEOUT_SECONDS = 30.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def flag(argv, name):
    if argv.count(name) != 1:
        raise ValueError('exactly one argument required: ' + name)
    return argv[argv.index(name) + 1]


def replace(argv, name, value):
    flag(argv, name)
    argv[argv.index(name) + 1] = str(value)


def valid_deadline(deadline):
    return type(deadline) in (int, float) and math.isfinite(deadline) and 0 < deadline-time.time() <= 48*3600


def declared_vram(measured):
    if type(measured) not in (int, float) or not math.isfinite(measured) or measured <= 0:
        raise ValueError('positive finite measured CUDA MiB required')
    return math.ceil(measured * 1.15 / 256) * 256



def own_gpu_mib():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    values = [float(line.split(',')[1]) for line in result.stdout.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
        raise ValueError('unique finite own GPU PID measurement required')
    return values[0]


def measured_vram(memory):
    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative own-process memory evidence required')
        return value
    if type(memory.get('pid')) is not int or memory['pid'] <= 0 or not memory.get('samples'):
        raise ValueError('own PID and nonempty GPU samples required')
    baseline = number(memory['context_process_mib'])
    reserved0 = number(memory['context_reserved_mib'])
    if baseline <= 0:
        raise ValueError('measured own CUDA context baseline required')
    overhead = max(0., baseline-reserved0)
    allocator_peak = 0.
    own_peak = baseline
    for row in memory['samples']:
        process = number(row['own_process_mib'])
        reserved = number(row['reserved_mib'])
        maximum = number(row['max_reserved_mib'])
        allocated = number(row['max_allocated_mib'])
        if process <= 0 or maximum < reserved or maximum < allocated:
            raise ValueError('inconsistent process/allocator sample')
        overhead = max(overhead, process-reserved)
        allocator_peak = max(allocator_peak, maximum, allocated)
        own_peak = max(own_peak, process)
    return max(own_peak, allocator_peak+overhead)


def validate_vram(proof):
    if 'memory_measurement' not in proof:
        raise ValueError('context-inclusive own-process VRAM proof required')
    measured = measured_vram(proof['memory_measurement'])
    wanted = declared_vram(measured)
    if proof.get('measured_mib') != measured or proof.get('declared_vram_mib') != wanted:
        raise ValueError('context-inclusive VRAM measurement/declaration mismatch')
    return wanted

def identity(req):
    root = Path(req['deployment'])
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != req['source_commit'] or len(actual) != 40:
        raise ValueError('deployed commit differs from pinned source')
    if not req['files']:
        raise ValueError('explicit deployed file hashes required')
    for name, expected in req['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha(root/path) != expected:
            raise ValueError('deployed source byte mismatch: ' + name)
    required = {'src/vkgr/ranking_r2.py', 'src/vkgr/ranking_r2_run.py', 'src/vkgr/ranking_r2_data.py', 'scripts/ranking_r2_ops.py'}
    if not required <= set(req['files']):
        raise ValueError('new runtime and ops files must be pinned')
    sys.path.insert(0, str(root/'src'))
    from vkgr import ranking_r2_run as run
    if not set(run.SOURCE_MEMBERS) <= set(req['files']):
        raise ValueError('all scientific source members must be deployment-pinned')
    return run


def derive_job(req, seed, vram=6656):
    if seed not in SEEDS:
        raise ValueError('fixed three R2 seeds only')
    root, runs = Path(req['deployment']), Path(req['runs'])
    template = load(req['templates'][str(seed)])
    if [template[k] for k in ('cpu_cores', 'ram_mib', 'vram_mib', 'wall_hours', 'disk_gib', 'no_admission')] != [1,18432,6656,24.,4,False]:
        raise ValueError('original A1/T3 resources changed')
    a = template['argv']
    if a[a.index('-m')+1:a.index('-m')+3] != ['vkgr.generative_g6_run','train']:
        raise ValueError('original same-seed A1/T3 template required')
    config_path = root/f'configs/ranking_r2/g-ranking-r2-s{seed}.json'
    config = load(config_path)
    from vkgr import ranking_r2_config as contract
    contract.validate(config)
    if config['sft'] != load(template['config']) or config['seed'] != seed:
        raise ValueError('R2 SFT must equal original same-seed A1/T3')
    control_path = Path(flag(a,'--output'))/'manifest.json'
    job = copy.deepcopy(template)
    rid = f'g-ranking-r2-s{seed}'
    job.update(run_id=rid,seed=seed,cwd=str(root),config=str(config_path),vram_mib=vram,pinned_manifest=req['pinned_manifest'])
    a = job['argv']
    a[0] = req['python'];a[a.index('-m')+1] = ENTRY
    for name,value in {'--config':str(config_path.relative_to(root)),'--config-sha':sha(config_path),
                       '--output':str(runs/rid/'out'),'--scratch':str(runs/rid/'scratch')}.items():
        replace(a,name,value)
    a += ['--control-manifest',str(control_path),'--control-sha',sha(control_path)]
    job['inputs'] = list(dict.fromkeys([*job['inputs'],str(control_path)]))
    job['env'] = [e for e in job['env'] if not e.startswith(('PYTHONPATH=','VKGR_PINNED_MANIFEST=','VKGR_DECODE','VKGR_EVAL_WORKERS=','VKGR_DATA_ROOT=','VKGR_RUN_ID=','VKGR_PROGRESS_PATH='))]
    job['env'] += ['PYTHONPATH='+str(root/'src'),'VKGR_PINNED_MANIFEST='+req['pinned_manifest'],'VKGR_DECODE=batched','VKGR_DECODE_BATCH=1',
                   'VKGR_EVAL_WORKERS=1','VKGR_DATA_ROOT='+str(Path(req['data_root']).resolve(strict=True))]
    return job


def derive_verify(job, manifest_sha, runs):
    verify = copy.deepcopy(job)
    rid = job['run_id']
    manifest = str(Path(runs)/rid/'out/manifest.json')
    vrid = rid+'-verify'
    verify.update(run_id=vrid, config=None, wall_hours=2., disk_gib=1,
                  inputs=list(dict.fromkeys([*job['inputs'], manifest])))
    a = verify['argv']
    a[a.index('-m')+2] = 'verify'
    for name in ('--config', '--config-sha'):
        i = a.index(name)
        del a[i:i+2]
    i = a.index('--output')
    a[i:i] = ['--manifest', manifest, '--manifest-sha', manifest_sha]
    replace(a, '--output', Path(runs)/vrid/'out')
    replace(a, '--scratch', Path(runs)/vrid/'scratch')
    return verify


def configure(job):
    for key in list(os.environ):
        if key.startswith('VKGR_'):
            del os.environ[key]
    os.environ.update(dict(e.split('=', 1) for e in job['env']))
    import torch
    torch.set_num_threads(1)


def prepare(req):
    run = identity(req)
    job = derive_job(req,42)
    configure(job)
    a = job['argv'];config=load(job['config'])
    stage=run.contract.sft_config(config)
    scratch=Path(req['ops'])/f'preflight-scratch-{time.time_ns()}'
    context,engineering,sid=run.sft._contexts(Path(req['deployment']),Path(flag(a,'--prepared')),flag(a,'--prepared-sha'),
        Path(flag(a,'--history-manifest')),flag(a,'--history-sha'),Path(flag(a,'--sid-manifest')),flag(a,'--sid-sha'),
        Path(flag(a,'--content')),flag(a,'--content-sha'),stage,scratch)
    run.contract.validate(config,engineering)
    if engineering is not False or context.binding['synthetic'] is not False or sid.info['engineering'] is not False:
        raise ValueError('formal R2 engineering=False data path required')
    if sid.info['sid_version'] != config['sft']['sid'] or sid.info['manifest_sha256'] != flag(a,'--sid-sha'):
        raise ValueError('original T3 SID binding changed')
    runtime=run.sft._runtime(False)
    return run,context,sid,runtime,scratch


def preflight(req):
    run,context,sid,runtime,scratch=prepare(req)
    entries=[]
    for seed in SEEDS:
        job=derive_job(req,seed);config=load(job['config']);run.contract.validate(config)
        a=job['argv']
        for name in ('--prepared','--history-manifest','--sid-manifest','--content'):
            pin_name={'--history-manifest':'--history-sha','--sid-manifest':'--sid-sha'}.get(name,name+'-sha')
            if flag(a,name)!=flag(derive_job(req,42)['argv'],name) or sha(flag(a,name))!=flag(a,pin_name):
                raise ValueError('original same-input binding drift: '+name)
        run._control(flag(a,'--control-manifest'),flag(a,'--control-sha'),config)
        run.sft.gr._seed(seed)
        model=run.sft.make_model_arch_d(context.codec,run.contract.sft_config(config),device=runtime['device'])
        binding=run.input_binding(Path(req['deployment']),config,context,runtime,flag(a,'--control-sha'))
        entries.append({'job':job,'expected_input_binding':binding,'config_sha256':sha(job['config'])})
        model_shape=model.spec;del model
    return {'status':'preflight_ok','source_commit':req['source_commit'],'entries':entries,'runtime':runtime,
            'data_binding_sha256':run.engine.digest(context.binding),'sid_manifest_sha256':sid.info['manifest_sha256'],
            'model_shape':model_shape,'shape':{'vocab_size':context.codec.vocab_size,'depth':context.codec.depth,
                'items':len(context.catalog),'examples':len(context.examples),'queries':len(context.queries)},
            'test_read':False,'old_checkpoint_read':False,'scratch':str(scratch)}


def cuda_preview(req):
    import torch
    run,context,sid,runtime,scratch=prepare(req)
    config=load(derive_job(req,42)['config']);stage=run.contract.sft_config(config)
    run.sft.gr._seed(42)
    torch.cuda.init();torch.cuda.empty_cache();torch.cuda.synchronize()
    memory={'pid':os.getpid(),'context_process_mib':own_gpu_mib(),'context_reserved_mib':torch.cuda.memory_reserved()/2**20,'samples':[]}
    torch.cuda.reset_peak_memory_stats()
    actor=run.sft.make_model_arch_d(context.codec,stage,device=runtime['device'])
    def sample(name):
        torch.cuda.synchronize()
        memory['samples'].append({'stage':name,'own_process_mib':own_gpu_mib(),'reserved_mib':torch.cuda.memory_reserved()/2**20,
            'max_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'max_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
    from vkgr.sid_depth import SIDTokensLD
    tokens=SIDTokensLD(context.codec,50);ids=[int(i) for i in context.catalog]
    rows=[{'history':[ids[(i+j)%len(ids)] for j in range(50)],'target':ids[(i+50)%len(ids)],'negative':ids[(i+51)%len(ids)]} for i in range(512)]
    guard=run.sft._guard(False,scratch,runtime['device'])
    sft_optimizer=run.sft.ga.make_optimizer_arch(actor,stage)
    for index in range(2):
        guard();run.sft.train_batch_d(actor,sft_optimizer,rows,context.codec,tokens,stage,runtime['device']);sample('sft-train-'+str(index+1))
    del sft_optimizer
    reference=run.r.frozen_reference(actor);reference_sha=run.sft.ev.model_digest(reference)
    optimizer=run.engine.make_optimizer(actor,config)
    for index in range(2):
        guard();run.engine.train_step(actor,reference,optimizer,rows,tokens,microbatch=config['dpo_microbatch'],device=runtime['device']);sample('dpo-train-'+str(index+1))
    if run.sft.ev.model_digest(reference)!=reference_sha or reference.training or any(p.requires_grad or p.grad is not None for p in reference.parameters()):
        raise ValueError('DPO reference changed or has gradients')
    actor.eval();before=run.sft.sdb.decode_histories_batched_d(actor,tokens,[rows[0]['history']],100,1);sample('dpo-beam100-b1')
    actor_count=sum(p.numel() for p in actor.parameters());reference_count=sum(p.numel() for p in reference.parameters())
    buffer=io.BytesIO();torch.save(actor.state_dict(),buffer);buffer.seek(0)
    state=torch.load(buffer,map_location='cpu',weights_only=True)
    del actor,optimizer;torch.cuda.empty_cache()
    reloaded=run.sft.make_model_arch_d(context.codec,stage,device=runtime['device']);reloaded.load_state_dict(state)
    after=run.sft.sdb.decode_histories_batched_d(reloaded,tokens,[rows[0]['history']],100,1);sample('dpo-reload-beam100-b1')
    if before!=after:raise ValueError('new own engineering state reload/decode mismatch')
    measured=measured_vram(memory)
    return {'status':'cuda_engineering_verified','source_commit':req['source_commit'],'runtime':runtime,
            'config_sha256':sha(derive_job(req,42)['config']),'synthetic_grad_steps':2,'synthetic_sft_steps':2,
            'reference_unchanged':True,'optimizer_actor_only':True,'actor_parameter_count':actor_count,'reference_parameter_count':reference_count,
            'reload_decode_equal':True,'data_binding_sha256':run.engine.digest(context.binding),'sid_manifest_sha256':sid.info['manifest_sha256'],
            'model_shape':reloaded.spec,'memory_measurement':memory,'measured_mib':measured,'declared_vram_mib':declared_vram(measured),'test_read':False,'old_checkpoint_read':False}


def where(queue, rid):
    return {k: sorted((Path(queue)/k).glob('*-'+rid+'.json')) for k in ('pending', 'claimed', 'rejected')}


def exists(queue, runs, rid):
    return (Path(runs)/rid).exists() or any(where(queue, rid).values()) or any(rid in p.name for p in Path(queue).glob('*/.*.tmp'))


def publish(queue, runs, job):
    queue = Path(queue)
    with (queue/'.ranking-r2-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if exists(queue, runs, job['run_id']):
            raise FileExistsError(job['run_id'])
        name = f"{time.time_ns():020d}-{job['run_id']}.json"
        temp = queue/'pending'/('.'+name+'.tmp')
        new(temp, job)
        temp.rename(queue/'pending'/name)


def launcher_prepare(req, job):
    sys.path.insert(0, req['scheduler_ops'])
    from vkgr_ops import launcher
    return launcher.prepare(job)


def dry_run(req):
    identity(req)
    pf, proof = load(req['preflight_receipt']), load(req['cuda_receipt'])
    if pf['status'] != 'preflight_ok' or proof['status'] != 'cuda_engineering_verified' or any(x['source_commit'] != req['source_commit'] for x in (pf, proof)):
        raise ValueError('matching real preflight and CUDA proof required')
    if proof.get('reference_unchanged') is not True or proof.get('optimizer_actor_only') is not True or proof.get('synthetic_sft_steps') != 2 or type(proof.get('actor_parameter_count')) is not int or proof['actor_parameter_count'] <= 0 or proof.get('reference_parameter_count') != proof['actor_parameter_count']:
        raise ValueError('true-shape actor/reference/optimizer CUDA proof required')
    if proof['reload_decode_equal'] is not True or proof['synthetic_grad_steps'] != 2 or proof['declared_vram_mib'] != validate_vram(proof):
        raise ValueError('full-shape CUDA proof mismatch')
    if any(proof.get(k) != pf.get(k) or k not in proof or k not in pf for k in ('data_binding_sha256', 'sid_manifest_sha256', 'runtime', 'model_shape')):
        raise ValueError('preflight/CUDA binding mismatch')
    if len(pf['entries']) != 3:
        raise ValueError('all three preflight seeds required')
    if proof.get('config_sha256') != pf['entries'][0]['config_sha256']:
        raise ValueError('DPO CUDA proof config differs from seed42 preflight')
    entries = []
    for seed, entry in zip(SEEDS, pf['entries']):
        wanted = derive_job(req, seed, proof['declared_vram_mib'])
        original = dict(wanted, vram_mib=6656)
        if entry['job'] != original or entry['config_sha256'] != sha(wanted['config']):
            raise ValueError('preflight job/config differs from current deployment')
        launcher_prepare(req, wanted)
        entries.append(dict(entry, job=wanted))
    if len(entries) != 3:
        raise ValueError('all three seeds required')
    return {'status': 'dry_run_ok', 'entries': entries, 'source_commit': req['source_commit'],
            'preflight_sha256': sha(req['preflight_receipt']), 'cuda_sha256': sha(req['cuda_receipt'])}


def submit(req):
    plan = dry_run(req)
    if not valid_deadline(req['deadline_epoch']):
        raise ValueError('fixed future deadline at most 48h required')
    for entry in plan['entries']:
        if exists(req['queue'], req['runs'], entry['job']['run_id']):
            raise FileExistsError(entry['job']['run_id'])
    plan.update(deadline_epoch=req['deadline_epoch'], submitted_at=time.time())
    ops = Path(req['ops'])
    new(ops/'plan.json', plan)
    sealed_request = ops/'submitted-request.json'
    new(sealed_request, req)
    # No controller publication. A ready detached owner receives a bound release.
    return start_driver(req, sealed_request)


def bound_published(req, job):
    records = [load(path) for paths in where(req['queue'], job['run_id']).values() for path in paths]
    return len(records) == 1 and all(records[0].get(k) == v for k, v in job.items())


def publish_intents(req, plan):
    ops = Path(req['ops'])
    results = []
    stopped = False
    for index, entry in enumerate(plan['entries']):
        job = entry['job']
        result = {'run_id': job['run_id'], 'job_sha256': digest(job), 'status': 'not_attempted'}
        if not stopped:
            new(ops/f'publication-intent-{index:02d}.json', dict(result, status='attempting'))
            try:
                launcher_prepare(req, job)
                publish(req['queue'], req['runs'], job)
                result['status'] = 'published'
            except Exception as exc:
                stopped = True
                visible = bound_published(req, job)
                result.update(status='published' if visible else 'failed', queue_visible=visible, error=type(exc).__name__+': '+str(exc))
                new(ops/'publication-ALERT.json', {'status': 'ALERT', 'failed_intent': result, 'remaining': 'not attempted; no retry'})
        new(ops/f'publication-result-{index:02d}.json', result)
        results.append(result)
    new(ops/'publication-final.json', {'status': 'ALERT' if stopped else 'published_not_accepted', 'runs': results})
    return {row['run_id']: row for row in results}, stopped


def ready_handshake(req, plan, launcher_pid):
    ops = Path(req['ops'])
    if os.getppid() != 1 or os.getsid(os.getpid()) != os.getpid():
        raise ValueError('driver must be PPID1 and own detached session before ready')
    ready = {'pid': os.getpid(), 'ppid': os.getppid(), 'launcher_pid': launcher_pid,
             'plan_sha256': digest(plan), 'deadline_epoch': plan['deadline_epoch']}
    new(ops/'driver-ready.json', ready)
    until = min(plan['deadline_epoch'], time.time()+READY_TIMEOUT_SECONDS)
    while time.time() < until:
        release = ops/'publish-release.json'
        if release.exists():
            if load(release) != {'plan_sha256': digest(plan), 'ready_sha256': sha(ops/'driver-ready.json'), 'driver_pid': os.getpid()}:
                raise ValueError('controller release does not bind ready driver and sealed plan')
            return
        time.sleep(.05)
    raise TimeoutError('controller never released ready owner; zero publications')


def observe(req, entry):
    runs, queue = Path(req['runs']), Path(req['queue'])
    job = entry['job']
    rid = job['run_id']
    train_terminal, verify_terminal = runs/rid/'terminal.json', runs/(rid+'-verify')/'terminal.json'
    status = load(train_terminal)['status'] if train_terminal.exists() else None
    state = {'run_id': rid, 'train_status': status, 'verified': False, 'settled': False}
    if any(where(queue, x)['rejected'] for x in (rid, rid+'-verify')):
        reasons = {}
        for rejected_rid in (rid, rid+'-verify'):
            for rejected in where(queue, rejected_rid)['rejected']:
                for reason in (rejected.with_suffix('.reason'), Path(str(rejected)+'.reason')):
                    if reason.exists():
                        reasons[reason.name] = reason.read_text()[:4096]
        return dict(state, settled=True, status='ALERT', reason='queue rejection; no retry', rejection_reasons=reasons)
    if status is not None and status != 'succeeded':
        return dict(state, settled=True, status='ALERT', reason='training terminal failure; no retry')
    if status == 'succeeded':
        claimed = where(queue, rid)['claimed']
        if len(claimed) != 1 or any(load(claimed[0]).get(k) != v for k, v in job.items()):
            raise ValueError('claimed training job differs from submitted plan')
        manifest_path = runs/rid/'out/manifest.json'
        manifest = load(manifest_path)
        run = identity(req)
        if (manifest.get('schema'),manifest.get('kind'),manifest.get('status')) != (run.SCHEMA,run.KIND,'complete') or manifest['config'] != load(job['config']) or any(manifest['binding'].get(k) != v for k,v in entry['expected_input_binding'].items()):
            raise ValueError('training manifest binding/config mismatch')
        verify = derive_verify(job, sha(manifest_path), runs)
        if not exists(queue, runs, rid+'-verify'):
            launcher_prepare(req, verify)
            publish(queue, runs, verify)
        if verify_terminal.exists():
            candidates = where(queue, rid+'-verify')['claimed']
            if len(candidates) != 1 or any(load(candidates[0]).get(k) != v for k, v in verify.items()):
                raise ValueError('claimed verify job differs from expected')
            records = []
            for line in (runs/(rid+'-verify')/'logs/stdout.log').read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
            verified = load(verify_terminal)['status'] == 'succeeded' and any(isinstance(x, dict) and x.get('status') == 'verified' for x in records)
            state.update(verified=verified, settled=True, status='verified' if verified else 'ALERT')
    return state


def driver(req, plan_sha):
    ops = Path(req['ops'])
    if sha(ops/'plan.json') != plan_sha:
        raise ValueError('sealed driver plan hash mismatch')
    plan = load(ops/'plan.json')
    interval = plan['deadline_epoch']-plan['submitted_at']
    if plan['source_commit'] != req['source_commit'] or not math.isfinite(interval) or not 0 < interval <= 48*3600:
        raise ValueError('immutable finite driver plan required')
    launcher_pid = os.getpid()
    if os.getppid() != 1:
        child = os.fork()
        if child:
            os._exit(0)
        os.setsid()
    until = time.monotonic()+5
    while os.getppid() != 1 and time.monotonic() < until:
        time.sleep(.01)
    with (ops/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        new(ops/'driver-pid.json', {'pid': os.getpid(), 'ppid': os.getppid(), 'deadline_epoch': plan['deadline_epoch'], 'poll_seconds': 900})
        try:
            identity(req)
            ready_handshake(req, plan, launcher_pid)
            identity(req)
        except Exception as exc:
            result = {'status': 'ALERT', 'reason': 'startup gate: '+type(exc).__name__+': '+str(exc), 'published_intents': 0}
            new(ops/'summary-final.json', result)
            return result
        publications, publication_failed = publish_intents(req, plan)
        while time.time() < plan['deadline_epoch']:
            identity(req)
            states = []
            for entry in plan['entries']:
                rid = entry['job']['run_id']
                if publications[rid]['status'] != 'published':
                    states.append({'run_id': rid, 'settled': True, 'verified': False, 'status': 'ALERT', 'reason': 'unpublished intent; no retry'})
                    continue
                try:
                    states.append(observe(req, entry))
                except Exception as exc:
                    states.append({'run_id': rid, 'settled': True, 'verified': False, 'status': 'ALERT', 'reason': type(exc).__name__+': '+str(exc)})
            value = {'utc_epoch': time.time(), 'runs': states, 'publication_alert': publication_failed,
                     'status': 'verified' if all(row['verified'] for row in states) and not publication_failed else
                               'ALERT' if publication_failed or any(row.get('status') == 'ALERT' for row in states) else 'running'}
            new(ops/f'summary-{time.time_ns()}.json', value)
            if all(row['settled'] for row in states):
                new(ops/'summary-final.json', value)
                return value
            time.sleep(min(900, max(0, plan['deadline_epoch']-time.time())))
        result = {'status': 'ALERT', 'reason': 'original 48h deadline expired; no job killed or retried'}
        new(ops/'summary-final.json', result)
        return result


def start_driver(req, request_path):
    ops = Path(req['ops'])
    plan = load(ops/'plan.json')
    proc = None
    try:
        identity(req)
        if plan['source_commit'] != req['source_commit'] or not valid_deadline(plan['deadline_epoch']):
            raise ValueError('unexpired fixed driver plan required')
        with (ops/'driver.log').open('x') as log:
            proc = subprocess.Popen([req['python'], '-B', str(Path(__file__).resolve()), 'driver', '--request', str(Path(request_path).resolve()),
                                     '--request-sha', sha(request_path), '--plan-sha', sha(ops/'plan.json')],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        until = min(plan['deadline_epoch'], time.time()+READY_TIMEOUT_SECONDS)
        while time.time() < until:
            code = proc.poll()
            if code is not None and code != 0:
                raise RuntimeError('driver launcher failed before ready')
            ready_path = ops/'driver-ready.json'
            if ready_path.exists():
                ready = load(ready_path)
                if (ready['launcher_pid'] != proc.pid or ready['ppid'] != 1 or ready['plan_sha256'] != digest(plan)
                    or ready['deadline_epoch'] != plan['deadline_epoch'] or os.getsid(ready['pid']) != ready['pid']):
                    raise ValueError('detached owner ready identity mismatch')
                new(ops/'publish-release.json', {'plan_sha256': digest(plan), 'ready_sha256': sha(ready_path), 'driver_pid': ready['pid']})
                return {'status': 'driver_released_not_accepted', 'pid': ready['pid'], 'deadline_epoch': plan['deadline_epoch'], 'controller_publications': 0}
            if (ops/'summary-final.json').exists():
                raise RuntimeError('driver startup failed before ready/release')
            time.sleep(.05)
        raise TimeoutError('driver ready deadline; no controller publications')
    except Exception as exc:
        alert = {'status': 'ALERT', 'reason': 'controller startup: '+type(exc).__name__+': '+str(exc), 'controller_publications': 0}
        new(ops/'controller-ALERT.json', alert)
        if proc is None:
            new(ops/'summary-final.json', alert)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'cuda-preview', 'dry-run', 'submit', 'driver'))
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha')
    parser.add_argument('--plan-sha')
    args = parser.parse_args(argv)
    if args.action == 'driver' and (not args.request_sha or not args.plan_sha or sha(args.request) != args.request_sha):
        raise ValueError('driver requires controller-sealed request and plan hashes')
    req = load(args.request)
    actions = {'preflight': preflight, 'cuda-preview': cuda_preview, 'dry-run': dry_run, 'submit': submit, 'driver': lambda value: driver(value, args.plan_sha)}
    print(json.dumps(actions[args.action](req), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
