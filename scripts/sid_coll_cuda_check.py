"""S1 real-input read-only preflight and synthetic full-shape CUDA rehearsal.

Only probe parameters are updated; no validation/test score is produced. Reload is
restricted to this probe's newly produced checkpoint, as explicitly authorized.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def declare_vram(measured):
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError('finite positive measured VRAM required')
    return math.ceil(measured*1.15/256)*256


def synthetic_rows(catalog,count,width):
    if len(catalog)<2 or not 1<=count<=512 or not 1<=width<=50:
        raise ValueError('bounded synthetic shape required')
    return [{'history':[int(catalog[(i+j)%len(catalog)]) for j in range(width)],
             'target':int(catalog[(i+width)%len(catalog)])} for i in range(count)]


def check_seed_configs(configs):
    if set(configs)!={42,2024,2025}:raise ValueError('all frozen seeds required')
    base={k:v for k,v in configs[42].items() if k!='seed'}
    for seed,c in configs.items():
        if c.get('seed')!=seed or {k:v for k,v in c.items() if k!='seed'}!=base:
            raise ValueError('three configurations must differ only by seed')


def check_decoder_shape(codec,config):
    # _ValidatedCodec exposes its semantic depth through the catalog protocol.
    if codec.catalog.depth!=3 or config['beam']!=100:
        raise ValueError('frozen decoder shape changed')

def _checked(spec,role):
    from vkgr import generative_data as gd
    r=spec['refs'][role]
    return gd._checked_file(r['path'],r['sha256'])


def _write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)


def context(spec,scratch):
    import torch
    from vkgr import generative_coll_run as g
    from vkgr.data_audit import file_sha256
    root=Path(__file__).resolve().parents[1]
    configs={seed:json.loads(_checked(spec,'config'+str(seed)).read_text()) for seed in g.SEEDS}
    check_seed_configs(configs)
    for c in configs.values():g.validate_config(c,False)
    # The old prepared data channel is independent from the clean new source root.
    params=spec['parameters'];data_root=Path(params.get('data_root',params.get('root','')))
    if not data_root.is_absolute() or not data_root.is_dir():raise ValueError('fixed existing data channel required')
    os.environ['VKGR_DATA_ROOT']=str(data_root)
    refs=spec['refs']
    for role in ('prepared','history','sid_manifest','content','input_manifest','vectors'):_checked(spec,role)
    def path(role):return refs[role]['path']
    def pin(role):return refs[role]['sha256']
    ctx,engineering,sid=g._contexts(root,path('prepared'),pin('prepared'),path('history'),pin('history'),
        path('sid_manifest'),pin('sid_manifest'),path('content'),pin('content'),configs[42],scratch,
        input_manifest=path('input_manifest'),input_sha=pin('input_manifest'),vectors_path=path('vectors'),vectors_sha=pin('vectors'))
    if engineering or ctx.binding.get('synthetic') is not False or sid.info['engineering']:
        raise ValueError('formal real input required')
    if (len(ctx.catalog),len(ctx.examples),len(ctx.queries))!=(120415,86785,2520):
        raise ValueError('real input shape differs from frozen T3')
    check_decoder_shape(ctx.codec,configs[42])
    torch.set_num_threads(1)
    runtime=g._runtime(False)
    return g,ctx,sid,configs,runtime


def preflight(spec,output):
    import torch
    out=Path(output);out.mkdir(exist_ok=False)
    g,ctx,sid,configs,runtime=context(spec,out/'context')
    root=Path(__file__).resolve().parents[1];entries={}
    for seed,cfg in configs.items():
        g.gr._seed(seed);model=g.make_model_b1(ctx.codec,cfg,device=runtime['device'])
        binding=g._binding(root,ctx,cfg,runtime,model,spec['refs']['prepared']['sha256'],spec['refs']['history']['sha256'],sid)
        if binding['test_gate']!='closed' or binding['synthetic'] is not False or binding['source']['git']['git_base']!=spec['commit']:
            raise ValueError('formal source/data binding mismatch')
        entries[str(seed)]={'config_sha256':spec['refs']['config'+str(seed)]['sha256'],
            'binding_sha256':g._digest(binding),'source':binding['source'],'data_root':binding['data_root'],
            'parameters':sum(p.numel() for p in model.parameters()),'engineering':False}
        del model
    proof={'status':'preflight_ok','engineering':False,'entries':entries,'source_commit':spec['commit'],
           'test_read':False,'existing_checkpoint_read':False,'new_training':False,
           'real_shape':{'items':len(ctx.catalog),'examples':len(ctx.examples),'queries':len(ctx.queries),'vocab':ctx.codec.vocab_size}}
    _write(out/'preflight.json',proof);return proof


def rehearse(spec,output):
    import torch
    from vkgr import tiger
    out=Path(output);out.mkdir(exist_ok=False)
    g,ctx,sid,configs,runtime=context(spec,out/'context')
    device=runtime['device'];cfg=configs[42]
    torch.cuda.synchronize();free,_=torch.cuda.mem_get_info();torch.cuda.reset_peak_memory_stats()
    g.gr._seed(42);model=g.make_model_b1(ctx.codec,cfg,device=device)
    opt=g.gt.make_optimizer(model,g.core_config(cfg));tokens=g.SIDTokensL(ctx.codec,50)
    rows=synthetic_rows(list(ctx.catalog),512,50);samples=[];steps=[]
    def observe(stage):
        torch.cuda.synchronize();own=None
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        if r.returncode!=0:raise ValueError('process GPU memory measurement unavailable')
        for line in r.stdout.splitlines():
            p=line.split(',')
            if len(p)==2 and p[0].strip()==str(os.getpid()):own=float(p[1].strip())
        if own is None or own<=0:raise ValueError('own process GPU sample missing')
        samples.append({'stage':stage,'process_gpu_mib':own,'reserved_mib':torch.cuda.max_memory_reserved()/2**20,
          'allocated_mib':torch.cuda.max_memory_allocated()/2**20,'global_delta_mib':(free-torch.cuda.mem_get_info()[0])/2**20})
    for _ in range(3):
        begin=time.monotonic();step=g.train_batch(model,opt,rows,ctx,tokens,cfg,device)
        torch.cuda.synchronize();steps.append({'seconds':time.monotonic()-begin,'examples':step.examples,
          'microbatches':step.microbatch_sizes,'finite_loss':math.isfinite(step.loss_mean)})
        observe('train')
    if not all(x['finite_loss'] and x['examples']==512 for x in steps):raise ValueError('full-shape update failed')
    histories=[r['history'] for r in rows[:3]];model.eval()
    with torch.no_grad():
        batched=g.gb.decode_histories_batched(model,tokens,histories,100,1);observe('batched')
        serial=[tiger.decode_history(model,ctx.codec,h,100) for h in histories];observe('serial')
    if batched!=serial:raise ValueError('serial/batch1 decode mismatch')
    cp=out/'probe.pt';torch.save(model.state_dict(),cp);pin=hashlib.sha256(cp.read_bytes()).hexdigest()
    state=torch.load(cp,map_location='cpu',weights_only=True)
    clone=g.make_model_b1(ctx.codec,cfg,device=device);clone.load_state_dict(state,strict=True);clone.eval()
    with torch.no_grad():replay=g.gb.decode_histories_batched(clone,tokens,histories,100,1)
    observe('new-probe-reload')
    if replay!=batched:raise ValueError('new probe replay differs')
    # Process-local peak, not another job's global footprint, determines declaration.
    measured=max(max(s['process_gpu_mib'],s['reserved_mib']) for s in samples)
    proof={'status':'cuda_engineering_verified','engineering':False,'source_commit':spec['commit'],
      'synthetic_grad_steps':3,'real_training_rows_used':False,'effective_batch':512,'microbatch':cfg['microbatch'],
      'history_len':50,'encoder_tokens':200,'beam':100,'items':len(ctx.catalog),'vocab_size':ctx.codec.vocab_size,
      'steps':steps,'gpu_samples':samples,'measured_mib':measured,'declared_vram_mib':declare_vram(measured),
      'serial_batched_equal':True,'new_checkpoint_replay_equal':True,'probe_checkpoint_sha256':pin,
      'test_read':False,'existing_checkpoint_read':False}
    _write(out/'engineering.json',proof);return proof


def dry(spec,output):
    out=Path(output);out.mkdir(exist_ok=False)
    proof=json.loads(_checked(spec,'cuda').read_text())
    if proof.get('schema')=='coll-stage-receipt-v1':proof=proof['raw_proof']
    if proof.get('status')!='cuda_engineering_verified' or proof.get('source_commit')!=spec['commit']:
        raise ValueError('raw CUDA proof required')
    if proof.get('declared_vram_mib')!=declare_vram(proof.get('measured_mib',0)):
        raise ValueError('VRAM declaration differs from measurement')
    configs={seed:json.loads(_checked(spec,'config'+str(seed)).read_text()) for seed in (42,2024,2025)}
    check_seed_configs(configs)
    value={'status':'dry_run_ok','source_commit':spec['commit'],'declared_vram_mib':proof['declared_vram_mib'],
           'configs':{str(s):spec['refs']['config'+str(s)]['sha256'] for s in configs},'test_read':False,
           'launcher_prepare_required_before_each_submission':True}
    _write(out/'dry.json',value);return value
