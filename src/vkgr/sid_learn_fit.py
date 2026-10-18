"""Frozen S2 optimizer loop. Only complete results contain a usable state.

The original L2 core/repair bytes are pinned; model initialization, batch order,
optimizer and reset functions are reused. The additional loss is the ruled S2
objective. No file writer, checkpoint reuse, validation selection or queue here.
"""
import time
from pathlib import Path
import numpy as np
import torch
from . import sid_learn_v1 as contract,sid_learn_loss as learn_loss,sid_rqvae as rq,sid_rqvae_r as fixed
from .data_audit import file_sha256
FROZEN_L2={rq:'52173bea9b80ffeb40e25943e3d7069ec53f9bd34a4509c97278194714065067',
           fixed:'c786848ab85c6ce2a593ccdfc626688484ae58d2ee0406c889a5ad48c5f54392'}

def check_frozen_l2():
    for module,pin in FROZEN_L2.items():
        if file_sha256(Path(module.__file__).resolve())!=pin:raise ValueError('fixed L2 source bytes differ')

def diagnostic_due(step,max_steps):
    return step==2000 or step==max_steps

def _inputs(content,target,fallback):
    if not isinstance(content,np.ndarray) or content.dtype!=np.float32 or content.ndim!=2 or content.shape[1]!=64 or not len(content) or not isinstance(target,np.ndarray) or target.dtype!=np.float32 or target.shape!=(len(content),32) or not isinstance(fallback,np.ndarray) or fallback.dtype!=np.bool_ or fallback.shape!=(len(content),):
        raise ValueError('float32 content64/target32 and bool fallback rows required')
    if not np.isfinite(content).all() or not np.isfinite(target).all():raise ValueError('STOP: input NaN or Inf')

def fit_learned(content,target,fallback,*,config,device='cpu',engineering=False,options=None,clock=time.monotonic,guard=None):
    contract.validate_config(config)
    if type(engineering) is not bool:raise ValueError('explicit engineering boolean required')
    if options is not None and not engineering:raise ValueError('formal fit cannot override frozen schedule')
    options={} if options is None else dict(options)
    if set(options)-{'K','batch_size','max_steps','check_every'}:raise ValueError('unknown engineering option')
    _inputs(content,target,fallback)
    if not engineering:
        contract.check_fallback_gate(fallback)
        if len(content)!=contract.CATALOG_ITEMS:raise ValueError('formal catalog count differs')
    k=options.get('K',config['codebook_size']);batch_size=options.get('batch_size',config['batch_size'])
    max_steps=options.get('max_steps',config['max_steps']);check_every=options.get('check_every',rq.PREREG['check_every'])
    if any(type(v) is not int or v<1 for v in (k,batch_size,max_steps,check_every)) or k<2 or min(len(content),batch_size)<k:
        raise ValueError('positive schedule and initialization batch >= K required')
    check_frozen_l2()
    if guard is not None:guard()
    start=clock();seed=config['seed'];torch.set_num_threads(1);torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=rq.RQVAE(k).to(device)
    x=torch.from_numpy(np.ascontiguousarray(content)).to(device)
    frozen_target=torch.from_numpy(np.ascontiguousarray(target)).to(device)
    mask=torch.from_numpy(np.ascontiguousarray(fallback)).to(device)
    rng=np.random.RandomState(seed);order=rng.permutation(len(x));position=0
    rq.init_codebooks(model,rq._first_batch(x,order,batch_size),seed,rq.PREREG['kmeans_iters'])
    init_digest=rq.state_digest(model.state_dict());optimizer=fixed.make_optimizer(model.parameters(),'l2')
    reset_cfg=fixed.PREREG_R['arms']['l2'];last_used=fixed.new_last_used(model)
    rng_reset=np.random.RandomState(seed+reset_cfg['rng_reset_offset'])
    bad=torch.zeros((),dtype=torch.bool,device=device)
    names=('total','l2_total','recon','align','div')
    running={name:torch.zeros((),dtype=torch.float64,device=device) for name in names};counted=0
    loss_log=[];usage_log=[];distance_log=[];reset_log=[]
    def report(status,step,elapsed):
        return {'status':status,'step':step,'max_steps':max_steps,'elapsed':elapsed,'wall_cap_seconds':config['wall_cap_seconds'],
                'K':k,'seed':seed,'batch_size':batch_size,'engineering':engineering,'init_digest':init_digest,
                'loss_log':loss_log,'usage_log':usage_log,'distance_log':distance_log,'reset_log':reset_log,
                'timing_scope':'core_includes_initialization','usage_dtype':'float32_hard_argmin_diagnostic_only',
                'distance_scope':'actual_pre_update_forward_d_over_32_all_rows'}
    for step in range(1,max_steps+1):
        if position>=len(x):order=rng.permutation(len(x));position=0
        rows=torch.from_numpy(order[position:position+batch_size]).to(device);position+=batch_size
        batch=x[rows];output=model(batch);books=list(model.codebooks)
        try:
            terms=learn_loss.loss_terms(batch,output,frozen_target[rows],mask[rows],books,
                    alpha=config['alpha'],gamma=config['gamma'],temperature=config['temperature'],tau_div=config['diversity']['tau'])
        except ValueError as exc:
            if str(exc).startswith('STOP:'):return report('fail_nonfinite',step,clock()-start)
            raise
        distances=learn_loss.distance_diagnostics(output['q'],books) if diagnostic_due(step,max_steps) else None
        optimizer.zero_grad(set_to_none=True);terms['total'].backward();optimizer.step()
        fixed.touch_last_used(last_used,output['q']['codes'],step)
        reset_log.extend(fixed.reset_dead_codes(model,optimizer,output['q']['residuals'],last_used,step,rng_reset,reset_cfg))
        bad |= ~torch.isfinite(terms['total'].detach())
        for name in names:running[name]+=terms[name].detach().double()
        counted+=1
        if step%check_every==0 or step==max_steps:
            if bool(bad) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):return report('fail_nonfinite',step,clock()-start)
            elapsed=clock()-start
            if elapsed>config['wall_cap_seconds']:return report('fail_wall',step,elapsed)
            if guard is not None:guard()
        if step%rq.PREREG['log_every']==0 or step==max_steps:
            loss_log.append({'step':step,**{name:float(value)/counted for name,value in running.items()}})
            for value in running.values():value.zero_()
            counted=0
        if distances is not None:
            usage_log.append((step,rq._usage(model,x)))
            distance_log.append({'step':step,'batch_rows':len(rows),'layers':distances})
    elapsed=clock()-start
    if elapsed>config['wall_cap_seconds']:return report('fail_wall',max_steps,elapsed)
    state={'K':k,'state_dict':{name:value.detach().cpu().clone() for name,value in model.state_dict().items()}}
    digest=rq.state_digest(state['state_dict'])
    elapsed=clock()-start
    if elapsed>config['wall_cap_seconds']:return report('fail_wall',max_steps,elapsed)
    result=report('complete',max_steps,elapsed)
    result.update(state=state,final_state_digest=digest)
    return result
