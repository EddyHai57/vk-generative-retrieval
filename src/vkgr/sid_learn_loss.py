"""Original S2 loss implementation of the frozen Kiro equations.

Quantization/straight-through/residuals/export remain in the unchanged L2 core.
No target construction, file IO, dataset selection or parameter updates here.
"""
import math
import torch
from torch.nn import functional as F
from . import sid_rqvae as rq
LATENT_DIM=32
EPS=1e-12

def _finite_cpu(value):
    # CUDA nonfinites stay on-device for the fit loop's inherited accumulated
    # bad flag/check cadence; avoid a host synchronization per distance matrix.
    if value.device.type=='cpu' and not bool(torch.isfinite(value).all()):
        raise ValueError('STOP: loss input/output holds NaN or Inf')
    return value

def _positive(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be finite and positive')

def squared_distances(residual,codebook):
    if not isinstance(residual,torch.Tensor) or not isinstance(codebook,torch.Tensor) or residual.ndim!=2 or codebook.ndim!=2 or not len(residual) or len(codebook)<2 or residual.shape[1]!=LATENT_DIM or codebook.shape[1]!=LATENT_DIM or residual.dtype!=codebook.dtype or residual.device!=codebook.device:
        raise ValueError('matching nonempty latent32 tensors required')
    _finite_cpu(residual);_finite_cpu(codebook)
    # Exact arithmetic expression used by rq._nearest; neither operand detached.
    result=(residual*residual).sum(1,keepdim=True)-2*(residual@codebook.T)+(codebook*codebook).sum(1)[None,:]
    return _finite_cpu(result)

def _distances(q,codebooks):
    if type(q) is not dict or not isinstance(codebooks,(list,tuple)) or not codebooks or len(q.get('residuals',()))!=len(codebooks)+1:
        raise ValueError('actual forward residual chain and codebooks required')
    return [squared_distances(r,e) for r,e in zip(q['residuals'][:-1],codebooks)]

def diversity_loss(q,codebooks,*,tau=1.0):
    """Negative normalized entropy of batch-average SOFT assignments.

    All rows participate, no EMA or extra straight-through path. q.residuals is
    the actual hard forward chain, including L2's existing detached prior embeds.
    """
    _positive(tau,'tau_div');terms=[]
    for distance,book in zip(_distances(q,codebooks),codebooks):
        probability=torch.softmax(-distance/(LATENT_DIM*tau),dim=1)
        average=probability.mean(dim=0)
        terms.append((average*torch.log(average+EPS)).sum()/math.log(len(book)))
    return _finite_cpu(torch.stack(terms).mean())

def alignment_loss(zst,target,fallback,*,temperature=.07):
    """One-way normalized zst->fixed PCA target; fallback absent from BOTH axes."""
    _positive(temperature,'alignment temperature')
    if not isinstance(zst,torch.Tensor) or not isinstance(target,torch.Tensor) or zst.ndim!=2 or zst.shape!=target.shape or zst.shape[1]!=LATENT_DIM or zst.dtype!=target.dtype or zst.device!=target.device or not isinstance(fallback,torch.Tensor) or fallback.dtype!=torch.bool or fallback.shape!=(len(zst),):
        raise ValueError('matching latent32/target and bool fallback rows required')
    _finite_cpu(zst);_finite_cpu(target)
    valid=~fallback.to(zst.device)
    latent=zst[valid];fixed=target.detach()[valid]
    if not len(latent):return zst.sum()*0.
    # Undefined zero norms become nonfinite and STOP, never silently imputed.
    latent=_finite_cpu(latent/torch.linalg.vector_norm(latent,dim=1,keepdim=True))
    fixed=_finite_cpu(fixed/torch.linalg.vector_norm(fixed,dim=1,keepdim=True))
    logits=(latent@fixed.T)/temperature
    return _finite_cpu(F.cross_entropy(logits,torch.arange(len(latent),device=logits.device)))

def loss_terms(x,output,target,fallback,codebooks,*,alpha=.1,gamma=.01,temperature=.07,tau_div=1.0):
    if type(alpha) not in (int,float) or type(gamma) not in (int,float) or not math.isfinite(alpha) or not math.isfinite(gamma) or alpha<0 or gamma<0:
        raise ValueError('finite nonnegative loss weights required')
    base=rq.loss_terms(x,output['xhat'],output['q'])
    align=alignment_loss(output['zst'],target,fallback,temperature=temperature)
    div=diversity_loss(output['q'],codebooks,tau=tau_div)
    unchanged=base['total']+alpha*align
    total=unchanged if gamma==0 else unchanged+gamma*div
    return {'recon':base['recon'],'codebook':base['codebook'],'commit':base['commit'],
            'l2_total':base['total'],'align':align,'div':div,'total':_finite_cpu(total)}

def distance_diagnostics(q,codebooks):
    """Only for step2000/final logging, using pre-update forward d/D."""
    result=[]
    with torch.no_grad():
        for distance in _distances(q,codebooks):
            values=(distance.detach()/LATENT_DIM).reshape(-1).double().cpu()
            percentiles=torch.quantile(values,torch.tensor([.5,.9],dtype=torch.float64))
            result.append({'mean':float(values.mean()),'p50':float(percentiles[0]),'p90':float(percentiles[1])})
    return result
