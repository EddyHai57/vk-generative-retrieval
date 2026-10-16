"""M2: causal K4 attention readout, one zero gate, shared M1-mean draw order."""
import hashlib
import json
import torch
from .retrieval_seqm1 import (DIM, Parts, SeqM1IDMean, assert_left_aligned,
    branch_spec as seqm1_spec, make_seqm1_model, parameter_report_seqm1)

INTERESTS = 4

class SeqM2IDMean(SeqM1IDMean):
    def _readout(self, z, valid):
        # where before the readout prevents invalid-position NaNs entering matmul.
        z = torch.where(valid[..., None], z, torch.zeros_like(z))
        scores = torch.tanh(self.attn_proj(z)) @ self.attn_w.T
        alpha = scores.transpose(1, 2).masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        r = alpha @ z
        if not bool(torch.isfinite(r).all()):
            raise ValueError('nonfinite multi-interest readout')
        return r, alpha

    def users(self, histories, side=None):
        p = self.parts(histories, side)
        users = p.m[:, None, :] + self.gate * p.r
        if not bool(torch.isfinite(users).all()):
            raise ValueError('nonfinite users')
        return users

    def diagnose(self, histories, side):
        with torch.no_grad():
            p = self.parts(histories, side)
            u = p.m[:, None, :] + self.gate * p.r
            ratios = (self.gate * p.r).norm(dim=-1) / p.m.norm(dim=-1).clamp_min(1e-12)[:, None]
            n = p.valid.sum(1).float()
            entropy = -(p.alpha * p.alpha.clamp_min(1e-30).log()).sum(-1)
            entropy = torch.where(n[:, None] > 1, entropy / n.clamp_min(2).log()[:, None], torch.ones_like(entropy))
            cos = torch.nn.functional.cosine_similarity
            pair = [cos(u[:, j], u[:, k], dim=-1) for j in range(4) for k in range(j+1,4)]
            return {'branch_ratio': ratios.max(1).values, 'head_branch_ratio': ratios,
                    'attn_entropy_ratio': entropy, 'pairwise_user_cosine': torch.stack(pair,1),
                    'cos_u_m': cos(u,p.m[:,None,:],dim=-1),
                    'attn_last_weight': p.alpha.gather(2,(p.valid.sum(1)-1)[:,None,None].expand(-1,4,1)).squeeze(-1)}

def make_seqm2_model(content, supported_indices, max_len):
    # Build every shared parameter using the frozen factory, before ANY extra draw.
    # Reclassification adds methods only; no constructor or shared parameter is rerun.
    model = make_seqm1_model(content, supported_indices, max_len, 'mean')
    model.__class__ = SeqM2IDMean
    model.attn_proj = torch.nn.Linear(DIM, DIM)  # identical W1/b1 code to M1 attn
    model.attn_w = torch.nn.Parameter(torch.empty(INTERESTS, DIM))
    torch.nn.init.normal_(model.attn_w, mean=0., std=.02)
    object.__setattr__(model, 'arm', 'k4-causal')
    object.__setattr__(model, 'readout', 'multi-interest-attention')
    return model

def parameter_report_seqm2(model):
    report = parameter_report_seqm1(model)
    extra = sum(p.numel() for p in model.attn_proj.parameters()) + model.attn_w.numel()
    return {**report, 'multi_interest_readout': extra, 'total': report['total'] + extra}

def branch_spec_sha256(model):
    spec = {**seqm1_spec(model), 'interests': INTERESTS, 'readout': 'multi-interest-attention',
            'readout_weight_shape': list(model.attn_w.shape), 'projection_bias': model.attn_proj.bias is not None,
            'projection_in_out': [model.attn_proj.in_features, model.attn_proj.out_features],
            'soft_optimizer_steps': 100, 'assignment_temperature': .10}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
