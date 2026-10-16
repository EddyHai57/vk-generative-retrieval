"""A1 B1024 full-shape CUDA rehearsal: synthetic rows, real verified codec.
No real training trajectory or validation metric; only new probe checkpoint reload.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

def synthetic_rows(catalog,count,width):
 if len(catalog)<2 or not 1<=count<=512 or not 1<=width<=50:raise ValueError('bounded shape required')
 return [{'history':[int(catalog[(i+j)%len(catalog)]) for j in range(width)],'target':int(catalog[(i+width)%len(catalog)])} for i in range(count)]
def declare_vram(measured):
 if not math.isfinite(measured) or measured<=0:raise ValueError('finite positive measured VRAM required')
 return math.ceil(measured*1.15/256)*256

def step_report(result,seconds):return {'seconds':seconds,'examples':result.examples,'microbatches':result.microbatch_sizes,'finite_loss':math.isfinite(result.loss_mean)}

def main():
 import torch
 import gtok02a1_ops as ops
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
 g6,ctx,engineering,sid,cfg,runtime=ops.prepare_context(42,out/'context-scratch')
 if engineering or ctx.binding['synthetic'] or sid.info['engineering']:raise ValueError('real codec/formal context required')
 torch.set_num_threads(1);device=runtime['device'];torch.cuda.synchronize();free_before,total=torch.cuda.mem_get_info();torch.cuda.reset_peak_memory_stats()
 g6.gr._seed(42);model=g6.make_model_arch_d(ctx.codec,cfg,device=device);opt=g6.ga.make_optimizer_arch(model,cfg);tokens=g6.sd.SIDTokensLD(ctx.codec,50)
 rows=synthetic_rows(list(ctx.catalog),512,50);samples=[]
 def observe(stage):
  torch.cuda.synchronize();own=None
  r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  if r.returncode==0:
   for line in r.stdout.splitlines():
    parts=line.split(',')
    if len(parts)==2 and parts[0].strip()==str(os.getpid()):
     try:own=float(parts[1].strip())
     except ValueError:pass
  samples.append({'stage':stage,'process_gpu_mib':own,'max_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'max_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'global_used_delta_mib':(free_before-torch.cuda.mem_get_info()[0])/2**20})
 steps=[]
 for _ in range(3):
  begin=time.monotonic();result=g6.train_batch_d(model,opt,rows,ctx.codec,tokens,cfg,device);torch.cuda.synchronize();steps.append(step_report(result,time.monotonic()-begin));observe('train')
 if not all(v['finite_loss'] and v['examples']==512 for v in steps):raise ValueError('full-shape synthetic step failed')
 histories=[r['history'] for r in rows[:3]];model.eval()
 with torch.no_grad():
  batched=g6.sdb.decode_histories_batched_d(model,tokens,histories,100,1);observe('batched-decode')
  serial=[g6.sdb.decode_history_d(model,ctx.codec,h,100) for h in histories];observe('serial-decode')
 if batched!=serial:raise ValueError('CUDA serial/batched decode mismatch')
 checkpoint=out/'probe-model.pt';torch.save(model.state_dict(),checkpoint);pin=hashlib.sha256(checkpoint.read_bytes()).hexdigest();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 clone=g6.make_model_arch_d(ctx.codec,cfg,device=device);clone.load_state_dict(state);clone.eval()
 with torch.no_grad():reloaded=g6.sdb.decode_histories_batched_d(clone,tokens,histories,100,1)
 observe('reload-decode')
 if reloaded!=batched:raise ValueError('new CUDA probe checkpoint replay differs')
 measured=max(v for sample in samples for k,v in sample.items() if k in ('process_gpu_mib','max_reserved_mib','global_used_delta_mib') and v is not None)
 proof={'status':'cuda_engineering_verified','synthetic_grad_steps':3,'real_training_rows_used':False,'real_shape':{'vocab_size':ctx.codec.vocab_size,'depth':ctx.codec.depth,'items':len(ctx.catalog),'examples':len(ctx.examples),'queries':len(ctx.queries)},'history_items':50,'encoder_tokens':200,'effective_batch':512,'microbatch':cfg['microbatch'],'steps':steps,'beam':100,'decode_queries':3,'serial_batched_equal':True,'reload_decode_equal':True,'probe_checkpoint_sha256':pin,'measured_mib':measured,'declared_vram_mib':declare_vram(measured),'gpu_samples':samples,'test_read':False,'existing_checkpoint_read':False}
 with (out/'engineering.json').open('x') as f:json.dump(proof,f,sort_keys=True)
 print(json.dumps(proof,sort_keys=True))
if __name__=='__main__':main()
