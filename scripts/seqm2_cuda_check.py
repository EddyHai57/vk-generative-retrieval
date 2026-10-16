"""M2 CUDA rehearsal: synthetic gradients/rows, true full dimensions, no old checkpoint."""
import gc,json,math,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True

def own_gpu_mib():
 r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 if r.returncode:raise ValueError('own GPU measurement unavailable')
 v=[float(x.split(',')[1]) for x in r.stdout.splitlines() if x.split(',')[0].strip()==str(os.getpid())]
 if len(v)!=1:raise ValueError('unique own GPU PID required')
 return v[0]

def zero_start(model,h,side,items,labels,mask,logq):
 import torch
 from vkgr.retrieval_seqm2_run import multi_interest_loss
 from vkgr.retrieval_negatives import masked_ce_logq
 model.train();model.zero_grad(set_to_none=True);u=model.users(h,side);m=model.anchor_vector(h,side)
 if not torch.equal(u,m[:,None,:].expand_as(u)):raise ValueError('CUDA zero users mismatch')
 ref,_=masked_ce_logq(m,items,labels,mask,.1,logq);actual,_,_=multi_interest_loss(u,items,labels,mask,.1,logq,1)
 error=abs(float(ref.detach())-float(actual.detach()))
 if error>1e-6:raise ValueError('zero start loss tolerance exceeded')
 actual.backward();branch_prefix=('order_table.','layer.','final_norm.','attn_proj.','attn_w')
 if not torch.isfinite(model.gate.grad) or not bool(model.gate.grad.abs()>0):raise ValueError('zero gate gradient missing')
 if any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in model.named_parameters() if n.startswith(branch_prefix)):raise ValueError('zero branch gradient mismatch')
 result={'loss_error':error,'gate_gradient':float(model.gate.grad),'branch_nonzero_gradient_elements':0};model.zero_grad(set_to_none=True);return result

def small_replay(run,out,device='cuda'):
 data=run.fr.synthetic_data();data={**data,'rows':data['rows']*math.ceil(512/len(data['rows']))}
 cfg={**run.default_config(),'batch_size':4,'max_epochs':2,'min_epochs':2}
 result=run.run_session(out/'synthetic',data,cfg,device)
 manifest=json.loads((out/'synthetic/manifest.json').read_text())
 phases={r['assignment']['phase'] for r in manifest['records'] if r['loss'] is not None}
 if phases!={'soft','hard'} or sum(r['loss'] is not None for r in manifest['records'])<201:raise ValueError('natural soft-to-hard transition not exercised')
 verified=run.verify_run(out/'synthetic/manifest.json',result['manifest_sha256'],data,cfg,device)
 if verified['status']!='verified' or not manifest['ablation_g0']:raise ValueError('synthetic verify/g0 missing')
 return verified,sum(r['loss'] is not None for r in manifest['records']),manifest['evaluations'][-1]['diagnostics']['hard_window']

def full_data(run):
 import torch
 from php01_vram_smoke import synthetic_batch
 gen=torch.Generator().manual_seed(7);h,s=synthetic_batch(gen,512,21496,120416)
 proto=run.fr.synthetic_data()['queries'][0];queries=[]
 for n in range(2520):
  i=n%512;ids=h[i].tolist();side=s[i,:,:3].tolist()
  queries.append({**proto,'history':ids,'history_positions':[[0,k] for k in range(50)],'side':side,'ctx':[[0,0] for _ in range(50)],'target':(n%21496)+1,'oov':False})
 data={'synthetic':True,'catalog':list(range(1,120416)),'idmap':{i:i for i in range(120416)},'content':torch.zeros(120416,64),'supported_indices':list(range(1,21497)),'support':set(range(1,21497)),'queries':queries,'rows':queries}
 return data,h,s,gen

def main():
 import torch
 import seqm2_ops as ops
 ops.identity();run=ops.modules();torch.set_num_threads(1)
 if not torch.cuda.is_available():raise ValueError('actual CUDA required')
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
 verified,count,window=small_replay(run,out)
 gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 context=own_gpu_mib();reserved0=torch.cuda.memory_reserved()/2**20
 data,h,s,gen=full_data(run);cfg=run.default_config();model=run._model(data,cfg,'cuda');opt=run.make_optimizer(model,cfg)
 targets=torch.randperm(21496,generator=gen)[:512]+1;labels=torch.arange(512,device='cuda');mask=torch.ones(512,512,dtype=torch.bool,device='cuda');logq=torch.full((512,),math.log(1/21496),device='cuda');h=h.cuda();s=s.cuda();targets=targets.cuda()
 zero=zero_start(model,h,s,model.items(targets),labels,mask,logq);samples=[];phases=[]
 def observe(stage):
  torch.cuda.synchronize();samples.append({'stage':stage,'own_process_mib':own_gpu_mib(),'max_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'max_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
 for step in (1,100,101):
  model.train();opt.zero_grad(set_to_none=True);loss,_,d=run.multi_interest_loss(model.users(h,s),model.items(targets),labels,mask,.1,logq,step);loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite CUDA gradient')
  if step==100 and (not torch.isfinite(model.attn_w.grad).all() or not bool((model.attn_w.grad.abs().sum(1)>0).all())):raise ValueError('CUDA soft W2 per-head gradient missing')
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();phases.append(d['phase']);observe('full-shape-step-'+str(step))
 if phases!=['soft','soft','hard']:raise ValueError('boundary mismatch')
 result,diag=run.evaluate_model(model,data,data['queries'],cfg);observe('full-catalog-evaluate')
 if len(result['predictions'])!=2520 or any(len(p)!=50 or len(set(p))!=50 for p in result['predictions']):raise ValueError('full shape exact top50 mismatch')
 g0=run.ablation_g0(model,data,data['queries'],cfg,1,result['metrics']['all']['recall@50']);observe('full-shape-g0')
 if len(g0['predictions'])!=2520:raise ValueError('full g0 incomplete')
 measured=max([x['own_process_mib'] for x in samples]+[x['max_reserved_mib']+max(0,context-reserved0) for x in samples]);declared=ops.declared_vram(measured)
 proof={'status':'cuda_engineering_verified','read_at_utc':ops.now(),'commit':ops.identity()['commit'],'synthetic_verify':verified,'natural_optimizer_steps':count,'hard_window_diagnostic':window,'soft_hard_boundary_passed':True,'full_shape':{'batch':512,'history':50,'interests':4,'catalog':120415},'evaluation_queries':2520,'full_shape_g0':True,'zero_start_loss_error':zero['loss_error'],'zero_start':zero,'gpu_samples':samples,'measured_vram_mib':measured,'declared_vram_mib':declared,'test_read':False,'existing_checkpoint_read':False,'real_training_rows_used':False,'scope':'synthetic CUDA engineering only; no real recall result'}
 with (out/'engineering.json').open('x') as f:json.dump(proof,f,sort_keys=True,allow_nan=False)
 print(json.dumps(proof,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
