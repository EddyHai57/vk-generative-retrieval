"""User tower timing only, initialized weights and synthetic feature values.

The historical accepted timing functions are imported by immutable source hash.
No training checkpoint, item vector export, retrieval, ANN or network is read.
"""
import argparse,datetime,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path

def scope():return 'model.users() only; not end-to-end'

def make_model(name,catalog=120416):
 import torch
 from vkgr.training import seed_all
 from vkgr.retrieval_seqm1 import make_seqm1_model
 from vkgr.retrieval_seq_combo_stack import make_model as stack
 from vkgr.retrieval_feat04combo import STAT_LAYOUT,E_FEATURE
 if name not in ('mean','bi','stack'):raise ValueError('registered arms only')
 seed_all(42);content=torch.zeros(catalog,64);supported=list(range(1,min(21496,catalog-1)+1))
 if name=='stack':
  buckets={buffer:torch.zeros(catalog,dtype=torch.long) for _,buffer,_ in STAT_LAYOUT[E_FEATURE]}
  return stack(content,supported,50,buckets)
 return make_seqm1_model(content,supported,50,name)

def checked_source(req):
 for key in ('stack_source','bench_source'):
  ref=req[key];root=Path(ref['root'])
  if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=ref['commit']:raise ValueError('source commit mismatch')
  if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():raise ValueError('dirty source')
  for rel,pin in ref['files'].items():
   p=root/rel
   if p.is_symlink() or '..' in Path(rel).parts or Path(rel).is_absolute() or hashlib.sha256(p.read_bytes()).hexdigest()!=pin:raise ValueError('source SHA mismatch')
 p=Path(req['bench_source']['root'])/'scripts/seqm1_latency_bench.py'
 s=importlib.util.spec_from_file_location('frozen_users_bench',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def execute(req,preflight=False):
 sys.path.insert(0,str(Path(req['stack_source']['root'])/'src'))
 m=checked_source(req);source=m.load_lengths(req['lengths'],req['lengths_sha256'],2520)
 out={'status':'latency_preflight_ok' if preflight else 'latency_measured','scope':scope(),'checkpoint_read':False,'test_read':False,
      'length_source':{k:v for k,v in source.items() if k!='lengths'},'source':{k:req[k] for k in ('stack_source','bench_source')},'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 if preflight:return out
 import torch
 torch.set_num_threads(1)
 if not torch.cuda.is_available():raise ValueError('A6000 required')
 reports={}
 for device in ('cpu','cuda'):
  models={}
  for name in ('mean','bi','stack'):
   model=make_model(name).to(device).eval()
   report=m.measure(name,model,device,120416,512,1000,50,source,7)
   report['validation_export_pass']=m.export_pass(model,source['lengths'],device,120416,512,7)
   models[name]=report;del model
  for mode in ('offline','online'):
   field=mode+'_ms_per_query'
   if len({r[field]['input_sha256'] for r in models.values()})!=1:raise ValueError('unpaired synthetic inputs')
   if any((r[field]['timed_calls'],r[field]['warmup_calls'])!=(1000,50) for r in models.values()):raise ValueError('formal timing count')
  reports[device]={'threads':torch.get_num_threads(),'device':device,'models':models,'ratio_vs_mean':{name:{mode:models[name][mode+'_ms_per_query']['mean']/models['mean'][mode+'_ms_per_query']['mean'] for mode in ('offline','online')} for name in ('bi','stack')}}
 out['reports']=reports;out['torch']=torch.__version__;out['limitations']='real lengths only; feature values and weights initialized; no feature store/item tower/retrieval/network; no learned-weight timing claim'
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--parameters',required=True);p.add_argument('--parameters-sha',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--output');a=p.parse_args();raw=Path(a.parameters).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=a.parameters_sha:raise ValueError('parameters mismatch')
 out=execute(json.loads(raw),a.preflight)
 if a.output:
  target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False)
  print(json.dumps({'status':out['status'],'report_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
 else:print(json.dumps(out,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
