"""Frozen user-profile validation acceptance and users-only latency; no checkpoints.

Paired bootstrap/exact pooling reuse the already registered analysis functions.
Latency uses initialized weights and the accepted real-length timing protocol.
"""
import argparse,datetime,hashlib,importlib.util,json,math,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import postbatch_acceptance as prior
SEEDS=(42,2024,2025)
def registered():return {f'{stem}-s{s}' for stem in ('p-profile-u2','p-profile-u3','p-seqm1-mean') for s in SEEDS}
def raw_verified(records,rid,pin):
 good=[x for x in records if type(x) is dict and x.get('status')=='verified' and x.get('manifest_sha256',pin)==pin]
 if rid.startswith('p-profile-'):good=[x for x in good if x.get('run_id')==rid+'-verify' and x.get('train_run_id')==rid and x.get('manifest_sha256')==pin]
 if not good:raise ValueError('original stdout verified identity absent: '+rid)
def classify(d,k,p):
 if not math.isfinite(d) or type(k) is not int or not 0<=k<=3 or not 0<=p<=1:raise ValueError('finite frozen statistics required')
 return 1 if d<=-6 else 2 if d>=6 and k>=2 and p<.05 else 3 if d>=6 else 4
def slices(lengths,new,ctl):
 if not len(lengths)==len(new)==len(ctl) or any(type(n) is not int or not 0<=n<=50 for n in lengths) or any(type(x) is not int or x not in (0,1) for x in new+ctl):raise ValueError('H50/binary query arrays required')
 out={}
 for name,full in [('H50_underfull',False),('H50_full',True)]:
  inds=[i for i,n in enumerate(lengths) if (n==50)==full];n=len(inds);a=sum(new[i] for i in inds);b=sum(ctl[i] for i in inds)
  out[name]={'queries':n,'new_hits':a,'control_hits':b,'delta_hits':a-b,'new_recall':a/n if n else None,'control_recall':b/n if n else None,'only_new':sum(new[i] and not ctl[i] for i in inds),'only_control':sum(ctl[i] and not new[i] for i in inds)}
 return out

def paired(req,preflight=False):
 r=prior.load_phase(req['phase2']);r.ROOT=Path(req['root'])
 if r.ROOT!=Path('/workspace/vkgr-infra01-root') or set(req['manifests'])!=registered() or set(req['identities'])!=registered():raise ValueError('exact registered root and nine runs required')
 plan,_=r.read(req['plan'],req['plan_sha256']);entries={e['job']['run_id']:e for e in plan['entries']}
 if set(entries)!={x for x in registered() if x.startswith('p-profile-')}:raise ValueError('frozen six-run plan required')
 states={};manifests={}
 for rid in sorted(registered()):
  state=r.inspect_run(rid);pin=req['manifests'][rid]
  if not state['verified'] or state.get('manifest_sha256')!=pin or state['queue']['rejected'] or len(state['queue']['claimed'])!=1:raise ValueError('unverified/mismatched original run: '+rid)
  manifest,_=r.read(r.ROOT/'runs'/rid/'out/manifest.json',pin);job,_=r.read(r.ROOT/'queue/claimed'/state['queue']['claimed'][0]);cfg,_=r.read(job['config']);expected=req['identities'][rid]
  prior.check_job(rid,manifest,job,cfg,expected['deployment'],expected['binding_sha256'])
  if manifest['binding'].get('synthetic') is not False or manifest['binding'].get('test_gate')!='closed_hash_only':raise ValueError('real validation-only binding required')
  accepted=[v for v in state['verify']['jobs'] if v['verified']]
  if len(accepted)!=1:raise ValueError('unique accepted verifier required')
  vr=accepted[0]['run_id'];log=r.ROOT/'runs'/vr/'logs/stdout.log';raw=log.read_bytes();records=[]
  for line in raw.decode().splitlines():
   try:records.append(json.loads(line))
   except ValueError:pass
  raw_verified(records,rid,pin);r.READS[str(log)]=hashlib.sha256(raw).hexdigest()
  if rid in entries:
   e=entries[rid]
   if any(job.get(k)!=v for k,v in e['job'].items()) or expected['binding_sha256']!=e['expected']['expected_binding_sha256']:raise ValueError('claimed train differs from frozen plan')
   vjob,_=r.read(r.ROOT/'queue/claimed'/next(iter(r.queue(vr)['claimed'])))
   argv=vjob['argv'];a=job['argv'];idx=a.index('--output');out=Path(a[idx+1]);expected_args=a[:idx]+a[idx+2:];expected_args[expected_args.index('vkgr.retrieval_profile_run')+1]='verify';expected_args+=['--manifest',str(out/'manifest.json'),'--manifest-sha',pin]
   if argv!=expected_args or vjob.get('config')!=job['config'] or vjob.get('seed')!=job['seed'] or vjob.get('cwd')!=job['cwd']:raise ValueError('verifier source/config/input/seed mismatch')
  states[rid]=state;manifests[rid]=manifest
 hist,_=r.read(req['history'],r.HISTORY_PIN);queries,_=r.read(Path(req['history']).parent/'validation-l50.json',hist['artifacts']['validation-l50.json']);cat,catsha=r.read(req['catalog'])
 if len(queries)!=2520 or sum(q['oov'] for q in queries)!=623 or len(cat)!=120415 or len(set(cat))!=120415 or catsha!=hist['binding']['catalog_sha'] or r.plain_digest(cat)!=hist['catalog_sha256']:raise ValueError('frozen population mismatch')
 catalog=set(cat)
 if any(type(q['oov']) is not bool or q['oov']!=(q['target'] not in catalog) for q in queries):raise ValueError('OOV contract')
 for manifest in manifests.values():prior.check_inputs(manifest,hist,r.HISTORY_PIN)
 if len({m['binding']['support_sha256'] for m in manifests.values()})!=1:raise ValueError('support differs')
 out={'status':'preflight_ok' if preflight else 'paired_complete','engineering':False,'test_read':False,'checkpoint_read':False,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifests':req['manifests']}
 if not preflight:
  archives={rid:r.load_archive(rid,queries,hist,states[rid]) for rid in sorted(registered())}
  for rid,m in manifests.items():
   rec=next(e for e in m['evaluations'] if e['epoch']==m['selection']['best_epoch']);pred,_=r.read(r.ROOT/'runs'/rid/'out'/rec['predictions_ref'],rec['predictions_sha256'])
   if any(any(i not in catalog for i in p) for p in pred):raise ValueError('candidate outside frozen catalog')
  reports={}
  for arm in ('u2','u3'):
   seeds={str(s):r.paired(archives[f'p-profile-{arm}-s{s}'],archives[f'p-seqm1-mean-s{s}']) for s in SEEDS};pooled=prior.pool(seeds,SEEDS)
   row=classify(pooled['mean_delta_hits'],pooled['seeds_ge6'],pooled['p_pool'])
   reports[arm]={'seeds':seeds,'pool':pooled,'frozen_row':row,'conditions':{'row2':row==2,'two_seed_exact_p_lt_005':pooled['seeds_p_lt_005']>=2,'leakage_review_approved':True,'latency_report':'separate report; required before promotion'},'H50':{str(s):slices([len(q['history']) for q in queries],archives[f'p-profile-{arm}-s{s}']['hits'],archives[f'p-seqm1-mean-s{s}']['hits']) for s in SEEDS}}
  profile,pin=r.read(req['profile_manifest'],req['profile_sha256']);profiles,_=r.read(Path(req['profile_manifest']).parent/'profiles.json',profile['artifacts']['profiles.json'])
  out.update(comparisons=reports,runs={rid:x['summary'] for rid,x in archives.items()},validation_profile_missing={'no_train_events':sum(x['counts'][0]==0 for x in profiles['validation']),'undefined_H50_like_rate':sum(x['counts'][3]==0 for x in profiles['validation'])},limitations=['pooled seeds share queries and are not independent; p_pool optimistic','H50==50 includes histories with exactly50; cannot isolate >50','no test; no checkpoint; no automatic promotion or anchor change'])
 out['artifacts_sha256']=r.READS;return out

def pack_side(h,s):
 import torch
 ids=torch.stack((torch.arange(len(h))%17,torch.arange(len(h))%17,torch.arange(len(h))%10),dim=-1).to(s.device)
 return torch.cat((s,ids[:,None,:].expand(-1,s.shape[1],-1)),dim=-1)

def latency(req,preflight=False):
 # Reuse the established source-checker and immutable real-length timing functions.
 from analysis.postbatch_latency import checked_source
 check={**req,'stack_source':req['profile_source']};m=checked_source(check);source=m.load_lengths(req['lengths'],req['lengths_sha256'],2520)
 out={'status':'latency_preflight_ok' if preflight else 'latency_measured','scope':'model.users() only; not end-to-end','test_read':False,'checkpoint_read':False,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'length_source':{k:v for k,v in source.items() if k!='lengths'},'source':{'profile_source':req['profile_source'],'bench_source':req['bench_source']}}
 if preflight:return out
 sys.path.insert(0,str(Path(req['profile_source']['root'])/'src'))
 import torch
 from vkgr.training import seed_all
 from vkgr.retrieval_seqm1 import make_seqm1_model
 from vkgr.retrieval_profile import make_profile_model
 from vkgr.profile_io import input_guard
 torch.set_num_threads(1)
 if not torch.cuda.is_available():raise ValueError('actual CUDA required')
 original=m.synthetic_batch
 def packed(*args,**kw):
  h,s=original(*args,**kw);return h,pack_side(h,s)
 m.synthetic_batch=packed
 class MeanInputs(torch.nn.Module):
  def __init__(self,base):super().__init__();self.base=base
  def users(self,h,s):return self.base.users(h,s[...,:6])
 reports={}
 try:
  with input_guard():
   for device in ('cpu','cuda'):
    models={}
    for name in ('mean','u2','u3'):
     seed_all(42);content=torch.zeros(120416,64);supported=list(range(1,21497));base=make_seqm1_model(content,supported,50,'mean') if name=='mean' else make_profile_model(content,supported,50,name);model=MeanInputs(base) if name=='mean' else base;model=model.to(device).eval()
     report=m.measure(name,model,device,120416,512,1000,50,source,7);report['validation_export_pass']=m.export_pass(model,source['lengths'],device,120416,512,7);models[name]=report;del model,base
    for mode in ('offline','online'):
     f=mode+'_ms_per_query'
     if len({r[f]['input_sha256'] for r in models.values()})!=1 or any((r[f]['timed_calls'],r[f]['warmup_calls'])!=(1000,50) for r in models.values()):raise ValueError('unpaired timing inputs/counts')
    reports[device]={'threads':1,'models':models,'ratio_vs_mean':{a:{mode:models[a][mode+'_ms_per_query']['mean']/models['mean'][mode+'_ms_per_query']['mean'] for mode in ('offline','online')} for a in ('u2','u3')}}
 finally:m.synthetic_batch=original
 out.update(reports=reports,torch=torch.__version__,limitations='initialized weights/synthetic features with fixed real history lengths; profile IDs pre-gathered outside timer; mean adapter strips last3 columns; no learned checkpoint/feature fetch/item tower/retrieval/network; not end-to-end')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--parameters',required=True);p.add_argument('--parameters-sha',required=True);p.add_argument('--mode',choices=('paired','latency'),required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();raw=Path(a.parameters).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=a.parameters_sha:raise ValueError('request SHA mismatch')
 out=(paired if a.mode=='paired' else latency)(json.loads(raw),a.preflight);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False)
 print(json.dumps({'status':out['status'],'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report':out},sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
