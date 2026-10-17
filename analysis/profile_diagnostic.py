"""One authorized post-close validation-only diagnostic; aggregate outputs only.

No model/checkpoint import, fit, train, decoding, new candidate depth or test I/O.
Missing ranks are right-censored at50, never imputed as51.
"""
import argparse,datetime,hashlib,importlib.util,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import postbatch_acceptance as prior
from analysis.profile_readout import raw_verified
FSEEDS=(42,2024,2025);BSEEDS=(314159,271828,161803)
def registered():return {f'{stem}-s{s}' for stem,seeds in [('p-profile-u2',FSEEDS),('p-profile-u3',FSEEDS),('p-seqm1-mean',FSEEDS),('p-biconfirm-bi',BSEEDS),('p-biconfirm-mean',BSEEDS)] for s in seeds}
def compare(new,control,targets):
 if not new or not len(new)==len(control)==len(targets):raise ValueError('same nonempty ordered validation queries required')
 if any(len(row)!=50 or len(set(row))!=50 or any(type(i) is not int for i in row) for row in new+control):raise ValueError('unique integer top50 archive required')
 hist={str(k):0 for k in range(51)};same_order=0;groups={k:[] for k in ('only_new','only_control')};deltas=[]
 for a,b,target in zip(new,control,targets):
  overlap=len(set(a)&set(b));hist[str(overlap)]+=1;same_order+=a==b;deltas.append(overlap)
  ha=target in a;hb=target in b
  if ha and not hb:groups['only_new'].append(a.index(target)+1)
  if hb and not ha:groups['only_control'].append(b.index(target)+1)
 ordered=sorted(deltas);n=len(new);out={'queries':n,'overlap_histogram':hist,'mean_overlap':sum(deltas)/n,'min_overlap':ordered[0],'max_overlap':ordered[-1],'overlap_percentiles_nearest_rank':{k:ordered[math.ceil(q*n)-1] for k,q in [('p05',.05),('p50',.5),('p95',.95)]},'identical_set_queries':hist['50'],'identical_order_queries':same_order}
 for label,ranks in groups.items():out[label]={'queries':len(ranks),'hit_side_rank_histogram':{str(i):ranks.count(i) for i in range(1,51)},'hit_side_rank45_50_count':sum(i>=45 for i in ranks),'hit_side_share_rank45_50':sum(i>=45 for i in ranks)/len(ranks) if ranks else None,'miss_side_exact_rank':'UNKNOWN (>50)','miss_side_rank_51_55_count':None,'miss_side_rank_above55_count':None}
 return out

def buckets(users,ids):
 if not users or len(users)!=len(ids) or any(type(u) is not int for u in users):raise ValueError('query/user grain mismatch')
 limits=(17,17,10);byuser={}
 for user,row in zip(users,ids):
  if len(row)!=3 or any(type(v) is not int or not 0<=v<n for v,n in zip(row,limits)):raise ValueError('bucket domain')
  if user in byuser and byuser[user]!=row:raise ValueError('fixed validation snapshot differs for same user')
  byuser[user]=row
 fields={}
 for j,(name,size) in enumerate(zip(('N_exp','N_like','H50_like_rate'),limits)):
  fields[name]={'query_counts':{str(k):sum(row[j]==k for row in ids) for k in range(size)},'user_counts':{str(k):sum(row[j]==k for row in byuser.values()) for k in range(size)},'occupied_buckets':sorted({row[j] for row in ids}),'constant_on_validation':len({row[j] for row in ids})==1}
 return {'queries':len(users),'distinct_users':len(byuser),'fields':fields,'H50_like_rate_zero_bucket':0,'H50_like_rate_undefined_bucket':9,'scope':'stored validation sidecar only; nonconstant buckets do not prove learned vector magnitude or runtime contribution'}

def execute(req,preflight=False):
 r=prior.load_phase(req['phase2']);r.ROOT=Path(req['root'])
 if r.ROOT!=Path('/workspace/vkgr-infra01-root') or set(req['manifests'])!=registered() or set(req['identities'])!=registered():raise ValueError('exact authorized15 runs/root required')
 hist,_=r.read(req['history'],r.HISTORY_PIN);queries,_=r.read(Path(req['history']).parent/'validation-l50.json',hist['artifacts']['validation-l50.json'])
 if len(queries)!=2520 or sum(q['oov'] for q in queries)!=623:raise ValueError('frozen validation population')
 states={};manifests={}
 for rid in sorted(registered()):
  state=r.inspect_run(rid);pin=req['manifests'][rid]
  if not state['verified'] or state.get('manifest_sha256')!=pin or state['queue']['rejected'] or len(state['queue']['claimed'])!=1:raise ValueError('original verifier/manifest mismatch: '+rid)
  m,_=r.read(r.ROOT/'runs'/rid/'out/manifest.json',pin);job,_=r.read(r.ROOT/'queue/claimed'/state['queue']['claimed'][0]);cfg,_=r.read(job['config']);expected=req['identities'][rid]
  prior.check_job(rid,m,job,cfg,expected['deployment'],expected['binding_sha256']);prior.check_inputs(m,hist,r.HISTORY_PIN)
  if m['binding'].get('synthetic') is not False or m['binding'].get('test_gate')!='closed_hash_only':raise ValueError('real validation-only binding')
  verifiers=[v for v in state['verify']['jobs'] if v['verified']]
  if len(verifiers)!=1:raise ValueError('unique verifier')
  log=r.ROOT/'runs'/verifiers[0]['run_id']/'logs/stdout.log';raw=log.read_bytes();records=[]
  for line in raw.decode().splitlines():
   try:records.append(json.loads(line))
   except ValueError:pass
  raw_verified(records,rid,pin);r.READS[str(log)]=hashlib.sha256(raw).hexdigest();states[rid]=state;manifests[rid]=m
 profile,pin=r.read(req['profile_manifest'],req['profile_sha256']);profiles,_=r.read(Path(req['profile_manifest']).parent/'profiles.json',profile['artifacts']['profiles.json']);bounds,_=r.read(Path(req['profile_manifest']).parent/'quantiles.json',profile['artifacts']['quantiles.json'])
 if profile.get('status')!='complete' or len(profiles['validation'])!=2520:raise ValueError('stored validation sidecar mismatch')
 for rid,m in manifests.items():
  if rid.startswith('p-profile-') and (m['binding']['profile_manifest_sha256']!=pin or m['binding']['profile_fit_sha256']!=profile['artifacts']['quantiles.json']):raise ValueError('run sidecar/quantile binding mismatch')
 producer=Path(req['profile_producer']);code=producer.read_bytes()
 if producer.is_symlink() or hashlib.sha256(code).hexdigest()!=profile['profile_script_sha256']:raise ValueError('frozen pure feature producer mismatch')
 r.READS[str(producer)]=profile['profile_script_sha256'];spec=importlib.util.spec_from_file_location('frozen_profile_diagnostic_producer',producer);pf=importlib.util.module_from_spec(spec);spec.loader.exec_module(pf)
 attached=pf.attach(queries,profiles['validation'],bounds,'validation')
 out={'status':'preflight_ok' if preflight else 'diagnostic_complete','engineering':False,'test_read':False,'checkpoint_read':False,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank_scope':'only stored top50; rank51-55 and >55 on missed side UNKNOWN; no decoding/export','population':{'queries':2520,'structural_oov':623},'manifest_pins':req['manifests']}
 if not preflight:
  predictions={}
  for rid,state in states.items():
   r.load_archive(rid,queries,hist,state) # Original best archive integrity/R/N checks.
   m=manifests[rid];epoch=m['selection']['best_epoch'];ev=next(x for x in m['evaluations'] if x['epoch']==epoch);pred,_=r.read(r.ROOT/'runs'/rid/'out'/ev['predictions_ref'],ev['predictions_sha256']);predictions[rid]=pred
  out['comparisons']={label:{str(s):compare(predictions[new+'-s'+str(s)],predictions[ctl+'-s'+str(s)],[q['target'] for q in queries]) for s in seeds} for label,new,ctl,seeds in [('u2_vs_mean','p-profile-u2','p-seqm1-mean',FSEEDS),('u3_vs_mean','p-profile-u3','p-seqm1-mean',FSEEDS),('bi_vs_mean_reference','p-biconfirm-bi','p-biconfirm-mean',BSEEDS)]}
  out['validation_buckets']=buckets([q['user_id'] for q in queries],[q['profile_ids'] for q in attached]);out['limitations']=['reference bi uses separately preregistered seeds; descriptive comparator, not cross-experiment paired test','nonconstant stored IDs and verified bindings do not prove nonzero learned p/validation contribution','no checkpoint/vector norm; mechanism(a)/(b)/(c) not identifiable solely by this report','no new significance or promotion gate; exact overlap hist/rank hist are descriptive']
 out['artifacts_sha256']=r.READS;return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--parameters',required=True);p.add_argument('--parameters-sha',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();raw=Path(a.parameters).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=a.parameters_sha:raise ValueError('request SHA mismatch')
 out=execute(json.loads(raw),a.preflight);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False)
 print(json.dumps({'status':out['status'],'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report':out},sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
