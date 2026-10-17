"""Post-batch registered paired analysis; metadata-only preflight, no model imports.

Only existing validation predictions are read for paired analysis. Curve helpers
consume metric tuples only. Private absolute paths are supplied in pinned JSON.
"""
import argparse, datetime, hashlib, json, math, sys, types
from pathlib import Path
PHASE_SHA='548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8'
PAIRS=(('stack_vs_m1','p-seq-combo-stack','p-seqm1-mean',(42,2024,2025)),
       ('bi_vs_mean','p-biconfirm-bi','p-biconfirm-mean',(314159,271828,161803)))

def exact(b,c):
 if any(type(x) is not int or x<0 for x in (b,c)):raise ValueError('nonnegative integer discordants required')
 return min(1.,2*sum(math.comb(b+c,k) for k in range(min(b,c)+1))/2**(b+c))

def pool(rows,seeds):
 if set(rows)!={str(s) for s in seeds} or len(seeds)!=3:raise ValueError('registered seed set mismatch')
 for r in rows.values():
  exact(r['only_new'],r['only_control'])
  if r['delta_hits']!=r['only_new']-r['only_control']:raise ValueError('discordants inconsistent')
 b=sum(r['only_new'] for r in rows.values());c=sum(r['only_control'] for r in rows.values())
 return {'only_new':b,'only_control':c,'p_pool':exact(b,c),'mean_delta_hits':sum(r['delta_hits'] for r in rows.values())/3,
         'seeds_ge6':sum(r['delta_hits']>=6 for r in rows.values()),'seeds_p_lt_005':sum(r['mcnemar_exact_two_sided']<.05 for r in rows.values()),
         'independent_observations':False,'limitation':'same queries across seeds; pooled p is optimistic, required gate not evidence strength'}

def plateaus(rows):
 if not rows or any(e!=i+1 or not all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in (r,n)) for i,(e,r,n) in enumerate(rows)):
  raise ValueError('contiguous epochs and finite R/N required')
 flat=[];lex=[];flat_start=1;best=(rows[0][1],rows[0][2]);best_epoch=1;bad=0
 for i,(epoch,r,n) in enumerate(rows[1:],1):
  prev=rows[i-1][1]
  if r!=prev:
   if epoch-flat_start>=3 and r>prev:flat.append({'start':flat_start,'end':epoch-1,'rise':epoch})
   flat_start=epoch
  if (r,n)>best:
   if bad>=3:lex.append({'best':best_epoch,'start':best_epoch+1,'end':epoch-1,'rise':epoch})
   best=(r,n);best_epoch=epoch;bad=0
  else:bad+=1
 return {'three_flat_recall_then_up':flat,'three_no_new_lex_best_then_up':lex,'terminal_no_new_best':bad,
         'right_censored_after_stop':True,'unobserved_post_stop_recovery':'UNKNOWN'}

def load_phase(path):
 p=Path(path)
 if p.is_symlink() or p.suffix!='.py' or p.stat().st_size>1000000:raise ValueError('pinned source required')
 raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PHASE_SHA:raise ValueError('frozen phase reader SHA mismatch')
 m=types.ModuleType('pinned_postbatch_phase');exec(compile(raw,str(p),'exec'),m.__dict__)
 m.READS[str(p)]=PHASE_SHA;return m

def plain_digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def check_job(rid,manifest,job,config,deployment,binding_pin):
 if job.get('run_id')!=rid or job.get('seed')!=manifest['config']['seed'] or job.get('cwd')!=deployment or config!=manifest['config']:
  raise ValueError('claimed config/seed/deployment mismatch')
 if plain_digest(manifest['binding'])!=binding_pin:raise ValueError('frozen source/config/data binding mismatch')

def check_inputs(manifest,hist,history_pin):
 b=manifest['binding'];i=b['inputs']
 if i.get('history_manifest_sha256')!=history_pin or i.get('query_sha')!=hist['binding']['query_sha'] or i.get('catalog_sha')!=hist['binding']['catalog_sha'] or b.get('catalog_sha256')!=hist['catalog_sha256']:
  raise ValueError('preflight query/history/catalog binding mismatch')

def pinned_runs(req,r):
 wanted={f'{stem}-s{s}' for _,a,b,seeds in PAIRS for stem in (a,b) for s in seeds}
 if set(req['manifests'])!=wanted:raise ValueError('exact registered run identities required')
 result={}
 for rid in sorted(wanted):
  state=r.inspect_run(rid);pin=req['manifests'][rid]
  if state.get('manifest_sha256')!=pin or not state['verified'] or state['queue']['rejected']:raise ValueError('raw verification/manifest identity mismatch: '+rid)
  if len(state['queue']['claimed'])!=1:raise ValueError('unique claimed train required')
  manifest,_=r.read(r.ROOT/'runs'/rid/'out/manifest.json',pin)
  binding=manifest['binding']
  if manifest['config']['seed']!=int(rid.rsplit('-s',1)[1]) or binding.get('synthetic') is not False or binding.get('test_gate')!='closed_hash_only':raise ValueError('real seed/test gate mismatch')
  for v in state['verify']['jobs']:
   if v['verified']:
    raw=(r.ROOT/'runs'/v['run_id']/'logs/stdout.log').read_bytes()
    lines=[]
    for line in raw.decode().splitlines():
     try:lines.append(json.loads(line))
     except ValueError:pass
    if not any(type(x) is dict and x.get('status')=='verified' for x in lines):raise ValueError('raw stdout verified absent')
    r.READS[str(r.ROOT/'runs'/v['run_id']/'logs/stdout.log')]=hashlib.sha256(raw).hexdigest()
  job,_=r.read(r.ROOT/'queue/claimed'/state['queue']['claimed'][0])
  config,_=r.read(job['config'])
  expected=req['identities'][rid]
  check_job(rid,manifest,job,config,expected['deployment'],expected['binding_sha256'])
  result[rid]=(state,manifest)
 return result

def execute(req,preflight=False):
 r=load_phase(req['phase2']);r.ROOT=Path(req['root']);states=pinned_runs(req,r)
 hist,_=r.read(Path(req['history']),r.HISTORY_PIN)
 qp=Path(req['history']).parent/'validation-l50.json'
 queries,_=r.read(qp,hist['artifacts']['validation-l50.json'])
 cat,catsha=r.read(req['catalog'])
 if len(queries)!=2520 or sum(q['oov'] for q in queries)!=623 or len(cat)!=120415 or len(set(cat))!=120415:raise ValueError('frozen population mismatch')
 if r.plain_digest(cat)!=hist['catalog_sha256'] or catsha!=hist['binding']['catalog_sha']:raise ValueError('catalog identity mismatch')
 catalog=set(cat)
 if any(type(q['oov']) is not bool or q['oov']!=(q['target'] not in catalog) for q in queries):raise ValueError('OOV mismatch')
 hist['_base_queries_canonical_digest']=hist['binding']['query_sha']
 for _,m in states.values():check_inputs(m,hist,r.HISTORY_PIN)
 support={m['binding']['support_sha256'] for _,m in states.values()}
 if len(support)!=1:raise ValueError('shared support mismatch')
 out={'status':'preflight_ok' if preflight else 'paired_complete','test_read':False,'checkpoint_read':False,'preflight_metadata_only':preflight,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 if not preflight:
  data={}
  for rid,(state,m) in states.items():
   data[rid]=r.load_archive(rid,queries,hist,state)
   if any(any(item not in catalog for item in pred) for pred in r.read(r.ROOT/'runs'/rid/'out'/next(x for x in m['evaluations'] if x['epoch']==m['selection']['best_epoch'])['predictions_ref'])[0]):raise ValueError('candidate outside catalog')
  reports={}
  for label,new,ctl,seeds in PAIRS:
   pairs={str(s):r.paired(data[f'{new}-s{s}'],data[f'{ctl}-s{s}']) for s in seeds}
   reports[label]={'seeds':pairs,'pool':pool(pairs,seeds)}
  out['comparisons']=reports
 out['manifest_pins']=req['manifests'];out['artifacts_sha256']=r.READS;return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--parameters',required=True);p.add_argument('--parameters-sha',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--output')
 a=p.parse_args();raw=Path(a.parameters).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=a.parameters_sha:raise ValueError('parameter pin mismatch')
 out=execute(json.loads(raw),a.preflight)
 if a.output:
  path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False)
  print(json.dumps({'status':out['status'],'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
 else:print(json.dumps(out,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
