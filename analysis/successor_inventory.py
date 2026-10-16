"""One-shot metadata inventory for approved A1 B1024 confirmation and M1 latency.
No tensors, checkpoints, full embeddings or prediction arrays; no writes.
"""
import datetime
import hashlib
import json
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root');Q=R/'queue'
READS={}
def read(p):
 p=Path(p)
 if p.suffix!='.json' or any(k in p.name.lower() for k in ('test','checkpoint','embedding')):raise ValueError('metadata JSON only')
 b=p.read_bytes();READS[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def where(r):
 d={k:sorted(p.name for p in (Q/k).glob('*-'+r+'.json')) for k in ('pending','claimed','rejected')};d['reasons']={}
 for n in d['rejected']:
  for p in ((Q/'rejected'/n).with_suffix('.reason'),Q/'rejected'/(n+'.reason')):
   if p.exists():d['reasons'][p.name]=p.read_text()[:1000]
 return d
def main():
 out={'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'templates':{},'future':{}}
 for seed in (42,2024,2025):
  rid=f'g-gtok01b-t3-b256d3-s{seed}';fs=list((Q/'claimed').glob('*-'+rid+'.json'))
  if len(fs)!=1:raise ValueError('unique A1 control template required')
  job=read(fs[0]);manifest=read(R/'runs'/rid/'out/manifest.json')
  out['templates'][rid]={'job':job,'config':read(job['config']),'binding':manifest['binding'],'selection':manifest['selection']}
 fit=R/'runs/g-tok2-b1024d3-fit/out/run_manifest.json';out['sid_manifest_path']=str(fit);out['sid_manifest']=read(fit);out['sid_manifest_sha256']=READS[str(fit)]
 out['sid_verify']=read(R/'runs/g-tok2-b1024d3-fit-verify/out/verify.json')
 for rid in [*(f'g-gtok02-a1-b1024d3-s{s}' for s in (42,2024,2025)),'g-gtok02-a1-b1024d3-cuda-v1','p-seqm1-latency-report-v1']:
  for r in (rid,rid+'-verify'):out['future'][r]={'queue':where(r),'run_exists':(R/'runs'/r).exists()}
 terms=[]
 for p in (R/'runs').glob('*/terminal.json'):
  v=read(p)
  if isinstance(v.get('ended_at'),(int,float)):terms.append({'run_id':p.parent.name,'ended_at':v['ended_at'],'status':v['status']})
 out['last_project_terminal']=max(terms,key=lambda v:v['ended_at']) if terms else None;out['read_artifacts_sha256']=READS;out['test_read']=False;out['new_submit']=False
 print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
