"""Independent, bounded raw-receipt readback; no writes, weights or arrays.
Scientific deployment remains fixed; this later reader does not redeploy it.
"""
import datetime,hashlib,json,os
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root');OPS=Path('/workspace/vkgr-sid-coll-ops-v1')
COMMIT='079d6bd35ae74317b29408cd43d3ec35de77c1eb'
STAGES=('input_preflight','input','input_verify','sid_fit','sid_verify','preflight','cuda','dry','train42','verify42','train2024','verify2024','train2025','verify2025')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path,pin=None):
 p=Path(path)
 if p.suffix not in ('.json','.log','.reason') or '..' in p.parts or not p.is_absolute() or any(x.is_symlink() for x in (p,*p.parents)):
  raise ValueError('bounded metadata only; no checkpoint/array/parquet reads')
 if p.stat().st_size>32*1024*1024:raise ValueError('bounded metadata limit')
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if pin is not None and sha!=pin:raise ValueError('metadata SHA mismatch')
 return json.loads(b) if p.suffix=='.json' else b.decode(),{'path':str(p),'sha256':sha,'read_at_utc':now()}

def accept(terminal,claimed,rejected,receipt,expected,raw_status=None):
 if terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0 or len(claimed)!=1 or rejected or not receipt or not expected:return False
 if receipt.get('schema')!='coll-stage-receipt-v1' or receipt.get('run_id')!=expected['run_id'] or receipt.get('binding',{}).get('commit')!=COMMIT or receipt['binding'].get('spec_sha256')!=expected['spec_sha256']:return False
 if receipt.get('status')=='verified' and raw_status not in ('verified','preflight_ok','cuda_engineering_verified','dry_run_ok'):return False
 return receipt.get('status') in ('verified','complete','produced')

def result_fields(raw):
 return {k:v for k,v in raw.items() if k in ('Dprime','aliases','epochs','epoch_runs','read_at_utc','task_id','restrictions')}

def run(root=R,ops=OPS):
 root=Path(root);ops=Path(ops);result={'read_at_utc':now(),'scientific_commit':COMMIT,'runs':{},'checkpoint_read':False,'test_read':False}
 ids=['sid-coll-'+v+'-'+s+'-v1' for v in ('c1','c2') for s in STAGES]+['sid-coll-s0-prime-v1','sid-coll-legacy-epochs-v1']
 for rid in ids:
  q={k:sorted((root/'queue'/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')};value={'queue':{k:[p.name for p in ps] for k,ps in q.items()},'accepted':False,'state':'UNKNOWN','reasons':[]};result['runs'][rid]=value
  for p in q['rejected']:
   for reason in (p.with_suffix('.reason'),Path(str(p)+'.reason')):
    if reason.exists():value['reasons'].append(read(reason)[0])
  rd=root/'runs'/rid
  if not (rd/'terminal.json').exists():
   value['state']='RUNNING_OR_QUEUED' if q['claimed'] or q['pending'] else 'NOT_SUBMITTED';continue
  terminal,tref=read(rd/'terminal.json');value.update(terminal=terminal,terminal_ref=tref)
  if (rd/'out/receipt.json').exists():
   receipt,ref=read(rd/'out/receipt.json');value.update(receipt_ref=ref,receipt_status=receipt.get('status'),raw_proof_status=receipt.get('raw_proof',{}).get('status'),binding=receipt.get('binding'),artifacts=receipt.get('proofs'),summary=receipt.get('summary'))
   raw_status=receipt.get('raw_proof',{}).get('status');expected=None
   if len(q['claimed'])==1:
    job,jref=read(q['claimed'][0]);a=job['argv'];p=a[a.index('--spec')+1];pin=a[a.index('--spec-sha')+1];read(p,pin)
    expected={'run_id':rid,'spec_sha256':pin};value['claimed_ref']=jref
   proof=receipt.get('proofs',{}).get('verification')
   if proof:
    raw,rref=read(proof['path'],proof['sha256']);raw_status=raw.get('status');value['raw_verification']=raw;value['raw_verification_ref']=rref
   value['accepted']=accept(terminal,q['claimed'],q['rejected'],receipt,expected,raw_status=raw_status)
   value['state']='ACCEPTED' if value['accepted'] else 'UNACCEPTED'
  elif (rd/'out/diagnostic/result.json').exists():
   raw,ref=read(rd/'out/diagnostic/result.json');value['result_ref']=ref
   value['result']=result_fields(raw)
   value['accepted']=terminal.get('status')=='succeeded' and terminal.get('exit_code')==0 and len(q['claimed'])==1 and not q['rejected'];value['state']='ACCEPTED_READONLY' if value['accepted'] else 'UNACCEPTED'
  else:value['state']='FAILED' if terminal.get('status')!='succeeded' else 'UNACCEPTED'
  if terminal.get('status')!='succeeded' and (rd/'logs/stderr.log').exists():value['stderr_tail']=read(rd/'logs/stderr.log')[0][-4000:]
 pidfile=ops/'controller/driver-pid.json'
 if pidfile.exists():
  p,ref=read(pidfile);p['reference']=ref
  try:
   os.kill(p['pid'],0);p['alive']=True;stat=Path('/proc')/str(p['pid'])/'stat';p['actual_ppid']=int(stat.read_text().rsplit(')',1)[1].split()[1]) if stat.exists() else 'UNKNOWN'
  except ProcessLookupError:p['alive']=False
  result['driver']=p
 summaries=sorted((ops/'controller').glob('summary-*.json'))
 if summaries:
  value,ref=read(summaries[-1]);result['latest_summary_ref']=ref;result['driver_status']={k:value.get(k) for k in ('status','event_wait','deadline_epoch')}
 return result
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,allow_nan=False))
