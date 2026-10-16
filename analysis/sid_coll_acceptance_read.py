"""Independent, bounded raw-receipt readback; no writes, weights or arrays.
Scientific deployment remains fixed; this later reader does not redeploy it.
"""
import datetime,hashlib,json,os,math
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

def producer_fields(value):
 result={k:value[k] for k in ('status','variant','dimension','catalog_sha256','vectors_sha256','content_feature_sha256','train_sequence_sha256','checkpoint_sha256','script_sha256','commit') if k in value}
 allowed=('config','epochs_completed','positive_pairs_per_epoch','processed_positive_pairs','steps','learning_rates','fallback_count','fallback_proportion','health','training_contract','supported_items','supported_fraction','unsupported_items','supported_zero_items','collaborative_block_items','content_only_items','limitations')
 result['audit']={k:v for k,v in value.get('audit',{}).items() if k in allowed}
 result['vectors_digest_kind']='canonical float32 array digest, distinct from physical .npy SHA'
 result['vectors_file_sha256']=value.get('artifacts',{}).get('vectors.npy')
 return result

def epoch_rows(manifest):
 rows=[]
 for value in manifest.get('epochs',[]):
  a=value.get('metrics',{}).get('metrics',{}).get('all_eligible',{});n=a.get('count');r=a.get('recall@50');h='UNKNOWN'
  if type(n) is int and n>0 and type(r) in (int,float) and math.isfinite(r) and abs(r*n-round(r*n))<1e-7:h=round(r*n)
  rows.append({'epoch':value.get('epoch'),'R50':r,'N50':a.get('ndcg@50'),'count':n,'hits_derived':h})
 return {'best_epoch':manifest.get('selection',{}).get('best_epoch'),'stop_epoch':manifest.get('selection',{}).get('epoch'),'epochs':rows,'hits_method':'derived from exact stored recall times count; descriptive only'}

def small_metadata(root):
 # These are immutable small manifests only; no failed job is rerun.
 first,fref=read(root/'runs/sid-fix-s0-first-v1/out/diagnostic/result.json','a2e93d69bfb79af2c94e344aa5368e7604825777acc3ed6f2f6cde37c5db1924')
 ids=['g-hist-v2-l50-s2025','g-g4-rand-s42','g-g4-rand-s2024','g-g4-rand-s2025']
 targets={rid:first['inventory']['runs'][rid] for rid in ids};aliases={rid:{'status':'UNKNOWN','receipts':[]} for rid in ids};skipped=0
 for q in sorted((root/'queue/claimed').glob('*.json')):
  # Avoid even incidental historical test-named metadata; never open test artifacts.
  if 'test' in q.name.lower():skipped+=1;continue
  job,jref=read(q);argv=job.get('argv',[])
  if argv.count('--manifest')!=1 or argv.count('--manifest-sha')!=1:continue
  path=argv[argv.index('--manifest')+1];pin=argv[argv.index('--manifest-sha')+1]
  for rid,target in targets.items():
   if path!=target['manifest_path'] or pin!=target['manifest_sha256']:continue
   alias=job['run_id'];rd=root/'runs'/alias;records=[];terminal={}
   if (rd/'terminal.json').exists():terminal,tref=read(rd/'terminal.json')
   for file in (rd/'out/verify.json',rd/'out/manifest.json',rd/'logs/stdout.log'):
    if not file.exists():continue
    body,bref=read(file)
    candidates=[body] if isinstance(body,dict) else []
    if isinstance(body,str):
     for line in body.splitlines():
      try:candidates.append(json.loads(line))
      except ValueError:pass
    if any(isinstance(x,dict) and x.get('status')=='verified' and all(x.get(k,pin)==pin for k in ('manifest_sha256','run_manifest_sha256')) for x in candidates):records.append(bref)
   found=terminal.get('status')=='succeeded' and terminal.get('exit_code')==0 and bool(records)
   aliases[rid]['receipts'].append({'alias_run_id':alias,'target_manifest_sha256':pin,'claimed':jref,'raw_verified_evidence':records,'terminal':terminal,'verified_receipt_found':found})
   if found:aliases[rid]['status']='verified_receipt_found'
 a1,aref=read(root/'runs/sid-fix-s0-a1b1024-v1/out/diagnostic/result.json','ce587ba5513109652945bf656eba5fcef156f7471daadba79b9ee50be10beb41');epochs={}
 for seed in (42,2024,2025):
  rid='g-gtok02-a1-b1024d3-s'+str(seed);v=a1['inventory']['runs'][rid]
  raw,ref=read(v['manifest_path'],v['manifest_sha256']);epochs[rid]={**epoch_rows(raw),'manifest_ref':ref,'original_verify_recorded':v['verified']}
 return {'legacy_aliases':aliases,'alias_scan_limit':'claimed metadata only; historical test-named queue files skipped','skipped_test_named_metadata':skipped,'a1_epochs':epochs,'immutable_source_refs':[fref,aref]}

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
   producer=receipt.get('proofs',{}).get('producer_manifest')
   if producer:
    pv,pref=read(producer['path'],producer['sha256'])
    if 'audit.json' in pv.get('artifacts',{}):
     audit,aref=read(Path(producer['path']).parent/'audit.json',pv['artifacts']['audit.json']);pv['audit']=audit;value['sgns_audit_ref']=aref
    value['producer_summary']=producer_fields(pv);value['producer_ref']=pref
   value['accepted']=accept(terminal,q['claimed'],q['rejected'],receipt,expected,raw_status=raw_status)
   value['state']='ACCEPTED' if value['accepted'] else 'UNACCEPTED'
  elif (rd/'out/diagnostic/result.json').exists():
   raw,ref=read(rd/'out/diagnostic/result.json');value['result_ref']=ref
   value['result']=result_fields(raw)
   value['accepted']=terminal.get('status')=='succeeded' and terminal.get('exit_code')==0 and len(q['claimed'])==1 and not q['rejected'];value['state']='ACCEPTED_READONLY' if value['accepted'] else 'UNACCEPTED'
  else:value['state']='FAILED' if terminal.get('status')!='succeeded' else 'UNACCEPTED'
  if terminal.get('status')!='succeeded' and (rd/'logs/stderr.log').exists():value['stderr_tail']=read(rd/'logs/stderr.log')[0][-4000:]
 pidfile=ops/'controller/driver-pid-v2.json'
 if not pidfile.exists():pidfile=ops/'controller/driver-pid.json'
 if pidfile.exists():
  p,ref=read(pidfile);p['reference']=ref
  try:
   os.kill(p['pid'],0);p['alive']=True;stat=Path('/proc')/str(p['pid'])/'stat';p['actual_ppid']=int(stat.read_text().rsplit(')',1)[1].split()[1]) if stat.exists() else 'UNKNOWN'
  except ProcessLookupError:p['alive']=False
  result['driver']=p
 summaries=sorted((ops/'controller').glob('summary-*.json'))
 if summaries:
  value,ref=read(summaries[-1]);result['latest_summary_ref']=ref;result['driver_status']={k:value.get(k) for k in ('status','event_wait','deadline_epoch')}
 result.update(small_metadata(root))
 return result
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,allow_nan=False))
