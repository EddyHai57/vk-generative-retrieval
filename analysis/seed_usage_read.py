"""One-shot queue/run seed usage audit; registered JSON metadata only."""
import datetime,hashlib,json,sys
from pathlib import Path
SEEDS={314159,271828,161803}

def scan(root):
 root=Path(root);matches=[];errors=[];count=0;reads={};queues=0;runs=0
 files=[]
 for state in ('pending','claimed','rejected'):
  directory=root/'queue'/state
  if directory.exists():files.extend(sorted(directory.glob('*.json')))
 queues=len(files)
 if not (root/'runs').is_dir():errors.append('run directory missing')
 for rd in sorted((root/'runs').iterdir()) if (root/'runs').is_dir() else []:
  if not rd.is_dir():continue
  runs+=1
  for parent in (rd,rd/'out'):
   for name in ('job.json','run.json','identity.json','manifest.json','run_manifest.json','config.json'):
    p=parent/name
    if p.exists():files.append(p)
 for p in files:
  try:
   if p.is_symlink() or p.stat().st_size>16*1024*1024:raise ValueError('unsafe metadata')
   raw=p.read_bytes();d=json.loads(raw);count+=1;sha=hashlib.sha256(raw).hexdigest();reads[str(p)]=sha
   if not isinstance(d,dict):continue
   candidates=[d.get('seed')]
   for key in ('config','job','training','tokenizer'):
    sub=d.get(key)
    if isinstance(sub,dict):candidates.append(sub.get('seed'))
   for seed in sorted(set(v for v in candidates if type(v)is int) & SEEDS):matches.append(dict(path=str(p),sha256=sha,seed=seed,run_id=d.get('run_id',p.parent.name)))
  except (OSError,ValueError) as exc:errors.append({'path':str(p),'error':str(exc)})
 return dict(status='INCOMPLETE' if errors else 'COLLISION' if matches else 'CLEAR',seeds=sorted(SEEDS),matches=matches,errors=errors,read_files=count,queue_records=queues,run_directories=runs,read_artifacts_sha256=reads,scope='all queue JSON plus per-run root/out named job/run/identity/config/manifest/run_manifest JSON; no data or checkpoint files',test_data_read=False,checkpoint_read=False,server_write=False)

def main():
 req=json.loads(sys.stdin.read(8193));root=Path(req['root'])
 if set(req)!={'root'} or not root.is_absolute():raise ValueError('private absolute root required')
 result=scan(root);result['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
