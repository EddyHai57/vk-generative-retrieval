"""Fixed-run engineering diagnostic: metadata and bounded owned logs only."""
import datetime
import json
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root')
RID='g-gtok02-a1-b1024d3-cuda-v1'
def main():
 run=R/'runs'/RID
 result={'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':RID,'test_read':False,'new_submit':False,'files':{},'metadata':{},'logs':{}}
 if run.exists():
  for p in run.rglob('*'):
   if p.is_file():result['files'][str(p.relative_to(run))]={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime}
  for rel in ['terminal.json','launch.json','worker.json','supervisor.json','registration.json','status.json','job.json']:
   p=run/rel
   if p.exists() and p.stat().st_size<100000:result['metadata'][rel]=json.loads(p.read_text())
  for rel in ['logs/stdout.log','logs/stderr.log','logs/supervisor.log','logs/launcher.log']:
   p=run/rel
   if p.exists():
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));result['logs'][rel]=f.read().decode(errors='replace')
 for p in (R/'queue').glob('*/*'+RID+'*'):
  if p.is_file() and p.stat().st_size<100000:result.setdefault('queue',{})[str(p.relative_to(R/'queue'))]=p.read_text()
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
