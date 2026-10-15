"""Read only the fixed M1 engineering queue, terminal and result receipts."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path('/workspace/vkgr-infra01-root')
rid='p-seqm1-cuda-engineering-v1'
out={'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':rid,'queue':{},'files':{}}
for kind in ('pending','claimed','rejected'):
    out['queue'][kind]=[]
    for p in (root/'queue'/kind).glob('*-'+rid+'.json'):
        row={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        if kind=='rejected':
            row['reasons']={str(q):q.read_text() for q in p.parent.glob(p.name+'*reason*')}
            for q in (p.with_suffix('.reason'),Path(str(p)+'.reason')):
                if q.exists():row['reasons'][str(q)]=q.read_text()
        out['queue'][kind].append(row)
for rel in ('terminal.json','status.json','out/rehearsal/engineering.json','logs/stdout.log','logs/stderr.log'):
    p=root/'runs'/rid/rel
    if p.exists():
        data=p.read_bytes()
        out['files'][rel]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),
                          'content':json.loads(data) if rel.endswith('.json') else data.decode(errors='replace')[-14000:]}
print(json.dumps(out,sort_keys=True))
