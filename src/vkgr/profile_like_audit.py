"""F7 training-only like audit. Pinned metadata and parquet column `like` only."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

def require(ok,msg):
    if not ok:raise ValueError(msg)
def summarize(values):
    out={'rows':0,'missing':0,'true':0,'false':0,'invalid':0,'invalid_types':{}}
    for v in values:
        out['rows']+=1
        if v is None:out['missing']+=1
        elif type(v) is bool:out['true' if v else 'false']+=1
        else:
            out['invalid']+=1;k=type(v).__name__;out['invalid_types'][k]=out['invalid_types'].get(k,0)+1
    out['missing_rate']=out['missing']/out['rows'] if out['rows'] else None
    return out

def safe(p):
    p=Path(p);require(p.is_absolute() and '..' not in p.parts,'absolute input path')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'symlink input')
    require(not any(re.search(r'(^|[-_])test([._-]|$)',x.lower()) or 'checkpoint' in x.lower() for x in p.parts),'test/model input prohibited')
    require(p.suffix not in ('.pt','.pth','.ckpt') and p.is_file(),'input file required')
    return p

def read(path,pin):
    p=safe(path);data=p.read_bytes();require(hashlib.sha256(data).hexdigest()==pin,'input SHA mismatch');return json.loads(data)

def train_records(history):
    rows=history['binding']['raw_records'];expected=['subsamples/ur0.01_ir0.01/train/week_%02d.parquet'%i for i in range(25)]
    require([r['role'] for r in rows]==expected,'exact train weeks 00-24 only')
    require(all(re.fullmatch('[a-f0-9]{64}',r['sha256']) and type(r['size_bytes']) is int and r['size_bytes']>=0 for r in rows),'train pins')
    return rows

def finalize_audit(result,audit):
    result={**result,'audit':audit}
    if audit['rows']==0 or audit['invalid']:
        result.update(status='STOP',reason='invalid like domain or empty train; no retry')
    return result

def execute(inputs,inputs_sha,preflight=False):
    import pyarrow.parquet as pq
    spec=read(inputs,inputs_sha);history=read(spec['history_manifest'],spec['history_sha256']);side=read(spec['side_manifest'],spec['side_sha256'])
    require(side['fields']==['feedback_mask','timespent','duration'] and side['feedback_bits'][0]=='like' and side['history_manifest_sha256']==spec['history_sha256'],'STOP: prepared/history lacks registered like field; do not extend contract')
    reads={str(inputs):inputs_sha,spec['history_manifest']:spec['history_sha256'],spec['side_manifest']:spec['side_sha256']};schemas=[];files=[]
    for record in train_records(history):
        p=safe(Path(spec['raw_root'])/record['role']);require(p.stat().st_size==record['size_bytes'],'raw size mismatch');h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        require(h.hexdigest()==record['sha256'],'raw SHA mismatch');reads[str(p)]=h.hexdigest();parquet=pq.ParquetFile(p)
        require('like' in parquet.schema_arrow.names,'STOP: like column absent; no contract extension')
        schemas.append(str(parquet.schema_arrow.field('like').type));files.append(p)
    result={'status':'like_audit_preflight_ok' if preflight else 'like_audit_complete','read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_read':False,'checkpoint_read':False,'u1_read':False,'columns_read':['like'],'prepared_like_source':'HIST_V2_SIDE_V1 feedback_mask bit0','side_manifest_sha256':spec['side_sha256'],'raw_schema_types':sorted(set(schemas)),'input_sha256':reads}
    if not preflight:
        def values():
            for p in files:
                for batch in pq.ParquetFile(p).iter_batches(batch_size=65536,columns=['like']):yield from batch.column(0).to_pylist()
        result=finalize_audit(result,summarize(values()))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--inputs-sha',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();result=execute(a.inputs,a.inputs_sha,a.preflight);print(json.dumps(result,sort_keys=True,allow_nan=False));return 2 if result['status']=='STOP' else 0
if __name__=='__main__':raise SystemExit(main())
