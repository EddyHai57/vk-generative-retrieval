"""Bounded queue alias and epoch metadata reads. Never opens model artifacts.

Plan: root; targets {registered_run_id: {path,sha256}}; epoch_manifests
{run_id: {path,sha256}}. --preflight emits only manifest schema/one epoch.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
try:
    from . import sid_diag as source
    from .sid_diag_prime import write_result
except ImportError:
    import sid_diag as source
    from sid_diag_prime import write_result
READS={}
MAX_BYTES=2*1024*1024


def safe_bytes(path,pin=None):
    path=source.safe_path(str(path))
    if any(any(w in part.lower() for w in ('test','checkpoint','embedding','weights')) for part in path.parts):raise ValueError('prohibited metadata path')
    if path.stat().st_size>MAX_BYTES:raise ValueError('bounded metadata required')
    payload=path.read_bytes();sha=hashlib.sha256(payload).hexdigest()
    if pin is not None and (not re.fullmatch('[0-9a-f]{64}',pin) or sha!=pin):raise ValueError('SHA mismatch')
    READS[str(path)]={'sha256':sha,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return payload


def safe_json(path,pin=None):
    if Path(path).suffix!='.json':raise ValueError('JSON metadata required')
    return json.loads(safe_bytes(path,pin))


def flag(argv,name):
    if not isinstance(argv,list) or any(not isinstance(v,str) for v in argv):raise ValueError('invalid argv')
    if argv.count(name)!=1:return None
    idx=argv.index(name)
    return argv[idx+1] if idx+1<len(argv) else None


def verified_record(record,pin):
    if not isinstance(record,dict) or record.get('status')!='verified':return False
    pins=[record[k] for k in ('manifest_sha256','run_manifest_sha256') if k in record]
    return all(p==pin for p in pins)


def find_aliases(root,targets):
    root=Path(root);result={rid:{'status':'UNKNOWN','aliases':[]} for rid in targets}
    for folder in ('pending','claimed','rejected'):
        for path in sorted((root/'queue'/folder).glob('*.json')):
            job=safe_json(path)
            if not isinstance(job,dict):continue
            argv=job.get('argv',[]);target=flag(argv,'--manifest');pin=flag(argv,'--manifest-sha')
            for rid,ref in targets.items():
                if target!=ref['path'] or pin!=ref['sha256']:continue
                alias=job.get('run_id')
                if not isinstance(alias,str) or not re.fullmatch('[a-zA-Z0-9_-]+',alias):raise ValueError('unsafe alias run ID')
                rd=root/'runs'/alias;terminal={};records=[]
                if (rd/'terminal.json').exists():terminal=safe_json(rd/'terminal.json')
                for name in ('verify.json','manifest.json'):
                    p=rd/'out'/name
                    if p.exists():records.append(safe_json(p))
                stdout=rd/'logs/stdout.log'
                if stdout.exists():
                    for line in safe_bytes(stdout).decode('utf-8').splitlines():
                        try:record=json.loads(line)
                        except ValueError:continue
                        if isinstance(record,dict) and record.get('status')=='verified':records.append(record)
                accepted=folder=='claimed' and terminal.get('status')=='succeeded' and terminal.get('exit_code')==0 and any(verified_record(r,pin) for r in records)
                result[rid]['aliases'].append({'run_id':alias,'queue_state':folder,'queue_path':str(path),'target_binding_matches':True,'terminal_status':terminal.get('status'),'exit_code':terminal.get('exit_code'),'verified':accepted})
                if accepted:result[rid]['status']='verified'
    return result


METRICS={'recall@50','ndcg@50','hits@50','R50','N50','hits','recall_50','ndcg_50','recall_at_50','ndcg_at_50','hits_at_50'}
def metric_leaves(value,prefix=''):
    result={}
    if isinstance(value,dict):
        for key,v in value.items():
            name=prefix+'.'+key if prefix else key
            if key in METRICS and type(v) in (int,float) and math.isfinite(v):result[name]=v
            elif isinstance(v,dict):result.update(metric_leaves(v,name))
    return result


def epoch_summary(manifest):
    selection=manifest.get('selection',{});entries=manifest.get('epochs',[])
    if not isinstance(entries,list):raise ValueError('epochs must be list')
    rows=[]
    for entry in entries:
        if not isinstance(entry,dict):raise ValueError('epoch entry must be dict')
        # G6 tok2 writer stores aggregate at entry.metrics; evaluation holds refs only.
        aggregate=entry.get('metrics',{})
        metrics=metric_leaves(aggregate)
        if not metrics:metrics=metric_leaves(entry.get('evaluation',{}))
        main=aggregate.get('metrics',{}).get('all_eligible',{})
        r,n,c=main.get('recall@50'),main.get('ndcg@50'),main.get('count')
        selection_metrics={'R50':r,'N50':n,'count':c,'hits':None,'hits_source':'UNKNOWN'}
        if type(c) is int and c>0 and type(r) in (int,float) and math.isfinite(r) and 0<=r<=1:
            raw=r*c;rounded=round(raw)
            if abs(raw-rounded)<1e-7:
                selection_metrics.update(hits=rounded,hits_source='derived_recall_times_count')
        rows.append({'epoch':entry.get('epoch'),'metrics':metrics,'selection_metrics':selection_metrics,'status':'OK' if type(r) in (int,float) and type(n) in (int,float) and math.isfinite(r) and math.isfinite(n) and 0<=r<=1 and 0<=n<=1 and type(c) is int and c>0 and selection_metrics['hits'] is not None else 'UNKNOWN'})
    best=selection.get('best_epoch');stop=selection.get('epoch',manifest.get('stop_epoch'))
    return {'status':'OK' if best is not None and stop is not None and rows and all(r['status']=='OK' for r in rows) else 'UNKNOWN','best_epoch':best,'stop_epoch':stop,'selection':selection,'epochs':rows,'manifest_keys':sorted(manifest)}


def run(plan,out,preflight=False,plan_ref=None):
    if Path(out).exists():raise ValueError('new exclusive output directory required')
    READS.clear();targets=plan.get('targets',{});epochs=plan.get('epoch_manifests',{})
    if set(targets)-source.LEGACY_RECEIPT_GAPS:raise ValueError('only four registered legacy target runs allowed')
    for rid,ref in targets.items():
        if Path(ref['path'])!=Path(plan['root'])/'runs'/rid/'out/manifest.json':raise ValueError('target path must be exact registered run manifest')
        safe_json(ref['path'],ref['sha256'])
    result={'task_id':'SID-FIX-S1-aux','plan_reference':plan_ref,'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'aliases':find_aliases(plan['root'],targets),'epoch_runs':{}}
    for rid,ref in epochs.items():
        if rid not in {f'g-gtok02-a1-b1024d3-s{s}' for s in (42,2024,2025)}:raise ValueError('only A1B1024 seed manifests allowed')
        m=safe_json(ref['path'],ref['sha256'])
        result['epoch_runs'][rid]=({'manifest_keys':sorted(m),'selection':m.get('selection'),'first_epoch':m.get('epochs',[None])[:1]} if preflight else epoch_summary(m))
    result['read_artifacts_sha256']=dict(READS);result['restrictions']={'checkpoint_read':False,'predictions_read':False,'new_verifier_submitted':False}
    write_result(out,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--preflight',action='store_true')
    args=p.parse_args();plan=safe_json(args.plan);receipt={'path':args.plan,**READS[args.plan]};run(plan,args.out,args.preflight,receipt)
if __name__=='__main__':main()
