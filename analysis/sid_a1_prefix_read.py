"""Once-only queued S5 prefix supplement. Reuse immutable S0 B; aggregates only.
No checkpoint, raw event, embedding or model loading. Input request is SHA-pinned.
"""
import argparse
import datetime
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

SEEDS=(42,2024,2025)
RUNS=tuple('g-coll-a1-c1-s'+str(s) for s in SEEDS)
FROZEN={
 'sid_diag.py':'e7db6e37cb00f7c165cfd373da53a4538cfa4e54b84113fbbae3bde6d2523161',
 'sid_diag_metrics.py':'ad966940d682aa0f4e576eeacfece8cef92b2f9597fec4a41522b7e535e55dc4',
 'phase2_validation_read.py':'548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8'}

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def safe(path,roots):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts,'absolute path required')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'symlink forbidden')
    require(not any(re.search(r'(^|[-_])test([._-]|$)',x.lower()) or any(w in x.lower() for w in ('checkpoint','embedding')) for x in p.parts),'forbidden artifact ancestry')
    require(p.suffix not in ('.pt','.pth','.ckpt') and p.is_file(),'no model files')
    require(any(p.is_relative_to(Path(r)) for r in roots),'outside readonly scope')
    return p

def frozen(root):
    root=Path(root)/'analysis'
    for name,pin in FROZEN.items():
        p=root/name
        require(p.is_file() and not p.is_symlink() and sha(p)==pin,'frozen S0 source mismatch')
    sys.path.insert(0,str(root))
    for name in ('phase2_validation_read','sid_diag_metrics','sid_diag'):
        previous=sys.modules.get(name)
        require(previous is None or Path(previous.__file__).resolve()==(root/(name+'.py')).resolve(),'foreign module already loaded')
    diag=importlib.import_module('sid_diag');metrics=importlib.import_module('sid_diag_metrics')
    require(Path(metrics.__file__).resolve()==(root/'sid_diag_metrics.py').resolve(),'B import identity')
    return diag,metrics

def validate_run_ids(ids):
    require(type(ids) is list and ids==list(RUNS),'exact three authorized seeds required')

def aggregate(targets,predictions,mapping,metrics):
    return metrics.prediction_metrics(targets,predictions,mapping)

def population(b,expected_hits,total=2520,in_catalog=1897,oov=623):
    require((b['n_total'],b['n_in_catalog'],b['n_oov'])==(total,in_catalog,oov),'frozen population mismatch')
    require(b['exact_item']['top50_hits']==expected_hits,'accepted item hits mismatch')

def execute(req,preflight=False):
    require(set(req)=={'schema','frozen_root','scheduler_root','run_ids','manifest_pins','history','catalog','sid_manifest'},'request fields')
    require(req['schema']=='sid-a1-prefix-request-v1','request schema');validate_run_ids(req['run_ids'])
    diag,metrics=frozen(req['frozen_root']);archive=diag.archive
    root=Path(req['scheduler_root']);require(root==archive.ROOT,'original archive root required')
    allowed=[root/'runs'/r for r in (*RUNS,*(r+'-verify' for r in RUNS))]
    allowed.extend([root/'queue/claimed',Path(req['history']['path']).parent,Path(req['catalog']['path']).parent,Path(req['sid_manifest']['path']).parent])
    original=archive.read
    def guarded(path,pin=None):
        safe(path,allowed);return original(path,pin)
    archive.read=guarded
    try:
        hist=diag.read_json(req['history']);catalog=diag.read_json(req['catalog']);sid=diag.read_json(req['sid_manifest'])
        require(len(catalog)==120415 and len(set(catalog))==120415,'catalog population')
        require(archive.plain_digest(catalog)==hist['catalog_sha256'] and req['catalog']['sha256']==hist['binding']['catalog_sha'],'catalog identity')
        require(req['history']['sha256']==archive.HISTORY_PIN,'history identity')
        queries=diag.read_json({'path':str(Path(req['history']['path']).parent/'validation-l50.json'),'sha256':hist['artifacts']['validation-l50.json']})
        require(len(queries)==2520 and sum(q['oov'] for q in queries)==623,'query population')
        hist['_g_catalog_digest']=archive.digest(catalog);hist['_base_queries_canonical_digest']=hist['binding']['query_sha']
        require(req['sid_manifest']['sha256']=='c53b81ddfcf8d45fd55903b30f8538372110c8e1df7376156719509994625187' and sid['sid_version']=='COLL_C1_B256D3_S42','accepted SID identity')
        ref={'path':str(Path(req['sid_manifest']['path']).parent/'mapping.npz'),'sha256':sid['artifacts']['mapping.npz']}
        safe(ref['path'],allowed);diag.stat_ref(ref)
        states={}
        for rid in RUNS:
            state=archive.inspect_run(rid);require(state['verified'] and state['manifest_sha256']==req['manifest_pins'][rid],'verified source manifest required')
            require(not state['queue']['rejected'],'rejected input run')
            states[rid]=state
        result={'status':'preflight_ok' if preflight else 'prefix_aggregate_complete','read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_read':False,'checkpoint_read':False,'model_run':False,'frozen_B_sha256':FROZEN['sid_diag_metrics.py'],'population':{'all':2520,'in_catalog':1897,'oov':623},'B':{}}
        if not preflight:
            mapping=diag.load_mapping(ref,catalog)
            for rid in RUNS:
                predictions,summary=diag.load_predictions(rid,queries,hist,states[rid],catalog)
                b=aggregate([q['target'] for q in queries],predictions,mapping,metrics)
                population(b,summary['hits']);result['B'][rid]=b
        result['input_sha256']={**archive.READS,**diag.READS}
        return result
    finally:archive.read=original

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--request-sha',required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    request=Path(args.request);require(request.suffix=='.json' and not request.is_symlink() and request.stat().st_size<65536 and sha(request)==args.request_sha,'request pin')
    print(json.dumps(execute(json.loads(request.read_bytes()),args.preflight),sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
