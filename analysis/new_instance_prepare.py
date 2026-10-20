"""New-instance preparation CLI. Tonight only `dry`/synthetic tests are allowed.

Execution commands are explicit, re-use pinned workers and never use SSH.
Run on the newly approved instance only, after formal launcher preflight.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA = '8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063'
MEMBERSHIP_SHA = '541ca824327f3a714fb078e26d2f0cdebcdafdd8fc1b19c5e0a507ffa2996fd9'
A_INVENTORY_SHA = '19cd71f02aff4fce93349907dbc5fcdbf006afc679ee899d9c98965123270fa0'
VERSIONS = dict(python='3.11.16', numpy='1.26.4', pyarrow='19.0.1', pyyaml='6.0.2', torch='2.11.0+cu128', cuda='12.8')


def inventory():
    path=ROOT/'configs/ur001-preparation/hf-files.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=A_INVENTORY_SHA: raise ValueError('fixed 29-file inventory SHA mismatch')
    source=json.loads(raw)
    from analysis import ur001_week26_bounded as w
    return {**source, 'files': source['files'] + [{'path':w.OBJECT,'size':w.OBJECT_SIZE,'sha256':w.OBJECT_SHA256}]}


def check_runtime(values):
    if values != VERSIONS: raise ValueError('STOP: original runtime identity differs; no fallback')


def runtime(lock, cuda_smoke=False):
    if hashlib.sha256(Path(lock).read_bytes()).hexdigest()!=LOCK_SHA: raise ValueError('original anchor uv.lock required')
    import numpy as np
    import pyarrow
    import yaml
    import torch
    values=dict(python=platform.python_version(),numpy=np.__version__,pyarrow=pyarrow.__version__,
                pyyaml=yaml.__version__,torch=torch.__version__,cuda=torch.version.cuda)
    check_runtime(values)
    result={'runtime':values,'lock_sha256':LOCK_SHA,'cuda_smoke_executed':False}
    if cuda_smoke:
        if not torch.cuda.is_available(): raise ValueError('STOP: CUDA unavailable')
        a=torch.tensor([1.,2.],device='cuda'); b=a*a
        torch.cuda.synchronize()
        if b.cpu().tolist()!=[1.,4.]: raise ValueError('STOP: CUDA small operator mismatch')
        result['cuda_smoke_executed']=True
    return result


def check_rebuilt(value):
    expected=dict(filtered_rows=14881823,filtered_users=81790,filtered_items=1183510,
                  union_items=12467294,users_sha256=MEMBERSHIP_SHA)
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('STOP: new-instance membership/four counts differ from original; return to Kiro')
    return {'status':'rebuilt_counts_exact','test_metrics_read':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for name in ('env-check','cuda-smoke'):
        c=sub.add_parser(name); c.add_argument('--lock',required=True)
        if name=='cuda-smoke': c.add_argument('--execute-on-new-instance',action='store_true',required=True)
    d=sub.add_parser('download-a'); d.add_argument('--root',required=True); d.add_argument('--reference-root',required=True)
    d.add_argument('--execute-on-new-instance',action='store_true',required=True)
    for name in ('source-preflight','prepare-users','download-week26','filter-preflight','filter'):
        c=sub.add_parser(name); c.add_argument('--request',required=True); c.add_argument('--execute-on-new-instance',action='store_true',required=True)
    v=sub.add_parser('check-rebuilt');v.add_argument('--receipt',required=True)
    a=p.parse_args(argv)
    if a.command=='dry':
        fixed=inventory(); result={'status':'ready','network_executed':False,'data_read':False,
            'dataset':fixed['dataset'],'revision':fixed['revision'],'files':fixed['files'],
            'expected_runtime':VERSIONS,'lock_sha256':LOCK_SHA}
    elif a.command in ('env-check','cuda-smoke'):
        result=runtime(a.lock,a.command=='cuda-smoke')
    elif a.command=='check-rebuilt':
        result=check_rebuilt(json.loads(Path(a.receipt).read_text()))
    elif a.command=='download-a':
        from analysis import ur001_prepare as w
        from analysis.ur001_week26_bounded import OLD_WORKER_SHA256
        if hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest()!=OLD_WORKER_SHA256:
            raise ValueError('original A worker identity mismatch')
        fixed=w.load_file_manifest()
        result=w.download_all(Path(a.root),Path(a.reference_root),fixed)
    else:
        from analysis import ur001_week26_bounded as w
        request=w._load_request(Path(a.request))
        functions={'source-preflight':w.source_preflight,'prepare-users':w.prepare_users,
                   'download-week26':w.download,'filter-preflight':w.filter_preflight,'filter':w.filter_week26}
        result=functions[a.command](request)
        if a.command=='prepare-users' and result.get('users_sha256')!=MEMBERSHIP_SHA:
            raise ValueError('STOP: membership SHA differs; preserve output, no filtering')
        if a.command=='filter': check_rebuilt(result)
    print(json.dumps(result,sort_keys=True)); return 0


if __name__=='__main__':
    raise SystemExit(main())
