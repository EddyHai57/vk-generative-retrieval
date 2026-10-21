"""Actual pinned OPS prepare -> metadata -> same-module dry; transport fixture only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo=Path(__file__).resolve().parents[2]
base=repo/'cache/sgns-subset-a1'/('launcher-chain-'+str(time.time_ns()))
base.mkdir(parents=True)
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
def clone(name,ref):
    target=base/name
    subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(target)],capture_output=True,check=True)
    git(target,'checkout','-q','--detach',ref)
    return target
code=clone('CODE',git(repo,'rev-parse','HEAD'));ops=clone('OPS','5840e6b686ec9651314a9e28a1a2c0befcc05158')
sys.path.insert(0,str(code/'src'))
from vkgr.sgns_subset_a1 import A1_ADDITIONS,SCHEMA
transport=base/'synthetic-transport';transport.write_bytes(b'fixture bytes, not an accepted raw input or producer')
ref={'path':str(transport),'sha256':hashlib.sha256(transport.read_bytes()).hexdigest()}
producer={name:ref for name in ('run','terminal','stdout','receipt')}
frozen_lock=base/'original-anchor.lock';frozen_lock.write_bytes(subprocess.check_output(['git','-C',str(repo),'show','b5a4d1c:uv.lock']))
value={'schema':SCHEMA,'source_commit':git(code,'rev-parse','HEAD'),
       'lock':{'path':str(frozen_lock),'sha256':hashlib.sha256(frozen_lock.read_bytes()).hexdigest()},
       'data_root':str(base/'synthetic-data'),'prepared':ref,'history':ref,'content':ref,
       'input_producer':producer,'cpu_sgns':producer,'gpu_sgns':producer,
       'cpu_sid':{'producer':producer,'manifest':ref,'input_manifest':ref},
       'gpu_sid':{'producer':producer,'manifest':ref,'input_manifest':ref},
       'output_root':str(base/'new-output'),'scratch_root':str(base/'new-scratch'),'ops_root':str(base/'OPS_ROOT'),
       'producer_delta':{p:hashlib.sha256((code/p).read_bytes()).hexdigest() for p in A1_ADDITIONS}}
request=base/'request.json';request.write_text(json.dumps(value,sort_keys=True));sha=hashlib.sha256(request.read_bytes()).hexdigest()
env={**os.environ,'PYTHONPATH':str(code/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
cmd=[sys.executable,'-B','-m','analysis.sgns_subset_a1','prepare','--stage','preflight','--request',str(request),'--request-sha256',sha,'--ops-code',str(ops)]
p=subprocess.run(cmd,cwd=code,env=env,capture_output=True,text=True)
assert p.returncode==0,p.stderr
result=json.loads(p.stdout);spec=result['spec'];metadata=result['metadata']
assert metadata['git']['commit']==value['source_commit'] and metadata['git']['dirty_files']==[]
assert spec['seed']==42 and spec['run_id']=='sgns-gpu-subset-a1-preflight-v1'
assert spec['argv'][3:5]==['analysis.sgns_subset_a1','preflight']
assert metadata['config']['sha256']==sha
(base/'prepared.json').write_text(p.stdout)
dry=subprocess.run(spec['argv'][:4]+['dry'],cwd=code,env=env,capture_output=True,text=True)
assert dry.returncode==0,dry.stderr
assert json.loads(dry.stdout)['training'] is False
(base/'dry.json').write_text(dry.stdout)
print(json.dumps({'status':'launcher_metadata_dry_verified','source':value['source_commit'],
    'payload_read':False,'real_preflight':False,'science_training':False,'submitted':False,
    'fixture_scope':'synthetic transport hashes only; no accepted raw or producer claim',
    'p3c_overlay':'not consumed by transport dry; formal historical data channel remains required',
    'evidence':str(base.relative_to(repo))},sort_keys=True))
# The added input preparation entry uses the actual same launcher, still no payload read.
ivalue={'schema':'gpu-sgns-subset-a1-input-request-v1','source_commit':value['source_commit'],'lock':value['lock'],
        'raw_data':value['data_root'],'ur001':ref,'content':ref,'gpu_sid_manifest_sha256':'1'*64,
        'output_root':str(base/'new-input-output'),'ops_root':str(base/'INPUT_OPS')}
irequest=base/'input-request.json';irequest.write_text(json.dumps(ivalue,sort_keys=True));isha=hashlib.sha256(irequest.read_bytes()).hexdigest()
p=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_a1_inputs','prepare','--request',str(irequest),'--request-sha256',isha,'--ops-code',str(ops)],cwd=code,env=env,capture_output=True,text=True)
assert p.returncode==0,p.stderr
prepared=json.loads(p.stdout);assert prepared['metadata']['git']['dirty_files']==[]
assert prepared['spec']['argv'][3:5]==['analysis.sgns_subset_a1_inputs','derive']
(base/'inputs-prepared.json').write_text(p.stdout)
p=subprocess.run(prepared['spec']['argv'][:4]+['dry'],cwd=code,env=env,capture_output=True,text=True)
assert p.returncode==0,p.stderr
assert json.loads(p.stdout)['subset_only'] is True
(base/'inputs-dry.json').write_text(p.stdout)
print(json.dumps({'status':'input_launcher_metadata_dry_verified','source':value['source_commit'],
                  'real_preflight':False,'payload_read':False,'submitted':False,'evidence':str(base.relative_to(repo))}))
