"""Real fixed launcher -> metadata -> same-module dry; no real data read."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo=Path(__file__).resolve().parents[2]
base=repo/'cache/ad1c428-sprint-20261021'/('sgns-chain-'+str(time.time_ns()))
base.mkdir(parents=True)
def git(root,*argv):
    return subprocess.check_output(['git','-C',str(root),*argv],stderr=subprocess.PIPE,text=True).strip()
def clone(name,ref):
    target=base/name
    subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(target)],capture_output=True,check=True)
    git(target,'checkout','-q','--detach',ref)
    return target
code=clone('CODE',git(repo,'rev-parse','HEAD'))
ops=clone('OPS_CODE','5840e6b686ec9651314a9e28a1a2c0befcc05158')
env={**os.environ,'PYTHONPATH':str(code/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
command=[sys.executable,'-B','-m','analysis.sgns_subset_ops','queue-dry','--stage','input-preflight',
         '--data',str(base/'raw'),'--embedding-root',str(base/'embed'),'--lock',str(code/'uv.lock'),
         '--bundle',str(base/'bundle'),'--output',str(base/'not-used'),'--ops-code',str(ops),
         '--ops-root',str(base/'OPS_ROOT')]
prepared=subprocess.run(command,cwd=code,env=env,capture_output=True,text=True)
assert prepared.returncode==0,prepared.stderr
value=json.loads(prepared.stdout)
assert value['metadata']['git']['commit']==git(code,'rev-parse','HEAD')
assert value['spec']['argv'][3:5]==['analysis.sgns_gpu_subset','preflight']
assert value['spec']['seed']==42
assert value['metadata']['git']['dirty_files']==[]
(base/'prepared.json').write_text(prepared.stdout)
# Same module invocation form; dry avoids pretending synthetic files are real inputs.
dry=subprocess.run(value['spec']['argv'][:4]+['dry'],cwd=code,env=env,capture_output=True,text=True)
assert dry.returncode==0,dry.stderr
assert json.loads(dry.stdout)['files']==26
(base/'dry.json').write_text(dry.stdout)
print(json.dumps({'status':'launcher_metadata_dry_verified','source':git(code,'rev-parse','HEAD'),
                  'source_files':len(git(code,'ls-tree','-r','--name-only','HEAD').splitlines()),
                  'payload_read':False,'real_preflight':False,'submitted':False,
                  'p3c_overlay':'not_applicable: worker does not read P3-D contract',
                  'evidence':str(base.relative_to(repo))},sort_keys=True))
