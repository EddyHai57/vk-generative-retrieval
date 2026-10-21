"""Real pinned OPS prepare -> same-entry dry, with no server submission."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
repo=Path(__file__).resolve().parents[2]
base=repo/'cache/p-query-block-infra'/('sprint-chain-'+str(time.time_ns()));base.mkdir(parents=True)
def git(root,*args,binary=False):
    value=subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE)
    return value if binary else value.decode().strip()
def clone(name,ref):
    root=base/name
    subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(root)],check=True,capture_output=True)
    git(root,'checkout','-q','--detach',ref);return root
runner=clone('runner',git(repo,'rev-parse','HEAD'))
engine=clone('engine','ad1c428f5b7ad3558ec7db4ce3b8ea0527a8050d')
ops=clone('ops-code','5840e6b686ec9651314a9e28a1a2c0befcc05158')
shutil.copyfile(repo/'analysis/p_query_sprint_ops.py',runner/'analysis/p_query_sprint_ops.py')
git(runner,'add','analysis/p_query_sprint_ops.py')
p3c='docs/p3c_validation.md';public=git(repo,'show','418a421:'+p3c,binary=True)
(runner/p3c).write_bytes(public);git(runner,'add',p3c)
if git(runner,'diff','--cached','--name-only'):
    git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-q','-m','local sprint transport fixture')
private=(Path(os.environ['VKGR_P3C_DATA_SOURCE']).read_bytes() if os.environ.get('VKGR_P3C_DATA_SOURCE')
         else git(repo,'show','7041c90:'+p3c,binary=True))
assert hashlib.sha256(private).hexdigest()=='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
(base/'p3c-public-original.md').write_bytes(public);(runner/p3c).write_bytes(private)
data=base/'synthetic';data.mkdir();original_ops=base/'original-ops';lock=base/'uv.lock';lock.write_text('synthetic structural fixture only\n')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
sys.path[:0]=[str(repo),str(repo/'src')]
from analysis import a40_cost as cost
oldroot=cost.ROOT;cost.ROOT=engine
try:
    generator={'schema':'a40-p-exact-synthetic-v1','data_kind':'synthetic','items':12000000,'queries':26827,
               'dim':64,'top_k':50,'synthetic_seed':42,'generator_sha256':sha(engine/'analysis/a40_cost.py'),
               'selected_yaml_sha256':cost.YAML_SHA,'baseline_sha256':cost.REFERENCE_SHA,'files':[]}
    (data/'inputs.json').write_text(json.dumps(generator))
    def write(stage,payload):
        rid='a40-exact-'+stage+'-v1';target=original_ops/'runs'/rid;(target/'logs').mkdir(parents=True)
        inputs=[lock]+([data/'inputs.json'] if stage=='preflight' else [])
        record={'schema':'vkgr-run-v1','run_id':rid,'kind':'ops','git':{'commit':git(engine,'rev-parse','HEAD'),'dirty':False},
                'cwd':str(engine),'argv':cost.worker_argv(stage,data,lock,original_ops),
                'config':{'path':str(engine/'configs/throughput/a40-m153999.yaml'),'sha256':cost.YAML_SHA},
                'inputs':[{'path':str(path),'sha256':sha(path)} for path in inputs]}
        (target/'run.json').write_text(json.dumps(record));(target/'terminal.json').write_text(json.dumps(
            {'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded','exit_code':0}))
        (target/'logs/stdout.log').write_text(json.dumps(payload)+'\n');return sha(target/'logs/stdout.log')
    generation_sha=write('generate',generator)
    pre={'status':'ready','engineering':False,'inputs_manifest_sha256':sha(data/'inputs.json'),
         'script_sha256':sha(engine/'analysis/a40_cost.py'),'selected_yaml_sha256':cost.YAML_SHA,
         'worker_parity':[1,2,8],'score_bytes_equal_to_original':True,'synthetic_input':True,'science_data_read':False,
         'items':12000000,'queries':26827,'dim':64,'top_k':50,'baseline_sha256':cost.REFERENCE_SHA,
         'generation_stdout_sha256':generation_sha,'parity_scope':'35x4103 synthetic; full fixed block and tails'}
    (data/'preflight.json').write_text(json.dumps(pre));write('preflight',pre)
finally:cost.ROOT=oldroot
request={'schema':'p-query-sprint-v1','stage':'diagnostic','run_id':'p-query-diagnostic-q32-w2-v1',
         'runner_source':git(runner,'rev-parse','HEAD'),'original_source':git(engine,'rev-parse','HEAD'),
         'workers':2,'query_count':32,'engine':str(engine),'root':str(data),'lock':str(lock),
         'original_ops_root':str(original_ops),'ops_root':str(base/'ops-root'),'output':str(base/'diagnostic-output')}
request_path=base/'request.json';request_path.write_text(json.dumps(request))
env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
cmd=[sys.executable,'-B','-m','analysis.p_query_sprint_ops']
prepared=subprocess.run(cmd+['prepare','--request',str(request_path),'--ops-code',str(ops)],cwd=runner,env=env,capture_output=True,text=True)
assert prepared.returncode==0,prepared.stderr
value=json.loads(prepared.stdout)
assert value['metadata']['git']['commit']==request['runner_source']
assert value['metadata']['pinned_files']=={p3c:hashlib.sha256(private).hexdigest()}
assert value['spec']['argv']==cmd+['run','--request',str(request_path),'--execute-on-new-instance']
assert value['resources']['measurement']=='not_measured'
assert value['spec']['cpu_cores']==3 and value['spec']['ram_mib']==98304
(base/'prepare.json').write_text(prepared.stdout)
dry=subprocess.run(cmd+['dry','--request',str(request_path)],cwd=runner,env=env,capture_output=True,text=True)
assert dry.returncode==0,dry.stderr
assert json.loads(dry.stdout)['status']=='dry';(base/'dry.json').write_text(dry.stdout)
# A failed original preflight is rejected before any catalog array exists.
terminal=original_ops/'runs/a40-exact-preflight-v1/terminal.json';bad=json.loads(terminal.read_text());bad['status']='failed';terminal.write_text(json.dumps(bad))
rejected=subprocess.run(cmd+['prepare','--request',str(request_path),'--ops-code',str(ops)],cwd=runner,env=env,capture_output=True,text=True)
assert rejected.returncode!=0 and 'binding differs' in rejected.stderr
(base/'failed-original.stderr').write_text(rejected.stderr)
print(json.dumps({'status':'real_launcher_prepare_and_dry_verified','no_submit':True,'synthetic_structure_only':True,
                  'original_failed_terminal_rejected':True,'real_array_preflight_executed':False,'evidence':str(base.relative_to(repo))}))
