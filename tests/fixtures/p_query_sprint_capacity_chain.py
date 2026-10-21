"""Real launcher metadata/dry with fixed old producer and closed source delta."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

repo=Path(__file__).resolve().parents[2]
# Reuse the existing original-input metadata fixture, including fixed D0 bytes.
with redirect_stdout(io.StringIO()):
    ns=runpy.run_path(str(repo/'tests/fixtures/p_query_sprint_chain.py'))
base=ns['base'];git=ns['git'];clone=ns['clone'];sha=ns['sha'];private=ns['private'];p3c=ns['p3c']
engine=ns['engine'];ops=ns['ops'];data=ns['data'];lock=ns['lock'];original_ops=ns['original_ops'];cmd=ns['cmd']
terminal=original_ops/'runs/a40-exact-preflight-v1/terminal.json'
value=json.loads(terminal.read_text());value['status']='succeeded';terminal.write_text(json.dumps(value))
oldsha='43eb5bec6d32a5d37746a033452824fd399f2dc4'
oldroot=clone('capacity-old-runner',oldsha);runner=clone('capacity-new-runner',oldsha)
from analysis import p_query_sprint_ops as sprint
for name in sorted(sprint.SOURCE_DELTA_PATHS):
    path=repo/name
    if path.exists():
        (runner/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,runner/name);git(runner,'add',name)
git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-q','-m','capacity closed transport fixture')
newsha=git(runner,'rev-parse','HEAD')
for root in (oldroot,runner):
    (root/p3c).parent.mkdir(parents=True,exist_ok=True);(root/p3c).write_bytes(private)
def call(root,command,path):
    env={**ns['env'],'PYTHONPATH':str(root/'src')}
    extra=['--ops-code',str(ops)] if command=='prepare' else []
    result=subprocess.run(cmd+[command,'--request',str(path),*extra],cwd=root,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout)
old={**ns['request'],'workers':4,'runner_source':oldsha,'run_id':'p-query-diagnostic-q32-w4-v1',
     'output':str(base/'capacity-old-output')}
oldpath=base/'capacity-old-request.json';oldpath.write_text(json.dumps(old))
prepared=call(oldroot,'prepare',oldpath)
res=sprint.ops_resources(prepared['resources']);meta=prepared['metadata']
record={'schema':'vkgr-run-v1','run_id':old['run_id'],'kind':'ops',
        **{key:meta[key] for key in ('cwd','git','config','inputs')},'argv':prepared['spec']['argv'],'resources':res}
payload={'schema':'p-query-sprint-result-v1','status':'verified','stage':'diagnostic','queries':32,'workers':4,
         'run_id':old['run_id'],'request_sha256':sha(oldpath),'argv':record['argv'],
         'runner_source':prepared['runner_source'],'input_binding':prepared['input_binding'],
         'resources':{**prepared['resources'],'process_tree_peak_rss_upper_bound_bytes':1024**3}}
output=Path(old['output']);output.mkdir();(output/'result.json').write_text(json.dumps(payload))
run=Path(old['ops_root'])/'runs'/old['run_id'];(run/'logs').mkdir(parents=True)
(run/'run.json').write_text(json.dumps(record));(run/'terminal.json').write_text(json.dumps(
    {'schema':'vkgr-terminal-v1','run_id':old['run_id'],'status':'succeeded','exit_code':0}))
(run/'logs/stdout.log').write_text(json.dumps(payload))
names=git(runner,'diff','--name-only',oldsha,newsha).splitlines()
delta={'schema':'p-query-source-delta-v1','old_source':oldsha,'new_source':newsha,'files':{
    name:{'old_sha256':sha(oldroot/name) if (oldroot/name).exists() else None,'new_sha256':sha(runner/name)} for name in names}}
delta_path=base/'capacity-source-delta.json';delta_path.write_text(json.dumps(delta))
request={**old,'run_id':'p-query-diagnostic-q64-w4-v1','query_count':64,'runner_source':newsha,
         'output':str(base/'capacity-new-output'),'measurement_request':str(oldpath),
         'source_delta':str(delta_path),'source_delta_sha256':sha(delta_path)}
path=base/'capacity-new-request.json';path.write_text(json.dumps(request))
prepared=call(runner,'prepare',path);dry=call(runner,'dry',path)
assert prepared['resources']['measurement']=='measured_upper_bound'
assert prepared['spec']['cpu_cores']==5 and prepared['spec']['ram_mib']==8192
assert {pin['path']:pin['sha256'] for pin in prepared['metadata']['inputs']}[str(delta_path)]==sha(delta_path)
assert prepared['measurement_producer']['run_id']==old['run_id']
assert prepared['measurement_producer']['source_delta_sha256']==sha(delta_path)
assert dry['status']=='dry'
(base/'capacity-prepare.json').write_text(json.dumps(prepared));(base/'capacity-dry.json').write_text(json.dumps(dry))
print(json.dumps({'status':'cross_source_real_launcher_prepare_and_dry_verified','no_submit':True,
                  'synthetic_structure_only':True,'real_array_preflight_executed':False,
                  'delta_paths':names,'evidence':str(base.relative_to(repo))}))
