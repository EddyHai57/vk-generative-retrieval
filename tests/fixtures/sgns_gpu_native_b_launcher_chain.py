"""Actual launcher metadata -> job binding -> same-entry dry; never submit."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
repo=Path(__file__).resolve().parents[2];base=repo/'cache/sgns-gpu-native-b'/('launcher-'+str(time.time_ns()));base.mkdir(parents=True)
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE).decode().strip()
def clone(name,ref):
    root=base/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(root)],check=True,capture_output=True)
    git(root,'checkout','-q','--detach',ref);return root
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
runner=clone('runner',git(repo,'rev-parse','HEAD'));ops=clone('ops-code','5840e6b686ec9651314a9e28a1a2c0befcc05158')
for name in ('src/vkgr/sgns_gpu_native.py','analysis/sgns_gpu_native_b.py'):
    shutil.copyfile(repo/name,runner/name);git(runner,'add',name)
git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-q','-m','B transport fixture')
bundle=base/'bundle';bundle.mkdir()
for name in ('manifest.json','catalog.json','content.npy','histories.jsonl'):(bundle/name).write_bytes(b'transport fixture only\n')
lock=base/'lock';lock.write_bytes(b'transport fixture only\n')
value={'schema':'sgns-gpu-native-b-request-v1','runner_source':git(runner,'rev-parse','HEAD'),'bundle':str(bundle),'bundle_sha256':sha(bundle/'manifest.json'),
       'lock':str(lock),'ops_root':str(base/'ops-root'),'output':str(base/'output'),'preflight_run_id':'sgns-gpu-native-b-preflight-fixture-v1',
       'benchmark_run_id':'sgns-gpu-native-b-benchmark-fixture-v1','warmup_batches':8,'measured_batches':256,'compiled':True}
path=base/'request.json';path.write_text(json.dumps(value));request_sha=sha(path)
env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'};cmd=[sys.executable,'-B','-m','analysis.sgns_gpu_native_b']
r=subprocess.run(cmd+['prepare','--stage','preflight','--request',str(path),'--request-sha256',request_sha,'--ops-code',str(ops)],cwd=runner,env=env,capture_output=True,text=True)
assert r.returncode==0,(r.stdout,r.stderr)
prepared=json.loads(r.stdout);(base/'prepare.json').write_text(r.stdout)
for extra in ({'VKGR_UNREGISTERED':'1'},{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
    attacked=subprocess.run(cmd+['prepare','--stage','preflight','--request',str(path),'--request-sha256',request_sha,'--ops-code',str(ops)],cwd=runner,env={**env,**extra},capture_output=True,text=True)
    assert attacked.returncode!=0 and 'unexpected inherited launcher environment' in attacked.stdout,(attacked.stdout,attacked.stderr)

assert prepared['spec']['argv']==cmd+['preflight','--request',str(path),'--request-sha256',request_sha,'--execute-on-new-instance']
assert prepared['spec']['cpu_cores']==2 and prepared['spec']['ram_mib']==16384 and prepared['spec']['vram_mib']==8192
sys.path[:0]=[str(runner),str(runner/'src')]
from analysis import sgns_gpu_native_b as p
run=Path(value['ops_root'])/'runs'/value['preflight_run_id'];run.mkdir(parents=True)
from analysis.a40_ops import load_ops
_,launcher,_,_=load_ops(ops)
meta=prepared['metadata'];spec=prepared['spec'];record={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'seed':spec['seed'],
    **{k:meta[k] for k in ('cwd','git','config','inputs')},'argv':spec['argv'],'resources':p.resources(),'env':launcher._recorded_env(meta['env'],launcher._job_env(meta['env'],run,spec['run_id']))}
(run/'run.json').write_text(json.dumps(record));assert p.job(value,path,request_sha,'preflight')==sha(run/'run.json')
for key in ('seed','resources','argv','inputs','env_absent','env_pythonpath','env_extra'):
    wrong=json.loads(json.dumps(record))
    if key=='seed':wrong[key]=43
    elif key=='resources':wrong[key]['ram_mib']+=1
    elif key=='argv':wrong[key][-1]='--invalid'
    elif key=='inputs':wrong[key]=list(reversed(wrong[key]))
    elif key=='env_absent':del wrong['env']
    elif key=='env_pythonpath':wrong['env']['PYTHONPATH']='/unexpected/src'
    else:wrong['env']['VKGR_UNREGISTERED']='1'
    (run/'run.json').write_text(json.dumps(wrong))
    try:p.job(value,path,request_sha,'preflight')
    except ValueError:pass
    else:raise AssertionError('own job mutation accepted: '+key)
for command in ('dry','invalid'):
    r=subprocess.run(cmd+[command],cwd=runner,env=env,capture_output=True,text=True)
    assert r.returncode==(0 if command=='dry' else 2),(r.stdout,r.stderr)
    if command=='dry':assert json.loads(r.stdout)['status']=='dry';(base/'dry.json').write_text(r.stdout)
print(json.dumps({'status':'real_launcher_metadata_job_binding_dry_verified','synthetic_structure_only':True,'input_roles_checked':len(meta['inputs']),
                  'negative_job_mutations':7,'inherited_environment_mutations':2,'real_array_preflight_executed':False,'no_submit':True,'evidence':str(base.relative_to(repo))}))
