"""Real launcher prepare -> clean deployment metadata -> same-module dry only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo))

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()

with tempfile.TemporaryDirectory(prefix='full-sgns-chain-') as temp:
    root = Path(temp).resolve()
    def clone(name,ref):
        target = root/name
        subprocess.run(['git','clone','-q','--shared','--no-checkout',str(repo),str(target)],check=True,capture_output=True)
        git(target,'checkout','-q','--detach',ref)
        return target
    base = 'c9a60c5ce6eeb02db862455a5cafbbc861c4effb'
    code = clone('CODE',base)
    from analysis.sgns_gpu_full import NEW_PATHS
    for relative in sorted(NEW_PATHS):
        path = code/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        # Snapshot committed task files only; never copy untracked/process files.
        path.write_bytes(subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+relative]))
    git(code,'add','--',*sorted(NEW_PATHS))
    git(code,'-c','user.name=Local engineering fixture','-c','user.email=fixture@example.invalid',
        '-c','commit.gpgsign=false','commit','-q','-m','Synthetic public-base full SGNS tree')
    assert set(git(code,'diff','--name-only',base,'HEAD').splitlines()) == NEW_PATHS
    lock = root/'LOCK'/'uv.lock'; lock.parent.mkdir()
    lock.write_text('# Synthetic launcher input; original runtime is not exercised.\n')
    ops = clone('OPS_CODE','5840e6b686ec9651314a9e28a1a2c0befcc05158')
    stage2 = root/'STAGE2'; stage2.mkdir()
    (stage2/'UR001-manifest.json').write_text('{}')
    env = {**os.environ,'PYTHONPATH':str(code/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
    argv = [sys.executable,'-B','-m','analysis.sgns_gpu_full','queue-dry','--stage','preflight',
            '--data',str(root/'RAW'),'--embedding-root',str(root/'EMBED'),
            '--stage2',str(stage2),'--stage2-run',str(root/'STAGE2_RUN'),
            '--lock',str(lock),'--output',str(root/'BUNDLE'),
            '--deployment-sha',git(code,'rev-parse','HEAD'),
            '--ops-code',str(ops),'--ops-root',str(root/'OPS_ROOT'),'--run-id','sgns-gpu-full-input-preflight-v1']
    run = subprocess.run(argv,cwd=code,env=env,capture_output=True,text=True)
    assert run.returncode == 0,run.stderr
    result = json.loads(run.stdout)
    assert result['metadata']['git']['commit'] == git(code,'rev-parse','HEAD')
    assert result['metadata']['git']['dirty_files'] == []
    assert result['spec']['seed'] == 42
    assert result['spec']['argv'][3:5] == ['analysis.sgns_gpu_full','preflight']
    dry = subprocess.run(result['spec']['argv'][:4]+['dry'],cwd=code,env=env,capture_output=True,text=True)
    assert dry.returncode == 0,dry.stderr
    assert json.loads(dry.stdout)['train_files'] == 25
    assert json.loads(dry.stdout)['training_enabled'] is False
    identity = subprocess.run([sys.executable,'-B','-c',
        'from analysis.sgns_gpu_full import check_source; from analysis.ur001_derive_cli import source_identity; s=source_identity(); check_source(s,s["commit"]); print("exact-source-delta-verified")'],
        cwd=code,env=env,capture_output=True,text=True)
    assert identity.returncode == 0,identity.stderr
    assert identity.stdout.strip() == 'exact-source-delta-verified'
    regression = subprocess.run([sys.executable,'-B','-m','unittest','tests.test_sgns_gpu_full',
        'tests.test_sgns_gpu_subset','tests.test_sgns_subset_ops'],cwd=code,env=env,capture_output=True,text=True)
    assert regression.returncode == 0,regression.stdout + regression.stderr
    print(json.dumps({'regression':regression.stderr.splitlines()[-4:], 'status':'launcher_metadata_dry_verified','source':git(code,'rev-parse','HEAD'),
                      'public_base':base,'delta_files':len(NEW_PATHS),'seed':42,'submitted':False,'real_preflight':False,'payload_read':False,
                      'p3c_overlay':'not_applicable: SGNS does not read P3-D',
                      'synthetic_manifest':'launcher hashing only; not represented as accepted full DAG',
                      'synthetic_lock':'launcher hashing only; not represented as original runtime lock'},sort_keys=True))
