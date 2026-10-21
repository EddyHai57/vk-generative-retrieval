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
    # Separate four-gate transport metadata chain. These are inert bytes for
    # hashing only, never fake accepted SGNS/SID/A1 receipts or runtime locks.
    transport=root/'TRANSPORT';transport.mkdir();inert=transport/'inert.json';inert.write_text('{}')
    import hashlib
    inert_record={'path':str(inert),'sha256':hashlib.sha256(inert.read_bytes()).hexdigest()}
    request=transport/'request.json'
    request.write_text(json.dumps({'schema':'gpu-sgns-subset-four-gate-request-v1',
        'a1_code_root':str(root/'A1_CODE_NOT_READ'),'a1_request':inert_record,
        'a1_verify':{k:inert_record for k in ('run','terminal','stdout','receipt')},
        'ops_root':str(root/'AGG_OPS'),'output_root':str(root/'AGG_OUTPUT'),'scratch_root':str(root/'AGG_SCRATCH'),
        'lock':inert_record,'deployment_sha':git(code,'rev-parse','HEAD')},sort_keys=True))
    aggregate=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_acceptance','queue-dry',
        '--request',str(request),'--request-sha256',hashlib.sha256(request.read_bytes()).hexdigest(),
        '--ops-code',str(ops)],cwd=code,env=env,capture_output=True,text=True)
    assert aggregate.returncode==0,aggregate.stderr
    aggregate_value=json.loads(aggregate.stdout)
    assert aggregate_value['metadata']['git']['commit']==git(code,'rev-parse','HEAD')
    assert aggregate_value['metadata']['git']['dirty_files']==[]
    assert aggregate_value['spec']['argv'][3:5]==['analysis.sgns_subset_acceptance','verify']
    assert aggregate_value['submitted'] is False and aggregate_value['subset_four_gate_acceptance'] is False
    aggregate_dry=subprocess.run(aggregate_value['spec']['argv'][:4]+['dry'],cwd=code,env=env,capture_output=True,text=True)
    assert aggregate_dry.returncode==0,aggregate_dry.stderr
    assert json.loads(aggregate_dry.stdout)['subset_four_gate_acceptance'] is False
    # New prepare RAM is an engineering declaration, never a measured peak.
    pf=root/'PREPARE_TRANSPORT_PF';(pf/'logs').mkdir(parents=True)
    for name in ('run.json','terminal.json','logs/stdout.log'):(pf/name).write_text('{}')
    prep_argv=list(argv)
    prep_argv[prep_argv.index('--stage')+1]='prepare'
    prep_argv[prep_argv.index('--run-id')+1]='full-input-prepare-transport-only'
    prep_argv+=['--preflight-run',str(pf)]
    prep=subprocess.run(prep_argv,cwd=code,env=env,capture_output=True,text=True)
    assert prep.returncode==0,prep.stderr
    prep_value=json.loads(prep.stdout)
    assert prep_value['spec']['ram_mib']==196608
    bridge_transport=False
    private_dir=os.environ.get('FULL_SGNS_BRIDGE_PRIVATE_DIR')
    if private_dir:
        private=Path(private_dir).resolve()
        fixed={}
        for name,filename in (('plan','full-pf-resource-override-plan-v1-private.json'),
                              ('before','full-pf-queue-before-v1-private.json'),('after','full-pf-queue-after-v1-private.json')):
            path=private/filename
            fixed[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        identity=subprocess.run([sys.executable,'-B','-c','import json; from analysis.ur001_derive_cli import source_identity; print(json.dumps(source_identity()))'],cwd=code,env=env,capture_output=True,text=True)
        assert identity.returncode==0,identity.stderr
        request=transport/'bridge-transport-only.json'
        request.write_text(json.dumps({'schema':'gpu-sgns-full-input-source-bridge-request-v1',
            'source':json.loads(identity.stdout),'old_code_root':str(root/'OLD_CODE_NOT_READ'),
            'preflight_run':str(pf),'preflight_argv':[],
            'resource_plan':fixed,'apply_output':str(root/'BACKUP_NOT_CREATED')},sort_keys=True))
        bridge=subprocess.run(prep_argv+['--source-bridge-request',str(request),'--source-bridge-sha256',hashlib.sha256(request.read_bytes()).hexdigest()],cwd=code,env=env,capture_output=True,text=True)
        assert bridge.returncode==0,bridge.stderr
        bridge_value=json.loads(bridge.stdout)
        assert bridge_value['metadata']['git']['dirty_files']==[]
        assert bridge_value['spec']['ram_mib']==196608
        assert bridge_value['spec']['argv'][19:23:2]==['--source-bridge-request','--source-bridge-sha256']
        assert len(bridge_value['metadata']['inputs'])==9
        assert not (root/'BACKUP_NOT_CREATED').exists()
        bridge_transport=True
    regression = subprocess.run([sys.executable,'-B','-m','unittest','tests.test_sgns_gpu_full',
        'tests.test_sgns_gpu_subset','tests.test_sgns_subset_ops','tests.test_sgns_subset_acceptance','tests.test_sgns_full_input_bridge'],cwd=code,env=env,capture_output=True,text=True)
    assert regression.returncode == 0,regression.stdout + regression.stderr
    print(json.dumps({'bridge_transport_metadata_dry':bridge_transport,'prepare_ram_mib':196608,'real_source_bridge_verified':False,'four_gate_metadata_dry':True,'regression':regression.stderr.splitlines()[-4:], 'status':'launcher_metadata_dry_verified','source':git(code,'rev-parse','HEAD'),
                      'public_base':base,'delta_files':len(NEW_PATHS),'seed':42,'submitted':False,'real_preflight':False,'payload_read':False,
                      'p3c_overlay':'not_applicable: SGNS does not read P3-D',
                      'synthetic_manifest':'launcher hashing only; not represented as accepted full DAG',
                      'synthetic_lock':'launcher hashing only; not represented as original runtime lock'},sort_keys=True))
