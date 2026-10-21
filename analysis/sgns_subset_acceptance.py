"""Read-only four-gate evidence aggregation; never trains or submits a job."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PINS_SHA='0603166e98bbc880fbc2aa2c7b5e9943d211e639a9fc6696324aeff4f10528d3'
RUN_ID='sgns-gpu-subset-four-gate-verify-v1'
SCHEMA='gpu-sgns-subset-four-gate-request-v1'
RECEIPT='gpu-sgns-subset-four-gate-receipt-v1'
REQUEST_KEYS={'schema','a1_code_root','a1_request','a1_verify','ops_root','output_root','scratch_root','lock','deployment_sha'}


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(raw):return hashlib.sha256(raw).hexdigest()


def pin(value,length=64):
    if type(value) is not str or len(value)!=length or set(value)-set('0123456789abcdef') or value=='0'*length:
        raise ValueError('explicit nonzero digest required')
    return value


def absolute(value):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):
        raise ValueError('absolute path without symlinks required')
    return p


def record_bytes(record):
    if type(record) is not dict or set(record)!={'path','sha256'}:raise ValueError('exact path/SHA record required')
    raw=absolute(record['path']).read_bytes()
    if sha(raw)!=pin(record['sha256']):raise ValueError('consumed record hash differs')
    return raw


def code_pins(*,required=True):
    path=ROOT/'configs/full-gpu-sgns/acceptance-pins.json'
    raw=path.read_bytes()
    if sha(raw)!=PINS_SHA:raise ValueError('four-gate receipt code pins hash differs')
    value=json.loads(raw)
    ready=type(value.get('a1_source')) is dict and type(value.get('a1_api')) is dict
    if required and not ready:raise ValueError('four-gate receipt code pins not frozen; STOP')
    return value,ready


def filesystem_tree(root,source):
    root=absolute(root);expected=source.get('files',{});actual={}
    for parent,dirs,names in os.walk(root,followlinks=False):
        # Git's administrative files are the only excluded namespace. Ignored
        # Python/native modules are included and cannot shadow pinned modules.
        if Path(parent)==root:dirs[:]=[d for d in dirs if d!='.git']
        if any((Path(parent)/d).is_symlink() for d in dirs):raise ValueError('code tree contains symlink')
        for name in names:
            path=Path(parent)/name
            if path==root/'.git':continue
            if path.is_symlink():raise ValueError('code tree contains symlink')
            actual[str(path.relative_to(root))]=sha(path.read_bytes())
    if actual!=expected:raise ValueError('code tree differs from complete frozen file ledger')
    return actual


def source_tree(root,source):
    root=absolute(root)
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
    if git('rev-parse','HEAD')!=pin(source.get('commit'),40) or git('status','--porcelain','--untracked-files=all'):
        raise ValueError('clean fixed original code tree required')
    names=set(git('ls-tree','-r','--name-only','HEAD').splitlines())
    if names!=set(source['files']):raise ValueError('Git tree differs from frozen ledger')
    filesystem_tree(root,source)
    return source


def module_origins(modules,root,source):
    root=absolute(root)
    for name,module in tuple(modules.items()):
        if name.split('.')[0] not in ('analysis','vkgr') or module is None:continue
        file=getattr(module,'__file__',None)
        if file:
            path=absolute(file)
            try:relative=str(path.relative_to(root))
            except ValueError:raise ValueError('module origin outside original pinned tree') from None
            if source.get('files',{}).get(relative)!=sha(path.read_bytes()):raise ValueError('module origin hash differs')
        elif name not in ('analysis',):raise ValueError('module origin file required')
        for entry in getattr(module,'__path__',[]):
            expected=root/('analysis' if name=='analysis' else 'src/vkgr')
            if absolute(entry)!=expected:raise ValueError('module namespace origin differs')


def read_request(path,digest):
    pins,_=code_pins()  # Must precede request, imports, data reads and scratch.
    value=json.loads(record_bytes({'path':str(path),'sha256':digest}))
    if type(value) is not dict or set(value)!=REQUEST_KEYS or value.get('schema')!=SCHEMA:
        raise ValueError('complete fixed four-gate request required')
    pin(value['deployment_sha'],40)
    for key in ('a1_code_root','ops_root','output_root','scratch_root'):absolute(value[key])
    for key in ('a1_request','lock'):record_bytes(value[key])
    if value['lock']['sha256']!=pins['lock_sha256']:raise ValueError('original runtime lock differs')
    if type(value['a1_verify']) is not dict or set(value['a1_verify'])!={'run','terminal','stdout','receipt'}:
        raise ValueError('complete original A1 verify producer required')
    for record in value['a1_verify'].values():record_bytes(record)
    roots=[absolute(value[k]) for k in ('a1_code_root','output_root','scratch_root')]+[ROOT]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('exclusive external output/scratch and separate original A1 tree required')
    source_tree(value['a1_code_root'],pins['a1_source'])
    return value


def last_receipt(raw):
    return json.loads(next((s for s in reversed(raw.splitlines()) if s.strip()),b'{}'))


def producer_records(record):
    if type(record) is not dict or set(record)!={'run','terminal','stdout','receipt'}:
        raise ValueError('complete original producer required')
    raw={name:record_bytes(rec) for name,rec in record.items()}
    return producer_bytes(raw,run_path=record['run']['path'])


def producer_bytes(raw, *, run_path, physical=False):
    run=json.loads(raw['run']);terminal=json.loads(raw['terminal']);receipt=json.loads(raw['receipt'])
    rid=run.get('run_id')
    required={'schema','run_id','status','exit_code','signal','wall_cap_exceeded','started_at','ended_at','wall_seconds'}
    wall=terminal.get('wall_seconds')
    if (set(terminal)!=required or terminal.get('schema')!='vkgr-terminal-v1'
            or terminal.get('run_id')!=rid or terminal.get('status')!='succeeded'
            or type(terminal.get('exit_code')) is not int or terminal['exit_code']!=0
            or terminal.get('signal') is not None or terminal.get('wall_cap_exceeded') is not False
            or type(wall) not in (int,float) or not math.isfinite(wall) or wall<0
            or run.get('schema')!='vkgr-run-v1' or run.get('kind')!='ops' or run.get('seed')!=42
            or run.get('git',{}).get('dirty') is not False or run.get('git',{}).get('dirty_files')!=[]
            or absolute(run_path).parent.name!=rid
            or (receipt.get('run_id')!=rid and not (physical and receipt.get('schema')=='sgns-subset-physical-input-v1' and 'run_id' not in receipt))
            or last_receipt(raw['stdout'])!=receipt):
        raise ValueError('original producer terminal/stdout binding differs')
    return run,receipt,raw


def check_proof(proof):
    keys={'health','sid_gates','collision','validation_hits','hits_gate','a1_native','evidence'}
    if type(proof) is not dict or set(proof)!=keys:raise ValueError('complete recomputed four-gate proof required')
    collision=proof.get('collision',{});a=collision.get('cpu_count');b=collision.get('gpu_count');n=collision.get('items')
    if (proof.get('health',{}).get('passed') is not True
            or set(proof.get('sid_gates',{}))!={'cpu','cuda'}
            or any(proof['sid_gates'][d].get('passed') is not True for d in ('cpu','cuda'))
            or type(proof.get('validation_hits')) is not int or proof['validation_hits']<378
            or type(proof.get('hits_gate')) is not int or proof['hits_gate']!=378
            or type(n) is not int or n<=0 or any(type(x) is not int or not 0<=x<=n for x in (a,b))
            or b*10>a*11 or collision.get('passed') is not True
            or collision.get('comparison')!='gpu_count * 10 <= cpu_count * 11'
            or proof.get('a1_native',{}).get('status')!='verified'
            or type(proof.get('evidence')) is not dict or not proof['evidence']):
        raise ValueError('recomputed four-gate acceptance failed')
    for path,digest in proof['evidence'].items():absolute(path);pin(digest)
    return proof


def invocation(request,digest):
    return [sys.executable,'-B','-m','analysis.sgns_subset_acceptance','verify','--request',str(request),'--request-sha256',digest]


def job_contract(value,request,digest,source):
    from analysis.sgns_gpu_full import file_record
    run=absolute(value['ops_root'])/'runs'/RUN_ID
    records=[{'path':str(request),'sha256':digest},value['lock'],value['a1_request'],*value['a1_verify'].values()]
    inputs=[];seen=set()
    for rec in records:
        if rec['path'] not in seen:inputs.append(file_record(Path(rec['path']),rec['sha256']));seen.add(rec['path'])
    return {'schema':'vkgr-run-v1','run_id':RUN_ID,'kind':'ops','cwd':str(ROOT),
        'argv':invocation(request,digest),'seed':42,'git':{'commit':source['commit'],'dirty':False,'dirty_files':[]},
        'config':{'path':str(request),'sha256':digest},'inputs':inputs,
        'env':{'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
               'CUBLAS_WORKSPACE_CONFIG':':4096:8','VKGR_RUN_ID':RUN_ID,'VKGR_PROGRESS_PATH':str(run/'progress.json')},
        'resources':{'cpu_cores':1,'ram_mib':98304,'vram_mib':32768,'disk_gib':4,'wall_seconds':172800.0},
        'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},
        'overlay':None,'pinned_files':{},'stop_mode':'signal'}


def check_job(value,request,digest,source):
    expected=job_contract(value,request,digest,source)
    raw=(absolute(value['ops_root'])/'runs'/RUN_ID/'run.json').read_bytes();run=json.loads(raw)
    if (set(run)-set(expected)-{'created_at','admission'}
            or any(encoded(run.get(k))!=encoded(v) for k,v in expected.items())):
        raise ValueError('own four-gate launcher contract differs')
    return sha(raw)


def replay(value,request,digest,scratch):
    code_pins();scratch=absolute(scratch)
    forbidden=[ROOT,absolute(value['a1_code_root']),absolute(value['output_root']),absolute(value['a1_request']['path']).parent]
    if scratch.exists() or any(scratch==p or scratch in p.parents or p in scratch.parents for p in forbidden):
        raise ValueError('new exclusive acceptance scratch required')
    adapter=ROOT/'analysis/sgns_subset_acceptance_a1.py'
    env={k:v for k,v in os.environ.items() if not k.startswith('PYTHON')}
    env.update({'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    run=subprocess.run([sys.executable,'-I','-B',str(adapter),'--request',str(request),
        '--request-sha256',digest,'--scratch',str(scratch)],cwd=value['a1_code_root'],env=env,capture_output=True,text=True)
    if run.returncode:raise ValueError('isolated original evidence verification failed: '+run.stderr[-1500:])
    return check_proof(json.loads(run.stdout))


def verify(request,digest):
    value=read_request(request,digest)
    from analysis.sgns_gpu_full import check_source,runtime
    from analysis.ur001_derive_cli import source_identity
    source=source_identity();check_source(source,value['deployment_sha']);env=runtime(value['lock']['path'])
    run_sha=check_job(value,request,digest,source)
    if os.environ.get('VKGR_RUN_ID')!=RUN_ID:raise ValueError('original launcher execution required')
    output=absolute(value['output_root'])
    if output.exists():raise FileExistsError('exclusive total receipt output required')
    proof=replay(value,request,digest,value['scratch_root'])
    if (read_request(request,digest)!=value or source_identity()!=source
            or runtime(value['lock']['path'])!=env or check_job(value,request,digest,source)!=run_sha):
        raise ValueError('four-gate input/source/runtime changed')
    receipt={'schema':RECEIPT,'authority':'ad1c428','status':'verified','verified':True,
        'subset_four_gate_acceptance':True,'engineering':False,'test_read':False,'training':False,
        'run_id':RUN_ID,'run_manifest_sha256':run_sha,'source':source,'runtime':env,
        'request':{'path':str(request),'sha256':digest},'code_pins_sha256':PINS_SHA,'proof':proof,
        'scratch_written':True,'resource_scope':'not_measured conservative first declaration; own context/input checks only'}
    output.mkdir(parents=True,exist_ok=False)
    with (output/'stage-receipt.json').open('xb') as f:f.write(encoded(receipt))
    return receipt


def consume(path,digest,scratch):
    code_pins()
    raw=record_bytes({'path':str(path),'sha256':digest});receipt=json.loads(raw)
    if (receipt.get('schema')!=RECEIPT or receipt.get('authority')!='ad1c428'
            or receipt.get('status')!='verified' or receipt.get('verified') is not True
            or receipt.get('subset_four_gate_acceptance') is not True or receipt.get('engineering') is not False
            or receipt.get('test_read') is not False or receipt.get('training') is not False
            or receipt.get('run_id')!=RUN_ID or receipt.get('code_pins_sha256')!=PINS_SHA):
        raise ValueError('original total acceptance receipt required')
    req=receipt['request'];value=read_request(Path(req['path']),req['sha256'])
    from analysis.sgns_gpu_full import producer,check_source,runtime
    from analysis.ur001_derive_cli import source_identity
    source=source_identity();check_source(source,value['deployment_sha']);env=runtime(value['lock']['path'])
    if (absolute(path)!=absolute(value['output_root'])/'stage-receipt.json'
            or receipt.get('source')!=source or receipt.get('runtime')!=env):raise ValueError('total receipt source/runtime/output differs')
    run=absolute(value['ops_root'])/'runs'/RUN_ID
    own=producer(run,invocation(Path(req['path']),req['sha256']),source,receipt,
        expected_job=job_contract(value,Path(req['path']),req['sha256'],source))
    if receipt.get('run_manifest_sha256')!=own['run.json']:raise ValueError('total receipt original job hash differs')
    check_proof(receipt['proof'])
    if replay(value,Path(req['path']),req['sha256'],scratch)!=receipt['proof']:
        raise ValueError('total receipt differs from independently recomputed original evidence')
    if (record_bytes({'path':str(path),'sha256':digest})!=raw or read_request(Path(req['path']),req['sha256'])!=value
            or producer(run,invocation(Path(req['path']),req['sha256']),source,receipt,
                expected_job=job_contract(value,Path(req['path']),req['sha256'],source))!=own
            or source_identity()!=source or runtime(value['lock']['path'])!=env):
        raise ValueError('total acceptance snapshot changed')
    return {'receipt_sha256':digest,'request_sha256':req['sha256'],'producer':own,'proof':receipt['proof']}


def queue_dry(request,digest,ops_code):
    # This path only hashes declared transport inputs. It deliberately does not
    # call accepted read_request or represent the hashes as scientific evidence.
    value=json.loads(record_bytes({'path':str(request),'sha256':digest}))
    if type(value) is not dict or set(value)!=REQUEST_KEYS or value.get('schema')!=SCHEMA:
        raise ValueError('complete fixed four-gate transport request required')
    from analysis.sgns_gpu_full import check_source
    from analysis.ur001_derive_cli import source_identity
    from analysis.a40_ops import load_ops
    source=source_identity();check_source(source,value['deployment_sha'])
    contract=job_contract(value,request,digest,source)
    _,launcher,_,_=load_ops(ops_code)
    flags=['--root',value['ops_root'],'--run-id',RUN_ID,'--kind','ops','--cwd',str(ROOT),
        '--config',str(request),'--seed','42','--cpu-cores','1','--ram-mib','98304','--vram-mib','32768',
        '--disk-gib','4','--wall-hours','48']
    for k,val in contract['env'].items():
        if k not in ('VKGR_RUN_ID','VKGR_PROGRESS_PATH'):flags+=['--env',k+'='+val]
    for record in contract['inputs']:flags+=['--input',record['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*invocation(request,digest)]))
    metadata=launcher.prepare(spec)
    prepared={k:contract[k] for k in ('cwd','git','overlay','pinned_files','config','inputs')}
    prepared['env']={k:val for k,val in contract['env'].items() if k not in ('VKGR_RUN_ID','VKGR_PROGRESS_PATH')}
    if encoded(metadata)!=encoded(prepared):
        raise ValueError('real launcher metadata differs from fixed acceptance contract')
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,'real_preflight':False,
        'scientific_payload_read':False,'scratch_written':False,'subset_four_gate_acceptance':False,
        'resource_scope':'not_measured conservative first declaration; requires live capacity gate'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in ('verify','queue-dry'):
        c=sub.add_parser(name);c.add_argument('--request',type=Path,required=True);c.add_argument('--request-sha256',required=True)
        if name=='queue-dry':c.add_argument('--ops-code',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='dry':
        _,ready=code_pins(required=False)
        value={'status':'dry','code_pins_ready':ready,'training':False,'test_read':False,'subset_four_gate_acceptance':False}
    elif args.command=='queue-dry':value=queue_dry(args.request,args.request_sha256,args.ops_code)
    else:value=verify(args.request,args.request_sha256)
    print(json.dumps(value,sort_keys=True,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
