"""Fixed SGNS subset jobs; original scheduler and scientific worker unchanged."""
import argparse
import json
from pathlib import Path
import sys

from analysis.sgns_gpu_subset import ROOT, fixed_records
from analysis.ur001_derive_cli import digest, secure

STAGES = ('input-preflight','input-prepare','cpu-fit','cuda-fit')


def worker_argv(stage,data,embedding,lock,bundle,output,bundle_sha=None):
    if stage not in STAGES:
        raise ValueError('fixed subset stage required')
    for path in (data,embedding,lock,bundle,output):secure(path)
    argv=[sys.executable,'-B','-m','analysis.sgns_gpu_subset']
    if stage in STAGES[:2]:
        return argv+['preflight' if stage=='input-preflight' else 'prepare',
                     '--data',str(data),'--embedding-root',str(embedding),
                     '--lock',str(lock),'--output',str(bundle)]
    if type(bundle_sha) is not str or len(bundle_sha)!=64 or any(c not in '0123456789abcdef' for c in bundle_sha):
        raise ValueError('fixed input manifest SHA required')
    return argv+['fit','--bundle',str(bundle),'--bundle-sha',bundle_sha,
                 '--device','cpu' if stage=='cpu-fit' else 'cuda',
                 '--lock',str(lock),'--output',str(output)]


def predecessor(args,source):
    if args.stage=='input-preflight':return {}
    path=secure(args.predecessor_run)
    run=json.loads((path/'run.json').read_bytes())
    terminal=json.loads((path/'terminal.json').read_bytes())
    lines=(path/'logs/stdout.log').read_bytes().splitlines()
    receipt=json.loads(next((line for line in reversed(lines) if line.strip()),b'{}'))
    prior='input-preflight' if args.stage=='input-prepare' else 'input-prepare'
    expected=worker_argv(prior,args.data,args.embedding_root,args.lock,args.bundle,args.output)
    if (path.name!='sgns-gpu-subset-'+prior+'-v1' or run.get('run_id')!=path.name
            or terminal.get('run_id')!=path.name or terminal.get('status')!='succeeded'
            or terminal.get('exit_code')!=0 or run.get('kind')!='ops' or run.get('argv')!=expected
            or run.get('git',{}).get('commit')!=source['commit'] or receipt.get('source')!=source
            or receipt.get('records')!=fixed_records() or receipt.get('engineering') is not False
            or receipt.get('training') is not False
            or receipt.get('status')!=('verified' if prior=='input-preflight' else 'prepared')
            or receipt.get('payload_written')!=(prior=='input-prepare')):
        raise ValueError('formal input predecessor binding differs')
    if prior=='input-prepare':
        manifest=secure(args.bundle/'manifest.json')
        if digest(manifest)!=args.bundle_sha or json.loads(manifest.read_bytes())!=receipt:
            raise ValueError('bundle differs from original prepared stdout')
    return {name:digest(path/name) for name in ('run.json','terminal.json','logs/stdout.log')}


def job(args):
    from analysis.a40_ops import load_ops
    from analysis.ur001_derive_cli import source_identity
    _,launcher,scheduler,_=load_ops(args.ops_code)
    source=source_identity()
    proof=predecessor(args,source)
    argv=worker_argv(args.stage,args.data,args.embedding_root,args.lock,args.bundle,args.output,args.bundle_sha)
    run_id='sgns-gpu-subset-'+args.stage+'-v1'
    flags=['--root',str(args.ops_root),'--run-id',run_id,'--kind','ops','--cwd',str(ROOT),
           '--config',str(ROOT/'configs/a40-subset/hf-files.json'),'--seed','42','--input',str(args.lock),
           '--env','PYTHONPATH='+str(ROOT/'src'),'--env','OMP_NUM_THREADS=1',
           '--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
           '--env','CUBLAS_WORKSPACE_CONFIG=:4096:8','--cpu-cores','1','--ram-mib','24576',
           '--vram-mib','8192' if args.stage=='cuda-fit' else '0','--disk-gib','2',
           '--wall-hours','24' if args.stage.endswith('fit') else '2']
    if args.predecessor_run:
        for name in proof:flags+=['--input',str(args.predecessor_run/name)]
    if args.bundle_sha:flags+=['--input',str(args.bundle/'manifest.json')]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]))
    metadata=launcher.prepare(spec)
    result={'status':'prepared_dry','spec':spec,'metadata':metadata,'predecessor':proof,
            'resource_scope':'not_measured conservative first limits; subsequent declarations require measured peak',
            'subset_four_gate_acceptance':False}
    if args.command=='queue-dry':return result
    root=launcher.resolve_root(str(args.ops_root))
    if (root/'runs'/run_id).exists():raise FileExistsError('original run exists')
    for state in ('pending','claimed','rejected'):
        if list((root/'queue'/state).glob('*-'+run_id+'.json')):raise FileExistsError('run already queued')
    path=scheduler.submit(root,spec)
    return {**result,'status':'written','queue_file':path.name}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for name in ('queue-dry','queue-submit'):
        c=sub.add_parser(name);c.add_argument('--stage',choices=STAGES,required=True)
        for flag in ('data','embedding-root','lock','bundle','output','ops-code','ops-root'):
            c.add_argument('--'+flag,type=Path,required=True)
        c.add_argument('--bundle-sha');c.add_argument('--predecessor-run',type=Path)
    args=p.parse_args(argv)
    if args.command=='dry':value={'status':'dry','stages':STAGES,'training':False,'payload_read':False}
    else:
        if args.stage!='input-preflight' and args.predecessor_run is None:p.error('formal predecessor required')
        value=job(args)
    print(json.dumps(value,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
