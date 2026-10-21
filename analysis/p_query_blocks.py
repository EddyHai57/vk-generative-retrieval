"""Read-only query-block parity tooling; never submits or alters original jobs."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace
import numpy as np
from vkgr.p_query_blocks import parallel_topk, cpu_capacity
from vkgr.throughput_config import Engineering, guard
from analysis.a40_cost import REFERENCE, REFERENCE_SHA, sha_file

ROOT=Path(__file__).resolve().parents[1]


def baseline():
    if sha_file(REFERENCE)!=REFERENCE_SHA:raise ValueError('original baseline source drift')
    namespace={};exec(compile(REFERENCE.read_bytes(),str(REFERENCE),'exec'),namespace)
    return namespace['blockwise_topk']


def parity(queries,items,ids,config,*,k=50):
    reference=baseline();expected_scores=[]
    def observed(*args,**kwargs):
        scores=np.einsum(*args,**kwargs)
        expected_scores.append((scores.shape,hashlib.sha256(scores.tobytes()).hexdigest()))
        return scores
    proxy=SimpleNamespace(**{**vars(np),'einsum':observed})
    oracle=FunctionType(reference.__code__,{**reference.__globals__,'np':proxy},reference.__name__,reference.__defaults__)
    # Reserve validation/allocation space for the serial oracle as well.
    guard(config,Path.cwd(),projected_memory=max(queries.size,items.size)+4*ids.nbytes+32*4096*8)
    expected=oracle(queries,items,ids,k,query_block=32,item_block=4096)
    started=time.perf_counter()
    actual,trace=parallel_topk(queries,items,ids,k,config,trace=True)
    elapsed=time.perf_counter()-started
    if expected.tobytes()!=actual.tobytes():raise ValueError('original prediction bytes differ')
    if expected_scores!=[entry for block in trace for entry in block]:raise ValueError('original score bytes differ')
    return {'status':'verified','scope':'same-input original 32/4096 reference and query-block workers',
            'queries':len(queries),'items':len(items),'dim':items.shape[1],'top_k':min(k,len(ids)),
            'score_bytes_equal_to_original':True,'prediction_bytes_equal_to_original':True,
            'score_blocks_checked':len(expected_scores),'baseline_sha256':REFERENCE_SHA,
            'prediction_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
            'parallel_wall_seconds':elapsed,'original_cost_terminal_compared':False}


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE,text=True).strip()


def bind_original_cost(engine,source,root,lock,ops_root):
    """Require original successful terminal/stdout before opening any array.

    The original producer module stays in its own clean fixed checkout. Its
    existing input/runtime/generator/preflight proof functions remain binding.
    """
    engine=Path(engine).resolve(strict=True)
    if git(engine,'rev-parse','HEAD')!=source or git(engine,'status','--porcelain','--untracked-files=all'):
        raise ValueError('original engine identity differs')
    # Do not execute unpinned external Python, even from an asserted git SHA.
    script=engine/'analysis/a40_cost.py'
    if sha_file(script)!=sha_file(ROOT/'analysis/a40_cost.py'):raise ValueError('original cost source bytes differ')
    spec=importlib.util.spec_from_file_location('_p_original_cost',script)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cost=json.loads((root/'cost.json').read_text())
    pins={str(lock.resolve()):sha_file(lock),str((root/'inputs.json').resolve()):sha_file(root/'inputs.json'),
          str((root/'preflight.json').resolve()):sha_file(root/'preflight.json')}
    producer=old.accepted_stage(ops_root,'run',source,old.worker_argv('run',root,lock,ops_root),pins,
                               {'path':str(old.YAML),'sha256':old.YAML_SHA},cost)
    arrays,env=old.inputs(root,lock,ops_root)
    pre=json.loads((root/'preflight.json').read_text())
    old.require_preflight(pre,sha_file(root/'inputs.json'),sha_file(script),old.YAML_SHA,env['generation_stdout_sha256'])
    predecessor=old.proof(ops_root,'preflight',root,lock,pre)
    expected={'status':'exact_cost_measured','engineering':False,'synthetic_input':True,'science_data_read':False,
              'items':old.N_ITEMS,'queries':old.N_QUERIES,'dim':old.DIM,'top_k':old.TOP_K,'workers':1,
              'baseline_sha256':REFERENCE_SHA,'script_sha256':sha_file(script),'selected_yaml_sha256':old.YAML_SHA,
              'inputs_manifest_sha256':sha_file(root/'inputs.json'),'preflight_stdout_sha256':predecessor['stdout_sha256'],
              'source':source,**env}
    if any(cost.get(key)!=value or type(cost.get(key))is not type(value) for key,value in expected.items()):
        raise ValueError('original cost payload binding differs')
    return arrays,cost,producer


def cost_parity(args,config):
    arrays,cost,producer=bind_original_cost(args.original_engine,args.original_source,args.root,args.lock,args.ops_root)
    items,queries,ids=arrays
    started=time.perf_counter();predicted,_=parallel_topk(queries,items,ids,50,config)
    elapsed=time.perf_counter()-started
    digest=hashlib.sha256(predicted.tobytes()).hexdigest()
    if digest!=cost['prediction_sha256']:raise ValueError('original cost prediction hash differs')
    return {'status':'verified','original_cost_terminal_compared':True,'prediction_sha256':digest,
            'original_cost_stdout_sha256':producer['stdout_sha256'],'original_source':args.original_source,
            'baseline_sha256':REFERENCE_SHA,'parallel_wall_seconds':elapsed,
            'score_bytes_equal_to_original':'UNKNOWN: original cost did not retain score traces',
            'scope':'full original cost prediction hash; per-score evidence requires parity command'}


def prepare(args):
    from analysis.a40_ops import load_ops
    _,launcher,_,_=load_ops(args.ops_code)
    # Prepare only; this adapter deliberately exposes no submit path.
    argv=[sys.executable,'-B','-m','analysis.p_query_blocks','dry']
    raw=['--root',str(args.ops_root),'--run-id',args.run_id,'--kind','ops','--cwd',str(ROOT),
         '--env','PYTHONPATH='+str(ROOT/'src'),'--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
         '--env','MKL_NUM_THREADS=1','--cpu-cores','1','--ram-mib','1024','--vram-mib','0','--disk-gib','0',
         '--wall-hours','1','--',*argv]
    if args.p3c_data_channel:
        expected='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
        if sha_file(ROOT/'docs/p3c_validation.md')!=expected:raise ValueError('P3C data-channel bytes differ')
        raw[-len(argv)-1:-len(argv)-1]=['--pinned-file','docs/p3c_validation.md='+expected]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args(raw))
    return {'status':'prepared_dry','spec':spec,'metadata':launcher.prepare(spec),'submit_exposed':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    p=sub.add_parser('prepare');p.add_argument('--ops-code',required=True);p.add_argument('--ops-root',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--p3c-data-channel',action='store_true')
    for name in ('parity','cost-parity'):
        p=sub.add_parser(name);p.add_argument('--workers',type=int,required=True)
        p.add_argument('--memory-limit-bytes',type=int,required=True);p.add_argument('--output',type=Path,required=True)
        if name=='parity':
            for flag in ('queries','items','ids'):p.add_argument('--'+flag,type=Path,required=True)
        else:
            for flag in ('original-engine','root','lock','ops-root'):p.add_argument('--'+flag,type=Path,required=True)
            p.add_argument('--original-source',required=True)
    args=parser.parse_args(argv)
    if args.command=='dry':value={'status':'dry','baseline_sha256':REFERENCE_SHA,'query_block':32,'item_block':4096,'data_read':False,'submit_exposed':False}
    elif args.command=='prepare':value=prepare(args)
    else:
        if args.output.exists() or args.output.is_symlink():raise FileExistsError('exclusive output required')
        if args.memory_limit_bytes<=0:raise ValueError('positive memory limit required')
        # Cap native BLAS pools before spawn. Fixed einsum optimize=False remains.
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
        config=Engineering(eval_workers=args.workers,use_mmap=True,prefetch=1,
                           memory_limit_bytes=args.memory_limit_bytes,max_in_flight_bytes=64*1024**2)
        guard(config,args.output.parent,projected_disk=65536)
        if args.command=='parity':
            paths=[args.queries,args.items,args.ids];pins=[sha_file(path) for path in paths]
            value=parity(*(np.load(path,mmap_mode='r',allow_pickle=False) for path in paths),config)
            if pins!=[sha_file(path) for path in paths]:raise ValueError('input bytes changed during parity')
            value['input_sha256']=dict(zip(('queries','items','ids'),pins))
        else:value=cost_parity(args,config)
        value.update({'engineering':True,'script_sha256':sha_file(__file__),'configuration':asdict(config),
                      'runtime':{'python':platform.python_version(),'numpy':np.__version__},'cpu_capacity':cpu_capacity(),
                      'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                      'process_tree_peak_rss_bytes':'UNKNOWN','measured_server_resources':False})
        with args.output.open('x') as f:json.dump(value,f,sort_keys=True)
    print(json.dumps(value,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
