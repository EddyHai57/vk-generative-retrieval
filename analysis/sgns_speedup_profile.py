"""Bounded, non-scientific investigation of the unchanged frozen SGNS function.

Runs at most 16 original batches into temporary in-memory tables, then aborts
before vector export/health/acceptance. Does not save vectors or submit jobs.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
NATIVE_SHA='2cc25d1997c12fe1e262e455e6644980186b5b9d0e00805014dee3148675876d'
FULL_PAIRS=18787015620
OBSERVED_RATE=95922.48482331433


def native():
    from vkgr import sid_coll_sgns as original
    path=ROOT/'src/vkgr/sid_coll_sgns.py'
    if Path(original.__file__).resolve()!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=NATIVE_SHA:
        raise ValueError('unchanged original SGNS bytes required')
    return original


def frozen_configuration():
    original=native()
    return {'source_sha256':NATIVE_SHA,'config':asdict(original.SGNSConfig()),
            'high_frequency_subsampling':False,'frequency_power_for_negatives':.75,
            'note':'frequency**.75 changes negative distribution only; no deletion of high-frequency events'}


def keep_probability(frequency,threshold):
    """word2vec sample-style candidate, not an approved new SGNS algorithm."""
    import math
    if not 0<frequency<=1 or not 0<threshold<1:raise ValueError('positive relative frequency and threshold required')
    return min(1.,math.sqrt(threshold/frequency)+threshold/frequency)


def pair_count(histories,window):
    if type(window) is not int or window<1:raise ValueError('positive window required')
    return sum(1 for _ in native().iter_positive_pairs(histories,window))


def scenarios():
    baseline=FULL_PAIRS/OBSERVED_RATE/3600
    return {'baseline':{'positive_pairs':FULL_PAIRS,'positive_pairs_per_epoch':FULL_PAIRS//5,'fit_hours':baseline,
                        'rate_scope':'prior subset original fit elapsed, includes its setup/finalization; not full total wall'},
            'B':{'positive_pairs':FULL_PAIRS,'fit_hours':'UNKNOWN','hyperparameters_changed':False,
                 'idealized_speedup_scenarios':[{'speedup':s,'fit_hours':baseline/s} for s in (2,4,8)],
                 'warning':'hypothetical multipliers, not measured speedup; GPU RNG can alter negative stream'},
            'C1':[{'threshold':t,'positive_pairs':'UNKNOWN','fit_hours':'UNKNOWN','hyperparameters_changed':True}
                  for t in (1e-3,1e-4,1e-5)],
            'C2':[{'batch_size':b,'positive_pairs':FULL_PAIRS,'steps_per_epoch':-(-(FULL_PAIRS//5)//b),
                   'fit_hours':'UNKNOWN','hyperparameters_changed':True} for b in (8192,16384,32768)],
            'C3_epochs':[{'epochs':e,'positive_pairs':FULL_PAIRS//5*e,'linear_fit_hours':baseline*e/5,
                          'hyperparameters_changed':True,'total_wall':'UNKNOWN'} for e in (3,1)],
            'C3_window':[{'window':w,'positive_pairs':'UNKNOWN','fit_hours':'UNKNOWN','hyperparameters_changed':True}
                         for w in (3,2,1)],
            'total_wall':'UNKNOWN: input/setup+fit+verification+SID/trie+G acceptance costs not all measured'}


def synthetic_inputs():
    import numpy as np
    # Modulo strides give repeatable nonidentical neighbors and >16 batches.
    ids=np.arange(1,257,dtype=np.uint32)
    histories={u:[1+(u*17+j*7)%256 for j in range(96)] for u in range(128)}
    content=np.ones((len(ids),64),dtype=np.float32)
    return ids,histories,content


class _PrefixComplete(BaseException):
    pass


def _section(line):
    # Native SHA pins the exact original line mapping below. No statement changes.
    if line==225:return 'assembly'
    if line==226:return 'negative_sampling'
    if line in (233,234):return 'host_to_device'
    if 235<=line<=243:return 'forward_backward_finite_checks_optimizer'
    return 'python_bookkeeping_and_loss_device_to_host'


def cuda_times(events,device):
    if device!='cuda':return 'UNKNOWN','UNKNOWN'
    events=[e for e in events if str(e.device_type).endswith('CUDA')]
    kernels=sum(e.time_range.elapsed_us() for e in events if 'memcpy' not in e.name.lower() and 'memset' not in e.name.lower())
    copies=sum(e.time_range.elapsed_us() for e in events if 'memcpy' in e.name.lower())
    return kernels,copies


def profile_prefix(inputs,device,batches):
    """Trace original nested batch; scope timings are instrumented host wall.

    CUDA profiler events separately expose kernels/copies. Host scopes include
    waits from finite checks and loss.cpu; never sum CUDA and host durations.
    """
    import torch
    from torch.profiler import profile,ProfilerActivity,record_function
    if type(batches) is not int or not 1<=batches<=16:raise ValueError('bounded batches 1..16 required')
    if device not in ('cpu','cuda'):raise ValueError('cpu/cuda only')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no CPU fallback')
    original=native();fit=original.fit_sgns
    from itertools import islice
    required_pairs=batches*original.SGNSConfig().batch_size
    if sum(1 for _ in islice(original.iter_positive_pairs(inputs[1],original.SGNSConfig().window),required_pairs))<required_pairs:
        raise ValueError('insufficient original positive pairs for bounded full batches')
    step_code=next(code for code in fit.__code__.co_consts if inspect.iscode(code) and code.co_name=='train_batch')
    previous_trace=sys.gettrace()
    if previous_trace is not None:raise ValueError('existing trace must not be replaced')
    activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device=='cuda' else [])
    profiler=profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False)
    sections={};completed=0;positives=0;batch_wall=0.;first_batch_at=None
    active=None;active_name=None;active_at=None;batch_at=None
    previous_batch_end=None;pair_loop_seconds=0.;pair_loop_intervals=0
    def close():
        nonlocal active,active_at
        if active is not None:
            elapsed=time.perf_counter()-active_at
            active.__exit__(None,None,None)
            sections[active_name]=sections.get(active_name,0.)+elapsed
            active=None
    def lines(frame,event,arg):
        nonlocal active,active_name,active_at,completed,positives,batch_wall,previous_batch_end
        if event=='line':
            name=_section(frame.f_lineno)
            if name!=active_name or active is None:
                close();active_name=name;active=record_function('sgns_profile::'+name)
                active.__enter__();active_at=time.perf_counter()
        elif event=='return':
            close();completed+=1;positives+=len(frame.f_locals['batch'])
            batch_wall+=time.perf_counter()-batch_at
            previous_batch_end=time.perf_counter()
            if completed>=batches:raise _PrefixComplete()
        elif event=='exception':close()
        return lines
    def calls(frame,event,arg):
        nonlocal first_batch_at,batch_at,active_name,pair_loop_seconds,pair_loop_intervals
        if event=='call' and frame.f_code is step_code:
            batch_at=time.perf_counter();active_name=None
            if previous_batch_end is not None:
                pair_loop_seconds+=batch_at-previous_batch_end;pair_loop_intervals+=1
            if first_batch_at is None:first_batch_at=batch_at
            return lines
        return None
    start=time.perf_counter();aborted=False
    # Exception exits the unchanged original try/finally, restoring native torch
    # settings and dropping temporary tables. The function cannot reach exports.
    with profiler:
        try:
            sys.settrace(calls)
            fit(*inputs,device=device,engineering=False)
        except _PrefixComplete:aborted=True
        finally:
            sys.settrace(previous_trace);close()
    elapsed=time.perf_counter()-start
    if not aborted:raise ValueError('input too small: fit reached completion instead of bounded prefix; results discarded')
    if native().fit_sgns is not fit:raise ValueError('original fit callable changed')
    events=profiler.events()
    cuda_kernel_us,cuda_copy_us=cuda_times(events,device)
    top=sorted(profiler.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:15]
    return {'schema':'sgns-bounded-prefix-profile-v1','device':device,'synthetic':False,
            'completed_batches':completed,'positive_pairs':positives,'original_config':asdict(original.SGNSConfig()),
            'original_source_sha256':NATIVE_SHA,'profiler_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_fit_code_unchanged':True,
            'instrumented_sections_seconds':sections,'instrumented_pair_loop_between_batches_seconds':pair_loop_seconds,
            'pair_loop_intervals':pair_loop_intervals,'instrumented_batch_host_wall_seconds':batch_wall,
            'elapsed_including_setup_profiler_finalize_seconds':elapsed,
            'setup_to_first_batch_seconds':first_batch_at-start,'cuda_kernel_us':cuda_kernel_us,'cuda_memcpy_us':cuda_copy_us,
            'top_cpu_ops':[{'name':e.key,'calls':e.count,'self_cpu_us':e.self_cpu_time_total} for e in top],
            'vectors_returned':False,'vectors_saved':False,'science_training':False,'scientific_verified':False,
            'temporary_parameter_updates':True,'test_read':False,
            'runtime':{'python':sys.version.split()[0],'torch':str(torch.__version__),'cuda':torch.version.cuda},
            'limits':['instrumented prefix only; no steady-state throughput claim','host scopes include GPU waits; do not add to CUDA duration',
                      'CUDA kernel/memcpy totals include setup and bounded prefix, not per-batch-only',
                      'between-batch pair-loop intervals instrumented; first batch assembly included in setup','no context-inclusive peak VRAM measurement']}


def subset_inputs(bundle,sha,lock):
    from analysis.ur001_derive_cli import secure,digest
    from analysis.sgns_gpu_subset import read_bundle
    from analysis.new_instance_prepare import runtime
    from vkgr.sgns_subset_a1 import NATIVE_LOCK_SHA
    if lock is None:raise ValueError('frozen runtime lock required')
    lock=secure(lock)
    if digest(lock)!=NATIVE_LOCK_SHA:raise ValueError('frozen runtime lock SHA differs')
    bundle=secure(bundle)
    path=bundle/'manifest.json'
    if digest(path)!=sha:raise ValueError('fixed subset manifest SHA required')
    producer=json.loads(path.read_bytes())
    if runtime(lock)!=producer.get('runtime'):raise ValueError('subset producer runtime differs')
    # The existing reader validates original subset pins and all consumed bytes.
    # Its source argument is the producer identity, not a claim that this new
    # profiler has that producer commit. Consumer scientific bytes pin above.
    return read_bundle(bundle,sha,producer['source'])[0]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry');sub.add_parser('estimates')
    run=sub.add_parser('profile');run.add_argument('--device',choices=('cpu','cuda'),required=True)
    run.add_argument('--batches',type=int,default=4)
    source=run.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic',action='store_true');source.add_argument('--subset-bundle',type=Path)
    run.add_argument('--subset-manifest-sha256');run.add_argument('--lock',type=Path)
    args=p.parse_args(argv)
    try:
        if args.command=='dry':result={'status':'dry','configuration':frozen_configuration(),'science_training':False,'cuda_measured':False}
        elif args.command=='estimates':result=scenarios()
        else:
            if args.subset_bundle is not None and args.subset_manifest_sha256 is None:raise ValueError('subset manifest SHA required')
            inputs=synthetic_inputs() if args.synthetic else subset_inputs(args.subset_bundle,args.subset_manifest_sha256,args.lock)
            result=profile_prefix(inputs,args.device,args.batches)
            result['synthetic']=args.synthetic
            if not args.synthetic:
                subset_inputs(args.subset_bundle,args.subset_manifest_sha256,args.lock)
                result['subset_manifest_sha256']=args.subset_manifest_sha256
        print(json.dumps(result,sort_keys=True));return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'science_training':False},sort_keys=True));return 2


if __name__=='__main__':raise SystemExit(main())
