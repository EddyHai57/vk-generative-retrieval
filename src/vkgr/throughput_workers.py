"""Spawn-only bounded ordered pure tasks. No sampler or RNG lives in workers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import copy
import multiprocessing as mp
import pickle
import random
import sys
from pathlib import Path


def rng_state():
    state={'python':random.getstate()}
    if 'numpy'in sys.modules:
        np=sys.modules['numpy'];state['numpy']=np.random.get_state()
    if 'torch'in sys.modules:
        torch=sys.modules['torch'];state['torch']=torch.get_rng_state().numpy().tobytes()
        if torch.cuda.is_initialized():state['cuda']=[x.cpu().numpy().tobytes()for x in torch.cuda.get_rng_state_all()]
    return pickle.dumps(state,protocol=5)


def _task(function,ordinal,value,result_bound=0,memory_limit_bytes=0):
    from .throughput_config import rss_bytes
    if memory_limit_bytes and rss_bytes()>memory_limit_bytes:raise MemoryError('worker RSS budget exceeded')
    before=rng_state();result=function(value)
    if rng_state()!=before:raise ValueError('worker consumed random state')
    if result_bound and len(pickle.dumps(result,protocol=5))>result_bound:raise MemoryError('worker result exceeded reserved output bound')
    if memory_limit_bytes and rss_bytes()>memory_limit_bytes:raise MemoryError('worker RSS budget exceeded')
    return ordinal,result


def ordered_map(function,values,*,workers=1,prefetch=1,initializer=None,initargs=(),max_in_flight_bytes=0,result_size_hint=None,memory_limit_bytes=0):
    if any(type(x)is not int or x<1 for x in (workers,prefetch)):raise ValueError('positive workers/prefetch required')
    if workers==1:
        if initializer:initializer(*initargs)
        for ordinal,value in enumerate(values):
            bound=result_size_hint(value)if result_size_hint else 0
            if result_size_hint is not None and (type(bound)is not int or bound<1):raise ValueError('positive output-size bound required')
            if max_in_flight_bytes and (not bound or 3*(len(pickle.dumps(value,protocol=5))+bound)>max_in_flight_bytes):raise MemoryError('input/output reservation exceeds in-flight cap')
            _,result=_task(function,ordinal,value,bound,memory_limit_bytes);yield result
        return
    iterator=iter(enumerate(values));pending=deque();used=0;waiting=None
    if max_in_flight_bytes and result_size_hint is None:raise ValueError('bounded output-size contract required')
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initializer,initargs=initargs)
    try:
        ended=False
        while pending or waiting is not None or not ended:
            while not ended and len(pending)<workers*prefetch:
                if waiting is None:
                    try:ordinal,value=next(iterator)
                    except StopIteration:ended=True;break
                    bound=result_size_hint(value)if result_size_hint else 0
                    if result_size_hint is not None and (type(bound)is not int or bound<1):raise ValueError('positive output-size bound required')
                    size=3*(len(pickle.dumps(value,protocol=5))+bound)
                    waiting=(ordinal,value,bound,size)
                ordinal,value,bound,size=waiting
                if max_in_flight_bytes and size>max_in_flight_bytes:raise MemoryError('one task exceeds input/output reservation cap')
                if max_in_flight_bytes and used+size>max_in_flight_bytes:break
                waiting=None
                pending.append((ordinal,pool.submit(_task,function,ordinal,value,bound,memory_limit_bytes),size));used+=size
            if pending:
                expected,future,size=pending.popleft();ordinal,result=future.result();used-=size
                if ordinal!=expected:raise ValueError('worker ordinal mismatch')
                yield result
    finally:
        for _,future,_ in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)


_STORE=None


def _store_init(store):
    global _STORE
    if isinstance(store,str):
        from .throughput_sequences import SequenceStore
        _STORE=SequenceStore.load(store,mmap=True)
    else:_STORE=store


def _rows(task):
    indices,length=task
    return _STORE.rows(indices,length=length)


def assemble_rows(store,batches,config,*,length=50):
    # Caller advances the original cursor and makes all random choices BEFORE
    # yielding each indices list. Worker initializer is once per process.
    if config.use_mmap and store.root is None:raise ValueError('persist/load mmap store before parallel use')
    if config.assembly_workers>1 and not config.use_mmap:
        from .throughput_config import guard
        copies=sum(getattr(store,n).nbytes for n in store.COLUMNS)*config.assembly_workers
        guard(config,Path.cwd(),projected_memory=copies)
    descriptor=store.root if config.use_mmap else store
    worker_memory=0
    if config.memory_limit_bytes:
        from .throughput_config import rss_bytes
        worker_memory=(config.memory_limit_bytes-rss_bytes()-config.max_in_flight_bytes)//config.assembly_workers
        if worker_memory<=0:raise MemoryError('no memory budget for worker processes')
    def checked():
        from .throughput_config import guard
        for rows in ordered_map(_rows,((list(b),length)for b in batches),workers=config.assembly_workers,
            prefetch=config.prefetch,initializer=_store_init,initargs=(descriptor,),max_in_flight_bytes=config.max_in_flight_bytes,
            result_size_hint=lambda task:store.rows_bound(task[0],task[1]),memory_limit_bytes=worker_memory):
            guard(config,Path.cwd())
            yield rows
    return checked()


def cursor_rows(store,cursor,config,*,batch_size,length=50,method='next'):
    """Speculate cursor on a COPY; commit original cursor only at step delivery.

    Random negative choices remain in the original main-process train step.
    Prefetch must not advance the live scientific cursor/RNG into future steps.
    """
    probe=copy.deepcopy(cursor)
    def batches():
        while True:
            indices=getattr(probe,method)(batch_size)
            if not indices:return
            yield indices
            # P cursor rolls epochs itself; this bounded stream covers one epoch.
            offset=getattr(probe,'offset',getattr(probe,'next_offset',None))
            count=getattr(probe,'size',getattr(probe,'example_count',None))
            if offset==count:return
    expected=copy.deepcopy(cursor)
    for rows in assemble_rows(store,batches(),config,length=length):
        planned=getattr(expected,method)(batch_size)
        actual=getattr(cursor,method)(batch_size)
        if actual!=planned or [r['example_id']for r in rows]!=actual:raise ValueError('live cursor differs from planned ordinal')
        yield rows
