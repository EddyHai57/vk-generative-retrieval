"""Original P exact scoring arithmetic, parallel only over complete query blocks.

Owned implementation mirrors 0ca61bc baselines.blockwise_topk's fixed 32/4096
calls and item-ID tie breaks. No GPU formula, partition selection, or ANN.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
from .throughput_workers import ordered_map
from .throughput_config import guard, rss_bytes

_DATA=None


def _descriptor(array):
    if isinstance(array,np.memmap):
        path=Path(array.filename).resolve()
        if not path.is_file()or path.suffix!='.npy' or not array.flags.c_contiguous:
            raise ValueError('mmap input must be a full .npy mmap')
        try:full=np.load(path,mmap_mode='r',allow_pickle=False)
        except (OSError,ValueError):raise ValueError('mmap input must be a full .npy mmap')from None
        if (not isinstance(full,np.memmap)or full.shape!=array.shape or full.dtype!=array.dtype
            or full.strides!=array.strides or getattr(full,'offset',None)!=getattr(array,'offset',None)):
            raise ValueError('mmap input must be a full .npy mmap')
        return {'mmap':str(path),'shape':array.shape,'dtype':array.dtype.str}
    return array


def _load(value):
    if isinstance(value,dict):
        array=np.load(value['mmap'],mmap_mode='r',allow_pickle=False)
        if array.shape!=value['shape']or array.dtype.str!=value['dtype']:raise ValueError('mmap descriptor identity differs')
        return array
    return value


def _init(q,items,ids,k,trace):
    global _DATA
    _DATA=(_load(q),_load(items),_load(ids),k,trace)


def _block(start):
    queries,items,ids,k,trace=_DATA;q=queries[start:start+32]
    best_ids=[np.empty(0,dtype=ids.dtype)for _ in q];best_scores=[np.empty(0,dtype=np.float64)for _ in q];digests=[]
    for offset in range(0,len(items),4096):
        scores=np.einsum('qd,id->qi',q,items[offset:offset+4096],dtype=np.float64,optimize=False)
        if trace:digests.append((scores.shape,hashlib.sha256(scores.tobytes()).hexdigest()))
        block_ids=ids[offset:offset+4096]
        for row in range(len(q)):
            combined_ids=np.concatenate((best_ids[row],block_ids));combined_scores=np.concatenate((best_scores[row],scores[row]))
            take=np.lexsort((combined_ids,-combined_scores))[:k]
            best_ids[row]=combined_ids[take];best_scores[row]=combined_scores[take]
    return np.array(best_ids,dtype=ids.dtype).reshape(len(q),k),digests


def exact_topk(queries,items,ids,k,config,*,trace=False):
    original=(queries,items,ids)
    if config.memory_limit_bytes and any(not isinstance(a,np.ndarray)for a in original):
        raise ValueError('budgeted exact inputs require shape-known numpy arrays/memmaps')
    # Establish an upper bound for dtype conversion and validation temporaries
    # before np.asarray, isfinite, or unique can allocate proportional arrays.
    q_shape=getattr(queries,'shape',());item_shape=getattr(items,'shape',());id_shape=getattr(ids,'shape',())
    q_size=math.prod(q_shape);item_size=math.prod(item_shape);id_size=math.prod(id_shape)
    q_bytes=q_size*np.dtype('f4').itemsize;item_bytes=item_size*np.dtype('f4').itemsize
    id_itemsize=getattr(getattr(ids,'dtype',None),'itemsize',8);id_bytes=id_size*id_itemsize
    cast_bytes=(q_bytes if getattr(queries,'dtype',None)!=np.dtype('f4') else 0)+(item_bytes if getattr(items,'dtype',None)!=np.dtype('f4') else 0)
    validation_bytes=max(q_size,item_size)+4*id_bytes
    guard(config,Path.cwd(),projected_memory=cast_bytes+validation_bytes)
    if config.eval_workers>1 and config.use_mmap:
        if not isinstance(items,np.memmap):raise ValueError('parallel mmap catalog must be a full .npy memmap')
        if items.dtype!=np.dtype('f4'):raise ValueError('parallel mmap catalog must be float32')
        _descriptor(items)
    # Reject every mmap that would be handed to a worker as an ambiguous slice.
    if config.eval_workers>1:
        for value in original:
            if isinstance(value,np.memmap):_descriptor(value)
    queries=np.asarray(queries,dtype=np.float32);items=np.asarray(items,dtype=np.float32);ids=np.asarray(ids)
    if queries.ndim!=2 or items.ndim!=2 or queries.shape[1]!=items.shape[1]or len(items)!=len(ids)or ids.ndim!=1 or not len(ids):raise ValueError('invalid score shapes')
    if not np.isfinite(queries).all()or not np.isfinite(items).all()or len(np.unique(ids))!=len(ids)or not np.issubdtype(ids.dtype,np.integer)or np.any(ids<0):raise ValueError('invalid scores/IDs')
    if type(k)is not int or k<1:raise ValueError('invalid k')
    k=min(k,len(ids))
    data=[]
    for before,after in zip(original,(queries,items,ids)):
        data.append(_descriptor(before)if config.eval_workers>1 and isinstance(before,np.memmap)and before.dtype==after.dtype else after)
    worker_bytes=sum(a.nbytes for a in (queries,items,ids))
    projected=worker_bytes*config.eval_workers
    guard(config,Path.cwd(),projected_memory=projected)
    worker_memory=0
    if config.memory_limit_bytes:
        if config.eval_workers==1:worker_memory=config.memory_limit_bytes
        else:worker_memory=worker_bytes+max(0,(config.memory_limit_bytes-rss_bytes()-projected)//config.eval_workers)
    output=[];digests=[]
    for predicted,trace_block in ordered_map(_block,range(0,len(queries),32),workers=config.eval_workers,prefetch=config.prefetch,
        initializer=_init,initargs=(*data,k,trace),max_in_flight_bytes=config.max_in_flight_bytes,
        result_size_hint=lambda _:32*k*ids.dtype.itemsize+8192*(1+len(items)//4096),memory_limit_bytes=worker_memory):
        guard(config,Path.cwd());output.append(predicted);digests.append(trace_block)
    return np.concatenate(output)if output else np.empty((0,k),dtype=ids.dtype),digests
