"""Config-only orchestration of the optional engineering kernels.

This is NOT a scientific backend: callers supply the frozen original functions,
sample references and identities. No model/data path is discovered implicitly.
"""
import copy
import json
import os
from pathlib import Path
from .throughput_config import Engineering,manifest,guard,rss_bytes
from .throughput_sequences import SequenceStore
from .throughput_workers import cursor_rows
from .throughput_tensors import cursor_prepared_batches
from .throughput_bhkm import fit_configured
from .throughput_weekly import derive_weeks
from .throughput_exact import exact_topk
from .throughput_decode import decode_queries
from .throughput_probe import resource_manifest


class Toolchain:
    def __init__(self,config,science_identity,manifest_path):
        self.config=config if isinstance(config,Engineering)else Engineering.load(config)
        if self.config.use_mmap and not self.config.sequence_enabled:
            raise ValueError('mmap requires configured sequence store')
        self.science=copy.deepcopy(science_identity)
        record=manifest(self.science,self.config)
        with Path(manifest_path).open('x')as f:
            json.dump(record,f,sort_keys=True);f.flush();os.fsync(f.fileno())

    def examples(self,original,columns,*,run_id):
        if not self.config.sequence_enabled:return original
        guard(self.config,Path.cwd())
        limit=0
        if self.config.memory_limit_bytes:
            limit=self.config.memory_limit_bytes-rss_bytes(include_children=True)
            if limit<=0:raise MemoryError('no memory for sequence construction')
        store=SequenceStore.from_columns(**columns,byte_limit=limit)
        if self.config.use_mmap:
            if not self.config.temp_dir or not isinstance(run_id,str)or not run_id or '/'in run_id or run_id in ('.','..'):
                raise ValueError('exclusive configured sequence namespace required')
            path=Path(self.config.temp_dir)/run_id/'sequence'
            if any(p.is_symlink()for p in (path,*path.parents)):raise ValueError('sequence path symlink forbidden')
            parent=path.parent
            while not parent.exists():parent=parent.parent
            guard(self.config,parent,projected_disk=sum(getattr(store,k).nbytes+4096 for k in store.COLUMNS))
            store.save(path);store=SequenceStore.load(path,mmap=True)
        guard(self.config,Path.cwd());return store

    def batches(self,store,cursor,*,batch_size,length=50,method='next'):
        if not self.config.sequence_enabled and self.config.assembly_workers==1:
            while True:
                indices=getattr(cursor,method)(batch_size)
                if not indices:return
                yield store.rows(indices)if hasattr(store,'rows')else[store[i]for i in indices]
                offset=getattr(cursor,'offset',getattr(cursor,'next_offset',None))
                count=getattr(cursor,'size',getattr(cursor,'example_count',None))
                if offset==count:return
        else:
            if not self.config.sequence_enabled:raise ValueError('parallel assembler requires configured sequence store')
            yield from cursor_rows(store,cursor,self.config,batch_size=batch_size,length=length,method=method)

    def tensor_batches(self,store,cursor,**kwargs):
        if not self.config.sequence_enabled:
            raise ValueError('CPU tensor workers require configured sequence store')
        return cursor_prepared_batches(store,cursor,self.config,**kwargs)

    def sid(self,original,ids,features,*,run_id,**science):
        if not self.config.bhkm_external and (self.config.memory_limit_bytes or self.config.disk_limit_bytes):
            if not hasattr(features,'nbytes'):raise ValueError('budgeted dense BHKM requires shape-known arrays')
            # Conservative planning model for the unchanged dense algorithm,
            # not a measured peak; reference per-dimension/sort arithmetic stays.
            branch=science.get('branching',16)
            if type(branch)is int and branch>=2:
                projected=64*len(ids)*min(branch,len(ids))+4*features.nbytes
                guard(self.config,Path.cwd(),projected_memory=projected)
            original_boundary=science.get('on_boundary')
            def boundary():
                guard(self.config,Path.cwd())
                if original_boundary is not None:original_boundary()
            science={**science,'on_boundary':boundary}
        return fit_configured(original,ids,features,self.config,run_id=run_id,**science)

    def weeks(self,partitions,parse_week,original_builder,**kwargs):
        return derive_weeks(partitions,parse_week,original_builder,self.config,**kwargs)

    def evaluate(self,queries,items,ids,k,**kwargs):
        return exact_topk(queries,items,ids,k,self.config,**kwargs)

    def decode(self,queries,original_decode,**kwargs):
        return decode_queries(queries,original_decode,self.config,**kwargs)

    def probe(self,directory):return resource_manifest(directory,self.config)
