"""Parallel stateless week parsing, followed by the original serial state machine."""
from itertools import chain
from os import PathLike,fsdecode,fspath
from pathlib import Path
from .throughput_config import guard,rss_bytes
from .throughput_workers import ordered_map


def derive_weeks(weeks,parse_week,original_builder,config,*,result_size_hint=None):
    partitions=list(weeks)
    order=[w for w,_ in partitions]
    if any(type(w)is not int or not 0<=w<=25 for w in order)or any(a>=b for a,b in zip(order,order[1:])):
        raise ValueError('authorized train/validation week order required; no week26')
    def reject(value):
        if isinstance(value,(str,PathLike)):
            raw=fsdecode(fspath(value))
            parts=Path(raw.replace('\\','/')).parts
            if any(k in raw.lower()for k in ('holdout','week_26','week26'))or any(
                    part.lower()=='test'or Path(part).stem.lower()=='test'for part in parts):
                raise ValueError('holdout is not an authorized derivation input')
        if isinstance(value,dict):
            for v in value.values():reject(v)
        elif isinstance(value,(tuple,list)):
            for v in value:reject(v)
    reject(partitions)
    guard(config,Path.cwd())
    worker_memory=0
    if config.memory_limit_bytes:
        worker_memory=(config.memory_limit_bytes-rss_bytes()-config.max_in_flight_bytes)//config.week_workers
        if worker_memory<=0:raise MemoryError('no budget for week parser workers')
    parsed=ordered_map(parse_week,partitions,workers=config.week_workers,prefetch=config.prefetch,
        max_in_flight_bytes=config.max_in_flight_bytes,result_size_hint=result_size_hint,memory_limit_bytes=worker_memory)
    # No history is built independently per week. Existing cross-week prefix
    # state, target identity and original physical ordinals belong to builder.
    def checked():
        for block in parsed:
            guard(config,Path.cwd());yield block;guard(config,Path.cwd())
    result=original_builder(chain.from_iterable(checked()))
    guard(config,Path.cwd());return result
