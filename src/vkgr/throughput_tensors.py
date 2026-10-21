"""Ordered CPU constructor work, preserving the original main-process step.

Factories supply an immutable namespace and a cheap validate_call contract for
all non-row constructor arguments. Workers never load a model or sample RNG.
Only CPU contiguous, gradient-free tensors cross IPC as NumPy bytes.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import re
from .tensor_cache import bind_step
from .throughput_workers import ordered_map, rng_state
from .throughput_config import guard, rss_bytes

_STORE = _BUILDER = None


def _init(descriptor, factory, namespace):
    import torch
    from .throughput_sequences import SequenceStore
    global _STORE, _BUILDER
    before = rng_state()
    _BUILDER = factory()
    if _BUILDER.identity() != namespace or not callable(getattr(_BUILDER, 'validate_call', None)):
        raise ValueError('immutable constructor namespace/validator required')
    _STORE = SequenceStore.load(descriptor, mmap=True) if isinstance(descriptor, str) else descriptor
    if rng_state() != before:
        raise ValueError('constructor initializer consumed RNG')


def _build(task):
    indices, length, microbatch = task
    rows = _STORE.rows(indices, length=length)
    chunks = []
    for start in range(0, len(rows), microbatch):
        part = rows[start:start + microbatch]
        tensors = tuple(_BUILDER.build(part))
        if not tensors or any(t.device.type != 'cpu' or t.requires_grad or not t.is_contiguous() for t in tensors):
            raise ValueError('only contiguous deterministic CPU tensors allowed')
        chunks.append((part, tuple(t.numpy() for t in tensors)))
    return rows, chunks


@dataclass
class PreparedBatch:
    rows: list
    chunks: list
    namespace: str
    validator: object

    def __post_init__(self):
        self._row_keys = tuple(json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False) for rows, _ in self.chunks)

    def bind(self, original, builder_name, *, namespace):
        if namespace != self.namespace or self.validator.identity() != self.namespace:
            raise ValueError('prepared constructor namespace mismatch')
        pending = iter(zip(self.chunks, self._row_keys))
        def constructor(rows, *args, **kwargs):
            import torch
            try:
                (expected, arrays), row_key = next(pending)
            except StopIteration:
                raise ValueError('prepared constructor consumed twice') from None
            if self.validator.identity() != self.namespace or json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False) != row_key or not self.validator.validate_call(rows, args, kwargs):
                raise ValueError('constructor row order or immutable arguments mismatch')
            return tuple(torch.from_numpy(a) for a in arrays)
        return bind_step(original, builder_name, constructor)


def prepared_batches(store, batches, config, *, factory, namespace, role, microbatch=512, length=50, tensor_bytes_bound=None):
    import torch
    if role not in ('g', 'p') or type(microbatch) is not int or microbatch < 1:
        raise ValueError('role and positive original microbatch required')
    if not isinstance(namespace, str) or re.fullmatch('[0-9a-f]{64}', namespace) is None:
        raise ValueError('pinned constructor namespace SHA required')
    if config.use_mmap and store.root is None:
        raise ValueError('mmap store must be persisted before workers start')
    if (config.max_in_flight_bytes or config.memory_limit_bytes) and tensor_bytes_bound is None:
        raise ValueError('bounded constructor tensor output contract required')
    before = rng_state()
    validator = factory()
    if validator.identity() != namespace or not callable(getattr(validator, 'validate_call', None)):
        raise ValueError('immutable constructor namespace/validator required')
    if rng_state() != before:
        raise ValueError('constructor factory consumed main-process RNG')
    copies = sum(getattr(store, k).nbytes for k in store.COLUMNS)
    guard(config, Path.cwd(), projected_memory=0 if config.use_mmap else copies * config.assembly_workers)
    worker_memory = 0
    in_flight = config.max_in_flight_bytes
    if config.memory_limit_bytes:
        available = config.memory_limit_bytes - rss_bytes()
        if not in_flight:in_flight = max(1, available // 2)
        worker_memory = (available - in_flight) // config.assembly_workers
        if worker_memory <= 0:
            raise MemoryError('no memory budget for CPU tensor workers')
    def bound(task):
        indices, width, _ = task
        extra = tensor_bytes_bound(len(indices), width) if tensor_bytes_bound else 0
        if type(extra) is not int or extra < 0 or (tensor_bytes_bound and not extra):
            raise ValueError('positive conservative tensor output bound required')
        return 4 * store.rows_bound(indices, width) + extra + 4096
    tasks = ((list(indices), length, microbatch if role == 'g' else 2**63-1) for indices in batches)
    for rows, chunks in ordered_map(_build, tasks, workers=config.assembly_workers, prefetch=config.prefetch,
            initializer=_init, initargs=(store.root if config.use_mmap else store, factory, namespace),
            max_in_flight_bytes=in_flight, result_size_hint=bound if tensor_bytes_bound else None,
            memory_limit_bytes=worker_memory):
        guard(config, Path.cwd())
        yield PreparedBatch(rows, chunks, namespace, validator)


def cursor_prepared_batches(store, cursor, config, *, batch_size, method='next', **kwargs):
    probe = copy.deepcopy(cursor)
    def indices():
        while True:
            value = getattr(probe, method)(batch_size)
            if not value:
                return
            yield value
            if getattr(probe, 'offset', getattr(probe, 'next_offset', None)) == getattr(probe, 'size', getattr(probe, 'example_count', None)):
                return
    expected = copy.deepcopy(cursor)
    for prepared in prepared_batches(store, indices(), config, **kwargs):
        planned = getattr(expected, method)(batch_size)
        actual = getattr(cursor, method)(batch_size)
        if planned != actual or [r['example_id'] for r in prepared.rows] != actual:
            raise ValueError('live cursor differs from prepared ordinal')
        yield prepared
