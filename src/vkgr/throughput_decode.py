"""Query processes only; each invokes the original batch=1 decoder unchanged."""
from dataclasses import dataclass
import hashlib
import inspect
import math
import numbers
import re
import sys
from pathlib import Path

from .throughput_config import guard, rss_bytes
from .throughput_workers import ordered_map

_SESSION = None
_OWNER_TYPE = None


def _source_bytes(value):
    """Return fail-closed source identity for a function/class and its module."""
    module = sys.modules.get(getattr(value, '__module__', ''))
    path = getattr(module, '__file__', None)
    if not path:
        raise ValueError('source module file required for decoder identity')
    try:
        module_source = Path(path).read_bytes()
        object_source = inspect.getsource(value).encode()
    except (OSError, TypeError, IndentationError) as exc:
        raise ValueError('complete decoder/codec source unavailable') from exc
    return (getattr(value, '__module__', '').encode() + b'\0' +
            getattr(value, '__qualname__', repr(value)).encode() + b'\0' +
            module_source + b'\0' + object_source)


def _array_digest(hasher, name, array):
    import numpy as np
    if not isinstance(array, np.ndarray) or array.flags.writeable:
        raise ValueError('immutable numpy arrays required in TorchSession codec')
    hasher.update(name.encode()); hasher.update(str((array.dtype.str, array.shape, array.strides)).encode())
    hasher.update(array.tobytes(order='A'))


def _hash_value(hasher, name, value, seen=None):
    """Hash complete codec/callback state with deterministic ordering."""
    if seen is None: seen = set()
    if value is None or type(value) in (bool, int, float, str, bytes):
        hasher.update(name.encode() + b'=' + type(value).__name__.encode() + b':' + repr(value).encode()); return
    try:
        import torch
        if isinstance(value, torch.device):
            hasher.update((name + '=torch.device:' + str(value)).encode()); return
    except ImportError:
        pass
    if _is_validated_codecd(value):
        _hash_validated_codecd(hasher, name, value, seen)
        return
    ident = id(value)
    if ident in seen: raise ValueError('cyclic decoder or codec namespace is unsupported')
    seen.add(ident)
    try:
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                _array_digest(hasher, name, value); return
            if isinstance(value, np.generic):
                _hash_value(hasher, name, value.item(), seen); return
        except ImportError:
            pass
        if isinstance(value, (tuple, list)):
            hasher.update((name + ':' + type(value).__name__ + ':' + str(len(value))).encode())
            for index, item in enumerate(value): _hash_value(hasher, name + '[' + str(index) + ']', item, seen)
            return
        if isinstance(value, (set, frozenset)):
            hasher.update((name + ':' + type(value).__name__ + ':' + str(len(value))).encode())
            for index, item in enumerate(sorted(value, key=repr)): _hash_value(hasher, name + '[' + str(index) + ']', item, seen)
            return
        if isinstance(value, dict):
            hasher.update((name + ':dict:' + str(len(value))).encode())
            for key in sorted(value, key=lambda item: repr(item)):
                _hash_value(hasher, name + '.key', key, seen); _hash_value(hasher, name + '[' + repr(key) + ']', value[key], seen)
            return
        if inspect.isfunction(value) or inspect.ismethod(value):
            hasher.update(_source_bytes(value))
            _hash_value(hasher, name + '.defaults', getattr(value, '__defaults__', None), seen)
            _hash_value(hasher, name + '.kwdefaults', getattr(value, '__kwdefaults__', None), seen)
            for index, cell in enumerate(getattr(value, '__closure__', None) or ()):
                _hash_value(hasher, name + '.closure[' + str(index) + ']', cell.cell_contents, seen)
            if inspect.ismethod(value) and value.__self__ is not None:
                _hash_value(hasher, name + '.bound_owner', value.__self__, seen)
            return
        namespace = getattr(value, '__dict__', None)
        slots = getattr(type(value), '__slots__', ())
        if isinstance(slots, str): slots = (slots,)
        if namespace is not None or slots:
            cls = type(value)
            hasher.update((name + ':' + cls.__module__ + '.' + cls.__qualname__).encode())
            if cls.__module__ not in ('types', 'builtins'):
                hasher.update(_source_bytes(cls))
            fields = dict(namespace or {})
            for slot in slots:
                if slot not in ('__dict__', '__weakref__') and hasattr(value, slot): fields.setdefault(slot, getattr(value, slot))
            for key in sorted(fields): _hash_value(hasher, name + '.' + key, fields[key], seen)
            return
        raise ValueError('unsupported decoder/codec namespace value: ' + name)
    finally:
        seen.remove(ident)


def _is_validated_codecd(value):
    """Recognize only the exact depth codec and validated base implementation."""
    if type(value).__module__ != 'vkgr.sid_depth' or type(value).__qualname__ != 'ValidatedCodecD':
        return False
    try:
        from .sid_depth import ValidatedCodecD
        from .generative_data import _ValidatedCodec
    except ImportError:
        return False
    return type(value) is ValidatedCodecD and _ValidatedCodec in ValidatedCodecD.__mro__


def _hash_validated_codecd(hasher, name, codec, seen):
    """Normalize only authenticated process-local guard caches from ValidatedCodecD."""
    from .generative_data import _ValidatedCodec
    arrays = ('ids', 'sids', 'keys', 'key_items')
    if tuple(getattr(codec, '_arrays', ())) != arrays:
        raise ValueError('unexpected ValidatedCodecD array contract')
    if codec.catalog is not codec._model or not callable(getattr(codec, 'check_identity', None)):
        raise ValueError('ValidatedCodecD owner/reference contract mismatch')
    refs = codec._references; metadata = codec._array_metadata
    if type(refs) is not list or len(refs) != len(arrays) or type(metadata) is not list or len(metadata) != len(arrays):
        raise ValueError('ValidatedCodecD identity cache shape mismatch')
    for field, ref, record in zip(arrays, refs, metadata):
        value = getattr(codec.catalog, field)
        semantic = (value.shape, value.dtype.str, value.nbytes, value.strides)
        if ref is not value or value.flags.writeable or type(record) is not tuple or len(record) != 5 or record[:4] != semantic or record[4] != value.ctypes.data:
            raise ValueError('ValidatedCodecD reference or local pointer guard mismatch')
    codec.check_identity()
    namespace_bytes = codec._namespace_bytes()
    if namespace_bytes != codec._namespace:
        raise ValueError('ValidatedCodecD semantic namespace cache mismatch')
    # _model, _references and _array_metadata are verified process-local aliases
    # and guards. Hash the model once, semantic shape metadata, and canonical
    # namespace bytes; never hash ctypes.data or repeat arrays as alias objects.
    hasher.update(_source_bytes(type(codec)))
    hasher.update(_source_bytes(_ValidatedCodec))
    fields = dict(getattr(codec, '__dict__', {}))
    for cached in ('_model', '_references', '_array_metadata', '_namespace'):
        fields.pop(cached, None)
    for key in sorted(fields): _hash_value(hasher, name + '.' + key, fields[key], seen)
    for field, record in zip(arrays, metadata):
        _hash_value(hasher, name + '.array_metadata.' + field, record[:4], seen)
    _hash_value(hasher, name + '.namespace_bytes', namespace_bytes, seen)


def _array_storage_bytes(value, active=None, arrays=None):
    """Count each immutable NumPy array once, independent of alias count."""
    if active is None: active = set()
    if arrays is None: arrays = {}
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            arrays[id(value)] = value.nbytes
            return sum(arrays.values())
        if isinstance(value, np.generic) or value is None or type(value) in (bool, int, float, str, bytes):
            return sum(arrays.values())
    except ImportError:
        pass
    ident = id(value)
    if ident in active: return sum(arrays.values())
    active.add(ident)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                _array_storage_bytes(key, active, arrays); _array_storage_bytes(item, active, arrays)
        elif isinstance(value, (tuple, list)):
            for item in value: _array_storage_bytes(item, active, arrays)
        else:
            namespace = getattr(value, '__dict__', None)
            if namespace is not None:
                for item in namespace.values(): _array_storage_bytes(item, active, arrays)
            slots = getattr(type(value), '__slots__', ())
            if isinstance(slots, str): slots = (slots,)
            for slot in slots:
                if slot not in ('__dict__', '__weakref__') and hasattr(value, slot):
                    _array_storage_bytes(getattr(value, slot), active, arrays)
        return sum(arrays.values())
    finally:
        active.remove(ident)


def _hash_torch_tensor(hasher, name, tensor, aliases):
    """Hash tensor contents and placement while retaining alias topology."""
    previous = aliases.get(id(tensor))
    if previous is not None:
        hasher.update((name + ':tensor-alias:' + previous).encode())
        return 0
    aliases[id(tensor)] = name
    import torch
    metadata = (str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()),
                str(tensor.device), str(tensor.layout), bool(tensor.requires_grad))
    hasher.update((name + ':tensor:' + repr(metadata)).encode())
    size = tensor.numel() * tensor.element_size()
    if getattr(tensor, '_backward_hooks', None) is not None:
        _hash_value(hasher, name + '.backward_hooks', tensor._backward_hooks)
    if tensor.device.type == 'meta':
        hasher.update((name + ':meta-no-storage').encode())
        return size
    if tensor.layout != torch.strided:
        raise ValueError('unsupported non-strided model tensor: ' + name)
    try:
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    except (RuntimeError, TypeError) as exc:
        raise ValueError('unsupported model tensor representation: ' + name) from exc
    hasher.update(raw)
    return size


def _hash_torch_module(hasher, name, module, modules, tensors):
    """Hash the full torch module tree, runtime attributes, and hook registry."""
    previous = modules.get(id(module))
    if previous is not None:
        hasher.update((name + ':module-alias:' + previous).encode())
        return 0
    modules[id(module)] = name
    hasher.update((name + ':module:' + module.__class__.__module__ + '.' + module.__class__.__qualname__).encode())
    hasher.update(_source_bytes(type(module)))
    hasher.update((name + ':training:' + repr(bool(module.training))).encode())
    # Custom class data may affect forward even when absent from __dict__.
    # Source hashes alone do not detect runtime edits to class constants.
    import torch
    for cls in type(module).__mro__:
        if cls is torch.nn.Module:break
        for key, value in sorted(vars(cls).items()):
            if key in ('__dict__','__weakref__','__module__','__doc__','__qualname__','__annotations__','__slots__'):continue
            label=name+'.class.'+cls.__qualname__+'.'+key
            def class_callable(function):
                hasher.update(_source_bytes(function))
                _hash_value(hasher,label+'.defaults',getattr(function,'__defaults__',None))
                _hash_value(hasher,label+'.kwdefaults',getattr(function,'__kwdefaults__',None))
                for cell in getattr(function,'__closure__',None)or():
                    closed=cell.cell_contents
                    if inspect.ismodule(closed):
                        if closed.__name__ not in ('torch','numpy') or sys.modules.get(closed.__name__)is not closed:
                            raise ValueError('unsupported model class module closure')
                        hasher.update((closed.__name__+str(closed.__version__)).encode())
                        hasher.update(Path(closed.__file__).read_bytes())
                    elif isinstance(closed,type) and closed in type(module).__mro__:
                        hasher.update(_source_bytes(closed))
                    else:_hash_value(hasher,label+'.closure',closed)
            if isinstance(value, property):
                for getter in (value.fget,value.fset,value.fdel):
                    if getter is not None:class_callable(getter)
            elif isinstance(value,(staticmethod,classmethod)):
                class_callable(value.__func__)
            elif inspect.isfunction(value):
                class_callable(value)
            else:
                _hash_value(hasher,label,value)
    size = 0
    for collection_name in ('_parameters', '_buffers'):
        values = getattr(module, collection_name)
        hasher.update((name + ':' + collection_name).encode())
        for key in sorted(values):
            value = values[key]
            if value is None:
                hasher.update((name + ':' + collection_name + '.' + key + ':none').encode())
            else:
                import torch
                if not isinstance(value, torch.Tensor):
                    raise ValueError('unsupported model state entry: ' + name + '.' + key)
                size += _hash_torch_tensor(hasher, name + '.' + collection_name + '.' + key, value, tensors)
                if collection_name == '_parameters':
                    hasher.update((name + '.' + key + ':requires_grad:' + repr(bool(value.requires_grad))).encode())
                    if value.grad is not None:
                        size += _hash_torch_tensor(hasher, name + '.' + key + '.grad', value.grad, tensors)
    for hook_name in sorted(key for key in module.__dict__ if 'hook' in key.lower()):
        _hash_value(hasher, name + '.hook_registry.' + hook_name, module.__dict__[hook_name])
    _hash_value(hasher, name + '.non_persistent_buffers', tuple(sorted(module._non_persistent_buffers_set)))
    core = {
        '_parameters', '_buffers', '_modules', '_non_persistent_buffers_set', 'training',
        '_backward_hooks', '_backward_pre_hooks', '_forward_hooks', '_forward_hooks_with_kwargs',
        '_forward_hooks_always_called', '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs',
        '_state_dict_hooks', '_state_dict_pre_hooks', '_load_state_dict_pre_hooks',
        '_load_state_dict_post_hooks', '_is_full_backward_hook',
    }
    fields = dict(module.__dict__)
    for cls in type(module).__mro__:
        slots = getattr(cls, '__slots__', ())
        if isinstance(slots, str): slots = (slots,)
        for slot in slots:
            if slot not in ('__dict__', '__weakref__') and hasattr(module, slot):
                fields.setdefault(slot, getattr(module, slot))
    for key in sorted(fields):
        if key not in core and 'hook' not in key.lower():
            _hash_value(hasher, name + '.attribute.' + key, fields[key])
    for key in sorted(module._modules):
        child = module._modules[key]
        hasher.update((name + '.child:' + key).encode())
        if child is None:
            hasher.update((name + '.child:' + key + ':none').encode())
        else:
            size += _hash_torch_module(hasher, name + '.child.' + key, child, modules, tensors)
    return size


class TorchSession:
    """Read-only model/codec identity used by both the caller and worker factory.

    Codec identity includes its class/module namespace and catalog bytes. Source
    that cannot be inspected is rejected instead of silently receiving a weak
    digest. ``storage_bytes`` is a validated lower bound for initializer planning,
    not a measured peak-RSS claim.
    """
    def __init__(self, model, codec, decode, *, beam):
        if not inspect.ismethod(decode) and not inspect.isfunction(decode):
            raise ValueError('inspectable decoder callback required')
        if not isinstance(beam, int) or isinstance(beam, bool) or beam < 1:
            raise ValueError('positive beam required')
        self.model, self.codec, self.callback, self.beam = model, codec, decode, beam
        model.eval()
        self._sha = self._identity()

    def _identity(self):
        h = hashlib.sha256()
        h.update(_source_bytes(self.callback))
        for label, values in (('defaults', getattr(self.callback, '__defaults__', None)),
                              ('kwdefaults', getattr(self.callback, '__kwdefaults__', None))):
            _hash_value(h, label, values)
        closure = getattr(self.callback, '__closure__', None) or ()
        for index, cell in enumerate(closure): _hash_value(h, 'closure[' + str(index) + ']', cell.cell_contents)
        h.update(_source_bytes(type(self.codec)))
        h.update(_source_bytes(type(self.model)))
        import torch
        try:
            import numpy as np
            numpy_version = np.__version__
        except ImportError:
            numpy_version = 'unavailable'
        runtime = (
            sys.version, numpy_version, torch.__version__, torch.version.cuda,
            torch.cuda.is_available(), torch.get_num_threads(), torch.get_num_interop_threads(),
            torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        )
        h.update(repr(runtime).encode()); h.update(repr(self.beam).encode())
        model_size = _hash_torch_module(h, 'model', self.model, {}, {})
        catalog = getattr(self.codec, 'catalog', None)
        if catalog is None or not hasattr(catalog, 'ids') or not hasattr(catalog, 'sids'):
            raise ValueError('codec catalog with ids and sids required')
        for name in ('ids', 'sids'):
            array = getattr(catalog, name)
            _array_digest(h, 'catalog.' + name, array)
        # Include the complete codec instance namespace, not only catalog keys.
        _hash_value(h, 'codec.namespace', self.codec)
        return h.hexdigest(), model_size + _array_storage_bytes(self.codec)

    def identity(self):
        return self._identity()[0]

    def storage_bytes(self):
        return self._identity()[1]

    def token(self):
        return (self.identity(), tuple((key, value._version) for key, value in self.model.state_dict().items()))

    def decode(self, query):
        return self.callback(self.model, self.codec, query['history'], beam=self.beam)


@dataclass(frozen=True)
class InitializerBudget:
    """Caller-supplied cap tied to an inspected TorchSession identity and size floor."""
    session_sha256: str
    estimated_init_bytes: int
    budget_bytes_per_worker: int

    @classmethod
    def for_session(cls, session, *, estimated_init_bytes, budget_bytes_per_worker):
        if not isinstance(session, TorchSession):
            raise ValueError('initializer estimate requires TorchSession evidence')
        if type(estimated_init_bytes) is not int or estimated_init_bytes < session.storage_bytes():
            raise ValueError('initializer estimate below validated model/codec storage floor')
        if type(budget_bytes_per_worker) is not int or budget_bytes_per_worker < estimated_init_bytes:
            raise ValueError('initializer budget below estimate')
        return cls(session.identity(), estimated_init_bytes, budget_bytes_per_worker)


@dataclass(frozen=True)
class VRAMReceipt:
    per_worker_peak_mib: float
    live_free_mib: float
    context_inclusive_measured: bool
    live_free_measured_this_call: bool


def _owner(original_decode, state_sha):
    owner = getattr(original_decode, '__self__', None)
    function = getattr(original_decode, '__func__', None)
    if owner is None or function is None or function is not getattr(type(owner), 'decode', None):
        raise ValueError('original_decode must be the owner-bound decode method')
    identity = getattr(owner, 'identity', None)
    token = getattr(owner, 'token', None)
    if not callable(identity) or not callable(token):
        raise ValueError('original decoder owner must expose identity() and token()')
    if identity() != state_sha:
        raise ValueError('state_sha does not match original decoder identity')
    owner_type = hashlib.sha256(_source_bytes(type(owner))).hexdigest()
    return owner, owner_type, token()


def _init(factory, state_sha, owner_type, init_budget, worker_memory):
    global _SESSION, _OWNER_TYPE
    # This estimate gate runs before factory invocation/model loading. It is a
    # planning cap; post-load RSS checks remain necessary because Python cannot
    # enforce a portable hard transient-RSS ceiling for a spawned process.
    if init_budget is not None:
        if init_budget.session_sha256 != state_sha or init_budget.estimated_init_bytes > init_budget.budget_bytes_per_worker:
            raise MemoryError('initializer budget identity/cap mismatch')
        if worker_memory and init_budget.budget_bytes_per_worker > worker_memory:
            raise MemoryError('initializer budget exceeds worker memory cap')
    _OWNER_TYPE = owner_type
    _SESSION = factory()
    if hashlib.sha256(_source_bytes(type(_SESSION))).hexdigest() != owner_type:
        raise ValueError('worker session owner type mismatch')
    if not callable(getattr(_SESSION, 'identity', None)) or _SESSION.identity() != state_sha:
        raise ValueError('worker model/codec/source identity mismatch')
    if init_budget is not None:
        if not isinstance(_SESSION, TorchSession) or _SESSION.storage_bytes() > init_budget.estimated_init_bytes:
            raise MemoryError('initializer exceeded validated model/codec storage floor')
    if worker_memory and rss_bytes() > worker_memory:
        raise MemoryError('worker RSS exceeded cap during initializer')


def _decode(query):
    before = _SESSION.token(); result = _SESSION.decode(query)
    if _SESSION.token() != before:
        raise ValueError('decoder changed frozen model/codec state')
    return result


def _valid_positive_real(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value)) and value > 0


def decode_queries(queries, original_decode, config, *, session_factory=None, state_sha=None,
                  device='cpu', vram_receipt=None,
                  result_bound=8192, initializer_budget=None):
    if config.decode_workers == 1:
        return [original_decode(q) for q in queries]
    if type(result_bound) is not int or result_bound < 1:
        raise ValueError('positive integer result bound required')
    if session_factory is None or state_sha is None:
        raise ValueError('pinned worker session factory required')
    if not isinstance(state_sha, str) or re.fullmatch('[0-9a-f]{64}', state_sha) is None:
        raise ValueError('SHA256 decoder state identity required')
    owner, owner_type, owner_token = _owner(original_decode, state_sha)
    if isinstance(owner, TorchSession):
        if not isinstance(initializer_budget, InitializerBudget):
            raise ValueError('validated TorchSession initializer budget required')
        if (initializer_budget.session_sha256 != state_sha or
                type(initializer_budget.estimated_init_bytes) is not int or
                initializer_budget.estimated_init_bytes < owner.storage_bytes() or
                type(initializer_budget.budget_bytes_per_worker) is not int or
                initializer_budget.budget_bytes_per_worker < initializer_budget.estimated_init_bytes):
            raise ValueError('initializer budget identity mismatch')
    elif initializer_budget is not None:
        raise ValueError('initializer budget evidence is only accepted for TorchSession')
    if device.startswith('cuda'):
        if (not isinstance(vram_receipt, VRAMReceipt) or
                not _valid_positive_real(vram_receipt.per_worker_peak_mib) or
                not _valid_positive_real(vram_receipt.live_free_mib) or
                vram_receipt.context_inclusive_measured is not True or
                vram_receipt.live_free_measured_this_call is not True):
            raise ValueError('positive context-inclusive measured peak and this-call live-free receipt required')
        if config.decode_workers * vram_receipt.per_worker_peak_mib > vram_receipt.live_free_mib * .8:
            raise MemoryError('parallel decoder VRAM budget exceeded')
    guard(config, Path.cwd())
    worker_memory = 0
    if config.memory_limit_bytes:
        worker_memory = (config.memory_limit_bytes - rss_bytes() - config.max_in_flight_bytes) // config.decode_workers
        if worker_memory <= 0:
            raise MemoryError('no worker memory available')
        if isinstance(owner, TorchSession) and initializer_budget.budget_bytes_per_worker > worker_memory:
            raise MemoryError('initializer budget exceeds available worker memory')
    output = []
    try:
        for value in ordered_map(_decode, queries, workers=config.decode_workers, prefetch=config.prefetch,
                initializer=_init, initargs=(session_factory, state_sha, owner_type, initializer_budget, worker_memory),
                max_in_flight_bytes=config.max_in_flight_bytes,
                result_size_hint=lambda _: result_bound, memory_limit_bytes=worker_memory):
            guard(config, Path.cwd()); output.append(value)
    finally:
        if owner.token() != owner_token or owner.identity() != state_sha:
            raise ValueError('original decoder owner identity/state changed during parallel decode')
    return output
