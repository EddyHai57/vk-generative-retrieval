"""New-instance-only engineering capture/compare. Do not run capture overnight.

Uses the original profile loaders for the pinned subset anchor, no evaluation,
no checkpoint. Run original and cached captures in separate fresh processes.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import struct
import sys
from types import FunctionType, SimpleNamespace, CellType


def freeze(value):
    """Byte identities, including signed zero; unknown state is never omitted."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'tensor': str(value.dtype), 'shape': list(value.shape), 'stride': list(value.stride()),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if isinstance(value, np.ndarray):
        return {'array': str(value.dtype), 'shape': list(value.shape),
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, torch.Generator):
        return {'torch_generator': freeze(value.get_state())}
    if isinstance(value, random.Random):
        return {'python_generator': freeze(value.getstate())}
    if isinstance(value, np.random.Generator):
        return {'numpy_generator': freeze(value.bit_generator.state)}
    if isinstance(value, np.generic):
        return freeze(value.item())
    if isinstance(value, dict):
        return {'dict': [[freeze(k), freeze(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [freeze(x) for x in value]}
    if type(value) is float:
        return {'float64_bytes': struct.pack('!d', value).hex()}
    if value is None or type(value) in (str, int, bool):
        return value
    if hasattr(value, '_asdict'):
        return freeze(value._asdict())
    raise TypeError('unregistered parity state type: ' + type(value).__name__)


def assert_equal(a, b):
    if a != b:
        raise ValueError('ALERT: exact state/byte identity mismatch')


def _rng(cursor, negative):
    import numpy as np
    import torch
    return freeze({'python': random.getstate(), 'numpy': np.random.get_state(),
                   'torch_cpu': torch.get_rng_state(), 'torch_cuda_all': torch.cuda.get_rng_state_all(),
                   'cursor': vars(cursor), 'negative': negative})


def _replace_cells(function, replacements):
    cells = tuple(CellType(replacements[name]) if name in replacements else cell
                  for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()))
    return FunctionType(function.__code__, function.__globals__, function.__name__, function.__defaults__, cells)


def observe_p_inputs(step, trace):
    """Private b1 proxy for evidence only; no replacement of scientific outputs."""
    b1 = step.__globals__['b1']
    proxy = SimpleNamespace(**vars(b1))
    def observe(callback, label):
        def call(*a, **kw):
            value = callback(*a, **kw)
            trace.setdefault(label, []).append(freeze(value))
            return value
        return call
    proxy._columns = observe(b1._columns, 'columns_labels_mask_negatives')
    proxy._log_q = observe(b1._log_q, 'logq')
    globals_copy = {**step.__globals__, 'b1': proxy}
    result = FunctionType(step.__code__, globals_copy, step.__name__, step.__defaults__, step.__closure__)
    result.__kwdefaults__ = step.__kwdefaults__
    return result


def warm_cache(cache, closure, branch, steps, constructor):
    """Replay constructors with an independent cursor, never model/negative RNG."""
    import copy
    probe = copy.deepcopy(closure['cursor'])
    for _ in range(steps):
        indices = probe.next_batch(512) if branch == 'g' else probe.next(closure['config']['batch_size'])
        if branch == 'g':
            batch = closure['context'].examples.rows(indices)
            micro = closure['config']['microbatch']
            for offset in range(0, len(batch), micro):
                cache.g(batch[offset:offset+micro], closure['tokens'], constructor)
        else:
            batch = [closure['rows'][j] for j in indices]
            cache.p(batch, closure['view']['idmap'], max(len(r['history']) for r in batch),
                    closure['config']['feature'], constructor)


def capture(request_path, output, mode, steps, cache_state='cold'):
    import torch
    if not torch.cuda.is_available():
        raise ValueError('registered new-instance CUDA runtime required')
    from analysis import profile_anchor_steps as profile
    # Load this new module independently: vkgr must come only from the original
    # source tree installed by the pinned loader, not this scaffold checkout.
    module_path = Path(__file__).resolve().parents[1] / 'src/vkgr/tensor_cache.py'
    spec = importlib.util.spec_from_file_location('engineering_tensor_cache', module_path)
    cache_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cache_mod)
    raw = json.loads(Path(request_path).read_text())
    def reject(value):
        if isinstance(value, dict):
            for v in value.values(): reject(v)
        elif isinstance(value, list):
            for v in value: reject(v)
        elif isinstance(value, str) and ('holdout' in value.lower() or 'week_26' in value.lower()):
            raise ValueError('holdout input forbidden')
    reject(raw)
    args = profile._load_request(request_path)
    _, _, setup, details = profile._load_g(args) if args.branch == 'g' else profile._load_p(args)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    __import__('numpy').random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, optimizer, step, builders = setup()
    closure = inspect.getclosurevars(step).nonlocals
    cursor = closure['cursor']; negative = closure.get('negative_rng')
    module = closure['runner'] if args.branch == 'g' else closure['feat']
    name = 'sid_targets_l' if args.branch == 'g' else '_tensors2'
    original_builder = getattr(builders[0][0], name)
    sha = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
    identity = cache_mod.namespace(input=sha(details), sid=details.get('sid_sha256', sha('P:no-SID')),
        config=args.config_sha, source=details['source_ledger_sha256'], runtime=sha(details['runtime']),
        constructor=hashlib.sha256(inspect.getsource(original_builder).encode()).hexdigest())
    cache = cache_mod.TensorCache(identity)
    if mode == 'cached' and cache_state == 'warm':
        # Deterministic constructors only, using a *copy* of the original
        # cursor. No model calls/negative draws and no original cursor advance.
        warm_cache(cache, closure, args.branch, steps, original_builder)
    trace = {}
    def builder(*a, **kw):
        value = (cache.g(*a, original_builder) if args.branch == 'g' else cache.p(*a, original_builder)) if mode == 'cached' else original_builder(*a, **kw)
        trace.setdefault('tensors', []).append(freeze(value))
        return value
    original_train = module.train_batch_d if args.branch == 'g' else module.train_step
    wrapped_train = cache_mod.bind_step(original_train, name, builder)
    if args.branch == 'p':
        wrapped_train = observe_p_inputs(wrapped_train, trace)
    def training(*a, **kw):
        result = wrapped_train(*a, **kw)
        trace['loss_result'] = freeze(vars(result) if hasattr(result, '__dict__') else result)
        return result
    proxy = SimpleNamespace(**vars(module))
    setattr(proxy, 'train_batch_d' if args.branch == 'g' else 'train_step', training)
    step = _replace_cells(step, {'runner' if args.branch == 'g' else 'feat': proxy})
    handles = []
    for name_p, param in model.named_parameters():
        def gradient(g, key=name_p):
            trace.setdefault('backward', {}).setdefault(key, []).append(freeze(g))
        handles.append(param.register_hook(gradient))
    def before_opt(opt, a, kw):
        trace['clipped_gradients'] = freeze({k: p.grad for k, p in model.named_parameters()})
        trace['optimizer_before'] = freeze(opt.state_dict())
    handles.append(optimizer.register_step_pre_hook(before_opt))
    try:
        with (target / 'trace.jsonl').open('x') as stream:
            initial = {'parameters': freeze(model.state_dict()), 'optimizer': freeze(optimizer.state_dict()), 'rng_cursor': _rng(cursor, negative)}
            stream.write(json.dumps({'initial': initial}, sort_keys=True) + '\n')
            for index in range(steps):
                trace.clear(); trace['before_rng_cursor'] = _rng(cursor, negative)
                step(); torch.cuda.synchronize()
                trace['parameters'] = freeze(model.state_dict())
                trace['optimizer_after'] = freeze(optimizer.state_dict())
                trace['after_rng_cursor'] = _rng(cursor, negative)
                if 'clipped_gradients' not in trace:
                    raise ValueError('ALERT: missing optimizer/gradient evidence')
                stream.write(json.dumps({'step': index, 'state': trace}, sort_keys=True) + '\n'); stream.flush()
    finally:
        for handle in handles: handle.remove()
    summary = {'schema': 'tensor-cache-capture-v1', 'mode': mode, 'cache_state': cache_state, 'steps': steps,
               'trace_sha256': hashlib.sha256((target/'trace.jsonl').read_bytes()).hexdigest(),
               'cache_hits': cache.hits, 'cache_misses': cache.misses, 'quality_metrics': False,
               'checkpoint': False, 'binding': details}
    with (target/'summary.json').open('x') as f: json.dump(summary, f, sort_keys=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('dry')
    cap = commands.add_parser('capture')
    cap.add_argument('--request', required=True); cap.add_argument('--output', required=True)
    cap.add_argument('--mode', choices=('original', 'cached'), required=True)
    cap.add_argument('--cache-state', choices=('cold', 'warm'), default='cold')
    cap.add_argument('--steps', type=int, default=8)
    comp = commands.add_parser('compare'); comp.add_argument('--original', required=True); comp.add_argument('--cached', required=True)
    args = p.parse_args(argv)
    if args.command == 'dry':
        result = {'status': 'ready', 'capture_executed': False, 'real_parity': 'NOT_RUN'}
    elif args.command == 'capture':
        if not 2 <= args.steps <= 500: p.error('bounded 2..500 engineering steps required')
        result = capture(args.request, args.output, args.mode, args.steps, args.cache_state)
    else:
        a, b = Path(args.original), Path(args.cached)
        sa, sb = (json.loads((x/'summary.json').read_text()) for x in (a, b))
        assert_equal(sa['steps'], sb['steps']); assert_equal(sa['binding'], sb['binding'])
        if sa['mode'] != 'original' or sb['mode'] != 'cached':
            raise ValueError('original vs cached required')
        if sb['cache_state'] == 'warm' and sb['cache_hits'] < 1:
            raise ValueError('warm capture must demonstrate a cache hit')
        assert_equal((a/'trace.jsonl').read_bytes(), (b/'trace.jsonl').read_bytes())
        result = {'status': 'byte_parity_verified', 'steps': sa['steps']}
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
