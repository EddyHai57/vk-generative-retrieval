"""G-HP-01 server preflight PH1-PH3 before the counted G1-G5 runs; prints ONE JSON line.

The reference run (``--manifest``) is the anchor A1, trained by the GARCH-01 entry (``vkgr.generative_b1_arch``).
  PH1  new factory vs GARCH factory at the reference (A1) values, real data: same-seed init; K frozen training steps with the
       RNG re-seeded before every step on both sides (dropout masks equal) -> model digest after every step; eval logits
       (torch.equal, both sides are the same hand-written class); the reference best.pt loaded into a model built by the
       new factory, serial decode of every validation query -> R@50 vs the reference manifest (in integer hits).
  PH2  for every config whose structure (ignoring dropout) differs from the reference (G1, G4): random seed-42 weights,
       serial decode vs batched decode with batch size 1 on every validation query. LR-only / dropout-only configs evaluate
       with the reference structure (dropout is off in eval), already covered by the GARCH-01 preflight.
  PH3  every config: K training steps with its own LR / structure / microbatch under the real runtime flags: finite loss,
       seconds per step, peak VRAM, measured parameter count vs the closed form.
Only booleans, counts, wall times and magnitudes are printed (no per-query rows, no test split).
The only files written are the temporary input copies the context loader puts under ``--scratch``.
Exit 0 = the comparisons ran (read the booleans); 1 = gate/input failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _modules(root):
    if str(root / 'src') not in sys.path:
        sys.path.insert(0, str(root / 'src'))
    import torch
    from vkgr import generative_arch as ga
    from vkgr import generative_b1_arch as gba
    from vkgr import generative_b1_run as b1
    from vkgr import generative_batched as gb
    from vkgr import generative_data as gd
    from vkgr import generative_evaluation as ev
    from vkgr import generative_hp_arch as hp
    from vkgr import generative_training as gt
    from vkgr import tiger
    from vkgr.data_audit import canonical_bytes
    return SimpleNamespace(torch=torch, ga=ga, gba=gba, b1=b1, gb=gb, gd=gd, ev=ev, hp=hp, gt=gt, tiger=tiger, canonical_bytes=canonical_bytes)


def _same_model(m, a, b):
    return m.ev.model_digest(a) == m.ev.model_digest(b)


def _equal_steps(first_diff):
    return first_diff is None


def _max_abs_diff(a, b):
    return max(float((a[k].detach().double() - b[k].detach().double()).abs().max()) for k in a)


def _batch_indices(count, step, size=512):
    return [(step * size + i) % count for i in range(min(size, count))]


def _lists(results):
    return [(list(items), dict(audit)) for items, audit in results]


def _recall50(m, context, lists):
    aggregate = m.ev.aggregate_items([items for items, _ in lists], context.queries, context.catalog, slice_definition=context.slice_definition)
    return float(aggregate['metrics']['all_eligible']['recall@50'])


def _device(model):
    return next(model.parameters()).device


def ph1(m, context, config, runtime, reference_best, reference_r50, steps):
    torch, codec, device = m.torch, context.codec, runtime['device']
    values = {**config, 'task_id': m.hp.TASK_ID}
    m.b1.gr._seed(config['seed'])
    old = m.ga.make_model_arch(codec, config, device=device)
    m.b1.gr._seed(config['seed'])
    new = m.hp.make_model_hp(codec, values, device=device)
    init_equal = _same_model(m, old, new)
    opt_old, opt_new = m.ga.make_optimizer_arch(old, config), m.hp.make_optimizer_hp(new, values)
    first_diff = None
    for step in range(1, steps + 1):
        rows = context.examples.rows(_batch_indices(len(context.examples), step - 1))
        m.b1.gr._seed(1000 + step)
        a = m.gt.train_effective_batch(old, opt_old, rows, codec, microbatch=config['microbatch'], device=device)
        m.b1.gr._seed(1000 + step)
        b = m.gt.train_effective_batch(new, opt_new, rows, codec, microbatch=config['microbatch'], device=device)
        if first_diff is None and not (a.loss_mean == b.loss_mean and _same_model(m, old, new)):
            first_diff = step
    source, decoder, _ = (x.to(device) for x in m.gd.sid_targets(context.examples.rows(list(range(min(64, len(context.examples))))), codec))
    old.eval()
    new.eval()
    with torch.no_grad():
        eval_equal = bool(torch.equal(old(source, decoder), new(source, decoder)))
    reloaded = m.hp.make_model_hp(codec, values, device=device)
    reloaded.load_state_dict(reference_best.state_dict())
    reloaded.eval()
    histories = [q['history'] for q in context.queries]
    r50 = _recall50(m, context, _lists([m.tiger.decode_history(reloaded, codec, h, 100) for h in histories]))
    queries = len(histories)
    return {'init_equal': init_equal, 'train_steps': steps, 'train_steps_equal': _equal_steps(first_diff), 'first_diff_step': first_diff,
            'max_abs_param_diff': _max_abs_diff(old.state_dict(), new.state_dict()), 'eval_logits_equal': eval_equal, 'queries': queries,
            'r50_reference': reference_r50, 'r50_reloaded': r50, 'reload_hits_delta': round(r50 * queries) - round(reference_r50 * queries)}


def _timed(device, fn):
    torch = sys.modules['torch']
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    value = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - begin, None if device.type == 'cpu' else torch.cuda.max_memory_allocated(device)


def ph2(m, context, runtime, config):
    codec = context.codec
    m.b1.gr._seed(42)
    model = m.hp.make_model_hp(codec, config, device=runtime['device'])
    model.eval()
    histories = [q['history'] for q in context.queries]
    device = _device(model)
    serial, serial_wall, serial_peak = _timed(device, lambda: [m.tiger.decode_history(model, codec, h, 100) for h in histories])
    batched, batched_wall, batched_peak = _timed(device, lambda: m.gb.decode_histories_batched(model, codec, histories, 100, 1))
    peaks = [p for p in (serial_peak, batched_peak) if p is not None]
    return {'queries': len(histories), 'equal': len(serial) == len(batched) and _lists(serial) == _lists(batched),
            'serial_wall_seconds': serial_wall, 'batched_b1_wall_seconds': batched_wall, 'peak_vram_bytes': max(peaks) if peaks else None}


def ph3(m, context, runtime, config, steps):
    torch, codec = m.torch, context.codec
    m.b1.gr._seed(config['seed'])
    model = m.hp.make_model_hp(codec, config, device=runtime['device'])
    report = model.init_report['arch']
    optimizer = m.hp.make_optimizer_hp(model, config)
    device = _device(model)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    seconds, done, error = [], 0, None
    try:
        for step in range(steps):
            rows = context.examples.rows(_batch_indices(len(context.examples), step))
            begin = time.perf_counter()
            result = m.gt.train_effective_batch(model, optimizer, rows, codec, microbatch=config['microbatch'], device=runtime['device'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            seconds.append(time.perf_counter() - begin)
            done += 1 if torch.isfinite(torch.tensor(result.loss_mean)) else 0
    except (ValueError, RuntimeError) as exc:
        error = f'{type(exc).__name__}: {str(exc)[:160]}'
    value = {'steps': done, 'finite': error is None and done == steps, 'error': error, 'step_seconds_mean': sum(seconds) / len(seconds) if seconds else None,
             'peak_vram_bytes': None if device.type == 'cpu' else torch.cuda.max_memory_allocated(device),
             'params_total': report['params_total'], 'params_expected': report['params_expected']}
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return value


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise ValueError('positive integer required')
    return value


def _structure(m, config):
    return tuple(config[k] for k in m.hp.SHAPE_KEYS if k != 'dropout')


def main(argv=None, root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    parser = argparse.ArgumentParser(description='G-HP-01 preflight PH1-PH3 (booleans, counts, wall times and magnitudes only)')
    for name in ('prepared', 'prepared-sha', 'history-manifest', 'history-sha', 'manifest', 'manifest-sha', 'scratch', 'train-steps', 'configs'):
        parser.add_argument(f'--{name}', required=True)
    args = parser.parse_args(argv)
    try:
        steps = _positive_int(args.train_steps)
        m = _modules(root)
        paths = [Path(p) if Path(p).is_absolute() else root / p for p in args.configs.split(',')]
        if not paths or len({p.stem for p in paths}) != len(paths):
            raise ValueError('distinct config files required')
        manifest_path = Path(args.manifest)
        value = m.b1._read_run(manifest_path, args.manifest_sha)
        config = value['config']
        if tuple(sorted(value['binding']['source']['files'])) != tuple(sorted(m.gba.MEMBERS)):
            raise ValueError('reference run was not trained by the GARCH-01 entry')
        with m.gba._runner(None):
            context, engineering = m.b1._contexts(root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha, config, Path(args.scratch))
            m.b1.validate_config(config, engineering)
            runtime = m.b1._runtime(engineering)
            m.b1.gr._seed(config['seed'])
            fresh = m.b1.make_model_b1(context.codec, config, device=runtime['device'])
            binding = m.b1._binding(root, context, config, runtime, fresh, args.prepared_sha, args.history_sha)
            if m.canonical_bytes(value['binding']) != m.canonical_bytes(binding):
                raise ValueError('reference run input/source/runtime/init binding mismatch')
            best = m.b1._load_best(manifest_path.parent, value, context, config, runtime, binding)
        if m.ev.model_digest(best) != value['epochs'][value['best']['epoch'] - 1]['model_sha256']:
            raise ValueError('best model digest mismatch')
        configs = {p.stem: m.hp.validate_config(m.gd._json(p), engineering) for p in paths}
        report = {'ph1_a1_factory': ph1(m, context, config, runtime, best, float(value['selection']['best_metric'][0]), steps),
                  'ph2_decode': {name: ph2(m, context, runtime, c) for name, c in configs.items() if _structure(m, c) != _structure(m, config)},
                  'ph3_train': {'per_config': {name: ph3(m, context, runtime, c, steps) for name, c in configs.items()},
                                'runtime': {**{k: runtime.get(k) for k in ('device', 'deterministic_algorithms', 'allow_tf32', 'cuda_version', 'gpu_name')},
                                            'torch': m.torch.__version__}}}
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G-HP-01 parity gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
