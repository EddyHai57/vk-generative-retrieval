"""GARCH-01 server preflight PF1-PF3 for the independent-head G architecture; prints ONE JSON line.

Run before the counted A1/A2 runs (a failed counted run does not give the slot back, so the checks come first).
  PF1  anchor-shaped new model vs the frozen model, real data: same-seed init, K frozen training steps (model digest after
       every step), eval logits with the stock fast path off (bit-equal expected) and on (tolerance), serial decode of
       the reference run's best checkpoint on every validation query (candidate lists + recall@50 old vs new).
  PF2  paper-scale model (seed 42, random weights): serial decode vs batched decode with batch size 1 on every validation
       query (random weights have near-tied logits: a stricter test than a trained model).
  PF3  paper-scale training steps with dropout 0.1 under the real runtime flags, per microbatch size: finite loss,
       seconds per step, peak VRAM, measured parameter count vs the closed form.
Only booleans, counts, wall times and magnitudes are printed (no per-query rows, no test split).
The only files written are the temporary input copies the runner's context loader puts under ``--scratch``.
Exit 0 = the comparisons ran (read the booleans); 1 = gate/input failure.
"""
import argparse
import contextlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def _timed(device, fn):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    value = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return value, {'wall_seconds': time.perf_counter() - begin, 'peak_vram_bytes': None if device.type == 'cpu' else torch.cuda.max_memory_allocated(device)}


@contextlib.contextmanager
def _fastpath_off():
    import torch
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


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


def _modules(root):
    if str(root / 'src') not in sys.path:
        sys.path.insert(0, str(root / 'src'))
    import torch
    from vkgr import generative_arch as ga
    from vkgr import generative_b1_batched as bb
    from vkgr import generative_b1_run as b1
    from vkgr import generative_batched as gb
    from vkgr import generative_data as gd
    from vkgr import generative_evaluation as ev
    from vkgr import generative_hp as hp
    from vkgr import generative_training as gt
    from vkgr import tiger
    from vkgr.data_audit import canonical_bytes
    return SimpleNamespace(torch=torch, ga=ga, bb=bb, b1=b1, gb=gb, gd=gd, ev=ev, hp=hp, gt=gt, tiger=tiger, canonical_bytes=canonical_bytes)


def _arch_config(m, shape, **overrides):
    config = {**m.ga.default_config(), **dict(zip(m.ga.SHAPE_KEYS, shape))}
    config.update(overrides)
    return config


def _recall50(m, context, lists):
    aggregate = m.ev.aggregate_items([items for items, _ in lists], context.queries, context.catalog, slice_definition=context.slice_definition)
    return float(aggregate['metrics']['all_eligible']['recall@50'])


def pf1(m, context, config, runtime, old_best, steps):
    torch, codec, device = m.torch, context.codec, runtime['device']
    shape = (config['d_model'], config['heads'], config['d_model'] // config['heads'], config['encoder_layers'], config['decoder_layers'], config['ff'], config['dropout'])
    arch = _arch_config(m, shape, learning_rate=config['learning_rate'], seed=config['seed'], init=config['init'])
    m.b1.gr._seed(config['seed'])
    old = m.b1.make_model_b1(codec, config, device=device)
    m.b1.gr._seed(config['seed'])
    new = m.ga.make_model_arch(codec, arch, device=device)
    init_equal = _same_model(m, old, new)
    opt_old, opt_new = m.gt.make_optimizer(old, m.hp.core_config(config)), m.ga.make_optimizer_arch(new, m.hp.core_config(config))
    first_diff = None
    for step in range(1, steps + 1):
        rows = context.examples.rows(_batch_indices(len(context.examples), step - 1))
        a = m.gt.train_effective_batch(old, opt_old, rows, codec, microbatch=config['microbatch'], device=device)
        b = m.gt.train_effective_batch(new, opt_new, rows, codec, microbatch=config['microbatch'], device=device)
        if first_diff is None and not (a.loss_mean == b.loss_mean and _same_model(m, old, new)):
            first_diff = step
    new_best = m.ga.make_model_arch(codec, arch, device=device)
    new_best.load_state_dict(old_best.state_dict())
    source, decoder, _ = (x.to(device) for x in m.gd.sid_targets(context.examples.rows(list(range(min(64, len(context.examples))))), codec))
    old_best.eval()
    new_best.eval()
    with torch.no_grad():
        on_old, on_new = old_best(source, decoder), new_best(source, decoder)
        with _fastpath_off():
            off_old, off_new = old_best(source, decoder), new_best(source, decoder)
    histories = [q['history'] for q in context.queries]
    old_lists = _lists([m.tiger.decode_history(old_best, codec, h, 100) for h in histories])
    new_lists = _lists([m.tiger.decode_history(new_best, codec, h, 100) for h in histories])
    return {'init_equal': init_equal, 'train_steps': steps, 'train_steps_equal': _equal_steps(first_diff), 'first_diff_step': first_diff,
            'max_abs_param_diff': _max_abs_diff(old.state_dict(), new.state_dict()),
            'eval_logits_max_abs_diff_fastpath_on': float((on_old - on_new).abs().max()), 'eval_logits_equal_fastpath_off': bool(torch.equal(off_old, off_new)),
            'decode_lists_equal': sum(1 for a, b in zip(old_lists, new_lists) if a == b), 'queries': len(histories),
            'r50_old': _recall50(m, context, old_lists), 'r50_new': _recall50(m, context, new_lists)}


def pf2(m, context, runtime):
    codec, device = context.codec, runtime['device']
    m.b1.gr._seed(42)
    model = m.ga.make_model_arch(codec, _arch_config(m, m.ga.PAPER_SCALE), device=device)
    histories = [q['history'] for q in context.queries]
    torch_device = next(model.parameters()).device
    serial, serial_obs = _timed(torch_device, lambda: [m.tiger.decode_history(model, codec, h, 100) for h in histories])
    batched, batched_obs = _timed(torch_device, lambda: m.gb.decode_histories_batched(model, codec, histories, 100, 1))
    peaks = [p for p in (serial_obs['peak_vram_bytes'], batched_obs['peak_vram_bytes']) if p is not None]
    return {'queries': len(histories), 'equal': len(serial) == len(batched) and _lists(serial) == _lists(batched),
            'serial_wall_seconds': serial_obs['wall_seconds'], 'batched_b1_wall_seconds': batched_obs['wall_seconds'], 'peak_vram_bytes': max(peaks) if peaks else None}


def pf3(m, context, runtime, steps, microbatches):
    torch, codec, device = m.torch, context.codec, runtime['device']
    per, report = {}, None
    for microbatch in microbatches:
        m.b1.gr._seed(42)
        model = m.ga.make_model_arch(codec, _arch_config(m, m.ga.PAPER_SCALE, microbatch=microbatch), device=device)
        report = model.init_report['arch']
        optimizer = m.ga.make_optimizer_arch(model, {'learning_rate': .0003, 'weight_decay': .0001})
        torch_device = next(model.parameters()).device
        if torch_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(torch_device)
        seconds, done, error = [], 0, None
        try:
            for step in range(steps):
                rows = context.examples.rows(_batch_indices(len(context.examples), step))
                begin = time.perf_counter()
                result = m.gt.train_effective_batch(model, optimizer, rows, codec, microbatch=microbatch, device=device)
                if torch_device.type == 'cuda':
                    torch.cuda.synchronize(torch_device)
                seconds.append(time.perf_counter() - begin)
                done += 1 if torch.isfinite(torch.tensor(result.loss_mean)) else 0
        except (ValueError, RuntimeError) as exc:
            error = f'{type(exc).__name__}: {str(exc)[:160]}'
        per[str(microbatch)] = {'steps': done, 'finite': error is None and done == steps, 'step_seconds_mean': sum(seconds) / len(seconds) if seconds else None,
                                'peak_vram_bytes': None if torch_device.type == 'cpu' else torch.cuda.max_memory_allocated(torch_device), 'error': error}
        del model, optimizer
        if torch_device.type == 'cuda':
            torch.cuda.empty_cache()
    keys = ('device', 'deterministic_algorithms', 'allow_tf32', 'cuda_version', 'gpu_name')
    return {'params_total': report['params_total'], 'params_expected': report['params_expected'], 'per_microbatch': per,
            'runtime': {**{k: runtime.get(k) for k in keys}, 'torch': torch.__version__}}


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise ValueError('positive integer required')
    return value


def main(argv=None, root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    parser = argparse.ArgumentParser(description='G-ARCH-01 preflight PF1-PF3 (booleans, counts, wall times and magnitudes only)')
    for name in ('prepared', 'prepared-sha', 'history-manifest', 'history-sha', 'manifest', 'manifest-sha', 'scratch'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--train-steps', required=True)
    parser.add_argument('--microbatches', required=True)
    args = parser.parse_args(argv)
    try:
        steps = _positive_int(args.train_steps)
        microbatches = [_positive_int(x) for x in args.microbatches.split(',')]
        m = _modules(root)
        manifest_path = Path(args.manifest)
        run = manifest_path.parent
        value = m.b1._read_run(manifest_path, args.manifest_sha)
        config = value['config']
        members = tuple(sorted(value['binding']['source']['files']))
        if members == tuple(sorted(m.bb.MEMBERS)):
            runner = m.bb._runner(None)
        elif members == tuple(sorted(m.b1.B1_MEMBERS)):
            runner = nullcontext()
        else:
            raise ValueError('reference run source member set is neither frozen G-B1 entry point')
        with runner:
            context, engineering = m.b1._contexts(root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha, config, Path(args.scratch))
            m.b1.validate_config(config, engineering)
            runtime = m.b1._runtime(engineering)
            m.b1.gr._seed(config['seed'])
            fresh = m.b1.make_model_b1(context.codec, config, device=runtime['device'])
            binding = m.b1._binding(root, context, config, runtime, fresh, args.prepared_sha, args.history_sha)
        if m.canonical_bytes(value['binding']) != m.canonical_bytes(binding):
            raise ValueError('reference run input/source/runtime/init binding mismatch')
        old_best = m.b1._load_best(run, value, context, config, runtime, binding)
        if m.ev.model_digest(old_best) != value['epochs'][value['best']['epoch'] - 1]['model_sha256']:
            raise ValueError('best model digest mismatch')
        report = {'pf1_anchor_shape': pf1(m, context, config, runtime, old_best, steps), 'pf2_paper_scale_decode': pf2(m, context, runtime),
                  'pf3_paper_scale_train': pf3(m, context, runtime, steps, microbatches)}
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'arch parity gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
