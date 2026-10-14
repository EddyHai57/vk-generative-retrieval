"""INFRA-02 server parity: serial vs batched beam=100 decode of one G-B1 best checkpoint.

Decodes every validation query of the run's context twice (frozen per-query decode,
then VKGR_DECODE=batched twin) and prints ONE JSON line:
  {"equal": bool, "queries": n, "batch_size": b,
   "serial": {"wall_seconds", "peak_vram_bytes"}, "batched": {...}}
``equal`` covers every query's candidate list (order, uniqueness, count) and audit.
No quality metric, no per-query rows and no artifacts are emitted; test split is never read.
Exit 0 = comparison ran (check ``equal``); 1 = gate/input failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _timed(device, fn):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter(); value = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - begin
    return value, {'wall_seconds': wall, 'peak_vram_bytes': None if device.type == 'cpu' else torch.cuda.max_memory_allocated(device)}


def main(argv=None, root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    if str(root / 'src') not in sys.path:
        sys.path.insert(0, str(root / 'src'))
    parser = argparse.ArgumentParser(description='serial vs batched G decode parity (boolean + wall + VRAM only)')
    for name in ('prepared', 'prepared-sha', 'history-manifest', 'history-sha', 'manifest', 'manifest-sha', 'scratch'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        from vkgr import generative_b1_run as b1
        from vkgr import generative_batched as gb
        from vkgr.data_audit import canonical_bytes
        from vkgr.generative_data_v2 import SIDTokensL
        from vkgr.generative_diagnostics import decode_history_l
        from vkgr.tiger import decode_history
        gb._check_batch(args.batch_size)
        manifest_path = Path(args.manifest); run = manifest_path.parent
        value = b1._read_run(manifest_path, args.manifest_sha); config = value['config']
        context, engineering = b1._contexts(root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha, config, Path(args.scratch))
        b1.validate_config(config, engineering); runtime = b1._runtime(engineering)
        b1.gr._seed(config['seed']); fresh = b1.make_model_b1(context.codec, config, device=runtime['device'])
        binding = b1._binding(root, context, config, runtime, fresh, args.prepared_sha, args.history_sha)
        if canonical_bytes(value['binding']) != canonical_bytes(binding):
            raise ValueError('run input/source/runtime/init binding mismatch')
        model = b1._load_best(run, value, context, config, runtime, binding)
        device = next(model.parameters()).device
        histories = [q['history'] for q in context.queries]
        if config['history_len'] > 50:
            tokens = SIDTokensL(context.codec, config['history_len'])
            serial_fn = lambda: [decode_history_l(model, tokens, h, 100) for h in histories]
        else:
            tokens = context.codec
            serial_fn = lambda: [decode_history(model, tokens, h, 100) for h in histories]
        serial, serial_obs = _timed(device, serial_fn)
        batched, batched_obs = _timed(device, lambda: gb.decode_histories_batched(model, tokens, histories, 100, args.batch_size))
        equal = [(list(i), dict(a)) for i, a in serial] == [(list(i), dict(a)) for i, a in batched]
        print(json.dumps({'equal': equal, 'queries': len(histories), 'batch_size': args.batch_size, 'serial': serial_obs, 'batched': batched_obs}, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'decode parity gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
