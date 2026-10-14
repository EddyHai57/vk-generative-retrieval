"""INFRA-02 server parity: serial vs batched beam=100 decode of one G-B1 checkpoint (best or last).

Decodes every validation query of the run's context twice (frozen per-query decode,
then the batched twin) and prints ONE JSON line:
  {"equal": bool, "queries": n, "batch_size": b, "checkpoint": "best"|"last", "epoch": e,
   "serial": {"wall_seconds", "peak_vram_bytes"}, "batched": {...}}
``equal`` covers every query's candidate list (order, uniqueness, count) and audit.
No quality metric and no per-query rows are emitted; test split is never read.
The only files written are the temporary input copies that the runner's context
loader puts under ``--scratch`` (a new empty directory; remove it afterwards).
Runs trained by either entry point (generative_b1_run / generative_b1_batched) are
accepted: the source member set is taken from the run's own binding and must match.
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


def _load_last(b1, run, value, context, config, runtime, binding):
    import torch
    from vkgr.data_audit import file_sha256
    last = value['last']
    if type(last) is not dict or last.get('path') != 'last.pt' or last.get('epoch') != len(value['epochs']) or file_sha256(run / 'last.pt') != last.get('sha256') \
            or last['sha256'] != value['epochs'][-1]['checkpoint_sha256']:
        raise ValueError('last checkpoint hash mismatch')
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    if state.get('schema') != 'g-b1-checkpoint-v1' or state.get('binding_sha256') != b1._digest(binding) or state.get('epoch') != last['epoch']:
        raise ValueError('last checkpoint identity mismatch')
    b1.gr._seed(config['seed']); model = b1.make_model_b1(context.codec, config, device=runtime['device'])
    model.load_state_dict(state['model'])
    return model


def main(argv=None, root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    if str(root / 'src') not in sys.path:
        sys.path.insert(0, str(root / 'src'))
    parser = argparse.ArgumentParser(description='serial vs batched G decode parity (boolean + wall + VRAM only)')
    for name in ('prepared', 'prepared-sha', 'history-manifest', 'history-sha', 'manifest', 'manifest-sha', 'scratch'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--checkpoint', choices=('best', 'last'), required=True)
    args = parser.parse_args(argv)
    try:
        from vkgr import generative_b1_run as b1
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_batched as gb
        from vkgr import generative_evaluation as ev
        from vkgr.data_audit import canonical_bytes
        from vkgr.generative_data_v2 import SIDTokensL
        from vkgr.generative_diagnostics import decode_history_l
        from vkgr.tiger import decode_history
        gb._check_batch(args.batch_size)
        manifest_path = Path(args.manifest); run = manifest_path.parent
        value = b1._read_run(manifest_path, args.manifest_sha); config = value['config']
        members = tuple(sorted(value['binding']['source']['files']))
        if members == tuple(sorted(bb.MEMBERS)):
            runner = bb._runner(None)
        elif members == tuple(sorted(b1.B1_MEMBERS)):
            from contextlib import nullcontext
            runner = nullcontext()
        else:
            raise ValueError('run source member set is neither G-B1 entry point')
        with runner:
            context, engineering = b1._contexts(root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha, config, Path(args.scratch))
            b1.validate_config(config, engineering); runtime = b1._runtime(engineering)
            b1.gr._seed(config['seed']); fresh = b1.make_model_b1(context.codec, config, device=runtime['device'])
            binding = b1._binding(root, context, config, runtime, fresh, args.prepared_sha, args.history_sha)
        if canonical_bytes(value['binding']) != canonical_bytes(binding):
            raise ValueError('run input/source/runtime/init binding mismatch')
        if args.checkpoint == 'best':
            model = b1._load_best(run, value, context, config, runtime, binding); epoch = value['best']['epoch']
        else:
            model = _load_last(b1, run, value, context, config, runtime, binding); epoch = value['last']['epoch']
        if ev.model_digest(model) != value['epochs'][epoch - 1]['model_sha256']:
            raise ValueError(f'{args.checkpoint} model digest mismatch')
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
        equal = len(serial) == len(batched) and [(list(i), dict(a)) for i, a in serial] == [(list(i), dict(a)) for i, a in batched]
        print(json.dumps({'equal': equal, 'queries': len(histories), 'batch_size': args.batch_size, 'checkpoint': args.checkpoint, 'epoch': epoch,
                          'serial': serial_obs, 'batched': batched_obs}, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'decode parity gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
