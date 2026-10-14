"""INFRA-02b localisation of a serial/batched decode mismatch (diagnostic only, never enables anything).

Same inputs/gates as g_decode_parity.py. Prints ONE JSON line with only counts, booleans, wall and
float32 logit-difference magnitudes; no quality metric, no item ids, no per-query rows; test never read.
Variants vs the frozen serial decode (all beam=100, all validation queries unless noted):
  serial_repeat      serial decode again (run-to-run determinism)
  batched_b1         batched twin with batch_size=1 (same encoder/decoder batch shapes as serial)
  serial_copy_memory serial beam, but memory rows materialised with index_select (batched layout) instead of expand
  batched_bN         batched twin with --batch-size
For each: differing queries, of which same-set-reordered / set-changed / audit-changed.
Logit probe (first --probe queries, step 0 and 1): max |delta| of float32 logits between the serial
layout and (a) encoder batch n, (b) decoder memory copy, (c) decoder rows of many queries at once.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _compare(ref, other):
    diff = reorder = setchg = audit = 0
    for (ri, ra), (oi, oa) in zip(ref, other):
        if (list(ri), dict(ra)) == (list(oi), dict(oa)):
            continue
        diff += 1
        if dict(ra) != dict(oa):
            audit += 1
        if set(ri) == set(oi) and len(ri) == len(oi):
            reorder += 1
        else:
            setchg += 1
    return {'differing_queries': diff, 'same_set_reordered': reorder, 'set_changed': setchg, 'audit_changed': audit, 'count_mismatch': len(ref) != len(other)}


def main(argv=None, root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    if str(root / 'src') not in sys.path:
        sys.path.insert(0, str(root / 'src'))
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'prepared-sha', 'history-manifest', 'history-sha', 'manifest', 'manifest-sha', 'scratch'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--checkpoint', choices=('best', 'last'), required=True)
    parser.add_argument('--probe', type=int, default=64)
    args = parser.parse_args(argv)
    try:
        import torch
        spec = __import__('importlib.util').util.spec_from_file_location('g_decode_parity', root / 'scripts' / 'g_decode_parity.py')
        parity = __import__('importlib.util').util.module_from_spec(spec); spec.loader.exec_module(parity)
        from vkgr import generative_b1_run as b1
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_batched as gb
        from vkgr.data_audit import canonical_bytes
        from vkgr.tiger import constrained_beam, decode_history
        gb._check_batch(args.batch_size)
        value = b1._read_run(Path(args.manifest), args.manifest_sha); config = value['config']; run = Path(args.manifest).parent
        if config['history_len'] > 50:
            raise ValueError('diagnostic covers L<=50 only')
        members = tuple(sorted(value['binding']['source']['files']))
        from contextlib import nullcontext
        runner = bb._runner(None) if members == tuple(sorted(bb.MEMBERS)) else nullcontext()
        with runner:
            context, engineering = b1._contexts(root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha, config, Path(args.scratch))
            b1.validate_config(config, engineering); runtime = b1._runtime(engineering)
            b1.gr._seed(config['seed']); fresh = b1.make_model_b1(context.codec, config, device=runtime['device'])
            binding = b1._binding(root, context, config, runtime, fresh, args.prepared_sha, args.history_sha)
        if canonical_bytes(value['binding']) != canonical_bytes(binding):
            raise ValueError('run input/source/runtime/init binding mismatch')
        model = b1._load_best(run, value, context, config, runtime, binding) if args.checkpoint == 'best' else parity._load_last(b1, run, value, context, config, runtime, binding)
        model.eval(); device = next(model.parameters()).device; codec = context.codec
        histories = [q['history'] for q in context.queries]

        def serial_copy(history):
            source = torch.tensor([codec.history(history)], dtype=torch.long, device=device)
            with torch.no_grad():
                memory, padding = model.encode(source)
                def score(prefixes):
                    rows = [[1] + [codec.offsets[j] + d for j, d in enumerate(p)] for p in prefixes]
                    index = torch.zeros(len(rows), dtype=torch.long, device=device)
                    return model.decode(memory.index_select(0, index), padding.index_select(0, index), torch.tensor(rows, dtype=torch.long, device=device))[:, -1]
                return constrained_beam(codec.catalog, codec, score, 100)

        out = {'checkpoint': args.checkpoint, 'queries': len(histories), 'batch_size': args.batch_size, 'variants': {}, 'wall_seconds': {}}
        begin = time.perf_counter(); ref = [decode_history(model, codec, h, 100) for h in histories]; out['wall_seconds']['serial'] = time.perf_counter() - begin
        variants = {'serial_repeat': lambda: [decode_history(model, codec, h, 100) for h in histories],
                    'batched_b1': lambda: gb.decode_histories_batched(model, codec, histories, 100, 1),
                    'serial_copy_memory': lambda: [serial_copy(h) for h in histories],
                    f'batched_b{args.batch_size}': lambda: gb.decode_histories_batched(model, codec, histories, 100, args.batch_size)}
        for name, fn in variants.items():
            begin = time.perf_counter(); got = fn(); out['wall_seconds'][name] = time.perf_counter() - begin
            out['variants'][name] = _compare(ref, got)

        # Logit probe on the first --probe queries of the most common source width.
        widths = {}
        for i, h in enumerate(histories):
            widths.setdefault(len(codec.history(h)), []).append(i)
        group = max(widths.values(), key=len)[:max(2, args.probe)]
        sources = torch.tensor([codec.history(histories[i]) for i in group], dtype=torch.long, device=device)
        probe = {'group_queries': len(group)}
        with torch.no_grad():
            single = [model.encode(sources[k:k + 1]) for k in range(len(group))]
            mem_n, pad_n = model.encode(sources)
            probe['encoder_batch_n_vs_1_max_abs'] = max(float((mem_n[k:k + 1] - single[k][0]).abs().max()) for k in range(len(group)))
            start = torch.tensor([[1]], dtype=torch.long, device=device)
            l_expand = [model.decode(m.expand(1, -1, -1), p.expand(1, -1), start)[:, -1] for m, p in single]
            zero = torch.zeros(1, dtype=torch.long, device=device)
            l_copy = [model.decode(m.index_select(0, zero), p.index_select(0, zero), start)[:, -1] for m, p in single]
            probe['decoder_memory_copy_vs_expand_step0_max_abs'] = max(float((a - b).abs().max()) for a, b in zip(l_expand, l_copy))
            l_many = model.decode(mem_n, pad_n, start.expand(len(group), 1))[:, -1]
            probe['decoder_rows_n_vs_1_step0_max_abs'] = max(float((l_many[k] - l_expand[k][0]).abs().max()) for k in range(len(group)))
            # step 1: 16 prefixes per query (all first digits), serial = one query's 16 rows, batched = all queries' rows
            first = torch.arange(codec.catalog.branching, device=device) + codec.offsets[0]
            dec = torch.stack([torch.ones_like(first), first], 1)
            per_query = [model.decode(m.expand(len(dec), -1, -1), p.expand(len(dec), -1), dec)[:, -1] for m, p in single]
            index = torch.arange(len(group), device=device).repeat_interleave(len(dec))
            many = model.decode(mem_n.index_select(0, index), pad_n.index_select(0, index), dec.repeat(len(group), 1))[:, -1].view(len(group), len(dec), -1)
            probe['decoder_rows_n_vs_1_step1_max_abs'] = max(float((many[k] - per_query[k]).abs().max()) for k in range(len(group)))
            probe['bitwise_equal'] = {k: v == 0. for k, v in probe.items() if k.endswith('_max_abs')}
        out['logit_probe'] = probe
        print(json.dumps(out, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'decode parity diag failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
