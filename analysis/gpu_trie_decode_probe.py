"""Synthetic-only exact CSR probe. CPU evidence never substitutes CUDA timing."""
import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import struct
import sys
import time

SIZES = (120000, 1200000, 12012895)
ROOT = Path(__file__).resolve().parents[1]


def dry():
    return {'status': 'dry', 'sizes': list(SIZES), 'branching': 229, 'steps': 4,
            'beam': 100, 'batch': 1, 'history': 50, 'seed': 42,
            'model': 'frozen A1 random initialization FP32', 'science_training': False,
            'science_data_read': False, 'synthetic_only': True,
            'cuda_parity': 'UNKNOWN until same-runtime CUDA probe and real subset receipt'}


def trace_digest(trace):
    payload = bytearray()
    for step in trace:
        payload.extend(struct.pack('<Q', len(step)))
        for path, value in step:
            payload.extend(struct.pack('<' + 'q' * len(path), *path))
            payload.extend(struct.pack('<d', value))
    return hashlib.sha256(payload).hexdigest()


def original_decode(model, codec, history, *, trace):
    """Invoke the frozen decoder; observe its selected states without editing it.

    Python's profile return event sees the original local active list directly;
    no independent reimplementation is used as the equivalence reference.
    """
    from vkgr.sid_depth_beam import constrained_beam_d, decode_history_d
    code = constrained_beam_d.__code__
    # At score callbacks the previous selected active list exists in the parent.
    # A line hook captures each completed assignment exactly once per position.
    seen = set()
    def observe_lines(frame, event, arg):
        if frame.f_code is code and event == 'line' and 'position' in frame.f_locals:
            position = frame.f_locals['position']
            active = frame.f_locals['active']
            if active and len(active[0][0]) == position + 1 and position not in seen:
                trace.append([(tuple(p), float(v)) for p, v in active])
                seen.add(position)
        return observe_lines
    previous = sys.gettrace()
    try:
        sys.settrace(observe_lines)
        return decode_history_d(model, codec, history, beam=100)
    finally:
        sys.settrace(previous)


def synchronized(device):
    import torch
    if device == 'cuda':
        torch.cuda.synchronize()


def run(device, sizes, queries):
    import numpy as np
    import torch
    from analysis.g_decode_timing import balanced_sids, references
    from vkgr.sid import SIDCatalog
    from vkgr.sid_depth import ValidatedCodecD, SIDTokensLD
    from vkgr.generative_arch import make_model_arch
    from vkgr.sid_depth_beam import decode_history_d
    from vkgr.gpu_trie_decode import CSRTrie, decode_history_csr
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pins = references()
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; CPU result cannot satisfy CUDA gate')
    results = []
    for n in sizes:
        c = SIDCatalog(np.arange(1, n + 1, dtype=np.uint32), balanced_sids(n, 229, 3), 229, 3, [])
        codec = SIDTokensLD(ValidatedCodecD(c, 3), 50)
        torch.manual_seed(42)
        model = make_model_arch(codec, pins['a1_config'], device=device)
        synchronized(device); started = time.perf_counter()
        trie = CSRTrie(c, device=device)
        synchronized(device); build = time.perf_counter() - started
        per_query = []
        for q in range(queries):
            history = [1 + (q + i) % n for i in range(50)]
            left_trace = []; right_trace = []
            left = original_decode(model, codec, history, trace=left_trace)
            right = decode_history_csr(model, codec, history, trie=trie, trace=right_trace)
            if len(left_trace) != 4 or trace_digest(left_trace) != trace_digest(right_trace) or left != right:
                raise ValueError('per-query beam paths/scores/order differs')
            # Separate uninstrumented timing from diagnostic tracing.
            times = {}
            profile = cProfile.Profile()
            for label, callback in [('original', lambda: decode_history_d(model, codec, history, 100)),
                                    ('csr', lambda: decode_history_csr(model, codec, history, trie=trie))]:
                synchronized(device); start = time.perf_counter()
                value = callback()
                synchronized(device); times[label] = time.perf_counter() - start
                if value != left:
                    raise ValueError('timed result differs')
            profile.enable(); decode_history_d(model, codec, history, 100); profile.disable()
            out = io.StringIO(); pstats.Stats(profile, stream=out).sort_stats('cumulative').print_stats(20)
            # Raw profiles remain private; public summary exposes no local paths.
            per_query.append({'ordinal': q, 'seconds': times, 'trace_sha256': trace_digest(left_trace),
                              'profile': out.getvalue(), 'parity': 'bitwise'})
        results.append({'n': n, 'csr_build_seconds': build, 'csr_storage_bytes': trie.storage_bytes,
                        'queries': per_query})
        print(json.dumps({'progress_n': n, 'device': device, 'queries_verified': len(per_query)}), file=sys.stderr, flush=True)
    return {**dry(), 'status': 'synthetic_parity_verified', 'device': device,
            'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                        'cuda': torch.version.cuda, 'threads': torch.get_num_threads()},
            'a1_config': pins['a1_config'], 'results': results,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'implementation_sha256': hashlib.sha256((ROOT / 'src/vkgr/gpu_trie_decode.py').read_bytes()).hexdigest(),
            'real_subset_parity': 'UNKNOWN', 'formal_timing': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('dry')
    probe = sub.add_parser('synthetic')
    probe.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    probe.add_argument('--sizes', nargs='+', type=int, choices=SIZES, default=list(SIZES))
    probe.add_argument('--queries', type=int, choices=range(1, 513), default=2)
    probe.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'dry':
        print(json.dumps(dry(), sort_keys=True)); return 0
    # Exclusive output creation before expensive work; no existing receipt overwritten.
    with args.output.open('x') as stream:
        value = run(args.device, args.sizes, args.queries)
        json.dump(value, stream, sort_keys=True, indent=2)
    print(json.dumps({'status': value['status'], 'device': args.device,
                      'sizes': args.sizes, 'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
