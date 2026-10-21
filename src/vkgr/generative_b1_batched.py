"""INFRA-02 opt-in entry point for the Batch1 G runner with batched beam decode.

``python -m vkgr.generative_b1_run`` stays byte-identical to batch1/g 6f3901d, so its
``binding.source`` (and every run trained by it) is unchanged by INFRA-02.
This module runs the same runner with two differences, scoped to the call:
  * ``binding.source`` additionally covers ``generative_batched.py`` and this file
    (runs trained here verify only here, and vice versa: fail-closed binding mismatch);
  * with ``VKGR_DECODE=batched`` evaluation uses the exact batched twin.
Without ``VKGR_DECODE`` it evaluates exactly like the default runner (serial / sharded).
"""
import sys
from contextlib import contextmanager

from . import generative_b1_run as b1
from . import generative_batched as gb

MEMBERS = b1.B1_MEMBERS + ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py')


def _evaluator(decode, base=None):
    """Drop-in for ``b1._evaluate`` honoring ``decode`` (None = the default runner's evaluator)."""
    gb.check_decode(decode); base = b1._evaluate if base is None else base

    def evaluate(model, context, tokens, config, runtime, output, checkpoint_sha, guard, workers, on_shard=None):
        if decode is None:
            return base(model, context, tokens, config, runtime, output, checkpoint_sha, guard, workers, on_shard=on_shard)
        if workers != 1:
            raise ValueError('decode env: batched decode excludes VKGR_EVAL_WORKERS>1')
        if config['history_len'] > 50:
            return 'l100_query_evaluation', gb.evaluate_l100_batched(model, context, tokens, output=output, checkpoint_sha256=checkpoint_sha,
                                                                     batch_size=decode['batch_size'], guard=guard)
        return 'query_evaluation', gb.evaluate_queries_batched(model, context, output=output, checkpoint_sha256=checkpoint_sha, shard_size=64,
                                                               batch_size=decode['batch_size'], budget_check=guard, on_shard=on_shard)
    return evaluate


@contextmanager
def _runner(decode):
    members, evaluate = b1.B1_MEMBERS, b1._evaluate
    b1.B1_MEMBERS = MEMBERS; b1._evaluate = _evaluator(decode, evaluate)
    try:
        yield b1
    finally:
        b1.B1_MEMBERS = members; b1._evaluate = evaluate


def run(*args, decode=None, **kwargs):
    with _runner(decode):
        return b1.run(*args, **kwargs)


def verify(*args, decode=None, **kwargs):
    with _runner(decode):
        return b1.verify(*args, **kwargs)


def main(argv=None):
    try:
        decode = gb.decode_mode_from_env()
        if decode is not None and b1.gs.eval_workers_from_env() != 1:
            raise ValueError('decode env: batched decode excludes VKGR_EVAL_WORKERS>1')
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G-B1 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 1
    with _runner(decode):
        return b1.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
