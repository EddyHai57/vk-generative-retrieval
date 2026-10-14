"""GARCH-01 opt-in entry point: the Batch1 G runner with the paper-scale independent-head architecture.

``python -m vkgr.generative_b1_run`` and ``python -m vkgr.generative_b1_batched`` stay untouched (their ``binding.source`` and
every run trained by them are unchanged). This module composes with the INFRA-02 batched entry and, scoped to each call,
swaps four names of ``generative_b1_run``: ``validate_config`` (arch registry), ``make_model_b1`` (arch factory),
``B1_MEMBERS`` (+ the three arch files, so arch runs bind their own source) and ``gt`` (frozen training module, except the
optimizer factory whose core check rejects any structure other than 1+1 layers / 4 heads). Everything is restored on exit.

Only ``train`` and ``verify`` are registered. Fail-closed before any output: ``VKGR_EVAL_WORKERS`` > 1 (sharded workers rebuild
the model with the frozen factory), batched decode with batch size != 1 (batch > 1 is not bit-equal on GPU, INFRA-02b),
``beam-diagnostic``. ``history_len`` 100 / random SID / other structures are rejected by ``generative_arch.validate_config``.
"""
import sys
from contextlib import contextmanager

from . import generative_arch as ga
from . import generative_b1_batched as bb
from . import generative_b1_run as b1
from . import generative_batched as gb

ARCH_MEMBERS = ('src/vkgr/tiger_arch.py', 'src/vkgr/generative_arch.py', 'src/vkgr/generative_b1_arch.py')
MEMBERS = bb.MEMBERS + ARCH_MEMBERS
COMMANDS = ('train', 'verify')


class _FrozenTraining:
    """Delegates to the frozen ``generative_training`` except ``make_optimizer`` (its ``_core`` only knows the anchor structure)."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def make_optimizer(model, config):
        return ga.make_optimizer_arch(model, config)


def _gate(decode, workers):
    gb.check_decode(decode)
    if workers != 1:
        raise ValueError('G-ARCH-01: VKGR_EVAL_WORKERS must be 1 (sharded workers rebuild the model with the frozen factory)')
    if decode is not None and decode['batch_size'] != 1:
        raise ValueError('G-ARCH-01: batched decode only with batch size 1 (batch > 1 is not bit-equal on GPU)')


def _swap(saved, name, value):
    saved.append((name, getattr(b1, name)))
    setattr(b1, name, value)


def _restore(saved):
    for name, value in reversed(saved):
        setattr(b1, name, value)


@contextmanager
def _runner(decode, workers=1):
    _gate(decode, workers)
    with bb._runner(decode):
        saved = []
        try:
            _swap(saved, 'B1_MEMBERS', MEMBERS)
            _swap(saved, 'validate_config', ga.validate_config)
            _swap(saved, 'make_model_b1', ga.make_model_arch)
            _swap(saved, 'gt', _FrozenTraining(b1.gt))
            yield b1
        finally:
            _restore(saved)


def run(*args, decode=None, **kwargs):
    with _runner(decode, kwargs.get('eval_workers', 1)):
        return b1.run(*args, **kwargs)


def verify(*args, decode=None, **kwargs):
    with _runner(decode, kwargs.get('eval_workers', 1)):
        return b1.verify(*args, **kwargs)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args[:1] not in ([c] for c in COMMANDS):
            raise ValueError('only the train and verify subcommands are registered')
        decode = gb.decode_mode_from_env()
        _gate(decode, b1.gs.eval_workers_from_env())
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G-ARCH-01 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    with _runner(decode):
        return b1.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
