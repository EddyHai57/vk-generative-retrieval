"""G-HP-01 opt-in entry point: the Batch1 G runner on the G-HP registry (one-knob tuning around the paper-scale anchor A1).

``python -m vkgr.generative_b1_run`` / ``generative_b1_batched`` / ``generative_b1_arch`` stay untouched (their
``binding.source`` and every run trained by them are unchanged). This module nests inside the GARCH-01 entry (which nests
the INFRA-02 batched entry) and, scoped to each call, re-swaps four names of ``generative_b1_run``: ``validate_config``
(G-HP registry), ``make_model_b1`` (G-HP factory, a thin delegation to the GARCH factory), ``B1_MEMBERS`` (+ the two G-HP
files, so G-HP runs bind their own source) and ``gt`` (the GARCH training proxy, except ``make_optimizer`` which takes the
G-HP learning rates). Everything is restored on exit, also on the exception path.

Guards are the GARCH-01 ones (fail-closed before any output): ``VKGR_EVAL_WORKERS`` > 1, batched decode with batch size != 1,
any subcommand other than ``train`` / ``verify``.
"""
import sys
from contextlib import contextmanager

from . import generative_b1_arch as gba
from . import generative_b1_run as b1
from . import generative_batched as gb
from . import generative_hp_arch as hp

HP_MEMBERS = ('src/vkgr/generative_hp_arch.py', 'src/vkgr/generative_b1_hp.py')
MEMBERS = gba.MEMBERS + HP_MEMBERS
COMMANDS = gba.COMMANDS


class _HPTraining:
    """Delegates to the training module in place when entered (the GARCH proxy) except ``make_optimizer``."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def make_optimizer(model, config):
        return hp.make_optimizer_hp(model, config)


@contextmanager
def _runner(decode, workers=1):
    gba._gate(decode, workers)
    with gba._runner(decode, workers):
        saved = []
        try:
            gba._swap(saved, 'B1_MEMBERS', MEMBERS)
            gba._swap(saved, 'validate_config', hp.validate_config)
            gba._swap(saved, 'make_model_b1', hp.make_model_hp)
            gba._swap(saved, 'gt', _HPTraining(b1.gt))
            yield b1
        finally:
            gba._restore(saved)


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
        gba._gate(decode, b1.gs.eval_workers_from_env())
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G-HP-01 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    with _runner(decode):
        return b1.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
