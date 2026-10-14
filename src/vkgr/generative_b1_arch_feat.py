"""GFEAT-02 opt-in entry: the Batch1 G runner with GARCH-01's paper-scale model (A1) and GFEAT-01's F2 / F3 encoder side features.

``generative_b1_run`` / ``generative_b1_batched`` / ``generative_b1_arch`` / ``generative_b1_feat`` stay byte-identical, so nothing
trained by them changes. Composition, scoped to each call and restored on exit (success or error):

  1. GARCH-01 gate (one evaluation worker; batched decode only with batch size 1)
  2. ``generative_b1_feat._runner`` (on top of the INFRA-02 batched runner): side contexts / tokenizer / training step / side
     evaluation / ``binding.feat01`` - unchanged GFEAT-01 code
  3. then four names of ``generative_b1_run`` (the GARCH-01 D2 pattern): ``B1_MEMBERS`` (+ arch and GFEAT-02 files),
     ``validate_config`` (A1 + feature registry), ``make_model_b1`` (arch factory with side tables), ``gt`` (frozen training module
     except the optimizer factory, whose core check only knows the anchor structure)

feature=none goes through GARCH-01's factory and GFEAT-01's none branches, so it trains A1 itself (the control).
"""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from . import generative_arch_feat as gaf
from . import generative_b1_arch as ba
from . import generative_b1_feat as feat
from . import generative_b1_run as b1
from . import generative_batched as gb
from . import generative_data as gd
from . import generative_sharded as gs

ADDED_MEMBERS = ('src/vkgr/generative_arch_feat.py', 'src/vkgr/generative_b1_arch_feat.py')
MEMBERS = feat.FEAT_MEMBERS + ba.ARCH_MEMBERS + ADDED_MEMBERS
COMMANDS = ('train', 'verify')


@contextmanager
def _runner(decode, side, workers=1):
    ba._gate(decode, workers)
    with feat._runner(decode, side):
        saved = []
        try:
            ba._swap(saved, 'B1_MEMBERS', MEMBERS)
            ba._swap(saved, 'validate_config', gaf.validate_config)
            ba._swap(saved, 'make_model_b1', gaf.make_model)
            ba._swap(saved, 'gt', ba._FrozenTraining(b1.gt))
            yield b1
        finally:
            ba._restore(saved)


def run(*args, decode=None, side=None, **kwargs):
    with _runner(decode, side, kwargs.get('eval_workers', 1)):
        return b1.run(*args, **kwargs)


def verify(*args, decode=None, side=None, **kwargs):
    with _runner(decode, side, kwargs.get('eval_workers', 1)):
        return b1.verify(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description='GFEAT-02 G runner (opt-in entry; A1 paper-scale model + F2/F3 history-side features); test split closed')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in COMMANDS:
        p = sub.add_parser(name)
        p.add_argument('--prepared', required=True); p.add_argument('--prepared-sha', required=True)
        p.add_argument('--history-manifest', required=True); p.add_argument('--history-sha', required=True)
        p.add_argument('--side-manifest'); p.add_argument('--side-sha')
        p.add_argument('--output', required=True); p.add_argument('--scratch', required=True)
        if name == 'train':
            p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True)
        else:
            p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        decode = feat.check_decode(gb.decode_mode_from_env())
        ba._gate(decode, gs.eval_workers_from_env())
        progress = gs.progress_target_from_env()
        if (args.side_manifest is None) != (args.side_sha is None):
            raise ValueError('side manifest and side sha are given together')
        side = None if args.side_manifest is None else (Path(args.side_manifest), args.side_sha)
        common = (root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha)
        if args.command == 'train':
            config = gd._json(gd._checked_file(Path(args.config), args.config_sha))
            value = run(*common, config, Path(args.output), temporary=Path(args.scratch), decode=decode, side=side, progress=progress)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']}))
        else:
            print(json.dumps(verify(*common, Path(args.manifest), args.manifest_sha, Path(args.output), temporary=Path(args.scratch), decode=decode, side=side)))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G-FEAT02 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
