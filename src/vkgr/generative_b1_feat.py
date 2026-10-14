"""GFEAT-01 opt-in entry: the Batch1 G runner with FEAT-01 F2 / F3 history-side features in the encoder.

``python -m vkgr.generative_b1_run`` and ``python -m vkgr.generative_b1_batched`` stay byte-identical to the INFRA-02 base, so
nothing trained by them changes. This module runs the same runner, on top of the INFRA-02 batched entry, with eight module-level
names of ``generative_b1_run`` replaced inside a context manager and restored on exit (success or error):

  feature=none              every replacement delegates to the original function; only ``binding.source`` (new members) and
                            ``binding.feat01`` differ, so the control reproduces the anchor's model_sha256 / computation_sha256 / selection
  feature=f2_feedback       six feedback-bit embeddings (multi-hot) summed per history item
  feature=f3_watch_buckets  timespent-bucket + watch-ratio-bucket embeddings per history item

The side data is the same ``HIST_V2_SIDE_V1`` sidecar the P line reads (sha-bound to the history-v2 prepare); the feature
definitions are imported from the modules shared byte-for-byte with the P line. Training evaluates with the batch-1 beam decode
(``VKGR_DECODE=batched VKGR_DECODE_BATCH=1``); verify decodes query by query. Only batch size 1 and a single evaluation worker
are registered: a decoder call that mixes queries is not bit-equal to the serial decode (INFRA-02b).
"""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from . import generative_b1_batched as bb
from . import generative_b1_run as b1
from . import generative_batched as gb
from . import generative_data as gd
from . import generative_feat as gf
from . import generative_feat_eval as gfe
from . import generative_sharded as gs
from . import history_v2
from .generative_data_v2 import SID_VERSIONS

_ANCHOR_VALIDATE = b1.validate_config  # fixed at import: b1.validate_config is replaced by validate_config_feat while the entry runs
SCHEMA = 'g-feat01-run-v1'
TASK = 'GFEAT-01'
ADDED_MEMBERS = tuple(f'src/vkgr/{n}.py' for n in ('history_side', 'retrieval_features', 'retrieval_hp', 'retrieval_sasrec', 'retrieval_models',
                                                   'generative_feat', 'generative_feat_eval', 'generative_b1_feat'))
FEAT_MEMBERS = bb.MEMBERS + ADDED_MEMBERS
REPLACED = ('B1_MEMBERS', 'validate_config', '_contexts', 'make_model_b1', 'SIDTokensL', 'train_batch', '_evaluate', '_binding')


def default_config(feature='none'):
    return {**b1.default_config(), 'schema': SCHEMA, 'task_id': TASK, 'feature': feature, 'side_version': gf.SIDE_VERSION}


def validate_config_feat(value, engineering=False):
    """Anchor config + {feature, side_version}; everything else is the anchor's frozen contract (no HPO, seed 42/2024/2025)."""
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered GFEAT-01 config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK or value['side_version'] != gf.SIDE_VERSION or value['feature'] not in gf.FEATURES:
        raise ValueError('unregistered GFEAT-01 schema/task/feature/side version')
    if type(value['history_len']) is not int or value['history_len'] != 50 or value['sid'] != SID_VERSIONS[0] or value['init'] != 'small':
        raise ValueError('GFEAT-01 registers history_len=50, the BHKM SID and init=small only')
    anchor = {k: v for k, v in value.items() if k not in ('feature', 'side_version')}
    anchor.update(schema=b1.default_config()['schema'], task_id='HIST-V2')
    _ANCHOR_VALIDATE(anchor, engineering)
    return value


def check_decode(decode):
    """None (serial) or the INFRA-02 batched switch with batch size 1."""
    gb.check_decode(decode)
    if decode is not None and decode['batch_size'] != 1:
        raise ValueError('decode env: GFEAT-01 registers batch size 1 only (batch>1 is not bit-equal to the serial decode)')
    return decode


def _check_side(side):
    if side is not None and (type(side) is not tuple or len(side) != 2):
        raise ValueError('side must be (manifest path, sha256)')
    return side


@contextmanager
def _runner(decode, side):
    check_decode(decode); _check_side(side)
    state = {'pack': None}
    with bb._runner(decode):
        saved = {name: getattr(b1, name) for name in REPLACED}
        inherited = dict(saved)

        def contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, config, temporary):
            # first statement: nothing (scratch, output) exists yet when the side arguments are inconsistent
            feature = config.get('feature') if type(config) is dict else None
            if feature not in gf.FEATURES:
                raise ValueError('unregistered GFEAT-01 feature')
            if (feature == 'none') != (side is None):
                raise ValueError('GFEAT-01: side manifest and sha are required for f2/f3 and forbidden for feature=none')
            context, engineering = inherited['_contexts'](root, prepared_path, prepared_sha, history_manifest, history_sha, config, temporary)
            if feature == 'none':
                state['pack'] = None
                return context, engineering
            loaded = history_v2.load_prepared(history_manifest, history_sha, config['history_len'])
            state['pack'] = gf.load_side_pack(side[0], side[1], history_sha, loaded, context, config['history_len'])
            return gf.attach_side(context, state['pack']), engineering

        def tokens(codec, n):
            return inherited['SIDTokensL'](codec, n) if state['pack'] is None else gf.SIDTokensSide(codec, n, state['pack'])

        def train_batch(model, optimizer, rows, context, tokens, config, device):
            if config['feature'] == 'none':
                return inherited['train_batch'](model, optimizer, rows, context, tokens, config, device)
            return gf.train_batch_feat(model, optimizer, rows, context, tokens, config, device)

        def evaluate(model, context, tokens, config, runtime, output, checkpoint_sha, guard, workers, on_shard=None):
            if config['feature'] == 'none':
                return inherited['_evaluate'](model, context, tokens, config, runtime, output, checkpoint_sha, guard, workers, on_shard=on_shard)
            if workers != 1 or config['history_len'] != 50 or state['pack'] is None:
                raise ValueError('GFEAT-01 side evaluation: one worker, history_len 50 and a loaded side pack required')
            return 'query_evaluation', gfe.evaluate_queries_side(model, context, state['pack'].validation, output=output, checkpoint_sha256=checkpoint_sha,
                                                                 batch_size=None if decode is None else decode['batch_size'], budget_check=guard, on_shard=on_shard)

        def binding(root, context, config, runtime, model, prepared_sha, history_sha):
            value = inherited['_binding'](root, context, config, runtime, model, prepared_sha, history_sha)
            return {**value, 'feat01': {'feature': config['feature'], 'side_version': None if config['feature'] == 'none' else gf.SIDE_VERSION}}

        try:
            b1.B1_MEMBERS = FEAT_MEMBERS; b1.validate_config = validate_config_feat; b1._contexts = contexts; b1.make_model_b1 = gf.make_model_feat
            b1.SIDTokensL = tokens; b1.train_batch = train_batch; b1._evaluate = evaluate; b1._binding = binding
            yield b1
        finally:
            for name, value in saved.items():
                setattr(b1, name, value)


def run(*args, decode=None, side=None, **kwargs):
    with _runner(decode, side):
        return b1.run(*args, **kwargs)


def verify(*args, decode=None, side=None, **kwargs):
    with _runner(decode, side):
        return b1.verify(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description='GFEAT-01 G runner (opt-in entry; FEAT-01 F2/F3 history-side features); test split closed')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'verify'):
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
        decode = check_decode(gb.decode_mode_from_env())
        if gs.eval_workers_from_env() != 1:
            raise ValueError('eval workers env: GFEAT-01 registers one evaluation worker')
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
        print(f'G-FEAT01 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
