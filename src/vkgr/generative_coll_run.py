"""S1 core03 independent runner, original project code adapted from generative_tok_run.
Same T3 training/selection; collaborative bundle/input family only. Old entries remain frozen.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import generative_b1_run as b1
from . import generative_batched as gb
from . import generative_data as gd
from . import generative_evaluation as ev
from . import generative_real as gr
from . import generative_sharded as gs
from . import generative_training as gt
from . import history_v2
from . import sid_coll_v1 as sid_v2
from . import sid_coll_run as coll_run
from .data_audit import canonical_bytes, file_sha256
from .generative_b1_batched import _evaluator
from .generative_data_v2 import SIDTokensL
from .generative_hp import EarlyStop, core_config, make_model_b1
from .generative_tok_data import HISTORY_LENGTH, build_tok_context

SCHEMA = 'g-coll-run-v1'
TASK_ID = 'SID-COLL-S1-EXEC'
MANIFEST_SCHEMA = 'g-coll-train-manifest-v1'
MANIFEST_KIND = 'g_coll_train'
CHECKPOINT_SCHEMA = 'g-coll-checkpoint-v1'
BUDGET = b1.BUDGET
SEEDS = (42, 2024, 2025)
CONTENT_DIM = 64
TOK_MEMBERS = b1.B1_MEMBERS + ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py', 'src/vkgr/sid_v2.py',
                               'src/vkgr/sid_rqvae.py', 'src/vkgr/sid_tok_audit.py', 'src/vkgr/generative_tok_data.py',
                               'src/vkgr/generative_coll_run.py', 'src/vkgr/sid_coll_v1.py', 'src/vkgr/sid_coll_run.py',
                               'src/vkgr/sid_coll_export.py', 'src/vkgr/sid_coll_sgns.py') + tuple(
                               f'configs/sid_coll/g-coll-{v.lower()}-s{seed}.json' for v in ('C1','C2') for seed in SEEDS)
_digest = b1._digest
_runtime = b1._runtime
_guard = b1._guard
_save = b1._save
_write_manifest = b1._write_manifest
_read_evaluation = b1._read_evaluation
train_batch = b1.train_batch


def default_config():
    """The anchor config (configs/batch1/g-hist-v2-l50-s42.json) with schema, task_id and sid replaced."""
    value = b1.default_config()
    value.update({'schema': SCHEMA, 'task_id': TASK_ID, 'sid': sid_v2.SID_VERSIONS[0]})
    return value


def validate_config(value, engineering=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered G-TOK config fields')
    fixed = ('schema', 'task_id', 'heads', 'encoder_layers', 'decoder_layers', 'dropout', 'precision', 'weight_decay', 'gradient_clip_norm',
             'effective_batch', 'init', 'history_version', 'history_len', 'patience', 'beam', 'test_gate')
    if any(value[k] != base[k] or type(value[k]) is not type(base[k]) for k in fixed):
        raise ValueError('frozen G-TOK field changed')
    allowed = sid_v2.SID_VERSIONS
    if type(value['sid']) is not str or value['sid'] not in allowed:
        raise ValueError('unregistered SIDMAP_V2 label for this context')
    if type(value['seed']) is not int or value['seed'] not in SEEDS or any(type(value[k]) is not int for k in ('d_model', 'ff', 'microbatch', 'max_epochs', 'min_epochs')):
        raise ValueError('typed seed/sizes')
    if engineering:
        if value['d_model'] not in (64, 128) or value['ff'] != 4 * value['d_model'] or value['learning_rate'] not in (.001, .0003) \
                or not 1 <= value['microbatch'] <= 512 or not 1 <= value['max_epochs'] <= 100 or not 0 <= value['min_epochs'] <= value['max_epochs']:
            raise ValueError('bounded engineering config')
    elif (value['d_model'], value['ff'], value['learning_rate'], value['max_epochs'], value['min_epochs']) != (128, 512, .0003, 100, 5) \
            or value['microbatch'] != 512:
        raise ValueError('formal G-TOK = core03 + BUDGET_E100_MIN5_P3')
    return value


def _source(root):
    files = {name: file_sha256(Path(root) / name) for name in TOK_MEMBERS}
    return {'files': files, 'source_sha256': _digest(files), 'git': gd._git(root)}


def _load_sid(sid_manifest, sid_sha, content_path, content_sha, catalog, engineering,
              input_manifest, input_sha, vectors_path, vectors_sha,
              expected_content_digest=sid_v2.CONTENT_V1_FEATURE_SHA256):
    array = np.load(gd._checked_file(content_path, content_sha), allow_pickle=False)
    ids = np.array([int(x) for x in catalog], dtype=np.uint32)
    if array.dtype != np.float32 or array.shape != (len(ids)+1,64) or not np.isfinite(array).all() or np.any(array[0]):
        raise ValueError('finite CONTENT_V1 with PAD required')
    from .sid_tok_audit import feature_digest
    digest=feature_digest(array[1:])
    if not engineering and digest != sid_v2.CONTENT_V1_FEATURE_SHA256: raise ValueError('H5a CONTENT_V1 mismatch')
    im=gd._json(gd._checked_file(input_manifest,input_sha))
    if im['content_feature_sha256']!=digest: raise ValueError('actual content provenance mismatch')
    vectors=np.load(gd._checked_file(vectors_path,vectors_sha),allow_pickle=False)
    return sid_v2.load_bundle(sid_manifest,sid_sha,catalog_ids=ids,vectors=vectors,input_manifest=im,engineering=engineering)


def _data_root(root):
    """Keep immutable new code separate from the pinned historical G3 input channel."""
    value=os.environ.get('VKGR_DATA_ROOT')
    if value is not None and not value.strip(): raise ValueError('VKGR_DATA_ROOT must name an existing directory')
    try:
        path=Path(root if value is None else value).resolve(strict=True)
    except (OSError,RuntimeError) as exc:
        raise ValueError('VKGR_DATA_ROOT must name an existing directory') from exc
    if not path.is_dir(): raise ValueError('VKGR_DATA_ROOT must name an existing directory')
    return path


def _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha, config,
              temporary, *, input_manifest, input_sha, vectors_path, vectors_sha, expected_content_digest=sid_v2.CONTENT_V1_FEATURE_SHA256):
    temporary = Path(temporary); temporary.mkdir(parents=True, exist_ok=False)
    base = gr.load_context(_data_root(root), prepared_path, prepared_sha, temporary)
    engineering = base.binding.get('engineering') is True and base.binding.get('synthetic') is True
    validate_config(config, engineering)
    loaded = history_v2.load_prepared(history_manifest, history_sha, HISTORY_LENGTH)
    sid = _load_sid(sid_manifest, sid_sha, content_path, content_sha, base.catalog, engineering, input_manifest, input_sha, vectors_path, vectors_sha, expected_content_digest)
    if sid.info['sid_version'] != config['sid']:
        raise ValueError('config sid differs from the bundle sid_version')
    return build_tok_context(base, loaded, HISTORY_LENGTH, sid), engineering, sid


def _binding(root, context, config, runtime, model, prepared_sha, history_sha, sid):
    info = sid.info
    return {'task': TASK_ID, 'budget': BUDGET, 'init_version': 'INIT_EMB_N002', 'config': config, 'config_sha256': _digest(config),
            'context_binding_sha256': _digest(context.binding), 'context_b1': context.binding['b1'], 'source': _source(root),
            'data_root': str(_data_root(root)),
            'runtime': runtime, 'init_report': model.init_report, 'prepared_manifest_sha256': prepared_sha, 'history_manifest_sha256': history_sha,
            'sid': {'sid_version': info['sid_version'], 'manifest_sha256': info['manifest_sha256'],
                    'content_feature_sha256': sid.manifest['content_feature_sha256'], 'branching': info['branching'],
                    'suffix_vocabulary': info['suffix_vocabulary'], 'vocab_size': info['vocab_size'], 'kind': info['kind'],
                    'engineering': info['engineering'], 'input_feature_sha256': sid.manifest['input_feature_sha256'],
                    'input_manifest_sha256': sid.manifest['input_manifest_sha256'], 'input_dimension': sid.manifest['input_dimension']},
            'synthetic': context.binding['synthetic'], 'test_gate': 'closed'}


def _check_batch_one(decode):
    """Batched decode is only proven equal to the serial decode at batch size 1; larger batches are refused."""
    if decode is not None and decode['batch_size'] != 1:
        raise ValueError('decode env: G-TOK batched decode is exact only with VKGR_DECODE_BATCH=1')


def run(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha, config, output, *,
        temporary, eval_workers=1, progress=None, decode=None, input_manifest=None, input_sha=None, vectors_path=None, vectors_sha=None, expected_content_digest=sid_v2.CONTENT_V1_FEATURE_SHA256):
    decode = {'mode':'batched','batch_size':1} if decode is None else decode
    root = Path(root); evaluate = _evaluator(decode)
    if decode is not None and eval_workers != 1:
        raise ValueError('decode env: batched decode excludes VKGR_EVAL_WORKERS>1')
    _check_batch_one(decode)
    context, engineering, sid = _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path,
                                          content_sha, config, temporary, input_manifest=input_manifest, input_sha=input_sha, vectors_path=vectors_path, vectors_sha=vectors_sha, expected_content_digest=expected_content_digest)
    runtime = _runtime(engineering); device = runtime['device']
    output = gd._no_symlink(Path(output)); output.mkdir(mode=0o700, exist_ok=False)
    gs.check_progress_outside(progress, output)
    guard = _guard(engineering, output, device)
    gr._seed(config['seed']); model = make_model_b1(context.codec, config, device=device)
    optimizer = gt.make_optimizer(model, core_config(config)); tokens = SIDTokensL(context.codec, config['history_len'])
    cursor = gt.EpochCursor(config['seed'], len(context.examples)); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    binding = _binding(root, context, config, runtime, model, prepared_sha, history_sha, sid)
    losses = []; epochs = []; best = None; last = None; wall = {'train': 0., 'evaluate': 0.}; progress_at = [None]
    def report(stage, completed=None, total=None, force=False):
        now = time.monotonic()
        if progress is None or not force and progress_at[0] is not None and now - progress_at[0] < 30:
            return
        progress_at[0] = now; gs.write_progress(progress, gs.progress_record(progress, stage, 'formal', cursor.epoch, config['max_epochs'], completed, total))
    while True:
        if cursor.phase == 'epoch_done':
            if stop.stopped:
                break
            cursor.advance_epoch()
        begin = time.perf_counter()
        while cursor.phase == 'train':
            guard(); indices = cursor.next_batch(512)
            result = train_batch(model, optimizer, context.examples.rows(indices), context, tokens, config, device)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            losses.append({'epoch': cursor.epoch, 'examples': result.examples, 'loss_mean': result.loss_mean}); report('train')
        wall['train'] += time.perf_counter() - begin
        epoch = cursor.epoch; guard()
        state = {'schema': CHECKPOINT_SCHEMA, 'binding_sha256': _digest(binding), 'epoch': epoch, 'model': dict(model.state_dict()),
                 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(), 'selection': stop.state(), 'rng': gt._rng(model)}
        checkpoint_sha = _save(output / 'last.pt', state); model_sha = ev.model_digest(model)
        begin = time.perf_counter(); report('evaluate', force=True)
        kind, evaluation = evaluate(model, context, tokens, config, runtime, output / f'evaluation-{epoch:03d}', checkpoint_sha, guard, eval_workers,
                                    on_shard=lambda completed, total: report('evaluate', completed, total))
        wall['evaluate'] += time.perf_counter() - begin
        aggregate = evaluation['aggregate']; main_metrics = aggregate['metrics']['all_eligible']
        improved = stop.observe(epoch, main_metrics['recall@50'], main_metrics['ndcg@50']); cursor.complete_evaluation()
        last = {'path': 'last.pt', 'sha256': checkpoint_sha, 'epoch': epoch}
        if improved:
            shutil.copyfile(output / 'last.pt', output / 'best.pt.partial'); (output / 'best.pt.partial').replace(output / 'best.pt')
            if file_sha256(output / 'best.pt') != checkpoint_sha:
                raise ValueError('best checkpoint copy mismatch')
            best = {'path': 'best.pt', 'sha256': checkpoint_sha, 'epoch': epoch, 'metric': list(stop.best_metric)}
        epochs.append({'epoch': epoch, 'metrics': aggregate, 'model_sha256': model_sha, 'checkpoint_sha256': checkpoint_sha,
                       'evaluation': {'kind': kind, 'path': f'evaluation-{epoch:03d}', 'manifest_sha256': evaluation['manifest_sha256'], 'computation_sha256': evaluation['manifest']['computation_sha256']},
                       'observations': evaluation['manifest'].get('observations')})
        value = {'schema': MANIFEST_SCHEMA, 'kind': MANIFEST_KIND, 'status': 'complete' if stop.stopped else 'running', 'config': config,
                 'binding': binding, 'selection': stop.state(), 'epochs': epochs, 'losses': losses, 'best': best, 'last': last, 'wall_seconds': dict(wall),
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience, 'eval_workers': eval_workers, 'decode': decode}
        sha = _write_manifest(output / 'manifest.json', value); report('complete' if stop.stopped else 'train', force=True)
    return {**value, 'manifest_path': output / 'manifest.json', 'manifest_sha256': sha}


def _read_run(path, pin):
    value = gd._json(gd._checked_file(Path(path), pin))
    keys = {'schema', 'kind', 'status', 'config', 'binding', 'selection', 'epochs', 'losses', 'best', 'last', 'wall_seconds', 'hit_max_epochs', 'eval_workers', 'decode'}
    if type(value) is not dict or set(value) != keys or value['schema'] != MANIFEST_SCHEMA or value['kind'] != MANIFEST_KIND or value['status'] != 'complete':
        raise ValueError('complete G-TOK train manifest required')
    return value


def _check_binding(recorded, fresh):
    if canonical_bytes(recorded) != canonical_bytes(fresh):
        raise ValueError('run input/source/runtime/init/SID binding mismatch')


def _check_best_hash(run_dir, best):
    if type(best) is not dict or best.get('path') != 'best.pt' or file_sha256(run_dir / 'best.pt') != best.get('sha256'):
        raise ValueError('best checkpoint hash mismatch')


def _load_best(run_dir, value, context, config, runtime, binding):
    best = value['best']
    _check_best_hash(run_dir, best)
    state = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    if state.get('schema') != CHECKPOINT_SCHEMA or state.get('binding_sha256') != _digest(binding) or state.get('epoch') != best['epoch']:
        raise ValueError('best checkpoint identity mismatch')
    gr._seed(config['seed']); model = make_model_b1(context.codec, config, device=runtime['device'])
    model.load_state_dict(state['model'])
    return model


def verify(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha, manifest_path, manifest_sha,
           output, *, temporary, eval_workers=1, decode=None, input_manifest=None, input_sha=None, vectors_path=None, vectors_sha=None, expected_content_digest=sid_v2.CONTENT_V1_FEATURE_SHA256):
    root = Path(root); manifest_path = Path(manifest_path); run_dir = manifest_path.parent; evaluate = _evaluator(decode)
    if decode is not None and eval_workers != 1:
        raise ValueError('decode env: batched decode excludes VKGR_EVAL_WORKERS>1')
    _check_batch_one(decode)
    value = _read_run(manifest_path, manifest_sha); config = value['config']
    context, engineering, sid = _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path,
                                          content_sha, config, temporary, input_manifest=input_manifest, input_sha=input_sha, vectors_path=vectors_path, vectors_sha=vectors_sha, expected_content_digest=expected_content_digest)
    runtime = _runtime(engineering)
    gr._seed(config['seed']); fresh = make_model_b1(context.codec, config, device=runtime['device'])
    binding = _binding(root, context, config, runtime, fresh, prepared_sha, history_sha, sid)
    _check_binding(value['binding'], binding)
    tokens = SIDTokensL(context.codec, config['history_len']); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    expected_best = None
    for epoch, record in enumerate(value['epochs'], 1):
        if record.get('epoch') != epoch:
            raise ValueError('epoch sequence')
        aggregate, computation, _ = _read_evaluation(record['evaluation']['kind'], run_dir, record, context, tokens)
        if canonical_bytes(aggregate) != canonical_bytes(record['metrics']) or computation != record['evaluation']['computation_sha256']:
            raise ValueError('recorded epoch metrics not reproducible from published rows')
        main_metrics = aggregate['metrics']['all_eligible']
        if stop.observe(epoch, main_metrics['recall@50'], main_metrics['ndcg@50']):
            expected_best = record
    if not stop.stopped or stop.state() != value['selection'] or expected_best is None:
        raise ValueError('selection replay mismatch')
    best = value['best']
    if best.get('epoch') != expected_best['epoch'] or best.get('sha256') != expected_best['checkpoint_sha256'] or best.get('metric') != list(stop.best_metric):
        raise ValueError('best selection mismatch')
    if type(value['last']) is not dict or value['last'].get('epoch') != len(value['epochs']) or file_sha256(run_dir / 'last.pt') != value['last'].get('sha256') \
            or value['last']['sha256'] != value['epochs'][-1]['checkpoint_sha256']:
        raise ValueError('last checkpoint mismatch')
    if sorted(p.name for p in run_dir.glob('*.pt')) != ['best.pt', 'last.pt']:
        raise ValueError('checkpoint policy best+last violated')
    model = _load_best(run_dir, value, context, config, runtime, binding)
    if ev.model_digest(model) != expected_best['model_sha256']:
        raise ValueError('best model digest mismatch')
    output = gd._no_symlink(Path(output)); output.mkdir(mode=0o700, exist_ok=False)
    _, replay = evaluate(model, context, tokens, config, runtime, output / f"replay-best-{best['epoch']:03d}", best['sha256'],
                         _guard(engineering, output, runtime['device']), eval_workers)
    if replay['manifest']['computation_sha256'] != expected_best['evaluation']['computation_sha256']:
        raise ValueError('best checkpoint re-evaluation mismatch')
    return {'status': 'verified', 'replayed_epochs': [best['epoch']], 'best_epoch': best['epoch'], 'epochs': len(value['epochs'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description='GTOK-01 G runner for SIDMAP_V2 tokenizers (new version; default runners untouched); test split closed')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--prepared', required=True); p.add_argument('--prepared-sha', required=True)
        p.add_argument('--history-manifest', required=True); p.add_argument('--history-sha', required=True)
        p.add_argument('--sid-manifest', required=True); p.add_argument('--sid-sha', required=True)
        p.add_argument('--content', required=True); p.add_argument('--content-sha', required=True)
        p.add_argument('--input-manifest', required=True); p.add_argument('--input-sha', required=True)
        p.add_argument('--vectors', required=True); p.add_argument('--vectors-sha', required=True)
        p.add_argument('--output', required=True); p.add_argument('--scratch', required=True)
        if name == 'train':
            p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True)
        else:
            p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        decode = gb.decode_mode_from_env(); workers = gs.eval_workers_from_env(); progress = gs.progress_target_from_env()
        if decode is not None and workers != 1:
            raise ValueError('decode env: batched decode excludes VKGR_EVAL_WORKERS>1')
        common = (root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha,
                  Path(args.sid_manifest), args.sid_sha, Path(args.content), args.content_sha)
        if args.command == 'train':
            config = gd._json(gd._checked_file(Path(args.config), args.config_sha))
            value = run(*common, config, Path(args.output), temporary=Path(args.scratch), eval_workers=workers, progress=progress, decode=decode, input_manifest=args.input_manifest, input_sha=args.input_sha, vectors_path=args.vectors, vectors_sha=args.vectors_sha)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']}))
        else:
            print(json.dumps(verify(*common, Path(args.manifest), args.manifest_sha, Path(args.output), temporary=Path(args.scratch),
                                    eval_workers=workers, decode=decode, input_manifest=args.input_manifest, input_sha=args.input_sha, vectors_path=args.vectors, vectors_sha=args.vectors_sha)))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G-TOK gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
