"""New SID A1 confirmation. Frozen A1/T3 procedure; only COLL input and run identity differ.
Original project implementation adapted from generative_g6_run; no old entry is modified.
"""
import argparse
import os
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from . import generative_arch as ga
from . import generative_b1_run as b1
from . import generative_batched as gb
from . import generative_data as gd
from . import generative_evaluation as ev
from . import generative_real as gr
from . import generative_sharded as gs
from . import generative_training as gt
from . import generative_coll_run as coll_base
from . import sid_structure_v1 as coll
from . import history_v2
from . import sid_depth as sd
from . import sid_depth_beam as sdb
from . import sid_v2
from . import sid_v2d
from . import tiger
from .data_audit import canonical_bytes, file_sha256
from .generative_b1_batched import _evaluator
from .generative_g6_data import HISTORY_LENGTH, build_g6_context
from .generative_g6_model import make_model_arch_d, train_batch_d
from .generative_hp import EarlyStop

SCHEMA = 'g-sid-structure-a1-run-v1'
TASK_ID = 'SID-STRUCTURE-MATCHED'
MANIFEST_SCHEMA = 'g-sid-structure-a1-train-manifest-v1'
MANIFEST_KIND = 'g_sid_structure_a1_train'
CHECKPOINT_SCHEMA = 'g-sid-structure-a1-checkpoint-v1'
SID_VERSION = 'STRUCT_C1_B48D3_S42'
SID_SHA = 'UNSEALED'
INPUT_SHA = '725fb211a342c518770623be81d490feaaec38a285c954b5b6f51f50a388e9a4'
VECTORS_SHA = 'bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7'
V2_DEPTH = 3
BUDGET = b1.BUDGET
SEEDS = ga.SEEDS
CONTENT_DIM = 64
G6_MEMBERS = b1.B1_MEMBERS + ('uv.lock',) + tuple(f'src/vkgr/{name}.py' for name in (
    'generative_batched', 'generative_b1_batched', 'tiger_arch', 'generative_arch', 'sid_depth', 'sid_depth_beam', 'sid_v2', 'sid_v2d',
    'sid_rqvae', 'sid_rqvae_d', 'sid_tok_audit', 'sid_tok_audit_d', 'generative_g6_model', 'generative_g6_data', 'generative_g6_run', 'coll_sid_a1_run',
    'generative_coll_run', 'sid_coll_v1', 'sid_coll_run', 'sid_coll_sgns', 'sid_coll_export', 'sid_structure_v1', 'sid_structure_run', 'sid_structure_a1_run')) + tuple(
    f'configs/sid_structure/g-structure-a1-s{seed}.json' for seed in SEEDS)
_digest = b1._digest
_runtime = b1._runtime
_guard = b1._guard
_save = b1._save
_write_manifest = b1._write_manifest
_read_evaluation = b1._read_evaluation


def default_config():
    value = ga.default_config()
    value.update(schema=SCHEMA, task_id=TASK_ID, sid=SID_VERSION, sid_manifest_sha256=SID_SHA)
    return value


def validate_config(value, engineering=False):
    base = default_config()
    if type(engineering) is not bool or type(value) is not dict or set(value) != set(base):
        raise ValueError('registered A1 collaborative config required')
    if type(value['seed']) is not int or value['seed'] not in SEEDS:
        raise ValueError('three frozen seeds only')
    pin=value['sid_manifest_sha256']
    if type(pin) is not str or (not engineering and (len(pin)!=64 or set(pin)-set('0123456789abcdef') or pin=='0'*64)):
        raise ValueError('sealed SID manifest sha256 required')
    variable = {'seed','sid_manifest_sha256'}
    if engineering:
        variable |= set(ga.SHAPE_KEYS) | {'microbatch', 'max_epochs', 'min_epochs'}
    if any(type(value[k]) is not type(base[k]) or value[k] != base[k] for k in base if k not in variable):
        raise ValueError('frozen A1 field changed')
    if engineering:
        adapted = dict(value, schema=ga.SCHEMA, task_id=ga.TASK_ID, sid=ga.FIXED['sid'])
        adapted.pop('sid_manifest_sha256')
        ga.validate_config(adapted, True)
    return value


def check_pins(sid_sha, input_sha, vectors_sha, engineering, *, config=None):
    if not engineering:
        if (input_sha,vectors_sha)!=(INPUT_SHA,VECTORS_SHA) or type(sid_sha) is not str or len(sid_sha)!=64 or set(sid_sha)-set('0123456789abcdef'):
            raise ValueError('frozen C1 input/vector and explicit SID byte pins required')
        if type(config) is not dict or config.get('sid_manifest_sha256')!=sid_sha:
            raise ValueError('SID hash differs from the frozen per-run config')


@contextmanager
def depth_decode():
    """Swap the two frozen decoder names the evaluators resolve at call time for the depth-aware twins; restored on exit."""
    saved = (tiger.decode_history, gb.decode_histories_batched)
    tiger.decode_history, gb.decode_histories_batched = sdb.decode_history_d, sdb.decode_histories_batched_d
    try:
        yield
    finally:
        tiger.decode_history, gb.decode_histories_batched = saved


def _evaluator_d(decode):
    inner = _evaluator(decode)

    def evaluate(*args, **kwargs):
        with depth_decode():
            return inner(*args, **kwargs)
    return evaluate


def _gate(decode, workers):
    gb.check_decode(decode)
    if workers != 1:
        raise ValueError('G6: VKGR_EVAL_WORKERS must be 1 (sharded workers rebuild the model with the frozen factory)')
    if decode is not None and decode['batch_size'] != 1:
        raise ValueError('G6: batched decode only with batch size 1 (batch > 1 is not bit-equal on GPU)')


def _source(root):
    files = {name: file_sha256(Path(root) / name) for name in G6_MEMBERS}
    return {'files': files, 'source_sha256': _digest(files), 'git': gd._git(root)}


def _content_rows(content_path, content_sha, catalog, engineering, expected_content_digest):
    """Pinned content rows (one per catalog item, PAD row dropped) and the catalog ids; formal contexts accept only CONTENT_V1."""
    if type(engineering) is not bool:
        raise ValueError('engineering flag must be a bool')
    if not engineering and expected_content_digest != sid_v2d.CONTENT_V1_FEATURE_SHA256:
        raise ValueError('formal context requires the CONTENT_V1 digest')
    array = np.load(gd._checked_file(content_path, content_sha), allow_pickle=False)
    ids = np.array([int(x) for x in catalog], dtype=np.uint32)
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype('<f4') or array.ndim != 2 or array.shape[1] != CONTENT_DIM \
            or len(array) != len(ids) + 1 or not np.isfinite(array).all() or np.any(array[0] != 0):
        raise ValueError('content must be a finite float32 [N+1, 64] array with a zero PAD row, one row per catalog item')
    rows = np.ascontiguousarray(array[1:], dtype=np.float32)
    return ids, rows


def _as_depth3(loaded):
    """A verified SIDMAP_V2 load as the depth-3 codec the G6 context requires (same ids/sids/branching; info gains depth 3)."""
    catalog = loaded.codec.catalog
    codec = sid_v2d.build_codec_d(np.array(catalog.ids, dtype=np.uint32), np.array(catalog.sids, dtype=np.uint32), loaded.info['branching'],
                                  V2_DEPTH, catalog_ids=catalog.ids)
    return sid_v2.LoadedSID(codec, loaded.manifest, {**loaded.info, 'depth': V2_DEPTH})


def _load_sid(sid_manifest, sid_sha, content_path, content_sha, catalog, engineering,
              input_manifest, input_sha, vectors_path, vectors_sha,
              expected_content_digest=coll.CONTENT_V1_FEATURE_SHA256):
    array = np.load(gd._checked_file(content_path, content_sha), allow_pickle=False)
    ids = np.array([int(x) for x in catalog], dtype=np.uint32)
    if array.dtype != np.float32 or array.shape != (len(ids)+1,64) or not np.isfinite(array).all() or np.any(array[0]):
        raise ValueError('finite CONTENT_V1 with PAD required')
    from .sid_tok_audit import feature_digest
    digest=feature_digest(array[1:])
    if not engineering and digest != coll.CONTENT_V1_FEATURE_SHA256: raise ValueError('H5a CONTENT_V1 mismatch')
    im=gd._json(gd._checked_file(input_manifest,input_sha))
    if im['content_feature_sha256']!=digest: raise ValueError('actual content provenance mismatch')
    vectors=np.load(gd._checked_file(vectors_path,vectors_sha),allow_pickle=False)
    return coll.load_bundle(sid_manifest,sid_sha,catalog_ids=ids,vectors=vectors,input_manifest=im,engineering=engineering)



def _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha, config,
              temporary, *, input_manifest, input_sha, vectors_path, vectors_sha,
              expected_content_digest=sid_v2d.CONTENT_V1_FEATURE_SHA256):
    temporary = Path(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    base = gr.load_context(coll_base._data_root(root), prepared_path, prepared_sha, temporary)
    engineering = base.binding.get('engineering') is True and base.binding.get('synthetic') is True
    validate_config(config, engineering)
    check_pins(sid_sha, input_sha, vectors_sha, engineering, config=config)
    loaded = history_v2.load_prepared(history_manifest, history_sha, HISTORY_LENGTH)
    sid = _load_sid(sid_manifest, sid_sha, content_path, content_sha, base.catalog, engineering,
        input_manifest, input_sha, vectors_path, vectors_sha, expected_content_digest)
    if sid.info['sid_version'] != SID_VERSION or sid.info['branching'] != 48 or sid.info['vocab_size'] != 48*3+sid.info['suffix_vocabulary']+2:
        raise ValueError('frozen collaborative codec identity/shape required')
    sid = _as_depth3(sid)
    return build_g6_context(base, loaded, HISTORY_LENGTH, sid), engineering, sid


def _binding(root, context, config, runtime, model, prepared_sha, history_sha, sid):
    info = sid.info
    return {'task': TASK_ID, 'budget': BUDGET, 'init_version': 'INIT_EMB_N002', 'config': config, 'config_sha256': _digest(config),
            'context_binding_sha256': _digest(context.binding), 'context_b1': context.binding['b1'], 'source': _source(root),
            'data_root': str(coll_base._data_root(root)), 'runtime': runtime, 'init_report': model.init_report, 'prepared_manifest_sha256': prepared_sha, 'history_manifest_sha256': history_sha,
            'sid': {'sid_version': info['sid_version'], 'manifest_sha256': info['manifest_sha256'],
                    'content_feature_sha256': sid.manifest['content_feature_sha256'], 'branching': info['branching'], 'depth': info['depth'],
                    'suffix_vocabulary': info['suffix_vocabulary'], 'vocab_size': info['vocab_size'], 'kind': info['kind'],
                    'engineering': info['engineering'], 'input_manifest_sha256': sid.manifest['input_manifest_sha256'],
                    'input_feature_sha256': sid.manifest['input_feature_sha256'], 'input_dimension': sid.manifest['input_dimension']},
            'synthetic': context.binding['synthetic'], 'test_gate': 'closed'}


def run(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha, config, output, *,
        temporary, eval_workers=1, progress=None, decode=None, input_manifest=None, input_sha=None, vectors_path=None, vectors_sha=None, expected_content_digest=sid_v2d.CONTENT_V1_FEATURE_SHA256):
    root = Path(root)
    _gate(decode, eval_workers)
    evaluate = _evaluator_d(decode)
    context, engineering, sid = _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path,
                                          content_sha, config, temporary, input_manifest=input_manifest, input_sha=input_sha, vectors_path=vectors_path, vectors_sha=vectors_sha, expected_content_digest=expected_content_digest)
    runtime = _runtime(engineering)
    device = runtime['device']
    output = gd._no_symlink(Path(output))
    output.mkdir(mode=0o700, exist_ok=False)
    gs.check_progress_outside(progress, output)
    guard = _guard(engineering, output, device)
    gr._seed(config['seed'])
    model = make_model_arch_d(context.codec, config, device=device)
    optimizer = ga.make_optimizer_arch(model, config)
    tokens = sd.SIDTokensLD(context.codec, config['history_len'])
    cursor = gt.EpochCursor(config['seed'], len(context.examples))
    stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    binding = _binding(root, context, config, runtime, model, prepared_sha, history_sha, sid)
    losses = []
    epochs = []
    best = None
    last = None
    wall = {'train': 0., 'evaluate': 0.}
    progress_at = [None]

    def report(stage, completed=None, total=None, force=False):
        now = time.monotonic()
        if progress is None or not force and progress_at[0] is not None and now - progress_at[0] < 30:
            return
        progress_at[0] = now
        gs.write_progress(progress, gs.progress_record(progress, stage, 'formal', cursor.epoch, config['max_epochs'], completed, total))
    while True:
        if cursor.phase == 'epoch_done':
            if stop.stopped:
                break
            cursor.advance_epoch()
        begin = time.perf_counter()
        while cursor.phase == 'train':
            guard()
            indices = cursor.next_batch(512)
            result = train_batch_d(model, optimizer, context.examples.rows(indices), context.codec, tokens, config, device)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            losses.append({'epoch': cursor.epoch, 'examples': result.examples, 'loss_mean': result.loss_mean})
            report('train')
        wall['train'] += time.perf_counter() - begin
        epoch = cursor.epoch
        guard()
        state = {'schema': CHECKPOINT_SCHEMA, 'binding_sha256': _digest(binding), 'epoch': epoch, 'model': dict(model.state_dict()),
                 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(), 'selection': stop.state(), 'rng': gt._rng(model)}
        checkpoint_sha = _save(output / 'last.pt', state)
        model_sha = ev.model_digest(model)
        begin = time.perf_counter()
        report('evaluate', force=True)
        kind, evaluation = evaluate(model, context, tokens, config, runtime, output / f'evaluation-{epoch:03d}', checkpoint_sha, guard, eval_workers,
                                    on_shard=lambda completed, total: report('evaluate', completed, total))
        wall['evaluate'] += time.perf_counter() - begin
        aggregate = evaluation['aggregate']
        main_metrics = aggregate['metrics']['all_eligible']
        improved = stop.observe(epoch, main_metrics['recall@50'], main_metrics['ndcg@50'])
        cursor.complete_evaluation()
        last = {'path': 'last.pt', 'sha256': checkpoint_sha, 'epoch': epoch}
        if improved:
            shutil.copyfile(output / 'last.pt', output / 'best.pt.partial')
            (output / 'best.pt.partial').replace(output / 'best.pt')
            if file_sha256(output / 'best.pt') != checkpoint_sha:
                raise ValueError('best checkpoint copy mismatch')
            best = {'path': 'best.pt', 'sha256': checkpoint_sha, 'epoch': epoch, 'metric': list(stop.best_metric)}
        epochs.append({'epoch': epoch, 'metrics': aggregate, 'model_sha256': model_sha, 'checkpoint_sha256': checkpoint_sha,
                       'evaluation': {'kind': kind, 'path': f'evaluation-{epoch:03d}', 'manifest_sha256': evaluation['manifest_sha256'],
                                      'computation_sha256': evaluation['manifest']['computation_sha256']},
                       'observations': evaluation['manifest'].get('observations')})
        value = {'schema': MANIFEST_SCHEMA, 'kind': MANIFEST_KIND, 'status': 'complete' if stop.stopped else 'running', 'config': config,
                 'binding': binding, 'selection': stop.state(), 'epochs': epochs, 'losses': losses, 'best': best, 'last': last,
                 'wall_seconds': dict(wall), 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience,
                 'eval_workers': eval_workers, 'decode': decode}
        sha = _write_manifest(output / 'manifest.json', value)
        report('complete' if stop.stopped else 'train', force=True)
    return {**value, 'manifest_path': output / 'manifest.json', 'manifest_sha256': sha}


def _read_run(path, pin):
    value = gd._json(gd._checked_file(Path(path), pin))
    keys = {'schema', 'kind', 'status', 'config', 'binding', 'selection', 'epochs', 'losses', 'best', 'last', 'wall_seconds', 'hit_max_epochs',
            'eval_workers', 'decode'}
    if type(value) is not dict or set(value) != keys or value['schema'] != MANIFEST_SCHEMA or value['kind'] != MANIFEST_KIND or value['status'] != 'complete':
        raise ValueError('complete G6 train manifest required')
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
    gr._seed(config['seed'])
    model = make_model_arch_d(context.codec, config, device=runtime['device'])
    model.load_state_dict(state['model'])
    return model


def verify(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha, manifest_path, manifest_sha,
           output, *, temporary, eval_workers=1, decode=None, input_manifest=None, input_sha=None, vectors_path=None, vectors_sha=None, expected_content_digest=sid_v2d.CONTENT_V1_FEATURE_SHA256):
    root = Path(root)
    manifest_path = Path(manifest_path)
    run_dir = manifest_path.parent
    _gate(decode, eval_workers)
    evaluate = _evaluator_d(decode)
    value = _read_run(manifest_path, manifest_sha)
    config = value['config']
    context, engineering, sid = _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path,
                                          content_sha, config, temporary, input_manifest=input_manifest, input_sha=input_sha, vectors_path=vectors_path, vectors_sha=vectors_sha, expected_content_digest=expected_content_digest)
    runtime = _runtime(engineering)
    gr._seed(config['seed'])
    fresh = make_model_arch_d(context.codec, config, device=runtime['device'])
    binding = _binding(root, context, config, runtime, fresh, prepared_sha, history_sha, sid)
    _check_binding(value['binding'], binding)
    tokens = sd.SIDTokensLD(context.codec, config['history_len'])
    stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
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
    output = gd._no_symlink(Path(output))
    output.mkdir(mode=0o700, exist_ok=False)
    _, replay = evaluate(model, context, tokens, config, runtime, output / f"replay-best-{best['epoch']:03d}", best['sha256'],
                         _guard(engineering, output, runtime['device']), eval_workers)
    if replay['manifest']['computation_sha256'] != expected_best['evaluation']['computation_sha256']:
        raise ValueError('best checkpoint re-evaluation mismatch')
    return {'status': 'verified', 'replayed_epochs': [best['epoch']], 'best_epoch': best['epoch'], 'epochs': len(value['epochs']), 'run_id':run_dir.parent.name,'manifest_sha256':manifest_sha,'verify_run_id':os.environ.get('VKGR_RUN_ID')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'train', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--prepared', required=True)
        p.add_argument('--prepared-sha', required=True)
        p.add_argument('--history-manifest', required=True)
        p.add_argument('--history-sha', required=True)
        p.add_argument('--sid-manifest', required=True)
        p.add_argument('--sid-sha', required=True)
        p.add_argument('--content', required=True)
        p.add_argument('--content-sha', required=True)
        for extra in ('input-manifest', 'input-sha', 'vectors', 'vectors-sha'):
            p.add_argument('--'+extra, required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--scratch', required=True)
        if name in ('preflight','train'):
            p.add_argument('--config', required=True)
            p.add_argument('--config-sha', required=True)
        else:
            p.add_argument('--manifest', required=True)
            p.add_argument('--manifest-sha', required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        decode = gb.decode_mode_from_env()
        workers = gs.eval_workers_from_env()
        progress = gs.progress_target_from_env()
        _gate(decode, workers)
        common = (root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha,
                  Path(args.sid_manifest), args.sid_sha, Path(args.content), args.content_sha)
        extra = dict(input_manifest=Path(args.input_manifest), input_sha=args.input_sha, vectors_path=Path(args.vectors), vectors_sha=args.vectors_sha)
        if args.command in ('preflight','train'):
            config = gd._json(gd._checked_file(Path(args.config), args.config_sha))
            if args.command=='preflight':
                context,engineering,sid=_contexts(*common,config,Path(args.scratch),**extra)
                if engineering:raise ValueError('formal real-input preflight required')
                print(json.dumps({'status':'preflight_ok','engineering':False,'written':False,'seed':config['seed'],'query_count':len(context.queries),'catalog_size':len(context.catalog),'sid':sid.info,'source_sha256':_source(root)['source_sha256']}));return 0
            value = run(*common, config, Path(args.output), temporary=Path(args.scratch), eval_workers=workers, progress=progress, decode=decode, **extra)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']}))
        else:
            print(json.dumps(verify(*common, Path(args.manifest), args.manifest_sha, Path(args.output), temporary=Path(args.scratch),
                                    eval_workers=workers, decode=decode, **extra)))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G6 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
