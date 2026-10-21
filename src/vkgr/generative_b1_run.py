"""Batch1 G runner: HIST-V2 / HIST-LEN / G4 (random SID control) + beam-width diagnostic.

New version only: the frozen G3 trajectory (generative_real) keeps max_epochs<=30 and all its
contracts. This runner reuses the verified G3 prepared context (gr.load_context), overlays the
shared HIST_V2 sidecar, and trains with explicit switches:
  init=small (INIT_EMB_N002), history_len=20|50|100, max_epochs=100/min_epochs=5/patience=3,
  sid=SID_BHKM_B16D3_S42 | SID_RANDPERM42_OF_BHKM_B16D3 (G4 only).
Evaluation is the frozen beam=100 evaluator (serial, or INFRA-01 sharded when
VKGR_EVAL_WORKERS>1) for L<=50, and a serial L-window twin for L=100. Only best.pt and
last.pt are kept; verify recomputes every epoch aggregate from published rows, replays the
selection and re-evaluates only the best checkpoint (fail-closed).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from . import generative_data as gd
from . import generative_evaluation as ev
from . import generative_experiment as ge
from . import generative_real as gr
from . import generative_sharded as gs
from . import generative_training as gt
from . import history_v2
from .data_audit import canonical_bytes, file_sha256
from .generative_data_v2 import SIDTokensL, SID_VERSIONS, build_v2_context, sid_targets_l
from .generative_diagnostics import decode_diagnostic, evaluate_l100, permuted_sid_codec, read_l100_evaluation
from .generative_hp import EarlyStop, core_config, make_model_b1

TASKS = ('HIST-V2', 'HIST-LEN', 'G4')
BUDGET = 'BUDGET_E100_MIN5_P3'
B1_MEMBERS = tuple(f'src/vkgr/{n}.py' for n in ('history_v2', 'generative_data_v2', 'generative_hp', 'generative_diagnostics', 'generative_b1_run',
                                                 'generative_sharded', 'generative_real', 'generative_training', 'generative_evaluation',
                                                 'generative_experiment', 'generative_data', 'tiger', 'sid'))


def default_config():
    return {'schema': 'g-b1-run-v1', 'task_id': 'HIST-V2', 'd_model': 128, 'heads': 4, 'encoder_layers': 1, 'decoder_layers': 1, 'ff': 512,
            'dropout': 0.0, 'precision': 'fp32', 'learning_rate': .0003, 'weight_decay': .0001, 'gradient_clip_norm': 1.0,
            'effective_batch': 512, 'microbatch': 512, 'seed': 42, 'init': 'small', 'history_version': 'HIST_V2_OBS', 'history_len': 50,
            'max_epochs': 100, 'min_epochs': 5, 'patience': 3, 'sid': SID_VERSIONS[0], 'beam': 100, 'test_gate': 'closed'}


def validate_config(value, engineering=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered G-B1 config fields')
    fixed = ('schema', 'heads', 'encoder_layers', 'decoder_layers', 'dropout', 'precision', 'weight_decay', 'gradient_clip_norm', 'effective_batch',
             'init', 'history_version', 'patience', 'beam', 'test_gate')
    if any(value[k] != base[k] or type(value[k]) is not type(base[k]) for k in fixed):
        raise ValueError('frozen G-B1 field changed')
    if value['task_id'] not in TASKS or value['sid'] not in SID_VERSIONS or (value['sid'] == SID_VERSIONS[1]) != (value['task_id'] == 'G4'):
        raise ValueError('random SID only for G4')
    if value['history_len'] not in (20, 50, 100) or type(value['history_len']) is not int or value['task_id'] == 'G4' and value['history_len'] != 50:
        raise ValueError('unregistered history length')
    if type(value['seed']) is not int or value['seed'] not in (42, 2024, 2025) or any(type(value[k]) is not int for k in ('d_model', 'ff', 'microbatch', 'max_epochs', 'min_epochs')):
        raise ValueError('typed seed/sizes')
    if engineering:
        if value['d_model'] not in (64, 128) or value['ff'] != 4 * value['d_model'] or value['learning_rate'] not in (.001, .0003) or not 1 <= value['microbatch'] <= 512 or not 1 <= value['max_epochs'] <= 100 or not 0 <= value['min_epochs'] <= value['max_epochs']:
            raise ValueError('bounded engineering config')
    elif (value['d_model'], value['ff'], value['learning_rate'], value['max_epochs'], value['min_epochs']) != (128, 512, .0003, 100, 5) or value['microbatch'] not in (512, 256, 128):
        raise ValueError('formal G-B1 = core03 + BUDGET_E100_MIN5_P3')
    return value


def _digest(value):
    return ge._digest(value)


def _source(root):
    files = {name: file_sha256(Path(root) / name) for name in B1_MEMBERS}
    return {'files': files, 'source_sha256': _digest(files), 'git': gd._git(root)}


def _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, config, temporary):
    temporary = Path(temporary); temporary.mkdir(parents=True, exist_ok=False)
    base = gr.load_context(root, prepared_path, prepared_sha, temporary)
    loaded = history_v2.load_prepared(history_manifest, history_sha, config['history_len'])
    codec = permuted_sid_codec(base.codec) if config['sid'] == SID_VERSIONS[1] else None
    context = build_v2_context(base, loaded, config['history_len'], config['sid'], codec)
    engineering = base.binding.get('engineering') is True and base.binding.get('synthetic') is True
    return context, engineering


def _runtime(engineering):
    if engineering:
        return gr.engineering_runtime()
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; no CPU fallback')
    prop = torch.cuda.get_device_properties(0)
    return gr.configure_runtime({'mode': 'real', 'device': 'cuda:0', 'deterministic_algorithms': True, 'allow_tf32': False, 'environment': gd._environment(),
                                 'cuda_version': torch.version.cuda, 'gpu_name': prop.name, 'gpu_uuid': str(getattr(prop, 'uuid', None)), 'threads': torch.get_num_threads()})


def _guard(engineering, output, device, sample_seconds=30.):
    if engineering:
        return lambda: None
    from .generative_pilot import observe_resources
    last = [None]
    def guard():
        now = time.monotonic()
        if last[0] is None or now - last[0] >= sample_seconds:
            last[0] = now; observe_resources(output, device)
    return guard


def _binding(root, context, config, runtime, model, prepared_sha, history_sha):
    return {'task': 'G-B1', 'budget': BUDGET, 'init_version': 'INIT_EMB_N002', 'config': config, 'config_sha256': _digest(config),
            'context_binding_sha256': _digest(context.binding), 'context_b1': context.binding['b1'], 'source': _source(root),
            'runtime': runtime, 'init_report': model.init_report, 'prepared_manifest_sha256': prepared_sha, 'history_manifest_sha256': history_sha,
            'synthetic': context.binding['synthetic'], 'test_gate': 'closed'}


def train_batch(model, optimizer, rows, context, tokens, config, device):
    """L<=50: the frozen G2 effective-batch step. L=100: same math with the L-window tokenizer."""
    if config['history_len'] <= 50:
        return gt.train_effective_batch(model, optimizer, rows, context.codec, microbatch=config['microbatch'], device=device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    n = len(rows); total = 0.; sizes = []
    try:
        for start in range(0, n, config['microbatch']):
            batch = rows[start:start + config['microbatch']]
            source, decoder, labels = (x.to(device) for x in sid_targets_l(batch, tokens))
            loss = gt.full_vocab_ce(model(source, decoder), labels)
            weighted = loss * (len(batch) / n)
            if not torch.isfinite(weighted):
                raise ValueError('nonfinite loss')
            weighted.backward(); total += float(loss.detach()) * len(batch) / n; sizes.append(len(batch))
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        except RuntimeError as exc:
            raise ValueError('nonfinite gradient') from exc
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite updated parameters')
        return gt.StepResult(total, n, tuple(sizes))
    finally:
        optimizer.zero_grad(set_to_none=True)


def _evaluate(model, context, tokens, config, runtime, output, checkpoint_sha, guard, workers, on_shard=None):
    if config['history_len'] > 50:
        result = evaluate_l100(model, context, tokens, output=output, checkpoint_sha256=checkpoint_sha, guard=guard)
        return 'l100_query_evaluation', result
    if workers == 1:
        return 'query_evaluation', ev.evaluate_queries(model, context, output=output, checkpoint_sha256=checkpoint_sha, shard_size=64, budget_check=guard)
    return 'query_evaluation', gs.evaluate_queries_sharded(model, context, output=output, checkpoint_sha256=checkpoint_sha, workers=workers, config=core_config(config),
                                                           runtime=runtime, shard_size=64, guard=guard, before_shard=guard, before_decode=guard, after_decode=guard, on_shard=on_shard)


def _read_evaluation(kind, run, record, context, tokens):
    path = run / record['evaluation']['path'] / 'manifest.json'
    if Path(record['evaluation']['path']).name != record['evaluation']['path']:
        raise ValueError('evaluation path escape')
    if kind == 'l100_query_evaluation':
        binding = {'context': context.binding, 'checkpoint_sha256': record['checkpoint_sha256'], 'model_sha256': record['model_sha256'], 'beam': 100, 'window': tokens.max_len}
        value, rows = read_l100_evaluation(path, record['evaluation']['manifest_sha256'], binding, context)
        return value['aggregate'], value['computation_sha256'], rows
    binding = {'context': context.binding, 'checkpoint_sha256': record['checkpoint_sha256'], 'model_sha256': record['model_sha256'], 'beam': 100, 'shard_size': 64}
    value, rows = ev._read_evaluation(path, record['evaluation']['manifest_sha256'], binding, context.queries, context.catalog)
    aggregate = ev.aggregate_items([r['items'] for r in rows], context.queries, context.catalog, slice_definition=context.slice_definition)
    if not value['complete'] or canonical_bytes(aggregate) != canonical_bytes(value['aggregate']) or ev._computation(rows, aggregate) != value['computation_sha256']:
        raise ValueError('evaluation aggregate/computation replay')
    return aggregate, value['computation_sha256'], rows


def _save(path, state):
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    return file_sha256(path)


def _write_manifest(path, value):
    tmp = path.with_name(path.name + '.partial')
    data = canonical_bytes(value)
    with tmp.open('wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    return file_sha256(path)


def run(root, prepared_path, prepared_sha, history_manifest, history_sha, config, output, *, temporary, eval_workers=1, progress=None):
    root = Path(root)
    context, engineering = _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, config, temporary)
    validate_config(config, engineering)
    runtime = _runtime(engineering); device = runtime['device']
    output = gd._no_symlink(Path(output)); output.mkdir(mode=0o700, exist_ok=False)
    gs.check_progress_outside(progress, output)
    guard = _guard(engineering, output, device)
    gr._seed(config['seed']); model = make_model_b1(context.codec, config, device=device)
    optimizer = gt.make_optimizer(model, core_config(config)); tokens = SIDTokensL(context.codec, config['history_len'])
    cursor = gt.EpochCursor(config['seed'], len(context.examples)); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    binding = _binding(root, context, config, runtime, model, prepared_sha, history_sha)
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
        state = {'schema': 'g-b1-checkpoint-v1', 'binding_sha256': _digest(binding), 'epoch': epoch, 'model': dict(model.state_dict()),
                 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(), 'selection': stop.state(), 'rng': gt._rng(model)}
        checkpoint_sha = _save(output / 'last.pt', state); model_sha = ev.model_digest(model)
        begin = time.perf_counter(); report('evaluate', force=True)
        kind, evaluation = _evaluate(model, context, tokens, config, runtime, output / f'evaluation-{epoch:03d}', checkpoint_sha, guard, eval_workers,
                                     on_shard=lambda completed, total: report('evaluate', completed, total))
        wall['evaluate'] += time.perf_counter() - begin
        aggregate = evaluation['aggregate']; main = aggregate['metrics']['all_eligible']
        improved = stop.observe(epoch, main['recall@50'], main['ndcg@50']); cursor.complete_evaluation()
        last = {'path': 'last.pt', 'sha256': checkpoint_sha, 'epoch': epoch}
        if improved:
            shutil.copyfile(output / 'last.pt', output / 'best.pt.partial'); os.replace(output / 'best.pt.partial', output / 'best.pt')
            if file_sha256(output / 'best.pt') != checkpoint_sha:
                raise ValueError('best checkpoint copy mismatch')
            best = {'path': 'best.pt', 'sha256': checkpoint_sha, 'epoch': epoch, 'metric': list(stop.best_metric)}
        epochs.append({'epoch': epoch, 'metrics': aggregate, 'model_sha256': model_sha, 'checkpoint_sha256': checkpoint_sha,
                       'evaluation': {'kind': kind, 'path': f'evaluation-{epoch:03d}', 'manifest_sha256': evaluation['manifest_sha256'], 'computation_sha256': evaluation['manifest']['computation_sha256']},
                       'observations': evaluation['manifest'].get('observations')})
        value = {'schema': 'g-b1-train-manifest-v1', 'kind': 'g_b1_train', 'status': 'complete' if stop.stopped else 'running', 'config': config,
                 'binding': binding, 'selection': stop.state(), 'epochs': epochs, 'losses': losses, 'best': best, 'last': last, 'wall_seconds': dict(wall),
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience, 'eval_workers': eval_workers}
        sha = _write_manifest(output / 'manifest.json', value); report('complete' if stop.stopped else 'train', force=True)
    return {**value, 'manifest_path': output / 'manifest.json', 'manifest_sha256': sha}


def _read_run(path, pin):
    value = gd._json(gd._checked_file(Path(path), pin))
    keys = {'schema', 'kind', 'status', 'config', 'binding', 'selection', 'epochs', 'losses', 'best', 'last', 'wall_seconds', 'hit_max_epochs', 'eval_workers'}
    if type(value) is not dict or set(value) != keys or value['schema'] != 'g-b1-train-manifest-v1' or value['kind'] != 'g_b1_train' or value['status'] != 'complete':
        raise ValueError('complete G-B1 train manifest required')
    return value


def _load_best(run, value, context, config, runtime, binding):
    best = value['best']
    if type(best) is not dict or best.get('path') != 'best.pt' or file_sha256(run / 'best.pt') != best.get('sha256'):
        raise ValueError('best checkpoint hash mismatch')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if state.get('schema') != 'g-b1-checkpoint-v1' or state.get('binding_sha256') != _digest(binding) or state.get('epoch') != best['epoch']:
        raise ValueError('best checkpoint identity mismatch')
    gr._seed(config['seed']); model = make_model_b1(context.codec, config, device=runtime['device'])
    model.load_state_dict(state['model'])
    return model


def verify(root, prepared_path, prepared_sha, history_manifest, history_sha, manifest_path, manifest_sha, output, *, temporary, eval_workers=1):
    root = Path(root); manifest_path = Path(manifest_path); run = manifest_path.parent
    value = _read_run(manifest_path, manifest_sha); config = value['config']
    context, engineering = _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, config, temporary)
    validate_config(config, engineering)
    runtime = _runtime(engineering)
    gr._seed(config['seed']); fresh = make_model_b1(context.codec, config, device=runtime['device'])
    binding = _binding(root, context, config, runtime, fresh, prepared_sha, history_sha)
    if canonical_bytes(value['binding']) != canonical_bytes(binding):
        raise ValueError('run input/source/runtime/init binding mismatch')
    tokens = SIDTokensL(context.codec, config['history_len']); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    expected_best = None
    for epoch, record in enumerate(value['epochs'], 1):
        if record.get('epoch') != epoch:
            raise ValueError('epoch sequence')
        aggregate, computation, _ = _read_evaluation(record['evaluation']['kind'], run, record, context, tokens)
        if canonical_bytes(aggregate) != canonical_bytes(record['metrics']) or computation != record['evaluation']['computation_sha256']:
            raise ValueError('recorded epoch metrics not reproducible from published rows')
        main = aggregate['metrics']['all_eligible']
        if stop.observe(epoch, main['recall@50'], main['ndcg@50']):
            expected_best = record
    if not stop.stopped or stop.state() != value['selection'] or expected_best is None:
        raise ValueError('selection replay mismatch')
    best = value['best']
    if best.get('epoch') != expected_best['epoch'] or best.get('sha256') != expected_best['checkpoint_sha256'] or best.get('metric') != list(stop.best_metric):
        raise ValueError('best selection mismatch')
    if type(value['last']) is not dict or value['last'].get('epoch') != len(value['epochs']) or file_sha256(run / 'last.pt') != value['last'].get('sha256') or value['last']['sha256'] != value['epochs'][-1]['checkpoint_sha256']:
        raise ValueError('last checkpoint mismatch')
    if sorted(p.name for p in run.glob('*.pt')) != ['best.pt', 'last.pt']:
        raise ValueError('checkpoint policy best+last violated')
    model = _load_best(run, value, context, config, runtime, binding)
    if ev.model_digest(model) != expected_best['model_sha256']:
        raise ValueError('best model digest mismatch')
    output = gd._no_symlink(Path(output)); output.mkdir(mode=0o700, exist_ok=False)
    _, replay = _evaluate(model, context, tokens, config, runtime, output / f"replay-best-{best['epoch']:03d}", best['sha256'], _guard(engineering, output, runtime['device']), eval_workers)
    if replay['manifest']['computation_sha256'] != expected_best['evaluation']['computation_sha256']:
        raise ValueError('best checkpoint re-evaluation mismatch')
    return {'status': 'verified', 'replayed_epochs': [best['epoch']], 'best_epoch': best['epoch'], 'epochs': len(value['epochs'])}


def beam_diagnostic(root, prepared_path, prepared_sha, history_manifest, history_sha, manifest_path, manifest_sha, beam, output, *, temporary):
    root = Path(root); manifest_path = Path(manifest_path); run = manifest_path.parent
    if beam not in (50, 200):
        raise ValueError('diagnostic beam must be 50 or 200; formal beam stays 100')
    value = _read_run(manifest_path, manifest_sha); config = value['config']
    context, engineering = _contexts(root, prepared_path, prepared_sha, history_manifest, history_sha, config, temporary)
    validate_config(config, engineering); runtime = _runtime(engineering)
    gr._seed(config['seed']); fresh = make_model_b1(context.codec, config, device=runtime['device'])
    binding = _binding(root, context, config, runtime, fresh, prepared_sha, history_sha)
    if canonical_bytes(value['binding']) != canonical_bytes(binding):
        raise ValueError('run input/source/runtime/init binding mismatch')
    model = _load_best(run, value, context, config, runtime, binding)
    record = value['epochs'][value['best']['epoch'] - 1]
    tokens = SIDTokensL(context.codec, config['history_len'])
    _, _, reference = _read_evaluation(record['evaluation']['kind'], run, record, context, tokens)
    diag_binding = {'run_manifest_sha256': manifest_sha, 'best_checkpoint_sha256': value['best']['sha256'], 'model_sha256': ev.model_digest(model),
                    'context_binding_sha256': _digest(context.binding), 'reference_beam': 100, 'kind': 'beam_diagnostic'}
    output = gd._no_symlink(Path(output)); output.parent.mkdir(parents=True, exist_ok=True)
    return decode_diagnostic(model, context, tokens, beam, output=output, binding=diag_binding, reference_rows=reference, guard=_guard(engineering, output.parent, runtime['device']))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Batch1 G runner (new version; frozen G3 untouched); test split closed')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'verify', 'beam-diagnostic'):
        p = sub.add_parser(name)
        p.add_argument('--prepared', required=True); p.add_argument('--prepared-sha', required=True)
        p.add_argument('--history-manifest', required=True); p.add_argument('--history-sha', required=True)
        p.add_argument('--output', required=True); p.add_argument('--scratch', required=True)
        if name == 'train':
            p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True)
        else:
            p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
        if name == 'beam-diagnostic':
            p.add_argument('--beam', type=int, choices=(50, 200), required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        workers = gs.eval_workers_from_env(); progress = gs.progress_target_from_env()
        common = (root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha)
        if args.command == 'train':
            config = gd._json(gd._checked_file(Path(args.config), args.config_sha))
            value = run(*common, config, Path(args.output), temporary=Path(args.scratch), eval_workers=workers, progress=progress)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']}))
        elif args.command == 'verify':
            print(json.dumps(verify(*common, Path(args.manifest), args.manifest_sha, Path(args.output), temporary=Path(args.scratch), eval_workers=workers)))
        else:
            result = beam_diagnostic(*common, Path(args.manifest), args.manifest_sha, args.beam, Path(args.output), temporary=Path(args.scratch))
            print(json.dumps({'status': 'diagnostic_complete', 'manifest_sha256': result['manifest_sha256']}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'G-B1 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
