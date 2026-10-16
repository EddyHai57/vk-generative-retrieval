"""R2 own SFT -> fixed train mining -> DPO+CE -> own verify, with no old checkpoint input."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import torch
from . import ranking_r2 as r
from . import ranking_r2_config as contract
from . import ranking_r2_engine as engine
from . import ranking_r2_sft as sft
from .ranking_r2_data import data_root
from .sid_depth import SIDTokensLD
from .generative_experiment import _write_json

SCHEMA = 'g-ranking-r2-manifest-v1'
KIND = 'g_ranking_r2_train'
CHECKPOINT_SCHEMA = 'g-ranking-r2-dpo-checkpoint-v1'
SOURCE_MEMBERS = sft.G6_MEMBERS


def _control(path, sha, config):
    value = sft.gd._json(sft.gd._checked_file(Path(path), sha))
    if (value.get('schema'), value.get('kind'), value.get('status')) != ('g-g6-train-manifest-v1', 'g_g6_train', 'complete') or value.get('config') != config['sft']:
        raise ValueError('pinned same-seed original A1/T3 control manifest metadata required; no control checkpoint is read')
    matches = [e for e in value['epochs'] if e['epoch'] == value['best']['epoch']]
    if len(matches) != 1:
        raise ValueError('unique original control best epoch required')
    metrics = matches[0]['metrics']['metrics']['all_eligible']
    if any(type(metrics[k]) not in (int, float) or not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in ('recall@50', 'ndcg@50')):
        raise ValueError('finite original control metrics required')
    return metrics


def input_binding(root, config, context, runtime, control_sha):
    return {'task': contract.TASK_ID, 'config': config, 'config_sha256': engine.digest(config), 'seed': config['seed'],
            'source': sft._source(root), 'data_root': str(data_root(root)), 'runtime': runtime,
            'context_binding_sha256': engine.digest(context.binding), 'control_manifest_sha256': control_sha, 'test_gate': 'closed'}


def _binding(root, config, context, runtime, sft_sha, sft_value, mining_sha, reference_sha, control_sha):
    return dict(input_binding(root, config, context, runtime, control_sha), sft_manifest_sha256=sft_sha,
                sft_best_checkpoint_sha256=sft_value['best']['sha256'], mining_sha256=mining_sha, reference_model_sha256=reference_sha)


def _own_sft(common, directory, value, context, runtime, config):
    # This is reachable only after this very run's SFT verifier has passed.
    stage = contract.sft_config(config)
    if value.get('config') != stage:
        raise ValueError('own SFT config/seed differs from this R2 run')
    return sft._load_best(directory, value, context, stage, runtime, value['binding'])


def _record(kind, result, checkpoint_sha, model, epoch, directory):
    return {'epoch': epoch, 'checkpoint_sha256': checkpoint_sha, 'model_sha256': sft.ev.model_digest(model), 'metrics': result['aggregate'],
            'evaluation': {'kind': kind, 'path': directory, 'manifest_sha256': result['manifest_sha256'],
                           'computation_sha256': result['manifest']['computation_sha256']}}


def _guard(engineering, output, device, deadline):
    original = sft._guard(engineering, output, device)
    def guard():
        if time.monotonic() >= deadline:
            raise ValueError('R2 finite 48h wall deadline')
        original()
    return guard


def run(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha,
        config, output, *, control_manifest, control_sha, engineering_indices=None, progress=None):
    contract.validate(config, engineering=True)  # unresolved scientific fields stop before any training/data read
    started = time.monotonic()
    last_progress = [0.]
    def report(stage_name, epoch=0, completed=None, total=None, force=False):
        now = time.monotonic()
        if progress is not None and (force or now-last_progress[0] >= 30):
            last_progress[0] = now
            sft.gs.write_progress(progress, sft.gs.progress_record(progress, stage_name, 'formal', epoch, 5, completed, total))
    common = (Path(root), Path(prepared_path), prepared_sha, Path(history_manifest), history_sha, Path(sid_manifest), sid_sha, Path(content_path), content_sha)
    output = sft.gd._no_symlink(Path(output))
    output.mkdir(mode=0o700, exist_ok=False)
    stage = contract.sft_config(config)
    context, engineering, sid = sft._contexts(*common, stage, output/'context-scratch')
    contract.validate(config, engineering)
    if not engineering and engineering_indices is not None:
        raise ValueError('formal R2 has no mining sample override')
    control = _control(control_manifest, control_sha, config)
    runtime = sft._runtime(engineering)
    decode = {'mode': 'batched', 'batch_size': 1}
    sft._gate(decode, 1)
    sft_value = sft.run(*common, stage, output/'sft', temporary=output/'sft-scratch', decode=decode, progress=progress)
    sft_sha = sft_value['manifest_sha256']
    report('r2-sft-verify', force=True)
    sft_verification = sft.verify(*common, output/'sft/manifest.json', sft_sha, output/'sft-verification', temporary=output/'sft-verify-scratch', decode=decode)
    if sft_verification['status'] != 'verified':
        raise ValueError('own SFT must directly verify before mining')
    actor = _own_sft(common, output/'sft', sft_value, context, runtime, config)
    reference = r.frozen_reference(actor)
    reference_sha = sft.ev.model_digest(reference)
    tokens = SIDTokensLD(context.codec, 50)
    guard = _guard(engineering, output, runtime['device'], started+48*3600)
    rows = context.examples.rows(list(range(len(context.examples))))
    indices = engineering_indices if engineering else None
    if indices is None:
        indices = r.sample_training_indices(len(rows))
    report('r2-mining', force=True)
    mined = engine.mine_rows(actor, tokens, rows, indices, engineering=engineering, guard=guard, on_progress=lambda done, total: report('r2-mining', completed=done, total=total))
    mined.update(sft_best_checkpoint_sha256=sft_value['best']['sha256'], sft_model_sha256=reference_sha)
    (output/'mining').mkdir()
    mining_sha = _write_json(output/'mining/records.json', mined)
    pairs = engine.validate_mining(mined, rows, tokens, engineering=engineering)
    if not pairs:
        raise ValueError('all fixed sampled examples skipped; no DPO data, no resampling')
    binding = _binding(root, config, context, runtime, sft_sha, sft_value, mining_sha, reference_sha, control_sha)
    dpo_dir = output/'dpo'
    dpo_dir.mkdir()
    evaluate = sft._evaluator_d(decode)
    report('r2-epoch0', force=True)
    kind, epoch0_eval = evaluate(actor, context, tokens, stage, runtime, dpo_dir/'epoch0-evaluation', sft_value['best']['sha256'], guard, 1)
    epoch0 = _record(kind, epoch0_eval, sft_value['best']['sha256'], actor, 0, 'epoch0-evaluation')
    main0 = epoch0['metrics']['metrics']['all_eligible']
    epoch0['control_delta'] = {k: main0[k]-control[k] for k in ('recall@50', 'ndcg@50')}
    epoch0['control_metrics'] = control
    epoch0['comparison_is_gate'] = False
    optimizer = engine.make_optimizer(actor, config)
    cursor = sft.gt.EpochCursor(config['seed'], len(pairs))
    stop = engine.dpo_stop(config)
    epochs, losses = [], []
    best, last = None, None
    wall = {'dpo_train': 0., 'dpo_evaluate': 0.}
    while not stop.stopped:
        if cursor.phase == 'epoch_done':
            cursor.advance_epoch()
        begin = time.perf_counter()
        while cursor.phase == 'train':
            guard()
            report('r2-dpo-train', epoch=cursor.epoch)
            batch_indices = cursor.next_batch(512)
            result = engine.train_step(actor, reference, optimizer, [pairs[i] for i in batch_indices], tokens,
                                       microbatch=config['dpo_microbatch'], device=runtime['device'])
            losses.append({'epoch': cursor.epoch, 'examples': result.examples, 'loss_mean': result.loss_mean})
        wall['dpo_train'] += time.perf_counter()-begin
        if sft.ev.model_digest(reference) != reference_sha:
            raise ValueError('frozen reference changed during DPO')
        epoch = cursor.epoch
        state = {'schema': CHECKPOINT_SCHEMA, 'binding_sha256': engine.digest(binding), 'epoch': epoch,
                 'model': dict(actor.state_dict()), 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(),
                 'selection': stop.state(), 'rng': sft.gt._rng(actor), 'reference_model_sha256': reference_sha}
        checkpoint_sha = sft._save(dpo_dir/'last.pt', state)
        begin = time.perf_counter()
        report('r2-dpo-evaluate', epoch=epoch, force=True)
        kind, evaluated = evaluate(actor, context, tokens, stage, runtime, dpo_dir/f'evaluation-{epoch:03d}', checkpoint_sha, guard, 1)
        wall['dpo_evaluate'] += time.perf_counter()-begin
        record = _record(kind, evaluated, checkpoint_sha, actor, epoch, f'evaluation-{epoch:03d}')
        epochs.append(record)
        metrics = record['metrics']['metrics']['all_eligible']
        improved = stop.observe(epoch, metrics['recall@50'], metrics['ndcg@50'])
        cursor.complete_evaluation()
        last = {'path': 'last.pt', 'sha256': checkpoint_sha, 'epoch': epoch}
        if improved:
            shutil.copyfile(dpo_dir/'last.pt', dpo_dir/'best.pt.partial')
            (dpo_dir/'best.pt.partial').replace(dpo_dir/'best.pt')
            if sft.file_sha256(dpo_dir/'best.pt') != checkpoint_sha:
                raise ValueError('DPO best copy mismatch')
            best = {'path': 'best.pt', 'sha256': checkpoint_sha, 'epoch': epoch, 'metric': list(stop.best_metric)}
        value = {'schema': SCHEMA, 'kind': KIND, 'status': 'complete' if stop.stopped else 'running', 'config': config, 'binding': binding,
                 'sft': {'path': 'sft/manifest.json', 'sha256': sft_sha, 'best': sft_value['best'], 'verify': sft_verification},
                 'mining': {'path': 'mining/records.json', 'sha256': mining_sha, 'sample_count': len(indices), 'skipped': mined['skipped'], 'skip_fraction': mined['skip_fraction']},
                 'epoch0': epoch0, 'dpo': {'selection': stop.state(), 'epochs': epochs, 'losses': losses, 'best': best, 'last': last,
                                         'reference_model_sha256': reference_sha, 'wall_seconds': wall}, 'wall_seconds': time.monotonic()-started}
        manifest_sha = sft._write_manifest(output/'manifest.json', value)
    report('complete', epoch=stop.epoch, force=True)
    return dict(value, manifest_sha256=manifest_sha)


def verify(root, prepared_path, prepared_sha, history_manifest, history_sha, sid_manifest, sid_sha, content_path, content_sha,
           manifest_path, manifest_sha, output, *, control_manifest, control_sha):
    value = sft.gd._json(sft.gd._checked_file(Path(manifest_path), manifest_sha))
    if (value.get('schema'), value.get('kind'), value.get('status')) != (SCHEMA, KIND, 'complete'):
        raise ValueError('complete R2 manifest required')
    config = value['config']
    contract.validate(config, engineering=True)
    run_dir = Path(manifest_path).parent
    output = sft.gd._no_symlink(Path(output))
    output.mkdir(mode=0o700, exist_ok=False)
    common = (Path(root), Path(prepared_path), prepared_sha, Path(history_manifest), history_sha, Path(sid_manifest), sid_sha, Path(content_path), content_sha)
    stage = contract.sft_config(config)
    context, engineering, sid = sft._contexts(*common, stage, output/'context-scratch')
    contract.validate(config, engineering)
    control = _control(control_manifest, control_sha, config)
    runtime = sft._runtime(engineering)
    if value['sft']['path'] != 'sft/manifest.json' or value['mining']['path'] != 'mining/records.json':
        raise ValueError('fixed R2 stage paths required')
    sft_value = sft._read_run(run_dir/'sft/manifest.json', value['sft']['sha256'])
    if sft_value.get('config') != stage:
        raise ValueError('own SFT config/seed differs before stage checkpoint verification')
    verified_sft = sft.verify(*common, run_dir/'sft/manifest.json', value['sft']['sha256'], output/'sft', temporary=output/'sft-scratch', decode={'mode': 'batched', 'batch_size': 1})
    if verified_sft['status'] != 'verified' or value['sft']['verify']['status'] != 'verified' or value['sft']['best'] != sft_value['best']:
        raise ValueError('own SFT verification/best binding mismatch')
    actor = _own_sft(common, run_dir/'sft', sft_value, context, runtime, config)
    reference_sha = sft.ev.model_digest(actor)
    tokens = SIDTokensLD(context.codec, 50)
    mined = sft.gd._json(sft.gd._checked_file(run_dir/'mining/records.json', value['mining']['sha256']))
    rows = context.examples.rows(list(range(len(context.examples))))
    pairs = engine.validate_mining(mined, rows, tokens, engineering=engineering)
    if not pairs or mined.get('sft_best_checkpoint_sha256') != sft_value['best']['sha256'] or mined.get('sft_model_sha256') != reference_sha:
        raise ValueError('mining source is not this run SFT best')
    if any(value['mining'][k] != mined[k] for k in ('sample_count', 'skipped', 'skip_fraction')):
        raise ValueError('mining skip report differs')
    fresh = _binding(root, config, context, runtime, value['sft']['sha256'], sft_value, value['mining']['sha256'], reference_sha, control_sha)
    sft._check_binding(value['binding'], fresh)
    dpo = value['dpo']
    if dpo['reference_model_sha256'] != reference_sha:
        raise ValueError('reference differs from own SFT best')
    dpo_dir = run_dir/'dpo'
    epoch0 = value['epoch0']
    aggregate, computation, _ = sft._read_evaluation(epoch0['evaluation']['kind'], dpo_dir, epoch0, context, tokens)
    expected_delta = {k: aggregate['metrics']['all_eligible'][k]-control[k] for k in ('recall@50', 'ndcg@50')}
    if (engine.digest(aggregate) != engine.digest(epoch0['metrics']) or computation != epoch0['evaluation']['computation_sha256']
        or epoch0['model_sha256'] != reference_sha or epoch0['checkpoint_sha256'] != sft_value['best']['sha256']):
        raise ValueError('epoch0 must reproduce this run SFT best model and published rows')
    if epoch0['epoch'] != 0 or epoch0['control_delta'] != expected_delta or epoch0['comparison_is_gate'] is not False or epoch0['control_metrics'] != control:
        raise ValueError('epoch0 descriptive comparison mismatch')
    stop = engine.dpo_stop(config)
    expected_best = None
    for epoch, record in enumerate(dpo['epochs'], 1):
        if record['epoch'] != epoch:
            raise ValueError('DPO epoch sequence; epoch0 cannot be a candidate')
        aggregate, computation, _ = sft._read_evaluation(record['evaluation']['kind'], dpo_dir, record, context, tokens)
        if engine.digest(aggregate) != engine.digest(record['metrics']) or computation != record['evaluation']['computation_sha256']:
            raise ValueError('DPO epoch aggregate/computation mismatch')
        metrics = aggregate['metrics']['all_eligible']
        if stop.observe(epoch, metrics['recall@50'], metrics['ndcg@50']):
            expected_best = record
    if not stop.stopped or stop.state() != dpo['selection'] or expected_best is None:
        raise ValueError('DPO selection replay mismatch')
    best = dpo['best']
    if best != {'path': 'best.pt', 'sha256': expected_best['checkpoint_sha256'], 'epoch': expected_best['epoch'], 'metric': list(stop.best_metric)}:
        raise ValueError('DPO best selection mismatch')
    last = dpo['last']
    if last != {'path': 'last.pt', 'sha256': dpo['epochs'][-1]['checkpoint_sha256'], 'epoch': len(dpo['epochs'])} or sft.file_sha256(dpo_dir/'last.pt') != last['sha256']:
        raise ValueError('DPO last checkpoint mismatch')
    if sorted(p.name for p in dpo_dir.glob('*.pt')) != ['best.pt', 'last.pt'] or sft.file_sha256(dpo_dir/'best.pt') != best['sha256']:
        raise ValueError('DPO best+last hash/policy mismatch')
    state = torch.load(dpo_dir/'best.pt', map_location='cpu', weights_only=True)
    if state.get('schema') != CHECKPOINT_SCHEMA or state.get('epoch') != best['epoch'] or state.get('binding_sha256') != engine.digest(fresh) or state.get('reference_model_sha256') != reference_sha:
        raise ValueError('DPO own checkpoint identity mismatch')
    actor.load_state_dict(state['model'])
    if sft.ev.model_digest(actor) != expected_best['model_sha256']:
        raise ValueError('DPO best model digest mismatch')
    guard = _guard(engineering, output, runtime['device'], time.monotonic()+48*3600)
    evaluate = sft._evaluator_d({'mode': 'batched', 'batch_size': 1})
    _, replay = evaluate(actor, context, tokens, stage, runtime, output/'replay-dpo-best', best['sha256'], guard, 1)
    if replay['manifest']['computation_sha256'] != expected_best['evaluation']['computation_sha256']:
        raise ValueError('DPO best re-evaluation mismatch')
    return {'status': 'verified', 'sft_status': 'verified', 'dpo_best_epoch': best['epoch'], 'skipped': mined['skipped'],
            'skip_fraction': mined['skip_fraction'], 'mining_verification': 'pinned beam records, sample and highest-eligible legality; no second full mining pass'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'verify'):
        p = sub.add_parser(name)
        for key in ('prepared', 'history-manifest', 'sid-manifest', 'content', 'control-manifest'):
            p.add_argument('--'+key, required=True)
        for key in ('prepared-sha', 'history-sha', 'sid-sha', 'content-sha', 'control-sha'):
            p.add_argument('--'+key, required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--scratch', required=True, help='reserved new-run scratch role; stages allocate only inside output')
        if name == 'train':
            p.add_argument('--config', required=True);p.add_argument('--config-sha', required=True)
        else:
            p.add_argument('--manifest', required=True);p.add_argument('--manifest-sha', required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        sft._gate(sft.gb.decode_mode_from_env(), sft.gs.eval_workers_from_env())
        common = (root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha, Path(args.sid_manifest), args.sid_sha, Path(args.content), args.content_sha)
        if args.command == 'train':
            config = sft.gd._json(sft.gd._checked_file(Path(args.config), args.config_sha))
            value = run(*common, config, Path(args.output), control_manifest=Path(args.control_manifest), control_sha=args.control_sha,
                        progress=sft.gs.progress_target_from_env())
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'skipped': value['mining']['skipped'],
                              'skip_fraction': value['mining']['skip_fraction'], 'dpo_best_epoch': value['dpo']['best']['epoch']}))
        else:
            print(json.dumps(verify(*common, Path(args.manifest), args.manifest_sha, Path(args.output), control_manifest=Path(args.control_manifest), control_sha=args.control_sha)))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'R2 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
