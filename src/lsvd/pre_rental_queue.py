"""Run an approved two-tower plan serially and recover from durable checkpoints."""

import argparse
from contextlib import redirect_stdout
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import time
import traceback

from .pre_rental_deploy import code_manifest, sha256_file, verify_manifest, verify_results
from .pre_rental_probe import write_receipt


def load_plan(path, device):
    from .two_tower_run import STRUCTURES

    plan = json.loads(path.read_text())['runs']
    run_ids = set()
    if not isinstance(plan, list) or not plan:
        raise ValueError('plan must contain a nonempty runs list')
    for run in plan:
        generator = run.get('kind') == 'generator'
        if generator:
            validate_generator_entry(run)
        elif set(run) != {'id', 'structure', 'seed', 'id_min_count', 'precision', 'epochs', 'patience'}:
            raise ValueError('each run requires id, structure, seed, id_min_count, precision, epochs and patience')
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', run['id']) or run['id'] in run_ids:
            raise ValueError('run IDs must be unique and contain only ASCII letters, digits, dots, dashes and underscores')
        run_ids.add(run['id'])
        if (not generator and run['structure'] not in STRUCTURES) or run['precision'] not in ('fp32', 'tf32', 'bf16'):
            raise ValueError('unknown tower structure or precision')
        if device == 'cpu' and run['precision'] != 'fp32':
            raise ValueError('CPU runs require fp32')
        numeric = [('seed', 0), ('epochs', 1), ('patience', 1)]
        if not generator:
            numeric.append(('id_min_count', 1))
        for name, minimum in numeric:
            if type(run[name]) is not int or run[name] < minimum:
                raise ValueError(f'invalid {name} in run {run["id"]}')
    return plan


def validate_generator_entry(run):
    from .generator_decode import DecodeConfig

    required = {'id', 'kind', 'seed', 'epochs', 'patience', 'precision', 'history_mode',
                'shared_head', 'sid_bundle', 'sid_sha256', 'item_vectors', 'item_vectors_sha256'}
    external = {'external_checkpoint', 'external_checkpoint_sha256'}
    optional = {'batch_size', 'decode', 'early_beam_widths'} | external
    if required - set(run) or set(run) - required - optional:
        raise ValueError('generator run has missing or unknown fields')
    if run['history_mode'] not in ('event', 'tiger', 'event_no_feedback') or type(run['shared_head']) is not bool:
        raise ValueError('invalid generator history mode or shared head')
    if run['shared_head'] and external & set(run) or not run['shared_head'] and external - set(run):
        raise ValueError('shared head off requires an external checkpoint and SHA256')
    path_fields = ['sid_bundle', 'item_vectors'] + (['external_checkpoint'] if not run['shared_head'] else [])
    for name in path_fields:
        if not isinstance(run[name], str) or not run[name]:
            raise ValueError(f'invalid {name}')
    for name in ['sid_sha256', 'item_vectors_sha256'] + (['external_checkpoint_sha256'] if not run['shared_head'] else []):
        if not re.fullmatch('[0-9a-f]{64}', run[name]):
            raise ValueError(f'invalid {name}')
    batch = run.get('batch_size', 1024)
    if type(batch) is not int or batch < 2:
        raise ValueError('generator batch size must be at least two')
    settings = DecodeConfig(**run.get('decode', {}))
    if settings.ranking == 'beam_product':
        raise ValueError('the generator queue produces five routes; use beam_product on full_sid only')
    early = DecodeConfig(k=200, beam_widths=run.get('early_beam_widths', (256, 256, 256)))
    if (settings.k == 1000 and settings.beam_widths[-1] < 1024) or early.beam_widths[-1] < 256:
        raise ValueError('generator plan needs final width >=1024 at K=1000 and >=256 for early stop')


def latest_checkpoint(run_root):
    from .two_tower_checkpoint import CheckpointManager

    for attempt in sorted(run_root.glob('attempt-*'), reverse=True):
        directory = attempt / 'checkpoints'
        if (directory / 'checkpoints.json').is_file():
            path = CheckpointManager(directory, create=False).latest
            if path is not None:
                if not path.is_file():
                    raise FileNotFoundError('published latest checkpoint is missing')
                return path
    return None


class Progress:
    """The clock counts active queue wall time across launches, including evaluation."""

    def __init__(self, root, device, limit_hours):
        self.path = root / 'status.json'
        self.device = device
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        if limit_hours is None:
            limit_hours = previous.get('gpu_hours_limit')
        self.limit_seconds = None if limit_hours is None else limit_hours * 3600
        self.previous_seconds = previous.get('elapsed_seconds', 0.)
        self.started = time.monotonic()
        self.detail = dict(status='running', current_run=None, epoch=None, step=None,
                           remaining_training_seconds=None)
        self.stop = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat)

    def elapsed(self):
        return self.previous_seconds + time.monotonic() - self.started

    def stop_reason(self):
        if self.stop.is_set():
            return 'signal'
        if self.limit_seconds is not None and self.elapsed() >= self.limit_seconds:
            return 'gpu_hours'
        return None

    def write(self):
        elapsed = self.elapsed()
        write_receipt(self.path, dict(self.detail, elapsed_seconds=elapsed, device=self.device,
                                    used_gpu_hours=elapsed / 3600 if self.device == 'cuda' else None,
                                    gpu_hours_limit=(self.limit_seconds / 3600
                                                     if self.limit_seconds is not None else None)))

    def heartbeat(self):
        while not self.done.wait(1):
            self.write()

    def update_session(self, session, phase):
        recent = session.records[-100:]
        seconds = sum(row['seconds'] for row in recent)
        rate = sum(row['targets'] for row in recent) / seconds if seconds else None
        remaining = max(0, (session.config.epochs - session.epoch) * session.epoch_targets
                        - session.cursor)
        self.detail = dict(self.detail, phase=phase, epoch=session.epoch + 1,
                           step=session.global_step, recent_targets_per_second=rate,
                           remaining_training_seconds=remaining / rate if rate else None)
        return self.stop_reason() is not None


def queue_status(root):
    if not (root / 'status.json').is_file():
        return dict(status='not_started')
    with (root / 'queue.lock').open('rb') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            active = True
        else:
            active = False
        result = json.loads((root / 'status.json').read_text())
        if not active and result['status'] == 'running':
            result['status'] = 'interrupted'
            if result.get('current_run'):
                run = root / 'runs' / result['current_run']
                checkpoint = latest_checkpoint(run)
                if checkpoint is not None:
                    match = re.search(r'-epoch-(\d+)-step-(\d+)-', checkpoint.name)
                    if match is not None:
                        result.update(epoch=int(match[1]), step=int(match[2]),
                                      durable_checkpoint=checkpoint.relative_to(run).as_posix())
        return result


def write_step_log(output, records):
    with (output / 'training_steps.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def publish_result(run_root, attempt, result, cold, identity, stop_requested, *, generator=False):
    bundle = attempt / 'result-bundle'
    bundle.mkdir()
    best = (attempt / result['best_checkpoint']).resolve()
    if attempt.resolve() not in best.parents:
        raise ValueError('best checkpoint must belong to this attempt')
    os.link(best, bundle / 'best.pt')
    if not generator:
        shutil.copytree(attempt / 'best-cold-vectors', bundle / 'best-cold-vectors', copy_function=os.link)
    for name in (('evaluation',) if generator else ('evaluation', 'cold-evaluation')):
        suffixes = ('.csv', '.json', '.npz') if generator else ('.csv', '.json')
        shutil.copytree(attempt / name, bundle / name,
                        copy_function=os.link if generator else shutil.copy2,
                        ignore=lambda _, entries: [entry for entry in entries
                            if '.' in entry and not entry.endswith(suffixes)])
    logs = bundle / 'logs'
    logs.mkdir()
    shutil.copy2(attempt / 'training_steps.csv', logs / 'training_steps.csv')
    for path in sorted(run_root.glob('attempt-*.log')):
        shutil.copy2(path, logs / path.name)
    write_receipt(bundle / 'run_summary.json',
                  dict(result, **identity, best_checkpoint='best.pt', cold_vectors=cold))
    with (bundle / 'SHA256SUMS').open('x') as handle:
        for path in sorted(bundle.rglob('*')):
            if stop_requested():
                return False
            if path.is_file() and path.name != 'SHA256SUMS':
                handle.write(f'{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n')
        handle.flush()
        os.fsync(handle.fileno())
    if stop_requested():
        return False
    bundle.rename(run_root / 'results')
    return True


def train_generator_run(args, run, run_root, attempt, progress, identity):
    from .generator_data import GeneratorData, ItemVectors, SIDBundle
    from .generator_decode import DecodeConfig
    from .generator_training import config_from_plan, external_query, fit_generator

    checkpoint = latest_checkpoint(run_root)
    progress.detail = dict(progress.detail, current_run=run['id'], phase='loading',
                           epoch=None, step=None, remaining_training_seconds=None,
                           recent_targets_per_second=None, attempt=attempt.name,
                           resumed_from=checkpoint.relative_to(run_root).as_posix() if checkpoint else None)
    sid = SIDBundle(run['sid_bundle'], expected_sha256=run['sid_sha256'])
    vectors = ItemVectors(run['item_vectors'], expected_sha256=run['item_vectors_sha256'])
    dataset = GeneratorData(args.dataset, sid, vectors, device=args.device,
                            expected_tensor_sha256=identity['data_sha256'])
    config = config_from_plan(run, sid, vectors)
    scorer = None
    if not config.shared_head:
        scorer = external_query(args.dataset, run['external_checkpoint'], args.device,
                                checkpoint_sha256=run['external_checkpoint_sha256'],
                                vectors=vectors)

    def control(session, phase):
        stop = progress.update_session(session, phase)
        if not stop and phase == 'final_evaluation':
            write_step_log(attempt, session.records)
        return stop

    result = fit_generator(dataset, config, args.device, attempt,
        identity=dict(data_sha256=identity['data_sha256'], sid_sha256=sid.sha256,
                      item_vectors_sha256=vectors.sha256, plan_sha256=identity['plan_sha256'],
                      external_checkpoint_sha256=run.get('external_checkpoint_sha256')),
        precision=run['precision'], resume=checkpoint, checkpoint_every=args.checkpoint_every,
        external_scorer=scorer, control=control, decode=DecodeConfig(**run.get('decode', {})),
        early_beam_widths=run.get('early_beam_widths', (256, 256, 256)))
    if result['status'] == 'paused' or progress.stop_reason() is not None:
        return 'paused', None
    result['effective_data_placement'] = 'cpu_memmap'
    return result, None


def train_run(args, run, run_root, attempt, progress, identity):
    import torch

    from .two_tower import TowerConfig, TwoTower
    from .two_tower_prepare import load_tensor_dataset
    from .two_tower_retrieval import export_cold_embeddings
    from .two_tower_run import STRUCTURES
    from .two_tower_training import fit_tensor

    config = TowerConfig(batch_size=1024, seed=run['seed'], id_min_count=run['id_min_count'],
                         epochs=run['epochs'], patience=run['patience'])
    checkpoint = latest_checkpoint(run_root)
    progress.detail = dict(progress.detail, current_run=run['id'], phase='loading',
                           epoch=None, step=None, remaining_training_seconds=None,
                           recent_targets_per_second=None, attempt=attempt.name,
                           resumed_from=checkpoint.relative_to(run_root).as_posix() if checkpoint else None)
    # fit_tensor creates its own attempt directory.
    _, catalog, store, data = load_tensor_dataset(args.dataset, config,
        storage_device=args.data_placement, pinned=args.device == 'cuda' and args.data_placement == 'cpu')

    def control(session, phase):
        stop = progress.update_session(session, phase)
        if not stop and phase == 'final_evaluation':
            write_step_log(attempt, session.records)
        return stop

    result = fit_tensor(catalog, store, data, config, run['structure'], args.device, attempt,
                        0, 1000, 8, max_train_seconds=math.inf, precision=run['precision'],
                        resume=checkpoint, checkpoint_every=args.checkpoint_every,
                        data_identity=identity['data_sha256'], control=control)
    if result.get('status') == 'paused' or progress.stop_reason() is not None:
        return 'paused', None
    progress.detail = dict(progress.detail, phase='exporting', remaining_training_seconds=0.)
    saved = torch.load(attempt / result['best_checkpoint'], map_location='cpu', weights_only=True)
    user_structure, _ = STRUCTURES[run['structure']]
    model = TwoTower(catalog, TowerConfig(**saved['config']), user_structure).to(args.device)
    model.load_state_dict(saved['model'])
    del saved
    cold = export_cold_embeddings(model, catalog, args.device, attempt / 'best-cold-vectors',
                                  run['structure'], stop_requested=progress.stop_reason)
    if cold is None or progress.stop_reason() is not None:
        return 'paused', None
    return result, cold


def run_queue(args):
    import torch

    from .two_tower_runtime import configure_threads

    if args.device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError('the queue requires exactly one visible CUDA card')
    if args.device == 'cpu' and (args.data_placement == 'cuda' or args.gpu_hours is not None):
        raise ValueError('CUDA data and GPU-hour limits require a CUDA run')
    if args.gpu_hours is not None and (not math.isfinite(args.gpu_hours) or args.gpu_hours <= 0):
        raise ValueError('GPU-hour limit must be finite and positive')
    if args.checkpoint_every < 1:
        raise ValueError('checkpoint interval must be positive')
    plan = load_plan(args.plan, args.device)
    limits = configure_threads(args.threads)
    source = code_manifest(Path(__file__).resolve().parents[2])
    identity = dict(plan_sha256=sha256_file(args.plan), data_sha256=sha256_file(args.dataset / 'manifest.json'),
                    source_sha256=source['source_sha256'], git_commit=source['git_commit'],
                    device=args.device, data_placement=args.data_placement,
                    checkpoint_every=args.checkpoint_every, threads=limits['configured_threads'])
    root = args.queue
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = root / 'queue-plan.json'
        if binding.is_file():
            if json.loads(binding.read_text()) != identity:
                raise ValueError('queue plan, code, data or execution configuration differs')
        else:
            write_receipt(binding, identity)
        progress = Progress(root, args.device, args.gpu_hours)
        handlers = {sig: signal.signal(sig, lambda *_: progress.stop.set())
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        progress.write()
        progress.thread.start()
        try:
            progress.detail = dict(progress.detail, phase='verifying_dataset')
            verify_manifest(args.dataset, args.dataset / 'manifest.json', identity['data_sha256'])
            for run in plan:
                generator = run.get('kind') == 'generator'
                run_root = root / 'runs' / run['id']
                if (run_root / 'results').exists():
                    verify_results(run_root / 'results', formal=True, generator=generator)
                    summary = json.loads((run_root / 'results/run_summary.json').read_text())
                    if any(summary.get(key) != value
                           for key, value in dict(identity, run=run).items()):
                        raise ValueError('completed results belong to another run or queue')
                    print('skip completed', run['id'], flush=True)
                    continue
                if progress.stop_reason() is not None:
                    break
                run_root.mkdir(exist_ok=True, parents=True)
                index = len(list(run_root.glob('attempt-*.log'))) + 1
                attempt = run_root / f'attempt-{index:06d}'
                print('start', run['id'], attempt.name, flush=True)
                path = run_root / (attempt.name + '.log')
                with path.open('x', buffering=1) as log, redirect_stdout(log):
                    try:
                        execute = train_generator_run if generator else train_run
                        result, cold = execute(args, run, run_root, attempt, progress, identity)
                    except Exception:
                        traceback.print_exc(file=log)
                        raise
                if result == 'paused':
                    progress.detail = dict(progress.detail, status='paused', reason=progress.stop_reason())
                    break
                progress.detail = dict(progress.detail, phase='publishing')
                if not publish_result(run_root, attempt, result, cold, dict(identity, run=run),
                                      progress.stop_reason, generator=generator):
                    progress.detail = dict(progress.detail, status='paused', reason=progress.stop_reason())
                    break
                print('complete', run['id'], flush=True)
            else:
                progress.detail = dict(status='complete', current_run=None, epoch=None, step=None,
                                       remaining_training_seconds=0.)
            if progress.detail['status'] == 'running':
                progress.detail = dict(progress.detail, status='paused', reason=progress.stop_reason())
        except Exception:
            progress.detail = dict(progress.detail, status='failed')
            raise
        finally:
            progress.done.set()
            progress.thread.join()
            progress.write()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    print(json.dumps(queue_status(root)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--dataset', type=Path, required=True)
    run.add_argument('--queue', type=Path, required=True)
    run.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run.add_argument('--data-placement', choices=['cpu', 'cuda'], default='cpu')
    run.add_argument('--threads', type=int)
    run.add_argument('--checkpoint-every', type=int, default=1000)
    run.add_argument('--gpu-hours', type=float)
    status = sub.add_parser('status')
    status.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(json.dumps(queue_status(args.queue), indent=2))
        return 0
    return run_queue(args)


if __name__ == '__main__':
    raise SystemExit(main())
