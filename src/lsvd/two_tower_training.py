"""Resumable full-target shuffle using the existing two-tower update."""

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .two_tower import TwoTower
from .two_tower_checkpoint import CheckpointManager
from .two_tower_runtime import precision_mode
from .two_tower_tensors import disk_shuffle


class TensorTrainingSession:
    def __init__(self, catalog, store, config, structure, device, output, *,
                 precision='fp32', resume=None, max_samples=0, data_identity=None):
        from .two_tower_run import STRUCTURES

        if config.packed_window:
            raise ValueError('full-target shuffle requires the unpacked user tower')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.catalog, self.store = catalog, store
        self.structure, self.device, self.precision = structure, device, precision
        user_structure, cross_layers = STRUCTURES[structure]
        self.config = replace(config, cross_layers=cross_layers)
        torch.manual_seed(config.seed)
        self.model = TwoTower(catalog, self.config, user_structure).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.scheduler = None
        self.scaler = None  # bf16 uses FP32 master parameters without a scaler.
        self.manager = CheckpointManager(self.output / 'checkpoints')
        self.order_rng = np.random.default_rng(config.seed)
        self.order = None
        self.shuffle_before = None
        self.epoch = self.global_step = self.cursor = 0
        self.epoch_targets = min(len(store.samples), max_samples or len(store.samples))
        if not self.epoch_targets:
            raise ValueError('no training targets')
        self.records = []
        self.training_state = dict(best=-float('inf'), bad=0, epochs=[], best_checkpoint=None)
        self.data_identity = data_identity
        self.last_batch_users = 0
        self.restore_timings = {}
        if resume is not None:
            saved = self.manager.restore(resume, self.model, self.optimizer,
                                          scheduler=self.scheduler, scaler=self.scaler,
                                          map_location='cpu', timings=self.restore_timings)
            if saved['config'] != asdict(self.config) or saved['structure'] != structure:
                raise ValueError('checkpoint model configuration differs')
            state = saved['training_state']
            if state['data_identity'] != data_identity:
                raise ValueError('checkpoint dataset identity differs')
            if state['epoch_targets'] != self.epoch_targets:
                raise ValueError('checkpoint target budget differs')
            if state.get('precision', 'fp32') != precision:
                raise ValueError('checkpoint precision differs')
            self.epoch = state.get('epoch_index', saved['epoch'])
            self.global_step = saved['step']
            self.cursor = saved['shuffle']['cursor']
            self.shuffle_before = saved['shuffle']['before']
            self.order_rng.bit_generator.state = saved['shuffle']['after']
            self.records = state['records']
            self.training_state = state['progress']
            payload = self.training_state['best_checkpoint']
            if payload is not None:
                self.manager.import_best(payload)
            elif saved.get('is_best'):
                self.training_state['best_checkpoint'] = torch.from_numpy(
                    np.memmap(resume, dtype=np.uint8, mode='c'))
                self.manager.import_best(self.training_state['best_checkpoint'])

    def _ensure_order(self):
        if self.cursor >= self.epoch_targets:
            self.epoch += 1
            self.cursor = 0
            self.order = None
            self.shuffle_before = None
        if self.order is None:
            if self.shuffle_before is None:
                self.shuffle_before = copy.deepcopy(self.order_rng.bit_generator.state)
            else:
                self.order_rng.bit_generator.state = self.shuffle_before
            self.order = disk_shuffle(self.output / f'epoch-{self.epoch + 1}-order.npy',
                                       len(self.store.samples), self.order_rng)

    def step(self, *, profile=False):
        from .two_tower_run import training_step
        from .two_tower_measure import StepTimer

        started = time.perf_counter()
        self._ensure_order()
        selected = self.order[self.cursor:min(self.cursor + self.config.batch_size,
                                              self.epoch_targets)]
        self.model.train()
        timer = StepTimer(self.device) if profile else None
        self.store.timer = self.catalog.timer = timer
        try:
            with precision_mode(self.device, self.precision):
                loss, skipped, _ = training_step(self.model, self.catalog, self.store,
                                                 self.config, self.device,
                                                 self.optimizer, selected, timer=timer)
        finally:
            self.store.timer = self.catalog.timer = None
        timings = timer.finish() if timer is not None else None
        positions = self.store.target_positions[
            torch.as_tensor(selected, device=self.store.storage_device).long()
        ].long()
        self.last_batch_users = int(self.store.tables['user_row'][positions].unique().numel())
        self.cursor += len(selected)
        self.global_step += 1
        row = dict(epoch=self.epoch + 1, step=self.global_step, loss=loss,
                   skipped=skipped, targets=len(selected),
                   distinct_users=self.last_batch_users,
                   seconds=time.perf_counter() - started,
                   epoch_finished=self.cursor == self.epoch_targets)
        if timings is not None:
            row['timings'] = timings
        self.records.append(row)
        return row

    def save(self, *, is_best=False):
        if self.shuffle_before is None:
            self._ensure_order()
        state = dict(data_identity=self.data_identity, epoch_targets=self.epoch_targets,
                     epoch_index=self.epoch, precision=self.precision,
                     records=self.records, progress=self.training_state)
        return self.manager.save(
            self.model, self.optimizer, epoch=self.epoch + 1, step=self.global_step,
            shuffle=dict(seed=self.config.seed, before=self.shuffle_before,
                         after=copy.deepcopy(self.order_rng.bit_generator.state),
                         cursor=self.cursor), scheduler=self.scheduler, scaler=self.scaler,
            config=self.config, structure=self.structure, training_state=state,
            is_best=is_best,
        )

    def register_epoch(self, score, metrics):
        improved = score is not None and score > self.training_state['best']
        self.training_state['epochs'].append(metrics)
        if improved:
            self.training_state['best'] = score
            self.training_state['bad'] = 0
            # Best files have no nested predecessor. Later saves carry its full bytes.
            self.training_state['best_checkpoint'] = None
        else:
            self.training_state['bad'] += 1
        checkpoint = self.save(is_best=improved)
        if improved:
            self.training_state['best_checkpoint'] = torch.from_numpy(
                np.memmap(checkpoint, dtype=np.uint8, mode='c'))
        return checkpoint


def fit_tensor(catalog, store, data, config, structure, device, output, max_samples,
               replicates, exact_users, max_train_seconds=7200, *, precision='fp32',
               resume=None, checkpoint_every=1000, data_identity=None, max_steps=0):
    from .evaluation import evaluate
    from .two_tower import user_hash
    from .two_tower_data import subset_evaluation
    from .two_tower_retrieval import retrieve
    from .two_tower_run import evaluate_model, primary

    session = TensorTrainingSession(catalog, store, config, structure, device, output,
                                     precision=precision, resume=resume,
                                     max_samples=max_samples, data_identity=data_identity)
    early_ids = np.array([u for u in data.user_ids
                          if user_hash(u, config.user_hash_seed) % 10 == 0])
    if not len(early_ids):
        raise ValueError('fixed 10% hash sample is empty')
    early_data = subset_evaluation(data, early_ids)
    def finish_epoch():
        predictions, search = retrieve(session.model, catalog, store, early_data,
                                         device, exact_users=0)
        score = primary(evaluate(early_data, predictions, model_name=structure,
                                  ks=(200,), replicates=1))['macro_capped']
        records = [r for r in session.records if r['epoch'] == session.epoch + 1]
        seconds = sum(r['seconds'] for r in records)
        metrics = dict(epoch=session.epoch + 1, mean_loss=float(np.mean([r['loss'] for r in records])),
                       samples=sum(r['targets'] for r in records),
                       skipped_rows=sum(r['skipped'] for r in records),
                       training_seconds=seconds, samples_per_second=sum(r['targets'] for r in records)/seconds,
                       early_recall200=score, early_users=len(early_ids), retrieval=search)
        session.register_epoch(score, metrics)

    started = time.perf_counter()
    if (session.cursor == session.epoch_targets
            and len(session.training_state['epochs']) <= session.epoch):
        finish_epoch()
    while (len(session.training_state['epochs']) < config.epochs
           and session.training_state['bad'] < config.patience):
        if time.perf_counter() - started >= max_train_seconds:
            break
        row = session.step()
        if row['epoch_finished']:
            finish_epoch()
        elif checkpoint_every and session.global_step % checkpoint_every == 0:
            session.save()
        if max_steps and session.global_step >= max_steps:
            break
    latest = session.save()
    best = session.manager.best
    if best is None:
        record = dict(status='finite_steps_only', steps=session.global_step,
                      checkpoint=str(latest.relative_to(output)),
                      epochs=session.training_state['epochs'])
        (Path(output) / 'finite_steps.json').write_text(json.dumps(record, indent=2))
        return record
    session.model.load_state_dict(torch.load(best, map_location='cpu', weights_only=True)['model'])
    return evaluate_model(session.model, catalog, store, data, structure, device,
                           Path(output), session.training_state['epochs'],
                           best, replicates, exact_users)


def run_tensor_command(args, config):
    from .two_tower_prepare import load_tensor_dataset, sha256_file
    from .two_tower_run import forward_only_preflight, STRUCTURES
    from .two_tower_retrieval import export_cold_embeddings

    if args.data_placement == 'cuda' and args.device != 'cuda':
        raise ValueError('CUDA-resident data requires a CUDA compute device')
    if args.resume is not None and len(args.structures) != 1:
        raise ValueError('a resume checkpoint identifies one structure')
    arrays, catalog, store, data = load_tensor_dataset(
        args.tensor_dataset, config, storage_device=args.data_placement,
        pinned=args.device == 'cuda' and args.data_placement == 'cpu',
    )
    identity = sha256_file(args.tensor_dataset / 'manifest.json')
    manifest = json.loads((args.tensor_dataset / 'manifest.json').read_text())
    record = dict(config=asdict(config), data=manifest['counts'],
                   data_sha256=identity, precision=args.precision,
                   data_placement=args.data_placement)
    (args.output / 'effective_config.json').open('x').write(json.dumps(record, indent=2))
    if args.dry_run:
        with precision_mode(args.device, args.precision):
            forward_only_preflight(catalog, store, data, config, args.structures,
                                    args.device, args.output)
        return 0
    results = []
    for structure in args.structures:
        if args.benchmark_steps:
            session = TensorTrainingSession(catalog, store, config, structure,
                                             args.device, args.output / structure,
                                             precision=args.precision,
                                             data_identity=identity,
                                             max_samples=args.max_train_samples)
            rows = [session.step(profile=True) for _ in range(args.benchmark_steps)]
            result = dict(structure=structure, rows=rows)
        else:
            result = fit_tensor(
                catalog, store, data, config, structure, args.device,
                args.output / structure, args.max_train_samples,
                args.replicates, args.exact_users, args.max_train_seconds,
                precision=args.precision, resume=args.resume,
                checkpoint_every=args.checkpoint_every, data_identity=identity,
                max_steps=args.max_train_steps,
            )
        results.append(result)
    if args.export_best_cold:
        candidates = [row for row in results if row.get('main') is not None]
        if candidates:
            best = max(candidates, key=lambda row: row['main']['macro_capped'])
            path = args.output / best['structure'] / best['best_checkpoint']
            saved = torch.load(path, map_location=args.device, weights_only=True)
            from .two_tower import TowerConfig
            model = TwoTower(catalog, TowerConfig(**saved['config']),
                              STRUCTURES[best['structure']][0]).to(args.device)
            model.load_state_dict(saved['model'])
            export_cold_embeddings(model, catalog, args.device,
                                   args.output / 'best-cold-vectors', best['structure'])
    (args.output / 'tensor_summary.json').open('x').write(
        json.dumps(dict(models=results, data_sha256=identity), indent=2))
    print(json.dumps(dict(status='completed', structures=len(results))))
    return 0
