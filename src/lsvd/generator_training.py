"""Generator training and week25 evaluation on the frozen tensor inputs."""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .generator_decode import DecodeConfig, GeneratorRanker, PrefixTrie
from .generator_eval import ROUTES, early_recall, evaluate_predictions, predict, report_ks, supply_summary
from .generator_model import GeneratorConfig, GeneratorModel
from .pre_rental_deploy import sha256_file, verify_manifest
from .pre_rental_probe import write_receipt
from .two_tower import user_hash
from .two_tower_checkpoint import CheckpointManager
from .two_tower_data import encode_batch, subset_evaluation
from .two_tower_runtime import configure_threads, precision_mode
from .two_tower_tensors import disk_shuffle


class GeneratorSession:
    def __init__(self, dataset, config, device, output, *, identity, precision='fp32',
                 positions=None, resume=None):
        self.dataset, self.config, self.device = dataset, config, device
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.precision, self.identity = precision, identity
        self.positions = dataset.main_positions if positions is None else positions
        self.epoch_targets = len(self.positions)
        if not self.epoch_targets:
            raise ValueError('no main-ruler training targets')
        random.seed(config.seed)
        np.random.seed(config.seed % (2**32))
        torch.manual_seed(config.seed)
        self.model = GeneratorModel(config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.manager = CheckpointManager(self.output / 'checkpoints')
        self.rng = np.random.default_rng(config.seed)
        self.order = self.shuffle_before = None
        self.epoch = self.cursor = self.global_step = 0
        self.records = []
        self.progress = dict(best=-float('inf'), bad=0, epochs=[], validation_sid_loss=[])
        self.best_payload = None
        if resume is not None:
            saved = self.manager.restore(resume, self.model, self.optimizer)
            state = saved['training_state']
            if (state['identity'] != identity or saved['config'] != asdict(config)
                    or state['precision'] != precision or state['epoch_targets'] != self.epoch_targets):
                raise ValueError('generator checkpoint configuration or input identity differs')
            self.epoch, self.global_step = saved['epoch'] - 1, saved['step']
            self.cursor = saved['shuffle']['cursor']
            self.shuffle_before = saved['shuffle']['before']
            self.rng.bit_generator.state = saved['shuffle']['after']
            self.records, self.progress = state['records'], state['progress']
            self.best_payload = state['best_payload']
            if self.best_payload is None and saved['is_best']:
                self.best_payload = torch.from_numpy(np.memmap(resume, dtype=np.uint8, mode='c'))
            if self.best_payload is not None:
                self.manager.import_best(self.best_payload)

    def _ensure_order(self):
        if self.cursor == self.epoch_targets:
            self.epoch += 1
            self.cursor = 0
            self.order = self.shuffle_before = None
        if self.order is None:
            if self.shuffle_before is None:
                self.shuffle_before = copy.deepcopy(self.rng.bit_generator.state)
            else:
                self.rng.bit_generator.state = self.shuffle_before
            self.order = disk_shuffle(self.output / f'epoch-{self.epoch + 1}-order.npy',
                                      self.epoch_targets, self.rng)

    def step(self):
        started = time.perf_counter()
        self._ensure_order()
        selected = self.order[self.cursor:self.cursor + self.config.batch_size]
        batch = self.dataset.training_batch(self.positions[selected])
        prepared = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with precision_mode(self.device, self.precision):
            total, sid, dense = self.model.loss(*batch)
        loss, sid_loss, dense_loss = map(float, (total.detach(), sid.detach(), dense.detach()))
        if not np.isfinite([loss, sid_loss, dense_loss]).all():
            raise ValueError('nonfinite generator loss')
        total.backward()
        self.optimizer.step()
        self.cursor += len(selected)
        self.global_step += 1
        row = dict(epoch=self.epoch + 1, step=self.global_step, loss=loss,
                   sid_loss=sid_loss, dense_loss=dense_loss, targets=len(selected),
                   assembly_seconds=prepared - started, seconds=time.perf_counter() - started,
                   epoch_finished=self.cursor == self.epoch_targets)
        self.records.append(row)
        return row

    def save(self, *, is_best=False):
        if self.shuffle_before is None:
            self._ensure_order()
        state = dict(identity=self.identity, precision=self.precision, epoch_targets=self.epoch_targets,
                     records=self.records, progress=self.progress,
                     best_payload=None if is_best else self.best_payload)
        saved = self.manager.save(self.model, self.optimizer, epoch=self.epoch + 1,
                                  step=self.global_step, config=self.config, structure='generator',
                                  is_best=is_best, training_state=state,
                                  shuffle=dict(before=self.shuffle_before,
                                               after=self.rng.bit_generator.state, cursor=self.cursor))
        if is_best:
            self.best_payload = torch.from_numpy(np.memmap(saved, dtype=np.uint8, mode='c'))
        return saved


def external_query(dataset_path, checkpoint, device, *, checkpoint_sha256=None,
                   vectors=None, **binding):
    from .generator_data import load_external_tower

    if checkpoint is None or (checkpoint_sha256 is not None and
                              sha256_file(checkpoint) != checkpoint_sha256):
        raise ValueError('external two-tower checkpoint SHA256 differs or is missing')
    if (vectors is not None and
            vectors.manifest['checkpoint']['checkpoint_sha256'] != checkpoint_sha256):
        raise ValueError('external query and frozen vectors must share the best checkpoint')
    model, catalog, store = load_external_tower(dataset_path, checkpoint, device, **binding)
    model.eval()

    def encode(user_ids):
        batch = store.query_batch(user_ids, model.config, device)
        with torch.no_grad(), precision_mode(device, 'fp32'):
            return encode_batch(model, catalog, batch, device)
    return encode


def fixed_hash_users(data, *, limit=0):
    return np.sort(np.asarray(sorted(data.user_ids, key=user_hash)[:limit]
                              if limit else data.user_ids))


def positions_for_users(dataset, users):
    store = dataset.store
    users = np.asarray(users)
    user_rows = np.searchsorted(store.user_ids, users)
    if not len(store.user_ids):
        return dataset.main_positions[:0]
    bounded = np.minimum(user_rows, len(store.user_ids) - 1)
    user_rows = user_rows[(user_rows < len(store.user_ids)) & (store.user_ids[bounded] == users)]
    boundaries = np.searchsorted(dataset.main_positions,
                                 np.stack((store.offsets[user_rows], store.offsets[user_rows + 1]), 1))
    return np.concatenate([dataset.main_positions[a:b] for a, b in boundaries]) if len(boundaries) else dataset.main_positions[:0]


def make_ranker(dataset, settings, *, ties=None):
    return GeneratorRanker(dataset.sid, dataset.vectors.full, dataset.catalog.columns['train_count'],
                           dataset.arrays['catalog_target_count'], settings, ties=ties)


def config_from_plan(run, sid, vectors):
    return GeneratorConfig(size=sid.size, vector_dim=vectors.full.shape[1],
                           history_mode=run['history_mode'], shared_head=run['shared_head'],
                           seed=run['seed'], epochs=run['epochs'], patience=run['patience'],
                           batch_size=run.get('batch_size', GeneratorConfig.batch_size))


def validation_examples(data, limit=256):
    examples = []
    offsets = data.targets['offsets']
    for row in sorted(range(len(data.user_ids)), key=lambda i: user_hash(data.user_ids[i])):
        a, b = offsets[row:row + 2]
        keep = (data.targets['main_positive'][a:b] != 0) & (data.targets['place'][a:b] <= 1)
        matches = data.targets['item_id'][a:b][keep]
        if len(matches):
            examples.append((data.user_ids[row], matches[0]))
            if len(examples) == limit:
                break
    return examples


def validation_loss(session, examples, batch_size):
    model, dataset, device = session.model, session.dataset, session.device
    model.eval()
    total = 0.
    with torch.no_grad(), precision_mode(device, session.precision):
        for start in range(0, len(examples), batch_size):
            events = examples[start:start + batch_size]
            users, items = map(np.asarray, zip(*events))
            history = dataset.query_batch(users)
            rows = np.searchsorted(dataset.sid.ids, items)
            codes = torch.as_tensor(dataset.sid.codes[rows].astype(np.int64), device=device)
            total += float(model.sid_loss(history, codes)) * len(events)
    return total / len(examples) if examples else None


def fit_generator(dataset, config, device, output, *, identity, precision='fp32',
                  positions=None, eval_data=None, decode=None, resume=None,
                  checkpoint_every=1000, sid_eval_every=1000, external_scorer=None,
                  control=None, max_steps=0, query_batch_size=8, replicates=1000,
                  early_beam_widths=(256, 256, 256)):
    output = Path(output)
    decode = decode or DecodeConfig()
    if decode.k == 1000 and decode.beam_widths[-1] < 1024:
        raise ValueError('formal K=1000 requires final beam width >=1024')
    if eval_data is None:
        eval_data = dataset.evaluation
    early_users = np.asarray([u for u in eval_data.user_ids
                              if user_hash(u, config.user_hash_seed) % 10 == 0])
    if not len(early_users):
        raise ValueError('fixed early-stop hash cohort is empty')
    early = subset_evaluation(eval_data, early_users)
    examples = validation_examples(early)
    early_decode = replace(decode, k=200, beam_widths=early_beam_widths,
                           candidate_budget=max(200, decode.candidate_budget),
                           ranking='item', popularity_weight=.02,
                           new_item_quota=None if decode.new_item_quota is None
                           else min(200, decode.new_item_quota))
    if early_decode.beam_widths[-1] < 256:
        raise ValueError('early-stop K=200 requires final beam width >=256')
    early_ranker = make_ranker(dataset, early_decode)
    trie = PrefixTrie(dataset.sid.prefixes, dataset.sid.size, device)
    session = GeneratorSession(dataset, config, device, output, identity=identity,
                               precision=precision, positions=positions, resume=resume)
    def check(phase):
        return control is not None and control(session, phase)

    def pause(saved=None):
        path = saved if saved is not None else session.save()
        return dict(status='paused', steps=session.global_step, checkpoint=path.name)

    def record_sid_validation():
        records = session.progress['validation_sid_loss']
        if not records or records[-1]['step'] != session.global_step:
            records.append(dict(step=session.global_step, targets=len(examples),
                                loss=validation_loss(session, examples, query_batch_size)))

    def finish_epoch():
        record_sid_validation()
        supply = {}
        score = early_recall(session.model, dataset, early, early_ranker, early_decode,
                             device, precision, external_query=external_scorer, trie=trie,
                             query_batch_size=query_batch_size, supply=supply)
        improved = score is not None and score > session.progress['best']
        if improved:
            session.progress.update(best=score, bad=0)
        else:
            session.progress['bad'] += 1
        session.progress['epochs'].append(dict(epoch=session.epoch + 1, recall_at_200=score,
                                                beam_widths=early_decode.beam_widths, supply=supply))
        return session.save(is_best=improved)

    if check('training'):
        return pause()
    if (session.cursor == session.epoch_targets
            and len(session.progress['epochs']) <= session.epoch):
        saved = finish_epoch()
        if check('training'):
            return pause(saved)
    while len(session.progress['epochs']) < config.epochs and session.progress['bad'] < config.patience:
        if max_steps and session.global_step >= max_steps:
            break
        row = session.step()
        if check('training'):
            return pause()
        if row['epoch_finished']:
            if check('epoch_evaluation'):
                return pause()
            saved = finish_epoch()
            if check('training'):
                return pause(saved)
        else:
            if sid_eval_every and session.global_step % sid_eval_every == 0:
                record_sid_validation()
            if checkpoint_every and session.global_step % checkpoint_every == 0:
                session.save()
    latest = session.save()
    if check('final_evaluation'):
        return pause(latest)
    record_sid_validation()
    best = session.manager.best
    if best is None and not max_steps:
        raise ValueError('no epoch produced a valid early-stop score')
    checkpoint = best or latest
    weights = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    session.model.load_state_dict(weights['model'])
    del weights
    ranker = make_ranker(dataset, decode, ties=early_ranker.ties)
    routes = ('full_sid',) if decode.ranking == 'beam_product' else ROUTES
    rankings, counts = predict(session.model, dataset, eval_data, ranker, decode, device, precision,
                               external_query=external_scorer, trie=trie,
                               query_batch_size=query_batch_size, routes=routes)
    evaluate_predictions(eval_data, rankings, dataset.catalog.groups, replicates=replicates, k=decode.k,
                         output=output / 'evaluation')
    result = dict(status='complete', best_checkpoint=str(checkpoint.relative_to(output)),
                  checkpoint_source='early_stop_best' if best else 'latest_smoke_step',
                  steps=session.global_step, epochs=session.progress['epochs'],
                  validation_sid_loss=session.progress['validation_sid_loss'],
                  decode=asdict(decode), candidates=counts, queries=len(eval_data.user_ids),
                  supply=supply_summary(rankings, report_ks(decode.k)),
                  config=asdict(config), precision=precision, identity=identity)
    write_receipt(output / 'generator_summary.json', result)
    return result


def main(argv=None):
    from .generator_data import GeneratorData, ItemVectors, SIDBundle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sid', type=Path, required=True)
    parser.add_argument('--sid-sha256', required=True)
    parser.add_argument('--item-vectors', type=Path, required=True)
    parser.add_argument('--item-vectors-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--precision', choices=('fp32', 'tf32', 'bf16'), default='fp32')
    parser.add_argument('--history-mode', choices=('event', 'tiger', 'event_no_feedback'), default='event')
    parser.add_argument('--two-tower-checkpoint', type=Path)
    parser.add_argument('--two-tower-sha256')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--smoke-users', type=int, default=0)
    parser.add_argument('--beam-widths', type=int, nargs=3, default=(1024, 1024, 1024))
    parser.add_argument('--early-beam-widths', type=int, nargs=3, default=(256, 256, 256))
    parser.add_argument('--prefix-beam-width', type=int, default=64)
    parser.add_argument('--ranking-workers', type=int, default=4)
    parser.add_argument('--prefix-limit', type=int, default=256)
    parser.add_argument('--candidate-budget', type=int, default=2000)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--new-item-quota', type=int)
    parser.add_argument('--ranking', choices=('item', 'beam_product'), default='item')
    parser.add_argument('--popularity-weight', type=float, default=.02)
    parser.add_argument('--ann-nlist', type=int, default=4096)
    parser.add_argument('--ann-nprobe', type=int, default=64)
    parser.add_argument('--query-batch-size', type=int, default=8)
    parser.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args(argv)
    if bool(args.two_tower_checkpoint) != bool(args.two_tower_sha256):
        parser.error('external scoring requires checkpoint and SHA256')
    configure_threads(args.threads)
    manifest = args.dataset / 'manifest.json'
    verify_manifest(args.dataset, manifest)
    sid = SIDBundle(args.sid, expected_sha256=args.sid_sha256)
    vectors = ItemVectors(args.item_vectors, expected_sha256=args.item_vectors_sha256)
    dataset = GeneratorData(args.dataset, sid, vectors, device=args.device)
    config = GeneratorConfig(size=sid.size, vector_dim=vectors.full.shape[1],
                             shared_head=args.two_tower_checkpoint is None,
                             history_mode=args.history_mode, batch_size=args.batch_size,
                             seed=args.seed, epochs=args.epochs)
    settings = DecodeConfig(beam_widths=args.beam_widths, prefix_beam_width=args.prefix_beam_width,
                            prefix_limit=args.prefix_limit, ranking_workers=args.ranking_workers,
                            candidate_budget=args.candidate_budget, k=args.k,
                            new_item_quota=args.new_item_quota, ranking=args.ranking,
                            popularity_weight=args.popularity_weight,
                            ann_nlist=args.ann_nlist, ann_nprobe=args.ann_nprobe)
    scorer = (external_query(args.dataset, args.two_tower_checkpoint, args.device,
                             checkpoint_sha256=args.two_tower_sha256, vectors=vectors)
              if args.two_tower_checkpoint else None)
    users = fixed_hash_users(dataset.evaluation, limit=args.smoke_users)
    subset = subset_evaluation(dataset.evaluation, users) if args.smoke_users else dataset.evaluation
    positions = positions_for_users(dataset, users) if args.smoke_users else None
    identity = dict(data_sha256=sha256_file(manifest),
                    sid_sha256=sid.sha256, item_vectors_sha256=vectors.sha256,
                    external_checkpoint_sha256=args.two_tower_sha256,
                    users_sha256=hashlib.sha256(users.tobytes()).hexdigest(),
                    decode=asdict(settings), early_beam_widths=list(args.early_beam_widths))
    result = fit_generator(dataset, config, args.device, args.output, identity=identity,
                           precision=args.precision, positions=positions, eval_data=subset,
                           decode=settings, resume=args.resume, max_steps=args.max_steps,
                           external_scorer=scorer, query_batch_size=args.query_batch_size,
                           replicates=args.replicates, early_beam_widths=args.early_beam_widths,
                           sid_eval_every=min(50, args.max_steps) if args.max_steps else 1000)
    print(json.dumps({key: result.get(key) for key in ('status', 'steps', 'queries',
                                                       'checkpoint_source')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
