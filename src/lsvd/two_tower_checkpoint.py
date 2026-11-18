"""Atomic full-state checkpoints in a newly owned run directory."""

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch


def capture_rng():
    state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=(state[0], torch.from_numpy(state[1].copy()),
                       state[2], state[3], state[4]),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    values = state['numpy']
    np.random.set_state((values[0], values[1].cpu().numpy(),
                         values[2], values[3], values[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class CheckpointManager:
    def __init__(self, root, *, create=True):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            self.registry = dict(counter=0, latest=[], best=None)
        else:
            self.registry = json.loads((self.root / 'checkpoints.json').read_text())

    @property
    def latest(self):
        names = self.registry['latest']
        return self.root / names[-1] if names else None

    @property
    def best(self):
        name = self.registry['best']
        return self.root / name if name else None

    def save(self, model, optimizer, *, epoch, step, shuffle, scheduler=None,
             scaler=None, config=None, structure=None, training_state=None,
             is_best=False):
        number = self.registry['counter'] + 1
        name = (f'checkpoint-{number:06d}-epoch-{epoch:04d}-step-{step:012d}-'
                + uuid.uuid4().hex + '.pt')
        target = self.root / name
        if target.exists():
            raise FileExistsError(target)
        temporary = self.root / (name + '.' + uuid.uuid4().hex + '.tmp')
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict() if scheduler else None,
                     scaler=scaler.state_dict() if scaler is not None else None,
                     rng=capture_rng(), epoch=epoch, step=step, shuffle=shuffle,
                     config=asdict(config) if is_dataclass(config) else config,
                     structure=structure, training_state=training_state or {},
                     model_training=model.training, is_best=is_best)
        with temporary.open('xb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        previous = set(self.registry['latest'])
        if self.registry['best']:
            previous.add(self.registry['best'])
        latest = (self.registry['latest'] + [name])[-2:]
        best = name if is_best else self.registry['best']
        registry = dict(counter=number, latest=latest, best=best)
        _atomic_json(self.root / 'checkpoints.json', registry)
        self.registry = registry
        retained = set(latest) | ({best} if best else set())
        # Only checkpoint filenames previously published by this manager are pruned.
        for old in previous - retained:
            (self.root / old).unlink()
        return target

    def import_best(self, payload):
        name = 'checkpoint-recovered-best-' + uuid.uuid4().hex + '.pt'
        target = self.root / name
        temporary = self.root / (name + '.tmp')
        with temporary.open('xb') as handle:
            payload.cpu().numpy().tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        registry = dict(self.registry, best=name)
        _atomic_json(self.root / 'checkpoints.json', registry)
        self.registry = registry
        return target

    def restore(self, path, model, optimizer, *, scheduler=None, scaler=None,
                map_location='cpu', timings=None):
        started = time.perf_counter()
        state = torch.load(path, map_location=map_location, weights_only=True)
        if timings is not None:
            timings['read_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None and state['scheduler'] is not None:
            scheduler.load_state_dict(state['scheduler'])
        if scaler is not None and state['scaler'] is not None:
            scaler.load_state_dict(state['scaler'])
        model.train(state['model_training'])
        restore_rng(state['rng'])
        if timings is not None:
            timings['restore_seconds'] = time.perf_counter() - started
        return state
