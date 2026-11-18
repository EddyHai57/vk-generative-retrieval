"""CPU wall-clock regions and CUDA event regions for the same training step."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import time

import torch


class StepTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.host = defaultdict(float)
        self.events = defaultdict(list)
        self.started = time.perf_counter()

    @contextmanager
    def region(self, name):
        if self.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.host[name] += time.perf_counter() - started
            if self.device.type == 'cuda':
                stop.record()
                self.events[name].append((start, stop))

    def finish(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        cuda = {name: sum(start.elapsed_time(stop) / 1000 for start, stop in pairs)
                for name, pairs in self.events.items()}
        return dict(host_seconds=dict(self.host), cuda_seconds=cuda,
                    wall_seconds=time.perf_counter() - self.started)


def region(timer, name):
    return timer.region(name) if timer is not None else nullcontext()


def phase_clock(phase, device, placement):
    if torch.device(device).type != 'cuda' or phase in ['take_batch', 'history']:
        return 'host_seconds'
    if phase == 'mask' and placement == 'cpu':
        return 'host_seconds'
    return 'cuda_seconds'
