"""Worker-owned CSR session; original query timing boundaries and model remain fixed."""
from dataclasses import dataclass
import hashlib
import time

import torch

from .throughput_decode import TorchSession, _hash_torch_tensor
from .gpu_trie_decode import CSRTrie, decode_history_csr


def timed_decode(model, codec, history, *, beam, trie):
    from analysis import g_decode_timing as original
    from .sid_depth import SIDTokensLD
    from analysis.gpu_trie_decode_probe import trace_digest
    tokens = SIDTokensLD(codec, original.HISTORY)
    trace = []
    cuda = next(model.parameters()).device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = decode_history_csr(model, tokens, history, beam=beam, trie=trie, trace=trace)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {'result': {'ranked': value, 'trace_sha256': trace_digest(trace)}, 'seconds': elapsed,
            'vram_mib': original._SAMPLER.sample() if cuda else 0,
            'vram_samples': original._SAMPLER.samples if cuda else 0,
            'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved() if cuda else 0}


class CSRSession(TorchSession):
    def __init__(self, model, codec, *, beam=100):
        if beam != 100 or type(beam) is not int:
            raise ValueError('registered beam100 required')
        started = time.perf_counter()
        self.trie = CSRTrie(codec.catalog, device=next(model.parameters()).device)
        build = time.perf_counter() - started
        super().__init__(model, codec, timed_decode, beam=beam)
        self.trie_measurement = {'index': 'device-resident layer CSR',
                                 'csr_build_seconds': build, 'csr_storage_bytes': self.trie.storage_bytes}

    def _identity(self):
        from pathlib import Path
        from . import gpu_trie_decode
        from analysis import gpu_trie_decode_probe
        base, size = super()._identity()
        h = hashlib.sha256(base.encode())
        h.update(Path(gpu_trie_decode.__file__).read_bytes())
        h.update(Path(__file__).read_bytes())
        h.update(Path(gpu_trie_decode_probe.__file__).read_bytes())
        aliases = {}
        for i, layer in enumerate(self.trie.layers):
            for j, tensor in enumerate(layer):
                size += _hash_torch_tensor(h, f'csr.{i}.{j}', tensor, aliases)
        h.update(repr((self.trie.bases, str(self.trie.device))).encode())
        return h.hexdigest(), size

    def decode(self, query):
        return timed_decode(self.model, self.codec, query['history'], beam=self.beam, trie=self.trie)


@dataclass(frozen=True)
class SessionFactory:
    root: str
    device: str = 'cuda'

    def __call__(self):
        from analysis.g_decode_timing import SessionFactory as OriginalFactory
        original = OriginalFactory(self.root, device=self.device)()
        result = CSRSession(original.model, original.codec, beam=100)
        result.trie_measurement['original_catalog'] = original.trie_measurement
        return result
