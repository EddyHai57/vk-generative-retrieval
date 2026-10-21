import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest

import numpy as np
import torch

from analysis.g_decode_timing import balanced_sids
from vkgr.sid import SIDCatalog
from vkgr.sid_depth import SIDTokensD
from vkgr.sid_depth_beam import constrained_beam_d


def fixture(n=71, k=3):
    catalog = SIDCatalog(np.arange(1, n + 1, dtype=np.uint32), balanced_sids(n, k, 3), k, 3, [])
    return catalog, SIDTokensD(catalog)


def original_trace(catalog, codec, score, beam):
    # Observe the unchanged original function at its score boundary and catalog decode.
    traces = []
    previous = None
    active_scores = [0.]
    def observed(paths):
        nonlocal previous, active_scores
        if previous is not None:
            old_paths, old_logits = previous
            expanded = []
            probs = torch.log_softmax(old_logits, -1).detach().cpu()
            for row, (path, total) in enumerate(zip(old_paths, active_scores)):
                for digit in catalog.allowed_next(path):
                    expanded.append((path + (digit,), total + float(probs[row, codec.offsets[len(path)] + digit])))
            expanded.sort(key=lambda x: (-x[1], x[0]))
            selected = expanded[:beam]
            assert paths == [p for p, _ in selected]
            active_scores = [v for _, v in selected]
            traces.append(selected)
        logits = score(paths)
        previous = (paths, logits)
        return logits
    result = constrained_beam_d(catalog, codec, observed, beam)
    paths, logits = previous
    expanded = []
    probs = torch.log_softmax(logits, -1).detach().cpu()
    for row, (path, total) in enumerate(zip(paths, active_scores)):
        for digit in catalog.allowed_next(path):
            expanded.append((path + (digit,), total + float(probs[row, codec.offsets[len(path)] + digit])))
    expanded.sort(key=lambda x: (-x[1], x[0]))
    traces.append(expanded[:beam])
    return result, traces


class GPUTrieTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_csr_matches_every_reachable_prefix(self):
        from vkgr.gpu_trie_decode import CSRTrie
        c, codec = fixture()
        trie = CSRTrie(c, device='cpu')
        for position in range(4):
            prefixes = sorted(set(tuple(map(int, row[:position])) for row in c.sids))
            states = torch.tensor([trie.prefix_key(p) for p in prefixes])
            digits, valid = trie.allowed(states, position)
            for i, prefix in enumerate(prefixes):
                self.assertEqual(digits[i][valid[i]].tolist(), c.allowed_next(prefix))
        with self.assertRaises(ValueError):
            trie.allowed(torch.tensor([999999]), 2)

    def test_random_and_tied_logits_have_identical_all_step_bytes(self):
        from vkgr.gpu_trie_decode import CSRTrie, constrained_beam_csr
        for n in [2, 71, 130]:
            c, codec = fixture(n)
            for tie in [False, True]:
                def score(paths):
                    g = torch.Generator().manual_seed(42 + len(paths[0]))
                    return torch.zeros((len(paths), codec.vocab_size)) if tie else torch.randn((len(paths), codec.vocab_size), generator=g)
                expected, trace = original_trace(c, codec, score, 100)
                actual_trace = []
                actual = constrained_beam_csr(c, codec, score, CSRTrie(c, device='cpu'), beam=100, trace=actual_trace)
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual_trace), 4)
                for left, right in zip(trace, actual_trace):
                    self.assertEqual([p for p, _ in left], [p for p, _ in right])
                    self.assertEqual(b''.join(struct.pack('d', v) for _, v in left), b''.join(struct.pack('d', v) for _, v in right))

    def test_no_query_catalog_scans_and_source_binding(self):
        from vkgr.gpu_trie_decode import CSRTrie, constrained_beam_csr
        c, codec = fixture()
        trie = CSRTrie(c, device='cpu')
        def forbidden(*args):
            raise AssertionError('catalog scan in query')
        c.allowed_next = forbidden
        c.decode = forbidden
        constrained_beam_csr(c, codec, lambda p: torch.zeros((len(p), codec.vocab_size)), trie, beam=100)
        other, other_codec = fixture()
        with self.assertRaises(ValueError):
            constrained_beam_csr(other, other_codec, lambda p: torch.zeros((len(p), codec.vocab_size)), trie, beam=100)

    def test_device_alias_resolves_to_tensor_device(self):
        from vkgr.gpu_trie_decode import CSRTrie, constrained_beam_csr
        c, codec = fixture()
        trie = CSRTrie(c, device='cpu:0')
        result = constrained_beam_csr(c, codec, lambda p: torch.zeros((len(p), codec.vocab_size)), trie)
        self.assertEqual(len(result[0]), len(c.ids))

    def test_invalid_beam_logits_rejected(self):
        from vkgr.gpu_trie_decode import CSRTrie, constrained_beam_csr
        c, codec = fixture()
        trie = CSRTrie(c, device='cpu')
        for beam in [0, True, 1.0]:
            with self.assertRaises(ValueError):
                constrained_beam_csr(c, codec, lambda p: torch.zeros((len(p), codec.vocab_size)), trie, beam=beam)
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                constrained_beam_csr(c, codec, lambda p: torch.full((len(p), codec.vocab_size), value), trie, beam=100)

    @unittest.skipUnless(torch.cuda.is_available(), 'same-runtime CUDA device unavailable')
    def test_cuda_matches_original_random_tied_logits(self):
        from vkgr.gpu_trie_decode import CSRTrie, constrained_beam_csr
        c, codec = fixture(130)
        for tie in [False, True]:
            def score(paths):
                g = torch.Generator(device='cuda').manual_seed(42 + len(paths[0]))
                return torch.zeros((len(paths), codec.vocab_size), device='cuda') if tie else torch.randn((len(paths), codec.vocab_size), generator=g, device='cuda')
            expected, trace = original_trace(c, codec, score, 100)
            actual_trace = []
            actual = constrained_beam_csr(c, codec, score, CSRTrie(c, device='cuda'), beam=100, trace=actual_trace)
            self.assertEqual(actual, expected)
            self.assertEqual(trace, actual_trace)

    def test_real_launcher_prepare_and_dry(self):
        result = subprocess.run([sys.executable, '-B', 'tests/fixtures/gpu_trie_decode_launcher_chain.py'], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'real_launcher_prepare_and_dry_verified')

    def test_cli_dry_and_invalid(self):
        cmd = [sys.executable, '-B', '-m', 'analysis.gpu_trie_decode_probe']
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path.cwd()), str(Path.cwd() / 'src')])}
        good = subprocess.run(cmd + ['dry'], env=env, text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        value = json.loads(good.stdout)
        self.assertEqual(value['sizes'], [120000, 1200000, 12012895])
        self.assertEqual((value['beam'], value['steps'], value['batch']), (100, 4, 1))
        self.assertFalse(value['science_training'])
        bad = subprocess.run(cmd + ['invalid'], env=env, text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
