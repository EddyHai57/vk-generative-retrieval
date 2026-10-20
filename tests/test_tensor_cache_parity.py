import unittest
import subprocess
import sys
import json
import torch
from analysis.tensor_cache_parity import freeze, assert_equal, warm_cache, observe_p_inputs
from types import SimpleNamespace


class SnapshotTests(unittest.TestCase):
    def test_observation_proxy_preserves_outputs_and_code_object(self):
        global b1
        b1 = SimpleNamespace(_columns=lambda x: (x, False), _log_q=lambda: [0.25])
        original_module = b1
        def step(x):
            return b1._columns(x), b1._log_q()
        trace = {}
        observed = observe_p_inputs(step, trace)
        self.assertIs(step.__code__, observed.__code__)
        self.assertIs(b1, original_module)
        self.assertEqual(step(3), observed(3))
        self.assertEqual(set(trace), {'columns_labels_mask_negatives','logq'})

    def test_p_warmup_uses_real_step_closure_view_without_advancing_cursor(self):
        class Cursor:
            offset = 0
            def next(self, n):
                self.offset += n
                return [0]
        class Cache:
            calls = []
            def p(self, *args): self.calls.append(args)
        cursor, cache = Cursor(), Cache()
        row = {'history': [1]}
        closure = dict(cursor=cursor, config={'batch_size': 1, 'feature':'id'},
                       rows=[row], view={'idmap':{1:2}})
        warm_cache(cache, closure, 'p', 2, None)
        self.assertEqual(cursor.offset,0)
        self.assertEqual(len(cache.calls),2)
        self.assertEqual(cache.calls[0][1],{1:2})

    def test_tensor_bits_signed_zero_and_state_keys(self):
        a = freeze({'parameters': torch.tensor([0.]), 'rng': (1, 2)})
        assert_equal(a, a)
        for b in (freeze({'parameters': torch.tensor([-0.]), 'rng': (1, 2)}),
                  freeze({'parameters': torch.tensor([0.]), 'rng': (2, 1)}),
                  freeze({'parameters': torch.tensor([0.])})):
            with self.assertRaises(ValueError):
                assert_equal(a, b)

    def test_float_exact_and_unknown_objects_rejected(self):
        with self.assertRaises(ValueError):
            assert_equal(freeze(1.), freeze(1. + 1e-15))
        with self.assertRaises(TypeError):
            freeze(object())

    def test_formal_cli_dry_and_invalid(self):
        base = [sys.executable, '-B', '-m', 'analysis.tensor_cache_parity']
        dry = subprocess.run(base+['dry'], capture_output=True, text=True)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertFalse(json.loads(dry.stdout)['capture_executed'])
        bad = subprocess.run(base+['not-a-command'], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
