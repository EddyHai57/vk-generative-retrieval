"""GTOK-01B mutants M79..M87 (SIDMAP_V2 depth-3 path of the G6 runner: dispatch, registry, depth adaptation, formal pins): all must be killed.

Cached runs are built in setUpClass before any patch is active, so every kill list names tests that execute the mutated code itself
(fresh runs, direct loader/config calls).
"""
import unittest

from tests.mutation_util import mutated, run_ids
from vkgr import generative_g6_run as g6

V = 'test_generative_g6_v2.'
W = 'test_generative_g6.'
CONFIG = [V + 'ConfigV2Tests']
DISPATCH = [V + 'DispatchTests', V + 'FormalSidPath']
FRESH_V2 = [V + 'V2DepthThreeEquivalence.test_fresh_one_epoch_v2_run_equals_the_arch_runner', V + 'V2Cli']
ALL_V2 = CONFIG + DISPATCH + [V + 'V2DepthThreeEquivalence', V + 'V2FailClosed', V + 'V2Cli']


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_generative_g6 as world
        import test_generative_g6_v2 as v2world
        for name in ('arch', 'legacy3'):  # build the cached worlds, bundles and runs before any patch is active
            world.fx().cached(name)
        for name in ('replay', 'formal-t3'):
            v2world.v2().bundle(name)
        v2world.v2().cached('v2')

    def test_unmutated_selection_is_green(self):
        result = run_ids(ALL_V2 + [W + 'FreshRuns'])
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 27)

    def survive(self, old, new, ids):
        from vkgr import generative_batched as gb
        from vkgr import tiger
        names = (tiger.decode_history, gb.decode_histories_batched)
        try:
            with mutated(g6, old, new):
                result = run_ids(ids)
        finally:  # a mutant that breaks the scoped decoder swap must not leak into the next test
            tiger.decode_history, gb.decode_histories_batched = names
        return result.wasSuccessful()

    def kill(self, old, new, ids):
        self.assertFalse(self.survive(old, new, ids), f'run mutant survived: {old!r} -> {new!r}')

    def test_M79_dispatch_always_uses_the_v2d_loader(self):
        self.kill("loader = _load_sid_v2 if config['sid'] in _v2_labels(engineering) else _load_sid", 'loader = _load_sid',
                  FRESH_V2 + [V + 'V2FailClosed.test_label_and_bundle_format_must_agree_before_any_output'])

    def test_M80_dispatch_always_uses_the_v2_loader(self):
        self.kill("loader = _load_sid_v2 if config['sid'] in _v2_labels(engineering) else _load_sid", 'loader = _load_sid_v2', [W + 'FreshRuns'])

    def test_M81_adapter_uses_the_wrong_depth(self):
        self.kill('V2_DEPTH = 3', 'V2_DEPTH = 4', DISPATCH)

    def test_M82_adapter_drops_the_depth_from_the_info(self):
        self.kill("{**loaded.info, 'depth': V2_DEPTH}", 'dict(loaded.info)', DISPATCH)

    def test_M83_formal_context_uses_the_engineering_v2_labels(self):
        self.kill('return V2_ENGINEERING_SID_VERSIONS if engineering else V2_SID_VERSIONS', 'return V2_ENGINEERING_SID_VERSIONS', CONFIG)

    def test_M84_formal_v2_registry_opened_to_every_sid_v2_label(self):
        self.kill("V2_SID_VERSIONS = ('SID_BHKM_B256D3_S42',)", 'V2_SID_VERSIONS = sid_v2.SID_VERSIONS', CONFIG)

    def test_M85_v2_loader_always_called_as_engineering(self):
        self.kill('engineering=engineering)\n    return _as_depth3(loaded)', 'engineering=True)\n    return _as_depth3(loaded)', DISPATCH)

    def test_M86_validate_config_does_not_add_the_v2_labels(self):
        self.kill('    allowed = allowed + _v2_labels(engineering)\n', '', CONFIG)

    def test_M87_formal_content_v1_guard_removed(self):
        self.kill('    if not engineering and expected_content_digest != sid_v2d.CONTENT_V1_FEATURE_SHA256:', '    if False:', DISPATCH)


if __name__ == '__main__':
    unittest.main()
