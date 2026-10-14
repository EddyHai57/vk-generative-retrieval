"""G6-PREP step 7 mutants M60..M75 (position table, step positions, decoder swap, gates, binding, schema): all must be killed.

The mutated module is re-executed from source; because ``generative_g6_run`` binds the model-module functions by name at import,
the run module is re-pointed at the mutated twins for the duration of a model mutant.
"""
import contextlib
import unittest
from unittest import mock

from tests.mutation_util import mutated, run_ids
from vkgr import generative_g6_data as gdata
from vkgr import generative_g6_model as gm
from vkgr import generative_g6_run as g6

U = 'test_generative_g6_unit.'
W = 'test_generative_g6.'
UNIT = [U + 'ConfigTests', U + 'ModelTests', U + 'TrainingStepTests']
FRESH = [W + 'FreshRuns']
WORLD = [W + 'DepthThreeEquivalence', W + 'DepthFour', W + 'FailClosed']


@contextlib.contextmanager
def mutate_model(old, new):
    with mutated(gm, old, new):
        with mock.patch.multiple(g6, make_model_arch_d=gm.make_model_arch_d, train_batch_d=gm.train_batch_d):
            yield


@contextlib.contextmanager
def mutate_data(old, new):
    with mutated(gdata, old, new):
        with mock.patch.object(g6, 'build_g6_context', gdata.build_g6_context):
            yield


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_generative_g6 as world
        for name in ('arch', 'legacy3', 'd4', 'd4-batched'):  # build the cached worlds and runs before any patch is active
            world.fx().cached(name)

    def test_unmutated_selection_is_green(self):
        result = run_ids(UNIT + FRESH + WORLD)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 31)

    def survive(self, manager, ids):
        from vkgr import generative_batched as gb
        from vkgr import tiger
        names = (tiger.decode_history, gb.decode_histories_batched)
        try:
            with manager:
                result = run_ids(ids)
        finally:  # a mutant that forgets to restore the swapped decoder names must not leak into the next test
            tiger.decode_history, gb.decode_histories_batched = names
        return result.wasSuccessful()

    def kill_model(self, old, new, ids):
        self.assertFalse(self.survive(mutate_model(old, new), ids), f'model mutant survived: {old!r} -> {new!r}')

    def kill_run(self, old, new, ids):
        self.assertFalse(self.survive(mutated(g6, old, new), ids), f'run mutant survived: {old!r} -> {new!r}')

    def test_M60_position_table_fixed_at_200_rows(self):
        self.kill_model("max_tokens = (codec.depth + 1) * config['history_len']", 'max_tokens = 200', [U + 'ModelTests'] + FRESH)

    def test_M61_training_step_keeps_four_positions(self):
        self.kill_model('positions = codec.depth + 1', 'positions = 4', [U + 'TrainingStepTests'] + FRESH)

    def test_M62_loss_does_not_check_the_position_count(self):
        self.kill_model('logits.shape[1] != positions or ', '', [U + 'TrainingStepTests'])

    def test_M63_embed_limit_ignores_max_tokens(self):
        self.kill_model('not 1 <= tokens.shape[1] <= self.max_tokens', 'not 1 <= tokens.shape[1] <= POSITIONS', [U + 'ModelTests'] + FRESH)

    def test_M64_position_table_created_after_the_transformer(self):
        self.kill_model("        self.position = nn.Embedding(max_tokens, d_model)\n        self.transformer = _Core(d_model, heads, head_dim, encoder_layers, decoder_layers, ff, dropout)\n",
                        "        self.transformer = _Core(d_model, heads, head_dim, encoder_layers, decoder_layers, ff, dropout)\n        self.position = nn.Embedding(max_tokens, d_model)\n",
                        [U + 'ModelTests'] + FRESH)

    def test_M65_microbatch_weight_dropped(self):
        self.kill_model('weighted = loss * (len(batch) / n)', 'weighted = loss / n', [U + 'TrainingStepTests'])

    def test_M66_factory_accepts_a_plain_codec(self):
        self.kill_model('    if not isinstance(codec, ValidatedCodecD):\n        raise ValueError(\'depth-validated codec required\')\n', '', [U + 'ModelTests'])

    def test_M67_depth_swap_not_applied(self):
        self.kill_run('tiger.decode_history, gb.decode_histories_batched = sdb.decode_history_d, sdb.decode_histories_batched_d',
                      'tiger.decode_history, gb.decode_histories_batched = tiger.decode_history, gb.decode_histories_batched', FRESH)

    def test_M68_depth_swap_not_restored(self):
        self.kill_run('        tiger.decode_history, gb.decode_histories_batched = saved', '        pass', FRESH)

    def test_M69_worker_gate_removed(self):
        self.kill_run('    if workers != 1:\n        raise ValueError(\'G6: VKGR_EVAL_WORKERS', '    if False:\n        raise ValueError(\'G6: VKGR_EVAL_WORKERS', FRESH)

    def test_M70_batch_gate_removed(self):
        self.kill_run("    if decode is not None and decode['batch_size'] != 1:", '    if False:', FRESH)

    def test_M71_sid_label_not_compared_with_the_bundle(self):
        self.kill_run("    if sid.info['sid_version'] != config['sid']:", '    if False:', [W + 'FailClosed.test_config_and_bundle_must_agree'])

    def test_M72_manifest_kind_of_the_other_runner(self):
        self.kill_run("MANIFEST_KIND = 'g_g6_train'", "MANIFEST_KIND = 'g_tok_train'", FRESH + [W + 'FailClosed'])

    def test_M73_depth_missing_from_the_binding(self):
        self.kill_run("'manifest_sha256': info['manifest_sha256'],\n                    'content_feature_sha256': sid.manifest['content_feature_sha256'], 'branching': info['branching'], 'depth': info['depth'],",
                      "'manifest_sha256': info['manifest_sha256'],\n                    'content_feature_sha256': sid.manifest['content_feature_sha256'], 'branching': info['branching'],", FRESH)

    def test_M74_formal_config_accepts_any_structure(self):
        self.kill_run('    elif tuple(value[k] for k in ga.SHAPE_KEYS) != ga.PAPER_SCALE or', '    elif False or', [U + 'ConfigTests'])

    def test_M75_formal_context_accepts_the_replay_label(self):
        self.kill_run('allowed = sid_v2d.ENGINEERING_SID_VERSIONS if engineering else sid_v2d.SID_VERSIONS', 'allowed = sid_v2d.ENGINEERING_SID_VERSIONS', [U + 'ConfigTests'])

    def test_M76_binding_not_compared_on_verify(self):
        self.kill_run("    if canonical_bytes(recorded) != canonical_bytes(fresh):\n        raise ValueError('run input/source/runtime/init/SID binding mismatch')",
                      '    return None', [W + 'FailClosed.test_a_forged_binding_in_the_manifest_is_refused'])

    def test_M77_best_checkpoint_hash_not_checked(self):
        self.kill_run("    if type(best) is not dict or best.get('path') != 'best.pt' or file_sha256(run_dir / 'best.pt') != best.get('sha256'):\n        raise ValueError('best checkpoint hash mismatch')",
                      '    return None', [W + 'FailClosed.test_tampered_manifest_wrong_pin_and_tampered_checkpoint'])

    def test_M78_data_binding_without_the_g6_section(self):
        self.assertFalse(self.survive(mutate_data("'g6': {'sid_version'", "'tok': {'sid_version'"), FRESH + WORLD), 'data mutant survived')


if __name__ == '__main__':
    unittest.main()
