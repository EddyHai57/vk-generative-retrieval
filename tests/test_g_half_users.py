from pathlib import Path
import json
import os
import random
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkgr import g_half_users_a1_run as g_half_users


class GHalfUsersTests(unittest.TestCase):
    def test_training_user_pool_is_unique_and_sorted(self):
        rows = [
            {'example_id': 0, 'user_id': 9, 'target': 1, 'history': [0]},
            {'example_id': 1, 'user_id': 4, 'target': 2, 'history': [1]},
            {'example_id': 2, 'user_id': 9, 'target': 3, 'history': [1, 2]},
        ]
        self.assertEqual(g_half_users.training_user_ids(rows), (4, 9))

    def test_retention_keeps_every_example_of_selected_training_users(self):
        rows = [
            {'example_id': 0, 'user_id': 9, 'target': 1, 'history': [0]},
            {'example_id': 1, 'user_id': 4, 'target': 2, 'history': [1]},
            {'example_id': 2, 'user_id': 9, 'target': 3, 'history': [1, 2]},
        ]
        retained = g_half_users.retain_training_rows(rows, (9,))
        self.assertEqual(retained, (rows[0], rows[2]))
        self.assertEqual([row['history'] for row in retained], [[0], [1, 2]])

    def test_retention_rejects_unknown_or_empty_user_selection(self):
        rows = [{'example_id': 0, 'user_id': 9, 'target': 1, 'history': [0]}]
        for users in ((), (3,), (9, 9)):
            with self.subTest(users=users), self.assertRaises(ValueError):
                g_half_users.retain_training_rows(rows, users)

    def test_sample_selection_requires_frozen_count_and_seed(self):
        users = (4, 9, 12, 18)
        self.assertEqual(g_half_users.select_users(users, 2, seed=20261020), (12, 18))
        for count, seed in ((0, 20261020), (5, 20261020), (2, 20261021)):
            with self.subTest(count=count, seed=seed), self.assertRaises(ValueError):
                g_half_users.select_users(users, count, seed=seed)

    def test_frozen_half_size_uses_floor_for_odd_and_even_pools(self):
        self.assertEqual(g_half_users.half_sample_size(8), 4)
        self.assertEqual(g_half_users.half_sample_size(7), 3)
        with self.assertRaises(ValueError):
            g_half_users.half_sample_size(1)

    def test_seeded_selection_uses_random_sample_over_sorted_pool(self):
        users = tuple(range(100, 108))
        before = random.getstate()
        selected = g_half_users.seeded_user_selection(users, 4, seed=20261020)
        self.assertEqual(selected, (102, 105, 106, 107))
        self.assertEqual(g_half_users.seeded_user_selection(users, 4, seed=20261020), selected)
        self.assertEqual(random.getstate(), before)
        with self.assertRaises(ValueError):
            g_half_users.seeded_user_selection(users, 4, seed=20261021)

    def test_config_carries_the_frozen_floor_rule_and_only_new_sampling_fields(self):
        for seed in (42, 2024, 2025):
            config = {**g_half_users.default_config(), 'seed': seed,
                      'sid_manifest_sha256': 'a' * 64}
            g_half_users.validate_config(config, False)
            root_config = json.loads((ROOT / f'configs/sid_structure/g-structure-a1-s{seed}.json').read_text())
            expected = {**root_config, 'schema': g_half_users.SCHEMA, 'task_id': g_half_users.TASK_ID,
                        'sid_manifest_sha256': 'a' * 64,
                        'user_sample_fraction': .5, 'user_sample_seed': 20261020,
                        'user_sample_count_rule': g_half_users.SAMPLE_COUNT_RULE,
                        'user_sample_method': g_half_users.SAMPLE_METHOD}
            self.assertEqual(config, expected)

    def test_user_id_sha_uses_compact_utf8_json_without_final_lf(self):
        users = (2, 13, 25)
        raw = b'[2,13,25]'
        import hashlib
        self.assertEqual(g_half_users.user_ids_sha256(users), hashlib.sha256(raw).hexdigest())

    def test_selected_user_pin_is_required_when_training_or_verifying(self):
        pin = 'a' * 64
        self.assertEqual(g_half_users.check_selected_users_pin(pin, pin, required=True), pin)
        for actual, expected, required in ((pin, 'b' * 64, True), (None, None, True),
                                           (pin, 'bad', True)):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                g_half_users.check_selected_users_pin(actual, expected, required=required)

    def test_validation_bytes_bind_to_anchor_context_and_survive_sampling(self):
        queries = [{'user_id': 1, 'history': [4], 'target': 6}]
        binding = {'queries_sha256': g_half_users.digest(queries), 'validation_file_sha256': 'b' * 64,
                   'consumer': {'git_base': 'c' * 40, 'git_dirty': False}}
        anchor = type('Context', (), {'binding': binding, 'queries': queries})()
        anchor_git = {'git_base': 'a' * 40, 'git_dirty': False}
        normalized = {**binding, 'consumer': anchor_git}
        context_sha = g_half_users.digest(normalized)
        self.assertEqual(g_half_users.validate_anchor_context(anchor, context_binding_sha256=context_sha, anchor_git=anchor_git),
                         {'context_binding_sha256': context_sha, 'queries_sha256': g_half_users.digest(queries),
                          'validation_file_sha256': 'b' * 64})
        sampled = type('Context', (), {'binding': dict(binding), 'queries': list(queries)})()
        self.assertEqual(g_half_users.validate_sampled_validation(anchor, sampled),
                         {'queries_sha256': g_half_users.digest(queries), 'validation_file_sha256': 'b' * 64})
        changed_queries = type('Context', (), {'binding': dict(binding),
                                               'queries': [{'user_id': 1, 'history': [5], 'target': 6}]})()
        with self.assertRaises(ValueError):
            g_half_users.validate_sampled_validation(anchor, changed_queries)
        with self.assertRaises(ValueError):
            g_half_users.validate_anchor_context(anchor, context_binding_sha256='c' * 64, anchor_git=anchor_git)

    def test_sample_context_filters_all_train_rows_and_leaves_validation_and_sid_untouched(self):
        from types import SimpleNamespace
        from vkgr import generative_experiment as ge
        rows = [
            {'example_id': 0, 'user_id': 5, 'target': 10, 'history': [20]},
            {'example_id': 1, 'user_id': 9, 'target': 20, 'history': [10]},
            {'example_id': 2, 'user_id': 5, 'target': 30, 'history': [10, 20]},
            {'example_id': 3, 'user_id': 12, 'target': 10, 'history': [30]},
        ]
        queries = [{'user_id': 9, 'history': [10], 'target': 30},
                   {'user_id': 12, 'history': [20], 'target': 10}]
        class Examples:
            def __len__(self): return len(rows)
            def rows(self, indices): return [dict(rows[index]) for index in indices]
        class Codec:
            catalog = SimpleNamespace(ids=(10, 20, 30))
            def check_identity(self): return self
        full = SimpleNamespace(examples=Examples(), codec=Codec(), queries=queries,
                               slice_definition=object(),
                               binding={'queries_sha256': ge._digest(queries),
                                        'catalog_sha256': ge._digest([10, 20, 30]),
                                        'b1': {'train_history_sha256': 'a' * 64}})
        full.check = lambda: full
        selected = g_half_users.seeded_user_selection(g_half_users.training_user_ids(rows), 1, seed=20261020)
        selected_sha = g_half_users.user_ids_sha256(selected)
        sampled = g_half_users.sample_context(full, expected_selected_users_sha256=selected_sha)
        sample_rows = sampled.examples.rows(list(range(len(sampled.examples))))
        self.assertIs(sampled.codec, full.codec)
        self.assertIs(sampled.slice_definition, full.slice_definition)
        self.assertEqual(g_half_users.canonical_bytes(sampled.queries), g_half_users.canonical_bytes(full.queries))
        self.assertEqual(sampled.binding['queries_sha256'], full.binding['queries_sha256'])
        self.assertEqual(sampled.binding['g_half_users']['selected_users_sha256'], selected_sha)
        self.assertEqual(sample_rows, [row for row in rows if row['user_id'] in set(selected)])
        self.assertTrue(all(row['user_id'] in set(selected) for row in sample_rows))
        with self.assertRaises(ValueError):
            g_half_users.sample_context(full, expected_selected_users_sha256='f' * 64)

    def test_a1_context_adapter_reads_config_from_original_context_signature(self):
        from vkgr import sid_structure_a1_run as base
        sentinel_context, sentinel_sid = object(), object()
        original_contexts = mock.Mock(return_value=(sentinel_context, False, sentinel_sid))
        with mock.patch.object(base, '_contexts', original_contexts), \
                mock.patch.object(g_half_users, 'sample_context', return_value='sampled') as sample, \
                mock.patch.dict(os.environ, {'VKGR_EXPECTED_SELECTED_USERS_SHA256': 'a' * 64}):
            with g_half_users._patched_base()() as patched:
                config = {**g_half_users.default_config(), 'user_sample_count_rule': 'floor(U/2)'}
                args = (Path('.'),) * 9 + (config, Path('scratch'))
                result = patched._contexts(*args)
                self.assertEqual(result, ('sampled', False, sentinel_sid))
                original_contexts.assert_called_once_with(*args)
                sample.assert_called_once_with(sentinel_context, expected_selected_users_sha256='a' * 64)
        self.assertEqual(base.TASK_ID, 'SID-STRUCTURE-MATCHED')


if __name__ == '__main__':
    unittest.main()
