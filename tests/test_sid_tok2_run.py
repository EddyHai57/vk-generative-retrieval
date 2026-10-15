"""GTOK-02: resolution-sweep tokenizer pipeline (config, gates, fit, verify, compare).

All data is synthetic (N=1500) and runs on CPU with engineering=True unless a test says otherwise. The six variants use the real
frozen ``sid.fit``; the pre-registered ``expected`` integers describe the real 120,415-item catalog, so they are only compared
through the ``_expected_for`` hook (G2), which returns None in engineering mode.
"""
import atexit
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_sid_v2 import World as OldWorld, balanced_sids
from tests.test_sid_v2s import D2_LABEL, SMALL_N, collision_mapping, small_ids, tamper, tamper_json, unit_rows
from vkgr import sid as bhkm
from vkgr import sid_tok_audit as au
from vkgr import sid_tok_run as old_run
from vkgr import sid_tok2_run as run
from vkgr import sid_v2
from vkgr import sid_v2s as s2
from vkgr.data_audit import canonical_bytes, file_sha256

VARIANTS = ('S32', 'S64', 'S128', 'S512', 'S1024', 'S256D2')
LABEL_OF = {'S32': 'SID_BHKM_B32D3_S42', 'S64': 'SID_BHKM_B64D3_S42', 'S128': 'SID_BHKM_B128D3_S42', 'S512': 'SID_BHKM_B512D3_S42',
            'S1024': 'SID_BHKM_B1024D3_S42', 'S256D2': 'SID_BHKM_B256D2_S42'}
T3_SHA = 'a7af78a3fc04f0f0d174b983987c34d7efa754edd88a05d1350cdadd06fd47b8'
CONFIG_SHA = 'c' * 64
INT_FIELDS = ('leaves_count', 'leaves_max', 'vocab_size', 'items_needing_suffix')


def leaf_paths(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaf_paths(item, path + (index,))
    else:
        yield path


def get_at(value, path):
    for part in path:
        value = value[part]
    return value


def set_at(value, path, new):
    parent = get_at(value, path[:-1])
    parent[path[-1]] = new


def fake_audit(branching, expected):
    """An audit-shaped dict holding exactly the numbers ``check_expected`` reads."""
    return {'branching': branching, 'items_needing_suffix': expected['items_needing_suffix'],
            'leaves': {'count': expected['leaves_count'], 'max': expected['leaves_max']},
            'positions': [{'distinct': d} for d in expected['positions_distinct']]}


_INPUTS = {}


def synthetic_inputs():
    if not _INPUTS:
        ids, content = small_ids(), unit_rows(SMALL_N)
        _INPUTS.update(ids=ids, content=content, content_sha256=au.feature_digest(content), catalog_sha256=sid_v2.catalog_digest(ids))
    return _INPUTS


_PARENT = {}


def parent_bundle():
    """The depth-3 256-ary parent of the S256D2 fixture as an engineering SIDMAP_V2 bundle (built once per process):
    returns (manifest path, sha, model)."""
    if not _PARENT:
        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        inputs = synthetic_inputs()
        model = bhkm.fit(inputs['ids'], inputs['content'], branching=256, depth=3, seed=42, max_iter=10)
        root = Path(tmp.name).resolve()
        OldWorld('SID_BHKM_B256D3_S42', 'bhkm', 256, True, model.ids, model.sids, inputs['content']).write(root / 'parent')
        manifest = root / 'parent' / 'run_manifest.json'
        _PARENT.update(path=manifest, sha=file_sha256(manifest), model=model)
    return _PARENT['path'], _PARENT['sha'], _PARENT['model']


_FITTED = {}


def fitted(variant):
    """fit_sweep_variant result on the synthetic catalog (engineering). S256D2 derives from a real 256-ary depth-3 parent bundle."""
    if variant not in _FITTED:
        inputs, config = synthetic_inputs(), run.expected_config()
        parent = None
        if variant == 'S256D2':
            _, parent_sha, model = parent_bundle()
            parent = {'ids': model.ids, 'sids': model.sids, 'manifest_sha256': parent_sha}
        _FITTED[variant] = run.fit_sweep_variant(config, variant, inputs['ids'], inputs['content'], expected_content_sha256=inputs['content_sha256'],
                                                 engineering=True, parent=parent)
    return _FITTED[variant]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.count = 0

    def out(self, name='out'):
        self.count += 1
        return self.root / f'{name}-{self.count}'

    def bundle(self, variant):
        output = self.out(variant)
        run.write_success_bundle(output, fitted(variant), config_sha=CONFIG_SHA)
        return output, file_sha256(output / 'run_manifest.json')

    def verify(self, variant, output, sha, **kwargs):
        args = dict(engineering=True)
        if variant == 'S256D2':
            args.update(parent=parent_bundle()[:2])
        args.update(kwargs)
        return run.verify_bundle(output / 'run_manifest.json', sha, synthetic_inputs(), run.expected_config(), CONFIG_SHA, **args)


class ConfigTests(unittest.TestCase):
    def test_constants(self):
        config = run.expected_config()
        self.assertEqual((config['schema'], config['task_id']), ('g-tok-02-fit-v1', 'G-TOK-02'))
        self.assertEqual(config['content'], {'dim': 64, 'feature_array_sha256': au.CONTENT_V1_FEATURE_SHA256})
        self.assertEqual(tuple(config['variants']), VARIANTS)
        self.assertEqual(run.FAILURE_SCHEMA, 'g-tok-02-failure-v1')

    def test_variants_describe_the_registry(self):
        config = run.expected_config()
        for name, label in LABEL_OF.items():
            variant = config['variants'][name]
            kind, branching, depth = s2.VERSION_SPEC[label]
            self.assertEqual((variant['sid_version'], variant['branching'], variant['depth']), (label, branching, depth))
        for name in VARIANTS[:-1]:
            self.assertEqual({k: config['variants'][name][k] for k in ('kind', 'max_iter', 'seed')}, {'kind': 'bhkm', 'max_iter': 10, 'seed': 42})
        d2 = config['variants']['S256D2']
        self.assertEqual(d2['kind'], 'derive_depth2')
        self.assertEqual(d2['parent'], {'manifest_sha256': T3_SHA, 'sid_version': 'SID_BHKM_B256D3_S42', 'variant': 'T3'})
        self.assertNotIn('seed', d2)
        self.assertNotIn('max_iter', d2)

    def test_expected_table_is_internally_consistent(self):
        for name, variant in run.expected_config()['variants'].items():
            expected = variant['expected']
            self.assertEqual(set(expected), set(INT_FIELDS) | {'positions_distinct'})
            self.assertEqual(expected['vocab_size'], 2 + 3 * variant['branching'] + expected['leaves_max'], name)
            self.assertEqual(expected['items_needing_suffix'], 120415 - expected['leaves_count'], name)
            self.assertEqual(len(expected['positions_distinct']), 4)
            self.assertTrue(all(type(x) is int for x in expected['positions_distinct'] + [expected[k] for k in INT_FIELDS]))

    def test_validate_config_accepts_the_approved_config(self):
        config = run.expected_config()
        self.assertIs(run.validate_config(config), config)

    def test_every_constant_is_frozen(self):
        base = run.expected_config()
        for path in leaf_paths(base):
            value = get_at(base, path)
            changed = (not value) if type(value) is bool else value + 1 if type(value) is int else value + 'x'
            mutated = copy.deepcopy(base)
            set_at(mutated, path, changed)
            with self.subTest(path=path), self.assertRaises(ValueError):
                run.validate_config(mutated)

    def test_key_set_and_types_are_frozen(self):
        base = run.expected_config()
        extra, missing, confused = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
        extra['variants']['S32']['note'] = 1
        del missing['variants']['S64']['seed']
        confused['variants']['S128']['seed'] = 42.0
        for name, bad in (('extra key', extra), ('missing key', missing), ('float seed', confused), ('list', []), ('v1 config', old_run.expected_config())):
            with self.subTest(name), self.assertRaises(ValueError):
                run.validate_config(bad)


class CheckExpectedTests(unittest.TestCase):
    def test_exact_numbers_give_no_mismatch(self):
        for name, variant in run.expected_config()['variants'].items():
            self.assertEqual(run.check_expected(fake_audit(variant['branching'], variant['expected']), variant['expected']), [], name)

    def test_any_single_integer_off_by_one_is_named(self):
        for name, variant in run.expected_config()['variants'].items():
            expected = variant['expected']
            for field in INT_FIELDS:
                broken = copy.deepcopy(expected)
                broken[field] += 1
                found = run.check_expected(fake_audit(variant['branching'], expected), broken)
                self.assertEqual([m['field'] for m in found], [field], (name, field))
                self.assertEqual((found[0]['expected'], found[0]['actual']), (broken[field], expected[field]))
            for index in range(4):
                broken = copy.deepcopy(expected)
                broken['positions_distinct'][index] += 1
                found = run.check_expected(fake_audit(variant['branching'], expected), broken)
                self.assertEqual([m['field'] for m in found], [f'positions_distinct[{index}]'], (name, index))

    def test_there_is_no_tolerance(self):
        variant = run.expected_config()['variants']['S32']
        audit = fake_audit(32, variant['expected'])
        audit['items_needing_suffix'] += 1
        self.assertEqual([m['field'] for m in run.check_expected(audit, variant['expected'])], ['items_needing_suffix'])


class GateTests(unittest.TestCase):
    def audit(self):
        return copy.deepcopy(fitted('S32')['audit']['mapping'])

    def check(self, audit, **kwargs):
        return run.gates_sweep(audit, expected_content_sha256=synthetic_inputs()['content_sha256'], **kwargs)

    def failed_ids(self, result):
        return sorted(c['id'] for c in result['checks'] if not c['ok'])

    def test_clean_mapping_passes_all_seven_gates(self):
        result = self.check(self.audit())
        self.assertTrue(result['passed'], result)
        self.assertEqual([c['id'] for c in result['checks']], ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'B1'])

    def test_each_gate_triggers_on_its_own_violation(self):
        n = self.audit()['items']
        mutations = {
            'H1': lambda a: a.__setitem__('full_sid_collisions', 1),
            'H2': lambda a: a.__setitem__('roundtrip_items', n - 1),
            'H3': lambda a: a.__setitem__('valid_prefix_deadends', 1),
            'H4': lambda a: a.__setitem__('ids_ascending', False),
            'H5': lambda a: a.__setitem__('content_sha256', 'f' * 64),
            'H6': lambda a: a.__setitem__('all_finite', False),
            'B1': lambda a: a['balanced_children'].__setitem__('balanced', False),
        }
        for gate, mutate in mutations.items():
            audit = self.audit()
            mutate(audit)
            result = self.check(audit)
            with self.subTest(gate):
                self.assertFalse(result['passed'])
                self.assertIn(gate, self.failed_ids(result))

    def test_wrong_pinned_content_digest_fails_h5(self):
        result = run.gates_sweep(self.audit(), expected_content_sha256='f' * 64)
        self.assertEqual(self.failed_ids(result), ['H5'])

    def test_rqvae_gates_are_not_used(self):
        self.assertEqual([c['id'] for c in self.check(self.audit())['checks']][-1], 'B1')
        self.assertFalse(any(c['id'].startswith('R') for c in self.check(self.audit())['checks']))

    def test_depth2_balance_uses_the_three_unpadded_columns(self):
        ids, parent, content = collision_mapping()
        sids4, raw3 = s2.derive_depth2_from_parent(ids, parent)
        digest = au.feature_digest(content)
        good = s2.sweep_audit(ids, sids4, content, branching=256, depth=2)
        self.assertTrue(run.gates_sweep(good, expected_content_sha256=digest)['passed'])
        # negative control: the same mapping judged on four padded columns fails B1 and nothing else
        padded = au.audit_mapping(ids, sids4, content, branching=256, kind='bhkm')
        result = run.gates_sweep(padded, expected_content_sha256=digest)
        self.assertEqual(self.failed_ids(result), ['B1'])

    def test_gates_sweep_delegates_to_the_frozen_gate_function(self):
        sentinel = {'passed': True, 'checks': []}
        with mock.patch.object(au, 'gates', return_value=sentinel) as patched:
            self.assertIs(run.gates_sweep(self.audit(), expected_content_sha256='a' * 64), sentinel)
        self.assertEqual(patched.call_args.args[1], 'bhkm')
        self.assertIsNone(patched.call_args.args[2])


class FitTests(Base):
    def test_every_variant_fits_judges_writes_and_verifies(self):
        for variant in VARIANTS:
            with self.subTest(variant):
                result = fitted(variant)
                self.assertTrue(result['ok'], result.get('gates'))
                self.assertEqual((result['status'], result['variant'], result['sid_version'], result['engineering']),
                                 ('complete', variant, LABEL_OF[variant], True))
                self.assertTrue(result['gates']['passed'])
                self.assertEqual(result['audit']['sweep']['depth'], 2 if variant == 'S256D2' else 3)
                output, sha = self.bundle(variant)
                report = self.verify(variant, output, sha)
                self.assertEqual((report['status'], report['sid_version'], report['manifest_sha256']), ('verified', LABEL_OF[variant], sha))
                self.assertIn('config_binding', report['checks'])

    def test_depth3_replay_fit_reproduces_the_stored_mapping(self):
        for variant in VARIANTS[:-1]:
            output, sha = self.bundle(variant)
            report = self.verify(variant, output, sha, replay_fit=True)
            with self.subTest(variant):
                self.assertIn('replay_fit', report['checks'])

    def test_stored_catalog_has_no_nodes_and_summary_is_kept(self):
        result = fitted('S32')
        self.assertEqual(result['catalog'].nodes, [])
        self.assertEqual(set(result['tokenizer']['summary']), {'nodes_per_level', 'child_size_histogram', 'unconverged_nodes'})
        self.assertEqual(result['tokenizer']['fit'], {'branching': 32, 'depth': 3, 'max_iter': 10, 'seed': 42})

    def test_depth3_fit_is_the_unmodified_frozen_fit(self):
        inputs = synthetic_inputs()
        model = bhkm.fit(inputs['ids'], inputs['content'], branching=64, depth=3, seed=42, max_iter=10)
        self.assertTrue(np.array_equal(fitted('S64')['sids'], model.sids))
        self.assertTrue(np.array_equal(fitted('S64')['ids'], model.ids))

    def test_depth2_is_derived_without_clustering(self):
        result = fitted('S256D2')
        parent_sha = parent_bundle()[1]
        self.assertEqual(result['tokenizer']['fit'], {'branching': 256, 'depth': 2, 'parent_manifest_sha256': parent_sha})
        self.assertEqual(result['tokenizer']['parent_manifest_sha256'], parent_sha)
        self.assertEqual(int(result['sids'][:, 2].max()), 0)
        with mock.patch.object(bhkm, 'fit', side_effect=AssertionError('no clustering for depth 2')):
            run.fit_sweep_variant(run.expected_config(), 'S256D2', synthetic_inputs()['ids'], synthetic_inputs()['content'],
                                  expected_content_sha256=synthetic_inputs()['content_sha256'], engineering=True,
                                  parent={'ids': parent_bundle()[2].ids, 'sids': parent_bundle()[2].sids, 'manifest_sha256': parent_bundle()[1]})

    def test_depth2_needs_a_matching_parent_and_formal_needs_the_pinned_sha(self):
        inputs, config = synthetic_inputs(), run.expected_config()
        parent = {'ids': parent_bundle()[2].ids, 'sids': parent_bundle()[2].sids, 'manifest_sha256': '0' * 64}
        common = dict(expected_content_sha256=inputs['content_sha256'])
        with self.assertRaises(ValueError):
            run.fit_sweep_variant(config, 'S256D2', inputs['ids'], inputs['content'], engineering=True, **common)
        with self.assertRaises(ValueError):
            run.fit_sweep_variant(config, 'S256D2', inputs['ids'], inputs['content'], engineering=False, parent=parent, **common)
        with self.assertRaises(ValueError):
            run.fit_sweep_variant(config, 'S32', inputs['ids'], inputs['content'], engineering=True, parent=parent, **common)
        with self.assertRaises(ValueError):
            run.fit_sweep_variant(config, 'S99', inputs['ids'], inputs['content'], engineering=True, **common)

    def test_failed_gates_give_a_failure_record_and_no_bundle(self):
        inputs = synthetic_inputs()
        result = run.fit_sweep_variant(run.expected_config(), 'S32', inputs['ids'], inputs['content'], expected_content_sha256='f' * 64, engineering=True)
        self.assertFalse(result['ok'])
        self.assertEqual(result['status'], 'gate_failed')
        self.assertNotIn('catalog', result)
        with self.assertRaises(ValueError):
            run.write_success_bundle(self.out('never'), result, config_sha=CONFIG_SHA)
        self.assertFalse((self.root / 'never-1').exists())
        record = run.failure_record_s(result, config_sha=CONFIG_SHA, catalog_sha256=inputs['catalog_sha256'])
        self.assertEqual((record['schema'], record['task_id'], record['reasons']), ('g-tok-02-failure-v1', 'G-TOK-02', ['H5']))
        old_run.write_failure_record(self.out('failure'), record)
        self.assertEqual(sorted(p.name for p in (self.root / 'failure-2').iterdir()), ['failure.json'])

    def test_producer_records_the_six_source_files(self):
        output, _ = self.bundle('S32')
        manifest = json.loads((output / 'run_manifest.json').read_text())
        self.assertEqual(sorted(manifest['producer']), sorted(f'src/vkgr/{n}' for n in (
            'sid_tok2_run.py', 'sid_v2s.py', 'sid_v2.py', 'sid_tok_audit.py', 'sid.py', 'sid_tok_run.py')))
        self.assertEqual(manifest['fit_config_sha256'], CONFIG_SHA)


class VerifyTests(Base):
    def test_wrong_config_sha_or_tokenizer_record_is_rejected(self):
        output, sha = self.bundle('S64')
        with self.assertRaises(ValueError):
            run.verify_bundle(output / 'run_manifest.json', sha, synthetic_inputs(), run.expected_config(), 'd' * 64, engineering=True)
        for name, mutate in {'seed': lambda t: t['config']['fit'].__setitem__('seed', 7),
                             'overrides': lambda t: t['config'].__setitem__('engineering_overrides', {'x': 1}),
                             'branching': lambda t: t['config']['fit'].__setitem__('branching', 32)}.items():
            copy_dir, copy_sha = self.bundle('S64')
            copy_sha = tamper_json(copy_dir, 'tokenizer.json', mutate)
            with self.subTest(name), self.assertRaises(ValueError):
                self.verify('S64', copy_dir, copy_sha)

    def test_a_bundle_of_another_variant_is_not_this_variants_bundle(self):
        output, sha = self.bundle('S32')
        config = copy.deepcopy(run.expected_config())
        with self.assertRaises(ValueError):  # the tokenizer record says branching 32, the S64 constants say 64
            run.verify_bundle(output / 'run_manifest.json', sha, synthetic_inputs(), config, CONFIG_SHA, engineering=True, variant='S64')

    def test_wrong_inputs_are_rejected(self):
        output, sha = self.bundle('S32')
        inputs = dict(synthetic_inputs())
        inputs['ids'] = inputs['ids'] + 1
        with self.assertRaises(ValueError):
            run.verify_bundle(output / 'run_manifest.json', sha, inputs, run.expected_config(), CONFIG_SHA, engineering=True)

    def test_formal_mode_pins_content_v1(self):
        output, sha = self.bundle('S32')
        with self.assertRaises(ValueError):
            run.verify_bundle(output / 'run_manifest.json', sha, synthetic_inputs(), run.expected_config(), CONFIG_SHA, engineering=False)

    def test_expected_comparison_is_skipped_in_engineering_and_reported_in_formal_hook(self):
        self.assertIsNone(run._expected_for(run.expected_config(), 'S32', True))
        self.assertEqual(run._expected_for(run.expected_config(), 'S32', False), run.expected_config()['variants']['S32']['expected'])
        output, sha = self.bundle('S32')
        audit = fitted('S32')['audit']['mapping']
        truth = {'leaves_count': audit['leaves']['count'], 'leaves_max': audit['leaves']['max'], 'vocab_size': audit['vocab_size'],
                 'items_needing_suffix': audit['items_needing_suffix'], 'positions_distinct': [p['distinct'] for p in audit['positions']]}
        with mock.patch.object(run, '_expected_for', return_value=truth):
            self.assertEqual(self.verify('S32', output, sha)['status'], 'verified')
        for field in INT_FIELDS:
            broken = copy.deepcopy(truth)
            broken[field] += 1
            with self.subTest(field), mock.patch.object(run, '_expected_for', return_value=broken):
                report = self.verify('S32', output, sha)
                self.assertEqual(report['status'], 'expected_mismatch')
                self.assertEqual([m['field'] for m in report['mismatches']], [field])

    def test_depth2_rederivation_is_bit_exact(self):
        output, sha = self.bundle('S256D2')
        report = self.verify('S256D2', output, sha)
        self.assertIn('rederive_depth2', report['checks'])
        # another parent bundle is refused because the recorded parent sha differs
        inputs = synthetic_inputs()
        other = bhkm.fit(inputs['ids'], inputs['content'], branching=256, depth=3, seed=43, max_iter=10)
        world = OldWorld('SID_BHKM_B256D3_S42', 'bhkm', 256, True, other.ids, other.sids, inputs['content'])
        target = self.out('other-parent')
        world.write(target)
        other_parent = (target / 'run_manifest.json', file_sha256(target / 'run_manifest.json'))
        with self.assertRaises(ValueError):
            self.verify('S256D2', output, sha, parent=other_parent)
        # a bundle that claims the other parent but was derived from the first one is caught by the bit-exact rederivation only
        liar, _ = self.bundle('S256D2')
        tamper_json(liar, 'tokenizer.json', lambda t: (t['config']['fit'].__setitem__('parent_manifest_sha256', other_parent[1]),
                                                         t['config'].__setitem__('parent_manifest_sha256', other_parent[1])))
        liar_sha = tamper_json(liar, 'audit.json', lambda a: a['sweep'].__setitem__('parent_manifest_sha256', other_parent[1]))
        with self.assertRaisesRegex(ValueError, 'rederiv'):
            self.verify('S256D2', liar, liar_sha, parent=other_parent)

    def test_depth2_parent_arguments_are_required_and_only_for_depth2(self):
        output, sha = self.bundle('S256D2')
        with self.assertRaises(ValueError):
            self.verify('S256D2', output, sha, parent=None)
        output3, sha3 = self.bundle('S32')
        with self.assertRaises(ValueError):
            self.verify('S32', output3, sha3, parent=parent_bundle()[:2])

    def test_depth2_parent_must_be_the_pinned_t3_in_formal_mode(self):
        # formal mode cannot run on synthetic content; the sha rule is tested through the helper
        with self.assertRaises(ValueError):
            run._check_parent_sha('0' * 64, run.expected_config()['variants']['S256D2'], engineering=False)
        run._check_parent_sha(T3_SHA, run.expected_config()['variants']['S256D2'], engineering=False)
        run._check_parent_sha('0' * 64, run.expected_config()['variants']['S256D2'], engineering=True)

    def test_replay_fit_catches_a_mapping_that_is_not_the_fit_output(self):
        output, _ = self.bundle('S32')
        w = fitted('S32')
        swapped = w['sids'].copy()
        swapped[[0, 1], :3] = swapped[[1, 0], :3]  # leaves are singletons here: still a legal, balanced, canonical mapping
        data = self.root / 'swapped.npz'
        sid_v2._write_npz(data, {'ids': w['ids'], 'sids': swapped})
        audit = s2.sweep_audit(w['ids'], swapped, synthetic_inputs()['content'], branching=32, depth=3)
        tamper(output, 'mapping.npz', data.read_bytes())
        sha = tamper_json(output, 'audit.json', lambda a: a.__setitem__('mapping', json.loads(canonical_bytes(audit))))
        self.assertEqual(self.verify('S32', output, sha)['status'], 'verified')  # loads fine: only the replay can notice
        with self.assertRaisesRegex(ValueError, 'replay'):
            self.verify('S32', output, sha, replay_fit=True)


class CompareTests(Base):
    def baseline(self):
        inputs = synthetic_inputs()
        model = bhkm.fit(inputs['ids'], inputs['content'], branching=16, depth=3, seed=42, max_iter=10)
        path = self.out('baseline').with_suffix('.npz')
        np.savez(path, ids=model.ids, sids=model.sids)
        return path, file_sha256(path)

    def t3_bundle(self):
        inputs = synthetic_inputs()
        model = bhkm.fit(inputs['ids'], inputs['content'], branching=256, depth=3, seed=42, max_iter=10)
        world = OldWorld('SID_BHKM_B256D3_S42', 'bhkm', 256, True, model.ids, model.sids, inputs['content'])
        target = self.out('t3')
        world.write(target)
        return target / 'run_manifest.json', file_sha256(target / 'run_manifest.json')

    def test_subset_compare_keeps_label_order_and_skips_missing_points(self):
        b32, b64 = self.bundle('S32'), self.bundle('S64')
        result = run.compare_bundles([(b64[0] / 'run_manifest.json', b64[1]), self.t3_bundle(), (b32[0] / 'run_manifest.json', b32[1])],
                                     self.baseline(), synthetic_inputs(), engineering=True)
        self.assertEqual(result['schema'], 'g-tok-02-compare-v1')
        self.assertEqual(result['table']['labels'], ['T0', 'T3', 'S32', 'S64'])
        self.assertEqual(set(result['bundle_manifest_sha256']), {'T3', 'S32', 'S64'})
        rows = {row['item']: row['values'] for row in result['table']['rows']}
        self.assertEqual(rows['items'], {label: SMALL_N for label in result['table']['labels']})

    def test_depth2_point_shows_balanced_on_the_unpadded_columns(self):
        b256d2 = self.bundle('S256D2')
        result = run.compare_bundles([(b256d2[0] / 'run_manifest.json', b256d2[1])], self.baseline(), synthetic_inputs(), engineering=True)
        rows = {row['item']: row['values'] for row in result['table']['rows']}
        self.assertTrue(rows['balanced_children.balanced']['S256D2'])
        self.assertEqual(result['table']['labels'], ['T0', 'S256D2'])

    def test_duplicate_or_unknown_bundles_are_rejected(self):
        b32 = self.bundle('S32')
        entry = (b32[0] / 'run_manifest.json', b32[1])
        with self.assertRaises(ValueError):
            run.compare_bundles([entry, entry], self.baseline(), synthetic_inputs(), engineering=True)
        old = OldWorld('SID_V2_REPLAY_BHKM_B16D3', 'bhkm', 16, True, small_ids(), balanced_sids(SMALL_N, 16), unit_rows(SMALL_N))
        old.write(self.root / 'old')
        with self.assertRaises(ValueError):
            run.compare_bundles([(self.root / 'old' / 'run_manifest.json', file_sha256(self.root / 'old' / 'run_manifest.json'))],
                                self.baseline(), synthetic_inputs(), engineering=True)

    def test_wrong_sha_is_rejected(self):
        b32 = self.bundle('S32')
        with self.assertRaises(ValueError):
            run.compare_bundles([(b32[0] / 'run_manifest.json', '0' * 64)], self.baseline(), synthetic_inputs(), engineering=True)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'gtok02' / 'g-tok-02.json'


def write_files(root):
    """Pinned content (row 0 = PAD) and catalog files for the synthetic catalog, plus the committed-equal config file."""
    inputs = synthetic_inputs()
    content = Path(root) / 'content.npy'
    catalog = Path(root) / 'catalog.json'
    config = Path(root) / 'config.json'
    array = np.zeros((len(inputs['ids']) + 1, 64), np.float32)
    array[1:] = inputs['content']
    np.save(content, array)
    catalog.write_bytes(canonical_bytes([int(x) for x in inputs['ids']]))
    config.write_text(json.dumps(run.expected_config(), sort_keys=True, indent=1) + '\n')
    return {'content': content, 'content_sha': file_sha256(content), 'catalog': catalog, 'catalog_sha': file_sha256(catalog),
            'config': config, 'config_sha': file_sha256(config)}


def fit_argv(variant, files, output, *extra, engineering=True):
    argv = ['fit', '--variant', variant, '--config', str(files['config']), '--config-sha', files['config_sha'],
            '--content', str(files['content']), '--content-sha', files['content_sha'],
            '--catalog', str(files['catalog']), '--catalog-sha', files['catalog_sha'], '--output', str(output)]
    return argv + (['--engineering'] if engineering else []) + list(extra)


def verify_argv(manifest, files, output, *extra, engineering=True):
    manifest = Path(manifest)
    argv = ['verify', '--bundle', str(manifest), '--sha', file_sha256(manifest), '--config', str(files['config']),
            '--config-sha', files['config_sha'], '--content', str(files['content']), '--content-sha', files['content_sha'],
            '--catalog', str(files['catalog']), '--catalog-sha', files['catalog_sha'], '--output', str(output)]
    return argv + (['--engineering'] if engineering else []) + list(extra)


def cli(argv, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run.main(argv, **kwargs)
    return code, out.getvalue(), err.getvalue()


class CliTests(Base):
    def setUp(self):
        super().setUp()
        self.files = write_files(self.root)

    def parent_args(self):
        path, sha, _ = parent_bundle()
        return ['--parent-manifest', str(path), '--parent-sha', sha]

    def fit(self, variant, *extra, **kwargs):
        output = self.out(variant.lower())
        code, stdout, stderr = cli(fit_argv(variant, self.files, output, *extra, **kwargs))
        return output, code, stdout, stderr

    def test_committed_config_is_the_approved_config(self):
        text = CONFIG_PATH.read_text()
        self.assertEqual(text, json.dumps(run.expected_config(), sort_keys=True, indent=1) + '\n')
        self.assertEqual(run.validate_config(json.loads(text)), run.expected_config())
        self.assertEqual(run.load_config(CONFIG_PATH, file_sha256(CONFIG_PATH)), run.expected_config())

    def test_end_to_end_fit_verify_compare_for_all_six_variants(self):
        bundles = []
        for variant in VARIANTS:
            with self.subTest(variant):
                extra = self.parent_args() if variant == 'S256D2' else []
                output, code, stdout, stderr = self.fit(variant, *extra)
                self.assertEqual((code, stderr), (0, ''))
                line = json.loads(stdout)
                manifest = output / 'run_manifest.json'
                self.assertEqual((line['status'], line['variant'], line['sid_version'], line['engineering']),
                                 ('success', variant, LABEL_OF[variant], True))
                self.assertEqual(line['manifest_sha256'], file_sha256(manifest))
                self.assertIn(f'"manifest_sha256": "{file_sha256(manifest)}"', stdout)
                verified = self.out('verify')
                code, stdout, stderr = cli(verify_argv(manifest, self.files, verified, *extra))
                self.assertEqual((code, stderr), (0, ''))
                self.assertIn('"status": "verified"', stdout)
                self.assertEqual(json.loads((verified / 'verify.json').read_text())['status'], 'verified')
                self.assertEqual(sorted(p.name for p in verified.iterdir()), ['verify.json'])
                bundles.append(manifest)
        t3 = self.out('t3')
        inputs = synthetic_inputs()
        model = bhkm.fit(inputs['ids'], inputs['content'], branching=256, depth=3, seed=42, max_iter=10)
        OldWorld('SID_BHKM_B256D3_S42', 'bhkm', 256, True, model.ids, model.sids, inputs['content']).write(t3)
        base = bhkm.fit(inputs['ids'], inputs['content'], branching=16, depth=3, seed=42, max_iter=10)
        baseline = self.root / 'baseline.npz'
        np.savez(baseline, ids=base.ids, sids=base.sids)
        compared = self.out('compare')
        argv = ['compare', '--baseline-mapping', str(baseline), '--baseline-sha', file_sha256(baseline), '--content', str(self.files['content']),
                '--content-sha', self.files['content_sha'], '--catalog', str(self.files['catalog']), '--catalog-sha', self.files['catalog_sha'],
                '--output', str(compared), '--engineering']
        for manifest in bundles + [t3 / 'run_manifest.json']:
            argv += ['--bundle', str(manifest), '--sha', file_sha256(manifest)]
        code, stdout, stderr = cli(argv)
        self.assertEqual((code, stderr), (0, ''))
        self.assertEqual(json.loads(stdout)['labels'], ['T0', 'T3', 'S32', 'S64', 'S128', 'S512', 'S1024', 'S256D2'])
        self.assertEqual(sorted(p.name for p in compared.iterdir()), ['comparison.json', 'comparison.md'])

    def test_gate_failure_exits_3_and_leaves_only_failure_json(self):
        failing = {'passed': False, 'checks': [{'id': 'H1', 'ok': False, 'value': 1, 'threshold': 0}]}
        with mock.patch.object(run, 'gates_sweep', return_value=failing):
            output, code, stdout, _ = self.fit('S32')
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(stdout)['reasons'], ['H1'])
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
        record = json.loads((output / 'failure.json').read_text())
        self.assertEqual((record['schema'], record['task_id']), ('g-tok-02-failure-v1', 'G-TOK-02'))

    def test_expected_mismatch_exits_4_and_does_not_touch_the_bundle(self):
        output, code, _, _ = self.fit('S32')
        self.assertEqual(code, 0)
        manifest = output / 'run_manifest.json'
        before = {p.name: file_sha256(p) for p in output.iterdir()}
        audit = fitted('S32')['audit']['mapping']
        truth = {'leaves_count': audit['leaves']['count'] + 1, 'leaves_max': audit['leaves']['max'], 'vocab_size': audit['vocab_size'],
                 'items_needing_suffix': audit['items_needing_suffix'], 'positions_distinct': [p['distinct'] for p in audit['positions']]}
        verified = self.out('verify')
        with mock.patch.object(run, '_expected_for', return_value=truth):
            code, stdout, _ = cli(verify_argv(manifest, self.files, verified))
        self.assertEqual(code, 4)
        self.assertEqual(json.loads(stdout)['mismatches'][0]['field'], 'leaves_count')
        self.assertEqual(sorted(p.name for p in verified.iterdir()), ['expected_mismatch.json'])
        self.assertEqual({p.name: file_sha256(p) for p in output.iterdir()}, before)

    def test_structural_verify_failure_exits_1_and_writes_nothing(self):
        output, _, _, _ = self.fit('S32')
        verified = self.out('verify')
        argv = verify_argv(output / 'run_manifest.json', self.files, verified)
        argv[argv.index('--sha') + 1] = '0' * 64
        code, _, stderr = cli(argv)
        self.assertEqual(code, 1)
        self.assertIn('ERROR', stderr)
        self.assertFalse(verified.exists())

    def test_argument_errors_exit_1_before_anything_is_written(self):
        output, code, _, _ = self.fit('S32')
        self.assertEqual(code, 0)
        parent = self.parent_args()
        cases = {
            'existing output directory': (['--variant', 'S32'], output),
            'unknown variant': (['--variant', 'S99'], self.out('x')),
            'S256D2 without parent': (['--variant', 'S256D2'], self.out('x')),
            'parent on a depth-3 variant': (['--variant', 'S32'] + parent, self.out('x')),
            'parent manifest alone': (['--variant', 'S256D2', '--parent-manifest', parent[1]], self.out('x')),
            'device cuda': (['--variant', 'S32', '--device', 'cuda'], self.out('x')),
            'output parent missing': (['--variant', 'S32'], self.root / 'nope' / 'out'),
        }
        for name, (args, target) in cases.items():
            argv = ['fit'] + args + ['--config', str(self.files['config']), '--config-sha', self.files['config_sha'], '--content', str(self.files['content']),
                                     '--content-sha', self.files['content_sha'], '--catalog', str(self.files['catalog']),
                                     '--catalog-sha', self.files['catalog_sha'], '--output', str(target), '--engineering']
            existed = target.exists()
            with self.subTest(name):
                code, _, stderr = cli(argv)
                self.assertEqual(code, 1, stderr)
                self.assertEqual(target.exists(), existed)

    def test_missing_required_flag_and_unknown_flag_exit_1(self):
        for argv in (['fit', '--variant', 'S32'], ['fit', '--variant', 'S32', '--max-steps', '5'], ['verify'], ['compare'], ['nothing']):
            with self.subTest(argv):
                self.assertEqual(cli(argv)[0], 1)

    def test_formal_mode_refuses_synthetic_content(self):
        output, code, _, stderr = self.fit('S32', engineering=False)
        self.assertEqual(code, 1)
        self.assertIn('CONTENT_V1', stderr)
        self.assertFalse(output.exists())

    def test_wrong_config_file_is_refused(self):
        wrong = self.root / 'wrong.json'
        value = run.expected_config()
        value['variants']['S32']['seed'] = 7
        wrong.write_text(json.dumps(value, sort_keys=True, indent=1) + '\n')
        files = dict(self.files, config=wrong, config_sha=file_sha256(wrong))
        output = self.out('wrong')
        code, _, stderr = cli(fit_argv('S32', files, output))
        self.assertEqual(code, 1)
        self.assertIn('unapproved fit config', stderr)
        self.assertFalse(output.exists())

    def test_help_exits_zero_for_every_subcommand(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
        for command in ('fit', 'verify', 'compare'):
            with self.subTest(command):
                done = subprocess.run([sys.executable, '-B', '-m', 'vkgr.sid_tok2_run', command, '--help'], env=env, capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn('--output', done.stdout)
        done = subprocess.run([sys.executable, '-B', '-m', 'vkgr.sid_tok2_run', 'fit', '--help'], env=env, capture_output=True, text=True)
        self.assertIn('--parent-manifest', done.stdout)
        self.assertNotIn('--max-steps', done.stdout)


if __name__ == '__main__':
    unittest.main()
