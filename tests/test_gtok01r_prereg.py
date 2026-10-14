"""GTOK-01R FEAT-004: four-way consistency of the pre-registered repair constants.
Sources: (1) sid_rqvae_r.PREREG_R and PREREG_R_SELECT, (2) configs/gtok01r/g-tok-01r.json (prereg_r / select_r),
(3) the machine-readable block in docs/g_tok_01r_validation.md, (4) the independent literal copy below.
Changing any single leaf in any one of the sources (or removing / adding a key) makes the check fail.
"""
import contextlib
import copy
import io
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from vkgr import sid_rqvae_r as rr
ROOT = Path(__file__).resolve().parents[1]
DOC_REL = 'docs/g_tok_01r_validation.md'
CFG_REL = 'configs/gtok01r/g-tok-01r.json'
BEGIN, END = '<!-- gtok01r-prereg:begin -->', '<!-- gtok01r-prereg:end -->'
_RESET = {'reset_check_every': 100, 'dead_window': 500, 'reset_until': 270000, 'rng_reset_offset': 7919}
# Independent literal copy (JSON form: tuples are lists). Written down before any probe has run.
EXPECTED = {
    'prereg_r': {
        'arms': {
            'p0': {'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': False},
            'l1': {'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': True,
                   **_RESET},
            'l2': {'optimizer': 'adagrad', 'lr': 0.1, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': True,
                   **_RESET},
            'l3': {'optimizer': 'adagrad', 'lr': 0.025, 'initial_accumulator_value': 0.1, 'eps': 1e-7,
                   'reset': True, **_RESET},
            'l4': {'optimizer': 'adam', 'lr': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0,
                   'reset': True, **_RESET},
        },
        'arm_order': ['l1', 'l2', 'l3', 'l4'],
        'reset': dict(_RESET),
        'probe_steps': 30000,
        'probe_wall_cap_seconds': 3000,
        'usage_diag_every': 2000,
        'diag_subset_rows': 8192,
        'diag_subset_seed': 42,
        'diag_fine_steps': [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                            2500, 3000, 4000, 5000],
        'diag_every_after': 1000,
    },
    'select_r': {
        'variant_K': {'T1': 256, 'T2': 16},
        'p0_loss_windows': {'T1': [0.8199, 1.01, 0.8115], 'T2': [0.8754, 0.8116, 0.8115]},
        'p0_loss_tol': 0.001,
        'p0_usage_ref': [1, 1, 2],
        'p0_usage_slack': 1,
        'p0_final_usage_max': 3,
        'usage_log_steps': [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000,
                            26000, 28000, 30000],
        'a2_from_step': 20000,
        'a2_to_step': 30000,
        'd_points': [20000, 22000, 24000, 26000, 28000, 30000],
        'e_r2_min': 0.05,
        'i1_z_std_min': 0.001,
        'i1_dead_frac': 0.99,
        'i1_max_step': 3000,
        'i1_usage_max': 3,
        'i2_step': 30000,
        'i2_rel_tol': 0.001,
        'i4_norm_range': [0.99, 1.01],
        'old_fit_wall_seconds': {'T1': 4354, 'T2': 4302},
        'preflight_limit_seconds': 6000,
    },
}
def _listify(value):
    if isinstance(value, dict):
        return {key: _listify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listify(item) for item in value]
    return value
def doc_block(root):
    text = (Path(root) / DOC_REL).read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('doc must contain exactly one prereg block')
    block = text.split(BEGIN)[1].split(END)[0]
    match = re.search(r'```json\n(.*?)\n```', block, re.S)
    if match is None:
        raise ValueError('prereg block has no json fence')
    return json.loads(match.group(1))
def differences(want, got, path=''):
    """Paths where got differs from want: values, types, missing keys, extra keys, list lengths."""
    if type(want) is not type(got):
        return [path or '<root>']
    if isinstance(want, dict):
        out = []
        for key in sorted(set(want) | set(got)):
            if key not in want or key not in got:
                out.append(f'{path}.{key}')
            else:
                out += differences(want[key], got[key], f'{path}.{key}')
        return out
    if isinstance(want, list):
        if len(want) != len(got):
            return [path]
        out = []
        for index, (a, b) in enumerate(zip(want, got)):
            out += differences(a, b, f'{path}[{index}]')
        return out
    return [] if want == got else [path]
def _module_problems(module_values):
    return [f'module {p}' for p in differences(EXPECTED, module_values)]
def _config_problems(root):
    config = json.loads((Path(root) / CFG_REL).read_text())
    got = {'prereg_r': config.get('prereg_r'), 'select_r': config.get('select_r')}
    return [f'config {p}' for p in differences(EXPECTED, got)]
def _select_doc_problems(block):
    got = {'select_r': block.get('select_r')}
    return [f'doc {p}' for p in differences({'select_r': EXPECTED['select_r']}, got)]
def _prereg_doc_problems(block):
    got = {'prereg_r': block.get('prereg_r')}
    return [f'doc {p}' for p in differences({'prereg_r': EXPECTED['prereg_r']}, got)]
def check_consistency(root, module_values=None):
    """Human readable mismatches (empty = all four sources agree). module_values lets tests tamper source (1)."""
    if module_values is None:
        module_values = {'prereg_r': _listify(rr.PREREG_R), 'select_r': _listify(rr.PREREG_R_SELECT)}
    block = doc_block(root)
    problems = _module_problems(module_values) + _config_problems(root)
    problems += _prereg_doc_problems(block)
    problems += _select_doc_problems(block)
    if set(block) != {'prereg_r', 'select_r'}:
        problems.append('doc block keys')
    return problems
def leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaf_paths(item, prefix + (index,))
    else:
        yield prefix
def _get(value, path):
    for step in path:
        value = value[step]
    return value
def _tampered(value, path):
    """A copy with one leaf changed to something of the same kind but different."""
    out = copy.deepcopy(value)
    parent = _get(out, path[:-1])
    old = parent[path[-1]]
    if isinstance(old, bool):
        parent[path[-1]] = not old
    elif isinstance(old, str):
        parent[path[-1]] = old + 'x'
    else:
        parent[path[-1]] = old + 1 if isinstance(old, int) else old * 1.5 + 0.001
    return out
def _copy_tree(destination):
    for rel in (DOC_REL, CFG_REL):
        target = Path(destination) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
def _rewrite_doc(tmp, new_block):
    path = Path(tmp) / DOC_REL
    text = path.read_text()
    head, rest = text.split(BEGIN)
    tail = rest.split(END)[1]
    path.write_text(head + BEGIN + '\n```json\n' + json.dumps(new_block, indent=2, sort_keys=True) + '\n```\n'
                    + END + tail)
def _rewrite_config(tmp, section, new_value):
    path = Path(tmp) / CFG_REL
    config = json.loads(path.read_text())
    config[section] = new_value
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
ALL_LEAVES = [(section, path) for section in ('prereg_r', 'select_r') for path in leaf_paths(EXPECTED[section])]
class Gtok01rPrereg(unittest.TestCase):
    def test_four_sources_agree(self):
        self.assertEqual(check_consistency(ROOT), [])
    def test_literal_copy_covers_every_module_key(self):
        self.assertEqual(set(rr.PREREG_R), set(EXPECTED['prereg_r']))
        self.assertEqual(set(rr.PREREG_R_SELECT), set(EXPECTED['select_r']))
        self.assertGreaterEqual(len(ALL_LEAVES), 120)
    def test_literal_copy_is_not_vacuous_for_the_named_groups(self):
        names = {(section, path[0]) for section, path in ALL_LEAVES}
        for key in ('probe_wall_cap_seconds', 'reset', 'arms', 'probe_steps'):
            self.assertIn(('prereg_r', key), names)
        for key in ('p0_loss_windows', 'p0_usage_ref', 'p0_final_usage_max', 'preflight_limit_seconds',
                    'old_fit_wall_seconds', 'e_r2_min', 'd_points', 'i4_norm_range', 'a2_from_step'):
            self.assertIn(('select_r', key), names)
    def test_tamper_every_leaf_in_the_module_values_is_caught(self):
        for section, path in ALL_LEAVES:
            with self.subTest(section=section, path=path):
                values = {'prereg_r': _listify(rr.PREREG_R), 'select_r': _listify(rr.PREREG_R_SELECT)}
                values[section] = _tampered(values[section], path)
                self.assertNotEqual(check_consistency(ROOT, values), [])
    def test_tamper_every_leaf_in_the_config_is_caught(self):
        for section, path in ALL_LEAVES:
            with self.subTest(section=section, path=path), tempfile.TemporaryDirectory() as tmp:
                _copy_tree(tmp)
                config = json.loads((Path(tmp) / CFG_REL).read_text())
                _rewrite_config(tmp, section, _tampered(config[section], path))
                self.assertNotEqual(check_consistency(tmp), [])
    def test_tamper_every_leaf_in_the_doc_block_is_caught(self):
        for section, path in ALL_LEAVES:
            with self.subTest(section=section, path=path), tempfile.TemporaryDirectory() as tmp:
                _copy_tree(tmp)
                block = doc_block(tmp)
                block[section] = _tampered(block[section], path)
                _rewrite_doc(tmp, block)
                self.assertNotEqual(check_consistency(tmp), [])
    def test_tamper_removed_and_extra_keys_are_caught(self):
        for section in ('prereg_r', 'select_r'):
            for key in EXPECTED[section]:
                with self.subTest(section=section, removed=key), tempfile.TemporaryDirectory() as tmp:
                    _copy_tree(tmp)
                    block = doc_block(tmp)
                    del block[section][key]
                    _rewrite_doc(tmp, block)
                    self.assertNotEqual(check_consistency(tmp), [])
                with self.subTest(section=section, removed_cfg=key), tempfile.TemporaryDirectory() as tmp:
                    _copy_tree(tmp)
                    config = json.loads((Path(tmp) / CFG_REL).read_text())
                    del config[section][key]
                    _rewrite_config(tmp, section, config[section])
                    self.assertNotEqual(check_consistency(tmp), [])
            with self.subTest(section=section, extra='doc'), tempfile.TemporaryDirectory() as tmp:
                _copy_tree(tmp)
                block = doc_block(tmp)
                block[section]['unapproved'] = 1
                _rewrite_doc(tmp, block)
                self.assertNotEqual(check_consistency(tmp), [])
            with self.subTest(section=section, extra='config'), tempfile.TemporaryDirectory() as tmp:
                _copy_tree(tmp)
                config = json.loads((Path(tmp) / CFG_REL).read_text())
                config[section]['unapproved'] = 1
                _rewrite_config(tmp, section, config[section])
                self.assertNotEqual(check_consistency(tmp), [])
    def test_type_change_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            _copy_tree(tmp)
            block = doc_block(tmp)
            block['prereg_r']['probe_steps'] = 30000.0
            _rewrite_doc(tmp, block)
            self.assertNotEqual(check_consistency(tmp), [])
    def test_missing_or_duplicated_block_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _copy_tree(tmp)
            path = Path(tmp) / DOC_REL
            text = path.read_text()
            path.write_text(text.replace(BEGIN, ''))
            with self.assertRaises(ValueError):
                check_consistency(tmp)
            path.write_text(text + '\n' + BEGIN + '\n```json\n{}\n```\n' + END + '\n')
            with self.assertRaises(ValueError):
                check_consistency(tmp)
    def test_config_sections_match_the_module_exactly(self):
        config = json.loads((ROOT / CFG_REL).read_text())
        self.assertEqual(differences(_listify(rr.PREREG_R), config['prereg_r']), [])
        self.assertEqual(differences(_listify(rr.PREREG_R_SELECT), config['select_r']), [])
class DocText(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / DOC_REL).read_text()
    def test_public_hygiene(self):
        for marker in ('/Us' + 'ers/', '/work' + 'space', 'Hello' + 'Bike', '哈' + '啰', '/pri' + 'vate/', 'gh' + 'p_',
                       'github' + '_pat_', 'AK' + 'IA', 'BEGIN ' + 'PRIVATE'):
            self.assertNotIn(marker, self.text)
        self.assertIsNone(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', self.text))
        self.assertLess(len(self.text.encode()), 100000)
    def test_required_sections_and_claim_boundary(self):
        for heading in ('### 1.', '### 2.', '### 3.', '### 4.', '### 5.', '### 6.'):
            self.assertIn(heading, self.text)
        for phrase in ('偏离', 'ESCALATE', 'STOP', 'INVALID_CONTROL', 'seed 42', '30,000', '不是 TIGER 复现',
                       '以偏离冻结卡的优化器设置训练', 'experiment_log'):
            self.assertIn(phrase, self.text)
    def test_no_result_numbers_section(self):
        section = self.text.split('### 6.')[1].split('\n### ')[0]
        self.assertIn('experiment_log', section)
        self.assertEqual(re.findall(r'\d', section), [])
    def test_reproduction_commands_use_placeholders_only(self):
        self.assertIn('<WORKTREE>', self.text)
        self.assertIn('--override-arm', self.text)
        for sub in ('probe', 'select', 'fit', 'verify'):
            self.assertIn('sid_tok_run_r ' + sub, self.text)
def _killed(patches, test_names):
    """True when the named tests fail by assertion (not by error) while the patches are active."""
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromNames(
        [f'{__name__}.Gtok01rPrereg.{name}' for name in test_names]))
    stream = io.StringIO()
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return (not result.wasSuccessful()) and bool(result.failures) and not result.errors
class MutantF4(unittest.TestCase):
    """F4: the select-rule numerics (section 2 criteria) are left out of the doc comparison."""
    def test_f4_skipping_the_doc_select_comparison_is_killed_by_the_doc_tamper_test(self):
        self.assertTrue(_killed([mock.patch(f'{__name__}._select_doc_problems', lambda block: [])],
                                ['test_tamper_every_leaf_in_the_doc_block_is_caught']))
    def test_f4_skipping_the_module_comparison_is_killed(self):
        self.assertTrue(_killed([mock.patch(f'{__name__}._module_problems', lambda values: [])],
                                ['test_tamper_every_leaf_in_the_module_values_is_caught']))
    def test_f4_skipping_the_config_comparison_is_killed(self):
        self.assertTrue(_killed([mock.patch(f'{__name__}._config_problems', lambda root: [])],
                                ['test_tamper_every_leaf_in_the_config_is_caught']))
    def test_f4_skipping_the_doc_prereg_comparison_is_killed(self):
        self.assertTrue(_killed([mock.patch(f'{__name__}._prereg_doc_problems', lambda block: [])],
                                ['test_tamper_every_leaf_in_the_doc_block_is_caught']))
    def test_f4_leaf_walk_that_skips_select_r_is_killed_by_the_coverage_test(self):
        short = [(s, p) for s, p in ALL_LEAVES if s != 'select_r']
        self.assertTrue(_killed([mock.patch(f'{__name__}.ALL_LEAVES', short)],
                                ['test_literal_copy_covers_every_module_key']))
if __name__ == '__main__':
    unittest.main()
