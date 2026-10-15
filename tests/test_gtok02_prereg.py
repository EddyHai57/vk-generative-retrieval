"""GTOK-02: five-way consistency of the pre-registered constants.

Sources: (1) ``sid_tok2_run.expected_config()``, (2) configs/gtok02/g-tok-02.json, (3) the 18 G training configs,
(4) the machine-readable block in docs/g_tok_02_validation.md, (5) the independent literal table below (it carries every ``expected``
integer of the six variants). Changing any single source, or the table, makes the check fail.
"""
import importlib.util
import inspect
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from vkgr import sid_tok2_run as run
from vkgr import sid_tok_audit as au

ROOT = Path(__file__).resolve().parents[1]
DOC_REL = 'docs/g_tok_02_validation.md'
FIT_REL = 'configs/gtok02/g-tok-02.json'
ANCHOR_REL = 'configs/batch1/g-hist-v2-l50-s42.json'
BEGIN, END = '<!-- gtok02-prereg:begin -->', '<!-- gtok02-prereg:end -->'
T3_SHA = 'a7af78a3fc04f0f0d174b983987c34d7efa754edd88a05d1350cdadd06fd47b8'
STEMS = {'S32': 'b32d3', 'S64': 'b64d3', 'S128': 'b128d3', 'S512': 'b512d3', 'S1024': 'b1024d3', 'S256D2': 'b256d2'}
SEEDS = (42, 2024, 2025)
# Independent literal copy (design 3.2). Do not import these from the code under test.
LITERAL_VARIANTS = {
    'S32': {'sid_version': 'SID_BHKM_B32D3_S42', 'kind': 'bhkm', 'branching': 32, 'depth': 3, 'max_iter': 10, 'seed': 42,
            'expected': {'items_needing_suffix': 87647, 'leaves_count': 32768, 'leaves_max': 4, 'positions_distinct': [32, 32, 32, 4], 'vocab_size': 102}},
    'S64': {'sid_version': 'SID_BHKM_B64D3_S42', 'kind': 'bhkm', 'branching': 64, 'depth': 3, 'max_iter': 10, 'seed': 42,
            'expected': {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1, 'positions_distinct': [64, 64, 30, 1], 'vocab_size': 195}},
    'S128': {'sid_version': 'SID_BHKM_B128D3_S42', 'kind': 'bhkm', 'branching': 128, 'depth': 3, 'max_iter': 10, 'seed': 42,
             'expected': {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1, 'positions_distinct': [128, 128, 8, 1], 'vocab_size': 387}},
    'S512': {'sid_version': 'SID_BHKM_B512D3_S42', 'kind': 'bhkm', 'branching': 512, 'depth': 3, 'max_iter': 10, 'seed': 42,
             'expected': {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1, 'positions_distinct': [512, 236, 1, 1], 'vocab_size': 1539}},
    'S1024': {'sid_version': 'SID_BHKM_B1024D3_S42', 'kind': 'bhkm', 'branching': 1024, 'depth': 3, 'max_iter': 10, 'seed': 42,
              'expected': {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1, 'positions_distinct': [1024, 118, 1, 1], 'vocab_size': 3075}},
    'S256D2': {'sid_version': 'SID_BHKM_B256D2_S42', 'kind': 'derive_depth2', 'branching': 256, 'depth': 2,
               'parent': {'manifest_sha256': T3_SHA, 'sid_version': 'SID_BHKM_B256D3_S42', 'variant': 'T3'},
               'expected': {'items_needing_suffix': 54879, 'leaves_count': 65536, 'leaves_max': 2, 'positions_distinct': [256, 256, 1, 2], 'vocab_size': 772}},
}
LITERAL_FIT = {'schema': 'g-tok-02-fit-v1', 'task_id': 'G-TOK-02',
               'content': {'feature_array_sha256': '766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84', 'dim': 64},
               'variants': LITERAL_VARIANTS}
LITERAL_JUDGEMENT = {'threshold_hits': 6, 'validation_queries': 2520, 'seeds': [42, 2024, 2025], 'near_hits': 10, 'top_k': 2}
GATE_IDS = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'B1']


def doc_block(root):
    text = (Path(root) / DOC_REL).read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('doc must contain exactly one prereg block')
    match = re.search(r'```json\n(.*?)\n```', text.split(BEGIN)[1].split(END)[0], re.S)
    if match is None:
        raise ValueError('prereg block has no json fence')
    return json.loads(match.group(1))


def train_paths(root):
    return {(name, seed): Path(root) / f'configs/gtok02/g-tok2-bhkm-{stem}-s{seed}.json' for name, stem in STEMS.items() for seed in SEEDS}


def check_consistency(root, literal=LITERAL_FIT):
    """Human readable mismatches; empty means all five sources agree."""
    root = Path(root)
    problems = []
    fit = json.loads((root / FIT_REL).read_text())
    block = doc_block(root)
    for name, source in (('expected_config()', run.expected_config()), ('fit-config file', fit), ('doc block', block['fit_config'])):
        if source != literal:
            problems.append(f'{name} differs from the literal table')
    if block['fit_config'] != fit:
        problems.append('doc fit_config != fit-config file')
    if block['judgement']['threshold_hits'] != LITERAL_JUDGEMENT['threshold_hits'] or block['judgement']['validation_queries'] != 2520 \
            or block['judgement']['seeds'] != [42, 2024, 2025] or block['judgement']['near_hits'] != 10 or block['judgement']['top_k'] != 2:
        problems.append('doc judgement block')
    if block['gates']['groups'] != {'H': ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'], 'B': ['B1']} or block['gates']['balance_tolerance'] != 1 \
            or block['gates']['b256d2_balance_basis'] != 'raw3' or block['gates']['not_applicable'] != ['R1', 'R2', 'R3', 'R4']:
        problems.append('doc gates block')
    if block['exit_codes'] != {'success': 0, 'argument_or_validation_error': 1, 'gate_failure': 3, 'expected_mismatch': 4}:
        problems.append('doc exit codes')
    anchor = json.loads((root / ANCHOR_REL).read_text())
    expected_runs = {name: {'sid': literal['variants'][name]['sid_version'],
                            'configs': [f'configs/gtok02/g-tok2-bhkm-{stem}-s{seed}.json' for seed in SEEDS]} for name, stem in STEMS.items()}
    if block['g_runs'] != expected_runs:
        problems.append('doc g_runs')
    for (name, seed), path in train_paths(root).items():
        cfg = json.loads(path.read_text())
        if cfg.get('sid') != literal['variants'][name]['sid_version'] or cfg.get('seed') != seed:
            problems.append(f'{path.name}: sid/seed')
        if cfg.get('schema') != 'g-tok2-run-v1' or cfg.get('task_id') != 'G-TOK-02':
            problems.append(f'{path.name}: schema/task_id')
        differing = sorted(key for key in set(cfg) | set(anchor) if cfg.get(key) != anchor.get(key))
        allowed = ['schema', 'sid', 'task_id'] + ([] if seed == 42 else ['seed'])
        if differing != sorted(allowed):
            problems.append(f'{path.name} differs from the anchor in {differing}')
    return problems


def copy_tree(destination):
    rels = [DOC_REL, FIT_REL, ANCHOR_REL] + [str(p.relative_to(ROOT)) for p in train_paths(ROOT).values()]
    for rel in rels:
        target = Path(destination) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)


class Gtok02Prereg(unittest.TestCase):
    def test_five_sources_agree(self):
        self.assertEqual(check_consistency(ROOT), [])

    def test_the_literal_table_is_internally_consistent(self):
        for name, variant in LITERAL_VARIANTS.items():
            expected = variant['expected']
            self.assertEqual(expected['vocab_size'], 2 + 3 * variant['branching'] + expected['leaves_max'], name)
            self.assertEqual(expected['items_needing_suffix'], 120415 - expected['leaves_count'], name)
            self.assertEqual(len(expected['positions_distinct']), 4)

    def test_gate_ids_are_h1_to_h6_and_b1_only(self):
        from tests.test_sid_v2s import SMALL_N, small_ids, unit_rows
        from vkgr import sid as bhkm
        ids, content = small_ids(), unit_rows(SMALL_N)
        sids = bhkm.fit(ids, content, branching=32, depth=3, seed=42, max_iter=10).sids
        audit = au.audit_mapping(ids, sids, content, branching=32, kind='bhkm')
        result = run.gates_sweep(audit, expected_content_sha256=au.feature_digest(content))
        self.assertEqual([c['id'] for c in result['checks']], GATE_IDS)
        self.assertEqual(au.BALANCE_TOLERANCE, 1)

    def test_gates_sweep_calls_the_frozen_gate_function_and_copies_none_of_it(self):
        source = inspect.getsource(run.gates_sweep)
        self.assertIn('au.gates(', source)
        self.assertNotIn('_check(', source)
        self.assertNotIn('BALANCE_TOLERANCE', source)
        self.assertNotIn('rqvae', source.replace("'bhkm'", ''))  # R1-R4 are not reachable from here

    def test_judgement_thresholds_in_the_scripts_match_the_doc(self):
        def load(name):
            spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        curve, select = load('gtok02_curve'), load('gtok02_select')
        block = doc_block(ROOT)['judgement']
        self.assertEqual((curve.THRESHOLD_HITS, select.N_QUERIES, select.NEAR_HITS, select.SEEDS, sorted(curve.T3_HITS)),
                         (block['threshold_hits'], block['validation_queries'], block['near_hits'], block['seeds'][1:], block['seeds']))
        self.assertEqual(len(select.select({'B16': {'recall50': 0.1, 'ndcg50': 0.1, 'verified': False}})['ranking']), 0)

    def test_negative_changed_fit_config_constant_is_caught(self):
        edits = (('S32', ['expected', 'leaves_count'], 32767), ('S64', ['expected', 'positions_distinct', 2], 31),
                 ('S128', ['seed'], 43), ('S512', ['max_iter'], 11), ('S1024', ['branching'], 1023), ('S256D2', ['depth'], 3),
                 ('S256D2', ['parent', 'manifest_sha256'], '0' * 64), ('S256D2', ['expected', 'items_needing_suffix'], 54880),
                 ('S32', ['sid_version'], 'SID_BHKM_B32D3_S43'))
        for name, path, value in edits:
            with self.subTest(name=name, path=path), tempfile.TemporaryDirectory() as tmp:
                copy_tree(tmp)
                target = Path(tmp) / FIT_REL
                data = json.loads(target.read_text())
                node = data['variants'][name]
                for part in path[:-1]:
                    node = node[part]
                node[path[-1]] = value
                target.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
                self.assertNotEqual(check_consistency(tmp), [])

    def test_negative_changed_doc_constant_is_caught(self):
        for old, new in (('"near_hits": 10', '"near_hits": 11'), ('"threshold_hits": 6', '"threshold_hits": 5'),
                         ('"balance_tolerance": 1', '"balance_tolerance": 2'), ('"b256d2_balance_basis": "raw3"', '"b256d2_balance_basis": "full"'),
                         ('"leaves_count": 65536', '"leaves_count": 65537'), ('"vocab_size": 3075', '"vocab_size": 3076'),
                         ('"gate_failure": 3', '"gate_failure": 2'), ('"top_k": 2', '"top_k": 1')):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as tmp:
                copy_tree(tmp)
                path = Path(tmp) / DOC_REL
                text = path.read_text()
                self.assertIn(old, text)
                path.write_text(text.replace(old, new, 1))
                self.assertNotEqual(check_consistency(tmp), [])

    def test_negative_changed_train_config_is_caught(self):
        for (name, seed), path in train_paths(ROOT).items():
            for key, value in (('sid', 'SID_BHKM_B16D3_S42'), ('learning_rate', 0.001), ('history_len', 20), ('seed', 7),
                               ('task_id', 'G-TOK-01'), ('schema', 'g-tok-run-v1')):
                with self.subTest(name=name, seed=seed, key=key), tempfile.TemporaryDirectory() as tmp:
                    copy_tree(tmp)
                    target = Path(tmp) / path.relative_to(ROOT)
                    data = json.loads(target.read_text())
                    data[key] = value
                    target.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
                    self.assertNotEqual(check_consistency(tmp), [])

    def test_negative_changed_literal_table_is_caught(self):
        import copy
        for name, path in (('S32', ['expected', 'vocab_size']), ('S64', ['expected', 'positions_distinct', 3]), ('S256D2', ['expected', 'leaves_max']),
                           ('S1024', ['branching'])):
            with self.subTest(name=name, path=path):
                broken = copy.deepcopy(LITERAL_FIT)
                node = broken['variants'][name]
                for part in path[:-1]:
                    node = node[part]
                node[path[-1]] += 1
                self.assertNotEqual(check_consistency(ROOT, broken), [])

    def test_negative_missing_block_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            path = Path(tmp) / DOC_REL
            path.write_text(path.read_text().replace(BEGIN, ''))
            with self.assertRaises(ValueError):
                check_consistency(tmp)

    def test_doc_results_section_has_no_numbers_and_no_local_paths(self):
        text = (ROOT / DOC_REL).read_text()
        section = text.split('### 11. 结果')[1].split('\n### ')[0]
        self.assertIn('experiment_log', section)
        self.assertEqual(re.findall(r'\d', section), [])
        # split literals: this file must itself pass the repository hygiene scan
        for marker in ('/Us' + 'ers/', '/work' + 'space', 'Hello' + 'Bike', '/pri' + 'vate/', '/ho' + 'me/'):
            self.assertNotIn(marker, text)
        self.assertNotIn('Eddy', text)
        self.assertIsNone(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', text))

    def test_doc_states_the_b256d2_ruling_premise_and_the_reading_rules(self):
        text = (ROOT / DOC_REL).read_text()
        for needle in ('裁定权在项目负责人', 'HELD', '补零前的三列', 'E 规则', '赢家诅咒', '假阳性率', '不是碰撞对照', 'violations > 0',
                       '不等于', '256 是峰值'):
            self.assertIn(needle, text)


if __name__ == '__main__':
    unittest.main()
