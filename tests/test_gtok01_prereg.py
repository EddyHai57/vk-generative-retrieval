"""GTOK-01 FEAT-005: four-way consistency of the pre-registered constants.
Sources: (1) sid_rqvae.PREREG, (2) configs/gtok01/g-tok-01.json, (3) the three G run configs,
(4) the machine-readable block in docs/g_tok_01_validation.md. The literals below are a fifth, independent
copy: changing any single source (or this table) makes the check fail.
"""
import copy
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from vkgr import sid_rqvae as rq

ROOT = Path(__file__).resolve().parents[1]
DOC_REL = 'docs/g_tok_01_validation.md'
FIT_REL = 'configs/gtok01/g-tok-01.json'
ANCHOR_REL = 'configs/batch1/g-hist-v2-l50-s42.json'
BEGIN, END = '<!-- gtok01-prereg:begin -->', '<!-- gtok01-prereg:end -->'

# Independent literal copy of the frozen constants (task description of FEAT-005 / plan section 2.1).
EXPECTED_RQVAE = {
    'max_steps': 300000, 'wall_cap_seconds': 7200, 'batch_size': 1024, 'lr': 0.4,
    'initial_accumulator_value': 0.1, 'eps': 1e-07, 'beta': 0.25, 'levels': 3, 'latent_dim': 32,
    'encoder_hidden': [512, 256, 128], 'usage_gate_num_den': [4, 5], 'kmeans_iters': 10, 'seed': 42,
    'content_dim': 64, 'optimizer': 'adagrad',
}
EXPECTED_VARIANTS = {
    'T1': {'sid_version': 'SID_RQVAE_B256D3_S42', 'kind': 'rqvae', 'codebook_size': 256},
    'T2': {'sid_version': 'SID_RQVAE_B16D3_S42', 'kind': 'rqvae', 'codebook_size': 16},
    'T3': {'sid_version': 'SID_BHKM_B256D3_S42', 'kind': 'bhkm', 'branching': 256, 'depth': 3,
           'max_iter': 10, 'seed': 42},
}
EXPECTED_RUNS = {
    'T1': 'configs/gtok01/g-tok-t1-rqvae-b256-s42.json',
    'T2': 'configs/gtok01/g-tok-t2-rqvae-b16-s42.json',
    'T3': 'configs/gtok01/g-tok-t3-bhkm-b256-s42.json',
}


def _listify(value):
    if isinstance(value, tuple):
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


def check_consistency(root):
    """Return a list of human readable mismatches (empty = all four sources agree)."""
    root = Path(root)
    problems = []
    fit = json.loads((root / FIT_REL).read_text())
    block = doc_block(root)
    prereg = {key: _listify(value) for key, value in rq.PREREG.items()}
    rqvae = fit['rqvae']
    for key, want in EXPECTED_RQVAE.items():
        for name, source in (('PREREG', prereg), ('fit-config', rqvae), ('doc', block['fit_config']['rqvae'])):
            if source.get(key) != want or type(source.get(key)) is not type(want):
                problems.append(f'{name}.{key}={source.get(key)!r} want {want!r}')
    for key, value in prereg.items():
        if rqvae.get(key) != value:
            problems.append(f'PREREG vs fit-config: {key}')
        if block['fit_config']['rqvae'].get(key) != value:
            problems.append(f'PREREG vs doc: {key}')
    if block['fit_config'] != fit:
        problems.append('doc fit_config != fit-config file')
    for name, source in (('fit-config', fit['variants']), ('doc', block['fit_config']['variants'])):
        if source != EXPECTED_VARIANTS:
            problems.append(f'{name}.variants differ from the pre-registered table')
    if block.get('gates', {}).get('usage_gate_num_den') != [4, 5]:
        problems.append('doc gates.usage_gate_num_den')
    if block.get('gates', {}).get('required_steps') != 300000:
        problems.append('doc gates.required_steps')
    if block.get('gates', {}).get('wall_cap_seconds') != 7200:
        problems.append('doc gates.wall_cap_seconds')
    if block.get('g_runs') != {tag: {'config': rel, 'sid': EXPECTED_VARIANTS[tag]['sid_version']}
                               for tag, rel in EXPECTED_RUNS.items()}:
        problems.append('doc g_runs')
    anchor = json.loads((root / ANCHOR_REL).read_text())
    for tag, rel in EXPECTED_RUNS.items():
        run = json.loads((root / rel).read_text())
        if run.get('sid') != fit['variants'][tag]['sid_version']:
            problems.append(f'{tag} run config sid')
        if run.get('schema') != 'g-tok-run-v1' or run.get('task_id') != 'G-TOK-01':
            problems.append(f'{tag} run config schema/task_id')
        differing = sorted(key for key in set(run) | set(anchor) if run.get(key) != anchor.get(key))
        if differing != ['schema', 'sid', 'task_id']:
            problems.append(f'{tag} run config differs from anchor in {differing}')
    return problems


def _copy_tree(destination):
    for rel in [DOC_REL, FIT_REL, ANCHOR_REL] + list(EXPECTED_RUNS.values()):
        target = Path(destination) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)


class Gtok01Prereg(unittest.TestCase):
    def test_four_sources_agree(self):
        self.assertEqual(check_consistency(ROOT), [])

    def test_fifth_literal_copy_covers_every_prereg_key(self):
        self.assertEqual(set(rq.PREREG) - set(EXPECTED_RQVAE), {'log_every', 'eval_every', 'check_every',
                                                                'probe_from', 'probe_step'})

    def test_negative_changed_fit_config_constant_is_caught(self):
        for key, value in (('max_steps', 299999), ('lr', 0.3), ('beta', 0.5), ('seed', 43), ('batch_size', 512),
                           ('eps', 1e-6), ('initial_accumulator_value', 0.0), ('kmeans_iters', 9),
                           ('latent_dim', 16), ('levels', 4), ('wall_cap_seconds', 3600),
                           ('encoder_hidden', [512, 256, 64]), ('usage_gate_num_den', [3, 4])):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                _copy_tree(tmp)
                path = Path(tmp) / FIT_REL
                value_json = json.loads(path.read_text())
                value_json['rqvae'][key] = value
                path.write_text(json.dumps(value_json, indent=1, sort_keys=True) + '\n')
                self.assertNotEqual(check_consistency(tmp), [])

    def test_negative_changed_doc_constant_is_caught(self):
        for old, new in (('"max_steps": 300000', '"max_steps": 300001'), ('"lr": 0.4', '"lr": 0.41'),
                         ('"codebook_size": 256', '"codebook_size": 128'), ('"max_iter": 10', '"max_iter": 11'),
                         ('"wall_cap_seconds": 7200', '"wall_cap_seconds": 7201'),
                         ('"required_steps": 300000', '"required_steps": 1')):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as tmp:
                _copy_tree(tmp)
                path = Path(tmp) / DOC_REL
                text = path.read_text()
                self.assertIn(old, text)
                path.write_text(text.replace(old, new, 1))
                self.assertNotEqual(check_consistency(tmp), [])

    def test_negative_changed_run_config_is_caught(self):
        for tag, rel in EXPECTED_RUNS.items():
            for key, value in (('sid', 'SID_BHKM_B16D3_S42'), ('learning_rate', 0.001), ('history_len', 20)):
                with self.subTest(tag=tag, key=key), tempfile.TemporaryDirectory() as tmp:
                    _copy_tree(tmp)
                    path = Path(tmp) / rel
                    value_json = json.loads(path.read_text())
                    value_json[key] = value
                    path.write_text(json.dumps(value_json, indent=1, sort_keys=True) + '\n')
                    self.assertNotEqual(check_consistency(tmp), [])

    def test_negative_missing_block_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _copy_tree(tmp)
            path = Path(tmp) / DOC_REL
            path.write_text(path.read_text().replace(BEGIN, ''))
            with self.assertRaises(ValueError):
                check_consistency(tmp)

    def test_doc_results_section_has_no_numbers_and_no_local_paths(self):
        text = (ROOT / DOC_REL).read_text()
        section = text.split('### 9. 结果')[1].split('\n### ')[0]
        self.assertIn('experiment_log', section)
        self.assertEqual(re.findall(r'\d', section), [])
        for marker in ('/Us' + 'ers/', '/work' + 'space', 'Hello' + 'Bike', '/pri' + 'vate/'):
            self.assertNotIn(marker, text)
        for heading in ('### 6.', '### 10.'):
            self.assertIn(heading, text)


if __name__ == '__main__':
    unittest.main()
