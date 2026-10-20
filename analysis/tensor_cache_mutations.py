"""Local synthetic mutation campaign, isolated fresh copies and cache prefixes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


MUTATIONS = {
    'padding_width': ('width = max(v[0].shape[1] for v in values)', 'width = max(v[0].shape[1] for v in values) + 1'),
    'tail_batch': ('source = torch.zeros((len(rows), width)', 'source = torch.zeros((len(rows) + 1, width)'),
    'cursor_order': ('for row in rows:\n', 'for row in reversed(rows):\n'),
    'rng_consumption': ('self.hits += 1', "__import__('random').random(); self.hits += 1"),
    'map_key_omission': ('[self.identity, kind, row, dependency]', '[self.identity, kind, row]'),
    'input_hash_omission': ('return _digest(values)', "values.pop('input'); return _digest(values)"),
    'sid_hash_omission': ('return _digest(values)', "values.pop('sid'); return _digest(values)"),
    'config_hash_omission': ('return _digest(values)', "values.pop('config'); return _digest(values)"),
    'source_hash_omission': ('return _digest(values)', "values.pop('source'); return _digest(values)"),
    'runtime_hash_omission': ('return _digest(values)', "values.pop('runtime'); return _digest(values)"),
    'constructor_hash_omission': ('return _digest(values)', "values.pop('constructor'); return _digest(values)"),
    'negative_passthrough': ('globals_copy = dict(original.__globals__)', "globals_copy = dict(original.__globals__)\n    old = globals_copy['_negative_inputs']\n    globals_copy['_negative_inputs'] = lambda rng: (['changed'], *old(rng)[1:])"),
    'mask_passthrough': ('globals_copy = dict(original.__globals__)', "globals_copy = dict(original.__globals__)\n    old = globals_copy['_negative_inputs']\n    globals_copy['_negative_inputs'] = lambda rng: ((v := old(rng))[0], __import__('torch').tensor([True, False]), v[2])"),
    'logq_passthrough': ('globals_copy = dict(original.__globals__)', "globals_copy = dict(original.__globals__)\n    old = globals_copy['_negative_inputs']\n    globals_copy['_negative_inputs'] = lambda rng: ((v := old(rng))[0], v[1], __import__('numpy').array([0.5, 0.5]))"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    runs = []
    for label, mutation in [('baseline', None), *MUTATIONS.items()]:
        sandbox = Path(tempfile.mkdtemp(prefix='vkgr-tensor-mutation-' + label + '-'))
        (sandbox/'src/vkgr').mkdir(parents=True); (sandbox/'tests').mkdir()
        (sandbox/'src/vkgr/__init__.py').touch()
        source = (root/'src/vkgr/tensor_cache.py').read_text()
        if mutation:
            before, after = mutation
            if before not in source: raise ValueError('mutation anchor absent: ' + label)
            source = source.replace(before, after)
        (sandbox/'src/vkgr/tensor_cache.py').write_text(source)
        shutil.copy2(root/'tests/test_tensor_cache.py', sandbox/'tests/test_tensor_cache.py')
        env = {**os.environ, 'PYTHONPATH': str(sandbox/'src'), 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTHONPYCACHEPREFIX': str(sandbox/'unique-pycache'), 'VKGR_CACHE_PREFIX': str(sandbox/'unique-data-cache')}
        with (output/(label+'.log')).open('xb') as stream:
            run = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                 cwd=sandbox, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=90)
        text = (output/(label+'.log')).read_text()
        # A syntax/import failure or killed interpreter does not count as a kill.
        assertions = 'FAIL:' in text and 'AssertionError' in text
        target = {'negative_passthrough': 'negative input passthrough',
                  'mask_passthrough': 'mask input passthrough',
                  'logq_passthrough': 'logq input passthrough'}.get(label)
        target_killed = target in text if target else assertions
        good = run.returncode == 0 if mutation is None else run.returncode != 0 and assertions and target_killed
        runs.append({'case': label, 'returncode': run.returncode, 'assertion_killed': assertions,
                     'accepted': good, 'fresh_copy': True, 'bytecode_disabled': True,
                     'unique_cache_prefix': True, 'target_assertion': target, 'target_killed': target_killed})
    receipt = {'schema': 'tensor-cache-isolated-mutations-v1', 'results': runs, 'pass': all(r['accepted'] for r in runs)}
    with (output/'receipt.json').open('x') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt)); return 0 if receipt['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
