"""Isolated bounded engineering mutations; no remote or scientific jobs."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = {
    'readonly-preflight-steps-admitted': ('if not read_only:', 'if True:'),
    'validated-job-raw-sha-unbound': ('return hashlib.sha256(raw).hexdigest()', 'return backend.file_sha(job)'),
    'template-drift-admitted': ("value.get(key) is not None and backend.encoded(value[key]) != backend.encoded(actual)", 'False'),
    'original-config-bytes-unbound': ("hashlib.sha256(raw).hexdigest() != digest", 'False'),
    'original-assembly-uncompared': ("identity(adapted) != identity(reference)", 'False'),
    'scoring-uncompared': ("left != right", 'False'),
    'end-job-unbound': ("check_job() != job_sha", 'False'),
    'loss-summary-forged': ("'record': freeze(record)", "'record': freeze({**record, 'loss': 0.})"),
    'own-job-unbound': ("any(backend.encoded(actual.get(k)) != backend.encoded(v) for k, v in expected.items())", 'False'),
}


def main():
    run = Path(tempfile.mkdtemp(prefix='p-full-engineering-mutants-')); results = []
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    names += ['analysis/p_full_engineering.py', 'tests/test_p_full_engineering.py']
    for index, (name, (before, after)) in enumerate(MUTATIONS.items()):
        tree = run / name; tree.mkdir()
        for rel in sorted(set(names)):
            source = ROOT / rel
            if source.is_file():
                target = tree / rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        path = tree / 'analysis/p_full_engineering.py'; raw = path.read_text()
        if before not in raw: raise ValueError('mutation source missing: ' + name)
        path.write_text(raw.replace(before, after))
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(tree / 'src') + os.pathsep + str(tree),
               'P_FULL_TEST_GIT_ROOT': str(ROOT), 'PYTHONPYCACHEPREFIX': str(run / ('unused-cache-' + str(index)))}
        got = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', 'tests/test_p_full_engineering.py', '-k', 'not helper_real_launcher_chain',
                              '--basetemp', str(run / ('pytest-' + str(index)))], cwd=tree, env=env, capture_output=True, text=True)
        (run / (name + '.log')).write_text(got.stdout + got.stderr)
        killed = got.returncode == 1 and 'FAILED tests/test_p_full_engineering.py::' in got.stdout
        results.append({'name': name, 'killed': killed, 'returncode': got.returncode})
    receipt = {'isolated': True, 'copy_per_mutant': True, 'python_B': True, 'no_bytecode': True,
               'unique_cache_prefix': True, 'results': results}
    (run / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps(receipt, sort_keys=True)); print('evidence_root=' + str(run))
    return 0 if all(row['killed'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
