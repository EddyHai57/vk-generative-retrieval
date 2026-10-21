"""Synthetic deployed-layout check using the real pinned launcher, never submit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[2]
base = repo / 'cache/g-preflight-retry1' / ('launcher-chain-' + str(time.time_ns()))
base.mkdir(parents=True)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE, text=True).strip()


def clone(name, ref):
    target = base / name
    subprocess.run(['git', 'clone', '--quiet', '--no-checkout', '--shared', str(repo), str(target)],
                   check=True, capture_output=True)
    git(target, 'checkout', '--quiet', '--detach', ref)
    return target


engine = clone('engine', '87159f720afaf42be6252223442ece56a20e0ef9')
ops_code = clone('ops-code', '5840e6b686ec9651314a9e28a1a2c0befcc05158')
runner = clone('runner', git(repo, 'rev-parse', 'HEAD'))
shutil.copyfile(repo / 'analysis/g_decode_preflight_retry.py', runner / 'analysis/g_decode_preflight_retry.py')
git(runner, 'add', 'analysis/g_decode_preflight_retry.py')
# Commit only when the parent HEAD does not already include these exact bytes.
if git(runner, 'diff', '--cached', '--name-only'):
    git(runner, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '--quiet', '-m', 'synthetic retry transport fixture')

sys.path[:0] = [str(engine), str(engine / 'src')]
from analysis import g_decode_timing as original
from analysis.a40_cost import YAML, YAML_SHA

root = base / 'synthetic'; root.mkdir()
ops_root = base / 'ops-root'
lock = base / 'uv.lock'; lock.write_text('synthetic launcher input; not a runtime preflight\n')
# No arrays, real data or CUDA: this chain stops after prepare -> dry, before run.
doc = {'status': 'synthetic_generated', 'synthetic_input': True, 'science_data_read': False}
(root / 'inputs.json').write_text(json.dumps(doc))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_run(stage, payload, status):
    rid = 'g-decode-timing-' + stage + '-v1'
    target = ops_root / 'runs' / rid
    (target / 'logs').mkdir(parents=True)
    pins = [{'path': str(lock), 'sha256': sha(lock)},
            {'path': str(original.PINS), 'sha256': original.PINS_SHA}]
    if stage == 'preflight':
        pins.append({'path': str(root / 'inputs.json'), 'sha256': sha(root / 'inputs.json')})
    run = {'run_id': rid, 'kind': 'ops', 'git': {'commit': git(engine, 'rev-parse', 'HEAD'),
                                             'dirty': False, 'dirty_files': []},
           'cwd': str(engine), 'argv': original.invocation(stage, root, lock, ops_root),
           'config': {'path': str(YAML), 'sha256': YAML_SHA}, 'inputs': pins}
    (target / 'run.json').write_text(json.dumps(run))
    (target / 'terminal.json').write_text(json.dumps({'run_id': rid, 'status': status,
                                                      'exit_code': 0 if status == 'succeeded' else None,
                                                      'signal': None if status == 'succeeded' else 15}))
    (target / 'logs/stdout.log').write_text(json.dumps(payload) + '\n' if stage == 'generate' else '')
    return target


write_run('generate', doc, 'succeeded')
failed = write_run('preflight', {}, 'wall_cap_exceeded')
for state in ('pending', 'claimed', 'rejected'):
    (ops_root / 'queue' / state).mkdir(parents=True)
argv = [sys.executable, '-B', '-m', 'analysis.g_decode_preflight_retry']
flags = ['--engine', str(engine), '--root', str(root), '--lock', str(lock),
         '--ops-root', str(ops_root), '--ops-code', str(ops_code)]
env = {**os.environ, 'PYTHONPATH': str(runner / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}


def run(args):
    return subprocess.run(argv + args, cwd=runner, env=env, capture_output=True, text=True)


prepared = run(['prepare', *flags])
assert prepared.returncode == 0, prepared.stderr
value = json.loads(prepared.stdout)
assert value['status'] == 'prepared_dry'
assert value['metadata']['git']['dirty'] is False
assert value['spec']['wall_hours'] == 8
assert value['spec']['cpu_cores'] == 1
assert value['spec']['ram_mib'] == 98304 and value['spec']['vram_mib'] == 8192
assert value['engine_source']['commit'] == '87159f720afaf42be6252223442ece56a20e0ef9'
assert value['metadata']['git']['commit'] == git(runner, 'rev-parse', 'HEAD')
assert len(value['metadata']['inputs']) == 7
(base / 'prepare.json').write_text(prepared.stdout)
dry = run(['dry'])
assert dry.returncode == 0 and json.loads(dry.stdout)['status'] == 'dry', dry.stderr
(base / 'dry.json').write_text(dry.stdout)
# Alter only disposable synthetic receipt variants; the original job is never read.
terminal = failed / 'terminal.json'
original_terminal = terminal.read_bytes()
wrong_terminal = json.loads(original_terminal)
wrong_terminal['status'] = 'succeeded'
terminal.write_text(json.dumps(wrong_terminal))
bad = run(['prepare', *flags])
assert bad.returncode != 0 and 'original failed preflight binding differs' in bad.stderr
(base / 'wrong-original-terminal.stderr').write_text(bad.stderr)
# Keep this deliberately invalid fixture on disk; do not delete or overwrite outputs.
print(json.dumps({'status': 'real_launcher_prepare_and_dry_verified',
                  'engine_files': len(value['engine_source']['files']),
                  'runner_files': len(value['runner_source']['files']),
                  'declared_inputs': len(value['metadata']['inputs']),
                  'wrong_original_terminal_rejected': True,
                  'synthetic_only': True, 'runtime_preflight_executed': False,
                  'no_submit': True, 'evidence_directory': str(base.relative_to(repo))}))
