"""Progress-only first engineering retry of the pinned original GPU preflight."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SHA = '87159f720afaf42be6252223442ece56a20e0ef9'
RUN_ID = 'g-decode-timing-preflight-retry1-v1'
OLD_RUN_ID = 'g-decode-timing-preflight-v1'
RECEIPT = 'preflight-retry1-transport.json'
RESOURCES = {'cpu_cores': 1, 'ram_mib': 98304, 'vram_mib': 8192, 'wall_hours': 8}


def absolute(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('absolute path without symlinks required')
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_ledger(root, expected=None):
    root = absolute(root)
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(root), *argv])
    sha = git('rev-parse', 'HEAD').decode().strip()
    if expected is not None and sha != expected:
        raise ValueError('fixed source SHA differs')
    if git('status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('clean fixed source required')
    files = {}
    for name in git('ls-tree', '-r', '--name-only', sha).decode().splitlines():
        target = absolute(root / name)
        expected_digest = hashlib.sha256(git('show', sha + ':' + name)).hexdigest()
        if digest(target) != expected_digest:
            raise ValueError('tracked source bytes differ: ' + name)
        files[name] = expected_digest
    return {'commit': sha, 'files': files}


def load_engine(root):
    root = absolute(root)
    ledger = source_ledger(root, ENGINE_SHA)
    sys.path[:0] = [str(root), str(root / 'src')]
    # Fail closed if this interpreter has already imported another engine tree.
    for name, module in list(sys.modules.items()):
        if name.startswith(('vkgr.', 'analysis.')) and name != __name__:
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).resolve().is_relative_to(ROOT):
                raise ValueError('another source tree is already imported: ' + name)
    engine = importlib.import_module('analysis.g_decode_timing')
    if Path(engine.__file__).resolve() != root / 'analysis/g_decode_timing.py':
        raise ValueError('original engine module identity differs')
    engine.references()
    return engine, ledger


@contextmanager
def phase(stage, emit, **fields):
    emit(stage, 'begin', **fields)
    try:
        yield
    except BaseException:
        emit(stage, 'error', **fields)
        raise
    else:
        emit(stage, 'end', **fields)


def observed_preflight(engine, root, lock, ops_root):
    """Wrap original stage boundaries; never enter or replace timed_decode."""
    started = time.perf_counter()
    def emit(stage, event, **fields):
        print(json.dumps({'stage': stage, 'event': event,
                          'utc': datetime.now(timezone.utc).isoformat(),
                          'elapsed_seconds': time.perf_counter() - started, **fields}, sort_keys=True),
              file=sys.stderr, flush=True)
    original_inputs = engine.inputs
    factory = engine.SessionFactory
    original_call = factory.__call__
    sessions = []
    def inputs(*args, **kwargs):
        with phase('inputs', emit):
            return original_inputs(*args, **kwargs)
    def build(instance, *args, **kwargs):
        with phase('session_build', emit):
            session = original_call(instance, *args, **kwargs)
        original_decode = session.decode
        previous = vars(session).get('decode')
        had_decode = 'decode' in vars(session)
        count = 0
        def decode(*args, **kwargs):
            nonlocal count
            count += 1
            with phase('query', emit, query_index=count):
                return original_decode(*args, **kwargs)
        sessions.append((session, had_decode, previous))
        session.decode = decode
        return session
    engine.inputs = inputs
    factory.__call__ = build
    try:
        with phase('preflight', emit):
            return engine.preflight(root, lock, ops_root)
    finally:
        engine.inputs = original_inputs
        factory.__call__ = original_call
        for session, had_decode, previous in sessions:
            if had_decode:
                session.decode = previous
            else:
                del session.decode


def request(args, engine):
    root = engine.fresh_path(args.root)
    lock = engine.fresh_path(args.lock)
    ops_root = engine.fresh_path(args.ops_root)
    if root == ROOT or ROOT in root.parents or root in ROOT.parents:
        raise ValueError('output must be outside runner source')
    if (root / 'preflight.json').exists() or (root / RECEIPT).exists():
        raise FileExistsError('retry output already exists; never overwrite')
    doc = json.loads((root / 'inputs.json').read_text())
    generation_sha = engine.predecessor('generate', root, lock, ops_root, doc)
    base = ops_root / 'runs' / OLD_RUN_ID
    run = json.loads((base / 'run.json').read_text())
    terminal = json.loads((base / 'terminal.json').read_text())
    from analysis.a40_cost import YAML, YAML_SHA
    expected_inputs = {str(lock): digest(lock), str(engine.PINS): engine.PINS_SHA,
                       str(root / 'inputs.json'): digest(root / 'inputs.json')}
    if (run.get('run_id') != OLD_RUN_ID or terminal.get('run_id') != OLD_RUN_ID or
            terminal.get('status') != 'wall_cap_exceeded' or terminal.get('signal') != 15 or
            run.get('kind') != 'ops' or run.get('git') != {'commit': ENGINE_SHA, 'dirty': False, 'dirty_files': []} or
            run.get('cwd') != str(engine.ROOT) or
            run.get('argv') != engine.invocation('preflight', root, lock, ops_root) or
            run.get('config') != {'path': str(YAML), 'sha256': YAML_SHA} or
            {p['path']: p['sha256'] for p in run['inputs']} != expected_inputs or
            len(run['inputs']) != len(expected_inputs)):
        raise ValueError('original failed preflight binding differs')
    return {'generation_stdout_sha256': generation_sha,
            'failed_run_sha256': digest(base / 'run.json'),
            'failed_terminal_sha256': digest(base / 'terminal.json'),
            'inputs_manifest_sha256': digest(root / 'inputs.json')}


def invocation(args):
    return [sys.executable, '-B', '-m', 'analysis.g_decode_preflight_retry', 'run',
            '--engine', str(args.engine), '--root', str(args.root), '--lock', str(args.lock),
            '--ops-root', str(args.ops_root), '--execute-on-new-instance']


def unoccupied(ops_root):
    if (ops_root / 'runs' / RUN_ID).exists():
        raise FileExistsError('retry run already exists')
    for state in ('pending', 'claimed', 'rejected'):
        if list((ops_root / 'queue' / state).glob('*-' + RUN_ID + '.json')):
            raise FileExistsError('retry queue identity already exists')


def job(args):
    runner = source_ledger(ROOT)
    engine, ledger = load_engine(args.engine)
    binding = request(args, engine)
    from analysis.a40_cost import YAML
    from analysis.a40_ops import load_ops
    _, launcher, scheduler, _ = load_ops(args.ops_code)
    ops_root = absolute(args.ops_root)
    unoccupied(ops_root)
    raw = ['--root', str(ops_root), '--run-id', RUN_ID, '--kind', 'ops', '--cwd', str(ROOT),
           '--config', str(YAML), '--seed', '42', '--input', str(absolute(args.lock)),
           '--input', str(engine.PINS), '--input', str(absolute(args.root) / 'inputs.json'),
           '--input', str(engine.ROOT / 'analysis/g_decode_timing.py'),
           '--input', str(Path(__file__).resolve()),
           '--input', str(ops_root / 'runs' / OLD_RUN_ID / 'run.json'),
           '--input', str(ops_root / 'runs' / OLD_RUN_ID / 'terminal.json'),
           '--env', 'PYTHONPATH=' + str(ROOT / 'src'), '--env', 'OMP_NUM_THREADS=1',
           '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1',
           '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '--cpu-cores', '1',
           '--ram-mib', '98304', '--vram-mib', '8192', '--disk-gib', '2', '--wall-hours', '8']
    spec = launcher.spec_from_args(launcher.build_parser().parse_args([*raw, '--', *invocation(args)]))
    metadata = launcher.prepare(spec)
    value = {'status': 'prepared_dry', 'spec': spec, 'metadata': metadata,
             'runner_source': runner, 'engine_source': ledger, 'binding': binding,
             'declaration_scope': 'RAM/VRAM conservative not_measured first measurement'}
    if args.command == 'submit':
        path = scheduler.submit(launcher.resolve_root(str(ops_root)), spec)
        value.update(status='written', queue_file=path.name)
    return value


def execute(args):
    runner = source_ledger(ROOT)
    engine, ledger = load_engine(args.engine)
    binding = request(args, engine)
    try:
        value = observed_preflight(engine, args.root, args.lock, args.ops_root)
        # Preserve original stdout/artifact payload; bind the new transport separately.
        engine.write_new(absolute(args.root) / RECEIPT,
                         {'schema': 'G_PREFLIGHT_RETRY1_TRANSPORT_V1', 'run_id': RUN_ID,
                          'runner_source': runner, 'engine_source': ledger, 'binding': binding,
                          'argv': invocation(args), 'runner_script_sha256': digest(__file__),
                          'preflight_sha256': digest(absolute(args.root) / 'preflight.json')})
        return value
    finally:
        if engine._SAMPLER is not None:
            engine._SAMPLER.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('dry')
    for command in ('run', 'prepare', 'submit'):
        item = sub.add_parser(command)
        for flag in ('engine', 'root', 'lock', 'ops-root'):
            item.add_argument('--' + flag, type=Path, required=True)
        if command == 'run':
            item.add_argument('--execute-on-new-instance', action='store_true', required=True)
        else:
            item.add_argument('--ops-code', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'dry':
        value = {'status': 'dry', 'run_id': RUN_ID, 'engine_sha': ENGINE_SHA,
                 'resources': RESOURCES, 'science_training': False, 'automatic_workers_measurement': False}
    else:
        value = execute(args) if args.command == 'run' else job(args)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
