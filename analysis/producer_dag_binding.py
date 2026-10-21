"""Closed producer receipts for stage-3 consumers (decision aa72a86, section 7).

Callers must first authenticate the complete pin through a fixed SHA/request.
This module authenticates provenance, not a consumer's scientific data contract.
No producer/consumer commit equality is inferred or required.
"""
import json
from pathlib import PurePosixPath

from analysis.sgns_subset_acceptance import encoded, pin, sha
from analysis.ur001_derive_cli import secure

JOB_FILES = ('run.json', 'terminal.json', 'logs/stdout.log')
PIN_KEYS = {'manifest_sha256', 'source', 'runtime', 'job_sha256', 'receipt'}
RUN_KEYS = {'schema', 'run_id', 'kind', 'cwd', 'argv', 'seed', 'git', 'config',
            'inputs', 'env', 'resources', 'ops_source', 'overlay', 'pinned_files', 'stop_mode'}
TERMINAL_KEYS = {'schema', 'run_id', 'status', 'exit_code', 'signal', 'wall_cap_exceeded',
                 'started_at', 'ended_at', 'wall_seconds'}


def _source(value):
    if type(value) is not dict or set(value) != {'commit', 'files'}:
        raise ValueError('complete source ledger required')
    pin(value['commit'], 40)
    if type(value['files']) is not dict or not value['files']:
        raise ValueError('nonempty source files required')
    for name, digest in value['files'].items():
        if (type(name) is not str or not name or PurePosixPath(name).is_absolute()
                or '..' in PurePosixPath(name).parts or str(PurePosixPath(name)) != name):
            raise ValueError('canonical relative source path required')
        pin(digest)


def _runtime(value):
    if (type(value) is not dict or type(value.get('runtime')) is not dict
            or not value['runtime'].get('python')):
        raise ValueError('complete Python/dependency runtime required')
    pin(value.get('lock_sha256'))


def check_consumer(actual_source, expected_source, actual_runtime, expected_runtime):
    """Compare the consumer's own clean source/runtime to its complete fixed pin."""
    _source(expected_source)
    _runtime(expected_runtime)
    if (encoded(actual_source) != encoded(expected_source)
            or encoded(actual_runtime) != encoded(expected_runtime)):
        raise ValueError('consumer exact source/runtime binding differs')


def verify_producer(manifest_path, run_dir, pin_record):
    """Read pinned original manifest/job/stdout; return manifest, receipt, binding.

    pin_record has exactly manifest_sha256, source, runtime, job_sha256 and
    receipt. job_sha256 fixes all three JOB_FILES; receipt fixes the complete
    last nonblank stdout JSON. It must either equal the manifest or name its
    exact SHA through manifest_sha256. Reinvoke before committing outputs to
    detect upstream drift across a long-running consumer operation.
    """
    if type(pin_record) is not dict or set(pin_record) != PIN_KEYS:
        raise ValueError('closed producer DAG pin required')
    _source(pin_record['source'])
    _runtime(pin_record['runtime'])
    hashes = pin_record['job_sha256']
    if type(hashes) is not dict or set(hashes) != set(JOB_FILES):
        raise ValueError('all original job receipt hashes required')
    expected = {'manifest': pin(pin_record['manifest_sha256']),
                **{name: pin(hashes[name]) for name in JOB_FILES}}
    manifest_path, run_dir = secure(manifest_path), secure(run_dir)
    paths = {'manifest': manifest_path, **{name: secure(run_dir/name) for name in JOB_FILES}}
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    if any(sha(blobs[name]) != digest for name, digest in expected.items()):
        raise ValueError('producer original manifest/job bytes differ')
    manifest = json.loads(blobs['manifest'])
    run = json.loads(blobs['run.json'])
    terminal = json.loads(blobs['terminal.json'])
    lines = [line for line in blobs['logs/stdout.log'].splitlines() if line.strip()]
    receipt = json.loads(lines[-1]) if lines else None
    if (type(manifest) is not dict or type(receipt) is not dict
            or encoded(manifest.get('source')) != encoded(pin_record['source'])
            or encoded(manifest.get('runtime')) != encoded(pin_record['runtime'])
            or encoded(receipt) != encoded(pin_record['receipt'])
            or receipt.get('verified') is not True
            or (encoded(receipt) != encoded(manifest)
                and receipt.get('manifest_sha256') != expected['manifest'])):
        raise ValueError('producer manifest/source/runtime/stdout binding differs')
    if (type(run) is not dict or not RUN_KEYS <= set(run)
            or run.get('schema') != 'vkgr-run-v1' or run.get('run_id') != run_dir.name
            or encoded(run.get('git')) != encoded({'commit': pin_record['source']['commit'],
                                                   'dirty': False, 'dirty_files': []})
            or type(terminal) is not dict or set(terminal) != TERMINAL_KEYS
            or terminal.get('schema') != 'vkgr-terminal-v1' or terminal.get('run_id') != run_dir.name
            or terminal.get('status') != 'succeeded' or type(terminal.get('exit_code')) is not int
            or terminal['exit_code'] != 0 or terminal.get('signal') is not None
            or terminal.get('wall_cap_exceeded') is not False):
        raise ValueError('successful original producer job/source required')
    if any(secure(path).read_bytes() != blobs[name] for name, path in paths.items()):
        raise ValueError('producer evidence changed during verification')
    return {'manifest': manifest, 'receipt': receipt,
            'binding': json.loads(encoded(pin_record))}
