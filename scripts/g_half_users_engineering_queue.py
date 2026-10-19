"""Metadata-only queue bridge for the frozen G half-user real-input preflight/CUDA gates."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys


STAGES = {'g-half-users-preflight-v1': 'preflight', 'g-half-users-cuda-v1': 'cuda-preview'}
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
LAUNCH_ENVELOPE_LIMIT = 48 * 1024 - 2048


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def safe_absolute(value, role):
    if type(value) is not str or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('absolute non-traversing path required: ' + role)
    path = Path(value)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('symlink path forbidden: ' + role)
    return path


def queue_state(root, run_id):
    root = Path(root)
    value = {name: sorted((root / 'queue' / name).glob('*-' + run_id + '.json'))
             for name in ('pending', 'claimed', 'rejected')}
    reasons = {}
    for path in value['rejected']:
        for reason in (path.with_suffix('.reason'), Path(str(path) + '.reason')):
            if reason.is_file() and not reason.is_symlink():
                reasons[reason.name] = reason.read_text()[:8192]
    return {'pending': [p.name for p in value['pending']],
            'claimed': [p.name for p in value['claimed']],
            'rejected': [p.name for p in value['rejected']], 'reasons': reasons}


def inventory(req):
    job = req['job']
    root, ops = Path(req['root']), Path(req['ops'])
    run_id = job['run_id']
    return {'run_id': run_id, 'queue': queue_state(root, run_id),
            'run_exists': (root / 'runs' / run_id).exists(),
            'intent_exists': (ops / ('intent-' + run_id + '.json')).exists(),
            'job_sha256': digest(job)}


def validate(req):
    required = {'root', 'ops', 'deployment', 'scheduler_code', 'python', 'job', 'action'}
    if type(req) is not dict or not required <= set(req):
        raise ValueError('engineering queue request fields required')
    root = safe_absolute(req['root'], 'root')
    ops = safe_absolute(req['ops'], 'ops')
    deployment = safe_absolute(req['deployment'], 'deployment')
    scheduler_code = safe_absolute(req['scheduler_code'], 'scheduler_code')
    if type(req['python']) is not str or not Path(req['python']).is_absolute():
        raise ValueError('absolute scheduler Python executable required')
    python = Path(req['python']).resolve(strict=True)
    job = req['job']
    if type(job) is not dict or type(job.get('run_id')) is not str or job['run_id'] not in STAGES:
        raise ValueError('fixed G half-user engineering run IDs only')
    stage = STAGES[job['run_id']]
    expected_script = deployment / 'scripts/g_half_users_ops.py'
    request_path = safe_absolute(job.get('config'), 'private ops request')
    if not request_path.is_relative_to(ops) or not request_path.is_file() or request_path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError('bounded private request must live under the control directory')
    argv = job.get('argv')
    expected = ['-B', str(expected_script), stage, '--request', str(request_path), '--request-sha', sha(request_path)]
    if type(argv) is not list or len(argv) != len(expected)+1 or argv[1:] != expected:
        raise ValueError('fixed interpreter/ops-stage/private-request command required')
    if Path(argv[0]).resolve() != python or python != Path(sys.executable).resolve():
        raise ValueError('engineering bridge interpreter must equal scheduler interpreter')
    if job.get('cwd') != str(deployment) or job.get('schema') != 'vkgr-queue-job-v1' or job.get('kind') != 'ops' \
            or job.get('cpu_cores') != 1 or job.get('no_admission') is not False \
            or type(job.get('vram_mib')) is not int or not 0 <= job['vram_mib'] <= 39168:
        raise ValueError('fixed engineering job schema/resource scope required')
    env = job.get('env')
    if type(env) is not list or any(type(value) is not str for value in env):
        raise ValueError('engineering job env must be a string list')
    if any(value.partition('=')[0] in ('VKGR_RUN_ID', 'VKGR_PROGRESS_PATH') for value in env):
        raise ValueError('engineering job identity/progress env is injected by the launcher')
    selected = [value for value in env if value.startswith('VKGR_EXPECTED_SELECTED_USERS_SHA256=')]
    if selected and (len(selected) != 1 or len(selected[0].split('=', 1)[1]) != 64
                     or any(char not in '0123456789abcdef' for char in selected[0].split('=', 1)[1])):
        raise ValueError('expected selected-users SHA must be one lowercase SHA-256 env value')
    if req.get('action') not in ('inventory', 'dry', 'submit', 'status'):
        raise ValueError('inventory/dry/submit/status only')
    return job


def _fresh(req):
    value = inventory(req)
    if value['run_exists'] or value['intent_exists'] or any(value['queue'][name] for name in ('pending', 'claimed', 'rejected')):
        raise FileExistsError('fixed engineering run already exists; no resubmit')
    return value


def _launcher(req, job):
    sys.path.insert(0, str(Path(req['scheduler_code']) / 'ops'))
    from vkgr_ops import launcher
    prepared = launcher.prepare(job)
    envelope = json.dumps({'spec': job, 'prepared': prepared}, allow_nan=False).encode()
    if len(envelope) >= LAUNCH_ENVELOPE_LIMIT:
        raise ValueError('engineering launch envelope too large')
    return prepared, len(envelope)


def dry(req):
    value = _fresh(req)
    _, size = _launcher(req, req['job'])
    return {'status': 'dry_ok', 'run_id': req['job']['run_id'],
            'job_sha256': value['job_sha256'], 'launch_bytes': size}


def submit(req):
    value = dry(req)
    if req.get('dry_job_sha256') != value['job_sha256']:
        raise ValueError('dry job hash differs from the sealed request')
    root, ops = Path(req['root']), Path(req['ops'])
    sys.path.insert(0, str(Path(req['scheduler_code']) / 'ops'))
    from vkgr_ops import scheduler
    with (ops / '.g-half-users-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _fresh(req)
        intent = ops / ('intent-' + req['job']['run_id'] + '.json')
        new(intent, value)
        queue_path = scheduler.submit(root, req['job'])
    return {'status': 'written_unconfirmed', 'run_id': req['job']['run_id'],
            'intent_sha256': sha(intent), 'queue_path': str(queue_path)}


def status(req):
    value = inventory(req)
    run_dir = Path(req['root']) / 'runs' / req['job']['run_id']
    for path in (run_dir / 'terminal.json', run_dir / 'logs/stdout.log', run_dir / 'logs/stderr.log'):
        if not path.exists():
            continue
        if path.is_symlink() or path.stat().st_size > MAX_RECEIPT_BYTES:
            raise ValueError('bounded nonsymlink terminal/stdout/stderr receipt required')
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw.decode('utf-8', errors='replace')
        value[path.name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'payload': payload}
    return value


def main():
    raw = sys.stdin.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError('engineering metadata request too large')
    req = json.loads(raw)
    validate(req)
    actions = {'inventory': inventory, 'dry': dry, 'submit': submit, 'status': status}
    result = actions[req['action']](req)
    result['read_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
