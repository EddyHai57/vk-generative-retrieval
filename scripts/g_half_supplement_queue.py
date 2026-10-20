"""Fixed, one-shot queue bridge for the G-half read-only supplement."""
import datetime
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

PF_ID = 'g-half-supplement-preflight-v3'
REPORT_ID = 'g-half-supplement-report-v1'
DEP = '/workspace/vkgr-g-half-supplement-v3'
OPS = '/workspace/vkgr-g-half-supplement-ops-v3'
SCHED = '/workspace/vkgr-infra01-root'
SCHEDULER_CODE = '/workspace/vkgr-infra03-ops'
SCRIPT = 'analysis/g_half_supplement.py'
QUEUE_SCRIPT = 'scripts/g_half_supplement_queue.py'
READER = 'analysis/phase2_validation_read.py'
LIMIT = 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(path):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in (p, *p.parents)), 'absolute nonsymlink metadata path required')
    return p


def validate_request(req):
    fields = {'schema', 'action', 'root', 'ops', 'scheduler_code', 'source_commit', 'files', 'job', 'preflight_ref', 'dry_job_sha256'}
    require(type(req) is dict and set(req) == fields, 'closed transport request fields required')
    require(req['schema'] == 'g-half-supplement-transport-v1' and req['action'] in ('inventory', 'dry', 'submit', 'status'), 'fixed transport action required')
    require(req['root'] == SCHED and req['ops'] == OPS and req['scheduler_code'] == SCHEDULER_CODE, 'fixed project scope required')
    require(type(req['source_commit']) is str and re.fullmatch('[a-f0-9]{40}', req['source_commit']), 'fixed source commit required')
    require(type(req['files']) is dict and set(req['files']) == {SCRIPT, READER, QUEUE_SCRIPT, 'uv.lock'}, 'exact four production source files required')
    for name, pin in req['files'].items():
        require(type(name) is str and type(pin) is str and re.fullmatch('[a-f0-9]{64}', pin), 'relative production source SHA required')
    job = req['job']
    require(type(job) is dict and job.get('run_id') in (PF_ID, REPORT_ID), 'two fixed report stages only')
    job_fields = {'run_id', 'cwd', 'argv', 'kind', 'schema', 'no_admission', 'cpu_cores', 'vram_mib',
                  'ram_mib', 'wall_hours', 'disk_gib', 'engineering', 'test_read', 'checkpoint_read',
                  'model_run', 'stop_mode', 'pinned_manifest', 'env', 'pinned_files', 'config', 'seed', 'inputs'}
    require(set(job) == job_fields, 'closed report job fields required')
    preflight = job['run_id'] == PF_ID
    require(job.get('cwd') == DEP and job.get('kind') == 'ops' and job.get('schema') == 'vkgr-queue-job-v1', 'fixed deployed job identity required')
    require(job.get('no_admission') is False, 'formal scheduler admission path required')
    require(all(type(job.get(k)) is int for k in ('cpu_cores', 'vram_mib', 'ram_mib', 'disk_gib')) and
            (job['cpu_cores'], job['vram_mib'], job['ram_mib'], job['disk_gib']) == (1, 0, 8192, 1) and
            type(job.get('wall_hours')) in (int, float) and job['wall_hours'] == 2, 'fixed CPU report resources required')
    require(job.get('engineering') is False and job.get('test_read') is False and job.get('checkpoint_read') is False and job.get('model_run') is False, 'read-only job boundary required')
    require(job.get('stop_mode') == 'none' and job.get('seed') is None and job.get('inputs') == [], 'fixed nontraining job controls required')
    argv = job.get('argv')
    require(type(argv) is list and len(argv) == (8 if preflight else 7) and Path(argv[0]).is_absolute() and Path(argv[0]).resolve() == Path(sys.executable).resolve(), 'formal Python argv required')
    require(argv[1:6] == ['-B', DEP+'/'+SCRIPT, '--request', OPS+'/report-request.json', '--request-sha'], 'fixed report request argv required')
    require(job.get('config') == OPS+'/report-request.json', 'fixed report config binding required')
    require(re.fullmatch('[a-f0-9]{64}', argv[6]) is not None and ((argv[-1] == '--preflight') if preflight else True), 'request physical SHA/preflight argv required')
    if not preflight:
        require(req['preflight_ref'] is not None and type(req['preflight_ref']) is dict, 'original preflight receipt required')
    else:
        require(req['preflight_ref'] is None, 'preflight cannot bind itself')
    require(req['dry_job_sha256'] is None or (type(req['dry_job_sha256']) is str and re.fullmatch('[a-f0-9]{64}', req['dry_job_sha256'])), 'dry job SHA syntax required')
    required_env = ['PYTHONPATH='+DEP, 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                    'OPENBLAS_NUM_THREADS=1', 'VKGR_PINNED_MANIFEST='+OPS+'/pinned-manifest.json']
    require(job.get('env') == required_env and job.get('pinned_manifest') == OPS+'/pinned-manifest.json', 'fixed readonly runtime environment required')
    require(job.get('pinned_files') == [k+'='+v for k, v in sorted(req['files'].items())], 'exact four source files must be pinned')
    return job


def identity(req):
    dep = safe(DEP)
    commit = subprocess.check_output(['git', '-C', str(dep), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == req['source_commit'] and not subprocess.check_output(['git', '-C', str(dep), 'status', '--porcelain', '--untracked-files=all'], text=True), 'clean fixed report source required')
    for name, pin in req['files'].items():
        p = safe(dep / name)
        require(sha(p) == pin, 'deployed source byte differs')
        blob = subprocess.check_output(['git', '-C', str(dep), 'show', commit+':'+name])
        require(hashlib.sha256(blob).hexdigest() == pin, 'source differs from original Git bytes')
    pinned_path = safe(OPS+'/pinned-manifest.json')
    pinned = json.loads(pinned_path.read_bytes())
    require(pinned == {'schema': 'vkgr-pinned-v1', 'base_commit': req['source_commit'],
                       'target_commit': req['source_commit'], 'overrides': {}, 'files': req['files']}, 'scheduler pinned source manifest differs')
    config = safe(req['job']['config'])
    require(sha(config) == req['job']['argv'][6], 'private report request physical SHA differs')
    request = json.loads(config.read_bytes())
    require(type(request) is dict and request.get('schema') == 'g-half-supplement-request-v1' and
            request.get('action') == 'report' and request.get('source_commit') == req['source_commit'] and
            request.get('source_sha256') == req['files'][SCRIPT], 'private report request source identity differs')


def inventory(root, ops, run_id):
    root, ops = Path(root), Path(ops)
    return {'run_id': run_id, 'run_exists': (root/'runs'/run_id).exists(),
            'intent_exists': (ops/('intent-'+run_id+'.json')).exists(),
            'queue': {status: sorted(x.name for x in (root/'queue'/status).glob('*-'+run_id+'.json'))
                      for status in ('pending', 'claimed', 'rejected')}}


def ensure_fresh(root, ops, run_id):
    result = inventory(root, ops, run_id)
    if result['run_exists'] or result['intent_exists'] or any(result['queue'].values()):
        raise FileExistsError('original report identity already occupied; no retry')
    return result


def check_preflight(req):
    ref = req['preflight_ref']
    stdout = Path(SCHED)/'runs'/PF_ID/'logs/stdout.log'
    require(type(ref) is dict and set(ref) == {'path', 'sha256'} and ref['path'] == str(stdout), 'fixed original preflight stdout required')
    p = safe(stdout)
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'preflight stdout physical SHA differs')
    obj = json.loads(raw)
    require(obj.get('status') == 'preflight_ok' and obj.get('run_id') == PF_ID and obj.get('source_commit') == req['source_commit'], 'preflight receipt identity differs')
    require(obj.get('source_sha256') == req['files'][SCRIPT], 'preflight report source SHA differs')
    require(obj.get('request_sha256') == req['job']['argv'][6], 'preflight private request SHA differs')
    require(obj.get('engineering') is False and obj.get('test_read') is False and obj.get('checkpoint_read') is False and obj.get('model_run') is False, 'preflight read-only boundary differs')
    run_dir = Path(SCHED)/'runs'/PF_ID
    run_path = safe(run_dir/'run.json')
    terminal_path = safe(run_dir/'terminal.json')
    run_raw, terminal_raw = run_path.read_bytes(), terminal_path.read_bytes()
    require(len(run_raw) <= LIMIT and len(terminal_raw) <= LIMIT, 'bounded preflight run/terminal receipts required')
    run, terminal = json.loads(run_raw), json.loads(terminal_raw)
    require(run.get('run_id') == PF_ID and run.get('cwd') == DEP and
            run.get('argv') == req['job']['argv']+['--preflight'], 'preflight run manifest argv/cwd differs')
    require(run.get('git') == {'commit': req['source_commit'], 'dirty': False, 'dirty_files': []}, 'preflight source tree identity differs')
    config = run.get('config') or {}
    require(config.get('path') == OPS+'/report-request.json' and config.get('sha256') == req['job']['argv'][6], 'preflight run config binding differs')
    require(terminal.get('run_id') == PF_ID and terminal.get('status') == 'succeeded' and terminal.get('exit_code') == 0 and terminal.get('wall_cap_exceeded') is False, 'preflight terminal receipt differs')
    queued = inventory(SCHED, OPS, PF_ID)['queue']
    require(len(queued['claimed']) == 1 and not queued['pending'] and not queued['rejected'], 'preflight claim uniqueness differs')
    claimed_path = safe(Path(SCHED)/'queue/claimed'/queued['claimed'][0])
    claimed_raw = claimed_path.read_bytes()
    require(len(claimed_raw) <= LIMIT, 'bounded preflight claimed job required')
    claimed = json.loads(claimed_raw)
    pinned_files = [name+'='+value for name, value in sorted((run.get('pinned_files') or {}).items())]
    require(claimed.get('run_id') == PF_ID and claimed.get('argv') == run['argv'] and claimed.get('cwd') == DEP and
            claimed.get('pinned_files') == pinned_files and
            claimed.get('cpu_cores') == 1 and claimed.get('vram_mib') == 0 and claimed.get('ram_mib') == 8192 and
            claimed.get('wall_hours') == 2 and claimed.get('disk_gib') == 1, 'preflight claimed job binding differs')
    return {'path': str(stdout), 'sha256': ref['sha256'], 'run_manifest_sha256': hashlib.sha256(run_raw).hexdigest(),
            'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(), 'claimed_job_sha256': hashlib.sha256(claimed_raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description='Fixed read-only G-half report queue bridge; request is bounded JSON on stdin')
    parser.parse_args()
    raw = sys.stdin.buffer.read(LIMIT+1)
    require(len(raw) <= LIMIT, 'bounded private transport request required')
    req = json.loads(raw)
    job = validate_request(req)
    identity(req)
    action, rid = req['action'], job['run_id']
    result = None
    if action == 'status':
        result = inventory(req['root'], req['ops'], rid)
        run_dir = Path(req['root'])/'runs'/rid
        for name in ('run.json', 'terminal.json', 'logs/stdout.log', 'logs/stderr.log'):
            p = run_dir/name
            if p.exists():
                p = safe(p)
                data = p.read_bytes()
                require(len(data) <= LIMIT, 'oversized status receipt')
                try: value = json.loads(data)
                except ValueError: value = data.decode()
                result[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'payload': value}
    elif action == 'inventory':
        result = ensure_fresh(req['root'], req['ops'], rid)
    else:
        ensure_fresh(req['root'], req['ops'], rid)
        if rid == REPORT_ID:
            check_preflight(req)
        sys.path.insert(0, str(Path(req['scheduler_code'])/'ops'))
        from vkgr_ops import launcher, scheduler
        prepared = launcher.prepare(job)
        require(len(json.dumps({'spec': job, 'prepared': prepared}, allow_nan=False).encode()) < 48*1024-2048, 'bounded launcher envelope required')
        if action == 'dry':
            result = {'status': 'dry_ok', 'run_id': rid, 'job_sha256': digest(job)}
        else:
            require(req['dry_job_sha256'] == digest(job), 'original dry job SHA differs')
            with (Path(req['ops'])/'publish.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ensure_fresh(req['root'], req['ops'], rid)
                intent = Path(req['ops'])/('intent-'+rid+'.json')
                with intent.open('x') as f:
                    json.dump({'run_id': rid, 'job_sha256': digest(job), 'status': 'attempting'}, f)
                try:
                    path = scheduler.submit(Path(req['root']), job)
                except Exception as exc:
                    alert = Path(req['ops'])/('publication-ALERT-'+rid+'.json')
                    with alert.open('x') as f:
                        json.dump({'status': 'ALERT', 'reason': type(exc).__name__+': '+str(exc), 'retry': False}, f)
                    raise
            result = {'status': 'written_unconfirmed', 'run_id': rid, 'job_sha256': digest(job), 'queue_path': str(path)}
    result['read_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
