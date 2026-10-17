"""Server-side enqueue tool for the G5 P top-100 export jobs (stages: pf, export, verify). Python stdlib only; reads JSON from stdin.

Clones the exact job envelope of the accepted ``p-seqm1-mean-s<seed>-verify`` claimed job (same interpreter, env, inputs, pinned
files) and swaps in the new deployment, command and run id. ``dry`` validates and prints the job specs; ``submit`` additionally
writes them to queue/pending (tmp + rename). Refuses any run id that already exists in runs/ or in any queue directory, and any
P anchor identity that differs from the frozen table. Never touches an existing run directory.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
DEPLOY = '/workspace/vkgr-g5-p-export100-v1'
MODULE = 'vkgr.retrieval_seqm1_export100'
FROZEN = {  # frozen identity table (G5 FINAL card section 7): seed -> (config sha256, train manifest sha256)
    42: ('e88dc757ff4085af020efc1f070ebba8df54876e85adf68120dff730f983b621', 'bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e'),
    2024: ('3e1e40f200c3a09a39e15048df33657bbb54080a7da33d604d92c41518a5dde5', '8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8'),
    2025: ('e4f8f612991ee960b496067b905cb57bb46d7d2b9e4c4af9026004a53e8713f3', '783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb'),
}
STAGES = {'pf': 'preflight', 'export': 'export', 'verify': 'verify'}


def template(root, seed):
    matches = sorted((root / 'queue/claimed').glob(f'*-p-seqm1-mean-s{seed}-verify.json'))
    if len(matches) != 1:
        raise ValueError('unique claimed template required')
    return json.loads(matches[0].read_text())


def run_id(stage, seed, gen=''):
    return f'g5-p-top100-{gen + "-" if gen else ""}{"pf-" if stage == "pf" else ""}s{seed}{"-verify" if stage == "verify" else ""}'


def build(root, stage, seed, vram_mib, wall_hours, export_sha=None, gen=''):
    if gen and (len(gen) > 8 or not gen.isalnum()):
        raise ValueError('bad generation tag')
    if stage not in STAGES or seed not in FROZEN or type(vram_mib) is not int or vram_mib < 256 or not 0 < wall_hours <= 4:
        raise ValueError('bad stage/seed/resources')
    config_sha, manifest_sha = FROZEN[seed]
    t = template(root, seed)
    argv = t['argv']; get = lambda flag: argv[argv.index(flag) + 1]
    if get('--config-sha') != config_sha or get('--manifest-sha') != manifest_sha:
        raise ValueError('template identity differs from the frozen table')
    pm = get('--manifest'); source_run = f'p-seqm1-mean-s{seed}'
    if pm != str(root / 'runs' / source_run / 'out/manifest.json'):
        raise ValueError('unexpected source manifest path')
    out_dir = root / 'runs' / run_id('export', seed, gen) / 'out'
    cmd = [argv[0], '-B', '-m', MODULE, STAGES[stage], '--inputs', get('--inputs'), '--inputs-sha', get('--inputs-sha'), '--config', get('--config'),
           '--config-sha', config_sha, '--manifest', pm, '--manifest-sha', manifest_sha, '--run-id', source_run]
    inputs = list(t['inputs'])
    if stage == 'export':
        cmd += ['--output', str(out_dir)]
    if stage == 'verify':
        if type(export_sha) is not str or len(export_sha) != 64:
            raise ValueError('export manifest sha required for verify')
        cmd += ['--export-manifest', str(out_dir / 'export-manifest.json'), '--export-manifest-sha', export_sha]
        inputs.append(str(out_dir / 'export-manifest.json'))
    job = dict(t)
    job.update(argv=cmd, config=str(Path(DEPLOY) / 'configs/seqm1' / f'{source_run}.json'), cwd=DEPLOY, inputs=inputs,
               run_id=run_id(stage, seed, gen), vram_mib=vram_mib, wall_hours=float(wall_hours), disk_gib=1 if stage != 'export' else 1)
    return job


def exists_anywhere(root, rid):
    if (root / 'runs' / rid).exists():
        return True
    return any(rid in p.name for sub in ('pending', 'claimed', 'rejected') for p in (root / 'queue' / sub).glob('*'))


def main():
    req = json.loads(sys.stdin.read(65537))
    root = Path(req.get('root', ROOT)); jobs = []
    for seed in req['seeds']:
        job = build(root, req['stage'], seed, req['vram_mib'], req['wall_hours'], (req.get('export_sha') or {}).get(str(seed)), req.get('gen', ''))
        if exists_anywhere(root, job['run_id']):
            raise ValueError('run id already exists: ' + job['run_id'])
        jobs.append(job)
    written = []
    if req['action'] == 'submit':
        for job in jobs:
            name = f'{time.time_ns():020d}-{job["run_id"]}.json'; pending = root / 'queue/pending'
            tmp = pending / ('.' + name + '.tmp')
            tmp.write_text(json.dumps(job)); os.replace(tmp, pending / name); written.append(name)
    elif req['action'] != 'dry':
        raise ValueError('unknown action')
    print(json.dumps({'action': req['action'], 'jobs': jobs, 'written': written,
                      'job_sha256': {j['run_id']: hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest() for j in jobs}}, sort_keys=True))


if __name__ == '__main__':
    main()
