"""Server-side enqueue tool for the G5 G-side assembly jobs (stdlib only; JSON request on stdin).

Stages ``asm-pf`` (check-only preflight) and ``asm`` (writing run), one job per G anchor seed. Each job clones the envelope of the accepted
``g-coll-a1-c1-s<seed>-verify`` claimed job (same interpreter, env, resources, scheduler pin, every input pin), replaces the command by
``python -B -m vkgr_g5.assemble_g`` and appends the fusion deployment's source after the G deployment's source on PYTHONPATH (G first, so G's ``vkgr`` shadows any other). The queue root is never taken
from the request. ``dry`` validates and prints; ``submit`` writes to queue/pending (tmp + rename). Refuses a run id that already exists.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
G_DEPLOY_PATTERN = r'/workspace/vkgr-coll-a1-g-v[0-9]+'
FUSION_DEPLOY_PATTERN = r'/workspace/vkgr-g5-fusion-v[0-9]+'
MANIFEST_SHA = {42: 'a431a5b1734930671cf7ca092e469a6da16e3c130ffb30836b9aa57aaca8b36b', 2024: '09a0b790c6b027613ef15bb6c9855d7b1adee79d9bf6ea093c16d09a1a5b8d36',
                2025: '86767e4d30ababc8087898f6f960197021ce54fd09cc6d775da4c373b242c712'}      # G5 FINAL card section 7
FLAGS = ('--prepared', '--prepared-sha', '--history-manifest', '--history-sha', '--sid-manifest', '--sid-sha', '--content', '--content-sha',
         '--input-manifest', '--input-sha', '--vectors', '--vectors-sha', '--manifest', '--manifest-sha')
STAGES = {'asm-pf': ('g5-fusion-asm-pf', True), 'asm': ('g5-fusion-asm', False)}


def run_id(stage, seed, gen=''):
    return f'{STAGES[stage][0]}-s{seed}' + (f'-{gen}' if gen else '')


def template(root, seed):
    matches = sorted((root / 'queue/claimed').glob(f'*-g-coll-a1-c1-s{seed}-verify.json'))
    if len(matches) != 1:
        raise ValueError('unique claimed template required')
    return json.loads(matches[0].read_text())


def build(root, stage, seed, g_deploy, fusion_deploy, vram_mib, wall_hours, gen=''):
    if stage not in STAGES or seed not in MANIFEST_SHA or type(vram_mib) is not int or not 256 <= vram_mib <= 6656 or not 0 < wall_hours <= 4:
        raise ValueError('bad stage/seed/resources')
    if gen and (len(gen) > 8 or not gen.isalnum()):
        raise ValueError('bad generation tag')
    if type(g_deploy) is not str or re.fullmatch(G_DEPLOY_PATTERN, g_deploy) is None or type(fusion_deploy) is not str or re.fullmatch(FUSION_DEPLOY_PATTERN, fusion_deploy) is None:
        raise ValueError('bad deployment path')
    t = template(root, seed); argv = t['argv']; get = lambda flag: argv[argv.index(flag) + 1]
    if get('--manifest-sha') != MANIFEST_SHA[seed] or t['cwd'] != g_deploy or argv[argv.index('-m') + 1:argv.index('-m') + 3] != ['vkgr.coll_sid_a1_run', 'verify']:
        raise ValueError('template identity differs from the frozen table')
    rid = run_id(stage, seed, gen); base = root / 'runs' / rid
    cmd = [argv[0], '-B', '-m', 'vkgr_g5.assemble_g', '--line-root', g_deploy]
    for flag in FLAGS:
        cmd += [flag, get(flag)]
    cmd += ['--output', str(base / 'out' / 'g-assembly.json'), '--scratch', str(base / 'scratch'), '--split', 'validation']
    if STAGES[stage][1]:
        cmd.append('--check-only')
    env = [f'PYTHONPATH={g_deploy}/src:{fusion_deploy}/src' if e.startswith('PYTHONPATH=') else e for e in t['env']]
    if env == t['env'] or f'PYTHONPATH={g_deploy}/src' not in t['env']:
        raise ValueError('template PYTHONPATH is not the G deployment source')
    job = dict(t)
    job.update(argv=cmd, env=env, run_id=rid, vram_mib=vram_mib, wall_hours=float(wall_hours))
    return job


def exists_anywhere(root, rid):
    if (root / 'runs' / rid).exists():
        return True
    pattern = re.compile(r'\.?\d+-' + re.escape(rid) + r'\.(json|reason|json\.reason)(\.tmp)?')
    return any(pattern.fullmatch(p.name) for sub in ('pending', 'claimed', 'rejected') for p in (root / 'queue' / sub).glob('*'))


def main(root=ROOT):
    req = json.loads(sys.stdin.read(65537)); root = Path(root); jobs = []
    if type(req.get('seeds')) is not list or len(set(req['seeds'])) != len(req['seeds']):
        raise ValueError('seeds must be a list without duplicates')
    for seed in req['seeds']:
        job = build(root, req['stage'], seed, req.get('g_deploy'), req.get('fusion_deploy'), req['vram_mib'], req['wall_hours'], req.get('gen', ''))
        if exists_anywhere(root, job['run_id']):
            raise ValueError('run id already exists: ' + job['run_id'])
        jobs.append(job)
    written = []
    if req['action'] == 'submit':
        for job in jobs:
            name = f'{time.time_ns():020d}-{job["run_id"]}.json'; pending = root / 'queue/pending'
            tmp = pending / ('.' + name + '.tmp'); tmp.write_text(json.dumps(job)); os.replace(tmp, pending / name); written.append(name)
    elif req['action'] != 'dry':
        raise ValueError('unknown action')
    print(json.dumps({'action': req['action'], 'jobs': jobs, 'written': written,
                      'job_sha256': {j['run_id']: hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest() for j in jobs}}, sort_keys=True))


if __name__ == '__main__':
    main()
