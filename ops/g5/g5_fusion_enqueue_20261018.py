"""Server-side enqueue tool for the G5 fusion-stage jobs (stdlib only; JSON request on stdin).

Stages: ``qmeta-pf`` / ``qmeta`` (P-side validation query metadata: check-only preflight, then the writing run). Jobs clone the envelope of
the accepted ``p-seqm1-mean-s<seed>-verify`` claimed job (same interpreter, env, inputs, pinned files), run the fusion module from a
separate fusion deployment with the P deployment's ``vkgr`` first on PYTHONPATH, and write only below ``runs/<run_id>/out``.
``dry`` validates and prints; ``submit`` writes to queue/pending (tmp + rename). Refuses run ids that exist anywhere. Never touches an
existing run directory.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
P_DEPLOY_PATTERN = r'/workspace/vkgr-g5-p-export100-v[0-9]+'
FUSION_DEPLOY_PATTERN = r'/workspace/vkgr-g5-fusion-v[0-9]+'
CONFIG_SHA = 'e88dc757ff4085af020efc1f070ebba8df54876e85adf68120dff730f983b621'   # p-seqm1-mean-s42 (history_len 50, feature f3_f4_recency)
TEMPLATE_RUN = 'p-seqm1-mean-s42'
STAGES = {'qmeta-pf': ('g5-fusion-qmeta-pf', True), 'qmeta': ('g5-fusion-qmeta', False)}
# Accepted P anchor train manifests (G5 FINAL card section 7): seed -> manifest sha256. The query metadata must equal each manifest's binding.
P_MANIFESTS = {42: 'bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e', 2024: '8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8',
               2025: '783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb'}
EXPECT_QUERIES, EXPECT_OOV = 2520, 623      # registered validation set (experiment_log: 2520 queries, 623 OOV)


def template(root):
    matches = sorted((root / 'queue/claimed').glob(f'*-{TEMPLATE_RUN}-verify.json'))
    if len(matches) != 1:
        raise ValueError('unique claimed template required')
    return json.loads(matches[0].read_text())


def run_id(stage, gen=''):
    return STAGES[stage][0] + (f'-{gen}' if gen else '')


def build(root, stage, p_deploy, fusion_deploy, vram_mib, wall_hours, gen=''):
    if stage not in STAGES or type(vram_mib) is not int or vram_mib < 256 or not 0 < wall_hours <= 4:
        raise ValueError('bad stage/resources')
    if gen and (len(gen) > 8 or not gen.isalnum()):
        raise ValueError('bad generation tag')
    if type(p_deploy) is not str or re.fullmatch(P_DEPLOY_PATTERN, p_deploy) is None or type(fusion_deploy) is not str or re.fullmatch(FUSION_DEPLOY_PATTERN, fusion_deploy) is None:
        raise ValueError('bad deployment path')
    t = template(root); argv = t['argv']; get = lambda flag: argv[argv.index(flag) + 1]
    if get('--config-sha') != CONFIG_SHA:
        raise ValueError('template identity differs from the frozen table')
    rid = run_id(stage, gen)
    cmd = [argv[0], '-B', '-m', 'vkgr_g5.query_meta_p', '--line-root', p_deploy, '--inputs', get('--inputs'), '--inputs-sha', get('--inputs-sha'),
           '--config', get('--config'), '--config-sha', CONFIG_SHA, '--split', 'validation',
           '--output', str(root / 'runs' / rid / 'out' / 'p-query-meta.json'), '--expect-queries', str(EXPECT_QUERIES), '--expect-oov', str(EXPECT_OOV)]
    manifests = [str(root / 'runs' / f'p-seqm1-mean-s{seed}' / 'out' / 'manifest.json') for seed in sorted(P_MANIFESTS)]
    for seed, path in zip(sorted(P_MANIFESTS), manifests):
        cmd += ['--manifest', path, '--manifest-sha', P_MANIFESTS[seed]]
    if STAGES[stage][1]:
        cmd.append('--check-only')
    env = [f'PYTHONPATH=src:{fusion_deploy}/src' if e.startswith('PYTHONPATH=') else e for e in t['env']]
    if env == t['env']:
        raise ValueError('template env has no PYTHONPATH')
    job = dict(t)
    job.update(argv=cmd, config=str(Path(p_deploy) / get('--config')), cwd=p_deploy, env=env, inputs=[i for i in t['inputs'] if i != get('--manifest')] + manifests,
               run_id=rid, vram_mib=vram_mib, wall_hours=float(wall_hours), disk_gib=1)
    return job


def exists_anywhere(root, rid):
    if (root / 'runs' / rid).exists():
        return True
    pattern = re.compile(r'\.?\d+-' + re.escape(rid) + r'\.(json|reason)(\.tmp)?')    # exact run id; a longer id sharing this prefix does not match
    return any(pattern.fullmatch(p.name) for sub in ('pending', 'claimed', 'rejected') for p in (root / 'queue' / sub).glob('*'))


def main(root=ROOT):
    req = json.loads(sys.stdin.read(65537)); root = Path(root)      # the queue root is never taken from the request
    job = build(root, req['stage'], req.get('p_deploy'), req.get('fusion_deploy'), req['vram_mib'], req['wall_hours'], req.get('gen', ''))
    if exists_anywhere(root, job['run_id']):
        raise ValueError('run id already exists: ' + job['run_id'])
    written = []
    if req['action'] == 'submit':
        name = f'{time.time_ns():020d}-{job["run_id"]}.json'; pending = root / 'queue/pending'
        tmp = pending / ('.' + name + '.tmp'); tmp.write_text(json.dumps(job)); os.replace(tmp, pending / name); written.append(name)
    elif req['action'] != 'dry':
        raise ValueError('unknown action')
    print(json.dumps({'action': req['action'], 'job': job, 'written': written, 'job_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
