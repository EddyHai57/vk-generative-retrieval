"""Server-side enqueue tool for the G5 fuse job (stdlib only; JSON request on stdin).

Stages ``fuse-pf`` (``--check-only``: every input gate, no fusion, nothing written) and ``fuse`` (writes the one new result directory). The job
is CPU-only and pure standard library. It reads the three accepted P export manifests (sha pinned here), the three G assemblies and the P
query metadata (their shas come from the request, after those jobs have run). The queue root is never taken from the request.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
FUSION_DEPLOY_PATTERN = r'/workspace/vkgr-g5-fusion-v[0-9]+'
P_EXPORT_MANIFEST_SHA = {42: 'be5eb969d47c453105c538f7f57a87ee3eff64b28666bf321fe196c677ce576b', 2024: '531a6c28eea6952f583fe6286648b8e0375755696ad2329802d245c719e3b989',
                         2025: '38ad413b07ed8b82342749d3f58be8438e99a2388253c06709b838f160413439'}      # experiment_log 2026-10-18 G5 top-100 export, third attempt
STAGES = {'fuse-pf': ('g5-fusion-fuse-pf', True), 'fuse': ('g5-fusion-fuse', False)}
TEMPLATE_RUN = 'p-seqm1-mean-s42'
SHA = re.compile(r'[0-9a-f]{64}')


def template(root):
    matches = sorted((root / 'queue/claimed').glob(f'*-{TEMPLATE_RUN}-verify.json'))
    if len(matches) != 1:
        raise ValueError('unique claimed template required')
    return json.loads(matches[0].read_text())


def build(root, stage, fusion_deploy, vram_mib, wall_hours, g_assembly_sha, meta_sha, gen=''):
    if stage not in STAGES or type(vram_mib) is not int or not 256 <= vram_mib <= 6656 or not 0 < wall_hours <= 4:
        raise ValueError('bad stage/resources')
    if gen and (len(gen) > 8 or not gen.isalnum()):
        raise ValueError('bad generation tag')
    if type(fusion_deploy) is not str or re.fullmatch(FUSION_DEPLOY_PATTERN, fusion_deploy) is None:
        raise ValueError('bad deployment path')
    if type(g_assembly_sha) is not dict or set(g_assembly_sha) != {'42', '2024', '2025'} or any(type(v) is not str or SHA.fullmatch(v) is None for v in g_assembly_sha.values()) \
            or type(meta_sha) is not str or SHA.fullmatch(meta_sha) is None:
        raise ValueError('assembly / metadata shas required')
    t = template(root); argv = t['argv']
    rid = STAGES[stage][0] + (f'-{gen}' if gen else '')
    runs = root / 'runs'
    cmd = [argv[0], '-B', '-m', 'vkgr_g5.fuse']
    inputs = []
    for seed in (42, 2024, 2025):
        export = runs / f'g5-p-top100-v3-s{seed}' / 'out' / 'export-manifest.json'
        assembly = runs / f'g5-fusion-asm-s{seed}' / 'out' / 'g-assembly.json'
        cmd += ['--p-export', f'{seed}={export}@{P_EXPORT_MANIFEST_SHA[seed]}', '--g-assembly', f'{seed}={assembly}@{g_assembly_sha[str(seed)]}']
        inputs += [str(export), str(export.with_name('top100.json')), str(assembly)]
    meta = runs / 'g5-fusion-qmeta' / 'out' / 'p-query-meta.json'
    cmd += ['--p-query-meta', f'{meta}@{meta_sha}', '--output-dir', str(runs / rid / 'out'), '--split', 'validation']
    inputs.append(str(meta))
    if STAGES[stage][1]:
        cmd.append('--check-only')
    env = [f'PYTHONPATH={fusion_deploy}/src' if e.startswith('PYTHONPATH=') else e for e in t['env']]
    if env == t['env']:
        raise ValueError('template env has no PYTHONPATH')
    job = dict(t)
    job.update(argv=cmd, config=None, cwd=fusion_deploy, env=env, inputs=inputs, run_id=rid, vram_mib=vram_mib, wall_hours=float(wall_hours), disk_gib=1, pinned_files=[])
    return job


def exists_anywhere(root, rid):
    if (root / 'runs' / rid).exists():
        return True
    pattern = re.compile(r'\.?\d+-' + re.escape(rid) + r'\.(json|reason|json\.reason)(\.tmp)?')
    return any(pattern.fullmatch(p.name) for sub in ('pending', 'claimed', 'rejected') for p in (root / 'queue' / sub).glob('*'))


def main(root=ROOT):
    req = json.loads(sys.stdin.read(65537)); root = Path(root)
    job = build(root, req['stage'], req.get('fusion_deploy'), req['vram_mib'], req['wall_hours'], req.get('g_assembly_sha'), req.get('meta_sha'), req.get('gen', ''))
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
