"""Bounded raw subset restore: train/validation only, no Parquet inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / 'configs/a40-subset/hf-files.json'
FILES_SHA = 'd7df6268431bb0d0bcd77d56809bf8920914b4b8858f58f0dac409c7df81ff49'
REVISION = '084a18386e149a454dd6dbe0123478e75c6b09ce'
PREFIX = 'subsamples/ur0.01_ir0.01'


def validate_inventory(value):
    expected = [f'{PREFIX}/train/week_{n:02d}.parquet' for n in range(25)] + [f'{PREFIX}/validation/week_25.parquet']
    if set(value) != {'dataset', 'revision', 'files'} or value['dataset'] != 'deepvk/VK-LSVD' or value['revision'] != REVISION:
        raise ValueError('fixed subset dataset/revision required')
    if [r['path'] for r in value['files']] != expected:
        raise ValueError('only the fixed 26 train/validation files are authorized')
    for r in value['files']:
        if set(r) != {'path', 'size', 'sha256'} or type(r['size']) is not int or r['size'] <= 0:
            raise ValueError('invalid fixed file record')
        h = r['sha256']
        if type(h) is not str or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('invalid LFS SHA')
    return value


def inventory():
    raw = FILES.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FILES_SHA: raise ValueError('subset inventory drift')
    return validate_inventory(json.loads(raw))


def check_tree(tree, records):
    if type(tree) is not list: raise ValueError('invalid HF metadata response')
    for r in records:
        matches = [v for v in tree if v.get('path') == r['path']]
        if len(matches) != 1: raise ValueError('missing or duplicate source file')
        v = matches[0]
        oid = (v.get('lfs') or {}).get('oid', '').removeprefix('sha256:')
        if v.get('size') != r['size'] or oid != r['sha256']: raise ValueError('source size/LFS differs')


def preflight(root, reference, lock):
    if not all(p.is_absolute() and '..' not in p.parts for p in (root, reference, lock)):
        raise ValueError('absolute non-traversing task paths required')
    if root == ROOT or ROOT in root.parents or root in ROOT.parents:
        raise ValueError('raw output must be outside the deployment')
    from analysis import ur001_prepare as old
    from analysis.ur001_week26_bounded import OLD_WORKER_SHA256
    from analysis.new_instance_prepare import runtime
    if hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() != OLD_WORKER_SHA256:
        raise ValueError('original download worker drift')
    root = old.require_new_root(root, reference)
    env = runtime(lock)
    fixed = inventory()
    for section in ('train', 'validation'):
        url = f'https://huggingface.co/api/datasets/deepvk/VK-LSVD/tree/{REVISION}/{PREFIX}/{section}?recursive=false&expand=true'
        with urlopen(Request(url, headers={'User-Agent':'vkgr-a40-subset/1'}), timeout=30) as response:
            tree = json.load(response)
        check_tree(tree, [r for r in fixed['files'] if r['path'].startswith(f'{PREFIX}/{section}/')])
    disk = shutil.disk_usage(root.parent)
    required = sum(r['size'] for r in fixed['files'])
    if disk.free - required < disk.total * .2: raise ValueError('disk headroom below 20%')
    return {'status':'ready', 'engineering':False, 'payload_read':False, 'files':26,
            'inventory_sha256':FILES_SHA, 'required_bytes':required, **env}


def download(root, reference, lock):
    ready = preflight(root, reference, lock)
    from analysis import ur001_prepare as old
    root.mkdir(exist_ok=False)
    records = [old.download_record(root, r, REVISION) for r in inventory()['files']]
    value = {'status':'downloaded', 'schema':'a40-subset-raw-v1', 'files':records,
             'inventory_sha256':FILES_SHA, 'total_bytes':sum(r['size'] for r in records),
             'preflight':ready, 'scientific_derived':False}
    with (root/'download-manifest.json').open('x') as f: json.dump(value, f, sort_keys=True)
    return value


def job(args):
    from analysis.a40_ops import load_ops
    common, launcher, scheduler, status = load_ops(args.ops_code)
    run_id = 'a40-subset-'+args.stage+'-v1'
    argv = [sys.executable, '-B', '-m', 'analysis.a40_subset_prepare', args.stage,
            '--root', str(args.root), '--reference-root', str(args.reference_root),
            '--lock', str(args.lock), '--execute-on-new-instance']
    ns = launcher.build_parser().parse_args([
        '--root', str(args.ops_root), '--run-id', run_id, '--kind', 'ops', '--cwd', str(ROOT),
        '--config', str(FILES), '--input', str(args.lock), '--env', 'OMP_NUM_THREADS=1',
        '--cpu-cores', '1', '--ram-mib', '2048', '--vram-mib', '0', '--disk-gib', '1',
        '--wall-hours', '2', '--', *argv])
    spec = launcher.spec_from_args(ns)
    metadata = launcher.prepare(spec)
    if args.command == 'prepare':
        return {'status':'prepared_dry', 'spec':spec, 'metadata':metadata}
    root = launcher.resolve_root(str(args.ops_root))
    if (root/'runs'/run_id).exists():
        raise ValueError('run already exists')
    for state in ('pending','claimed','rejected'):
        if list((root/'queue'/state).glob('*-'+run_id+'.json')): raise ValueError('run already queued')
    path = scheduler.submit(root, spec)
    return {'status':'written', 'queue_file':path.name, 'run_id':run_id}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for name in ('preflight','download','prepare','submit'):
        c = sub.add_parser(name)
        for flag in ('root','reference-root','lock'): c.add_argument('--'+flag,type=Path,required=True)
        if name in ('preflight','download'):
            c.add_argument('--execute-on-new-instance',action='store_true',required=True)
        else:
            c.add_argument('--stage',choices=('preflight','download'),required=True)
            c.add_argument('--ops-code',required=True)
            c.add_argument('--ops-root',type=Path,required=True)
    a = p.parse_args(argv)
    if a.command == 'dry': result = {'status':'ready','files':len(inventory()['files']),'payload_read':False}
    elif a.command == 'preflight': result = preflight(a.root,a.reference_root,a.lock)
    elif a.command == 'download': result = download(a.root,a.reference_root,a.lock)
    else: result = job(a)
    print(json.dumps(result,sort_keys=True));return 0


if __name__ == '__main__': raise SystemExit(main())
