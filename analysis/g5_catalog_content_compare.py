"""One bounded read-only catalog comparison. No models/checkpoints/predictions.

Side loaders run sequentially under their original pinned deployment and loader.
Catalog IDs cross only an in-job pipe; the public report contains aggregates and
at most five differing indices. No input, existing run or bundle is written.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import datetime


def describe(values, newline=False):
    container_type = type(values).__name__
    dtype = str(getattr(values, 'dtype', 'python_integer_objects'))
    items = list(values)
    # Accept numpy integral scalars via __index__, reject lossy float conversion.
    import operator
    try:
        if any(isinstance(x, bool) for x in items):
            raise ValueError('boolean item id')
        clean = [operator.index(x) for x in items]
    except TypeError as exc:
        raise ValueError('integer item ids required') from exc
    if not clean or len(set(clean)) != len(clean):
        raise ValueError('nonempty unique catalog required')
    encoded = json.dumps(clean, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    original = encoded + (b'\n' if newline else b'')
    canonical = json.dumps(sorted(clean), separators=(',', ':')).encode()
    return {'values': clean, 'count': len(clean), 'container_type': container_type,
            'dtype': dtype, 'element_types': sorted({type(x).__name__ for x in items}),
            'digest_object': 'ordered list of integer item_id; no extra fields',
            'serialization': 'compact sorted-keys JSON UTF-8' + (' plus LF' if newline else ', no LF'),
            'original_digest': hashlib.sha256(original).hexdigest(),
            'serialized_bytes': len(original), 'normalised_digest': hashlib.sha256(canonical).hexdigest(),
            'strictly_ascending': all(a < b for a, b in zip(clean, clean[1:]))}


def compare(p, g):
    a, b = p['values'], g['values']
    same_set, same_order = set(a) == set(b), a == b
    differences = [i for i in range(max(len(a), len(b)))
                   if i >= len(a) or i >= len(b) or a[i] != b[i]]
    return {'branch': 3 if not same_set else (1 if same_order else 2),
            'same_set': same_set, 'same_order': same_order,
            'only_p_count': len(set(a) - set(b)), 'only_g_count': len(set(b) - set(a)),
            'different_positions_count': len(differences), 'different_positions': differences[:5],
            'normalisation': 'ascending integer item_id, compact UTF-8 JSON, no LF; no intersection or filling',
            'p': {k: v for k, v in p.items() if k != 'values'},
            'g': {k: v for k, v in g.items() if k != 'values'}}


def read_json(path, pin):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('absolute nonsymlink metadata path required')
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError('metadata sha256 mismatch')
    return json.loads(data)


def read_receipt(path, pin):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p, *p.parents)) or p.stat().st_size > 65536:
        raise ValueError('small absolute nonsymlink stdout receipt required')
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError('producer receipt sha256 mismatch')
    for line in reversed(data.decode().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'catalog_digest' in value:
            return value
    raise ValueError('catalog producer receipt missing')


def check_scratch(scratch, root=Path('/workspace/vkgr-infra01-root')):
    rid = os.environ.get('VKGR_RUN_ID')
    if rid != 'g5-catalog-content-diag-v1':
        raise ValueError('explicit new diagnostic run identity required')
    base = Path(root) / 'runs' / rid
    p = Path(scratch)
    if not p.is_absolute() or '..' in p.parts or p != base / 'catalog-scratch' or p.exists() \
            or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('scratch must be the fixed fresh child of this new run')
    meta = base / 'run.json'
    if meta.is_symlink() or not meta.is_file() or json.loads(meta.read_text()).get('run_id') != rid:
        raise ValueError('new run metadata identity required')
    return p


def native_description(catalog, digest):
    native = digest(list(catalog))
    for lf in (False, True):
        r = describe(catalog, newline=lf)
        if r['original_digest'] == native:
            return r
    raise ValueError('ALERT: native digest object/algorithm outside registered representation branches')


def side_load(side, job, scratch):
    """Only the accepted formal loader; G scratch is fresh within this new run."""
    check_scratch(scratch)
    argv = job['argv']; get = lambda key: argv[argv.index(key) + 1]
    root = Path(job['cwd'])
    import vkgr
    if Path(vkgr.__file__).resolve().parent != root / 'src/vkgr':
        raise ValueError('original deployment import identity differs')
    if side == 'p':
        from vkgr import retrieval_b1_run as b1, retrieval_feat02_run as fr, retrieval_seqm1_run as sm
        from vkgr.retrieval_php01 import FEATURE
        config = read_json(root / get('--config'), get('--config-sha'))
        sm.validate_config(config)
        spec = fr.validate_inputs(read_json(root / get('--inputs'), get('--inputs-sha')), FEATURE)
        data, _, _ = b1.real_inputs(spec)
        data, _ = fr.apply_feat02(data, spec, config['history_len'])
        fr._view(data, config['history_len'])
        r = native_description(data['catalog'], b1._digest)
        r['loader'] = 'real_inputs + apply_feat02 + fr._view; same catalog object'
    else:
        from vkgr import coll_sid_a1_run as cr
        value = cr._read_run(Path(get('--manifest')), get('--manifest-sha'))
        context, engineering, _ = cr._contexts(root, Path(get('--prepared')), get('--prepared-sha'),
            Path(get('--history-manifest')), get('--history-sha'), Path(get('--sid-manifest')), get('--sid-sha'),
            Path(get('--content')), get('--content-sha'), value['config'], scratch,
            input_manifest=Path(get('--input-manifest')), input_sha=get('--input-sha'),
            vectors_path=Path(get('--vectors')), vectors_sha=get('--vectors-sha'))
        if engineering or cr._digest(context.binding) != value['binding']['context_binding_sha256']:
            raise ValueError('formal context/accepted binding required')
        fresh = cr._source(root)
        if fresh['files'] != value['binding']['source']['files']:
            raise ValueError('accepted G source ledger differs')
        r = native_description(context.catalog, cr._digest)
        r['loader'] = 'coll_sid_a1_run._contexts().catalog, engineering=False'
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('diagnostic must not initialise CUDA')
    r['cuda_initialized'] = False
    return r


def collect(req):
    if set(req) != {'p_job', 'p_job_sha', 'g_job', 'g_job_sha', 'p_metadata', 'p_metadata_sha', 'g_metadata', 'g_metadata_sha', 'scratch'}:
        raise ValueError('exact diagnostic request required')
    check_scratch(req['scratch'])
    result = {}
    for side in ('p', 'g'):
        job = read_json(req[side + '_job'], req[side + '_job_sha'])
        if job['run_id'] != ('g5-fusion-qmeta' if side == 'p' else 'g5-fusion-asm-s42'):
            raise ValueError('only accepted qmeta / G seed42 assembly loaders')
        # Job argv input paths were relative to its accepted cwd; retain that cwd.
        env = dict(os.environ)
        entries = job['env'].items() if isinstance(job['env'], dict) else (x.split('=', 1) for x in job['env'])
        env.update(entries); env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['VKGR_RUN_ID'] = os.environ['VKGR_RUN_ID']
        payload = {'side': side, 'job': job, 'scratch': req['scratch']}
        child = subprocess.run([job['argv'][0], '-B', str(Path(__file__).resolve()), '--side'],
            input=json.dumps(payload), text=True, capture_output=True, env=env, cwd=job['cwd'], timeout=1500, check=True)
        r = json.loads(child.stdout)
        header = read_receipt(req[side + '_metadata'], req[side + '_metadata_sha'])
        if r['original_digest'] != header['catalog_digest'] or r['count'] != 120415:
            raise ValueError('actual catalog does not reproduce its accepted artifact header')
        result[side] = r
    report = compare(result['p'], result['g'])
    report.update(status='catalog_content_compared', read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_pins=req,
        test_read=False, checkpoint_read=False, prediction_read=False,
        original_artifacts_modified=False, scratch_scope='new run only; original formal loader scratch',
        catalog_exported=False)
    run_meta = Path('/workspace/vkgr-infra01-root/runs') / os.environ['VKGR_RUN_ID'] / 'run.json'
    report.update(run_id=os.environ['VKGR_RUN_ID'], run_manifest_sha256=hashlib.sha256(run_meta.read_bytes()).hexdigest())
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--request'); parser.add_argument('--request-sha'); parser.add_argument('--side', action='store_true')
    args = parser.parse_args()
    if args.side:
        p = json.loads(sys.stdin.read()); answer = side_load(p['side'], p['job'], p['scratch'])
    else:
        answer = collect(read_json(args.request, args.request_sha))
    print(json.dumps(answer, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
