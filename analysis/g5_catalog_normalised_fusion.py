"""G5 branch-one representation normalisation. Original artifacts stay intact.

One finite CPU queue job: check normalisation in memory, write fresh inputs,
original fuse check-only, original primary fuse, full deterministic replay.
No model/checkpoint/test, no new arm, no changed fusion or statistical method.
"""
import hashlib
import json
import contextlib
import io
import os
from pathlib import Path
import sys
import datetime

PROOF_SHA = '42959f7c75501e940662b3b6e50db270ee4ef86f93ba60403d39cfc7be28b521'
DIAG_SCRIPT_SHA = 'f5a255cb9dc1e0f04d8f5997d537c86a2da758b14918929d824c8446a00708eb'
DIAG_RUN_SHA = '2e396cc9267ee95ade04a444ef1fe16cec24b5c6293b0aa79fffe4bd6e23a2c6'
P_DIGEST = 'abe2e7a3291f45eb7d535cac574c3ffae8192c79ff726fc25cd6349543152cf5'
G_DIGEST = '80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3'
FUSE_SHA = '015f30acb630226b5a787170b1dc1d038b7c3fd7ac60d2cce00f095fc3432cd7'
CORE_SHA = '46c562257c372a46b550ca442ab417dc8551345ddf4c80f3d9f8a18fc1fe9537'
ROLES = ('p', '42', '2024', '2025')
FORMULA = 'ascending integer item_id; compact UTF-8 JSON; no trailing LF'
META_SHA = '719fd2e28b41f49097cc3407f6d67604faa5d4bb637d8468e5480ac23ac7af54'
G_SHAS = {'42': '194294946cafe2c861a55aa51ae5a32d2f4f2274f4b43e558070d4139e016ef9',
          '2024': 'be27305e5f0910a30a46208472d10a0fbb536dc77d3ebc932e069b2d32da8be2',
          '2025': '1b6978a8b016326822bac206b8f92c5a11f54497d033f1ae4952e379b9d967fd'}
P_SHAS = {'42': 'be5eb969d47c453105c538f7f57a87ee3eff64b28666bf321fe196c677ce576b',
          '2024': '531a6c28eea6952f583fe6286648b8e0375755696ad2329802d245c719e3b989',
          '2025': '38ad413b07ed8b82342749d3f58be8438e99a2388253c06709b838f160413439'}


def normalise(records, proof, source_pins):
    expected = {'status': 'catalog_content_compared', 'run_id': 'g5-catalog-content-diag-v1',
        'script_sha256': DIAG_SCRIPT_SHA, 'run_manifest_sha256': DIAG_RUN_SHA,
        'same_set': True, 'same_order': True, 'branch': 1, 'only_p_count': 0, 'only_g_count': 0,
        'different_positions_count': 0, 'test_read': False, 'checkpoint_read': False, 'prediction_read': False}
    if any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items()):
        raise ValueError('accepted branch-one diagnostic binding required')
    for role, digest in (('p', P_DIGEST), ('g', G_DIGEST)):
        side = proof[role]
        if side['count'] != 120415 or side['strictly_ascending'] is not True \
                or side['original_digest'] != digest or side['normalised_digest'] != P_DIGEST:
            raise ValueError('diagnostic catalog facts differ')
    if set(records) != set(ROLES) or set(source_pins) != set(ROLES):
        raise ValueError('all four original inputs required')
    out = {}
    for role, value in records.items():
        expected_digest = P_DIGEST if role == 'p' else G_DIGEST
        if value.get('catalog_size') != 120415 or value.get('catalog_digest') != expected_digest or 'catalog_normalisation' in value:
            raise ValueError('native catalog identity differs or normalisation repeated')
        out[role] = {**value, 'catalog_digest': P_DIGEST, 'catalog_normalisation': {
            'schema': 'g5-catalog-normalisation-v1', 'proof_sha256': PROOF_SHA,
            'original_file_sha256': source_pins[role], 'original_catalog_digest': expected_digest,
            'formula': FORMULA, 'normalised_digest': P_DIGEST}}
    return out


def run_commands(fuse, args):
    logs = []
    for suffix in (['--check-only'], []):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = fuse.main([*args, *suffix])
        logs.append(stream.getvalue())
        if code != 0:
            raise ValueError('ALERT: original fusion preflight/formal command failed; no retry')
        print(json.dumps({'stage': 'preflight' if suffix else 'primary', 'receipt': logs[-1]}), flush=True)
    return logs


def verify_primary(fuse, result, p_inputs, g_inputs, meta_input):
    inputs = {'p_export': {str(s):v[1] for s,v in p_inputs.items()},
              'g_assembly': {str(s):v[1] for s,v in g_inputs.items()}, 'p_query_meta': meta_input[1]}
    provenance = result.get('provenance', {})
    if result.get('schema') != 'g5-fusion-results-v1' or result.get('inputs') != inputs \
            or provenance.get('fuse_script_sha256') != FUSE_SHA or provenance.get('fusion_v2_script_sha256') != CORE_SHA:
        raise ValueError('fusion result provenance or input binding differs')
    routes, queries, size, checks = fuse.build_inputs(p_inputs, g_inputs, meta_input)
    replay = fuse.fv.run(routes, queries, size)
    # Compare JSON representations: tuple/list distinctions are not file semantics.
    encode = lambda x: json.dumps(x, sort_keys=True, allow_nan=False)
    if encode(result['fusion']) != encode(replay) or encode(result['checks']) != encode(checks):
        raise ValueError('fusion deterministic full replay mismatch')
    return True


def new_output(path, root=Path('/workspace/vkgr-infra01-root')):
    rid = os.environ.get('VKGR_RUN_ID')
    base = Path(root) / 'runs/g5-fusion-primary-v1'
    out = Path(path)
    if rid != base.name or out != base/'out' or not out.is_absolute() or '..' in out.parts or out.exists() \
            or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('fresh output of this one new primary run required')
    run = base/'run.json'
    if not run.is_file() or run.is_symlink() or json.loads(run.read_text()).get('run_id') != rid:
        raise ValueError('primary run metadata identity required')
    return out


def write_new(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, pin):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('absolute nonsymlink input required')
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError('pinned input hash mismatch')
    return json.loads(data)


def pipeline(req, fuse):
    if set(req) != {'proof', 'meta', 'g', 'p', 'output_root'} or set(req['p']) != set(P_SHAS) or set(req['g']) != set(G_SHAS):
        raise ValueError('exact frozen fusion request required')
    references = [req['proof'], req['meta'], *req['p'].values(), *req['g'].values()]
    if req['proof'][1] != PROOF_SHA or req['meta'][1] != META_SHA \
            or any(req['p'][s][1] != P_SHAS[s] or req['g'][s][1] != G_SHAS[s] for s in P_SHAS):
        raise ValueError('only the accepted original artifacts and diagnostic are allowed')
    fuse.refuse_held_out_paths([x[0] for x in references] + [req['output_root']])
    out = new_output(req['output_root'])
    proof = read_pinned(*req['proof'])
    records = {'p': read_pinned(*req['meta']), **{s: read_pinned(*req['g'][s]) for s in G_SHAS}}
    pins = {'p': META_SHA, **G_SHAS}
    # This is the normaliser's true-input read-only preflight; no output yet.
    normal = normalise(records, proof, pins)
    print(json.dumps({'status': 'normalisation_preflight_ok', 'written': False,
                      'proof_sha256': PROOF_SHA, 'original_files_sha256': pins,
                      'normalised_digest': P_DIGEST}), flush=True)
    out.mkdir(exist_ok=False); normal_dir = out/'normalised-inputs'; normal_dir.mkdir()
    paths = {}; normal_shas = {}
    for role in ROLES:
        path = normal_dir/(role+'.json'); normal_shas[role] = write_new(path, normal[role]); paths[role] = str(path)
    normal_manifest = {'schema': 'g5-catalog-normalisation-manifest-v1', 'proof_sha256': PROOF_SHA,
        'formula': FORMULA, 'normalised_digest': P_DIGEST, 'original_files_sha256': pins,
        'output_files_sha256': normal_shas,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    normal_manifest_sha = write_new(out/'catalog-normalisation-manifest.json', normal_manifest)
    # Exact round trip: no candidates, labels, ranks or other fields were changed.
    for role in ROLES:
        reloaded = read_pinned(paths[role], normal_shas[role])
        if reloaded != normal[role] or {k:v for k,v in reloaded.items() if k not in ('catalog_digest','catalog_normalisation')} \
                != {k:v for k,v in records[role].items() if k != 'catalog_digest'}:
            raise ValueError('normalisation full-field replay mismatch')
    p_inputs = {int(s): tuple(req['p'][s]) for s in P_SHAS}
    g_inputs = {int(s): (paths[s], normal_shas[s]) for s in G_SHAS}
    meta = (paths['p'], normal_shas['p'])
    args = ['--split', 'validation', '--output-dir', str(out/'fusion'), '--p-query-meta', '@'.join(meta)]
    for s in P_SHAS:
        args += ['--p-export', s+'='+'@'.join(p_inputs[int(s)]), '--g-assembly', s+'='+'@'.join(g_inputs[int(s)])]
    command_receipts = run_commands(fuse, args)
    result_path = out/'fusion/fusion-results.json'; result_data = result_path.read_bytes()
    result = json.loads(result_data)
    verify_primary(fuse, result, p_inputs, g_inputs, meta)
    import resource
    run_file = out.parent/'run.json'
    manifest = {'schema': 'g5-normalised-fusion-verification-v1', 'status': 'verified', 'run_id': out.parent.name,
        'run_manifest_sha256': hashlib.sha256(run_file.read_bytes()).hexdigest(),
        'normalisation_manifest_sha256': normal_manifest_sha, 'proof_sha256': PROOF_SHA,
        'results_sha256': hashlib.sha256(result_data).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fuse_script_sha256': FUSE_SHA, 'fusion_v2_script_sha256': CORE_SHA,
        'command_receipts_sha256': [hashlib.sha256(x.encode()).hexdigest() for x in command_receipts],
        'normalised_digest': P_DIGEST, 'normalisation': FORMULA, 'full_replay': True,
        'test_read': False, 'checkpoint_read': False, 'old_artifacts_modified': False,
        'cuda_initialized': False, 'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'read_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    manifest_sha = write_new(out/'manifest.json', manifest)
    return {**manifest, 'manifest_sha256': manifest_sha}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True); parser.add_argument('--request-sha', required=True)
    args = parser.parse_args()
    from vkgr_g5 import fuse
    for module, pin in ((fuse, FUSE_SHA), (fuse.fv, CORE_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != pin:
            raise ValueError('frozen original fusion source differs')
    print(json.dumps(pipeline(read_pinned(args.request, args.request_sha), fuse), sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(json.dumps({'status': 'ALERT', 'error': str(exc), 'no_retry': True}), flush=True)
        raise SystemExit(2)
