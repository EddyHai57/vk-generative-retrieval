"""G6-PREP tokenizer pipeline for depth-4 semantic IDs: fit and verify SIDMAP_V2D bundles (new file; ``sid_tok_run`` is imported).

Exactly two data inputs are accepted, each pinned by sha256: the CONTENT_V1 item vectors (.npy, row 0 is the all-zero PAD row)
and the catalog id list (.json). No label, query, split or interaction file is opened.

Variants (every constant comes from the frozen fit config ``configs/g6prep/g-g6-tok.json``):
  B16D4          the frozen balanced k-means ``sid.fit`` with branching 16, depth 4, 10 iterations, seed 42  (SID_BHKM_B16D4_S42)
  B16D3_REPLAY   the same pipeline at depth 3; its mapping must equal the anchor mapping byte for byte        (SID_V2D_REPLAY_BHKM_B16D3)
  RQ256D4        RQ-VAE, four codebooks of 256, all other constants as pre-registered for GTOK-01             (SID_RQVAE_B256D4_S42)
Only the T1 line of GTOK-01 has a depth-4 RQ-VAE version; there is no 16-ary RQ-VAE variant here.

Exit codes: 0 success, 1 argument / input / validation error (nothing written), 3 gate failure (the output directory then
holds failure.json and never a run_manifest.json).
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

from . import generative_data as gd
from . import sid as bhkm
from . import sid_depth as sd
from . import sid_rqvae as rq
from . import sid_rqvae_d as rd
from . import sid_tok_audit as au
from . import sid_tok_audit_d as ad
from . import sid_tok_run as run
from . import sid_v2d
from .data_audit import file_sha256

CONFIG_SCHEMA = 'g-g6-tok-fit-v1'
FAILURE_SCHEMA = 'g-g6-failure-v1'
TASK_ID = sid_v2d.TASK_ID
EXIT_OK, EXIT_ERROR, EXIT_GATE = run.EXIT_OK, run.EXIT_ERROR, run.EXIT_GATE
VARIANTS = ('B16D4', 'B16D3_REPLAY', 'RQ256D4')
VARIANT_LABELS = {'SID_BHKM_B16D4_S42': 'B16D4', 'SID_V2D_REPLAY_BHKM_B16D3': 'B16D3_REPLAY', 'SID_RQVAE_B256D4_S42': 'RQ256D4'}
PRODUCER_FILES = ('sid_tok_run_d.py', 'sid_v2d.py', 'sid_depth.py', 'sid_tok_audit_d.py', 'sid_rqvae_d.py', 'sid_rqvae.py',
                  'sid_tok_run.py', 'sid_v2.py', 'sid_tok_audit.py', 'sid.py')


def expected_config():
    """The one approved fit configuration (every pre-registered constant); configs/g6prep/g-g6-tok.json equals it."""
    rqvae = copy.deepcopy(run._FROZEN_PREREG)
    rqvae['levels'] = 4  # the only pre-registered constant that differs from GTOK-01
    rqvae['usage_diag_every'] = run.USAGE_DIAG_EVERY
    return {
        'schema': CONFIG_SCHEMA, 'task_id': TASK_ID,
        'content': {'feature_array_sha256': au.CONTENT_V1_FEATURE_SHA256, 'dim': rqvae['content_dim']},
        'rqvae': rqvae,
        'variants': {
            'B16D4': {'sid_version': 'SID_BHKM_B16D4_S42', 'kind': 'bhkm', 'branching': 16, 'depth': 4, 'max_iter': 10, 'seed': 42,
                      'prereg': copy.deepcopy(ad.PREREG_BHKM_B16D4)},
            'B16D3_REPLAY': {'sid_version': 'SID_V2D_REPLAY_BHKM_B16D3', 'kind': 'bhkm', 'branching': 16, 'depth': 3, 'max_iter': 10, 'seed': 42},
            'RQ256D4': {'sid_version': 'SID_RQVAE_B256D4_S42', 'kind': 'rqvae', 'codebook_size': 256, 'levels': 4},
        },
    }


def validate_config(value):
    """Strict whitelist: key set, types and values must equal the approved configuration exactly."""
    found = run._first_difference(value, expected_config())
    if found:
        raise ValueError(f'unapproved fit config at {found}')
    return value


def load_config(path, digest):
    return validate_config(gd._json(run._check_file_sha(path, digest)))


def _fit_params(config, variant, overrides):
    params = run._fit_params(config, overrides)
    params['levels'] = variant['levels']
    return params


def canonical_sids_d(ids, codes):
    """Rows sorted by ascending item id, suffix assigned canonically; returns (ids uint32, sids uint32 [N, depth+1])."""
    ids, codes = np.asarray(ids), np.asarray(codes)
    if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= 2 ** 32 or codes.ndim != 2 or len(codes) != len(ids):
        raise ValueError('invalid ids/codes')
    order = np.argsort(ids, kind='stable')
    ordered = ids[order].astype(np.uint32)
    return ordered, sd.assign_suffix(ordered, codes[order])


def _fit_frozen_kmeans(ids, content, spec):
    """The unmodified frozen sid.fit; returns the nodes-free catalog and a compact summary."""
    model = bhkm.fit(ids, content, branching=spec['branching'], depth=spec['depth'], seed=spec['seed'], max_iter=spec['max_iter'])
    summary = run.bhkm_summary(model.nodes)
    catalog = run._drop_nodes(model)
    return {'catalog': catalog, 'ids': catalog.ids, 'sids': catalog.sids, 'summary': summary}


def fit_rqvae_variant_d(config, name, ids, content, *, device, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256,
                        engineering=False, overrides=None, clock=time.monotonic):
    """Fit and judge the depth-4 RQ-VAE. ok is True only when training completed and every gate passed; only then is the
    model (weights, mapping) part of the result."""
    variant = config['variants'][name]
    if variant['kind'] != 'rqvae':
        raise ValueError('not an RQ-VAE variant')
    overrides = dict(overrides or {})
    K, levels = variant['codebook_size'], variant['levels']
    params = _fit_params(config, variant, overrides)
    started = clock()
    raw = rd.fit_d(content, K=K, device=device, clock=clock, **params)
    base = run._base(name, variant, engineering, content, run._observations(device, clock() - started), params, K)
    base['depth'] = levels
    if raw['status'] != 'complete':
        return run._failed_result(base, raw)
    state = raw['state']
    codes = rd.export_codes_d(state, content)
    final_ids, sids = canonical_sids_d(ids, codes)
    mapping = ad.audit_mapping_d(final_ids, sids, content, branching=K, kind='rqvae', depth=levels)
    info = {'status': raw['status'], 'steps': int(raw['step']), 'max_steps': params['max_steps'], 'elapsed': float(raw['elapsed']),
            'wall_cap_seconds': params['wall_cap_seconds'], 'required_steps': params['max_steps'],
            'reconstruction': rd.reconstruction_report_d(state, content)}
    gates = ad.gates_d(mapping, 'rqvae', info, depth=levels, expected_content_sha256=expected_content_sha256, required_steps=params['max_steps'])
    diagnostic = run.usage_diagnostic(K, raw['usage_log'], raw['step'], rd.codebook_usage_d(codes, K))
    detail = {'step': raw['step'], 'elapsed': float(raw['elapsed']), 'loss_log': raw['loss_log'], 'usage_log': raw['usage_log'],
              'usage_diagnostic': diagnostic}
    audit = {'mapping': mapping, 'rq': info}
    if not gates['passed']:
        return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, 'detail': detail}
    tokenizer = {'variant': name, 'schema': CONFIG_SCHEMA, 'fit': dict(params), 'effective_prereg': run._jsonable(rq.PREREG),
                 'engineering_overrides': overrides, 'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum())}
    train_log = {'K': K, 'seed': params['seed'], 'init_digest': raw['init_digest'], 'prefix_digest': raw['prefix_digest'],
                 'loss_log': raw['loss_log'], 'usage_log': raw['usage_log'], 'usage_diagnostic': diagnostic,
                 'observations': base['observations']}
    return {**base, 'ok': True, 'status': 'complete', 'ids': final_ids, 'sids': sids, 'weights': state, 'audit': audit, 'gates': gates,
            'tokenizer': tokenizer, 'train_log': train_log, 'detail': detail}


def fit_bhkm_variant_d(config, name, ids, content, *, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256, engineering=False,
                       clock=time.monotonic):
    """Fit and judge B16D4 or the depth-3 replay with the unmodified frozen sid.fit; the stored catalog has no nodes."""
    spec = config['variants'][name]
    if spec['kind'] != 'bhkm':
        raise ValueError('not a balanced k-means variant')
    params = {key: spec[key] for key in ('branching', 'depth', 'max_iter', 'seed')}
    started = clock()
    fitted = _fit_frozen_kmeans(ids, content, spec)
    base = run._base(name, spec, engineering, content, run._observations('cpu', clock() - started), params, spec['branching'])
    base['depth'] = spec['depth']
    mapping = ad.audit_mapping_d(fitted['ids'], fitted['sids'], content, branching=spec['branching'], kind='bhkm', depth=spec['depth'])
    prereg = sid_v2d.prereg_for(spec['sid_version'], len(fitted['ids']), engineering)
    gates = ad.gates_d(mapping, 'bhkm', None, depth=spec['depth'], expected_content_sha256=expected_content_sha256, prereg=prereg)
    audit = {'mapping': mapping, 'rq': None}
    detail = {'summary': fitted['summary']}
    if not gates['passed']:
        return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, 'detail': detail}
    tokenizer = {'variant': name, 'schema': CONFIG_SCHEMA, 'fit': params, 'summary': fitted['summary'], 'engineering_overrides': {},
                 'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum()), 'observations': base['observations']}
    return {**base, 'ok': True, 'status': 'complete', 'catalog': fitted['catalog'], 'ids': fitted['ids'], 'sids': fitted['sids'],
            'weights': None, 'train_log': None, 'audit': audit, 'gates': gates, 'tokenizer': tokenizer, 'detail': detail}


def _producer():
    root = Path(__file__).resolve().parent
    return {f'src/vkgr/{name}': file_sha256(root / name) for name in PRODUCER_FILES}


def write_success_bundle_d(output, result, *, config_sha):
    if result['ok'] is not True:
        raise ValueError('only a variant whose gates passed can be published')
    return sid_v2d.write_bundle_d(
        output, sid_version=result['sid_version'], engineering=result['engineering'], ids=result['ids'], sids=result['sids'],
        content_feature_sha256=result['content_digest'], tokenizer=result['tokenizer'], audit=result['audit'], gates=result['gates'],
        fit_config_sha256=config_sha, producer=_producer(), weights=result['weights'], train_log=result['train_log'])


def failure_record_d(result, *, config_sha, catalog_sha256):
    record = run.failure_record(result, config_sha=config_sha, catalog_sha256=catalog_sha256)
    record.update({'schema': FAILURE_SCHEMA, 'task_id': TASK_ID, 'depth': result['depth']})
    return record


def _label(sid_version):
    try:
        return VARIANT_LABELS[sid_version]
    except KeyError as exc:
        raise ValueError('bundle is not one of the G6 tokenizers') from exc


def _check_config_binding_d(directory, loaded, config, config_sha, engineering):
    """The bundle must have been fitted with exactly this (validated) config file and with nothing else overridden."""
    if loaded.manifest['fit_config_sha256'] != config_sha:
        raise ValueError('bundle was fitted with a different config file')
    tokenizer = gd._json(directory / 'tokenizer.json')['config']
    name = _label(loaded.info['sid_version'])
    variant = config['variants'][name]
    if variant['kind'] == 'bhkm':
        ok = tokenizer['fit'] == {key: variant[key] for key in ('branching', 'depth', 'max_iter', 'seed')} and tokenizer['engineering_overrides'] == {}
    else:
        overrides = tokenizer['engineering_overrides']
        ok = type(overrides) is dict and set(overrides) <= set(run.OVERRIDE_OPTIONS) and (engineering or not overrides) \
            and tokenizer['fit'] == run._jsonable(_fit_params(config, variant, overrides)) \
            and (engineering or tokenizer['effective_prereg'] == run._FROZEN_PREREG)
    if not ok:
        raise ValueError('tokenizer.json does not match the config it was fitted with')


def _replay_prefix_digest(config, K, levels, content, device, engineering):
    """Re-run the first probe_step training steps under the constants the bundle recorded; returns the digest."""
    params = config['fit']
    with run._prereg_context(config['effective_prereg'], engineering):
        out = rd.fit_d(content, K=K, levels=levels, device=device, seed=params['seed'], max_steps=params['probe_step'],
                       batch_size=params['batch_size'], wall_cap_seconds=run.NO_WALL_CAP, log_every=params['log_every'],
                       eval_every=params['eval_every'], check_every=params['check_every'], probe_from=params['probe_from'],
                       probe_step=params['probe_step'])
    return out['prefix_digest'] if out['status'] == 'complete' else None


def _replay_prefix(directory, loaded, content, device):
    if loaded.info['kind'] != 'rqvae':
        raise ValueError('--replay-prefix applies to RQ-VAE bundles only')
    config = gd._json(directory / 'tokenizer.json')['config']
    log = gd._json(directory / 'train-log.json')
    if log['prefix_digest'] is None:
        raise ValueError('the bundle recorded no prefix digest (fewer steps than the probe step)')
    device = run.resolve_device('rqvae', device or log['observations']['device'], True)
    if _replay_prefix_digest(config, loaded.info['branching'], loaded.info['depth'], content, device, loaded.manifest['engineering']) != log['prefix_digest']:
        raise ValueError('prefix replay does not reproduce the recorded digest')


def _replay_fit(directory, loaded, inputs):
    if loaded.info['kind'] != 'bhkm':
        raise ValueError('--replay-fit applies to balanced k-means bundles only')
    config = gd._json(directory / 'tokenizer.json')['config']
    again = _fit_frozen_kmeans(inputs['ids'], inputs['content'], config['fit'])
    stored = loaded.codec.catalog
    if not (np.array_equal(again['ids'], stored.ids) and np.array_equal(again['sids'], stored.sids)) or again['summary'] != config['summary']:
        raise ValueError('fit replay does not reproduce the stored mapping')


def _baseline_report(loaded, baseline, inputs):
    """Compare with the anchor mapping npz (ids, sids). Same depth: must be byte-equal. Deeper: report the prefix agreement."""
    data = np.load(run._check_file_sha(baseline[0], baseline[1]), allow_pickle=False)
    if not hasattr(data, 'files') or set(data.files) != {'ids', 'sids'}:
        raise ValueError('baseline mapping must hold exactly ids and sids')
    ids, sids = data['ids'], data['sids']
    if ids.dtype != np.uint32 or sids.dtype != np.uint32 or sids.ndim != 2 or not np.array_equal(ids, inputs['ids']):
        raise ValueError('baseline mapping does not match the catalog')
    depth, base_depth = loaded.info['depth'], sids.shape[1] - 1
    stored = loaded.codec.catalog
    if depth == base_depth:
        if not np.array_equal(stored.sids, sids):
            raise ValueError('baseline mapping differs from the bundle mapping')
        return {'mode': 'exact', 'equal': True}
    if depth < base_depth:
        raise ValueError('baseline is deeper than the bundle')
    return {'mode': 'prefix_agreement', **ad.prefix_agreement(stored.ids, stored.sids, ids, sids, base_depth)}


def verify_bundle_d(manifest_path, manifest_sha, inputs, config, config_sha, *, engineering=False, replay_prefix=False, replay_fit=False,
                    baseline=None, device=None):
    loaded = sid_v2d.load_bundle_d(manifest_path, manifest_sha, catalog_ids=inputs['ids'], content=inputs['content'],
                                   expected_content_digest=run._expected_digest(inputs, engineering), engineering=engineering)
    directory = Path(manifest_path).absolute().parent
    _check_config_binding_d(directory, loaded, config, config_sha, engineering)
    checks = ['load_bundle', 'config_binding']
    if replay_prefix:
        _replay_prefix(directory, loaded, inputs['content'], device)
        checks.append('replay_prefix')
    if replay_fit:
        _replay_fit(directory, loaded, inputs)
        checks.append('replay_fit')
    result = {'sid_version': loaded.info['sid_version'], 'depth': loaded.info['depth'], 'manifest_sha256': manifest_sha,
              'engineering': engineering, 'checks': checks, 'replay_prefix': replay_prefix, 'replay_fit': replay_fit}
    if baseline is not None:
        result['baseline'] = _baseline_report(loaded, baseline, inputs)
    return result


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f'argument error: {message}')


def _parser():
    parser = _Parser(prog='python -m vkgr.sid_tok_run_d', allow_abbrev=False, description='G6-PREP depth-4 tokenizer fit / verify')
    sub = parser.add_subparsers(dest='command', required=True)

    def pinned(p, name):
        p.add_argument(f'--{name}', required=True)
        p.add_argument(f'--{name}-sha', required=True)

    fit = sub.add_parser('fit', allow_abbrev=False)
    fit.add_argument('--variant', required=True)
    for name in ('config', 'content', 'catalog'):
        pinned(fit, name)
    fit.add_argument('--output', required=True)
    fit.add_argument('--device')
    fit.add_argument('--engineering', action='store_true')
    fit.add_argument('--max-steps', type=int)
    fit.add_argument('--batch-size', type=int)
    fit.add_argument('--wall-cap-seconds', type=float)
    fit.add_argument('--probe-from', type=int)
    fit.add_argument('--probe-step', type=int)
    verify = sub.add_parser('verify', allow_abbrev=False)
    verify.add_argument('--bundle', required=True)
    verify.add_argument('--sha', required=True)
    for name in ('config', 'content', 'catalog'):
        pinned(verify, name)
    verify.add_argument('--engineering', action='store_true')
    verify.add_argument('--replay-prefix', action='store_true')
    verify.add_argument('--replay-fit', action='store_true')
    verify.add_argument('--device')
    verify.add_argument('--baseline-mapping')
    verify.add_argument('--baseline-sha')
    return parser


def _cmd_fit(args, clock):
    if args.variant not in VARIANTS:
        raise ValueError('variant must be B16D4, B16D3_REPLAY or RQ256D4')
    overrides = {name: getattr(args, name) for name in run.OVERRIDE_OPTIONS if getattr(args, name) is not None}
    kind = expected_config()['variants'][args.variant]['kind']
    if overrides and not args.engineering:
        raise ValueError('--max-steps, --batch-size, --wall-cap-seconds and --probe-* are engineering-only')
    if overrides and kind != 'rqvae':
        raise ValueError('training length options apply to the RQ-VAE variant only')
    if args.engineering and kind == 'rqvae' and 'max_steps' not in overrides:
        raise ValueError('--engineering needs an explicit --max-steps')
    output = run._new_output(args.output)
    run.choose_device(kind, args.device, args.engineering)
    config = load_config(args.config, args.config_sha)
    inputs = run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    device = run.resolve_device(kind, args.device, args.engineering)
    if not args.engineering and run._jsonable(rq.PREREG) != run._FROZEN_PREREG:
        raise ValueError('live RQ-VAE constants differ from the pre-registered ones')
    common = {'expected_content_sha256': run._expected_digest(inputs, args.engineering), 'engineering': args.engineering, 'clock': clock}
    if kind == 'rqvae':
        result = fit_rqvae_variant_d(config, args.variant, inputs['ids'], inputs['content'], device=device, overrides=overrides, **common)
    else:
        result = fit_bhkm_variant_d(config, args.variant, inputs['ids'], inputs['content'], **common)
    if result['ok']:
        manifest = write_success_bundle_d(output, result, config_sha=args.config_sha)
        run._say({'status': 'success', 'variant': args.variant, 'sid_version': manifest['sid_version'], 'depth': manifest['depth'],
                  'manifest_sha256': file_sha256(output / sid_v2d.MANIFEST_NAME), 'engineering': args.engineering})
        return EXIT_OK
    record = failure_record_d(result, config_sha=args.config_sha, catalog_sha256=inputs['catalog_sha256'])
    run.write_failure_record(output, record)
    run._say({'status': record['status'], 'variant': args.variant, 'reasons': record['reasons']})
    return EXIT_GATE


def _cmd_verify(args):
    if (args.baseline_mapping is None) != (args.baseline_sha is None):
        raise ValueError('--baseline-mapping and --baseline-sha go together')
    config = load_config(args.config, args.config_sha)
    inputs = run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    baseline = None if args.baseline_mapping is None else (args.baseline_mapping, args.baseline_sha)
    run._say(verify_bundle_d(args.bundle, args.sha, inputs, config, args.config_sha, engineering=args.engineering,
                             replay_prefix=args.replay_prefix, replay_fit=args.replay_fit, baseline=baseline, device=args.device))
    return EXIT_OK


def main(argv=None, *, clock=time.monotonic):
    try:
        args = _parser().parse_args(argv)
        return _cmd_fit(args, clock) if args.command == 'fit' else _cmd_verify(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return EXIT_ERROR


if __name__ == '__main__':
    raise SystemExit(main())
