"""GTOK-01 tokenizer pipeline: fit (T1, T2, T3), verify and compare for SIDMAP_V2 bundles.

Exactly two data inputs are accepted, each pinned by sha256: the CONTENT_V1 item vectors (.npy, row 0 is the
all-zero PAD row) and the catalog id list (.json). No label, query, split or interaction file is opened.

Variants (all constants come from the frozen fit config, which mirrors sid_rqvae.PREREG):
  T1  RQ-VAE, three codebooks of 256   (SID_RQVAE_B256D3_S42)
  T2  RQ-VAE, three codebooks of 16    (SID_RQVAE_B16D3_S42)
  T3  the frozen balanced k-means sid.fit, 256-ary, depth 3, 10 iterations, seed 42 (SID_BHKM_B256D3_S42);
      its node list is dropped right after the fit and only a compact summary is kept.

Exit codes: 0 success, 1 argument / input / validation error (nothing written), 3 gate failure (the output
directory then holds failure.json and never a run_manifest.json).
"""
import argparse
import contextlib
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import generative_data as gd
from . import sid as bhkm
from . import sid_rqvae as rq
from . import sid_tok_audit as au
from . import sid_v2
from .data_audit import canonical_bytes, file_sha256
from .sid import SIDCatalog

CONFIG_SCHEMA = 'g-tok-01-fit-v1'
FAILURE_SCHEMA = 'g-tok-01-failure-v1'
COMPARE_SCHEMA = 'g-tok-01-compare-v1'
TASK_ID = sid_v2.TASK_ID
EXIT_OK, EXIT_ERROR, EXIT_GATE = 0, 1, 3
USAGE_DIAG_EVERY = 2000  # per-level codebook usage is logged at this cadence; step 2000 and the last step are reported
USAGE_DIAG_STEP = 2000
VARIANTS = ('T1', 'T2', 'T3')
VARIANT_LABELS = {'SID_RQVAE_B256D3_S42': 'T1', 'SID_RQVAE_B16D3_S42': 'T2', 'SID_BHKM_B256D3_S42': 'T3'}
BASELINE_LABEL = 'T0'
BASELINE_BRANCHING = 16  # T0 = SID_BHKM_B16D3_S42, the frozen 16-ary mapping
OVERRIDE_OPTIONS = ('max_steps', 'batch_size', 'wall_cap_seconds', 'probe_from', 'probe_step')
PRODUCER_FILES = ('sid_tok_run.py', 'sid_rqvae.py', 'sid_v2.py', 'sid_tok_audit.py', 'sid.py')
NO_WALL_CAP = 1e18  # replay runs measure nothing about speed


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


_FROZEN_PREREG = _jsonable(rq.PREREG)  # snapshot at import: later patches of the live dict cannot move the frozen config


def expected_config():
    """The one approved fit configuration (every pre-registered constant); configs/gtok01/g-tok-01.json equals it."""
    rqvae = copy.deepcopy(_FROZEN_PREREG)
    rqvae['usage_diag_every'] = USAGE_DIAG_EVERY
    return {
        'schema': CONFIG_SCHEMA, 'task_id': TASK_ID,
        'content': {'feature_array_sha256': au.CONTENT_V1_FEATURE_SHA256, 'dim': rqvae['content_dim']},
        'rqvae': rqvae,
        'variants': {
            'T1': {'sid_version': 'SID_RQVAE_B256D3_S42', 'kind': 'rqvae', 'codebook_size': 256},
            'T2': {'sid_version': 'SID_RQVAE_B16D3_S42', 'kind': 'rqvae', 'codebook_size': 16},
            'T3': {'sid_version': 'SID_BHKM_B256D3_S42', 'kind': 'bhkm', 'branching': 256, 'depth': 3,
                   'max_iter': 10, 'seed': 42},
        },
    }


def _first_difference(left, right, path='config'):
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                return f'{path}.{key}'
            found = _first_difference(left[key], right[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f'{path}[{index}]')
            if found:
                return found
        return None
    return None if left == right else path


def validate_config(value):
    """Strict whitelist: key set, types and values must equal the approved configuration exactly."""
    found = _first_difference(value, expected_config())
    if found:
        raise ValueError(f'unapproved fit config at {found}')
    return value


def _check_file_sha(path, digest):
    """Regular file, no symlink component, sha256 equal to the pin."""
    return gd._checked_file(path, digest)


def load_config(path, digest):
    return validate_config(gd._json(_check_file_sha(path, digest)))


def load_inputs(content_path, content_sha, catalog_path, catalog_sha, *, engineering=False):
    """Pinned content and catalog. Formal mode additionally requires the CONTENT_V1 digest."""
    content_file = _check_file_sha(content_path, content_sha)
    catalog_file = _check_file_sha(catalog_path, catalog_sha)
    array = np.load(content_file, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype('<f4') or array.ndim != 2 \
            or array.shape[1] != rq.PREREG['content_dim'] or len(array) < 3:
        raise ValueError('content must be a float32 [N+1, 64] array')
    if not np.isfinite(array).all() or np.any(array[0] != 0):
        raise ValueError('content must be finite and row 0 must be the all-zero PAD row')
    ids = gd._json(catalog_file)
    if type(ids) is not list or len(ids) != len(array) - 1 or any(type(x) is not int or not 0 <= x < 2 ** 32 for x in ids) \
            or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError('catalog must be a strictly ascending list of uint32 ids, one per content row')
    rows = np.ascontiguousarray(array[1:], dtype=np.float32)
    digest = au.feature_digest(rows)
    if not engineering and digest != au.CONTENT_V1_FEATURE_SHA256:
        raise ValueError('content digest differs from CONTENT_V1')
    ids = np.array(ids, dtype=np.uint32)
    return {'content': rows, 'ids': ids, 'content_sha256': digest, 'catalog_sha256': sid_v2.catalog_digest(ids)}


def choose_device(kind, device, engineering):
    """Argument-level device rules (no hardware query): formal RQ-VAE fits are CUDA-only."""
    if kind == 'bhkm':
        if device not in (None, 'cpu'):
            raise ValueError('the balanced k-means variant runs on CPU only (device option)')
        return 'cpu'
    device = ('cpu' if engineering else 'cuda') if device is None else device
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if not engineering and device != 'cuda':
        raise ValueError('formal RQ-VAE fits require device cuda; CPU is engineering-only')
    return device


def resolve_device(kind, device, engineering):
    device = choose_device(kind, device, engineering)
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but not available')
    return device


def _fit_params(config, overrides):
    r = config['rqvae']
    pick = lambda name: overrides.get(name, r[name])
    return {'seed': r['seed'], 'max_steps': pick('max_steps'), 'batch_size': pick('batch_size'),
            'wall_cap_seconds': pick('wall_cap_seconds'), 'log_every': r['log_every'], 'eval_every': r['usage_diag_every'],
            'check_every': r['check_every'], 'probe_from': pick('probe_from'), 'probe_step': pick('probe_step')}


def _drop_nodes(model):
    """The fitted node list (centres of every node) is large and never needed again: keep the mapping only."""
    return SIDCatalog(model.ids, model.sids, model.branching, model.depth, [])


def bhkm_summary(nodes):
    levels = Counter(len(node['prefix']) for node in nodes)
    sizes = Counter(size for node in nodes for size in node['sizes'])
    return {'nodes_per_level': {str(k): levels[k] for k in sorted(levels)},
            'child_size_histogram': {str(k): sizes[k] for k in sorted(sizes)},
            'unconverged_nodes': int(sum(not node['converged'] for node in nodes))}


def _fit_frozen_kmeans(ids, content, spec):
    """The unmodified frozen sid.fit; returns the nodes-free catalog and a compact summary."""
    model = bhkm.fit(ids, content, branching=spec['branching'], depth=spec['depth'], seed=spec['seed'],
                     max_iter=spec['max_iter'])
    summary = bhkm_summary(model.nodes)
    catalog = _drop_nodes(model)
    return {'catalog': catalog, 'ids': catalog.ids, 'sids': catalog.sids, 'summary': summary}


def usage_diagnostic(K, usage_log, final_step=None, final_used=None):
    """Per-level codebook usage (used codes / K) at step 2000 and at the final step. Diagnostic only: no early stop."""
    logged = {step: used for step, used in usage_log}

    def entry(step, used, source):
        return {'step': int(step), 'used': [int(u) for u in used], 'fraction': [u / K for u in used], 'source': source}

    at_2000 = logged.get(USAGE_DIAG_STEP)
    return {'K': K, 'step_2000': None if at_2000 is None else entry(USAGE_DIAG_STEP, at_2000, 'training_model'),
            'final': None if final_used is None else entry(final_step, final_used, 'canonical_export'),
            'last_logged': None if not usage_log else entry(usage_log[-1][0], usage_log[-1][1], 'training_model')}


def _observations(device, wall_seconds):
    return {'device': device, 'wall_seconds': float(wall_seconds), 'torch_version': torch.__version__,
            'numpy_version': np.__version__, 'python_version': '.'.join(map(str, sys.version_info[:3]))}


def _base(name, variant, engineering, content, observations, params, branching):
    return {'variant': name, 'sid_version': variant['sid_version'], 'kind': variant['kind'], 'engineering': engineering,
            'content_digest': au.feature_digest(content), 'K': branching, 'branching': branching, 'params': params,
            'observations': observations}


def _failed_result(base, raw):
    """Result of a fit that did not complete: log records only, never a model, mapping or weights."""
    detail = {'step': raw['step'], 'elapsed': float(raw['elapsed']), 'loss_log': raw['loss_log'],
              'usage_log': raw['usage_log'], 'usage_diagnostic': usage_diagnostic(base['K'], raw['usage_log'])}
    return {**base, 'ok': False, 'status': raw['status'], 'detail': detail}


def fit_rqvae_variant(config, name, ids, content, *, device, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256,
                      engineering=False, max_steps=None, overrides=None, clock=time.monotonic):
    """Fit and judge T1 or T2. ok is True only when training completed and every gate passed; only then is the
    model (weights, mapping) part of the result."""
    variant = config['variants'][name]
    if variant['kind'] != 'rqvae':
        raise ValueError('not an RQ-VAE variant')
    overrides = dict(overrides or {})
    if max_steps is not None:
        overrides['max_steps'] = max_steps
    K = variant['codebook_size']
    params = _fit_params(config, overrides)
    started = clock()
    raw = rq.fit(content, K=K, device=device, clock=clock, **params)
    base = _base(name, variant, engineering, content, _observations(device, clock() - started), params, K)
    if raw['status'] != 'complete':
        return _failed_result(base, raw)
    state = raw['state']
    codes = rq.export_codes(state, content)
    final_ids, sids = sid_v2.canonical_sids(ids, codes)
    mapping = au.audit_mapping(final_ids, sids, content, branching=K, kind='rqvae')
    info = {'status': raw['status'], 'steps': int(raw['step']), 'max_steps': params['max_steps'],
            'elapsed': float(raw['elapsed']), 'wall_cap_seconds': params['wall_cap_seconds'],
            'required_steps': params['max_steps'], 'reconstruction': rq.reconstruction_report(state, content)}
    gates = au.gates(mapping, 'rqvae', info, expected_content_sha256=expected_content_sha256,
                     required_steps=params['max_steps'])
    diagnostic = usage_diagnostic(K, raw['usage_log'], raw['step'], rq.codebook_usage(codes, K))
    detail = {'step': raw['step'], 'elapsed': float(raw['elapsed']), 'loss_log': raw['loss_log'],
              'usage_log': raw['usage_log'], 'usage_diagnostic': diagnostic}
    audit = {'mapping': mapping, 'rq': info}
    if not gates['passed']:
        return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, 'detail': detail}
    tokenizer = {'variant': name, 'schema': CONFIG_SCHEMA, 'fit': dict(params), 'effective_prereg': _jsonable(rq.PREREG),
                 'engineering_overrides': overrides, 'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum())}
    train_log = {'K': K, 'seed': params['seed'], 'init_digest': raw['init_digest'], 'prefix_digest': raw['prefix_digest'],
                 'loss_log': raw['loss_log'], 'usage_log': raw['usage_log'], 'usage_diagnostic': diagnostic,
                 'observations': base['observations']}
    return {**base, 'ok': True, 'status': 'complete', 'ids': final_ids, 'sids': sids, 'weights': state, 'audit': audit,
            'gates': gates, 'tokenizer': tokenizer, 'train_log': train_log, 'detail': detail}


def fit_bhkm_variant(config, ids, content, *, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256, engineering=False,
                     clock=time.monotonic):
    """Fit and judge T3 with the unmodified frozen sid.fit; the stored catalog has no nodes."""
    spec = config['variants']['T3']
    params = {key: spec[key] for key in ('branching', 'depth', 'max_iter', 'seed')}
    started = clock()
    fitted = _fit_frozen_kmeans(ids, content, spec)
    base = _base('T3', spec, engineering, content, _observations('cpu', clock() - started), params, spec['branching'])
    mapping = au.audit_mapping(fitted['ids'], fitted['sids'], content, branching=spec['branching'], kind='bhkm')
    gates = au.gates(mapping, 'bhkm', None, expected_content_sha256=expected_content_sha256)
    audit = {'mapping': mapping, 'rq': None}
    detail = {'summary': fitted['summary']}
    if not gates['passed']:
        return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, 'detail': detail}
    tokenizer = {'variant': 'T3', 'schema': CONFIG_SCHEMA, 'fit': params, 'summary': fitted['summary'],
                 'engineering_overrides': {}, 'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum()),
                 'observations': base['observations']}
    return {**base, 'ok': True, 'status': 'complete', 'catalog': fitted['catalog'], 'ids': fitted['ids'],
            'sids': fitted['sids'], 'weights': None, 'train_log': None, 'audit': audit, 'gates': gates,
            'tokenizer': tokenizer, 'detail': detail}


def _producer():
    root = Path(__file__).resolve().parent
    return {f'src/vkgr/{name}': file_sha256(root / name) for name in PRODUCER_FILES}


def write_success_bundle(output, result, *, config_sha):
    if result['ok'] is not True:
        raise ValueError('only a variant whose gates passed can be published')
    return sid_v2.write_bundle(
        output, sid_version=result['sid_version'], engineering=result['engineering'], ids=result['ids'],
        sids=result['sids'], content_feature_sha256=result['content_digest'], tokenizer=result['tokenizer'],
        audit=result['audit'], gates=result['gates'], fit_config_sha256=config_sha, producer=_producer(),
        weights=result['weights'], train_log=result['train_log'])


def _sanitize(value):
    """Training logs of a failed fit may hold nan/inf, which canonical JSON refuses."""
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def failure_record(result, *, config_sha, catalog_sha256):
    gates, audit = result.get('gates'), result.get('audit')
    reasons = ['fit:' + result['status']] if gates is None else [c['id'] for c in gates['checks'] if not c['ok']]
    return _sanitize({
        'schema': FAILURE_SCHEMA, 'task_id': TASK_ID, 'status': result['status'], 'reasons': reasons,
        'variant': result['variant'], 'sid_version': result['sid_version'], 'engineering': result['engineering'],
        'gates': gates, 'audit': None if audit is None else audit, 'fit_config_sha256': config_sha,
        'content_feature_sha256': result['content_digest'], 'catalog_sha256': catalog_sha256, 'params': result['params'],
        'observations': result['observations'], 'detail': result['detail']})


def write_failure_record(output, record):
    """failure.json only: a failed tokenizer never leaves a manifest, a mapping or weights behind."""
    output = Path(output).absolute()
    gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory inside an existing one')
    output.mkdir()
    temporary = output / 'failure.json.tmp'
    temporary.write_bytes(canonical_bytes(record))
    os.replace(temporary, output / 'failure.json')
    return file_sha256(output / 'failure.json')


@contextlib.contextmanager
def _prereg_context(effective, engineering):
    """Replay under the constants a bundle recorded. Formal bundles must match the frozen constants exactly."""
    if _jsonable(effective) != _FROZEN_PREREG and not engineering:
        raise ValueError('formal bundle recorded constants other than the pre-registered ones')
    if set(effective) != set(rq.PREREG):
        raise ValueError('recorded constants do not match the module constants')
    saved = dict(rq.PREREG)
    rq.PREREG.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in effective.items()})
    try:
        yield
    finally:
        rq.PREREG.clear()
        rq.PREREG.update(saved)


def _replay_prefix_digest(config, K, content, device, engineering):
    """Re-run the first probe_step training steps under the constants the bundle recorded; returns the digest."""
    params = config['fit']
    with _prereg_context(config['effective_prereg'], engineering):
        out = rq.fit(content, K=K, device=device, seed=params['seed'], max_steps=params['probe_step'],
                     batch_size=params['batch_size'], wall_cap_seconds=NO_WALL_CAP, log_every=params['log_every'],
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
    device = resolve_device('rqvae', device or log['observations']['device'], True)
    if _replay_prefix_digest(config, loaded.info['branching'], content, device, loaded.manifest['engineering']) \
            != log['prefix_digest']:
        raise ValueError('prefix replay does not reproduce the recorded digest')


def _replay_fit(directory, loaded, inputs):
    if loaded.info['kind'] != 'bhkm':
        raise ValueError('--replay-fit applies to balanced k-means bundles only')
    config = gd._json(directory / 'tokenizer.json')['config']
    again = _fit_frozen_kmeans(inputs['ids'], inputs['content'], config['fit'])
    stored = loaded.codec.catalog
    if not (np.array_equal(again['ids'], stored.ids) and np.array_equal(again['sids'], stored.sids)) \
            or again['summary'] != config['summary']:
        raise ValueError('fit replay does not reproduce the stored mapping')


def _expected_digest(inputs, engineering):
    return inputs['content_sha256'] if engineering else au.CONTENT_V1_FEATURE_SHA256


def _label(sid_version):
    try:
        return VARIANT_LABELS[sid_version]
    except KeyError as exc:
        raise ValueError('bundle is not one of the three GTOK-01 tokenizers') from exc


def _check_config_binding(directory, loaded, config, config_sha, engineering):
    """The bundle must have been fitted with exactly this (validated) config file and with nothing else overridden."""
    if loaded.manifest['fit_config_sha256'] != config_sha:
        raise ValueError('bundle was fitted with a different config file')
    tokenizer = gd._json(directory / 'tokenizer.json')['config']
    variant = config['variants'][_label(loaded.info['sid_version'])]
    if variant['kind'] == 'bhkm':
        ok = tokenizer['fit'] == {key: variant[key] for key in ('branching', 'depth', 'max_iter', 'seed')} \
            and tokenizer['engineering_overrides'] == {}
    else:
        overrides = tokenizer['engineering_overrides']
        ok = type(overrides) is dict and set(overrides) <= set(OVERRIDE_OPTIONS) and (engineering or not overrides) \
            and tokenizer['fit'] == _jsonable(_fit_params(config, overrides)) \
            and (engineering or tokenizer['effective_prereg'] == _FROZEN_PREREG)
    if not ok:
        raise ValueError('tokenizer.json does not match the config it was fitted with')


def verify_bundle(manifest_path, manifest_sha, inputs, config, config_sha, *, engineering=False, replay_prefix=False,
                  replay_fit=False, device=None):
    loaded = sid_v2.load_bundle(manifest_path, manifest_sha, catalog_ids=inputs['ids'], content=inputs['content'],
                                expected_content_digest=_expected_digest(inputs, engineering), engineering=engineering)
    directory = Path(manifest_path).absolute().parent
    _check_config_binding(directory, loaded, config, config_sha, engineering)
    checks = ['load_bundle', 'config_binding']
    if replay_prefix:
        _replay_prefix(directory, loaded, inputs['content'], device)
        checks.append('replay_prefix')
    if replay_fit:
        _replay_fit(directory, loaded, inputs)
        checks.append('replay_fit')
    return {'sid_version': loaded.info['sid_version'], 'manifest_sha256': manifest_sha, 'engineering': engineering,
            'checks': checks, 'replay_prefix': replay_prefix, 'replay_fit': replay_fit}


def _load_baseline(path, digest, inputs):
    data = np.load(_check_file_sha(path, digest), allow_pickle=False)
    if not hasattr(data, 'files') or set(data.files) != {'ids', 'sids'}:
        raise ValueError('baseline mapping must hold exactly ids and sids')
    ids, sids = data['ids'], data['sids']
    if ids.dtype != np.uint32 or sids.dtype != np.uint32 or not np.array_equal(ids, inputs['ids']):
        raise ValueError('baseline mapping does not match the catalog')
    sid_v2.build_codec(ids, sids, BASELINE_BRANCHING, catalog_ids=inputs['ids'])
    return au.audit_mapping(ids, sids, inputs['content'], branching=BASELINE_BRANCHING, kind='bhkm')


def compare_bundles(bundles, baseline, inputs, *, engineering=False):
    """bundles: [(manifest_path, sha)], baseline: (mapping.npz path, sha). Labels are T0 plus the bundles' T1..T3."""
    audits = {BASELINE_LABEL: _load_baseline(baseline[0], baseline[1], inputs)}
    shas = {}
    for manifest_path, sha in bundles:
        loaded = sid_v2.load_bundle(manifest_path, sha, catalog_ids=inputs['ids'], content=inputs['content'],
                                    expected_content_digest=_expected_digest(inputs, engineering), engineering=engineering)
        label = _label(loaded.info['sid_version'])
        if label in audits:
            raise ValueError('duplicate tokenizer label in compare')
        audits[label] = gd._json(Path(manifest_path).absolute().parent / 'audit.json')['mapping']
        shas[label] = sha
    return {'schema': COMPARE_SCHEMA, 'engineering': engineering, 'bundle_manifest_sha256': shas,
            'baseline_mapping_sha256': baseline[1], 'table': au.compare({label: audits[label] for label in sorted(audits)})}


def write_comparison(output, result):
    output = Path(output).absolute()
    gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory inside an existing one')
    output.mkdir()
    (output / 'comparison.json').write_bytes(canonical_bytes(result))
    (output / 'comparison.md').write_text(au.render_markdown(result['table']), encoding='utf-8')


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f'argument error: {message}')


def _parser():
    parser = _Parser(prog='python -m vkgr.sid_tok_run', allow_abbrev=False,
                     description='GTOK-01 tokenizer fit / verify / compare')
    sub = parser.add_subparsers(dest='command', required=True)

    def pinned(p, name):
        p.add_argument(f'--{name}', required=True)
        p.add_argument(f'--{name}-sha', required=True)

    fit = sub.add_parser('fit', allow_abbrev=False)
    fit.add_argument('--variant', required=True)
    pinned(fit, 'config')
    pinned(fit, 'content')
    pinned(fit, 'catalog')
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
    pinned(verify, 'config')
    pinned(verify, 'content')
    pinned(verify, 'catalog')
    verify.add_argument('--engineering', action='store_true')
    verify.add_argument('--replay-prefix', action='store_true')
    verify.add_argument('--replay-fit', action='store_true')
    verify.add_argument('--device')
    compare = sub.add_parser('compare', allow_abbrev=False)
    compare.add_argument('--bundle', action='append', required=True)
    compare.add_argument('--sha', action='append', required=True)
    compare.add_argument('--baseline-mapping', required=True)
    compare.add_argument('--baseline-sha', required=True)
    pinned(compare, 'content')
    pinned(compare, 'catalog')
    compare.add_argument('--output', required=True)
    compare.add_argument('--engineering', action='store_true')
    return parser


def _new_output(path):
    output = Path(path).absolute()
    gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory inside an existing one')
    return output


def _say(value):
    print(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def _cmd_fit(args, clock):
    if args.variant not in VARIANTS:
        raise ValueError('variant must be T1, T2 or T3')
    overrides = {name: getattr(args, name) for name in OVERRIDE_OPTIONS if getattr(args, name) is not None}
    kind = expected_config()['variants'][args.variant]['kind']
    if overrides and not args.engineering:
        raise ValueError('--max-steps, --batch-size, --wall-cap-seconds and --probe-* are engineering-only')
    if overrides and kind != 'rqvae':
        raise ValueError('training length options apply to the RQ-VAE variants only')
    if args.engineering and kind == 'rqvae' and 'max_steps' not in overrides:
        raise ValueError('--engineering needs an explicit --max-steps')
    output = _new_output(args.output)
    choose_device(kind, args.device, args.engineering)
    config = load_config(args.config, args.config_sha)
    inputs = load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    device = resolve_device(kind, args.device, args.engineering)
    if not args.engineering and _jsonable(rq.PREREG) != _FROZEN_PREREG:
        raise ValueError('live RQ-VAE constants differ from the pre-registered ones')
    common = {'expected_content_sha256': _expected_digest(inputs, args.engineering), 'engineering': args.engineering,
              'clock': clock}
    if kind == 'rqvae':
        result = fit_rqvae_variant(config, args.variant, inputs['ids'], inputs['content'], device=device,
                                   overrides=overrides, **common)
    else:
        result = fit_bhkm_variant(config, inputs['ids'], inputs['content'], **common)
    if result['ok']:
        manifest = write_success_bundle(output, result, config_sha=args.config_sha)
        _say({'status': 'success', 'variant': args.variant, 'sid_version': manifest['sid_version'],
              'manifest_sha256': file_sha256(output / sid_v2.MANIFEST_NAME), 'engineering': args.engineering})
        return EXIT_OK
    record = failure_record(result, config_sha=args.config_sha, catalog_sha256=inputs['catalog_sha256'])
    write_failure_record(output, record)
    _say({'status': record['status'], 'variant': args.variant, 'reasons': record['reasons']})
    return EXIT_GATE


def _cmd_verify(args):
    config = load_config(args.config, args.config_sha)
    inputs = load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    _say(verify_bundle(args.bundle, args.sha, inputs, config, args.config_sha, engineering=args.engineering,
                       replay_prefix=args.replay_prefix, replay_fit=args.replay_fit, device=args.device))
    return EXIT_OK


def _cmd_compare(args):
    if len(args.bundle) != len(args.sha):
        raise ValueError('every --bundle needs its own --sha')
    output = _new_output(args.output)
    inputs = load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    result = compare_bundles(list(zip(args.bundle, args.sha)), (args.baseline_mapping, args.baseline_sha), inputs,
                             engineering=args.engineering)
    write_comparison(output, result)
    _say({'status': 'compared', 'labels': result['table']['labels']})
    return EXIT_OK


def main(argv=None, *, clock=time.monotonic):
    try:
        args = _parser().parse_args(argv)
        if args.command == 'fit':
            return _cmd_fit(args, clock)
        if args.command == 'verify':
            return _cmd_verify(args)
        return _cmd_compare(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return EXIT_ERROR


if __name__ == '__main__':
    raise SystemExit(main())
