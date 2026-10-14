"""GTOK-01R tokenizer pipeline: probe, select, fit and verify for the repaired RQ-VAE recipe.

New CLI next to sid_tok_run.py (which stays untouched and is reused by import). Exactly two data inputs are
accepted, each pinned by sha256: the CONTENT_V1 item vectors (.npy, row 0 is the all-zero PAD row) and the catalog id
list (.json). No label, query, split or interaction file is opened.

  probe   one 30,000-step run of one arm (p0, l1..l4) on one variant (T1: K=256, T2: K=16) -> probe-result.json
  select  ten probe-results -> decision.json by the pre-registered rule (sid_rqvae_r.select_arm); optional release
          of one ESCALATE-listed arm with --override-arm / --override-source
  fit     the formal 300,000-step fit of the arm chosen in decision.json -> SIDMAP_V2 bundle (same file whitelist as
          GTOK-01; recipe / arm / decision_sha256 / arm_hparams live in tokenizer.json['config'], diagnostics and
          reset_log in train-log.json)
  verify  sid_v2.load_bundle plus the bindings to config, decision, arm and arm_hparams (optionally a prefix replay)

Exit codes: 0 success, 1 argument / input / validation error (nothing written; INVALID_CONTROL writes the decision
and exits 1), 3 gate failure of a fit (failure.json, never a manifest) or a STOP / ESCALATE decision.
"""
import argparse
import contextlib
import copy
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import generative_data as gd
from . import sid_rqvae as rq
from . import sid_rqvae_r as rr
from . import sid_tok_audit as au
from . import sid_tok_run as run
from . import sid_v2
from .data_audit import canonical_bytes, file_sha256

CONFIG_SCHEMA_R = 'g-tok-01r-fit-v1'
FAILURE_SCHEMA_R = 'g-tok-01r-failure-v1'
RECIPE = 'gtok01r'
TASK_ID = sid_v2.TASK_ID
EXIT_OK, EXIT_ERROR, EXIT_GATE = run.EXIT_OK, run.EXIT_ERROR, run.EXIT_GATE
TRAIN_LOG_MAX_BYTES = 1_000_000
VARIANTS_R = ('T1', 'T2')
OVERRIDE_OPTIONS_R = run.OVERRIDE_OPTIONS + ('log_every', 'eval_every')
PRODUCER_FILES_R = ('sid_tok_run_r.py', 'sid_rqvae_r.py', 'sid_tok_run.py', 'sid_rqvae.py', 'sid_v2.py',
                    'sid_tok_audit.py', 'sid.py')
TOKENIZER_CONFIG_KEYS = frozenset({'variant', 'schema', 'fit', 'effective_prereg', 'engineering_overrides',
                                   'zero_norm_rows', 'recipe', 'arm', 'decision_sha256', 'arm_hparams'})
_HEX40 = re.compile(r'[0-9a-f]{40}')
_jsonable = run._jsonable
# Snapshot at import: later patches of the live dicts cannot move the frozen configuration.
_FROZEN_R = {'prereg_r': _jsonable(rr.PREREG_R), 'select_r': _jsonable(rr.PREREG_R_SELECT)}


def expected_config_r():
    """The one approved configuration; configs/gtok01r/g-tok-01r.json equals it."""
    rqvae = copy.deepcopy(run._FROZEN_PREREG)
    rqvae['usage_diag_every'] = run.USAGE_DIAG_EVERY
    return {
        'schema': CONFIG_SCHEMA_R, 'task_id': TASK_ID, 'recipe': RECIPE,
        'content': {'feature_array_sha256': au.CONTENT_V1_FEATURE_SHA256, 'dim': rqvae['content_dim']},
        'rqvae': rqvae,
        'variants': {
            'T1': {'sid_version': 'SID_RQVAE_B256D3_S42', 'kind': 'rqvae', 'codebook_size': 256},
            'T2': {'sid_version': 'SID_RQVAE_B16D3_S42', 'kind': 'rqvae', 'codebook_size': 16},
        },
        'prereg_r': copy.deepcopy(_FROZEN_R['prereg_r']),
        'select_r': copy.deepcopy(_FROZEN_R['select_r']),
    }


def validate_config_r(value):
    """Strict whitelist: key set, types and values must equal the approved configuration exactly."""
    found = run._first_difference(value, expected_config_r())
    if found:
        raise ValueError(f'unapproved fit config at {found}')
    return value


def require_frozen_constants():
    """Formal commands run only while every live constant equals its pre-registered value."""
    if _jsonable(rr.PREREG_R) != _FROZEN_R['prereg_r'] or _jsonable(rr.PREREG_R_SELECT) != _FROZEN_R['select_r'] \
            or _jsonable(rq.PREREG) != run._FROZEN_PREREG:
        raise ValueError('live constants differ from the pre-registered ones')


def _check_file_sha(path, digest):
    return run._check_file_sha(path, digest)


def _new_output(path):
    return run._new_output(path)


def load_config_r(path, digest):
    return validate_config_r(gd._json(_check_file_sha(path, digest)))


def _write_new(output, name, value):
    """New directory holding one canonical JSON file; the bytes are produced before anything is created."""
    data = canonical_bytes(value)
    output = Path(output).absolute()
    gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory inside an existing one')
    output.mkdir()
    temporary = output / (name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, output / name)
    return file_sha256(output / name)


def fit_params(config, overrides):
    params = run._fit_params(config, overrides)
    params.update({name: overrides[name] for name in ('log_every', 'eval_every') if name in overrides})
    return params


def probe_params(config, overrides):
    """Probe length and wall cap come from the pre-registered probe spec, not from the 300,000-step fit spec."""
    params = fit_params(config, {k: v for k, v in overrides.items() if k not in ('max_steps', 'wall_cap_seconds')})
    params['max_steps'] = overrides.get('max_steps', config['prereg_r']['probe_steps'])
    params['wall_cap_seconds'] = overrides.get('wall_cap_seconds', config['prereg_r']['probe_wall_cap_seconds'])
    return params


# ------------------------------------------------------------------------------------------------------ probe
def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, float):
        return value == value and abs(value) != float('inf')
    return True


def _finite_prefix(log):
    """A diverged run may log nan; the record keeps the entries before the first non-finite one."""
    out = []
    for entry in log:
        if not _finite(entry):
            break
        out.append([entry[0], list(entry[1]) if isinstance(entry[1], (list, tuple)) else entry[1]])
    return out


def probe_record(raw, *, arm, variant, K, config_sha, code_commit, content_file_sha, inputs):
    final = None
    if raw['status'] == 'complete':
        codes = rq.export_codes(raw['state'], inputs['content'])
        report = rq.reconstruction_report(raw['state'], inputs['content'])
        final = {'canonical_usage': rq.codebook_usage(codes, K), 'r2': report['r2'],
                 'residual_norms': report['residual_norms']}
    return {
        'schema': rr.PROBE_SCHEMA, 'variant': variant, 'arm': arm, 'K': K, 'config_sha256': config_sha,
        'code_commit': code_commit, 'content_file_sha256': content_file_sha,
        'content_array_sha256': inputs['content_sha256'], 'catalog_sha256': inputs['catalog_sha256'],
        'status': raw['status'], 'steps': raw['step'], 'elapsed': float(raw['elapsed']),
        'ms_per_step': float(raw['ms_per_step']), 'loss_log': _finite_prefix(raw['loss_log']),
        'recon_log': _finite_prefix(raw['recon_log']), 'usage_log': _finite_prefix(raw['usage_log']),
        'diagnostics': [r for r in raw['diagnostics'] if _finite(r)], 'final': final}


def run_probe(config, arm, variant, inputs, *, config_sha, content_file_sha, code_commit, device, overrides,
              clock=time.monotonic):
    K = config['variants'][variant]['codebook_size']
    params = probe_params(config, overrides)
    raw = rr.fit_r(inputs['content'], K=K, arm=arm, device=device, clock=clock, project=False, diagnostics=True,
                   **params)
    record = probe_record(raw, arm=arm, variant=variant, K=K, config_sha=config_sha, code_commit=code_commit,
                          content_file_sha=content_file_sha, inputs=inputs)
    rr.validate_probe(record)
    return record


# ----------------------------------------------------------------------------------------------- decision I/O
def load_decision(path, digest, *, config_sha):
    """decision.json written by `select`: pinned by sha256, strict schema, produced under this very config."""
    decision = gd._json(_check_file_sha(path, digest))
    rr.validate_decision(decision)
    if decision['config_sha256'] != config_sha:
        raise ValueError('decision was produced under a different config')
    return decision


def require_arm(decision):
    """fit and verify accept l1..l4 only; STOP, ESCALATE and INVALID_CONTROL decisions never reach a fit."""
    if decision['decision'] not in rr.PREREG_R['arm_order']:
        raise ValueError('decision is not an l1..l4 arm')
    return decision['decision']


def check_decision_inputs(decision, inputs, content_sha):
    if decision['content_file_sha256'] != content_sha or decision['content_array_sha256'] != inputs['content_sha256'] \
            or decision['catalog_sha256'] != inputs['catalog_sha256']:
        raise ValueError('decision does not match the inputs')


def check_variant(decision, variant, K):
    if rr.PREREG_R_SELECT['variant_K'][variant] != K:
        raise ValueError('variant K differs from the pre-registered K')
    if decision['per_arm_per_K'][decision['decision']][variant]['status'] != 'complete':
        raise ValueError('decision does not cover this variant')


# -------------------------------------------------------------------------------------------------------- fit
def fit_variant_r(config, name, decision, decision_sha, ids, content, *, device,
                  expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256, engineering=False, overrides=None,
                  clock=time.monotonic):
    """Fit and judge T1 or T2 with the decision's arm. ok is True only when training completed and every gate
    passed; only then is the model (weights, mapping) part of the result."""
    variant = config['variants'][name]
    arm = decision['decision']
    K = variant['codebook_size']
    overrides = dict(overrides or {})
    params = fit_params(config, overrides)
    started = clock()
    raw = rr.fit_r(content, K=K, arm=arm, device=device, clock=clock, project=True, diagnostics=True, **params)
    base = run._base(name, variant, engineering, content, run._observations(device, clock() - started), params, K)
    extra = {'arm': arm, 'recon_log': raw['recon_log'], 'reset_log': raw['reset_log'],
             'diagnostics': raw['diagnostics'], 'ms_per_step': raw['ms_per_step']}
    if raw['status'] != 'complete':
        result = run._failed_result(base, raw)
        result['detail'].update(extra)
        return result
    state = raw['state']
    codes = rq.export_codes(state, content)
    final_ids, sids = sid_v2.canonical_sids(ids, codes)
    mapping = au.audit_mapping(final_ids, sids, content, branching=K, kind='rqvae')
    info = {'status': raw['status'], 'steps': int(raw['step']), 'max_steps': params['max_steps'],
            'elapsed': float(raw['elapsed']), 'wall_cap_seconds': params['wall_cap_seconds'],
            'required_steps': params['max_steps'], 'reconstruction': rq.reconstruction_report(state, content)}
    gates = au.gates(mapping, 'rqvae', info, expected_content_sha256=expected_content_sha256,
                     required_steps=params['max_steps'])
    diagnostic = run.usage_diagnostic(K, raw['usage_log'], raw['step'], rq.codebook_usage(codes, K))
    detail = {'step': raw['step'], 'elapsed': float(raw['elapsed']), 'loss_log': raw['loss_log'],
              'usage_log': raw['usage_log'], 'usage_diagnostic': diagnostic, **extra}
    audit = {'mapping': mapping, 'rq': info}
    if not gates['passed']:
        return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, 'detail': detail}
    tokenizer = {'variant': name, 'schema': CONFIG_SCHEMA_R, 'fit': dict(params),
                 'effective_prereg': _jsonable(rq.PREREG), 'engineering_overrides': overrides,
                 'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum()), 'recipe': RECIPE, 'arm': arm,
                 'decision_sha256': decision_sha, 'arm_hparams': _jsonable(rr.PREREG_R['arms'][arm])}
    train_log = {'K': K, 'seed': params['seed'], 'init_digest': raw['init_digest'], 'prefix_digest': raw['prefix_digest'],
                 'loss_log': raw['loss_log'], 'usage_log': raw['usage_log'], 'usage_diagnostic': diagnostic,
                 'observations': base['observations'], **extra}
    return {**base, 'ok': True, 'status': 'complete', 'ids': final_ids, 'sids': sids, 'weights': state, 'audit': audit,
            'gates': gates, 'tokenizer': tokenizer, 'train_log': train_log, 'detail': detail}


def assert_train_log_size(train_log):
    """diagnostics and reset_log live inside train-log.json (the bundle file set is a strict whitelist): < 1 MB."""
    size = len(canonical_bytes(train_log))
    if size >= TRAIN_LOG_MAX_BYTES:
        raise ValueError(f'train-log.json exceeds the {TRAIN_LOG_MAX_BYTES}-byte limit ({size} bytes)')


def _producer_r():
    root = Path(__file__).resolve().parent
    return {f'src/vkgr/{name}': file_sha256(root / name) for name in PRODUCER_FILES_R}


def write_success_bundle_r(output, result, *, config_sha):
    if result['ok'] is not True:
        raise ValueError('only a variant whose gates passed can be published')
    assert_train_log_size(result['train_log'])
    return sid_v2.write_bundle(
        output, sid_version=result['sid_version'], engineering=result['engineering'], ids=result['ids'],
        sids=result['sids'], content_feature_sha256=result['content_digest'], tokenizer=result['tokenizer'],
        audit=result['audit'], gates=result['gates'], fit_config_sha256=config_sha, producer=_producer_r(),
        weights=result['weights'], train_log=result['train_log'])


def failure_record_r(result, *, config_sha, catalog_sha256, decision_sha):
    record = run.failure_record(result, config_sha=config_sha, catalog_sha256=catalog_sha256)
    record.update({'schema': FAILURE_SCHEMA_R, 'recipe': RECIPE, 'arm': result['detail']['arm'],
                   'decision_sha256': decision_sha})
    return record


# ----------------------------------------------------------------------------------------------------- verify
def _expected_arm_hparams(arm, engineering):
    return _jsonable(rr.PREREG_R['arms'][arm]) if engineering else _FROZEN_R['prereg_r']['arms'][arm]


def check_recipe(tok):
    if tok.get('recipe') != RECIPE:
        raise ValueError('bundle recipe is not gtok01r (recipe missing or different)')


def check_arm(tok, decision):
    arm = decision['decision']
    if tok.get('arm') != arm or (decision['override'] is not None and decision['override']['arm'] != tok.get('arm')):
        raise ValueError('bundle arm differs from the decision')


def check_arm_hparams(tok, decision, engineering):
    if tok.get('arm_hparams') != _expected_arm_hparams(decision['decision'], engineering):
        raise ValueError('bundle arm_hparams differ from the arm table of the decision')


def check_decision_sha(tok, decision_sha):
    if tok.get('decision_sha256') != decision_sha:
        raise ValueError('decision_sha256 recorded in the bundle differs from --decision-sha')


def check_fit_binding(tok, config, loaded, engineering):
    overrides = tok.get('engineering_overrides')
    ok = set(tok) == TOKENIZER_CONFIG_KEYS and tok['variant'] == run._label(loaded.info['sid_version']) \
        and tok['schema'] == CONFIG_SCHEMA_R and type(overrides) is dict \
        and set(overrides) <= set(OVERRIDE_OPTIONS_R) and (engineering or not overrides) \
        and tok['fit'] == _jsonable(fit_params(config, overrides)) and tok['effective_prereg'] == _jsonable(rq.PREREG) \
        and type(tok['zero_norm_rows']) is int
    if not ok:
        raise ValueError('tokenizer.json does not match the config it was fitted with')


def check_binding(directory, loaded, config, config_sha, decision, decision_sha, engineering):
    """The bundle must have been fitted with this config and this decision's arm, not just claim so."""
    if loaded.manifest['fit_config_sha256'] != config_sha:
        raise ValueError('bundle was fitted with a different config file')
    if loaded.info['kind'] != 'rqvae' or run._label(loaded.info['sid_version']) not in VARIANTS_R:
        raise ValueError('bundle is not a T1 or T2 RQ-VAE tokenizer')
    tok = gd._json(directory / 'tokenizer.json')['config']
    check_recipe(tok)
    check_arm(tok, decision)
    check_arm_hparams(tok, decision, engineering)
    check_decision_sha(tok, decision_sha)
    check_fit_binding(tok, config, loaded, engineering)


@contextlib.contextmanager
def _arm_context(arm, hparams):
    """Replay under the arm table the bundle recorded."""
    saved = rr.PREREG_R['arms'][arm]
    rr.PREREG_R['arms'][arm] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in hparams.items()}
    try:
        yield
    finally:
        rr.PREREG_R['arms'][arm] = saved


def _replay_prefix_digest_r(tok, K, content, device, engineering):
    """Re-run the first probe_step steps with the recorded arm_hparams and constants; returns the digest."""
    params, arm = tok['fit'], tok['arm']
    with run._prereg_context(tok['effective_prereg'], engineering), _arm_context(arm, tok['arm_hparams']):
        out = rr.fit_r(content, K=K, arm=arm, device=device, seed=params['seed'], max_steps=params['probe_step'],
                       batch_size=params['batch_size'], wall_cap_seconds=run.NO_WALL_CAP, log_every=params['log_every'],
                       eval_every=params['eval_every'], check_every=params['check_every'],
                       probe_from=params['probe_from'], probe_step=params['probe_step'], diagnostics=False)
    return out['prefix_digest'] if out['status'] == 'complete' else None


def _replay_prefix_r(directory, loaded, content, device):
    tok = gd._json(directory / 'tokenizer.json')['config']
    log = gd._json(directory / 'train-log.json')
    if log['prefix_digest'] is None:
        raise ValueError('the bundle recorded no prefix digest (fewer steps than the probe step)')
    device = run.resolve_device('rqvae', device or log['observations']['device'], True)
    if _replay_prefix_digest_r(tok, loaded.info['branching'], content, device, loaded.manifest['engineering']) \
            != log['prefix_digest']:
        raise ValueError('prefix replay does not reproduce the recorded digest')


def verify_bundle_r(manifest_path, manifest_sha, inputs, config, config_sha, decision, decision_sha, content_sha, *,
                    engineering=False, replay_prefix=False, device=None):
    check_decision_inputs(decision, inputs, content_sha)
    loaded = sid_v2.load_bundle(manifest_path, manifest_sha, catalog_ids=inputs['ids'], content=inputs['content'],
                                expected_content_digest=run._expected_digest(inputs, engineering),
                                engineering=engineering)
    directory = Path(manifest_path).absolute().parent
    check_binding(directory, loaded, config, config_sha, decision, decision_sha, engineering)
    checks = ['load_bundle', 'decision_inputs', 'config_binding', 'arm', 'arm_hparams']
    if replay_prefix:
        _replay_prefix_r(directory, loaded, inputs['content'], device)
        checks.append('replay_prefix')
    return {'sid_version': loaded.info['sid_version'], 'manifest_sha256': manifest_sha, 'engineering': engineering,
            'arm': decision['decision'], 'decision_sha256': decision_sha, 'checks': checks,
            'replay_prefix': replay_prefix}


# ------------------------------------------------------------------------------------------------------- CLI
def _parser():
    parser = run._Parser(prog='python -m vkgr.sid_tok_run_r', allow_abbrev=False,
                         description='GTOK-01R probe / select / fit / verify')
    sub = parser.add_subparsers(dest='command', required=True)

    def pinned(p, name):
        p.add_argument(f'--{name}', required=True)
        p.add_argument(f'--{name}-sha', required=True)

    def overrides(p):
        for name in OVERRIDE_OPTIONS_R:
            p.add_argument('--' + name.replace('_', '-'), type=float if name == 'wall_cap_seconds' else int)

    probe = sub.add_parser('probe', allow_abbrev=False)
    probe.add_argument('--arm', required=True, choices=tuple(rr.PREREG_R['arms']))
    probe.add_argument('--variant', required=True, choices=VARIANTS_R)
    probe.add_argument('--code-commit', required=True)
    pinned(probe, 'config')
    pinned(probe, 'content')
    pinned(probe, 'catalog')
    probe.add_argument('--output', required=True)
    probe.add_argument('--device')
    probe.add_argument('--engineering', action='store_true')
    overrides(probe)
    select = sub.add_parser('select', allow_abbrev=False)
    select.add_argument('--probe', action='append', required=True)
    select.add_argument('--probe-sha', action='append', required=True)
    pinned(select, 'config')
    select.add_argument('--output', required=True)
    select.add_argument('--override-arm', choices=tuple(rr.PREREG_R['arm_order']))
    select.add_argument('--override-source')
    select.add_argument('--engineering', action='store_true')
    fit = sub.add_parser('fit', allow_abbrev=False)
    fit.add_argument('--variant', required=True)
    fit.add_argument('--decision', required=True)
    fit.add_argument('--decision-sha', required=True)
    pinned(fit, 'config')
    pinned(fit, 'content')
    pinned(fit, 'catalog')
    fit.add_argument('--output', required=True)
    fit.add_argument('--device')
    fit.add_argument('--engineering', action='store_true')
    overrides(fit)
    verify = sub.add_parser('verify', allow_abbrev=False)
    verify.add_argument('--bundle', required=True)
    verify.add_argument('--sha', required=True)
    verify.add_argument('--decision', required=True)
    verify.add_argument('--decision-sha', required=True)
    pinned(verify, 'config')
    pinned(verify, 'content')
    pinned(verify, 'catalog')
    verify.add_argument('--engineering', action='store_true')
    verify.add_argument('--replay-prefix', action='store_true')
    verify.add_argument('--device')
    return parser


def _overrides(args):
    given = {name: getattr(args, name) for name in OVERRIDE_OPTIONS_R if getattr(args, name) is not None}
    if given and not args.engineering:
        raise ValueError('--max-steps, --batch-size, --wall-cap-seconds, --probe-*, --log-every and --eval-every '
                         'are engineering-only')
    if args.engineering and 'max_steps' not in given:
        raise ValueError('--engineering needs an explicit --max-steps')
    return given


def _cmd_probe(args, clock):
    if not _HEX40.fullmatch(args.code_commit):
        raise ValueError('--code-commit must be a lower-case 40-hex commit id')
    overrides = _overrides(args)
    output = _new_output(args.output)
    run.choose_device('rqvae', args.device, args.engineering)
    config = load_config_r(args.config, args.config_sha)
    inputs = run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha,
                             engineering=args.engineering)
    device = run.resolve_device('rqvae', args.device, args.engineering)
    if not args.engineering:
        require_frozen_constants()
    record = run_probe(config, args.arm, args.variant, inputs, config_sha=args.config_sha,
                       content_file_sha=args.content_sha, code_commit=args.code_commit, device=device,
                       overrides=overrides, clock=clock)
    sha = _write_new(output, 'probe-result.json', record)
    run._say({'status': 'probed', 'arm': args.arm, 'variant': args.variant, 'probe_status': record['status'],
              'probe_sha256': sha})
    return EXIT_OK


def _cmd_select(args):
    if len(args.probe) != len(args.probe_sha):
        raise ValueError('every --probe needs its own --probe-sha')
    if (args.override_arm is None) != (args.override_source is None):
        raise ValueError('--override-arm and --override-source must be given together')
    output = _new_output(args.output)
    if not args.engineering:
        require_frozen_constants()
    load_config_r(args.config, args.config_sha)
    probes = [gd._json(_check_file_sha(path, sha)) for path, sha in zip(args.probe, args.probe_sha)]
    decision = rr.build_decision(probes, list(args.probe_sha), config_sha256=args.config_sha,
                                 override_arm=args.override_arm, override_source=args.override_source)
    sha = _write_new(output, 'decision.json', decision)
    chosen = decision['decision']
    pre = decision['preflight']
    run._say({'status': 'selected', 'decision': chosen, 'decision_sha256': sha,
              'escalate_listed_arms': decision['escalate_listed_arms'],
              'preflight_exceeds': None if pre is None else pre['exceeds']})
    if chosen in rr.PREREG_R['arm_order']:
        return EXIT_OK
    return EXIT_ERROR if chosen == 'INVALID_CONTROL' else EXIT_GATE


def _cmd_fit(args, clock):
    if args.variant not in VARIANTS_R:
        raise ValueError('variant must be T1 or T2')
    overrides = _overrides(args)
    output = _new_output(args.output)
    run.choose_device('rqvae', args.device, args.engineering)
    config = load_config_r(args.config, args.config_sha)
    decision = load_decision(args.decision, args.decision_sha, config_sha=args.config_sha)
    require_arm(decision)
    inputs = run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha,
                             engineering=args.engineering)
    check_decision_inputs(decision, inputs, args.content_sha)
    check_variant(decision, args.variant, config['variants'][args.variant]['codebook_size'])
    device = run.resolve_device('rqvae', args.device, args.engineering)
    if not args.engineering:
        require_frozen_constants()
    result = fit_variant_r(config, args.variant, decision, args.decision_sha, inputs['ids'], inputs['content'],
                           device=device, expected_content_sha256=run._expected_digest(inputs, args.engineering),
                           engineering=args.engineering, overrides=overrides, clock=clock)
    if result['ok']:
        manifest = write_success_bundle_r(output, result, config_sha=args.config_sha)
        run._say({'status': 'success', 'variant': args.variant, 'sid_version': manifest['sid_version'],
                  'manifest_sha256': file_sha256(output / sid_v2.MANIFEST_NAME), 'engineering': args.engineering,
                  'arm': decision['decision']})
        return EXIT_OK
    record = failure_record_r(result, config_sha=args.config_sha, catalog_sha256=inputs['catalog_sha256'],
                              decision_sha=args.decision_sha)
    run.write_failure_record(output, record)
    run._say({'status': record['status'], 'variant': args.variant, 'reasons': record['reasons']})
    return EXIT_GATE


def _cmd_verify(args):
    config = load_config_r(args.config, args.config_sha)
    if not args.engineering:
        require_frozen_constants()
    inputs = run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha,
                             engineering=args.engineering)
    decision = load_decision(args.decision, args.decision_sha, config_sha=args.config_sha)
    require_arm(decision)
    run._say(verify_bundle_r(args.bundle, args.sha, inputs, config, args.config_sha, decision, args.decision_sha,
                             args.content_sha, engineering=args.engineering, replay_prefix=args.replay_prefix,
                             device=args.device))
    return EXIT_OK


def main(argv=None, *, clock=time.monotonic):
    try:
        args = _parser().parse_args(argv)
        if args.command == 'probe':
            return _cmd_probe(args, clock)
        if args.command == 'select':
            return _cmd_select(args)
        if args.command == 'fit':
            return _cmd_fit(args, clock)
        return _cmd_verify(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return EXIT_ERROR


if __name__ == '__main__':
    raise SystemExit(main())
