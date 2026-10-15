"""GTOK-02 tokenizer pipeline: fit (six resolution-sweep variants), verify and compare for SIDMAP_V2S bundles.

Variants (all constants come from the frozen fit config ``configs/gtok02/g-tok-02.json``):
  S32 S64 S128 S512 S1024  the frozen balanced k-means ``sid.fit`` at depth 3 with that branching (seed 42, 10 iterations)
  S256D2                   256-ary depth 2, derived from the pinned 256-ary depth-3 T3 mapping: the (c0, c1) partition is kept and
                           items are numbered by ascending item id inside each group; stored as [c0, c1, 0, rank]

Exit codes: 0 success, 1 argument / input / validation error (nothing written), 3 gate failure (the output directory then holds
failure.json and never a run_manifest.json), 4 the audited integers differ from the pre-registered ``expected`` block
(``verify`` only; expected_mismatch.json is written next to nothing else, the bundle is not touched).

Exactly two data inputs are accepted, each pinned by sha256: the CONTENT_V1 item vectors and the catalog id list. No label, query,
split or interaction file is opened. Reuse of GTOK-01 code is by import only; nothing there is edited.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import generative_data as gd
from . import sid_tok_audit as au
from . import sid_tok_run as old_run
from . import sid_v2
from . import sid_v2s as s2
from .data_audit import canonical_bytes, file_sha256

CONFIG_SCHEMA = 'g-tok-02-fit-v1'
FAILURE_SCHEMA = 'g-tok-02-failure-v1'
COMPARE_SCHEMA = 'g-tok-02-compare-v1'
TASK_ID = s2.TASK_ID
EXIT_OK, EXIT_ERROR, EXIT_GATE, EXIT_EXPECTED = 0, 1, 3, 4
VARIANTS = ('S32', 'S64', 'S128', 'S512', 'S1024', 'S256D2')
VARIANT_LABELS = {'SID_BHKM_B32D3_S42': 'S32', 'SID_BHKM_B64D3_S42': 'S64', 'SID_BHKM_B128D3_S42': 'S128',
                  'SID_BHKM_B512D3_S42': 'S512', 'SID_BHKM_B1024D3_S42': 'S1024', 'SID_BHKM_B256D2_S42': 'S256D2'}
BASELINE_LABEL = old_run.BASELINE_LABEL
T3_LABEL = 'T3'
T3_SID_VERSION = 'SID_BHKM_B256D3_S42'
T3_MANIFEST_SHA256 = 'a7af78a3fc04f0f0d174b983987c34d7efa754edd88a05d1350cdadd06fd47b8'
COMPARE_ORDER = (BASELINE_LABEL, T3_LABEL, 'S32', 'S64', 'S128', 'S512', 'S1024', 'S256D2')
PRODUCER_FILES = ('sid_tok2_run.py', 'sid_v2s.py', 'sid_v2.py', 'sid_tok_audit.py', 'sid.py', 'sid_tok_run.py')
EXPECTED_INT_FIELDS = ('leaves_count', 'leaves_max', 'vocab_size', 'items_needing_suffix')


def _bhkm(branching, sid_version, expected):
    return {'sid_version': sid_version, 'kind': 'bhkm', 'branching': branching, 'depth': 3, 'max_iter': 10, 'seed': 42,
            'expected': expected}


def expected_config():
    """The one approved fit configuration; configs/gtok02/g-tok-02.json equals it. ``expected`` holds the integers the real
    120,415-item audit must reproduce exactly (derived from the sid.fit capacity rule; the server audit is what counts)."""
    return {
        'schema': CONFIG_SCHEMA, 'task_id': TASK_ID,
        'content': {'feature_array_sha256': au.CONTENT_V1_FEATURE_SHA256, 'dim': 64},
        'variants': {
            'S32': _bhkm(32, 'SID_BHKM_B32D3_S42', {'items_needing_suffix': 87647, 'leaves_count': 32768, 'leaves_max': 4,
                                                    'positions_distinct': [32, 32, 32, 4], 'vocab_size': 102}),
            'S64': _bhkm(64, 'SID_BHKM_B64D3_S42', {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1,
                                                    'positions_distinct': [64, 64, 30, 1], 'vocab_size': 195}),
            'S128': _bhkm(128, 'SID_BHKM_B128D3_S42', {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1,
                                                       'positions_distinct': [128, 128, 8, 1], 'vocab_size': 387}),
            'S512': _bhkm(512, 'SID_BHKM_B512D3_S42', {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1,
                                                       'positions_distinct': [512, 236, 1, 1], 'vocab_size': 1539}),
            'S1024': _bhkm(1024, 'SID_BHKM_B1024D3_S42', {'items_needing_suffix': 0, 'leaves_count': 120415, 'leaves_max': 1,
                                                          'positions_distinct': [1024, 118, 1, 1], 'vocab_size': 3075}),
            'S256D2': {'sid_version': 'SID_BHKM_B256D2_S42', 'kind': 'derive_depth2', 'branching': 256, 'depth': 2,
                       'parent': {'manifest_sha256': T3_MANIFEST_SHA256, 'sid_version': T3_SID_VERSION, 'variant': 'T3'},
                       'expected': {'items_needing_suffix': 54879, 'leaves_count': 65536, 'leaves_max': 2,
                                    'positions_distinct': [256, 256, 1, 2], 'vocab_size': 772}},
        },
    }


def validate_config(value):
    """Strict whitelist: key set, types and values must equal the approved configuration exactly."""
    found = old_run._first_difference(value, expected_config())
    if found:
        raise ValueError(f'unapproved fit config at {found}')
    return value


def load_config(path, digest):
    return validate_config(gd._json(old_run._check_file_sha(path, digest)))


def gates_sweep(audit_mapping, *, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256):
    """The pre-registered gates H1-H6 and B1, unchanged. For depth 2 the audit given here already holds B1 computed on the
    unpadded [c0, c1, rank] columns (``sid_v2s.sweep_audit``); R1-R4 do not apply to balanced k-means."""
    return au.gates(audit_mapping, 'bhkm', None, expected_content_sha256=expected_content_sha256)


def check_expected(audit_mapping, expected):
    """Exact integer comparison (no tolerance) of the eight pre-registered numbers; returns the list of mismatches."""
    actual = {'leaves_count': audit_mapping['leaves']['count'], 'leaves_max': audit_mapping['leaves']['max'],
              'vocab_size': 2 + 3 * audit_mapping['branching'] + audit_mapping['leaves']['max'],
              'items_needing_suffix': audit_mapping['items_needing_suffix']}
    pairs = [(field, expected[field], actual[field]) for field in EXPECTED_INT_FIELDS]
    distinct = [position['distinct'] for position in audit_mapping['positions']]
    pairs += [(f'positions_distinct[{i}]', expected['positions_distinct'][i], distinct[i]) for i in range(4)]
    return [{'field': field, 'expected': want, 'actual': got} for field, want, got in pairs if want != got]


def _expected_for(config, name, engineering):
    """The ``expected`` block to enforce, or None. Synthetic engineering data cannot reproduce the real integers (G2)."""
    return None if engineering else config['variants'][name]['expected']


def _check_parent_sha(sha, variant, *, engineering):
    if not engineering and sha != variant['parent']['manifest_sha256']:
        raise ValueError('parent bundle is not the pinned T3 bundle')


def fit_sweep_variant(config, name, ids, content, *, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256, engineering=False,
                      clock=time.monotonic, parent=None):
    """Fit (or derive) and judge one variant. ok is True only when every gate passed; only then are ids/sids/tokenizer in the result.

    ``parent`` (S256D2 only): {'ids', 'sids', 'manifest_sha256'} of the loaded depth-3 parent mapping.
    """
    if name not in VARIANTS:
        raise ValueError('variant must be one of ' + ', '.join(VARIANTS))
    variant = config['variants'][name]
    derived = variant['kind'] == 'derive_depth2'
    if derived != (parent is not None):
        raise ValueError('only S256D2 takes (and requires) a parent mapping')
    started = clock()
    parent_sha = None
    if derived:
        _check_parent_sha(parent['manifest_sha256'], variant, engineering=engineering)
        if not np.array_equal(np.asarray(parent['ids']), np.asarray(ids)):
            raise ValueError('parent mapping does not cover this catalog')
        parent_sha = parent['manifest_sha256']
        sids, _ = s2.derive_depth2_from_parent(np.asarray(ids, dtype=np.uint32), parent['sids'])
        fitted = {'ids': np.asarray(ids, dtype=np.uint32), 'sids': sids, 'summary': None, 'catalog': None}
        params = {'branching': variant['branching'], 'depth': 2, 'parent_manifest_sha256': parent_sha}
    else:
        fitted = old_run._fit_frozen_kmeans(ids, content, variant)
        params = {key: variant[key] for key in ('branching', 'depth', 'max_iter', 'seed')}
    base = old_run._base(name, variant, engineering, content, old_run._observations('cpu', clock() - started), params, variant['branching'])
    mapping = s2.sweep_audit(fitted['ids'], fitted['sids'], content, branching=variant['branching'], depth=variant['depth'])
    gates = gates_sweep(mapping, expected_content_sha256=expected_content_sha256)
    audit = {'mapping': mapping, 'sweep': s2.sweep_section(variant['sid_version'], engineering, parent_sha)}
    detail = {} if fitted['summary'] is None else {'summary': fitted['summary']}
    if not gates['passed']:
        return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, 'detail': detail}
    tokenizer = {'variant': name, 'schema': CONFIG_SCHEMA, 'fit': params, 'engineering_overrides': {},
                 'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum()), 'observations': base['observations']}
    if derived:
        tokenizer['parent_manifest_sha256'] = parent_sha
    else:
        tokenizer['summary'] = fitted['summary']
    result = {**base, 'ok': True, 'status': 'complete', 'ids': fitted['ids'], 'sids': fitted['sids'], 'audit': audit,
              'gates': gates, 'tokenizer': tokenizer, 'detail': detail}
    if fitted['catalog'] is not None:
        result['catalog'] = fitted['catalog']
    return result


def _producer():
    root = Path(__file__).resolve().parent
    return {f'src/vkgr/{name}': file_sha256(root / name) for name in PRODUCER_FILES}


def write_success_bundle(output, result, *, config_sha):
    if result['ok'] is not True:
        raise ValueError('only a variant whose gates passed can be published')
    return s2.write_bundle(
        output, sid_version=result['sid_version'], engineering=result['engineering'], ids=result['ids'], sids=result['sids'],
        content_feature_sha256=result['content_digest'], tokenizer=result['tokenizer'], audit=result['audit'], gates=result['gates'],
        fit_config_sha256=config_sha, producer=_producer())


def failure_record_s(result, *, config_sha, catalog_sha256):
    """GTOK-01's failure record with this task's schema and task id (G3)."""
    record = old_run.failure_record(result, config_sha=config_sha, catalog_sha256=catalog_sha256)
    record['schema'], record['task_id'] = FAILURE_SCHEMA, TASK_ID
    return record


def _variant_name(sid_version):
    try:
        return VARIANT_LABELS[sid_version]
    except KeyError as exc:
        raise ValueError('bundle is not one of the six GTOK-02 tokenizers') from exc


def _check_config_binding(directory, loaded, name, config, config_sha, parent_sha):
    """The bundle must have been fitted with exactly this (validated) config file and with nothing else overridden."""
    if loaded.manifest['fit_config_sha256'] != config_sha:
        raise ValueError('bundle was fitted with a different config file')
    variant = config['variants'][name]
    tokenizer = gd._json(directory / 'tokenizer.json')['config']
    if variant['kind'] == 'derive_depth2':
        fit = {'branching': variant['branching'], 'depth': 2, 'parent_manifest_sha256': parent_sha}
    else:
        fit = {key: variant[key] for key in ('branching', 'depth', 'max_iter', 'seed')}
    if tokenizer['fit'] != fit or tokenizer['engineering_overrides'] != {}:
        raise ValueError('tokenizer.json does not match the config it was fitted with')


def _rederive(loaded, parent, variant, inputs, engineering):
    """Bit-exact: load the parent bundle (old loader), derive again, compare with the stored mapping."""
    parent_path, parent_sha = parent
    _check_parent_sha(parent_sha, variant, engineering=engineering)
    again = sid_v2.load_bundle(parent_path, parent_sha, catalog_ids=inputs['ids'], content=inputs['content'],
                               expected_content_digest=old_run._expected_digest(inputs, engineering), engineering=engineering)
    if again.info['sid_version'] != variant['parent']['sid_version']:
        raise ValueError('parent bundle has the wrong sid_version')
    catalog = again.codec.catalog
    sids, _ = s2.derive_depth2_from_parent(catalog.ids, catalog.sids)
    stored = loaded.codec.catalog
    if not (np.array_equal(stored.ids, catalog.ids) and np.array_equal(stored.sids, sids)):
        raise ValueError('depth-2 rederivation does not reproduce the stored mapping')


def verify_bundle(manifest_path, manifest_sha, inputs, config, config_sha, *, engineering=False, replay_fit=False, parent=None,
                  variant=None):
    """Re-verify a published bundle. Returns status 'verified', or 'expected_mismatch' (with the list) when every structural check
    passed but the audited integers differ from the pre-registered ones; structural failures raise ValueError.

    ``parent`` = (manifest path, sha) of the T3 bundle, required for S256D2 and refused otherwise.
    """
    loaded = s2.load_bundle(manifest_path, manifest_sha, catalog_ids=inputs['ids'], content=inputs['content'],
                            expected_content_digest=old_run._expected_digest(inputs, engineering), engineering=engineering)
    name = _variant_name(loaded.info['sid_version'])
    if variant is not None and variant != name:
        raise ValueError(f'bundle is {name}, not {variant}')
    spec = config['variants'][name]
    depth2 = spec['kind'] == 'derive_depth2'
    if depth2 != (parent is not None):
        raise ValueError('the parent bundle is required for S256D2 and refused for every other variant')
    if depth2 and replay_fit:
        raise ValueError('--replay-fit applies to the depth-3 variants; S256D2 is always rederived from its parent')
    directory = Path(manifest_path).absolute().parent
    _check_config_binding(directory, loaded, name, config, config_sha, parent[1] if depth2 else None)
    checks = ['load_bundle', 'config_binding']
    report = {'sid_version': loaded.info['sid_version'], 'manifest_sha256': manifest_sha, 'engineering': engineering, 'checks': checks}
    expected = _expected_for(config, name, engineering)
    if expected is not None:
        mismatches = check_expected(gd._json(directory / 'audit.json')['mapping'], expected)
        if mismatches:
            return {**report, 'status': 'expected_mismatch', 'mismatches': mismatches}
        checks.append('expected')
    if replay_fit:
        old_run._replay_fit(directory, loaded, inputs)
        checks.append('replay_fit')
    if depth2:
        _rederive(loaded, parent, spec, inputs, engineering)
        checks.append('rederive_depth2')
    return {**report, 'status': 'verified', 'replay_fit': replay_fit}


def _load_any(manifest_path, sha, inputs, engineering):
    """(label, audit mapping) of a T3 bundle (SIDMAP_V2) or of one of the six sweep bundles (SIDMAP_V2S)."""
    manifest = gd._json(gd._checked_file(manifest_path, sha))
    version = manifest.get('sid_version') if type(manifest) is dict else None
    kwargs = dict(catalog_ids=inputs['ids'], content=inputs['content'], engineering=engineering,
                  expected_content_digest=old_run._expected_digest(inputs, engineering))
    if version in VARIANT_LABELS:
        label, loaded = VARIANT_LABELS[version], s2.load_bundle(manifest_path, sha, **kwargs)
    elif version == T3_SID_VERSION:
        label, loaded = T3_LABEL, sid_v2.load_bundle(manifest_path, sha, **kwargs)
    else:
        raise ValueError('bundle is neither T3 nor one of the six GTOK-02 tokenizers')
    return label, gd._json(Path(manifest_path).absolute().parent / 'audit.json')['mapping']


def compare_bundles(bundles, baseline, inputs, *, engineering=False):
    """bundles: [(manifest_path, sha)], baseline: (mapping.npz path, sha). Any subset of the points is accepted; a missing point is
    simply absent from the table (never interpolated)."""
    audits, shas = {BASELINE_LABEL: old_run._load_baseline(baseline[0], baseline[1], inputs)}, {}
    for manifest_path, sha in bundles:
        label, audit = _load_any(manifest_path, sha, inputs, engineering)
        if label in audits:
            raise ValueError('duplicate tokenizer label in compare')
        audits[label], shas[label] = audit, sha
    ordered = sorted(audits, key=COMPARE_ORDER.index)
    return {'schema': COMPARE_SCHEMA, 'engineering': engineering, 'bundle_manifest_sha256': shas,
            'baseline_mapping_sha256': baseline[1], 'table': au.compare({label: audits[label] for label in ordered})}


class _Parser(old_run._Parser):
    pass


def _parser():
    parser = _Parser(prog='python -m vkgr.sid_tok2_run', allow_abbrev=False, description='GTOK-02 tokenizer fit / verify / compare')
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
    fit.add_argument('--parent-manifest')
    fit.add_argument('--parent-sha')
    verify = sub.add_parser('verify', allow_abbrev=False)
    verify.add_argument('--bundle', required=True)
    verify.add_argument('--sha', required=True)
    pinned(verify, 'config')
    pinned(verify, 'content')
    pinned(verify, 'catalog')
    verify.add_argument('--output', required=True)
    verify.add_argument('--engineering', action='store_true')
    verify.add_argument('--replay-fit', action='store_true')
    verify.add_argument('--parent-manifest')
    verify.add_argument('--parent-sha')
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


def _say(value):
    """One JSON line with default separators: drivers look for the literal '"status": "verified"' and '"manifest_sha256": "<hex>"'."""
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def _parent_argument(args, name):
    """(manifest path, sha) for S256D2, None for every other variant; the pair is all-or-nothing."""
    given = (args.parent_manifest is not None, args.parent_sha is not None)
    if given == (False, False):
        if name == 'S256D2':
            raise ValueError('S256D2 needs --parent-manifest and --parent-sha')
        return None
    if given != (True, True):
        raise ValueError('--parent-manifest and --parent-sha go together')
    if name != 'S256D2':
        raise ValueError('--parent-manifest and --parent-sha apply to S256D2 only')
    return args.parent_manifest, args.parent_sha


def _load_parent(parent, variant, inputs, engineering):
    _check_parent_sha(parent[1], variant, engineering=engineering)
    loaded = sid_v2.load_bundle(parent[0], parent[1], catalog_ids=inputs['ids'], content=inputs['content'],
                                expected_content_digest=old_run._expected_digest(inputs, engineering), engineering=engineering)
    if loaded.info['sid_version'] != variant['parent']['sid_version']:
        raise ValueError('parent bundle has the wrong sid_version')
    return {'ids': loaded.codec.catalog.ids, 'sids': loaded.codec.catalog.sids, 'manifest_sha256': parent[1]}


def _cmd_fit(args, clock):
    if args.variant not in VARIANTS:
        raise ValueError('variant must be one of ' + ', '.join(VARIANTS))
    parent = _parent_argument(args, args.variant)
    output = old_run._new_output(args.output)
    old_run.choose_device('bhkm', args.device, args.engineering)
    config = load_config(args.config, args.config_sha)
    inputs = old_run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    variant = config['variants'][args.variant]
    loaded_parent = None if parent is None else _load_parent(parent, variant, inputs, args.engineering)
    result = fit_sweep_variant(config, args.variant, inputs['ids'], inputs['content'], engineering=args.engineering, clock=clock,
                               expected_content_sha256=old_run._expected_digest(inputs, args.engineering), parent=loaded_parent)
    if result['ok']:
        manifest = write_success_bundle(output, result, config_sha=args.config_sha)
        _say({'status': 'success', 'variant': args.variant, 'sid_version': manifest['sid_version'],
              'manifest_sha256': file_sha256(output / s2.MANIFEST_NAME), 'engineering': args.engineering})
        return EXIT_OK
    record = failure_record_s(result, config_sha=args.config_sha, catalog_sha256=inputs['catalog_sha256'])
    old_run.write_failure_record(output, record)
    _say({'status': record['status'], 'variant': args.variant, 'reasons': record['reasons']})
    return EXIT_GATE


def _write_json(output, name, value):
    output.mkdir()
    (output / name).write_bytes(canonical_bytes(value))


def _cmd_verify(args):
    output = old_run._new_output(args.output)
    config = load_config(args.config, args.config_sha)
    inputs = old_run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    name = _variant_name(gd._json(gd._checked_file(args.bundle, args.sha)).get('sid_version'))
    parent = _parent_argument(args, name)
    report = verify_bundle(args.bundle, args.sha, inputs, config, args.config_sha, engineering=args.engineering,
                           replay_fit=args.replay_fit, parent=parent)
    if report['status'] == 'verified':
        _write_json(output, 'verify.json', report)
        _say(report)
        return EXIT_OK
    _write_json(output, 'expected_mismatch.json', report)
    _say({'status': report['status'], 'sid_version': report['sid_version'], 'mismatches': report['mismatches']})
    return EXIT_EXPECTED


def _cmd_compare(args):
    if len(args.bundle) != len(args.sha):
        raise ValueError('every --bundle needs its own --sha')
    output = old_run._new_output(args.output)
    inputs = old_run.load_inputs(args.content, args.content_sha, args.catalog, args.catalog_sha, engineering=args.engineering)
    result = compare_bundles(list(zip(args.bundle, args.sha)), (args.baseline_mapping, args.baseline_sha), inputs, engineering=args.engineering)
    old_run.write_comparison(output, result)
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
