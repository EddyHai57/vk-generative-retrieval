"""G1 read-only inputs. Historical producer identity is independent of this consumer."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from .data_audit import canonical_bytes, file_sha256
from .retrieval_data import FROZEN as P3D_SHA

G0_SHA = 'fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59'
G0_FILE = 'artifacts/g0-foundation-manifest.json'
P3D_FILE = 'artifacts/p3d-retrieval-contract-manifest.json'


def _same(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _path(root, name):
    if type(name) is not str or not name or '\\' in name:
        raise ValueError('invalid relative path')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '.' in path.parts or path.as_posix() != name:
        raise ValueError('invalid relative path')
    result = Path(root).absolute() / name
    _no_symlink(result)
    return result


def _no_symlink(path):
    path = Path(path).absolute()
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError('symlink or symlink parent forbidden')
    return path


def _checked_file(path, digest):
    path = _no_symlink(path)
    if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid SHA256 pin')
    if not stat.S_ISREG(path.stat().st_mode) or file_sha256(path) != digest:
        raise ValueError('input hash or file type mismatch')
    return path


def _json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate JSON field')
            value[key] = item
        return value
    def constant(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def _legacy_members(root):
    g0 = _json(_checked_file(_path(root, G0_FILE), G0_SHA))
    p3d = _json(_checked_file(_path(root, P3D_FILE), P3D_SHA))
    expected = {}
    def add(name, digest, source):
        _path(root, name)
        if name in expected and expected[name]['sha256'] != digest:
            raise ValueError('historical anchor conflict')
        expected.setdefault(name, {'sha256': digest, 'sources': []})['sources'].append(source)
    for name, digest in g0['consumer_source']['files'].items():
        add(name, digest, 'g0.consumer_source')
    for record in p3d['bound_files']:
        add(record['path'], record['sha256'], 'p3d.bound_files')
    add('tests/fixtures/g0_synthetic.json', g0['input_hashes']['fixture'], 'g0.input_hashes.fixture')
    add(G0_FILE, G0_SHA, 'g0.external_pin')
    if len(expected) != 37:
        raise ValueError('historical member count mismatch')
    return expected


def validate_legacy_capsule(root: Path, inventory_sha256: str) -> dict:
    """Inventory rehash cannot replace anchored historical bytes or members."""
    root = _no_symlink(root)
    value = _json(_checked_file(_path(root, 'inventory.json'), inventory_sha256))
    anchors = {'g0_manifest_sha256': G0_SHA, 'p3d_manifest_sha256': P3D_SHA}
    if type(value) is not dict or set(value) != {'schema_version', 'kind', 'anchors', 'members'} or not _same(value['schema_version'], 1) or value['kind'] != 'immutable_g0_public_capsule' or not _same(value['anchors'], anchors) or type(value['members']) is not list:
        raise ValueError('invalid capsule schema or historical anchors')
    expected = _legacy_members(root)
    seen = set()
    for record in value['members']:
        if type(record) is not dict or set(record) != {'path', 'bytes', 'sha256', 'sources'}:
            raise ValueError('invalid capsule member schema')
        name = record['path']; path = _path(root, name)
        if name in seen or name not in expected or record['sha256'] != expected[name]['sha256'] or not _same(record['sources'], expected[name]['sources']):
            raise ValueError('historical member identity mismatch')
        seen.add(name)
        if type(record['bytes']) is not int or record['bytes'] < 0 or _checked_file(path, record['sha256']).stat().st_size != record['bytes']:
            raise ValueError('historical bytes mismatch')
    files = set()
    directories = {str(parent) for p in seen for parent in PurePosixPath(p).parents} | {'.'}
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            directory = Path(parent) / name; _no_symlink(directory)
            if directory.relative_to(root).as_posix() not in directories:
                raise ValueError('extra capsule directory')
        for name in names:
            path = Path(parent) / name; _no_symlink(path)
            files.add(path.relative_to(root).as_posix())
    if seen != set(expected) or files != seen | {'inventory.json'}:
        raise ValueError('missing or extra capsule member')
    from .retrieval_data import validate_contract
    validate_contract(root)
    return value

CONFIG_SHA = '21bac9b6313c27f33880cd05501f0d5ac9631d4e3f45fefa070198d78db20480'
PUBLIC_PINS = {
    'artifacts/p4b-pipeline-manifest.json': 'd31cc460273c3d94afc48b70f976cd0697aa893af5107d2495f772fd031781f0',
    'artifacts/p5-sid-manifest.json': '26b9073f6bf5685965f12e528a4b8803710bdb108229da7b94d1247a1f5453cc',
}


def _git(root):
    import subprocess
    return {'git_base': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True))}


def _environment():
    import platform
    from importlib.metadata import version
    if tuple(map(int, platform.python_version_tuple()[:2])) != (3, 11):
        raise ValueError('Python 3.11 required')
    return {'python': platform.python_version(), 'exact_dependencies': {name: version(name) for name in ('numpy', 'torch', 'pyarrow', 'PyYAML')}}


def _declarations(root):
    p3d = _json(_checked_file(_path(root, P3D_FILE), P3D_SHA))
    public = {name: _json(_checked_file(_path(root, name), digest)) for name, digest in PUBLIC_PINS.items()}
    p4b = public['artifacts/p4b-pipeline-manifest.json']; p5 = public['artifacts/p5-sid-manifest.json']
    frozen = _json(_path(root, 'artifacts/p3-data-contract-manifest.json'))
    p4b_producer = dict(p4b['consumer_runtime'], source_fingerprint=p4b['real_consumer_source_fingerprint'])
    producers = {'p3': p3d['legacy_p3_producer'], 'p4b_real': p4b_producer,
                 'p5': {k: p5[k] for k in ('source_fingerprint', 'git_base', 'python_major_minor', 'dependency_versions')}}
    query_pins = {x['role']: x['sha256'] for x in p3d['full_query_pins']}
    inputs = {
        'prepared': {'manifest_sha256': p4b['manifest_sha256'], 'producer_ref': 'p4b_real',
                     'artifacts': p4b['artifacts'], 'examples': p4b['examples'], 'catalog_items': p4b['catalog_items'],
                     'binding_checks': {'consumer': p4b_producer, 'input_records': p4b['input_records'],
                                        'query_pins': p4b['query_pins'], 'legacy_producer': p3d['legacy_p3_producer'],
                                        'reconciliation': p4b['reconciliation'], 'test_gate': 'closed_hash_only'}},
        'sid': {'manifest_sha256': p5['server_manifest_sha256'], 'producer_ref': 'p5',
                'artifacts': {k: v['sha256'] for k, v in p5['artifacts'].items()},
                'effective_config': p5['effective_config'], 'feature_array_sha256': p5['feature_array_sha256'],
                'catalog_items': p4b['catalog_items'],
                'evidence_inputs': p5['input_fingerprints'],
                'frozen_manifest_sha256': frozen_sha(root),
                'evidence_checks': {'policy_sha256': frozen['policy_sha256'], 'p3_producer': p3d['legacy_p3_producer'],
                    'p4_manifest_sha256': next(x['sha256'] for x in p5['input_fingerprints'] if x['role']=='p4_manifest'),
                    'p4_source_fingerprint': '51db4ea2320b0f8aae0a92d2036ba273bd1984966c6db553b640a5f17956176e'}},
        'validation': {'role': 'validation-queries.json', 'artifact_sha256': query_pins['validation-queries.json'], 'producer_ref': 'p3'},
        'test_declaration': {'role': 'test-queries.json', 'artifact_sha256': query_pins['test-queries.json'], 'read_allowed': False},
    }
    for role, name in [('catalog', 'catalog.json'), ('features', 'content.npy'), ('examples', 'examples.jsonl'), ('idmap', 'idmap.json'), ('support', 'support.json')]:
        inputs[role] = {'artifact_sha256': p4b['artifacts'][name], 'producer_ref': 'p4b_real'}
    if p4b['frozen_contract_manifest_sha256'] != P3D_SHA or not _same(p4b['query_pins'], p3d['full_query_pins']) or p5['input_fingerprints'][0]['sha256'] != frozen_sha(root) or frozen['train_evidence']['producer'] != p3d['legacy_p3_producer']:
        raise ValueError('historical producer declarations conflict')
    return producers, inputs


def frozen_sha(root):
    # This file is already anchored by P3-D; no raw input is read.
    return file_sha256(_path(root, 'artifacts/p3-data-contract-manifest.json'))


def build_contract(root: Path) -> dict:
    """Freeze public declarations and actual consumer bytes; never open real inputs."""
    root = _no_symlink(root)
    from .retrieval_data import validate_contract
    validate_contract(root)
    capsule = _path(root, 'provenance/g0-v1')
    inventory_sha = file_sha256(_path(capsule, 'inventory.json'))
    validate_legacy_capsule(capsule, inventory_sha)
    _checked_file(_path(root, 'configs/generative-core-v1.json'), CONFIG_SHA)
    old = _json(capsule / G0_FILE)
    names = sorted(set(old['consumer_source']['files']) | set(PUBLIC_PINS) | {
        'src/vkgr/generative_data.py', 'tests/test_generative_data.py', 'docs/generative_contract_v1.md',
        'configs/generative-core-v1.json', 'provenance/g0-v1/inventory.json'})
    files = {name: file_sha256(_no_symlink(_path(root, name))) for name in names}
    producer, inputs = _declarations(root)
    consumer = {**_git(root), **_environment(), 'source_members': files,
                'source_sha256': hashlib.sha256(canonical_bytes(files)).hexdigest(),
                'lock_sha256': files['uv.lock']}
    bound = [{'role': role, 'path': name, 'bytes': _path(root, name).stat().st_size, 'sha256': files[name]}
             for role, name in [('technical_contract', 'docs/generative_contract_v1.md'), ('core_config', 'configs/generative-core-v1.json')]]
    return {'schema_version': 1, 'task_id': 'G1', 'protocol_version': 'generative-like-sid-v1', 'status': 'contract_frozen',
            'legacy_capsule': {'inventory_sha256': inventory_sha, 'p3d_manifest_sha256': P3D_SHA, 'g0_manifest_sha256': G0_SHA},
            'bound_files': bound, 'historical_producers': producer, 'inputs': inputs, 'consumer': consumer,
            'test_gate': 'closed', 'real_sid_audit': 'pending_server_authorization'}


class _Contract(dict):
    """Sealed context from load_contract; synthetic unit oracles isolate loaders."""
    def __init__(self, value, root):
        super().__init__(json.loads(canonical_bytes(value)))
        self.root = Path(root)
        self._bytes = canonical_bytes(value)


def _require_contract(value):
    if type(value) is not _Contract or canonical_bytes(dict(value)) != value._bytes:
        raise ValueError('validated unchanged contract context required')
    return value


def load_contract(root: Path, manifest_path: Path, expected_sha256: str) -> dict:
    value = _json(_checked_file(manifest_path, expected_sha256))
    expected = build_contract(root)
    if not _same(value, expected):
        raise ValueError('contract schema, consumer or historical identity mismatch')
    return _Contract(value, root)


def _strict_example(row, idmap, index):
    from .retrieval_data import validate_example
    if type(row) is not dict or type(row.get('example_id')) is not int or any(type(row.get(k)) is not list for k in ('history', 'history_positions', 'past_likes', 'target_position')):
        raise ValueError('invalid typed training example')
    validate_example(row, idmap, index)


def _compare_fields(actual, expected, role):
    if type(actual) is not dict or any(k not in actual or not _same(actual[k], v) for k, v in expected.items()):
        raise ValueError(role + ' binding mismatch')


class _Prepared(dict):
    def __init__(self, value):
        super().__init__(value)
        self._identity = _prepared_identity(self)


def _prepared_identity(value):
    from .sid_run import feature_digest
    return canonical_bytes({'catalog': value['catalog'].tolist(), 'features': feature_digest(value['features']),
                            'support': value['support'], 'idmap': value['idmap'], 'manifest': value['manifest'],
                            'manifest_sha256': value['manifest_sha256'], 'examples_path': str(value['examples_path']),
                            'example_count': value['example_count']})


def _require_prepared(value):
    if type(value) is not _Prepared or _prepared_identity(value) != value._identity:
        raise ValueError('validated unchanged prepared context required')
    return value


def load_prepared(manifest_path: Path, contract: dict) -> dict:
    """Static P4B consumer validation, including every sequential training prefix."""
    import numpy as np
    from .retrieval_data import verify_prepared, catalog_map
    contract = _require_contract(contract)
    spec = contract['inputs']['prepared']
    # Caller pin precedes parsing or access to any bundle artifact.
    path = _checked_file(manifest_path, spec['manifest_sha256'])
    if path.name != 'prepare-manifest.json':
        raise ValueError('fixed prepared manifest role required')
    value = _json(path)
    required = {'schema_version', 'task_id', 'kind', 'status', 'binding', 'examples', 'catalog_items', 'artifacts'}
    if type(value) is not dict or set(value) != required or not _same(value['schema_version'], 1) or value['task_id'] != 'P4B' or value['kind'] != 'prepare' or value['status'] != 'success' or type(value['examples']) is not int or type(value['catalog_items']) is not int or value['examples'] < 1 or value['catalog_items'] < 1:
        raise ValueError('invalid typed prepared manifest schema')
    if not _same(value['artifacts'], spec['artifacts']) or value['examples'] != spec['examples'] or value['catalog_items'] != spec['catalog_items']:
        raise ValueError('prepared artifact/count declaration mismatch')
    binding = value['binding']; checks = spec['binding_checks']
    keys = {'consumer', 'input_records', 'query_pins', 'legacy_producer', 'legacy_training_examples_sha256', 'test_gate', 'resources_start', 'resources_before_write', 'reconciliation'}
    if type(binding) is not dict or set(binding) != keys:
        raise ValueError('invalid prepared binding schema')
    consumer_checks = checks['consumer']
    _compare_fields(binding['consumer'], consumer_checks, 'prepared producer')
    if set(binding['consumer']) - set(consumer_checks) not in (set(), {'source_transport_origin'}):
        raise ValueError('unexpected prepared producer identity field')
    _compare_fields(binding, {k: v for k, v in checks.items() if k != 'consumer'}, 'prepared input')
    legacy_hash = binding['legacy_training_examples_sha256']
    if type(legacy_hash) is not str or len(legacy_hash) != 64 or any(c not in '0123456789abcdef' for c in legacy_hash):
        raise ValueError('invalid legacy training examples pin')
    names = {'catalog.json', 'content.npy', 'examples.jsonl', 'idmap.json', 'support.json'}
    if type(value['artifacts']) is not dict or set(value['artifacts']) != names:
        raise ValueError('prepared fixed artifact roles required')
    paths = {name: _checked_file(_path(path.parent, name), digest) for name, digest in value['artifacts'].items()}
    catalog = _json(paths['catalog.json']); idmap = catalog_map(catalog)
    support = _json(paths['support.json'])
    if type(support) is not list or any(type(i) is not int or i not in idmap for i in support) or support != sorted(set(support)):
        raise ValueError('invalid support identity')
    expected_idmap = {'items': catalog, 'pad': 0, 'start': 1, 'supported_indices': sorted(idmap[i] for i in support)}
    actual_idmap = _json(paths['idmap.json'])
    if not _same(actual_idmap, expected_idmap):
        raise ValueError('typed idmap identity mismatch')
    features = np.load(paths['content.npy'], mmap_mode='r', allow_pickle=False)
    if features.dtype != np.float32 or features.shape != (len(catalog)+1, 64) or not np.isfinite(features).all() or np.any(features[0]):
        raise ValueError('prepared content schema mismatch')
    previous = {}; previous_target = None; count = 0; supported = set(support)
    with paths['examples.jsonl'].open('rb') as stream:
        for line in stream:
            # Strict parser also refuses duplicate fields and nonfinite values.
            row = json.loads(line, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
            _strict_example(row, idmap, count)
            if not set(row['past_likes'] + [row['target']]) <= supported:
                raise ValueError('training prefix support mismatch')
            if previous_target is not None and row['target_position'] <= previous_target:
                raise ValueError('global training prefix ordering conflict')
            user = row['user_id']; prior = previous.get(user)
            if prior is None:
                if len(row['history']) != 1 or row['past_likes'] != sorted(set(row['history'])):
                    raise ValueError('first prefix evidence mismatch')
            elif row['history'] != (prior['history']+[prior['target']])[-50:] or row['history_positions'] != (prior['history_positions']+[prior['target_position']])[-50:] or row['past_likes'] != sorted(set(prior['past_likes']+[prior['target']])):
                raise ValueError('sequential full prefix identity mismatch')
            previous[user] = row; previous_target = row['target_position']; count += 1
    if count != value['examples'] or len(catalog) != value['catalog_items']:
        raise ValueError('prepared example/catalog denominator mismatch')
    verify_prepared(path.parent)
    ids = np.array(catalog, dtype=np.uint32); ids.setflags(write=False)
    return _Prepared({'catalog': ids, 'features': features, 'support': support, 'idmap': actual_idmap,
                      'manifest': value, 'manifest_sha256': spec['manifest_sha256'],
                      'examples_path': paths['examples.jsonl'], 'example_count': count})


def _unique_pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('duplicate JSON field')
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError('nonfinite JSON value')


def load_sid(manifest_path: Path, contract: dict, prepared: dict):
    """Pinned static SID artifacts + exact prepared catalog/content agreement."""
    import numpy as np
    from .sid_run import load_static, feature_digest
    contract = _require_contract(contract); prepared = _require_prepared(prepared)
    if prepared['manifest_sha256'] != contract['inputs']['prepared']['manifest_sha256']:
        raise ValueError('prepared context/contract identity binding mismatch')
    spec = contract['inputs']['sid']; path = _checked_file(manifest_path, spec['manifest_sha256'])
    if path.name != 'run_manifest.json':
        raise ValueError('fixed SID manifest role required')
    value = _json(path)
    if not _same(value.get('schema_version'), 1) or type(value.get('catalog_items')) is not int or value.get('catalog_items') != spec['catalog_items'] or not _same(value.get('artifacts'), spec['artifacts']) or not _same(value.get('effective_config'), spec['effective_config']) or value.get('feature_array_sha256') != spec['feature_array_sha256']:
        raise ValueError('SID declared artifact/config identity mismatch')
    producer = contract['historical_producers']['p5']
    _compare_fields(value.get('producer'), {'source_fingerprint': producer['source_fingerprint'],
        'git_commit': producer['git_base'], 'python_major_minor': producer['python_major_minor'],
        'dependencies': producer['dependency_versions']}, 'SID producer')
    evidence = value.get('evidence')
    if type(evidence) is not dict or evidence.get('frozen_manifest_sha256') != spec['frozen_manifest_sha256'] or type(evidence.get('inputs')) is not list:
        raise ValueError('SID evidence schema mismatch')
    records = []
    for row in evidence['inputs']:
        if type(row) is not dict or set(row) != {'role', 'path', 'sha256'}:
            raise ValueError('SID evidence input schema mismatch')
        records.append({'role': row['role'], 'sha256': row['sha256']})
    if not _same(records, spec['evidence_inputs']):
        raise ValueError('SID input lineage mismatch')
    _compare_fields(evidence, spec['evidence_checks'], 'SID evidence')
    for name, digest in spec['artifacts'].items():
        _checked_file(_path(path.parent, name), digest)
    model = load_static(path, spec['manifest_sha256'])
    if not np.array_equal(model.ids, prepared['catalog']):
        raise ValueError('SID/prepared catalog identity mismatch')
    features = np.load(_path(path.parent, 'normalized-features.npy'), mmap_mode='r', allow_pickle=False)
    if feature_digest(features) != feature_digest(prepared['features'][1:]):
        raise ValueError('SID/prepared content bytes mismatch')
    if type(evidence.get('feature_zero_norm')) is not int or evidence['feature_zero_norm'] != int(np.count_nonzero(np.linalg.norm(features, axis=1) == 0)):
        raise ValueError('SID zero-feature evidence mismatch')
    model.ids.setflags(write=False); model.sids.setflags(write=False)
    model.keys.setflags(write=False); model.key_items.setflags(write=False)
    return _ValidatedCodec(model)


def _check_codec(codec, prepared=None):
    import numpy as np
    from .tiger import SIDTokens
    if not isinstance(codec, SIDTokens):
        raise ValueError('SIDTokens codec required')
    model = codec.catalog
    if model.ids.dtype != np.uint32 or model.ids.ndim != 1 or not len(model.ids) or np.any(model.ids[1:] <= model.ids[:-1]) or model.sids.dtype != np.uint32 or model.sids.shape != (len(model.ids), 4) or model.depth != 3 or type(model.branching) is not int or model.branching < 2 or np.any(model.sids[:, :3] >= model.branching):
        raise ValueError('invalid codec catalog/SID schema')
    if model.bases != [model.branching]*3+[int(model.sids[:, 3].max())+1] or codec.offsets != list(np.cumsum([2]+model.bases[:-1])) or codec.vocab_size != 2+sum(model.bases):
        raise ValueError('codec namespace identity mismatch')
    order = np.lexsort((model.ids, model.sids[:, 2], model.sids[:, 1], model.sids[:, 0]))
    codes = model.sids[order]
    starts = np.r_[True, np.any(codes[1:, :3] != codes[:-1, :3], axis=1)]
    row_indices = np.arange(len(codes))
    suffix = row_indices - np.maximum.accumulate(np.where(starts, row_indices, 0))
    if not np.array_equal(codes[:, 3], suffix):
        raise ValueError('noncanonical suffix or duplicate SID')
    if prepared is not None and not np.array_equal(model.ids, prepared['catalog']):
        raise ValueError('codec/prepared catalog mismatch')
    return model


def audit_sid(codec, prepared: dict) -> dict:
    """Full legality audit; bounded train-defined content diagnostic, no N²."""
    import numpy as np
    prepared = _require_prepared(prepared); model = _check_codec(codec, prepared)
    audit = model.audit()
    if audit['full_sid_collisions'] or audit['roundtrip_items'] != len(model.ids) or audit['reachable_items'] != len(model.ids) or audit['valid_prefix_deadends']:
        raise ValueError('SID roundtrip/reachability gate failed')
    support = set(prepared['support'])
    indices = np.array([i for i, item in enumerate(model.ids) if int(item) in support], dtype=np.int64)
    codes = model.sids[indices, :3]
    order = np.lexsort((model.ids[indices], codes[:, 2], codes[:, 1], codes[:, 0]))
    indices = indices[order]
    distances = []; eligible_pairs = 0; singleton_items = 0; group = []
    def finish(rows):
        nonlocal eligible_pairs, singleton_items
        eligible_pairs += max(0, len(rows)-1)
        singleton_items += int(len(rows) == 1)
        for a, b in zip(rows, rows[1:]):
            if len(distances) == 4096:
                break
            delta = prepared['features'][a+1].astype(np.float64) - prepared['features'][b+1].astype(np.float64)
            distances.append(float(np.dot(delta, delta)))
    for index in indices.tolist():
        if group and not np.array_equal(model.sids[group[-1], :3], model.sids[index, :3]):
            finish(group); group = []
        group.append(index)
    finish(group)
    return {'sid': audit, 'resources': 'not_measured',
            'content_distance': {'population': 'train_positive_supported_items_only',
                'pairing': 'adjacent_item_ids_within_semantic_leaf', 'max_pairs': 4096,
                'order': 'lexicographic_semantic_leaf_then_item_id', 'metric': 'float64_squared_l2_on_frozen_float32_features',
                'supported_items': len(indices), 'unpaired_singleton_items': singleton_items,
                'eligible_pair_count': eligible_pairs, 'pair_count': len(distances),
                'truncated_pairs': eligible_pairs-len(distances),
                'mean_squared_l2': float(np.mean(distances)) if distances else None,
                'max_squared_l2': max(distances) if distances else None},
            'claim_boundary': 'legality and bounded train-only distance; no semantic quality or real resource claim'}


def _position(value, max_week):
    if type(value) is not list or len(value) != 2 or any(type(x) is not int or x < 0 for x in value) or value[0] > max_week:
        raise ValueError('invalid typed ordinal position')
    return value


def load_validation(path: Path, contract: dict, catalog) -> list[dict]:
    """Accept only fixed validation identity; reject test role/hash before opening."""
    import numpy as np
    from .retrieval_data import catalog_map
    contract = _require_contract(contract); spec = contract['inputs']['validation']
    path = Path(path)
    if contract['test_gate'] != 'closed' or spec['role'] != 'validation-queries.json' or path.name != spec['role'] or spec['artifact_sha256'] == contract['inputs']['test_declaration']['artifact_sha256']:
        raise ValueError('validation-only identity; test gate closed')
    if not isinstance(catalog, np.ndarray) or catalog.dtype != np.uint32 or catalog.ndim != 1:
        raise ValueError('typed catalog array required')
    members = catalog_map(catalog.tolist()); rows = _json(_checked_file(path, spec['artifact_sha256']))
    if type(rows) is not list or not rows:
        raise ValueError('nonempty fixed validation population required')
    users = set(); keys = {'user_id', 'history', 'history_positions', 'target', 'target_position', 'oov'}
    for row in rows:
        if type(row) is not dict or set(row) != keys or any(type(row[k]) is not int or not 0 <= row[k] < 2**32 for k in ('user_id', 'target')) or row['user_id'] in users or type(row['oov']) is not bool or row['oov'] != (row['target'] not in members):
            raise ValueError('invalid typed validation identity/OOV')
        users.add(row['user_id'])
        history = row['history']; positions = row['history_positions']
        if type(history) is not list or not 1 <= len(history) <= 50 or any(type(x) is not int or x not in members for x in history) or type(positions) is not list or len(positions) != len(history):
            raise ValueError('validation history mismatch')
        target = _position(row['target_position'], 25)
        if target[0] != 25:
            raise ValueError('validation week mismatch')
        for position in positions:
            _position(position, 24)
        if any(a >= b for a, b in zip(positions, positions[1:])) or positions[-1] >= target:
            raise ValueError('validation strict-past ordering conflict')
    return rows


def sid_targets(rows, codec):
    """Four full-vocabulary teacher-forced positions; no target/negative masking."""
    from collections.abc import Sequence
    if type(codec) is _ValidatedCodec:
        codec.check_identity()
    else:
        _check_codec(codec)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError('nonempty training row sequence required')
    histories = []; targets = []
    for row in rows:
        if type(row) is not dict or 'history' not in row or 'target' not in row or type(row['history']) is not list or not 1 <= len(row['history']) <= 50 or type(row['target']) is not int or any(type(i) is not int for i in row['history']):
            raise ValueError('invalid training history/target schema')
        for item in row['history'] + [row['target']]:
            codec.catalog.encode(item)
        histories.append(row['history']); targets.append(row['target'])
    return codec.batch(histories, targets, device='cpu')


from .tiger import SIDTokens


class _ValidatedCodec(SIDTokens):
    """Once-validated immutable mapping; batch identity checks are O(1)."""
    _arrays = ('ids', 'sids', 'keys', 'key_items')

    def __init__(self, model):
        import numpy as np
        # Immutable bytes backing prevents setflags(write=True), even after return.
        for name in self._arrays:
            value = getattr(model, name)
            setattr(model, name, np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape))
        super().__init__(model)
        _check_codec(self)
        self._model = model
        self._references = [getattr(model, name) for name in self._arrays]
        self._array_metadata = [(v.shape, v.dtype.str, v.nbytes, v.strides, v.ctypes.data) for v in self._references]
        self._namespace = self._namespace_bytes()

    def _namespace_bytes(self):
        return canonical_bytes([self.catalog.branching, self.catalog.depth, self.catalog.suffix_vocabulary,
                                self.catalog.bases, self.offsets, self.vocab_size])

    def check_identity(self):
        if self.catalog is not self._model or self._namespace_bytes() != self._namespace:
            raise ValueError('readonly codec namespace identity mismatch')
        for name, ref, metadata in zip(self._arrays, self._references, self._array_metadata):
            value = getattr(self.catalog, name)
            if value is not ref or value.flags.writeable or (value.shape, value.dtype.str, value.nbytes, value.strides, value.ctypes.data) != metadata:
                raise ValueError('readonly codec array identity mismatch')
