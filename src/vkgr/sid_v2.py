"""SIDMAP_V2: the GTOK-01 semantic-ID bundle contract, isolated from the frozen P5 contract.

The P5 loader (sid_run.load_static / gd.load_sid) is pinned to a 16-ary balanced k-means mapping, so the new
tokenizers use their own bundle: mapping.npz, tokenizer.json, audit.json, run_manifest.json (published last) and,
for RQ-VAE, weights.npz and train-log.json. ``load_bundle`` is fail-closed: every pinned hash, the catalog and
content identity, the codec rules of the frozen G code (``gd._check_codec``), the recomputed audit and gates, and
the weights-to-mapping export are re-verified. Nothing here edits or calls the P5 loader.
"""
import hashlib
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import generative_data as gd
from . import sid_rqvae as rq
from . import sid_tok_audit as au
from .data_audit import canonical_bytes, file_sha256
from .sid import SIDCatalog

SCHEMA = 'sidmap-v2'
PROTOCOL = 'generative-like-sid-v2'
TASK_ID = 'G-TOK-01'
SID_VERSIONS = ('SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D3_S42', 'SID_BHKM_B256D3_S42')
ENGINEERING_SID_VERSIONS = SID_VERSIONS + ('SID_V2_REPLAY_BHKM_B16D3',)
CONTENT_V1_FEATURE_SHA256 = au.CONTENT_V1_FEATURE_SHA256
VERSION_SPEC = {'SID_RQVAE_B256D3_S42': ('rqvae', 256), 'SID_RQVAE_B16D3_S42': ('rqvae', 16),
                'SID_BHKM_B256D3_S42': ('bhkm', 256), 'SID_V2_REPLAY_BHKM_B16D3': ('bhkm', 16)}
MANIFEST_NAME = 'run_manifest.json'
ARTIFACTS = {'rqvae': ('mapping.npz', 'tokenizer.json', 'weights.npz', 'audit.json', 'train-log.json'),
             'bhkm': ('mapping.npz', 'tokenizer.json', 'audit.json')}
MANIFEST_KEYS = frozenset({
    'schema', 'protocol', 'task_id', 'status', 'sid_version', 'kind', 'engineering', 'branching', 'catalog_items',
    'suffix_vocabulary', 'vocab_size', 'content_feature_sha256', 'catalog_sha256', 'artifacts', 'fit_config_sha256',
    'producer', 'gates'})
RQ_INFO_KEYS = frozenset({'status', 'steps', 'max_steps', 'elapsed', 'wall_cap_seconds', 'required_steps', 'reconstruction'})
TOKENIZER_KEYS = frozenset({'sid_version', 'kind', 'branching', 'config'})
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class LoadedSID:
    codec: object
    manifest: dict
    info: dict


def _hex64(value, what):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{what}: lowercase sha256 hex required')
    return value


def _spec(sid_version, engineering):
    if type(engineering) is not bool:
        raise ValueError('engineering flag must be a bool')
    allowed = ENGINEERING_SID_VERSIONS if engineering else SID_VERSIONS
    if type(sid_version) is not str or sid_version not in allowed:
        raise ValueError('unregistered sid_version for this context')
    return VERSION_SPEC[sid_version]


def catalog_digest(ids):
    """Same function as the G context's catalog_sha256: sha256 of the canonical JSON list of item ids."""
    return hashlib.sha256(canonical_bytes([int(x) for x in np.asarray(ids).tolist()])).hexdigest()


def canonical_sids(ids, codes):
    """Rows sorted by ascending item id, 4th code assigned canonically; returns (ids uint32, sids uint32 [N, 4])."""
    ids, codes = np.asarray(ids), np.asarray(codes)
    if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= 2 ** 32 or codes.shape != (len(ids), 3):
        raise ValueError('invalid ids/codes')
    order = np.argsort(ids, kind='stable')
    ordered = ids[order].astype(np.uint32)
    return ordered, rq.assign_fourth_code(ordered, codes[order])


def build_codec(ids, sids, branching, catalog_ids=None):
    """Frozen G codec for an arbitrary (unbalanced) mapping; the catalog keeps nodes=[] so that pickles stay small."""
    if not isinstance(ids, np.ndarray) or not isinstance(sids, np.ndarray) or ids.dtype != np.uint32 \
            or sids.dtype != np.uint32 or ids.ndim != 1 or not len(ids) or sids.shape != (len(ids), 4):
        raise ValueError('uint32 ids [N] and sids [N, 4] required')
    if type(branching) is not int or branching < 2:
        raise ValueError('branching must be an int >= 2')
    codec = gd._ValidatedCodec(SIDCatalog(ids, sids, branching, 3, []))
    gd._check_codec(codec, {'catalog': ids if catalog_ids is None else np.asarray(catalog_ids)})
    return codec


def _write_npz(path, arrays):
    """Deterministic uncompressed npz (fixed member timestamps) so that equal arrays give equal bytes."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.array(array, order='C'), allow_pickle=False)  # keeps 0-d arrays 0-d
            info = zipfile.ZipInfo(name + '.npy', date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, buffer.getvalue())


def _load_npz(path, names=None):
    try:
        with np.load(path, allow_pickle=False) as data:
            if names is not None and set(data.files) != set(names):
                raise ValueError('npz member set mismatch')
            return {key: data[key] for key in data.files}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('unreadable npz artifact') from exc


def _check_state_arrays(arrays, branching):
    """Validate {'K', tensors...} against a freshly built RQVAE(branching); returns {'K', 'state_dict'}."""
    k = arrays.get('K')
    if k is None or k.shape != () or k.dtype != np.int64 or int(k) != branching:
        raise ValueError('weights K must equal the branching')
    expected = rq.RQVAE(branching).state_dict()
    if set(arrays) - {'K'} != set(expected):
        raise ValueError('weights tensor names mismatch')
    state_dict = {}
    for name, reference in expected.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != tuple(reference.shape) or not np.isfinite(value).all():
            raise ValueError('weights tensor dtype/shape/finiteness mismatch')
        state_dict[name] = torch.from_numpy(np.array(value, dtype=np.float32))
    return {'K': branching, 'state_dict': state_dict}


def _weights_arrays(state, branching):
    if type(state) is not dict or set(state) != {'K', 'state_dict'} or state['K'] != branching or type(state['state_dict']) is not dict:
        raise ValueError('invalid weights state')
    arrays = {'K': np.array(branching, dtype=np.int64)}
    for name in rq.RQVAE(branching).state_dict():
        if name not in state['state_dict']:
            raise ValueError('weights tensor names mismatch')
        arrays[name] = state['state_dict'][name].detach().cpu().numpy().astype(np.float32)
    _check_state_arrays(arrays, branching)
    return arrays


def write_bundle(output, *, sid_version, engineering, ids, sids, content_feature_sha256, tokenizer, audit, gates,
                 fit_config_sha256, producer, weights=None, train_log=None, require_passed=True):
    """Write a new bundle directory; ``run_manifest.json`` is replaced into place last.

    ``require_passed=False`` exists only to produce honestly-failing bundles for loader tests; load_bundle
    rejects them regardless.
    """
    kind, branching = _spec(sid_version, engineering)
    codec = build_codec(ids, sids, branching)
    _hex64(content_feature_sha256, 'content digest')
    _hex64(fit_config_sha256, 'fit config digest')
    if type(producer) is not dict or not producer or any(type(k) is not str or not k for k in producer):
        raise ValueError('producer source digests required')
    for digest in producer.values():
        _hex64(digest, 'producer digest')
    if type(tokenizer) is not dict:
        raise ValueError('tokenizer config must be a dict')
    if type(audit) is not dict or set(audit) != {'mapping', 'rq'} or type(audit['mapping']) is not dict:
        raise ValueError('audit must hold exactly mapping and rq')
    mapping_audit = audit['mapping']
    if mapping_audit.get('kind') != kind or mapping_audit.get('branching') != branching \
            or mapping_audit.get('items') != len(ids) or mapping_audit.get('content_sha256') != content_feature_sha256:
        raise ValueError('audit does not describe this mapping/content')
    if (kind == 'rqvae') != (audit['rq'] is not None) or (kind == 'rqvae') != (weights is not None) \
            or (kind == 'rqvae') != (train_log is not None):
        raise ValueError('rqvae bundles need rq info, weights and train log; bhkm bundles none of them')
    if type(gates) is not dict or set(gates) != {'passed', 'checks'}:
        raise ValueError('gates must hold exactly passed and checks')
    if require_passed and gates['passed'] is not True:
        raise ValueError('refusing to publish a bundle whose gates did not pass')
    files = {'mapping.npz': None,
             'tokenizer.json': canonical_bytes({'sid_version': sid_version, 'kind': kind, 'branching': branching,
                                                'config': tokenizer}),
             'audit.json': canonical_bytes(audit)}
    if kind == 'rqvae':
        if type(train_log) is not dict:
            raise ValueError('train log must be a dict')
        arrays = _weights_arrays(weights, branching)
        files['train-log.json'] = canonical_bytes(train_log)
    output = Path(output).absolute()
    gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory inside an existing one')
    output.mkdir()
    _write_npz(output / 'mapping.npz', {'ids': ids, 'sids': sids})
    for name, data in files.items():
        if data is not None:
            (output / name).write_bytes(data)
    if kind == 'rqvae':
        _write_npz(output / 'weights.npz', arrays)
    manifest = {
        'schema': SCHEMA, 'protocol': PROTOCOL, 'task_id': TASK_ID, 'status': 'complete', 'sid_version': sid_version,
        'kind': kind, 'engineering': engineering, 'branching': branching, 'catalog_items': int(len(ids)),
        'suffix_vocabulary': int(codec.catalog.suffix_vocabulary), 'vocab_size': int(codec.vocab_size),
        'content_feature_sha256': content_feature_sha256, 'catalog_sha256': catalog_digest(ids),
        'artifacts': {name: file_sha256(output / name) for name in ARTIFACTS[kind]},
        'fit_config_sha256': fit_config_sha256, 'producer': dict(producer), 'gates': gates}
    temporary = output / (MANIFEST_NAME + '.tmp')
    temporary.write_bytes(canonical_bytes(manifest))
    os.replace(temporary, output / MANIFEST_NAME)
    return manifest


def _check_gates_passed(gates):
    if type(gates) is not dict or set(gates) != {'passed', 'checks'} or gates['passed'] is not True:
        raise ValueError('SID bundle gates did not pass')


def _check_content(manifest, content, expected_digest):
    _hex64(expected_digest, 'expected content digest')
    if not isinstance(content, np.ndarray) or content.dtype != np.float32 or content.ndim != 2:
        raise ValueError('float32 content rows required')
    if manifest['content_feature_sha256'] != expected_digest or au.feature_digest(content) != expected_digest:
        raise ValueError('content digest mismatch')


def _check_catalog(manifest, catalog_ids, mapping_ids):
    catalog_ids = np.asarray(catalog_ids)
    if catalog_ids.ndim != 1 or manifest['catalog_sha256'] != catalog_digest(catalog_ids) \
            or manifest['catalog_items'] != len(catalog_ids) or not np.array_equal(catalog_ids, mapping_ids):
        raise ValueError('catalog identity mismatch')


def _verify_artifact(directory, name, digest):
    return gd._checked_file(gd._path(directory, name), digest)


def _check_weights(state, ids, sids, content):
    if not np.array_equal(rq.assign_fourth_code(ids, rq.export_codes(state, content)), sids):
        raise ValueError('weights do not reproduce the mapping')


def _check_audit(manifest, stored, kind, branching, ids, sids, content, state, engineering):
    if type(stored) is not dict or set(stored) != {'mapping', 'rq'}:
        raise ValueError('audit.json must hold exactly mapping and rq')
    recomputed = au.audit_mapping(ids, sids, content, branching=branching, kind=kind)
    if not au.audits_equal(stored['mapping'], recomputed):
        raise ValueError('audit.json differs from the recomputed audit')
    info, required = None, au.REQUIRED_STEPS
    if kind == 'rqvae':
        info = stored['rq']
        if type(info) is not dict or set(info) != RQ_INFO_KEYS:
            raise ValueError('audit.json rq section schema mismatch')
        if not au.audits_equal(info['reconstruction'], rq.reconstruction_report(state, content)):
            raise ValueError('reconstruction report differs from the recomputation')
        if not engineering and (info['required_steps'] != rq.PREREG['max_steps']
                                or info['wall_cap_seconds'] != rq.PREREG['wall_cap_seconds']):
            raise ValueError('formal bundles must carry the pre-registered training length and wall cap')
        required = info['required_steps']
    elif stored['rq'] is not None:
        raise ValueError('bhkm bundles carry no rq section')
    expected = au.gates(recomputed, kind, info, expected_content_sha256=manifest['content_feature_sha256'],
                        required_steps=required)
    if not au.audits_equal(manifest['gates'], expected):
        raise ValueError('manifest gates differ from the recomputed gates')


def load_bundle(manifest_path, manifest_sha256, *, catalog_ids, content, expected_content_digest=CONTENT_V1_FEATURE_SHA256,
                engineering=False):
    """Fail-closed loader. ``content`` is the float32 [N, 64] item rows (needed to recompute audit and export)."""
    try:
        return _load_bundle(manifest_path, manifest_sha256, catalog_ids, content, expected_content_digest, engineering)
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
        if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
            raise
        raise ValueError(f'malformed SIDMAP_V2 bundle: {type(exc).__name__}') from exc


def _load_bundle(manifest_path, manifest_sha256, catalog_ids, content, expected_content_digest, engineering):
    path = gd._checked_file(manifest_path, manifest_sha256)
    if path.name != MANIFEST_NAME:
        raise ValueError('fixed run_manifest.json role required')
    manifest = gd._json(path)
    if type(manifest) is not dict or set(manifest) != MANIFEST_KEYS:
        raise ValueError('manifest key set mismatch')
    for key, value in (('schema', SCHEMA), ('protocol', PROTOCOL), ('task_id', TASK_ID), ('status', 'complete')):
        if manifest[key] != value:
            raise ValueError(f'manifest {key} mismatch')
    kind, branching = _spec(manifest['sid_version'], engineering)
    if manifest['engineering'] is not engineering or manifest['kind'] != kind or type(manifest['branching']) is not int \
            or manifest['branching'] != branching:
        raise ValueError('manifest engineering/kind/branching mismatch')
    for key in ('catalog_items', 'suffix_vocabulary', 'vocab_size'):
        if type(manifest[key]) is not int:
            raise ValueError(f'manifest {key} must be an int')
    _check_gates_passed(manifest['gates'])
    _check_content(manifest, content, expected_content_digest)
    directory, names = path.parent, ARTIFACTS[kind]
    artifacts = manifest['artifacts']
    if type(artifacts) is not dict or set(artifacts) != set(names) or set(os.listdir(directory)) != set(names) | {MANIFEST_NAME}:
        raise ValueError('artifact set mismatch')
    for name in names:
        _verify_artifact(directory, name, artifacts[name])
    mapping = _load_npz(directory / 'mapping.npz', {'ids', 'sids'})
    ids, sids = mapping['ids'], mapping['sids']
    _check_catalog(manifest, catalog_ids, ids)
    if len(content) != len(ids):
        raise ValueError('content rows do not match the catalog')
    codec = build_codec(ids, sids, branching)
    if manifest['suffix_vocabulary'] != codec.catalog.suffix_vocabulary or manifest['vocab_size'] != codec.vocab_size:
        raise ValueError('manifest vocabulary does not match the mapping')
    tokenizer = gd._json(directory / 'tokenizer.json')
    if type(tokenizer) is not dict or set(tokenizer) != TOKENIZER_KEYS or type(tokenizer['config']) is not dict \
            or (tokenizer['sid_version'], tokenizer['kind'], tokenizer['branching']) != (manifest['sid_version'], kind, branching):
        raise ValueError('tokenizer.json does not match the manifest')
    state = None
    if kind == 'rqvae':
        if type(gd._json(directory / 'train-log.json')) is not dict:
            raise ValueError('train-log.json must be an object')
        state = _check_state_arrays(_load_npz(directory / 'weights.npz'), branching)
        _check_weights(state, ids, sids, content)
    _check_audit(manifest, gd._json(directory / 'audit.json'), kind, branching, ids, sids, content, state, engineering)
    info = {'sid_version': manifest['sid_version'], 'manifest_sha256': manifest_sha256, 'branching': branching,
            'suffix_vocabulary': codec.catalog.suffix_vocabulary, 'vocab_size': codec.vocab_size, 'kind': kind,
            'engineering': engineering}
    return LoadedSID(codec, manifest, info)
