"""SIDMAP_V2S: the GTOK-02 resolution-sweep bundle contract, isolated from SIDMAP_V2 (GTOK-01) and from the frozen P5 contract.

Six new balanced k-means tokenizers: branching 32 / 64 / 128 / 512 / 1024 at depth 3, and branching 256 at depth 2.
Depth 2 lives in the frozen four-position framework (``gd._check_codec`` demands depth 3 and sids [N, 4]) as
``[c0, c1, 0, rank]``: the third position is a constant zero and ``rank`` numbers the items of one (c0, c1) group by
ascending item id. Nothing here edits SIDMAP_V2; its helpers are imported.
"""
import os
from pathlib import Path

import numpy as np

from . import generative_data as gd
from . import sid_tok_audit as au
from . import sid_v2
from .data_audit import canonical_bytes, file_sha256

SCHEMA = 'sidmap-v2s'
PROTOCOL = 'generative-like-sid-v2s'
TASK_ID = 'G-TOK-02'
SID_VERSIONS = ('SID_BHKM_B32D3_S42', 'SID_BHKM_B64D3_S42', 'SID_BHKM_B128D3_S42', 'SID_BHKM_B512D3_S42',
                'SID_BHKM_B1024D3_S42', 'SID_BHKM_B256D2_S42')
ENGINEERING_SID_VERSIONS = SID_VERSIONS + ('SID_V2S_REPLAY_BHKM_B16D3',)
VERSION_SPEC = {'SID_BHKM_B32D3_S42': ('bhkm', 32, 3), 'SID_BHKM_B64D3_S42': ('bhkm', 64, 3),
                'SID_BHKM_B128D3_S42': ('bhkm', 128, 3), 'SID_BHKM_B512D3_S42': ('bhkm', 512, 3),
                'SID_BHKM_B1024D3_S42': ('bhkm', 1024, 3), 'SID_BHKM_B256D2_S42': ('bhkm', 256, 2),
                'SID_V2S_REPLAY_BHKM_B16D3': ('bhkm', 16, 3)}
CONTENT_V1_FEATURE_SHA256 = au.CONTENT_V1_FEATURE_SHA256
PARENT_BRANCHING = 256
MANIFEST_NAME = sid_v2.MANIFEST_NAME
ARTIFACTS = ('mapping.npz', 'tokenizer.json', 'audit.json')
MANIFEST_KEYS = sid_v2.MANIFEST_KEYS | frozenset({'depth'})
TOKENIZER_KEYS = sid_v2.TOKENIZER_KEYS
SWEEP_KEYS = frozenset({'depth', 'pad_digit_zero', 'balanced_basis', 'parent_manifest_sha256'})


def _spec(sid_version, engineering):
    """(kind, branching, depth) of a registered label; formal contexts do not know the engineering-only label."""
    if type(engineering) is not bool:
        raise ValueError('engineering flag must be a bool')
    allowed = ENGINEERING_SID_VERSIONS if engineering else SID_VERSIONS
    if type(sid_version) is not str or sid_version not in allowed:
        raise ValueError('unregistered sid_version for this context')
    return VERSION_SPEC[sid_version]


def pad_depth2(raw3):
    """[c0, c1, rank] -> [c0, c1, 0, rank] (uint32 [N, 4])."""
    raw3 = np.asarray(raw3)
    if raw3.dtype != np.uint32 or raw3.ndim != 2 or raw3.shape[1] != 3:
        raise ValueError('uint32 [N, 3] required')
    padded = np.zeros((len(raw3), 4), dtype=np.uint32)
    padded[:, 0], padded[:, 1], padded[:, 3] = raw3[:, 0], raw3[:, 1], raw3[:, 2]
    return padded


def unpad_depth2(sids):
    """[c0, c1, 0, rank] -> [c0, c1, rank]: the three columns the balance gate B1 is applied to."""
    sids = np.asarray(sids)
    if sids.dtype != np.uint32 or sids.ndim != 2 or sids.shape[1] != 4:
        raise ValueError('uint32 [N, 4] required')
    return np.ascontiguousarray(sids[:, [0, 1, 3]])


def derive_depth2_from_parent(ids, parent_sids):
    """Deterministic depth-2 mapping from a depth-3 parent: keep (c0, c1), number each group by ascending item id.

    Returns (sids4 [N, 4], raw3 [N, 3]) with sids4 = pad_depth2(raw3). No clustering; the inputs are not modified.
    """
    ids, parent = np.asarray(ids), np.asarray(parent_sids)
    if ids.dtype != np.uint32 or ids.ndim != 1 or not len(ids) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError('ids must be strictly ascending uint32')
    if parent.dtype != np.uint32 or parent.shape != (len(ids), 4):
        raise ValueError('parent sids must be uint32 [N, 4] aligned with ids')
    if parent[:, :2].max() >= PARENT_BRANCHING:
        raise ValueError('parent digits exceed the parent branching')
    key = parent[:, 0].astype(np.int64) * PARENT_BRANCHING + parent[:, 1].astype(np.int64)
    order = np.lexsort((np.arange(len(ids)), key))  # stable: ascending item id inside a group
    grouped = key[order]
    starts = np.r_[True, grouped[1:] != grouped[:-1]]
    rows = np.arange(len(ids))
    rank_sorted = rows - np.maximum.accumulate(np.where(starts, rows, 0))
    rank = np.empty(len(ids), dtype=np.uint32)
    rank[order] = rank_sorted.astype(np.uint32)
    raw3 = np.stack([parent[:, 0], parent[:, 1], rank], axis=1).astype(np.uint32)
    return pad_depth2(raw3), raw3


def sweep_section(sid_version, engineering, parent_manifest_sha256):
    """The ``sweep`` block of audit.json; depth, padding flag and balance basis follow from the registered label."""
    _, _, depth = _spec(sid_version, engineering)
    return {'depth': depth, 'pad_digit_zero': depth == 2, 'balanced_basis': 'raw3' if depth == 2 else 'full',
            'parent_manifest_sha256': parent_manifest_sha256}


def sweep_audit(ids, sids, content, *, branching, depth):
    """``sid_tok_audit.audit_mapping`` with one difference: for depth 2 the balance report B1 is computed on the
    three columns [c0, c1, rank] before padding. The function, the tolerance and the threshold are unchanged; the
    padded zero column is a constant, not a tree level, so B1 on four columns would call every collision a violation."""
    mapping = au.audit_mapping(ids, sids, content, branching=branching, kind='bhkm')
    if depth == 2:
        mapping['balanced_children'] = au.balanced_children(unpad_depth2(sids), branching)
    return mapping


def _hex64(value, what):
    return sid_v2._hex64(value, what)


def _check_padding(sids):
    if np.any(np.asarray(sids)[:, 2] != 0):
        raise ValueError('depth-2 mapping must carry a constant zero pad digit at position 3')


def _check_sweep(sweep, depth, tokenizer_config):
    """Structure of the sweep block against the registry and the tokenizer record (not against the data)."""
    if type(sweep) is not dict or set(sweep) != SWEEP_KEYS:
        raise ValueError('sweep section schema mismatch')
    if type(sweep['depth']) is not int or sweep['depth'] != depth \
            or sweep['pad_digit_zero'] is not (depth == 2) or sweep['balanced_basis'] != ('raw3' if depth == 2 else 'full'):
        raise ValueError('sweep section differs from the registered depth')
    parent = sweep['parent_manifest_sha256']
    if depth == 2:
        _hex64(parent, 'parent manifest digest')
        if tokenizer_config.get('parent_manifest_sha256') != parent:
            raise ValueError('sweep parent differs from the tokenizer record')
    elif parent is not None or 'parent_manifest_sha256' in tokenizer_config:
        raise ValueError('only depth-2 bundles have a parent')


def write_bundle(output, *, sid_version, engineering, ids, sids, content_feature_sha256, tokenizer, audit, gates,
                 fit_config_sha256, producer, require_passed=True):
    """Write a new bundle directory; ``run_manifest.json`` is replaced into place last.

    ``require_passed=False`` exists only to produce honestly-failing bundles for loader tests; load_bundle rejects them regardless.
    """
    kind, branching, depth = _spec(sid_version, engineering)
    codec = sid_v2.build_codec(ids, sids, branching)
    if depth == 2:
        _check_padding(sids)
    _hex64(content_feature_sha256, 'content digest')
    _hex64(fit_config_sha256, 'fit config digest')
    if type(producer) is not dict or not producer or any(type(k) is not str or not k for k in producer):
        raise ValueError('producer source digests required')
    for digest in producer.values():
        _hex64(digest, 'producer digest')
    if type(tokenizer) is not dict:
        raise ValueError('tokenizer config must be a dict')
    if type(audit) is not dict or set(audit) != {'mapping', 'sweep'} or type(audit['mapping']) is not dict:
        raise ValueError('audit must hold exactly mapping and sweep')
    mapping_audit = audit['mapping']
    if mapping_audit.get('kind') != kind or mapping_audit.get('branching') != branching \
            or mapping_audit.get('items') != len(ids) or mapping_audit.get('content_sha256') != content_feature_sha256:
        raise ValueError('audit does not describe this mapping/content')
    _check_sweep(audit['sweep'], depth, tokenizer)
    if type(gates) is not dict or set(gates) != {'passed', 'checks'}:
        raise ValueError('gates must hold exactly passed and checks')
    if require_passed and gates['passed'] is not True:
        raise ValueError('refusing to publish a bundle whose gates did not pass')
    files = {'tokenizer.json': canonical_bytes({'sid_version': sid_version, 'kind': kind, 'branching': branching, 'config': tokenizer}),
             'audit.json': canonical_bytes(audit)}
    output = Path(output).absolute()
    gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory inside an existing one')
    output.mkdir()
    sid_v2._write_npz(output / 'mapping.npz', {'ids': ids, 'sids': sids})
    for name, data in files.items():
        (output / name).write_bytes(data)
    manifest = {
        'schema': SCHEMA, 'protocol': PROTOCOL, 'task_id': TASK_ID, 'status': 'complete', 'sid_version': sid_version,
        'kind': kind, 'engineering': engineering, 'branching': branching, 'depth': depth, 'catalog_items': int(len(ids)),
        'suffix_vocabulary': int(codec.catalog.suffix_vocabulary), 'vocab_size': int(codec.vocab_size),
        'content_feature_sha256': content_feature_sha256, 'catalog_sha256': sid_v2.catalog_digest(ids),
        'artifacts': {name: file_sha256(output / name) for name in ARTIFACTS},
        'fit_config_sha256': fit_config_sha256, 'producer': dict(producer), 'gates': gates}
    temporary = output / (MANIFEST_NAME + '.tmp')
    temporary.write_bytes(canonical_bytes(manifest))
    os.replace(temporary, output / MANIFEST_NAME)
    return manifest


def _check_audit(manifest, stored, branching, depth, ids, sids, content, tokenizer_config):
    if type(stored) is not dict or set(stored) != {'mapping', 'sweep'}:
        raise ValueError('audit.json must hold exactly mapping and sweep')
    _check_sweep(stored['sweep'], depth, tokenizer_config)
    recomputed = sweep_audit(ids, sids, content, branching=branching, depth=depth)
    if not au.audits_equal(stored['mapping'], recomputed):
        raise ValueError('audit.json differs from the recomputed audit')
    expected = au.gates(recomputed, 'bhkm', None, expected_content_sha256=manifest['content_feature_sha256'])
    if not au.audits_equal(manifest['gates'], expected):
        raise ValueError('manifest gates differ from the recomputed gates')


def load_bundle(manifest_path, manifest_sha256, *, catalog_ids, content, expected_content_digest=CONTENT_V1_FEATURE_SHA256,
                engineering=False):
    """Fail-closed loader; returns the same ``LoadedSID`` as SIDMAP_V2 (``info`` additionally has ``depth``)."""
    try:
        return _load_bundle(manifest_path, manifest_sha256, catalog_ids, content, expected_content_digest, engineering)
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
        if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
            raise
        raise ValueError(f'malformed SIDMAP_V2S bundle: {type(exc).__name__}') from exc


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
    kind, branching, depth = _spec(manifest['sid_version'], engineering)
    if manifest['engineering'] is not engineering or manifest['kind'] != kind or type(manifest['branching']) is not int \
            or manifest['branching'] != branching or type(manifest['depth']) is not int or manifest['depth'] != depth:
        raise ValueError('manifest engineering/kind/branching/depth mismatch')
    for key in ('catalog_items', 'suffix_vocabulary', 'vocab_size'):
        if type(manifest[key]) is not int:
            raise ValueError(f'manifest {key} must be an int')
    sid_v2._check_gates_passed(manifest['gates'])
    sid_v2._check_content(manifest, content, expected_content_digest)
    directory = path.parent
    artifacts = manifest['artifacts']
    if type(artifacts) is not dict or set(artifacts) != set(ARTIFACTS) or set(os.listdir(directory)) != set(ARTIFACTS) | {MANIFEST_NAME}:
        raise ValueError('artifact set mismatch')
    for name in ARTIFACTS:
        sid_v2._verify_artifact(directory, name, artifacts[name])
    mapping = sid_v2._load_npz(directory / 'mapping.npz', {'ids', 'sids'})
    ids, sids = mapping['ids'], mapping['sids']
    sid_v2._check_catalog(manifest, catalog_ids, ids)
    if len(content) != len(ids):
        raise ValueError('content rows do not match the catalog')
    codec = sid_v2.build_codec(ids, sids, branching)
    if manifest['suffix_vocabulary'] != codec.catalog.suffix_vocabulary or manifest['vocab_size'] != codec.vocab_size:
        raise ValueError('manifest vocabulary does not match the mapping')
    if depth == 2:
        _check_padding(sids)
    tokenizer = gd._json(directory / 'tokenizer.json')
    if type(tokenizer) is not dict or set(tokenizer) != TOKENIZER_KEYS or type(tokenizer['config']) is not dict \
            or (tokenizer['sid_version'], tokenizer['kind'], tokenizer['branching']) != (manifest['sid_version'], kind, branching):
        raise ValueError('tokenizer.json does not match the manifest')
    _check_audit(manifest, gd._json(directory / 'audit.json'), branching, depth, ids, sids, content, tokenizer['config'])
    info = {'sid_version': manifest['sid_version'], 'manifest_sha256': manifest_sha256, 'branching': branching, 'depth': depth,
            'suffix_vocabulary': codec.catalog.suffix_vocabulary, 'vocab_size': codec.vocab_size, 'kind': kind,
            'engineering': engineering}
    return sid_v2.LoadedSID(codec, manifest, info)
