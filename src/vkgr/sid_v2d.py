"""SIDMAP_V2D: the G6-PREP semantic-ID bundle contract for ``depth`` semantic digits (default 4), sibling of SIDMAP_V2.

SIDMAP_V2 is pinned to three semantic digits (manifest key set, version whitelist, ``gd._check_codec``). This contract is a
new schema (``sidmap-v2d``): the SIDMAP_V2 manifest keys plus ``depth``, its own version registry, and loaders that refuse
each other's bundles. Files and publishing order are those of SIDMAP_V2 (``run_manifest.json`` is replaced into place last).
``load_bundle_d`` is fail-closed: every pinned hash, the catalog and content identity, the depth-aware codec rules
(``sid_depth.check_codec_v2``), the recomputed audit and gates (including B2 for balanced k-means) and the weights-to-mapping
export are re-verified. Helpers that do not depend on the depth are imported from ``sid_v2``; nothing there is edited.
"""
import os
from pathlib import Path

import numpy as np
import torch

from . import generative_data as gd
from . import sid_depth as sd
from . import sid_rqvae as rq
from . import sid_rqvae_d as rd
from . import sid_tok_audit as au
from . import sid_tok_audit_d as ad
from . import sid_v2
from .data_audit import canonical_bytes, file_sha256
from .sid import SIDCatalog

SCHEMA = 'sidmap-v2d'
PROTOCOL = 'generative-like-sid-v2d'
TASK_ID = 'G-HP-01-G6'
SID_VERSIONS = ('SID_BHKM_B16D4_S42', 'SID_RQVAE_B256D4_S42')
ENGINEERING_SID_VERSIONS = SID_VERSIONS + ('SID_V2D_REPLAY_BHKM_B16D3',)
VERSION_SPEC = {'SID_BHKM_B16D4_S42': ('bhkm', 16, 4), 'SID_RQVAE_B256D4_S42': ('rqvae', 256, 4),
                'SID_V2D_REPLAY_BHKM_B16D3': ('bhkm', 16, 3)}
CONTENT_V1_FEATURE_SHA256 = au.CONTENT_V1_FEATURE_SHA256
MANIFEST_NAME = sid_v2.MANIFEST_NAME
ARTIFACTS = sid_v2.ARTIFACTS
MANIFEST_KEYS = frozenset(sid_v2.MANIFEST_KEYS | {'depth'})
TOKENIZER_KEYS = frozenset({'sid_version', 'kind', 'branching', 'depth', 'config'})
LoadedSID = sid_v2.LoadedSID


def _spec_d(sid_version, engineering):
    """(kind, branching, depth) of a registered version; the replay label is accepted in engineering contexts only."""
    if type(engineering) is not bool:
        raise ValueError('engineering flag must be a bool')
    allowed = ENGINEERING_SID_VERSIONS if engineering else SID_VERSIONS
    if type(sid_version) is not str or sid_version not in allowed:
        raise ValueError('unregistered sid_version for this context')
    return VERSION_SPEC[sid_version]


def prereg_for(sid_version, n, engineering):
    """Gate B2 target. Formal balanced k-means B16D4 is judged against the frozen literal block (so it only passes at the
    registered catalog size); engineering mappings against the block the capacity rule gives for their own size."""
    kind, branching, depth = _spec_d(sid_version, engineering)
    if kind != 'bhkm':
        return None
    if not engineering and sid_version == 'SID_BHKM_B16D4_S42':
        return ad.PREREG_BHKM_B16D4
    return ad.prereg_block(n, branching, depth)


def build_codec_d(ids, sids, branching, depth, catalog_ids=None):
    """Depth-aware validated codec for an arbitrary (unbalanced) mapping; the catalog keeps nodes=[] so that pickles stay small."""
    if not isinstance(ids, np.ndarray) or not isinstance(sids, np.ndarray) or ids.dtype != np.uint32 \
            or sids.dtype != np.uint32 or ids.ndim != 1 or not len(ids) or sids.ndim != 2 or sids.shape != (len(ids), depth + 1):
        raise ValueError('uint32 ids [N] and sids [N, depth+1] required')
    if type(branching) is not int or branching < 2:
        raise ValueError('branching must be an int >= 2')
    codec = sd.ValidatedCodecD(SIDCatalog(ids, sids, branching, depth, []), depth)
    sd.check_codec_v2(codec, depth, {'catalog': ids if catalog_ids is None else np.asarray(catalog_ids)})
    return codec


def _check_state_arrays_d(arrays, branching, depth):
    """Validate {'K', tensors...} against a freshly built RQVAED(branching, depth); returns {'K', 'state_dict'}."""
    k = arrays.get('K')
    if k is None or k.shape != () or k.dtype != np.int64 or int(k) != branching:
        raise ValueError('weights K must equal the branching')
    expected = rd.RQVAED(branching, depth).state_dict()
    if set(arrays) - {'K'} != set(expected):
        raise ValueError('weights tensor names mismatch')
    state_dict = {}
    for name, reference in expected.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != tuple(reference.shape) or not np.isfinite(value).all():
            raise ValueError('weights tensor dtype/shape/finiteness mismatch')
        state_dict[name] = torch.from_numpy(np.array(value, dtype=np.float32))
    return {'K': branching, 'state_dict': state_dict}


def _weights_arrays_d(state, branching, depth):
    if type(state) is not dict or set(state) != {'K', 'state_dict'} or state['K'] != branching or type(state['state_dict']) is not dict:
        raise ValueError('invalid weights state')
    arrays = {'K': np.array(branching, dtype=np.int64)}
    for name in rd.RQVAED(branching, depth).state_dict():
        if name not in state['state_dict']:
            raise ValueError('weights tensor names mismatch')
        arrays[name] = state['state_dict'][name].detach().cpu().numpy().astype(np.float32)
    _check_state_arrays_d(arrays, branching, depth)
    return arrays


def write_bundle_d(output, *, sid_version, engineering, ids, sids, content_feature_sha256, tokenizer, audit, gates,
                   fit_config_sha256, producer, weights=None, train_log=None, require_passed=True):
    """Write a new bundle directory; ``run_manifest.json`` is replaced into place last.

    ``require_passed=False`` exists only to produce honestly-failing bundles for loader tests; load_bundle_d rejects them regardless.
    """
    kind, branching, depth = _spec_d(sid_version, engineering)
    codec = build_codec_d(ids, sids, branching, depth)
    sid_v2._hex64(content_feature_sha256, 'content digest')
    sid_v2._hex64(fit_config_sha256, 'fit config digest')
    if type(producer) is not dict or not producer or any(type(k) is not str or not k for k in producer):
        raise ValueError('producer source digests required')
    for digest in producer.values():
        sid_v2._hex64(digest, 'producer digest')
    if type(tokenizer) is not dict:
        raise ValueError('tokenizer config must be a dict')
    if type(audit) is not dict or set(audit) != {'mapping', 'rq'} or type(audit['mapping']) is not dict:
        raise ValueError('audit must hold exactly mapping and rq')
    mapping_audit = audit['mapping']
    if mapping_audit.get('kind') != kind or mapping_audit.get('branching') != branching or mapping_audit.get('items') != len(ids) \
            or mapping_audit.get('content_sha256') != content_feature_sha256 or len(mapping_audit.get('positions', ())) != depth + 1:
        raise ValueError('audit does not describe this mapping/content')
    if (kind == 'rqvae') != (audit['rq'] is not None) or (kind == 'rqvae') != (weights is not None) \
            or (kind == 'rqvae') != (train_log is not None):
        raise ValueError('rqvae bundles need rq info, weights and train log; bhkm bundles none of them')
    if type(gates) is not dict or set(gates) != {'passed', 'checks'}:
        raise ValueError('gates must hold exactly passed and checks')
    if require_passed and gates['passed'] is not True:
        raise ValueError('refusing to publish a bundle whose gates did not pass')
    files = {'mapping.npz': None,
             'tokenizer.json': canonical_bytes({'sid_version': sid_version, 'kind': kind, 'branching': branching, 'depth': depth,
                                                'config': tokenizer}),
             'audit.json': canonical_bytes(audit)}
    if kind == 'rqvae':
        if type(train_log) is not dict:
            raise ValueError('train log must be a dict')
        arrays = _weights_arrays_d(weights, branching, depth)
        files['train-log.json'] = canonical_bytes(train_log)
    output = Path(output).absolute()
    gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output must be a new directory inside an existing one')
    output.mkdir()
    sid_v2._write_npz(output / 'mapping.npz', {'ids': ids, 'sids': sids})
    for name, data in files.items():
        if data is not None:
            (output / name).write_bytes(data)
    if kind == 'rqvae':
        sid_v2._write_npz(output / 'weights.npz', arrays)
    manifest = {
        'schema': SCHEMA, 'protocol': PROTOCOL, 'task_id': TASK_ID, 'status': 'complete', 'sid_version': sid_version,
        'kind': kind, 'engineering': engineering, 'branching': branching, 'depth': depth, 'catalog_items': int(len(ids)),
        'suffix_vocabulary': int(codec.catalog.suffix_vocabulary), 'vocab_size': int(codec.vocab_size),
        'content_feature_sha256': content_feature_sha256, 'catalog_sha256': sid_v2.catalog_digest(ids),
        'artifacts': {name: file_sha256(output / name) for name in ARTIFACTS[kind]},
        'fit_config_sha256': fit_config_sha256, 'producer': dict(producer), 'gates': gates}
    temporary = output / (MANIFEST_NAME + '.tmp')
    temporary.write_bytes(canonical_bytes(manifest))
    os.replace(temporary, output / MANIFEST_NAME)
    return manifest


def _check_weights_d(state, ids, sids, content):
    if not np.array_equal(sd.assign_suffix(ids, rd.export_codes_d(state, content)), sids):
        raise ValueError('weights do not reproduce the mapping')


def _check_audit_d(manifest, stored, kind, branching, depth, ids, sids, content, state, engineering):
    if type(stored) is not dict or set(stored) != {'mapping', 'rq'}:
        raise ValueError('audit.json must hold exactly mapping and rq')
    recomputed = ad.audit_mapping_d(ids, sids, content, branching=branching, kind=kind, depth=depth)
    if not au.audits_equal(stored['mapping'], recomputed):
        raise ValueError('audit.json differs from the recomputed audit')
    info, required = None, au.REQUIRED_STEPS
    if kind == 'rqvae':
        info = stored['rq']
        if type(info) is not dict or set(info) != sid_v2.RQ_INFO_KEYS:
            raise ValueError('audit.json rq section schema mismatch')
        if not au.audits_equal(info['reconstruction'], rd.reconstruction_report_d(state, content)):
            raise ValueError('reconstruction report differs from the recomputation')
        if not engineering and (info['required_steps'] != rq.PREREG['max_steps']
                                or info['wall_cap_seconds'] != rq.PREREG['wall_cap_seconds']):
            raise ValueError('formal bundles must carry the pre-registered training length and wall cap')
        required = info['required_steps']
    elif stored['rq'] is not None:
        raise ValueError('bhkm bundles carry no rq section')
    expected = ad.gates_d(recomputed, kind, info, depth=depth, expected_content_sha256=manifest['content_feature_sha256'],
                          required_steps=required, prereg=prereg_for(manifest['sid_version'], len(ids), engineering))
    if not au.audits_equal(manifest['gates'], expected):
        raise ValueError('manifest gates differ from the recomputed gates')


def load_bundle_d(manifest_path, manifest_sha256, *, catalog_ids, content, expected_content_digest=CONTENT_V1_FEATURE_SHA256,
                  engineering=False):
    """Fail-closed loader. ``content`` is the float32 [N, 64] item rows (needed to recompute audit and export)."""
    try:
        return _load_bundle_d(manifest_path, manifest_sha256, catalog_ids, content, expected_content_digest, engineering)
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
        if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
            raise
        raise ValueError(f'malformed SIDMAP_V2D bundle: {type(exc).__name__}') from exc


def _load_bundle_d(manifest_path, manifest_sha256, catalog_ids, content, expected_content_digest, engineering):
    path = gd._checked_file(manifest_path, manifest_sha256)
    if path.name != MANIFEST_NAME:
        raise ValueError('fixed run_manifest.json role required')
    manifest = gd._json(path)
    if type(manifest) is not dict or set(manifest) != MANIFEST_KEYS:
        raise ValueError('manifest key set mismatch')
    for key, value in (('schema', SCHEMA), ('protocol', PROTOCOL), ('task_id', TASK_ID), ('status', 'complete')):
        if manifest[key] != value:
            raise ValueError(f'manifest {key} mismatch')
    kind, branching, depth = _spec_d(manifest['sid_version'], engineering)
    if manifest['engineering'] is not engineering or manifest['kind'] != kind or type(manifest['branching']) is not int \
            or manifest['branching'] != branching or type(manifest['depth']) is not int or manifest['depth'] != depth:
        raise ValueError('manifest engineering/kind/branching/depth mismatch')
    for key in ('catalog_items', 'suffix_vocabulary', 'vocab_size'):
        if type(manifest[key]) is not int:
            raise ValueError(f'manifest {key} must be an int')
    sid_v2._check_gates_passed(manifest['gates'])
    sid_v2._check_content(manifest, content, expected_content_digest)
    directory, names = path.parent, ARTIFACTS[kind]
    artifacts = manifest['artifacts']
    if type(artifacts) is not dict or set(artifacts) != set(names) or set(os.listdir(directory)) != set(names) | {MANIFEST_NAME}:
        raise ValueError('artifact set mismatch')
    for name in names:
        sid_v2._verify_artifact(directory, name, artifacts[name])
    mapping = sid_v2._load_npz(directory / 'mapping.npz', {'ids', 'sids'})
    ids, sids = mapping['ids'], mapping['sids']
    sid_v2._check_catalog(manifest, catalog_ids, ids)
    if len(content) != len(ids):
        raise ValueError('content rows do not match the catalog')
    codec = build_codec_d(ids, sids, branching, depth)
    if manifest['suffix_vocabulary'] != codec.catalog.suffix_vocabulary or manifest['vocab_size'] != codec.vocab_size:
        raise ValueError('manifest vocabulary does not match the mapping')
    tokenizer = gd._json(directory / 'tokenizer.json')
    if type(tokenizer) is not dict or set(tokenizer) != TOKENIZER_KEYS or type(tokenizer['config']) is not dict \
            or (tokenizer['sid_version'], tokenizer['kind'], tokenizer['branching'], tokenizer['depth']) \
            != (manifest['sid_version'], kind, branching, depth):
        raise ValueError('tokenizer.json does not match the manifest')
    state = None
    if kind == 'rqvae':
        if type(gd._json(directory / 'train-log.json')) is not dict:
            raise ValueError('train-log.json must be an object')
        state = _check_state_arrays_d(sid_v2._load_npz(directory / 'weights.npz'), branching, depth)
        _check_weights_d(state, ids, sids, content)
    _check_audit_d(manifest, gd._json(directory / 'audit.json'), kind, branching, depth, ids, sids, content, state, engineering)
    info = {'sid_version': manifest['sid_version'], 'manifest_sha256': manifest_sha256, 'branching': branching, 'depth': depth,
            'suffix_vocabulary': codec.catalog.suffix_vocabulary, 'vocab_size': codec.vocab_size, 'kind': kind,
            'engineering': engineering}
    return LoadedSID(codec, manifest, info)
