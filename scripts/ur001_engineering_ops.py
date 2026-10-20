"""Bounded engineering transport for the approved ur0.01 download/audit and anchor profiles.

Requests are JSON on stdin. The module never accepts shell snippets or arbitrary commands.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_ENVELOPE_BYTES = 48 * 1024
MAX_WALL_HOURS = 48
MAX_VRAM_MIB = 39168
P3C_PATH = 'docs/p3c_validation.md'
P3C_OPS_COPY = 'p3c_validation.md'
P3C_GIT_SHA256 = '5449378c3e2c00d5af4102da73e93a28839898107c139df110f29d50c321c466'
P3C_DATA_SHA256 = 'c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
A_DATA_ROOT_PARTS = ('/', 'workspace', 'vkgr-ur001-data-20261020-v1')
A_OLD_ROOT_PARTS = ('/', 'workspace', 'vkgr-p2-20261006t1755', 'data', 'raw', 'vk-lsvd-084a183')
PROFILE_ROOTS = {'g': ('/', 'workspace', 'vkgr-sid-structure-a1-v1'),
                 'p': ('/', 'workspace', 'vkgr-seqm1-p-v1')}
PROFILE_DATA_ROOTS = {'g': '/workspace/vkgr-b1-g', 'p': None}
PROFILE_SOURCE_LEDGER_PINS = {
    'g': {'identity_sha256': '54e96df0492f70dd06d98bde7e662d6f7402e097d5cebf8c57d39ae578e0816c',
          'map_sha256': '9b9c168ee455ac2cc168b750ae832668c9bce0fa60ed2cb553d2e45eab8d8c85', 'paths': 41},
    'p': {'identity_sha256': '4f9aa8c0e66856aebaf1ce304a9d28ec58a9078db30628cace295676114243ed',
          'map_sha256': 'c91833d57b1f8436c60fbbd2e4ecbe617757a4a4ebd7a825865d329b1c968429', 'paths': 51},
}
LEGACY_A_PF = {
    'run_id': 'ur001-download-preflight-v1',
    'source_sha': '1b560a9faa2b1b2621157785313d58ca3a828bc0',
    'helper_sha256': 'c6441c3412236902983fe6e613992a9a215201ec37e666d37e0fa2d4243445e8',
    'metadata_sha256': '9dd4c863e16e9a203c571638beebc11bbf5ddf00cf77f1a498b4e3e9006f2bad',
    'intent_sha256': '5c4e33e11be910531c36bf26c55cb67819a28c9f14b4ee151c42136362c64b69',
    'job_sha256': '54a3d17ee268ae2086fec050825dff114f9d26ae73ec3a9be64cf0bf432266a6',
    'scheduler_commit': '5840e6b686ec9651314a9e28a1a2c0befcc05158',
    'stdout_sha256': 'aefe93b8483a4ecd8a05c999fc82c67834f0c2a0604a19edae173c59db531096',
    'stdout_bytes': 5143,
    'source_files': {
        'analysis/ur001_prepare.py': '86e3ea98142b762c844d0deb66b2cf3d22f239df591e31f8f5d5f7d57263f94b',
        'configs/ur001-preparation/hf-files.json': '19cd71f02aff4fce93349907dbc5fcdbf006afc679ee899d9c98965123270fa0',
        'scripts/ur001_engineering_ops.py': 'c6441c3412236902983fe6e613992a9a215201ec37e666d37e0fa2d4243445e8',
        'uv.lock': '1a778881fd23cf8d3b4215e66ac7f118427783767a043d6ae1a0cfa2c7ce1420',
    },
}
PROFILE_COMMITS = {'g': 'b5a4d1c99ba2d8361a2d90c23bac27af5b71769d',
                   'p': '0ca61bc35ad68bd79b86103b54e470445cbab07f'}
PROFILE_CONFIGS = {
    'g': ('configs/sid_structure/g-structure-a1-s42.json', '9438c9574e687496e42d3ac2453c649fd7d94adc9e51371e4a14aa96d92c69d5'),
    'p': ('configs/seqm1/p-seqm1-mean-s42.json', 'e88dc757ff4085af020efc1f070ebba8df54876e85adf68120dff730f983b621'),
}
PROFILE_INPUTS = {
    'g': {
        'prepared': ('vkgr-infra01-root/runs/g-prepare-5983cf6-01/prepared/manifest.json', 'a55df0a3b285ce4cf623126fe53df53378bf5d7c0645c6484f0f405641bddd82'),
        'history_manifest': ('vkgr-infra01-root/runs/h2-prepare/out/manifest.json', '8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43'),
        'sid_manifest': ('vkgr-infra01-root/runs/sid-structure-fit-v1/out/run_manifest.json', '67b1f404e7b97a96ffc2a6c303906fa62e3f48234d5f55c771bb8480741fc3d4'),
        'content': ('vkgr-p4b-20261007t-pipeline-03/prepared-v3/content.npy', '644b2d7a656999a8962355d7c32dae07134ad46e9557b4a8418f137d62f73581'),
        'input_manifest': ('vkgr-infra01-root/runs/sid-coll-c1-input_verify-v1/out/input-manifest.json', '725fb211a342c518770623be81d490feaaec38a285c954b5b6f51f50a388e9a4'),
        'vectors': ('vkgr-infra01-root/runs/sid-coll-c1-input-v1/out/payload/vectors.npy', 'bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7'),
    },
    'p': {'inputs': ('vkgr-b1-inputs/p-feat02-inputs-base-v1.json', 'a12b98003aa2fbb58df83bc70bb448e426d01c2a4cc2e6357bf1f95778f23e38')},
}
JOB_ENV = ('OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8')
HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')

# Values are (script, subcommand). A run id cannot select a different stage.
RUNS = {
    'ur001-download-preflight-v1': ('analysis/ur001_prepare.py', 'preflight'),
    'ur001-download-v1': ('analysis/ur001_prepare.py', 'download'),
    'ur001-audit-preflight-v1': ('analysis/ur001_prepare.py', 'audit-preflight'),
    'ur001-audit-v1': ('analysis/ur001_prepare.py', 'audit'),
    'anchor-profile-g-preflight-v1': ('analysis/profile_anchor_steps.py', 'preflight'),
    'anchor-profile-g-preflight-v2': ('analysis/profile_anchor_steps.py', 'preflight'),
    'anchor-profile-g-v1': ('analysis/profile_anchor_steps.py', 'profile'),
    'anchor-profile-p-preflight-v1': ('analysis/profile_anchor_steps.py', 'preflight'),
    'anchor-profile-p-preflight-v2': ('analysis/profile_anchor_steps.py', 'preflight'),
    'anchor-profile-p-v1': ('analysis/profile_anchor_steps.py', 'profile'),
}
def _expected_seed(run_id):
    return 42 if type(run_id) is str and run_id.startswith('anchor-profile-') else None


TAKEOVER_RUN_IDS = tuple(
    [f'p-watch-positive-s{s}{suffix}' for s in ('42', '2024', '2025') for suffix in ('', '-verify')]
    + [f'g-half-users-r2-s{s}{suffix}' for s in ('42', '2024', '2025') for suffix in ('', '-verify')]
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    return digest_bytes(Path(path).read_bytes())


def safe_path(value, role, must_exist=False):
    if type(value) is not str or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError(f'{role} must be an absolute non-traversing path')
    path = Path(value)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'{role} may not traverse symlinks')
    if must_exist and not path.exists():
        raise ValueError(f'{role} does not exist')
    return path


def _git(repo, *args):
    result = subprocess.run(['git', '-C', str(repo), *args], stdin=subprocess.DEVNULL,
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f'git {args[0]} failed: {result.stderr.decode(errors="replace")[:1000]}')
    return result.stdout


def _tracked_tree(repo, ref='HEAD'):
    raw = _git(repo, 'ls-tree', '-rz', '--full-tree', ref)
    blobs = []
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        meta, rel = entry.split(b'\t', 1)
        mode, kind, blob = meta.decode('ascii').split()
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError('deployment tree may contain only regular tracked files')
        blobs.append((rel.decode('utf-8'), blob))
    if not blobs:
        return {}
    result = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'],
                            input=('\n'.join(blob for _, blob in blobs) + '\n').encode(),
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError('git cat-file batch failed')
    content, offset, files = result.stdout, 0, {}
    for name, _ in blobs:
        newline = content.find(b'\n', offset)
        if newline < 0:
            raise ValueError('truncated git cat-file batch header')
        header = content[offset:newline].split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('unexpected git cat-file batch object')
        size = int(header[2])
        start, end = newline + 1, newline + 1 + size
        if end >= len(content) or content[end:end + 1] != b'\n':
            raise ValueError('truncated git cat-file batch body')
        files[name] = digest_bytes(content[start:end])
        offset = end + 1
    if offset != len(content):
        raise ValueError('unexpected trailing git cat-file output')
    return files


def _parts_match(path, parts):
    return tuple(Path(path).parts) == tuple(parts)


def _copy_p3c_to_ops(ops_root, source):
    target = Path(ops_root) / P3C_OPS_COPY
    if target.exists() or target.is_symlink():
        raise FileExistsError('fixed P3C OPS copy already exists; never overwrite it')
    source = safe_path(source, 'P3C data-channel source', True)
    raw = source.read_bytes()
    if digest_bytes(raw) != P3C_DATA_SHA256:
        raise ValueError('P3C data-channel bytes differ from the approved fixed SHA-256')
    with target.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if digest_file(target) != P3C_DATA_SHA256:
        raise ValueError('new private P3C OPS copy failed fixed SHA-256 verification')
    return target


def _deployment(req):
    required = {'action', 'source_repo', 'branch', 'expected_sha', 'deployment', 'files'}
    allowed = required | {'ops', 'p3c_data_source'}
    if set(req) - allowed or not required <= set(req):
        raise ValueError('deploy request accepts fixed source, branch, SHA, deployment and exact file pins')
    repo = safe_path(req['source_repo'], 'source_repo', True)
    dest = safe_path(req['deployment'], 'deployment')
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('deployment path already exists; existing trees are never replaced')
    branch, expected = req['branch'], req['expected_sha']
    if type(branch) is not str or not re.fullmatch(r'[A-Za-z0-9._/-]{1,200}', branch) or '..' in branch.split('/'):
        raise ValueError('fixed branch name required')
    if type(expected) is not str or not HEX40.fullmatch(expected):
        raise ValueError('expected_sha must be a full lowercase Git SHA')
    files = req['files']
    if type(files) is not dict or any(type(k) is not str or type(v) is not str or not HEX64.fullmatch(v) for k, v in files.items()):
        raise ValueError('files must map relative paths to SHA-256')
    remote = _git(repo, 'remote', 'get-url', 'origin').decode().strip()
    if not remote:
        raise ValueError('origin remote required')
    _git(repo, 'fetch', 'origin', branch)
    fetched = _git(repo, 'rev-parse', 'FETCH_HEAD').decode().strip()
    if fetched != expected:
        raise ValueError('fetched source SHA differs from expected_sha')
    source_files = _tracked_tree(repo, expected)
    if set(source_files) != set(files):
        raise ValueError('requested exact deployment path set differs from pinned Git tree')
    override = None
    if P3C_PATH in source_files:
        if source_files[P3C_PATH] != P3C_GIT_SHA256:
            raise ValueError('deployment Git P3C base is not the approved P3C redacted SHA')
        target_sha = files[P3C_PATH]
        if target_sha == P3C_DATA_SHA256:
            if 'ops' not in req or 'p3c_data_source' not in req:
                raise ValueError('approved P3C overlay requires a new private OPS copy destination and source')
            ops_root = safe_path(req['ops'], 'ops', True)
            _copy_p3c_to_ops(ops_root, req['p3c_data_source'])
            override = {'base_sha256': P3C_GIT_SHA256, 'sha256': P3C_DATA_SHA256,
                        'reason': 'fixed INFRA-01 D0 P3C data-channel copy'}
        elif target_sha != P3C_GIT_SHA256:
            raise ValueError('P3C deployment SHA is not the approved data-channel or Git redacted version')
    elif P3C_PATH in files:
        raise ValueError('P3C path was not tracked by the fetched source commit')
    if 'p3c_data_source' in req and override is None:
        raise ValueError('P3C data source supplied without the exact approved overlay pin')
    for rel, sha in source_files.items():
        if rel == P3C_PATH and override is not None:
            continue
        if files[rel] != sha:
            raise ValueError(f'non-overlay pin differs from Git blob: {rel}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, 'worktree', 'add', '--detach', str(dest), expected)
    actual = {}
    for rel in sorted(files):
        path = dest / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'deployed path missing or unsafe: {rel}')
        actual[rel] = digest_file(path)
        if actual[rel] != source_files[rel]:
            raise ValueError(f'worktree bytes differ from expected Git tree: {rel}')
    if override is not None:
        target = dest / P3C_PATH
        private_copy = Path(req['ops']) / P3C_OPS_COPY
        raw = private_copy.read_bytes()
        if digest_bytes(raw) != P3C_DATA_SHA256:
            raise ValueError('private OPS P3C copy changed before deployment overlay')
        with target.open('wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        actual[P3C_PATH] = digest_file(target)
    if actual != files:
        raise ValueError('deployed exact file set or SHA differs after the fixed P3C overlay')
    overrides = {P3C_PATH: override} if override is not None else {}
    return {'status': 'deployed', 'deployment': str(dest), 'commit': expected,
            'source_paths': len(actual), 'source_tree_sha256': digest_bytes(canonical(actual)),
            'source_files': actual, 'source_overrides': overrides,
            'private_p3c_sha256': P3C_DATA_SHA256 if override is not None else None}





def _workspace_path(relative):
    return str(Path('/workspace') / relative)


def _read_private_json(path):
    raw = Path(path).read_bytes()
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError('private request exceeds bounded size')
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError('private request must be a JSON object')
    return value


def _parse_a_request(path):
    value = _read_private_json(path)
    if set(value) != {'schema', 'data_root', 'old_subset_root'} or value.get('schema') != 'ur001-engineering-request-v1':
        raise ValueError('A private request must have exactly the frozen schema and two roots')
    data_root = safe_path(value['data_root'], 'A data_root')
    old_root = safe_path(value['old_subset_root'], 'A old_subset_root')
    if tuple(data_root.parts) != A_DATA_ROOT_PARTS or tuple(old_root.parts) != A_OLD_ROOT_PARTS:
        raise ValueError('A data roots differ from the fixed ur0.01 and existing subset roots')
    return {'data_root': str(data_root), 'old_subset_root': str(old_root), 'schema': value['schema']}


def _profile_request(path, run_id, ops_root, root):
    value = _read_private_json(path)
    common = {'schema', 'branch', 'data_root', 'source_root', 'expected_source_commit', 'source_ledger',
              'config', 'scratch_root', 'output_root', 'inputs'}
    branch = 'g' if run_id.startswith('anchor-profile-g-') else 'p'
    if set(value) != common or value.get('schema') != 'anchor-profile-request-v1' or value.get('branch') != branch:
        raise ValueError('B private request does not match its fixed profile branch schema')
    if value.get('data_root') != PROFILE_DATA_ROOTS[branch]:
        raise ValueError('profile data_root differs from the fixed branch data channel')
    source_root = safe_path(value['source_root'], 'profile source_root')
    if tuple(source_root.parts) != PROFILE_ROOTS[branch]:
        raise ValueError('profile source_root differs from the registered anchor source root')
    if value.get('expected_source_commit') != PROFILE_COMMITS[branch]:
        raise ValueError('profile source commit differs from its registered anchor commit')
    cfg = value['config']
    if type(cfg) is not dict or set(cfg) != {'path', 'sha256'}:
        raise ValueError('profile config must have exactly path and sha256')
    config_rel, config_sha = PROFILE_CONFIGS[branch]
    expected_config = source_root / config_rel
    if cfg['path'] != config_rel or cfg['sha256'] != config_sha:
        raise ValueError('profile model config path or SHA differs from its fixed anchor')
    roles = PROFILE_INPUTS[branch]
    pins = value['inputs']
    if type(pins) is not dict or set(pins) != set(roles):
        raise ValueError('profile request contains an unregistered input role or omits a fixed role')
    role_paths, role_map = [], {}
    for role, (relative, expected_sha) in roles.items():
        item = pins[role]
        expected_path = _workspace_path(relative)
        if type(item) is not dict or set(item) != {'path', 'sha256'}:
            raise ValueError(f'profile input role schema invalid: {role}')
        if item['path'] != expected_path or item['sha256'] != expected_sha:
            raise ValueError(f'profile input role path/SHA differs from its registered pin: {role}')
        role_path = safe_path(item['path'], 'profile input role ' + role)
        if any(part.lower() in ('test', 'tests', 'ckpt', 'checkpoint', 'checkpoints') for part in role_path.parts):
            raise ValueError(f'profile input role may not reference test/checkpoint data: {role}')
        role_paths.append(str(role_path))
        role_map[role] = {'path': str(role_path), 'sha256': expected_sha}
    ledger = value['source_ledger']
    if type(ledger) is not dict or set(ledger) != {'sha256', 'files'} or type(ledger['files']) is not dict or not ledger['files']:
        raise ValueError('profile source ledger required')
    ledger_sha = ledger['sha256']
    if type(ledger_sha) is not str or not HEX64.fullmatch(ledger_sha):
        raise ValueError('profile source ledger SHA must be lowercase SHA-256')
    ledger_pin = PROFILE_SOURCE_LEDGER_PINS[branch]
    ledger_map_sha = digest_bytes(canonical(ledger['files']))
    if (ledger_sha != ledger_pin['identity_sha256'] or ledger_map_sha != ledger_pin['map_sha256']
            or len(ledger['files']) != ledger_pin['paths']):
        raise ValueError('profile source ledger differs from its frozen identity or complete path map')
    ledger_paths = []
    for rel, sha in sorted(ledger['files'].items()):
        member = Path(rel)
        if (type(rel) is not str or member.is_absolute() or '..' in member.parts or '.' in member.parts
                or member.as_posix() != rel or type(sha) is not str or not HEX64.fullmatch(sha)):
            raise ValueError('profile source ledger path or SHA invalid')
        ledger_paths.append(str(source_root / rel))
    scratch = safe_path(value['scratch_root'], 'profile scratch_root')
    output = safe_path(value['output_root'], 'profile output_root')
    run_dir = Path(root) / 'runs' / run_id
    if scratch != run_dir / 'scratch' or output != run_dir / 'out':
        raise ValueError('profile scratch/output roots must be the fresh per-run scratch and out directories')
    model_config = safe_path(str(expected_config), 'profile model config')
    extra_inputs = [str(model_config), *ledger_paths, *role_paths]
    if branch == 'p':
        extra_inputs.append(str(source_root / P3C_PATH))
    return {'branch': branch, 'data_root': value['data_root'], 'source_root': str(source_root),
            'expected_source_commit': PROFILE_COMMITS[branch],
            'source_ledger': ledger, 'source_ledger_sha256': ledger_sha,
            'source_ledger_json_sha256': ledger_map_sha, 'model_config': str(model_config),
            'model_config_sha256': config_sha, 'registered_inputs': role_map,
            'extra_inputs': list(dict.fromkeys(extra_inputs)), 'scratch_root': str(scratch),
            'output_root': str(output), 'private_request': value}


def _verify_profile_files(profile, ops_root):
    branch, source_root = profile['branch'], Path(profile['source_root'])
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError('profile source root missing or unsafe')
    head = _git(source_root, 'rev-parse', 'HEAD').decode().strip()
    if head != profile['expected_source_commit']:
        raise ValueError('profile source root commit differs from registered anchor commit')
    dirty = [line for line in _git(source_root, 'status', '--porcelain=v1', '--untracked-files=all').decode().splitlines() if line]
    if branch == 'g':
        if dirty:
            raise ValueError('G anchor source root must be clean')
    else:
        base_p3c = _git(source_root, 'show', f'{head}:{P3C_PATH}')
        p3c_path = source_root / P3C_PATH
        dirty_paths = [line[3:] for line in dirty]
        private_copy = Path(ops_root) / P3C_OPS_COPY
        if (digest_bytes(base_p3c) != P3C_GIT_SHA256 or dirty_paths != [P3C_PATH]
                or p3c_path.is_symlink() or not p3c_path.is_file()
                or digest_file(p3c_path) != P3C_DATA_SHA256
                or private_copy.is_symlink() or not private_copy.is_file()
                or digest_file(private_copy) != P3C_DATA_SHA256):
            raise ValueError('P anchor source root must use the exact approved P3C data-channel overlay')
    config = Path(profile['model_config'])
    if config.is_symlink() or not config.is_file() or digest_file(config) != profile['model_config_sha256']:
        raise ValueError('profile model config file differs from its registered SHA')
    for rel, expected in profile['source_ledger']['files'].items():
        path = source_root / rel
        if path.is_symlink() or not path.is_file() or digest_file(path) != expected:
            raise ValueError(f'profile source ledger file differs from its SHA: {rel}')
    for role, item in profile['registered_inputs'].items():
        path = Path(item['path'])
        if path.is_symlink() or not path.is_file() or digest_file(path) != item['sha256']:
            raise ValueError(f'registered profile data input differs from its SHA: {role}')
    return {'branch': branch, 'source_root_commit': head,
            'source_ledger_sha256': digest_bytes(canonical(profile['source_ledger']['files'])),
            'registered_inputs_sha256': digest_bytes(canonical(profile['registered_inputs'])),
            'model_config_sha256': profile['model_config_sha256'],
            'p3c_data_sha256': P3C_DATA_SHA256 if branch == 'p' else None}


def _expected_command(job, deployment, config, config_sha, *, root=None, ops_root=None, require_script=True):
    run_id = job.get('run_id')
    if run_id not in RUNS:
        raise ValueError('run_id is outside the fixed ur001/profile engineering set')
    rel_script, command = RUNS[run_id]
    script = deployment / rel_script
    if require_script and (not script.is_file() or script.is_symlink()):
        raise ValueError(f'fixed job script missing or unsafe: {rel_script}')
    if run_id.startswith('ur001-'):
        request = _parse_a_request(config)
        expected = [sys.executable, '-I', '-B', str(script), command, '--root', request['data_root'],
                    '--old-subset-root', request['old_subset_root']]
        binding = {'kind': 'A', **request}
        expected_inputs = [str(script), str(config)]
    else:
        if root is None or ops_root is None:
            raise ValueError('profile command binding requires root and ops')
        profile = _profile_request(config, run_id, Path(ops_root), Path(root))
        expected = [sys.executable, '-I', '-B', str(script), command, '--request', str(config)]
        binding = {'kind': 'B', **{k: v for k, v in profile.items() if k != 'private_request'}}
        expected_inputs = [str(script), str(config), *profile['extra_inputs']]
    return rel_script, command, expected, expected_inputs, binding


def _job_paths(req, job):
    deployment = safe_path(req['deployment'], 'deployment', True)
    ops_root = safe_path(req['ops'], 'ops', True)
    config = safe_path(job.get('config'), 'private config', True)
    if not config.is_relative_to(ops_root) or config.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError('bounded private config must live under ops')
    return deployment, ops_root, config


def _deployment_binding(req):
    deployment = Path(req['deployment'])
    expected = req['source_sha']
    files, overrides = req['source_files'], req['source_overrides']
    if type(expected) is not str or not HEX40.fullmatch(expected):
        raise ValueError('source SHA must be a full lowercase Git commit')
    head = _git(deployment, 'rev-parse', 'HEAD').decode().strip()
    if head != expected:
        raise ValueError('deployment source commit differs from source_sha')
    base = _tracked_tree(deployment, expected)
    if (type(files) is not dict or set(files) != set(base)
            or any(type(v) is not str or not HEX64.fullmatch(v) for v in files.values())):
        raise ValueError('source_files must pin the exact tracked path set and SHA-256 values')
    if type(overrides) is not dict or set(overrides) - {P3C_PATH}:
        raise ValueError('only the fixed P3C data-channel path may be overlaid')
    if P3C_PATH in base:
        if base[P3C_PATH] != P3C_GIT_SHA256:
            raise ValueError('deployment P3C base differs from approved Git redacted SHA')
        if P3C_PATH in overrides:
            override = overrides[P3C_PATH]
            if (type(override) is not dict or set(override) != {'base_sha256', 'sha256', 'reason'}
                    or override['base_sha256'] != P3C_GIT_SHA256 or override['sha256'] != P3C_DATA_SHA256
                    or files[P3C_PATH] != P3C_DATA_SHA256
                    or override['reason'] != 'fixed INFRA-01 D0 P3C data-channel copy'):
                raise ValueError('P3C override differs from the approved fixed Git/data-channel SHA pair')
            ops_copy = Path(req['ops']) / P3C_OPS_COPY
            if ops_copy.is_symlink() or not ops_copy.is_file() or digest_file(ops_copy) != P3C_DATA_SHA256:
                raise ValueError('private OPS P3C copy is absent or differs from the approved data-channel SHA')
        elif files[P3C_PATH] != P3C_GIT_SHA256:
            raise ValueError('P3C path requires the approved data-channel override or exact Git SHA')
    elif overrides:
        raise ValueError('P3C overlay is not present in the pinned Git tree')
    for rel, base_sha in base.items():
        if rel == P3C_PATH and P3C_PATH in overrides:
            continue
        if files[rel] != base_sha:
            raise ValueError(f'source file differs from Git without an approved overlay: {rel}')
        path = deployment / rel
        if path.is_symlink() or not path.is_file() or digest_file(path) != files[rel]:
            raise ValueError(f'deployment source bytes differ from exact source_files pin: {rel}')
    if P3C_PATH in base:
        path = deployment / P3C_PATH
        expected_p3c = P3C_DATA_SHA256 if P3C_PATH in overrides else P3C_GIT_SHA256
        if path.is_symlink() or not path.is_file() or digest_file(path) != expected_p3c:
            raise ValueError('deployed P3C bytes differ from the approved exact SHA')
    dirty = [line for line in _git(deployment, 'status', '--porcelain=v1', '--untracked-files=all').decode().splitlines() if line]
    allowed_dirty = set(overrides)
    found_dirty = set()
    for line in dirty:
        rel = line[3:]
        if ' -> ' in rel:
            rel = rel.split(' -> ', 1)[1]
        found_dirty.add(rel)
    if found_dirty != allowed_dirty:
        raise ValueError('deployment dirty paths differ from the exact approved source_overrides set')
    return {'commit': head, 'files_sha256': digest_bytes(canonical(files)), 'paths': len(files),
            'overrides_sha256': digest_bytes(canonical(overrides)),
            'p3c_data_sha256': P3C_DATA_SHA256 if P3C_PATH in overrides else None}


def scheduler_binding(req):
    path = safe_path(req['scheduler_code'], 'scheduler_code', True)
    expected = req['scheduler_commit']
    if type(expected) is not str or not HEX40.fullmatch(expected):
        raise ValueError('scheduler_commit must be a full lowercase Git commit')
    head = _git(path, 'rev-parse', 'HEAD').decode().strip()
    if head != expected:
        raise ValueError('scheduler source commit differs from scheduler_commit')
    if _git(path, 'status', '--porcelain=v1', '--untracked-files=all').strip():
        raise ValueError('scheduler dependency checkout must be clean')
    tracked = _tracked_tree(path, expected)
    files = {rel: value for rel, value in tracked.items() if rel.startswith('ops/vkgr_ops/') and rel.endswith('.py')}
    if not files or 'ops/vkgr_ops/launcher.py' not in files or 'ops/vkgr_ops/scheduler.py' not in files:
        raise ValueError('versioned launcher and scheduler Python sources required')
    for rel, sha in files.items():
        if digest_file(path / rel) != sha:
            raise ValueError(f'scheduler dependency bytes differ from pinned commit: {rel}')
    return {'commit': head, 'files_sha256': digest_bytes(canonical(files)), 'paths': len(files)}


def _helper_environment_binding(req):
    ignored_python = sorted(key for key in os.environ if key.startswith('PYTHON'))
    for key in ('LD_PRELOAD', 'NVIDIA_TF32_OVERRIDE'):
        if os.environ.get(key):
            raise ValueError(f'helper ambient environment contains unregistered influential value: {key}')
    for key, value in os.environ.items():
        if key.startswith('VKGR_') and not (key == 'VKGR_OPS_ROOT' and value == req['root']):
            raise ValueError(f'helper ambient environment contains unregistered VKGR variable: {key}')
    cuda_value = os.environ.get('CUDA_VISIBLE_DEVICES')
    return {'python_isolated': True, 'ignored_by_isolated_python': ignored_python,
            'cuda_visible_devices_sha256': digest_bytes(cuda_value.encode()) if cuda_value is not None else None}


def _profile_source_binding(profile, ops_root):
    branch = profile['branch']
    source_root = Path(profile['source_root'])
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError('profile source root missing or unsafe')
    head = _git(source_root, 'rev-parse', 'HEAD').decode().strip()
    if head != profile['expected_source_commit']:
        raise ValueError('profile source root commit differs from registered anchor commit')
    dirty = [line for line in _git(source_root, 'status', '--porcelain=v1', '--untracked-files=all').decode().splitlines() if line]
    if branch == 'g':
        if dirty:
            raise ValueError('G anchor source root must be clean')
    else:
        base_p3c = _git(source_root, 'show', f'{head}:{P3C_PATH}')
        p3c_path = source_root / P3C_PATH
        dirty_paths = [line[3:] for line in dirty]
        private_copy = Path(ops_root) / P3C_OPS_COPY
        if (digest_bytes(base_p3c) != P3C_GIT_SHA256 or dirty_paths != [P3C_PATH]
                or p3c_path.is_symlink() or not p3c_path.is_file()
                or digest_file(p3c_path) != P3C_DATA_SHA256
                or private_copy.is_symlink() or not private_copy.is_file()
                or digest_file(private_copy) != P3C_DATA_SHA256):
            raise ValueError('P anchor source root must use the exact approved P3C data-channel overlay')
    config = Path(profile['model_config'])
    if config.is_symlink() or not config.is_file() or digest_file(config) != profile['model_config_sha256']:
        raise ValueError('profile model config file differs from its registered SHA')
    ledger_files = profile['source_ledger']['files']
    for rel, expected in ledger_files.items():
        path = source_root / rel
        if path.is_symlink() or not path.is_file() or digest_file(path) != expected:
            raise ValueError(f'profile source ledger file differs from its SHA: {rel}')
    for role, item in profile['registered_inputs'].items():
        path = Path(item['path'])
        if path.is_symlink() or not path.is_file() or digest_file(path) != item['sha256']:
            raise ValueError(f'registered profile data input differs from its SHA: {role}')
    source_files_sha = profile['source_ledger']['sha256']
    source_map_sha = digest_bytes(canonical(ledger_files))
    roles_sha = digest_bytes(canonical(profile['registered_inputs']))
    return {'branch': branch, 'source_root_commit': head, 'source_ledger_sha256': source_files_sha,
            'source_ledger_json_sha256': source_map_sha,
            'registered_inputs_sha256': roles_sha, 'model_config_sha256': profile['model_config_sha256'],
            'p3c_data_sha256': P3C_DATA_SHA256 if branch == 'p' else None}


def _fixed_request_binding(req, job, *, verify_profile=True, require_script=True):
    rel, command, expected, expected_inputs, binding = _expected_command(
        job, Path(req['deployment']), Path(job['config']), digest_file(job['config']),
        root=Path(req['root']), ops_root=Path(req['ops']), require_script=require_script)
    if job['argv'] != expected:
        raise ValueError('job argv differs from fixed script, run id, stage and CLI contract')
    if job['inputs'] != expected_inputs:
        raise ValueError('job.inputs include an unregistered input or differ from fixed request roles/source files')
    if binding['kind'] == 'B' and verify_profile:
        binding['verified_source'] = _profile_source_binding(binding, req['ops'])
    return {'script': rel, 'command': command, 'argv': expected, 'inputs': expected_inputs,
            'binding': binding}


def validate(req):
    if type(req) is not dict or type(req.get('action')) is not str:
        raise ValueError('JSON object with action required')
    action = req['action']
    if action == 'deploy':
        return req
    if action == 'inventory':
        _validate_inventory(req)
        return req
    if action not in ('metadata', 'dry', 'submit', 'status', 'reconcile'):
        raise ValueError('deploy/metadata/dry/submit/status/reconcile/inventory only')
    required = {'action', 'root', 'ops', 'deployment', 'source_sha', 'source_files', 'source_overrides',
                'scheduler_code', 'scheduler_commit', 'python', 'job'}
    if not required <= set(req) or set(req) - (required | {'dry_job_sha256'}):
        raise ValueError('unexpected or missing transport request fields')
    for field in ('root', 'ops', 'deployment', 'scheduler_code'):
        safe_path(req[field], field, must_exist=field in ('root', 'ops'))
    if type(req['python']) is not str or not Path(req['python']).is_absolute() or '..' in Path(req['python']).parts:
        raise ValueError('python must be an absolute executable path')
    python = Path(req['python']).resolve(strict=True)
    if not python.is_file():
        raise ValueError('python must resolve to an executable file')
    if python != Path(sys.executable).resolve():
        raise ValueError('request Python must be this transport interpreter')
    if action != 'status':
        _deployment_binding(req)
        scheduler_binding(req)
    else:
        if type(req['source_sha']) is not str or not HEX40.fullmatch(req['source_sha']):
            raise ValueError('status source_sha must be a full lowercase Git SHA')
        if type(req['source_files']) is not dict or any(type(v) is not str or not HEX64.fullmatch(v) for v in req['source_files'].values()):
            raise ValueError('status source_files map invalid')
        if type(req['source_overrides']) is not dict:
            raise ValueError('status source_overrides must be an object')
        if type(req['scheduler_commit']) is not str or not HEX40.fullmatch(req['scheduler_commit']):
            raise ValueError('status scheduler_commit must be a full lowercase Git SHA')
    job = req['job']
    allowed = {'run_id', 'kind', 'cwd', 'config', 'seed', 'inputs', 'pinned_manifest', 'pinned_files', 'env',
               'cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib', 'wall_hours', 'stop_mode', 'no_admission', 'argv'}
    if type(job) is not dict or set(job) != allowed:
        raise ValueError('job must have the exact fixed launcher schema')
    if job['seed'] != _expected_seed(job.get('run_id')):
        raise ValueError('job seed must equal the fixed seed for its engineering stage')
    deployment, ops_root, config = _job_paths(req, job)
    config_sha = digest_file(config)
    if action == 'status':
        if job.get('run_id') not in RUNS:
            raise ValueError('status run_id is outside the fixed engineering stage set')
        request_binding = {'script': RUNS[job['run_id']][0], 'binding': None}
    else:
        request_binding = _fixed_request_binding(req, job)
    if job['pinned_manifest'] is not None:
        raise ValueError('pinned_manifest is not permitted; full source_files and fixed overlays bind this tree')
    rel_script = request_binding['script']
    pin_map = {}
    for item in job['pinned_files']:
        rel, _, value = item.rpartition('=')
        if rel in pin_map:
            raise ValueError('duplicate pinned source path')
        pin_map[rel] = value
        if rel not in req['source_files'] or value != req['source_files'][rel]:
            raise ValueError(f'pinned source path is outside the exact source tree or has a different SHA: {rel}')
    expected_pins = {rel_script: req['source_files'][rel_script]}
    for rel, override in req['source_overrides'].items():
        if override.get('sha256') != P3C_DATA_SHA256:
            raise ValueError('deployment overlay does not match the approved P3C data-channel SHA')
        expected_pins[rel] = P3C_DATA_SHA256
    if pin_map != expected_pins:
        raise ValueError('pinned_files must equal the fixed entrypoint and approved P3C overlay pins')
    if job['env'] != list(JOB_ENV):
        raise ValueError('engineering job environment must equal the fixed minimal environment')
    if req['action'] != 'status':
        _helper_environment_binding(req)
    if (job['kind'] != 'ops' or job['cwd'] != str(deployment) or type(job['cpu_cores']) is not int or job['cpu_cores'] != 1
            or job['no_admission'] is not False or job['stop_mode'] not in ('none', 'signal')
            or type(job['vram_mib']) is not int
            or not 0 <= job['vram_mib'] <= MAX_VRAM_MIB):
        raise ValueError('engineering job kind, cwd, seed, admission or resource scope invalid')
    for key in ('ram_mib', 'disk_gib'):
        if type(job[key]) is not int or job[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    wall = job['wall_hours']
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0 or wall > MAX_WALL_HOURS:
        raise ValueError('wall_hours must be finite and in (0, 48]')
    if type(job['env']) is not list or any(type(v) is not str for v in job['env']):
        raise ValueError('env must be a string list')
    if type(job['inputs']) is not list or any(type(v) is not str for v in job['inputs']):
        raise ValueError('inputs must be a string list')
    if type(job['pinned_files']) is not list or any(type(v) is not str for v in job['pinned_files']):
        raise ValueError('pinned_files must be a string list')
    for item in job['inputs']:
        safe_path(item, 'input', req['action'] != 'status')
    for item in job['pinned_files']:
        rel, sep, sha = item.rpartition('=')
        if not sep or not rel or not HEX64.fullmatch(sha) or rel.startswith('/') or '..' in Path(rel).parts:
            raise ValueError('pinned_files must be relative path=lowercase SHA-256 values')
    if req['action'] == 'submit' and (type(req.get('dry_job_sha256')) is not str or not HEX64.fullmatch(req['dry_job_sha256'])):
        raise ValueError('submit requires a lowercase dry_job_sha256')
    if action == 'reconcile':
        if (set(req) != required or job['run_id'] != LEGACY_A_PF['run_id']
                or req['source_sha'] != LEGACY_A_PF['source_sha']
                or req['source_files'] != LEGACY_A_PF['source_files'] or req['source_overrides'] != {}
                or req['scheduler_commit'] != LEGACY_A_PF['scheduler_commit']):
            raise ValueError('legacy reconcile only accepts the fixed original A preflight source and scheduler')
    return job


def queue_state(root, run_id):
    root = Path(root)
    out = {}
    for state in ('pending', 'claimed', 'rejected'):
        directory = root / 'queue' / state
        out[state] = sorted(p.name for p in directory.glob('*-' + run_id + '.json')) \
            if directory.is_dir() and not directory.is_symlink() else []
    return out


def _queue_rows(root, run_id, state):
    directory = Path(root) / 'queue' / state
    rows = []
    if not directory.is_dir() or directory.is_symlink():
        return rows
    for path in sorted(directory.glob('*-' + run_id + '.json')):
        row = {'name': path.name}
        if path.is_symlink() or not path.is_file():
            row['error'] = 'queue entry is not a regular file'
        elif path.stat().st_size > MAX_RECEIPT_BYTES:
            row.update({'error': 'queue entry exceeds bound', 'bytes': path.stat().st_size})
        else:
            raw = path.read_bytes()
            row['sha256'] = digest_bytes(raw)
            row['bytes'] = len(raw)
            try:
                row['payload'] = json.loads(raw)
            except ValueError:
                row['error'] = 'queue entry is not valid JSON'
        rows.append(row)
    return rows


def _bounded_receipt(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError('status receipt must be a regular non-symlink file')
    size = path.stat().st_size
    if size > MAX_RECEIPT_BYTES:
        raise ValueError(f'status receipt exceeds {MAX_RECEIPT_BYTES} bytes: {path.name}')
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = raw.decode('utf-8', errors='replace')
    return {'path': str(path), 'bytes': len(raw), 'sha256': digest_bytes(raw), 'payload': payload}


def _status_run_binding(run, job, metadata):
    prepared = metadata.get('launcher_prepared') or {}
    reasons = []
    checks = {
        'schema': 'vkgr-run-v1', 'run_id': job['run_id'], 'kind': job['kind'], 'cwd': job['cwd'],
        'argv': job['argv'], 'seed': job['seed'], 'git': prepared.get('git'),
        'overlay': prepared.get('overlay'), 'pinned_files': prepared.get('pinned_files'),
        'config': prepared.get('config'), 'inputs': prepared.get('inputs'),
        'ops_source': metadata.get('launcher_ops_source'), 'stop_mode': job['stop_mode'],
    }
    for key, expected in checks.items():
        if run.get(key) != expected:
            reasons.append(f'run.json {key} differs from sealed metadata')
    resources = {name: job[name] for name in ('cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib')}
    resources['wall_seconds'] = job['wall_hours'] * 3600.0
    if run.get('resources') != resources:
        reasons.append('run.json resources differ from sealed job')
    env = run.get('env')
    if type(env) is not dict:
        reasons.append('run.json env is missing or malformed')
    else:
        required_env = dict(item.split('=', 1) for item in job['env'])
        required_env['VKGR_RUN_ID'] = job['run_id']
        expected_run_dir = Path(metadata['runtime_root']) / 'runs' / job['run_id']
        required_env['VKGR_PROGRESS_PATH'] = str(expected_run_dir / 'progress.json')
        for key, expected in required_env.items():
            if env.get(key) != expected:
                reasons.append(f'run.json env {key} differs from fixed launcher environment')
        allowed_vkgr = set(required_env) | {'VKGR_OPS_ROOT'}
        if set(key for key in env if key.startswith('VKGR_')) - allowed_vkgr:
            reasons.append('run.json env contains an unregistered VKGR variable')
        if 'VKGR_OPS_ROOT' in env and env['VKGR_OPS_ROOT'] != metadata['runtime_root']:
            reasons.append('run.json VKGR_OPS_ROOT differs from fixed ops root')
    return {'passed': not reasons, 'reasons': reasons}


def inventory(req):
    _validate_inventory(req)
    root = Path(req['root'])
    output = {'runs': []}
    for run_id in req['run_ids']:
        run_dir = root / 'runs' / run_id
        record = {'run_id': run_id, 'exists': run_dir.is_dir(), 'queue': queue_state(root, run_id)}
        if run_dir.is_dir():
            for name in ('run.json', 'manifest.json', 'source.json', 'owner.json', 'terminal.json'):
                path = run_dir / name
                if path.is_symlink() or not path.is_file():
                    continue
                raw = path.read_bytes()
                if len(raw) > MAX_RECEIPT_BYTES:
                    record[name] = {'oversize': True, 'bytes': len(raw), 'sha256': digest_bytes(raw)}
                else:
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        payload = None
                    record[name] = {'bytes': len(raw), 'sha256': digest_bytes(raw), 'payload': payload}
        output['runs'].append(record)
    return output


def _validate_inventory(req):
    required = {'action', 'root', 'run_ids'}
    if set(req) != required or req.get('action') != 'inventory':
        raise ValueError('inventory accepts only root and run_ids')
    safe_path(req['root'], 'root', True)
    ids = req['run_ids']
    if type(ids) is not list or tuple(ids) != TAKEOVER_RUN_IDS:
        raise ValueError('inventory must name exact original twelve run IDs in fixed order')


def _assert_run_unclaimed(req):
    job = req['job']
    root, ops_root = Path(req['root']), Path(req['ops'])
    run_id = job['run_id']
    queue = queue_state(root, run_id)
    run_exists = (root / 'runs' / run_id).exists()
    intent = ops_root / ('intent-' + run_id + '.json')
    metadata = ops_root / ('metadata-' + run_id + '.json')
    if run_exists or any(queue.values()):
        raise FileExistsError('fixed run id already exists in runs/queue; no retry or overwrite')
    if intent.exists():
        raise FileExistsError('intent already exists; no retry or overwrite')
    return queue, metadata, intent


def _fresh(req):
    queue, metadata, intent = _assert_run_unclaimed(req)
    if not metadata.is_file() or metadata.is_symlink():
        raise ValueError('metadata must be written before dry/submit')
    return queue, metadata, intent


def _launcher(req, job):
    ops_code = Path(req['scheduler_code'])
    if not (ops_code / 'ops' / 'vkgr_ops' / 'launcher.py').is_file():
        raise ValueError('versioned vkgr_ops launcher checkout required')
    sys.path.insert(0, str(ops_code / 'ops'))
    from vkgr_ops import launcher
    prepared = launcher.prepare(job)
    envelope = canonical({'spec': job, 'prepared': prepared})
    if len(envelope) >= MAX_ENVELOPE_BYTES:
        raise ValueError('engineering launch envelope exceeds 48 KiB')
    return prepared, len(envelope)


def _job_sha(job):
    return digest_bytes(canonical(job))


def _launcher_ops_files(req):
    path = Path(req['scheduler_code']) / 'ops' / 'vkgr_ops'
    files = {}
    for item in sorted(path.glob('*.py')):
        if item.is_symlink() or not item.is_file():
            raise ValueError('launcher OPS source ledger contains a non-regular Python file')
        files[f'vkgr_ops/{item.name}'] = digest_file(item)
    if not files:
        raise ValueError('launcher OPS source ledger is empty')
    return files


def _launcher_ops_source(req):
    files = _launcher_ops_files(req)
    # Match vkgr_ops.launcher._ops_source(): canonical_json includes the final LF.
    return {'commit': req['scheduler_commit'], 'sha256': digest_bytes(canonical(files) + b'\n')}


def _helper_identity():
    path = Path(__file__).resolve()
    repo = next((parent for parent in path.parents if (parent / '.git').exists()), None)
    if repo is None:
        return {'commit': None, 'file_sha256': digest_file(path), 'tracked_sha256': None,
                'source_matches_commit': False}
    commit = _git(repo, 'rev-parse', 'HEAD').decode().strip()
    relative = path.relative_to(repo).as_posix()
    tracked = _git(repo, 'show', f'{commit}:{relative}')
    tracked_sha = digest_bytes(tracked)
    file_sha = digest_file(path)
    return {'commit': commit, 'file_sha256': file_sha, 'tracked_sha256': tracked_sha,
            'source_matches_commit': file_sha == tracked_sha}


def _transport_request_sha(req, job, request_binding, source_binding, dependency_binding,
                           environment_binding, config_sha):
    value = {'root': req['root'], 'ops': req['ops'], 'deployment': req['deployment'],
             'source_binding': source_binding, 'scheduler_code': req['scheduler_code'],
             'scheduler_binding': dependency_binding, 'job': job, 'request_binding': request_binding,
             'environment_binding': environment_binding, 'config_sha256': config_sha}
    return digest_bytes(canonical(value))


def metadata(req):
    job = validate(req)
    _assert_run_unclaimed(req)
    prepared, size = _launcher(req, job)
    ops_root = Path(req['ops'])
    target = ops_root / ('metadata-' + job['run_id'] + '.json')
    source_binding = _deployment_binding(req)
    dependency_binding = scheduler_binding(req)
    request_binding = _fixed_request_binding(req, job)
    environment_binding = _helper_environment_binding(req)
    config_sha = digest_file(job['config'])
    body = {'schema': 'ur001-engineering-metadata-v1', 'job_sha256': _job_sha(job),
            'request_binding': request_binding, 'environment_binding': environment_binding,
            'transport_request_sha256': _transport_request_sha(req, job, request_binding, source_binding,
                                                                dependency_binding, environment_binding, config_sha),
            'source_binding': source_binding, 'scheduler_binding': dependency_binding,
            'request_binding': request_binding, 'environment_binding': environment_binding,
            'launcher_ops_source': _launcher_ops_source(req),
            'config_sha256': config_sha, 'launcher_prepared': prepared,
            'launch_bytes': size, 'job': job}
    with target.open('x') as stream:
        json.dump(body, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {'status': 'metadata_written', 'run_id': job['run_id'], 'metadata_sha256': digest_file(target),
            'job_sha256': body['job_sha256'], 'launch_bytes': size}


def dry(req):
    job = validate(req)
    queue, meta, _ = _fresh(req)
    body = json.loads(meta.read_text())
    job_sha = _job_sha(job)
    if body.get('job_sha256') != job_sha:
        raise ValueError('metadata/job binding mismatch')
    source_binding = _deployment_binding(req)
    dependency_binding = scheduler_binding(req)
    request_binding = _fixed_request_binding(req, job)
    environment_binding = _helper_environment_binding(req)
    config_sha = digest_file(job['config'])
    if body.get('source_binding') != source_binding or body.get('scheduler_binding') != dependency_binding:
        raise ValueError('metadata source/dependency binding differs from current deployment')
    if body.get('request_binding') != request_binding or body.get('environment_binding') != environment_binding:
        raise ValueError('metadata request/environment binding differs from current dry preparation')
    if body.get('config_sha256') != config_sha:
        raise ValueError('metadata config hash differs from current private request')
    request_sha = _transport_request_sha(req, job, request_binding, source_binding,
                                         dependency_binding, environment_binding, config_sha)
    if body.get('transport_request_sha256') != request_sha:
        raise ValueError('metadata sealed transport request digest differs from current dry request')
    if body.get('launcher_ops_source') != _launcher_ops_source(req):
        raise ValueError('metadata launcher OPS source differs from current dry preparation')
    prepared, size = _launcher(req, job)
    if body.get('launcher_prepared') != prepared:
        raise ValueError('metadata launcher prepared inputs/source differ from current dry preparation')
    if body.get('launch_bytes') != size:
        raise ValueError('metadata launcher envelope size differs from current dry preparation')
    return {'status': 'dry_ok', 'run_id': job['run_id'], 'job_sha256': job_sha,
            'metadata_sha256': digest_file(meta), 'launch_bytes': size, 'queue': queue}


def submit(req):
    job = validate(req)
    receipt = dry(dict(req, action='dry'))
    if req['dry_job_sha256'] != receipt['job_sha256']:
        raise ValueError('dry job SHA differs from current fixed request')
    root, ops_root = Path(req['root']), Path(req['ops'])
    lock_path = ops_root / '.ur001-submit.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _fresh(req)
        intent = ops_root / ('intent-' + job['run_id'] + '.json')
        with intent.open('x') as stream:
            json.dump({'schema': 'ur001-submit-intent-v1', 'job_sha256': receipt['job_sha256'],
                       'metadata_sha256': receipt['metadata_sha256']}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        ops_code = Path(req['scheduler_code'])
        sys.path.insert(0, str(ops_code / 'ops'))
        from vkgr_ops import scheduler
        queue_path = scheduler.submit(root, job)
    return {'status': 'written_unconfirmed', 'run_id': job['run_id'],
            'intent_sha256': digest_file(intent), 'queue_path': str(queue_path)}


def status(req):
    job = validate(req)
    root, ops_root = Path(req['root']), Path(req['ops'])
    run_id = job['run_id']
    run_dir = root / 'runs' / run_id
    queue = {state: _queue_rows(root, run_id, state) for state in ('pending', 'claimed', 'rejected')}
    reasons = []
    claimed = queue['claimed']
    if len(claimed) != 1:
        reasons.append('queue must contain exactly one claimed entry for this run id')
    if queue['pending']:
        reasons.append('pending entries remain for this run id')
    if queue['rejected']:
        reasons.append('rejected entries exist for this run id')
    claimed_payload = None
    if len(claimed) == 1:
        row = claimed[0]
        payload = row.get('payload')
        if type(payload) is not dict or payload.get('schema') != 'vkgr-queue-job-v1':
            reasons.append('claimed payload is missing or has the wrong queue schema')
        else:
            payload_job = dict(payload)
            payload_job.pop('schema', None)
            if payload_job != job:
                reasons.append('claimed queue payload differs from the sealed job')
            claimed_payload = payload_job
    metadata_path = ops_root / ('metadata-' + run_id + '.json')
    intent_path = ops_root / ('intent-' + run_id + '.json')
    metadata_receipt = _bounded_receipt(metadata_path)
    intent_receipt = _bounded_receipt(intent_path)
    metadata = metadata_receipt.get('payload') if metadata_receipt and isinstance(metadata_receipt.get('payload'), dict) else None
    if metadata is None:
        reasons.append('sealed metadata receipt is missing or invalid')
    else:
        if metadata.get('schema') != 'ur001-engineering-metadata-v1' or metadata.get('job') != job or metadata.get('job_sha256') != _job_sha(job):
            reasons.append('metadata job binding differs from the requested fixed job')
        if metadata.get('config_sha256') != digest_file(job['config']):
            reasons.append('private config differs from sealed metadata')
        source_bind = {'commit': req['source_sha'], 'files_sha256': digest_bytes(canonical(req['source_files'])),
                       'paths': len(req['source_files']),
                       'overrides_sha256': digest_bytes(canonical(req['source_overrides'])),
                       'p3c_data_sha256': P3C_DATA_SHA256 if P3C_PATH in req['source_overrides'] else None}
        if metadata.get('source_binding') != source_bind:
            reasons.append('request source binding differs from sealed metadata')
        if metadata.get('scheduler_binding', {}).get('commit') != req['scheduler_commit']:
            reasons.append('request scheduler commit differs from sealed metadata')
        try:
            status_sha = _transport_request_sha(req, job, metadata.get('request_binding'),
                source_bind, metadata.get('scheduler_binding'), metadata.get('environment_binding'),
                metadata.get('config_sha256'))
            if metadata.get('transport_request_sha256') != status_sha:
                reasons.append('status request differs from the sealed transport request')
        except (TypeError, ValueError):
            reasons.append('metadata transport request fingerprint is invalid')
    intent = intent_receipt.get('payload') if intent_receipt and isinstance(intent_receipt.get('payload'), dict) else None
    if intent is None:
        reasons.append('submit intent receipt is missing or invalid')
    elif metadata_receipt and (intent.get('job_sha256') != _job_sha(job)
                               or intent.get('metadata_sha256') != metadata_receipt['sha256']):
        reasons.append('intent does not bind the exact job and metadata bytes')
    queue_accepted = not reasons
    run_receipt = _bounded_receipt(run_dir / 'run.json')
    runtime = {'passed': False, 'reasons': ['run.json missing']}
    run_payload = run_receipt.get('payload') if run_receipt and isinstance(run_receipt.get('payload'), dict) else None
    if run_payload is not None and metadata is not None:
        status_meta = dict(metadata)
        status_meta['runtime_root'] = str(root)
        runtime = _status_run_binding(run_payload, job, status_meta)
    raw_receipts = {}
    for name in ('terminal.json', 'owner.json', 'manifest.json', 'logs/stdout.log', 'logs/stderr.log',
                 'out/manifest.json', 'out/preflight.json', 'out/profile.json'):
        receipt = _bounded_receipt(run_dir / name)
        if receipt is not None:
            raw_receipts[name] = receipt
    if run_receipt is not None:
        raw_receipts['run.json'] = run_receipt
    result_status = 'pending_root_mechanical_acceptance'
    if not queue_accepted:
        result_status = 'queue_not_accepted'
    elif not runtime['passed']:
        result_status = 'runtime_binding_mismatch_or_missing'
    return {'run_id': run_id, 'queue': queue, 'queue_accepted': queue_accepted,
            'queue_acceptance_reasons': reasons, 'runtime_binding_passed': runtime['passed'],
            'runtime_binding_reasons': runtime['reasons'], 'engineering_verified': False,
            'engineering_result_status': result_status, 'raw_receipts': raw_receipts,
            'metadata_sha256': metadata_receipt['sha256'] if metadata_receipt else None,
            'intent_sha256': intent_receipt['sha256'] if intent_receipt else None}


def reconcile(req):
    """Read-only reconciliation for the one pinned legacy A preflight receipt."""
    job = validate(req)
    ordinary = status(dict(req, action='status'))
    if not ordinary['queue_accepted']:
        raise ValueError('legacy preflight queue/job/metadata/intent binding was not accepted')
    if ordinary['runtime_binding_reasons'] != ['run.json ops_source differs from sealed metadata']:
        raise ValueError('legacy preflight has runtime binding differences beyond the pinned ops_source format')
    if ordinary['metadata_sha256'] != LEGACY_A_PF['metadata_sha256']:
        raise ValueError('legacy preflight metadata bytes differ from the one approved historical receipt')
    if ordinary['intent_sha256'] != LEGACY_A_PF['intent_sha256']:
        raise ValueError('legacy preflight intent bytes differ from the one approved historical receipt')
    if _job_sha(job) != LEGACY_A_PF['job_sha256']:
        raise ValueError('legacy preflight job differs from its fixed historical job SHA')
    if ordinary['queue']['claimed'] and len(ordinary['queue']['claimed']) != 1:
        raise ValueError('legacy preflight must have exactly one claimed queue entry')

    metadata_path = Path(req['ops']) / ('metadata-' + job['run_id'] + '.json')
    metadata_receipt = _bounded_receipt(metadata_path)
    metadata = metadata_receipt['payload']
    intent_receipt = _bounded_receipt(Path(req['ops']) / ('intent-' + job['run_id'] + '.json'))
    intent = intent_receipt['payload']
    source_binding = _deployment_binding(req)
    dependency_binding = scheduler_binding(req)
    request_binding = _fixed_request_binding(req, job)
    environment_binding = _helper_environment_binding(req)
    config_sha = digest_file(job['config'])
    expected_bindings = {
        'schema': 'ur001-engineering-metadata-v1', 'job': job, 'job_sha256': LEGACY_A_PF['job_sha256'],
        'source_binding': source_binding, 'scheduler_binding': dependency_binding,
        'request_binding': request_binding, 'environment_binding': environment_binding,
        'config_sha256': config_sha,
        'transport_request_sha256': _transport_request_sha(req, job, request_binding, source_binding,
                                                           dependency_binding, environment_binding, config_sha),
    }
    for key, expected in expected_bindings.items():
        if metadata.get(key) != expected:
            raise ValueError(f'legacy preflight sealed metadata {key} differs from the fixed request')
    if (intent != {'schema': 'ur001-submit-intent-v1', 'job_sha256': LEGACY_A_PF['job_sha256'],
                   'metadata_sha256': LEGACY_A_PF['metadata_sha256']}):
        raise ValueError('legacy preflight intent payload differs from its fixed job/metadata binding')

    files = _launcher_ops_files(req)
    file_map_sha = digest_bytes(canonical(files))
    legacy_ops_source = {'commit': LEGACY_A_PF['scheduler_commit'], 'sha256': file_map_sha}
    runtime_ops_source = _launcher_ops_source(req)
    if metadata.get('launcher_ops_source') != legacy_ops_source:
        raise ValueError('legacy metadata ops_source is not the no-LF digest of the fixed launcher file ledger')
    if runtime_ops_source != {'commit': LEGACY_A_PF['scheduler_commit'],
                              'sha256': digest_bytes(canonical(files) + b'\n')}:
        raise ValueError('current launcher ops_source is not the LF digest of the same fixed file ledger')

    receipts = ordinary['raw_receipts']
    run_receipt = receipts.get('run.json')
    run_payload = run_receipt.get('payload') if run_receipt and isinstance(run_receipt.get('payload'), dict) else None
    if run_payload is None or run_payload.get('ops_source') != runtime_ops_source:
        raise ValueError('legacy run.json ops_source differs from the LF digest of the fixed launcher file ledger')
    stdout = receipts.get('logs/stdout.log')
    if not stdout or stdout['sha256'] != LEGACY_A_PF['stdout_sha256'] or stdout['bytes'] != LEGACY_A_PF['stdout_bytes']:
        raise ValueError('legacy preflight stdout receipt differs from its fixed SHA/size')

    reconciled_metadata = dict(metadata)
    reconciled_metadata['launcher_ops_source'] = runtime_ops_source
    runtime = _status_run_binding(run_payload, job, dict(reconciled_metadata, runtime_root=req['root']))
    if not runtime['passed']:
        raise ValueError('legacy preflight runtime binding still has differences after the one approved ops_source normalization')

    helper_identity = _helper_identity()
    if not helper_identity['source_matches_commit']:
        raise ValueError('reconcile helper file bytes differ from the helper Git commit')
    receipt_summary = {name: {'bytes': item['bytes'], 'sha256': item['sha256']}
                       for name, item in receipts.items()}
    return {
        'status': 'legacy_pf_reconciled_pending_root_acceptance', 'run_id': job['run_id'],
        'queue_accepted': True, 'runtime_binding_passed': True, 'engineering_verified': False,
        'engineering_result_status': 'pending_root_mechanical_acceptance',
        'legacy_pf_protocol': {
            'source_sha': req['source_sha'], 'source_files_sha256': digest_bytes(canonical(req['source_files'])),
            'helper_sha256': req['source_files']['scripts/ur001_engineering_ops.py'],
            'metadata_sha256': metadata_receipt['sha256'], 'intent_sha256': intent_receipt['sha256'],
            'job_sha256': _job_sha(job), 'scheduler_commit': req['scheduler_commit'],
            'deployment_binding': source_binding, 'scheduler_binding': dependency_binding,
            'transport_request_sha256': metadata['transport_request_sha256'],
            'request_binding_sha256': digest_bytes(canonical(request_binding)),
            'config_sha256': config_sha,
            'launcher_ops_paths': len(files), 'launcher_ops_file_map_sha256': file_map_sha,
            'metadata_ops_source_no_lf_sha256': legacy_ops_source['sha256'],
            'run_ops_source_lf_sha256': runtime_ops_source['sha256'],
            'same_launcher_file_ledger': True, 'original_stdout_sha256': stdout['sha256'],
            'original_stdout_bytes': stdout['bytes'], 'helper_identity': helper_identity,
        },
        'claimed_queue_receipts': [{'name': row['name'], 'bytes': row['bytes'], 'sha256': row['sha256']}
                                   for row in ordinary['queue']['claimed']],
        'raw_receipt_sha256': receipt_summary,
        'runtime_binding_reasons': [],
    }


def main():
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError('request exceeds bounded stdin size')
    req = json.loads(raw)
    if type(req) is not dict:
        raise ValueError('request must be a JSON object')
    action = req.get('action')
    if action == 'deploy':
        result = _deployment(req)
    elif action == 'inventory':
        result = inventory(req)
    else:
        validate(req)
        result = {'metadata': metadata, 'dry': dry, 'submit': submit, 'status': status,
                  'reconcile': reconcile}[action](req)
    result['read_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
