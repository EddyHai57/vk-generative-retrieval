"""Read-only G user-half request checks, source identity, and formal preflight adapter."""
import argparse
import hashlib
import json
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import contextmanager


SEEDS = (42, 2024, 2025)
CONTROL_HITS = {42: 398, 2024: 384, 2025: 384}
ANCHOR_SOURCE_REQUEST_SHA256 = 'c6995e313011f69f7acdd3e2af1205ae8f1b71a4fa4e0cd996d1fa77fe07dd55'
ANCHOR_SOURCE_COMMIT = 'b5a4d1c99ba2d8361a2d90c23bac27af5b71769d'
ANCHOR_MANIFEST_SHA256 = '2f6744f4cafae340666410b87912b8cf1c71d9447c55b740f1280f2c52afefae'
ANCHOR_SCIENCE_FILE_COUNT = 41
ANCHOR_OPS_FILE_COUNT = 45
MAX_ANCHOR_REQUEST_BYTES = 16 * 1024 * 1024
MAX_ANCHOR_MANIFEST_BYTES = 64 * 1024 * 1024
DELTA_PATHS = (
    'src/vkgr/g_half_users_a1_run.py',
    'scripts/g_half_users_ops.py',
    'tests/test_g_half_users.py',
    'tests/test_g_half_users_ops.py',
    'scripts/g_half_users_engineering_queue.py',
    'tests/test_g_half_users_engineering_queue.py',
    'docs/superpowers/plans/2026-10-20-g-half-implementation.md',
    'configs/g_half_users/g-half-users-s42.json',
    'configs/g_half_users/g-half-users-s2024.json',
    'configs/g_half_users/g-half-users-s2025.json',
)
DEPENDENCIES = ('numpy', 'torch', 'pyarrow', 'PyYAML')
_OLD_ORIGINALS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def replace(argv, name, value):
    if argv.count(name) != 1:
        raise ValueError('exactly one argument required: ' + name)
    argv[argv.index(name)+1] = str(value)


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate request field')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite request number')
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def _sha_map(value, role):
    if type(value) is not dict or not value:
        raise ValueError(role + ' nonempty path-to-SHA mapping required')
    for name, pin in value.items():
        path = Path(name)
        if type(name) is not str or path.is_absolute() or '..' in path.parts or not name or type(pin) is not str \
                or len(pin) != 64 or any(char not in '0123456789abcdef' for char in pin):
            raise ValueError(role + ' contains invalid path/SHA')
    return value


def check_member_sets(files, training, delta, *, accepted_training_files):
    files = _sha_map(files, 'deployed files')
    training = _sha_map(training, 'anchor training files')
    delta = _sha_map(delta, 'frozen pinned delta')
    accepted_training_files = _sha_map(accepted_training_files, 'independently accepted anchor files')
    if training != accepted_training_files:
        raise ValueError('training ledger differs from the independently accepted anchor file map')
    if set(delta) != set(DELTA_PATHS):
        raise ValueError('pinned delta must equal the frozen half-user runtime/test path set')
    if set(files) != set(training) | set(delta) or set(training) & set(delta):
        raise ValueError('deployed source files must equal anchor training files union the exact pinned delta')
    for name, pin in {**training, **delta}.items():
        if files.get(name) != pin:
            raise ValueError('deployed file SHA differs from its anchor/delta pin: ' + name)
    return {'training_file_count': len(training), 'pinned_delta_file_count': len(delta)}


def read_accepted_source_request(reference):
    if type(reference) is not dict or set(reference) != {'path', 'sha256'} \
            or type(reference['path']) is not str or not Path(reference['path']).is_absolute() \
            or '..' in Path(reference['path']).parts \
            or reference['sha256'] != ANCHOR_SOURCE_REQUEST_SHA256:
        raise ValueError('original A1 source request SHA/path reference required')
    path = Path(reference['path'])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ANCHOR_REQUEST_BYTES \
            or sha(path) != ANCHOR_SOURCE_REQUEST_SHA256:
        raise ValueError('original accepted A1 request bytes differ from the frozen SHA')
    value = load(path)
    if type(value) is not dict or value.get('source_commit') != ANCHOR_SOURCE_COMMIT:
        raise ValueError('original A1 request source commit differs from the frozen source')
    files = _sha_map(value.get('files'), 'original accepted A1 ops files')
    if len(files) != ANCHOR_OPS_FILE_COUNT:
        raise ValueError('original A1 ops/launcher source ledger must contain 45 files')
    return {'source_commit': value['source_commit'], 'files': files}


def read_accepted_anchor_manifest(reference):
    if type(reference) is not dict or set(reference) != {'path', 'sha256'} \
            or type(reference['path']) is not str or not Path(reference['path']).is_absolute() \
            or '..' in Path(reference['path']).parts \
            or reference['sha256'] != ANCHOR_MANIFEST_SHA256:
        raise ValueError('accepted seed42 anchor manifest SHA/path reference required')
    path = Path(reference['path'])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ANCHOR_MANIFEST_BYTES \
            or sha(path) != ANCHOR_MANIFEST_SHA256:
        raise ValueError('accepted seed42 anchor manifest bytes differ from the frozen SHA')
    value = load(path)
    if (type(value) is not dict or value.get('schema') != 'g-sid-structure-a1-train-manifest-v1'
            or value.get('kind') != 'g_sid_structure_a1_train' or value.get('status') != 'complete'
            or value.get('config', {}).get('seed') != 42
            or value.get('config', {}).get('sid') != 'STRUCT_C1_B48D3_S42'):
        raise ValueError('accepted complete STRUCT A1 seed42 anchor manifest required')
    binding = value.get('binding')
    if type(binding) is not dict or binding.get('task') != 'SID-STRUCTURE-MATCHED':
        raise ValueError('accepted STRUCT A1 anchor binding required')
    source = binding.get('source')
    if type(source) is not dict:
        raise ValueError('anchor scientific source ledger required')
    science_files = _sha_map(source.get('files'), 'anchor scientific source files')
    if len(science_files) != ANCHOR_SCIENCE_FILE_COUNT:
        raise ValueError('original scientific source ledger must contain 41 files')
    runtime = binding.get('runtime')
    environment = runtime.get('environment') if type(runtime) is dict else None
    if type(environment) is not dict or set(environment) != {'python', 'exact_dependencies'} \
            or 'uv.lock' not in science_files:
        raise ValueError('anchor exact Python/dependency/lock runtime required')
    expected_runtime = {**environment, 'lock_sha256': science_files['uv.lock']}
    context_pin = binding.get('context_binding_sha256')
    git = source.get('git')
    if type(context_pin) is not str or len(context_pin) != 64 or any(c not in '0123456789abcdef' for c in context_pin) \
            or type(git) is not dict \
            or set(git) != {'git_base', 'git_dirty'}:
        raise ValueError('anchor context/source Git binding required')
    return {'source_files': science_files, 'runtime': expected_runtime,
            'context_binding_sha256': context_pin, 'git': git, 'manifest': value}


def accepted_baselines(req):
    old_ops = read_accepted_source_request(req.get('anchor_source_request'))
    anchor = read_accepted_anchor_manifest(req.get('anchor_manifest'))
    science_files = anchor['source_files']
    if not set(science_files) <= set(old_ops['files']) \
            or any(old_ops['files'].get(name) != pin for name, pin in science_files.items()):
        raise ValueError('anchor manifest 41-file science ledger differs from the accepted 45-file ops ledger')
    return old_ops, anchor


def verify_deployed_tree_delta(root, current_commit):
    """Require the actual tree to be the accepted original source plus exactly the frozen new paths."""
    try:
        diff = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-status', '--no-renames',
                                        ANCHOR_SOURCE_COMMIT, current_commit], text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError('accepted original source commit must be available for full-tree comparison') from exc
    rows = [line.split('\t', 1) for line in diff.splitlines()]
    if any(len(row) != 2 or row[0] != 'A' for row in rows):
        raise ValueError('every change from the accepted original tree must be a pinned path addition')
    paths = [row[1] for row in rows]
    if len(paths) != len(set(paths)) or set(paths) != set(DELTA_PATHS):
        raise ValueError('deployment tree must add exactly the frozen pinned-delta paths')
    return {'base_commit': ANCHOR_SOURCE_COMMIT, 'current_commit': current_commit,
            'added_paths': tuple(sorted(paths)), 'added_path_count': len(paths)}


def validate_source_contract(req):
    """Enforce exact original source + frozen pinned delta and actual deployed runtime."""
    required = {'deployment', 'source_commit', 'files', 'training_files', 'pinned_delta', 'training_runtime', 'python',
                'anchor_source_request', 'anchor_manifest'}
    if type(req) is not dict or not required <= set(req):
        raise ValueError('half-user request lacks source/runtime identity fields')
    accepted_ops, anchor = accepted_baselines(req)
    root = Path(req['deployment']).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('deployment root must be a directory')
    actual_commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != req['source_commit'] or len(actual_commit) != 40:
        raise ValueError('deployed source commit differs from the request')
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain=v1', '--untracked-files=all'], text=True)
    if dirty:
        raise ValueError('deployed source tree must be clean')
    tree_delta = verify_deployed_tree_delta(root, actual_commit)
    files = _sha_map(req['files'], 'deployed files')
    ops_counts = check_member_sets(files, req['training_files'], req['pinned_delta'],
                                   accepted_training_files=accepted_ops['files'])
    training = req['training_files']
    delta = req['pinned_delta']
    expected_science_files = {**anchor['source_files'], **delta}
    science_files = {name: files.get(name) for name in expected_science_files}
    science_counts = check_member_sets(science_files, anchor['source_files'], delta,
                                      accepted_training_files=anchor['source_files'])
    for name, pin in {**training, **delta}.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha(path) != pin:
            raise ValueError('deployed source byte mismatch: ' + name)
        blob = subprocess.check_output(['git', '-C', str(root), 'show', actual_commit + ':' + name])
        if hashlib.sha256(blob).hexdigest() != pin:
            raise ValueError('deployed file differs from its fixed Git blob: ' + name)
    runtime = actual_runtime(req['python'], root / 'uv.lock')
    if canonical(req['training_runtime']) != canonical(anchor['runtime']) \
            or canonical(runtime) != canonical(anchor['runtime']):
        raise ValueError('actual deployed Python/dependencies/lock SHA differ from the anchor training runtime')
    return {'deployment_commit': actual_commit, 'source_sha256': digest(files),
            'science_source_sha256': digest(science_files), 'runtime': runtime,
            'tree_delta': tree_delta,
            'ops_source_file_count': ops_counts['training_file_count'] + ops_counts['pinned_delta_file_count'],
            'science_source_file_count': science_counts['training_file_count'] + science_counts['pinned_delta_file_count'],
            'anchor_ops_file_count': ops_counts['training_file_count'],
            'anchor_science_file_count': science_counts['training_file_count'],
            'pinned_delta_file_count': len(delta)}


def actual_runtime(python, lock_path):
    """Probe the selected deployed interpreter instead of trusting request metadata."""
    if type(python) is not str or not python:
        raise ValueError('deployed Python executable required')
    script = "import importlib.metadata,json,platform; print(json.dumps({'python':platform.python_version(),'exact_dependencies':{n:importlib.metadata.version(n) for n in ('numpy','torch','pyarrow','PyYAML')}},sort_keys=True))"
    result = subprocess.run([python, '-B', '-c', script], check=True, capture_output=True, text=True, timeout=30)
    value = json.loads(result.stdout)
    if type(value) is not dict or set(value) != {'python', 'exact_dependencies'} or type(value['exact_dependencies']) is not dict \
            or tuple(sorted(value['exact_dependencies'])) != tuple(sorted(DEPENDENCIES)):
        raise ValueError('deployed interpreter runtime probe schema')
    value['lock_sha256'] = sha(lock_path)
    return value


def validate_request_file(path, request_sha256):
    if Path(path).is_symlink() or type(request_sha256) is not str or len(request_sha256) != 64 or sha(path) != request_sha256:
        raise ValueError('private request SHA mismatch')
    req = load(path)
    source = validate_source_contract(req)
    anchor = req.get('anchor_manifest')
    if type(anchor) is not dict or set(anchor) != {'path', 'sha256'} or type(anchor['path']) is not str \
            or type(anchor['sha256']) is not str or len(anchor['sha256']) != 64 \
            or any(c not in '0123456789abcdef' for c in anchor['sha256']):
        raise ValueError('accepted original anchor manifest path/SHA required')
    return req, source


def _old_ops(root):
    sys.path.insert(0, str(Path(root) / 'scripts'))
    import sid_structure_ops
    return sid_structure_ops


def _half_module(root):
    sys.path.insert(0, str(Path(root) / 'src'))
    from vkgr import g_half_users_a1_run
    return g_half_users_a1_run


def derive_job(req, seed, vram_mib=None):
    """Derive a half-user train job by editing only the old A1 job envelope and registered config."""
    if seed not in SEEDS:
        raise ValueError('fixed three seeds only')
    expected_users_sha = _require_expected_selected(req)
    root, runs = Path(req['deployment']), Path(req['runs'])
    ops = _old_ops(root)
    half = _half_module(root)
    original_derive = _OLD_ORIGINALS.get(id(ops), {}).get('derive_job', ops.derive_job)
    job = original_derive(req, seed, vram_mib)
    anchor_config = ops.load(job['config'])
    config_path = root / f'configs/g_half_users/g-half-users-s{seed}.json'
    config = ops.load(config_path)
    expected = dict(anchor_config, schema=half.SCHEMA, task_id=half.TASK_ID,
                    sid='STRUCT_C1_B48D3_S42',
                    sid_manifest_sha256=req['coll_refs']['sid_manifest']['sha256'],
                    user_sample_fraction=0.5, user_sample_seed=20261020,
                    user_sample_count_rule=half.SAMPLE_COUNT_RULE,
                    user_sample_method=half.SAMPLE_METHOD)
    if config != expected:
        raise ValueError('half-user config differs from the same-seed A1 config only by frozen sampling identity')
    half.validate_config(config, False)
    rid = f'g-half-users-r2-s{seed}'
    config_rel = config_path.relative_to(root)
    job.update(run_id=rid, seed=seed, cwd=str(root), config=str(config_path),
               vram_mib=vram_mib if vram_mib is not None else job['vram_mib'])
    argv = job['argv']
    argv[argv.index('-m')+1] = 'vkgr.g_half_users_a1_run'
    for name, value in {'--config': str(config_rel), '--config-sha': sha(config_path),
                        '--output': str(runs/rid/'out'), '--scratch': str(runs/rid/'scratch')}.items():
        ops.replace(argv, name, value)
    job['inputs'] = list(dict.fromkeys([*job['inputs'], str(config_path)]))
    job['pinned_files'] = [name + '=' + pin for name, pin in sorted(req['files'].items())]
    job['env'] = [value for value in job['env'] if not value.startswith(
        ('VKGR_RUN_ID=', 'VKGR_PROGRESS_PATH=', 'VKGR_EXPECTED_SELECTED_USERS_SHA256='))]
    job['env'].append('VKGR_EXPECTED_SELECTED_USERS_SHA256=' + expected_users_sha)
    deadline = req.get('deadline_epoch')
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError('finite driver deadline_epoch required for r2 train job')
    job['env'].append('VKGR_DRIVER_DEADLINE_EPOCH=' + str(int(deadline)))
    return job


def derive_verify(job, manifest_sha256, runs):
    """Derive a new verify job from the accepted half-user train job."""
    if type(manifest_sha256) is not str or len(manifest_sha256) != 64 or any(c not in '0123456789abcdef' for c in manifest_sha256):
        raise ValueError('completed train manifest SHA required')
    verify = copy.deepcopy(job)
    verify_id = job['run_id'] + '-verify'
    manifest = str(Path(runs) / job['run_id'] / 'out/manifest.json')
    verify.update(run_id=verify_id, config=None, wall_hours=2., disk_gib=1,
                  inputs=list(dict.fromkeys([*job['inputs'], manifest])))
    argv = verify['argv']
    argv[argv.index('-m')+2] = 'verify'
    for name in ('--config', '--config-sha'):
        index = argv.index(name)
        del argv[index:index+2]
    index = argv.index('--output')
    argv[index:index] = ['--manifest', manifest, '--manifest-sha', manifest_sha256]
    replace(argv, '--output', Path(runs)/verify_id/'out')
    replace(argv, '--scratch', Path(runs)/verify_id/'scratch')
    verify['env'] = [value for value in verify['env'] if not value.startswith(
        ('VKGR_RUN_ID=', 'VKGR_PROGRESS_PATH=', 'VKGR_EXPECTED_SELECTED_USERS_SHA256='))]
    expected = next((value.split('=', 1)[1] for value in job['env']
                     if value.startswith('VKGR_EXPECTED_SELECTED_USERS_SHA256=')), None)
    _half_module(job['cwd']).check_selected_users_pin(None, expected, required=True)
    verify['env'].append('VKGR_EXPECTED_SELECTED_USERS_SHA256=' + expected)
    return verify


def identity(req):
    """Keep the original A1 inventory gate and add exact pinned delta/runtime checks."""
    validate_source_contract(req)
    root = Path(req['deployment']).resolve(strict=True)
    ops = _old_ops(root)
    original = _OLD_ORIGINALS.get(id(ops), {}).get('identity', ops.identity)
    return original(req)


def observe(req, entry):
    """Report G-half deltas against the accepted matched A1 anchor controls."""
    ops = _old_ops(req['deployment'])
    originals = _OLD_ORIGINALS.get(id(ops))
    old_observe = originals['observe'] if originals else ops.observe
    result = old_observe(req, entry)
    if result.get('status') == 'verified' and result.get('verified') is True:
        seed = entry['job']['seed']
        if seed not in CONTROL_HITS:
            raise ValueError('verified observation has no frozen anchor control for seed')
        result = dict(result)
        result['control_hits'] = CONTROL_HITS[seed]
        result['delta_hits'] = result['hits'] - CONTROL_HITS[seed]
    return result


def _anchor_context(req, run, full_context):
    ref = req['anchor_manifest']
    anchor = run._read_run(Path(ref['path']), ref['sha256'])
    if (anchor['config'].get('seed') != 42 or anchor['config'].get('sid') != 'STRUCT_C1_B48D3_S42'
            or anchor['binding'].get('task') != run.TASK_ID):
        raise ValueError('accepted seed42 STRUCT A1 anchor manifest required')
    half = _half_module(req['deployment'])
    validation = half.validate_anchor_context(full_context,
        context_binding_sha256=anchor['binding']['context_binding_sha256'],
        anchor_git=anchor['binding']['source']['git'])
    return half, anchor, validation


def prepare(req):
    """Reload original formal A1 inputs, pin the original context, then filter train examples."""
    expected = _require_expected_selected(req)
    root = Path(req['deployment']).resolve(strict=True)
    ops = _old_ops(root)
    originals = _OLD_ORIGINALS.get(id(ops))
    if not originals:
        raise ValueError('ops prepare requires the isolated new-ops adapter')
    # The original prepare must derive the unmodified anchor config before this wrapper samples rows.
    current_derive = ops.derive_job
    ops.derive_job = originals['derive_job']
    try:
        run, full_context, sid, runtime, scratch = originals['prepare'](req)
    finally:
        ops.derive_job = current_derive
    half, _, _ = _anchor_context(req, run, full_context)
    context = half.sample_context(full_context, expected_selected_users_sha256=expected)
    return run, context, sid, runtime, scratch


@contextmanager
def adapted_old_ops(req):
    """Reuse the reviewed A1 CUDA, queue, and detached-driver gates with half-user bindings."""
    ops = _old_ops(req['deployment'])
    names = ('identity', 'derive_job', 'derive_verify', 'prepare', 'start_driver', '__file__')
    originals = {name: getattr(ops, name) for name in names}
    if hasattr(ops, 'observe'):
        originals['observe'] = ops.observe
    _OLD_ORIGINALS[id(ops)] = originals

    def start_driver(request, request_path):
        old_file = ops.__file__
        ops.__file__ = str(Path(__file__).resolve())
        try:
            return originals['start_driver'](request, request_path)
        finally:
            ops.__file__ = old_file

    try:
        ops.identity = identity
        ops.derive_job = derive_job
        ops.derive_verify = derive_verify
        ops.prepare = prepare
        ops.observe = observe
        ops.start_driver = start_driver
        yield ops
    finally:
        for name, value in originals.items():
            setattr(ops, name, value)
        _OLD_ORIGINALS.pop(id(ops), None)


def _require_expected_selected(req):
    value = req.get('expected_selected_users_sha256')
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('sealed preflight selected-users SHA required before dry-run/submit')
    return value


def cuda_preview(req):
    _require_expected_selected(req)
    with adapted_old_ops(req) as ops:
        return ops.cuda_preview(req)


def dry_run(req):
    expected = _require_expected_selected(req)
    preflight = load(req['preflight_receipt'])
    check_request_selection_against_preflight(req, preflight)
    if expected != preflight['expected_selected_users_sha256']:
        raise ValueError('selected-user pin changed after formal preflight')
    with adapted_old_ops(req) as ops:
        return ops.dry_run(req)


def submit(req):
    _require_expected_selected(req)
    with adapted_old_ops(req) as ops:
        return ops.submit(req)


def driver(req, plan_sha256):
    _require_expected_selected(req)
    with adapted_old_ops(req) as ops:
        plan = ops.load(Path(req['ops']) / 'plan.json')
        preflight = load(req['preflight_receipt'])
        check_request_selection_against_preflight(req, preflight)
        if any(entry.get('job', {}).get('env', []).count('VKGR_EXPECTED_SELECTED_USERS_SHA256=' + req['expected_selected_users_sha256']) != 1
               for entry in plan.get('entries', [])):
            raise ValueError('sealed driver plan selected-user SHA drift')
        return ops.driver(req, plan_sha256)


def _compare_module_validation_pins(receipt, validation):
    required = ('context_binding_sha256', 'queries_sha256', 'validation_file_sha256')
    for key in required:
        if type(validation.get(key)) is not str or receipt.get(key) != validation[key]:
            raise ValueError('module preflight validation pin mismatch: ' + key)


def run_module_preflight(req, seed_job, expected_source_sha256):
    """Run and bind the formal -m preflight receipt before parent data loading."""
    root = Path(req['deployment']).resolve(strict=True)
    argv = list(seed_job['argv'])
    if argv.count('train') != 1:
        raise ValueError('derived training argv must contain exactly one train command')
    argv[argv.index('train')] = 'preflight'
    ops_dir = Path(req['ops']).resolve(strict=True)
    call_dir = Path(tempfile.mkdtemp(prefix='g-half-r2-cli-pf-', dir=ops_dir))
    replace(argv, '--output', call_dir / 'output')
    replace(argv, '--scratch', call_dir / 'scratch')
    anchor_ref = req['anchor_manifest']
    for flag, value in (('--anchor-manifest', anchor_ref['path']),
                        ('--anchor-manifest-sha', anchor_ref['sha256'])):
        if flag in argv:
            replace(argv, flag, value)
        else:
            argv.extend((flag, value))
    argv[0] = req['python']
    child_env = os.environ.copy()
    for item in seed_job['env']:
        key, _, value = item.partition('=')
        child_env[key] = value
    child_env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                      'MKL_THREADING_LAYER': 'SEQUENTIAL', 'KMP_USE_SHM': '0'})
    (call_dir / 'argv.json').write_text(json.dumps(argv, sort_keys=True) + '\n')
    try:
        child = subprocess.run(argv, cwd=root, env=child_env, capture_output=True,
                               text=True, check=False)
        stdout, stderr, returncode = child.stdout, child.stderr, child.returncode
    except Exception as exc:
        stdout, stderr, returncode = '', repr(exc), None
        (call_dir / 'stdout.log').write_text(stdout)
        (call_dir / 'stderr.log').write_text(stderr)
        (call_dir / 'returncode.txt').write_text('unavailable\n')
        (call_dir / 'evidence.json').write_text(json.dumps({
            'argv': argv, 'argv_sha256': hashlib.sha256(canonical(argv)).hexdigest(),
            'stdout_sha256': hashlib.sha256(stdout.encode()).hexdigest(),
            'stderr_sha256': hashlib.sha256(stderr.encode()).hexdigest(),
            'returncode': None, 'launch_error': repr(exc)}, sort_keys=True) + '\n')
        raise RuntimeError('module preflight subprocess launch failed; evidence retained at ' + str(call_dir)) from exc
    (call_dir / 'stdout.log').write_text(stdout)
    (call_dir / 'stderr.log').write_text(stderr)
    (call_dir / 'returncode.txt').write_text(str(returncode) + '\n')
    stdout_sha256 = hashlib.sha256(stdout.encode()).hexdigest()
    stderr_sha256 = hashlib.sha256(stderr.encode()).hexdigest()
    (call_dir / 'evidence.json').write_text(json.dumps({
        'argv': argv, 'argv_sha256': hashlib.sha256(canonical(argv)).hexdigest(),
        'stdout_sha256': stdout_sha256, 'stderr_sha256': stderr_sha256,
        'returncode': returncode}, sort_keys=True) + '\n')
    try:
        rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError('module preflight subprocess emitted malformed JSON; evidence retained at ' + str(call_dir)) from exc
    if returncode != 0 or len(rows) != 1:
        raise ValueError('module preflight subprocess failed or emitted an ambiguous receipt; evidence retained at ' + str(call_dir))
    receipt = rows[0]
    if (receipt.get('status') != 'preflight_ok' or receipt.get('engineering') is not False
            or receipt.get('written') is not False
            or receipt.get('training_subset', {}).get('selected_users_sha256')
            != req['expected_selected_users_sha256']
            or receipt.get('source_sha256') != expected_source_sha256):
        raise ValueError('module preflight subprocess receipt does not match sealed request/source; evidence retained at ' + str(call_dir))
    return {'argv': argv, 'returncode': returncode,
            'stdout_sha256': stdout_sha256, 'stderr_sha256': stderr_sha256,
            'argv_sha256': hashlib.sha256(canonical(argv)).hexdigest(),
            'receipt': receipt, 'call_dir': str(call_dir)}


def formal_preflight(req):
    """Reuse original verified STRUCT fit/input gates, then prove the sampled formal context."""
    root = Path(req['deployment']).resolve(strict=True)
    identity(req)
    source_identity = validate_source_contract(req)
    ops = _old_ops(root)
    # Exercise the exact module invocation used by the launcher before the parent
    # loads the full context. This receipt is intentionally only the CLI receipt;
    # the existing full three-seed formal receipt below remains authoritative.
    seed_job = derive_job(req, 42)
    cli = run_module_preflight(req, seed_job, source_identity['science_source_sha256'])
    ops = _old_ops(root)
    run, full_context, sid, runtime, scratch = ops.prepare(req)
    anchor_job = ops.derive_job(req, 42)
    anchor_ref = req['anchor_manifest']
    anchor = run._read_run(Path(anchor_ref['path']), anchor_ref['sha256'])
    if (anchor['config'].get('seed') != 42 or anchor['config'].get('sid') != 'STRUCT_C1_B48D3_S42'
            or anchor['binding'].get('task') != run.TASK_ID):
        raise ValueError('accepted seed42 STRUCT A1 anchor manifest required')
    half = _half_module(root)
    validation = half.validate_anchor_context(full_context,
        context_binding_sha256=anchor['binding']['context_binding_sha256'],
        anchor_git=anchor['binding']['source']['git'])
    _compare_module_validation_pins(cli['receipt'], validation)
    if (len(full_context.catalog), len(full_context.examples), len(full_context.queries)) != (120415, 86785, 2520):
        raise ValueError('formal anchor catalog/example/query shape changed')
    rows = []
    model_shape = None
    sample_binding_sha = None
    selected_users_sha256 = None
    expected_science_files = {**accepted_baselines(req)[1]['source_files'], **req['pinned_delta']}
    for seed in SEEDS:
        config_path = root / f'configs/g_half_users/g-half-users-s{seed}.json'
        config_sha = sha(config_path)
        config = ops.load(config_path)
        half.validate_config(config, False)
        if config['seed'] != seed or config['sid_manifest_sha256'] != req['coll_refs']['sid_manifest']['sha256']:
            raise ValueError('half-user seed/SID config differs from current full-train anchor')
        sampled = half.sample_context(full_context)
        half.validate_sampled_validation(full_context, sampled)
        current_context_sha = run._digest(sampled.binding)
        if sample_binding_sha is not None and sample_binding_sha != current_context_sha:
            raise ValueError('three training seeds must share one user subset/context')
        sample_binding_sha = current_context_sha
        selected = sampled.binding['g_half_users']['selected_users_sha256']
        if selected_users_sha256 is not None and selected_users_sha256 != selected:
            raise ValueError('the three seeds must use the same selected-user set')
        selected_users_sha256 = selected
        run.gr._seed(seed)
        model = run.make_model_arch_d(sampled.codec, config, device=runtime['device'])
        with half._patched_base()() as patched:
            binding = patched._binding(root, sampled, config, runtime, model,
                ops.flag(anchor_job['argv'], '--prepared-sha'),
                ops.flag(anchor_job['argv'], '--history-sha'), sid)
        if binding['source']['files'] != expected_science_files:
            raise ValueError('training scientific source ledger differs from accepted 41+10 pinned delta')
        if req.get('expected_selected_users_sha256') not in (None, selected_users_sha256):
            raise ValueError('sealed selected-users SHA differs from formal preflight recomputation')
        derived_req = dict(req, expected_selected_users_sha256=selected_users_sha256)
        job = derive_job(derived_req, seed)
        rows.append({'seed': seed, 'job': job, 'config_sha256': config_sha,
                     'expected_binding_sha256': run._digest(binding),
                     'data_hashes': {key: binding[key] for key in ('prepared_manifest_sha256', 'history_manifest_sha256', 'context_binding_sha256')},
                     'training_subset': sampled.binding['g_half_users'], 'source': binding['source']})
        model_shape = model.spec
        del model
    return {'status': 'preflight_ok', 'engineering': False, 'written': False,
            'module_preflight_argv': cli['argv'],
            'module_preflight_returncode': cli['returncode'],
            'module_preflight_stdout_sha256': cli['stdout_sha256'],
            'module_preflight_stderr_sha256': cli['stderr_sha256'],
            'module_preflight_argv_sha256': cli['argv_sha256'],
            'module_preflight_call_dir': cli['call_dir'],
            'module_preflight_receipt': cli['receipt'],
            'source_commit': req['source_commit'],
            'validation': validation, 'catalog_size': len(full_context.catalog), 'query_count': len(full_context.queries),
            'anchor_training_example_count': len(full_context.examples), 'data_binding_sha256': sample_binding_sha,
            'expected_selected_users_sha256': selected_users_sha256,
            'model_shape': model_shape, 'entries': rows, 'sid_version': sid.info['sid_version'],
            'sid_manifest_sha256': sid.info['manifest_sha256'], 'anchor_manifest_sha256': anchor_ref['sha256'], 'rows': rows,
            'runtime': runtime, 'scratch': str(scratch)}


def check_request_selection_against_preflight(req, preflight):
    expected = _require_expected_selected(req)
    if type(preflight) is not dict or preflight.get('status') != 'preflight_ok' \
            or preflight.get('expected_selected_users_sha256') != expected:
        raise ValueError('sealed request selected-user SHA differs from formal preflight')
    entries = preflight.get('entries')
    if type(entries) is not list or len(entries) != 3:
        raise ValueError('all three frozen preflight entries required')
    for seed, entry in zip(SEEDS, entries):
        if entry.get('seed') != seed or entry.get('training_subset', {}).get('selected_users_sha256') != expected \
                or entry.get('job', {}).get('run_id') != f'g-half-users-r2-s{seed}' \
                or f'VKGR_EXPECTED_SELECTED_USERS_SHA256={expected}' not in entry['job'].get('env', []):
            raise ValueError('preflight seed/job selected-user binding mismatch')
    return expected


def verify_stdout(path, expected_sha256, *, run_id, manifest_sha256, verify_run_id):
    """Read original verify stdout and require its direct verified run/manifest receipt."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != expected_sha256:
        raise ValueError('original verify stdout file/SHA mismatch')
    matches = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if type(value) is dict and value.get('status') == 'verified':
            matches.append(value)
    expected = {'status': 'verified', 'run_id': run_id, 'manifest_sha256': manifest_sha256,
                'verify_run_id': verify_run_id}
    if len(matches) != 1 or any(matches[0].get(k) != v for k, v in expected.items()):
        raise ValueError('stdout lacks one directly verified, run_id/manifest-bound receipt')
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('action', choices=('preflight', 'cuda-preview', 'dry-run', 'submit', 'driver', 'verify-check'))
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha')
    parser.add_argument('--plan-sha')
    parser.add_argument('--stdout')
    parser.add_argument('--stdout-sha')
    parser.add_argument('--run-id')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--verify-run-id')
    args = parser.parse_args(argv)
    if args.action == 'driver' and (not args.request_sha or not args.plan_sha):
        raise ValueError('driver requires controller-sealed request and plan SHA')
    if not args.request_sha:
        raise ValueError('pinned private request SHA required')
    req, source = validate_request_file(args.request, args.request_sha)
    if args.action == 'preflight':
        result = formal_preflight(req)
        result['source_identity'] = source
    elif args.action == 'cuda-preview':
        result = cuda_preview(req)
    elif args.action == 'dry-run':
        result = dry_run(req)
    elif args.action == 'submit':
        result = submit(req)
    elif args.action == 'driver':
        result = driver(req, args.plan_sha)
    else:
        if not all((args.stdout, args.stdout_sha, args.run_id, args.manifest_sha256, args.verify_run_id)):
            raise ValueError('verify-check requires pinned stdout/run/manifest identities')
        result = verify_stdout(args.stdout, args.stdout_sha, run_id=args.run_id,
                               manifest_sha256=args.manifest_sha256, verify_run_id=args.verify_run_id)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
