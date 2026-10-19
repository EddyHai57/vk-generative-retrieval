"""Frozen user-level half-sample adapter for the STRUCT_C1_B48D3_S42 G runner.

The original A1 runner remains byte-identical.  This module owns the only new
training-data transformation; validation queries, catalog, codec, and histories
are retained from the original checked context.
"""
from __future__ import annotations

import hashlib
import json
import random
import os
import sys
import io
from contextlib import redirect_stdout
from pathlib import Path

from .data_audit import canonical_bytes


TASK_ID = 'SID-STRUCTURE-G-HALF-USERS'
SAMPLE_SEED = 20261020
SAMPLE_FRACTION = 0.5
SAMPLE_METHOD = 'random.Random(seed).sample(sorted_training_example_user_ids, floor(U/2)); sort selected IDs'
SAMPLE_COUNT_RULE = 'floor(U/2)'
SCHEMA = 'g-sid-structure-a1-half-users-run-v1'
MANIFEST_SCHEMA = 'g-sid-structure-a1-half-users-train-manifest-v1'
MANIFEST_KIND = 'g_sid_structure_a1_half_users_train'
CHECKPOINT_SCHEMA = 'g-sid-structure-a1-half-users-checkpoint-v1'
CONFIG_KEYS = {'user_sample_fraction', 'user_sample_seed', 'user_sample_count_rule', 'user_sample_method'}
SOURCE_MEMBERS = ('src/vkgr/g_half_users_a1_run.py', 'scripts/g_half_users_ops.py',
                  'tests/test_g_half_users.py', 'tests/test_g_half_users_ops.py',
                  'scripts/g_half_users_engineering_queue.py', 'tests/test_g_half_users_engineering_queue.py',
                  'docs/superpowers/plans/2026-10-20-g-half-implementation.md')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def user_ids_sha256(user_ids):
    """SHA of an ascending integer list as compact UTF-8 JSON with no final LF."""
    if type(user_ids) is not tuple or tuple(sorted(set(user_ids))) != user_ids:
        raise ValueError('sorted unique user ID tuple required')
    raw = json.dumps(list(user_ids), separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def check_selected_users_pin(actual_sha256, expected_sha256, *, required=False):
    if expected_sha256 is None:
        if required:
            raise ValueError('sealed selected training user SHA required')
        return actual_sha256
    if type(expected_sha256) is not str or len(expected_sha256) != 64 \
            or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('sealed selected training user SHA has invalid syntax')
    if actual_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError('selected training user SHA differs from the sealed formal preflight')
    return expected_sha256 if actual_sha256 is None else actual_sha256


def training_user_ids(rows):
    """Return the canonical pool of users who have at least one train example."""
    users = set()
    for row in rows:
        if type(row) is not dict or type(row.get('user_id')) is not int or not 0 <= row['user_id'] < 2**32:
            raise ValueError('training example user_id must be uint32')
        users.add(row['user_id'])
    if not users:
        raise ValueError('nonempty training-example user pool required')
    return tuple(sorted(users))


def half_sample_size(user_count):
    """Frozen floor(U/2) sample size."""
    if type(user_count) is not int or user_count < 2:
        raise ValueError('typed user count >= 2 required')
    return user_count // 2


def select_users(user_ids, sample_count, *, seed):
    """Use the frozen isolated stdlib RNG and return selected IDs sorted."""
    if type(user_ids) is not tuple or not user_ids or tuple(sorted(set(user_ids))) != user_ids:
        raise ValueError('canonical sorted unique user pool required')
    if type(sample_count) is not int or not 1 <= sample_count < len(user_ids):
        raise ValueError('proper nonempty user subset size required')
    if type(seed) is not int or seed != SAMPLE_SEED:
        raise ValueError('frozen user selection seed required')
    return tuple(sorted(random.Random(seed).sample(user_ids, sample_count)))


def seeded_user_selection(user_ids, sample_count, *, seed):
    """Public name for the frozen isolated standard-library sampler."""
    return select_users(user_ids, sample_count, seed=seed)


def retain_training_rows(rows, selected_user_ids):
    """Keep every training example of each selected training user, preserving row order/bytes."""
    users = training_user_ids(rows)
    if type(selected_user_ids) is not tuple or not selected_user_ids or tuple(sorted(set(selected_user_ids))) != selected_user_ids \
            or not set(selected_user_ids) < set(users):
        raise ValueError('selected users must be a nonempty proper canonical subset of the training pool')
    retained = tuple(row for row in rows if row['user_id'] in set(selected_user_ids))
    if not retained or {row['user_id'] for row in retained} != set(selected_user_ids):
        raise ValueError('selected training users must retain all and only their examples')
    return retained


def sample_context(context, *, expected_selected_users_sha256=None):
    """Filter only train examples and make the unchanged validation/SID identities explicit."""
    context.check()
    rows = context.examples.rows(list(range(len(context.examples))))
    users = training_user_ids(rows)
    count = half_sample_size(len(users))
    selected = seeded_user_selection(users, count, seed=SAMPLE_SEED)
    selected_sha256 = user_ids_sha256(selected)
    check_selected_users_pin(selected_sha256, expected_selected_users_sha256)
    retained = retain_training_rows(rows, selected)
    queries = context.queries
    queries_sha = digest(queries)
    if context.binding.get('queries_sha256') != queries_sha:
        raise ValueError('validation query byte identity does not match the checked context')
    binding = dict(context.binding)
    b1 = dict(binding['b1'])
    b1['train_history_sha256'] = digest(retained)
    binding['b1'] = b1
    binding['queries_sha256'] = queries_sha
    binding['g_half_users'] = {
        'fraction': SAMPLE_FRACTION,
        'user_count_rule': SAMPLE_COUNT_RULE,
        'selection_seed': SAMPLE_SEED,
        'selection_method': SAMPLE_METHOD,
        'population_users': len(users),
        'selected_users': len(selected),
        'population_training_examples': len(rows),
        'selected_training_examples': len(retained),
        'population_users_sha256': user_ids_sha256(users),
        'selected_users_sha256': selected_sha256,
        'selected_examples_sha256': digest(retained),
    }
    from . import generative_data_v2 as gd2
    from . import generative_experiment as ge
    sampled = ge.RunContext(ge._CONTEXT_KEY, examples=gd2.ListExamples(retained), codec=context.codec,
                            queries=queries, binding=binding, stats=context.slice_definition).check()
    validate_sampled_validation(context, sampled)
    if sampled.queries != context.queries or sampled.codec is not context.codec or sampled.slice_definition is not context.slice_definition:
        raise ValueError('user sampling changed validation queries, codec, or full-train slice definition')
    return sampled


def validate_anchor_context(context, *, context_binding_sha256, anchor_git):
    """Bind the regenerated full anchor context to the accepted anchor manifest receipt."""
    if type(context_binding_sha256) is not str or len(context_binding_sha256) != 64 \
            or any(c not in '0123456789abcdef' for c in context_binding_sha256):
        raise ValueError('external anchor context-binding SHA required')
    binding = context.binding
    if type(anchor_git) is not dict or set(anchor_git) != {'git_base', 'git_dirty'} \
            or type(anchor_git['git_base']) is not str or len(anchor_git['git_base']) != 40 \
            or any(c not in '0123456789abcdef' for c in anchor_git['git_base']) or type(anchor_git['git_dirty']) is not bool:
        raise ValueError('anchor manifest source Git identity required')
    actual_binding_sha = digest(binding)
    if actual_binding_sha != context_binding_sha256:
        raise ValueError('regenerated full context differs from the accepted anchor manifest')
    queries_sha256 = digest(context.queries)
    validation_file_sha256 = binding.get('validation_file_sha256')
    for value in (queries_sha256, validation_file_sha256):
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('anchor context lacks a bound validation file/query SHA')
    if binding.get('queries_sha256') != queries_sha256:
        raise ValueError('anchor context query/history bytes do not match its binding')
    return {'context_binding_sha256': actual_binding_sha,
            'queries_sha256': queries_sha256, 'validation_file_sha256': validation_file_sha256}


def validate_sampled_validation(anchor_context, sampled_context):
    """Prove user sub-sampling leaves validation query/history bytes and source file unchanged."""
    anchor_queries = canonical_bytes(anchor_context.queries)
    sampled_queries = canonical_bytes(sampled_context.queries)
    anchor_file_sha = anchor_context.binding.get('validation_file_sha256')
    if sampled_queries != anchor_queries or digest(sampled_context.queries) != digest(anchor_context.queries) \
            or sampled_context.binding.get('validation_file_sha256') != anchor_file_sha:
        raise ValueError('user sampling changed validation query/history or source-file identity')
    return {'queries_sha256': digest(anchor_context.queries), 'validation_file_sha256': anchor_file_sha}


def default_config():
    from . import sid_structure_a1_run as base
    value = base.default_config()
    value.update(schema=SCHEMA, task_id=TASK_ID,
                 user_sample_fraction=SAMPLE_FRACTION,
                 user_sample_seed=SAMPLE_SEED,
                 user_sample_count_rule=SAMPLE_COUNT_RULE,
                 user_sample_method=SAMPLE_METHOD)
    return value


def validate_config(value, engineering=False, *, base_validator=None):
    if type(value) is not dict or not CONFIG_KEYS <= set(value):
        raise ValueError('registered user-sampling fields required')
    if (type(value['user_sample_fraction']) is not float or value['user_sample_fraction'] != SAMPLE_FRACTION
            or type(value['user_sample_seed']) is not int or value['user_sample_seed'] != SAMPLE_SEED
            or value['user_sample_count_rule'] != SAMPLE_COUNT_RULE
            or value['user_sample_method'] != SAMPLE_METHOD):
        raise ValueError('frozen user-sampling contract changed')
    from . import sid_structure_a1_run as base
    old_schema, old_task = base.SCHEMA, base.TASK_ID
    base.SCHEMA, base.TASK_ID = SCHEMA, TASK_ID
    try:
        stripped = {k: v for k, v in value.items() if k not in CONFIG_KEYS}
        (base.validate_config if base_validator is None else base_validator)(stripped, engineering)
    finally:
        base.SCHEMA, base.TASK_ID = old_schema, old_task
    return value


def _patched_base():
    """Patch the isolated legacy A1 engine for this process, restoring all globals on exit."""
    from contextlib import contextmanager
    from . import sid_structure_a1_run as base

    @contextmanager
    def patched():
        names = ('SCHEMA', 'TASK_ID', 'MANIFEST_SCHEMA', 'MANIFEST_KIND', 'CHECKPOINT_SCHEMA',
                 'default_config', 'validate_config', '_contexts', '_source', '_binding')
        saved = {name: getattr(base, name) for name in names}
        original_contexts = base._contexts
        original_source = base._source
        original_binding = base._binding
        original_validator = base.validate_config

        def check_config(config, engineering=False):
            return validate_config(config, engineering, base_validator=original_validator)

        def contexts(*args, **kwargs):
            context, engineering, sid = original_contexts(*args, **kwargs)
            expected = os.environ.get('VKGR_EXPECTED_SELECTED_USERS_SHA256')
            return sample_context(context, expected_selected_users_sha256=expected), engineering, sid

        def source(root):
            value = original_source(root)
            files = dict(value['files'])
            for name in SOURCE_MEMBERS:
                files[name] = base.file_sha256(Path(root) / name)
            for seed in (42, 2024, 2025):
                name = f'configs/g_half_users/g-half-users-s{seed}.json'
                files[name] = base.file_sha256(Path(root) / name)
            return {'files': files, 'source_sha256': base._digest(files), 'git': value['git']}

        def binding(*args, **kwargs):
            value = original_binding(*args, **kwargs)
            value['g_half_users'] = args[1].binding['g_half_users']
            return value

        try:
            base.SCHEMA = SCHEMA
            base.TASK_ID = TASK_ID
            base.MANIFEST_SCHEMA = MANIFEST_SCHEMA
            base.MANIFEST_KIND = MANIFEST_KIND
            base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
            base.validate_config = check_config
            base._contexts = contexts
            base._source = source
            base._binding = binding
            yield base
        finally:
            for name, value in saved.items():
                setattr(base, name, value)
    return patched


def _preflight(argv):
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--prepared-sha', required=True)
    parser.add_argument('--history-manifest', required=True)
    parser.add_argument('--history-sha', required=True)
    parser.add_argument('--sid-manifest', required=True)
    parser.add_argument('--sid-sha', required=True)
    parser.add_argument('--content', required=True)
    parser.add_argument('--content-sha', required=True)
    for extra in ('input-manifest', 'input-sha', 'vectors', 'vectors-sha'):
        parser.add_argument('--' + extra, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--anchor-manifest', required=True)
    parser.add_argument('--anchor-manifest-sha', required=True)
    args = parser.parse_args(argv)
    from . import sid_structure_a1_run as base
    from . import generative_data as gd
    config = gd._json(gd._checked_file(Path(args.config), args.config_sha))
    validate_config(config, False)
    root = Path(__file__).resolve().parents[2]
    output = gd._no_symlink(Path(args.output))
    if output.exists():
        raise ValueError('preflight output must not exist')
    base_config = {key: value for key, value in config.items() if key not in CONFIG_KEYS}
    base_config.update(schema=base.SCHEMA, task_id=base.TASK_ID)
    common = (root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha,
              Path(args.sid_manifest), args.sid_sha, Path(args.content), args.content_sha,
              base_config, Path(args.scratch))
    extra = {'input_manifest': Path(args.input_manifest), 'input_sha': args.input_sha,
             'vectors_path': Path(args.vectors), 'vectors_sha': args.vectors_sha}
    full_context, engineering, sid = base._contexts(*common, **extra)
    if engineering is not False or full_context.binding.get('synthetic') is not False or sid.info.get('engineering') is not False:
        raise ValueError('formal real-input preflight requires engineering=False')
    if sid.info.get('sid_version') != 'STRUCT_C1_B48D3_S42':
        raise ValueError('frozen full-train SID mapping required')
    anchor = base._read_run(Path(args.anchor_manifest), args.anchor_manifest_sha)
    if (anchor['config'].get('seed') != 42 or anchor['config'].get('sid') != 'STRUCT_C1_B48D3_S42'
            or anchor['binding'].get('task') != base.TASK_ID):
        raise ValueError('accepted seed42 STRUCT A1 anchor manifest required')
    validation_pins = validate_anchor_context(full_context,
        context_binding_sha256=anchor['binding']['context_binding_sha256'],
        anchor_git=anchor['binding']['source']['git'])
    context = sample_context(full_context)
    validate_sampled_validation(full_context, context)
    if (len(full_context.catalog), len(full_context.examples), len(full_context.queries)) != (120415, 86785, 2520):
        raise ValueError('formal anchor catalog/example/query shape changed')
    with _patched_base()() as patched:
        print(json.dumps({'status': 'preflight_ok', 'engineering': False, 'written': False,
                          'seed': config['seed'], 'query_count': len(context.queries),
                          'catalog_size': len(context.catalog), **validation_pins,
                          'sid_manifest_sha256': sid.info['manifest_sha256'],
                          'training_subset': context.binding['g_half_users'],
                          'source_sha256': patched._source(root)['source_sha256']},
                         sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    """Keep A1's train/verify CLI while adding a sampled real-path preflight."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == 'preflight':
        return _preflight(values[1:])
    from . import sid_structure_a1_run as base
    if not values or values[0] not in ('train', 'verify'):
        raise SystemExit('registered commands are preflight, train, and verify')
    run_id = os.environ.get('VKGR_RUN_ID')
    if type(run_id) is not str or not run_id:
        raise ValueError('bound VKGR_RUN_ID required for train/verify')
    expected_users_sha = os.environ.get('VKGR_EXPECTED_SELECTED_USERS_SHA256')
    check_selected_users_pin(None, expected_users_sha, required=True)
    if values[0] == 'verify':
        def flag(name):
            if values.count(name) != 1:
                raise ValueError('exactly one argument required: ' + name)
            return values[values.index(name)+1]
        verify_path, verify_sha = flag('--manifest'), flag('--manifest-sha')
        train_id = Path(verify_path).parent.parent.name
    with _patched_base()() as _:
        captured = io.StringIO()
        with redirect_stdout(captured):
            code = base.main(values)
        raw = captured.getvalue().strip().splitlines()
        if code != 0:
            if captured.getvalue():
                print(captured.getvalue(), end='')
            return code
        try:
            record = json.loads(raw[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ValueError('A1 engine emitted no parseable result receipt') from exc
        if values[0] == 'train':
            receipt = {'status': record.get('status'), 'run_id': run_id,
                       'manifest_sha256': record.get('manifest_sha256'), 'best_epoch': record.get('best_epoch')}
        else:
            expected_verify_id = run_id
            if (record.get('status') != 'verified' or record.get('run_id') != train_id
                    or record.get('verify_run_id') != expected_verify_id
                    or record.get('manifest_sha256') != verify_sha
                    or Path(verify_path).name != 'manifest.json'):
                raise ValueError('direct verified stdout receipt is not bound to the requested run/manifest/verify job')
            receipt = {'status': 'verified', 'run_id': train_id,
                       'verify_run_id': expected_verify_id, 'manifest_sha256': verify_sha,
                       'best_epoch': record.get('best_epoch'), 'replayed_epochs': record.get('replayed_epochs')}
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
        return code
