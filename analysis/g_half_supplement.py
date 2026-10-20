"""Bounded read-only G user-half supplementary report. No model imports."""
import argparse
import datetime
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
HISTORY_PIN = '8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43'
BASE_QUERIES_CANONICAL_DIGEST = '9077a1c9495d494bc8ec209c0bf3305fd68db5bfcca24abcadc4a1e552e60a97'
SLICES_PIN = '6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26'
SELECTED_USERS_SHA = '94c72d1db3e5b3e39de75c91b1e0163ca4df68b9439704ad4548cd71738e06d8'
POPULATION_USERS_SHA = 'e382450f9b92830600c4eda85f77dfab087cbf06918eaa21d21caa63b50096b4'
SELECTED_EXAMPLES_SHA = '304914f943a7e90afdfd88f56d074832cb68b39e546f1d3535ec9ed1ef76aacc'
HALF_PREPARED_MANIFEST_SHA = 'a55df0a3b285ce4cf623126fe53df53378bf5d7c0645c6484f0f405641bddd82'
BIN_DENOMINATORS = {'high': 963, 'mid': 548, 'low': 155, 'unsupported': 231, 'oov': 623}
N = 2520
SEEDS = (42, 2024, 2025)
HALF_RUNS = {42: 'g-half-users-r2-s42', 2024: 'g-half-users-r2-s2024', 2025: 'g-half-users-r2-s2025'}
FULL_RUNS = {42: 'g-structure-a1-s42', 2024: 'g-structure-a1-s2024', 2025: 'g-structure-a1-s2025'}
P_RUNS = {42: 'p-seqm1-mean-s42', 2024: 'p-seqm1-mean-s2024', 2025: 'p-seqm1-mean-s2025'}
HALF_MANIFESTS = {
    42: 'b515c5a10bc366646be7d87890d4c8d8c3724d650bd52d470c77bafa0894b51d',
    2024: 'c21894b5290ea89215756423c87d181454e6467adafdb17050f2fe20194ad86d',
    2025: 'f66c78912a69ebd5f3b1a92e3f71de9b4b35bdc8d0711568b924cfca3f9fcbdf',
}
FULL_MANIFESTS = {
    42: '2f6744f4cafae340666410b87912b8cf1c71d9447c55b740f1280f2c52afefae',
    2024: 'a5adbc141ca6a55c7edade908e990dc19ab3048b26ba744b9f1ce783c46cdbb2',
    2025: 'ca87efedda954d3df7342551d8fb4790b71710b029a23886d95e3e58263f7f0b',
}
P_MANIFESTS = {
    42: 'bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e',
    2024: '8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8',
    2025: '783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb',
}
HALF_STDOUT = {
    42: '23f5812b86597e2404dabb93a769cc697ace9a52b8966fae417f151c3c98d18a',
    2024: 'a3ae26cb0a68bac18fda03acde2d45494177ff0bfe3db75d103e0519791ac170',
    2025: '3fdccd6236bc872ce9cb9de2d220c5132ed625a3d547d5c50b6c530c063fe1d9',
}
FULL_STDOUT = {
    42: 'f6bf1588ca2761dca116392776b68c73b38e66f61392c1803d7136a332c701ae',
    2024: 'f17e10a9856ddc1b2fdc0a1929e1f1c6b1089f973e6d6ae72a71f864f10e67bf',
    2025: '7175023924828773c6c5795ddc93a11042a6f1a266ae8b6899c2dc56757166e1',
}
P_STDOUT = {
    42: '12ef936b63e42a289b3c1d61a04499c375cf55c3a4d65db3b6a534cc9b6bb18c',
    2024: '0422a9b5d324ff79b3d37c51fbfbff09240ed2bea9b98589dc3294f211aaf591',
    2025: 'cd0eeb35e2f28038dc9374e9648a9d74174ca360f3c85f5ac07a3ecb7a8583ec',
}
VERIFY_STDOUT_PINS = {**{HALF_RUNS[s]: HALF_STDOUT[s] for s in SEEDS},
                      **{FULL_RUNS[s]: FULL_STDOUT[s] for s in SEEDS},
                      **{P_RUNS[s]: P_STDOUT[s] for s in SEEDS}}
ALL_IDS = tuple(HALF_RUNS[s] for s in SEEDS) + tuple(FULL_RUNS[s] for s in SEEDS) + tuple(P_RUNS[s] for s in SEEDS)
SOURCE_SCHEMA = 'g-half-supplement-request-v1'
MAX_REQUEST = 1024 * 1024
MAX_JSON = 32 * 1024 * 1024
MAX_EXAMPLES = 256 * 1024 * 1024


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def selected_user_digest(user_ids):
    ids = sorted(set(user_ids))
    if any(type(x) is not int or x < 0 or x >= 2**32 for x in ids):
        raise ValueError('selected user IDs must be uint32 integers')
    raw = json.dumps(ids, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def select_training_users(user_ids):
    ordered = tuple(sorted(set(user_ids)))
    if len(ordered) < 2 or any(type(x) is not int or not 0 <= x < 2**32 for x in ordered):
        raise ValueError('at least two uint32 training users required')
    selected = tuple(sorted(random.Random(20261020).sample(ordered, len(ordered)//2)))
    return ordered, selected


def pair_seed_hits(seed, half_by_seed, full_by_seed, p_by_seed):
    if seed not in SEEDS or any(seed not in mapping for mapping in (half_by_seed, full_by_seed, p_by_seed)):
        raise ValueError('same-seed half/full/P evidence required')
    return half_by_seed[seed], full_by_seed[seed], p_by_seed[seed]


def aggregate_seed(queries, half_hits, full_hits, p_hits, bins, selected_users, expected_bin_denominators=BIN_DENOMINATORS):
    n = len(queries)
    if not n or any(len(v) != n for v in (half_hits, full_hits, p_hits, bins)):
        raise ValueError('query coverage mismatch')
    if set(expected_bin_denominators) != set(BIN_DENOMINATORS):
        raise ValueError('frozen bin denominator schema mismatch')
    if any(x not in (0, 1) for values in (half_hits, full_hits, p_hits) for x in values):
        raise ValueError('hit vectors must be binary')
    for q, bin_name in zip(queries, bins):
        uid = q.get('user_id') if type(q) is dict else None
        if type(uid) is not int or not 0 <= uid < 2**32:
            raise ValueError('validation query user_id must be uint32')
        if bool(q.get('oov')) != (bin_name == 'oov'):
            raise ValueError('OOV query axis differs from heat-bin axis')
    if any(bool(h) and bool(q.get('oov')) for values in (half_hits, full_hits, p_hits) for h, q in zip(values, queries)):
        raise ValueError('OOV hit violates structural zero')
    def stats(indices):
        count = len(indices)
        half = sum(half_hits[i] for i in indices)
        full = sum(full_hits[i] for i in indices)
        return {'denominator': count, 'half_hits': half, 'full_hits': full, 'delta_hits': half-full}
    all_indices = list(range(n))
    in_indices = [i for i, q in enumerate(queries) if not q.get('oov')]
    oov_indices = [i for i, q in enumerate(queries) if q.get('oov')]
    bins_seen = {name: [i for i, label in enumerate(bins) if label == name] for name in BIN_DENOMINATORS}
    if any(label not in bins_seen for label in bins):
        raise ValueError('unknown frozen heat bin')
    hotness = {name: {'denominator': expected_bin_denominators[name],
                      'half_hits': sum(half_hits[i] for i in idx),
                      'full_hits': sum(full_hits[i] for i in idx),
                      'delta_hits': sum(half_hits[i]-full_hits[i] for i in idx)}
               for name, idx in bins_seen.items()}
    selected_idx = [i for i, q in enumerate(queries) if q.get('user_id') in selected_users]
    control_idx = [i for i, q in enumerate(queries) if q.get('user_id') not in selected_users]
    group = {'selected': stats(selected_idx), 'not_selected': stats(control_idx)}
    delta = sum(half_hits[i]-full_hits[i] for i in all_indices)
    if sum(x['delta_hits'] for x in group.values()) != delta or sum(x['denominator'] for x in group.values()) != n:
        raise ValueError('user group delta does not conserve total delta')
    if sum(x['half_hits'] for x in group.values()) != sum(half_hits) or sum(x['full_hits'] for x in group.values()) != sum(full_hits):
        raise ValueError('user group hits do not conserve totals')
    if sum(v['delta_hits'] for v in hotness.values()) != delta:
        raise ValueError('heat-bin deltas do not conserve total delta')
    only = sum(bool(half_hits[i]) and not bool(p_hits[i]) for i in all_indices)
    catalog_delta = sum(half_hits[i]-full_hits[i] for i in in_indices)
    oov_delta = sum(half_hits[i]-full_hits[i] for i in oov_indices)
    require(catalog_delta+oov_delta == delta, 'catalog/OOV deltas do not conserve total delta')
    return {
        'population': {'all': n, 'in_catalog': len(in_indices), 'oov': len(oov_indices)},
        'half_hits': {'all': sum(half_hits), 'in_catalog': sum(half_hits[i] for i in in_indices), 'oov': sum(half_hits[i] for i in oov_indices)},
        'full_hits': {'all': sum(full_hits), 'in_catalog': sum(full_hits[i] for i in in_indices), 'oov': sum(full_hits[i] for i in oov_indices)},
        'delta_hits': {'all': delta, 'in_catalog': sum(half_hits[i]-full_hits[i] for i in in_indices), 'oov': sum(half_hits[i]-full_hits[i] for i in oov_indices)},
        'hotness': hotness,
        'user_groups': group,
        'g_only_vs_p': {'all': only, 'in_catalog': sum(bool(half_hits[i]) and not bool(p_hits[i]) for i in in_indices), 'oov': sum(bool(half_hits[i]) and not bool(p_hits[i]) for i in oov_indices)},
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_source_commit(expected, source_sha):
    here = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(here), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(here), 'status', '--porcelain', '--untracked-files=all'], text=True)
    require(commit == expected and not status, 'clean pinned report source required')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_sha, 'report source bytes differ')


def assert_nonsymlink_path(path, root):
    p, root = Path(path), Path(root)
    require(p.is_absolute() and '..' not in p.parts and (p == root or root in p.parents), 'registered absolute receipt path required')
    current = p
    while True:
        require(not current.is_symlink(), 'receipt symlink refused')
        if current == root:
            return p
        current = current.parent


def safe_path(value, *, allowed_roots, max_bytes=MAX_JSON, suffix='.json'):
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts and p.suffix == suffix, 'absolute pinned path required')
    require(not any(part.lower().find('test') >= 0 or part.lower().find('checkpoint') >= 0 or part.lower().find('embedding') >= 0 for part in p.parts), 'sensitive path refused')
    roots = [Path(root) for root in allowed_roots]
    root = next((candidate for candidate in roots if p == candidate or candidate in p.parents), None)
    require(root is not None, 'path outside registered roots')
    current = p
    while True:
        require(not current.is_symlink(), 'symlink path refused')
        if current == root:
            break
        current = current.parent
    require(p.is_file() and p.stat().st_size <= max_bytes, 'bounded regular input required')
    return p


def load_pinned_json(ref, allowed_roots):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'closed path/SHA ref required')
    require(re.fullmatch('[a-f0-9]{64}', ref['sha256']) is not None, 'SHA256 required')
    p = safe_path(ref['path'], allowed_roots=allowed_roots)
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'input SHA mismatch')
    return json.loads(raw)


def validate_request(req):
    fields = {'schema', 'action', 'source_commit', 'source_sha256', 'inputs'}
    require(type(req) is dict and set(req) == fields, 'closed supplement request fields required')
    require(req['schema'] == SOURCE_SCHEMA and req['action'] == 'report', 'fixed report request schema/action required')
    require(type(req['source_commit']) is str and re.fullmatch('[a-f0-9]{40}', req['source_commit']), 'pinned source commit required')
    require(type(req['source_sha256']) is str and re.fullmatch('[a-f0-9]{64}', req['source_sha256']), 'report script SHA required')
    require(type(req['inputs']) is dict and set(req['inputs']) == {'history', 'catalog', 'queries', 'slices', 'examples', 'run_roots'}, 'fixed input registry required')
    for key in ('history', 'catalog', 'queries', 'slices', 'examples'):
        ref = req['inputs'][key]
        require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'closed path/SHA source reference required')
        require(type(ref['sha256']) is str and re.fullmatch('[a-f0-9]{64}', ref['sha256']), 'input SHA syntax required')
    expected_paths = {
        'history': '/workspace/vkgr-infra01-root/runs/h2-prepare/out/manifest.json',
        'catalog': '/workspace/vkgr-p4b-20261007t-pipeline-03/prepared-v3/catalog.json',
        'queries': '/workspace/vkgr-infra01-root/runs/h2-prepare/out/validation-l50.json',
        'slices': '/workspace/vkgr-infra01-root/runs/h2-prepare/out/train-slices.json',
        'examples': '/workspace/vkgr-p4b-20261007t-pipeline-03/prepared-v3/examples.jsonl',
    }
    for key, path in expected_paths.items():
        require(req['inputs'][key].get('path') == path, 'fixed '+key+' source path required')
    require(req['inputs']['history']['sha256'] == HISTORY_PIN, 'frozen history SHA differs')
    require(req['inputs']['slices']['sha256'] == SLICES_PIN, 'frozen train-only slices SHA differs')
    require(req['inputs']['catalog']['sha256'] == '80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3', 'frozen catalog SHA differs')
    require(req['inputs']['examples']['sha256'] == 'c1476db25680b30969343189f3e63cccddeff504bf9b073ae1f9727d42440b62', 'prepared examples SHA differs')
    require(type(req['inputs']['run_roots']) is dict and set(req['inputs']['run_roots']) == {'half', 'full', 'p'}, 'run roots by role required')
    for root in req['inputs']['run_roots'].values():
        require(root == str(ROOT), 'fixed original run root required')
    return req


def read_training_population(ref):
    p = Path(ref['path'])
    require(p.is_absolute() and p.name == 'examples.jsonl' and not p.is_symlink(), 'only pinned prepared examples.jsonl may be read')
    require(not any(x.is_symlink() for x in p.parents) and p.stat().st_size <= MAX_EXAMPLES, 'symlink or oversized prepared examples')
    raw_hash = hashlib.sha256()
    users = set()
    with p.open('rb') as stream:
        for line in stream:
            raw_hash.update(line)
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError('invalid prepared examples row') from exc
            uid = row.get('user_id') if type(row) is dict else None
            require(type(uid) is int and 0 <= uid < 2**32, 'training example user_id must be uint32')
            users.add(uid)
    require(raw_hash.hexdigest() == ref['sha256'], 'prepared examples physical SHA differs')
    ordered, selected = select_training_users(users)
    require(len(ordered) == 10863 and selected_user_digest(ordered) == POPULATION_USERS_SHA, 'prepared user population differs')
    require(len(selected) == 5431 and selected_user_digest(selected) == SELECTED_USERS_SHA, 'frozen selected users SHA differs')
    # The registered selected-examples SHA describes history_v2 overlay rows;
    # this raw prepared JSONL is only the frozen user/count source.
    selected_set = set(selected)
    total = retained = 0
    raw_hash2 = hashlib.sha256()
    with p.open('rb') as stream:
        for line in stream:
            raw_hash2.update(line)
            row = json.loads(line)
            total += 1
            if row['user_id'] in selected_set:
                retained += 1
    require(raw_hash2.hexdigest() == ref['sha256'], 'prepared examples changed during read')
    require((total, retained) == (86785, 43060), 'frozen training sample counts differ')
    return {'population_users': len(ordered), 'selected_users': len(selected),
            'population_training_examples': total, 'selected_training_examples': retained,
            'population_users_sha256': POPULATION_USERS_SHA,
            'selected_users_sha256': SELECTED_USERS_SHA}, selected_set


def derive_bins(queries, slices, catalog):
    thresholds = slices.get('popularity_thresholds', {})
    median, p90 = thresholds.get('median'), thresholds.get('p90')
    require(median == 1 and p90 == 11, 'frozen train-only heat thresholds differ')
    counts = {int(k): v for k, v in slices.get('item_like_counts', {}).items()}
    bins = []
    for q in queries:
        target = q.get('target')
        if type(target) is not int:
            raise ValueError('integer target required')
        require(type(q.get('oov')) is bool and q['oov'] == (target not in catalog), 'query OOV label/catalog membership differs')
        if q['oov']:
            bins.append('oov')
        else:
            count = counts.get(target, 0)
            bins.append('unsupported' if count == 0 else 'low' if count <= median else 'mid' if count <= p90 else 'high')
    actual = {k: bins.count(k) for k in BIN_DENOMINATORS}
    require(actual == BIN_DENOMINATORS, 'fixed train-only heat-bin denominators differ')
    require(sum(bool(q.get('oov')) for q in queries) == 623, 'fixed OOV population differs')
    return bins


def guarded_archive(req):
    """Import the pinned reader only after restricting all JSON reads to named roots."""
    from analysis import phase2_validation_read as archive
    run_roots = {Path(x) for x in req['inputs']['run_roots'].values()}
    history_path = Path(req['inputs']['history']['path'])
    catalog_path = Path(req['inputs']['catalog']['path'])
    allowed_fixed = {history_path, catalog_path, Path(req['inputs']['queries']['path']), Path(req['inputs']['slices']['path'])}
    original = archive.read
    def guarded_read(path, pin=None):
        lexical = Path(path)
        require(not lexical.is_symlink() and '..' not in lexical.parts, 'archive lexical path refused')
        require(lexical in allowed_fixed or any(root == lexical or root in lexical.parents for root in run_roots), 'unregistered archive path refused')
        require(lexical.suffix == '.json' and not any(w in lexical.name.lower() for w in ('test', 'checkpoint', 'embedding')), 'sensitive archive path refused')
        require(not any(parent.is_symlink() for parent in lexical.parents), 'archive symlink refused')
        if lexical in allowed_fixed:
            ref = next(r for r in req['inputs'].values() if isinstance(r, dict) and r.get('path') == str(lexical))
            require(pin == ref['sha256'] or pin is None, 'fixed archive pin differs')
        else:
            require(any(root in lexical.parents for root in run_roots) or lexical in allowed_fixed, 'archive path outside registered run roots')
        return original(path, pin)
    archive.read = guarded_read
    return archive


def checked_state(archive, rid, root, manifest_sha, stdout_sha):
    precheck_run_files(root, rid)
    state = archive.inspect_run(rid)
    require(state['verified'] is True and state['manifest_sha256'] == manifest_sha, 'raw verified/manifest identity differs')
    queue = state.get('queue', {})
    require(len(queue.get('claimed', [])) == 1 and not queue.get('pending') and not queue.get('rejected'), 'run queue terminal binding differs')
    verify_queue = archive.queue(rid+'-verify')
    require(len(verify_queue.get('claimed', [])) == 1 and not verify_queue.get('pending') and not verify_queue.get('rejected'), 'verify queue terminal binding differs')
    require(len(state.get('verify', {}).get('jobs', [])) == 1, 'exactly one original verify receipt required')
    validate_raw_stdout(rid, root, stdout_sha, manifest_sha)
    return state


def precheck_run_files(root, run_id):
    root = Path(root)
    run_dir = root/'runs'/run_id
    verify_dir = root/'runs'/(run_id+'-verify')
    for path in (run_dir, run_dir/'terminal.json', run_dir/'out/manifest.json',
                 verify_dir, verify_dir/'terminal.json', verify_dir/'logs/stdout.log'):
        assert_nonsymlink_path(path, root)
    require(run_dir.is_dir() and (run_dir/'terminal.json').is_file() and (run_dir/'out/manifest.json').is_file(), 'original run receipts missing')
    require(verify_dir.is_dir() and (verify_dir/'terminal.json').is_file() and (verify_dir/'logs/stdout.log').is_file(), 'original verifier receipts missing')
    for identity in (run_id, run_id+'-verify'):
        for status in ('pending', 'claimed', 'rejected'):
            folder = root/'queue'/status
            assert_nonsymlink_path(folder, root)
            paths = sorted(folder.glob('*-'+identity+'.json'))
            for path in paths:
                assert_nonsymlink_path(path, root)
            if status in ('pending', 'rejected'):
                require(not paths, 'original queue has pending or rejected entries')
            else:
                require(len(paths) == 1, 'one unique claimed job required')


def validate_raw_stdout(run_id, root, expected_sha, expected_manifest_sha):
    run_dir = Path(root) / 'runs' / (run_id + '-verify')
    raw_path = run_dir / 'logs/stdout.log'
    require(not raw_path.is_symlink() and not any(p.is_symlink() for p in raw_path.parents), 'train stdout symlink refused')
    raw = raw_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, f'{run_id}-verify stdout SHA differs')
    records = []
    for line in raw.decode('utf-8').splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict): records.append(item)
        except ValueError:
            continue
    if run_id in P_RUNS.values():
        # Legacy P verifier stdout omits run/manifest fields; the exact byte pin
        # plus archive reader's claimed job --manifest-sha binding is mandatory.
        valid = any(x.get('status') == 'verified' for x in records)
    else:
        valid = any(x.get('status') == 'verified' and x.get('run_id') == run_id and
                    x.get('verify_run_id', run_id+'-verify') == run_id+'-verify' and
                    x.get('manifest_sha256', x.get('run_manifest_sha')) == expected_manifest_sha for x in records)
    require(valid, f'{run_id}-verify stdout lacks verified manifest-bound record')


def validate_half_manifest(archive, root, run_id, manifest_sha):
    manifest, _ = archive.read(Path(root)/'runs'/run_id/'out/manifest.json', manifest_sha)
    g = manifest.get('binding', {}).get('g_half_users', {})
    expected = {'fraction': 0.5, 'population_users': 10863, 'selected_users': 5431,
                'population_training_examples': 86785, 'selected_training_examples': 43060,
                'population_users_sha256': POPULATION_USERS_SHA, 'selected_users_sha256': SELECTED_USERS_SHA,
                'selected_examples_sha256': SELECTED_EXAMPLES_SHA,
                'selection_method': 'random.Random(seed).sample(sorted_training_example_user_ids, floor(U/2)); sort selected IDs',
                'selection_seed': 20261020, 'user_count_rule': 'floor(U/2)'}
    require(all(g.get(key) == value for key, value in expected.items()), 'half-arm manifest user-sample binding differs')
    require(manifest.get('binding', {}).get('prepared_manifest_sha256') == HALF_PREPARED_MANIFEST_SHA,
            'half-arm prepared descriptor binding differs')
    return g['selected_examples_sha256']


def precheck_prediction_paths(archive, run_id, manifest_sha):
    manifest_path = archive.ROOT/'runs'/run_id/'out/manifest.json'
    manifest, _ = archive.read(manifest_path, manifest_sha)
    best = manifest['selection']['best_epoch']
    base = manifest_path.parent
    if 'evaluations' in manifest:
        entries = [x for x in manifest['evaluations'] if x['epoch'] == best]
        require(len(entries) == 1 and entries[0]['predictions_ref'] == f'predictions-epoch-{best:03d}.json', 'P prediction archive role differs')
        lexical = base/entries[0]['predictions_ref']
        require(not lexical.is_symlink() and lexical.parent == base and not any(x.is_symlink() for x in lexical.parents), 'P prediction lexical path unsafe')
    else:
        entries = [x for x in manifest['epochs'] if x['epoch'] == best]
        require(len(entries) == 1 and entries[0]['evaluation']['path'] == f'evaluation-{best:03d}', 'G evaluation archive role differs')
        evdir = base/entries[0]['evaluation']['path']
        evmanifest = evdir/'manifest.json'
        require(not evdir.is_symlink() and not evmanifest.is_symlink() and not any(x.is_symlink() for x in evmanifest.parents), 'G evaluation lexical path unsafe')
        ev, _ = archive.read(evmanifest, entries[0]['evaluation']['manifest_sha256'])
        for part in ev['shards']:
            lexical = Path(part['path'])
            require(not lexical.is_symlink() and lexical.parent == evdir and lexical.name == f"shard-{part['start']:06d}-{part['end']:06d}.json" and not any(x.is_symlink() for x in lexical.parents), 'G shard lexical path unsafe')


def execute_report(req):
    req = validate_request(req)
    check_source_commit(req['source_commit'], req['source_sha256'])
    base_roots = {Path(req['inputs']['history']['path']).parent, Path(req['inputs']['catalog']['path']).parent}
    hist = load_pinned_json(req['inputs']['history'], base_roots)
    require(hist.get('artifacts', {}).get('validation-l50.json') == req['inputs']['queries']['sha256'], 'history query artifact pin differs')
    require(hist.get('artifacts', {}).get('train-slices.json') == req['inputs']['slices']['sha256'], 'history train-slices pin differs')
    queries = load_pinned_json(req['inputs']['queries'], {Path(req['inputs']['queries']['path']).parent})
    slices = load_pinned_json(req['inputs']['slices'], {Path(req['inputs']['slices']['path']).parent})
    catalog = load_pinned_json(req['inputs']['catalog'], {Path(req['inputs']['catalog']['path']).parent})
    archive = guarded_archive(req)
    require(isinstance(queries, list) and len(queries) == N and isinstance(catalog, list) and len(catalog) == 120415, 'frozen validation/catalog shape differs')
    require(len(set(catalog)) == len(catalog), 'catalog duplicates')
    require(archive.plain_digest(catalog) == hist.get('catalog_sha256') and req['inputs']['catalog']['sha256'] == hist.get('binding', {}).get('catalog_sha'), 'frozen catalog canonical binding differs')
    require(hist.get('binding', {}).get('query_sha') == BASE_QUERIES_CANONICAL_DIGEST,
            'frozen base-query canonical identity differs')
    hist['_base_queries_canonical_digest'] = hist['binding']['query_sha']
    hist['_g_catalog_digest'] = archive.digest(catalog)
    data, summaries, records, user_sample_manifest_pins = {}, {}, {}, {}
    role_data = [('half', HALF_RUNS, HALF_MANIFESTS, HALF_STDOUT), ('full', FULL_RUNS, FULL_MANIFESTS, FULL_STDOUT), ('p', P_RUNS, P_MANIFESTS, P_STDOUT)]
    for role, ids, pins, stdout_pins in role_data:
        for seed in SEEDS:
            rid = ids[seed]
            root = req['inputs']['run_roots'][role]
            require(Path(root).is_absolute() and not Path(root).is_symlink(), 'fixed nonsymlink run root required')
            evidence = checked_state(archive, rid, root, pins[seed], VERIFY_STDOUT_PINS[rid])
            archive.ROOT = Path(root)
            precheck_prediction_paths(archive, rid, pins[seed])
            records[rid] = evidence
            data[rid] = archive.load_archive(rid, queries, hist, evidence)
            if role == 'half':
                user_sample_manifest_pins[rid] = validate_half_manifest(archive, root, rid, pins[seed])
            summaries[rid] = data[rid]['summary']
    user_stats, selected_users = read_training_population(req['inputs']['examples'])
    require(user_stats['population_users'] == 10863 and user_stats['selected_users'] == 5431, 'frozen user split differs')
    bins = derive_bins(queries, slices, set(catalog))
    by_seed = {}
    half_by_seed = {seed: data[HALF_RUNS[seed]]['hits'] for seed in SEEDS}
    full_by_seed = {seed: data[FULL_RUNS[seed]]['hits'] for seed in SEEDS}
    p_by_seed = {seed: data[P_RUNS[seed]]['hits'] for seed in SEEDS}
    for seed in SEEDS:
        half_hits, full_hits, p_hits = pair_seed_hits(seed, half_by_seed, full_by_seed, p_by_seed)
        by_seed[str(seed)] = aggregate_seed(queries, half_hits, full_hits, p_hits, bins, selected_users)
        require(by_seed[str(seed)]['population'] == {'all': 2520, 'in_catalog': 1897, 'oov': 623}, 'query population split differs')
        require(by_seed[str(seed)]['delta_hits']['all'] == data[HALF_RUNS[seed]]['summary']['hits'] - data[FULL_RUNS[seed]]['summary']['hits'], 'paired delta does not match summaries')
        require(sum(group['denominator'] for group in by_seed[str(seed)]['user_groups'].values()) == 2520, 'user group denominators do not sum to population')
    for seed, expected in zip(SEEDS, (335, 357, 344)):
        require(summaries[HALF_RUNS[seed]]['hits'] == expected, 'frozen half-arm hit totals differ')
    for seed, expected in zip(SEEDS, (398, 384, 384)):
        require(summaries[FULL_RUNS[seed]]['hits'] == expected, 'frozen full G anchor hit totals differ')
    for seed, expected in zip(SEEDS, (466, 481, 471)):
        require(summaries[P_RUNS[seed]]['hits'] == expected, 'frozen P anchor hit totals differ')
    out = {'schema': 'g-half-supplement-report-v1', 'status': 'read_only_complete', 'test_read': False,
           'checkpoint_read': False, 'embedding_read': False, 'model_run': False, 'submit': False,
           'history_sha256': HISTORY_PIN, 'selected_users_sha256': SELECTED_USERS_SHA,
           'inputs_sha256': {k: v['sha256'] for k, v in req['inputs'].items() if isinstance(v, dict) and 'sha256' in v},
           'training_users': user_stats, 'runs': summaries, 'seed_details': by_seed,
           'user_sample_manifest_binding': {'selected_examples_sha256_by_run': user_sample_manifest_pins,
                                            'source': 'binding.g_half_users in the three raw-verified half-run manifests'},
           'run_evidence': {rid: {'manifest_sha256': records[rid]['manifest_sha256'],
                                  'verify_stdout_sha256': VERIFY_STDOUT_PINS[rid],
                                  'verified': records[rid]['verified']} for rid in ALL_IDS},
           'selection': {rid: {'best_epoch': summary['best_epoch'], 'stop_epoch': summary['stop_epoch']} for rid, summary in summaries.items()},
           'comparison_scope': {'paired_by_seed': True, 'user_group_denominator': 'same fixed 2520-query population',
                                'g_only_is_descriptive': True, 'fusion_constructed': False,
                                'anchor_changed': False, 'causal_attribution': False},
           'read_artifacts_sha256': archive.READS}
    out['read_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return out


def preflight_report(req, request_sha):
    req = validate_request(req)
    check_source_commit(req['source_commit'], req['source_sha256'])
    inputs = req['inputs']
    hist_root = Path(inputs['history']['path']).parent
    hist = load_pinned_json(inputs['history'], {hist_root})
    require(hist.get('artifacts', {}).get('validation-l50.json') == inputs['queries']['sha256'], 'history query artifact pin differs')
    require(hist.get('artifacts', {}).get('train-slices.json') == inputs['slices']['sha256'], 'history train-slices pin differs')
    queries = load_pinned_json(inputs['queries'], {hist_root})
    slices = load_pinned_json(inputs['slices'], {hist_root})
    catalog = load_pinned_json(inputs['catalog'], {Path(inputs['catalog']['path']).parent})
    require(isinstance(queries, list) and len(queries) == N and isinstance(catalog, list) and len(catalog) == 120415, 'frozen population/catalog shape differs')
    archive = guarded_archive(req)
    require(archive.plain_digest(catalog) == hist.get('catalog_sha256') and inputs['catalog']['sha256'] == hist.get('binding', {}).get('catalog_sha'), 'frozen catalog canonical binding differs')
    require(hist.get('binding', {}).get('query_sha') == BASE_QUERIES_CANONICAL_DIGEST,
            'frozen base-query canonical identity differs')
    bins = derive_bins(queries, slices, set(catalog))
    for query in queries:
        require(type(query.get('user_id')) is int and 0 <= query['user_id'] < 2**32, 'validation query user_id must be uint32')
    _, selected = read_training_population(inputs['examples'])
    role_data = [('half', HALF_RUNS, HALF_MANIFESTS, HALF_STDOUT), ('full', FULL_RUNS, FULL_MANIFESTS, FULL_STDOUT), ('p', P_RUNS, P_MANIFESTS, P_STDOUT)]
    check, user_sample_manifest_pins = {}, {}
    for role, ids, pins, _ in role_data:
        root = inputs['run_roots'][role]
        for seed in SEEDS:
            rid = ids[seed]
            state = checked_state(archive, rid, root, pins[seed], VERIFY_STDOUT_PINS[rid])
            if role == 'half':
                user_sample_manifest_pins[rid] = validate_half_manifest(archive, root, rid, pins[seed])
            check[rid] = {'manifest_sha256': state['manifest_sha256'], 'verified': True}
    return {'schema': 'g-half-supplement-preflight-v3', 'status': 'preflight_ok',
            'run_id': 'g-half-supplement-preflight-v3', 'source_commit': req['source_commit'],
            'source_sha256': req['source_sha256'],
            'request_sha256': request_sha,
            'query_count': len(queries), 'heat_bin_counts': {name: bins.count(name) for name in BIN_DENOMINATORS},
            'selected_query_count': sum(q['user_id'] in selected for q in queries),
            'user_sample_manifest_binding': {'selected_examples_sha256_by_run': user_sample_manifest_pins,
                                             'source': 'binding.g_half_users in the three raw-verified half-run manifests'},
            'runs': check, 'test_read': False, 'checkpoint_read': False, 'embedding_read': False,
            'model_run': False, 'engineering': False, 'read_artifacts_sha256': archive.READS}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only G half-user supplement report')
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch('[a-f0-9]{64}', args.request_sha):
        raise ValueError('request SHA syntax required')
    rp = Path(args.request)
    raw = rp.read_bytes()
    if len(raw) > MAX_REQUEST or hashlib.sha256(raw).hexdigest() != args.request_sha:
        raise ValueError('bounded private request SHA mismatch')
    req = validate_request(json.loads(raw))
    if args.preflight:
        require(args.request_sha == hashlib.sha256(raw).hexdigest(), 'preflight request SHA mismatch')
        out = preflight_report(req, args.request_sha)
    else:
        out = execute_report(req)
        out['request_sha256'] = args.request_sha
        out['source_commit'] = req['source_commit']
        out['source_sha256'] = req['source_sha256']
    print(json.dumps(out, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

