"""P anchor (M1-mean, 3 seeds) top-100 archive metadata inventory; stdout only.

Read-only. Lists directory entries (name, kind, size) and reads only small metadata:
each run's manifest.json (key names plus a few scalar config fields, never values of
metrics), verifier stdout.log (status fields) and verify claimed receipts (manifest pin).
Never opens predictions, checkpoints, embeddings, optimizer state, training rows or test,
and issues no queue write. Supports python -B - < source: no __file__ dependency.
"""
import datetime
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
RUN_IDS = ('p-seqm1-mean-s42', 'p-seqm1-mean-s2024', 'p-seqm1-mean-s2025')
VERIFY_SUFFIXES = ('-verify', '-v1-verify')
MAX_ENTRIES = 400
MAX_JSON = 8 * 1024 * 1024
MAX_LOG = 64 * 1024
MAX_RECEIPT = 64 * 1024
DEPTH_KEY = re.compile(r'(@\s*\d+|top[_-]?\d+|top[_-]?k|(^|_)k\d+|(^|_)k$|cutoff|depth|candidate|beam|eval_k)', re.I)
DISCOVERY = re.compile(r'(top[_-]?100|top[_-]?k|p-seqm1|export|g5|fusion|candidate)', re.I)
BLOCKED_READ = ('test', 'checkpoint', 'embedding', 'prediction', 'vector', 'optimizer', 'weights', 'state', 'model')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def listing(directory):
    directory = Path(directory)
    if directory.is_symlink():
        return {'exists': True, 'refused': 'symlink'}
    if not directory.is_dir():
        return {'exists': False}
    entries = []
    names = sorted(os.listdir(directory))
    for name in names[:MAX_ENTRIES]:
        st = os.lstat(directory / name)
        kind = 'symlink' if os.path.islink(directory / name) else ('dir' if os.path.isdir(directory / name) else 'file')
        entries.append({'name': name, 'kind': kind, 'size': None if kind == 'dir' else st.st_size})
    return {'exists': True, 'count': len(names), 'truncated': len(names) > MAX_ENTRIES, 'entries': entries}


def small_bytes(path, limit):
    """Return (bytes, None) or (None, reason). Refuses blocked names, symlinks and oversize files."""
    path = Path(path)
    if any(x in path.name.lower() for x in BLOCKED_READ):
        return None, 'blocked name'
    if path.is_symlink():
        return None, 'symlink'
    if not path.is_file():
        return None, 'absent'
    if path.stat().st_size > limit:
        return None, 'oversize'
    return path.read_bytes(), None


def collect_depth_keys(value, found, level=0):
    if level > 8:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if DEPTH_KEY.search(str(key)):
                found.add(str(key))
            collect_depth_keys(item, found, level + 1)
    elif isinstance(value, list):
        for item in value[:50]:
            collect_depth_keys(item, found, level + 1)


def manifest_summary(path):
    payload, reason = small_bytes(path, MAX_JSON)
    if payload is None:
        return {'read': False, 'reason': reason}
    manifest = json.loads(payload)
    keys = set()
    collect_depth_keys(manifest, keys)
    config = manifest.get('config') if isinstance(manifest.get('config'), dict) else {}
    scalars = {k: v for k, v in config.items() if DEPTH_KEY.search(str(k)) and isinstance(v, (int, float, str, bool)) and len(str(v)) <= 40}
    selection = manifest.get('selection') if isinstance(manifest.get('selection'), dict) else {}
    evaluations = manifest.get('evaluations') if isinstance(manifest.get('evaluations'), list) else []
    refs = sorted({str(e.get('predictions_ref')) for e in evaluations if isinstance(e, dict) and 'predictions_ref' in e})
    return {'read': True, 'sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload),
            'top_level_keys': sorted(manifest)[:60], 'depth_key_names': sorted(keys)[:60],
            'config_depth_scalars': scalars, 'best_epoch': selection.get('best_epoch'), 'stop_epoch': selection.get('epoch'),
            'evaluation_count': len(evaluations), 'predictions_refs': refs[:60]}


def verify_status(log_path):
    payload, reason = small_bytes(log_path, MAX_LOG)
    if payload is None:
        return {'read': False, 'reason': reason}
    lines = [x for x in payload.decode('utf-8', 'replace').splitlines() if x.strip()]
    out = {'read': True, 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'parsed': False}
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            keep = {k: value[k] for k in ('status', 'best_epoch', 'epochs', 'replayed_epochs', 'run_id') if k in value}
            out.update(parsed=True, fields=keep, other_keys=sorted(set(value) - set(keep))[:20])
            break
    return out


def claimed_receipts(root, name, actual_manifest_sha):
    out = []
    directory = Path(root) / 'queue/claimed'
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob('*-' + name + '.json')):
        payload, reason = small_bytes(path, MAX_RECEIPT)
        item = {'file': path.name}
        if payload is None:
            item.update(read=False, reason=reason)
        else:
            value = json.loads(payload)
            job = value if 'argv' in value else value.get('job', {}) if isinstance(value.get('job'), dict) else {}
            argv = job.get('argv') if isinstance(job.get('argv'), list) else []
            pin = argv[argv.index('--manifest-sha') + 1] if '--manifest-sha' in argv and argv.index('--manifest-sha') + 1 < len(argv) else None
            item.update(read=True, manifest_sha_pin=pin, manifest_sha_matches_actual=(pin is not None and pin == actual_manifest_sha))
        out.append(item)
    return out


def inventory(root=ROOT):
    root = Path(root)
    started = now()
    runs = {}
    for rid in RUN_IDS:
        run = root / 'runs' / rid
        out_listing = listing(run / 'out')
        manifest = manifest_summary(run / 'out/manifest.json')
        files = [e for e in out_listing.get('entries', []) if e['name'].startswith('predictions') and e['name'].endswith('.json')]
        verifies = []
        for suffix in VERIFY_SUFFIXES:
            vdir = root / 'runs' / (rid + suffix)
            verifies.append({'run_id': rid + suffix, 'exists': vdir.exists() or vdir.is_symlink(), 'listing': listing(vdir),
                             'stdout_status': verify_status(vdir / 'logs/stdout.log') if vdir.is_dir() and not vdir.is_symlink() else {'read': False, 'reason': 'absent or symlink'},
                             'claimed_receipts': claimed_receipts(root, rid + suffix, manifest.get('sha256')),
                             'rejected_names': sorted(p.name for p in (root / 'queue/rejected').glob('*-' + rid + suffix + '.*'))[:10] if (root / 'queue/rejected').is_dir() else []})
        runs[rid] = {'run_dir_listing': listing(run), 'out_listing': out_listing, 'prediction_files': files,
                     'manifest': manifest, 'verify': verifies,
                     'train_claimed_receipts': claimed_receipts(root, rid, manifest.get('sha256'))}
    own = set(RUN_IDS) | {rid + s for rid in RUN_IDS for s in VERIFY_SUFFIXES}
    runs_dir = root / 'runs'
    matches = sorted(n for n in (os.listdir(runs_dir) if runs_dir.is_dir() else []) if DISCOVERY.search(n) and n not in own)
    queue_matches = {}
    queue = root / 'queue'
    for sub in (sorted(os.listdir(queue)) if queue.is_dir() else []):
        if (queue / sub).is_dir() and not (queue / sub).is_symlink():
            queue_matches[sub] = sorted(n for n in os.listdir(queue / sub) if DISCOVERY.search(n) and not any(r in n for r in RUN_IDS))[:50]
    return {'read_at_utc': started, 'finished_at_utc': now(), 'root': str(root), 'runs': runs,
            'discovery': {'runs_dir_matches': matches[:100], 'runs_dir_match_count': len(matches), 'queue_matches': queue_matches,
                          'pattern': DISCOVERY.pattern, 'scope': 'single-level name listing of runs/ and queue/*; not an exhaustive server search'},
            'boundaries': {'prediction_content_read': False, 'checkpoint_read': False, 'test_read': False, 'embedding_read': False,
                           'new_submit': False, 'write_performed': False}}


def main():
    print(json.dumps(inventory(), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
