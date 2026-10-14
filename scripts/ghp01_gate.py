"""G-HP-01 entry gate on the anchor A1 (pure standard library; prints ONE JSON line).

A1 becomes the G-HP anchor only if: its train job succeeded with a complete manifest; a verify job of exactly that manifest
(``--manifest-sha`` = sha256 of the train manifest file) succeeded and printed ``status == 'verified'`` with the same best
epoch; and its best R@50 is at least 196 of 2520 validation hits. 196 hits = 0.077778 is the first count at or above the
threshold 0.077397 (core03 anchor 190 hits + 0.002); 195 hits = 0.077381 does not pass. Counting integer hits avoids the
floating-point edge. Missing inputs give ``passed=false`` with a ``missing`` list instead of a traceback.
"""
import argparse
import hashlib
import json
from pathlib import Path

QUERIES = 2520
MIN_HITS = 196
R50_THRESHOLD = 0.077397


def hits(r50, queries=QUERIES):
    """R@50 -> integer hit count; refuses values that are not k / queries."""
    count = round(r50 * queries)
    if abs(count / queries - r50) >= 1e-12:
        raise ValueError(f'R@50 {r50} is not an integer hit count over {queries} queries')
    return count


def _arg(argv, name):
    argv = list(argv or ())
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def decide(train_terminal, train_manifest, train_manifest_sha256, verify_terminal, verify_last_line, verify_run):
    inputs = {'train_terminal': train_terminal, 'train_manifest': train_manifest, 'verify_terminal': verify_terminal,
              'verify_stdout': verify_last_line, 'verify_run': verify_run}
    missing = [k for k, v in inputs.items() if v is None]
    manifest = train_manifest or {}
    selection = manifest.get('selection') or {}
    train_ok = (train_terminal or {}).get('status') == 'succeeded' and manifest.get('status') == 'complete'
    verified = ((verify_terminal or {}).get('status') == 'succeeded' and (verify_last_line or {}).get('status') == 'verified'
                and selection.get('best_epoch') is not None and (verify_last_line or {}).get('best_epoch') == selection.get('best_epoch')
                and train_manifest_sha256 is not None and _arg((verify_run or {}).get('argv'), '--manifest-sha') == train_manifest_sha256)
    metric = selection.get('best_metric')
    r50, n50 = (metric[0], metric[1]) if metric else (None, None)
    count = hits(r50) if r50 is not None else None
    r50_ok = count is not None and count >= MIN_HITS
    return {'passed': bool(train_ok and verified and r50_ok and not missing), 'checks': {'train_succeeded': train_ok, 'verified': verified, 'r50_ok': r50_ok},
            'r50': r50, 'n50': n50, 'hits': count, 'threshold_hits': MIN_HITS, 'r50_threshold': R50_THRESHOLD, 'queries': QUERIES,
            'best_epoch': selection.get('best_epoch'), 'epochs': len(manifest['epochs']) if 'epochs' in manifest else None,
            'wall_seconds': manifest.get('wall_seconds'), 'hit_max_epochs': manifest.get('hit_max_epochs'), 'missing': missing}


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _last_json_line(path):
    path = Path(path)
    if not path.is_file():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    try:
        return json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description='G-HP-01 A1 entry gate (one JSON line)')
    for name in ('train-terminal', 'train-manifest', 'verify-terminal', 'verify-stdout', 'verify-run'):
        parser.add_argument(f'--{name}', required=True)
    args = parser.parse_args(argv)
    manifest_path = Path(args.train_manifest)
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
    value = decide(_read_json(args.train_terminal), _read_json(manifest_path), sha, _read_json(args.verify_terminal),
                   _last_json_line(args.verify_stdout), _read_json(args.verify_run))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
