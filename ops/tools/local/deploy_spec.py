"""Build stdin specs for the remote deploy writers from a LOCAL git repo (no network, no ssh).

  tree      --repo R --commit C [--branch B --expect-old O]   -> {'commit','tree'} (+ 'branch','expect_old' for ops_update.py)
  g-overlay --repo R --base B --target T --data-root D [--data PATH]... [--override PATH=SHA]...
            -> {'base','target','target_tree','data','overrides'} for g_deploy.py; data files are read from D/PATH (base64).
Output goes to --out (exclusive create) or stdout. Paths inside the spec are repo-relative.
"""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def git(repo, *args, text=True):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=text, check=True).stdout


def tree(repo, commit):
    """{path: {'sha256', 'mode'}} of every blob in ``commit`` (content sha256, not git blob id)."""
    out = {}
    for line in git(repo, 'ls-tree', '-r', '-z', commit).split('\0'):
        if not line:
            continue
        meta, path = line.split('\t', 1)
        mode, kind, _ = meta.split()
        if kind != 'blob':
            continue
        out[path] = {'sha256': hashlib.sha256(git(repo, 'show', f'{commit}:{path}', text=False)).hexdigest(), 'mode': mode}
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    s = p.add_subparsers(dest='cmd', required=True)
    t = s.add_parser('tree')
    t.add_argument('--repo', required=True)
    t.add_argument('--commit', required=True)
    t.add_argument('--branch')
    t.add_argument('--expect-old')
    g = s.add_parser('g-overlay')
    g.add_argument('--repo', required=True)
    g.add_argument('--base', required=True)
    g.add_argument('--target', required=True)
    g.add_argument('--data-root')
    g.add_argument('--data', action='append', default=[])
    g.add_argument('--override', action='append', default=[])
    for sp in (t, g):
        sp.add_argument('--out')
    a = p.parse_args(argv)
    if a.cmd == 'tree' and (a.branch is None) != (a.expect_old is None):
        p.error('--branch and --expect-old go together (ops_update spec)')
    if a.cmd == 'g-overlay':
        if a.data and not a.data_root:
            p.error('--data needs --data-root')
        for o in a.override:
            path, _, sha = o.partition('=')
            if not path or len(sha) != 64:
                p.error(f'--override must be PATH=SHA256: {o}')
        if any(Path(d).is_absolute() or '..' in Path(d).parts for d in a.data):
            p.error('--data paths must be repo-relative without ..')
    return a


def build(a):
    if a.cmd == 'tree':
        commit = git(a.repo, 'rev-parse', a.commit + '^{commit}').strip()
        spec = {'commit': commit, 'tree': tree(a.repo, commit)}
        if a.branch:
            spec.update(branch=a.branch, expect_old=a.expect_old)
        return spec
    data = {}
    for rel in a.data:
        b = (Path(a.data_root) / rel).read_bytes()
        data[rel] = {'b64': base64.b64encode(b).decode(), 'sha256': hashlib.sha256(b).hexdigest()}
    return {'base': git(a.repo, 'rev-parse', a.base + '^{commit}').strip(),
            'target': git(a.repo, 'rev-parse', a.target + '^{commit}').strip(),
            'target_tree': tree(a.repo, a.target), 'data': data,
            'overrides': dict(o.split('=', 1) for o in a.override)}


def main(argv=None):
    a = parse_args(argv)
    text = json.dumps(build(a), sort_keys=True)
    if a.out:
        with open(a.out, 'x') as f:
            f.write(text)
    else:
        sys.stdout.write(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
