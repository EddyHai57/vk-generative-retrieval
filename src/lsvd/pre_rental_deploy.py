"""Prepare source/data receipts and optional deployment commands from local config."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.request

from .two_tower_prepare import sha256_file


def upload_manifest(dataset):
    dataset = Path(dataset)
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = []
    for row in manifest['arrays'].values():
        path = dataset / row['file']
        digest = sha256_file(path)
        if digest != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('prepared data changed after its manifest was written')
        records.append(dict(path=row['file'], bytes=path.stat().st_size, sha256=digest))
    records.append(dict(path='manifest.json', bytes=manifest_path.stat().st_size,
                        sha256=sha256_file(manifest_path)))
    result = dict(kind='two-tower-data-upload', train_weeks=[0, 24],
                  validation_week=25, files=records,
                  total_bytes=sum(row['bytes'] for row in records))
    return result


def code_manifest(source):
    source = Path(source)
    files = [path for path in (source / 'src/lsvd').rglob('*') if path.is_file()
             and '__pycache__' not in path.parts
             and (path.suffix in {'.py', '.md', '.txt', '.sh'}
                  or path.name in {'probe_fields.json', 'versions.json', 'Dockerfile.cuda128'})]
    files += [path for path in [source / 'src/vkgr/__init__.py',
                               source / 'src/vkgr/tiger_arch.py'] if path.is_file()]
    rows = [dict(path=path.relative_to(source).as_posix(),
                 bytes=path.stat().st_size, sha256=sha256_file(path))
            for path in sorted(files)]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    git = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True)
    commit = git.stdout.strip() if git.returncode == 0 else None
    return dict(kind='working-tree-source', files=rows, source_sha256=digest,
                git_commit=commit,
                total_bytes=sum(row['bytes'] for row in rows))


def verify_manifest(root, manifest, expected_manifest_sha256=None):
    root = Path(root).resolve()
    if expected_manifest_sha256 is not None and sha256_file(manifest) != expected_manifest_sha256:
        raise ValueError('manifest identity mismatch')
    manifest = json.loads(Path(manifest).read_text())
    files = manifest.get('files')
    if files is None:
        files = [dict(path=row['file'], bytes=row['bytes'], sha256=row['sha256'])
                 for row in manifest['arrays'].values()]
    for row in files:
        path = (root / row['path']).resolve()
        if root not in path.parents or path.stat().st_size != row['bytes']:
            raise ValueError('manifest file path or size mismatch')
        if sha256_file(path) != row['sha256']:
            raise ValueError('manifest sha256 mismatch')
    return dict(status='verified', files=len(files),
                total_bytes=sum(row['bytes'] for row in files))


def deployment_commands(mode, source, connection, *, commit=None, files_from=None):
    target = connection['target']
    destination = connection['destination']
    if not target or not destination:
        raise ValueError('fill the ignored local connection configuration first')
    ssh = ['ssh', *connection.get('ssh_arguments', [])]
    if mode == 'git':
        if commit is None or re.fullmatch('[0-9a-fA-F]{40}', commit) is None:
            raise ValueError('git deployment requires a complete commit SHA')
        command = ('git clone --no-checkout ' + shlex.quote(connection['repository'])
                   + ' ' + shlex.quote(destination) + ' && git -C '
                   + shlex.quote(destination) + ' checkout --detach ' + commit)
        return [dict(phase='clone-and-checkout', argv=[*ssh, target, command])]
    if mode != 'rsync' or files_from is None:
        raise ValueError('rsync deployment requires its exact file list')
    transport = shlex.join(ssh)
    return [dict(phase='make-destination', argv=[*ssh, target,
                 'mkdir -p ' + shlex.quote(destination)]),
            dict(phase='rsync-source', argv=['rsync', '--archive', '--partial',
                 '--append-verify', '--protect-args', '--files-from=' + str(files_from),
                 '-e', transport, str(Path(source)) + '/', target + ':' + destination + '/'])]


def stop_request(environment):
    instance = environment.get('CONTAINER_ID', '')
    credential = environment.get('CONTAINER_API_KEY', '')
    if not instance.isdigit() or not credential:
        raise ValueError('per-instance credentials are unavailable')
    return urllib.request.Request(
        'https://console.vast.ai/api/v0/instances/' + instance + '/',
        data=json.dumps({'state': 'stopped'}).encode(), method='PUT',
        headers={'Authorization': 'Bearer ' + credential,
                 'Content-Type': 'application/json'},
    )


def stop_self():
    request = stop_request(os.environ)
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not result.get('success'):
        raise RuntimeError('instance stop request was not accepted')
    return dict(action='stop_requested')


def launch_detached(command, run_directory):
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'stdout.log').open('xb') as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(pid=process.pid, started_utc=datetime.now(timezone.utc).isoformat(),
                  detached=True, command_sha256=hashlib.sha256(
                      json.dumps(command).encode()).hexdigest())
    (root / 'launch.json').open('x').write(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    package = sub.add_parser('package')
    package.add_argument('--source', type=Path, default=Path('.'))
    package.add_argument('--dataset', type=Path, required=True)
    package.add_argument('--output', type=Path, required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--root', type=Path, required=True)
    verify.add_argument('--manifest', type=Path, required=True)
    verify.add_argument('--expected-manifest-sha256')
    deploy = sub.add_parser('deploy')
    deploy.add_argument('--mode', choices=['git', 'rsync'], required=True)
    deploy.add_argument('--source', type=Path, default=Path('.'))
    deploy.add_argument('--package', type=Path, required=True)
    deploy.add_argument('--connection', type=Path, default=Path('.local/lsvd-connection.json'))
    deploy.add_argument('--commit')
    deploy.add_argument('--dry-run', action='store_true')
    upload = sub.add_parser('upload')
    upload.add_argument('--dataset', type=Path, required=True)
    upload.add_argument('--manifest', type=Path, required=True)
    upload.add_argument('--connection', type=Path, default=Path('.local/lsvd-connection.json'))
    upload.add_argument('--dry-run', action='store_true')
    launch = sub.add_parser('launch')
    launch.add_argument('--run-directory', type=Path, required=True)
    launch.add_argument('--dry-run', action='store_true')
    launch.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == 'package':
        args.output.mkdir(parents=True, exist_ok=False)
        source = code_manifest(args.source)
        (args.output / 'source-manifest.json').open('x').write(json.dumps(source, indent=2))
        (args.output / 'source-files.txt').open('x').write(
            ''.join(row['path'] + '\n' for row in source['files']))
        data = upload_manifest(args.dataset)
        (args.output / 'data-manifest.json').open('x').write(json.dumps(data, indent=2))
        print(json.dumps(dict(source_sha256=source['source_sha256'],
                              source_files=len(source['files']), data_bytes=data['total_bytes'])))
    elif args.command == 'verify':
        print(json.dumps(verify_manifest(args.root, args.manifest,
                                         args.expected_manifest_sha256)))
    elif args.command == 'deploy':
        connection = json.loads(args.connection.read_text())
        commands = deployment_commands(args.mode, args.source, connection, commit=args.commit,
                                        files_from=args.package / 'source-files.txt')
        ssh = ['ssh', *connection.get('ssh_arguments', [])]
        destination = connection['destination']
        receipt = args.package / 'source-manifest.json'
        remote_receipt = destination + '/lsvd-source-manifest.json'
        commands.append(dict(phase='copy-source-manifest', argv=[
            'rsync', '--archive', '--partial', '--protect-args', '-e', shlex.join(ssh),
            str(receipt), connection['target'] + ':' + remote_receipt]))
        verifier = ['env', 'PYTHONPATH=' + destination + '/src',
                    connection.get('python', 'python3'), '-m', 'lsvd.pre_rental_deploy',
                    'verify', '--root', destination, '--manifest', remote_receipt,
                    '--expected-manifest-sha256', sha256_file(receipt)]
        commands.append(dict(phase='verify-source-sha256',
                              argv=[*ssh, connection['target'], shlex.join(verifier)]))
        if args.dry_run:
            print(json.dumps(dict(action='dry_run', phases=[x['phase'] for x in commands])))
        else:
            for row in commands:
                subprocess.run(row['argv'], check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            print(json.dumps(dict(action='source_transferred', verification='verified')))
    elif args.command == 'upload':
        connection = json.loads(args.connection.read_text())
        manifest = json.loads(args.manifest.read_text())
        files = args.manifest.with_name('data-files.txt')
        if not files.exists():
            files.open('x').write(''.join(row['path'] + '\n' for row in manifest['files']))
        destination = connection['data_destination']
        command = ['rsync', '--archive', '--partial', '--append-verify', '--protect-args',
                   '--files-from=' + str(files), '-e',
                   shlex.join(['ssh', *connection.get('ssh_arguments', [])]),
                   str(args.dataset) + '/', connection['target'] + ':' + destination + '/']
        if not args.dry_run:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            verifier = ['env', 'PYTHONPATH=' + connection['destination'] + '/src',
                        connection.get('python', 'python3'), '-m', 'lsvd.pre_rental_deploy',
                        'verify', '--root', destination, '--manifest', destination + '/manifest.json',
                        '--expected-manifest-sha256', next(row['sha256'] for row in manifest['files']
                                                         if row['path'] == 'manifest.json')]
            subprocess.run(['ssh', *connection.get('ssh_arguments', []),
                            connection['target'], shlex.join(verifier)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(json.dumps(dict(action='dry_run' if args.dry_run else 'data_transferred',
                              files=len(manifest['files']), total_bytes=manifest['total_bytes'],
                              verification='planned' if args.dry_run else 'verified')))
    else:
        command = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
        if not command:
            parser.error('launch requires a command')
        print(json.dumps(dict(action='dry_run', detached=True) if args.dry_run
                         else launch_detached(command, args.run_directory)))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(json.dumps(dict(status='transport_failed', returncode=error.returncode)),
              file=sys.stderr)
        raise SystemExit(error.returncode)
