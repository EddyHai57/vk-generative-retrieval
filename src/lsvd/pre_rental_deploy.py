"""Prepare source/data receipts and optional deployment commands from local config."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request

PUBLIC_REPOSITORY = 'https://github.com/EddyHai57/vk-generative-retrieval.git'
PUBLIC_BRANCH = 'claude/vk-generative-retrieval-takeover-f62c2a'


def sha256_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def detect_rsync():
    executable = shutil.which('rsync')
    if executable is None:
        raise FileNotFoundError('rsync is required')
    lines = subprocess.check_output([executable, '--version'], text=True).splitlines()
    implementation = 'openrsync' if lines[0].startswith('openrsync:') else 'rsync'
    return dict(executable=executable, implementation=implementation,
                version='; '.join(lines[:2] if implementation == 'openrsync' else lines[:1]))


def ascii_path(path):
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', path) or '..' in Path(path).parts:
        raise ValueError('expected an ASCII path without spaces or shell metacharacters')
    return path


def rsync_command(source, destination, *, ssh=None, files_from=None, compress=True,
                  executable='rsync'):
    command = [executable, '--archive', '--partial']
    if compress:
        command.append('--compress')
    if files_from is not None:
        command.append('--files-from=' + str(files_from))
    if ssh is not None:
        command.extend(['-e', shlex.join(ssh)])
    return [*command, str(source), str(destination)]


def result_digests(sums, *, formal=False):
    files = {}
    for line in sums.decode('ascii').splitlines():
        digest, name = line.split('  ', 1)
        if (not re.fullmatch('[0-9a-f]{64}', digest) or Path(ascii_path(name)).is_absolute()
                or name in files or name in ('SHA256SUMS', 'pull-receipt.json')):
            raise ValueError('invalid result SHA256SUMS entry')
        files[name] = digest
    if not files:
        raise ValueError('result SHA256SUMS is empty')
    if formal:
        required = {'best.pt', 'run_summary.json', 'logs/training_steps.csv',
                    'best-cold-vectors/item_ids.npy', 'best-cold-vectors/vectors.npy',
                    'best-cold-vectors/manifest.json', 'evaluation/all/metrics.csv',
                    'cold-evaluation/all/metrics.csv'}
        if not required.issubset(files) or not any(name.startswith('logs/attempt-')
                                                   and name.endswith('.log') for name in files):
            raise ValueError('formal result SHA256SUMS is missing required artifacts')
    return files


def verify_results(root, *, formal=False):
    root = Path(root).resolve()
    sums = (root / 'SHA256SUMS').read_bytes()
    rows = []
    for name, expected in result_digests(sums, formal=formal).items():
        path = root / name
        if root not in path.resolve().parents or not path.is_file():
            raise ValueError('result file missing or outside result directory: ' + name)
        if sha256_file(path) != expected:
            raise ValueError('result sha256 mismatch: ' + name)
        rows.append(dict(path=name, sha256=expected, bytes=path.stat().st_size))
    return dict(status='verified', files=rows, manifest_sha256=hashlib.sha256(sums).hexdigest(),
                total_bytes=sum(row['bytes'] for row in rows))


def pull_results(source, output, *, ssh, executable='rsync', formal=False):
    from .pre_rental_probe import write_receipt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'pull-receipt.json'
    receipt_path = output / 'SHA256SUMS'
    source = source.rstrip('/') + '/'
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        remote_sums = Path(temporary) / 'SHA256SUMS'
        subprocess.run(rsync_command(source + 'SHA256SUMS', remote_sums, ssh=ssh,
                                    executable=executable), check=True)
        sums = remote_sums.read_bytes()
        files = result_digests(sums, formal=formal)
        if receipt_path.exists():
            if receipt_path.read_bytes() != sums:
                raise ValueError('result SHA256SUMS differs from the existing pull')
        else:
            if any(output.iterdir()):
                raise ValueError('choose a new result directory')
            remote_sums.rename(receipt_path)
        if not receipt.exists():
            command = rsync_command(source, str(output) + '/', ssh=ssh, files_from='-',
                                    executable=executable)
            subprocess.run(command, input=''.join(name + '\n' for name in files),
                           text=True, check=True)
    result = verify_results(output, formal=formal)
    if not receipt.exists():
        write_receipt(receipt, result)
    return result


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
                  or path.name in {'probe_fields.json', 'versions.json', 'Dockerfile.cuda128',
                                   'training-queue.example.json'})]
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


def deployment_commands(mode, source, connection, *, commit=None, files_from=None,
                        executable='rsync'):
    target = connection['target']
    destination = connection['destination']
    if not target or not destination:
        raise ValueError('fill the ignored local connection configuration first')
    ascii_path(destination)
    ssh = ['ssh', *connection.get('ssh_arguments', [])]
    if mode == 'git':
        if commit is None or re.fullmatch('[0-9a-fA-F]{40}', commit) is None:
            raise ValueError('git deployment requires a complete commit SHA')
        clone = ['git', 'clone', '--no-checkout', '--single-branch', '--branch',
                 PUBLIC_BRANCH, PUBLIC_REPOSITORY, destination]
        git = ['git', '-C', destination]
        on_branch = git + ['merge-base', '--is-ancestor', commit,
                           'refs/remotes/origin/' + PUBLIC_BRANCH]
        checkout = git + ['checkout', '--detach', commit]
        command = ' && '.join(shlex.join(part) for part in (clone, on_branch, checkout))
        return [dict(phase='clone-and-checkout', argv=[*ssh, target, command])]
    if mode != 'rsync' or files_from is None:
        raise ValueError('rsync deployment requires its exact file list')
    return [dict(phase='make-destination', argv=[*ssh, target,
                 'mkdir -p ' + shlex.quote(destination)]),
            dict(phase='rsync-source', argv=rsync_command(str(Path(source)) + '/',
                 target + ':' + destination + '/', files_from=files_from, ssh=ssh,
                 executable=executable))]


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
    pull = sub.add_parser('pull')
    pull.add_argument('--remote-result', required=True)
    pull.add_argument('--output', type=Path, required=True)
    pull.add_argument('--connection', type=Path, default=Path('.local/lsvd-connection.json'))
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
        rsync = detect_rsync()
        connection = json.loads(args.connection.read_text())
        commands = deployment_commands(args.mode, args.source, connection, commit=args.commit,
                                        files_from=args.package / 'source-files.txt',
                                        executable=rsync['executable'])
        ssh = ['ssh', *connection.get('ssh_arguments', [])]
        destination = connection['destination']
        receipt = args.package / 'source-manifest.json'
        remote_receipt = destination + '/lsvd-source-manifest.json'
        commands.append(dict(phase='copy-source-manifest', argv=rsync_command(
            receipt, connection['target'] + ':' + remote_receipt, ssh=ssh,
            executable=rsync['executable'])))
        verifier = ['env', 'PYTHONPATH=' + destination + '/src',
                    connection.get('python', 'python3'), '-m', 'lsvd.pre_rental_deploy',
                    'verify', '--root', destination, '--manifest', remote_receipt,
                    '--expected-manifest-sha256', sha256_file(receipt)]
        commands.append(dict(phase='verify-source-sha256',
                              argv=[*ssh, connection['target'], shlex.join(verifier)]))
        if args.dry_run:
            print(json.dumps(dict(action='dry_run', rsync=rsync,
                                  phases=[x['phase'] for x in commands])))
        else:
            for row in commands:
                subprocess.run(row['argv'], check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            print(json.dumps(dict(action='source_transferred', verification='verified', rsync=rsync)))
    elif args.command == 'upload':
        rsync = detect_rsync()
        connection = json.loads(args.connection.read_text())
        manifest = json.loads(args.manifest.read_text())
        destination = ascii_path(connection['data_destination'])
        source = ascii_path(connection['destination'])
        files = ''.join(ascii_path(row['path']) + '\n' for row in manifest['files'])
        ssh = ['ssh', *connection.get('ssh_arguments', [])]
        command = rsync_command(str(args.dataset) + '/', connection['target'] + ':' + destination + '/',
                                files_from='-', ssh=ssh, executable=rsync['executable'])
        if not args.dry_run:
            subprocess.run(command, input=files, text=True, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            verifier = ['env', 'PYTHONPATH=' + source + '/src',
                        connection.get('python', 'python3'), '-m', 'lsvd.pre_rental_deploy',
                        'verify', '--root', destination, '--manifest', destination + '/manifest.json',
                        '--expected-manifest-sha256', next(row['sha256'] for row in manifest['files']
                                                         if row['path'] == 'manifest.json')]
            subprocess.run(['ssh', *connection.get('ssh_arguments', []),
                            connection['target'], shlex.join(verifier)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(json.dumps(dict(action='dry_run' if args.dry_run else 'data_transferred',
                              files=len(manifest['files']), total_bytes=manifest['total_bytes'],
                              verification='planned' if args.dry_run else 'verified',
                              compression=True, rsync=rsync)))
    elif args.command == 'pull':
        rsync = detect_rsync()
        connection = json.loads(args.connection.read_text())
        source = connection['target'] + ':' + ascii_path(args.remote_result)
        result = pull_results(source, args.output, ssh=['ssh', *connection.get('ssh_arguments', [])],
                              executable=rsync['executable'], formal=True)
        print(json.dumps(dict(result, rsync=rsync)))
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
