"""Send one remote script over an existing authenticated SSH control master (no fresh auth).

usage: transport.py --config C --n N SCRIPT PREFIX [--input FILE | --readonly] [--timeout S] [--dry-run]

- SCRIPT is rendered with :func:`toolcfg.render` (``CFG`` prepended).
- write mode: ``python -B -c <script>`` on the server, stdin = ``--input`` bytes.
- ``--readonly``: the script itself goes on stdin (``python -B`` reads the program), no input allowed.
- Receipts ``PREFIX.{stdout,stderr,transport.json}`` go to ``state_dir``.
- ``--dry-run``: print the ssh argv and rendered-script sha256; no socket check, no ssh.
Derived from DEPLOY-01 ``transport_mux_n.py`` (see ops/tools/PROVENANCE.md).
"""
import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolcfg  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('script')
    p.add_argument('prefix')
    p.add_argument('--config')
    p.add_argument('--n', type=int, required=True, help='control master number (formalNN)')
    p.add_argument('--input')
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--readonly', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.readonly and a.input:
        p.error('readonly source stdin cannot carry a write package')
    if not a.readonly and not a.input and not a.dry_run:
        p.error('write mode needs --input (or use --readonly)')
    if not 1 <= a.n <= 99:
        p.error('--n must be 1..99')
    return a


def control_socket(cfg, n):
    state = json.loads((toolcfg.state_dir(cfg) / f'ssh-control-state-formal{n:02d}.json').read_text())
    return Path(state['socket_path'])


def check_socket(cfg, socket):
    directory = socket.parent
    if (not str(directory).startswith(cfg['socket_prefix']) or directory.is_symlink()
            or stat.S_IMODE(directory.stat().st_mode) != 0o700 or directory.stat().st_uid != os.getuid()
            or socket.is_symlink() or not stat.S_ISSOCK(socket.stat().st_mode)
            or socket.stat().st_uid != os.getuid()):
        raise ValueError('owned actual private control socket required')


def ssh_argv(cfg, socket, readonly, script):
    command = shlex.quote(cfg['server']['python']) + ' -B'
    if not readonly:
        command += ' -c ' + shlex.quote(script)
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ControlMaster=no',
            '-o', 'ProxyCommand=false', '-S', str(socket), cfg['ssh_host'], command]


def main(argv=None):
    a = parse_args(argv)
    cfg = toolcfg.load(a.config)
    script = toolcfg.render(a.script, cfg)
    if a.dry_run:
        socket = Path('<socket of formal%02d>' % a.n)
        args = ssh_argv(cfg, socket, a.readonly, '<rendered script>')
        print(json.dumps({'dry_run': True, 'ssh_argv': args, 'script': a.script,
                          'rendered_sha256': toolcfg.sha256_text(script), 'readonly': a.readonly,
                          'input': a.input, 'prefix': a.prefix}, indent=1))
        return 0
    socket = control_socket(cfg, a.n)
    check_socket(cfg, socket)
    check = subprocess.run(['ssh', '-S', str(socket), '-O', 'check', cfg['ssh_host']],
                           capture_output=True, timeout=5)
    if check.returncode:
        raise RuntimeError('control connection lost; no new authentication fallback')
    data = script.encode() if a.readonly else Path(a.input).read_bytes()
    out_dir = toolcfg.state_dir(cfg)
    start = time.monotonic()
    with (out_dir / (a.prefix + '.stdout')).open('wb') as out, (out_dir / (a.prefix + '.stderr')).open('wb') as err:
        result = subprocess.run(ssh_argv(cfg, socket, a.readonly, script), input=data, stdout=out,
                                stderr=err, timeout=a.timeout)
    value = {'exit_code': result.returncode, 'elapsed_seconds': time.monotonic() - start,
             'script_path': Path(a.script).name, 'script_sha256': toolcfg.sha256_text(script),
             'stdout_sha256': hashlib.sha256((out_dir / (a.prefix + '.stdout')).read_bytes()).hexdigest(),
             'reused_authenticated_control': True, 'readonly_stdin_source': a.readonly,
             'fresh_auth_fallback': False}
    (out_dir / (a.prefix + '.transport.json')).write_text(json.dumps(value))
    print(json.dumps(value))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
