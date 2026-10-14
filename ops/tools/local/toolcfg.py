"""Shared config for the local ops tools (standard library only).

Config is a JSON file passed with ``--config`` (or env ``VKGR_TOOLS_CONFIG``); nothing about the
local machine is hard-coded here. See ``ops/tools/config.example.json`` for the schema.

Remote scripts under ``ops/tools/remote/`` are sent to the server through the transport and read
their server paths from a ``CFG`` global that :func:`render` prepends (they are run with
``python -c`` or from stdin, so they cannot take argv).
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
REMOTE = TOOLS / 'remote'
SERVER = TOOLS / 'server'

REQUIRED = {'ssh_host': str, 'state_dir': str, 'server': dict}
SERVER_KEYS = ('python', 'ops_repo', 'ops_root', 'g_deploy', 'p_deploy', 'pinned_dir', 'work_dir',
               'permits_dir', 'tools_dir', 'repo_url', 'scope_prefix')


class ConfigError(ValueError):
    pass


def load(path=None):
    """Read and validate the config. ``path`` falls back to ``VKGR_TOOLS_CONFIG``."""
    path = path or os.environ.get('VKGR_TOOLS_CONFIG')
    if not path:
        raise ConfigError('config required: pass --config or set VKGR_TOOLS_CONFIG')
    try:
        cfg = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f'config unreadable: {error}') from error
    for key, kind in REQUIRED.items():
        if not isinstance(cfg.get(key), kind) or not cfg[key]:
            raise ConfigError(f'config key {key!r} must be a non-empty {kind.__name__}')
    missing = [k for k in SERVER_KEYS if not isinstance(cfg['server'].get(k), str) or not cfg['server'][k]]
    if missing:
        raise ConfigError(f'config server keys missing: {missing}')
    cfg.setdefault('socket_prefix', os.path.join(tempfile.gettempdir(), 'vkgr-ssh-'))
    cfg.setdefault('mux_max_seconds', 57600)
    return cfg


def state_dir(cfg):
    return Path(cfg['state_dir']).expanduser()


def render(script_path, cfg):
    """Remote script source with ``CFG`` (the server section + ssh-free keys) prepended."""
    payload = json.dumps({'server': cfg['server']}, sort_keys=True)
    head = f'import json as _cfg_json\nCFG = _cfg_json.loads({payload!r})\n'
    return head + Path(script_path).read_text()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()
