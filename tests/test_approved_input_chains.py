import importlib.util
import json
from pathlib import Path

import pytest


def load():
    spec = importlib.util.spec_from_file_location('approved_input_chains', Path(__file__).parents[1] / 'ops/tools/remote/approved_input_chains.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_requires_original_success(tmp_path):
    m = load()
    (tmp_path / 'terminal.json').write_text(json.dumps({'run_id': 'r', 'status': 'succeeded', 'exit_code': 0, 'signal': None, 'wall_cap_exceeded': False}))
    assert m.terminal(tmp_path, 'r') is True
    (tmp_path / 'terminal.json').write_text(json.dumps({'run_id': 'r', 'status': 'failed', 'exit_code': 0, 'signal': None}))
    with pytest.raises(ValueError):
        m.terminal(tmp_path, 'r')


def test_pending_is_not_success(tmp_path):
    assert load().terminal(tmp_path, 'r') is False


def test_no_fit_in_any_chain():
    m = load()
    assert m.CHAINS['a1'] == ('preflight', 'train', 'verify')
    assert m.CHAINS['full-input'] == ('preflight', 'prepare')
    assert all('fit' not in chain for chain in m.CHAINS.values())


def test_submit_keeps_exact_dry_spec():
    m = load()
    spec = dict(run_id='r', kind='ops', cwd='/code', config='/request', seed=42,
                cpu_cores=1, ram_mib=123, vram_mib=0, disk_gib=4, wall_hours=2,
                env=['K=V'], inputs=['/input'], argv=['py', '-B', '-m', 'module', 'stage'])
    argv = m.submit_argv(spec)
    assert argv[argv.index('--ram-mib') + 1] == '123'
    assert argv[argv.index('--') + 1:] == spec['argv']
