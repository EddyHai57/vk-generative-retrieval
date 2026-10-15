"""FEAT-04a CUDA parity (card F4): the same synthetic config run with the FEAT-04a runner (f4_control, T0.10) and with the P-HP-01 runner (H1)
on one device; every per-step record, selection, per-epoch result and diagnostic, and the state-dict bytes must be equal (bit for bit).
Runs on synthetic data only, reads nothing, writes only to a temporary directory. Prints one JSON line; exit 0 = equal, 2 = differs/failed.
Usage: PYTHONPATH=src python scripts/feat04a_cuda_parity.py --device cuda
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from vkgr import retrieval_feat02_run as fr
from vkgr import retrieval_feat04a_run as run4
from vkgr import retrieval_php01_run as php


def compare(a, b, dir_a, dir_b):
    errors = []
    for key in ('loss', 'gradient_norm', 'drawn', 'skipped', 'indices_sha256', 'negatives_sha256', 'step', 'epoch'):
        if [r[key] for r in a['records']] != [r[key] for r in b['records']]:
            errors.append('records.' + key)
    if a['selection'] != b['selection']:
        errors.append('selection')
    if [e['result'] for e in run4.full_evaluations(dir_a, a)] != [e['result'] for e in php.full_evaluations(dir_b, b)]:
        errors.append('per-epoch results')
    if [e['diagnostics'] for e in a['evaluations']] != [e['diagnostics'] for e in b['evaluations']]:
        errors.append('per-epoch diagnostics')
    if a['data_diagnostics'] != b['data_diagnostics']:
        errors.append('data diagnostics')
    for name in ('best.pt', 'last.pt'):
        sa = torch.load(Path(dir_a) / name, map_location='cpu', weights_only=True)['model']
        sb = torch.load(Path(dir_b) / name, map_location='cpu', weights_only=True)['model']
        if list(sa) != list(sb) or any(sa[k].numpy().tobytes() != sb[k].numpy().tobytes() for k in sa):
            errors.append(name + ' state')
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description='FEAT-04a vs P-HP-01 H1 parity on one device (synthetic data)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('ERROR FEAT-04A parity: CUDA requested but not available', file=sys.stderr); return 2
    data = fr.synthetic_data()
    small = {'batch_size': 16, 'max_epochs': args.epochs, 'min_epochs': min(2, args.epochs)}
    config4 = {**run4.default_config(), 'feature': 'f4_control', 'logq': True, 'temperature': 0.1, **small}
    config_php = {**php.default_config(), 'knob': 'H1', 'temperature': 0.1, 'dim': 64, **small}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = run4.run_session(tmp / 'feat04a', data, config4, args.device)
        b = php.run_session(tmp / 'php01', data, config_php, args.device)
        errors = compare(a, b, tmp / 'feat04a', tmp / 'php01')
    print(json.dumps({'status': 'parity_ok' if not errors else 'parity_differs', 'device': args.device, 'epochs': len(a['evaluations']), 'steps': a['steps'], 'differences': errors}))
    return 0 if not errors else 2


if __name__ == '__main__':
    raise SystemExit(main())
