"""One bounded entry for inventory, tower speed, synthetic A1, resume and retrieval."""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


STAGE_SECONDS = dict(machine=300, two_tower=1200, generator=600,
                     checkpoint=300, evaluate=300)


def case_grid(stage, *, local_smoke=False):
    if stage == 'two_tower':
        if local_smoke:
            return [dict(structure=structure, precision='fp32', id_min_count=100,
                          data_placement='cpu', batch_size=8)
                    for structure in ['mlp_mean', 'mlp_causal']]
        return [dict(structure=structure, precision=precision, id_min_count=minimum,
                      data_placement=placement, batch_size=1024)
                for structure, precision, minimum, placement in itertools.product(
                    ['mlp_mean', 'mlp_causal'], ['fp32', 'tf32', 'bf16'],
                    [5, 20, 100], ['cpu', 'cuda'])]
    if local_smoke:
        return [dict(positions=positions, batch_size=1, precision='fp32')
                for positions in [200, 600]]
    return [dict(positions=positions, batch_size=batch, precision=precision)
            for positions, batch, precision in itertools.product(
                [200, 600], [256, 512, 1024], ['fp32', 'bf16'])]


def run_child(command, output_prefix, seconds, *, own_group=True):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output_prefix.with_name(output_prefix.name + '.log').open('xb') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=own_group)
        receipt = dict(pid=process.pid, command_sha256=hashlib.sha256(
            json.dumps(command).encode()).hexdigest(), timeout_seconds=seconds,
            started_utc=datetime.now(timezone.utc).isoformat())
        with output_prefix.with_name(output_prefix.name + '.launch.json').open('x') as receipt_file:
            json.dump(receipt, receipt_file, indent=2)
        status = 'complete'
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            status = 'timeout'
            if own_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
    result = dict(status=status, returncode=process.returncode,
                   seconds=time.monotonic()-started)
    path = output_prefix / 'result.json'
    if path.exists():
        result['result'] = json.loads(path.read_text())
        if status != 'timeout':
            result['status'] = result['result'].get('status', status)
    elif status != 'timeout' and process.returncode != 0:
        result['status'] = 'failed'
    if status == 'timeout' and output_prefix.is_dir():
        result['partial_cases'] = [json.loads(path.read_text())
                                    for path in output_prefix.glob('*/result.json')]
    return result


def child_command(args, stage, output, *, case_file=None, shared_order=None):
    command = [sys.executable, '-m', 'lsvd.pre_rental_probe', '--stage', stage,
               '--output', str(output), '--device', args.device,
               '--warmup', str(args.warmup), '--steps', str(args.steps),
               '--total-seconds', str(args.total_seconds),
               '--generator-vocab', str(args.generator_vocab),
               '--decoder-positions', str(args.decoder_positions)]
    for flag, value in [('dataset', args.dataset), ('threads', args.threads),
                        ('case-file', case_file), ('shared-order', shared_order)]:
        if value is not None:
            command += ['--' + flag, str(value)]
    if args.synthetic:
        command.append('--synthetic')
    if args.local_smoke:
        command.append('--local-smoke')
    if args.download_url:
        command += ['--download-url', args.download_url]
    return command


def run_cases(args, stage):
    from .pre_rental_cases import prepare_shared_order

    shared = prepare_shared_order(args, args.output) if stage == 'two_tower' else None
    started = time.monotonic()
    cases = []
    budget = min(STAGE_SECONDS[stage], args.total_seconds)
    for specification in case_grid(stage, local_smoke=args.local_smoke):
        remaining = budget - (time.monotonic()-started)
        label = ('-'.join(str(specification[key]) for key in
                           ['structure', 'precision', 'id_min_count', 'data_placement'])
                 if stage == 'two_tower' else
                 f"positions-{specification['positions']}-batch-{specification['batch_size']}-{specification['precision']}")
        if remaining <= 0:
            cases.append(dict(case=specification, status='skipped_stage_deadline'))
            continue
        case_file = args.output / (label + '.json')
        with case_file.open('x') as handle:
            json.dump(specification, handle)
        command = child_command(args, stage + '_case', args.output / label,
                                 case_file=case_file, shared_order=shared)
        result = run_child(command, args.output / label, remaining, own_group=False)
        result['case'] = specification
        cases.append(result)
    complete = all(row['status'] in ['complete', 'oom'] for row in cases)
    return dict(status='complete' if complete else 'partial', cases=cases,
                cuda_grid_skipped=args.local_smoke,
                cuda_skip_reason='Mac/local CPU smoke' if args.local_smoke else None)


def run_worker(args):
    from . import pre_rental_cases

    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.stage in ['two_tower', 'generator']:
            result = run_cases(args, args.stage)
        else:
            result = getattr(pre_rental_cases, args.stage)(args)
    except Exception as error:
        result = pre_rental_cases.empty_record()
        result.update(status='oom' if type(error).__name__ in
                      ['OutOfMemoryError', 'MemoryError'] else 'failed',
                      error_type=type(error).__name__, error=str(error)[:240])
        if args.device == 'cuda':
            from .two_tower_runtime import process_metrics
            result['resources'] = process_metrics('cuda')
    with (args.output / 'result.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--local-smoke', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--total-seconds', type=float, default=2700)
    parser.add_argument('--generator-vocab', type=int, default=3074)
    parser.add_argument('--decoder-positions', type=int, default=3)
    parser.add_argument('--download-url', default='https://speed.cloudflare.com/__down?bytes=33554432')
    parser.add_argument('--auto-stop', action='store_true')
    parser.add_argument('--stage', choices=[*STAGE_SECONDS, 'two_tower_case', 'generator_case'])
    parser.add_argument('--case-file', type=Path)
    parser.add_argument('--shared-order', type=Path)
    args = parser.parse_args(argv)
    if not args.synthetic and args.dataset is None:
        parser.error('--dataset or --synthetic is required')
    if not 0 < args.total_seconds <= 2700 or args.steps < 1 or args.warmup < 0:
        parser.error('positive steps and total <=2700 seconds required')
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.local_smoke:
        args.device = 'cpu'
    if args.device == 'cpu':
        args.local_smoke = True
    if args.stage is not None:
        return run_worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reserve = 40 if args.auto_stop else 2
    deadline = started + args.total_seconds - reserve
    stages = []
    for stage, limit in STAGE_SECONDS.items():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stages.append(dict(stage=stage, status='skipped_total_deadline'))
            continue
        command = child_command(args, stage, args.output / stage)
        result = run_child(command, args.output / stage, min(limit, remaining))
        result['stage'] = stage
        stages.append(result)
    summary = dict(execution='local_cpu_smoke' if args.local_smoke else 'cuda_probe',
                    cuda_cases_skipped=args.local_smoke, stages=stages,
                    seconds=time.monotonic()-started,
                    maximum_seconds=args.total_seconds,
                    settings=dict(device=args.device, warmup=args.warmup,
                                   measured_steps=args.steps, threads_requested=args.threads,
                                   generator_vocab=args.generator_vocab,
                                   decoder_positions=args.decoder_positions))
    with (args.output / 'probe.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
    print('一小时探针汇总')
    for row in stages:
        print(row['stage'], row['status'], round(row.get('seconds', 0), 3), 'seconds')
    print('total', round(summary['seconds'], 3), 'seconds')
    if args.auto_stop:
        from .pre_rental_deploy import stop_self
        try:
            stopped = stop_self()
        except Exception as error:
            stopped = dict(action='stop_failed', error_type=type(error).__name__)
        with (args.output / 'auto-stop.json').open('x') as handle:
            json.dump(stopped, handle)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
