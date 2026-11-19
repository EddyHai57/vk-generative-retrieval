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
import uuid


STAGE_SECONDS = dict(machine=300, two_tower=1200, generator=600,
                     checkpoint=300, evaluate=300)
RETRIEVE_RESULTS = '先拉回结果，再由Eddy销毁'


def case_grid(stage, *, local_smoke=False):
    if stage == 'two_tower':
        if local_smoke:
            return [dict(structure=structure, precision='fp32', id_min_count=100,
                          data_placement='cpu', batch_size=8)
                    for structure in ['mlp_mean', 'mlp_causal']]
        priority = [
            ('mlp_causal', 'bf16', 20, 'cuda'),
            ('mlp_causal', 'fp32', 20, 'cuda'),
            ('mlp_causal', 'bf16', 20, 'cpu'),
            ('mlp_causal', 'bf16', 5, 'cuda'),
            ('mlp_causal', 'tf32', 20, 'cuda'),
            ('mlp_mean', 'bf16', 20, 'cuda'),
            ('mlp_mean', 'fp32', 20, 'cuda'),
        ]
        grid = itertools.product(['mlp_mean', 'mlp_causal'], ['fp32', 'tf32', 'bf16'],
                                 [5, 20, 100], ['cpu', 'cuda'])
        ordered = priority + [case for case in grid if case not in priority]
        return [dict(structure=structure, precision=precision, id_min_count=minimum,
                     data_placement=placement, batch_size=1024)
                for structure, precision, minimum, placement in ordered]
    if local_smoke:
        return [dict(positions=positions, batch_size=1, precision='fp32')
                for positions in [200, 600]]
    return [dict(positions=positions, batch_size=batch, precision=precision)
            for positions, precision, batch in itertools.product(
                [200, 600], ['bf16', 'fp32'], [256, 512, 1024])]


def case_label(stage, case):
    if stage == 'two_tower':
        return '-'.join(str(case[key]) for key in
                        ('structure', 'precision', 'id_min_count', 'data_placement'))
    return f"positions-{case['positions']}-batch-{case['batch_size']}-{case['precision']}"


def case_progress(output, stage, *, local_smoke):
    grid = case_grid(stage, local_smoke=local_smoke)
    report = dict(planned=len(grid), last_attempted_position=0,
                  last_reported_position=0, last_attempted_case=None)
    for position, case in enumerate(grid, 1):
        path = output / case_label(stage, case)
        if path.with_suffix('.launch.json').is_file():
            report.update(last_attempted_position=position, last_attempted_case=case)
        if (path / 'result.json').is_file():
            report['last_reported_position'] = position
    return report


def write_receipt(path, record):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as handle:
        json.dump(record, handle, indent=2)
    temporary.rename(path)


def fp32_summary(stage_output, *, local_smoke):
    path = stage_output / 'fp32-cpu-cuda'
    outcome = path.with_suffix('.outcome.json')
    if local_smoke:
        return dict(status='skipped_cpu')
    if outcome.is_file():
        child = json.loads(outcome.read_text())
        result = child.get('result', {})
        status = child['status'] if child['status'] != 'complete' else 'missing_result'
        result.update(status=status, seconds=child['seconds'], returncode=child['returncode'])
    elif (path / 'result.json').is_file():
        result = json.loads((path / 'result.json').read_text())
    elif path.with_suffix('.launch.json').is_file():
        result = dict(status='interrupted_without_result')
    else:
        return dict(status='not_started')
    if result['status'] not in ('pass', 'skipped_cpu'):
        result['severity'] = 'red'
    return result


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
    command += ['--download-url', args.download_url]
    return command


def run_cases(args, stage):
    from .pre_rental_cases import prepare_shared_order

    started = time.monotonic()
    budget = min(STAGE_SECONDS[stage], args.total_seconds)
    shared = prepare_shared_order(args, args.output) if stage == 'two_tower' else None
    if stage == 'two_tower' and args.device == 'cuda':
        remaining = min(120, budget - (time.monotonic()-started))
        if remaining > 0:
            output = args.output / 'fp32-cpu-cuda'
            command = child_command(args, 'fp32_cpu_cuda', output, shared_order=shared)
            outcome = run_child(command, output, remaining, own_group=False)
            write_receipt(output.with_suffix('.outcome.json'), outcome)
    cases = []
    for specification in case_grid(stage, local_smoke=args.local_smoke):
        remaining = budget - (time.monotonic()-started)
        label = case_label(stage, specification)
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
    write_receipt(args.output / 'result.json', result)
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
    parser.add_argument('--stage', choices=[*STAGE_SECONDS, 'two_tower_case', 'generator_case',
                                           'fp32_cpu_cuda'])
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
            result = dict(status='skipped_total_deadline')
        else:
            command = child_command(args, stage, args.output / stage)
            result = run_child(command, args.output / stage, min(limit, remaining))
        result['stage'] = stage
        if stage in ('two_tower', 'generator'):
            result['case_progress'] = case_progress(args.output / stage, stage,
                                                     local_smoke=args.local_smoke)
        stages.append(result)
    parity = fp32_summary(args.output / 'two_tower', local_smoke=args.local_smoke)
    summary = dict(execution='local_cpu_smoke' if args.local_smoke else 'cuda_probe',
                    cuda_cases_skipped=args.local_smoke, stages=stages,
                    fp32_cpu_cuda=parity, auto_stop=args.auto_stop,
                    evaluation_model_source=stages[-1].get('result', {}).get('model_source'),
                    next_action=RETRIEVE_RESULTS,
                    seconds=time.monotonic()-started,
                    maximum_seconds=args.total_seconds,
                    settings=dict(device=args.device, warmup=args.warmup,
                                   measured_steps=args.steps, threads_requested=args.threads,
                                   generator_vocab=args.generator_vocab,
                                   decoder_positions=args.decoder_positions))
    write_receipt(args.output / 'probe.json', summary)
    print('一小时探针汇总')
    for row in stages:
        print(row['stage'], row['status'], round(row.get('seconds', 0), 3), 'seconds')
        if 'case_progress' in row:
            progress = row['case_progress']
            print('  已启动到第', progress['last_attempted_position'], '组，已记录到第',
                  progress['last_reported_position'], '组，共', progress['planned'], '组')
    if parity.get('severity') == 'red':
        print(f"\033[31mFP32 CPU/CUDA 对拍未通过（红）{parity['status']}\033[0m")
    else:
        print('FP32 CPU/CUDA 对拍', parity['status'])
    print('total', round(summary['seconds'], 3), 'seconds')
    print('evaluation model', summary['evaluation_model_source'])
    print(RETRIEVE_RESULTS)
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
