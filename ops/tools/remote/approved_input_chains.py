"""Finite aa72a86 input/A1 handoff only; never submit full SGNS fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/workspace')
PYTHON = BASE / 'vkgr-env-a40-v1/.venv/bin/python'
OPS_CODE = BASE / 'vkgr-ops-code-a40-v1'
OPS = BASE / 'vkgr-ops-root-a40-v1'
CHAINS = {'a1': ('preflight', 'train', 'verify'), 'full-input': ('preflight', 'prepare')}
SOURCES = {'a1': 'b7254990e8f8babbd77ed9b61c6339599616eaea',
           'full-input': '951c680cc77ceaf17cd9fb91e294f1da193fa447'}
CODES = {'a1': BASE / 'vkgr-sgns-subset-a1-context-retry1-code-v1',
         'full-input': BASE / 'vkgr-full-gpu-sgns-ready-code-v2'}
A1_REQUEST = BASE / 'vkgr-sgns-subset-a1-acceptance-request-retry1-v1.json'
A1_SHA = '0874f7efb77ed0d04845ff4ce30583b267388f4ebc76700a33acce28e136525e'


def terminal(run, run_id):
    path = run / 'terminal.json'
    if not path.exists():
        return False
    value = json.loads(path.read_bytes())
    if (value.get('run_id') != run_id or value.get('status') != 'succeeded'
            or value.get('exit_code') != 0 or value.get('signal') is not None
            or value.get('wall_cap_exceeded') is not False):
        raise ValueError('original terminal is not successful')
    return True


def submit_argv(spec):
    argv = [str(PYTHON), '-B', str(OPS_CODE / 'ops/bin/vkgr-sched'), 'submit', '--root', str(OPS)]
    for name in ('run_id', 'kind', 'cwd', 'config', 'seed', 'cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib', 'wall_hours'):
        argv += ['--' + name.replace('_', '-'), str(spec[name])]
    for name, flag in (('env', '--env'), ('inputs', '--input')):
        for value in spec[name]:
            argv += [flag, value]
    return argv + ['--'] + spec['argv']


def command(argv, code, evidence, name):
    result = subprocess.run(argv, cwd=code, capture_output=True, check=False)
    with (evidence / (name + '.stdout')).open('xb') as handle:
        handle.write(result.stdout)
    with (evidence / (name + '.stderr')).open('xb') as handle:
        handle.write(result.stderr)
    if result.returncode:
        raise ValueError('command failed: ' + name)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', choices=tuple(CHAINS), required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if os.getppid() != 1:
        raise ValueError('detached PPID=1 required')
    args.evidence.mkdir(parents=True, exist_ok=False)
    return run_chain(args)


def run_chain(args):
    code = CODES[args.chain]
    os.environ.update(PYTHONPATH=str(code / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path[:0] = [str(code), str(code / 'src')]
    os.chdir(code)
    deadline = time.time() + (96 if args.chain == 'a1' else 48) * 3600
    if args.chain == 'a1':
        from analysis import sgns_subset_a1 as module
        if hashlib.sha256(A1_REQUEST.read_bytes()).hexdigest() != A1_SHA:
            raise ValueError('fixed A1 request differs')
        value = module.read_request(A1_REQUEST, A1_SHA, consume=False)
        ids = module.core.RUN_IDS
        def dry(stage):
            return [str(PYTHON), '-B', '-m', 'analysis.sgns_subset_a1', 'prepare', '--stage', stage,
                    '--request', str(A1_REQUEST), '--request-sha256', A1_SHA, '--ops-code', str(OPS_CODE)]
        def accepted(stage):
            source, env = module.identity(value)
            if stage != 'verify':
                module.predecessor(value, 'train' if stage == 'preflight' else 'verify', A1_SHA, source, request=A1_REQUEST)
                return
            run = OPS / 'runs' / ids[stage]
            lines = [line for line in (run / 'logs/stdout.log').read_bytes().splitlines() if line.strip()]
            receipt = json.loads(lines[-1])
            artifact = Path(value['output_root']) / stage / 'stage-receipt.json'
            run_sha = module.job_binding(value, A1_REQUEST, A1_SHA, stage)
            _, _, _, proof = module.core.engine_for(value, A1_SHA, source)
            expected = {'source': source, 'runtime': env, 'request_sha256': A1_SHA,
                        'run_manifest_sha256': run_sha, 'run_id': ids[stage], 'stage': stage,
                        'verified': True, 'status': 'verified', 'test_read': False, 'hits_gate': 378,
                        'schema': 'gpu-sgns-subset-a1-stage-v1', 'authority': 'ad1c428',
                        'engineering': False, 'training': False, 'subset_four_gate_acceptance': False,
                        'resource_scope': module.RESOURCE_SCOPE,
                        'config': module.core.registered_config(value['gpu_sid']['manifest']['sha256']),
                        'input_producers': proof}
            if (json.loads(artifact.read_bytes()) != receipt
                    or set(receipt) != set(expected) | {'validation_hits', 'native_verification',
                        'context_inclusive_peak_mib', 'context_samples', 'rss_tree_upper_bound_mib'}
                    or any(module.canonical(receipt.get(k)) != module.canonical(v) for k, v in expected.items())
                    or type(receipt.get('validation_hits')) is not int or receipt['validation_hits'] < 378
                    or receipt.get('native_verification', {}).get('status') != 'verified'):
                raise ValueError('original A1 verification closure differs')
            module.measured_resources(receipt)
    else:
        from analysis import sgns_gpu_full as module
        ids = {stage: 'full-gpu-sgns-input-' + stage + '-ready-v2' for stage in CHAINS[args.chain]}
        def dry(stage):
            return [str(PYTHON), '-B', '-m', 'analysis.sgns_gpu_full', 'queue-dry',
                    '--data', str(BASE / 'vkgr-ur001-data-20261020-v1'), '--embedding-root', str(BASE / 'vkgr-ur001-data-20261020-v1'),
                    '--stage2', str(BASE / 'vkgr-UR001-full-derivation-v1'), '--stage2-run', str(OPS / 'runs/ur001-full-derive-v1'),
                    '--lock', str(BASE / 'vkgr-env-a40-v1/uv.lock'), '--output', str(BASE / 'vkgr-full-gpu-sgns-input-ready-v2'),
                    '--deployment-sha', SOURCES[args.chain], '--stage', stage, '--ops-code', str(OPS_CODE),
                    '--ops-root', str(OPS), '--run-id', ids[stage],
                    '--preflight-run', str(OPS / 'runs' / ids['preflight'])]
        def accepted(stage):
            run = OPS / 'runs' / ids[stage]
            receipt = module.last_receipt(run / 'logs/stdout.log')
            source = module.source_identity()
            expected = {'schema': 'gpu-sgns-full-input-v1', 'engineering': False,
                        'training': False, 'test_read': False,
                        'status': 'verified' if stage == 'preflight' else 'prepared',
                        'verified': stage == 'preflight', 'payload_written': stage == 'prepare'}
            if (any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items())
                    or receipt.get('source') != source
                    or receipt.get('runtime') != module.runtime(BASE / 'vkgr-env-a40-v1/uv.lock')):
                raise ValueError('original full input formal/source/runtime differs')
            job = json.loads((run / 'run.json').read_bytes())
            module.producer(run, job['argv'], source, receipt,
                            expected_job=module.full_job_contract(run, job['argv'], source))
            if stage == 'prepare':
                output = BASE / 'vkgr-full-gpu-sgns-input-ready-v2'
                if receipt.get('status') != 'prepared' or json.loads((output / 'manifest.json').read_bytes()) != receipt:
                    raise ValueError('original prepared manifest differs')
                if set(receipt.get('artifacts', {})) != {'catalog.json', 'content.npy', 'histories.jsonl'}:
                    raise ValueError('full artifact inventory differs')
                for name, pin in receipt['artifacts'].items():
                    if module.digest(output / name) != pin:
                        raise ValueError('full prepared artifact differs')
    stages = CHAINS[args.chain]
    for index, stage in enumerate(stages):
        run_id = ids[stage]
        while True:
            if time.time() >= deadline:
                raise TimeoutError('finite chain deadline')
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() != SOURCES[args.chain]:
                raise ValueError('frozen deployment source differs')
            run = OPS / 'runs' / run_id
            states = {state: list((OPS / 'queue' / state).glob('*-' + run_id + '.json')) for state in ('pending', 'claimed', 'rejected')}
            if sum(map(len, states.values())) > 1:
                raise ValueError('duplicate original run ID in queue')
            if states['rejected']:
                raise ValueError('original job rejected; no resubmit')
            if terminal(run, run_id):
                accepted(stage)
                break
            if not run.exists() and not any(states.values()):
                if index == 0:
                    raise ValueError('initial job must already be submitted by owner')
                raw = command(dry(stage), code, args.evidence, stage + '-dry')
                proof = json.loads(raw)
                if proof.get('status') != 'prepared_dry' or proof['spec']['run_id'] != run_id or proof['metadata']['git'] != {'commit': SOURCES[args.chain], 'dirty': False, 'dirty_files': []}:
                    raise ValueError('formal dry identity differs')
                command(submit_argv(proof['spec']), code, args.evidence, stage + '-submit')
            with (args.evidence / 'progress.jsonl').open('a') as handle:
                handle.write(json.dumps({'stage': stage, 'time': time.time(), 'queue': {key: len(paths) for key, paths in states.items()}}) + '\n')
            time.sleep(900)
    with (args.evidence / 'summary.json').open('x') as handle:
        json.dump({'status': 'stages_terminal', 'chain': args.chain, 'full_fit_submitted': False,
                   'requires_owner_final_acceptance': True, 'ended_at': time.time()}, handle)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # The normal driver stdout is redirected to a unique private receipt.
        print(json.dumps({'status': 'stopped', 'error_type': type(error).__name__,
                          'error': str(error), 'full_fit_submitted': False}), flush=True)
        raise
