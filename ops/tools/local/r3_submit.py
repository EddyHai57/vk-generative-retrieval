"""R3 reference-run submit driver (LOCAL). Reads an R3 manifest (schema deploy01-r3-manifest-v2) and sends exactly one
step per invocation through transport.py. Nothing retries automatically.

usage: r3_submit.py --config C --manifest M --n N --step sign|train|parity (--confirm-reviewed | --dry-run)
  sign  : launch the permit-sign job (permit + dry validation with the real loaders).
  train : read the sign run's receipt (fresh readonly read), require status=permit_signed_and_dry_validated,
          fill <PERMIT_SHA256>, then submit/launch the train job (VKGR_EVAL_WORKERS=W from the manifest).
  parity: only after the train run terminal is succeeded (checked again by the server tool): submit the decode-only
          parity job (own pool / own decode budget; w1 vs wW on the same best checkpoint).
--confirm-reviewed is required for any real send (formal-run review gate). --dry-run prints the exact payload(s)
and transport argv and sends nothing (no socket needed); for 'train' it shows the template with <PERMIT_SHA256>.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolcfg  # noqa: E402

SCHEMA = 'deploy01-r3-manifest-v2'
STEPS = {'sign': 'step1_sign_permit', 'train': 'step2_train', 'parity': 'step3_parity_decode_only'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='R3 submit driver (one step per call)')
    p.add_argument('--config')
    p.add_argument('--manifest', required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--step', choices=tuple(STEPS), required=True)
    p.add_argument('--confirm-reviewed', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.dry_run and a.confirm_reviewed:
        p.error('--dry-run and --confirm-reviewed are exclusive')
    if not a.dry_run and not a.confirm_reviewed:
        p.error('refuse: formal run submission needs --confirm-reviewed (orchestrator review gate)')
    return a


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    if m.get('schema') != SCHEMA:
        raise SystemExit(f'manifest schema must be {SCHEMA}')
    for key in STEPS.values():
        step = m.get(key)
        if not isinstance(step, dict) or step.get('tool') not in ('launch', 'submit'):
            raise SystemExit(f'manifest {key}.tool must be launch|submit')
        args = step.get('launch_args') or step.get('launch_args_template')
        if not isinstance(args, list) or '--run-id' not in args or '--cwd' not in args:
            raise SystemExit(f'manifest {key} launch args need --run-id and --cwd')
    return m


def payload(m, step, permit_sha=None):
    s = m[STEPS[step]]
    if step == 'train':
        args = s['launch_args_template']
        if permit_sha is not None:
            if len(permit_sha) != 64:
                raise SystemExit('permit sha must be 64 hex')
            args = [permit_sha if x == '<PERMIT_SHA256>' else x for x in args]
    else:
        args = s['launch_args']
    return {'tool': s['tool'], 'launch_args': list(args), 'tamper': None}


def transport_argv(script, prefix, n, config, inp=None):
    argv = [sys.executable, '-B', str(Path(__file__).with_name('transport.py')), str(toolcfg.REMOTE / script), prefix,
            '--n', str(n), '--timeout', '180']
    if config:
        argv += ['--config', config]
    return argv + (['--input', str(inp)] if inp else ['--readonly'])


def sign_receipt(logs, run_id):
    """Validated permit sha from the sign run in a runlogs_read.py result."""
    s = logs.get(run_id)
    if not s or not s.get('terminal') or s['terminal'].get('status') != 'succeeded':
        raise SystemExit('sign run not succeeded')
    rec = json.loads(s['stdout.log'].strip().splitlines()[-1])
    if rec.get('status') != 'permit_signed_and_dry_validated':
        raise SystemExit('permit not validated')
    return rec['permit_sha256']


def main(argv=None):
    a = parse_args(argv)
    m = load_manifest(a.manifest)
    if a.dry_run:
        out = {'dry_run': True, 'step': a.step, 'payload': payload(m, a.step),
               'transport_argv': transport_argv('launch.py', f'r3-{a.step}-<ts>', a.n, a.config, '<payload file>')}
        if a.step == 'train':
            out['first_reads'] = transport_argv('runlogs_read.py', 'r3-signread-<ts>', a.n, a.config)
            out['sign_run_id'] = m[STEPS['sign']]['launch_args'][m[STEPS['sign']]['launch_args'].index('--run-id') + 1]
        print(json.dumps(out, indent=1))
        return 0
    cfg = toolcfg.load(a.config)
    here = toolcfg.state_dir(cfg)
    ts = time.strftime('%Y%m%dt%H%M%S', time.gmtime())

    def send(script, prefix, inp=None):
        r = subprocess.run(transport_argv(script, prefix, a.n, a.config, inp), capture_output=True, text=True)
        print(r.stdout, r.stderr, file=sys.stderr)
        return json.loads((here / (prefix + '.stdout')).read_text())
    permit_sha = None
    if a.step == 'train':
        sign_args = m[STEPS['sign']]['launch_args']
        logs = send('runlogs_read.py', f'r3-signread-{ts}')
        permit_sha = sign_receipt(logs, sign_args[sign_args.index('--run-id') + 1])
    body = payload(m, a.step, permit_sha)
    inp = here / f'r3-in-{a.step}-{ts}.json'
    inp.write_text(json.dumps(body))
    print(json.dumps(send('launch.py', f'r3-{a.step}-{ts}', inp), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
