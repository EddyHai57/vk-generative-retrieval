"""Read-only installed-copy and execpolicy checks. Does not install rules or run remote actions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]


def helper(name):
    spec = importlib.util.spec_from_file_location('fixed_check_'+name, ROOT/'ops/fixed'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_install(root):
    local, package = helper('local'), helper('package')
    local.ROOT = root
    manifest = local.installation()
    for rel, source in package.RUNTIME.items():
        if (root/rel).read_bytes() != (ROOT/source).read_bytes():
            raise ValueError('installed runtime differs from this reviewed release')
    return {'version': local.VERSION, 'manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
            'runtime_files': {rel: {'sha256': manifest['files'][rel], 'uid': (root/rel).stat().st_uid,
                                  'mode': oct((root/rel).stat().st_mode & 0o777)} for rel in manifest['files']}}


def checks(root, codex, output):
    snapshot = inspect_install(root)
    output.mkdir(mode=0o700, exist_ok=False)
    entry = str(root/'bin/vkgr-fixed')
    rules = output/'rendered.rules.draft'
    rules.write_text(helper('package').render_rules(Path(entry)))
    cases = [[entry, 'plan']] + [[entry, a, 'execute'] for a in ('status','resources','queue','receipts','deploy')]
    cases += [['git','push','origin','main'], [entry,'deploy','existing'],
              ['bash','-lc',entry+' status execute && rm arbitrary'], [entry,'deploy','execute','--target','existing']]
    results = []
    for i, argv in enumerate(cases):
        command = [str(codex), 'execpolicy', 'check', '--pretty', '--rules', str(rules), '--', *argv]
        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (output/f'policy-{i:02}.stdout').write_text(run.stdout)
        (output/f'policy-{i:02}.stderr').write_text(run.stderr)
        doc = json.loads(run.stdout)
        expected = i < 6 or i == 9
        if run.returncode or bool(doc.get('matchedRules')) != expected or (expected and doc.get('decision') != 'allow'):
            raise ValueError('installed-path execpolicy check differs')
        results.append({'argv': argv, 'exit': run.returncode, 'result': doc})
    # Only the dry plan and deliberately invalid argv are executed locally.
    plan = subprocess.run([entry, 'plan'], check=True, capture_output=True, text=True, timeout=10)
    if json.loads(plan.stdout).get('version') != snapshot['version']:
        raise ValueError('installed plan identity differs')
    for action in ('status','resources','queue','receipts','deploy'):
        run = subprocess.run([entry, action, 'execute', 'reject-extra'], capture_output=True, text=True, timeout=10)
        if run.returncode != 2:
            raise ValueError('installed parser accepted extra argument')
    report = {'status': 'checked_not_installed_rules', 'install': snapshot, 'policy': results,
              'installed_plan': json.loads(plan.stdout), 'invalid_argument_tests': 5,
              'remote_actions_executed': 0, 'rules_installed': False}
    (output/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-root', required=True, type=Path)
    p.add_argument('--codex', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args(argv)
    try:
        report = checks(args.install_root, args.codex, args.output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('PENDING/FAILED installed-copy verification: '+type(error).__name__, file=sys.stderr)
        return 1
    print(json.dumps({'status':report['status'], 'policy_cases':len(report['policy']), 'rules_installed':False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
