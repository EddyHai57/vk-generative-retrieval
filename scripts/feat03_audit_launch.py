P = None
# FEAT-03 audit launcher. Committed with the placeholder above; the sender replaces ONLY line 1 with a one-line JSON
# object {worktree, commit, tree, script_sha256, inputs_spec, src}. Everything below is read-only: it verifies that the
# checked-out worktree is exactly the pushed commit (HEAD, tree, business-script hash, no edits or extra files under
# src/ and scripts/) and then runs the business script with its stdout passed through untouched.
import hashlib
import json
import runpy
import subprocess
import sys

sys.dont_write_bytecode = True


def _refuse(reason):
    raise SystemExit('audit launcher refused: ' + reason)


def _git(*args):
    try:
        return subprocess.run(['git', '-C', P['worktree'], *args], capture_output=True, text=True)
    except OSError:
        _refuse('git unavailable')


if type(P) is not dict or set(P) != {'worktree', 'commit', 'tree', 'script_sha256', 'inputs_spec', 'src'}:
    _refuse('parameters missing or malformed')
if any(type(value) is not str or not value for value in P.values()):
    _refuse('parameters must be non-empty strings')

head = _git('rev-parse', 'HEAD')
if head.returncode != 0 or head.stdout.strip() != P['commit']:
    _refuse('HEAD does not match the expected commit')
tree = _git('rev-parse', 'HEAD^{tree}')
if tree.returncode != 0 or tree.stdout.strip() != P['tree']:
    _refuse('tree does not match the expected tree')

SCRIPT = P['worktree'].rstrip('/') + '/scripts/feat03_audit_constants.py'
try:
    with open(SCRIPT, 'rb') as handle:
        script_sha256 = hashlib.sha256(handle.read()).hexdigest()
except OSError:
    _refuse('business script unreadable')
if script_sha256 != P['script_sha256']:
    _refuse('business script hash does not match')

if _git('diff', '--quiet', 'HEAD', '--', 'src', 'scripts').returncode != 0:
    _refuse('tracked files under src/ or scripts/ differ from the commit')
untracked = _git('ls-files', '--others', '--exclude-standard', '--', 'src', 'scripts')
if untracked.returncode != 0 or untracked.stdout.strip():
    _refuse('untracked files present under src/ or scripts/')

sys.stderr.write(json.dumps({'commit': P['commit'], 'tree': P['tree'], 'script_sha256': script_sha256}, sort_keys=True) + '\n')
sys.stderr.flush()
sys.argv = [SCRIPT, '--inputs-spec', P['inputs_spec'], '--src', P['src']]
runpy.run_path(SCRIPT, run_name='__main__')
