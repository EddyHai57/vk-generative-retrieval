import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'ops/bin/vkgr-fixed'


def load_server():
    spec = importlib.util.spec_from_file_location('fixed_server', ROOT / 'ops/fixed/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedOpsTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([str(ENTRY), *args], capture_output=True, text=True)

    def test_formal_plan_nonempty_and_fixed(self):
        run = self.cli('plan')
        self.assertEqual(run.returncode, 0, run.stderr)
        doc = json.loads(run.stdout)
        self.assertEqual(doc['commands'], ['status', 'resources', 'queue', 'receipts', 'deploy', 'plan'])
        self.assertEqual(doc['deploy_commit'], '661a444870dbb07b4fbcd69cebb57f25c691bb26')

    def test_all_actions_have_formal_dry_receipts(self):
        for verb in ('status', 'resources', 'queue', 'receipts', 'deploy'):
            run=self.cli(verb, 'plan')
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout)['action'],verb)

    def test_all_extra_arguments_and_forbidden_verbs_fail_before_network(self):
        for args in [(), ('push', 'main'), ('submit',), ('rm',), ('holdout',), ('deploy-existing',)]:
            with self.subTest(args=args):
                self.assertNotEqual(self.cli(*args).returncode, 0)
        for verb in ('status', 'resources', 'queue', 'receipts', 'deploy', 'plan'):
            for extra in ('week26', '/tmp/test', ';', '--config', '--help', 'main', '$(id)'):
                with self.subTest(verb=verb, extra=extra):
                    self.assertEqual(self.cli(verb, 'execute', extra).returncode, 2)

    def test_remote_parser_rejects_own_extra_arguments(self):
        for args in [('status', 'extra'), ('deploy', 'existing'), ('push', 'main'), ('test',)]:
            run = subprocess.run([sys.executable, '-I', str(ROOT/'ops/fixed/server.py'), *args], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)

    def test_bounded_receipt_refuses_symlink_and_oversize(self):
        m = load_server()
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp).resolve()/'receipt.json'
            p.write_text('{"ok":true}')
            self.assertEqual(m.small_json(p), {'ok': True})
            alias = Path(temp).resolve()/'alias.json'; alias.symlink_to(p)
            with self.assertRaises(ValueError): m.small_json(alias)
            p.write_bytes(b'{}' + b' ' * (m.RECEIPT_LIMIT - 1))
            with self.assertRaises(ValueError): m.small_json(p)

    def test_deploy_existing_targets_refused_before_any_git(self):
        m = load_server()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();target=root/'target';pins=root/'pins'
            target.mkdir()
            with self.assertRaises(ValueError): m.new_targets(target,pins)
            dangling=root/'dangling';dangling.symlink_to(root/'absent')
            with self.assertRaises(ValueError): m.new_targets(dangling,pins)
            pins.mkdir()
            with self.assertRaises(ValueError): m.new_targets(root/'fresh',pins)

    def test_status_uses_json_mode_and_receipts_omit_env(self):
        m=load_server()
        calls=[]
        def capture(argv, **kwargs):
            calls.append([str(a) for a in argv])
            return '{"host": {"cpu_total": 8}, "runs": []}'
        m.check_ops=lambda: None
        m.call=capture
        self.assertEqual(m.dispatch('resources'),{'host':{'cpu_total':8}})
        self.assertIn('--json',calls[0])
        with tempfile.TemporaryDirectory() as temp:
            m.ROOT=Path(temp).resolve()
            for rid in m.RUNS:
                d=m.ROOT/'runs'/rid;d.mkdir(parents=True)
                (d/'run.json').write_text(json.dumps({'run_id':rid,'env':{'SECRET':'fixture'},'resources':{'cpu':1}}))
            result=m.dispatch('receipts')
            self.assertNotIn('env',result[m.RUNS[0]]['run.json'])
            self.assertEqual(result[m.RUNS[0]]['run.json']['run_id'],m.RUNS[0])

    def test_transport_banner_and_schema_validation(self):
        spec=importlib.util.spec_from_file_location('fixed_local',ROOT/'ops/fixed/local.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        body={'schema':'vkgr-fixed-ops-receipt-v1','action':'queue','result':{}}
        self.assertEqual(m.remote_receipt('untrusted login banner\n'+json.dumps(body)+'\n','queue'),body)
        for text in ('', '{}',json.dumps(dict(body,action='deploy'))):
            with self.assertRaises(ValueError):m.remote_receipt(text,'queue')

    def test_pin_directory_race_is_refused(self):
        m=load_server()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();target=root/'code';pins=root/'pins'
            original=m.new_targets
            def race(a,b):
                original(a,b)
                pins.mkdir()
                (pins/'foreign').write_text('preserve')
            m.new_targets=race
            m.call=lambda *a, **k: json.dumps({'tree':{}})
            with self.assertRaises(FileExistsError):
                m.deploy_tree(root,target,pins,'0'*40,tools=ROOT)
            self.assertEqual((pins/'foreign').read_text(),'preserve')

    def test_reused_tools_match_original_blobs(self):
        # Pinned from original 5840e6b blobs; independent public clones need no private history.
        expected = {
            'ops/tools/local/deploy_spec.py': '1382787f43a240729a0adf6b5873e4fbd2bb9ec8a21a52951bae1a4455a8999c',
            'ops/tools/remote/g_deploy.py': '5338241831645bf861c341548968a465afc30677280131e112ac11a490c4b85b',
        }
        for rel, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(), digest)

    def test_deploy_git_worktree_and_hashes_real(self):
        m = load_server()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();repo=root/'repo';repo.mkdir()
            def git(*args):
                return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
            git('init','-q');git('config','user.email','fixture@example.invalid');git('config','user.name','fixture')
            (repo/'sample.txt').write_text('synthetic fixture\n');git('add','.');git('commit','-qm','fixture')
            sha=git('rev-parse','HEAD')
            result=m.deploy_tree(repo,root/'fresh',root/'pins',sha,tools=ROOT)
            self.assertEqual(result['target_tree_bad'],[])
            self.assertEqual(result['head'],sha)
            self.assertEqual((root/'fresh/sample.txt').read_bytes(),(repo/'sample.txt').read_bytes())
            with self.assertRaises(ValueError):m.deploy_tree(repo,root/'fresh',root/'pins',sha,tools=ROOT)


if __name__ == '__main__':
    unittest.main()
