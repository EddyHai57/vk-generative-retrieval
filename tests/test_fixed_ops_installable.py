import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location('installable_'+name,ROOT/'ops/fixed'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


class InstallableOpsTests(unittest.TestCase):
    def test_source_execute_refused_without_published_lookup_or_ssh(self):
        # Repository bytes are development source, never an authorized runtime.
        r=subprocess.run([str(ROOT/'ops/bin/vkgr-fixed'),'status','execute'],capture_output=True,text=True)
        self.assertEqual(r.returncode,1)
        self.assertIn('installation',r.stderr)

    def test_package_new_only_root_controlled_destination_and_six_absolute_rules(self):
        m=load('package')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve()/'bundle'
            install=Path('/Library/VKGRFixedOps/fixture-v2')
            doc=m.build_package(out,install)
            self.assertEqual(doc['version'],'vkgr-fixed-ops-v2')
            for rel in ('bin/vkgr-fixed','fixed/local.py','fixed/server.py','install.json','manifest.json'):
                self.assertTrue((out/rel).is_file(),rel)
            cfg=json.loads((out/'install.json').read_text())
            self.assertEqual(cfg['install_root'],str(install))
            self.assertEqual(Path(cfg['private_receipts']),m.private_receipts())
            manifest=json.loads((out/'manifest.json').read_text())
            for rel,digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((out/rel).read_bytes()).hexdigest(),digest)
            draft=(out/'rendered.rules.draft').read_text()
            self.assertEqual(draft.count('prefix_rule('),6)
            self.assertNotIn('@INSTALLED_ENTRY@',draft)
            self.assertIn(str(install/'bin/vkgr-fixed'),draft)
            self.assertIn('"deploy", "execute"',draft)
            with self.assertRaises(FileExistsError):m.build_package(out,install)
            with self.assertRaises(ValueError):m.build_package(Path(temp)/'bad',Path(temp)/'runtime')

    def test_installed_layout_bytes_modes_owner_and_ancestors(self):
        package=load('package');local=load('local')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()/'bundle'
            package.build_package(root,Path('/Library/VKGRFixedOps/fixture-v2'))
            # Test the payload validator separately. This is NOT an installed-root proof.
            self.assertEqual(local.payload(root,expected_owner=os.getuid(),require_readonly=False)['version'],local.VERSION)
            (root/'fixed/server.py').write_text('# mutated\n')
            with self.assertRaises(ValueError):local.payload(root,expected_owner=os.getuid(),require_readonly=False)
        with self.assertRaises(ValueError):local.installation()

    def test_private_receipts_fixed_not_repository_or_environment(self):
        package=load('package');local=load('local')
        self.assertEqual(local.private_receipts(),package.private_receipts())
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp).resolve()/'private';p.mkdir(mode=0o700)
            run=subprocess.CompletedProcess([],1,stdout='fixture-out',stderr='fixture-error')
            local.private_receipts=lambda:p
            a=Path(local.save_transport(run));b=Path(local.save_transport(run))
            self.assertNotEqual(a,b)
            self.assertEqual(a.parent,p)
            self.assertEqual(a.stat().st_mode&0o777,0o600)
            p.chmod(0o755)
            with self.assertRaises(ValueError):local.save_transport(run)

    def test_install_config_and_payload_identity_cannot_retarget(self):
        package=load('package');local=load('local')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()/'bundle'
            package.build_package(root,Path('/Library/VKGRFixedOps/fixture-v2'))
            (root/'install.json').write_text('{}')
            with self.assertRaises(ValueError):local.payload(root,expected_owner=os.getuid(),require_readonly=False)

    def test_installed_root_rejects_writable_ancestor_and_symlinks(self):
        local=load('local')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();p=root/'install';p.mkdir()
            with self.assertRaises(ValueError):local.root_chain(p)
            link=root/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):local.root_chain(link)

    def test_install_checks_and_probe_refuse_absent_layout_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            for script in ('verify_install.py','sandbox_write_probe.py'):
                argv=[sys.executable,'-I',str(ROOT/'ops/fixed'/script),'--install-root',str(root/'absent')]
                if script=='verify_install.py':argv += ['--codex','/usr/bin/false','--output',str(root/'out')]
                r=subprocess.run(argv,capture_output=True,text=True)
                self.assertEqual(r.returncode,1,r.stderr)
                self.assertIn('PENDING',r.stderr)
                self.assertFalse((root/'out').exists())

    def test_root_chain_rejects_acl_write_access(self):
        from unittest.mock import patch
        m=load('local')
        with patch.object(m.os,'access',return_value=True):
            with self.assertRaises(ValueError):m.root_chain(Path('/'))

    def test_acl_parser_fail_closed_and_deny_allowed(self):
        m=load('local')
        m.check_acl_text('dr-xr-xr-x 1 root wheel 0 fixture\n')
        m.check_acl_text('dr-xr-xr-x+ 1 root wheel 0 fixture\n 0: everyone deny delete\n')
        for ace in (' 0: everyone allow delete_child', ' 0: everyone allow writesecurity',
                    ' 0: everyone inherited allow read', 'unrecognized ACL'):
            with self.assertRaises(ValueError):m.check_acl_text('metadata\n'+ace+'\n')
        with self.assertRaises(ValueError):m.check_acl_text('')

    @unittest.skipUnless(sys.platform=='darwin','native macOS ACL fixture')
    def test_real_private_acl_refused_before_receipt_content(self):
        m=load('local')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp).resolve()/'private';p.mkdir(mode=0o700)
            subprocess.run(['/bin/chmod','+a','everyone allow read,readattr,readextattr,readsecurity,file_inherit,directory_inherit',str(p)],check=True)
            m.private_receipts=lambda:p
            with self.assertRaises(ValueError):m.save_transport(subprocess.CompletedProcess([],0,stdout='secret',stderr=''))
            self.assertEqual(list(p.iterdir()),[])
            subprocess.run(['/bin/chmod','-N',str(p)],check=True)
            m.check_private_dir()
            subprocess.run(['/bin/chmod','+a','everyone allow delete_child',str(p)],check=True)
            with self.assertRaises(ValueError):m.check_acl(p)
            subprocess.run(['/bin/chmod','-N',str(p)],check=True)



if __name__=='__main__':unittest.main()
