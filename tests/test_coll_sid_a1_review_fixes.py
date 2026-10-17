"""Fresh review regressions: committed clean bytes and real launcher envelope."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_coll_sid_a1_ops import module

class SourceIdentity(unittest.TestCase):
    def fixture(self,root,m):
        def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
        git('init','-q')
        files={}
        for name in ('src/vkgr/coll_sid_a1_run.py','scripts/coll_sid_a1_ops.py'):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('committed bytes\n');files[name]=m.sha(p)
        git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
        return {'deployment':str(root),'source_commit':git('rev-parse','HEAD'),'files':files},git
    def exercise(self,assume):
        from vkgr import coll_sid_a1_run as run
        m=module()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            root=Path(tmp);req,git=self.fixture(root,m)
            with patch.object(run,'G6_MEMBERS',()):
                m.identity(req)
                name='src/vkgr/coll_sid_a1_run.py'
                if assume:git('update-index','--assume-unchanged',name)
                (root/name).write_text('request-resigned dirty bytes\n');req['files'][name]=m.sha(root/name)
                if assume:self.assertEqual(git('status','--porcelain'), '')
                with self.assertRaises(ValueError):m.identity(req)
    def test_dirty_deployment_cannot_resign_hash(self):self.exercise(False)
    def test_hidden_dirty_file_must_match_pinned_git_blob(self):self.exercise(True)

class Envelope(unittest.TestCase):
    BIG={'payload':'x'*65500}  # prepared alone serializes to 65515; wrapper is larger.
    def test_dry_run_rejects_real_prepared_65552_byte_payload(self):
        m=module();job={'run_id':'g-coll-a1-c1-s42','config':'cfg','vram_mib':8192}
        pf={'status':'preflight_ok','source_commit':'a'*40,'entries':[{'job':dict(job,vram_mib=None),'config_sha256':'c'*64} for _ in range(3)]}
        proof={'status':'cuda_engineering_verified','source_commit':'a'*40,'reload_decode_equal':True,'synthetic_grad_steps':2,'declared_vram_mib':8192}
        for key in ('data_binding_sha256','sid_manifest_sha256','runtime','model_shape'):pf[key]=proof[key]='bound'
        req={'preflight_receipt':'pf','cuda_receipt':'cuda','source_commit':'a'*40}
        huge={'payload':'x'*65537};self.assertEqual(len(json.dumps(huge).encode()),65552)
        with patch.object(m,'identity'),patch.object(m,'load',side_effect=[pf,proof]),patch.object(m,'validate_vram',return_value=8192),patch.object(m,'derive_job',return_value=job),patch.object(m,'sha',return_value='c'*64),patch.object(m,'launcher_prepare',return_value=huge):
            with self.assertRaisesRegex(ValueError,'envelope'):m.dry_run(req)
    def test_train_publish_rejects_before_intent(self):
        m=module()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);ops=root/'ops';ops.mkdir();q=root/'queue'
            for name in ('pending','claimed','rejected'):(q/name).mkdir(parents=True)
            req={'ops':str(ops),'queue':str(q),'runs':str(root/'runs')}
            with patch.object(m,'launcher_prepare',return_value=self.BIG),patch.object(m,'publish') as publish:
                results,failed=m.publish_intents(req,{'entries':[{'job':{'run_id':'g-coll-a1-c1-s42'}}]})
                self.assertTrue(failed);publish.assert_not_called()
            self.assertEqual(list(ops.glob('publication-intent-*')),[])
    def test_verify_publish_rejects_before_intent(self):
        m=module()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);ops=root/'ops';ops.mkdir();q=root/'queue';runs=root/'runs'
            for name in ('pending','claimed','rejected'):(q/name).mkdir(parents=True)
            rid='g-coll-a1-c1-s42';out=runs/rid/'out';out.mkdir(parents=True)
            cp=root/'config.json';cp.write_text('{}');(runs/rid/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
            binding={'prepared_manifest_sha256':'p'*64,'history_manifest_sha256':'h'*64,'context_binding_sha256':'c'*64}
            (out/'manifest.json').write_text(json.dumps({'config':{},'binding':binding}))
            job={'run_id':rid,'seed':42,'config':str(cp),'inputs':[],'argv':['python','-m',m.ENTRY,'train','--config',str(cp),'--config-sha','a'*64,'--output','o','--scratch','s']}
            (q/'claimed'/('001-'+rid+'.json')).write_text(json.dumps(job))
            req={'ops':str(ops),'queue':str(q),'runs':str(runs),'source_commit':'a'*40,'coll_refs':{'sid_manifest':{'sha256':'s'*64}}};entry={'job':job,'config_sha256':'c'*64,'expected_binding_sha256':m.digest(binding)}
            with patch.object(m,'identity',return_value=SimpleNamespace(_digest=m.digest)),patch.object(m,'launcher_prepare',return_value=self.BIG),patch.object(m,'publish') as publish:
                with self.assertRaisesRegex(ValueError,'envelope'):m.observe(req,entry)
                publish.assert_not_called()
            self.assertEqual(list(ops.glob('verify-intent-*')),[])

class EnvelopeBoundary(unittest.TestCase):
    def test_final_envelope_47103_allowed_47104_refused(self):
        m=module();job={'run_id':'synthetic'}
        base=len(json.dumps({'spec':job,'prepared':{'payload':''}},allow_nan=False).encode())
        with patch.object(m,'launcher_prepare',return_value={'payload':'x'*(47103-base)}):
            m.prepare_envelope({},job)
        with patch.object(m,'launcher_prepare',return_value={'payload':'x'*(47104-base)}):
            with self.assertRaisesRegex(ValueError,'envelope'):m.prepare_envelope({},job)
