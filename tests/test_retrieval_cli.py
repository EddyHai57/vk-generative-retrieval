import unittest,subprocess,sys,tempfile,json
from pathlib import Path
class CLITests(unittest.TestCase):
 def test_synthetic_overfit_and_fresh_process_resume(self):
  import importlib.util
  self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval'),'CLI pipeline missing')
  def run(*args):
   proc=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval',*args],capture_output=True,text=True)
   self.assertEqual(proc.returncode,0,proc.stderr);return proc
  with tempfile.TemporaryDirectory() as tmp:
   a=Path(tmp)/'a';b=Path(tmp)/'b'
   run('train','--synthetic','--output',str(a),'--max-steps','20','--until-step','20')
   run('train','--synthetic','--output',str(b),'--max-steps','20','--until-step','7')
   run('train','--synthetic','--output',str(b),'--max-steps','20','--until-step','20','--resume',str(b/'manifest-step-7.json'))
   ma=json.loads((a/'manifest-step-20.json').read_text());mb=json.loads((b/'manifest-step-20.json').read_text())
   self.assertLess(ma['records'][-1]['loss'],ma['records'][0]['loss']*.1);self.assertEqual(ma['records'],mb['records']);self.assertEqual([len(r['batch']) for r in ma['records'][:4]],[3,2,3,2])
   import torch
   ca=torch.load(a/ma['last']['path'],weights_only=True);cb=torch.load(b/mb['last']['path'],weights_only=True)
   for name,value in ca['model'].items():self.assertTrue(torch.equal(value,cb['model'][name]))
   run('verify','--manifest',str(b/'manifest-step-20.json'))
   c=Path(tmp)/'c'
   run('train','--synthetic','--output',str(c),'--max-steps','20','--until-step','8')
   run('train','--synthetic','--output',str(c),'--max-steps','20','--until-step','20','--resume',str(c/'manifest-step-8.json'))
   mc=json.loads((c/'manifest-step-20.json').read_text());self.assertEqual(ma['records'],mc['records'])
   cc=torch.load(c/mc['last']['path'],weights_only=True)
   for name,value in ca['model'].items():self.assertTrue(torch.equal(value,cc['model'][name]))
 def test_audit_and_prepare_stop_on_missing_pins_before_output(self):
  from vkgr import retrieval
  self.assertTrue(hasattr(retrieval,'audit_inputs'),'audit input pipeline missing')
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises((ValueError,FileNotFoundError)):retrieval.audit_inputs(Path(tmp)/'missing',Path(tmp)/'missing',Path(tmp)/'missing')
 def test_resource_gate_rejects_insufficient_headroom(self):
  from vkgr import retrieval
  self.assertTrue(hasattr(retrieval,'require_resources'),'resource gate missing')
  retrieval.require_resources({'ram':.2,'vram':.3,'disk':.5})
  with self.assertRaises(ValueError):retrieval.require_resources({'ram':.19,'vram':.9,'disk':.9})
 def test_legacy_producer_manifest_filename(self):
  from vkgr.retrieval import audit_inputs
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'frozen-contract-manifest.json').write_text('{}')
   try:
    with self.assertRaisesRegex(ValueError,'hash mismatch'):audit_inputs(root,root,root)
   except FileNotFoundError:self.fail('consumer requested incorrect legacy manifest filename')
