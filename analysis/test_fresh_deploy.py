import json,tempfile,unittest
from pathlib import Path
from analysis.fresh_deploy import inventory,validate
class DeployTest(unittest.TestCase):
 def req(self,t):return {'root':str(Path(t)/'opsroot'),'source':str(Path(t)/'source'),'destination':str(Path(t)/'newcode'),'ops':str(Path(t)/'newops'),'branch':'codex/new-batch','commit':'a'*40,'files':{'src/x.py':'b'*64},'run_ids':['x-train']}
 def test_inventory_detects_existing_queue_and_run(self):
  with tempfile.TemporaryDirectory() as t:
   r=self.req(t);root=Path(r['root']);(root/'queue/claimed').mkdir(parents=True);(root/'runs/x-train').mkdir(parents=True)
   (root/'queue/claimed/001-x-train.json').write_text('{}')
   got=inventory(r);self.assertFalse(got['clear']);self.assertTrue(got['jobs']['x-train']['run_exists'])
 def test_private_paths_and_safe_file_names(self):
  with tempfile.TemporaryDirectory() as t:
   r=self.req(t);validate(r)
   r['files']={'../data':'b'*64}
   with self.assertRaises(ValueError):validate(r)

class NewOpsDocumentsTest(unittest.TestCase):
 def req(self,t):
  r=DeployTest().req(str(Path(t).resolve()))
  data=b'print("pinned source")\n';r['files']={'src/x.py':__import__('hashlib').sha256(data).hexdigest()}
  return r,data
 def mock_git(self,r,data):
  from unittest.mock import patch
  import contextlib
  stack=contextlib.ExitStack()
  def run(argv,**kwargs):
   if 'worktree' in argv:
    p=Path(r['destination'])/'src';p.mkdir(parents=True);(p/'x.py').write_bytes(data)
  stack.enter_context(patch('analysis.fresh_deploy.subprocess.run',side_effect=run))
  stack.enter_context(patch('analysis.fresh_deploy.subprocess.check_output',side_effect=[r['commit']+'\n','']))
  return stack
 def test_new_parameters_and_pinned_input_metadata_use_actual_file_bytes(self):
  import hashlib
  from analysis.fresh_deploy import deploy
  with tempfile.TemporaryDirectory(dir='/private/tmp') as t:
   r,data=self.req(t);inp=Path(t)/'accepted-inputs.json';inp.write_text('{"train": "private"}\n')
   r['inputs_file_pin']={'path':str(inp),'sha256':hashlib.sha256(inp.read_bytes()).hexdigest()}
   params={'root':str(Path(t)/'data'),'inputs':str(inp)}
   r['new_ops_documents']={'parameters.json':params,'templates.json':{'purpose':'new-only'},'approval.json':{'approved':True},'preflight-request.json':{'action':'preflight'}}
   pin=inventory(r)['request_sha256']
   with self.mock_git(r,data):result=deploy(r,pin)
   ops=Path(r['ops']);record=json.loads((ops/'deployment.json').read_text())
   self.assertEqual(json.loads((ops/'parameters.json').read_text()),params)
   self.assertEqual((ops/'parameters.json').read_bytes(),json.dumps(params,sort_keys=True).encode('utf-8'))
   self.assertTrue(all((ops/name).exists() for name in r['new_ops_documents']))
   self.assertEqual(record['parameters_sha256'],hashlib.sha256((ops/'parameters.json').read_bytes()).hexdigest())
   self.assertEqual(record['inputs_sha256'],r['inputs_file_pin']['sha256'])
   self.assertEqual(inp.read_text(),'{"train": "private"}\n')
   self.assertEqual(result['status'],'deployed')
 def test_legacy_request_result_and_metadata_field_sets_unchanged(self):
  from analysis.fresh_deploy import deploy
  with tempfile.TemporaryDirectory(dir='/private/tmp') as t:
   r,data=self.req(t);pin=inventory(r)['request_sha256']
   with self.mock_git(r,data):result=deploy(r,pin)
   self.assertEqual(set(result),{'status','commit','files_checked','data_overlay'})
   self.assertEqual(set(json.loads((Path(r['ops'])/'deployment.json').read_text())),{'commit','files','created_at_utc','data_overlay'})
 def test_bad_input_hash_or_non_json_rejected_before_any_git_or_directory_creation(self):
  import hashlib
  from unittest.mock import patch
  from analysis.fresh_deploy import deploy
  with tempfile.TemporaryDirectory(dir='/private/tmp') as t:
   r,data=self.req(t);inp=Path(t)/'inputs.json';inp.write_text('not json')
   for sha in ('0'*64,hashlib.sha256(inp.read_bytes()).hexdigest()):
    r['inputs_file_pin']={'path':str(inp),'sha256':sha}
    with patch('analysis.fresh_deploy.subprocess.run') as git:
     with self.assertRaises(ValueError):deploy(r,'0'*64)
     git.assert_not_called()
    self.assertFalse(Path(r['destination']).exists());self.assertFalse(Path(r['ops']).exists())
 def test_unsafe_names_reserved_metadata_or_relative_parameters_rejected(self):
  with tempfile.TemporaryDirectory(dir='/private/tmp') as t:
   r,_=self.req(t)
   for documents in ({'../parameters.json':{}},{'sub/parameters.json':{}},{'deployment.json':{}},{'parameters.json':{'root':'relative'}},{'parameters.json':{'root':3}}):
    r['new_ops_documents']=documents
    with self.assertRaises(ValueError):validate(r)
 def test_existing_ops_or_destination_never_overwritten(self):
  from unittest.mock import patch
  from analysis.fresh_deploy import deploy
  with tempfile.TemporaryDirectory(dir='/private/tmp') as t:
   r,_=self.req(t);r['new_ops_documents']={'parameters.json':{'root':str(Path(t)/'data')}}
   ops=Path(r['ops']);ops.mkdir();marker=ops/'parameters.json';marker.write_text('existing bytes')
   with patch('analysis.fresh_deploy.subprocess.run') as git:
    with self.assertRaises(ValueError):deploy(r,'0'*64)
    git.assert_not_called()
   self.assertEqual(marker.read_text(),'existing bytes')
 def test_input_changed_after_inventory_or_ambiguous_json_is_rejected(self):
  import hashlib
  from unittest.mock import patch
  from analysis.fresh_deploy import deploy
  with tempfile.TemporaryDirectory(dir='/private/tmp') as t:
   r,_=self.req(t);inp=Path(t)/'input.json';inp.write_text('{}')
   r['inputs_file_pin']={'path':str(inp),'sha256':hashlib.sha256(inp.read_bytes()).hexdigest()};pin=inventory(r)['request_sha256']
   inp.write_text('{"changed":true}')
   with patch('analysis.fresh_deploy.subprocess.run') as git:
    with self.assertRaises(ValueError):deploy(r,pin)
    git.assert_not_called()
   for text in ('{"duplicate":1,"duplicate":2}','{"value":NaN}','{"value":1e999}'):
    inp.write_text(text);r['inputs_file_pin']['sha256']=hashlib.sha256(inp.read_bytes()).hexdigest()
    with self.assertRaises(ValueError):validate(r)
 def test_existing_worktree_never_touched(self):
  from unittest.mock import patch
  from analysis.fresh_deploy import deploy
  with tempfile.TemporaryDirectory(dir='/private/tmp') as t:
   r,_=self.req(t);dest=Path(r['destination']);dest.mkdir();marker=dest/'keep.txt';marker.write_text('unchanged')
   r['new_ops_documents']={'parameters.json':{'root':str(Path(t)/'data')}}
   with patch('analysis.fresh_deploy.subprocess.run') as git:
    with self.assertRaises(ValueError):deploy(r,inventory(r)['request_sha256'])
    git.assert_not_called()
   self.assertEqual(marker.read_text(),'unchanged');self.assertFalse(Path(r['ops']).exists())
