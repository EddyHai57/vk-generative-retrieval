"""Queue metadata file transport and deployment import-root regression guards."""
import copy,hashlib,importlib.util,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
def ops():
 spec=importlib.util.spec_from_file_location('stack_request_ops',ROOT/'scripts/seq_combo_stack_ops.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class RequestTests(unittest.TestCase):
 def setup_request(self,tmp):
  m=ops();self.assertTrue(hasattr(m,'read_request'),'request file interface missing')
  t=Path(tmp);req={'action':'preflight','paths':{k:str(t/k) for k in m.PATH_KEYS}};p=t/'request.json';p.write_text(json.dumps(req));return m,req,p,hashlib.sha256(p.read_bytes()).hexdigest()
 def test_positional_request_file_works_with_stdin_eof(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,req,p,pin=self.setup_request(tmp);out=io.StringIO()
   with mock.patch.object(m.sys,'stdin',mock.Mock(read=mock.Mock(side_effect=AssertionError('file mode must not read stdin')))),mock.patch.object(m.sys,'stdout',out),mock.patch.object(m,'preflight',return_value={'status':'preflight_ok'}):m.main(['preflight','--request',str(p),'--request-sha',pin,'--action','preflight'])
   self.assertEqual(json.loads(out.getvalue())['status'],'preflight_ok')
 def test_file_tamper_rejected_before_dispatch(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,req,p,pin=self.setup_request(tmp);p.write_text(json.dumps({**req,'action':'submit'}))
   with self.assertRaises(ValueError):m.read_request(['preflight','--request',str(p),'--request-sha',pin])
 def test_all_actions_must_match_and_bad_schema_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,req,p,pin=self.setup_request(tmp)
   for args in (['dry','--request',str(p),'--request-sha',pin],['preflight','--action','submit','--request',str(p),'--request-sha',pin]):
    with self.assertRaises(ValueError):m.read_request(args)
   p.write_text('[]');pin=hashlib.sha256(p.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):m.read_request(['--request',str(p),'--request-sha',pin])
 def test_stdin_compatibility_and_empty_stdin_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,req,p,pin=self.setup_request(tmp)
   with mock.patch.object(m.sys,'stdin',io.StringIO(json.dumps(req))):self.assertEqual(m.read_request([]),req)
   with mock.patch.object(m.sys,'stdin',io.StringIO('')):
    with self.assertRaises(ValueError):m.read_request([])
 def test_checkpoint_and_test_split_metadata_paths_never_opened(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,req,p,pin=self.setup_request(tmp)
   for name in ('best.pt','test-queries.json'):
    bad=Path(tmp)/name;bad.write_text('{}')
    with mock.patch.object(Path,'read_bytes',side_effect=AssertionError('protected path must not be read')):
     with self.assertRaises(ValueError):m.read_request(['--request',str(bad),'--request-sha','a'*64])

class EnvironmentTests(unittest.TestCase):
 def template_job(self):
  return {'run_id':'old-template','argv':['python','-B','-m','old.module','train','--inputs','i','--inputs-sha','a','--config','c','--config-sha','b','--output','o'],'inputs':[],'env':['OMP_NUM_THREADS=1','PYTHONPATH=<OLD_ROOT>/src','KEEP_FROZEN=value'],'cpu_cores':1,'ram_mib':9216,'no_admission':False,'disk_gib':8,'wall_hours':8}
 def test_verifier_and_smoke_replace_old_pythonpath_preserve_other_env(self):
  m=ops()
  with tempfile.TemporaryDirectory() as tmp:
   m.configure({k:str(Path(tmp)/k) for k in m.PATH_KEYS});job=self.template_job();job['run_id']=m.RIDS[0]
   verify=m.derive_verify(job,'f'*64)
   self.assertIn('PYTHONPATH='+str(m.DEP/'src'),verify['env']);self.assertNotIn('PYTHONPATH=<OLD_ROOT>/src',verify['env']);self.assertIn('KEEP_FROZEN=value',verify['env'])
   self.assertTrue(hasattr(m,'make_smoke_job'),'smoke builder must bind deployment env')
   with mock.patch.object(m,'template',return_value=self.template_job()):smoke=m.make_smoke_job()
   self.assertEqual([e for e in smoke['env'] if e.startswith('PYTHONPATH=')],['PYTHONPATH='+str(m.DEP/'src')]);self.assertIn('OMP_NUM_THREADS=1',smoke['env'])
 def test_all_three_train_jobs_replace_old_template_import_root(self):
  m=ops()
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);m.configure({k:str(t/k) for k in m.PATH_KEYS});m.OPS.mkdir();(m.OPS/'parameters.json').write_text('{}');(m.DEP/'configs/seq_combo_stack').mkdir(parents=True)
   source={'git_commit':'a'*40,'git_dirty':False};meta={'commit':'a'*40,'inputs_sha256':'b'*64};entries={}
   for rid in m.RIDS:
    cp=m.DEP/f'configs/seq_combo_stack/{rid}.json';cp.write_bytes((ROOT/f'configs/seq_combo_stack/{rid}.json').read_bytes());entries[rid]={'engineering':False,'config_sha256':m.sha(cp),'leakage_receipt':{}}
   smoke=m.R/'runs'/m.SMOKE;(smoke/'out').mkdir(parents=True);(smoke/'terminal.json').write_text(json.dumps({'status':'succeeded'}));proof={'vram':{'variants':[{'declared_vram_mib':1024}]}};(smoke/'out/engineering.json').write_text(json.dumps(proof))
   req={'combo_review_sha256':'f'*64,'preflight':{'status':'preflight_ok','commit':meta['commit'],'inputs_sha256':meta['inputs_sha256'],'source':source,'entries':entries},'engineering':proof}
   with mock.patch.object(m,'check_combo_review'),mock.patch.object(m,'identity',return_value=meta),mock.patch('vkgr.retrieval_b1_run.source_identity',return_value=source),mock.patch.object(m,'check_engineering'),mock.patch.object(m,'check_leakage_receipt'),mock.patch.object(m,'template',return_value=self.template_job()):jobs=m.train_jobs(req)
   self.assertEqual(len(jobs),3)
   for e in jobs:
    self.assertEqual([v for v in e['job']['env'] if v.startswith('PYTHONPATH=')],['PYTHONPATH='+str(m.DEP/'src')]);self.assertIn('KEEP_FROZEN=value',e['job']['env']);self.assertIn('--data-parameters',e['job']['argv'])
