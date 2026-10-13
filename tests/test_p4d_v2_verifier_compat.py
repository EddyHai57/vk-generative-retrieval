"""Strict v2 stats serialization compatibility; no training rewrite."""
import unittest,tempfile
from pathlib import Path
import torch
from vkgr import retrieval_experiment as original
from scripts.p4d_v2_verifier_compat import compatible_module
producer,proof=compatible_module(original)

class CompatibilityTests(unittest.TestCase):
 def session(self,root,epochs):
  torch.set_num_threads(1)
  rows=[{'history':[i],'target':i+3,'past_likes':[i]} for i in range(1,4)]
  config={'architecture':'id_mean','learning_rate':.001,'temperature':.07,'seed':42,'batch_size':3}
  binding={'config':config,'origin':'explicit_synthetic_compatibility_fixture'}
  result=producer.run_session(root,rows,torch.zeros(7,64),list(range(1,7)),{i:i for i in range(1,7)},config,binding,lambda model,e:{'metrics':{'all':{'recall@50':e/40,'ndcg@50':e/80}}},'cpu',until_step=epochs)
  return root/result['manifest_name'],rows
 def test_eleven_and_thirty_epochs_accept_equal_checkpoint_json_stats(self):
  with tempfile.TemporaryDirectory() as d:
   for epochs in (11,30):
    path,rows=self.session(Path(d)/str(epochs),epochs)
    value=producer.verify_session(path)
    producer.verify_draw_provenance(value,{'rows':rows,'idmap':{i:i for i in range(1,7)}})
    self.assertEqual(value['completed_epochs'],epochs)
 def test_draw_reconciliation_accepts_thirty_equal_epochs(self):
  from vkgr.manifest import read_json
  with tempfile.TemporaryDirectory() as d:
   path,rows=self.session(Path(d)/'run',30)
   value=read_json(path)
   cursor=producer.verify_draw_provenance(value,{'rows':rows,'idmap':{i:i for i in range(1,7)}})
   self.assertEqual(cursor['epoch'],29)
 def test_illegal_epoch_representations_and_count_changes_are_rejected(self):
  import copy,json
  from vkgr.manifest import read_json
  from scripts.p4d_v2_verifier_compat import stats_bytes
  valid={'drawn':3,'skipped':0,'epochs':{0:[3,0]}}
  self.assertEqual(stats_bytes(valid),stats_bytes({'drawn':3,'skipped':0,'epochs':{'0':[3,0]}}))
  illegal=[{'00':[3,0]},{'-1':[3,0]},{-1:[3,0]},{True:[3,0]},{0.0:[3,0]},{0:[3,0],'0':[3,0]},{'30':[3,0]},{'0':[3.0,0]},{'0':[3,False]}]
  for epochs in illegal:
   with self.subTest(epochs=epochs),self.assertRaises(ValueError):stats_bytes({**valid,'epochs':epochs})
  with tempfile.TemporaryDirectory() as d:
   path,rows=self.session(Path(d)/'run',11);value=read_json(path)
   bad=copy.deepcopy(value);bad['stats']['drawn']+=1
   path.write_text(json.dumps(bad))
   with self.assertRaisesRegex(ValueError,'last checkpoint/manifest mismatch'):producer.verify_session(path)
   with self.assertRaisesRegex(ValueError,'skip denominator reconciliation mismatch'):producer.verify_draw_provenance(bad,{'rows':rows,'idmap':{i:i for i in range(1,7)}})
   for epochs in illegal:
    bad=copy.deepcopy(value);bad['stats']['epochs']=epochs
    with self.assertRaises(ValueError):producer.verify_draw_provenance(bad,{'rows':rows,'idmap':{i:i for i in range(1,7)}})
 def test_proof_changes_only_two_validators_and_original_module_is_untouched(self):
  import types
  from scripts.p4d_v2_verifier_compat import REPLACEMENTS
  self.assertEqual(set(proof['changed_validators']),set(REPLACEMENTS))
  for name,value in original.__dict__.items():
   if isinstance(value,types.FunctionType) and value.__module__==original.__name__:
    if name not in REPLACEMENTS:self.assertIs(getattr(producer,name).__code__,value.__code__)
    else:self.assertIsNot(getattr(producer,name).__code__,value.__code__)
  self.assertNotIn('_p4d_v2_stats_bytes',original.__dict__)
  for change in proof['changed_validators'].values():self.assertEqual(change['stats_calls_replaced'],2)
 def test_four_architectures_resume_after_eleven_epochs_matches_exact_states(self):
  from vkgr.retrieval_models import MODEL_ORDER
  def equal(a,b):
   if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
   elif isinstance(a,dict):
    self.assertEqual(a.keys(),b.keys())
    for k in a:equal(a[k],b[k])
   elif isinstance(a,(list,tuple)):
    self.assertEqual(len(a),len(b))
    for x,y in zip(a,b):equal(x,y)
   else:self.assertEqual(a,b)
  rows=[{'history':[i],'target':i+5,'past_likes':[i]} for i in range(1,6)]
  torch.set_num_threads(1)
  for arch in MODEL_ORDER:
   with self.subTest(architecture=arch),tempfile.TemporaryDirectory() as d:
    root=Path(d);cfg={'architecture':arch,'learning_rate':.001,'temperature':.07,'seed':42,'batch_size':3};binding={'config':cfg,'scope':'explicit synthetic long resume'}
    args=(rows,torch.zeros(11,64),list(range(1,11)),{i:i for i in range(1,11)},cfg,binding,lambda model,e:{'metrics':{'all':{'recall@50':e/40,'ndcg@50':e/80}}},'cpu')
    a=producer.run_session(root/'a',*args)
    b=producer.run_session(root/'b',*args,until_step=23)
    self.assertEqual(b['completed_epochs'],11)
    b=producer.run_session(root/'b',*args,until_step=26,resume=root/'b'/b['manifest_name'])
    b=producer.run_session(root/'b',*args,resume=root/'b'/b['manifest_name'])
    for key in ('records','evaluations','stats','selection','completed_epochs','steps','stop_reason'):equal(a[key],b[key])
    self.assertEqual((a['completed_epochs'],a['steps'],a['stop_reason']),(30,60,'max_epochs'))
    for result,path in ((a,root/'a'),(b,root/'b')):
     producer.verify_session(path/result['manifest_name'],binding)
     producer.verify_draw_provenance(result,{'rows':rows,'idmap':{i:i for i in range(1,11)}})
    ca=torch.load(root/'a'/a['last']['path'],weights_only=True);cb=torch.load(root/'b'/b['last']['path'],weights_only=True)
    for key in ('model','optimizer','cursor','stats','torch_rng','python_rng','numpy_rng','cuda_rng'):equal(ca[key],cb[key])
    # Durations and content hashes differ across execution schedules; consumed
    # budget, checkpoint logical refs and validation selection must be identical.
    for key in ('max_epochs','records','evaluations','selection','best'):equal(ca['budget'][key],cb['budget'][key])
 def cli_fixture(self,root):
  import json,hashlib
  project=Path(__file__).resolve().parents[1]
  # Frozen v2 source members from the reviewed 105-file transport ledger.
  # Ledger bytes SHA == adapter ENTRY_SHA; public_files digest == LEDGER_SHA.
  # Explicit allowlist: new consumers must never enter this historical producer.
  names=['.python-version','pyproject.toml','uv.lock','configs/traditional.yaml',
         'src/vkgr/__init__.py',
         'src/vkgr/__main__.py',
         'src/vkgr/baseline_run.py',
         'src/vkgr/baselines.py',
         'src/vkgr/config.py',
         'src/vkgr/data_audit.py',
         'src/vkgr/manifest.py',
         'src/vkgr/retrieval.py',
         'src/vkgr/retrieval_baselines.py',
         'src/vkgr/retrieval_data.py',
         'src/vkgr/retrieval_experiment.py',
         'src/vkgr/retrieval_models.py',
         'src/vkgr/retrieval_training.py',
         'src/vkgr/runtime.py',
         'src/vkgr/sid.py',
         'src/vkgr/sid_run.py',
         'src/vkgr/tiger.py',
         'src/vkgr/tiger_run.py',
         'src/vkgr/training.py',
        ]
  entries=[]
  for name in names:
   target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((project/name).read_bytes())
  for i in range(105-len(names)):
   target=root/f'fixture/public-{i}.txt';target.parent.mkdir(exist_ok=True);target.write_text('explicit synthetic public fixture')
   names.append(target.relative_to(root).as_posix())
  for name in sorted(names):
   b=(root/name).read_bytes();entries.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
  (root/'cache').mkdir();(root/'cache/p4d-source-entry.json').write_text(json.dumps({'source_fingerprint':proof['trainer_source']}))
  ledger=root.parent/'ledger.json';ledger.write_text(json.dumps({'source_fingerprint':proof['trainer_source'],'public_files':entries,'transport_origin_sha256':hashlib.sha256((root/'cache/p4d-source-entry.json').read_bytes()).hexdigest()}))
  runs=root.parent/'runs.json';runs.write_text('[]')
  return project,ledger,runs
 def cli_command(self,project,root,ledger,inv,runs,fixture_pin=True):
  import sys
  # Only the static ledger pin is replaced with the independently constructed
  # synthetic fixture pin. Real path/bytes/hashes/fingerprint and main execute.
  setup="import importlib.util,json,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('adapter',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
  if fixture_pin:setup+="m.LEDGER_SHA=m.digest(m.encoded(json.loads(Path(sys.argv[5]).read_text())['public_files']));m.ENTRY_SHA=json.loads(Path(sys.argv[5]).read_text())['transport_origin_sha256'];"
  setup+="raise SystemExit(m.main(sys.argv[2:]))"
  return [sys.executable,'-c',setup,str(project/'scripts/p4d_v2_verifier_compat.py'),'--producer-root',str(root),'--ledger',str(ledger),'--invocation',str(inv),'--','select','--run-list',str(runs),'--output',str(root.parent/'selection.json')]
 def test_complete_producer_pin_and_explicit_cli_identity(self):
  import subprocess,sys,json
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'producer';root.mkdir();project,ledger,runs=self.cli_fixture(root);inv=Path(d)/'invocation.json'
   cmd=self.cli_command(project,root,ledger,inv,runs)
   rejected=subprocess.run(self.cli_command(project,root,ledger,inv,runs,fixture_pin=False),capture_output=True,text=True)
   self.assertNotEqual(rejected.returncode,0);self.assertFalse(inv.exists())
   result=subprocess.run(cmd,capture_output=True,text=True)
   self.assertEqual(result.returncode,2,result.stderr)
   value=json.loads(inv.read_text());receipt=json.loads(Path(str(inv)+'.result.json').read_text())
   self.assertEqual(value['trainer_source'],proof['trainer_source']);self.assertEqual(value['proof'],proof)
   self.assertEqual(receipt['producer_exit'],2);self.assertEqual(receipt['status'],'failed_stop')
   (root/'src/vkgr/retrieval_models.py').write_text('tampered source')
   other=Path(d)/'other.json';cmd[cmd.index(str(inv))]=str(other)
   result=subprocess.run(cmd,capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertFalse(other.exists())
 def test_existing_result_sidecar_stops_before_producer_invocation(self):
  import subprocess,sys
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'producer';root.mkdir();project,ledger,runs=self.cli_fixture(root);inv=Path(d)/'invocation.json'
   Path(str(inv)+'.result.json').write_text('preserved previous evidence')
   result=subprocess.run(self.cli_command(project,root,ledger,inv,runs),capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertFalse(inv.exists(),'must preflight both exclusive sidecars before any producer command')
 def test_select_accepts_complete_sixteen_long_runs_using_same_adapter(self):
  import argparse,json
  torch.set_num_threads(1)
  rows=[{'history':[i],'target':i+3,'past_likes':[i]} for i in range(1,4)]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);paths=[]
   for index,base in enumerate(producer.core_configs()):
    cfg={**base,'batch_size':3};binding={'config':cfg,'origin':'fixed synthetic long selection','pilot_manifest_sha256':'fixed synthetic shared pilot','consumer':{'source_fingerprint':proof['trainer_source']}}
    value=producer.run_session(root/str(index),rows,torch.zeros(7,64),list(range(1,7)),{i:i for i in range(1,7)},cfg,binding,lambda model,e:{'metrics':{'all':{'recall@50':e/40,'ndcg@50':e/80}}},'cpu')
    paths.append(str(root/str(index)/value['manifest_name']))
   run_list=root/'run-list.json';run_list.write_text(json.dumps(paths))
   selection=producer.select_core(argparse.Namespace(run_list=str(run_list),output=str(root/'selection.json')))
   self.assertEqual(selection['status'],'success');self.assertEqual(len(selection['all_tuning_runs']),16)
   self.assertEqual(selection['input_binding']['shared_batch_size'],3)
   self.assertEqual(selection['best_architecture'],'id_mean')
   for name,item in selection['selected_configs'].items():
    self.assertEqual(item['config'],next(c for c in producer.core_configs() if c['architecture']==name))
 def test_transport_origin_metadata_tamper_stops_before_invocation(self):
  import subprocess,json
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'producer';root.mkdir();project,ledger,runs=self.cli_fixture(root);inv=Path(d)/'invocation.json'
   origin=root/'cache/p4d-source-entry.json';value=json.loads(origin.read_text());value['git_commit']='invented provenance';origin.write_text(json.dumps(value))
   result=subprocess.run(self.cli_command(project,root,ledger,inv,runs),capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertFalse(inv.exists(),'transport origin bytes must remain pinned')

 def test_frozen_fixture_ignores_new_consumers_and_pins_original_members(self):
  import json,hashlib,shutil
  from unittest.mock import patch
  from scripts.p4d_v2_verifier_compat import PRODUCER_SOURCE
  actual=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory() as d:
   shadow=Path(d)/'current';shadow.mkdir()
   for name in ('.python-version','pyproject.toml','uv.lock','configs/traditional.yaml'):
    dest=shadow/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(actual/name,dest)
   (shadow/'src/vkgr').mkdir(parents=True)
   # Deliberately simulate arbitrary future consumers; fixture must not glob them.
   for path in (actual/'src/vkgr').glob('*.py'):shutil.copyfile(path,shadow/'src/vkgr'/path.name)
   (shadow/'src/vkgr/future_consumer.py').write_text('raise RuntimeError("not a producer")')
   root=Path(d)/'producer';root.mkdir()
   with patch(__name__+'.__file__',str(shadow/'tests/test_p4d_v2_verifier_compat.py')):
    project,ledger,runs=self.cli_fixture(root)
   names={rec['path'] for rec in json.loads(ledger.read_text())['public_files']}
   self.assertNotIn('src/vkgr/generative_foundation.py',names)
   self.assertNotIn('src/vkgr/future_consumer.py',names)
   base=['.python-version','pyproject.toml','uv.lock']
   source=sorted(name for name in names if name.startswith('src/vkgr/') and name.endswith('.py'))
   self.assertEqual(len(source),19)
   encoded=''.join(f'{name}\0{hashlib.sha256((root/name).read_bytes()).hexdigest()}\n' for name in base+source)
   self.assertEqual(hashlib.sha256(encoded.encode()).hexdigest(),PRODUCER_SOURCE)

 def test_missing_or_rehashed_tampered_producer_members_rejected(self):
  import json,hashlib,subprocess
  for mutation in ('missing','tampered','rehashed'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
    root=Path(d)/'producer';root.mkdir();project,ledger,runs=self.cli_fixture(root)
    member=root/'src/vkgr/retrieval_models.py'
    if mutation=='missing':member.unlink()
    else:member.write_text('tampered source')
    if mutation=='rehashed':
     value=json.loads(ledger.read_text())
     for entry in value['public_files']:
      if entry['path']=='src/vkgr/retrieval_models.py':entry.update(bytes=member.stat().st_size,sha256=hashlib.sha256(member.read_bytes()).hexdigest())
     ledger.write_text(json.dumps(value))
    inv=Path(d)/'invocation.json'
    result=subprocess.run(self.cli_command(project,root,ledger,inv,runs),capture_output=True,text=True)
    self.assertNotEqual(result.returncode,0)
    self.assertFalse(inv.exists());self.assertFalse(Path(str(inv)+'.result.json').exists())
    self.assertIn('producer source mismatch' if mutation=='rehashed' else 'producer transport mismatch',result.stderr)
