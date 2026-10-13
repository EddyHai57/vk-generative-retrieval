"""Strict v2 stats serialization compatibility; no training rewrite."""
import unittest,tempfile
from pathlib import Path
import torch
from vkgr import retrieval_experiment as original
from scripts.p4d_v2_verifier_compat import compatible_module
producer,proof=compatible_module(original)

# Exact historical fingerprint members from the accepted P4D inheritance overlay.
# cache/p4e-entry-overlay.json SHA256:
# 09a6a2effee89cbec1b5bc139a6bc410771d9cfd5361bc2d36439fbe178c53ab
# Literal pins keep this public fixture independent of ignored cache availability.
HISTORICAL_SOURCE = {'.python-version': '49a506dd32096b010d75205acf3430c9ae6c40351888129499e5a5e487126c93',
 'pyproject.toml': '4ec83f6cfcd6318e9c5743804652b6a121ce96bd5d6e92916758313623f4c667',
 'uv.lock': '8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063',
 'src/vkgr/__init__.py': '323fedc682877b25bcd5250f80b22c8d9b213af4d5caaff2494e855ba9e6f1bc',
 'src/vkgr/__main__.py': 'b3261b3886687607d3cc83ebe722bf69719200e17d1d8bd2195d001f8247ec59',
 'src/vkgr/baseline_run.py': '26937152a2f996d9c5f58cde1aba76c70c36a05c3fe64bf66b1ea5ef02439c45',
 'src/vkgr/baselines.py': 'ec92bb0b9858e9da7e192f8cc32da96d5a274ad195413220695b093361b96a0b',
 'src/vkgr/config.py': '540a878c5cfc8945bdfcc083609770e32cae92cebf566f980f9235fe0ba24546',
 'src/vkgr/data_audit.py': 'c7f5f28cdb7e19181fc46f8d69140dba9711ff66f5b8537399692e37e4ded182',
 'src/vkgr/manifest.py': '3b06a12b3129aec87320eebbcc17a5859814e99fb479d57d10a6a8febe415f17',
 'src/vkgr/retrieval.py': '40a9d636cf1f31b08d1fcd0e1501ef5137083d920e40bb3109ae1b31cece810c',
 'src/vkgr/retrieval_baselines.py': 'd49c18a5a63de38f84b6b7ab14805a7d05e7726d99d3bf33bbdf13802754cc39',
 'src/vkgr/retrieval_data.py': '1fa39c85b98ba8aebfabfd0c23659c4a178a1f3f842a5985469656c9deadb867',
 'src/vkgr/retrieval_experiment.py': '4b29d02710582a933c2aed8fc908f0113ee70a7111f09da112c6f5a935f7caa7',
 'src/vkgr/retrieval_models.py': 'a482cfde4ec0b6bf5525fa51d3bf226d8c10a4dc4aac785c1440721ee68f30f2',
 'src/vkgr/retrieval_training.py': 'b947d301087fd8ab74d46ba93a976e01ed876f8dbeb2f615cb75087bffd8e9a6',
 'src/vkgr/runtime.py': 'ede19b05543327ec790f4c59301bd05006c750f89e4c149d75f0193cb4d8f4e8',
 'src/vkgr/sid.py': '9602580f907cc2e8cfa3684aec7828df2bdfc87b59e71b6fd3865a9a19de91a3',
 'src/vkgr/sid_run.py': 'bea7121443d7dd7afb19a96454bceb571a4a58b184416a0e7b7af908c771bfed',
 'src/vkgr/tiger.py': '09ab37d79da8e46dabf522d8d73d6dbd88b5b77985375e459e4977896a18c352',
 'src/vkgr/tiger_run.py': 'ead24365f8ed942f87c1213cc9d099b721f6363bb3f88411eb8a94e80110a517',
 'src/vkgr/training.py': 'b58f3ffe0d7313529c349fca86b00fa1589f65af600d42b957990968bc44ddbd'}

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
 def cli_fixture(self,root,project=None):
  import json,hashlib
  project=Path(__file__).resolve().parents[1] if project is None else project
  names=['.python-version','pyproject.toml','uv.lock']+sorted(name for name in HISTORICAL_SOURCE if name.startswith('src/vkgr/'))
  rebuilt=''.join(f'{name}\0{HISTORICAL_SOURCE[name]}\n' for name in names)
  self.assertEqual(hashlib.sha256(rebuilt.encode()).hexdigest(),proof['trainer_source'])
  pins={**HISTORICAL_SOURCE,'configs/traditional.yaml':'00b8122cc251465506779e79f3820fb7f5a2b674bd22e2d741dd1b65a54bc8d1'}
  payloads={name:(project/name).read_bytes() for name in pins}
  for name,payload in payloads.items():
   self.assertEqual(hashlib.sha256(payload).hexdigest(),pins[name],f'historical fixture member changed: {name}')
  names=list(pins)
  entries=[]
  for name,payload in payloads.items():
   target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
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
   self.assertIn("producer transport mismatch",result.stderr)
 def test_unrelated_project_module_does_not_pollute_historical_fixture(self):
  import shutil,subprocess
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);project=base/'current';project.mkdir()
   source=Path(__file__).resolve().parents[1]
   for name in ('.python-version','pyproject.toml','uv.lock','configs/traditional.yaml'):
    path=project/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,path)
   shutil.copytree(source/'src',project/'src')
   (project/'src/vkgr/future_unrelated.py').write_text('raise RuntimeError("must never import in producer")')
   root=base/'producer';root.mkdir();self.cli_fixture(root,project=project)
   self.assertFalse((root/'src/vkgr/future_unrelated.py').exists())
   result=subprocess.run([__import__('sys').executable,'-B','-c',
    'import sys;sys.path.insert(0,sys.argv[1]);from vkgr.manifest import source_fingerprint;print(source_fingerprint())',str(root/'src')],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr)
   self.assertEqual(result.stdout.strip(),proof['trainer_source'])
 def test_extra_module_in_producer_is_rejected_before_invocation(self):
  import subprocess
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'producer';root.mkdir();project,ledger,runs=self.cli_fixture(root);inv=Path(d)/'invocation.json'
   (root/'src/vkgr/unreviewed.py').write_text('EXTRA = True')
   result=subprocess.run(self.cli_command(project,root,ledger,inv,runs),capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0)
   self.assertIn('producer source mismatch',result.stderr)
   self.assertFalse(inv.exists())
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
