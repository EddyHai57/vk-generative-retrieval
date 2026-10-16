"""Fresh-seed entry isolation, replay, mutations and history oracles."""
import contextlib,copy,importlib,io,json,sys,tempfile,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_seqm1_model as tm
import test_seqm1_leak as leak
from test_seqm1_run import guard_holdout_never_opened
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(314159,271828,161803)
def runner():return importlib.import_module('vkgr.retrieval_biconfirm_run')
def small(arm='mean',seed=314159):return {**runner().default_config(),'arm':arm,'seed':seed,'batch_size':16,'max_epochs':3,'min_epochs':2}
class ConfirmationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_new_entry_exists(self):self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_biconfirm_run'),'dedicated fresh-seed entry required')
 def test_registry_accepts_only_new_seeds_two_masks_and_old_entry_stays_closed(self):
  run=runner()
  from vkgr import retrieval_seqm1_run as old
  for seed in SEEDS:
   for arm in ('mean','bi'):
    c={**run.default_config(),'seed':seed,'arm':arm};self.assertEqual(run.validate_config(c),c)
    with self.assertRaises(ValueError):old.validate_config({**old.default_config(),'seed':seed})
  for change in ({'seed':42},{'seed':2024},{'seed':2025},{'seed':True},{'seed':314159.0},{'arm':'attn'},{'arm':'pure'},{'temperature':.07},{'max_epochs':101},{'test_gate':'open'},{'extra_uniform':1},{'dim':128}):
   with self.assertRaises(ValueError,msg=change):run.validate_config({**run.default_config(),**change})
  for bad in ({},[],None,{**run.default_config(),'unknown':1}):
   with self.assertRaises(ValueError):run.validate_config(bad)
 def test_six_configs_frozen_and_one_variable_per_pair(self):
  run=runner();paths=sorted((ROOT/'configs/biconfirm').glob('*.json'));self.assertEqual(len(paths),6);seen=set()
  original=json.loads((ROOT/'configs/seqm1/p-seqm1-mean-s42.json').read_text())
  for path in paths:
   v=json.loads(path.read_text());run.validate_config(v);seen.add((v['arm'],v['seed']))
   self.assertEqual({k:x for k,x in v.items() if k not in ('schema','task_id','seed','arm')},{k:x for k,x in original.items() if k not in ('schema','task_id','seed','arm')})
  self.assertEqual(seen,{(a,s) for a in ('mean','bi') for s in SEEDS})
 def test_shared_initialization_and_zero_anchor_for_every_seed(self):
  run=runner();data=run.fr.synthetic_data();h,s=run.fr._tensors2(run.fr._view(data,50)[0][:8],data['idmap'],50,'f3_f4_recency')
  for seed in SEEDS:
   a=run._model(data,{**run.default_config(),'seed':seed,'arm':'mean'},'cpu').eval();b=run._model(data,{**run.default_config(),'seed':seed,'arm':'bi'},'cpu').eval()
   self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
   for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
   with torch.no_grad():
    self.assertTrue(torch.equal(a.users(h,s),a.anchor_vector(h,s)));self.assertTrue(torch.equal(b.users(h,s),b.anchor_vector(h,s)))
 def test_bi_later_history_sensitivity_and_mean_causality(self):
  run=runner();data=run.fr.synthetic_data();h,s=tm.controlled_row();changed=h.clone();changed[0,-1]=99
  for arm in ('mean','bi'):
   m=tm.activate(run._model(data,{**run.default_config(),'arm':arm},'cpu')).eval()
   with torch.no_grad():a=m.parts(h,s).z[:,0];b=m.parts(changed,s).z[:,0]
   self.assertEqual(torch.equal(a,b),arm=='mean')
 def test_target_pad_and_candidate_isolation(self):
  run=runner();data=run.fr.synthetic_data();batch=run.fr._view(data,50)[0][:12];altered=[leak.flip_target(q,data['catalog']) for q in batch]
  h,s=run.fr._tensors2(batch,data['idmap'],50,'f3_f4_recency');hh,ss=run.fr._tensors2(altered,data['idmap'],50,'f3_f4_recency')
  self.assertTrue(torch.equal(h,hh));self.assertTrue(torch.equal(s,ss))
  for arm in ('mean','bi'):
   m=tm.activate(run._model(data,{**run.default_config(),'arm':arm},'cpu')).eval();items=m.items(torch.arange(1,20));width=max(len(q['history']) for q in batch)
   with torch.no_grad():
    a=m.users(h,s);self.assertTrue(torch.equal(a,m.users(hh,ss)));self.assertTrue(torch.allclose(a,m.users(h[:,:width],s[:,:width]),atol=1e-6,rtol=1e-6));self.assertTrue(torch.equal(items,m.items(torch.arange(1,20))))
 def test_actual_train_verify_and_config_data_source_checkpoint_mutation_rejections(self):
  run=runner();data=run.fr.synthetic_data()
  for arm in ('mean','bi'):
   cfg=small(arm)
   with tempfile.TemporaryDirectory(prefix='biconfirm-') as d:
    out=Path(d)/'run';v=guard_holdout_never_opened(lambda:run.run_session(out,data,cfg));mp=out/'manifest.json'
    self.assertEqual(v['binding']['task'],'P-BI-CONFIRM');self.assertEqual(v['binding']['arm'],arm)
    self.assertEqual(v['binding']['source']['files']['src/vkgr/retrieval_biconfirm_run.py'],run.file_sha256(Path(run.__file__)))
    self.assertEqual(guard_holdout_never_opened(lambda:run.verify_run(mp,v['manifest_sha256'],data,cfg))['status'],'verified')
    with self.assertRaises(FileExistsError):run.run_session(out,data,cfg)
    with self.assertRaises(ValueError):run.verify_run(mp,'0'*64,data,cfg)
    with self.assertRaises(ValueError):run.verify_run(mp,v['manifest_sha256'],data,{**cfg,'seed':271828})
    drift=copy.deepcopy(data);drift['queries'][0]['target']=data['catalog'][-1]
    with self.assertRaises(ValueError):run.verify_run(mp,v['manifest_sha256'],drift,cfg)
    damaged=copy.deepcopy(v);damaged.pop('manifest_sha256');damaged['binding']['source']['source_sha']='0'*64;pin=run.b1._write_json_atomic(out/'tampered.json',damaged)
    with self.assertRaises(ValueError):run.verify_run(out/'tampered.json',pin,data,cfg)
    with (out/'best.pt').open('ab') as f:f.write(b'x')
    with self.assertRaises(ValueError):run.verify_run(mp,v['manifest_sha256'],data,cfg)
 def test_cli_bad_input_hash_creates_no_output(self):
  run=runner()
  with tempfile.TemporaryDirectory(prefix='biconfirm-cli-') as d:
   d=Path(d);cp=d/'config.json';cp.write_text(json.dumps(run.default_config()));ip=d/'inputs.json';ip.write_text('{}')
   with contextlib.redirect_stderr(io.StringIO()):code=run.main(['train','--inputs',str(ip),'--inputs-sha','0'*64,'--config',str(cp),'--config-sha',run.file_sha256(cp),'--output',str(d/'out')])
   self.assertEqual(code,2);self.assertFalse((d/'out').exists())
if __name__=='__main__':unittest.main()

class DataChannelTests(unittest.TestCase):
 def test_private_channel_is_explicit_pinned_and_never_overlays_source(self):
  from unittest import mock
  from vkgr import retrieval_data as rd
  run=runner()
  self.assertTrue(callable(getattr(run,'data_channel',None)),'separate explicit private contract channel required')
  with mock.patch.dict('os.environ',{},clear=True):
   with self.assertRaises(ValueError):run.data_channel()
  with self.assertRaises(ValueError):run.data_channel('relative')
  with tempfile.TemporaryDirectory(prefix='biconfirm-data-') as d:
   root=Path(d);(root/'artifacts').mkdir();(root/'docs').mkdir();doc=root/'docs/p3c_validation.md';doc.write_text('synthetic contract')
   value={'bound_files':[{'path':'docs/p3c_validation.md','sha256':run.file_sha256(doc)}],'full_query_pins':[{'role':'validation-queries.json','sha256':'f'*64}],'catalog_sha256':'e'*64}
   manifest=root/'artifacts/p3d-retrieval-contract-manifest.json';manifest.write_text(json.dumps(value));pin=run.file_sha256(manifest)
   with mock.patch.object(rd,'FROZEN',pin):
    pins,binding=run.data_channel(root)
    self.assertEqual(pins,value);self.assertEqual(binding['data_root'],str(root.resolve()));self.assertEqual(binding['contract_sha256'],pin);self.assertEqual(binding['bound_files'],value['bound_files'])
    doc.write_text('mutated')
    with self.assertRaises(ValueError):run.data_channel(root)
