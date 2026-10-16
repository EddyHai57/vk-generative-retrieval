import json
from pathlib import Path
import tempfile
import unittest
from analysis import sid_aux_reads as a
class AuxTests(unittest.TestCase):
 def test_guard_epoch(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'weights.json';p.write_text('{}')
   with self.assertRaises(ValueError):a.safe_json(p)
   p=Path(td)/'metadata.json';p.write_text('{}')
   with self.assertRaises(ValueError):a.safe_json(p,'0'*64)
  self.assertEqual(a.epoch_summary({})['status'],'UNKNOWN')
  v=a.epoch_summary({'selection':{'best_epoch':2,'epoch':3},'epochs':[{'epoch':2,'evaluation':{'aggregate':{'recall@50':.2,'ndcg@50':.1,'hits@50':3}}}]})
  self.assertEqual(v['best_epoch'],2);self.assertEqual(v['epochs'][0]['metrics']['aggregate.recall@50'],.2)
 def test_alias_binding(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td).resolve();(root/'queue/claimed').mkdir(parents=True)
   target={'path':str(root/'runs/source/out/manifest.json'),'sha256':'a'*64}
   def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
   write(root/'queue/claimed/x.json',{'run_id':'arbitrary-alias','argv':['python','verify','--manifest',target['path'],'--manifest-sha',target['sha256']]})
   rd=root/'runs/arbitrary-alias';write(rd/'terminal.json',{'status':'succeeded','exit_code':0})
   write(rd/'out/verify.json',{'status':'verified','manifest_sha256':'b'*64})
   self.assertEqual(a.find_aliases(root,{'source':target})['source']['status'],'UNKNOWN')
   write(rd/'out/verify.json',{'status':'verified','manifest_sha256':target['sha256']})
   self.assertEqual(a.find_aliases(root,{'source':target})['source']['status'],'verified')
   write(rd/'out/verify.json',{'status':'unverified','manifest_sha256':target['sha256']})
   self.assertEqual(a.find_aliases(root,{'source':target})['source']['status'],'UNKNOWN')
 def test_actual_g6_epoch_schema(self):
  aggregate={'metrics':{'all_eligible':{'count':2520,'recall@50':100/2520,'ndcg@50':.02}}}
  v=a.epoch_summary({'selection':{'best_epoch':1,'epoch':1},'epochs':[{'epoch':1,'metrics':aggregate,'evaluation':{'kind':'query_evaluation','path':'evaluation-001'}}]})
  self.assertEqual(v['epochs'][0]['selection_metrics'],{'R50':100/2520,'N50':.02,'hits':100,'count':2520,'hits_source':'derived_recall_times_count'})
  self.assertEqual(v['status'],'OK')
 def test_verified_requires_success_and_bound_flags(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td).resolve();(root/'queue/claimed').mkdir(parents=True)
   target={'path':str(root/'runs/source/out/manifest.json'),'sha256':'a'*64}
   rd=root/'runs/alias';(rd/'logs').mkdir(parents=True)
   (rd/'logs/stdout.log').write_text('prefix verified\n'+json.dumps({'status':'verified'})+'\n')
   job={'run_id':'alias','argv':['--manifest',target['path'],'--manifest-sha',target['sha256']]}
   jp=root/'queue/claimed/job.json';jp.write_text(json.dumps(job))
   (rd/'terminal.json').write_text(json.dumps({'status':'succeeded','exit_code':2}))
   self.assertEqual(a.find_aliases(root,{'source':target})['source']['status'],'UNKNOWN')
   (rd/'terminal.json').write_text(json.dumps({'status':'succeeded','exit_code':0}))
   self.assertEqual(a.find_aliases(root,{'source':target})['source']['status'],'verified')
   job['argv']+=['--manifest',target['path']];jp.write_text(json.dumps(job))
   self.assertEqual(a.find_aliases(root,{'source':target})['source']['status'],'UNKNOWN')
