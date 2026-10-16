import importlib.util,unittest,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'scripts/sid_coll_capacity_repair.py'
class Capacity(unittest.TestCase):
 def module(self):
  s=importlib.util.spec_from_file_location('capacity',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_real_host_capacity_rounds_down_with_unchanged_margin(self):
  m=self.module();self.assertEqual(m.capacity(49140,49139,0,0),39168)
  self.assertLess((49139-39312)/49140,.2);self.assertGreaterEqual((49139-39168)/49140,.2)
 def test_only_vram_bytes_change_and_wrong_job_rejected(self):
  m=self.module();old={'run_id':'sid-coll-c1-cuda-v1','vram_mib':39312,'seed':42,'argv':['x']};raw=(json.dumps(old,sort_keys=True,indent=2)+'\n').encode();new=m.patch_bytes(raw,39168)
  value=json.loads(new);self.assertEqual({**value,'vram_mib':39312},old);self.assertEqual(raw.replace(b'39312',b'39168'),new)
  with self.assertRaises(ValueError):m.patch_bytes(json.dumps({**old,'run_id':'not-own'}).encode(),39168)
 def test_fixed_original_plan_only_two_capacity_fields_change(self):
  m=self.module();old={'deadline_epoch':12,'branches':{b:{'stages':{'cuda':{'job':{'run_id':rid,'vram_mib':39312}},'train42':{'job':{'vram_mib':'@cuda.raw_proof.declared_vram_mib'}}}} for b,rid in [('C1p','sid-coll-c1-cuda-v1'),('C2','sid-coll-c2-cuda-v1')]}}
  p=m.patched_plan(old,39168);self.assertEqual(p['deadline_epoch'],12)
  for b in p['branches']:p['branches'][b]['stages']['cuda']['job']['vram_mib']=39312
  self.assertEqual(p,old)
if __name__=='__main__':unittest.main()
