import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).parents[1]
def module(name):
 p=ROOT/'scripts'/(name+'.py');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class ComboOpsChecks(unittest.TestCase):
 def test_frozen_configs_only_seed_and_old_e_identity_fields_change(self):
  m=module('feat04combo_freeze');rows=[json.loads((ROOT/f'configs/feat04combo/p-feat04combo-cde-s{s}.json').read_text()) for s in (42,2024,2025)]
  m.validate_configs(rows)
  for key,value in (('learning_rate',.0005),('feature','f4_recency_behavior'),('seed',7)):
   bad=copy.deepcopy(rows);bad[1][key]=value
   with self.assertRaises(ValueError):m.validate_configs(bad)
 def test_frozen_files_refuse_tamper(self):
  m=module('feat04combo_freeze')
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'x.py').write_text('first')
   pin={'files':{'x.py':m.sha(root/'x.py')}};m.verify(root,pin)
   (root/'x.py').write_text('changed')
   with self.assertRaises(ValueError):m.verify(root,pin)
 def test_publisher_is_exclusive_and_rejected_blocks_retry(self):
  m=module('feat04combo_ops')
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);q=root/'queue';runs=root/'runs';runs.mkdir()
   for k in ('pending','claimed','rejected'):(q/k).mkdir(parents=True)
   job={'run_id':'fixture','argv':['fixture']};name=m.publish(q,runs,job)
   self.assertTrue((q/'pending'/name).exists())
   with self.assertRaises(FileExistsError):m.publish(q,runs,job)
   rejected={'run_id':'rejected-fixture'};(q/'rejected'/'01-rejected-fixture.json').write_text('{}');(q/'rejected'/'01-rejected-fixture.reason').write_text('fixture failure')
   self.assertTrue(m.where(q,'rejected-fixture')['rejected'])
   with self.assertRaises(FileExistsError):m.publish(q,runs,rejected)
 def test_verify_proof_needs_succeeded_bound_original_stdout(self):
  m=module('feat04combo_ops')
  self.assertTrue(m.accepts_verify('succeeded',True,[{'status':'verified'}]))
  self.assertFalse(m.accepts_verify('succeeded',True,[{'exit_code':0}]))
  self.assertFalse(m.accepts_verify('succeeded',False,[{'status':'verified'}]))
  self.assertFalse(m.accepts_verify('failed',True,[{'status':'verified'}]))
 def test_partial_failure_reports_alert_while_other_verifies_continue(self):
  m=module('feat04combo_ops');states=[{'verified':False,'train_status':'failed','exceptions':[],'rejected':False},{'verified':False,'train_status':None,'exceptions':[],'rejected':False}]
  self.assertEqual(m.summary_status(states,False),'ALERT')
  self.assertEqual(m.summary_status([{'verified':True,'exceptions':[],'rejected':False}],True),'verified')
 def test_driver_starts_for_registered_rejected_or_terminal_without_retry(self):
  m=module('feat04combo_ops')
  for state,terminal in (({'pending':[],'claimed':[],'rejected':['fixture']},False),({'pending':[],'claimed':[],'rejected':[]},True)):
   self.assertTrue(m.can_start_observation(state,terminal))
  self.assertFalse(m.can_start_observation({'pending':[],'claimed':[],'rejected':[]},False))
 def test_expired_before_first_poll_keeps_machine_readable_unaccepted_entries(self):
  m=module('feat04combo_ops');v=m.deadline_fallback([{'job':{'run_id':'fixture'}}])
  self.assertEqual(v[0]['run_id'],'fixture');self.assertTrue(v[0]['unaccepted']);self.assertFalse(v[0]['verified'])
if __name__=='__main__':unittest.main()
