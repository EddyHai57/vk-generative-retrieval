import unittest
from analysis import postbatch_acceptance as p

class PostbatchTests(unittest.TestCase):
 def test_pool_uses_registered_new_seed_set(self):
  rows={str(s):{'only_new':1,'only_control':0,'delta_hits':1,'mcnemar_exact_two_sided':1} for s in (314159,271828,161803)}
  self.assertEqual(p.pool(rows,(314159,271828,161803))['p_pool'],.25)
  with self.assertRaises(ValueError):p.pool(rows,(42,2024,2025))
 def test_no_discordants(self):self.assertEqual(p.exact(0,0),1)
 def test_lex_plateau_and_recall_plateau_distinct(self):
  rows=[(1,.1,.01),(2,.1,.02),(3,.1,.03),(4,.1,.04),(5,.11,.01)]
  out=p.plateaus(rows)
  self.assertEqual(out['three_flat_recall_then_up'],[{'start':1,'end':4,'rise':5}])
  self.assertEqual(out['three_no_new_lex_best_then_up'],[])
 def test_three_nonbest_epochs_then_global_best(self):
  rows=[(1,.1,.01),(2,.09,.01),(3,.08,.01),(4,.09,.01),(5,.11,.01)]
  self.assertEqual(p.plateaus(rows)['three_no_new_lex_best_then_up'],[{'best':1,'start':2,'end':4,'rise':5}])
 def test_terminal_plateau_is_right_censored(self):
  out=p.plateaus([(1,.1,.01),(2,.09,.01),(3,.08,.01),(4,.07,.01)])
  self.assertEqual(out['three_no_new_lex_best_then_up'],[])
  self.assertEqual(out['terminal_no_new_best'],3)
 def test_invalid_curve_rejected(self):
  for rows in ([(2,.1,.1)],[(1,float('nan'),.1)],[(1,.1,.1),(1,.2,.2)]):
   with self.assertRaises(ValueError):p.plateaus(rows)
 def test_pool_rejects_inconsistent_counts(self):
  rows={str(s):{'only_new':1,'only_control':0,'delta_hits':2,'mcnemar_exact_two_sided':1} for s in (42,2024,2025)}
  with self.assertRaises(ValueError):p.pool(rows,(42,2024,2025))
class BindingGuards(unittest.TestCase):
 def test_preflight_rejects_foreign_query_binding(self):
  hist={'binding':{'query_sha':'q','catalog_sha':'cat'},'catalog_sha256':'canonical'}
  m={'binding':{'inputs':{'history_manifest_sha256':'hist','query_sha':'wrong','catalog_sha':'cat'},'catalog_sha256':'canonical'}}
  with self.assertRaises(ValueError):p.check_inputs(m,hist,'hist')
  m['binding']['inputs']['query_sha']='q';p.check_inputs(m,hist,'hist')
 def test_claimed_job_config_and_deployment_bound(self):
  m={'config':{'seed':42},'binding':{'config_sha256':'cfg'}}
  job={'run_id':'p-seq-combo-stack-s42','seed':42,'cwd':'/fixture/deployed'}
  p.check_job('p-seq-combo-stack-s42',m,job,{'seed':42},'/fixture/deployed',p.plain_digest(m['binding']))
  for patch in ({'seed':2024},{'cwd':'/fixture/other'},{'run_id':'other'}):
   with self.assertRaises(ValueError):p.check_job('p-seq-combo-stack-s42',m,{**job,**patch},{'seed':42},'/fixture/deployed',p.plain_digest(m['binding']))
  with self.assertRaises(ValueError):p.check_job('p-seq-combo-stack-s42',m,job,{'seed':2024},'/fixture/deployed',p.plain_digest(m['binding']))
if __name__=='__main__':unittest.main()
