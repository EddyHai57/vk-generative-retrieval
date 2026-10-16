import unittest
from analysis.b256d2_four_column_read import balance, audit_padding
class B1Test(unittest.TestCase):
 def test_padding_creates_preregistered_violation(self):
  r=audit_padding([[a,b,0,c] for a in range(2) for b in range(2) for c in range(2)],2)
  self.assertTrue(r['raw3']['balanced']);self.assertFalse(r['padded4_B1']['balanced'])
  self.assertEqual(r['padded4_B1']['violations'],4)
  self.assertEqual(r['padded4_B1']['max_spread'],0)
  self.assertEqual(r['padded4_all_levels']['violations'],4)
 def test_reject_bad_padding_duplicate_and_ragged(self):
  for rows in [[[0,0,1,0]],[[0,0,0,0],[0,0,0,0]],[[0,0,0]]]:
   with self.assertRaises(ValueError):audit_padding(rows,2)
 def test_spread_over_one(self):
  self.assertFalse(balance([[0],[0],[0],[1]],2,1)['balanced'])

class PrivatePathsTest(unittest.TestCase):
 def test_relative_private_location_rejected(self):
  from analysis import b256d2_four_column_read as r
  with self.assertRaises(ValueError):r.configure_paths({'mapping':'relative'})
