import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]

class EstimateTests(unittest.TestCase):
    def exhaustive(self,histories,threshold,window):
        from collections import Counter
        from analysis.sgns_speedup_profile import keep_probability,pair_count
        count=Counter(x for row in histories.values() for x in row);total=sum(count.values())
        result=0.
        for row in histories.values():
            probabilities=[keep_probability(count[x]/total,threshold) for x in row]
            for mask in itertools.product((False,True),repeat=len(row)):
                weight=1.
                for p,keep in zip(probabilities,mask):weight*=p if keep else 1-p
                kept=[x for x,keep in zip(row,mask) if keep]
                result+=weight*pair_count({1:kept},window)
        return result
    def test_exact_expectation_matches_exhaustive_bernoulli(self):
        from analysis import sgns_speedup_estimates as e
        histories={1:[1,2,1,3,2],2:[3,1,1]}
        for threshold in (.001,.01,.1):
            for window in (1,2,3,5):
                r=e.estimate(histories,[threshold],window=window,max_distance=8,chunk_size=3)
                c=r['C1'][0];oracle=self.exhaustive(histories,threshold,window)
                self.assertAlmostEqual(c['truncated_expected_pairs_per_epoch'],oracle,places=11)
                self.assertEqual(c['omitted_pairs_upper_bound'],0.)
                self.assertLessEqual(c['lower_pairs_per_epoch'],oracle)
                self.assertGreaterEqual(c['upper_pairs_per_epoch'],oracle)
    def test_truncation_upper_bound_contains_true_expectation(self):
        from analysis import sgns_speedup_estimates as e
        histories={1:[1,2,1,3,2,3]}
        oracle=self.exhaustive(histories,.01,2)
        c=e.estimate(histories,[.01],window=2,max_distance=1,chunk_size=2)['C1'][0]
        self.assertLess(c['truncated_expected_pairs_per_epoch'],oracle)
        self.assertGreater(c['omitted_pairs_upper_bound'],0.)
        self.assertLessEqual(c['lower_pairs_per_epoch'],oracle)
        self.assertGreaterEqual(c['upper_pairs_per_epoch'],oracle)
    def test_no_subsampling_reduces_to_original_pairs(self):
        from analysis import sgns_speedup_estimates as e
        from analysis.sgns_speedup_profile import pair_count
        h={1:[1,2,1,3,2,3],2:[1,1,2]}
        r=e.estimate(h,[.9],window=2,max_distance=8,chunk_size=2)
        self.assertAlmostEqual(r['C1'][0]['truncated_expected_pairs_per_epoch'],pair_count(h,2))
        for row in r['C3']:
            self.assertEqual(row['subset_pairs_per_epoch'],pair_count(h,row['window']))
    def test_chunk_invariance_and_no_cross_user_pairs(self):
        from analysis import sgns_speedup_estimates as e
        h={1:[1],2:[2],3:[3]}
        r=e.estimate(h,[.001],window=5,max_distance=4,chunk_size=1)
        self.assertEqual(r['C1'][0]['truncated_expected_pairs_per_epoch'],0.)
        self.assertTrue(all(row['subset_pairs_per_epoch']==0 for row in r['C3']))
        h={1:[1,2,3,1,2,3,1,2],2:[2,3,1,1]}
        a=e.estimate(h,[.001],max_distance=4,chunk_size=2)['C1'][0]
        b=e.estimate(h,[.001],max_distance=4,chunk_size=100)['C1'][0]
        self.assertAlmostEqual(a['truncated_expected_pairs_per_epoch'],b['truncated_expected_pairs_per_epoch'])
        self.assertAlmostEqual(a['omitted_pairs_upper_bound'],b['omitted_pairs_upper_bound'])
    def test_invalid_bounded_configuration(self):
        from analysis import sgns_speedup_estimates as e
        for kwargs in ({'max_distance':0},{'max_distance':513},{'chunk_size':0},{'window':0}):
            with self.assertRaises(ValueError):e.estimate({1:[1,2,3]},[.001],**kwargs)
        with self.assertRaises(ValueError):e.estimate({},[.001])
    def test_cli_synthetic_is_not_full_observation(self):
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
        for args,code in [(['dry'],0),(['synthetic','--max-distance','16'],0),(['illegal'],2)]:
            p=subprocess.run([sys.executable,'-B','-m','analysis.sgns_speedup_estimates',*args],cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(p.returncode,code,p.stderr)
            if args[0]=='synthetic':
                value=json.loads(p.stdout)
                self.assertTrue(value['synthetic']);self.assertFalse(value['science_training'])
                self.assertFalse(value['full_measured'])
                self.assertEqual(len(value['C1']),3)
if __name__=='__main__':unittest.main()
