"""Tiny arithmetic fixtures for the frozen two-component RQ cost definition."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

def module():
    path = ROOT / 'scripts/sid_scale_rq_components.py'
    assert path.is_file(), 'missing two-component cost implementation'
    spec = importlib.util.spec_from_file_location('rq_components', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

class RQComponentTests(unittest.TestCase):
    def test_frozen_formula_uses_floor_and_explicit_k_d_l(self):
        m = module()
        r = m.compose(n=10, steps=29, usage_interval=10, k=2, d=3, levels=4,
                      core_seconds_per_step=.5, assign_seconds_per_unit=.01)
        self.assertEqual(r['n_usage'], 2)
        self.assertAlmostEqual(r['usage_pass_seconds'], 2.4)
        self.assertAlmostEqual(r['total_fit_seconds'], 19.3)
        self.assertEqual(r['kind'], 'composed_from_components')
        s = m.compose(n=10, steps=29, usage_interval=10, k=4, d=3, levels=4,
                      core_seconds_per_step=.5, assign_seconds_per_unit=.01)
        self.assertAlmostEqual(s['usage_total_seconds'], 9.6)

    def test_core_calibration_rejects_usage_or_interval_boundary(self):
        m = module()
        good = {'status':'complete', 'step':3, 'elapsed':.6, 'usage_log':[]}
        self.assertAlmostEqual(m.core_constant(good, usage_interval=4), .2)
        for bad in ({**good,'step':4}, {**good,'usage_log':[(2,[2,2,2])]},
                    {**good,'elapsed':float('nan')}, {**good,'status':'fail_wall'}):
            with self.assertRaises(ValueError):
                m.core_constant(bad, usage_interval=4)

    def test_usage_regression_reports_residuals_and_noisy_fit(self):
        m = module()
        r = m.usage_constant([{'n':10,'seconds':2.}, {'n':20,'seconds':4.},
                              {'n':30,'seconds':6.}], k=2,d=5,levels=2)
        self.assertAlmostEqual(r['assign_seconds_per_unit'], .01)
        self.assertEqual(r['r_squared'], 1.)
        self.assertEqual([x['residual_seconds'] for x in r['points']], [0.,0.,0.])
        noisy = m.usage_constant([{'n':10,'seconds':2.}, {'n':20,'seconds':5.},
                                  {'n':30,'seconds':6.}], k=2,d=5,levels=2)
        self.assertLess(noisy['r_squared'], 1.)
        for points in ([{'n':10,'seconds':2.}],
                       [{'n':10,'seconds':2.},{'n':10,'seconds':4.}]):
            with self.assertRaises(ValueError):m.usage_constant(points,k=2,d=5,levels=2)

    def test_selfcheck_enforces_symmetric_twofold_boundary(self):
        m = module()
        for predicted in (50.,100.,200.):
            self.assertEqual(m.historical_check(predicted, 100.)['status'], 'passed')
        for predicted in (49.,201.):
            r = m.historical_check(predicted,100.)
            self.assertEqual(r['status'],'failed')
            self.assertEqual(r['extrapolation'],'UNKNOWN')
        self.assertAlmostEqual(m.historical_check(120.,100.)['relative_error'],.2)

    def test_source_interval_and_function_identity_come_from_fixed_bytes(self):
        m = module();r = m.l2_source(ROOT)
        self.assertEqual(r['usage_interval'],10000)
        self.assertEqual(r['interval_line'],20)
        self.assertEqual(r['usage_function'],'vkgr.sid_rqvae._usage')
        self.assertEqual(r['usage_function_line'],128)
        self.assertEqual(r['periodic_call_line'],262)
        self.assertEqual(len(r['files']),2)
        self.assertEqual(r['diagnostics'],False)

    def test_actual_chunked_memory_is_not_full_n_by_k(self):
        m = module();r = m.usage_memory(n=100000,k=48,d=32,levels=3,chunk=8192)
        self.assertEqual(r['input_float32_bytes'],100000*64*4)
        self.assertEqual(r['distance_float32_bytes'],8192*48*4)
        self.assertEqual(r['code_indices_int64_bytes'],8192*3*8)
        self.assertEqual(r['seen_bool_bytes'],48*3)
        self.assertEqual(r['scope'],'principal_arrays_not_peak_rss')


class UsageMeasurementTests(unittest.TestCase):
    def test_wall_ratio_uses_separately_measured_constants(self):
        m = module()
        # Same item-forward count: 10 usage passes * 100 items = 1000 core items.
        a = m.compose(n=100, steps=1000, usage_interval=100,k=2,d=3,levels=4,
                      core_seconds_per_step=1., assign_seconds_per_unit=.01)
        b = m.compose(n=100, steps=1000, usage_interval=100,k=2,d=3,levels=4,
                      core_seconds_per_step=1., assign_seconds_per_unit=.1)
        self.assertIn('usage_to_core_wall_ratio', a)
        self.assertAlmostEqual(a['usage_to_core_wall_ratio'],.24)
        self.assertAlmostEqual(b['usage_to_core_wall_ratio'],2.4)
        self.assertAlmostEqual(a['usage_wall_fraction'],240/1240)
        self.assertIn('not wall time',a['item_forward_count_limitation'])

    def test_direct_original_usage_is_called_once_without_fit(self):
        import sys
        from unittest.mock import patch
        import numpy as np
        root = str(ROOT/'src')
        if root not in sys.path:sys.path.insert(0,root)
        from vkgr import sid_rqvae as rq, sid_rqvae_r as repair
        m = module();self.assertTrue(callable(getattr(m,'measure_usage_once',None)));x = np.ones((19,64),dtype=np.float32)
        with patch.object(rq,'_usage',wraps=rq._usage) as usage, patch.object(repair,'fit_r',side_effect=AssertionError('training prohibited')):
            result = m.measure_usage_once(x,k=3,d=32,levels=3,engineering=True,root=ROOT)
        self.assertEqual(usage.call_count,1)
        self.assertEqual(result['n'],19)
        self.assertEqual(result['diagnostics'],False)
        self.assertEqual(result['usage_calls'],1)
        self.assertGreater(result['seconds'],0)
        self.assertEqual(result['function'],'vkgr.sid_rqvae._usage')
        self.assertEqual(len(result['source']['files']),2)
        self.assertEqual(len(result['input_sha256']),64)
        self.assertNotIn('features',result)

    def test_measurement_refuses_unapproved_local_scale_before_model_allocation(self):
        import sys
        from unittest.mock import patch
        import numpy as np
        root = str(ROOT/'src')
        if root not in sys.path:sys.path.insert(0,root)
        from vkgr import sid_rqvae as rq
        m = module();self.assertTrue(callable(getattr(m,'measure_usage_once',None)))
        with patch.object(rq,'RQVAE',side_effect=AssertionError('model allocated')):
            for kwargs in ({'engineering':True}, {'engineering':False}):
                with self.assertRaises(ValueError):
                    m.measure_usage_once(np.ones((1025,64),dtype=np.float32),k=3,d=32,levels=3,root=ROOT,**kwargs)
            with self.assertRaises(ValueError):
                m.measure_usage_once(np.ones((19,64),dtype=np.float32),k=3,d=16,levels=3,root=ROOT,engineering=True)

if __name__ == '__main__':unittest.main()
