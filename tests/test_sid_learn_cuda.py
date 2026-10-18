"""CPU tests only establish CUDA guard and inherited context-inclusive arithmetic."""
import unittest
from unittest.mock import patch
from vkgr import sid_learn_run as run,sid_learn_v1 as contract
class LearnCudaTests(unittest.TestCase):
    def test_no_cuda_rejects_before_model_or_allocation(self):
        self.assertTrue(hasattr(run,'cuda_preview'),'missing CUDA engineering entry')
        with patch.object(run.torch.cuda,'is_available',return_value=False),patch.object(run.fit,'fit_learned',side_effect=AssertionError('model started')) as fit:
            with self.assertRaises(RuntimeError):run.cuda_preview({},contract.expected_config())
        fit.assert_not_called()
    def test_memory_formula_counts_context_and_uses_margin(self):
        self.assertTrue(hasattr(run,'profile_ops'),'missing inherited measurement functions')
        memory={'pid':1,'context_process_mib':500.,'context_reserved_mib':0.,'samples':[{'own_process_mib':1100.,'reserved_mib':600.,'max_reserved_mib':1000.,'max_allocated_mib':900.}]}
        self.assertEqual(run.profile_ops.measured_vram(memory),1500.);self.assertEqual(run.profile_ops.declared_vram(1500.),1792)
if __name__=='__main__':unittest.main()
