"""RED→GREEN: T3 (SID_BHKM_B256D3_S42) whitelist registration.

Test that:
1. T3 is rejected before GREEN (RED phase)
2. T3 is accepted after adding to ENGINEERING_SID_VERSIONS and VERSION_SPEC (GREEN phase)
3. Existing version specs remain unchanged (regression guard)
"""
import unittest
from vkgr import sid_v2d


class TestT3Whitelist(unittest.TestCase):
    """Test T3 (256×3 BHKM) whitelist registration."""

    def test_t3_not_in_formal_versions(self):
        """T3 should not be in formal SID_VERSIONS (筛选候选，非正式生产)."""
        self.assertNotIn('SID_BHKM_B256D3_S42', sid_v2d.SID_VERSIONS)

    def test_existing_versions_unchanged(self):
        """Existing version specs must remain unchanged (regression guard)."""
        # Pin existing formal versions
        self.assertEqual(sid_v2d.SID_VERSIONS, ('SID_BHKM_B16D4_S42', 'SID_RQVAE_B256D4_S42'))
        
        # Pin existing engineering-only versions (before T3)
        self.assertIn('SID_V2D_REPLAY_BHKM_B16D3', sid_v2d.ENGINEERING_SID_VERSIONS)
        
        # Pin existing VERSION_SPEC entries
        self.assertEqual(sid_v2d.VERSION_SPEC['SID_BHKM_B16D4_S42'], ('bhkm', 16, 4))
        self.assertEqual(sid_v2d.VERSION_SPEC['SID_RQVAE_B256D4_S42'], ('rqvae', 256, 4))
        self.assertEqual(sid_v2d.VERSION_SPEC['SID_V2D_REPLAY_BHKM_B16D3'], ('bhkm', 16, 3))

    def test_t3_rejected_in_formal_context(self):
        """RED: T3 should be rejected by _spec_d in formal context."""
        with self.assertRaisesRegex(ValueError, 'unregistered sid_version'):
            sid_v2d._spec_d('SID_BHKM_B256D3_S42', engineering=False)


    def test_t3_in_engineering_versions_after_green(self):
        """GREEN: T3 should be in ENGINEERING_SID_VERSIONS after adding."""
        # This will pass after GREEN
        self.assertIn('SID_BHKM_B256D3_S42', sid_v2d.ENGINEERING_SID_VERSIONS)

    def test_t3_spec_correct_after_green(self):
        """GREEN: T3 spec should be (bhkm, 256, 3) after adding to VERSION_SPEC."""
        # This will pass after GREEN
        self.assertIn('SID_BHKM_B256D3_S42', sid_v2d.VERSION_SPEC)
        self.assertEqual(sid_v2d.VERSION_SPEC['SID_BHKM_B256D3_S42'], ('bhkm', 256, 3))

    def test_t3_accepted_in_engineering_context_after_green(self):
        """GREEN: T3 should be accepted by _spec_d in engineering context after GREEN."""
        # This will pass after GREEN
        kind, branching, depth = sid_v2d._spec_d('SID_BHKM_B256D3_S42', engineering=True)
        self.assertEqual(kind, 'bhkm')
        self.assertEqual(branching, 256)
        self.assertEqual(depth, 3)

    def test_t3_still_rejected_in_formal_context_after_green(self):
        """GREEN: T3 should still be rejected in formal context (engineering-only)."""
        # This will pass after GREEN (T3 is engineering-only, not formal)
        with self.assertRaisesRegex(ValueError, 'unregistered sid_version'):
            sid_v2d._spec_d('SID_BHKM_B256D3_S42', engineering=False)


if __name__ == '__main__':
    unittest.main()
