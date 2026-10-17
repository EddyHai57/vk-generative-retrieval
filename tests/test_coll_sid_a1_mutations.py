"""Mutants must be killed by behavioral guard tests, never source string assertions."""
import unittest
from mutation_util import assert_killed
from vkgr import coll_sid_a1_run as g

class Mutations(unittest.TestCase):
    def test_frozen_field_guard(self):
        assert_killed(self,g,"if any(type(value[k]) is not type(base[k]) or value[k] != base[k] for k in base if k not in variable):",
                     'if False:', ['test_coll_sid_a1_run.Registry.test_every_formal_field_is_frozen'])
    def test_bundle_pin_guard(self):
        assert_killed(self,g,'if not engineering and (sid_sha, input_sha, vectors_sha) != (SID_SHA, INPUT_SHA, VECTORS_SHA):',
                     'if False:', ['test_coll_sid_a1_run.Registry.test_strict_bundle_pins'])
    def test_schema_guard(self):
        assert_killed(self,g,"or value['schema'] != MANIFEST_SCHEMA", "or False", ['test_coll_sid_a1_run.Registry.test_wrong_family_manifest_fails_before_checkpoint'])
    def test_checkpoint_hash_guard(self):
        assert_killed(self,g,"or file_sha256(run_dir / 'best.pt') != best.get('sha256')", "or False",
                     ['test_coll_sid_a1_run.Registry.test_synthetic_full_train_own_verify_and_tamper'])

if __name__=='__main__':unittest.main()
