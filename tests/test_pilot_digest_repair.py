"""Legacy digest repair preserves actual predecessor bytes and science binding."""
import copy
import unittest
from vkgr import retrieval_sasrec_remote as R

class LegacyDigestRepairTests(unittest.TestCase):
    def legacy(self):
        artifacts=[{'role':'fixture','sha256':'a'*64,'size':3,'path':'/fixture'}]
        return {'artifact_hashes':artifacts,'complete_artifact_set_sha256':R.sha(R.canonical(artifacts)),'terminal_receipt_sha256':'b'*64}
    def test_normalizes_only_digest_and_preserves_original(self):
        original=self.legacy();before=copy.deepcopy(original)
        fixed=R.normalize_legacy_artifact_digest(original)
        self.assertEqual(original,before)
        self.assertEqual(fixed['complete_artifact_set_sha256'],R.sha(R.plan.canonical_bytes(original['artifact_hashes'])))
        fixed['complete_artifact_set_sha256']=original['complete_artifact_set_sha256']
        self.assertEqual(fixed,original)
    def test_corrupt_legacy_digest_rejected(self):
        value=self.legacy();value['complete_artifact_set_sha256']='0'*64
        with self.assertRaises(ValueError):R.normalize_legacy_artifact_digest(value)
    def test_already_correct_digest_cannot_claim_repair(self):
        value=self.legacy();value['complete_artifact_set_sha256']=R.sha(R.plan.canonical_bytes(value['artifact_hashes']))
        with self.assertRaises(ValueError):R.normalize_legacy_artifact_digest(value)
    def test_original_source_binding_only_for_explicit_pilot_verify(self):
        for stage in ('pilot','formal-train','formal-verify'):
            with self.assertRaises(ValueError):R.digest_repair_predecessor({'stage':stage},{'digest_repair':{}})
    def test_unapproved_original_hash_rejected_before_path(self):
        with self.assertRaises(ValueError):R.digest_repair_predecessor({'stage':'pilot-verify'},{'digest_repair':{'original_seal':{'path':'/missing','sha256':'0'*64}}})

class RepairBindingTests(unittest.TestCase):
    def fixture(self):
        original=LegacyDigestRepairTests().legacy()
        original['artifact_hashes'][0]['role']='pilot-artifact-index'
        original['predecessor_operation_manifest_sha256']='c'*64
        original['complete_artifact_set_sha256']=R.sha(R.canonical(original['artifact_hashes']))
        prior={'stage':'pilot','queue':{'fixture':7},'source_pins':{'src/vkgr/retrieval_sasrec_remote.py':R.PILOT07_CONSUMER_SHA256,'src/vkgr/retrieval_sasrec.py':'d'*64}}
        value={'stage':'pilot-verify','queue':copy.deepcopy(prior['queue']),'source_pins':{**prior['source_pins'],'src/vkgr/retrieval_sasrec_remote.py':'e'*64},'predecessor':R.normalize_legacy_artifact_digest(original)}
        auth={'digest_repair':{'schema_id':'p4h-pilot07-newline-digest-repair-v1','original_seal':{'path':'/workspace/vkgr-p4h-sol03-pilot-v1/attempt07/pilot-output/sealed.json','sha256':R.PILOT07_LEGACY_SEAL_SHA256}}}
        return original,prior,value,auth
    def call(self,original,prior,value,auth,index=None):
        from unittest.mock import patch
        if index is None:index={'artifacts':[],'complete_artifact_set_sha256':R.sha(R.canonical([]))}
        with patch.object(R,'_read_json',side_effect=[original,prior,index]),patch.object(R,'verify_source_tree') as source:
            result=R.digest_repair_predecessor(value,auth)
            source.assert_called_once_with(prior)
            return result
    def test_original_science_and_source_are_verified(self):
        original,prior,value,auth=self.fixture()
        root,actual=self.call(original,prior,value,auth)
        self.assertEqual(str(root),'/workspace/vkgr-p4h-sol03-pilot-v1/attempt07/pilot-output')
        self.assertEqual(actual,prior)
    def test_changed_queue_or_scientific_source_rejected(self):
        from unittest.mock import patch
        for field in ('queue','source','seal'):
            original,prior,value,auth=self.fixture()
            if field=='queue':value['queue']['fixture']=8
            elif field=='source':value['source_pins']['src/vkgr/retrieval_sasrec.py']='f'*64
            else:value['predecessor']['terminal_receipt_sha256']='f'*64
            with patch.object(R,'_read_json',side_effect=[original,prior]),self.assertRaises(ValueError):R.digest_repair_predecessor(value,auth)
    def test_incomplete_legacy_index_rejected(self):
        original,prior,value,auth=self.fixture()
        with self.assertRaises(ValueError):self.call(original,prior,value,auth,index={'artifacts':[],'complete_artifact_set_sha256':'0'*64})

if __name__=='__main__':unittest.main()
