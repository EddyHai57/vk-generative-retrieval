"""Synthetic-only tests for the P anchor top-100 archive metadata inventory."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('p100inv',Path(__file__).with_name('p_top100_inventory_read_20261018.py'))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SENTINEL='SENTINEL_PER_QUERY_CONTENT'


def build(root,*,verify=True,extra_run=None):
    runs=root/'runs';claimed=root/'queue/claimed';(root/'queue/rejected').mkdir(parents=True)
    claimed.mkdir(parents=True,exist_ok=True)
    for seed in (42,2024,2025):
        rid=f'p-seqm1-mean-s{seed}';out=runs/rid/'out';out.mkdir(parents=True)
        (out/'model').mkdir()
        (out/'model/state.pt').write_bytes(SENTINEL.encode())
        (out/'predictions-epoch-003.json').write_text(SENTINEL)
        manifest={'config':{'seed':seed,'eval_k':50,'lr':0.1},'selection':{'best_epoch':3,'epoch':6},
                  'evaluations':[{'epoch':3,'predictions_ref':'predictions-epoch-003.json','predictions_sha256':'a'*64,
                                  'diagnostics':{'recall@50':0.1,'ndcg@50':0.05}}]}
        (out/'manifest.json').write_text(json.dumps(manifest))
        if verify:
            logs=runs/(rid+'-verify')/'logs';logs.mkdir(parents=True)
            (logs/'stdout.log').write_text('noise\n'+json.dumps({'status':'verified','best_epoch':3,'epochs':6,'replayed_epochs':[3]})+'\n')
            sha=m.sha256_file(out/'manifest.json')
            (claimed/f'01-{rid}-verify.json').write_text(json.dumps({'run_id':rid+'-verify','argv':['python','verify','--manifest',str(out/'manifest.json'),'--manifest-sha',sha]}))
    if extra_run:(runs/extra_run).mkdir(parents=True)


class InventoryTests(unittest.TestCase):
    def test_no_prediction_checkpoint_or_test_content_is_read_or_emitted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build(root)
            opened=[]
            real=Path.read_bytes
            def spy(self,*a,**k):opened.append(self.name);return real(self,*a,**k)
            with mock.patch.object(Path,'read_bytes',spy):
                out=m.inventory(root)
            text=json.dumps(out)
            self.assertNotIn(SENTINEL,text)
            self.assertFalse([n for n in opened if n.startswith('predictions') or n.endswith('.pt')])
            self.assertFalse(out['boundaries']['prediction_content_read'])

    def test_reports_file_names_sizes_and_depth_evidence_without_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build(root);got=m.inventory(root)['runs']['p-seqm1-mean-s42']
            self.assertEqual(got['prediction_files'][0]['name'],'predictions-epoch-003.json')
            self.assertEqual(got['prediction_files'][0]['size'],len(SENTINEL))
            self.assertIn('eval_k',got['manifest']['depth_key_names'])
            self.assertIn('recall@50',got['manifest']['depth_key_names'])
            self.assertEqual(got['manifest']['config_depth_scalars'],{'eval_k':50})
            self.assertEqual(got['manifest']['best_epoch'],3)
            self.assertNotIn('top100_verdict',got)

    def test_verify_status_and_manifest_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build(root);v=m.inventory(root)['runs']['p-seqm1-mean-s2024']['verify'][0]
            self.assertEqual(v['stdout_status']['fields']['status'],'verified')
            self.assertTrue(v['claimed_receipts'][0]['manifest_sha_matches_actual'])

    def test_missing_verify_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build(root,verify=False);v=m.inventory(root)['runs']['p-seqm1-mean-s42']['verify']
            self.assertTrue(all(not x['exists'] for x in v))

    def test_discovery_finds_other_top100_named_dirs_but_not_own_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build(root,extra_run='p-seqm1-mean-s42-top100-export');got=m.inventory(root)['discovery']
            self.assertIn('p-seqm1-mean-s42-top100-export',got['runs_dir_matches'])
            self.assertNotIn('p-seqm1-mean-s42',got['runs_dir_matches'])

    def test_oversize_manifest_not_read(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build(root)
            big=root/'runs/p-seqm1-mean-s42/out/manifest.json';big.write_bytes(b' '*(m.MAX_JSON+1))
            got=m.inventory(root)['runs']['p-seqm1-mean-s42']['manifest']
            self.assertEqual(got['read'],False)

    def test_symlinked_run_dir_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build(root)
            target=root/'elsewhere';target.mkdir()
            link=root/'runs/p-seqm1-mean-s42-verify'
            if link.exists():
                import shutil;shutil.rmtree(link)
            link.symlink_to(target)
            v=m.inventory(root)['runs']['p-seqm1-mean-s42']['verify'][0]
            self.assertEqual(v['listing'].get('refused'),'symlink')


if __name__=='__main__':unittest.main()
