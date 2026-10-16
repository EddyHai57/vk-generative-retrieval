import unittest,tempfile,json
from pathlib import Path
from analysis.seed_usage_read import scan
class SeedScanTests(unittest.TestCase):
 def test_job_and_manifest_collisions_not_hash_mentions(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'queue/claimed').mkdir(parents=True);(p/'runs/a/out').mkdir(parents=True)
   (p/'queue/claimed/a.json').write_text(json.dumps({'run_id':'a','seed':314159}))
   (p/'runs/a/out/manifest.json').write_text(json.dumps({'config':{'seed':42},'hash':'314159'}))
   d=scan(p);self.assertEqual(len(d['matches']),1);self.assertEqual(d['matches'][0]['seed'],314159)
 def test_unreadable_or_symlink_is_not_clearance(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'queue/claimed').mkdir(parents=True);(p/'runs').mkdir()
   (p/'queue/claimed/a.json').write_text('bad')
   self.assertEqual(scan(p)['status'],'INCOMPLETE')
