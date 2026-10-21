import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class SubsetTests(unittest.TestCase):
    def test_inventory_bytes_are_frozen(self):
        from analysis import a40_subset_prepare as entry
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)/'inventory.json'
            path.write_bytes(entry.FILES.read_bytes()+b'\n')
            with patch.object(entry,'FILES',path), self.assertRaises(ValueError): entry.inventory()

    def test_inventory_is_exactly_train_and_validation(self):
        from analysis.a40_subset_prepare import inventory, validate_inventory
        value = inventory()
        self.assertEqual(len(value['files']), 26)
        self.assertEqual(value['files'][-1]['path'], 'subsamples/ur0.01_ir0.01/validation/week_25.parquet')
        for change in ('path', 'count', 'size', 'sha'):
            bad = copy.deepcopy(value)
            if change == 'path': bad['files'][0]['path'] = 'interactions/test/week_26.parquet'
            if change == 'count': bad['files'].pop()
            if change == 'size': bad['files'][0]['size'] = 0
            if change == 'sha': bad['files'][0]['sha256'] = 'not-a-hash'
            with self.assertRaises(ValueError): validate_inventory(bad)

    def test_remote_metadata_must_equal_frozen_record(self):
        from analysis.a40_subset_prepare import inventory, check_tree
        fixed = inventory()['files'][:1]
        r = fixed[0]
        tree = [{'path':r['path'], 'size':r['size'], 'lfs':{'oid':r['sha256']}}]
        check_tree(tree, fixed)
        for name in ('size', 'lfs'):
            bad = copy.deepcopy(tree)
            bad[0][name] = 1 if name == 'size' else {'oid':'0'*64}
            with self.assertRaises(ValueError): check_tree(bad, fixed)

    def test_cli_dry_and_illegal_and_execution_guard(self):
        cmd = [sys.executable, '-B', '-m', 'analysis.a40_subset_prepare']
        good = subprocess.run(cmd+['dry'],capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(json.loads(good.stdout)['files'],26)
        for tail in (['unknown'], ['download'], ['preflight']):
            self.assertNotEqual(subprocess.run(cmd+tail,capture_output=True).returncode,0)

    def test_real_preflight_reads_only_tree_and_never_creates_output(self):
        from analysis import a40_subset_prepare as entry
        fixed = entry.inventory()
        urls = []
        def open_tree(req, timeout):
            urls.append(req.full_url)
            section = 'validation' if '/validation?' in req.full_url else 'train'
            values = [{'path':r['path'],'size':r['size'],'lfs':{'oid':r['sha256']}}
                      for r in fixed['files'] if '/'+section+'/' in r['path']]
            return io.BytesIO(json.dumps(values).encode())
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw).resolve(); reference = base/'reference'; reference.mkdir()
            target = base/'subset'
            with patch.object(entry, 'urlopen', side_effect=open_tree), \
                 patch('analysis.new_instance_prepare.runtime', return_value={'runtime':'fixture'}), \
                 patch.object(entry.shutil, 'disk_usage', return_value=entry.shutil._ntuple_diskusage(10**11,10**10,9*10**10)):
                value = entry.preflight(target, reference, base/'uv.lock')
            self.assertFalse(value['payload_read'])
            self.assertFalse(target.exists())
            self.assertEqual(len(urls),2)
            self.assertTrue(all('/tree/' in u and '/test/' not in u and '/resolve/' not in u for u in urls))


if __name__ == '__main__': unittest.main()
