"""G5 P-side top-100 export tests (synthetic only): first-50 gate, best.pt-only weights_only read, no vectors, no touching of the run dir, verify."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_seqm1_run import small  # noqa: E402
from vkgr import retrieval_feat02_run as fr  # noqa: E402
from vkgr import retrieval_seqm1_export100 as ex  # noqa: E402
from vkgr import retrieval_seqm1_run as run  # noqa: E402
from vkgr.data_audit import file_sha256  # noqa: E402


def trained(tmp, arm='mean'):
    data = fr.synthetic_data(); config = small(arm)
    run_dir = Path(tmp) / 'runs' / 'p-synth' / 'out'; run_dir.parent.mkdir(parents=True)
    value = run.run_session(run_dir, data, config)
    return data, config, run_dir / 'manifest.json', value['manifest_sha256']


class ExportTests(unittest.TestCase):
    def test_first50_equal_depth_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            out = Path(tmp) / 'export'
            result = ex.export_run(manifest, sha, data, config, out, run_id='p-synth')
            self.assertEqual(sorted(p.name for p in out.iterdir()), ['export-manifest.json', 'top100.json'])
            top100 = json.loads((out / 'top100.json').read_text())
            self.assertEqual(len(top100), 24); self.assertTrue(all(len(r) == 100 and len(set(r)) == 100 for r in top100))
            value = json.loads(manifest.read_text()); best = value['best']['epoch']
            original = json.loads((manifest.parent / value['evaluations'][best - 1]['predictions_ref']).read_text())
            self.assertTrue(all(new[:50] == old for new, old in zip(top100, original)))
            self.assertEqual(result['gate'], {'first50_equal': True, 'queries': 24, 'mismatched_queries': 0, 'compared_depth': 50})
            self.assertEqual(result['best_epoch'], best); self.assertEqual(result['top100_sha256'], file_sha256(out / 'top100.json'))

    def test_gate_stops_when_first50_differs_and_nothing_is_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); out = Path(tmp) / 'export'
            real = ex.candidates
            def swapped(*a, **k):
                rows = real(*a, **k); rows[3][0], rows[3][1] = rows[3][1], rows[3][0]; return rows
            with mock.patch.object(ex, 'candidates', swapped):
                with self.assertRaisesRegex(ValueError, 'first-50 gate failed: 1/24'):
                    ex.export_run(manifest, sha, data, config, out)
            self.assertFalse(out.exists())

    def test_reads_only_best_checkpoint_with_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); loads = []; real = torch.load
            def spy(path, *a, **k):
                loads.append((Path(path).name, k.get('weights_only'))); return real(path, *a, **k)
            with mock.patch.object(torch, 'load', spy):
                ex.export_run(manifest, sha, data, config, Path(tmp) / 'export')
            self.assertEqual(loads, [('best.pt', True)])

    def test_no_vectors_and_run_directory_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            before = {p.name: (p.stat().st_mtime_ns, file_sha256(p)) for p in manifest.parent.iterdir()}
            out = Path(tmp) / 'export'; result = ex.export_run(manifest, sha, data, config, out)
            self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, file_sha256(p)) for p in manifest.parent.iterdir()})
            self.assertFalse(result['user_vectors_exported'] or result['item_embeddings_exported'] or result['holdout_split_opened'])
            self.assertEqual(result['checkpoints_opened'], ['best.pt'])
            self.assertFalse(list(out.glob('*.npy')) + list(out.glob('*.pt')))

    def test_output_must_be_outside_run_directory_and_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            for bad in (manifest.parent / 'export', manifest.parent, manifest.parent.parent):
                with self.assertRaisesRegex(ValueError, 'outside the existing run directory'):
                    ex.export_run(manifest, sha, data, config, bad)
            out = Path(tmp) / 'export'; ex.export_run(manifest, sha, data, config, out)
            with self.assertRaises(FileExistsError):
                ex.export_run(manifest, sha, data, config, out)

    def test_bad_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            with self.assertRaisesRegex(ValueError, 'external hash mismatch'):
                ex.export_run(manifest, '0' * 64, data, config, Path(tmp) / 'a')
            with self.assertRaisesRegex(ValueError, 'config mismatch'):
                ex.export_run(manifest, sha, data, {**config, 'seed': 2024}, Path(tmp) / 'b')
            with mock.patch.object(ex.sm, 'validate_config', lambda c, synthetic=False: c):
                with self.assertRaisesRegex(ValueError, 'real export requires producer CUDA'):
                    ex.export_run(manifest, sha, {**data, 'synthetic': False}, config, Path(tmp) / 'c')
            best = manifest.parent / 'best.pt'; best.write_bytes(best.read_bytes() + b'x')
            with self.assertRaisesRegex(ValueError, 'best checkpoint hash mismatch'):
                ex.export_run(manifest, sha, data, config, Path(tmp) / 'd')

    def test_verify_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); out = Path(tmp) / 'export'
            result = ex.export_run(manifest, sha, data, config, out, run_id='p-synth'); esha = result['export_manifest_sha256']
            verdict = ex.verify_export(out / 'export-manifest.json', esha, manifest, sha, data, config)
            self.assertEqual(verdict['status'], 'verified'); self.assertEqual(verdict['run_id'], 'p-synth')
            self.assertEqual(verdict['manifest_sha256'], sha); self.assertEqual(verdict['export_manifest_sha256'], esha)
            # wrong export-manifest pin
            with self.assertRaisesRegex(ValueError, 'external hash mismatch'):
                ex.verify_export(out / 'export-manifest.json', '1' * 64, manifest, sha, data, config)
            # extra file in the export directory
            (out / 'extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unexpected files'):
                ex.verify_export(out / 'export-manifest.json', esha, manifest, sha, data, config)
            (out / 'extra.json').unlink()
            # tampered top-100 (hash-consistent forgery): rewrite file and record, recomputation must disagree
            rows = json.loads((out / 'top100.json').read_text()); rows[0] = list(reversed(rows[0]))
            new_top = ex.b1._write_json_atomic(out / 'top100.json', rows)
            record = json.loads((out / 'export-manifest.json').read_text()); record['top100_sha256'] = new_top
            new_esha = ex.b1._write_json_atomic(out / 'export-manifest.json', record)
            with self.assertRaisesRegex(ValueError, 'differs from independent recomputation'):
                ex.verify_export(out / 'export-manifest.json', new_esha, manifest, sha, data, config)

    def test_verify_detects_forged_record_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); out = Path(tmp) / 'export'
            ex.export_run(manifest, sha, data, config, out)
            record = json.loads((out / 'export-manifest.json').read_text()); record['gate'] = {**record['gate'], 'queries': 5}
            esha = ex.b1._write_json_atomic(out / 'export-manifest.json', record)
            with self.assertRaisesRegex(ValueError, 'first-50 gate|field mismatch'):
                ex.verify_export(out / 'export-manifest.json', esha, manifest, sha, data, config)

    def test_candidate_stats_rejects_duplicates_short_rows_and_foreign_items(self):
        catalog = list(range(200))
        good = [list(range(100)), list(range(100, 200))]
        self.assertEqual(ex.candidate_stats(good, catalog, 100)['all_in_catalog'], True)
        for bad, message in (([list(range(99)) + [0], list(range(100, 200))], 'duplicate'), ([list(range(99))], 'depth mismatch'),
                             ([list(range(99)) + [999]], 'outside the catalog')):
            with self.assertRaisesRegex(ValueError, message):
                ex.candidate_stats(bad, catalog, 100)

    def test_preflight_checks_identity_without_forward_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            with mock.patch.object(ex, 'candidates', side_effect=AssertionError('forward pass must not run')):
                got = ex.preflight(manifest, sha, data, config)
            self.assertEqual((got['status'], got['forward_pass_run'], got['written'], got['queries']), ('preflight_ok', False, False, 24))
            self.assertEqual(got['original_depths'], [50])
            with self.assertRaisesRegex(ValueError, 'external hash mismatch'):
                ex.preflight(manifest, '2' * 64, data, config)

    def test_predictions_ref_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            real = ex.read_pinned
            def forged(path, digest):
                value = real(path, digest)
                if Path(path).name == 'manifest.json':
                    value['evaluations'][value['best']['epoch'] - 1]['predictions_ref'] = '../x.json'
                return value
            with mock.patch.object(ex, 'read_pinned', forged):
                with self.assertRaisesRegex(ValueError, 'predictions path escape'):
                    ex.preflight(manifest, sha, data, config)

    def test_cli_rejects_run_id_not_matching_run_directory(self):
        with mock.patch.object(torch.cuda, 'is_available', lambda: True), mock.patch.object(ex, 'read_pinned', lambda p, d: {}), \
                mock.patch.object(ex.sm, 'validate_config', lambda c: c), mock.patch.object(ex.fr, 'validate_inputs', lambda v, f: {'disk_cap_bytes': 1}):
            code = ex.main(['preflight', '--inputs', 'i', '--inputs-sha', 'a', '--config', 'c', '--config-sha', 'b', '--manifest', '/x/runs/p-a/out/manifest.json',
                            '--manifest-sha', 'c', '--run-id', 'p-other'])
        self.assertEqual(code, 2)

    def test_no_holdout_reference_in_source(self):
        text = Path(ex.__file__).read_text(); needle = 'te' + 'st'
        self.assertFalse([l for l in text.splitlines() if needle in l.lower() and 'holdout_split_opened' not in l and 'latest' not in l.lower() and 'attest' not in l.lower() and 'contest' not in l.lower() and 'test (week 26)' not in l.lower() and 'ValueError' not in l])

    def test_cli_requires_cuda_and_pins(self):
        with mock.patch.object(torch.cuda, 'is_available', lambda: False), mock.patch.object(ex, 'read_pinned', lambda p, d: {}), \
                mock.patch.object(ex.sm, 'validate_config', lambda c: c), mock.patch.object(ex.fr, 'validate_inputs', lambda v, f: {'disk_cap_bytes': 1}):
            code = ex.main(['export', '--inputs', 'i', '--inputs-sha', 'a', '--config', 'c', '--config-sha', 'b', '--manifest', 'm',
                            '--manifest-sha', 'c', '--run-id', 'r', '--output', 'o'])
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
