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


MODULE_PATH = 'src/vkgr/retrieval_seqm1_export100.py'


def pinned():
    return {MODULE_PATH: file_sha256(Path(ex.__file__))}


def train_identity(real=ex.b1.source_identity):
    """Training-time source identity as it was for the accepted runs: the export module did not exist yet."""
    def fake():
        value = real(); files = {k: v for k, v in value['files'].items() if k != MODULE_PATH}
        return {**value, 'files': files, 'git_commit': 'train-commit', 'git_dirty': False}
    return fake


def trained(tmp, arm='mean', identity=None):
    data = fr.synthetic_data(); config = small(arm)
    run_dir = Path(tmp) / 'runs' / 'p-synth' / 'out'; run_dir.parent.mkdir(parents=True)
    with mock.patch.object(ex.b1, 'source_identity', identity or train_identity()):
        value = run.run_session(run_dir, data, config)
    return data, config, run_dir / 'manifest.json', value['manifest_sha256']


def xr(*a, **k):
    k.setdefault('delta', pinned()); return ex.export_run(*a, **k)


def xv(*a, **k):
    k.setdefault('delta', pinned()); return ex.verify_export(*a, **k)


def xp(*a, **k):
    k.setdefault('delta', pinned()); return ex.preflight(*a, **k)


class ExportTests(unittest.TestCase):
    def test_first50_equal_depth_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            out = Path(tmp) / 'export'
            result = xr(manifest, sha, data, config, out, run_id='p-synth')
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
                    xr(manifest, sha, data, config, out)
            self.assertFalse(out.exists())

    def test_reads_only_best_checkpoint_with_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); loads = []; real = torch.load
            def spy(path, *a, **k):
                loads.append((Path(path).name, k.get('weights_only'))); return real(path, *a, **k)
            with mock.patch.object(torch, 'load', spy):
                xr(manifest, sha, data, config, Path(tmp) / 'export')
            self.assertEqual(loads, [('best.pt', True)])

    def test_no_vectors_and_run_directory_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            before = {p.name: (p.stat().st_mtime_ns, file_sha256(p)) for p in manifest.parent.iterdir()}
            out = Path(tmp) / 'export'; result = xr(manifest, sha, data, config, out)
            self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, file_sha256(p)) for p in manifest.parent.iterdir()})
            self.assertFalse(result['user_vectors_exported'] or result['item_embeddings_exported'] or result['holdout_split_opened'])
            self.assertEqual(result['checkpoints_opened'], ['best.pt'])
            self.assertFalse(list(out.glob('*.npy')) + list(out.glob('*.pt')))

    def test_output_must_be_outside_run_directory_and_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            for bad in (manifest.parent / 'export', manifest.parent, manifest.parent.parent):
                with self.assertRaisesRegex(ValueError, 'outside the existing run directory'):
                    xr(manifest, sha, data, config, bad)
            out = Path(tmp) / 'export'; xr(manifest, sha, data, config, out)
            with self.assertRaises(FileExistsError):
                xr(manifest, sha, data, config, out)

    def test_bad_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            with self.assertRaisesRegex(ValueError, 'external hash mismatch'):
                xr(manifest, '0' * 64, data, config, Path(tmp) / 'a')
            with self.assertRaisesRegex(ValueError, 'config mismatch'):
                xr(manifest, sha, data, {**config, 'seed': 2024}, Path(tmp) / 'b')
            with mock.patch.object(ex.sm, 'validate_config', lambda c, synthetic=False: c):
                with self.assertRaisesRegex(ValueError, 'real export requires producer CUDA'):
                    xr(manifest, sha, {**data, 'synthetic': False}, config, Path(tmp) / 'c')
            best = manifest.parent / 'best.pt'; best.write_bytes(best.read_bytes() + b'x')
            with self.assertRaisesRegex(ValueError, 'best checkpoint hash mismatch'):
                xr(manifest, sha, data, config, Path(tmp) / 'd')

    def test_verify_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); out = Path(tmp) / 'export'
            result = xr(manifest, sha, data, config, out, run_id='p-synth'); esha = result['export_manifest_sha256']
            verdict = xv(out / 'export-manifest.json', esha, manifest, sha, data, config)
            self.assertEqual(verdict['status'], 'verified'); self.assertEqual(verdict['run_id'], 'p-synth')
            self.assertEqual(verdict['manifest_sha256'], sha); self.assertEqual(verdict['export_manifest_sha256'], esha)
            # wrong export-manifest pin
            with self.assertRaisesRegex(ValueError, 'external hash mismatch'):
                xv(out / 'export-manifest.json', '1' * 64, manifest, sha, data, config)
            # extra file in the export directory
            (out / 'extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unexpected files'):
                xv(out / 'export-manifest.json', esha, manifest, sha, data, config)
            (out / 'extra.json').unlink()
            # tampered top-100 (hash-consistent forgery): rewrite file and record, recomputation must disagree
            rows = json.loads((out / 'top100.json').read_text()); rows[0] = list(reversed(rows[0]))
            new_top = ex.b1._write_json_atomic(out / 'top100.json', rows)
            record = json.loads((out / 'export-manifest.json').read_text()); record['top100_sha256'] = new_top
            new_esha = ex.b1._write_json_atomic(out / 'export-manifest.json', record)
            with self.assertRaisesRegex(ValueError, 'differs from independent recomputation'):
                xv(out / 'export-manifest.json', new_esha, manifest, sha, data, config)

    def test_verify_detects_forged_record_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); out = Path(tmp) / 'export'
            xr(manifest, sha, data, config, out)
            record = json.loads((out / 'export-manifest.json').read_text()); record['gate'] = {**record['gate'], 'queries': 5}
            esha = ex.b1._write_json_atomic(out / 'export-manifest.json', record)
            with self.assertRaisesRegex(ValueError, 'first-50 gate|field mismatch'):
                xv(out / 'export-manifest.json', esha, manifest, sha, data, config)

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
                got = xp(manifest, sha, data, config)
            self.assertEqual((got['status'], got['forward_pass_run'], got['written'], got['queries']), ('preflight_ok', False, False, 24))
            self.assertEqual(got['original_depths'], [50])
            with self.assertRaisesRegex(ValueError, 'external hash mismatch'):
                xp(manifest, '2' * 64, data, config)

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
                    xp(manifest, sha, data, config)

    def test_cli_rejects_run_id_not_matching_run_directory(self):
        with mock.patch.object(torch.cuda, 'is_available', lambda: True), mock.patch.object(ex, 'read_pinned', lambda p, d: {}), \
                mock.patch.object(ex.sm, 'validate_config', lambda c: c), mock.patch.object(ex.fr, 'validate_inputs', lambda v, f: {'disk_cap_bytes': 1}):
            with mock.patch('sys.stderr', new_callable=__import__('io').StringIO) as err:
                code = ex.main(['preflight', '--inputs', 'i', '--inputs-sha', 'a', '--config', 'c', '--config-sha', 'b', '--manifest', '/x/runs/p-a/out/manifest.json',
                                '--manifest-sha', 'c', '--run-id', 'p-other', '--pin-new-path', MODULE_PATH + '=' + 'a' * 64])
        self.assertEqual(code, 2); self.assertIn('run id does not match the run directory', err.getvalue())

    def test_peak_mib_none_without_cuda_and_never_in_manifest(self):
        with mock.patch.object(torch.cuda, 'is_available', lambda: False):
            self.assertIsNone(ex.peak_mib())
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); out = Path(tmp) / 'export'
            xr(manifest, sha, data, config, out)
            self.assertNotIn('cuda_peak', (out / 'export-manifest.json').read_text())

    def test_pinned_delta_passes_and_manifest_keeps_both_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp); out = Path(tmp) / 'export'
            result = xr(manifest, sha, data, config, out)
            self.assertEqual(result['train_source_commit'], 'train-commit'); self.assertNotEqual(result['export_source_commit'], 'train-commit')
            self.assertEqual(result['pinned_delta'], pinned())
            self.assertEqual(xp(manifest, sha, data, config)['source_identity'], 'pinned_delta_ok')

    def _export_with(self, mutate=None, delta=None, train_identity_fn=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp, identity=train_identity_fn)
            real = ex.b1.source_identity
            def current():
                value = real()
                return mutate(value) if mutate else value
            with mock.patch.object(ex.b1, 'source_identity', current):
                return xp(manifest, sha, data, config, **({'delta': delta} if delta is not None else {}), **kwargs)

    def test_changed_preexisting_file_fails(self):
        def mutate(value):
            files = dict(value['files']); key = next(k for k in files if k.endswith('retrieval_seqm1_run.py')); files[key] = '0' * 64
            return {**value, 'files': files}
        with self.assertRaisesRegex(ValueError, 'ledger differs'):
            self._export_with(mutate)

    def test_unpinned_new_path_fails(self):
        def mutate(value):
            return {**value, 'files': {**value['files'], 'src/vkgr/extra_unpinned.py': '1' * 64}}
        with self.assertRaisesRegex(ValueError, 'ledger differs'):
            self._export_with(mutate)

    def test_wrong_pinned_sha_and_missing_pinned_path_fail(self):
        with self.assertRaisesRegex(ValueError, 'ledger differs'):
            self._export_with(delta={MODULE_PATH: 'f' * 64})
        with self.assertRaisesRegex(ValueError, 'ledger differs'):
            self._export_with(delta={'src/vkgr/not_there.py': pinned()[MODULE_PATH]})

    def test_pinned_path_already_in_training_ledger_fails(self):
        with self.assertRaisesRegex(ValueError, 'already present'):
            self._export_with(train_identity_fn=ex.b1.source_identity)

    def test_python_dependency_lock_must_match_strictly(self):
        for field, value in (('python', '0.0.0'), ('dependencies', {'torch': '0'}), ('lock_sha', '2' * 64)):
            with self.assertRaisesRegex(ValueError, f'source identity mismatch: {field}'):
                self._export_with(lambda v, f=field, x=value: {**v, f: x})

    def test_other_binding_classes_stay_strict_and_commit_is_not_a_pass_condition(self):
        with self.assertRaisesRegex(ValueError, 'binding mismatch: inputs'):
            self._export_with(inputs_binding={'other': 1})
        # equal commit but an extra unpinned file still fails: the commit plays no role in passing
        def mutate(value):
            return {**value, 'git_commit': 'train-commit', 'files': {**value['files'], 'src/vkgr/x.py': '3' * 64}}
        with self.assertRaisesRegex(ValueError, 'ledger differs'):
            self._export_with(mutate)

    def test_every_non_source_binding_key_is_strict_and_commit_only_difference_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            rows, queries = fr._view(data, config['history_len']); model = run._model(data, config, 'cpu')
            real = ex.b1.source_identity
            stored = json.loads(manifest.read_text())['binding']
            current = run.binding_for4(data, config, rows, queries, model, 'cpu', None)
            delta = pinned()
            ex.check_binding(stored, current, delta)                                    # baseline passes
            self.assertEqual(stored['source']['git_commit'], 'train-commit')
            keys = [k for k in stored if k != 'source']
            self.assertGreaterEqual(len(keys), 10)
            for key in keys:
                forged = {**current, key: 'forged' if current[key] != 'forged' else 'forged2'}
                with self.assertRaisesRegex(ValueError, f'binding mismatch: {key}$'):
                    ex.check_binding(stored, forged, delta)
            commit_only = {**current, 'source': {**current['source'], 'git_commit': 'another', 'git_dirty': True, 'source_sha': '0' * 64}}
            self.assertEqual(ex.check_binding(stored, commit_only, delta)['export_source_commit'], 'another')

    def test_extra_or_missing_binding_field_fails(self):
        source = {'python': '3', 'dependencies': {}, 'lock_sha': 'l', 'files': {'a.py': '1' * 64}, 'git_commit': 'c', 'git_dirty': False, 'source_sha': 's'}
        delta = {'n.py': '2' * 64}; later = {**source, 'files': {'a.py': '1' * 64, 'n.py': '2' * 64}, 'git_commit': 'd'}
        base = {'a': 1, 'source': source}
        self.assertEqual(ex.check_binding(base, {'a': 1, 'source': later}, delta), {'train_source_commit': 'c', 'export_source_commit': 'd'})
        with self.assertRaisesRegex(ValueError, 'binding fields differ'):
            ex.check_binding(base, {'a': 1, 'b': 2, 'source': later}, delta)
        with self.assertRaisesRegex(ValueError, 'binding fields differ'):
            ex.check_binding({'a': 1, 'b': 2, 'source': source}, {'a': 1, 'source': later}, delta)
        with self.assertRaisesRegex(ValueError, 'source identity fields differ'):
            ex.check_binding(base, {'a': 1, 'source': {**later, 'extra': 1}}, delta)

    def test_missing_or_malformed_delta_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            for bad in (None, {}, {MODULE_PATH: 'short'}):
                with self.assertRaisesRegex(ValueError, 'pinned delta required'):
                    ex.preflight(manifest, sha, data, config, delta=bad)
        self.assertEqual(ex.parse_delta([MODULE_PATH + '=' + 'a' * 64]), {MODULE_PATH: 'a' * 64})
        for bad in ([], ['x'], ['/abs=' + 'a' * 64], ['../x=' + 'a' * 64], ['p=' + 'A' * 64], ['p=' + 'a' * 64, 'p=' + 'b' * 64]):
            with self.assertRaises(ValueError):
                ex.parse_delta(bad)

    def test_pinned_module_is_not_imported_by_model_or_data_construction(self):
        """The pinned new module must not influence numerics: importing every model/data/run module never imports it,
        and no other source file references it."""
        import subprocess
        root = Path(ex.__file__).resolve().parents[2]
        code = ("import sys; sys.path.insert(0, 'src');"
                "import vkgr.retrieval_seqm1_run, vkgr.retrieval_seqm1, vkgr.retrieval_feat02_run, vkgr.retrieval_feat01_run,"
                "vkgr.retrieval_php01_run, vkgr.retrieval_b1_run, vkgr.retrieval_data, vkgr.retrieval_experiment, vkgr.retrieval_training,"
                "vkgr.baselines, vkgr.retrieval_models, vkgr.retrieval_negatives, vkgr.history_v2, vkgr.history_side;"
                "assert 'vkgr.retrieval_seqm1_export100' not in sys.modules, 'imported'; print('clean')")
        out = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, 'clean'), out.stderr[-400:])
        referrers = [p.name for p in (root / 'src/vkgr').glob('*.py') if p.name != 'retrieval_seqm1_export100.py' and 'export100' in p.read_text()]
        self.assertEqual(referrers, [])

    def test_preflight_peak_is_labelled_not_valid_for_declaration_and_peak_cannot_break_publish(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            data, config, manifest, sha = trained(tmp)
            self.assertEqual(xp(manifest, sha, data, config)['cuda_peak_scope'], 'no_forward_pass_not_valid_for_declaration')
        with mock.patch.object(torch.cuda, 'is_available', lambda: True), \
                mock.patch.object(torch.cuda, 'max_memory_allocated', side_effect=RuntimeError('boom')):
            self.assertIsNone(ex.peak_mib())

    def test_no_holdout_reference_in_source(self):
        text = Path(ex.__file__).read_text(); needle = 'te' + 'st'
        self.assertFalse([l for l in text.splitlines() if needle in l.lower() and 'holdout_split_opened' not in l and 'latest' not in l.lower() and 'attest' not in l.lower() and 'contest' not in l.lower() and 'test (week 26)' not in l.lower() and 'ValueError' not in l])

    def test_cli_requires_cuda_and_pins(self):
        with mock.patch.object(torch.cuda, 'is_available', lambda: False), mock.patch.object(ex, 'read_pinned', lambda p, d: {}), \
                mock.patch.object(ex.sm, 'validate_config', lambda c: c), mock.patch.object(ex.fr, 'validate_inputs', lambda v, f: {'disk_cap_bytes': 1}):
            with mock.patch('sys.stderr', new_callable=__import__('io').StringIO) as err:
                code = ex.main(['export', '--inputs', 'i', '--inputs-sha', 'a', '--config', 'c', '--config-sha', 'b', '--manifest', 'm',
                                '--manifest-sha', 'c', '--run-id', 'r', '--output', 'o', '--pin-new-path', MODULE_PATH + '=' + 'a' * 64])
        self.assertEqual(code, 2); self.assertIn('actual CUDA environment required', err.getvalue())


if __name__ == '__main__':
    unittest.main()
