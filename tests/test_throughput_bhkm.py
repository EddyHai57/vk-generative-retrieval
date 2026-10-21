import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from vkgr import throughput_bhkm


REPO = Path(__file__).resolve().parents[1]
SID_BLOB = "b5a4d1c:src/vkgr/sid.py"
EVIDENCE_ROOT = REPO / "cache/throughput-bhkm-20261022"


def load_original():
    source = (REPO/'tests/fixtures/throughput_frozen_sid.py').read_bytes()
    if __import__('hashlib').sha256(source).hexdigest()!='9602580f907cc2e8cfa3684aec7828df2bdfc87b59e71b6fd3865a9a19de91a3':
        # Compare to the frozen local Git identity where available.
        expected=subprocess.check_output(["git","show",SID_BLOB],cwd=REPO)
        if source!=expected:raise ValueError('frozen original SID bytes differ')
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("sid_reference", loader=None))
    exec(compile(source, SID_BLOB, "exec"), module.__dict__)
    return module


def sample():
    rows = np.zeros((11, 64), dtype=np.float32)
    rows[1, 0] = 1
    rows[2, 1] = 1
    rows[3, 0] = 1
    rows[4, 2] = 1
    rows[5, 3] = 1
    rows[6, 1] = 1
    rows[7, 4] = 1
    rows[8, 5] = 1
    rows[9, 2] = 1
    rows[10, 0] = 1
    ids = np.array([91, 8, 80, 7, 70, 6, 60, 5, 50, 4, 40], dtype=np.uint32)
    return ids, rows


def reference_trace(module, ids, features, **kwargs):
    trace = {"distance": [], "labels": [], "centers": []}
    original_distance, original_assign = module._distance, module._assign

    def observed_distance(rows, centers):
        result = original_distance(rows, centers)
        trace["distance"].append(result.copy())
        trace["centers"].append(centers.copy())
        return result

    def observed_assign(distances, item_ids):
        result = original_assign(distances, item_ids)
        trace["labels"].append(result.copy())
        return result

    module._distance, module._assign = observed_distance, observed_assign
    try:
        catalog = module.fit(ids, features, **kwargs)
    finally:
        module._distance, module._assign = original_distance, original_assign
    trace["nodes"] = catalog.nodes
    trace["sids"] = catalog.sids.copy()
    trace["audit"] = catalog.audit()
    return catalog, trace


class BHKMExternalSortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        cls.run_root = Path(tempfile.mkdtemp(prefix="fit-verify-", dir=EVIDENCE_ROOT))

    def test_configured_default_and_external_preserve_catalog_api_and_boundary(self):
        from vkgr.throughput_config import Engineering
        ids,features=sample();old_calls=[];new_calls=[]
        old=throughput_bhkm.fit_configured(load_original(),ids,features,Engineering(),run_id='disabled',
            branching=3,depth=2,seed=7,max_iter=2,on_boundary=lambda:old_calls.append(1))
        config=Engineering(bhkm_external=True,bhkm_workers=2,prefetch=2,bhkm_rows=3,temp_dir=str(self.run_root.resolve()))
        new=throughput_bhkm.fit_configured(load_original(),ids,features,config,run_id='configured',
            branching=3,depth=2,seed=7,max_iter=2,on_boundary=lambda:new_calls.append(1))
        self.assertFalse(isinstance(new,tuple));self.assertEqual(old_calls,new_calls)
        np.testing.assert_array_equal(new.sids,old.sids)

    def test_exact_catalog_and_trace_across_worker_counts_and_block_edges(self):
        ids, features = sample()
        reference, ref_trace = reference_trace(
            load_original(), ids, features, branching=4, depth=2, seed=29, max_iter=4
        )
        for workers in (1, 2, 8):
            with self.subTest(workers=workers):
                catalog, actual = throughput_bhkm.fit(
                    load_original(), ids, features,
                    root=self.run_root / f"workers-{workers}", workers=workers,
                    block_rows=3, branching=4, depth=2, seed=29,
                    max_iter=4, trace=True,
                )
                np.testing.assert_array_equal(catalog.ids, reference.ids)
                np.testing.assert_array_equal(catalog.sids, reference.sids)
                np.testing.assert_array_equal(catalog.keys, reference.keys)
                np.testing.assert_array_equal(catalog.key_items, reference.key_items)
                self.assertEqual(catalog.nodes, reference.nodes)
                self.assertEqual(catalog.audit(), reference.audit())
                self.assertEqual(actual["nodes"], ref_trace["nodes"])
                self.assertEqual(actual["audit"], ref_trace["audit"])
                np.testing.assert_array_equal(actual["sids"], ref_trace["sids"])
                self.assertEqual(len(actual["rounds"]), len(ref_trace["labels"]))
                for got, want in zip(actual["rounds"], ref_trace["labels"]):
                    np.testing.assert_array_equal(got["labels"], want)
                self.assertEqual(len(actual["distances"]), len(ref_trace["distance"]))
                for got, want in zip(actual["distances"], ref_trace["distance"]):
                    np.testing.assert_array_equal(got, want)
                for i, round_trace in enumerate(actual["rounds"]):
                    np.testing.assert_array_equal(round_trace["centers_before"], ref_trace["centers"][2 * i])
                    np.testing.assert_array_equal(round_trace["centers"], ref_trace["centers"][2 * i + 1])
                    self.assertEqual(round_trace["objective"], reference.nodes[
                        next(j for j, node in enumerate(reference.nodes) if node["prefix"] == round_trace["prefix"])
                    ]["objectives"][round_trace["iteration"]])
                files = list((self.run_root / f"workers-{workers}").rglob("*.npy"))
                self.assertTrue(files)
                self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in files))
        replay, _ = throughput_bhkm.fit(
            load_original(), ids, features, root=self.run_root / "same-seed-replay",
            workers=2, block_rows=3, branching=4, depth=2, seed=29,
            max_iter=4, trace=False,
        )
        np.testing.assert_array_equal(replay.sids, reference.sids)
        self.assertEqual(replay.nodes, reference.nodes)

    def test_float64_dimension_reduction_matches_on_nontrivial_features(self):
        rng = np.random.default_rng(20261022)
        features = rng.normal(size=(9, 64)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        ids = np.arange(100, 109, dtype=np.uint32)
        reference, expected = reference_trace(load_original(), ids, features, branching=3, depth=1, seed=7, max_iter=3)
        actual, trace = throughput_bhkm.fit(
            load_original(), ids, features, root=self.run_root / "float64", workers=1,
            block_rows=4, branching=3, depth=1, seed=7, max_iter=3, trace=True,
        )
        self.assertEqual(actual.nodes, reference.nodes)
        for got, want in zip(trace["distances"], expected["distance"]):
            np.testing.assert_array_equal(got, want)

    def test_equal_distance_edges_follow_item_then_child_lexicographic_order(self):
        ids = np.array([8, 2, 7, 1, 6, 3, 5], dtype=np.uint32)
        features = np.zeros((len(ids), 64), dtype=np.float32)
        reference = load_original().fit(ids, features, branching=3, depth=1, seed=3, max_iter=2)
        actual, _ = throughput_bhkm.fit(
            load_original(), ids, features, root=self.run_root / "equal-distance-ties",
            workers=1, block_rows=2, branching=3, depth=1, seed=3, max_iter=2,
        )
        self.assertEqual(actual.nodes, reference.nodes)
        np.testing.assert_array_equal(actual.sids, reference.sids)

    def test_empty_input_and_existing_root_are_rejected_without_overwrite(self):
        module = load_original()
        root = self.run_root / "existing-root"
        with self.assertRaises(ValueError):
            throughput_bhkm.fit(module, np.array([], dtype=np.uint32), np.empty((0, 64), dtype=np.float32), root=root)
        ids, features = sample()
        throughput_bhkm.fit(module, ids, features, root=root, branching=2, depth=1, max_iter=1)
        with self.assertRaises(FileExistsError):
            throughput_bhkm.fit(module, ids, features, root=root, branching=2, depth=1, max_iter=1)

    def test_bad_limits_fail_before_any_run_is_written(self):
        ids, features = sample()
        root = self.run_root / "bad-limits"
        with self.assertRaises((MemoryError, OSError)):
            throughput_bhkm.fit(
                load_original(), ids, features, root=root,
                block_rows=3, branching=4, depth=2, max_iter=4,
                memory_limit_bytes=1,
            )
        self.assertFalse(root.exists())

    def test_bounded_submitter_limits_outstanding_futures_and_preserves_order(self):
        class Future:
            def __init__(self, owner, value):
                self.owner = owner
                self.value = value

            def result(self):
                self.owner.active -= 1
                return self.value

        class Executor:
            active = 0
            peak = 0

            def submit(self, function, value):
                self.active += 1
                self.peak = max(self.peak, self.active)
                return Future(self, function(value))

        executor = Executor()
        result = list(throughput_bhkm._bounded_submit(executor, lambda x: x * x, range(20), 3))
        self.assertEqual(result, [x * x for x in range(20)])
        self.assertEqual(executor.peak, 3)

    def test_prefetch_one_and_two_keep_same_catalog(self):
        ids, features = sample()
        outputs = []
        for prefetch in (1, 2):
            catalog, _ = throughput_bhkm.fit(
                load_original(), ids, features, root=self.run_root / f"prefetch-{prefetch}",
                workers=2, prefetch=prefetch, block_rows=3, branching=4,
                depth=2, seed=29, max_iter=4,
            )
            outputs.append(catalog)
        np.testing.assert_array_equal(outputs[0].sids, outputs[1].sids)
        self.assertEqual(outputs[0].nodes, outputs[1].nodes)

    def test_host_memory_limit_is_capped_by_cgroup(self):
        self.assertEqual(throughput_bhkm._minimum_memory_limit(64_000, 8_000), 8_000)
        self.assertEqual(throughput_bhkm._minimum_memory_limit(64_000, None), 64_000)
        self.assertEqual(throughput_bhkm._minimum_memory_limit(None, 8_000), 8_000)

    def test_trace_memory_is_budgeted_before_trace_arrays_are_allocated(self):
        ids, features = sample()
        captured = []
        real_guard = throughput_bhkm._guard
        expected = throughput_bhkm._trace_memory_bytes(len(ids), 4, 2, 4)
        real_zeros = np.zeros
        checked = []
        def zeros(shape, *args, **kwargs):
            if shape == (len(ids), 3):
                self.assertTrue(captured)
                self.assertGreaterEqual(max(captured), expected)
                checked.append(True)
            return real_zeros(shape, *args, **kwargs)

        def capture(*args, **kwargs):
            captured.append(kwargs.get("projected_memory", 0))
            return real_guard(*args, **kwargs)

        with mock.patch.object(throughput_bhkm, "_guard", side_effect=capture), mock.patch.object(np, "zeros", side_effect=zeros):
            throughput_bhkm.fit(
                load_original(), ids, features, root=self.run_root / "trace-budget",
                workers=1, prefetch=1, block_rows=3, branching=4,
                depth=2, seed=29, max_iter=4, trace=True,
            )
        expected = throughput_bhkm._trace_memory_bytes(len(ids), 4, 2, 4)
        self.assertTrue(captured)
        self.assertTrue(checked)
        self.assertGreaterEqual(max(captured), expected)


if __name__ == "__main__":
    unittest.main()
