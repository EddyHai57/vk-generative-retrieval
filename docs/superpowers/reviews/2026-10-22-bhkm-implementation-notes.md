# BHKM exact external-sort implementation notes

## Scope and interface

Added `src/vkgr/throughput_bhkm.py` as an opt-in implementation adapter. The caller supplies the frozen module loaded from `b5a4d1c:src/vkgr/sid.py`; the adapter returns `(catalog, trace_or_none)` and constructs the catalog through that module's existing `SIDCatalog`. The caller can keep the canonical `fit` path as the default until its engineering config explicitly selects external BHKM.

Each run requires a new `root`. Sorted `.npy` edge runs are stored under unique node/iteration/block names, changed to read-only after writing, memory-mapped for merging, and retained. A second invocation against the same root fails before writing. Configured disk and memory limits are checked before run creation and at run/objective boundaries; the existing 80% disk and 60% memory headroom checks also apply. The disk preflight bound includes all depth/iteration edge records and a conservative per-level run-header bound. Replays consume additional disk because old runs remain in place.

## Exactness argument

For each block, distances use the source implementation's float64 array and the same ascending 64-dimension `delta * delta` accumulation. Each edge record is exactly 16 bytes: `(distance: float64, item_id: uint32, child: uint32)`. `np.lexsort((child, item_id, distance))` gives the original global key `(distance, item_id, child)`; block runs are sorted by that full key and merged by a single-threaded heap. Greedy capacity mutation and label assignment stay on the caller process.

The original RNG instance and depth-first traversal stay in the caller process; each node makes the same single `rng.choice` request before descending children in ascending order. Centers still use `rows[labels == j].mean(axis=0, dtype=np.float64)` over the full selected row slice. The objective path computes the selected post-update distance for every row into one N-element vector and invokes one `sum(dtype=np.float64)`, preserving NumPy's original reduction shape rather than summing partial blocks.

## RED, GREEN, and evidence

The first test invocation failed as expected because `throughput_bhkm` did not exist. After implementation, all five local tests passed. They compare the complete catalog, node metadata, trace distances and labels, per-round centers/objectives/convergence, SIDs, suffix vocabulary through audit, and audit output against the original Git blob. They cover workers 1/2/3, non-divisible capacities, short final blocks, all-equal distances, repeated and zero vectors, empty input rejection, float64 dimension accumulation, fresh-root enforcement, configured resource rejection, and same-seed replay. Each successful fit directory and its run files remain under `cache/throughput-bhkm-20261022/fit-verify-*`.

Mutation checks used isolated source copies, `python -B`, `PYTHONDONTWRITEBYTECODE=1`, and a unique `PYTHONPYCACHEPREFIX` per copy. The final checks killed all four requested mutants: reversed tie child order, omitted capacity remainder, reversed feature dimension order, and an extra RNG draw before initialization. Evidence and individual outputs are retained in `cache/throughput-bhkm-20261022/mutation-report-*.txt` and corresponding `mutation*-*/` directories. An earlier mutation batch is retained but excluded because the copied package initializer was missing and Python imported the unmutated module; the corrected batches assert the imported module path before executing tests.

## Limits and handoff

Validation is local and uses synthetic tiny arrays only. No server, real data, scientific training, or performance claim was involved. This validates exact behavior and worker-count determinism for the tested cases; it does not establish full-catalog runtime or the sufficiency of any machine's disk. Retained multi-round, multi-node, and replay runs can exceed a single-run estimate, so reaching a resource cap stops the run and leaves existing evidence untouched.

This branch adds only the implementation, its tests, the requested cache evidence, and this note. The module has not been connected to the caller's engineering adapter. No commit or push was made; integration and the required independent Luna review remain with the parent execution task.

## Resource-guard follow-up

The follow-up adds `prefetch=1` to the API. Spawned workers now receive blocks lazily through an ordered bounded submitter, with no more than `workers * prefetch` futures outstanding; the former eager `jobs` list and eager `pool.map` submission are gone. Each submitted block reserves its estimated disk bytes before enqueue, and completions are consumed in input order. A fake executor test verifies the pending-future ceiling, and a two-worker fit comparison verifies `prefetch=1` and `prefetch=2` produce identical node metadata and SIDs.

Memory capacity now uses the minimum of host RAM and detected cgroup limits. Per-block planning includes a 96-byte-per-edge working-set estimate for distances, the 16-byte run record, row/child/order arrays, repeated item keys, and lexsort workspace; it also reserves serialized input blocks and a per-worker RSS baseline for the bounded queue. This is a conservative estimate rather than a measured worker peak.

For `trace=True`, the preflight reserves a whole-fit upper bound across all levels and maximum rounds for assignment and objective distance matrices, labels and convergence snapshots, per-round centers, final center lists, SID copies, trace objects, and concatenation peak. The remaining trace reservation is included in later stage guards before additional trace arrays are built. A test captures the pre-allocation guard and checks that it includes the trace estimate.

TDD follow-up: all four new targeted tests first failed against the prior API/helpers (`_bounded_submit` missing, `prefetch` unsupported, cgroup-min helper missing, and trace-budget API unavailable). After implementation, the same four tests passed with `PYTHONDONTWRITEBYTECODE=1` and `python -B`. The targeted command covered bounded outstanding futures and order, prefetch 1/2 catalog equality, host/cgroup minimum selection, and trace memory reservation before allocation. No broad test suite, server, or real data was used in this follow-up.
