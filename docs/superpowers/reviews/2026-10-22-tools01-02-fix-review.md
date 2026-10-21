# Tool 01/02 targeted follow-up review

- Reviewer/model: Codex GPT-6 (the exact Luna model identity was not available to this reviewer; do not attribute this report to Luna).
- Scope: only the fixes to the two prior blocking reviews, their stated receipts, and a small synthetic output-bound fixture. No server, real data, test split, CUDA, training, or full-suite run. No repository files changed.
- Context-count status: UNKNOWN.

## Tool 1 — `throughput_sequences.from_columns`

**Original value-corruption blocker: addressed.** `from_columns` now rejects non-integer side/context and references, checks side/context int64 range, rejects invalid reference IDs, and performs stable user sorting only after validation. The v2 G/P receipts claim the production `from_columns` path and original cursor at workers 1/2/3 with full synthetic step state equal. The fresh RED receipt shows the lossy-column assertion failed before the implementation change. The existing isolated mutation receipt covers future/order/mask/key mutations. These are accepted as supplied evidence; I did not rerun them.

**Original allocation-budget blocker: improved, but still partially unaddressed.** The estimate now includes twice the input column bytes plus a linear workspace allowance before `argsort`, `unique`, and output-column construction. However, `references` is reshaped and cast to `uint64` before the estimate/check. For a large signed or non-contiguous references input, that conversion may allocate a full copy before `byte_limit` can reject it. The current formula can account for the copy in its arithmetic, but the check does not prevent that pre-check peak allocation. This remains a narrow P2 gap if `byte_limit` is meant to cap constructor peak allocation. Run with already-normalized contiguous arrays, the later major sorting/output allocations are guarded by the new estimate.

## Tool 2 — `throughput_workers`

**Original in-flight input/output accounting blocker: addressed for the `assemble_rows` path, with limits.** Submission reserves `3 * (serialized task input + rows_bound)` and refuses a task that cannot fit; the assembler supplies a bound based on clipped history and the full prior-like prefix. The worker checks the actual serialized result against that bound before returning it. A local synthetic fixture with 1,200 events measured pickle output of 3,535 bytes at history length 1, 5,254 bytes at length 50, and 44,453 bytes at length 2,000; corresponding bounds were 85,952, 136,128, and 1,312,704 bytes. This supports the estimator for those fixtures, not a proof for every batch/configuration. The v3 log reports 24 regression passes, and the v2 worker mutation receipt reports isolated RNG/order/cursor mutations rejected; neither receipt is a measurement of real RSS or peak allocation.

The cap depends on callers supplying a truthful `result_size_hint`; the post-build serialized-size check detects an understated hint only after the result has been materialized. `assemble_rows` supplies its own conservative bound, while arbitrary `ordered_map` callers do not receive a general pre-allocation guarantee. Worker RSS checks plus the process-tree/cgroup guard add runtime detection, but they run at task/row boundaries and do not prove that transient peaks can never exceed the configured threshold.

**Cancellation improvement: addressed as requested, with a stop-latency limitation.** `finally` now calls `future.cancel()` for pending futures and `shutdown(wait=True, cancel_futures=True)`. This cancels work that has not started. Already-running tasks still finish before shutdown returns, so a stuck worker can still delay close/error propagation; the patch does not establish bounded cancellation latency.

## Overall disposition

- Tool 1: original schema/equivalence blockers addressed; constructor peak-budget guarantee remains partially unaddressed because reference coercion can allocate before the check.
- Tool 2: the original assembly-path output reservation blocker is addressed by the conservative row bound, reservation, post-build check, and RSS guards. Arbitrary `ordered_map` hints and transient peak memory remain limitations; cancellation does not stop already-running work.
- Real anchor, CUDA, real data, real RSS, mmap startup peak, and throughput gain: **NOT_RUN / UNKNOWN**. Synthetic parity and the small output-bound fixture do not establish those properties.
