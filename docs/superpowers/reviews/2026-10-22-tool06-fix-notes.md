# Tool 6 repair notes

## Changed

- Parallel decode now requires `original_decode` to be the owner-bound `decode` method. The owner's `identity()` must equal `state_sha`; the owner's type/source identity and token are checked against each worker session, and the original owner is rechecked after the run.
- `TorchSession` is exercised through the actual `decode_queries` worker path. Its identity covers decoder callback and module source/defaults/closure, model and codec class/module source, complete codec instance namespace and immutable catalog array bytes, model weights, Python/NumPy/Torch/CUDA runtime values, thread and deterministic flags, and beam. Unknown source or unsupported mutable namespace values fail closed.
- The exact `vkgr.sid_depth.ValidatedCodecD` type is normalized only when the exact class and `generative_data._ValidatedCodec` base are import-authenticated and `check_identity()` succeeds. Its `_model` and `_references` aliases are checked but not rehashed as repeated objects; `_array_metadata` contributes shape, dtype, nbytes and strides while its `ctypes.data` pointer is checked locally and excluded from cross-process identity. Other custom codec types keep the generic strict namespace path.
- Initializer storage accounting now counts each NumPy array in the codec namespace once by alias identity, including `ids`, `sids`, `keys` and `key_items`; alias IDs are used only for accounting, never as identity digest data.
- TorchSession multiprocessing requires an `InitializerBudget` tied to the source session identity. Its estimate must cover at least the inspected model/codec storage floor and fit the explicit per-worker cap before the pool starts. Post-initializer and per-task RSS checks remain.
- CUDA mode requires a `VRAMReceipt` with positive finite per-worker peak and live-free values plus context-inclusive measured peak and current-call free-space attestations. Unknown or malformed receipts fail closed.
- Parallel `result_bound` must be a positive exact integer. `decode_workers=1` still directly calls the original callback.

## Verification

The original focused suite passed: `9 passed in 4.23s`. After the codec compatibility addition, the full focused suite passed: `10 passed in 5.29s`. The tests include frozen `constrained_beam_d` on a synthetic catalog for workers 1/2/3; TorchSession worker identity matching; weight, codec class, and codec namespace changes; owner mismatch; initializer budget rejection before factory invocation; initializer failure; bad VRAM receipt fields; positive result-bound validation; and same-structure `ValidatedCodecD` pointer-independent identity and unique-array storage accounting.

Receipts: `cache/throughput-20261022/06-fix-green.log`, `cache/throughput-20261022/06-alias-red.log`, `cache/throughput-20261022/06-alias-green.log`, and `cache/throughput-20261022/06-final-green-v2.log`.

## Limits

- No CUDA, production loader, real catalog/data, anchor, or training was run.
- `InitializerBudget` is an explicit preflight estimate and cap, not a measured initializer peak. Its estimate is validated against the model/codec byte floor but Python multiprocessing provides no portable hard ceiling on transient RSS before factory completion. Post-load RSS checks catch persistent overage; a transient allocation can exceed the estimate before that check. A production caller must provide a defensible estimate/cap before enabling TorchSession multiprocessing.
- The VRAM receipt fields are caller attestations. This CPU-only work does not prove that a production caller measures them with the required context-inclusive and per-call semantics.
- Per-query identity recomputes full callback/model/codec bytes to detect mutation; performance impact is UNKNOWN and was not benchmarked. Validation was not weakened for speed.
- Initializer exceptions surface through `ProcessPoolExecutor` as `BrokenProcessPool`; tests prove the call fails without returning decoded results, but the worker's underlying exception is not wrapped into a parent-readable structured error.
- No independent review was performed or is claimed here.
