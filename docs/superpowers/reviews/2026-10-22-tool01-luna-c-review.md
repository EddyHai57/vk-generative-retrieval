# Tool 1 independent review — user-contiguous sequence + offsets

- Reviewer/model: gpt-6-luna/high
- Scope: tool 1 only (`throughput_sequences.py`, shared `throughput_config.py` and configs, its tests, harness boundary). Tools 2–7 were not reviewed; only shared interfaces were read.
- Safety: local source and synthetic fixtures only. No server, real data, test split, or real model training accessed. No repository files changed. Main checkout showed only the new throughput files as untracked; no tracked old entrypoint changes were present.
- References: `docs/decision_log.md` §2026-10-22 (recorded in commit `11249b6`), `docs/superpowers/specs/2026-10-22-throughput-options.md`, and implementation plan.

## Quality conclusion

**Do not treat tool 1 as ready for production integration yet.** The basic row reconstruction is plausible and fail-closed for missing, future, non-like, or otherwise ineligible targets. The supplied regression and mutation receipts pass. Two material gaps remain: the exact G/P step harness exercises the fixture adapter rather than the columnar production constructor, and the production constructor silently coerces some malformed columns in ways that can alter frozen feature values. The configured byte limit also does not currently bound constructor peak allocation. These are engineering/equivalence proof gaps; this review makes no scientific validity, anchor, or promotion judgment.

## Findings

1. **[P1] Frozen-step parity does not exercise the production columnar build.** `analysis/throughput_step_harness.py:46` constructs the store with `SequenceStore.from_events(...)`; the production interface is `from_columns(...)` (`src/vkgr/throughput_sequences.py:21`). The test at `tests/test_throughput_sequences.py:14-21` compares `rows()` output from both paths on one fixture, but the G/P receipts only establish full tensor/parameter/gradient/optimizer/loss/RNG parity for rows emitted from `from_events`. Thus they do not establish the required step parity for the columnar production path, including its sorting, casts, and references. Before integration, feed the same synthetic fixture through `from_columns` in the step harness and compare its rows and full frozen step state.

2. **[P1] Columnar input schema permits silent value changes.** `from_columns` casts `references` to `uint64` before validating (`throughput_sequences.py:24`), and casts `side`/`ctx` to `int64` at return (`:34`) without first requiring integer dtypes or lossless range. Negative/floating reference values can wrap/truncate, while floating side/context values truncate. This can change original sample indices or feature tensors without a schema error, contrary to the frozen dtype/bytes requirement. Validate shape, signedness/range and allowed exact dtypes before conversion; explicitly reject non-integral or lossy values. Add malformed-input checks and dtype/bytes parity coverage for the production path.

3. **[P2] `byte_limit` is not a reliable peak-allocation cap.** `from_columns` estimates source array bytes plus `n*8` only (`throughput_sequences.py:29-30`), but then allocates a stable sort index and sorted/cast output arrays (`:31-34`); the `references` coercion/reshape can itself allocate before the check (`:24`). The estimate therefore can accept an input whose constructor peak exceeds the declared budget. Either budget all known simultaneous buffers conservatively, or construct in bounded/chunked form and make the cap semantics explicit. Do not describe this parameter as a hard allocation cap until verified.

## Checks and evidence

- `cache/throughput-20261022/01-red.log`: initial import failed because the test runner did not expose `src` on `PYTHONPATH`; this is harness setup, not a product assertion failure.
- `01-02-regression-v2.log`: 9 tests passed (5 sequence tests plus 4 worker tests).
- `mutations-01-v1/receipt.json`: baseline accepted; future, order, mask, and key mutations each rejected. Receipt states isolated copy, unique cache, and bytecode disabled for all cases.
- `g-synthetic-step-v1.json` and `p-synthetic-step-v1.json`: report exact synthetic step state across workers 1/2/3, quality metric false, real anchor `NOT_RUN`. Harness source confirms the `from_events` limitation above.
- Full suite was not rerun, per task instruction. A direct ad hoc Python boundary run was attempted but unavailable in this shell (`python`/`uv` absent; `python3` lacks NumPy), so no extra runtime result is claimed.

## Other scoped observations

- Sequence ordering is built from stable user sort while within-user physical order remains increasing (`throughput_sequences.py:28,31-34`). Row retrieval cuts strictly before target and keeps target out of history (`:63-71`); `past_likes` uses all prior likes, separate from clipped history (`:66`); side/context are sliced to the same history tail (`:69-71`). Original reference order is retained and `example_id` is the reference index.
- Fixture adapter is explicitly marked non-production (`:37-40`), while columnar construction avoids per-event Python dicts (`:21-34`).
- `load(..., mmap=True)` uses NumPy read-only mmap and disables pickle (`:85-98`); the batch row materialization is a deliberate list conversion. The store is saved from already materialized arrays, so mmap support alone does not make initial construction memory-bounded.
- Shared defaults keep all optimizations off and workers at 1 (`configs/throughput/disabled.yaml:1-14`; config class defaults at `throughput_config.py:11-26`). `manifest()` places the config SHA under a separate `engineering` key (`:43-49`), leaving supplied science identity unchanged. Current tests verify this object boundary, though no consumer integration was reviewed.
