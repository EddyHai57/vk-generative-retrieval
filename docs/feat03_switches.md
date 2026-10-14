# FEAT-03 P switches (new version; FEAT-02 / FEAT-01 / Batch1 / P4D / P4E / P4H unchanged)
Design source: the FEAT-03 design and the frozen card in the main repo. This file only explains the code.
All FEAT-03 code is new files; no pre-existing tracked file is edited (`git diff --name-status <base>..HEAD` shows only `A`).
## Shared conditions
- Anchor: F3 (timespent and ratio buckets of FEAT-01). Every variant is F3 plus one more variable. N1 loss (in-batch logQ, alpha=1, no extra uniform), `init=small` (N(0,0.02), PAD rows 0), `HIST_V2_OBS` L50, 100/5/3 epochs, seed 42, attempt 1.
- Anti-leak rule: a feature may only use fields of history events strictly before the target. The week used by the recency features is the week of the history event itself; the anchor of the week gap is the last history event, never the target.
- Side tensor: `(B, W, 6)` long per history slot `[feedback_mask, timespent, duration, week, place, platform]`, PAD slots all 0. Place and platform are never filled by FEAT-03.
## Variants: `feature` (`src/vkgr/retrieval_feat03.py`)
| value | what it is | own new parameters (total / trainable) |
|---|---|---|
| `f4_control` | equivalence fixture for FEAT-02 F4 (F3 + position + week gap). **Tests and synthetic runs only; a real run (`train` / `verify`) rejects it. Never submitted.** | 960 / 832 |
| `f3_pos_only` | Q1a: F3 + `P[position bucket]` only (diagnostic) | 512 / 448 |
| `f3_gap_only` | Q1b: F3 + `G[week-gap bucket]` only (diagnostic) | 448 / 384 |
| `f4_recency_pool` | Q2: the F4 event vectors, pooled with 13 learnable logits (7 position + 6 week gap, initial value exactly 0) as a normalised weighted mean | 13 / 13 |
| `f3_user_activity` | Q3: F3 pooled vector + `Len[length bucket of n]` + `Act[bucket of the number of distinct history weeks]` (two user-level tables, no PAD row; edges come from the train-only audit) | 768 / 768 |
- The model is built with the FEAT-02 base feature string for every variant, so the item, T, R, P, G tables are drawn in the F4 order from the same seed; the tables a variant does not use are dropped after initialisation. The variant name is `model.variant`.
- `bucket_spec3_sha256` covers the FEAT-02 spec digest, the Q3 edges, the variant list and the Q2 logit sizes; it is stored in every run binding.
## Runner: `src/vkgr/retrieval_feat03_run.py`
- Config schema `p-feat03-run-v1`, `task_id=FEAT-03`. `configs/feat03/p-feat03-{q1a-pos-only,q1b-gap-only,q2-recency-pool,q3-user-activity}-s42.json` differ from `configs/feat02/p-feat02-f4-recency-s42.json` only in `schema`, `task_id` and `feature` (tested).
- Inputs spec: schema `p-feat02-inputs-v1`; all four sidecar pins (ctx and attr manifest and sha256) must be null, otherwise `ValueError`.
- Week column: filled for `f4_control`, `f3_gap_only`, `f4_recency_pool`, `f3_user_activity` (module constant `NEEDS_WEEK`), 0 for `f3_pos_only`.
- Extra diagnostics: Q2 records `recency_logits` (`position`: 7 values, `weekgap`: 6 values) in every epoch; Q3 records `data_diagnostics.user_activity_buckets` (counts per length and activity bucket for train examples and validation queries, feature tensors only) and `q3_scale_ratio` per epoch (median of `||Len + Act|| / ||pooled||` over the validation queries, computed on CPU/numpy). Verify recomputes the bucket counts from the pinned data and re-exports the best checkpoint, whose diagnostics must equal the archive.
## CLI (`python -m vkgr.retrieval_feat03_run <command>`)
| command | arguments | purpose |
|---|---|---|
| `synthetic-train` | `--config --output [--device {cpu,cuda}]` | train on the built-in synthetic fixture; `--device` defaults to `cpu` and then behaves like FEAT-02's `synthetic-train` |
| `synthetic-verify` | `--config --manifest --manifest-sha [--device {cpu,cuda}]` | verify a synthetic run; the device must equal the one used by `synthetic-train`, otherwise the binding differs and the command exits with code 2 |
| `train` | `--inputs --inputs-sha --config --config-sha --output` | real pinned data, always CUDA |
| `verify` | `--inputs --inputs-sha --config --config-sha --manifest --manifest-sha` | real pinned data, always CUDA |
- `--device cuda` first checks `torch.cuda.is_available()`; without CUDA the command prints `ERROR P-FEAT03 gate failed: ValueError: actual CUDA environment required` to stderr, nothing to stdout, and exits with code 2. A missing deterministic cuBLAS workspace setting also exits with code 2.
- Any gate failure prints `ERROR P-FEAT03 gate failed: <ErrorType>: <message>` to stderr and exits with code 2. An out-of-set `--device` value is rejected by argparse (also code 2).
- `synthetic-train` followed by `synthetic-verify` on CPU is the local round trip; the same two commands with `--device cuda` are the server rehearsal for Q2 and Q3 (synthetic data, no quota) before any real run.
## Verify coverage (fail-closed)
- Binding includes `feature`, `side_sha256`, `bucket_spec_sha256`, `param_report` and (for the week variants) `weeks_sha256`. Any change in data or constants gives a binding mismatch.
- `history_support_slices`, `support_metrics`, `top50_unsupported_share` are recomputed from the archived predictions; the best checkpoint is re-exported and its result and diagnostics must equal the archive. Only the best epoch's Q2 logits and Q3 scale ratio are verified; the other epochs' values are descriptive.
- Not verified locally: bitwise GPU re-export for the Q2 parameters and the Q3 tables (first server rehearsal).
