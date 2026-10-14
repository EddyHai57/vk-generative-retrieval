# FEAT-02 P switches (new version; Batch1 / FEAT-01 / P4D / P4E / P4H unchanged)
Design source: the FEAT-02 plan and the frozen card (section 9) in the main repo. This file only explains the code.
All FEAT-02 code is new files; no pre-existing tracked file is edited (`git diff --name-status <base>..HEAD` shows only `A`).
## Shared conditions
- Anchor: F3 (`f3_watch_buckets` of FEAT-01). Every variant is F3 plus one change. N1 loss (in-batch logQ, alpha=1, no extra uniform), `init=small` (N(0,0.02), PAD rows 0), `HIST_V2_OBS` L50, 100/5/3 epochs, seed 42, attempt 1.
- Anti-leak rule: a feature may only use fields of history events strictly before the target. Bucket edges, category tables and the author vocabulary are train-defined and hard-coded in `src/vkgr/retrieval_feat02.py`; changing any of them changes `bucket_spec_sha256`, so older manifests fail verify.
- Side tensor: `(B, W, 6)` long per history slot `[feedback_mask, timespent, duration, week, place, platform]`, PAD slots all 0. A variant reads only its own columns; the others cannot change its output (tested).
## Model: `feature` (`src/vkgr/retrieval_feat02.py`)
| value | change over F3 | own new tables | params total / trainable (F3 T+R included) |
|---|---|---|---|
| `f3` | equivalence control for tests, not submitted | none | 896 / 768 |
| `f3_masked` | the T and R vectors are added only on **supported** history events; unsupported events stay zero as in N1. Not counted as a candidate | none | 896 / 768 |
| `f3_f4_recency` | adds `P[position bucket]` and `G[week-gap bucket]` to every event; anchor = last history event, never the target | position 512 / 448, week gap 448 / 384 | 1856 / 1600 |
| `f3_f5_scene` | adds `Place[place id]` and `Platform[platform id]` of the event's own row; unseen value -> OOV row, which contributes 0 | place 1664 / 1536, platform 832 / 704 | 3392 / 3008 |
| `f3_f6_author_content` | item tower (history side and candidate side share it) gets `A[author row] + W c` added; unsupported items are no longer zero vectors | author 712640 / 712512, projection 4096 / 4096 | 717632 / 717376 |
- "trainable" = total minus the rows fixed at 0 (PAD; for place, platform and author tables also OOV). The F3 convention (T + R = 896 / 768) is included in the last column; the own-table column is the plan's convention.
- Train-defined constants: position edges `(1,2,4,8,16,32)` (7 buckets); week-gap edges `(1,2,3,5,9)` (6 buckets, the 17+ bucket was merged because it held 1.94% < 2%); place values 0..23 (24); platform values 0..10 (11); author vocabulary = authors of supported catalog items, V = 11133, its sha256 is hard-coded. Rows: 0 = PAD, 1 = OOV, 2+ = value.
- F4 week gap = week of the last history event minus the event's week; position = back index from the last event. Neither uses the target week or position.
- F6 item vector: supported = `emb + A + W c`; unsupported with a vocabulary author = `A + W c`; unsupported with an OOV author = `W c`; PAD item = 0. `W` is `Linear(64, 64, bias=False)` initialised N(0,0.02). `c` is the prepared official catalog content vector (L2-normalised, PAD row 0). The id embedding of an unsupported item is masked to 0, so an untrained id row cannot affect scores. `content` and `author_index` are checkpoint buffers and immutable on load.
- Draw order: item embedding, T, R, then the variant tables (position, week gap | place, platform | author, projection). Under the same seed the item table and T, R equal F3 bit for bit (tested). Add order in the user tower: `((((enc + T) + R) + X1) + X2)`.
- Known interpretation limits (also in the card): mean pooling makes the F4 position term a function of history length only; F6 couples the history side and the item tower; OOV targets are not retrievable by construction (not in the frozen catalog); single seed.
## Data sidecars
- `HIST_V2_SIDE_V1` (`src/vkgr/history_side.py`, unchanged): `[feedback_mask, timespent, duration]` per event. All variants use it.
- F4 needs no new sidecar: the week of each event comes from the pinned `history_positions`.
- `HIST_V2_CTX_V1` (`src/vkgr/history_ctx.py`, F5): `[place, platform]` per history event, bound by sha256 to one history-v2 prepare manifest; L100 produced, L50/L20 are suffixes. `agent` is never read. CLI `prepare-ctx` writes it, `verify-ctx` rebuilds it from raw train and compares (`raw ctx replay mismatch` on drift).
- `ITEM_ATTR_V1` (`src/vkgr/item_attr.py`, F6): catalog-ordered `author_id` plus the ascending author vocabulary of supported items; bound to the items metadata sha, catalog sha and support sha. Real path asserts the vocabulary size and sha256 equal the hard-coded values. CLI `prepare-attr` / `verify-attr` (`raw attr replay mismatch` on drift). Its functions take no query, holdout, valid, target or test argument (signature-guarded).
## Runner: `src/vkgr/retrieval_feat02_run.py`
- Config schema `p-feat02-run-v1`, `task_id=FEAT-02`: the FEAT-01 field set with `feature` taking the values above. `configs/feat02/p-feat02-{f3-masked,f4-recency,f5-scene,f6-author-content}-s42.json` differ from `configs/feat01/p-feat01-f3-watch-buckets-s42.json` only in `schema`, `task_id` and `feature` (tested).
- Inputs spec `p-feat02-inputs-v1` = FEAT-01 inputs plus `ctx_manifest/ctx_sha256` and `attr_manifest/attr_sha256` (each pair all-or-nothing). F5 requires the ctx pair, F6 requires the attr pair, other variants must keep both pairs null.
- CLI: `synthetic-train`, `train`, `verify`, `prepare-ctx`, `verify-ctx`, `prepare-attr`, `verify-attr`.
- Diagnostics in every epoch: `history_support_slices`, `support_metrics` (target supported / unsupported / OOV), `top50_unsupported_share`, `item_norm_stats` (raw L2 norms before normalisation). F6 adds `data_diagnostics.item_attr_coverage`.
## Verify coverage (fail-closed)
- Binding includes `feature`, `side_sha256`, `bucket_spec_sha256`, `param_report`; F4 `weeks_sha256`; F5 `ctx_sha256`; F6 `attr_sha256`. Any change in data or constants -> binding mismatch.
- `history_support_slices`, `support_metrics`, `top50_unsupported_share` and `item_attr_coverage` are recomputed from the archived predictions; the best checkpoint is re-exported and its result and diagnostics must equal the archive.
- Checkpoint immutability covers `supported`, `content` and `author_index`.
- Not verified locally: bitwise GPU re-export for the new tables (first server verify), real parquet path (mocked in tests).
