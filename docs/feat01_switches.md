# FEAT-01 P switches (new version; Batch1 / P4D / P4E / P4H unchanged)

Design source: `docs/superpowers/specs/2026-10-14-feat01-card.md` (main repo). This file only explains the code.

## Data: `HIST_V2_SIDE_V1` (`src/vkgr/history_side.py`)

- New sidecar bound by sha256 to one `HIST_V2_OBS` prepare manifest (`history_manifest_sha256`). History-v2 files are not modified.
- For every event already in the history-v2 sidecar, it stores that event's own fields: `[feedback_mask, timespent, duration]`.
  - `feedback_mask` bits (frozen order): like=1, share=2, bookmark=4, open_comments=8, click_on_author=16, dislike=32.
  - `timespent`: raw uint8 seconds, 1..255. `duration`: item duration, 1..255 seconds, from `metadata/items_metadata.parquet` pinned by sha in the inputs spec.
- Event set = pinned `history_positions` (strictly before the target, train weeks 00-24 only). The target row and later rows are never read. Shorter lengths are suffixes of the L100 side.
- `prepare-side` writes it. `verify-side` rebuilds it from raw train and compares.

## Model: `feature` (`src/vkgr/retrieval_features.py`)

| value | user tower | new parameters |
|---|---|---|
| `none` | plain id_mean; equals the Batch1 small-init id_mean (equivalence control, not submitted) | 0 |
| `f1_ratio_pool` | weighted mean, `w = clip(timespent/duration, 0.1, 2.0)`, PAD w=0 | 0 |
| `f2_feedback` | event = item emb + sum of set-bit rows of a 6x64 feedback table | 384 |
| `f3_watch_buckets` | event = item emb + timespent-bucket emb (7+PAD) + ratio-bucket emb (5+PAD) | 896 (768 trainable; PAD rows fixed at 0) |
| `c0_unsupported_const` | split control, not counted: event = item emb + one shared 1x64 vector, added to **unsupported** history events only; side values ignored | 64 |

- Timespent buckets: 1-3, 4-9, 10-30, 31-60, 61-134, 135-254, 255.
- Ratio buckets, computed with integer comparisons: `10*ts<dur`, `2*ts<dur`, `ts<dur`, `ts==dur`, `ts>dur`.
- The item tower is unchanged. Extra tables are drawn after the item embedding, so under the same seed the item table equals the anchor's bit for bit.
- F3 adds a vector to every unsupported history event, which id_mean otherwise zeroes. F2 does so only for unsupported events with at least one feedback bit (the like bit is always 0 there, because support = items with a train like). `c0_unsupported_const` makes unsupported history visible without any feedback/watch information; see the card §4.1.

## Runner: `src/vkgr/retrieval_feat01_run.py`

- Config schema `p-feat01-run-v1` = Batch1 fields + `feature` + `side_version`. FEAT-01 batch 1 is restricted to id_mean, L50 and INIT_EMB_N002. The logQ / extra_uniform switches are kept so the anchor can follow the P-NEG winner.
- `configs/feat01/p-feat01-f{1,2,3}-*-s42.json` equal the anchor `configs/batch1/p-neg-n1-logq.json` except for schema/task/feature/side_version.
- The inputs spec is `p-feat01-inputs-v1`: the Batch1 fields plus `side_manifest`, `side_sha256` and `items_metadata_sha256`.
- Verify is fail-closed, as in Batch1, and its binding also includes `side_sha256` and the feature.
- Slices (card §5): every evaluation stores `result.support_metrics` (target supported / unsupported / OOV, from `evaluate_vectors`) and `diagnostics.history_support_slices` (queries bucketed by the train-defined supported fraction of their history: `h0` = 0, `h_lt_half` < 1/2, `h_ge_half` in [1/2, 1), `h1` = 1; R@50 / N@50 / count). Verify recomputes the history slices from the stored predictions for every epoch.
- `data_diagnostics.side_visibility`: validation history events, unsupported events, unsupported events with >=1 feedback bit (what F2 can make visible), unsupported events with the like bit (expected 0).
