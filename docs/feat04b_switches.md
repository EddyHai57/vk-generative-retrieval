# FEAT-04b P switches (new version; FEAT-04a / FEAT-03 / FEAT-02 / FEAT-01 / PHP-01 / Batch1 / P4D / P4E / P4H unchanged)
Design source: the FEAT-04b design and the frozen card kept outside this repository. This file only explains the code.
All FEAT-04b code is new files; no pre-existing tracked file is edited (`git diff --name-status <base>..HEAD` shows only `A`; the pinned files are listed with their sha256 in `configs/feat04b/pinned-files.json`).
## Shared conditions
- Anchor: F4-T010 (id_mean + N1 in-batch logQ alpha=1 + F3 + F4, temperature 0.10, learning rate 0.001). Every real variant is the anchor plus one more variable. `init=small` (N(0,0.02), PAD rows 0), `HIST_V2_OBS` L50, 100/5/3 epochs, seeds 42 / 2024 / 2025, attempt 1.
- Anti-leak rule: a feature may only use fields of history events strictly before the target. The week used by the recency features is the week of the history event itself; the anchor of the week gap is the last history event, never the target. C/D statistics use train weeks 0-24 only (never week 25 validation, never week 26 test).
- Side tensor: `(B, W, 6)` long per history slot `[feedback_mask, timespent, duration, week, place, platform]`, PAD slots all 0. Place and platform are never filled.
## Variants: `feature` (`src/vkgr/retrieval_feat04b.py`)
| value | what it is | own new parameters |
|---|---|---|
| `f4_control` | equivalence fixture for the anchor. **Tests and synthetic runs only; a real run rejects it. Never submitted.** | none beyond the anchor |
| `f4_recency_pool` | chain A = Q2-T010: F4 event vectors pooled with 13 learnable logits (7 position + 6 week gap, initial value exactly 0). **Tests and synthetic runs only here; the real A runs belong to FEAT-04a.** | 13 |
| `f4_item_popularity` | C: candidate side only. `cand(i) = items(i) + supported(i) * (EX[expo bucket] + FW[first-week bucket])` | two bucket tables (a PAD row each) |
| `f4_item_quality` | D: candidate side only. `cand(i) = items(i) + supported(i) * (RT[watch-ratio bucket] + LK[smoothed like-rate bucket])` | two bucket tables (a PAD row each) |
| `f4_recency_behavior` | E: chain A plus six behaviour weights beta (initial value exactly 0): `l_k = u[pb_k] + v[gb_k] + sum_b bit_b(mask_k) * beta_b`; beta acts on the pooling logit only | 13 + 6 = 19 |
- The user tower of C and D is the anchor's, bit for bit; their bucket ids are long buffers (not weights) and a checkpoint with other ids is refused. The candidate vectors go through `model.candidate_items` in both `train_step` and the evaluation export.
- Bucket-id order of E follows `history_side.FEEDBACK_BITS` (like, share, bookmark, open_comments, click_on_author, dislike).
- `bucket_spec_sha256(variant)` is computed per variant (C and D only hash their own constants) and stored in that variant's run binding, so a drifted constant of one variant cannot break another variant's verify.
## Bucket constants (C/D)
- `EXPO_EDGES`, `FIRST_EDGES`, `RATIO_EDGES`, `LIKE_EDGES`, `LIKE_K`, `LIKE_P0` and `STATS_SHA256` live in `src/vkgr/retrieval_feat04b.py`. The edges, `LIKE_P0` and the sidecar digest come from the train-only audit (`scripts/feat04b_audit_constants.py`, stdout stored verbatim as `configs/feat04b/stats-bucket-constants.json`); the grid and the 2% merge rule were fixed before the audit was run.
- `src/vkgr/item_stats.py` builds the sidecar (`n`, `first_week`, `likes`, `ratio_pm_sum` per catalog item; integer arithmetic; `timespent == 0` is legal and counts as a row; a catalog item without any row is an error). It carries no bucket constants.
## Runner: `src/vkgr/retrieval_feat04b_run.py`
- Config schema `p-feat04b-run-v1`, `task_id=FEAT-04B`. The real path accepts temperature 0.1 (float) and learning rate 0.001 (float) only, seeds 42 / 2024 / 2025, and the real variants C, D, E. The synthetic path also accepts 0.15 and `f4_control` / `f4_recency_pool` for the equivalence fixtures.
- Nine real configs: `configs/feat04b/p-feat04b-{c-i4-popfresh,d-i5-quality,e-q2-behavior}-s{42,2024,2025}.json`. They differ from the anchor's H1 configuration only in `schema`, `task_id`, `feature`, `seed`, and have no `dim` / `knob` (tested).
- Inputs spec: schema `p-feat04b-inputs-v1`. A pin that is used must be set and a pin that is not used must be null: C/D need `stats_manifest` and `stats_sha256`; control, A and E need both null; the context and attribute pins are always null.
- Extra diagnostics: E records `recency_logits`, `behavior_beta` and `beta_bit_train_counts`; C/D record `top50_stat_profile`, `target_bucket_hist` and `logq_redundancy`. Verify recomputes them and re-exports the best checkpoint.
## CLI (`python -m vkgr.retrieval_feat04b_run <command>`)
| command | arguments | purpose |
|---|---|---|
| `synthetic-train` | `--config --output [--device {cpu,cuda}]` | train on the built-in synthetic fixture; `--device` defaults to `cpu` |
| `synthetic-verify` | `--config --manifest --manifest-sha [--device {cpu,cuda}]` | verify a synthetic run on the same device |
| `train` | `--inputs --inputs-sha --config --config-sha --output` | real pinned data, always CUDA |
| `verify` | `--inputs --inputs-sha --config --config-sha --manifest --manifest-sha` | real pinned data, always CUDA |
| `preflight` | `--inputs --inputs-sha --config --config-sha` | read-only check on the real inputs (nothing is trained or written); a failing assertion exits with code 2 |
| `range-check` | `--inputs --inputs-sha` | read-only scan of the train weeks 0-24: timespent and duration ranges; any violation exits with code 2 and blocks `prepare-stats` |
| `prepare-stats` / `verify-stats` | `--inputs --inputs-sha --output` / `--manifest --manifest-sha` | build the C/D sidecar / rebuild it from the raw partitions and compare |
- `--device cuda` first checks `torch.cuda.is_available()`; without CUDA the command prints one `ERROR P-FEAT04B gate failed: ...` line to stderr, nothing to stdout, and exits with code 2. Any gate failure exits with code 2.
## Scripts
| script | purpose |
|---|---|
| `scripts/feat04b_audit_constants.py` | train-only audit for C/D: edges by the pre-set grid and merge rule, `LIKE_P0`, sidecar digest, `P_LIKE_REPORT_THRESHOLD`, the statistics-versus-logQ facts; stdout only |
| `scripts/feat04b_audit_launch.py` | read-only launcher that checks the deployed commit, tree and script hash before the audit runs |
| `scripts/feat04b_vram_smoke.py` | full-size synthetic memory smoke per real variant; declaration = max(1024, ceil(measured * 1.15 / 256) * 256) MiB with measured = max(mem_get_info delta, max reserved) |
| `scripts/feat04b_freeze.py` | two-stage mechanical freeze (`--write` / `--check`, `--stage 1|2`): append-only stage records, behaviour pins, plan-text check, audit-threshold rule, item_stats unchanged since the deployed commit, K10 + E4 for E, review verdict |
## Verify coverage (fail-closed)
- The binding includes `feature`, `side_sha256`, `weeks_sha256`, `bucket_spec_sha256`, `param_report`; C/D add `stats_sha256`, `stat_cutoff_week` and `stat_bucket_ids_sha256`. Any change in data or constants gives a binding mismatch.
- Not verified locally: bitwise GPU re-export of E's `(B, W, 6) @ (6,)` product and of the long-buffer gather (first CUDA rehearsal on the server).
