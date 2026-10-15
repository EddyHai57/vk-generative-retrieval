密级: 公开

# P-HP-01 P switches (bounded single-knob tuning on the FEAT-02 F4 anchor)
Design source: the P-HP-01 plan and the frozen card (3A) in the main repo. This file only explains the code.
All P-HP-01 code is new files; no pre-existing tracked file is edited (`git diff --name-status <base>..HEAD` shows only `A`). Metrics live only in the experiment log.
## Anchor and knobs
- Anchor = F4 (`f3_f4_recency`): id_mean + N1 in-batch logQ (alpha=1) + F3 duration/ratio bucket embeddings + position bucket + week-gap bucket; config `configs/feat02/p-feat02-f4-recency-s42.json` (dim 64, T 0.15, LR 0.001). Its seed42 run already exists and is not re-run.
- Each H run changes exactly one coordinate; the other two keep the anchor value.

| knob | field | value | config |
|---|---|---|---|
| H1 | temperature | 0.10 | `configs/php01/p-php01-h1-t010-s42.json` |
| H2 | temperature | 0.25 | `configs/php01/p-php01-h2-t025-s42.json` |
| H3 | dim | 32 | `configs/php01/p-php01-h3-d32-s42.json` |
| H4 | dim | 128 | `configs/php01/p-php01-h4-d128-s42.json` |
| H5 | learning_rate | 0.002 | `configs/php01/p-php01-h5-lr002-s42.json` |
| H6 | learning_rate | 0.0005 | `configs/php01/p-php01-h6-lr0005-s42.json` |
| H7 | combo | conditional, generated | `configs/php01/p-php01-h7-combo-s42.json` (only after the trigger) |

## Model: `src/vkgr/retrieval_php01.py`
- `make_php01_model(content, supported, 50, dim)`, `dim` in `(32, 64, 128)`. All five tables (item id, T, R, position, week gap) have width `dim`; row counts, bucket edges and the F4 formula are imported from `retrieval_feat02` (`bucket_spec_sha256` unchanged). Init is N(0, 0.02) with PAD rows 0 for every dim (not scaled).
- The constructor repeats the id_mean branch of `TwoTower.__init__` instead of calling it (calling it would draw an extra 64-wide table). Draw order = FEAT-02: item, T, R, then P, G.
- Primary evidence: at dim=64 the model equals `make_feat02_model('f3_f4_recency')` at the same seed byte for byte (state_dict keys, dtypes, shapes, bytes, RNG state after construction, user/item vectors, one AdamW step) - `tests/test_php01_model.py`.
- `parameter_report_d` uses the FEAT-02 convention (item embedding excluded): dim 32 / 64 / 128 -> 928/800, 1856/1600, 3712/3200.
## Runner: `src/vkgr/retrieval_php01_run.py`
- Schema `p-php01-run-v1`, `task_id=P-HP-01` = F4 fields + `dim` (int) + `knob` (str). Every other field goes through the frozen FEAT-02 validator (temperature/LR projected back to the anchor values, because the frozen schema pins them). Types are strict (`64.0`, `True` rejected). `knob='anchor'` is synthetic-only (equivalence control).
- Batch tensors, train step, evaluation and diagnostics are the FEAT-02 functions; `run_session`/`verify_run` are copies with schema/kind, model, optimizer and binding changed. The anchor knob reproduces the FEAT-02 F4 runner bit for bit on synthetic data (records, selection, evaluations, checkpoints).
- Binding = FEAT-02 binding + `task=P-HP-01`, `knob`, `dim`, `param_report` at the run width. CLI: `synthetic-train`, `train`, `verify` (same inputs spec as FEAT-02 F4).
## H7 trigger and selection: `src/vkgr/retrieval_php01_select.py`
- Pre-registered rule, machine-readable in `configs/php01/p-php01-h7-combo-rule.json` and checked against the code constants at start.
- Winner: completed, verified H1-H6 seed42 run with delta R@50 >= 0.002 over the anchor. On n=2520 queries this is >= 6 more hits (5 hits = 0.001984 does not pass); compared on integer hits.
- Trigger: winners cover >= 2 different knobs (H1/H2 temperature, H3/H4 dim, H5/H6 learning rate); two levels of one knob count once. H7 = anchor values with each winning knob set to its best winning level (R@50 -> N@50 -> enumeration order); otherwise H7 is void.
- Selection: anchor + verified H runs (+ H7 once run) ranked by R@50 -> N@50 -> enumeration order (anchor first). Failed runs are reported, never winners, and their slot is not refunded.
- CLI `decide` (results spec `p-php01-results-v1` -> decision manifest `p-php01-combo-decision-v1` + generated H7 config) and `select` (re-derives the decision, checks the H7 run config equals the decided combo). Outputs are never overwritten. H4 results must be comparable to the anchor (inputs, rows, queries, catalog, support, side, weeks, bucket spec).
- Provenance chain: rule file (Git) -> decision manifest -> generated H7 config committed to Git as `configs/php01/p-php01-h7-combo-s42.json` in its own config-only commit, frozen again, then enqueued (no sha-only pin).
## H4 memory: `scripts/php01_vram_smoke.py`
- Synthetic full-size shapes only, no data or path argument, prints one JSON line. CUDA: peak allocated/reserved and the `mem_get_info` delta; declared VRAM = 3072 MiB if the measurement is <= 3072, else `ceil(measured * 1.2 / 256) * 256`. CPU: analytic parameter/optimizer bytes, VRAM `UNKNOWN`. The GPU measurement belongs to the deploy step.
## Not verified locally
- Bitwise GPU re-export for dim 32/128 (first server verify); the real inputs path (mocked in tests); H4 VRAM on the A6000.
