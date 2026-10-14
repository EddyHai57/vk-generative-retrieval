# GFEAT-01 G switches (new entry only; default and INFRA-02 paths unchanged)
Design source: the FEAT-01 experiment card (main repo, `docs/superpowers/specs/2026-10-14-feat01-card.md`). The feature definitions (feedback bit order, timespent / watch-ratio bucket edges, bucket counts) are NOT defined here: they live in `history_side.py`, `retrieval_features.py` (+ `retrieval_hp.py`, `retrieval_sasrec.py`), copied byte-for-byte from the P branch `feat01/p` @ `d70a993a800a`, together with the shared `history_v2.py`. Tests pin their sha256 on both lines. This file only explains the G code switches.

| Switch (`feature`) | Meaning | New parameters (d_model = 128) |
|---|---|---|
| `none` | control: new entry, no side data, no side arguments. Reproduces the anchor's `model_sha256`, per-epoch `computation_sha256`, losses and selection | 0 |
| `f2_feedback` | per history item: sum of the item's feedback-bit embeddings (like, share, bookmark, open_comments, click_on_author, dislike; multi-hot, 6 x d table) | 768 |
| `f3_watch_buckets` | per history item: timespent-bucket embedding (7 buckets + PAD) + watch-ratio-bucket embedding (5 buckets + PAD) | 1792 (1536 trainable; PAD rows are fixed zeros) |

- Where it enters: `x = (E[token] + side_vector(item)) * sqrt(d_model) + position`, encoder input only, decoder unchanged. The item's side vector is broadcast onto all four SID tokens of that item; PAD items get exactly 0.
- Initialisation: side tables are created after the anchor's initialisation (same random stream), N(0, 0.02), PAD rows 0, so every base parameter equals the anchor's under the same seed.
- Data: the `HIST_V2_SIDE_V1` sidecar (`[feedback_mask, timespent, duration]` per history event, strictly before the target, train weeks only), bound to the history-v2 prepare by sha256. The pack is checked row by row against the training context (example ids, histories, lengths), value ranges, and that an item has one duration (a row mix-up shows up as one item with two durations). Side data lives outside `RunContext.queries/examples`, so every `query_sha256` is the anchor's.
- Entry: `python -m vkgr.generative_b1_feat train|verify` (adds `--side-manifest/--side-sha`; required for f2/f3, forbidden for `none`). Training: `VKGR_DECODE=batched VKGR_DECODE_BATCH=1`; verify: decode switches unset (query-by-query decode).
- Registered: `history_len=50`, `SID_BHKM_B16D3_S42`, `init=small`, one evaluation worker, decode batch size 1 only. A decoder call that mixes several queries is not bit-equal to the serial decode (INFRA-02b), so it is rejected rather than used.
- `binding.source` of runs from this entry covers the added modules and `binding.feat01` records the feature; runs trained here verify only here (fail-closed binding mismatch), and vice versa.
Code: `src/vkgr/generative_feat.py`, `generative_feat_eval.py`, `generative_b1_feat.py`; configs `configs/gfeat01/g-feat01-*.json` (differ from `configs/batch1/g-hist-v2-l50-s42.json` only in `schema`, `task_id`, `feature`, `side_version`).
- Tests (`tests/test_generative_feat.py`, `tests/test_generative_b1_feat.py`) run in the INFRA-01 frozen-base harness (design 6.1) with `OMP_NUM_THREADS=1`.
