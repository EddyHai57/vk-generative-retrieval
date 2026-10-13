# Batch1 G switches (new version; frozen G1/G2/G3 contracts unchanged)

Design source: `docs/superpowers/specs/2026-10-14-experiment-cards-batch1.md` (main repo). This file only explains the code switches.

| Switch | Values | Meaning | Old behaviour |
|---|---|---|---|
| `history_len` | 20 / 50 / 100 | `HIST_V2_OBS_L*` sidecar (shared `history_v2.py`, byte-identical with the P branch) overlaid on the verified G3 context; examples/queries keep v1 identity | like history L50 |
| `init` | `small` | `INIT_EMB_N002`: token + position embeddings N(0, 0.02), PAD row 0. The forward keeps `embedding * sqrt(d_model)` | default N(0, 1) |
| `max_epochs`/`min_epochs`/`patience` | 100 / 5 / 3 | `BUDGET_E100_MIN5_P3`, same rule and shared case table as P | 30 / - / 3 |
| `sid` | `SID_BHKM_B16D3_S42` / `SID_RANDPERM42_OF_BHKM_B16D3` | G4 random-SID control: seed-42 permutation of leaf prefixes, leaf occupancy unchanged, suffix canonical by item id | original SID |
| beam diagnostic | 50 / 200 | decode-only, `kind=beam_diagnostic`, never selectable; formal beam stays 100 | - |

Code: `src/vkgr/generative_data_v2.py`, `generative_hp.py`, `generative_diagnostics.py`, `generative_b1_run.py`; configs `configs/batch1/g-*.json`.

- L<=50: model = frozen `generative_training.make_model` (+ small init); evaluation = frozen beam100 evaluator, or INFRA-01 sharded when `VKGR_EVAL_WORKERS>1` (needs `OMP_NUM_THREADS=1`).
- L=100: 400-row position table and a serial evaluator (`kind=l100_query_evaluation`); its `truncated_history` slice uses the last-50 view.
- Only `best.pt` / `last.pt` are kept. Verify recomputes every epoch aggregate from published rows, replays early stop, re-evaluates only the best checkpoint.
- Tests must run in the INFRA-01 frozen-base harness (design 6.1) with `OMP_NUM_THREADS=1`.
