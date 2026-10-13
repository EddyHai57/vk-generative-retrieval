# Batch1 P switches (new version; frozen P4D/P4E/P4H unchanged)

Source of truth for the experiment design: `docs/superpowers/specs/2026-10-14-experiment-cards-batch1.md` (main repo). This file only explains the code switches.

| Switch | Values | Meaning | Old behaviour |
|---|---|---|---|
| `history_len` | 20 / 50 / 100 | `HIST_V2_OBS_L*`: every train interaction row, physical order, strictly before target, duplicates kept (P4E `build_watch` definition, window parameterised) | P4E `watch_history` = L50 |
| `init` | `legacy` / `small` | `small` = `INIT_EMB_N002`: every `nn.Embedding` N(0, 0.02), PAD row 0; other layers untouched | `legacy` = PyTorch default N(0, 1), value-equal to the frozen models |
| `max_epochs`/`min_epochs`/`patience` | 100 / 5 / 3 | `BUDGET_E100_MIN5_P3`: bad epochs only counted after `min_epochs`; ties are bad; best over all epochs | 30 / 0 / 3 (`P-REPLAY` only) |
| `logq` | bool, alpha fixed 1.0 | subtract log(expected sample count) per column; in-batch only = log(count/N) | off |
| `extra_uniform` | 0 / 128 | 128 distinct uniform negatives from `SUPPORT_POOL_V1` minus batch targets | 0 |
| `checkpoint_policy` | `best_last` | only `best.pt` and `last.pt` (atomic replace) | P4E kept every 50 steps |

Code: `src/vkgr/history_v2.py` (shared byte-identical with the G branch), `retrieval_hp.py`, `retrieval_negatives.py`, `retrieval_b1_run.py`; configs `configs/batch1/p-*.json`.

Verify (`retrieval_b1_run verify`) is fail-closed: replays the cursor/negative draws and skip counts, recomputes every epoch's metrics from stored predictions, replays the early stop, re-exports only the best checkpoint and checks the last state. `zero_user_vector_fraction` and `history_supported_fraction` are recorded because id_mean only sees supported history items.

`P-REPLAY` (`init=legacy`, 30/0/3, L50) must equal P4E `watch_history` seed42 epoch by epoch: `retrieval_b1_run replay-compare`.
