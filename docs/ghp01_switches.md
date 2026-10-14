# G-HP-01: bounded one-knob tuning on the paper-scale anchor A1

Status: code and configs only. No metric is reported here (metrics live in the experiment log).

## Anchor
A1 = `configs/garch01/g-garch01-a1-lr3e-4-s42.json`: encoder/decoder 4+4 layers, 6 heads x 64 (inner 384), d_model 128,
FF 1024, dropout 0.1, LR 3e-4, microbatch 256 (effective batch 512), `SID_BHKM_B16D3_S42`, `HIST_V2_OBS` L=50,
`BUDGET_E100_MIN5_P3`, `INIT_EMB_N002`, seed 42, beam 100.

A1 is used as the G-HP anchor only if its train run succeeded, a verify of exactly that manifest returned `verified` with the
same best epoch, and its best R@50 is at least 196 of 2520 validation hits (`scripts/ghp01_gate.py`). 196 is the first hit
count at or above 0.077397 (core03 anchor 190 hits + 0.002); 195 hits = 0.077381 fails.

## Knobs (one factor at a time, seed 42)
| id | run_id | changed fields | A1 -> G | params (V=80) |
|---|---|---|---|---|
| G1 | g-ghp01-g1-width2x-s42 | d_model, ff, head_dim | 128->256, 1024->2048, 64->128 (heads stay 6, inner 384->768) | 17,978,448 |
| G2 | g-ghp01-g2-lr1e-4-s42 | learning_rate | 3e-4 -> 1e-4 | 4,532,816 |
| G3 | g-ghp01-g3-lr6e-4-s42 | learning_rate | 3e-4 -> 6e-4 | 4,532,816 |
| G4 | g-ghp01-g4-layers2-s42 | encoder_layers, decoder_layers | 4+4 -> 2+2 | 2,289,744 |
| G5 | g-ghp01-g5-dropout0-s42 | dropout | 0.1 -> 0.0 | 4,532,816 |
| G6 | - | SID depth | 3 -> 4 | deferred (needs a depth-4 RQ-VAE fit); still counted within the 7 slots |
| G7 | - | combination | see below | not enqueued in this batch |

Every config differs from the A1 file only in `task_id` (`G-HP-01`) and its knob fields (pinned by a test).

G1 head handling: the original card value is "d_model x2, FF = 4d, heads unchanged". On core03 heads were fixed at 4 and the
per-head width followed d_model, i.e. the whole model (attention width included) doubled. Applied to A1 the same direction
doubles d_model, FF and attention width with 6 heads, so head_dim 64 -> 128 (inner/d_model stays 3, FF/d_model stays 8).
Alternatives rejected: 6 x 64 on d_model 256 widens only the residual stream (inner/d_model drops to 1.5), 12 x 64 changes
the head count. G4: A1 is already 4+4, so the layer knob goes down to 2+2. G5: the original direction was 0 -> 0.1; A1 is
already at 0.1, so the knob takes the other side, 0.

Unchanged in every config: heads 6, microbatch 256, FP32, AdamW wd 1e-4, clip 1.0, beam 100, init small, history v2 L50,
the epoch budget, seed 42, SID version.

## Code
- `src/vkgr/generative_hp_arch.py`: registry, thin factory over the GARCH-01 factory (same `TigerArchModel`), optimizer.
  Whitelist = width {A1, G1} x layers {4+4, 2+2} x dropout {0.1, 0} (8 structures) and LR {1e-4, 3e-4, 6e-4}, so a later
  combination needs no code change. What is actually submitted is fixed by the frozen config sha256s.
- `src/vkgr/generative_b1_hp.py`: opt-in entry `python -m vkgr.generative_b1_hp train|verify`, nested inside the GARCH-01
  entry. Default, batched and GARCH-01 entries are unchanged; runs of different entries do not verify each other (binding
  source = 20 files). Guards: eval workers must be 1, batched decode only with batch size 1, only train/verify.
- `scripts/ghp01_parity.py`: server preflight PH1 (new vs GARCH factory at A1 values, bitwise; reload of A1 best), PH2
  (batched b=1 vs serial decode for new structures G1/G4), PH3 (10 training steps per config, finite, params, VRAM).
- `scripts/ghp01_gate.py`: A1 entry gate. `scripts/ghp01_combo.py`: G7 trigger and card 1.5 selection.

## Rules in integer hits (2520 validation queries)
- Win: succeeded + verified + at least 6 hits above the anchor (= dR@50 >= 0.002; 5 hits = 0.001984 is not a win).
- G7 trigger: at least 2 distinct knobs with a winner. Knob groups: width {G1}, LR {G2, G3}, layers {G4}, dropout {G5},
  SID depth {G6}. G2 and G3 together count as one knob. Combination = A1 with each winning knob at its best level
  (R@50 -> N@50 -> enumeration order). Judged now on G1-G5 (provisional), again after G6.
- Selection (card 1.5): anchor + verified G runs, R@50 -> N@50 -> enumeration order anchor < G1 < ... < G7.

## Tests (CPU, frozen-base harness, OMP_NUM_THREADS=1)
Bitwise only, no tolerance: at A1 values the new factory equals the GARCH factory and a directly built `TigerArchModel`
(same seed, init small and legacy); dropout 0.1 equals A1 in train mode (logits, loss, gradients, 3 steps); dropout 0 turns
every dropout off (train == eval, RNG-independent, RNG untouched). Mutation tests M1-M16 cover every knob. Removing only
the G-HP runner's own `finally` is an equivalent mutant: the enclosing GARCH-01 runner restores the same names on exit.

## Limits
Single seed; 2520 queries; early-stop budget min 5 / patience 3 may stop deeper or slower runs early; G1 costs about 4x A1
compute and may approach the 24 h wall cap; the G1 head handling is a conversion choice; GPU equality (PH1-PH3) is checked
on the server before submission; positional encoding stays the anchor's absolute table (the paper uses relative bias).
