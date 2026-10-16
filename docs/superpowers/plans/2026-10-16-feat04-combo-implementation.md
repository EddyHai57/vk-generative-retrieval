# FEAT-04 Conditional Combination Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; execute inline with TDD and one final fresh-context review.

**Goal:** Deliver the registered F4-T010 + C + D + E combination for seeds42/2024/2025, with verified engineering gates and one verify per real train.
**Architecture:** Add new model/runner/config/operational files only. User vectors call the existing E implementation; candidates add the four unchanged C/D tables with supported masking. Reuse the existing data and stats sidecar without recomputation. Preserve all old files and frozen artifacts byte for byte.
**Tech Stack:** Existing Python3.11/PyTorch stack, unittest, detached finite server drivers.
**Spec:** Main decision_log 2026-10-16 night §4; existing feat04-design §3/§7; main next-phase plan §1.2. User explicitly requested executing this registered combination on 2026-10-16; no new scientific choice.

## Global Constraints
- Real configs: three fixed seeds, T0.10/logQ/L50/small init/lr0.001 and original100/5/3 budget; no extra arms, sweeps or altered thresholds.
- Stats reused exactly from audited train-only ITEM_STATS_V1; old C/D constants and D-1/C9 report limitations unchanged. C9 is report-only; U1/test excluded.
- All operational code pushed before server execution, fixed SHA deployment with per-file hashes. No overwrite/deletion of existing run/data/deploy paths.
- Read-only real preflight before new real-data execution; synthetic CUDA train+verify and actual shape VRAM probe precede queue submission.
- Dry-run, fresh claimed/rejected read; no retry failed attempt/rejected. Driver PPID1/900s/48h/no model calls, machine final receipt.

## Review Focus
- Candidate C/D terms must never flow into history events; user E path preserves bit order/Q2 support mask.
- Unsupported candidate and PAD vectors remain exactly zero even with nonzero stat tables.
- Replay binds all four immutable bucket IDs, source/config/stats/history hashes and refuses mutation.
- Both behavior-beta and all four stat diagnostics survive train/verify and best-epoch reporting.
- Config, freeze and submission reject unregistered feature/seed/budget or pre-existing run IDs.

### Task 1: Model and runner (FEAT04-COMBO)
- [ ] Tests first: formula, nonzero E equality, four-table candidate sum/mask, history isolation, gradients, immutable state, parameter counts, bad config/pins, diagnostic union, train/verify tamper rejection and preflight label independence. Run fresh RED.
- [ ] Create src/vkgr/retrieval_feat04combo.py and retrieval_feat04combo_run.py; old files untouched. Run GREEN and old FEAT-04b regression.
- [ ] Three frozen-only-seed-different configs; source pin/freeze generator and leakage source guards. Commit implementation.

### Task 2: Freeze, deploy and real engineering gates
- [ ] One whole-branch review; fix blockers RED→GREEN. Scan all newly public history/tree; equal-tree commit on existing public base, push work branch.
- [ ] Deploy new fixed-SHA worktree; preserve original p3c channel override and hash; per-file verification. Reuse existing pinned stats/input paths after read-only check.
- [ ] Real CPU read-only preflight all3; synthetic CUDA train+verify; real-shape synthetic-step VRAM probe, no metrics-based choice. Declare ceil(1.15*measured/256)*256, use existing P1024 floor.
- [ ] Freeze source/config/input/preflight pins. Dry-run. Tell Eddy before queue; submit three registered trains only if all gates pass. Read claimed and rejected reasons; finite server verifier automatically supplies3 verifies.
- [ ] Update STATE/schedule/experiment receipts and Kiro prompt with exact commit/preflight/queue status. No anchor/promotional decision.
