# M2 causal K4 execution plan

> Execute with isolated implementation and one whole-artifact independent code review. User's single-review requirement overrides redundant per-subtask reviews. Use TDD and verification-before-completion. Do not delete any workspace or old attempt.

**Goal:** implement the finalized M2 design and admit exactly three seeds with automatic verification after every engineering gate passes.
**Architecture:** new M2 model/runner reuse immutable M1-mean shared initialization, inputs and training budget; new exact scorer handles four user vectors. Versioned operations own immutable deployment/preflight/probe/plan/driver receipts.
**Tech:** existing Python3.11/PyTorch/stdlib unittest, existing scheduler, one A6000.
**Spec:** `2026-10-16-m2-multi-interest-draft.md` now FINAL; decision_log afternoon section2 verbatim is authoritative.

## Global constraints
- One arm, seeds42/2024/2025; no old checkpoint/test/ANN or embedding export. Original M1 files unchanged.
- Shared initialization bitwise equal to M1-mean; all additional draws follow shared draws. Zero-step loss difference <=1e-6.
- Steps1–100 detached target soft assignment/N1; 101 onward hard tie-smallest; hard101–200 coverage is diagnostic only, never a stop/retry.
- NaN/Inf in loss/grad/user vectors stops attempt; design deviations or design-changing HIGH/MEDIUM go to Kiro.
- Full-catalog exact max top50, stable existing tie-break. Branch/g0 diagnostics are descriptive, not promotion gates.
- One independent code review before freeze. Fixed Git source/push/per-file SHA/three real preflights/CUDA fullshape measured VRAM/dry/claimed+rejected proof before claiming acceptance.
- Server driver PPID1, >=900s, 48h/no LLM, no retry; raw verified + run/manifest identity required.

## Review focus
- Target assignment must never enter user encoding; target flip/causal/padding/OOV/unsupported and runtime test interception mutants pin this.
- Four-head weighted row loss must retain N1 masks/logQ/skips and initial equivalence; manual loss/gradient oracles and all W2 row gradients pin this.
- Exact K4 scorer/tie-break must agree with a simple full-catalog oracle, including overlap/ties; no ANN fallback.
- Warm→hard global step and diagnostic window cannot alter early stopping or omit a seed; unit + synthetic train/verify cover the boundary.
- Preflight source/data/config identity, numerical failures, startup metadata readability and independent raw verifier binding must fail closed; ops tests + real receipts cover these.

## Bounded work and file ownership
1. Science implementer owns only new `retrieval_seqm2.py`, `retrieval_seqm2_run.py`, `test_seqm2_*.py`, and three `configs/seqm2` files. Persistent report `/private/tmp/vkgr-m2-science-report-20261016.md`. No server access.
2. Parent owns new seqm2 ops/freeze/job-entry/CUDA/read-only scripts, their tests, design copy and governance. Only parent deploys/submits/updates facts.
3. Independent combo statistics worker owns two new analysis files in the separate tools worktree. Parent executes the read-only tool and logs its receipts, never changes F4 comparison or promotion rule.

## Interfaces
- `make_seqm2_model(content, supported_indices, max_len)`; `users(histories,side)` returns B×4×64.
- New runner `default_config`, `validate_config`, `_model`, `run_session`, `verify_run`, `evaluate_model`, `ablation_g0` follow M1 call conventions; CLI keeps --inputs/sha, --config/sha, --output, --manifest/sha.
- Ops reads real P base input and accepted same-seed M1 validation archives once; creates expected binding hashes for every config and checks them after training.
- CUDA proof must include zero-start receipt, soft/hard boundary, synthetic train/verified replay/g0, full512×50×4 training/optimizer and fullcatalog evaluation peak, allocator/process memory and declared ceil(measured×1.15/256)×256.

## Execution ledger
- [x] Read afternoon ruling, mark design FINAL and copy five rulings verbatim; no second design review.
- [x] Create native isolated worktree at public b089b8f, branch codex/seqm2-causal-k4; record base for whole review.
- [ ] Science fresh RED→GREEN, immutable M1 regression, numerical/leakage/manual oracle tests, synthetic full train→verify.
- [ ] Ops/freeze/CUDA/driver TDD and source/input metadata contracts; final startup record budget <=48KiB before scheduler's64KiB reader.
- [ ] One full independent code review with full diff package and reports; design-changing issue stops for Kiro.
- [ ] Freeze then public scan/push/new fixed-SHA deployment/per-file hashes; preserve original p3c data-channel artifact copy, never old deployment.
- [ ] Three real config preflights and same-seed M1 archive binding check; no checkpoint read.
- [ ] CUDA engineering and real-shape memory; dry all three, tell Eddy, submit once, independently confirm claimed/no rejected, start and prove finite server driver.
- [ ] Update STATE/schedule/handoff/experiment/issue receipts and Kiro table; separate code completion from pending science acceptance.

Combo McNemar/bootstrap against F4 and descriptive E runs independently. P anchor is M1-mean per Kiro afternoon; G5 freeze waits combo verdict and stable G binding. Other noon approvals are recorded but not automatic expansion of this task. Context counter UNKNOWN, no new timer or persistent model monitor.
