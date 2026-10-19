# Learnable SID read-only report implementation plan

Native execution under Eddy's explicit two-report authorization and AGENTS sections 4/4A/8. No new design approval or training.
Goal: aggregate S0 A-D/Dprime, original-prefix and popularity differences for accepted S2 versus fixed control.
Architecture: new narrowly scoped runner and queue transport, derived from reviewed structure-report guardrails; frozen original S0 math unchanged.
Spec: S2 FINAL plus decision_log latest two-card ruling.
Constraints: no test/checkpoint/model, no full mapping/prediction export; CPU1, VRAM0, finite wall; fixed exact six manifests/stdouts, train weeks 00-24 physical order; new exclusive run IDs. No cross-radix conversion or anchor decision.
Review focus: wrong SID source; runtime/delta mismatch; invalid history week/order; zero random denominator; prefix-group mean versus marginal code count.

- [ ] Tests first: adapt exact-run/source/runtime/job/payload guards, add hand-calculated global-codebook occupancy and original D/Dprime cases, exact 25 train-file list.
- [ ] Fresh RED assertion for absent new module; implement new analysis/sid_learn_report_read.py and scripts/sid_learn_report_queue.py only; old entry bytes unchanged.
- [ ] GREEN related tests, isolated assertion mutants of source/stdout/axes/means/history; freeze new member SHA card and commit.
- [ ] One independent whole prequeue review, fix by AGENTS 4A, public flattened scan/push and fixed-SHA fresh deploy.
- [ ] Real metadata-only engineering=False preflight, dry, unique queue admission, original aggregate stdout bound to job/request/source; no retry on science failure.
- [ ] Numbers to experiment_log, STATE/map/handoff status; Kiro reviews descriptive results only. S4 remains independent engineering pending.
