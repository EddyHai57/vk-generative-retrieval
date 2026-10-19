# S4 formal two-component wiring plan

Native execution of Eddy's existing S4 authorization; AGENTS 4/4A and frozen two-component decision plus same-day correction govern.
Goal: compose measured core and direct original usage costs, retain bounded BHKM and historical two-factor selfcheck.
Architecture: new scripts/sid_scale_composed.py and queue bridge; reuse unchanged cost_v1 synthetic generation/BHKM/supervisor and rq_components pure arithmetic. No old entry or L2 byte changes.
Files: new two scripts, tests/test_sid_scale_composed.py and queue tests, new fixed source card. Existing source files unchanged.

Global constraints: CPU1, no training slot, original run_id sid-scale-cost-v1, synthetic-only calibration, K256 D32 L3, core3000<source interval, usage once each at N120415/250000/300000; BHKM only250000/300000 <=3600s each; parent8970s within approved9000s; own-tree RSS16384MiB. No test/checkpoint/full mapping exports, no usage interval change.
Historical selfcheck: pin original manifest/tokenizer/terminal of T1r/T2r; actual fit interval from each tokenizer (caller uses usage_diag_every), device CUDA versus new CPU disclosed; actual job wall, not internal elapsed. Shared K256 calibration applied to historical K16 explicitly a model assumption subject to unchanged factor2 rejection. Any failed selfcheck makes all RQ extrapolations UNKNOWN and alerts Kiro. Historical predictions still composed_from_components, not measurement.
Review focus: contaminated core, missing/duplicated usage, partial stages/censoring, old single-core projection escaping, history source/params/time scope mismatch, source/runtime/queue binding and original budget.

- [ ] Fresh RED pure composition/interval/history-factor checks, then minimal GREEN.
- [ ] New bounded workers and parent; tiny engineering tests only locally, verify no old byte changes.
- [ ] New PF/queue/source card; isolated mutants and related regressions, one independent prequeue review.
- [ ] Public flattened scan0, push, fresh fixed-SHA deploy/real metadata preflight/dry/unique queue admission. Do not submit before all gates pass.
- [ ] Original aggregate receipt bound to source/request/run, component timings/R2/residuals/memory and two step scopes only experiment_log; UNKNOWN/ALERT rules unchanged.
