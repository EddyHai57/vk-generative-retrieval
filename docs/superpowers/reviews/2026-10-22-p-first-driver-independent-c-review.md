# P first-batch single-seed driver independent C review

Reviewer: short-lived independent Codex subagent; precise model ID UNKNOWN. Fixed original c616554b200bb613e7583040ef4d8d4f19135225, only 3 added files/diff/report and relevant original interfaces. No SSH, real-data reads, test reads, server submit, or training.

Initial verdict: BLOCKED pending original-queue disappearance fix. CPU control: 16 passed, `<PRIVATE_REVIEW_CACHE>/p-first-driver-independent-control.log`.

Independent counterexample: run_chain initially observes the original train job as pending. During real-prepare boundary the entry disappears, no run directory exists, and the immediate pre-write queue recheck returns absent. Code uses only current absent/run-not-exists and calls scheduler.submit, replaying an already observed original job. The documented fail-closed disappearance rule requires STOP without submit. Sentinel execution independently observed submissions=['train']; `<PRIVATE_REVIEW_CACHE>/p-first-driver-independent-attacks.json`. All backend/complete boundaries are synthetic; scheduler is a no-write local recording sentinel. Author and owner notified.

Other inspected properties: independent driver/consumer source identities, original native config and 96-hour wall, original helper/DAG receipt closure, plan to original prepare and exact job comparison, rejected queue refusal, >=900-second poll and <=200h finite deadline, required PPID1, claimed confirmation before stage completion, no scheduler tick or original-consumer edit. No completion claim pending correction and independent re-review.

## Fixed-source re-review: b4c9ded73183cfb889361b2fb78c1770b1edc4fd

Final independent C verdict: PASS for the single-seed mechanical driver within existing owner/scientific admission. This review grants no new training authorization or budget.

- Narrow fix records whether original queue/run was present before prepare; if both disappear before the only write, it stops rather than resubmitting.
- Original independent sentinel rerun now observes `submissions=[]` and `observed original queue/run disappeared; no resubmit`. `<PRIVATE_REVIEW_CACHE>/p-first-driver-independent-fixed-attacks.json`. Original failed counterexample remains preserved.
- Independent driver regression: 17 passed, 1.06s; includes original actual backend-plan/launcher.prepare/scheduler.submit synthetic pending chain with launched=false. `<PRIVATE_REVIEW_CACHE>/p-first-driver-independent-fixed-control.log`.
- Independent mutations: 5/5 killed by target assertion (complete job check, rejection gate, final DAG closure, poll>=900, explicit driver SHA). Each fresh complete fixed Git tree, python -B, PYTHONDONTWRITEBYTECODE=1, unique PYTHONPYCACHEPREFIX. Old native reference fixture explicitly reads its fixed Git object from P_FULL_TEST_GIT_ROOT; mutated driver code imports only new copied tree. `<PRIVATE_REVIEW_CACHE>/p-first-driver-independent-mutations-valid-b4c9ded.json`.
- First mutation pass omitted native Git fixture root, so final-DAG case errored before assertion. That run is retained and excluded; corrected evidence above all reaches intended target assertions. Pytest `DID NOT RAISE` is counted as a killed assertion, alongside AssertionError.
- Independent driver and original consumer identities remain separated; consumer bytes, existing preflight, scientific implementation/config/seed, 96h train wall and scheduler are unchanged by this increment. No backend RAM or other uncommitted files were reviewed/included.

Limitations: synthetic local review does not replace owner checks of real producer receipts, source/layout/runtime, resource measurement or conditional scientific admission. Existing claimed jobs remain untouched. No SSH, server job, real-data, validation/test or checkpoint read occurred.
