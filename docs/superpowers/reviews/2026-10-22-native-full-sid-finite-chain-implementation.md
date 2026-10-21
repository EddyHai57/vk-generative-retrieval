# Full B → 原 fullSID 有限接续：实现报告

范围：仅新增 `ops/tools/remote/native_full_sid_finite_chain.py`、对应测试和本报告。未修改原科学模块、fullB 驱动、scheduler 或事实源；未 push、部署、启动服务器作业，task passes 未改。交 owner 派新独立 C review 后再决定发布。

## 固定调用链与边界

- 原 fullB source `9fac715df389761a4869e5a7118977d802f01c1e`，原 fullSID source `aa8c0910e407e92d757b91eae64d3ce3137cb027`。
- driver 要求 PPID=1、poll=900 秒、有限截止至多 200 小时。固定 driver/template SHA，每轮核原 source 及实际文件集合；忽略文件也不得额外存在。逐块 hash、Git 子命令、原 CLI、submit 前后核剩余截止，子进程 timeout 使用剩余预算。
- 等原 fullB 自然成功且 claimed、不在 rejected。原正式 `analysis.sgns_native_full verify --stage fit` 完整闭包通过后，对原 request、fit result、run/terminal/stdout 机械算 SHA。只回填 owner 模板内 producer request/receipt/proof；科学内容、source、run_id、output、资源原样保留。
- 对 fullSID preflight → fit → verify：原正式 CLI plan、真实 launcher.prepare、完整 transport_job 检查 argv/config/ordered inputs/env/source/resources/OPS；每 stage submit 一次；pending 只等原 scheduler，成功必须 claimed 并经原 accepted 闭包。最后原 consume；不新增消费判定或替换 parent 校验。
- queue rejected、终态失败、重复队列、作业消失、截止、字节/绑定差异一律 ALERT、独占 summary；无重投、无 scheduler tick、无科学 G 模型训练或 test 读取。
- driver evidence、SID output 与 template、原 request/lock/receipt、两消费者源码树和 OPS 互斥；新 request/stdout/stderr/summary 独占写入。

## 资源单位更正

Owner 初始建议 525312 MiB 源于把原规划当作约 513 GiB；原文更正为 `PLAN_BYTES=513000000000`、`PLAN_MIB=489234`。初始建议不采用。原 fullSID RSS 线仍是该规划值的 60%，CPU K=229/depth=3、H/R、fit-verify、replay 均未修改。

实现没有默认资源：owner 模板必须显式给 CPU/RAM/disk、VRAM=0、wall=48 小时。Owner 提出的 368640 MiB 仅是首次保守 `not_measured` 候选，**本次不登记正式资源**；正式入队仍由原 admission 结合实时 20% 余量及已有预约决定。fit 不冒充已有实测。

## 验证与限制

- Fresh RED：新增模块缺失导致测试导入失败；第一次系统 Python 无 pytest，记录保留；改用项目 Python 获得预期 RED。
- 最终相关回归：85 passed（新增驱动 38 项，原 A1/P 驱动及原 fullSID 47 项）。原测试入口首次遗漏 PYTHONPATH 得到 16 import failure；仅修测试环境为 src 后全部通过，未修改生产依赖。
- 真实 launcher synthetic E2E：隔离 clone 原 OPS、构造并提交测试专用 synthetic SID 模块，实际 launcher.prepare → CHECK_PLAN subprocess 与 dry；argv、ordered inputs、env、seed、RAM 五类漂移均拒绝。未 mock launcher。该 fixture 不代表真实数据 PF 或完整真实科学调用验收。
- 隔离变异 6/6 killed（exit 1、非收集错误）：删逐块 deadline、跳实际 file set、接受零 proof、跳 lineage、删 transport_job、跳 PPID。每项单独新 clone、唯一缓存前缀、python -B、PYTHONDONTWRITEBYTECODE=1；未复用作者缓存。
- 本地证据由 owner 私有映射保存：`full-sid-chain-red-v2.log`、`full-sid-chain-final-tests.log`、`full-sid-chain-mutants.json` 及其逐项日志；不在公开文档写本机绝对路径。
- 尚未独立 C 审、尚未真实部署/PF、尚未生成 owner 正式 template，未声明 fullSID 或 G backend 通过。

## Independent C blockers and transport correction (after eb4493d)

Independent C blocked the first candidate for two concrete gaps. (1) Another same-ID queue entry could appear while plan/plan-check ran, yet the driver would still submit. (2) The driver did not check the actual fixed OPS tree; the original OPS loader checks tracked bytes but accepts ignored bytecode, and `python -B` prevents writing rather than reading `.pyc`. The reviewer demonstrated a harmless marker in valid ignored launcher bytecode executing despite clean tracked Git bytes. The original blocked candidate must not be published.

Both reviewer controls reproduced fresh RED (2 fail). Corrections remain within the three new files:

- After plan-check, inspect all pending/claimed/rejected entries, own run and stage output again. Newly pending/claimed means observe it and wait, never submit an extra entry. A seen/reserved job disappearing remains ALERT. Unqueued run/output or rejected/duplicate queue remains ALERT.
- `execute` also has a final submit-only guard after its fixed-tree/hash checks and immediately before the scheduler subprocess, closing the longer hashing interval. If work appears there it returns without spawning or creating a submit receipt; callers then observe the original queue. This does not assert a new cross-process scheduler locking guarantee.
- `fixed` now checks OPS source `5840e6b686ec9651314a9e28a1a2c0befcc05158` with the same exact actual regular-file set and every tracked byte as driver/fullB/fullSID, including symlink and ignored-extra rejection, before each child process. Original OPS source and loader remain unchanged.

Validation: all seven reviewer controls pass, including the two original blocking cases and the valid ignored-bytecode demonstration. New local tests cover pending/claimed/rejected/run/output appearing during final fixed checks, actual ignored `.pyc` rejection, and persist both reviewer blocking controls. Final regression includes original fullB/SID and original A1/P drivers; detailed counts and isolated mutation receipt are recorded by owner in the task evidence. No server, push, deploy, scheduler tick, scientific input/config or task passes change occurred. This increment requires a new independent C review; author self-validation does not close the blockers by itself.

Final C-correction evidence: **121 passed** (46 author driver tests, 68 related original A1/P/fullB/fullSID tests, 7 reviewer controls). **9/9 isolated mutants killed**, including the original six plus omitting OPS-tree verification, omitting the post-plan queue check, and omitting the final submit guard. Private evidence: `full-sid-c-r1-red.log`, `full-sid-c-r1-green-v2.log`, `full-sid-c-r1-final-regression-v2.log`, `full-sid-c-r1-mutants.json`. First correction attempt made an unqueued failed terminal report the earlier run/output guard rather than its original failure reason; the check order was restored, and the failed test log was retained before the final GREEN. All mutations used independent clones, unique cache prefixes, disabled pyc and pytest cache; failures were test assertions/controls, not import/collection errors.

## Isolated clean OPS execution directory (after fd000c58)

The reviewed code was flattened and published as `858b0b0c2584d244ad469c38c2cee9b9cb679f26`; its 143 tracked deployment files matched SHA256, and its flattened driver suite passed 46 tests. Startup readonly `fixed` then correctly rejected the original active OPS actual file set: six ignored Python bytecode files existed. The driver was **not started** and no SID job was submitted. Original OPS bytes, bytecode, scheduler, jobs, template and failed readonly receipt are preserved; no deletion or source exemption was used.

Owner authorized a minimal transport correction. A separate clean worktree of the same original OPS `5840e6b686ec9651314a9e28a1a2c0befcc05158` was deployed at `/workspace/vkgr-native-full-sid-ops-code-v1`, with all 95 tracked SHA256 values matching. The new driver pins this path for template, source-tree checks and original plan/launcher. The imported original `approved_input_chains.submit_argv` has its own old hardcoded executable path, so merely changing the driver constant would still execute the old OPS. A local wrapper now checks all six original prefix fields exactly, then replaces only argv[2] with the clean OPS scheduler executable. Every other argv byte, the shared OPS_ROOT, source/runtime/metadata checks, queue guards and deadline rules stay unchanged. No original helper, OPS or science file was modified.

This reuses the existing fixed Git worktree deployment and original scheduler CLI; it introduces neither a new scheduler nor a scientific configuration. Original fullSID accepts an absolute nonsymlink OPS location and authenticates the same source; its metadata identifies the unchanged OPS source. Original fullB readonly verify does not load an OPS module. Request/plan/submit therefore use the clean executable tree while existing queued jobs and the active original scheduler keep their old installation. The future owner template must explicitly use the new clean path; the first immutable template is retained and is not reused or overwritten.

Fresh RED: seven targeted tests failed for the old path and missing checked wrapper. Minimal GREEN: 54 driver tests passed. Final regression: **142 passed in 53.38s** across driver/original A1/P/fullSID/fullB and the prior eight extra controls; original seven reviewer controls separately **7 passed in 2.01s**. The real launcher synthetic fixture now checks request/plan and relocated submit use the same clean OPS path and all remaining argv fields match the original helper. New tests reject each of six original-prefix drifts, preserve an old-tree ignored-bytecode marker, and reject an ignored extra in the clean tree. These remain synthetic controls, not real fullSID PF acceptance.

Self-validation isolated mutants: **13/13 killed**, each fresh clone plus exact three-file candidate overlay, unique cache prefix, `python -B`, disabled pyc/pytest cache. Original nine controls were rerun, plus reverting the OPS path, omitting the prefix gate, keeping the old executable, and altering an unrelated argv tail. All failures were assertions/control failures, not syntax/import/collection errors. Private evidence filenames: `clean-ops-red-private.log`, `clean-ops-green-private.log`, `clean-ops-regression-private.log`, `clean-ops-prior-seven-private.log`, `clean-ops-mutations-private.json` and individual mutant logs under the owner's deployment evidence cache.

This correction is **C class** because it changes the bound transport execution path. It is committed locally for a different independent reviewer; it has not been published, deployed as a new driver, or started. The clean OPS clone alone was created without touching the active installation. Original fullB/A1 inputs are still pending; no upstream pin was invented and no G model training, task pass or scientific acceptance is claimed.
