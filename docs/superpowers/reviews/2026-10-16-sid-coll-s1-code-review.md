# SID-COLL S1 唯一完整独立 code review

结论：**CHANGES_REQUESTED**。发现 1 个 P1 运维中断问题、1 个 P2 交付汇总缺项；两项都可在普通 fresh RED→GREEN→回归工程修复中解决，**不需要改变科学设计、不请求新一轮 design/code review**。

Reviewed base：`c1c4b7413b4ae794894fc78b98828ab26029a487`。
Reviewed HEAD：`61a039fc8e350d3eecb602a68aaf405e0f51d49b`（包含最后 aux 路由补丁）。
Context 当前可见计数：UNKNOWN。Review 为只读：无 SSH、无 push、无提交作业、无源文件修改；本报告是唯一持久新增文件。测试自己的 temporary fixtures 按原测试正常创建/清理。

## Findings

### [P1] 为 controller 汇总使用 48 KiB queue 限额会中断 driver

位置：`scripts/sid_coll_ops.py:343`（另见 `:59`、`:162`、`:345`）。

`inspect_stage` 把每一阶段的完整 `binding.data` 与 `binding.resolved_from` 都复制进 `tick` 返回值。两臂各有 14 阶段，这些重复证据会使 controller summary 超过 48 KiB。driver 在每一次 tick 后无条件使用 `new()` 写 summary，而 `new()` 在 `len(payload)>=49152` 时直接抛 `ValueError`。异常不在 driver 循环内处理，driver 因而退出，既不留下正常 final summary，也无法继续推进尚未提交的 verify/补 seed。该错误与 GPU、数据、模型训练是否正常无关。

独立复现使用真实 `sid_coll_plan.build_documents`、真实 `sid_coll_ops.inspect_stage/tick`，只 mock 只读 metadata/queue/terminal filesystem；为全部阶段填入真实 entry 的 receipt/binding 结构及合法 SHA、两臂 verified42 hits=260。输出：

```
REAL_TICK_STATUS complete BRANCHES {'C1p': 'complete', 'C2': 'complete'} SUMMARY_BYTES 65687
DRIVER_WRITE_FAILURE ValueError('output metadata must be <48KiB')
```

`new('<LOCAL_TMP>/sid-coll-review-summary-never-created.json', result)` 在打开文件前即拒绝，未创建该文件。这是对真实 assembler/controller 接口的检查，不依赖既有简化 ops fixture。

当两臂进入补 seed 后半程会越界；若其中一臂先完成，另一臂推进时也会越界。两臂均未达到筛选门槛时未必触发。因此现有只测单步/简化 receipt 的 tests 全绿不能覆盖它。

修复方向：给 controller summary 独立的有界大小合同（例如 2 MiB），或精简重复 binding、保留其 hash 与原 receipt ref。queue/job/launcher/supervisor 的 48 KiB 合同应保持。加入真实大小 binding 的双臂全流程回归，至少覆盖 driver 周期 summary、final summary、control tick/status 输出。

科学设计影响：**无**。

### [P2] 终态汇总丢失 A–D/D′ 和 prediction hash 的类型

位置：`scripts/sid_coll_ops.py:160-162`；producer 位置 `scripts/sid_coll_job_entry.py:343-351`、`:383-396`。

fit-verify 把 A/D/Dprime 写在 `mapping-diagnostics.json`，G verify 把 B/C/C_delta_vs_T3 写在 `prediction-diagnostics.json`；receipt 只在 `proofs` 中引用这些 JSON。controller 的 `inspect_stage` 不读取或保留这些 proof refs，只从 receipt 顶层 `summary`（或整个 receipt）抽取固定字段。故实际成功终态不会包含 A/B/C/D/Dprime，也没有明确 UNKNOWN，无法满足夜间授权 §5 的机器终态汇总要求。数据虽然仍在服务器文件里，自动交付产物却遗漏它们。

同一过滤器只保留含混的 `prediction_sha256`，丢弃 entry 已正确提供的 `prediction_sha256_kind`、`prediction_computation_sha256` 和 `prediction_manifest_file_sha256`，使 summary 失去 computation digest 与物理文件 SHA 的区分。原 entry 本身没有把两种 SHA 冒充为同一个。

证据：阅读真实 entry writer 与 controller whitelist，并在上面的真实 tick 合成完成场景核对 summary 结构。现有 ops fixtures 只把 hits 等直接置于 receipt 顶层，无法检查诊断 JSON 的 producer/controller 连接。

修复方向：从已验收 receipt 的诊断 proof 引用出发，按物理 SHA 读取有界小 JSON，将必要的 A/D/Dprime 与每 seed 的 B/C/差分放入 summary，保留引用与 SHA；未产生的诊断明确 UNKNOWN。保留明确命名的 computation/file SHA 字段或 kind。无需重算诊断、改变阈值或扩大输入读取范围。

科学设计影响：**无**。

## 已完成的独立检查

- 读 FINAL 冻结卡、execution plan 与主仓 decision_log 夜间 SID-FIX §4/§5；review 未改变 SGNS/C2/SID/G 科学超参、筛选阈值或 checkpoint 许可。
- 最终 HEAD 自行运行 S1 全部 unittest：**63 tests PASS，4.312s**。命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. <PYTHON> -m unittest discover -s tests -p 'test_sid_coll*.py'`。
- 自行运行 supplemental analysis unittest：**12 tests PASS，0.011s**，`-m unittest discover -s analysis -p 'test_sid*.py'`。
- 实际读取本机 infra03 `vkgr_ops.scheduler.submit(root,spec)` 与 `launcher.prepare(spec)` 源码，核实 queue schema、参数顺序、env list、prepare/run-dir/out-dir 行为；没有只凭 mock 声明接口兼容。
- 核对原 teacher commit `0ca61bc35ad68bd79b86103b54e470445cbab07f` 的 `make_seqm1_model` API、原 binding/digest/source ledger 语义；export 子进程只按物品索引调用 `items()`，使用 `weights_only=True`。未读取任何真实 checkpoint。
- 检查 SGNS 求和 loss、连续 negative RNG/reject、频次^0.75、固定 user/context 顺序、5 epoch LR、fallback/非有限 stop 与严格健康检查。
- 检查 C2 分块归一化、mask/zero/NaN、固定 teacher lock、源包 hash、64/128 新 SID schema、旧 loader 不变、BHKM clone 与 H5a/H5b/replay 链。
- 核对 new source root 与 old data channel 分离、三 seed 配置只差 seed、beam=100、动态 refs/expected statuses、own-new-checkpoint verify、query 顺序及 computation digest 复算接口。
- 核对 259/260 筛选、独立分支失败隔离、sticky ALERT/rejected 不重投、唯一 intent、PPID1/inotify/900s/固定48h流程，以及新增 aux-dry/aux-submit whitelist 和默认 stdin-source readonly 路由。
- Review 结束时源 worktree `git status --short` 为空。

## 验证边界

Local 为 Torch CPU runtime；没有实际 CUDA、正式 120415 item 数据、远端 scheduler 或服务器 supervisor 的执行验证。真实数据 event-order 证明、teacher best/source-lock 实际元数据、正式三 seed preflight、CUDA shape/显存、实际 launch record/supervisor 可读性和 claimed/no rejected 仍必须按已授权部署阶段检查，不能把此次 CPU unittest PASS 当成这些 gate 已通过。

上述两项修复由父 owner 自测关闭即可，不重复进行独立 review。本报告结论只适用于所列 reviewed HEAD；没有声称尚未执行的工程修复已通过。

## 执行owner关闭（独立review后，未追加review轮）
P1/P2均为工程finding，无科学设计更改。fresh RED观察到两个新API缺失及遗漏run列表；GREEN关闭：summary专用2MiB有界writer，queue/job/launcher仍48KiB；按物理SHA验读A/D/Dprime及B/C，保留诊断refs、computation/fileSHA与kind；所有未验收/被筛掉run明确列出。追加真实driver双支全阶段>48KiB周期及终态回归，均正常完成；总67项S1测试＋12项analysis测试通过。原CHANGES_REQUESTED保留为review历史，owner依项目普通工程fix自测关闭，不冒称reviewer再次批准。实际服务器门禁仍待部署验证。
