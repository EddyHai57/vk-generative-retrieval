# 新 SID 的 A1 确认：正式作业前独立 code review

密级：公开。2026-10-17。结论：**CHANGES_REQUESTED**。context UNKNOWN。

审查范围：`codex/coll-sid-a1-confirm`，base `ddf208e9b70a4c6c180d6dd1c600a464cf8958a2` → HEAD `34a20007f8e7f08f72d7f7bfff6bd378229abd0b`，10 个全新增文件。依据当前 AGENTS、decision_log 午间/午后/无人值守授权、S5 FINAL 卡与实现交接。独立 reviewer 未实现代码、连接服务器、部署、push、读取旧 checkpoint/test 或提交科学作业。

## 必须关闭的发现

### MEDIUM R1：source gate 未验证 worktree 干净，也未将请求中的文件 SHA 绑定到 Git blob

位置：`scripts/coll_sid_a1_ops.py:110–128`，特别是 112–120。

`identity()` 只核 HEAD 等于请求的 `source_commit`，并核工作目录文件 SHA 等于请求的 `files`。它没有检查 `git status --porcelain`，也没有核 `git show <source_commit>:<member>` 的 SHA。因此同一个 HEAD 下未提交的源码，只要请求记录其当前 SHA，也会被当作固定 commit 的源码接纳；额外 dirty 文件也不被拒绝。该函数用于 preflight、dry-run、submit 与 detached driver 的 source gate，缺口贯穿正式路径。`run._source()` 记录 `git_dirty`，但记录不等于拒绝，不补上此门禁。

独立探针：以实际全部 `G6_MEMBERS` 与新 ops 文件的 SHA 组装请求，仅 mock `subprocess.check_output` 返回指定 HEAD，调用 `identity()` 成功；记录到的全部 Git 调用只有 `rev-parse HEAD`，没有 status 或 blob 核验。这是直接验证缺失的检查，不把 synthetic fixture 的更早拒绝算作成功。

修复与验收：在共用 source gate 拒绝 dirty worktree，并把每个请求文件 SHA 与指定 commit 的 blob 校验。添加行为测试：clean/正确 blob 通过；dirty 失败；工作文件 SHA 与请求一致、但与 commit blob 不一致仍失败；正式发布前检查仍触发。不得修改旧入口。**纯工程修复，不需改科学设计。**

### MEDIUM R2：dry-run 丢弃 prepared record，遗漏最终启动 JSON 运载尺寸检查

位置：`scripts/coll_sid_a1_ops.py:347–350,370`；同一 helper 还在 410、471 用于正式 train/verify 发布前。

`launcher.prepare(job)` 的结果没有被校验或保存；`dry_run()` 只要调用不抛异常就返回 `dry_run_ok`。这不能验证 supervisor 实际读取的 `{spec, prepared}` 组合 JSON 是否可运载。本项目旧 metadata 事故对应的已有工程检查位于 `scripts/sid_coll_control.py:138–139`：对完整组合的 JSON 编码长度核验，并预留 envelope 余量。新增入口没有这一步，可能让 dry 通过、正式发布后才因元数据过大失败，消耗不可重跑的科学身份。

独立探针：在 `/private/tmp` 下构造匹配的三配置 preflight/CUDA receipts（完整 own-PID/context memory 样本、measured=1500 MiB、declared=1792 MiB）；mock identity 与配置派生，令 `launcher_prepare` 返回 JSON 编码 **65552 bytes** 的 prepared record。实际调用 `dry_run()`，三次 prepare 后返回 **`dry_run_ok`**。既有 control 的运载检查阈值为 `48*1024-2048 = 47104 bytes`；该探针明确越界，仍被接纳。没有访问 scheduler 或真实数据。

修复与验收：共用 helper 校验真实最终 `{spec, prepared}` JSON 的运载大小及可编码/可读取性，train 与 verify 发布前复用；采用实际 supervisor 的既定 envelope 与余量，不臆造新科学门槛。行为测试覆盖边界内、边界上/越界、无法编码的 metadata，证明 dry 和正式发布前均拒绝且 queue 不产生新记录；verify 路径也覆盖。**纯工程修复，不需改科学设计。**

## 已核内容与验证证据

- 三配置与同 seed 原 A1+T3 仅 schema/task/SID 不同；逐字段值/类型冻结。新训练 while 与原 G6 的训练/selection while AST 相同，verify 的 epoch selection replay AST 相同；model factory 与 train batch 为同一原函数。
- COLL loader 沿原正式 `engineering=False` 家族合同复用，SID/input/vector 三个固定文件 SHA、全 bundle artifact SHA、CONTENT_V1/H5a/H5b、codec 771/suffix1 均有检查；depth3 适配保留 digits。原 fit receipt 与原 stdout raw verified/receipt 指针有独立绑定门禁。
- raw config 文件 SHA 与 manifest canonical config digest 分开；preflight 提供三 seed expected binding，driver 比完整 manifest binding；每成功 train 派生唯一新 manifest verify。旧 manifest/CP schema 拒绝，CPU synthetic 自有新 checkpoint 完整 train→verify 及篡改拒绝已运行。
- CUDA helper 明确仅 full-A1 shape 的工程两步、beam100/b1、自有内存 checkpoint replay 与 own-PID/context-inclusive 峰值证据；未将它冒称服务器完整 runner/科学 verify。服务器真实输入 preflight/CUDA/科学作业均未由 reviewer 执行。
- detached ready/release、PPID1/session、900 秒轮询、48h 截止、发布 intent、失败 latch 与不重投路径已检查，detached 本机故障测试实际通过。不会自动修改 anchor。
- base→HEAD 全部为新增，旧生产模块无字节改动；未发现必须改变模型、数据、selection、beam 或科学设计的偏离。

独立运行命令（`<PYTHON>` 为 root 指定的项目 Python，线程均为 1，`PYTHONPATH=src:tests`，在 author worktree）：

```sh
<PYTHON> -B -m unittest test_coll_sid_a1_run test_coll_sid_a1_ops test_coll_sid_a1_submission test_coll_sid_a1_mutations -v
```

结果：**25 tests，2.599s，OK**。含 4 个行为 mutants、`/private/tmp` 非 symlink fixture 的错误 schema/CP hash 拒绝、完整 synthetic runner 与三个真实 detached 本机故障场景。

扩展相关回归另实际运行：

```sh
<PYTHON> -B -m unittest test_generative_g6_unit test_generative_g6_v2 test_sid_coll_contract test_sid_coll_g test_sid_coll_cuda_check -v
```

结果：**49 tests，3.331s，36 通过、13 ERROR**。13 项均为旧 `test_generative_g6_v2` 在 fixture 初始化时，经 `test_generative_g6.Fixture → engineering_world → prepare_engineering → retrieval_data.checked` 抛 `ValueError: input hash mismatch`，未到受测 G6 路径；本 diff 未修改这些旧模块/fixture。与实现交接报告的旧合同问题一致，但本 reviewer 未修旧合同，也未声称全库/完整旧 G6 回归通过。没有为满足本次 review 放宽旧合同或绕过拒绝。

R1/R2 关闭前不得把本次 review 记为 APPROVED 或发布科学训练。由唯一 author 按 fresh RED→GREEN→相关回归关闭这两项；按 AGENTS 普通工程修复规则继续，不重复完整 review。若修复意外要求科学设计变化，则停本支交 Kiro。此文仅为本次独立 review；root 仍是唯一服务器与事实源 owner。


## 主执行者工程修复关闭记录

原review CHANGES_REQUESTED保留，不伪称新增一次APPROVED。两个MEDIUM已在a5da4d6修复：clean +物理SHA/固定commit Git blob双核，dry/train/verify统一最终spec+prepared JSON门槛47104且intent前拒绝；verify超限写ALERT并latch，不重投。五例fresh RED、最终98相关tests含4mutants及47103/47104边界GREEN，主owner直接读原日志并核diff，旧入口无改动。最终FINAL卡同源原字节入Git。按AGENTS普通工程修复不重复整套review，科学设计未变；实际服务器合同/PF/CUDA/dry仍另需过闸门，未因review通过声称已验收科学结果。
