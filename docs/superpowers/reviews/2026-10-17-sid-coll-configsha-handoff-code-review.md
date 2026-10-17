# 协同向量 SID：config SHA metadata handoff 独立 code review

结论：**APPROVED（代码 review）**。未发现 HIGH / MEDIUM 阻塞项。对象 `023d0b19a8720fe4436ea87cf39f07f0d0f57834` 相对 `9e55dc7` 的三个新增文件：`analysis/sid_coll_configsha_handoff.py`、同名 `.md`、`tests/test_sid_coll_configsha_handoff.py`。本 reviewer 非作者，未修改源码、部署或启动恢复。context 计数 UNKNOWN。

依据：同名 handoff 说明、主仓 `2026-10-17-sid-coll-driver-alert-diagnosis.md`，以及 Eddy 直接批准「批准按提案恢复」的私有固定审批 JSON。该批准涵盖新版本工具、新 plan/control、保留原输出/STOP/intents/截止、采纳原 train42 和唯一 verify，不含训练重跑或预算/门槛变化。本 review 不扩大批准范围。

## 核对结果

- **仅改十二个 expected pins 与新 control_dir。** `patch_plan`（`:205–223`）深拷贝原 plan，只替换十二处原 null 的 `binding.config_sha256` 及 control_dir。每个 pin 同时核对冻结 spec 的内层 `parameters.seed`、job seed/run ID、固定科学 commit/source、配置 Git blob/物理文件 SHA 与原 `binding.data.config<seed>`。其它 job/source/data/resources 字段保持原值。原 created/deadline 相等并保留固定48小时窗口，过期 STOP（`:168–194`）；没有重置 spent、attempt 或截止。
- **既有作业与旧 intent 保护。** `inspect_runs`（`:269–324`）盘点全部28个原 ID：严格核 materialized queued job、唯一 claimed、run.json 的 config/input/env/resources/clean Git/ops identity，以及原 terminal 和 receipt。九个前置阶段（含 train42）必须 accepted，train42 的原九项 expected 必须恰好只差 null config SHA；缺失、失败、rejected、重复 queue、unresolved existing identity 或无 queue 的 uncertain intent 均 STOP。旧 intent 的 job digest 和字节 SHA 核对后复制到新 control；原 STOP/summary 只封存引用，不写回原目录（`:258–266`、`:327–390`）。
- **只读输入与版本身份。** 路径、文件类型与大小、重复 JSON key、非有限数均有约束（`:48–98`）；helper HEAD/clean tree/Git与物理脚本 SHA、全部科学 source/config、完整 ops package SHA 及 run ops identity 均检查（`:139–165`）。不打开 checkpoint、vectors、数组、预测或数据文件。inventory 只检查 receipt 身份，动态值禁止 summary 和未验收 raw score 寻址，仅允许既有 CUDA 显存声明（`:108–117`）。审批与 request 文件字节有 pin，启动时 request 对象、sealed request digest、plan SHA 和 guard 字节再次绑定。
- **唯一 ownership 与 release 边界。** `prepare` 全程持 global+old lock，以 new-only 文件封存新 plan/control（`:365–377`）。`start` 独占 launch intent，批准的当前 `sys.executable`、固定环境启动 helper，重试/部分新目录均拒绝（`:491–515`）。最终 owner 实际 PPID=1，持 global、old、new 三锁后写 ready，等待精确绑定的 release；release 前没有 tick/publication（`:443–488`）。旧锁用只读 fd 获取，旧 owner restart 无法同时持锁。
- **Linux ACK 验真。** `ack`（`:398–430`）不仅检查 ready 的声明，还读 actual `/proc/<pid>/stat` 的 PPID、`/proc/locks`，按设备 major/minor、inode、FLOCK ADVISORY WRITE 和**同一个 actual PID**确认三锁。`lock_owned_by` 不采纳其它 PID 或 POSIX lock。controller 完成 ACK 才写 release，owner 按 ready 字节 SHA/request/plan/PID/原 deadline 接受（`:433–454`）。控制连接在 release 前消失会有限等待后 STOP、零发布；release 后 owner 独立继续。
- **原科学 gate 与 publisher 未替换。** 已读取固定 `11d67681e0cbde09983f6bf482f55ab8a7191bdb` 的真实 `scripts/sid_coll_ops.py` 与 job-entry 接口：helper 直接复用原 `load_plan`、`tick`、intent publisher、`EventWaiter`、summary writer，没有 monkeypatch（`:467–488`）。原 tick 先逐项验收 train/verify，只有 verify42 的原始 verified receipt 与整数 hits≥260 才进入2024/2025；已存在 exact verify 采纳，缺失 ID 经原 intent-before-submit publisher 唯一提交。partial failure/uncertain intent 不重试，另一支继续；原900秒 fallback、own-terminal事件与原绝对 deadline 保留。launcher 的 run.json schema/ops source canonical hash 也与 Git 中既有 launcher 接口交叉核对。

## 独立本地验证与限制

使用项目已有 Python、`PYTHONPATH=.`、OMP/MKL/OPENBLAS threads=1，运行：

```sh
<APPROVED_PYTHON> -B -m unittest tests.test_sid_coll_configsha_handoff
```

实测 **26 tests PASS，13.569秒**。覆盖精确十二字段修改、旧字节保留、pending verify 采纳、uncertain/rejected/failed/身份篡改拒绝、source/helper/config/approval/request pins、原 tick 只补缺失 verify、真实本机 fork/PPID1/三锁/无 release 零发布/控制进程退出后 release 继续与重复 start 拒绝。

另做独立临时 synthetic ACK probe：真实持有三个 fixture flock，注入 Linux `/proc/locks` 文本和 PPID1；三锁同 PID 时 `ready_confirmed`，仅第三锁换成其它 PID 时正确拒绝。没有修改测试或生产源码。

**Linux 实机限制：** 开发机没有 Linux `/proc`，本地 fork test 使用显式 release，不代表完整生产 `start → Linux ACK → bound release` 已实测。首个服务器 metadata inventory 和实际 ACK 仍须由主执行者核验；工具的生产路径会强制通过该 ACK 才 release。原两 train42 的真实 succeeded/config 寻址证据来自上述诊断，review 未重新连接服务器，也未读取或解释未验收 scores。未声称服务器恢复、verify 发布或科学验收已完成。

本次为新增恢复组件的一次独立 review；后续普通工程修复按 fresh RED → GREEN → 回归收口，不重复整套 review。
