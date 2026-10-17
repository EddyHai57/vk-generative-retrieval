# 新 SID 的 A1 确认：实现交接

密级：公开。2026-10-17。context UNKNOWN。仅交接本地实现，不记录实验结果或裁决。

## 边界

- 依据 decision_log「午后：无人值守授权」及 FINAL 卡 `docs/superpowers/specs/2026-10-17-coll-sid-a1-confirmation-draft.md`。三个 seed 42/2024/2025 全跑；无筛选、无对照重训、无 bundle 重拟合。
- 新增文件；旧 A1/S1 入口没有修改。实现分支 `codex/coll-sid-a1-confirm`，base `ddf208e9b70a4c6c180d6dd1c600a464cf8958a2`；最终 HEAD 由本次 commit / Git 查阅，不在同一 commit 自引用。
- root 是唯一服务器/部署/job owner；本实现者没有连接服务器、push、部署、提交科学训练或读取旧 checkpoint/test。

## 新入口

`python -B -m vkgr.coll_sid_a1_run train|verify`：G6 原有全部 CLI 参数，额外必填 `--input-manifest --input-sha --vectors --vectors-sha`。Python run/verify 的额外 keyword 为 `input_manifest, input_sha, vectors_path, vectors_sha`。formal schema/task 为 `g-coll-a1-run-v1` / `SID-COLL-A1-CONFIRM`，训练/CP schema 独立；旧 G6/S1 manifest 被拒。

`configs/coll_sid_a1/g-coll-a1-c1-s{42,2024,2025}.json` 与原 `configs/gtok01b` 同 seed config 仅 schema/task_id/sid 不同，所有数值/类型逐字段锁定。model factory、train batch、训练/selection while 与 verify 的 selection replay 保持原 A1/G6 程序（AST 回归比对）。严格 C1 loader 沿用 `generative_coll_run._load_sid → sid_coll_v1.load_bundle`，H5a/b 不改，只有其 verified codec 的无损 depth3 适配。

formal 入口硬锁 FINAL 卡的 SID/input/vector 文件 SHA，SID 版本、词表 771、suffix 1；保留原 data_root、prepared/history/context、CONTENT_V1、输入 provenance 与原 source byte binding。verify 仅接受本入口新 manifest/CP 身份，重算 binding/selection/best digest，并对自己 best 新 CP 做重放。

## 私有运维 request

新 `scripts/coll_sid_a1_ops.py`：`preflight | cuda-preview | dry-run | submit | driver --request REQUEST`。路径、Python、scheduler、数据及旧原始 job 都由 root 的私有 JSON 提供，public 源码不含真实服务器路径/连接信息。

必需字段：

- `deployment`：固定新 commit 的完整干净 worktree；`source_commit`：40字节hex；`files`：全 `run.G6_MEMBERS` 加新 ops 的逐文件 SHA；`pinned_manifest`：部署 pin manifest 路径。
- `runs, queue, ops, python, scheduler_ops, data_root`：私有路径；ops 先创建空目录；scheduler_ops 供 import `vkgr_ops.launcher`。不由脚本部署或删除文件。
- `templates`：键 `42/2024/2025`（JSON string），值为原已验收同 seed A1+T3 **job JSON** 的路径。job.config 指原 config 文件；原 CPU/RAM/24h/disk/admission 固定检查。新 vram 在 preflight 中为 null（UNKNOWN），dry 后才由 CUDA 实测设整数；6656 只用于核旧模板身份。
- `coll_refs`：恰使用 `sid_manifest, input_manifest, vectors`，各为 `{path, sha256}`；输入只读，SID/input/vector SHA 沿 FINAL 卡。
- `fit_verify_receipt, fit_verify_stdout`：各 `{path,sha256}`。必须是原 C1 sid_verify 的原始 receipt/stdout。receipt 的 schema/stage/variant/raw verified、mapping_replay 与输入绑定校验；stdout 同时含 raw_proof 字典及最后 `{status:"verified", receipt:{path,sha256}}` 指针。不会重新执行 fit。
- `preflight_receipt, cuda_receipt`：分别指已保存的顶层 stdout JSON。不要混淆 raw config 文件 SHA 与 canonical config digest；preflight entries 同时报告 `config_sha256` 和 `config_canonical_sha256`，训练验收使用完整 `expected_binding_sha256`。
- `deadline_epoch`：首次 submit 固定的未来截止，最多48小时；后续 ready/driver 接续不重置。root 负责在实际 submit 前记录 STATE 带时间告知。

建议顺序：固定SHA deploy与文件核验 → preflight 保存JSON → CUDA工程job运行 cuda-preview 保存JSON → dry-run → submit。每一步单独留原始 stdout 与job身份；root 组装工程job及资源 admission。submit 先封 plan/request，再启动 detached owner；owner 实际 PPID1+新session 后 ready，controller 的 bound release 后才唯一 publish。前台 controller 发布数始终0。

queue/runs 身份检查包含 queue 全子目录；rejected 读取 reason，失败无重投。三个 train 固定新ID `g-coll-a1-c1-s{seed}`；每成功 train 仅唯一 `-verify`，verify 发布前不可覆写 intent，失败/已终态臂 latch，独立臂继续。driver poll 900s、固定48h，无模型调用；机器 summary 报 source/config/data/SID、best/stop、R/N/hits/差值与未验收/异常。只有原 verify stdout 的 `verified` 加 claimed job/manifest binding 才验收。

## 验证与尚未执行项

- fresh RED：新入口不存在、新 ops 不存在、全 queue 身份与 fit stdout 绑定新行为缺失；均执行后再实现。
- CPU synthetic 完整新 runner train → 自己新 checkpoint verify、tamper 拒绝已真实执行；不是旧 checkpoint。
- 新入口/ops/真实 detached 本机断线 fault/四个行为 mutants 与相关 A1/S1 回归已通过，最终命令及精确数量以交接消息附的新鲜终态输出为准。四 mutants 是 frozen field、bundle pin、schema、checkpoint hash；更正 macOS默认temp symlink会先被拒造成的伪阳性后重验。
- 全库 `unittest discover -s tests` 额外试跑已出现 E/F、P3 gate与sandbox sysctl warning，长时间未完成，按root授权只停止此本机 synthetic 测试并保留私有日志；没有全库全绿结论，没有为旧基线改scope。
- **服务器 preflight/CUDA/科学三train+各verify尚未执行**。CUDA工具只做真实工程 context/全A1 shape的两步 synthetic update、beam100 b1与新建内存 checkpoint reload/decode一致，ownPID/context-inclusive峰值按1.15×ceil256MiB。JSON明确 `engineering_only=true`、`memory_checkpoint_replay`、`server_full_runner_train_verify=false`；不得称科学train verified。服务器未执行完整 runner.run/verify，CPU完整runner证据与CUDA工程证据分开记账（root已确认此工程协议）。
- 真实 input/source/fit receipt 任一绑定不符、科学设计接入变化、NaN/Inf、test/旧CP越界即停该支回root/Kiro；不得降门禁。只有 root 一次独立 code review、public scan/push、真实工程门禁齐后才可发布三科学训练。

## 一次独立 review 后的工程修复

独立 review 的两个 MEDIUM 只改本次新增 ops：deployment 现在要求 Git status clean，并对每个 requested file 同时核物理 SHA 与固定 commit 的 Git blob SHA；request 自报 hash 无法批准 dirty 或 assume-unchanged 隐藏的修改。launcher.prepare 的真实返回值与 job 一起按实际 JSON 编码核最终 `{"spec":job,"prepared":prepared}` envelope，沿 S1 的 47104 字节上限（达到即拒绝）；dry、train publish、verify publish 都执行，先检查再写 intent。拒绝保留 ALERT 并不重投。

fresh RED 五个行为场景全 FAIL，GREEN 五项通过；另补 47103 允许 / 47104 拒绝的精确边界。FINAL 冻结卡从主仓原字节复制新增，SHA256 `da30eebb8caba93f19bfb0b4fc017ffe8f88d634eb75c0ae3f611e7f1b29e2ff`。没有改科学 runner、配置、算法、预算或原 A1/S1 文件；没有再跑全库 suite；真实服务器工程前置仍待 root。
