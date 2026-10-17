# 协同 SID aggregate 有限读取交接

授权：decision_log「2026-10-17 午后：无人值守授权」§2.4；按 S1 FINAL §7 与 pipeline §2 原定义读取。仅新增 reader、synthetic tests、本交接。无服务器连接，无原数据/预测/映射/checkpoint 读取，无作业或预算变更。context UNKNOWN。

## 调用

`python analysis/coll_sid_diagnostics_read.py --stdin < PRIVATE_REQUEST.json`

stdin 为 JSON：`schema` 固定 `coll-sid-aggregate-request-v1`；`files` 正好八条，每条字段为 `path`（私有绝对路径）、`sha256`（原 proof pin）、`run_id`、`role`（mapping 或 prediction）、`variant`（C1 或 C2）、`seed`（mapping 为 null；prediction 为 42/2024/2025）。每 variant 一 mapping、三 prediction。reader 不写文件，stdout 为 `coll-sid-aggregate-read-v1`，含 UTC、来源 pins、八个核验后 aggregate。

basename 只接受 mapping-diagnostics.json / prediction-diagnostics.json；路径必须对应 runs/run_id/out；禁止 symlink；单文件最多 2MiB；JSON duplicate/nonfinite 拒绝；严格小 aggregate schema，拒绝额外预测数组。失败 exit 非零，不输出部分科学结果。输出 verified 只表示有限 aggregate 核验，不代替训练原 raw verified 验收。

## 来源定位与下一动作

唯一服务器 owner 用原六 verify receipt 的 proofs.prediction_diagnostics 和 summary.branches 的 sid_verify.artifacts.mapping_diagnostics 组私有请求；八条都已有 SHA，不重跑诊断。prediction JSON 含 B/C/C_delta_vs_T3，mapping JSON 含 A/D/Dprime/physical_sources。

T3 参照可从已有私有缓存 `sidfix-s0-report-artifacts-retry1-20261016T101107Z.stdout` 的 runs[sid-fix-s0-first-v1].result.C.T3 取，原 result SHA 为 `a2e93d69bfb79af2c94e344aa5368e7604825777acc3ed6f2f6cde37c5db1924`。G4 train-slices SHA 与门槛看该 result.inventory.threshold_source 和各 C.thresholds，不读大 slices/counts 文件。工具 `compare_c(prediction_aggregate, same_seed_reference_C)` 验证门槛/target 分母并返回五档命中差；root 必须再与原 C_delta_vs_T3 相等，和总 hits 对齐。

本地测试含 synthetic 手算 B/C、OOV 排除、五档 delta、A occupancy、有效层数、D/Dprime STRUCT_NA 和零随机分母，以及 pin/run/path/schema/额外预测数组拒绝变异。首次真实八文件 intake 尚待唯一 owner 固定 SHA 部署后执行；若实际 schema 不符停止，不能据 synthetic 宣称真实 A–D 已收口。科学判断与数字事实源只由 root 写，避免跨 core03/A1 判赢家或把 lift 写成已学到协同结构。
