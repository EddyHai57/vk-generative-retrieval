密级: 公开

# SID-FIX S0 Implementation Plan

Goal: 用只读归档计算已冻结 A–D，S1 只出草稿。
Spec: docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md §2；Eddy 已选择直接实现/一次 review，不重开设计轮。
Architecture: 新增纯函数、严格输入/hash读取器与独立 queue 运维；不改旧入口。

## C 组来源（实现前登记）
G4 定义：experiment_log 的 G4 D1/D2 节；原代码 generative_evaluation.py ITEM_GROUPS 与 history_v2.build_observed。阈值来源是 h2-prepare/out/train-slices.json，由固定 history manifest（8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43）绑定。nearest-rank 对所有 train-positive item 的 train-like 事件次数：sorted_positive[ceil(N*.5)-1] 与 [ceil(N*.9)-1]。OOV 先排；count=0 unsupported0；0<count≤median longtail；median<count≤p90 mid；count>p90 head。不改变 ties 边界，不用 validation 重估阈值。真实数值和原始 SHA 在服务器 metadata preflight 核后才登记；不会填猜测值。

## 文件与验收
- [x] analysis/sid_diag_metrics.py / test_sid_diag_metrics.py：先 RED（模块缺失），A–D 手算 GREEN；语义 leaf 与完整 SID 分开，D2 pad 非有效语义层；0 分母 UNKNOWN。
- [x] analysis/sid_diag.py / test_sid_diag_io.py：只读 allowlist/所有输入 SHA，旧训练 verified 及预测绑定；训练期完整相邻事件而非重叠截断样本；输出新目录 JSON+报告。
- [x] analysis/sid_diag_ops.py / test_sid_diag_ops.py：新 deploy/只读真实 metadata preflight/dry/一次 queue/独立 claimed/rejected 收据，1 CPU/无 CUDA；失败不重试旧 attempt。
- [x] 一次 whole review；扫描新增全部 Git 历史、每文件≤1MiB；push 工作分支；固定 SHA 部署逐文件 hash。
- [x] 先 first_batch，排除 A1+B1024；三 verify 齐才补第二次，不改原 run/bundle。
- [x] experiment_log 数字/路径/SHA/实际UTC；STATE 只记机械读法与待 Kiro。

Stop: 输入/身份/顺序/hash不一致，无法核验 raw verified，旧 run 或重复身份，真实 preflight 失败；S0不读test/checkpoint/完整embedding。S1草稿的设计取舍只交Kiro，不在S0决定。

最终交付：两批均succeeded、输入与canonical result SHA核验、A–D与B均值/C双对照写experiment_log。4旧缺口UNKNOWN，不读预测；S1仍DRAFT。普通metadata/schema/性能修复保留旧部署，唯一pending覆盖获Eddy明确批准；无重投失败attempt。精确身份/收据/测试与版本差异见STATE/handoff与最终索引。
