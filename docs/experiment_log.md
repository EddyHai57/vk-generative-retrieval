密级: 公开

# Experiment Log

## 当前状态
**本节原文“尚无实验”已过期**（2026-10-15 更正）。已有大量实验结果，按时间顺序记在下面各节，最新的在文末。P 线锚点 F4-T010（三 seed R@50 0.17421/0.17222/0.17619）；G 线最强候选是 256 叉 SID（T3，core03 三 seed 254/274/275 hits）。当前阶段与在跑任务看 `STATE.md` 文末，排程看 `docs/schedule.md`。
公司内部信息、数据、schema、代码与实验结果不得进入本项目。
本文件是人工维护的实验与指标唯一摘要源。
原始 config、metric JSON、日志与大 artifact 不复制到本文，只记录路径别名与 hash。
## Run Summary

| Run ID | Status | Dataset | Split | Catalog | SID | Model | Seed | Hardware | Metrics | Resources | Artifact hashes | Claim boundary |
|---|---|---|---|---|---|---|---:|---|---|---|---|---|

当前表为空，不预填任何指标或预期收益。

## Run Record Template

### Identity

- Run ID:
- Date:
- Status: `planned | running | completed | failed | invalidated`
- Git commit:
- Config path:
- Parent run:
- Purpose:

### Dataset

- Dataset/subset:
- Source manifest hash:
- Processed dataset hash:
- Schema version:
- Row count:
- User count:
- Item count:
- Week range:
- Feedback policy version:
- Duplicate policy version:

### Split

- Split manifest hash:
- Train weeks:
- Validation weeks:
- Test/holdout weeks:
- Query set hash:
- Query count:
- History rule:
- Target rule:
- Holdout usage boundary:

### Catalog

- Catalog snapshot ID:
- Catalog cutoff:
- Train-observed item count:
- First-seen/OOV target count:
- Feature coverage:
- Catalog hash:

### SID

- Tokenizer name/version:
- Codebook config:
- SID depth/vocabulary:
- Collision suffix rule:
- SID map hash:
- Collision diagnostics artifact hash:
- Train/freeze week:

### Model

- Model family:
- Parameter count:
- Input features:
- History length policy:
- Objective:
- Decode policy:
- Constraint version:
- Checkpoint hash:
- Baseline fairness notes:

### Reproducibility

- Seed:
- Determinism settings:
- Dependency lock hash:
- Command manifest hash:
- Retry/resume notes:

### Hardware

- Provider:
- GPU model/count:
- GPU memory:
- CPU:
- RAM:
- Disk:
- CUDA/runtime:
- FAISS/backend:

### Metrics

- Item Recall@K:
- Item NDCG@K:
- SID-level metric:
- Collision-corrected item metric:
- Invalid rate:
- Dead-end rate:
- Duplicate rate:
- Effective valid K:
- Train-observed slice:
- First-seen/OOV slice:
- Warm/cold slices:
- ANN recall against exact audit:
- Statistical summary / seeds:
- Metric JSON hash:

### Resources

- Data preparation wall-clock:
- Training wall-clock:
- Evaluation wall-clock:
- Refresh wall-clock:
- Peak RAM:
- Peak VRAM:
- Disk footprint:
- Index/trie footprint:
- Latency p50/p95:
- Cost:
- Resource trace hash:

### Artifact Hashes

- Source manifest:
- Dataset manifest:
- Split manifest:
- Catalog manifest:
- Feature manifest:
- SID/codebook:
- Checkpoint:
- Index/trie:
- Predictions:
- Metrics:
- Logs:

### Claim Boundary

- This run supports:
- This run does not support:
- Known confounders:
- Failed gates:
- Follow-up required:
- Result classification: `positive | negative | inconclusive | invalid`

## 填写规则

没有 artifact hash 的指标不得作为正式结果写入 summary。

One-seed 结果必须明确写 one-seed，不外推稳定性。

SID-level 指标不得替代 collision-corrected item-level 指标。

ANN 指标必须附固定 query exact audit。

`first-seen` 只能解释为 observed-item proxy，不解释为真实 listing 或 cold-start ground truth。

不报告 CTR、online lift 或 A/B 结果。


## G1 — 冻结输入与 SID 真实审计（2026-10-08，Astra accepted）

本次是 bounded CPU input audit，不训练、不重新拟合 SID、不打开 actual test。执行来源为生成式 worktree 的 `cache/g1-real-audit-aggregate-evidence.json`；Astra 通过 SSH 独立读取实际三个小产物并核内容和 hash，证据 `/private/tmp/g1-astra-remote-verification.json`。

| 产物 | SHA-256 |
|---|---|
| aggregate.json | d03978b3e4d5346208c605866640b25d83a36e4164419364dc5bd4c0be817e47 |
| supervision.json | ce39dac731f7ff411a8831bdda255338dc1b1e4de11f476bc33d1173a4097d4c |
| server-contract.json | 4bd3d093d4f86f4a033eda293798b807e46dd9d8ea0af68240481972d3e000a7 |

- Catalog 120415；ordered training examples 86785；固定 validation queries 2520，其中 OOV 623。
- 全 catalog SID roundtrip/reachable 均为 120415；完整 SID collision 0，valid-prefix deadend 0；4 个非法 token/suffix 探针全部拒绝。
- Semantic leaf 4096 个全部占用；2465 个 leaf 各含 29 items，1631 个各含 30 items。语义前三位仍有碰撞，靠 suffix 区分，不能把完整 SID 无碰撞误称语义码无碰撞。
- 原 producer 的 273 个节点中 245 个 unconverged，如实保留；本审计没有修复或 refit。
- 内容距离仅对 21496 个 train-positive-supported items，按冻结 leaf/item 顺序取最多 4096 对相邻 items。共有 17498 对 eligible，实际 4096 对；float64 squared L2 mean 0.7429038667488203，max 2.1125659593192783。没有随机对照，不据此声称语义质量改善。
- Worker exit 0，wall 79.78798671788536 s；317 次采样，peak RSS 736935936 bytes，peak IO 440294040 bytes，minimum visible RAM free fraction 0.5054285365047034。单 CPU，CUDA 未初始化；均满足本次预注册 caps。

Claim boundary：证明冻结输入身份、catalog/SID 可逆可达与有限采样资源 gate；不证明模型质量、训练性能、硬资源隔离或线上效果。资源是 worker-only sampled controls，可能有采样间隔内 overshoot，hidden host ancestry unknown；`gd.audit_sid` 本身资源测量仍标 not_measured，与外层 supervisor 分开解释。G1 accepted 后才进入 G2 runner 设计。


### P4E prepare共享统计 — Astra fresh验收（2026-10-08）

服务器内独立train Counter与产物一致：catalog120415，positive items21496，known-zero98919，positive users18915，train-like events105700；原training examples86785、validation queries2520。prepare68.8686s/verify76.0935s，均exit0；operation产物63711384bytes，上限3600s/4GiB；0 optimizer更新。Astra只读验收42.9374s，当前余量RAM0.531963/VRAM0.987810/disk0.713521，仅验收时单点，不冒称原运行历史minimum（未记录）。证据/private/tmp/p4e-astra-prepare-acceptance.json；这是数据与执行验收，不是模型质量结果。


### P4E v2 pilot协调验收（2026-10-08）

Astra fresh核12worker exit0、训练增量180steps、三策略13项state字段及完整output exact proof绑定；outer exit0，wall1382.041129s，从保留原budget起点累计2416.285655s。新产物1127452867bytes，旧失败3586bytes，均计原预算。continuous30训练耗时依uniform/user_balance/watch为11.917006/4.348549/6.213294s；完整导出及receipt尾段37.016795/37.383047/38.854211s。独立证据/private/tmp/p4e-astra-pilot-final-check.json，completion3fb47d4e0655bba13941d8233e84d4c7330aa2ea375231664fddd5e0d7f50f5a。此为短CUDA恢复/产物闭环，未跨完整epoch、不报告质量结论，正式成本外推待补。


G3本地协调验收：fresh244tests64.029s PASS(skip1)，七CLI实际完成，state SHA d4aa162fa3c959b7eab10d0de73df57e69b2ec6b8bd948f7ef1a27e1a9a94ee9与G2一致；pool formal2/replay2/decode35符合CPU工程链。证据/private/tmp/g3-astra-cli-zzmm2rxi/receipts.json。无真实GPU/模型质量结果。

### PG-R01-P-RECOVER｜pilot07实际独立verify接收（2026-10-13）

原source9a59/完整86785 train、120415 catalog，matched train-only capacity两臂各2520、各30optimizer steps。新真实prepare184.113375469s；pilot actual wall302.720451731s。各30steps训练计时like6.924483347s、observed6.909634038s；完整epoch export/exact分别24.852717939s、25.254611623s，checkpoint各124422667bytes。计时为30steps总量，不能当单step。

Digest约定修复只追加wrapper/凭证，原checkpoint/queue/失败ledger/spent保留。独立verifier manifest db7f25e8…、terminal6f5a5ff3…，实际237.140014393s，两臂fresh tensor replay成功、5040 total train-only queries、0optimizer steps、validation_target_read=false/quality_metrics_computed=false；worker/controller/pub实际exit0/CLOSED、fresh所有owned absence。接受证据main05/pilot07-final-acceptance.json、verify-repair01-accept-cost03.stdout；不报告模型质量、validation或test收益。

绑定已实测部分的S02/S06 forecast为13024s/6248983966bytes（1.5时间余量、30epoch/5100steps/slot，保守metadata/atomic/cleanup储备），尚缺完整formal validation prepare/deploy与metrics聚合/序列化，因此admissible=false/submit_permit=false。Cost-completion01仅CPU成本探针已start：10:25:30.453+08 first-write＋2h、原树old+new20GiB、0新科学attempt/训练步数，zero-vector synthetic ranks绝不能当模型质量；实际成本待其终态绑定。新旧spent不reset。

Cost-completion01最终补齐：实际277.807206667s、worker actualexit0/CLOSED/fresh全部owned absence，terminale6295fe4…、registrationc0740d00…、sourcea3350863…。完整86785 train/full120415catalog/full2520 validation producer627b3b2b…；prepare240.649891755s/input137011897bytes，真实source copy0.443980665s/897241bytes。CPU zero-vector synthetic evaluator＋metrics/prediction序列化成本like15.415677129s、observed15.448801615s，各575369bytes；只成本，不是模型质量。没有optimizer/model selection/HPO/test或新科学attempt。

完整保守forecast在base上每slot加1104s，并额外计input/deployment/32份序列化；故两slot15232s/6561625858bytes、uniform7616s/slot，cost_complete/evidence_bound/time_fits/disk_fits=true。Full prepare与synthetic exact在已有forecast之外重复计入为明确保守overcount；预测不是账单保证或hard quota。原树总charged1502244950bytes、旧spent/queue/产物不改。最终绑定main05/p-recovery-final-acceptance.json、cost-completion01-final-accept.stdout；PG-R01-P-RECOVER完成。Formal仍admissible=false/submit_permit=false，需新队列14h/32GiB、source/input/fullvalidation/资源/身份/permit准入后另task执行，不能把成本通过当已训练或质量收益。

## P4H-05 recovery01｜SASRec-style S02/S06 validation 终态（2026-10-13，Kiro只读读取）
来源：服务器 recovery01 `P4H-S0x/formal-verify-output/epoch-1-metrics.json`（独立verify，optimizer_steps_during_verify=0）；本地收据 main05/p4h-metrics-01.stdout（SHA 44c3d12c…）。driver `S02_recovered+verified_S06_completed+verified+CLOSED`，test未读。配置：ID-only causal SASRec-style 64d/1层/4heads/最近50行为，LR0.001/T0.15/seed42，batch512，max30/patience3。
| Slot | history | completed epochs / best | steps | R@10 | R@20 | R@50 | N@50 | train-observed R@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P4H-S02 | like | 4 / 1 | 680 | 0.000000 | 0.000397 | 0.000397 | 0.000107 | 0.000527 |
| P4H-S06 | observed | 4 / 1 | 680 | 0.000397 | 0.000397 | 0.001587 | 0.000630 | 0.002109 |
n=2520（OOV 623计入分母，OOV各指标0）。**负结果**：R@50分别只有1个、4个命中，远低于Pop单路0.0500与双塔id_mean 0.0373；两者都在epoch1取best后连续3个epoch变差而早停。原因未诊断（可能为配置/学习动态或实现问题），不据此否定SASRec类方法；单seed，不做稳定性结论。诊断登记为后续issue。

## Batch1 W1｜history-v2 + init small + max100/min5（2026-10-14，validation，Kiro只读读取）
来源：服务器 /workspace/vkgr-infra01-root/runs/<run>/out/manifest.json 的 selection.best_metric；本地收据 main05/kiro-b-best-104607.stdout（SHA a3262401…）。代码 P batch1/p 457cad8、G batch1/g 6f3901d overlay。validation n=2520（OOV 623 计入分母），test 未读。各 run 均 attempt1/resume0、succeeded，verify 已通过（G 两个 verify 运行中）。
| Run | 线 | 设置 | R@50 | N@50 | best/stop epoch |
|---|---|---|---:|---:|---|
| p-hist-v2-l50-s42 | P | id_mean, v2 L50 | 0.1262 | 0.0330 | 3/8 |
| p-hist-v2-l50-s2024 | P | 同上 | 0.1254 | 0.0332 | 2/8 |
| p-hist-v2-l50-s2025 | P | 同上 | 0.1218 | 0.0323 | 3/8 |
| p-hist-len-l20-s42 | P | id_mean, L20 | 0.1262 | 0.0345 | 2/8 |
| p-hist-len-l100-s42 | P | id_mean, L100 | 0.1187 | 0.0317 | 3/8 |
| p-p4h-v2-s1-lr1e-3 | P | SASRec, L50, LR1e-3 | 0.1028 | 0.0306 | 24/27 |
| p-p4h-v2-s2-lr3e-4 | P | SASRec, L50, LR3e-4 | 0.0762 | 0.0204 | 11/14 |
| p-replay-s42 | P | 新代码 + legacy init + v2 L50 + 旧预算（不计数） | 0.0798 | 0.0216 | 30/30 |
| g-hist-v2-l50-s42 | G | core03, v2 L50, W=4 | 0.0754 | 0.0258 | 10/13 |
| g-hist-v2-l50-s2024 | G | 同上 | 0.0718 | 0.0256 | 9/12 |
| g-hist-v2-l50-s2025 | G | 同上 | 0.0825 | 0.0278 | 16/19 |
| g-hist-len-l100-s42 | G | core03, L100, W=1 | 0.0758 | 0.0271 | 9/12 |
| g-g4-rand-s42/s2024/s2025 | G | 随机 SID control, v2 L50 | 0.0520/0.0500/0.0500 | 0.0142/0.0141/0.0151 | — |
解读（只做方向判断，不是最终结论）：
- 参照（旧口径，非同版本）：P4D id_mean like-history 0.0373；P4F wRRF 0.0635；G core03 v1 0.0540；Pop 0.0500。
- P：history-v2 + 新初始化后，id_mean 3 seed 均值约 0.124，是旧值的 3 倍以上，也超过旧四路融合。P-REPLAY（旧 init）为 0.080，接近 P4E watch_history 的旧量级，说明大约一半增益来自观看历史、另一半来自初始化（粗略分解，单 seed）。
- G：v2 L50 3 seed 约 0.072–0.083；随机 SID control 约 0.050–0.052，真实语义 SID 高出约 0.025（≈60 次命中），支持“SID 语义有用”。
- SASRec-v2 低于 id_mean（0.103 vs 0.126，seed42），按卡 0c 规则 P anchor = id_mean。
- HIST-LEN：P L100 比 L50 低 0.0075（变差）→ L100 出局；P L20 与 L50 的 R@50 持平。G L20 尚未完成，L* 待 G L20 终态后按联合规则定（预计为 50）。

### Batch1 W1 补充 + W2 P-NEG（2026-10-14 11:35，validation，收据 main05/kiro-b-best-113*.stdout）
| Run | 设置 | R@50 | N@50 | best/stop |
|---|---|---:|---:|---|
| g-hist-len-l20-s42 | G core03, L20 | 0.0770 | 0.0243 | 14/17 |
| p-neg-n1-logq | id_mean L50 + in-batch logQ(α=1) | **0.1472** | 0.0399 | 10/13 |
| p-neg-n2-mixed | + supported 均匀负样本 | 0.1437 | 0.0376 | 11/14 |
| p-neg-n3-mixed-logq | 两者结合 | 0.1433 | 0.0383 | 14/17 |
对照 anchor p-hist-v2-l50-s42 = 0.1262。logQ 纠偏 +0.021（≈53 次命中），三种方案都明显高于对照；单 seed，verify 已入队。
HIST-LEN 联合选择：L20 相对 L50——P R@50 +0.0000、G +0.0016，均未达 +0.002 门槛；L100 P 变差。按预注册规则 **L* = 50**。

### FEAT-01 F1–F3 + C0（P 线，seed42，validation，2026-10-14，收据 main05/kiro-b-bb-141511.stdout；详细见 .agents/tasks/feat01b-report.md）
代码 P feat01/p d70a993（卡 FROZEN 2026-10-14T04:26:52Z）；基线 N1 = id_mean L50 + in-batch logQ，R@50 0.14722 / N@50 0.03987。validation n=2520（OOV 623 计入分母）；5 个 run 全部早停结束（hit_max_epochs=false），verify 全部 verified，test 未读。
| Run | 变量 | R@50 | ΔR@50 vs N1 | N@50 | best/stop |
|---|---|---:|---:|---:|---|
| p-feat01-f1-ratio-pool-s42 | F1 观看比例加权池化（无新参数） | 0.13373 | −0.01349 | 0.03603 | 9/12 |
| p-feat01-f2-feedback-s42 | F2 6 个反馈位 embedding | 0.14643 | −0.00079 | 0.04009 | 10/13 |
| **p-feat01-f3-watch-buckets-s42** | **F3 观看时长分桶 + 观看比例分桶 embedding** | **0.15556** | **+0.00833（+21 命中）** | 0.04424 | 10/13 |
| p-feat01-c0-unsupported-const-s42（对照，不计名额） | 仅给 unsupported 历史事件加同一个常数向量 | 0.14722 | 0.00000 | 0.04006 | 10/13 |
解读：F3 越过 +0.002 门槛，按预注册规则成为新 P anchor（F4 起叠加其上）；F1、F2 无效。C0 与 N1 完全相同，说明“unsupported 历史变得可见”本身几乎没有作用，F3 的增益主要来自时长/比例信息（两者可能有交互，这只是方向判断，不是精确拆分；F3-masked 对照已触发）。
切片（best epoch）：target 侧所有 run 的 unsupported / OOV target R@50 均为 0（id_mean 物品塔对这些 item 输出零向量，用户塔特征改变不了）；F3 增量全部来自 supported target（0.2227 → 0.2353）。历史侧：F3 增量落在 h_ge_half（+15 命中）与 h1（+5 命中）；零用户向量占比 5 个 run 均为 0。**卡里风险 1 的预测（F3/C0 让零向量→0、增量落在 h0/h_lt_half）因 v2 L50 上本来就没有零向量而落空**（h0 桶为空、h_lt_half 仅 13 个 query），如实记录。N1 的切片由冻结的同一份切片代码从存档 predictions 离线重算（与 F3 存档一致 = true）。
限制：单 seed；分桶边界与 F1 权重是经验定的、未做边界搜索；仅 id_mean / L50 / N1。待补 seed 2024/2025（feat01c-seeds 运行中）。

### INFRA-02b GPU 批量 beam 解码对拍（G，2026-10-14，来源 .agents/tasks/infra-02b-report.md，收据 main05/i02b-*）
参照 run g-infra02-parity-ref-s42（新入口 generative_b1_batched，默认模式）与旧 runner 的 g-hist-v2-l50-s42 逐 epoch 13 个 `model_sha256` / `computation_sha256` 完全相同。对拍覆盖全部 2520 个 validation query、best（epoch10）与 last（epoch13）两个 checkpoint：**batch_size=1 equal=true（解码 wall 237s→81s，约 2.9×）；batch 32/64/128 全部 equal=false**（cuBLAS 在 batch 维变大时选了不同归约顺序，float32 logits 差 1.4e-6～3.1e-6；b32 有 14/2520 个 query 在近似并列处顺序翻转，候选集合与 audit 不变）。结论：仅允许 batch_size=1，仅 L≤50（L100 批量路径未对拍），不启用任何容差；批量训练的 run 必须再用逐条解码 verify 一次。

### FEAT-01c｜F3 vs N1 补 seed（P 线，validation，2026-10-14，8 个 job 全部 succeeded 且 verified；报告 .agents/tasks/feat01c-seeds-report.md，收据 main05/kiro-b-bb-152404.stdout SHA fba18896…）
代码 feat01c/p 9f29922（仅新增 configs/feat01c/ 4 个配置，与 s42 只差 seed）；服务器 /workspace/vkgr-feat01c-p。N1 与 F3 均为 id_mean + HIST_V2_OBS_L50 + in-batch logQ(α=1) + init small + 100/5/3，F3 多了观看时长分桶+观看比例分桶 embedding。
| seed | N1 R@50 (hits) | F3 R@50 (hits) | ΔR@50 (Δhits) | N1 N@50 | F3 N@50 | N1 best/stop | F3 best/stop |
|---:|---|---|---|---:|---:|---|---|
| 42 | 0.14722 (371) | 0.15556 (392) | +0.00833 (+21) | 0.03987 | 0.04424 | 10/13 | 10/13 |
| 2024 | 0.14722 (371) | 0.14960 (377) | +0.00238 (+6) | 0.03870 | 0.04121 | 13/16 | 7/10 |
| 2025 | 0.14286 (360) | 0.15437 (389) | +0.01151 (+29) | 0.03807 | 0.04324 | 8/11 | 15/18 |
| 均值±样本sd | 0.14577±0.00252 | 0.15317±0.00315 | **+0.00741±0.00463** | 0.03888±0.00092 | 0.04290±0.00154 | | |
（N1 s42 与 s2024 命中数恰好都是 371，但 N@50 与 best epoch 不同，不是同一个 run。）
解读（如实）：F3 在 3 个 seed 上 R@50、N@50 全部为正，均值 +0.0074（约 +5% 相对），按预注册 ≥0.002 规则仍为有效 anchor；但 seed2024 只多 6 个命中，刚好压线；seed42 的 +0.0083 偏乐观（另外两个 seed 均值 +0.0069）；配对 t=2.77（df=2）**未达 5% 显著**，n=3 的效力有限。结论强度：“方向一致、幅度中等、统计上未充分确立”，不能写成稳定收益。N@50 的提升更一致（+0.0044/+0.0025/+0.0052，均值 +0.0040，sd 0.0014）。

### G4 D1/D2｜beam 宽度诊断（G，decode-only，2026-10-14，validation；报告待写，收据 main05/kiro-b-bs-154436.stdout SHA a13cebd5…、kiro-b-bm-154139.stdout）
对象：G anchor g-hist-v2-l50-s42 的 best checkpoint（binding：best_checkpoint_sha256 87a06aff…、model_sha256 a93266ac…、run_manifest_sha256 672e7731…，reference_beam=100）。两个 run 均 succeeded、optimizer_steps=0、n=2520（OOV 623 计入分母）、test 未读；manifest SHA beam50 6147e5b5…、beam200 94cf31d7…。属诊断，不是正式指标，正式 beam100 不变。
| 解码宽度 | R@50 (hits) | N@50 | in-catalog R@50 | coverage | 相对 beam100 救回/丢失 | p50 / p95 延迟/query | wall |
|---|---|---:|---:|---:|---|---|---:|
| beam50 | 0.07262 (183) | 0.02508 | 0.09647 | 0.0182 | 31 / 38（净 −7） | 0.057s / 0.059s | 235s |
| beam100（正式 anchor） | 0.0754 (190) | 0.0258 | — | — | — | — | — |
| beam200 | 0.07579 (191) | 0.02586 | 0.10069 | 0.0522 | 5 / 4（净 +1） | 0.166s / 0.172s | 512s |
（beam100 的 in-catalog 与 coverage 本次未重读，留空。延迟是 batch=1 单 query 解码，beam50 与 beam200 有约 200s 重叠并发，另有 GPU 共享，只作量级参考。）

逐层前缀命中率（in-catalog 1897 个 query，top50 候选；定义取自 manifest summary：前 k 位 SID 与目标一致）：
| 解码宽度 | 前1位 | 前2位 | 前3位 | 完整 SID | 前3位对时 suffix 命中 |
|---|---:|---:|---:|---:|---:|
| beam50 | 0.766 | 0.405 | 0.152 | 0.0965 | 0.633 |
| beam200 | 0.862 | 0.431 | 0.158 | 0.1007 | 0.639 |

按目标热度切片的 R@50（beam50 / beam200）：>p90（963 个）0.185 / 0.195；>median ≤p90（548 个）0.0091 / 0.0055；≤median（155 个）0 / 0；unsupported0（231 个）0 / 0；OOV（623 个）0 / 0。

解读（只对这一个 checkpoint 成立）：
- beam 宽度不是瓶颈：100→200 净多 1 个命中（救回 5、丢失 4），属于并列处的抖动；砍到 50 净少 7 个。
- 损失主要出在 SID 第 3 位：前2位 0.43 → 前3位 0.16 掉得最多；前3位对了之后 suffix 还能命中 64%。所以问题更像 tokenizer 的粒度/语义质量与模型的区分能力，而不是解码，也不是叶内 suffix。这是对 GTOK-01 与 GARCH-01 的动机，不是它们的结论。
- 命中几乎全在头部目标：>p90 的 963 个 query 命中率 18–20%，中间档不到 1%，其余为 0；与随机 SID control（0.050–0.052，约等于 Pop 0.0500）合起来看，G 目前的增益主要来自“热门 + 少量语义”。P 线同切片的对照还没做，不能据此写 P/G 差距的原因。
限制：单 checkpoint（seed42 best）、单次解码；更宽 beam 仍可能漏掉高分路径。

### G3 补 seed｜旧代码 selected-seed2024 / 2025（G，v1 口径，2026-10-14 读取，validation，收据 main05/kiro-b-g3-154227.stdout SHA cdeb32ef…）
代码/配置：旧 frozen-base 路径 + G3 select 选出的 winner（config_index 3 = core03 d128 / LR0.0003，like-history，旧 init）；服务器 /workspace/vkgr-g3-selected-seed{2024,2025}-v1，两个 run 均 replay verified（status=verified）、0 resume、n=2520、beam100、test 未读。
| seed | R@50 (hits) | N@50 | in-catalog R@50 | 训练到 epoch | formal / replay wall |
|---:|---|---:|---:|---:|---|
| 42（既有参照行） | 0.0540 (136) | — | — | — | — |
| 2024 | 0.00913 (23) | 0.00238 | 0.01212 | 14 | 17837s / 16714s |
| 2025 | 0.01270 (32) | 0.00611 | 0.01687 | 12 | 15414s / 14407s |
（seed42 取自上文“参照（旧口径，非同版本）”一行，本次没有重读它的 manifest。）
解读：同一配置、同一旧代码，三个 seed 的 R@50 是 0.054 / 0.009 / 0.013，两个新 seed 都低于 Pop（0.0500），seed42 的 0.054 是离群的乐观值。这说明 v1 的 G 对 seed 极不稳；v2（observed history + 小初始化 + min_epochs）3 个 seed 为 0.072–0.083，既更高也更稳。但 v2 同时改了好几样东西，P-DIAG-01 找到的“默认 N(0,1) 初始化 + 早停抖动”是否也是 G v1 的原因，**这里没有验证**，只能说方向一致。结论：以后引用 G v1 时不要再用 0.0540 当 G 的基线，G 的基线以 v2 三个 seed 为准。
限制：这两个 run 只是 v1 的补 seed，不进入任何 v2 的选择；不重跑。

### FEAT-02 seed42（P 线，validation，2026-10-14 18:3x 读取；**数字已于 18:5x 验收，见下方“FEAT-02 验收补记”**；收据 main05/kiro-b-f2m-183236.stdout SHA cdef918c…）
代码 feat02/p 94eb0bc（卡 FROZEN 2026-10-14T09:14:29Z），服务器 /workspace/vkgr-feat02-p。基线 F3 s42 = 0.15556 (392) / N@50 0.04424，其余条件同 F3（N1 + HIST_V2_OBS L50 + init small + 100/5/3，seed 42）。validation n=2520（OOV 623 计入分母），test 未读。
| Run | 变量（只改这一个） | R@50 (hits) | ΔR@50 vs F3 (Δhits) | N@50 | best/stop | 状态 |
|---|---|---|---|---:|---|---|
| F3-masked | F3 的 side 只加在 supported 事件上 | 0.15913 (401) | +0.00357 (+9) | 0.04571 | 10/13 | complete，verify 运行中 |
| F4 recency | F3 + 位置桶 + 周间隔 | 0.16825 (424) | **+0.01270 (+32)** | 0.04946 | 12/15 | complete，verify 排队 |
| F5 scene | F3 + 场景 | 训练中，epoch10 中间值 0.15595，不作结论 | — | — | — | running |
| F6 author+content | F3 + 作者 embedding + 内容 | 0.14087 (355) | **−0.01468 (−37)** | 0.04043 | 4/8 | complete，verify 未入队 |
卡 §9.1 的拆分（方向性，单 seed）：ΔM = F3-masked − N1(0.14722, 371) = +0.01190 (+30)；ΔV = F3 − F3-masked = −0.00357 (−9)。
解读（如实，全部单 seed、未 verify）：
- F4 按预注册规则（ΔR@50 ≥ 0.002）有效，也是目前最大的单项增益；N@50 同向 +0.0052。F3 三个 seed 的 R@50 sd 约 0.0032，F4 的 +0.0127 是它的 4 倍多，但只有一个 seed。按卡补 seed 2024/2025（不计名额）；成为新 anchor 之前要先做一次针对泄漏的 review（decision_log 16:3x 第 5 点）。均已排。
- F6 比 F3 低 0.0147，甚至低于 N1（0.1472），应记为“无效”。它第 4 个 epoch 就到最优、第 8 个停，和作者表有 71.7 万个可训练参数同时出现，过拟合是一个猜测，没有验证。unsupported 目标的分组指标要等 verify 输出再看。
- F3-masked 比 F3 高 9 个命中（在 F3 的 seed 间波动范围内）：这一次没有看到“让 unsupported 历史事件可见”带来增益，F3 的增益主要来自 supported 事件。方向性结论，不进入选择。
- F5 未完成，不下结论。

### GFEAT-01 seed42 + FEAT-02 F5 终值（2026-10-14 18:4x，**数字来自 workflow 轮询通知，4 位小数；manifest 精确值与 verify 结果待报告补写，未验收**）
**G 线 GFEAT-01**（代码 gfeat01/g e4d1118，服务器 /workspace/vkgr-gfeat01-g；其余条件同 G anchor g-hist-v2-l50-s42：observed history L50，init small，100/5/3，batch=1 批量解码训练；validation n=2520，OOV 623 计入分母，test 未读）。
| Run | 变量 | R@50 (hits) | ΔR@50 vs anchor 0.0754 (Δhits) | 状态 |
|---|---|---|---|---|
| none-ctrl | 新入口、不加特征 | 0.0754 (190) | 0（复现 anchor） | trained；verify 待跑 |
| F2 feedback | encoder 侧加反馈 side | 0.0806 (203) | **+0.0052 (+13)** | trained；verify 运行中 |
| F3 watch buckets | encoder 侧加观看时长/比例分桶 | 0.0802 (202) | **+0.0048 (+12)** | trained；verify 已入队 |
命中数由 4 位小数反推（每个 R@50 对应唯一的 k/2520）。解读（如实）：
- none 对照与 anchor 的 R@50 一致（4 位小数），说明新入口没有引入偏差；是否逐位一致要看 manifest 精确值。
- F2、F3 都超过 ≥0.002 门槛，按卡补 seed 2024/2025（不计名额）已排。**但 G 的 seed 间波动本身就大**：anchor 三个 seed 为 0.0754 / 0.0718 / 0.0825（极差 0.0107），+0.005 只相当于一个 seed 的波动，单 seed 不能下结论。
- P 线同一类特征（F3）在 P 上 seed42 为 +0.0083，G 上同向、幅度更小。

**P 线 FEAT-02 更新**：F5 scene 终值 R@50 0.1560（393 hits，+1 hit vs F3 的 392，ΔR@50 +0.0004，按卡为 inconclusive，不补 seed）。F3-masked / F4 / F6 的 verify 已 succeeded，F5 verify 运行中。其余数字同上一节（F4 0.16825、F6 0.14087、F3-masked 0.15913）。

### GFEAT-01 seed42 精确值补记（G，2026-10-14 18:58 读 manifest，收据 main05/kiro-b-gc-185832.stdout SHA a8c2f9be…；verify 输出内容尚未读）
| Run | R@50 (hits) | N@50 | best/stop epoch |
|---|---|---:|---|
| anchor g-hist-v2-l50-s42 | 0.075397 (190) | 0.025775 | 10/13 |
| none-ctrl | 0.075397 (190) | 0.025775 | 10/13 |
| F2 feedback | 0.080556 (203) | 0.026194 | 14/17 |
| F3 watch buckets | 0.080159 (202) | 0.025513 | 14/17 |
- none-ctrl 与 anchor 的 R@50、N@50 到浮点全精度一致，best/stop epoch 相同，13 个 epoch 的 metrics 相同。逐 epoch 比较中，已打印的前 6 个 epoch 的差异都只是 checkpoint_sha256，model_sha256 一致；总差异数 13 正好等于 epoch 数，与“每个 epoch 只有 checkpoint 文件 hash 不同”吻合（其余 7 个 epoch 没有逐条打印）。checkpoint 文件 hash 不同的原因没有查（可能是文件里带了 config 元数据）。
- F2 的 N@50 比 anchor 高 0.00042；F3 的 R@50 多 12 个命中，但 N@50 反而低 0.00026。增益主要体现在 R@50，top-50 内部的排序质量没有明显改善。
- 上一节里“来自轮询通知”的 4 位小数值与这里的精确值一致；轮询步骤的一条备注把 F2/F3 的差写成 +0.0152/+0.0148，是笔误，应为 +0.0052/+0.0048（G 线 report 已更正）。

### FEAT-02 验收补记（P 线，seed42，validation，2026-10-14；素材 .agents/tasks/feat02-report.md，其数字取自服务器 manifest；4 个 run 全部 verified）
- 验收：4 个 run 的 verify job 都 succeeded，stdout 为 `verified`（只重放 best epoch）。这是第一批在服务器 GPU 训练产物上做的 verify；verify job 实际用的设备没有直接读到，“在 GPU 上”是推断。review：feat02-review.json APPROVED（0 HIGH/MEDIUM，5 LOW）。
- F5 scene 终值：R@50 0.15595 (393) / N@50 0.04433，best/stop 10/13，ΔR@50 vs F3 +0.00040 (+1)，按卡为 inconclusive，不补 seed。
- N@50 相对 F3 的变化：F3-masked +0.00146，F4 +0.00522，F5 +0.00008，F6 −0.00381。
- F4 在 13 个可比 epoch 中有 12 个高于 F3（epoch 2–13）；F3 的 R@50 在 epoch 10→11 掉了 0.0115，所以 epoch 间波动约 ±0.01，大于 0.002 门槛。F4 的 +0.0127 明显大于这个波动，F3-masked 的 +0.0036 和 F5 的 +0.0004 则在波动范围内。

target 切片 R@50（括号为 N@50；supported n=1666 / unsupported n=231 / OOV n=623）：
| run | supported | unsupported | OOV |
|---|---:|---:|---:|
| N1 | 0.22269 (0.06031) | 0 | 0 |
| F3 | 0.23529 (0.06692) | 0 | 0 |
| F3-masked | 0.24070 (0.06914) | 0 | 0 |
| F4 | 0.25450 (0.07482) | 0 | 0 |
| F5 | 0.23589 (0.06705) | 0 | 0 |
| F6 | 0.21008 (0.06003) | 0.02165 (0.00814) | 0 |
- F3 起所有收益都来自 supported 目标；OOV 目标在所有变体里都是 0（OOV 不在冻结 catalog，检索不到；R@50 的上限是 1897/2520 = 75.3%）。F6 的 −37 个命中 = supported −42 + unsupported +5。
- 历史切片（按 query 历史里 supported item 占比 f）：h0（f=0）validation 里 n=0；h_lt_half n=13；h_ge_half n=2281；h1 n=226。F3 / F3-masked / F4 / F5 / F6 在 h_ge_half 的 R@50 分别为 0.1486 / 0.1512 / 0.1596 / 0.1482 / 0.1333，在 h1 为 0.2301 / 0.2434 / 0.2611 / 0.2389 / 0.2212。FEAT-01 卡预期的“增量主要落在 h0/h_lt_half”在这份 validation 上无法检验（h0 为空，h_lt_half 只有 13 个 query）。
- 零向量：所有 run 的用户零向量占比都是 0（validation 里没有“历史全是 unsupported”的 query）；item 侧零向量 F3-masked/F4/F5 为 98919（82.1%，等于 unsupported 个数），F6 为 0。

F6 诊断（best epoch）：top-50 里 unsupported item 占 12.3%（F3-masked/F4/F5 为 0）；unsupported item 的原始范数均值 0.2521 对 supported 0.4867（F3-masked 的 supported 为 0.3537）。作者在词表里的占 catalog 的 35.32%。与 FEAT-02 计划风险 R1（unsupported 没当过 in-batch 负样本、分数未校准、挤占 top-50）**一致，但这份数据分不开原因**：F6 同时改了 supported 的向量和历史侧向量，F6-hist-only 的拆分对照没有做。F6 的曲线在 epoch 1 最高（0.1238）、epoch 4 最优、epoch 8 早停。

F3-masked 的 2×2（只给方向，两个因素不可加）：
| | unsupported 历史不可见 | unsupported 历史可见 |
|---|---:|---:|
| 无时长信息 | N1 0.14722 | C0 0.14722 |
| 有时长信息 | F3-masked 0.15913 | F3 0.15556 |
dM = F3-masked − N1 = +0.01190 (+30)；dV = F3 − F3-masked = −0.00357 (−9)；C0 − N1 = 0。时长信息加在 supported 事件上拿到了 F3 增益的主要部分；让 unsupported 历史“可见”本身没有收益。

**F4 的机制限制（计划风险 R4）**：F4 是加性 embedding + 均值池化，位置项在均值池化后只是“历史长度 n”的函数，周间隔项只是“周间隔直方图”的函数，它们**不能让更近的 item 权重更大**；M4 变异（位置从序列开头数）在用户向量上不可见，就是这个性质的证据。所以 F4 为正不能解读成“近期加权有效”，更像是给用户向量加了“历史长度 + 活跃周分布”这类用户级信号。要检验真正的新近性，需要另一个变量（可学习的分桶池化权重），本批没做。

其他限制：单 seed；Δ 的显著性没做 paired 检验（逐 query predictions 有存档，未读）；best epoch 按 validation 选、数字略偏乐观；只在 L50/id_mean/N1 上测；verify 只重放 best epoch；mask 与作者词表依赖 full-train likes（对训练样本含其 target 之后的点赞，与 N1 已有的 embedding×supported 同量级，不是新引入，也不涉及 validation 信息）；真实 items_metadata.parquet 读取耗时/内存未读。

### F4 补 seed + 逐 query 配对分析（P 线，validation，2026-10-14 19:3x；素材 .agents/tasks/feat02-paired-report.md；收据 main05/kiro-b-f4s-193219.stdout SHA 1e10fb1b…）
代码 feat02/p 94eb0bc（s42）、feat02c/p 523d872（s2024/s2025，仅新增 2 个只差 seed 的配置）。validation n=2520（OOV 623 计入分母），test 未读。F4 s42 与 s2024 的 verify 已 succeeded；s2025 的 verify 待跑，以 workflow 报告为准。
| seed | N1 R@50 (hits) | F3 R@50 (hits) | F4 R@50 (hits) | F4−F3 (Δhits) | McNemar p（精确，双侧） | ΔR@50 95% bootstrap | F4 N@50 | ΔN@50 vs F3 |
|---:|---|---|---|---|---:|---|---:|---:|
| 42 | 0.14722 (371) | 0.15556 (392) | 0.16825 (424) | +0.01270 (+32) | 0.0024 | [+0.0048, +0.0206] | 0.04946 | +0.00522 |
| 2024 | 0.14722 (371) | 0.14960 (377) | 0.15992 (403) | +0.01032 (+26) | 0.0095 | [+0.0032, +0.0179] | 0.04625 | +0.00503 |
| 2025 | 0.14286 (360) | 0.15437 (389) | 0.17024 (429) | +0.01587 (+40) | 0.00011 | [+0.0079, +0.0238] | 0.04950 | +0.00626 |
| 均值 | 0.14577 | 0.15317 | 0.16614 | +0.01296（sd 0.0028） | | | 0.04840 | +0.00550 |
- F4 在三个 seed 上都比 F3 高，都过 ≥0.002 门槛；做 Bonferroni（×3）后 p = 0.0073 / 0.029 / 0.00033，仍都 <0.05。三个 seed 用同一批 2520 个 query，p 值不独立，不能合并。
- 自检：12 个 run 的逐 query 命中数、逐 query NDCG 均值都与 manifest 的 best_metric 对上（误差 <1e-9）。
- 历史长度分层（≤10 / 11–25 / 26–49 / ≥50）：≥50 的 query 占 62.7%，只贡献 3 个 seed 共 98 个增益命中中的 46 个（47%）；增益点估计最大的是 26–49（+0.022 / +0.025 / +0.036）和 51–99（+0.019 / +0.012 / +0.028），历史 ≥100（在 100 处饱和）基本没有增益（0.000 / +0.001 / +0.005，区间含 0）。独立 reviewer 提出的“增益主要来自占多数的 >50 条历史的 query”这个具体说法**没有被这份数据支持**。
- popularity：F4 的 98 个增益命中里 86 个（88%）落在 target 流行度最高一档，但 F3 自己的命中也有约 89% 在这一档，所以是“增益跟着命中走”（该档相对增益约 7–10%）。
- 其它（seed42）：F3-masked−F3 +0.00357，p=0.12，区间含 0；F5−F3 +0.00040，p=1；F6−F3 −0.01468，p=0.0008，区间不含 0（显著变差）。F3−N1：+0.00833（p=0.031）/ +0.00238（p=0.64，不显著）/ +0.01151（p=0.0082），F3 自己的增益对 seed 敏感；其净增益是交换：最热门一档显著变好，中等热度一档三个 seed 都显著变差（−0.018 / −0.040 / −0.018）。
- **限制**：机制没有归因。没跑 position-only / gap-only 对照，所以不能把 F4 称作“近期加权”，只能称“历史长度 + 时间跨度先验”；历史长度 >100 只能到 100 饱和；单 validation 窗口；best epoch 按 validation 选；没有统计推荐 item 的流行度。

### G 线 21:40 读数：A1 seed42 终值 + GFEAT-01C 补 seed（validation，收据 main05/kiro-b-gm-214002.stdout SHA 503d54e3…、verify 收据 main05/kiro-b-v-2140*.stdout）
n=2520（OOV 623 计入分母），test 未读。门槛按整数命中：ΔR@50 ≥ 0.002 等价于多 ≥6 个命中。

**A1（GARCH-01，论文规模 4+4 层、6×64 heads、d128、FF1024、dropout 0.1、LR 3e-4，seed42）训练终态**：R@50 **0.099206（250 hits）**，N@50 0.031776，best/stop 14/17。core03 anchor 是 0.075397（190），**多 60 个命中（+0.0238）**，远过门槛。A1 seed42 的 verify 尚未入队（由 GARCH-01 workflow 在训练终态后入队），**未验收**。A2（LR 1e-3）终值 0.070238（177），best/stop 6/9，低于 anchor。A1 seed2024/2025 已入队，21:40 时在训练（22 分钟）。按卡 0c 的预注册选择规则（seed42 判定），A1 verify 通过后即为 G anchor。

**GFEAT-01C 补 seed（G 线 F2 反馈 / F3 观看时长分桶，core03 上；相对同 seed 的 core03 anchor g-hist-v2-l50-s{42,2024,2025} = 190 / 181 / 208 hits）**：
| seed | anchor (hits) | F2 R@50 (hits) | F2 Δhits | F3 R@50 (hits) | F3 Δhits |
|---:|---:|---|---:|---|---:|
| 42 | 0.075397 (190) | 0.080556 (203) | +13 | 0.080159 (202) | +12 |
| 2024 | 0.071825 (181) | 0.075794 (191) | +10 | 0.086905 (219) | +38 |
| 2025 | 0.082540 (208) | 0.086111 (217) | +9 | 0.080556 (203) | −5 |
| 均值 | 0.076587 | 0.080820 | **+10.7（+0.0042）** | 0.082540 | **+15.0（+0.0060）** |
- F2：3/3 seed 都高于同 seed anchor，且每个都 ≥6 个命中，方向稳定，幅度约为 P 线同类特征（F3 −N1 均值 +0.0074）的一半多一点。
- F3：均值更大但不稳：seed2024 +38，seed2025 −5，单 seed 结论不可靠。
- verify：F2 三个 seed、F3 的 s42 与 s2025、none 对照都已 `verified`；F3 s2024 的 verify 在 21:40 读取时没有出现在列表里，之后 GFEAT-01C 报告（.agents/tasks/gfeat01c-seeds-report.md，22:1x）确认 4 个补 seed 的 verify 全部 `verified`。**至此 6 个 run 全部验收**（报告的数字取自 deploy 记录；与上表的 manifest 读数一致）。
- 限制：这些都在 core03 上。如果 A1 成为 G anchor，特征效果要在 A1 上重测（特征与模型规模可能有交互）；没有做逐 query 配对检验。

### A1 seed42 验收 + 补 seed 中间值（G 线，validation，2026-10-14 22:53 读取；收据 main05/kiro-b-v-2253*.stdout、kiro-b-gm-2253*.stdout）
- **A1 seed42 的 verify：`verified`**（g-garch01-a1-lr3e-4-s42-v1-verify，串行解码，只重放 best epoch 14，共 17 个 epoch）。训练终值 R@50 0.099206（250 hits）、N@50 0.031776，比 core03 anchor（0.075397，190 hits）多 60 个命中。至此 A1 seed42 验收完毕。
- 补 seed 训练中（均未终态，只是中间值，不能当结果）：A1 seed2024 当前最优 R@50 0.080952（204 hits）、N@50 0.027105，best epoch 10；A1 seed2025 当前最优 0.082937（209 hits）、0.027926，best epoch 9。对照同 seed 的 core03 anchor 181 / 208 hits。**提醒**：seed42 在第 10 个 epoch 时也只有 0.0921，最终到 0.0992，所以这两个值可能还会涨，但目前 seed 之间差别已经很大（seed42 远高于另外两个），A1 的领先幅度要等 3 个 seed 终态后才能下结论。

### A1 补 seed 终态（G 线，validation，2026-10-14 23:5x；数字来自 workflow 通知的 4 位小数，manifest 精确值与 verify 以 .agents/tasks/garch01c-seeds-report.md 为准；verify 运行中，**未验收**）
A1 = TIGER 论文规模（4+4 层、6×64 heads、d128、FF1024、dropout 0.1、LR 3e-4）。n=2520（OOV 623 计入分母），test 未读。命中数由 4 位小数反推。
| seed | core03 anchor R@50 (hits) | A1 R@50 (hits) | Δhits | A1 N@50 | best/stop |
|---:|---|---|---:|---:|---|
| 42 | 0.0754 (190) | 0.0992 (250，已 verified) | +60 | 0.0318 | 14/17 |
| 2024 | 0.0718 (181) | 0.0817 (206) | +25 | 0.0272 | 11/ |
| 2025 | 0.0825 (208) | 0.0929 (234) | +26 | 0.0286 | 11/ |
| 均值 | 0.0766 | 0.0913 | **+37（+0.0147）** | | |
- A1 在三个 seed 上都比同 seed 的 core03 多 ≥25 个命中，都远过 ≥6 个命中的门槛，方向一致。seed42 的领先最大（+60），另外两个 seed 约为一半；所以幅度的点估计要用三 seed 均值（约 +0.015，相对约 +19%），不要用 seed42 的 +0.024。
- A2（LR 1e-3）只有 seed42，0.0702，低于 core03，LR 1e-3 不用。

### P-HP-01 终态（P 线，F4 anchor 上的单旋钮调参，seed42，validation，2026-10-15 00:0x；素材 .agents/tasks/php01-report.md）
代码 php01/p 587198e（base = feat02/p 94eb0bc）。validation n=2520（OOV 623 计入分母），test 未读。6 个训练 + 6 个 verify 全部 succeeded / verified。门槛：ΔR@50 ≥ 0.002 ⇔ ≥6 个命中。
| run | 旋钮 | R@50 (hits) | N@50 | best/stop | ΔR@50 vs F4 | Δhits | 过门槛 |
|---|---|---|---:|---|---:|---:|---|
| F4（anchor，已有） | T 0.15 / dim 64 / LR 1e-3 | 0.16825 (424) | 0.04946 | 12/15 | — | — | — |
| H1 | T 0.10 | **0.17421 (439)** | 0.05144 | 10/13 | +0.00595 | +15 | 是 |
| H2 | T 0.25 | 0.14603 (368) | 0.04265 | 10/13 | −0.02222 | −56 | 否 |
| H3 | dim 32 | 0.16429 (414) | 0.04923 | 12/15 | −0.00397 | −10 | 否 |
| H4 | dim 128 | 0.16905 (426) | 0.05067 | 12/15 | +0.00079 | +2 | 否 |
| H5 | LR 0.002 | 0.16706 (421) | 0.04914 | 10/13 | −0.00119 | −3 | 否 |
| H6 | LR 0.0005 | 0.16349 (412) | 0.04768 | 17/20 | −0.00476 | −12 | 否 |
| H7 | 组合 | — | — | — | — | — | **void**（只有温度一个旋钮有赢家，条件组合未触发，未训练） |
- 预注册选择规则（R@50 → N@50 → 枚举序）的 winner = H1；排名 H1 > H4 > F4 > H5 > H3 > H6 > H2。
- **H1 vs F4 逐 query 配对（seed42）**：both 371，仅 H1 命中 68，仅 F4 命中 53，都不中 2028。McNemar 精确双侧 **p = 0.203**；ΔR@50 +0.00595，95% bootstrap CI [−0.0024, +0.0143]（跨 0）；ΔN@50 +0.00198，CI [−0.0004, +0.0044]（跨 0）。历史长度分层均跨 0（≤10 −0.0365，11–25 +0.0056，26–49 +0.0157，≥50 +0.0070）。
- **解读**：H1 过了预注册门槛，但不是统计意义上的证据。F4 三个 seed 的 R@50 样本标准差约 0.0055，与 H1 的 +0.0060 同量级，单 seed 分不开。H2（T 0.25）显著更差（−56 hits），说明温度方向是有信号的：0.25 变差，0.10 可能变好，0.15 附近可能不是最优。
- 限制：单 seed；只做了 H1 vs F4 的配对，其他 run 没有做；H4 显存只在合成数据上冒烟（max_reserved 456 MiB，≤3072 声明），真实训练峰值未读；H1 的 N@50 增益只有 bootstrap CI；dim 与温度的交互没有测（OFAT）。

### A1 三个 seed 验收完成 + 精确值（G 线，validation，2026-10-15 00:2x；素材 .agents/tasks/garch01c-seeds-report.md；覆盖上一节的 4 位小数与“未验收”）
n=2520（OOV 623 计入分母），test 未读。数字来自各 run 的 manifest `selection.best_metric`。**三个 seed 的 verify 均 `verified`**（s2024、s2025 的 replayed_epochs 都是 [11]）。
| seed | A1 R@50 (hits) | core03 anchor R@50 (hits) | ΔR@50 | Δhits | A1 N@50 | best/stop | verify |
|---:|---|---|---:|---:|---:|---|---|
| 42 | 0.0992063 (250) | 0.0753968 (190) | +0.0238095 | +60 | 0.0317760 | 14/17 | verified |
| 2024 | 0.0817460 (206) | 0.0718254 (181) | +0.0099206 | +25 | 0.0272094 | 11/14 | verified |
| 2025 | 0.0928571 (234) | 0.0825397 (208) | +0.0103175 | +26 | 0.0285864 | 11/14 | verified |
| 均值 | 0.0912698 | 0.0765873 | **+0.0146825（约 +37 hits）** | | | | |
- 样本 sd（n−1）：A1 0.0088377，anchor 0.0054554，成对差 Δ 0.0079067。
- 三个 seed 都过 ≥6 hits 门槛，方向一致；幅度用三 seed 均值（+0.0147），不用 seed42 的 +0.0238。
- 限制：只有 3 个 seed；seed42 是 A1 的筛选 seed（A1 因它的结果被选为 anchor），有选择偏差，另外两个 seed 的增益约为它的一半；没有做逐 query 配对检验。

### GTOK-01 T1/T2（RQ-VAE）拟合 gate_failed（G 线，2026-10-15 01:5x；读取 g-tok-t{1,2}-fit/out/failure.json，prefix main05/kiro-b-to-0020）
T1 = SID_RQVAE_B256D3_S42，T2 = SID_RQVAE_B16D3_S42。两个拟合都跑满 300,000 步（R2 通过；wall 4354 s / 4302 s），H1–H6 全过，**R1、R3、R4 失败 → status=gate_failed**，按预注册不进训练。
- R1 每层利用率：两者的实测 distinct 都是 [1, 1, 2]（T1 的 base 256、T2 的 base 16），门槛为 ≥80%。
- R3 重建 R²：T1 −7.519e-5，T2 −7.519e-5 量级（≈0，略为负）；R4 逐层残差范数没有递减。逐层 explained_variance：T1 {1: 0.0, 2: 0.0, 3: 4.15e-05}，T2 {1: 0.0, 2: 0.0, 3: 3.61e-05}。
- 损失曲线：两者在第 3000 步后都停在约 0.811，与 K（256 或 16）无关，到 30000 步仍是 0.8110x，没有继续下降。第 1000/2000 步有一次抖动（T1：0.820 → 1.010 → 0.812）。
- **读法（推断，未验证）**：K 相差 16 倍而曲线几乎相同，说明不是码本大小的问题，更像是整个拟合都塌到“预测均值”的状态（codebook collapse / encoder 死掉）。原因没有查，候选包括预注册的 Adagrad lr 0.4（卡里写死、决定“不改”）、输入尺度、初始化。不能据此说“RQ-VAE 不适合这份内容特征”。
- 对 T3（BHKM b256）不适用 R 类门禁，已按预注册放行并在 core03 上入队训练；它的第 3 位只用到 256 个取值里的 2 个，也有退化迹象。
- 后果：G6 的“RQ-VAE 4 层”只对应 T1，现已不可得，G6 回到 BHKM 16 叉 4 层的后备方案。

### FEAT-03B 终态（P 线，F4 机制拆分 + 可学习新近性 + 用户级活动，三 seed，validation，2026-10-15 04:5x；素材 .agents/tasks/feat03-report.md；预注册读法见冻结卡 §10）
代码 feat03/p 757bde2（seed 2024/2025 的配置在 feat03c ab53a8d，只差 seed）。validation n=2520（OOV 623 计入分母），test 未读。12 个训练 + 12 个 verify 全部 succeeded/verified；Q2、Q3 先过 CUDA 预演（含 best checkpoint 重导出精确相等）。报告里的数字是从逐 query 存档重算的，18 个 run 的 11 项自检与 manifest 的 best_metric 全部对上。
| 变体（相对谁） | seed 42 R@50 (hits) | seed 2024 | seed 2025 | 三 seed 均值 hits | 相对 F3 的 Δhits | 相对 F4 的 Δhits |
|---|---|---|---|---:|---:|---:|
| F3（对照，已有） | 0.15556 (392) | 0.14960 (377) | 0.15437 (389) | 386.00 | — | −32.67 |
| F4（对照，已有） | 0.16825 (424) | 0.15992 (403) | 0.17024 (429) | 418.67 | +32.67 | — |
| Q1a F3+只加位置桶 | 0.15714 (396) | 0.15238 (384) | 0.14802 (373) | 384.33 | −1.67 | −34.33 |
| Q1b F3+只加周间隔桶 | 0.16587 (418) | 0.16706 (421) | 0.16151 (407) | 415.33 | **+29.33** | −3.33 |
| Q2 F4+可学习新近性池化权重 | 0.17341 (437) | 0.17659 (445) | 0.17341 (437) | **439.67** | +53.67 | **+21.00** |
| Q3 F3+用户级活动桶（长度+活跃周数） | 0.15754 (397) | 0.15635 (394) | 0.16230 (409) | 400.00 | +14.00 | −18.67 |
均值 R@50：Q1a 0.15251，Q1b 0.16481，Q2 0.17447，Q3 0.15873（F3 0.15318，F4 0.16614）。
- **Q1（预注册读法表第 2 行）**：位置桶单独加入没有可检出增益（均值 −1.67，N），周间隔桶单独加入拿到 +29.33（三个 seed 都 ≥6 hits，C）；A+B=27.67，落在“近似可加”档（24.67–40.67）。允许措辞：“活动时间跨度（周间隔分布）先验”。不得写“近期加权”“共同贡献”。
- **Q2（第 2 行）**：D2=+21.00，k₊=3，p_pool=0.0014；形状偏新近（S_pos +1.06/+0.86/+1.25，S_gap +0.70/+0.71/+0.66，三个 seed 都 ≥ ln1.5）。允许措辞：“可学习新近性加权带来增益”，**必须附**：权重同时乘在 F4 的 P、G 加性项上，增益可能来自对这两项的重新加权，不一定是内容部分的近期加权。
  - 逐 seed Q2−F4：ΔH +13/+42/+8，exact McNemar p = 0.309 / 0.0001 / 0.539；只有 seed 2024 过 0.05。
  - 轨迹旗标：三个 seed 的 best epoch 前 logit 仍在移动（最大绝对变化 0.187/0.214/0.151 > 0.1），预算受限，增益可能被低估，不能写成已收敛。
  - 作 P anchor 的预注册条件（§7.2）：至少 2 个 seed 的 Q2−F4 p<0.05 ✗（只有 1 个）；对新池化代码的独立泄漏 review 未做。**不满足，不换 anchor。**
- **Q3（第 4 行）**：D3=+14.00，ρ=0.429，p_pool=0.0118，相对 F3 有效，ρ 离 13.07 的分界只高 0.93 个命中；seed 42 单 run 只有 +5。对 F4 的 E=−18.67，“劣于 F4”。允许措辞：“取得了 F4 增益的 40%–80%；说明增益量级约为 F4 的一半上下，不说明它们捕捉了同一部分信息”。增益 query 重合度（描述性）0.536/0.550/0.417。
- **限制**：三个 seed 评同一批 2520 个 query，p_pool 偏乐观，只是必要条件；结论仅限 P 线、L50、id_mean、N1、validation week 25；这一步没有做独立 fresh review（subagent 无法再派生）。

### G-HP-01 终态（G 线，A1 锚点上的单旋钮调参，seed42，validation，2026-10-15 05:0x；素材 .agents/tasks/ghp01-report.md）
代码 ghp01/g 4784686（overlay 部署，base 0376d9a）。n=2520（OOV 623 计入分母），test 未读。5 个训练 + 5 个 verify 全部 succeeded/verified，没有撞 100 epoch 上限；门槛 ΔR@50 ≥ 0.002 ⇔ ≥6 hits。
| run | 旋钮 | R@50 (hits) | N@50 | best/stop | Δhits vs A1 | ΔN@50 | 过门槛 |
|---|---|---|---:|---|---:|---:|---|
| A1（anchor） | d128、LR 3e-4、4+4 层、dropout 0.1 | 0.0992063 (250) | 0.0317760 | 14/17 | — | — | — |
| G1 | 宽度 ×2（d256，参数量约 3.97×） | **0.1019841 (257)** | 0.0305114 | 13/16 | **+7** | −0.00126 | 是 |
| G2 | LR 1e-4 | 0.0857143 (216) | 0.0296343 | 16/19 | −34 | −0.00214 | 否 |
| G3 | LR 6e-4 | 0.0797619 (201) | 0.0291957 | 5/8 | −49 | −0.00258 | 否 |
| G4 | 2+2 层 | 0.0849206 (214) | 0.0280070 | 11/14 | −36 | −0.00377 | 否 |
| G5 | dropout 0 | 0.0869048 (219) | 0.0269041 | 6/9 | −31 | −0.00487 | 否 |
- 预注册选择规则（R@50 → N@50 → 枚举序）下候选中 R@50 最高的是 G1；N@50 上 A1 仍最高。
- **组合 G7（provisional，只判 G1–G5）：不触发**，赢的旋钮只有宽度 1 个（需要 ≥2 个不同旋钮）。G6（SID 深度 +1）依赖 GTOK-01，未入队；`scripts/ghp01_combo.py` 把 run id 写死为 G1–G5，要判 G6 必须先扩展脚本。
- G1 总墙钟 3.53 h（train 7404 s + evaluate 5216 s），24 h 上限没有撞；每 epoch 约 795 s。
- 限制：单 seed。A1 自己三个 seed 的 R@50 是 0.0992 / 0.0817 / 0.0929（250 / 206 / 234 hits，sd 约 22 hits），G1 的 +7 hits 远小于这个 seed 间散布；N@50 反而更低；G1 的训练过程中实际 GPU 峰值没有再测。

### GFEAT-02 终态（G 线，A1 上复测 F2/F3，seed42，validation，2026-10-15 06:5x；素材 .agents/tasks/gfeat02-report.md）
代码 gfeat02/g afaa9db。n=2520（OOV 623 计入分母），test 未读。预检通过，3 个训练 + 3 个 verify 全部 succeeded/verified。
- **闸门通过**：none-ctrl 的 selection 与 A1 seed42 逐字段相同（best_metric [0.0992063492063492, 0.03177597666987997]、best 14、stop 17、bad_epochs 3），17/17 个 epoch 的 model_sha256 / computation_sha256 / metrics 逐个相同；F2/F3 结果有效、可解读。
| run | R@50 (hits) | N@50 | best/stop | ΔR@50 vs none-ctrl | Δhits | 预注册判定 |
|---|---|---:|---|---:|---:|---|
| none-ctrl（= A1 s42） | 0.0992063 (250) | 0.0317760 | 14/17 | — | — | — |
| F2 反馈 | 0.0988095 (249) | 0.0312760 | 14/17 | −0.000397 | −1 | inconclusive |
| F3 时长/比例分桶 | 0.0912698 (230) | 0.0305290 | 9/12 | −0.007937 | −20 | **有害**（≤ −0.002） |
- 没有特征有效，不选入选者，不补 seed。G 线特征名额：F2、F3（core03）+ F2、F3（A1）共 4 个已用。
- 事后观察（不参与判定）：同一条曲线 epoch 间波动可达 ±0.008（约 ±20 个命中），单个 best-epoch 数字对“取到哪个 epoch”很敏感；F2 在 epoch 5–12、15–17 多数时候高于 none-ctrl，只是 best epoch（14）上低 1 个命中，单 seed 分不清是小幅正效应被选择噪声盖住还是噪声。
- 限制：单 seed；F3 的 stop epoch 12 / best 9，是否因 patience 3 早停而吃亏未验证（patience 3 是 A1 预注册设置，不改）。
- **与 core03 的对比**：core03 上 F2 三个 seed 都 +9 到 +13 个命中、F3 均值 +15（但不稳）；搬到 A1（更大的模型）后 F2 ≈ 0、F3 −20。特征效果与模型规模有交互，core03 上的特征结论不能直接搬到 A1。

### GTOK-01R：RQ-VAE 重试，T1r / T2r 拟合通过全部门禁（G 线，2026-10-15 08:xx；素材 .agents/tasks/gtok01r-server-report.md，来自 coder 通知，未逐项核对）
代码 gtok01r/g，公开分支 gtok01r-pub/g 80f7eed。探针 10 个（P0 对照 + 4 个臂 × K=256/16，各 30000 步），P0 复现旧塌缩（逐层利用率 [1,1,2]，对照有效）；按预注册规则（L1→L4 顺序取第一个在两个 K 上都合格的臂）选 **L2 = Adagrad lr 0.1（加死码重置）**；L1（原 lr 0.4 + 死码重置）不合格。探针实测显存峰值 536 MiB。
| 变体 | 臂 | 逐层利用率（used/base） | 重建 R² | wall | 门禁 H1–H6、R1–R4 |
|---|---|---|---:|---:|---|
| T1r（RQ-VAE 256×3） | L2 | [256, 256, 256] | 0.655 | 4329 s | 全过 |
| T2r（RQ-VAE 16×3） | L2 | [16, 16, 16] | 0.450 | 4283 s | 全过 |
- 对照旧结果：T1、T2 的利用率是 [1, 1, 2]、R² ≈ 0；改用 lr 0.1 加死码重置后三层都用满，R² 0.655 / 0.450。
- **读法（推断）**：旧塌缩的主因更像是预注册写死的 Adagrad lr 0.4 过大；原代码已有的 k-means 初始化加死码重置、lr 0.1 的组合有效。探针只测了 30000 步，完整拟合 30 万步的门禁已通过，这一点是实测。没有单独测“只加死码重置、不降 lr”这一项在完整拟合上的效果（L1 在探针里不合格）。
- verify 与下游 core03 训练尚未完成，等结果再记。

### H1（F4 + 温度 0.10）三 seed 补完 + 逐 query 配对（P 线，validation，2026-10-15 10:3x；脚本 main05/php01c_paired_read.py，收据 kiro-b-pp-0033；H1 补 seed 的代码 php01c/p 9f543c1）
n=2520（OOV 623 计入分母），test 未读。自检全部 PASS（命中数由逐 query 预测重算，与 manifest 一致）。H1 的 seed 2024/2025 verify 已不在队列中（推断已完成，未逐个读 stdout）。
| seed | F4 (T0.15) R@50 (hits) | H1 (T0.10) R@50 (hits) | Δhits | N@50 F4 → H1 | best/stop (F4 → H1) | McNemar 精确双侧 p | ΔR@50 95% bootstrap |
|---:|---|---|---:|---|---|---:|---|
| 42 | 0.16825 (424) | 0.17421 (439) | +15 | 0.04946 → 0.05144 | 12/15 → 10/13 | 0.2029 | [−0.0024, +0.0143] |
| 2024 | 0.15992 (403) | 0.17222 (434) | **+31** | 0.04625 → 0.05087 | 7/10 → 7/10 | **0.00088** | [+0.0052, +0.0194] |
| 2025 | 0.17024 (429) | 0.17619 (444) | +15 | 0.04950 → 0.05286 | 11/14 → 11/14 | 0.1284 | [−0.0012, +0.0131] |
| 均值 | 0.16614 (418.67) | 0.17421 (439.00) | **+20.33（+0.0081）** | 0.04840 → 0.05172 | | | |
- 三个 seed 的 Δ 都为正且都 ≥6 hits（过 0.002 门槛）；逐 seed 显著的只有 seed 2024。合计 b/c = 175/114（三个 seed 评同一批 query，不独立，p 值只是描述）。
- H1 是从 6 个旋钮里按 seed42 选出的（赢者诅咒），但 seed 2024/2025 没参与选择，仍然 +31 / +15，方向复现。
- 历史长度分层（三 seed）：≥50 条历史的 query 在三个 seed 上增益点估计为 +0.0070 / +0.0082 / +0.0101，符号一致；≤10 条的短历史在 seed 42/2025 为负（−0.0365 / −0.0073），区间都含 0。
- 限制：只测了温度这一个旋钮的两个取值（0.10、0.15）；没有 0.05 或 0.08，最优温度可能更低；温度与 dim、LR 的交互没测（OFAT）。

### GTOK-01 core03 tokenizer 筛选（G 线，core03 小模型，seed42，validation，2026-10-15 10:3x；训练终值读自 manifest selection.best_metric，收据 kiro-b-bm-0032）
n=2520（OOV 623 计入分母），test 未读。core03 = d128、1+1 层、LR 3e-4；对照 core03 anchor（BHKM 16×3，SID_BHKM_B16D3_S42）seed42 = 0.0753968（190 hits）。
| tokenizer | R@50 (hits) | N@50 | best/stop | Δhits vs core03 anchor | verify |
|---|---|---:|---|---:|---|
| T3 BHKM 256×3 | **0.1007937 (254)** | 0.0315363 | 10/13 | **+64** | 驱动脚本记录 verify succeeded（wall 864 s），未读 stdout 的 verified 字样 |
| T1r RQ-VAE 256×3（L2 臂：Adagrad lr 0.1 + 死码重置） | 0.0924603 (233) | 0.0314094 | 10/13 | +43 | 运行中 |
| T2r RQ-VAE 16×3（L2 臂） | 0.0595238 (150) | 0.0182698 | 5/8 | −40 | 运行中 |
- 原 T1、T2（照搬预注册 lr 0.4）拟合门禁失败，没有训练，见前面的 gate_failed 一节。
- **主要读法**：把 SID 从 16 叉 3 位换成 256 叉 3 位，core03 小模型就从 190 涨到 254 hits（+64），追平 A1（250）。T1r 同样是 256 叉，也有 +43。T2r 是 16 叉，反而比 BHKM 16 叉差（−40）。说明 tokenizer 的分辨率（256 叉）比聚类方法（k-means 还是 RQ-VAE）对召回影响更大；256 叉 BHKM 比 256 叉 RQ-VAE 多 21 个命中，小于 A1 自己的 seed 间 sd（约 22 hits），不能据此断言 BHKM 优于 RQ-VAE。
- 机制没有归因：256 叉带来的好处可能来自叶子碰撞更少（16^3=4096 个叶子对 12 万个 item），这是推断，未做拆分实验。T3 的第 3 位只用了 256 个取值里的 2 个，T2r 的最大叶子 203 对旧 k-means 的 30，都说明分布不均，但对召回的影响没测。
- 限制：单 seed；T3、T1r、T2r 是 core03 规模，不是 A1；T3 的 seed 2024/2025 和 A1+T3 的确认 run 已在排程。

- 补记（2026-10-15 11:xx，素材 .agents/tasks/gtok01r-server-report.md）：T1r、T2r 在 core03 上的训练各有 1 个 verify，2 个训练对应 2 个 verify，全部 verified。上面 tokenizer 筛选表中 T1r、T2r 的“运行中”改为 verified；数字不变（T3 254 > T1r 233 > anchor 190 > T2r 150）。

### GTOK-01C：T3（BHKM 256×3）core03 补 seed 2024/2025（G 线，validation，2026-10-15 13:xx；素材 gtok01c 工作流通知，未读报告）
代码 gtok01c-pub/g 2bc1c28（只新增 2 个只差 seed 的配置）。n=2520（OOV 623 计入分母），test 未读。两个训练的 verify 均 verified。
| seed | T3 core03 R@50 (hits) | core03 anchor（BHKM16×3）(hits) | Δhits |
|---:|---|---:|---:|
| 42 | 0.1007937 (254) | 190 | +64 |
| 2024 | 0.1087302 (274) | 181 | +93 |
| 2025 | 0.1091270 (275) | 208 | +67 |
| 均值 | **0.1062169 (267.67)** | 193 | **+74.67** |
- 三个 seed 都远超 6 hits 门槛，方向一致，增益 +64 到 +93，不是 seed42 的偶然。
- **对照 A1**（论文规模 4+4 层，BHKM16×3）三 seed 均值 0.0912698（230 hits：250/206/234）：core03 小模型 + 256 叉 SID 的均值已高于 A1 + 16 叉 SID（+37.67 hits）。说明对这份数据，SID 的分辨率（256 叉）比模型规模更重要（推断，A1+T3 的确认 run 尚未出，不能说两者可叠加）。
- 限制：3 个 seed 评同一批 2520 个 query；N@50、best/stop 未读；T3 的第 3 位只用到 256 个取值中的 2 个这一退化迹象仍在，对召回的影响没拆分。

### Codex 接手阶段一读数（2026-10-15 16:32–16:34+08，只读；未验收结果）
validation n=2520（OOV 623 计入分母）；test 未读。以下只记录已结束训练的 selection，不代表 verify 通过，不据此改变 anchor。
来源：`main05/codex-takeover-php02-20261015T163540.stdout`（实际读取 16:32:28+08，SHA256 `38986157a2f9f678ffb884bc94eab10fc0c22368cfb8928cc1e207fd0ef96684`）；对应 transport exit=0、readonly=true。文件名前缀不是实际观测时间，时间取收据内容/本机收据 mtime。
| run_id | R@50 (hits) | N@50 | best/stop | train wall (s) | verify（同次读取） |
|---|---|---:|---|---:|---|
| `p-php02-t005-s42` | 0.1615079 (407) | 0.0450115 | 6/9 | 628.0 | claimed，未验收 |
| `p-php02-t005-s2024` | 0.1623016 (409) | 0.0454254 | 7/10 | 693.3 | pending，未验收 |
| `p-php02-t005-s2025` | 0.1650794 (416) | 0.0455817 | 7/10 | 696.1 | pending，未验收 |
| `p-php02-t007-s42` | 0.1702381 (429) | 0.0488967 | 9/12 | 843.1 | pending，未验收 |
| `p-php02-t007-s2024` | 0.1698413 (428) | 0.0491079 | 7/10 | 681.6 | 尚未入队（原驱动负责），未验收 |
- 六个 php02 训练均已独立读到 `claimed`，对应 rejected/.reason 均无记录；t007-s2025 仍在训练，未抄录其中间质量指标。四个 verify 已入队（一个 claimed、三个 pending）；其余两个由 pid 32766 原驱动补齐，不人工重复提交。
- A1+T3 seed42（`g-gtok01b-t3-b256d3-s42`）：terminal succeeded/exit0；manifest complete；R@50 **0.1146825（289 hits）**，N@50 **0.0347788**，best/stop **8/11**，train wall **5390.7 s**。对应 `-v1-verify` 尚无 run 目录，未验收；两个补 seed 仍在运行，不作三 seed 或 anchor 结论。来源：`main05/codex-takeover-a1s42-20261015T163800.stdout`（实际收据 16:33:36+08，SHA256 `3ce421659085cd2caf503a6a3f4e4df54f8a9ae53097403efa1a3d88db9cfad6`；stdout manifest SHA `0672ba747178756e8d872c0d5018dc90be3a6c12b9af74a237826558e6bbd3ae`）。
- T1r seed2024/2025 尚无 terminal，不能记最终结果。来源：`main05/codex-takeover-t1r-20261015T163800.stdout`，SHA256 `30ab1445e1de67aef2c1d2b8341bc2d2008f23817819bed28067d025567ec388`。
- 全部 transport/source/stdout hashes 与实际收据时间已本地核对，索引：`cache/codex-takeover-20261015-stage1/receipt-index.json`。局限：本轮没有重新执行科学 verify；没有读取 week26、checkpoint、完整 embedding 或逐 query 预测；P/G 命中并集预检尚未执行。

- 接收收口复核（16:39:03+08）：php02 六训练仍为五 succeeded、一 running；t005-s42-verify、t005-s2024-verify 的 supervisor terminal 均 succeeded/exit0，wall 250.2/271.6s；stdout 的 `verified` 字样本轮未直接读取，**仅确认 verifier 进程成功退出，仍不写科学验收完成**。t005-s2025-verify 已 claimed/running，t007-s42-verify 仍 pending，另两个 t007 verify 尚未入队。来源 `main05/codex-takeover-final-php02-20261015T163920.stdout`，SHA256 `e7667924c651b8d752792f7be28f77a6f1f9d177bfec7a9e5c66f1b998fcb398`。A1+T3 s42 的原驱动16:36:57已写verify，16:39独立只读确认在pending、无对应rejected/.reason，**尚未claimed**；来源 `main05/codex-takeover-final-a1-20261015T163920.stdout`，SHA256 `b177f5e5bc50cfe0c38fd888a8e4d352b387d2f1a95e92023bb5f365d10846ed`。不重复提交。

## Codex 阶段二第一步：已验收读数、机械判定与 G5 零成本预检（2026-10-15）
本节仅读数/记账，不改任何 anchor、不冻结新规则、不提交训练。validation n=2520，非OOV 1897，OOV 623保留分母；没有读test、checkpoint或完整embedding。
主收据 S1：`main05/codex-p2-verified-audits-20261015T190500.stdout`，SHA256 `597afa53b8d4131188c2b07fe55e9c8f2044d777cff7f49f510ea5c39deadca9`，实际服务器读取 2026-10-15T11:04:55.248335+00:00 至 2026-10-15T11:05:01.620417+00:00（UTC；北京时间+8h）。
分析代码：已push工作分支 `codex/phase2-readonly-tools`，commit `3235a15`，transport的source SHA与该Git文件一致。验收要求：train succeeded、verifier terminal succeeded/exit0，且直接读取 stdout/out manifest 中精确 `status=verified`；再将claimed verify job的`--manifest/--manifest-sha`绑定到当前训练manifest。不能用supervisor exit0代替。具体verify路径、stdout hashes、manifest hashes、队列job hashes及全部被读artifact hashes见S1与 `cache/codex-phase2-step1-20261015/receipt-index-v2.json`。
档案格式已先核源码和设计：P为manifest.evaluations中best epoch的`predictions-epoch-NNN.json`；G为best epoch的`evaluation-NNN/manifest.json`和固定范围shards。P采用无newline的JSON digest且query包含side字段，G采用canonical+newline；本脚本按各自规范核对，不错误要求两者feature-side query hash相等。共同原validation pin、history-v2 manifest、catalog文件与两种digest均核对；G每行ordinal/完整query SHA、shard与computation SHA核对；每个可用archive的R@50/N@50由真实item top50重算一致（<1e-9）。不读取张量或重导出模型。

### 1. php02 温度精扫（六训练与六verify均verified；来源S1）
| run_id | R@50（精确值） | hits | N@50（精确值） | best/stop | train / verify wall秒 | 科学验收 |
|---|---:|---:|---:|---|---|---|
| `p-php02-t005-s42` | 0.16150793650793652 | 407 | 0.045011539614802691 | 6/9 | 628.0 / 250.2 | verified |
| `p-php02-t005-s2024` | 0.16230158730158731 | 409 | 0.045425362487100412 | 7/10 | 693.3 / 271.6 | verified |
| `p-php02-t005-s2025` | 0.16507936507936508 | 416 | 0.045581718229173958 | 7/10 | 696.1 / 276.2 | verified |
| `p-php02-t007-s42` | 0.17023809523809524 | 429 | 0.0488966913211012 | 9/12 | 843.1 / 310.7 | verified |
| `p-php02-t007-s2024` | 0.16984126984126985 | 428 | 0.049107861117874414 | 7/10 | 681.6 / 275.5 | verified |
| `p-php02-t007-s2025` | 0.16944444444444445 | 427 | 0.048996163102463151 | 9/12 | 768.2 / 311.5 | verified |
对照F4-T010同seed hits=439/434/444，三份对照也直接读到verified；完整精确值与archive身份见S1。按decision_log 15:5x第5条机械套用：
| 温度 | 三seed Δhits | 均值Δhits | ≥6 hits的seed数 | 有效 | 三seed都≥6（采用条件） | 有害 | 分类 |
|---|---|---:|---:|---|---|---|---|
| 0.05 | -32 / -25 / -28 | -28.33333333 | 0 | 否 | 否 | 是 | 有害 |
| 0.07 | -10 / -6 / -17 | -11.00000000 | 0 | 否 | 否 | 是 | 有害 |
| 变体 / seed | Δhits | only-new / only-control | exact McNemar双侧p | ΔR@50 95% bootstrap CI | ΔN@50 95% bootstrap CI |
|---|---:|---|---:|---|---|
| p-php02-t005 / 42 | -32 | 56 / 88 | 0.009543281732 | [-0.02222222, -0.00357143] | [-0.00908485, -0.00383177] |
| p-php02-t005 / 2024 | -25 | 50 / 75 | 0.03140933357 | [-0.01865079, -0.00158730] | [-0.00812159, -0.00279435] |
| p-php02-t005 / 2025 | -28 | 64 / 92 | 0.03031031564 | [-0.02103175, -0.00158730] | [-0.01022924, -0.00434420] |
| p-php02-t007 / 42 | -10 | 45 / 55 | 0.3682016173 | [-0.01150794, +0.00357143] | [-0.00483741, -0.00034728] |
| p-php02-t007 / 2024 | -6 | 25 / 31 | 0.50440376 | [-0.00833333, +0.00317460] | [-0.00359592, +0.00009016] |
| p-php02-t007 / 2025 | -17 | 51 / 68 | 0.1421320281 | [-0.01507937, +0.00158730] | [-0.00643014, -0.00129484] |
配对方法沿用php01c：10000次按query有放回percentile bootstrap，seed0；同一组indices用于ΔR和ΔN。原php01c脚本run_ids固定在F4-vs-H1，不能原样跑php02；php02-design §6确认schema/kind/task/knob差异，预测路径与结构相同。已有php02_paired_read.py源码可供核口径，但它本身未绑定verified输出且只打印舍入值，本轮使用上述受版本管理的读数器补足验证与精确值。温度的McNemar仅描述，不作门槛。存在validation选择的赢者诅咒，FEAT-04对照温度不随本次预览静默改变。**这里只列采用条件不满足；不写anchor温度裁决，提交Kiro复查。**

### 2. FEAT-04a A/B（六训练与六verify均verified；来源S1）
| run_id | R@50（精确值） | hits | N@50（精确值） | best/stop | train / verify wall秒 | 科学验收 |
|---|---:|---:|---:|---|---|---|
| `p-feat04a-a-q2t010-s42` | 0.18293650793650792 | 461 | 0.055468506557224455 | 10/13 | 838.0 / 335.1 | verified |
| `p-feat04a-a-q2t010-s2024` | 0.18412698412698414 | 464 | 0.054594054793762663 | 12/15 | 976.0 / 369.0 | verified |
| `p-feat04a-a-q2t010-s2025` | 0.18214285714285713 | 459 | 0.055121278734368263 | 10/13 | 841.5 / 330.5 | verified |
| `p-feat04a-b-f6hist-s42` | 0.17063492063492064 | 430 | 0.049918994346905721 | 4/8 | 525.2 / 231.4 | verified |
| `p-feat04a-b-f6hist-s2024` | 0.17777777777777778 | 448 | 0.051418181790121018 | 3/8 | 523.6 / 227.9 | verified |
| `p-feat04a-b-f6hist-s2025` | 0.18055555555555555 | 455 | 0.054021014931877151 | 6/9 | 580.0 / 253.0 | verified |
| 变体 / seed | Δhits | only-new / only-control | exact McNemar双侧p | ΔR@50 95% bootstrap CI | ΔN@50 95% bootstrap CI |
|---|---:|---|---:|---|---|
| p-feat04a-a-q2t010 / 42 | +22 | 72 / 50 | 0.05682268892 | [+0.00039683, +0.01746032] | [+0.00148195, +0.00661435] |
| p-feat04a-a-q2t010 / 2024 | +30 | 96 / 66 | 0.0224056917 | [+0.00198413, +0.02182540] | [+0.00073115, +0.00662740] |
| p-feat04a-a-q2t010 / 2025 | +15 | 80 / 65 | 0.2448835934 | [-0.00317460, +0.01508929] | [-0.00043804, +0.00490297] |
| p-feat04a-b-f6hist / 42 | -9 | 68 / 77 | 0.5065964186 | [-0.01309524, +0.00595238] | [-0.00427418, +0.00123484] |
| p-feat04a-b-f6hist / 2024 | +14 | 73 / 59 | 0.2577610878 | [-0.00317460, +0.01468254] | [-0.00206202, +0.00311510] |
| p-feat04a-b-f6hist / 2025 | +11 | 86 / 75 | 0.4307246551 | [-0.00555556, +0.01428571] | [-0.00215461, +0.00451730] |
| 变体 | 均值Δhits / ≥6 seed数 | 机械分类 | 转正条件1：有效 | 条件2：≥2seed McNemar p<0.05 | 条件3：独立泄漏review |
|---|---|---|---|---|---|
| A Q2-T010 | +22.33333333 / 3 | 有效 | 满足 | **不满足，仅seed2024**（s42 p=0.05682，s2025 p=0.24488） | 有独立code review中的静态泄漏检查；是否满足专项验收口径待Kiro认定 |
| B F6-hist | +5.33333333 / 2 | inconclusive | **不满足** | **不满足，0个seed** | 同上 |
独立review证据：`.agents/tasks/feat04a-run/feat04a-code-review.md`（APPROVED，明确读过泄漏测试/被调用函数，未重跑测试）。本轮不冒称新做了泄漏review，也不把普通review自动等同专项验收。A的exact p与bootstrap区间在s42边界处结论不同，均按预定方法原样报告；转正只按exact McNemar，不按bootstrap跨零替代，不改阈值。
B解释限制：按decision_log 16:0x第2条，作者词表由full-train likes支持集合确定，训练样本会包含其target之后的信息，这是已登记偏离；validation行从不进统计。不能称为严格时间隔离的作者特征实验。按17:0x：作者PAD=0/OOV=1，OOV行经rows>=2掩码贡献为0且不可训练；**没有可学习共享OOV向量**，64D内容投影对所有事件仍生效。E2只证明runner接线与feat03等价，不是A池化正确性的证据，后者见guard_a_*与issue I-2026-10-15-g。
三个seed评同一批query，不能把它们合成独立样本或把p值直接合并。这里不写A/B转正结论，交Kiro复查。

### 3. A1+T3（三seed训练均succeeded，但s2024尚未验收；来源S1）
| run_id | R@50（精确值） | hits | N@50（精确值） | best/stop | train / verify wall秒 | 科学验收 |
|---|---:|---:|---:|---|---|---|
| `g-gtok01b-t3-b256d3-s42` | 0.11468253968253968 | 289 | 0.034778822893986427 | 8/11 | 5390.7 / 933.5 | verified |
| `g-gtok01b-t3-b256d3-s2024` | 0.13015873015873017 | 328 | 0.042026448215640728 | 20/23 | 11392.1 / UNKNOWN | **未验收** |
| `g-gtok01b-t3-b256d3-s2025` | 0.12301587301587301 | 310 | 0.038717616319113142 | 16/19 | 9727.8 / 895.3 | verified |
对照A1+BHKM16同seed=250/206/234 hits（本轮三份对照均直接读到verified）。A1+T3各训练终值为289/328/310 hits；s42及s2025 verified，s2024缺对应verified证据，不用三seed训练结果冒充已验收结果。
| decision_log 16:5x第1条 | 本次核对 |
|---|---|
| 三训练succeeded且三verify=verified | **未满足：s2024未验收** |
| 三seed均值Δhits≥6 | **正式判定UNKNOWN**；仅训练终值算术预览为+79 hits，包含未验收seed，不据此过门 |
| 至少2seed Δhits≥6 | **正式判定UNKNOWN**；仅训练终值预览+39/+122/+76，包含未验收seed |
不更换G anchor；等待原52031驱动补verify并产生新收据，不人工重提。**三seed和verified齐全时提醒Eddy在Kiro处切到Opus解读；本会话未声称已切换模型。**

### 4. T1r三seed vs T3（两组三seed均verified；来源S1）
| run_id | R@50（精确值） | hits | N@50（精确值） | best/stop | train / verify wall秒 | 科学验收 |
|---|---:|---:|---:|---|---|---|
| `g-tok-t1r-rqvae-b256-s42` | 0.092460317460317465 | 233 | 0.031409388406976893 | 10/13 | 5048.5 / 732.6 | verified |
| `g-tok-t1r-rqvae-b256-s2024` | 0.098809523809523805 | 249 | 0.03327598226971213 | 15/18 | 7916.2 / 796.8 | verified |
| `g-tok-t1r-rqvae-b256-s2025` | 0.10555555555555556 | 266 | 0.034027339686367605 | 17/20 | 8782.7 / 797.3 | verified |
T1r hits=[233, 249, 266]，均值249.33333333；T3 hits=[254, 274, 275]，均值267.66666667；差（T1r−T3）=-18.33333333 hits。按decision_log 16:5x第2条，绝对差>12，机械方向为**T3更高**，不落入“不能区分”区间；仍标为**推断**，不是方法的普遍优劣或因果结论。固定SID拟合seed=42，仅G训练seed变化；单一数据集、core03、同K与depth、同validation窗口。结果解释仍交Kiro。

### 5. GTOK-02六个SID拟合/审计（不是召回指标；来源S1的fits）
六fit与六fit-verify均succeeded；直接读取out/verify.json的status=verified与原bundle manifest SHA相同。审计audit.json逐文件SHA与已verified manifest的artifacts.audit.json匹配；未读取mapping.npz或内容embedding。fit/verify墙钟补充来源S2：`main05/codex-p2-tokfit-20261015T185410.stdout`，SHA256 `c224ae329af7ead9e9cce051308363a14d6f5f035171729811d70567357ed964`，实际收据18:52:28+08（按transport完成mtime）；bundle manifest SHA与S1逐一相同。
| 配置 | leaves / max / mean | 各位distinct（含suffix） | 各位utilization（含suffix） | items_needing_suffix | CER | LCR | full SID碰撞 | B1 | fit / verify秒 |
|---|---|---|---|---:|---:|---:|---:|---|---|
| b32d3 | 32768 / 4 / 3.67477417 | [32, 32, 32, 4] | [1.0, 1.0, 1.0, 1.0] | 87647 | 0.72787443 | 1.00000000 | 0 | 通过 | 130.8 / 33.1 |
| b64d3 | 120415 / 1 / 1.00000000 | [64, 64, 30, 1] | [1.0, 1.0, 0.46875, 1.0] | 0 | 0.00000000 | 0.00000000 | 0 | 通过 | 293.1 / 47.8 |
| b128d3 | 120415 / 1 / 1.00000000 | [128, 128, 8, 1] | [1.0, 1.0, 0.0625, 1.0] | 0 | 0.00000000 | 0.00000000 | 0 | 通过 | 502.1 / 49.3 |
| b512d3 | 120415 / 1 / 1.00000000 | [512, 236, 1, 1] | [1.0, 0.4609375, 0.001953125, 1.0] | 0 | 0.00000000 | 0.00000000 | 0 | 通过 | 1377.9 / 68.2 |
| b1024d3 | 120415 / 1 / 1.00000000 | [1024, 118, 1, 1] | [1.0, 0.115234375, 0.0009765625, 1.0] | 0 | 0.00000000 | 0.00000000 | 0 | 通过 | 2418.1 / 68.2 |
| b256d2 | 65536 / 2 / 1.83738708 | [256, 256, 1, 2] | [1.0, 1.0, 0.00390625, 1.0] | 54879 | 0.45574887 | 0.91149774 | 0 | raw3通过 | 107.8 / 105.1 |
CER=items_needing_suffix/N；LCR=semantic_collision_items/N，后者不是发生碰撞的leaf比例。N=120415；所有bundle engineering=false、gate passed、roundtrip/reachable完整。第三位在B512/B1024只剩1个取值，是审计结果，不将“高叉数”自动解读成更好的语义或召回。B256D2第三位是固定补零，utilization只是格式数字，不是学得语义码利用率。
**B256D2的B1并列记录**：补零前三列的B1由audit.mapping.balanced_children记录，balanced=true、max_spread=1、nodes=65793、violations=0，sweep.balanced_basis=raw3；补零后四列的B1检查结果**未在当前manifest/audit持久化，UNKNOWN**。设计/报告给出的“预期违反256处”不是本轮读取的服务器审计实测，不冒充实测。此缺口进STATE待Kiro补审计收据；不改门禁、不重拟合。
五个depth3 verify的replay_fit=false（检查load_bundle/config_binding/expected）；B256D2还包括rederive_depth2。不能称为重放拟合全部通过。GTOK-02 core03训练由Kiro原工作流安排；本轮只观察其队列，不入队或接管。

#### GTOK-02 core03 seed42 训练排程更新（2026-10-15；不新增作业）
Eddy本轮转交19:12时六个训练已全部在跑的更新；本轮只读独立确认六个均claimed、run目录存在、无对应rejected/.reason、尚无terminal，forecast均列running。19:12作为批次状态更新时点，不改写各run的实际first-write/start/spent/deadline。来源：`main05/codex-gtok02-update-queue-20261015T191600.stdout`，SHA256 `64a7118446b6427e7e97836e77f142dfc476ab2e95b72976556b228cd31b6802`，实际收据完成 `2026-10-15T11:15:26.757715+00:00`（UTC）；`main05/codex-gtok02-update-forecast-20261015T191600.stdout`，SHA256 `66263b865f2aae9a06a9ff85ceceaf156623170eda024fac73cca5b51f368f3f`，实际收据完成 `2026-10-15T11:15:34.881158+00:00`（UTC）。

| run_id | 声明显存MiB | 当前接纳 / 执行 | verify提交者 |
|---|---:|---|---|
| `g-tok2-b1024d3-s42` | 3072 | claimed / running | 原驱动96334自动补，不重复入队 |
| `g-tok2-b512d3-s42` | 3072 | claimed / running | 原驱动96334自动补，不重复入队 |
| `g-tok2-b128d3-s42` | 3072 | claimed / running | 原驱动96334自动补，不重复入队 |
| `g-tok2-b64d3-s42` | 3072 | claimed / running | 原驱动96334自动补，不重复入队 |
| `g-tok2-b32d3-s42` | 3072 | claimed / running | 原驱动96334自动补，不重复入队 |
| `g-tok2-b256d2-s42` | 3072 | claimed / running | 原驱动96334自动补，不重复入队 |
驱动96334本机实测存活、PPID=1，脚本gtok02t_driver.py；启动日志显示interval=900s。进程收据`main05/codex-gtok02-driver-20261015T191600.txt`的SHA见`cache/codex-gtok02-queue-update-20261015/receipt-index.json`。预计结果窗口20:45–21:05（北京时间）是Eddy转交的未实测估计，不是工具ETA，不保证训练或verify届时完成。此处只补run身份与资源声明，不读取召回中间指标，不新增训练或verify。

### 6. G5零成本预检：同seed命中集交并（非融合结果；来源S1）
P固定F4-T010；G分别用已验收的core03+T3与A1+BHKM16。每个组合的P/G对应训练与科学verify均通过，预测取各自validation-best checkpoint的归档item top50，无新模型推理。命中集以完整query顺序标识，只输出计数，不导出query/user IDs。
| G版本 / seed | P hits | G hits | 交集 | 仅P | 仅G | 并集oracle hits | Jaccard | 全2520 oracle R | 非OOV1897 oracle R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core03+T3 / 42 | 439 | 254 | 169 | 270 | 85 | 524 | 0.32251908 | 0.20793651 | 0.27622562 |
| core03+T3 / 2024 | 434 | 274 | 204 | 230 | 70 | 504 | 0.40476190 | 0.20000000 | 0.26568266 |
| core03+T3 / 2025 | 444 | 275 | 198 | 246 | 77 | 521 | 0.38003839 | 0.20674603 | 0.27464418 |
| A1+BHKM16 / 42 | 439 | 250 | 169 | 270 | 81 | 520 | 0.32500000 | 0.20634921 | 0.27411703 |
| A1+BHKM16 / 2024 | 434 | 206 | 146 | 288 | 60 | 494 | 0.29554656 | 0.19603175 | 0.26041118 |
| A1+BHKM16 / 2025 | 444 | 234 | 158 | 286 | 76 | 520 | 0.30384615 | 0.20634921 | 0.27411703 |
**OOV分层**：六个组合各623个OOV query，P/G命中、交集、并集、仅P、仅G均为0，both-miss=623，oracle R=0，Jaccard=UNKNOWN/未定义（0/0）。非OOV各1897，P/G/交并/仅P/仅G的计数与上表相同，分母与oracle R按最后一列单列。
**边界**：预检，oracle并集是知道真实target后选择命中路线得到的上限；不是可达指标，不是融合结果，也不是RRF/固定200候选预算下的收益。两路top50合起来最多100候选后能否排进final50未测；Pop/第二个P seed对照也未算，不能据此证明G值得采用。原候选按validation选best，三seed用同一批query，结果仅用于G5设计复查。没有读test，没有建融合框架。

## 2026-10-16 Codex训练安排：GTOK-02补seed与已结束批次读数

任务 TRAIN-SCHEDULE-20261016。仅按既有预注册机械套用；anchor、转正及新增条件批次均交Kiro。没有读取test、既有checkpoint或完整embedding。以下来源均为指定main05 wrapper生成；文件名前缀不代表实际读取时间。

### 来源收据
- **G1**：`main05/codex-seeds-finalcheck-20261016T004800.stdout`，SHA256 `9338fac3905399cc688adb488f9a57e5783baacd482d78aaa3ea5ad3806f5456`，服务器实际读取UTC `2026-10-15T16:42:34.000546+00:00`。含六个s42的manifest SHA、verify stdout路径/SHA、claimed verify argv与manifest绑定、原部署pin逐文件检查、补seed配置及队列、驱动身份。直接解析到`verified`，不是以exit0代替。
- **R1**：`main05/codex-schedule-results-20261016T004600.stdout`，SHA256 `6112d97617e36d6386620a12d829b683c6897c244df92ec4c6148a3408135c3d`，实际读取UTC `2026-10-15T16:41:26.482504+00:00` 至 `16:41:34.719897+00:00`。逐run的manifest/verify stdout路径、SHA与目标绑定、validation档案hash、query行序与population核验均保存在JSON中。配对bootstrap按query、10000次、seed0，区间为95%百分位；精确双侧McNemar。来源工具已push `8ff7b6bcdc7cac632f7c07ace1a500f68b9f5164`。
- **D1**：`main05/codex-seqm1-deploy-20261016T004500.stdout`，SHA256 `c1ab750899f32effab8cdf37f5e6d09980040157fafbe14f2214c0f758d4e7e7`，实际UTC `2026-10-15T16:41:13.045483+00:00`。
- **M1**：`main05/codex-seqm1-preflight-20261016T004700.stdout`，SHA256 `07a7e533182d2f40ce13be7d2f25b68458cfea10d038ec0887238aa5fb98cb71`，实际UTC `2026-10-15T16:42:16.479213+00:00`。
- 全部操作stdout/transport/source hashes、读取或收据完成时间：`cache/codex-training-schedule-20261016/receipt-index-v1.json`（后续事件见v2）。原始逐query预测不进Git。

### 1. GTOK-02六个s42：科学验收后排名（G1）

| run_id | hits/2520 | R@50 | N@50 | verify | source |
|---|---:|---:|---:|---|---|
| g-tok2-b1024d3-s42 | 280 | 0.1111111111111111 | 0.034541834269608385 | verified | G1 |
| g-tok2-b512d3-s42 | 263 | 0.10436507936507937 | 0.033095016709313634 | verified | G1 |
| g-tok2-b256d2-s42 | 233 | 0.09246031746031746 | 0.02949205959184739 | verified | G1 |
| g-tok2-b128d3-s42 | 229 | 0.09087301587301587 | 0.030631876937991735 | verified | G1 |
| g-tok2-b64d3-s42 | 211 | 0.08373015873015872 | 0.026563602686387265 | verified | G1 |
| g-tok2-b32d3-s42 | 205 | 0.08134920634920635 | 0.02320880065741259 | verified | G1 |

八点按hits排名：B1024 280 > B512 263 > T3/B256D3 254 > B256D2 233 > B128 229 > B64 211 > B32 205 > B16 190。B16/T3数值来自原已验收对照（decision_log及既有实验节）；六新点逐个读到verified。按P1最高两点加距最高≤10者，名单为 **B1024D3、B512D3**；阈值270，近邻规则没有额外点。此为补seed名单，不是G anchor选择。B256D2按P2无条件补跑，与P1去重。

补seed六run：`g-tok2-b256d2-s2024/s2025`、`g-tok2-b1024d3-s2024/s2025`、`g-tok2-b512d3-s2024/s2025`。G1中六个全部claimed、对应rejected为空；每个配置与其s42仅seed不同、冻结pin有效，每个声明1核/8192 MiB RAM/3072 MiB VRAM/4 GiB disk/30h wall。3072沿用本批冻结的保守声明；各新叉数的真实峰值未记录，不能将早期GPU占用称为峰值，见gtok02-report与本轮issue。总训练声明6核、18432 MiB VRAM，不把wall cap当ETA。

P2先在00:21前已独立确认两个claimed，满足00:40前恢复作业要求；P1曾有两个pending，G1时已全部claimed。详见submit/confirm收据索引。verify由服务器版本化驱动自动补：P2 pid2910284（ops `1c99d83`，目录`/workspace/vkgr-codex-seed-drivers/p2-20261016T0025`），P1 pid2911779（ops `37bd41f`，目录`/workspace/vkgr-codex-seed-drivers/p1-20261016T0035`）；G1确认两者存活、PPID1，900s、48h有限、不调用模型。新训练尚未科学验收，不将claimed当训练成功。

### 2. A1+T3三seed验收收口（R1）

| run_id | hits/2520 | R@50 | N@50 | best/stop epoch | verify | source |
|---|---:|---:|---:|---|---|---|
| g-gtok01b-t3-b256d3-s42 | 289 | 0.11468253968253968 | 0.03477882289398643 | 8/11 | verified | R1 |
| g-gtok01b-t3-b256d3-s2024 | 328 | 0.13015873015873017 | 0.04202644821564073 | 20/23 | verified | R1 |
| g-gtok01b-t3-b256d3-s2025 | 310 | 0.12301587301587301 | 0.03871761631911314 | 16/19 | verified | R1 |

对照A1+BHKM16同seed为250/206/234，三seed均值230；A1+T3均值309，Δ均值79，逐seedΔ为+39/+122/+76。decision_log 16:5x条件：(a)三训练succeeded且三个verify直接读到verified：满足；(b)均值Δ≥6：满足；(c)至少2 seed Δ≥6：3/3，满足。仅登记条件满足，**不写换G anchor结论，请Eddy在Kiro处切Opus解读**。s2024 verify stdout SHA `9bcd43b69168936f6324dc9f45c434e904fd712346b1db7b842e59774cea8a2b`，目标manifest SHA `afa32baf26ea25b45058b4527cc9cea5c8f07975f153cf9f7940c8a26f1e027a`，已逐项绑定（R1）。

### 3. FEAT-04b C/D三seed（R1）

| run_id | hits/2520 | R@50 | N@50 | best/stop epoch | verify | source |
|---|---:|---:|---:|---|---|---|
| p-feat04b-c-i4-popfresh-s42 | 449 | 0.17817460317460318 | 0.05147147814432139 | 6/9 | verified | R1 |
| p-feat04b-c-i4-popfresh-s2024 | 451 | 0.17896825396825397 | 0.052402657860703604 | 7/10 | verified | R1 |
| p-feat04b-c-i4-popfresh-s2025 | 452 | 0.17936507936507937 | 0.0536688309985621 | 11/14 | verified | R1 |
| p-feat04b-d-i5-quality-s42 | 441 | 0.175 | 0.05168149034245512 | 10/13 | verified | R1 |
| p-feat04b-d-i5-quality-s2024 | 451 | 0.17896825396825397 | 0.052488838394037673 | 7/10 | verified | R1 |
| p-feat04b-d-i5-quality-s2025 | 460 | 0.18253968253968253 | 0.05460135824147601 | 12/15 | verified | R1 |

| comparison / seed | exact McNemar p | bootstrap delta R@50 CI95 | delta N@50 CI95 | source |
|---|---:|---|---|---|
| p-feat04b-c-i4-popfresh / 42 | 0.4152851813125319 | [-0.004374999999999991, 0.012698412698412698] | [-0.002341654140997932, 0.0024147967759322133] | R1 |
| p-feat04b-c-i4-popfresh / 2024 | 0.11446824201255824 | [-0.0011904761904761906, 0.014285714285714285] | [-0.0006378746175067128, 0.003718038360149133] | R1 |
| p-feat04b-c-i4-popfresh / 2025 | 0.4841184136072915 | [-0.004365079365079365, 0.011111111111111112] | [-0.0017398798200123193, 0.003339548016520452] | R1 |
| p-feat04b-d-i5-quality / 42 | 0.899076313652859 | [-0.005158730158730159, 0.007142857142857143] | [-0.0013124935297069884, 0.0017976394313027153] | R1 |
| p-feat04b-d-i5-quality / 2024 | 0.05681467793283902 | [0.0003968253968253968, 0.013095238095238096] | [-0.0002522004268231315, 0.00345534017557458] | R1 |
| p-feat04b-d-i5-quality / 2025 | 0.11737015883134423 | [-0.0011904761904761906, 0.01389880952380938] | [-0.000546765392447197, 0.0039952547696745794] | R1 |

C Δhits为+10/+17/+8，均值+11.666666666666666，3/3过≥6；D为+2/+17/+16，均值相同，2/3过≥6。两者机械分类均**有效**，但各自McNemar p<0.05为0/3，均不满足转正的统计条件。独立泄漏review仍需按专项口径认定，不能由verified替代。C/D统计使用full-train，违反训练样本统计严格早于target的已知偏离保留；D likes近似target频次，解释必须带decision_log 23:2x，不作无泄漏或因果收益声明。C9仍只报告，不能因未触发警告而证明无偏差。C/D及其驱动未改、未重启；E由Kiro的`wf_2062b81bb2e46197`单独负责。

### 4. 其它已结束批次复核（R1；详数沿用上一阶段实验节）
- 温度T005三seed Δhits -32/-25/-28，均值-28.333333333333332；T007 -10/-6/-17，均值-11。六verify均直接verified；两者机械分类有害，采用条件均不满足。温度anchor解释交Kiro。
- FEAT-04a A Δ +22/+30/+15，均值+22.333333333333332，有效；McNemar p<.05仅1/3，不满足转正统计条件。B Δ -9/+14/+11，均值+5.333333333333333，inconclusive。六verify均verified；B OOV作者贡献固定为0、内容投影仍生效，full-train作者词表偏离继续披露。E2仅接线等价证据，不证明A池化正确性。
- T1r hits233/249/266，三verify均verified；均值249.33333333333334，对照T3 254/274/275均值267.6666666666667，差-18.333333333333343，超±12，机械方向T3较高，仅推断，不扩展为方法普遍优劣。

### 5. M1部署与真实输入preflight（工程准备，无召回指标）
D1：seq/m1原始`04d4edeb84826d4d56387f95dc820fda1dce10b2`压平到新公开分支`codex/seq-m1-pub`，提交`0ca61bc35ad68bd79b86103b54e470445cbab07f`，tree相同；基于已公开ancestor，新范围38文件全部新增，扫描0命中，无大文件，没有改写旧历史。部署新目录`/workspace/vkgr-seqm1-p-v1`，model242文件、ops55文件逐个SHA核过。p3c原Git文件移存ops目录后，用固定hash的数据入口创建新文件；未丢弃原文件。

M1：真实输入preflight_ok，train86785、validation2520、catalog120415；三个mean配置hash核过，test_read=false、training_started=false。ops提交`8ff7b6bcdc7cac632f7c07ace1a500f68b9f5164`，source SHA `20c1a8115d0321ddcdd5d62e22481cc94b0d0905f234480b302c93156f1fc6f0`。CUDA工程预演另记事件；正式M1三seed尚未入队，不能把合成工程verify当作真实模型验收。seq-report F3/F4分析冻结项与阈值标定未收口，进STATE待裁决，不自行改规则。


### M2: M1 CUDA engineering receipt

Source: `main05/codex-seqm1-smoke-confirm-20261016T005500.stdout`, SHA256 `3449f43a9e973208caa9ec4c82f6a38f3f4954f27ca4c921742b7e0fec2007ae`, read UTC `2026-10-15T16:45:18.762221+00:00`. Run `p-seqm1-cuda-engineering-v1`: claimed, rejected empty, terminal succeeded; `engineering_passed` and all five synthetic verifies explicitly `verified`. Engineering artifact SHA256 `afc9a92d45c38b3bff9878f58370107dc55a2825161c699026247be267b828fd`.

| arm | allocated peak MiB | reserved peak MiB | context-inclusive bound MiB | ceil(measured*1.15/256)*256 MiB | source |
|---|---:|---:|---:|---:|---|
| mean | 377.87109375 | 508.0 | 882.0 | 1024 | M2 |
| last | 377.875 | 508.0 | 882.0 | 1024 | M2 |
| attn | 391.1640625 | 510.0 | 884.0 | 1024 | M2 |
| bi | 356.30859375 | 488.0 | 862.0 | 1024 | M2 |
| pure | 367.07275390625 | 514.0 | 888.0 | 1024 | M2 |

Measured with per-process NVML observations and allocator peaks, not global free-memory differences. Full-shape B512/L50/120416 embedding rows, three AdamW steps per arm, full-catalog item export and 2520-user export; CUDA padding atol1e-6 passed. Terminal wall 18.635082006454468 seconds is engineering rehearsal only, not a real-training ETA or retrieval metric. No test read; no formal M1 training submitted.

## 2026-10-16 Codex全权接收：FEAT-04b、GTOK审计与SEQM1-000

本轮没有新训练或verify提交、没有重启旧驱动。Kiro工作流全部完成是Eddy本轮直接交接事实；服务器训练/verify状态另按以下收据验收。M1真实入队必须先在Opus下由Kiro复核decision_log02:xx；本轮只做已授权的前置检查。

- **F1**: `main05/codex-fulltake-science-acceptance-20261016T021600.stdout`, SHA256 `54d8c99eef7c2dfd6df00ebf03994902e42777b2a929b64ba9bcbc6764bda301`, actual read UTC `2026-10-15T17:44:31.401139+00:00`; transport completed UTC `2026-10-15T17:44:44.414102+00:00`.
- **N1**: `main05/codex-seqm1000-baseline-20261016T020100.stdout`, SHA256 `7ca4973f972b448219476d06535742f60c3c6ba0fc7dcb4b9af5baa971f8b24f`, actual read UTC `2026-10-15T17:36:10.379338+00:00`; transport completed UTC `2026-10-15T17:37:03.678396+00:00`.
- **A1**: `main05/codex-fulltake-cd-audit-20261016T020300.stdout`, SHA256 `6be7d119cad3edd6b247abfc772b64a4995fa0e812fe978eb70aada21b8cbba5`, actual read UTC `None`; transport completed UTC `2026-10-15T17:37:22.247936+00:00`.
- **H1**: `main05/codex-fulltake-final-status-20261016T021000.stdout`, SHA256 `75ed9a735c09ff70c370e58468e482d76bf08a7807835359811eaeebed7b703a`, actual read UTC `None`; transport completed UTC `2026-10-15T17:41:34.288768+00:00`.
- **T1**: `main05/codex-fulltake-final-forecast-20261016T021000.stdout`, SHA256 `74f07c41abcda9f024a2efa483c7d7e8677f795306c314aceb58639832bd8e00`, actual read UTC `None`; transport completed UTC `2026-10-15T17:41:38.542076+00:00`.
F1/N1 include server UTC; A1/H1/T1 have no internal timestamp, so use explicitly labelled transport completion time. All per-run manifest/verifier/archive hashes and claimed-job bindings are inside F1; full source/transport index: `cache/codex-fulltake-20261016/receipt-index-v2.json`.

### 1. FEAT-04b C/D阶段1（第一波s42；第二波s2024/s2025）
六训练succeeded，六verify均直接读到verified并核claimed job的manifest目标/SHA绑定；不是只看supervisor exit0。完整精确R@50/N@50、best/stop、逐seed McNemar与bootstrap见上一节“2026-10-16 Codex训练安排”§3，本轮复核数值一致。补报已追加到`.agents/tasks/feat04b-report.md`“阶段1/C、D：Codex读数补记”。

C Δhits +10/+17/+8，均值+11.666666666666666，3/3过6；D +2/+17/+16，均值相同，2/3过6。均机械有效；各自0/3 McNemar p<.05，未满足转正统计条件，不写转正结论。D-1：full-train0–24周统计包含训练侧自身target/未来行，validation从不进入统计；不得称严格时间隔离的物品特征有效。

审计stdout再次只读核对（A1）：sum_likes=105700、n_train_examples=86785、差18915；items_equal=111967/120415，比例0.9298426275796204（约93.0%），p_like=0.21152272858212826（约0.2115）。C9提醒阈值0.30未触发；C9仅报告项，不是转正门禁。不得称总点赞约等于训练样本数，也不得把D增益直接归因于观看质量。原统计manifest SHA `eb41bc060c67549a0cb7bf96c627000bb90b73cb5111ca11b21bf572573c036b`，未改统计文件。

### 2. FEAT-04b E：anchor对照、E−A与最佳epoch β

| seed | hits | R@50 | N@50 | best/stop | verify | source |
|---|---:|---:|---:|---|---|---|
| 42 | 460 | 0.18253968253968253 | 0.05420100549695798 | 11/14 | verified | F1 |
| 2024 | 466 | 0.18492063492063493 | 0.053963670161003764 | 11/14 | verified | F1 |
| 2025 | 461 | 0.18293650793650792 | 0.05436257555085179 | 9/12 | verified | F1 |

| seed / contrast | delta hits | exact McNemar p | delta R CI95 | delta N CI95 | source |
|---|---:|---:|---|---|---|
| 42 / E-anchor | 21 | 0.08481264433974636 | [-0.0007936507936507937, 0.01746031746031746] | [-1.471637436546826e-05, 0.005566978841652325] | F1 |
| 42 / E-A descriptive | -1 | 1.0 | [-0.00753968253968254, 0.006746031746031746] | [-0.003250629332043218, 0.0007060751616604249] | F1 |
| 2024 / E-anchor | 32 | 0.014612363079529685 | [0.002777777777777778, 0.02261904761904762] | [0.0002843468001881506, 0.005839929494801581] | F1 |
| 2024 / E-A descriptive | 2 | 0.9142116370824501 | [-0.006349206349206349, 0.007936507936507936] | [-0.002495686524008578, 0.0011812520249806645] | F1 |
| 2025 / E-anchor | 17 | 0.16188152458394658 | [-0.001984126984126984, 0.015873015873015872] | [-0.0013150693721783794, 0.004281875500046849] | F1 |
| 2025 / E-A descriptive | 2 | 0.9132023622245966 | [-0.006349206349206349, 0.007936507936507936] | [-0.0029183417985325154, 0.0014376462455781534] | F1 |

Mechanical classification `effective`, per-seed delta hits `[21, 32, 17]`, mean delta `23.333333333333332`, seeds >=6 `3/3`; exact p<.05 `1/3`. Independent leakage review still requires Kiro recognition; no promotion/anchor verdict here.

E-A descriptive delta hits `[-1, 2, 2]`, mean `1.0`. This is not an additional selection gate or a causal effect.

| seed | like | share | bookmark | open_comments | click_on_author | dislike | source |
|---|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.48290345072746277 | 0.04337453097105026 | 0.1818845570087433 | -0.03444068506360054 | 0.21290381252765656 | -0.20028406381607056 | F1 |
| 2024 | 0.4673086106777191 | 0.03119783289730549 | 0.18467408418655396 | -0.05830158665776253 | 0.24044296145439148 | -0.22464925050735474 | F1 |
| 2025 | 0.48476237058639526 | 0.07144404947757721 | 0.14413005113601685 | -0.025613104924559593 | 0.24962422251701355 | -0.20796239376068115 | F1 |

β加在pooling logits上，初值0；符号与大小仅描述训练后的权重，不是行为的因果效果。E−A是同seed描述性对照，不能把E相对F4的全部收益说成行为权重收益。缺任何一个verified就只报已验收seed，三seed分类与转正条件为UNKNOWN/未验收，不以训练终态补齐。来源F1，按query bootstrap10000次、seed0、95%百分位。

### 3. GTOK-02六s42与拟合审计复核

| run_id | hits | R@50 | N@50 | best/stop | verify | source |
|---|---:|---:|---:|---|---|---|
| g-tok2-b1024d3-s42 | 280 | 0.1111111111111111 | 0.034541834269608385 | 14/17 | verified | F1 |
| g-tok2-b512d3-s42 | 263 | 0.10436507936507937 | 0.033095016709313634 | 14/17 | verified | F1 |
| g-tok2-b256d2-s42 | 233 | 0.09246031746031746 | 0.02949205959184739 | 10/13 | verified | F1 |
| g-tok2-b128d3-s42 | 229 | 0.09087301587301587 | 0.030631876937991735 | 14/17 | verified | F1 |
| g-tok2-b64d3-s42 | 211 | 0.08373015873015872 | 0.026563602686387265 | 15/18 | verified | F1 |
| g-tok2-b32d3-s42 | 205 | 0.08134920634920635 | 0.02320880065741259 | 11/14 | verified | F1 |

六s42均直接verified；与B16=190、T3=254共同排名仍为B1024=280、B512=263、T3=254、B256D2=233、B128=229、B64=211、B32=205、B16=190。P1名单B1024/B512；最高点差≤10没有新增点。P2 B256D2无条件。六个补seed沿用上轮已claimed run_id与原driver，不重复入队。B256D2三seed位置对照/±12推断等补seed全部verified再套，不写碰撞率因果结论。

| fit variant | leaves | max leaf | semantic collision items | full SID collisions | position utilization (including suffix) | source |
|---|---:|---:|---:|---:|---|---|
| b1024d3 | 120415 | 1 | 0 | 0 | [1.0, 0.115234375, 0.0009765625, 1.0] | F1 |
| b512d3 | 120415 | 1 | 0 | 0 | [1.0, 0.4609375, 0.001953125, 1.0] | F1 |
| b256d2 | 65536 | 2 | 109758 | 0 | [1.0, 1.0, 0.00390625, 1.0] | F1 |
| b128d3 | 120415 | 1 | 0 | 0 | [1.0, 1.0, 0.0625, 1.0] | F1 |
| b64d3 | 120415 | 1 | 0 | 0 | [1.0, 1.0, 0.46875, 1.0] | F1 |
| b32d3 | 32768 | 4 | 120415 | 0 | [1.0, 1.0, 1.0, 1.0] | F1 |

拟合/fit-verify六组均verified；数字来自F1 fits逐项manifest/audit哈希绑定。利用率列按positions顺序，含suffix位，不等于召回指标。B256D2已存raw3 B1通过（violations0/max_spread1），补零后四列B1实测仍未归档，**UNKNOWN**；第三位全0的利用率本身不能替代B1审计结果，不把“预期256”写成实测。所有full_sid_collisions均0，不代表语义前缀无碰撞。

### 4. SEQM1-000：F4 best checkpoint的原始m范数（N1）

| seed | raw m p10 | raw m median | raw m p90 | fixed interval check | source |
|---|---:|---:|---:|---|---|
| 42 | 0.10629798082079726 | 0.14351252076545618 | 0.1825660695762527 | PASS | N1 |
| 2024 | 0.09682895493449363 | 0.1315106727486185 | 0.16584390205704844 | PASS | N1 |
| 2025 | 0.11121384781284774 | 0.1498281392174144 | 0.19138349981347788 | PASS | N1 |

三个seed的原始validation用户向量都未normalize；各2520 queries。44个原manifest中src/vkgr源码逐文件SHA与M1部署一致，三个seed的源码账本相同；php01模型/runner及其依赖均覆盖。三份validation预测档案可读、重算R/N一致，checkpoint SHA、binding、epoch/step已核，`weights_only=True`加载，只输出范数汇总，未导出用户/item向量或checkpoint。CPU单线程，入口测得已有约6.9908核、配额10.24、80%线8.192，另1核可放行；没有抢占其它作业。

三个中位数均位于固定[0.05,5]，前置尺度检查通过；**0.05/3阈值没改、没有重新标定**。此前M1真实输入preflight、CUDA五臂verified/padding与全shapeVRAM已通过；mean882 MiB→按×1.15向上取整256声明1024，详见上一节M2。02:xx由Sonnet起草，Opus/Kiro复核仍是正式入队先决，不以本工程检查替代。之后顺序按新裁决：先mean3seed，再其余4臂共12个，各有verify；本轮没有入队。

### 5. owner、驱动与其它机械结论
F1中三服务器驱动的PID、PPID、完整command与source SHA均核到：P2 2910284、P1 2911779、E 2914285；原目录及900s轮询保留。E driver的done只表示终态，不替代独立science verified。既有5d51228驱动压平发布为新分支`codex/feat04b-driver-pub@4f16de248af8fc67da8d3b987f6fc519cf26ddf7`，parent为已公开a91bcd38，树等同5d51228、只新增该脚本，完整新增范围扫描0命中；原源码SHA `04e722881033719b68560a73d260b749728b25607ca604631534716d87ff5ef6`与服务器副本一致。仅补版本归档，未重部署/重启，原copy来源偏差透明保留。

A1+T3三seed均verified，Δ+39/+122/+76，均值+79，原三条件均满足；T1r比T3均值低18.33，超±12但仅推断；两温度机械有害/不满足采用；A有效但显著seed1/3，B inconclusive。前节数字与条件表本轮复核一致，不追加anchor/转正结论。结果解读、设计/anchor决定请Eddy在Kiro处切Opus。

本轮源码工具`codex/phase2-readonly-tools@4f30d5f`；21项方法测试fresh通过，一次独立review指出的依赖漏项已RED→GREEN修复，无重复review。除Eddy明确授权的SEQM1-000三份固定F4 best checkpoint外，没有读取其它checkpoint、完整embedding导出或test。/tmp只汇总报告，清单见`docs/superpowers/plans/2026-10-15-server-tmp-residue-list.md`，没有删除、覆盖或改历史。

## 2026-10-16 M1夜间正式批次：十五配置preflight、mean接纳与金丝雀接续

Task M1-NIGHT-20261016。Eddy转交Opus复核通过，按decision_log“2026-10-16夜间”§1–2执行；不改模型/阈值，不读test，不根据mean分数增删臂或挑seed。本节只登记工程与入队事实，**M1召回效果尚未验收**。

- **DEP1**: `main05/codex-m1-night-deploy-20261016T030000.stdout`, SHA256 `438231dd773f3f649765de8d0ce71583855171496cae37bb1e03b52525f37417`, actual read UTC `2026-10-15T18:38:29.771620+00:00`, transport completed UTC `2026-10-15T18:38:29.974614+00:00`.
- **PF1**: `main05/codex-m1-night-preflight15-20261016T030200.stdout`, SHA256 `2773ca19eb7fab8bcbb4a7a5ba7b992bddcf8b7b36f7a998c0bb51241c0a3802`, actual read UTC `2026-10-15T18:39:24.616827+00:00`, transport completed UTC `2026-10-15T18:41:34.429603+00:00`.
- **D1**: `main05/codex-m1-night-dry2-20261016T030800.stdout`, SHA256 `bec37efeed92a022ff99011946f482fe5484dbf9706b8f9269ea699765e4942c`, actual read UTC `2026-10-15T18:43:08.952970+00:00`, transport completed UTC `2026-10-15T18:43:09.463576+00:00`.
- **S1**: `main05/codex-m1-night-submit-20261016T031000.stdout`, SHA256 `bc73da8f6e5a87eebf0a769b02c6ab0dca2ef0cd93d86cf38c6d49574f3823c7`, actual read UTC `2026-10-15T18:43:40.937083+00:00`, transport completed UTC `2026-10-15T18:43:41.474352+00:00`.
- **C1**: `main05/codex-m1-night-driver-check-20261016T031700.stdout`, SHA256 `76695e290c9cda3a8764008642ec61d08698a4044b21f66e3aa4820aa5ea643e`, actual read UTC `2026-10-15T18:47:24.511124+00:00`, transport completed UTC `2026-10-15T18:47:24.865043+00:00`.
- **H1**: `main05/codex-m1-night-final-status-20261016T032000.stdout`, SHA256 `4cfc249de101a643ca352f48dd5bd2ddaa6200296d9e04587daf84fe38b95d6a`, actual read UTC `None`, transport completed UTC `2026-10-15T18:49:07.695296+00:00`.
- **T1**: `main05/codex-m1-night-final-forecast-20261016T032000.stdout`, SHA256 `b397dd42f547a85824cc93ca04ca2ed0f7973df750a8b621f9a1e46e018b1b6c`, actual read UTC `None`, transport completed UTC `2026-10-15T18:49:09.360221+00:00`.
H1/T1 have no internal UTC; use explicitly labelled transport completion time. Full source/transport hashes including the failed first dry-run: `cache/codex-m1-night-20261016/receipt-index-v2.json`. Filename prefixes are not actual times.

### 1. 正式部署与十五配置真实只读preflight

模型固定`codex/seq-m1-pub@0ca61bc35ad68bd79b86103b54e470445cbab07f`，`/workspace/vkgr-seqm1-p-v1`；数据通道p3c override保持原hash。运维核心`codex/phase2-readonly-tools@6a8e057bb021ed50dbaacc92d75c7a048b5bf45a`，新目录`/workspace/vkgr-seqm1-night-20261016/code`，核心SHA `9e2b1a2e5114882e25a56233883ccacdebed4235be6f53e457c3eac3d8c7edda`。DEP1逐文件检查model242文件、ops60文件。只读入口57cb439、控制桥7bec3b8均已push；核心6a8e057字节未改变，不覆盖旧worktree或GTOK驱动。

PF1通过ro通道，对mean3和十二诊断配置逐一读冻结config、核arm/seed唯一变化、构造CPU模型并生成与真实CUDA入口对应的预期binding；所有配置train86785、validation2520、catalog120415。数据加载一次，逐配置源/参数/binding检查，不调用训练循环、不写队列。此前只核mean的范围缺口已补齐。

| config / run_id | arm | seed | total parameters | config SHA256 | source |
|---|---|---:|---:|---|---|
| p-seqm1-attn-s2024 | attn | 2024 | 7766081 | `ea46a5c37d4f9264df69f6a9bc10b158c9fcb2264d3e5e5a028ee728957570b2` | PF1 |
| p-seqm1-attn-s2025 | attn | 2025 | 7766081 | `8630b0a44dad0d60ecb4c5fe82ae404324f12d38ceda46bbab0677f5bbd54d54` | PF1 |
| p-seqm1-attn-s42 | attn | 42 | 7766081 | `c926d551cfc286f01d6ada490cef309d3d54f32e9f5a8d18104ceda20644af67` | PF1 |
| p-seqm1-bi-s2024 | bi | 2024 | 7761857 | `aab45c7afa4021e2ddb57a112a54da3d06f6ebbe5652bdc773d1696f3b17058b` | PF1 |
| p-seqm1-bi-s2025 | bi | 2025 | 7761857 | `f20ee82a7f7b2769b47c6ef1dcf0c1627b90b502e46cfae6228870405dfecad2` | PF1 |
| p-seqm1-bi-s42 | bi | 42 | 7761857 | `edde9a01dffe2838adccb72e7afb6479306b7dcc1ff15484eec8867114159f4b` | PF1 |
| p-seqm1-last-s2024 | last | 2024 | 7761857 | `da6994842ccc0177535a118ccd9716665d22865f5c954948413e9f9512fd604a` | PF1 |
| p-seqm1-last-s2025 | last | 2025 | 7761857 | `14c4c0917ffe4ad047cb23c92f1fe9b829ba990be48d6768df40c9e9abb5e46c` | PF1 |
| p-seqm1-last-s42 | last | 42 | 7761857 | `0cfe43b8d9ff270a99a0304d2bd0f76d5b29653f66e6cf63c7f427e580f92793` | PF1 |
| p-seqm1-mean-s2024 | mean | 2024 | 7761857 | `3e1e40f200c3a09a39e15048df33657bbb54080a7da33d604d92c41518a5dde5` | PF1 |
| p-seqm1-mean-s2025 | mean | 2025 | 7761857 | `e4f8f612991ee960b496067b905cb57bb46d7d2b9e4c4af9026004a53e8713f3` | PF1 |
| p-seqm1-mean-s42 | mean | 42 | 7761857 | `e88dc757ff4085af020efc1f070ebba8df54876e85adf68120dff730f983b621` | PF1 |
| p-seqm1-pure-s2024 | pure | 2024 | 7761856 | `12543bfa1b7e710f6161db84e811df02e7b4bb983a571d48d24b795546d1d39e` | PF1 |
| p-seqm1-pure-s2025 | pure | 2025 | 7761856 | `922564d5ad82f7e34960b7d0373428106c7d4ff6d7c8510c77227acf4843c287` | PF1 |
| p-seqm1-pure-s42 | pure | 42 | 7761856 | `4f9b209ebaaadda2827ea97e1c85a388ab9b8a6f9b6150afffbc90fe732d162f` | PF1 |

上述参数量/binding是工程身份，不是召回指标。配置hash和各数据hash完整保存在PF1及服务器plan.json；每个job都绑定其预期config/source/input身份。原SEQM1-000与五臂CUDA/padding/VRAM证据未因运维入口增加而改变；声明每个1024 MiB（原工程测量×1.15向上取整256）。

### 2. 提交前review、dry-run与mean三seed接纳

一次独立review覆盖正式工具、十五配置与接续状态机。发现`settled()`漏`launch_failed/wall_cap_exceeded`真实终态，已fresh RED→GREEN补齐；真实filesystem夹具验证verified绑定可放固定十二作业、supervisor exit0但没有verified会ALERT阻断、rejected不重投、失败不重跑。最终34项analysis tests通过（包含入口修复回归），没有重复review。

首次dry-run在写队列前失败（python -c无`__file__`，无法定位清单），原失败收据保留；固定SHA模块控制桥已修复，D1正式dry-run十五项launcher.prepare全部通过。没有因此消耗科学训练attempt、没有重启或覆盖。

S1只提交mean三seed；C1独立ro确认三个全部claimed、无对应rejected/.reason，十二诊断尚未写入队列。每个资源：1核、RAM9216 MiB、VRAM1024 MiB、disk8 GiB、wall8h；wall是上限，不是实测ETA。各自对应verify由下述verifier自动补（1核、RAM9216、VRAM1024、disk1 GiB、wall3h）。

| mean run_id | claimed queue file | seed | verify | source |
|---|---|---:|---|---|
| p-seqm1-mean-s42 | `01792089820907115150-p-seqm1-mean-s42.json` | 42 | planned, not yet accepted | C1 |
| p-seqm1-mean-s2024 | `01792089820917751090-p-seqm1-mean-s2024.json` | 2024 | planned, not yet accepted | C1 |
| p-seqm1-mean-s2025 | `01792089820927911397-p-seqm1-mean-s2025.json` | 2025 | planned, not yet accepted | C1 |

### 3. 已部署的自动接续：采用主方案A（金丝雀），未用十五项一次性入队备选

- dispatcher pid2916804、verifier/reporter pid2916803，C1实测两者alive、PPID1，启动UTC `2026-10-15T18:46:28Z`附近；900s轮询，不调用LLM，截止从plan创建起48h，绝不重置。
- dispatcher只看mean的训练/verify收据，至少一个训练succeeded且其verify直接verified后，固定提交last/attn/bi/pure×42/2024/2025十二训练。放行前检查全部mean已观测状态，不看R/N；任何已观测mean失败或终态verify未verified则写ALERT并停止接续。未完成/pending的verify属于等待，不当作失败。
- verifier/reporter独立继续给已经启动且成功的训练配verify，避免接续停止时其它mean漏验；rejected读取.reason、不重投；失败attempt不重跑、不覆盖。若放行后其它mean才出现失败，汇总写ALERT，已提交作业保留，不撤回或覆盖。
- C1时mean_gate=WAIT，三个mean running，诊断十二个not_submitted，所有M1尚未science verified。当前没有M1质量结论。

plan：`/workspace/vkgr-seqm1-night-20261016/plan.json`；角色journal分别`dispatcher.jsonl`、`verifier.jsonl`；接续终态`dispatch-terminal.json`。每次汇总新建`summary-<timestamp>.json`，不覆盖已有文件；全终态写`summary-final.json`，48h未收口写`summary-deadline.json`并ALERT。字段覆盖每run train/verify状态、run_id、预期/实际commit、config/data hashes、best/stop epoch、R/N、ablation_g0/Δhits、branch_ratio_median、异常及unaccepted；未verified的质量字段保持null，不当作结果。pure的ablation_g0/branch_ratio按模型定义为不适用。

### 4. 当前资源与后续验收

H1最后状态：`host  GPU 43997/49140 MiB free  RAM 163877/202084 MiB free  CPU quota 10.24 cores, used 5.07, idle 50%, throttled 35262 (17200s)  disk 213.2/500.0 GiB free` (H1)。M1三mean均真实worker运行；GTOK原批次继续由P1/P2自动补verify，Codex未重提。T1中M1尚无已结束样本，只能显示wall_cap，真实训练/verify耗时UNKNOWN；不把8h/3h当ETA。

本轮初始状态分类：已完成=十五preflight、正式dry-run、三mean接纳与双role运行证明；仍运行=三mean及原GTOK未收口作业；失败=新M1科学attempt暂未观测到失败（首次dry是已修复的运维入口错误）；未验收=全部M1十五run，其中十二诊断尚未启动。后续以新收据更新，不把尚未提交写成失败。

按Eddy明确选择**不创建晨间定时任务**。一次heartbeat创建曾被自动审批因项目禁新增timer拒绝，没有创建成功、没有绕行；原automations保持暂停。醒来后读取既有机器汇总并更新STATE四分类，不由模型夜间反复轮询。FEAT-04组合已由Kiro登记F4-T010+C+D+E，但本轮未实现/未入队；其它候选均不在今晚队列。

## 2026-10-16 晨间 M1 / GTOK-02 验收（Codex；最终，全部科学验收通过）

### 收据与边界
本节替换本晨连接阻塞时的临时快照；失败连接收据/旧脚本/旧socket保留。Eddy明确批准仅修本机连接恢复工具，修复已版本化push，未删除socket、未伪造actual exit、未重启或重投任何科学作业。计数口径：M1十五训练+十五verify，GTOK昨夜六补seed训练+六verify；另复核既有六s42及六fit，不重复算作昨夜新增。没有读test/checkpoint/完整embedding，没有新训练、timer或科学规则变更。context运行时计数UNKNOWN。

- **M1-R1**：`cache/pgr01-p-recovery-v1/main05/codex-morning-m1-20261016T080001.stdout`，SHA256 `795c21e6614418b1c6e5a662954eba8e0009c9ebf150fea579e05b9a1a0d2acd`，实际读取UTC `2026-10-16T02:04:15.274813+00:00`，完成UTC `None`，transport完成UTC `2026-10-16T02:04:15.935521+00:00`。
- **M1-R2**：`cache/pgr01-p-recovery-v1/main05/codex-morning-m1-fresh-20261016T102000.stdout`，SHA256 `9587030389f099aa3f2609ed31e1604227a4ae0f23e66758bf6b927ad8ac894e`，实际读取UTC `2026-10-16T02:20:51.870888+00:00`，完成UTC `2026-10-16T02:20:58.569008+00:00`，transport完成UTC `2026-10-16T02:20:59.235687+00:00`。
- **G-R1**：`cache/pgr01-p-recovery-v1/main05/codex-morning-gtok-repaired-20261016T104000.stdout`，SHA256 `4e156161c0bb608145c20047c09bdb857b61f6bf22f7d95376a015857e00e960`，实际读取UTC `2026-10-16T02:20:53.177999+00:00`，完成UTC `2026-10-16T02:21:05.697739+00:00`，transport完成UTC `2026-10-16T02:21:06.578146+00:00`。
- **H-R1**：`cache/pgr01-p-recovery-v1/main05/codex-morning-host-repaired-20261016T104100.stdout`，SHA256 `54aed0c4fd2c8a31409ce56718ece3d988164d30336bc868fc1085abca13d06d`，实际读取UTC `None`，完成UTC `None`，transport完成UTC `2026-10-16T02:21:09.086805+00:00`。
H-R1没有内部UTC，只使用明确标注的transport完成时间。前缀中的080001/104000等都不是实际读取时间。完整source/transport、artifact SHA、身份与逐项验证日志见上述stdout及`cache/codex-morning-acceptance-20261016/receipt-index-final.json`。

### 1. 自动接续、终态及当前资源
M1-R2重新直接读取全部十五train manifest、verify原始stdout与对应claimed job，逐项要求terminal=succeeded、stdout JSON `status=verified`、job argv指定同一个manifest和SHA；模型/config/真实preflight binding匹配。hash绑定的validation逐query归档重算R@50/N@50及固定2520query/623 OOV检查全PASS。**十五train succeeded、十五verify verified；异常/未验收/对应rejected均为零。**
M1-R2证明dispatcher在UTC `2026-10-15T19:16:28.294002+00:00`（03:16:28+08）自动发布恰好十二诊断；mean金丝雀verify的真实ended_at早于发布，历史与fresh manifest SHA相同。每项发布文件在当前claimed中，payload逐字段等于冻结plan，不是仅action=written。固定十二项都已完成，不是未提交/待接续。
M1-R1汇总final时间04:16:29+08；M1-R2扫描项目terminal得到最后科学终态`p-seqm1-pure-s2025-verify`，04:11:49.740447+08。M1两role及GTOK原P1/P2两driver均已退出，P1/P2 journal均有all_terminal=true和逐项verified；正常收口，不重启。H-R1在约10:21+08只有sched-serve-02运行，科学worker=0、pending=0；GPU空闲49139/49140MiB、CPU used0.01/10.24核、RAM空闲178182/202084MiB、磁盘空闲210.9/500GiB。
**空档说明**：项目最后终态至当前快照约6小时9分；并非整夜不断训练。该间隔依据项目terminal与当前队列，无持续host监测，外部未知进程历史不作推断。已主动告知Eddy；夜间有限批次用尽后，没有完成前置的其它训练可自动入队，不越界启动M2/A1新SID/FEAT组合。旧historical failed行不属于本批失败。

### 2. M1逐run验收与精确指标
所有行source=M1-R2；selection.best_metric顺序R@50、N@50，best/stop来自实际selection。g0Δ=带分支hits减g置零hits（不是对旧F4的增益）；gate与ratio取最佳epoch。每run完整commit/config/data hash、manifest SHA、verify stdout/job SHA和终态均在M1-R2 states及read_artifacts_sha256。

| run_id | hits | R@50 | N@50 | best/stop | g0 Δhits | ratio median | gate | train / verify |
|---|---:|---:|---:|---|---:|---:|---:|---|
| p-seqm1-mean-s42 | 466 | 0.18492063492063493 | 0.054675733733997554 | 12/15 | 109 | 0.573334097862 | -0.114620760083 | succeeded / verified |
| p-seqm1-mean-s2024 | 481 | 0.19087301587301586 | 0.055926352399775565 | 7/10 | 97 | 0.505180716515 | -0.0854303240776 | succeeded / verified |
| p-seqm1-mean-s2025 | 471 | 0.1869047619047619 | 0.05639525992000974 | 11/14 | 107 | 0.557538330555 | -0.109124176204 | succeeded / verified |
| p-seqm1-last-s42 | 475 | 0.1884920634920635 | 0.05635278408064815 | 10/13 | 129 | 0.458594024181 | -0.0656691715121 | succeeded / verified |
| p-seqm1-last-s2024 | 485 | 0.19246031746031747 | 0.054626618546255806 | 9/12 | 151 | 0.469797253609 | 0.0682647079229 | succeeded / verified |
| p-seqm1-last-s2025 | 476 | 0.18888888888888888 | 0.05611134440255852 | 11/14 | 123 | 0.482529222965 | -0.0690595954657 | succeeded / verified |
| p-seqm1-attn-s42 | 468 | 0.18571428571428572 | 0.05542851491860466 | 11/14 | 116 | 0.577811539173 | -0.102165065706 | succeeded / verified |
| p-seqm1-attn-s2024 | 473 | 0.1876984126984127 | 0.05673137459193701 | 7/10 | 89 | 0.509287178516 | -0.0766288787127 | succeeded / verified |
| p-seqm1-attn-s2025 | 472 | 0.1873015873015873 | 0.05742954042122517 | 11/14 | 97 | 0.55447781086 | -0.100818887353 | succeeded / verified |
| p-seqm1-bi-s42 | 482 | 0.19126984126984126 | 0.05846345452773587 | 10/13 | 120 | 0.560475945473 | -0.0922843068838 | succeeded / verified |
| p-seqm1-bi-s2024 | 484 | 0.19206349206349208 | 0.057259133807466195 | 11/14 | 120 | 0.575288236141 | -0.0942410975695 | succeeded / verified |
| p-seqm1-bi-s2025 | 486 | 0.19285714285714287 | 0.05797233087201667 | 10/13 | 113 | 0.559504389763 | -0.0951567143202 | succeeded / verified |
| p-seqm1-pure-s42 | 417 | 0.16547619047619047 | 0.05232136222172015 | 8/11 | N/A | N/A | N/A | succeeded / verified |
| p-seqm1-pure-s2024 | 413 | 0.1638888888888889 | 0.0511685199481289 | 7/10 | N/A | N/A | N/A | succeeded / verified |
| p-seqm1-pure-s2025 | 396 | 0.15714285714285714 | 0.05115700284883114 | 5/8 | N/A | N/A | N/A | succeeded / verified |

模型固定`0ca61bc35ad68bd79b86103b54e470445cbab07f`；核心ops `6a8e057bb021ed50dbaacc92d75c7a048b5bf45a`；pure的g0/ratio/gate为N/A，不当作零。

### 3. M1机械规则及配对统计（解读/转正交Kiro）
mean相对F4-T010同seed439/434/444：Δhits=+27/+47/+27，均值+33.6666666667，k+=3，p_pool=2.93683611418e-06（only_new281、only_control180）。§8.1机械第2行“有效”；逐seedexact p均<.05（3/3）。p_pool按冻结规则聚合重复query跨seed，非独立、偏乐观，不能当7560独立用户。
§8.2只看mean：D>−6、k0=3且分支启用3/3，两个条件机械满足；只意味着**可交Kiro裁决是否开始M2设计卡**，绝不自动入队。mean g0Δ=109/97/107，正贡献3/3；四个锚定臂ratio全部在[.05,3]，按§8.5分支启用但未压过anchor。Opus关于ratio可能受g噪声影响的限制保留，g0并列报告，不加新门槛。

| 臂 / seed | Δhits vs F4 | exact McNemar 双侧p | ΔR bootstrap 95%CI | ΔN bootstrap 95%CI |
|---|---:|---:|---|---|
| mean / 42 | +27 | 0.0376452399164 | [0.000793650793651, 0.0206349206349] | [0.000234128444237, 0.00620143548908] |
| mean / 2024 | +47 | 0.000146283947293 | [0.00912698412698, 0.0281746031746] | [0.00221547169944, 0.00795481103479] |
| mean / 2025 | +27 | 0.0364227525377 | [0.00119047619048, 0.0206349206349] | [0.000328246141244, 0.0066917706478] |
| last / 42 | +36 | 0.00170949278392 | [0.00555555555556, 0.0230158730159] | [0.00228264867736, 0.00753579503608] |
| last / 2024 | +51 | 5.73972746967e-05 | [0.0107142857143, 0.0297619047619] | [0.000825128560038, 0.0068111019836] |
| last / 2025 | +32 | 0.0055360251639 | [0.00396825396825, 0.0214285714286] | [0.000378849435578, 0.00616121784553] |
| attn / 42 | +29 | 0.0251280321903 | [0.0015873015873, 0.0214285714286] | [0.000862751294084, 0.00715814818305] |
| attn / 2024 | +39 | 0.00188773116028 | [0.00595238095238, 0.025] | [0.00280379435937, 0.00894065482506] |
| attn / 2025 | +28 | 0.0303103156438 | [0.0015873015873, 0.0210317460317] | [0.00116724348099, 0.00792825842819] |
| bi / 42 | +43 | 0.000489635748168 | [0.00793650793651, 0.0265873015873] | [0.00401148450463, 0.0100877085705] |
| bi / 2024 | +50 | 0.000684779960031 | [0.00873015873016, 0.0313492063492] | [0.00294358772959, 0.00995883181508] |
| bi / 2025 | +42 | 0.00191406448275 | [0.00634920634921, 0.0269841269841] | [0.00164189524922, 0.00860504565821] |
| pure / 42 | -22 | 0.182253428222 | [-0.0210317460317, 0.00357142857143] | [-0.00320514078536, 0.00498208145059] |
| pure / 2024 | -21 | 0.229424464629 | [-0.0210317460317, 0.0047619047619] | [-0.00393766580157, 0.00455386121469] |
| pure / 2025 | -48 | 0.00170879698089 | [-0.0305555555556, -0.00753968253968] | [-0.00584772726272, 0.00241029607423] |

query-paired bootstrap10000次、seed0；R/N均为绝对差，非百分比相对提升。每pair交并/discordants与ΔN精确值见M1-R2。

| 臂 | 三seed mean hits | mean Δhits vs F4 | p_pool（描述性） | 正g0 seed数 |
|---|---:|---:|---:|---:|
| mean | 472.666666667 | 33.6666666667 | 2.93683611418e-06 | 3 |
| last | 478.666666667 | 39.6666666667 | 4.22228888791e-09 | 3 |
| attn | 471 | 32 | 9.63956248968e-06 | 3 |
| bi | 484 | 45 | 5.39366942803e-09 | 3 |
| pure | 408.666666667 | -30.3333333333 | 0.00100998236243 | N/A |

按§3.6机械代表=bi，三seed均值484，比mean高11.3333333333≥6；Bonferroni×4 p_pool=2.15746777121e-08。这是预注册四选一的**探索性**代表，仍有赢者诅咒，不替代mean主结论、不能直接转正。bi对mean逐seedΔ=+16/+3/+15，exact p=0.198181341363/0.872440802605/0.199158398186，三者均不显著，不将“代表胜出”写成已证bi显著优于mean。
§8.3(A)描述性：last−mean均值+6（池化诊断）；attn−mean为−1.6666666667；bi−mean+11.3333333333。§8.3(B) E_start=mean(last−pure)=70，E_cond=mean(pure)−259=149.6666666667，机械落“两者都有贡献；不可加、不拆比例”行；S1仅seed42，条件是一整包，最终机制措辞交Kiro。
**转正未判**：统计条件已满足，仍需确认新模型独立泄漏review与§3.11延迟报告，另由Kiro写decision_log；P anchor仍F4-T010，不自动换mean/bi。

### 4. GTOK-02三seed及Q1/P2机械判读
source=G-R1。原六s42、六补seed的train均succeeded，verify原始stdout均直接verified且job/manifest SHA匹配；validation归档selfcheck PASS。六fit audit/verify也重读verified。昨夜新增六train+六verify全验收，不重复算s42和fit。

| run_id | hits | R@50精确 | N@50精确 | best/stop | train wall秒 | verify wall秒 | train / verify |
|---|---:|---:|---:|---|---:|---:|---|
| g-tok2-b1024d3-s42 | 280 | 0.1111111111111111 | 0.034541834269608385 | 14/17 | 6926.87976861 | 660.932022095 | succeeded / verified |
| g-tok2-b1024d3-s2024 | 285 | 0.1130952380952381 | 0.0342232368793491 | 18/21 | 8506.66200972 | 661.981784105 | succeeded / verified |
| g-tok2-b1024d3-s2025 | 283 | 0.1123015873015873 | 0.03491204370865006 | 16/19 | 7790.34998155 | 662.448660135 | succeeded / verified |
| g-tok2-b512d3-s42 | 263 | 0.10436507936507937 | 0.033095016709313634 | 14/17 | 6968.60335541 | 799.45651412 | succeeded / verified |
| g-tok2-b512d3-s2024 | 273 | 0.10833333333333334 | 0.03378161918376801 | 11/14 | 5820.28575945 | 817.700701952 | succeeded / verified |
| g-tok2-b512d3-s2025 | 267 | 0.10595238095238095 | 0.03142778200603079 | 13/16 | 6647.39197111 | 802.972028971 | succeeded / verified |
| g-tok2-b256d2-s42 | 233 | 0.09246031746031746 | 0.02949205959184739 | 10/13 | 5400.6532321 | 803.308275938 | succeeded / verified |
| g-tok2-b256d2-s2024 | 280 | 0.1111111111111111 | 0.03376347192346512 | 16/19 | 7767.62071419 | 823.402167559 | succeeded / verified |
| g-tok2-b256d2-s2025 | 262 | 0.10396825396825397 | 0.032031248326339426 | 20/23 | 9335.28413129 | 816.279796362 | succeeded / verified |

这些是core03，不是A1指标。三seed均值：B1024=282.6666666667、B512=267.6666666667、B256D2=258.3333333333；T3=267.6666666667。s42参与P1选型，正式Q1只用独立补seed2024/2025，不用三seed均值替代Q1。

| Q1比较（seed2024/2025） | Δhits | ±6机械状态 |
|---|---|---|
| B1024−T3 | +11 / +8 | UP |
| B512−T3 | −1 / −8 | FLAT |
| B1024−B512 | +12 / +16 | UP |
| B256D2−T3 | +6 / −13 | FLAT |

**Q1机械分支**：至少一个UP → still_rising；所测范围内1024较512高，按§3.8取1024侧。此处不选择A1确认run或换G anchor，入选范围交Kiro。大叉数同时改变有效语义层数/后缀，不能写成单一“更细语义”或因果解释。
**P2**：B256D2位置比较FLAT；三seed均值相对T3为−9.3333333333，绝对差≤12，按预注册±12行，对“第三层取值稀疏与前两层格子平均约1.8视频相关”的原推断落“得到支持”机械分支（仅推断；不构成因果证据），不据此证明碰撞/补零因果。B256D2补零四列B1实测未归档、常零位置logp(0)诊断仍UNKNOWN；raw3 B1通过，不把预期违规值当实测。新叉数VRAM峰值缺口仍在，每run3072是原冻结保守声明，不伪称本晨实测。

| fit | 叶子数 | 最大叶 | 完整SID碰撞 | 每层利用率（包括suffix） | B1 / verify |
|---|---:|---:|---:|---|---|
| g-tok2-b1024d3-fit | 120415 | 1 | 0 | 1/0.115234375/0.0009765625/1 | raw audit passed / verified |
| g-tok2-b128d3-fit | 120415 | 1 | 0 | 1/1/0.0625/1 | raw audit passed / verified |
| g-tok2-b256d2-fit | 65536 | 2 | 0 | 1/1/0.00390625/1 | raw audit passed / verified |
| g-tok2-b32d3-fit | 32768 | 4 | 0 | 1/1/1/1 | raw audit passed / verified |
| g-tok2-b512d3-fit | 120415 | 1 | 0 | 1/0.4609375/0.001953125/1 | raw audit passed / verified |
| g-tok2-b64d3-fit | 120415 | 1 | 0 | 1/1/0.46875/1 | raw audit passed / verified |

拟合表仅审计数字，不是召回；每个audit/manifest/verify SHA与path均在G-R1 fits。

### 5. 验收分类、遗留与工程验证
M1本批实测（15样本，M1-R2 terminal）：train最小/中位/最大 8.105/13.366/15.550min；verify 3.779/5.929/6.500min。GTOK补seed六样本（G-R1）：train 97.005/129.650/155.588min，verify11.033/13.494/13.723min。均为本批成本回溯，非下一批ETA；身份/逐项wall秒见experiment_log和收据。

- **已完成**：M1十五train+十五verify、十二诊断自动接续身份/时间/队列；GTOK六补seedtrain+六verify；既有六s42/六fit复核。
- **仍在运行**：本批科学作业0；仅scheduler服务，不计训练。
- **失败**：本批科学attempt0；本晨连接运维失败已修复，原错误记录保留；历史失败行不混入。
- **未验收**：本批science run0。M1转正所需专项review/延迟与GTOK审计/显存/常零诊断缺口属于尚未补齐的专项证据，不掩盖成全部报告完备。
工程工具：a2ee053为只读验收器（一次review修复fresh canary绑定和缺失值UNKNOWN），7511063为Eddy批准的本机连接恢复；38项tests通过，公开扫描无新增秘密/地址/本机路径命中。固定Git blob安装main05，旧脚本备份和旧socket保留；服务器仍只经kiro_b_run.sh读取。

## 2026-10-16 FEAT04-COMBO engineering and admission receipts (Codex)

Receipt paths are relative to repository root. UTC from the stdout field when present; otherwise use explicitly labelled transport completion. Filename prefixes are not read timestamps.

- **H0**: `cache/pgr01-p-recovery-v1/main05/codex-feat04combo-initial-20261016T104500.stdout`, SHA256 `6ca4321fc603190eb3297cb6a35b81ec2da335e4129a632dcc4b27409e9d254d`, actual read UTC `None`; transport completion UTC `2026-10-16T02:29:41.918775+00:00`.
- **I1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-inventory-20261016T105300.stdout`, SHA256 `f734f3f04dc0a028bac2b91b171e335c3fc6c2106d31f23a23a21762923e0e0a`, actual read UTC `2026-10-16T02:49:10.188712+00:00`; transport completion UTC `2026-10-16T02:49:10.316974+00:00`.
- **DEP1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-deploy-20261016T105400.stdout`, SHA256 `d9513c651441dddd99c86a53595b30673ad7a69c807a950d779b00928c3dd5a0`, actual read UTC `None`; transport completion UTC `2026-10-16T02:49:43.989557+00:00`.
- **PF1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-preflight-20261016T105600.stdout`, SHA256 `4be5665a07ef861c227d0eb884dcda92f487ef5a0e7db08bb2a45941993d2a9e`, actual read UTC `2026-10-16T02:51:27.660827+00:00`; transport completion UTC `2026-10-16T02:51:28.474476+00:00`.
- **SD1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-cuda-dry-20261016T105900.stdout`, SHA256 `3b911fdf6f6e1bee939483ca702f5e605ee97a8293a2d3dbf9fc612a1e120bc8`, actual read UTC `None`; transport completion UTC `2026-10-16T02:51:45.759612+00:00`.
- **SS1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-cuda-submit-20261016T110000.stdout`, SHA256 `7ed241a3fe3fbb302655f1ac7d36b60feabf4374edd77075eeef0cb1935d201f`, actual read UTC `None`; transport completion UTC `2026-10-16T02:52:09.537179+00:00`.
- **SC1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-cuda-accepted-20261016T110100.stdout`, SHA256 `b557203dd10710bfa6512aef1d67aa5ad01854f14c8ab05e5f8b8d5933c8300f`, actual read UTC `2026-10-16T02:52:52.521972+00:00`; transport completion UTC `2026-10-16T02:52:52.669976+00:00`.
- **D1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-real-dry-20261016T110200.stdout`, SHA256 `a1d297b1b63d18a9bcb114c79d7ff3f7de412db23629644fa8cfe7b617d01ada`, actual read UTC `None`; transport completion UTC `2026-10-16T02:53:35.929789+00:00`.
- **S1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-real-submit-20261016T110300.stdout`, SHA256 `0e680d2d4649d3c9fc6dd012af3ac293c0cd15c8f12b5fcd66fbeba298c6a666`, actual read UTC `None`; transport completion UTC `2026-10-16T02:54:44.980323+00:00`.
- **C0**: `cache/pgr01-p-recovery-v1/main05/codex-combo-real-accepted-20261016T110400.stdout`, SHA256 `1e0603dbe9ed68d7c56444d82bf40b43ac254a792dff88a7ae59fe5ba604edda`, actual read UTC `2026-10-16T02:55:13.157093+00:00`; transport completion UTC `2026-10-16T02:55:14.782722+00:00`.
- **DS1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-driver-start-20261016T110500.stdout`, SHA256 `53bb3bcd8be83e2fc11daab77c9df7490a6700003c00236d875c770f070d44d8`, actual read UTC `None`; transport completion UTC `2026-10-16T02:56:11.176502+00:00`.
- **C1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-final-acceptance-20261016T110600.stdout`, SHA256 `93c4923709c0ef1204bd61da80da339978646628ec54b3d6010c12034ce9e00c`, actual read UTC `2026-10-16T02:56:32.549162+00:00`; transport completion UTC `2026-10-16T02:56:34.221821+00:00`.
- **H1**: `cache/pgr01-p-recovery-v1/main05/codex-combo-final-host-20261016T110600.stdout`, SHA256 `da4da75a7f274e9763701602aa306a89f59ac5556e7167ccf2071c558afcb99f`, actual read UTC `None`; transport completion UTC `2026-10-16T02:56:40.510017+00:00`.

Full source/transport pins: `cache/codex-feat04combo-20261016/receipt-index.json`; machine summary: `engineering-acceptance-summary.json` in the same directory.

| run_id | seed | config SHA256 | preflight | final claimed file | VRAM MiB | source |
|---|---:|---|---|---|---:|---|
| p-feat04combo-cde-s42 | 42 | `1d80f5693f1df28b023609c12b944f1d30cd0438ac8a44509cfc3351114a69be` | preflight_ok | `01792119284823621063-p-feat04combo-cde-s42.json` | 1024 | PF1/C1/H1 |
| p-feat04combo-cde-s2024 | 2024 | `9bb49d82c1be1d2754a2aaee71ac43a49c0e37af58dd86edb88f7543bd2a58da` | preflight_ok | `01792119284837515097-p-feat04combo-cde-s2024.json` | 1024 | PF1/C1/H1 |
| p-feat04combo-cde-s2025 | 2025 | `0ff523e5c19cd0d3aa9dc9ea6c5cbfd506fbedd6dc810f4ff3e1902b7d8de450` | preflight_ok | `01792119284848626732-p-feat04combo-cde-s2025.json` | 1024 | PF1/C1/H1 |

### 范围、实现与验证
沿 decision_log 夜间§4执行 F4-T010+C+D+E，三个固定seed；用户塔完整复用E（Q2+6行为权重），候选塔为id+supported×(C两表+D两表)，history_items不加候选统计。旧A–E模块、配置与冻结件不变（相对5d51228仅新增文件）。T0.10/N1/L50/small/lr0.001/100-5-3及计划§1.2判据不变，不把E换A、不引入M1或新SID。组合−E仅描述性，C9仅报告。

本地旧FEAT04b基线236项PASS；fresh RED→GREEN后，最终相关联合回归130项PASS（其中新增15项，不能把重叠基线相加）。测试日志保存在本节cache目录。一次独立Sol review未发现组合公式或新增泄漏问题；快速rejected阻断其它seed verify的问题已RED→GREEN修复。首次poll已超期无机器汇总的问题按有限deadline交付要求上调处理，也已RED→GREEN修复。无第二次review、无科学规则变更。

冻结覆盖216文件；`configs/feat04combo/freeze-v1.json` SHA256 `04ae4dcc0fb1ec56ea50c4ae692606dd002198c8aa82f9080c722175bd0da90c`。本地commit `c4e4eb8c2a9d87d42303e2c0021e168942fda33a`，等树公开commit `84b160cb0eeb3c53b877f6cf6474276ba06a6ba5`，tree `acf222d2c34c6d5485d9184940018910d0596498`；parent为已公开4f16de24，其树等于原5d51228。新增公开历史扫描零命中，文件均≤1MiB；不force、不改旧历史、不push main。DEP1在新目录`/workspace/vkgr-feat04combo-p-v1`固定SHA检出，逐文件核对286文件；原p3c数据通道override保持既有c89b6466…64bb3274b5f，旧部署/数据不动。OPS=`/workspace/vkgr-feat04combo-ops-v1`。部署前I1确认所有新run_id在queue/runs均不存在。

### 真实只读preflight与CUDA/VRAM
PF1三个配置均`preflight_ok`；每seed检查512训练样本及全部2520个validation query，catalog120415、supported21496。用户向量起点与control逐位相等，候选统计只影响supported，unsupported严格零；翻target/target_position/oov不改变用户输入与向量。四组桶buffer的catalog/supported直方图、config/source/data和预期CUDA binding hash均已归档。仅CPU只读，无真实训练或test读取。

输入spec SHA256 `08ed500966e506a2599beb2bb2807330165b82f13ec2b8df01286bdd769493d6`，stats manifest `eb41bc060c67549a0cb7bf96c627000bb90b73cb5111ca11b21bf572573c036b`，统计内容digest仍25f6cc4c…f48b685；复用原审计sidecar，未重新准备/覆盖统计、改边界或用validation聚合。

SD1/SS1/SC1：合成CUDA工程run `p-feat04combo-cuda-v1`已唯一claimed且无rejected，terminal=succeeded，wall9.885975599秒。其新建合成checkpoint的verify直接`verified`（best epoch2、20steps），不是正式召回指标。fullshape probe为batch512/history50/catalog120415/queries2520；loss有限，max allocated233MiB、max reserved246MiB、mem_get_info delta358MiB，保守实测max=358MiB。声明max(1024,ceil(358×1.15/256)×256)=1024MiB。此为工程预演实测，不冒充正式训练全程峰值；H1当时640MiB进程占用也不作为峰值。

### 正式接纳、驱动与当前状态
D1三个真实job launcher.prepare全通过；Eddy在S1提交前已收到汇报。S1只提交三个固定seed。C0最初一个claimed、两个pending，未补投；C1独立只读核三个全部唯一claimed，pending/rejected/.reason均空。H1直接看到三个科学worker running（s42 evaluate、s2024 train、s2025初始化phase未标），pending0，不能仅凭written作此结论。

每train声明1CPU/RAM9216MiB/VRAM1024MiB/disk8GiB/wall8h cap；每verify1CPU/同RAM和VRAM/disk1GiB/wall3h cap。cap不是ETA，组合真实train/verify耗时UNKNOWN。H1 host CPU used3.78/10.24核（正常准入约8.19核）、GPU空闲47845/49140MiB、RAM空闲170608/202084MiB、disk空闲210.9/500GiB；三train声明合计3核，没有额外新实验填空。

DS1/C1驱动pid2921056实测alive、当前PPID1、起始PPID1；900秒轮询/no LLM，plan固定deadline_epoch1792292084.793988（创建起48h，不重置）。成功训练后独占补各verify，读取其原始stdout `verified`并绑定job/manifest SHA；失败/rejected读reason、写ALERT，不重跑/重投/覆盖，其它成功seed继续验收。OPS含plan.json、driver-pid.json、driver-launcher.log、不可变summary-<ns>.json；全终态summary-final.json，截止summary-deadline.json。逐run记录身份/hash、状态、best/stop、R@50→N@50、最佳beta、异常/unaccepted；质量字段只在verified后填。没有app timer或模型轮询，不依赖本机休眠状态。

分类：工程gate/三train接纳/driver证明已完成；三个真实训练仍运行；截至C1/H1未观测本批科学失败；三个science run及其计划verify均未验收，尚无组合效果或转正结论。原M1/GTOK维持晨间已验收，不混数。

原D-1及23:2x更正沿用：sum_likes105700对train examples86785，93.0%物品likes==target_freq，p_like0.2115<0.30；C9报告提醒不触发，也不恢复为门槛。train内全train统计含自己的target及后续train行，validation不入统计，不能作热度/质量因果解释。后续按原§1.2对F4-T010判定，另报组合−E及最佳beta符号/大小。

Kiro提示见`docs/kiro_opus_morning_review_20261016.md`；没有自主确认新SID的A1名单或启动M2/G-COLL/F7/G5，没有test/既有真实checkpoint/完整embedding读取、删除/tmp清理、付费、main/force push/改历史、U1/C9门槛变更或timer。context计数UNKNOWN。

F1 official forecast: `cache/pgr01-p-recovery-v1/main05/codex-combo-final-forecast-20261016T111000.stdout`, SHA256 `77d5ea7dff8a71c95698c571ac9da496a324f1b0735c27dadf35e12db455f6ec`; no internal read UTC, transport completion UTC `2026-10-16T03:02:29.964035+00:00`. The three combo jobs have no ended samples; displayed remaining 7h52m/7h53m is wall_cap, not ETA. Full source/transport pins are in receipt-index.json.

### 收尾接收Kiro最新晨间裁决（规则仍以decision_log为准）
本轮收尾发现decision_log新增Kiro/Opus晨间§1–5：认可已有M1独立review；服务器延迟口径核对前P仍F4，核对后按预注册切M1-mean且不追溯本组合；M2仅设计/评审；A1只确认B1024三seed且前置全齐才提交。未由Codex代写关键裁决、未据此在本轮追加新训练。
报告补上Kiro指出的g0读法：晨间M1-R2 mean带分支hits466/481/471，g0Δ109/97/107，因此g0后357/384/364，远低于原F4。m和分支已共同适应，g0Δ不是“额外约100命中”，净增益仍只看对F4的+33.67。来源晨间M1-R2原始manifest/selection/ablation（完整SHA/actual UTC见晨间receipt-index-final.json）及decision_log最新晨间§1.5。
静态工程预检：本机seq/m1的scripts/seqm1_latency_bench.py把非50长度近似为均匀分布、offline最多20次；尚未满足最新晨间要求的真实validation长度分布/1000计时，不得拿旧量级参考切anchor。后续先版本化适配，CPU/A6000同脚本报告，再交Kiro只核口径；当前未跑新延迟作业，未读既有真实checkpoint。


## 2026-10-16 SUCCESSOR：组合终态接收、M1正式延迟报告、B1024 A1工程

Codex执行；context计数UNKNOWN。规则仅沿decision_log最新Opus晨间裁决，未写anchor/转正结论。各stdout路径、完整SHA256、内部实际UTC或明确标记UNKNOWN见`cache/codex-successor-engineering-20261016/receipt-index.json`；文件名前缀不是实际时间。延迟原始1000次计时审计见同目录`latency-audit.json`。没有读test。

### 1. FEAT组合三seed已全验收（C2）
来源`main05/codex-successor-combo-read-20261016T112000.stdout`，实际读取UTC 2026-10-16T04:43:12.800296+00:00，SHA256 `7888e2fd00f3b4f056c3f13c3b4b299484903eac2dbdab561f0d440fe14ecf8f`。三个train/verify均succeeded，直接原始verify stdout的verified与claimed job/manifest SHA绑定逐项通过；无rejected/异常，原driver正常退出，不重启/重投。

| seed | hits | R@50 | N@50 | best/stop | ΔF4 | ΔE（描述性） |
|---|---:|---:|---:|---|---:|---:|
| 42 | 484 | 0.19206349206349208 | 0.056063402858042594 | 6/9 | +45 | +24 |
| 2024 | 502 | 0.1992063492063492 | 0.058068432008195094 | 7/10 | +68 | +36 |
| 2025 | 490 | 0.19444444444444445 | 0.0575455869204088 | 8/11 | +46 | +29 |

均值hits492，对F4平均+53，对E平均+29.666666666666668。3/3逐seedΔ≥6，效果量条件满足；**逐query McNemar/bootstrap尚未计算，转正统计条件未验收**。不以效果量替代配对检验，也不据此追溯更换本批对照。D-1、C9报告项和23:2x统计限制沿原文，不作因果解释。

| seed | like | share | bookmark | open_comments | click_on_author | dislike |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.43899357318878174 | 0.047422122210264206 | 0.13658851385116577 | 0.018275881186127663 | 0.13237041234970093 | -0.08134528994560242 |
| 2024 | 0.4396224319934845 | 0.031002836301922798 | 0.12849189341068268 | -0.00018890452338382602 | 0.16394829750061035 | -0.1976643204689026 |
| 2025 | 0.4329977333545685 | 0.0750708132982254 | 0.11950550973415375 | 0.003130621975287795 | 0.22469916939735413 | -0.20026381313800812 |

### 2. M1正式延迟报告（L2，口径待Kiro核）
run `p-seqm1-latency-report-v1`终态succeeded，wall395.69554686546326s。独立只读曾确认唯一claimed且无rejected；本项是工程基准作业，不是训练，没有伪造训练verify。读数来源`main05/codex-latency-result-20261016T132000.stdout`，实际读取UTC 2026-10-16T05:11:16.233790+00:00，stdout SHA256 `1a40d6226dbadb5804e6d3782e5ef552b76c5e4a93840fc7f8d1446a9253261e`；服务器`/workspace/vkgr-infra01-root/runs/p-seqm1-latency-report-v1/out/report.json` SHA256 `249f55574045d46c3087d88760034b187a77f98e1f3763ec0d401b34493624db`。源码codex/m1-latency-real@b089b8f2da3b72d115f3df743f1e5461392270f6，script SHA ed0daa90dc09d531238d2c4bbe2fa37a0f81f5653b4c9da07d53ad3a04c72c6f；运维8811c5284c1afdf55546ef735d5cc7c5a8f207f3；模型src/config与原冻结版本字节不变。

真实validation仅取2520条历史长度，源SHA c703bbcc3e3004a58a2a14c2c7c2420b51b3b7ae0c47f6b284d587e67819ab1a，长度数组SHA e4fe754fba97c5fe95d905b6d527f905350faed2ef7064ef38e659b75feac5c8，L50有1580条。最大余数法固定分配计时长度并seed7打乱；每模型同输入SHA。offline512、online1；CPU单线程后A6000顺序执行，50预热/1000计时每模型每模式；CUDA同步；数据生成和H2D不计入。另跑一次完整2520长度导出遍历（512/512/512/512/472五次）独立报告，不拿五次当1000次。

**边界**：仅`model.users()`，history/side的值是合成的，长度分布真实；初始化权重，没有读既有checkpoint。无特征取数、物品塔、ANN、网络、KV缓存，不是服务端到端延迟；合成id的cache行为不代表真实特征分布。离线单位是每query（batch耗时除512），在线batch1；不能把GPU batch1比CPU慢解读为完整服务表现。无数值通过阈值。

单位均ms/query，p50/p95/p99为nearest rank ceil(q*n)。已从全部raw样本重算mean和三个分位数，8行均一致；输入SHA、1000计时、50预热和完整导出五次全部通过。

| device | model | mode | mean | p50 | p95 | p99 |
|---|---|---|---:|---:|---:|---:|
| cpu | anchor | offline512 | 0.024146869 | 0.023963217 | 0.025876217 | 0.027719998 |
| cpu | mean | offline512 | 0.412940403 | 0.413399336 | 0.423254725 | 0.428272799 |
| cpu | anchor | online1 | 0.592868637 | 0.593383098 | 0.604384113 | 0.613934128 |
| cpu | mean | online1 | 1.744423641 | 1.761041116 | 1.853301190 | 1.875081100 |
| cuda | anchor | offline512 | 0.004220154 | 0.004222838 | 0.004317312 | 0.004508525 |
| cuda | mean | offline512 | 0.012925273 | 0.012898867 | 0.013242521 | 0.014133289 |
| cuda | anchor | online1 | 2.117529798 | 2.092363080 | 2.229033969 | 2.295193961 |
| cuda | mean | online1 | 5.853951691 | 5.843315972 | 5.962647032 | 6.087996997 |

mean/F4平均耗时比：CPU offline17.1012×、online2.9423×；A6000 offline3.06275×、online2.76452×。报告工程验收完成，**等待Kiro只核口径，P anchor暂仍F4-T010**；不以无门槛的耗时比例自行阻断或批准转正，不重复已有泄漏review。

### 3. B1024 A1工程（后续准入收据在下节补）
69项新增/相关G6回归通过，一次独立提交review无阻断。新入口仅正式sid_v2s registry/适配与实际部署source binding，原模型/train/decode不变，三config仅SID变化；SID fit verified。冻结源c1c4b7413b4ae794894fc78b98828ab26029a487已公开工作分支push，新增15文件/新历史扫描零命中且均≤1MiB（g-public-scan.json）。新DEP /workspace/vkgr-gtok02-a1-v1、OPS /workspace/vkgr-gtok02-a1-ops-v1，284逐文件SHA核对。沿既有prepared消费身份保留base0376+完整Git overlay/p3c数据通道，只读旧部署；pinned SHA a48995dd9de21cd6e60fb8d9b1fe9327bde03128198a1d13c9444c12f74bf07f，manifest另绑定真实public target，不能只写0376。真实preflight与CUDA/VRAM尚在工程gate，未称正式训练已入队。

B1024 PF-R/PF-S：独立ro真实preflight实际UTC2026-10-16T05:17:52.256947+00:00，随后prepare保存实际UTC2026-10-16T05:20:53.571694+00:00；三seed全部source/config/data/expected binding精确相同。路径、SHA、两次新scratch目录见receipt-index，均保留，不清理；没有真实训练或指标读取。


### 4. B1024 CUDA工程启动失败（未产生CUDA/VRAM证明）
最后一次独立读取工程证据实际UTC2026-10-16T05:28:37.914371+00:00仍无terminal/engineering.json。诊断实际UTC2026-10-16T05:29:37.638672+00:00：cuda-v1只有run.json（66316字节）及supervisor.log（67字节），日志在05:21:58Z写run.json unreadable; nothing to supervise；没有owner/stdout/stderr/terminal。worker未启动，**不是已完成的CUDA预演，也没有显存测量**；三真实训练仍从未入队。完整诊断stdout SHA/source及实际UTC见receipt-index；issue I-2026-10-16-b1024-launch-record。

本机对共享common.READ_LIMIT=65536重现：有效66316字节synthetic JSON被read_json_limited拒绝。prepare dry漏了最终run元数据上限；full Git overlay的dirty_files及pinned_files账本重复使记录越界。原源码科学入口/preflight有效性不受此启动问题否定，但不能据此绕过CUDA gate。原claim/run/log保留，不写假终态、不重启/重提。准备新clean Git launcher载体与原DEP全量逐文件guard、record-size余量守卫，只改运载层；修复及preflight/dry可先完成，重新工程执行前按失败attempt边界请Eddy批准。M1延迟报告及M2草稿不受此阻塞。


### 5. B1024启动层修复已工程验收，等待Eddy批准一次新工程执行
修复只改运载层：codex/phase2-readonly-tools@e3df6db5b712c8781bb683cec8fe2e41f1eb0aa3，clean CODE `/workspace/vkgr-gtok02-a1-launch-v2/code`、CTL其父目录，79文件全部Git blob/文件SHA及clean HEAD核对通过。每job入口先原DEP完整逐文件校验、固定c1c4b74模型target及pinned SHA，再从原DEP执行科学module/probe；model __file__、原prepared身份及配置不变。保留kind real/admission；不调共享64KiB限制、不删减原证据、不改旧DEP/pin/attempt。run记录改为clean launcher身份，科学manifest仍绑定原模型target/pin；两者分别可追。

回归：fresh RED→GREEN，新增10tests；实现全tests41/41、邻近analysis40/40通过；父端和一次独立focused review均运行10/10，review无阻断，限制为服务器前置需另外读取。源scan新增历史零命中/均≤1MiB后push。clean-launch-acceptance.json及bridge-public-scan.json保存工程验收摘要，收据完整路径/SHA/实际UTC或明确transport完成时刻见receipt-index。

新路由真实只读preflight实际UTC2026-10-16T05:40:52.521759+00:00，三seed entries与原保存PF逐项完全一致（engineering=False/synthetic=False/closed及source/config/data/expected binding）。新scratch保留、路径入索引，不清理。正式工程dry返回smoke_dry_ok：**spec+prepared预算3229字节**，小于48KiB守卫（留16KiB给运行元数据）；这是dry预算，不是尚未生成的新run.json实测大小。prepared为clean Git/no pins inline，原DEP证据由受guard的输入manifest和逐文件校验保留。

**状态**：新工程cuda-v2仅dry，未提交，三真实seed仍未提交，verify driver尚未启动。已向Eddy请求仅一次修复后工程执行授权；依据decision_log晨间§5“失败attempt不重跑不覆盖”，不因换ID绕过。Eddy批准后只提交一次cuda-v2并ro核claimed/rejected；成功后读真实shape/VRAM/重放证明，再正式三seed dry→submit→claimed验收→新900s/48h driver。旧cuda-v1所有记录保留；如新attempt失败不再自动重试，不伪造终态。


### 6. Eddy批准后的cuda-v2唯一执行（工程等待）
Eddy本轮明确批准一次修复后工程预演，通过后继续原三seed。授权边界记录approved-execution.json；先ro inventory实际UTC2026-10-16T05:48:21.545979+00:00确认新工程/三seed及所有verify不存在，再仅一次提交。独立只读05:49:15.083152Z确认cuda-v2唯一claimed文件01792129719661511098-g-gtok02-a1-b1024d3-cuda-v2.json，无pending/rejected/.reason。host读到真实worker2924171 running（加载输入，CPU约1核、GPU当时尚未分配），不只凭claimed判断。host没有内部读取UTC，transport完成时刻单独标在receipt-index。原cuda-v1未动，三真实train尚未提交。


### 7. cuda-v2工程通过，三真实seed正式提交与driver已启动
E1来源main05/codex-b1024-v2-engineering-read-20261016T055200Z.stdout，实际UTC2026-10-16T05:52:00.682084+00:00，stdout SHA256 `a9804011fcae521676cf6d87ace1cb88efa8bf853b806faca3defe5ec2905200`。cuda-v2终态succeeded/exit0/wall155.80976510047913s，产物直接status=cuda_engineering_verified。此为工程证明，不是召回训练或正式科学verify。完整真实codec shape vocab3075/depth3/items120415/examples86785/queries2520；batch512/micro256×2/history50/encoder200，三合成梯度步骤finite_loss，单步1.5418609641492367/1.1878685720730573/1.1390871340408921s。未用真实训练rows，未读test或既有checkpoint；beam100/3query serial-batched完全相同，新工程checkpoint reload相同，SHA d0fe43e10b4462380fec77640c2b6ed82d5d6a671e57acefeb1ba376ab9ebcc7。

VRAM：max allocated4960.294921875MiB，max reserved5332MiB，最大全局空闲差5442MiB，最大本PID nvidia-smi5704MiB；保守实测max=5704MiB。ceil(5704×1.15/256)×256=6656MiB。**声明恰与旧值相同，但由新独立实测计算，不是沿用旧峰值，也不是正式训练全程峰值。**

D2三个真实job launcher.prepare及48KiB metadata guard全通过，科学PF/新probe claimed逐字段绑定通过；S2随后只提交三个原seed，run_id g-gtok02-a1-b1024d3-s42/-s2024/-s2025。源模型c1c4b74、clean运载e3df6db、各config/pin/data哈希沿冻结，不增加臂/seed/规则。每个1CPU/RAM18432MiB/VRAM6656MiB/disk4GiB/wall24h cap；对应verify1核/同RAM VRAM/disk1GiB/wall2h cap，cap不当ETA，真实新配置墙钟UNKNOWN。三个训练声明合计3核/19968MiB VRAM，低于约8.19核/39312MiB准入预算，实际由scheduler逐项放行。

C3（实际UTC05:53:13.256920Z）为s42唯一claimed、另两pending，无rejected；C4（05:53:43.965694Z）当时仍同样状态，不能写三项全claimed。scheduler源码规定每tick最多启动一项、间隔30秒；继续独立核最终接纳，不重提。工具启动说明曾把三项claimed写早，已向Eddy更正，事实以这些raw receipts为准。

DS2/C4：新driver pid2924901，始/当前PPID1且alive，poll_seconds900、固定deadline_epoch1792302771.520567（plan创建起48h，不重置）。driver从clean bridge --driver启动，保留路由、逐job source guard和48KiB准入检查，no LLM。plan/driver/immutable summary写原OPS新文件；每train成功后独占补其verify、只认原始stdout verified+job/manifest绑定，failed/rejected不重跑，继续其它成功seed的verify。当前三科学run仍未验收，未生成科学效果/anchor结论。


### 8. 最终接纳/实际三worker复核（C5/H2）
C5实际读取UTC `2026-10-16T05:54:55.786188+00:00`，main05/codex-b1024-final-admission-20261016T055600Z.stdout SHA256 `217b1fc5968d690b757cb395d8c2167ba01df86d13cade37f021e75f8cf1093f`：三seed均唯一claimed，pending/rejected/.reason均空；三个planned verify均未入队、未验收，未把planned写成verified。driver2924901 alive、当前PPID1、初始PPID1、900秒，deadline本地 `2026-10-18T13:52:51.520567+08:00`，保持原plan固定截止。三state均无异常，train尚无terminal（正常running），不据此写效果或anchor判定。逐config/data/source身份记录final-admission-summary.json。

H2独立host确认真实worker s42=2924730、s2024=2925082、s2025=2925183均running，初始输入准备阶段CPU分别约100%/102%/100%，GPU当时尚未分配；host CPU3.02/10.24核，RAM free174845/202084MiB，GPU free49139/49140MiB，disk free210.3/500GiB，未超过约8.19核准入预算。没有内部UTC，明确使用receipt-index所列transport完成时刻；“0hxx/24h”是已用/cap，不是训练ETA。三个已并行，且后续仅固定各自verify自动接续；没有未冻结新实验的无限自动队列。

当前part：工程/正式接纳/driver交付完成；三科学train/verify仍待验收，按原A1+T3同seed控制289/328/310及冻结三条件机械套用后交Kiro/Opus，不直接改G anchor。M1延迟仍待Kiro核口径，M2仍DRAFT；旧v1全部保留，无重启/删除/伪造终态。本轮实现代码早已push，主事实源仅提交自己的文件，AGENTS/decision等既有改动保留。context计数UNKNOWN。

### 9. 15:07+08用户状态询问：当前是G线A1+B1024，非M1
本轮只读，不提交/重启任何job。状态源main05/codex-b1024-user-progress-20261016T061500Z.stdout，SHA256 06dab31f92e06bc7b3097fa115300067d31a1d43eaad755054bcb8c17bd98398，内部实际UTC2026-10-16T07:06:37.788595+00:00（文件名前缀不是实际时刻）：三train尚无terminal，verify均未开始/未verified，异常数组均空；driver2924901 alive/当前PPID1，无summary-final。
host源main05/codex-current-host-20261016T070700Z.stdout，SHA256 1ea025315a6cc5c8d249115087581f78ad84528fb7435ad4f1d3bb14d544a87e，内部UTC UNKNOWN，明确以transport完成UTC2026-10-16T07:06:52.118584+00:00标记；三worker均running/evaluate。s42当前轮1792/2520 query、已用约1h13m；s2024为1280/2520、约1h13m；s2025为1280/2520、约1h12m。这是当前评估轮的query进度，不是整体训练完成比例；未读取epoch序号，不猜总ETA。24h是cap。
host CPU3.02/10.24核，GPU free32002/49140MiB，三进程当时各5706MiB；后者是瞬时占用，不冒充全程峰值。M1十五训练/十五verify已晨间全验收，延迟补测也已完成；M2只获设计/评审授权，拟唯一K4因果多兴趣配置×三seed/各verify，未实现冻结或训练。完整source/transport哈希在receipt-index本轮新增两项。

## 2026-10-16 下午：P anchor执行记账与FEAT组合配对统计

按decision_log下午§1，P anchor自该条起为M1-mean（466/481/471 hits），来源仍为晨间M1-R2原verified/manifest/预测收据；M1延迟SUCCESSOR§2的口径已由Kiro/Opus通过。这是执行既有裁决，不由Codex新增决定。FEAT组合仍对原F4判，不追溯M1，G5等待组合判定。

新读数收据main05/codex-combo-paired-afternoon-20261016T080000Z.stdout，SHA256 9b8f02faaccab8bf84cf2d266cd354dfddba751a6bf446f14621c0095cad9a5b；内部实际读取UTC2026-10-16T07:26:59.452774+00:00、完成UTC07:27:01.851367+00:00，前缀不是实际时间。9个run（组合/F4/E各三）重新核raw verified、job/manifest/config/data/source及逐query预测SHA与顺序；2520queries/OOV623/catalog120415不变。完整READS在stdout，摘要与来源见cache/codex-m2-execution-20261016/combo-paired-summary.json、combo-receipt.json。没有test/checkpoint/embedding读取或新submit。

算法复用固定SHA的phase2原实现：exact双侧McNemar，10000次逐query配对bootstrap，numpy.default_rng(0)、10×1000分块，同样采样索引算deltaR/deltaN，percentile95CI；没有新pooled p或多重检验规则。三seed同一批query，区间不估计seed及总体不确定性。

机械结论：组合−F4为+45/+68/+46，均值+53，3/3过6且3/3逐seed p<.05；有效与转正统计条件通过。**独立泄漏review认可由Kiro判断，不由本工具裁决转正或换anchor。** 对E为+24/+36/+29，均值29.666666666666668，仅描述，不能分离C/D或声称交互/因果效果，不增设门槛。D-1保留：全train统计含自身target及较晚train行；sum_likes105700对86785，93.0%物品likes==target_freq，p_like=.2115；C9仍报告项。

| comparison | seed | delta hits | only new/control | exact p | deltaR CI95 | deltaN CI95 |
|---|---|---:|---|---:|---|---|
| combo-F4 (primary) | 42 | 45 | 118/73 | 0.0013902601073185607 | [0.007142857142857143, 0.02857142857142857] | [0.0014255026029736917, 0.007752843133269786] |
| combo-F4 (primary) | 2024 | 68 | 140/72 | 3.493232865141978e-06 | [0.015873015873015872, 0.038492063492063494] | [0.003868239613604296, 0.01056913831910389] |
| combo-F4 (primary) | 2025 | 46 | 118/72 | 0.001042653290765216 | [0.00753968253968254, 0.02896825396825397] | [0.0013509820033960449, 0.00800426840972733] |
| combo-E (descriptive) | 42 | 24 | 86/62 | 0.05831193147429581 | [0.0, 0.01904761904761905] | [-0.000817634529159758, 0.004448198085392684] |
| combo-E (descriptive) | 2024 | 36 | 98/62 | 0.00549054656634036 | [0.004761904761904762, 0.024206349206349207] | [0.0013150772418065503, 0.006944402854126449] |
| combo-E (descriptive) | 2025 | 29 | 85/56 | 0.01805614505270096 | [0.002380952380952381, 0.020634920634920634] | [0.0005821248101141796, 0.005776798148328422] |


## 2026-10-16 M2 工程与三 seed 正式接纳（科学结果待验收）

本节要点：一次独立代码review及原三项发现闭环后冻结，全部工程门禁通过，三真实seed已运行、服务器verify驱动存活。与G尾部verify实际并行，没有等G批次收口；工程合成证明不当作科学效果，M2真实结果与完整延迟仍待后续验收。详细数值和逐收据SHA/实际UTC如下。

Authoritative design/permission: decision_log afternoon sections2-3; FINAL card preserves all five rulings verbatim. No extra design round. Public source8688f09b9f58fc5f3b3ef3f396099403222cc206; code-reviewed2d5114e4961e3f95b96ec2a6202253d562ad268c. Independent whole review found two HIGH startup issues and one MEDIUM missing control binding; all closed within the same review, final APPROVED. Fresh RED/GREEN: science23, M1 related84, ops9, engineering natural-boundary/replay1. Prior fullsuite930 retained1FAIL/1ERROR/2skip from unchanged baseline P3C pin; parent reproduced the same two on b089b8f, preserved source/hash and verified the established server data-channel copy rather than weakening it. No claim that930all passed.

Freeze65files SHAe7b3385d6ac8f15dc35afbeed0de5f2e2b391af129829e2f7db9a48a6c8baf6f; pre-push newhistory scan zero hits/files<=1MiB. Fixed-SHA Git deploy checked264files. New DEP /workspace/vkgr-seqm2-p-v1, OPS /workspace/vkgr-seqm2-ops-v1. Git P3C original retained at OPS/p3c-git-original.md, frozen data-channel c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f separately checked. Original M1 and other deployments unchanged.

All three real config preflights passed at UTC2026-10-16T08:07:22.229739+00:00: engineering=False, synthetic=False, test closed_hash_only, 86785train/2520validation/120415catalog. Original accepted M1 controls466/481/471 and source/config/data/input/manifest/prediction/rawverified pins rechecked, no checkpoint. Control audit sealed at08:09:27.415758Z; SHA9dc09a9b71026ffdccbb8ff6a57c43b09c8b478136cc71af16a65a516fc7472e, PF file SHAfa2baaa4ad608740b88bf6232a9c02d1f882480dfe33d3dbbb9dd43564e33aca. Passed data/code/control pin is carried in job inputs/env and actual M2 manifest/verify; missing/drift fails closed.

CUDA engineering p-seqm2-cuda-v1 succeeded/exit0/wall136.97236466407776s; direct proof status cuda_engineering_verified, synthetic replay status verified (new own checkpoint only). Natural256optimizer steps exercised100soft to101hard and all100 diagnostic steps101-200. Synthetic head counts132/102/95/71 are engineering fixture only, not research results; zero head coverage would not stop training. CUDA zero-start loss difference4.76837158203125e-7<=1e-6, gate gradient0.08811980485916138, branch nonzero gradient elements0. Full512batch x50history x4interests, 120415catalog exact evaluation on2520synthetic queries plus fullshape g0 passed. No ANN or real training rows in this proof.

VRAM: own-PID observed peak932MiB, max allocated393.26416015625MiB, max reserved556MiB; measured bound932MiB. ceil(932*1.15/256)*256=1280MiB per job, newly measured rather than copied M1 declaration. Engineering samples/allocator/context accounting in rawproof. This is engineering fullshape evidence, not the actual real-training lifetime peak.

Formal dry3passed with48KiB startup-metadata budget; actualentry normalizes template-B and driver bootstraps infra03. Eddy was informed before one-shot submission of p-seqm2-k4-s42/-s2024/-s2025. C1 initially2claimed/1pending. Final C2 readUTC2026-10-16T08:22:09.571741+00:00: all3 uniquely claimed, no pending/rejected/reasons. Each declares1CPU,9216MiBRAM,1280MiBVRAM,8GiBdisk,8htrain cap; verify1GiB/3hcap, sameCPU/RAM/VRAM. Cap is not ETA. Total new declaration3cores/3840MiBVRAM; scheduler applies actual capacity alongside G.

Driver2925994 alive, initial/currentPPID1, poll900s/noLLM, fixed deadline_epoch1792311607.7094562 =2026-10-18T16:20:07.709456+08:00, not reset. Rawverified+fulljob/manifest/source binding required; failure/rejected ALERT no retry; unused heads descriptive only. Each successful train gets one verify, others continue independently. Immutable summaries/final/deadline in OPS; science3train+planned3verify not yet accepted, no M2 recall result or anchor verdict.

Final host read confirms worker2925927(s42) train,2925945(s2024) evaluate,2925974(s2025)evaluate; simultaneous G-s2024 verify2926036 stillrunning, so M2 started before G batch closed. Host CPU4.83/10.24cores, RAM free166050/202084MiB, GPUfree46408/49140MiB, diskfree209.3/500GiB. M2 process904MiB observations are instantaneous, not peaks. Host lacks internalUTC: use explicitly labelled transport completion in index.

All stdout/source/transport SHA and actualreadUTC (UNKNOWN where not emitted) are in cache/codex-m2-execution-20261016/receipt-index.json; full frozen-source/control/queue/driver proof in final-admission-summary.json, private approved report copies retained. No test/oldcheckpoint/ANN/embedding export/deletion/paid resource/main-or-force push/history rewrite/newtimer. Panchor M1-mean; combo primary remains F4 and pairedstatistics await Kiro verdict; no M1+combo stack/M3/bi/G-COLL/F7/G5 automatic training. Context counter UNKNOWN.

- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-inventory-20261016T080220Z.stdout`; SHA256 `9a486eb1e87eb5266a86c3c43fb3408f46c43e391f810c4fa17e7f319ee70a69`; readUTC `2026-10-16T08:02:21.070280+00:00`; transport-completedUTC `2026-10-16T08:02:21.196308+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-deploy-20261016T080244Z.stdout`; SHA256 `af3fa24b476877e215f133626fd7d2ea2916c9ab4365d0b509c3cc6ea1e09583`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T08:02:46.301681+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-preflight-20261016T080629Z.stdout`; SHA256 `688b45738dc87bb651d48106821300e0970afedf6d010e5f2958b8ef188694bd`; readUTC `2026-10-16T08:07:22.229739+00:00`; transport-completedUTC `2026-10-16T08:07:23.659417+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-control-seal-20261016T080924Z.stdout`; SHA256 `afea44ca94d7e5fff13cd0bd13350526d67854e71f4d4e9cda3852b61a1b0a2a`; readUTC `2026-10-16T08:09:27.415758+00:00`; transport-completedUTC `2026-10-16T08:09:27.922548+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-cuda-dry-20261016T081105Z.stdout`; SHA256 `d0411819087ab0177a1e431e1decdabb4a1e9ae17968d5a197d461b315bc73b0`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T08:11:06.043251+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-cuda-submit-20261016T081128Z.stdout`; SHA256 `fa15ae8177c1a3341d2cfc19c546e277ce8468d4806be8605b833fb6a83513e0`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T08:11:29.796096+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-cuda-claimed-20261016T081400Z.stdout`; SHA256 `b6e7fd4814842ee5638feefd0c2d93d2403968c21179c0bccc6b1a49b9004ae0`; readUTC `2026-10-16T08:14:01.475832+00:00`; transport-completedUTC `2026-10-16T08:14:01.614846+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-cuda-result-20261016T081833Z.stdout`; SHA256 `89451fd7b4e4f92fbb462afd1790b753c0a49a856833de2f7de67640475cb9ec`; readUTC `2026-10-16T08:18:33.970989+00:00`; transport-completedUTC `2026-10-16T08:18:34.114543+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-formal-dry-20261016T081944Z.stdout`; SHA256 `3c4929dda0a4614ab95260421df58586b2ca5bfde73c3afcb01b9694b2937744`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T08:19:47.245895+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-formal-submit-20261016T082005Z.stdout`; SHA256 `a078a5040304e5be75c2c2d9e60c2c37178d3d716d5ee63769b10baca2af3ab9`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T08:20:08.250232+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-first-acceptance-20261016T082114Z.stdout`; SHA256 `9463ae2cb7b4d09af9cf8d4d0f803bea1900e6c0a046b62eda386f6fcd81cb3e`; readUTC `2026-10-16T08:21:14.670709+00:00`; transport-completedUTC `2026-10-16T08:21:16.685874+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-driver-start-20261016T082148Z.stdout`; SHA256 `20a5a33c439ab7cdfa3e4f0f6cca8c17ff252c5189d28ea85a60230c7ae5eb70`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T08:21:49.051004+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-final-acceptance-20261016T082208Z.stdout`; SHA256 `76d8cd4fd88c7d942dcad163e90f8b44e795775ff356b5f50f8fd97f26cca6f7`; readUTC `2026-10-16T08:22:09.571741+00:00`; transport-completedUTC `2026-10-16T08:22:11.440352+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-running-host-20261016T082440Z.stdout`; SHA256 `ddd392ed74ce97bbec1f656e4f9d910879aae5ad01c28cfea17062de9c3b97c9`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T08:24:43.343418+00:00`.
- Receipt `cache/pgr01-p-recovery-v1/main05/codex-m2-preparation-host-20261016T075950Z.stdout`; SHA256 `e94c98a09219b555ef7784fd1c1655dab51ccfcb09e19eb9f3c825453019d7c5`; readUTC `UNKNOWN`; transport-completedUTC `2026-10-16T07:59:54.791119+00:00`.

## 2026-10-16 SID-FIX 前置：A1+B1024 三 raw verify 收口（不自动换 anchor）

来源 `main05/sidfix-a1-gate-20261016T085904Z.stdout`，SHA256 `7f1bdbba726c638c72b4924518b1559d1e6d6716cc7e8f4f45491efc931ca051`，实际读取 UTC `2026-10-16T08:59:05.636907+00:00`。原 driver 正常结束；三个 train/verify 均 succeeded，原始 stdout records 均直接 `status=verified`，job_bound=true、无异常/未验收；未重投。

| seed | hits | 对 A1+T3 Δhits | N@50 | manifest SHA256 | verify stdout SHA256 |
|---|---:|---:|---:|---|---|
| 42 | 313 | 24 | 0.0406882150170104 | a23ae9fbd417fa71767b62152a70a2dfac8622a0fb86c85f857d8575e9235e5e | 98dc13962dccd003f4fb5a598c0045044a4407fae4d25ecbedff9bf4abe6d523 |
| 2024 | 330 | 2 | 0.040103633490310917 | b042fb5eb9d6ddc9f3fbaad8d9011abcbd62e80a95116c3fa53f87ecdf23dbc8 | 48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67 |
| 2025 | 210 | -100 | 0.025632711740935014 | 16b7c7c135063153500c818a047f4aebd6dd814ea84652580b9ee728905dcfb1 | 16b0d5d8f24602777e2b2fa35e300b2dac5c6dd86a480af3ef4ed469d15569f0 |

原三条件机械核对：三 succeeded+verified 满足；Δ均值 -24.666666666666668 hits，不满足≥6；逐 seed ≥6 为 1/3，不满足≥2。仅事实记账与待 Kiro，G anchor 不变。S0 第一批仍排除 A1+B1024，之后单独补一次；S0 不读取任何 checkpoint。

### SID-FIX 准入前原 M2 状态只读核对
来源 `main05/sidfix-m2-resource-state-20261016T090151Z.stdout`，stdout SHA256 `ec8e1a6184e4ed633ca9925f1ce9569e6c8fa591cea608217236f2c6443a99da`，实际 UTC `2026-10-16T09:01:51.913535+00:00`。三原训练与三 verify 均 succeeded；每个 verify_receipt job_bound=true 且直接 records.status=verified，无 exceptions。仅收口状态核对，不替代 M2 逐 query/诊断/延迟的完整科学验收，不在此抄效果数字。未重启/重投，原 driver 仍在最后汇总前的正常轮询窗口。

## 2026-10-16 SID-FIX S0 工程与第一批接纳（结果另节）

冻结 pipeline §2；仅新增数学/IO/queue 工具。一次 whole review 的 R1–R3 由同 reviewer 关闭 APPROVED。随后真实 producer metadata 文件名、M1 nested binding、四旧缺口 UNKNOWN、schema 修复均 fresh RED→GREEN，最后40 tests PASS；复用 phase2 reader 9 tests PASS。全部新增公开历史扫描零命中、无>1MiB文件，只push工作分支。

科学工具 DEP /workspace/vkgr-sidfix-s0-code-v4 @ b2b4a1b783445712c8d8f003dec9b6f058c3e7a3；schema修复运维/授权pending工具13633c7，OPS /workspace/vkgr-sidfix-s0-ops-v4。v1–v3保留；S1未导出、拟合、训练。

C组：median=1、p90=11，nearest-rank来自train-positive item like次数；0 unsupported，0<count≤1 low，1<count≤11 mid，>11 high；OOV先单列。train-slices SHA256 6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26；history pin8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43。来源 main05/sidfix-s0-v2-discovery-20261016T090915Z.stdout，stdout SHA256 5ba6eede9515bcd92e89358a3c8226f5821be9b7b5e33fac3d4dd142983ab219，实际UTC 2026-10-16T09:09:16.636392+00:00。

33行中29行raw verified，四旧缺口B/C/三seed均值UNKNOWN，不读取它们的预测；9份映射+随机control同源确定性派生共10种SID。

第一批 sid-fix-s0-first-v1：1CPU/12288MiB RAM/0VRAM/1GiB/4h cap（非ETA）；原queue前缀01792142148793391058，worker2926716/supervisor2926715 detached。独立确认唯一claimed/pending0/rejected0/no reason，原pending仅schema修复获Eddy明确批准，备份后锁内只加字段、未重投。来源 main05/sidfix-s0-after-approved-repair-20261016T092104Z.stdout，stdout SHA256 ad1e53b8247c7835393e4dddb00b9a0325d3859bb82e22326579c4f11f8de195，实际UTC 2026-10-16T09:21:05.007169+00:00。

完整stdout/source/transport SHA及实际读取UTC见 cache/codex-sid-fix-20261016/receipt-index.json；无法取得内部UTC者标UNKNOWN，transport结束单列。结果以随后终态节为准。

### S0 A1 补批前置（已准备，尚未提交）
新DEP /workspace/vkgr-sidfix-s0-code-v5 @ 2193affb73c2405bf00170f43dec98c02a6c1f6d，OPS /workspace/vkgr-sidfix-s0-ops-v5，91文件SHA核对；45本地tests PASS。仅catalog成员检查从list改set，20个synthetic群体×200query，含OOV/重复/边界，优化前后完整C字典逐位相等（cache/popularity-equivalence.json），统计定义不变。运行中的第一批v4未改、不重启。

补批只A1T3/A1B1024×3共6行、2种SID；六原训练直接verified、无unaccepted。新真实只读preflight来源 main05/sidfix-s0-v5-supplement-preflight-20261016T093841962361Z.stdout，stdout SHA256 d3b85e446dd292d22df1dea0e0db99e835116c62a81d068e3900a8500f980621，实际UTC 2026-10-16T09:38:43.066304+00:00；dry通过。第一批终态未收之前不submit；cap仍是4h非ETA。事件等待器f9eaa60只读Linuxinotify、有限900s、无模型轮询/无后台timer，终态后再读结果。

## 2026-10-16 SID-FIX S0 最终诊断（两批 succeeded；四旧归档 B/C 未验收）

本节全部数字来自同一次最终只读收据：main05/sidfix-s0-final-results-20261016T095052Z.stdout，SHA256 055c781722a33e7b9297fa382f3611bdc783ff0e7843d5ef7e74ac48be628331，实际读取UTC 2026-10-16T09:50:53.695156+00:00。F = sid-fix-s0-first-v1；A = sid-fix-s0-a1b1024-v1。每行标 F/A，原输入/manifest/预测/verify的逐文件SHA在各result的read_artifacts_sha256及inventory.verify.jobs.evidence；canonical原字节在私有cache与服务器run/out/diagnostic/result.json，未入Git。

两个terminal均succeeded/exit0/未超cap，均唯一claimed且无pending/rejected/.reason。F wall1785.8050305843353s（29.7634171764min），A wall33.31781601905823s；工作量不同且F用了旧list成员检查，不将二者当性能对拍。F结果SHA a2e93d69bfb79af2c94e344aa5368e7604825777acc3ed6f2f6cde37c5db1924，A结果SHA ce587ba5513109652945bf656eba5fcef156f7471daadba79b9ee50be10beb41。F共33配置-seed行（29 raw verified，4 UNKNOWN）；A共6 raw verified。工具45tests、原reader9tests通过；真实重复A1T3三seed的B/C逐项相同，T3/B1024的A/D也逐项相同，确认set优化等价。

不读test/checkpoint/content或完整embedding；只读0–24周原始user/item事件、SID mapping和validation归档。以下小数展示12位有效数字，完整精度保留canonical JSON。

### A. SID结构（语义层排除suffix；B256D2排除第三位padding）

| SID | N | 语义叶子数 | N/叶子 | 最大叶 | 有效层数 | 完整SID碰撞(多余item) | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B16 | 120415 | 4096 | 29.3981933594 | 30 | 3 | 0 | F |
| random | 120415 | 4096 | 29.3981933594 | 30 | 3 | 0 | F |
| T3 | 120415 | 120415 | 1 | 1 | 3 | 0 | F/A |
| T1r | 120415 | 117105 | 1.02826523206 | 17 | 3 | 0 | F |
| B32 | 120415 | 32768 | 3.67477416992 | 4 | 3 | 0 | F |
| B64 | 120415 | 120415 | 1 | 1 | 3 | 0 | F |
| B128 | 120415 | 120415 | 1 | 1 | 3 | 0 | F |
| B512 | 120415 | 120415 | 1 | 1 | 2 | 0 | F |
| B1024 | 120415 | 120415 | 1 | 1 | 2 | 0 | F/A |
| B256D2 | 120415 | 65536 | 1.83738708496 | 2 | 2 | 0 | F |

有效层数仅为占用父节点平均孩子数>1的层数，不改GTOK-02 Q1。full SID碰撞与语义叶内多item不是同一个量。层熵为token边际分布H/log(branching)，children只对占用父prefix平均；同时保留prefix节点数与N/节点，便于换算，不外推全量效果。

逐层结构

| SID | k | 不同码数/branch | 利用率 | 归一化熵 | 父节点 | prefix节点 | 平均children | N/节点 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B16 | 1 | 16/16 | 1 | 0.999999999813 | 1 | 16 | 16 | 7525.9375 | F |
| B16 | 2 | 16/16 | 1 | 0.999999812791 | 16 | 256 | 16 | 470.37109375 | F |
| B16 | 3 | 16/16 | 1 | 0.999953147665 | 256 | 4096 | 16 | 29.3981933594 | F |
| random | 1 | 16/16 | 1 | 0.999999999813 | 1 | 16 | 16 | 7525.9375 | F |
| random | 2 | 16/16 | 1 | 0.999999812791 | 16 | 256 | 16 | 470.37109375 | F |
| random | 3 | 16/16 | 1 | 0.999953147665 | 256 | 4096 | 16 | 29.3981933594 | F |
| T3 | 1 | 256/256 | 1 | 0.999999904904 | 1 | 256 | 256 | 470.37109375 | F |
| T3 | 2 | 256/256 | 1 | 0.995801061317 | 256 | 65536 | 256 | 1.83738708496 | F |
| T3 | 3 | 2/256 | 0.0078125 | 0.124292817238 | 65536 | 120415 | 1.83738708496 | 1 | F |
| T1r | 1 | 256/256 | 1 | 0.995229264986 | 1 | 256 | 256 | 470.37109375 | F |
| T1r | 2 | 256/256 | 1 | 0.997609971967 | 256 | 47346 | 184.9453125 | 2.54329827229 | F |
| T1r | 3 | 256/256 | 1 | 0.998789009298 | 47346 | 117105 | 2.47338740337 | 1.02826523206 | F |
| B32 | 1 | 32/32 | 1 | 0.999999999692 | 1 | 32 | 32 | 3762.96875 | F |
| B32 | 2 | 32/32 | 1 | 0.999997490054 | 32 | 1024 | 32 | 117.592773438 | F |
| B32 | 3 | 32/32 | 1 | 0.997654336788 | 1024 | 32768 | 32 | 3.67477416992 | F |
| B64 | 1 | 64/64 | 1 | 0.999999991518 | 1 | 64 | 64 | 1881.484375 | F |
| B64 | 2 | 64/64 | 1 | 0.99996728108 | 64 | 4096 | 64 | 29.3981933594 | F |
| B64 | 3 | 30/64 | 0.46875 | 0.815941555956 | 4096 | 120415 | 29.3981933594 | 1 | F |
| B128 | 1 | 128/128 | 1 | 0.999999977716 | 1 | 128 | 128 | 940.7421875 | F |
| B128 | 2 | 128/128 | 1 | 0.99957452005 | 128 | 16384 | 128 | 7.34954833984 | F |
| B128 | 3 | 8/128 | 0.0625 | 0.421396868989 | 16384 | 120415 | 7.34954833984 | 1 | F |
| B512 | 1 | 512/512 | 1 | 0.999999781217 | 1 | 512 | 512 | 235.185546875 | F |
| B512 | 2 | 236/512 | 0.4609375 | 0.875508091886 | 512 | 120415 | 235.185546875 | 1 | F |
| B512 | 3 | 1/512 | 0.001953125 | -0 | 120415 | 120415 | 1 | 1 | F |
| B1024 | 1 | 1024/1024 | 1 | 0.999998740095 | 1 | 1024 | 1024 | 117.592773438 | F |
| B1024 | 2 | 118/1024 | 0.115234375 | 0.688145868662 | 1024 | 120415 | 117.592773438 | 1 | F |
| B1024 | 3 | 1/1024 | 0.0009765625 | -0 | 120415 | 120415 | 1 | 1 | F |
| B256D2 | 1 | 256/256 | 1 | 0.999999904904 | 1 | 256 | 256 | 470.37109375 | F |
| B256D2 | 2 | 256/256 | 1 | 0.995801061317 | 256 | 65536 | 256 | 1.83738708496 | F |

### B. 前缀准确率（每项精确比例=表中分子/1897；OOV623不进分母）

| 配置 | seed | top1前1/2/3分子 | top50前1/2/3分子 | 完整SID top1/top50 | item top50 hits | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| B16 | 42 | 454/88/21 | 1610/818/301 | 17/190 | 190 | F |
| B16 | 2024 | 496/86/26 | 1570/773/278 | 22/181 | 181 | F |
| B16 | 2025 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| random | 42 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| random | 2024 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| random | 2025 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| T3 | 42 | 104/16/16 | 912/257/254 | 16/254 | 254 | F |
| T3 | 2024 | 94/11/11 | 913/284/274 | 11/274 | 274 | F |
| T3 | 2025 | 93/13/13 | 895/284/275 | 13/275 | 275 | F |
| T1r | 42 | 106/19/18 | 895/243/233 | 18/233 | 233 | F |
| T1r | 2024 | 118/23/22 | 887/261/249 | 22/249 | 249 | F |
| T1r | 2025 | 91/17/17 | 908/282/266 | 17/266 | 266 | F |
| B32 | 42 | 310/18/5 | 1386/505/214 | 4/205 | 205 | F |
| B64 | 42 | 202/17/13 | 1217/326/211 | 13/211 | 211 | F |
| B128 | 42 | 158/17/16 | 1080/269/229 | 16/229 | 229 | F |
| B512 | 42 | 78/17/17 | 734/263/263 | 17/263 | 263 | F |
| B512 | 2024 | 69/17/17 | 737/273/273 | 17/273 | 273 | F |
| B512 | 2025 | 65/13/13 | 737/267/267 | 13/267 | 267 | F |
| B1024 | 42 | 54/16/16 | 619/280/280 | 16/280 | 280 | F |
| B1024 | 2024 | 58/14/14 | 605/285/285 | 14/285 | 285 | F |
| B1024 | 2025 | 47/13/13 | 597/283/283 | 13/283 | 283 | F |
| B256D2 | 42 | 78/13/13 | 898/236/236 | 13/233 | 233 | F |
| B256D2 | 2024 | 89/14/14 | 910/286/286 | 14/280 | 280 | F |
| B256D2 | 2025 | 103/15/15 | 918/271/271 | 15/262 | 262 | F |
| A1B16 | 42 | 387/77/22 | 1619/877/379 | 13/250 | 250 | F |
| A1B16 | 2024 | 430/64/17 | 1650/842/328 | 14/206 | 206 | F |
| A1B16 | 2025 | 402/72/19 | 1597/857/352 | 15/234 | 234 | F |
| A1T3 | 42 | 89/13/13 | 977/293/289 | 13/289 | 289 | F/A |
| A1T3 | 2024 | 90/19/19 | 1016/339/328 | 19/328 | 328 | F/A |
| A1T3 | 2025 | 85/19/19 | 1000/322/310 | 19/310 | 310 | F/A |
| Pmean | 42 | 不适用：P只作C参照 | — | — | — | F |
| Pmean | 2024 | 不适用：P只作C参照 | — | — | — | F |
| Pmean | 2025 | 不适用：P只作C参照 | — | — | — | F |
| A1B1024 | 42 | 41/20/20 | 642/313/313 | 20/313 | 313 | A |
| A1B1024 | 2024 | 42/16/16 | 655/330/330 | 16/330 | 330 | A |
| A1B1024 | 2025 | 18/10/10 | 466/210/210 | 10/210 | 210 | A |

B为in-catalog口径，完整SID命中与item命中分别计算后核对相同（本批full SID无碰撞）；训练Recall@50仍用2520分母。D2的k3保留存档中的常零位，因此前2/3位相同是结构事实。四旧缺口未读预测，不补零、不用训练成功代替verified。

B三seed均值（比例，不是另一次独立样本）

| 配置 | mean top1前1/2/3 | mean top50前1/2/3 | mean full SID top50 | 来源 |
| --- | --- | --- | --- | --- |
| B16 | UNKNOWN | UNKNOWN | UNKNOWN | F |
| random | UNKNOWN | UNKNOWN | UNKNOWN | F |
| T3 | 0.0511333684765/0.00702864171499/0.00702864171499 | 0.477947636619/0.144965735372/0.141099982428 | 0.141099982428 | F |
| T1r | 0.0553505535055/0.0103672465296/0.0100158144439 | 0.472676155333/0.1381128097/0.13143560007 | 0.13143560007 | F |
| B512 | 0.0372518010894/0.00825865401511/0.00825865401511 | 0.387981022667/0.141099982428/0.141099982428 | 0.141099982428 | F |
| B1024 | 0.0279388508171/0.00755578984361/0.00755578984361 | 0.319978914075/0.149007204358/0.149007204358 | 0.149007204358 | F |
| B256D2 | 0.0474433315762/0.00738007380074/0.00738007380074 | 0.479001932876/0.139342822/0.139342822 | 0.136179933228 | F |
| A1B16 | 0.214197856264/0.0374275171323/0.0101915304867 | 0.855034264628/0.452644526445/0.186083289404 | 0.121244069584 | F |
| A1T3 | 0.0463890353189/0.00896151818661/0.00896151818661 | 0.525918116324/0.167633104902/0.162888771745 | 0.162888771745 | F |
| A1B1024 | 0.0177473203303/0.00808293797224/0.00808293797224 | 0.309787383588/0.149885784572/0.149885784572 | 0.149885784572 | A |

### C. 原G4热度分档命中（相对core03 T3同seed；跨规模差只描述）

| 配置 | seed | head | mid | low | unsupported0 | OOV | 合计hits | 各档Δ对core03T3(h/m/l/u) | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B16 | 42 | 187 | 3 | 0 | 0 | 0 | 190 | -54/-9/-1/0 | F |
| B16 | 2024 | 175 | 5 | 0 | 1 | 0 | 181 | -83/-10/-1/1 | F |
| B16 | 2025 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| random | 42 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| random | 2024 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| random | 2025 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | F |
| T3 | 42 | 241 | 12 | 1 | 0 | 0 | 254 | 0/0/0/0 | F |
| T3 | 2024 | 258 | 15 | 1 | 0 | 0 | 274 | 0/0/0/0 | F |
| T3 | 2025 | 253 | 22 | 0 | 0 | 0 | 275 | 0/0/0/0 | F |
| T1r | 42 | 223 | 10 | 0 | 0 | 0 | 233 | -18/-2/-1/0 | F |
| T1r | 2024 | 235 | 14 | 0 | 0 | 0 | 249 | -23/-1/-1/0 | F |
| T1r | 2025 | 248 | 18 | 0 | 0 | 0 | 266 | -5/-4/0/0 | F |
| B32 | 42 | 203 | 2 | 0 | 0 | 0 | 205 | -38/-10/-1/0 | F |
| B64 | 42 | 202 | 9 | 0 | 0 | 0 | 211 | -39/-3/-1/0 | F |
| B128 | 42 | 222 | 7 | 0 | 0 | 0 | 229 | -19/-5/-1/0 | F |
| B512 | 42 | 243 | 20 | 0 | 0 | 0 | 263 | 2/8/-1/0 | F |
| B512 | 2024 | 252 | 21 | 0 | 0 | 0 | 273 | -6/6/-1/0 | F |
| B512 | 2025 | 243 | 23 | 1 | 0 | 0 | 267 | -10/1/1/0 | F |
| B1024 | 42 | 260 | 19 | 1 | 0 | 0 | 280 | 19/7/0/0 | F |
| B1024 | 2024 | 255 | 28 | 1 | 1 | 0 | 285 | -3/13/0/1 | F |
| B1024 | 2025 | 261 | 20 | 2 | 0 | 0 | 283 | 8/-2/2/0 | F |
| B256D2 | 42 | 221 | 12 | 0 | 0 | 0 | 233 | -20/0/-1/0 | F |
| B256D2 | 2024 | 262 | 18 | 0 | 0 | 0 | 280 | 4/3/-1/0 | F |
| B256D2 | 2025 | 234 | 27 | 1 | 0 | 0 | 262 | -19/5/1/0 | F |
| A1B16 | 42 | 236 | 14 | 0 | 0 | 0 | 250 | -5/2/-1/0 | F |
| A1B16 | 2024 | 201 | 5 | 0 | 0 | 0 | 206 | -57/-10/-1/0 | F |
| A1B16 | 2025 | 223 | 11 | 0 | 0 | 0 | 234 | -30/-11/0/0 | F |
| A1T3 | 42 | 277 | 12 | 0 | 0 | 0 | 289 | 36/0/-1/0 | F/A |
| A1T3 | 2024 | 287 | 41 | 0 | 0 | 0 | 328 | 29/26/-1/0 | F/A |
| A1T3 | 2025 | 281 | 29 | 0 | 0 | 0 | 310 | 28/7/0/0 | F/A |
| Pmean | 42 | 403 | 62 | 1 | 0 | 0 | 466 | 162/50/0/0 | F |
| Pmean | 2024 | 412 | 67 | 2 | 0 | 0 | 481 | 154/52/1/0 | F |
| Pmean | 2025 | 401 | 69 | 1 | 0 | 0 | 471 | 148/47/1/0 | F |
| A1B1024 | 42 | 296 | 17 | 0 | 0 | 0 | 313 | 55/5/-1/0 | A |
| A1B1024 | 2024 | 299 | 29 | 2 | 0 | 0 | 330 | 41/14/1/0 | A |
| A1B1024 | 2025 | 207 | 3 | 0 | 0 | 0 | 210 | -46/-19/0/0 | A |

每档query数固定：head963、mid548、low155、unsupported231、OOV623；精确每档R@50=上表hits/对应档query数。边界median1/p9011来自固定train-slices SHA6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26；低档与unsupported合计386只是补报，不合并原五档。服务器JSON的C_delta_vs_same_seed_T3对A1行实际用A1T3；本表明确重算core03 T3，下面另列同规模A1T3，避免名称歧义，不修改原产物。

A1B1024−A1T3同规模差分

| seed | Δhead | Δmid | Δlow | Δunsupported | Δhits | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 19 | 5 | 0 | 0 | 24 | A |
| 2024 | 12 | -12 | 2 | 0 | 2 | A |
| 2025 | -74 | -26 | 0 | 0 | -100 | A |

### D. 相邻共现与均匀随机前缀lift（只描述）

| SID | k1共现/随机matches | lift1 | k2共现/随机matches | lift2 | k3共现/随机matches | lift3 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B16 | 501529/226530 | 2.21396283053 | 70110/14174 | 4.94638069705 | 9331/872 | 10.7006880734 | F |
| random | 227738/227073 | 1.00292857363 | 14360/14140 | 1.01555869873 | 905/898 | 1.00779510022 | F |
| T3 | 82649/14144 | 5.84339649321 | 844/50 | 16.88 | 0/26 | 0 | F/A |
| T1r | 93739/15028 | 6.23762310354 | 2106/124 | 16.9838709677 | 131/28 | 4.67857142857 | F |
| B32 | 313223/113163 | 2.7678923323 | 28168/3448 | 8.16937354988 | 2124/114 | 18.6315789474 | F |
| B64 | 203491/56305 | 3.61408400675 | 11039/854 | 12.9262295082 | 0/26 | 0 | F |
| B128 | 131676/28298 | 4.65319103824 | 3614/219 | 16.502283105 | 0/26 | 0 | F |
| B512 | 52913/7120 | 7.4316011236 | 0/26 | 0 | 0/26 | 0 | F |
| B1024 | 32324/3421 | 9.44869921076 | 0/26 | 0 | 0/26 | 0 | F/A |
| B256D2 | 82649/14144 | 5.84339649321 | 844/50 | 16.88 | 844/50 | 16.88 | F |

所有SID均使用3630957个训练期用户内相邻对；本数据相邻同item剔除数0。随机也是3630957对，Python Random(0)，同一升序catalog、独立有放回抽两端，允许self-pair；随机分母0才UNKNOWN，本批未发生。D2的k3沿存储pad定义；D不设协同阈值。

读法限制待Kiro：T3/B1024等语义叶已单item，两个不同item的完整语义prefix共现不可能相同，故某些深层cooccurrence=0是结构必然；均匀随机含self-pair使对应random matches=26，不能把lift3=0直接读成“无协同”。random含self的技术选择已在看数之前写进工具doc与tests；冻结卡未明说随机是否排self，请Kiro核对意图，Codex不在看数后自改函数。

### §2.3冻结读法的机械分支（不下语义/记忆结论）

| 比较 | 使用seed | Δhead | Δmid | Δlow | Δunsupported | 总增益 | 头部份额 | 机械分支 | 来源 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1024−T3 | 2024,2025 | 5 | 11 | 2 | 1 | 19 | 0.263157894737 | head_share_lt_0_5 | F |
| A1B1024−A1T3 | 42,2024,2025 | -43 | -33 | 2 | 0 | -74 | UNKNOWN | no_positive_gain | A |

core03只用独立2024/2025，总增益19、头部5/19=0.2631578947368421<0.5，落原“正增益且头部不到50%”描述分支；s42只并列，不参与分支。A1三seed总增益−74≤0，机械为“无可分解的正增益”，头部份额UNKNOWN。50%没有更强依据，不是门槛；不据此修改GTOK02 Q1、G anchor或宣称机制/因果。所有结果交STATE待Kiro表，在实际Opus下复查。

S1仍DRAFT、未导出checkpoint/拟合/训练；S2–S5未立卡。

### S0 Markdown报告实际读回（最终补证）
只读观测工具98f23248ea29f55268b1e59fb566d028188cf05d已push；46相关tests PASS，原reader9tests PASS。服务器两report.md均实际读取，A–D章节齐；source main05/sidfix-s0-report-artifacts-retry1-20261016T101107Z.stdout，stdout SHA256 bee0fd2b10718a3f8f5d7bb41fabf85c2e9faf46fef11ef71d4c80e3e3ce91f2，实际UTC 2026-10-16T10:11:10.373407+00:00；两result/terminal SHA与上述最终验收完全相同。

- sid-fix-s0-a1b1024-v1：3038 bytes；report SHA256 f1cd8ca982a39e2c1719e57e5ee8a4ad2b292d4ccac5d94537d9832d5c4a5b27；路径 /workspace/vkgr-infra01-root/runs/sid-fix-s0-a1b1024-v1/out/diagnostic/report.md。
- sid-fix-s0-first-v1：9563 bytes；report SHA256 f80d10da1edc8d6febdb18542d22a89914de6f9d17779278320c75d10f602510；路径 /workspace/vkgr-infra01-root/runs/sid-fix-s0-first-v1/out/diagnostic/report.md。

完整交付索引更新为cache/codex-sid-fix-20261016/receipt-index-delivery.json（保留此前索引，不覆盖原run），报告证明report-acceptance.json，delivery-manifest.json记录科学与观测版本分离。一次自动权限审查超时发生在命令创建前，允许的一次只读重试成功，未变更科学作业。18:10+08已主动告知Eddy：本项目已知GPU科学训练在17:01检查时全部收口，此后仅CPU诊断，GPU训练空档至少1h；当前S1待Kiro冻结，没有擅自新排训练。

## 2026-10-16 SID-COLL S1执行前只读队列收据

来源main05/sidcoll-initial-queue-20261016T103741Z.stdout，SHA256 153f4e5c5670d0a648643727f35ea3a75f3ad77f48f36ca546759a618515a604；transport同前缀.json核exit0与stdoutSHA。实际服务器读取UTC未输出，故UNKNOWN（前缀不作实测时间）。pending0，只有scheduler运行，GPU空闲49139/49140MiB；RAM空闲176494/202084MiB；CPUquota10.24、当次用量0.17/idle98%；盘209.2/500GiB空闲。仅时点资源快照，不是S1训练/显存实测；本次无新作业提交。科学输入/配置仍须工程闸门，不为填空扩大实验。

## 2026-10-16 SID-COLL S1 engineering acceptance and S0 supplemental reads

Receipt: main05/sidcoll-raw-with-epochs-20261016T145310Z.stdout; SHA256 312bb2a42c69b2542e6817abb25f46e3fba6e767965e6b46fc21529657879ff2; actual read UTC 2026-10-16T14:53:11.456452+00:00. Scientific deployment079d6bd35ae74317b29408cd43d3ec35de77c1eb; observer event fix1fa96dc, original plan/deadline preserved. Per-stage immutable raw refs/hashes are in this stdout and cache/codex-sid-coll-s1-20261016/receipt-index-v1.json. No test data or unauthorized old checkpoint read.

中文读法：本节是工程验收与只读补充，不是S1召回结果。两输入独立推进；C1尚在SGNS，C2已导出验收、正在SID拟合。真实G的fit/replay、三配置PF、CUDA实测、dry、训练及逐条verify未全部通过前，R@50/N@50不写成已验收。原D保留，D′及A1曲线仅描述；四旧验收缺口不阻塞S1。

S1: input_preflight C1/C2 succeeded, unique claimed/no rejected, raw verified. C2 input/export succeeded and input_verify raw verified; C2 SID fit ongoing; C1 SGNS ongoing at this receipt. Core03 G training/verification/VRAM remain UNACCEPTED. Driver2927689 was alive/PPID1, inotify own_terminal confirmed, fallback900s, deadline 2026-10-18T14:36:46.754110+00:00. No training was restarted.

### S0 Dprime (original D unchanged)

Source main05/sidcoll-raw-acceptance-20261016T144436Z.stdout, SHA256 b92cd6fcf68f0c23d553b4365dd25913b87abcf4113d7cd2b013e55cdf4b2651; actual read UTC2026-10-16T14:44:37.360632Z. Result /workspace/vkgr-infra01-root/runs/sid-coll-s0-prime-v1/out/diagnostic/result.json SHA256 e52da316018e4e0708b9369c646828cff039bf5e53ab42f54c6a0a30b74505e8; computation UTC2026-10-16T14:44:00.035557Z. All rows use this receipt. Random(0) draws two distinct videos per pair; pairs may recur. STRUCT_NA means max prefix group<=1, never numeric zero. Descriptive, no new gate.

|SID|Pairs|Lift1|Lift2|Lift3|
|---|---:|---:|---:|---:|
|B1024|3630957|9.41567142441|STRUCT_NA|STRUCT_NA|
|B128|3630957|4.66654853457|19.9668508287|STRUCT_NA|
|B16|3630957|2.2106448627|4.95337007206|11.0165289256|
|B256D2|3630957|5.86329455165|31.2592592593|31.2592592593|
|B32|3630957|2.75804591122|8.19313554392|22.5957446809|
|B512|3630957|7.44938758271|STRUCT_NA|STRUCT_NA|
|B64|3630957|3.60518389909|13.4294403893|STRUCT_NA|
|T1r|3630957|6.34486259645|20.25|43.6666666667|
|T3|3630957|5.86329455165|31.2592592593|STRUCT_NA|
|random|3630957|1.00672363262|1.02791696492|1.0523255814|

### Four legacy archives

The original auxiliary job sid-coll-legacy-epochs-v1 FAILED on a historical queue metadata filename containing test, before epoch reads. It was not resubmitted. A separate pinned small-metadata read found no matching verified aliases; 5 historical test-named queue metadata files were skipped. This bounded scan is not an exhaustive claim about every possible alias.

- g-g4-rand-s2024: UNKNOWN (same receipt above).
- g-g4-rand-s2025: UNKNOWN (same receipt above).
- g-g4-rand-s42: UNKNOWN (same receipt above).
- g-hist-v2-l50-s2025: UNKNOWN (same receipt above).

### A1+B1024 full epoch curves (descriptive only)

Same main05 receipt/SHA/UTC as first paragraph. Each original train manifest is hash-checked against accepted S0 supplement inventory; no old checkpoint or predictions are read. Hits below are derived from exact stored R50 times count, integer tolerance checked. No rerun, seed deletion, anchor change or causal explanation.

g-gtok02-a1-b1024d3-s2024: best=14, stop=17; manifest /workspace/vkgr-infra01-root/runs/g-gtok02-a1-b1024d3-s2024/out/manifest.json SHA256 b042fb5eb9d6ddc9f3fbaad8d9011abcbd62e80a95116c3fa53f87ecdf23dbc8; actual read UTC 2026-10-16T14:53:11.539019+00:00.

|Epoch|R50 (exact stored)|N50 (exact stored)|Hits (derived)|
|---:|---:|---:|---:|
|1|0.04801587301587302|0.013202842072070517|121|
|2|0.04484126984126984|0.013774618458233396|113|
|3|0.04246031746031746|0.01918575326238369|107|
|4|0.05714285714285714|0.018396476593752903|144|
|5|0.09007936507936508|0.030095315570500693|227|
|6|0.09801587301587301|0.032145592232954684|247|
|7|0.09841269841269841|0.031146847646725703|248|
|8|0.10396825396825397|0.034507691315297594|262|
|9|0.1111111111111111|0.03406378077118028|280|
|10|0.1126984126984127|0.034206748714811025|284|
|11|0.1246031746031746|0.04083389541374624|314|
|12|0.12261904761904761|0.0389933997642364|309|
|13|0.12341269841269842|0.03967383783107948|311|
|14|0.13095238095238096|0.040103633490310917|330|
|15|0.12103174603174603|0.03683153652602335|305|
|16|0.12063492063492064|0.0355300782835684|304|
|17|0.12341269841269842|0.03753622545922327|311|

g-gtok02-a1-b1024d3-s2025: best=9, stop=12; manifest /workspace/vkgr-infra01-root/runs/g-gtok02-a1-b1024d3-s2025/out/manifest.json SHA256 16b7c7c135063153500c818a047f4aebd6dd814ea84652580b9ee728905dcfb1; actual read UTC 2026-10-16T14:53:11.543621+00:00.

|Epoch|R50 (exact stored)|N50 (exact stored)|Hits (derived)|
|---:|---:|---:|---:|
|1|0.044444444444444446|0.01404483052410095|112|
|2|0.04246031746031746|0.01396213937768721|107|
|3|0.04801587301587302|0.012813917317409752|121|
|4|0.04801587301587302|0.015899347903603107|121|
|5|0.04563492063492063|0.012411475373333035|115|
|6|0.04523809523809524|0.013702721223567424|114|
|7|0.061507936507936505|0.022861728741220555|155|
|8|0.07579365079365079|0.025302723397659772|191|
|9|0.08333333333333333|0.025632711740935014|210|
|10|0.08293650793650793|0.02847634386393113|209|
|11|0.08214285714285714|0.028508677709133543|207|
|12|0.08253968253968254|0.027137858106492437|208|

g-gtok02-a1-b1024d3-s42: best=11, stop=14; manifest /workspace/vkgr-infra01-root/runs/g-gtok02-a1-b1024d3-s42/out/manifest.json SHA256 a23ae9fbd417fa71767b62152a70a2dfac8622a0fb86c85f857d8575e9235e5e; actual read UTC 2026-10-16T14:53:11.532367+00:00.

|Epoch|R50 (exact stored)|N50 (exact stored)|Hits (derived)|
|---:|---:|---:|---:|
|1|0.04642857142857143|0.01701348543927274|117|
|2|0.048412698412698414|0.015325352789005847|122|
|3|0.054365079365079366|0.021885953525246797|137|
|4|0.07023809523809524|0.021594940335377486|177|
|5|0.08373015873015872|0.027087788587967968|211|
|6|0.09126984126984126|0.03291167157094427|230|
|7|0.0992063492063492|0.031034417054120356|250|
|8|0.11031746031746031|0.0367709265003472|278|
|9|0.10595238095238095|0.03490406044180958|267|
|10|0.11904761904761904|0.03913668218826616|300|
|11|0.12420634920634921|0.0406882150170104|313|
|12|0.12301587301587301|0.03900598736607893|310|
|13|0.12063492063492064|0.03755442290046605|304|
|14|0.12023809523809524|0.037773107356961724|303|

### C2 export/input verification audit (not recall metrics)

Receipt main05/sidcoll-producer-audit-20261016T145956Z.stdout; SHA256 74d867fc50f2722b50a2a9e974675ee13b1c36b1872a80de7257921d712dd44b; actual read UTC 2026-10-16T14:59:57.170755+00:00. Raw verification status=verified; verification JSON /workspace/vkgr-infra01-root/runs/sid-coll-c2-input_verify-v1/out/verification.json SHA256 755791b5de3480253983f0d0635c1ce87d56f4370adac3484063562f934f4f19. Producer JSON /workspace/vkgr-infra01-root/runs/sid-coll-c2-input-v1/out/payload/producer-manifest.json SHA256 9451639817dcb6703cc8f35be92ace127ecef2bbc5abd6ae993d2e931729bc0f. All numbers below use this receipt.

- Dimension 128; supported/collaborative-block 21496/120415; actual fraction 0.1785159656189013; content-only 98919; supported-zero 0.
- Fixed M1-mean s42 best checkpoint SHA256 d98feb9972c3a8c48f2f710c69770b2edd4c626299e4b8131a00077fde2df868; export script SHA256 0d7541e199dcb5397a37b6cc8eec24b9c94b3925b22100725b3fe2c935ab29fc; train sequence digest 957e52a270343380b7bcff63c0af00f0b1b899cdbe92561433d4316d9c638859.
- CONTENT_V1 array digest 766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84; transformed 128D array digest 060e98bf3bf2709d9b07120f4023ecd81fb31d3b643309e39ae9433d1d3022b8; physical vectors.npy SHA256 dd24db0083d4b9907977234ddfdacaf595298747ba013b73e6a441a2f52c8200. Array digest and physical file SHA are distinct. No vector arrays or user vectors exported to Git/chat.
- Mandatory limitations: only the supported group carries a collaborative block; the rest are content-only. The two groups are systematically separated by block presence in128D; first-layer BHKM may primarily split support/likedness. These are design properties, not evidence of improvement. C1 health/fallback UNKNOWN until its producer completes; newG R/N/VRAM UNKNOWN until their own verified gates.

## 2026-10-17 01:20 S1 progress: both G preflights FAILED; no G training submitted

Receipt main05/sidcoll-progress-20261016T172025Z.stdout SHA256 33a533c2a34f2005ee2ed37ebb58a7e8d97ae575f0fd6cdc8619168caf7ca055; actual read UTC 2026-10-16T17:20:26.765523+00:00. All stage timings/status and health values below use this receipt; original terminal/claimed/raw verified JSON refs are embedded. Global resource receipt main05/sidcoll-global-progress-20261016T172103Z.stdout SHA256 6af6a789ccf5795cde32510dc5ed1540f9fb74d065bcaa04362a2fdbf9664ade; internal read UTC UNKNOWN (prefix not substituted). Global pending0, only scheduler running, GPU49139/49140MiB free, CPU0.01/10.24 cores, idle100%, disk209.1/500GiB free.

C1/C2 input and BHKM fit-verify both directly verified (including full same-seed replay). C1 SGNS health PASS: cooccurrence0.5335246548108621 > random0.48878856149861677, difference0.04473609331224532; content fallback251/120415=0.0020844579163725447. This is an engineering health check, not evidence of recall improvement; fallback mixes representation spaces. Five epochs/173851080 positives/42445 steps recorded; no gradient/negative-trace full replay claimed.

|Stage|C1 wall seconds|C2 wall seconds|Acceptance|
|---|---:|---:|---|
|input|2335.1391184329987|15.991363525390625|succeeded; separate verify accepted|
|input_verify|64.40261220932007|14.43617033958435|raw verified|
|sid_fit|751.3833718299866|1045.2508418560028|succeeded; separate verify accepted|
|sid_verify|843.9050416946411|1149.69025182724|raw verified|
|preflight|140.4886040687561|142.45333337783813|FAILED|

Both preflights failed with AttributeError: _ValidatedCodec has no attribute depth, at sid_coll_cuda_check.context. The depth belongs to codec.catalog.depth. C2 stopped2026-10-16 23:32:17+08; C1 stopped2026-10-17 00:01:55+08. At01:20 readback, no project scientific jobs remained active for78.51min after the last failure. Driver stopped with ALERT as required, not an unexplained crash. Formal G train/verify count=0; CUDA/VRAM/dry remain UNACCEPTED. No failed attempt resubmitted. Local/pushed patch0f39d69 fixes only the property lookup, real codec regression fresh RED/GREEN,76 tests PASS; no server deployment/retry.

Schedule estimate (historical inference, not measured S1 ETA): after authorized engineering retry and remaining gates, two parallel s42+verify roughly2–3h; if both pass260, four supplementary jobs+verify add roughly2–3h if admission permits parallel execution; total about4–6h. Current automatically executable backlog=0h because both branches are sticky ALERT. Fixed48h is a deadline, not inventory; new SID training early-stop time and engineering work remain UNKNOWN. S2/S5/G5 not ready/automatically authorized.

## 2026-10-17 approved S1 repaired preflights: both passed, CUDA pending admission

G preflight receipt main05/sidcoll-retry-gates-20261016T180106Z.stdout SHA256 81273c8102a5772c1478dcd49482c05fb382aa42ad5435ca630d35b54b36aac9; actual read UTC 2026-10-16T18:01:07.435528+00:00. Both branches accepted with raw preflight_ok, engineering=False; all six configurations inspected; real shape120415 items/86785 examples/2520 queries/vocab771. Driver2930845 alive/actual PPID1, poll900, original deadline1792334206.75411.

New model deployment11d67681e0cbde09983f6bf482f55ab8a7191bdb;296 files checked, all scientific src/six G and two fit configs/job_entry bytes identical to079d6bd. Ten original successful stages reused unchanged. Only failed PF identities received approved -v2 attempts; both consumed once. No new SGNS/SID fit or teacher export. Initial admission dry placeholder defect fixed by b303f1c,79tests; this failed before any queue write and did not consume a PF attempt.

Capacity proposal receipt main05/sidcoll-capacity-proposal-20261016T181131Z.stdout SHA256 2551fc942404a31bea3459f3494fffbc06611cc8ef983b4e90a323178d2d92db; actual read UTC 2026-10-16T18:11:33.635935+00:00. CUDA probes remain unique pending, no claimed/rejected/run directory. Existing20% admission: total49140/free49139MiB, committed0/settling0; need39312 fails measured margin. Proposed39168 passes (measured_headroom0.2029100529100529). No policy threshold changed. Capacity is an engineering probe reservation, not a measured training peak.

|Pending run|Old queue SHA256|Proposed new queue SHA256|
|---|---|---|
|sid-coll-c1-cuda-v1|a43de5faea63fe602f8c649d1d695dfb1afdecac0ddd80dbf9327bb77f48d27b|29c8e3ba7b7efc95dd16d0a7dfc62bcc81e5a2f73d4ecaea7c36b01551b90acf|
|sid-coll-c2-cuda-v1|8c6d6d015ae6b9ccd8915a91952a7c2252f0ca579e9fca916db929e686e2f520|e9d0ee6b1678e28494c255a38d2fc8bfc22fb59c9c6eb71654cecb3f131d3369|

Proposed new plan SHA b14630f9b2be83b12e4d21503b7a5d9ce04f44c1b5b277ce50ff20d7a3d5753d, original deadline retained; original plan is not overwritten. Approval required by Eddy file-overwrite boundary, not by a new scientific gate. Prepared tool0b2becc,83tests, defaults readonly; not applied. No CUDA metric/VRAM measurement/G training recall exists yet.
## 2026-10-17 02:20：协同向量 SID 容量修正获批，CUDA已接续

Eddy明确批准两条never-started CUDA pending容量字段修正并继续。在scheduler锁内备份原字节/SHA，仅改vram_mib 39312→39168；原run_id/config/data/20%门禁/截止不变、不重投。原plan保留，新plan pipeline-plan-capacity-v1.json SHA256 b14630f9b2be83b12e4d21503b7a5d9ce04f44c1b5b277ce50ff20d7a3d5753d，备份在 /workspace/vkgr-sid-coll-ops-v2/capacity-repair-v1。新观察2931675实际PPID1/alive，900秒兜底，原deadline epoch1792334206.75411。

02:20:42+08只读验收：共现向量 SID（C1′）sid-coll-c1-cuda-v1已accepted，原始status=cuda_engineering_verified；真实shape B512/L50/200 tokens、beam100，3步synthetic有限梯度，serial/batched一致、新probe checkpoint回放一致。最大进程显存2036MiB，ceil256(2036×1.15)=2560MiB。未读既有checkpoint；该预演不是真实训练，未读test。dry-v1已自动pending；内容加 P 向量 SID（C2）cuda-v1已claimed、尚未终态。两CUDA均唯一claimed、无pending/rejected；正式G训练尚未提交。

来源收据（实际读取时间以stdout内部为准）：

- main05/sidcoll-capacity-approved-20261016T181735Z.stdout，SHA256 f58e8c14115c4dfa92cd0a7cf9359fa067f21b3c6109def7377911c670d77fdd，实际UTC 2026-10-16T18:17:37.826485+00:00，含两pending原/新SHA及修复记录。
- main05/sidcoll-capacity-confirm-20261016T182003Z.stdout，SHA256 0c07dc22ee52be7e9988ad3432c61b51678b45cddc4696c11099020a5e8d7227，实际UTC 2026-10-16T18:20:05.152596+00:00，含实际PPID1/alive和claimed/no rejected。
- main05/sidcoll-capacity-stage-20261016T182041Z.stdout，SHA256 f6678cd77254b7a68275489ab6c8899687304bcfd7cf6ce97e5afa7128de16be，实际UTC 2026-10-16T18:20:42.842500+00:00，含原始CUDA验收与显存样本。


## 2026-10-17 02:24：协同向量 SID 两路正式训练接纳

共现向量 SID（C1′）sid-coll-c1-train42-v1 与内容加 P 向量 SID（C2）sid-coll-c2-train42-v1均独立确认唯一claimed、无pending/rejected；全局状态两进程running（PID2932713/2933344），pending0。读取时仍处于启动加载阶段，GPU占用0不代表已产生epoch结果。尚无正式训练指标，两个verify及条件补seed未验收。

两路CUDA原始status=cuda_engineering_verified、dry原始dry_run_ok，均accepted；各测峰2036MiB、按ceil256(2036×1.15)=2560MiB用于各自训练声明。各B512/L50/200 tokens、beam100，3步synthetic有限梯度；serial/batched与新probe回放一致；未读其它既有checkpoint或test。原始verification.json SHA：C1 daefc97bf7d0e23314257c3175b1f434bb20163e827390e820959dc9486d030f；C2 16f4d7d67b4e2a089ec0f0746f9d7aeebfb74f54ceb37ae497914874830062f0。

服务器驱动2931675实际PPID1/alive，900秒兜底/文件事件、无模型，原deadline1792334206.75411不变；训练succeeded自动唯一verify，直接verified且hits≥260才补各2024/2025，失败按冻结STOP不重跑。两路工程门禁均已过，本轮83测试fresh回归通过；旧失败/旧plan/产物保留，不自动S2/S5。

来源：main05/sidcoll-two-trains-20261016T182434Z.stdout，SHA256 7e08d8e7da5d942c149aa85e6e80f6817570e2493c925323f24eb81022c2ea41，实际UTC2026-10-16T18:24:36.013170+00:00，含raw CUDA/dry验证及两训练claimed；main05/sidcoll-train-global-20261016T182426Z.stdout，SHA256 b2687320793160377d4c7a851c2fef3e9d2c79d7e4e1ca8ddb2a4c2182996a49，全局工具内部读取UTC未输出为UNKNOWN，不拿前缀冒充实测时间。详细索引cache/codex-sid-coll-s1-20261016/capacity-receipt-index.json。


## 2026-10-17 02:40：训练等待期只读复查补数（未启动新训练）

两路协同向量 SID（S1）s42的本地任务触发条件02:28重新独立核对：均唯一claimed、无pending/rejected，观察2931675实际PPID1/alive。来源main05/localtasks-trigger-20261016T182834Z.stdout SHA256 9e088adc58bbf7f02bfe40f3b4c0fa2fa8168384e13bd4b90bb83a256cf793e7，实际UTC 2026-10-16T18:28:36.086064+00:00。本轮不修改这些作业/驱动。只读工具6268807工作分支已push，22 tests；一次独立review的身份缺口经fresh RED/GREEN修复，12个已验收manifest固定SHA。所有实际部署路径已移入gitignored本地JSON，公开扫描无本机路径/IP/密钥/真实服务器部署路径。transport的rw仅以-c传入私有JSON，三个脚本没有任何服务器写操作。

### 物品热度+质量+行为权重组合（FEAT-04 条件组合）

原注册对照F4-T010：

| seed | 新/对照 hits | Δhits | b/c | exact McNemar p | ΔR@50 95% CI | ΔN@50 95% CI |
|---|---|---|---|---|---|---|
| 42 | 484/439 | +45 | 118/73 | 0.0013902601073185607 | [0.007142857142857143, 0.02857142857142857] | [0.0014255026029736917, 0.007752843133269786] |
| 2024 | 502/434 | +68 | 140/72 | 3.4932328651419779e-06 | [0.015873015873015872, 0.038492063492063494] | [0.003868239613604296, 0.01056913831910389] |
| 2025 | 490/444 | +46 | 118/72 | 0.0010426532907652161 | [0.00753968253968254, 0.02896825396825397] | [0.0013509820033960449, 0.00800426840972733] |

p_pool=6.7211836551213586e-11（Σb/Σc=376/217）；mean Δhits=53。

对序列用户塔（M1-mean），只描述，不更换注册对照：

| seed | 新/对照 hits | Δhits | b/c | exact McNemar p | ΔR@50 95% CI | ΔN@50 95% CI |
|---|---|---|---|---|---|---|
| 42 | 484/466 | +18 | 98/80 | 0.20246043236353428 | [-0.0031746031746031746, 0.01746031746031746] | [-0.0016637427087852572, 0.004429912977361445] |
| 2024 | 502/481 | +21 | 99/78 | 0.13253419223810542 | [-0.001984126984126984, 0.01865079365079365] | [-0.0009708181302984046, 0.005276939333005028] |
| 2025 | 490/471 | +19 | 102/83 | 0.18555890492457608 | [-0.0031746031746031746, 0.018253968253968255] | [-0.0022255414407312017, 0.004590511029794268] |

p_pool=0.014096472751947788（Σb/Σc=299/241）；mean Δhits=19.333333333333332。

四条件机械核对：均值Δ≥6满足；≥2 seed Δ≥6满足；≥2 seed逐seed exact p<.05满足（对F4，3/3）；独立泄漏review认可缺证据，既有C/D/E review和组合review记录供Kiro核定完整§7.6范围。不自行写转正或换anchor。C9仅报告项；D-1统计包括自身target及后续train行，validation未进统计，不能称严格时间隔离。23:2x审计sum_likes105700对训练样本86785，93.0%物品likes==target_freq，p_like=.2115≤.30提醒未触发；不证明无泄漏。细证据与review/test路径见docs/superpowers/reviews/2026-10-17-feature-combination-evidence.md。

### 多兴趣召回（M2）

固定序列用户塔（M1-mean）对照：

| seed | 新/对照 hits | Δhits | b/c | exact McNemar p | ΔR@50 95% CI | ΔN@50 95% CI |
|---|---|---|---|---|---|---|
| 42 | 441/466 | -25 | 87/112 | 0.088630250906162958 | [-0.021031746031746033, 0.0011904761904761906] | [-0.00425835978211307, 0.0033317351099699933] |
| 2024 | 447/481 | -34 | 51/85 | 0.0044812516752525244 | [-0.02261904761904762, -0.004365079365079365] | [-0.004160194240319254, 0.00210922413294579] |
| 2025 | 443/471 | -28 | 73/101 | 0.040363102046086119 | [-0.02142857142857143, -0.0007936507936507937] | [-0.00599688172160281, 0.0011137390625876006] |

p_pool=0.00013322720452796184（Σb/Σc=211/298）；mean Δhits=-29。

按冻结§8.1：D=-29、k+=0，满足行1 D≤-6；不作anchor裁决。各头wins/hard101–200选择比例/18个pairwise cos/max-ratio均已从raw verified manifest再核，完整表见docs/superpowers/reviews/2026-10-17-multi-interest-evidence.md，来源下面metadata收据；没有根据总hits反推配对。

来源main05/acceptance-paired-20261016T183959Z.stdout；SHA256 4a283900c081c0ee488a6fbfda16b16715eea487397559237eb10f3a6f023b2f；实际读取UTC 2026-10-16T18:40:00.703469+00:00，结束UTC 2026-10-16T18:40:04.197577+00:00。

所有比较共同2520 validation queries（OOV623结构零保留），固定有序query/history/catalog/support与12manifest SHA，best选择/预测SHA/R与N重算均通过。bootstrap沿原phase2 default_rng(0)、10000次query配对重采样、R/N共用索引、95% percentile；不能估计seed/user/总体不确定性。p_pool把相同query的三seed b/c相加，存在相关性而偏乐观；组合仅补充报告项，多兴趣沿原预注册定义，不新增显著性或多重比较规则。未读test/checkpoint/完整embedding，未新增训练。


### 多兴趣召回（M2）：g0、每头诊断与完整延迟盘点

| seed | best hits | g0 hits（R×2520） | best−g0 hits | g0 R@50 | g0 N@50 |
|---|---|---|---|---|---|
| 42 | 441 | 328 | +113 | 0.13015873015873017 | 0.037281042642595297 |
| 2024 | 447 | 290 | +157 | 0.11507936507936507 | 0.032299619022277411 |
| 2025 | 443 | 319 | +124 | 0.12658730158730158 | 0.034305321732547495 |

同一best模型归零g，仅消融描述，不能当相对序列用户塔的净收益；本轮读取manifest原始ablation_g0和原verify绑定，没有打开g0预测或checkpoint。每头计数/比例/cos/max-ratio完整表随证据包保留，meta收据再次固定三个原manifest SHA、source8688f09、原plan/job/config/data及原始verified。

延迟：固定OPS、三train/verify及out直层小JSON的有界盘点只有旧summary，未找到用户塔、四路exact、去重重排、合计的实际计时证据。因此各项和实测串/并行均UNKNOWN；生产代码是CPU NumPy float64分块四头打分后max再全序top50，不能把代码顺序或旧user-only延迟冒充完整计时。没有借此读取未授权的旧多兴趣checkpoint，缺口交Kiro。盘点非穷尽全服务器。

来源main05/multi-interest-meta-20261016T184013Z.stdout，SHA256 251a2fd69af94b4f5605bf0266f0e798564d7e158dcd5a68757987f84df3c26b，实际读取UTC 2026-10-16T18:40:14.237881+00:00；best/g0预测ref与SHA、所有读取小文件SHA均在此stdout，未将逐query预测进Git。

### GTOK-02：B256D2补零后四列B1只读实测

固定120415条映射，positions distinct=[256,256,1,2]。原B1函数只检查输入前三列：对补零前raw3为balanced=true/nodes65793/violations0/max_spread1；对补零后完整四列作为输入为balanced=false/nodes65793/violations54879/max_spread1。违例来自第三位常零导致某些父节点孩子数不足，不是spread超1。额外把四列都逐层统计的补充诊断nodes131329/violations54879/max_spread1/false，不作为新门禁。保留Eddy已批准的解释性偏离：正式验收仍在raw3上，不改旧结果或阈值。

来源main05/b256d2-four-column-20261016T184013Z.stdout，SHA256 9d2a249b093daef1d47a596897c7b11572c0554cba37639c2a97c7cf0697bce3，实际UTC 2026-10-16T18:40:16.209694+00:00；输入mapping SHA505d9c5ec258ea66afb653f7aa2f7be71700bfb186c021166c6a06b8e12d9453、原run_manifest SHAce565f6874fde90437e2c7595ad3a040d2f7351b04924ae5b31b51eb3647d2e8。未读test/CP，未更改mapping/bundle/门禁，没有新作业。

### SID 全量规模基准（S4）：本机合成有界成本探针

固定seed42、float32单位随机向量64维、BHKM256×3/RQ-VAE K256原PREREG300000steps/流式根节点原型。每方法独立子进程，NumPy BLAS与torch intra/inter-op均实测1线程；wall最多60s、RSS上限4096MiB，父ps采样与worker ru_maxrss分开，短暂采样间隔内超限不能绝对排除。无真实数据/旧CP读取，无模型落盘；不占服务器资源。工具ddf208e、15 tests及协同SID83回归通过；一次独立review的CLI参数忽略问题已经fresh RED/GREEN关闭，未再复审。

| N | 方法 | 状态 | 端到端墙钟s | worker墙钟s | parent采样峰值MiB | worker peak MiB | 完整suffix |
|---|---|---|---|---|---|---|---|
| 1250000 | bhkm | SKIPPED_RESOURCE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| 1250000 | rqvae | CENSORED_WALL | 21.104646 | 19.253611 | 950.828125 | 950.828125 | UNKNOWN |
| 1250000 | streaming | COMPLETE | 3.244019 | 2.326307 | 224.765625 | 224.656250 | UNKNOWN |
| 12470000 | bhkm | SKIPPED_RESOURCE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| 12470000 | rqvae | SKIPPED_RESOURCE | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| 12470000 | streaming | COMPLETE | 23.865845 | 23.130812 | 222.296875 | 222.296875 | UNKNOWN |

BHKM两规模已知数组下界为12160000000/121308160000 bytes（11.33/112.98GiB），超过4GiB工程cap因此未分配；这是估计下界，不是实测峰值。1247万RQ输入list+concat物化下界6384640000 bytes>4GiB，因此跳过（这是此worker物化方式的限制，不是RQ不可流式的证明）。125万RQ实际完成2000/300000步触发原fail_wall_projected；21.10秒是到停止的实测时间，不是完整拟合时间，也不能声称完整耗时≥60秒。流式两规模都完成一遍根节点分配、有限且计数总和等于N，但非balanced、无三层SID，无完整suffix证据。

来源cache/codex-local-evidence-20261017/synthetic-scale-results.json，SHA256 a146520289cae8266aa573207e5cc534144b6635a0a1b999bbd834294e878a4b，实际读取UTC2026-10-16T18:40:00.722782+00:00；原始输出<TMP>/vkgr-sid-scale-20261017-183816/results.json，producer未嵌时间，不能冒充其生成时间。源码与原SID实现SHA及commit见同目录receipt-index.json。新视频分配只出docs/superpowers/specs/2026-10-17-sid-assignment-draft.md草稿，不能用本基准称全量生产可行。


## 2026-10-17 上午并行批次：工程前置首轮收据

双向序列模型确认seed前置：服务器382个queue记录/396run目录，共936份固定job/config/manifest小JSON，三新seed314159/271828/161803无collision/errors；本地115refs、60distinct tip及可达git历史仅草稿登记命中，完整实验账无旧run命中。本地refs ledger SHA f5b444180271656b2568799a7684aecc66436b413762f2280e156377d734f03a。服务器来源main05/seed-usage-actual-20261016T185813Z.stdout，SHA00c0b72ce6e335cb4ac0449e0cb887f43d1ea9032071cce06b630d1907b953bb，实际UTC2026-10-16T18:58:15.092396Z。

组合补证原attempt p-combo-leakage-real-v1唯一claimed、无rejected，57.25143480300903s后exit1：输出父目录未建，FileNotFoundError。没有拿到机器可读检查结论，不认通过，也不归因科学检查失败；stdout原始为空（SHA e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855）。Eddy/Kiro明确批准仅一次工程重验；3dd6215仅修输出父目录与先stdout flush，audit_runtime源码逐字同ba96b1e，函数SHA2a2f1db4cbe3f53b10538a292f419bc1ec6ca87f974db624750bc90afeafd6ef；新v2部署/source/job保留原84b/512train+2520validation/四变体/seed/非零值与检查项，旧失败不覆盖。旧stdout无结果可逐项比，不能补造旧结果。新v2已提交，待独立终态验收。来源main05/leak-job-result-20261016T192949Z.stdout及leak-v2-*，各SHA/实际UTC在本轮receipt-index-current.json；未重跑训练。

双向确认六配置真实engineering=False PF p-biconfirm-real-preflight-v1 succeeded，直接stdout preflight_ok，86785train/2520validation/catalog120415，各source/config/data绑定齐全、Git clean；源码bfc9c99867b863d4ca23843064181dd8bf6a9fb5，wall62.127885580062866s。来源main05/bi-pf-result-20261016T194122Z.stdout，SHAadb61729b2f68a985fa039c1a443f0c905d8ec0272889569f9e4dcf381c6d9e5；实际UTC见索引。源PF file由固定生产serializer精确重建pin026e981d3d06f37f86952c024a16133335d08fed267a473fa1b14458fd719998，下一CUDA消费者将独立read_pinned重核实际文件，不把派生pin冒充额外服务器读取。CUDA工程job p-biconfirm-cuda-v1已提交，训练尚未提交。

自生成难负样本DPO（R2）工程输入PF g-rank-r2-mining-input-preflight-v1 succeeded/原始preflight_ok，wall73.9244441986084s；真实86785train/catalog120415/固定20k抽样，无validation/test/CP读取，源码974c6751f49a085680f4cbe75862bbf86b7d9f4e。来源main05/r2-pf-result-20261016T193330Z.stdout，源SHA/实际UTC见索引。

R2速度作业g-rank-r2-mining-speed-v1已独立claimed/no rejected并succeeded：128样本、beam10、b1、单核、A1新随机权重（4+4/d128/FF1024/drop.1/200positions/V771），decode+filter5.100290571106598s、25.096609343225744 query/s、128合法pairs。固定20k线性796.9204017354059s；整job79.69856977462769s，所以本次加载/构建/warmup等固定开销估74.59827920352109s，合计外推871.518680938927s（约14.5min）≤4h。仅工程微测，不是新SFTbest正式挖掘吞吐，不外推负样本命中/跳过比例；sample_seed0、sample集合SHA bc2a343ddac20bc0a9a7edc664de321bc5b5bb463408d7825e8b43c1bdfb61ca，source/runtime/input/cfg均绑receipt。来源main05/r2-speed-confirm-20261016T193909Z.stdout，SHA089148f65664ff0f7518ff517221ccb07ed0c3a9ad3b7898ec1221be347aee96，实际UTC见索引。原随机工程峰allocated仅66996736bytes，绝不是双模型DPO训练显存，声明仍UNKNOWN。按Kiro新补定可冻结R2，完整两阶段/verify/含referenceCUDA仍在实现，未提交科学训练。

本轮所有新事实来源索引cache/codex-morning-execution-20261017/receipt-index-current.json；raw stdout含运行环境等私有元数据，只存gitignored cache，不公开完整预测/embedding/checkpoint。科学指标未新增，不读test，不动原协同向量SID driver/作业。

## 2026-10-17 双向序列模型确认与组合泄漏补证：正式提交工程验收

双向确认六配置真实preflight为engineering=False；两臂CUDA实际shape B512/H50/catalog120415/validation2520。mean本PID峰866 MiB、bi862 MiB，均按乘1.15向上取整256声明1024 MiB；自身新工程checkpoint的synthetic verify两臂直接verified，未读旧checkpoint/test。source bfc9c99867b863d4ca23843064181dd8bf6a9fb5。

首轮正式dry因请求kind=train被原launcher拒绝，无训练发布、无科学结果；按常设一次工程重验仅新建请求/计划v2、kind改ops，科学绑定/数据/六seed及原created/deadline不变。v2 dry_ok；唯一driver2934201已release，实际PPID=1/alive，poll900，原deadline1792353901.9985971。六训练各一次发布：mean三条已独立claimed，bi三条pending、无rejected，属于容量排队，不能称六条均claimed。verify由同driver接续，当前均未验收。尚未读取训练selection指标，不报告科学效果。

组合p-combo-leakage-real-v2真实检查pass；原失败v1及空stdout保留，IO-only重验未改84b160c科学代码/512训练+2520validation/四变体/非零初始化/检查项。原reviewer仅补核P1，闭环文档2026-10-17-feature-combination-leakage-p1-closure.md SHA01ced7ea52f3f5b34b6c423bdf3b0f71aef1f42e1a0fc0185b98d112b8811595，APPROVED。L8实际24batches、L9只适用C/D；L6/L7/L10为静态review。P anchor未自动更改。

| 来源stdout | SHA256 | 实际读取UTC |
|---|---|---|
| main05/bi-pf-result-20261016T194122Z.stdout | adb61729b2f68a985fa039c1a443f0c905d8ec0272889569f9e4dcf381c6d9e5 | 2026-10-16T19:41:22.974883+00:00 |
| main05/bi-cuda-result-20261016T195352Z.stdout | 2bd440b51cbe3310d8a6d43963035fe1e9f459af8e4ed949938dca613942902d | 2026-10-16T19:53:53.281681+00:00 |
| main05/leak-v2-confirm-20261016T194804Z.stdout | e1f427e5390e7e32a261088a4482449da9469b43780f27930ea013b4af375cac | 2026-10-16T19:48:07.617594+00:00 |
| main05/bi-formal-dry-20261016T201100Z.stdout | 5797dc521d9c397a0a4143a91e4e023d480c0c299916d2bffd694c5945bd8a3d | 2026-10-16T20:09:35.809191+00:00 |
| main05/bi-formal-dry-v2-20261016T201630Z.stdout | 0d26ba17b80becb66dc8fab9f6bfb9d693e94042ed2ce528bd28f04cfc61b895 | 2026-10-16T20:11:46.180779+00:00 |
| main05/bi-formal-submit-20261016T201800Z.stdout | 7c3b3dcdee92ba83b9ec768a0e267f21d7b0d05238a072092bb8d59a1a7bdb72 | 2026-10-16T20:12:11.693640+00:00 |
| main05/bi-controller-read-20261016T202200Z.stdout | 97dec16ce32e175e3dd1c4752daae714b53c10d189aee174d812320860c1c8bc | 2026-10-16T20:13:15.132444+00:00 |
| main05/bi-train-confirm1-20261016T202500Z.stdout | e4a8405a1caa25b57db1458cbf5a5b7f694978d81e902d08007c388c58761547 | 2026-10-16T20:14:09.730675+00:00 |
| main05/bi-train-confirm2-20261016T202500Z.stdout | 4002eb20c15a206e1a8a50b8541d42f3e7548c2e9cc16266318f7326bff1a5ca | 2026-10-16T20:14:10.574364+00:00 |
| main05/bi-train-confirm3-20261016T202500Z.stdout | b0601a53f748c82c1aaed8a99209c3e008923783820b0892007752ea4e971542 | 2026-10-16T20:14:11.453306+00:00 |
| main05/bi-train-confirm4-20261016T202500Z.stdout | fee285eb8fbf362d3625774ed07eb58e535b719da2a89d5b0e86ff2da9747a13 | 2026-10-16T20:14:12.302952+00:00 |
| main05/bi-train-confirm5-20261016T202500Z.stdout | 1362050a09eb3f61806939e9afa83730564954134879bcfa1e0f5d3f79043973 | 2026-10-16T20:14:13.109256+00:00 |
| main05/bi-train-confirm0-20261016T202300Z.stdout | 1983337d242d4ebae278ac24c440e9e9b92827242ef4273a4671d5bc4d3db47c | 2026-10-16T20:13:34.370046+00:00 |

### 三个后续实验的部署与排队工程收据

温度交叉熵：source7a061788780333970748238cff792eca383052d5，逐文件242校验，clean新部署；新只读真实preflight已提交且最新pending/no rejected。自生成难负样本DPO：完整source277a1f14f33a82c0dd1eecf5330b4bb6ebaa0715，一次完整code review APPROVED（60 tests），逐文件250校验，clean新部署；完整runner三配置真实preflight已提交、最新pending/no rejected。旧microbench成本通过不替代该正式PF。序列用户塔+组合叠加：作者40eb70e与公开快进a7536e55e99bf7d126ae983176f205b19ef30be1树一致f317188f2416f5dca592001b6099aa0c32164277，逐文件307校验，新metadata绑定输入SHA08ed500966e506a2599beb2bb2807330165b82f13ec2b8df01286bdd769493d6、参数SHA5c92760fb51b75c62c7a154a32a2775ef77a80625f2defb9e91176463abe6870。自身真实泄漏preflight已提交且最新pending/no rejected，尚不算pass。三批CUDA dry已备，无真实训练提前入队，实际训练VRAM仍待测，不用工程预约上限冒充实测声明。

原协同向量SID两路s42仍claimed、无训练终态/verify验收，原自动观察驱动2931675仍alive、实际PPID1、900秒、截止1792334206.75411不变；本轮只读，无新提交，不改其plan。最新batch只读不做global扫描，不能称全服务器完整空闲/忙碌盘点。

| 来源stdout | SHA256 | 实际读取UTC |
|---|---|---|
| main05/r1-v2-deploy-apply-20261016T195357Z.stdout | f00cd75c8af4183a4465bd0317fd8d97b88801e9fb6f53918d60fc0ff5209fb7 | 2026-10-16T19:53:58.677678+00:00 |
| main05/r2-full-deploy-apply-20261016T202800Z.stdout | bdf4b6b6d061397610518663e8c95f961e34fc477454027e52c4f23743ee0d56 | 2026-10-16T20:15:08.544530+00:00 |
| main05/stack-deploy-apply-20261016T202330Z.stdout | c5c4af9bdbf34ab7db076744f159ece4fbabd1d4d8fdefbd46485964123ff97d | 2026-10-16T20:22:28.478520+00:00 |
| main05/r1-pf-confirm-20261016T202550Z.stdout | b1eac045fecae1e1d4411f1df654298123d52988c6fc51f376bbe9f4731cdb0d | 2026-10-16T20:14:18.895370+00:00 |
| main05/r2-full-pf-confirm-20261016T201810Z.stdout | 89d3d017e01133eb3d7d4e2cc3f5349bbe7001f695a952ed6ec60e295ea0a88c | 2026-10-16T20:17:34.959679+00:00 |
| main05/stack-preflight-confirm-20261016T202530Z.stdout | 114c3b32815fef2f7e2b3e99054a9dc4b2bfcbcb7df755e2b1fdc2075322f448 | 2026-10-16T20:23:27.559205+00:00 |
| main05/parallel-batch-read-20261016T202350Z.stdout | 2e696da829c6054fc38a13b7544970b5f3ba8f6e4d4ef1bf8fce08070dd9e42a | 2026-10-16T20:22:36.629933+00:00 |
| main05/sidcoll-readonly-20261016T201800Z.stdout | 32978a390b52a66b8fe764d776c187d3b3402d6d7db7985c7a31560fbc6dc632 | 2026-10-16T20:17:31.851935+00:00 |

双向确认首次driver时点更新：mean三个训练终态succeeded（对应314159/271828/161803墙钟751.2119781970978 / 580.7483870983124 / 575.6650147438049秒）；bi三个均已claimed，六训练无pending/rejected。尚未读取mean verify原始verified，故科学验收保持未验收，不报告效果。来源main05/first-driver-tick-read-20261016T202720Z.stdout，SHA256 3d58e41df8a16008315fcdf9331871b7528897d4de6b36ae9f13ae8fe6ff6565，实际UTC 2026-10-16T20:27:18.432371+00:00；同次R1/R2/叠加真实PF仍pending且无rejected。

补充排队解释（仅已存run快照）：bi s314159的run.json记录原admission settle_seconds=600、headroom=.2，CPUquota10.24；这只是该次放行时快照，不能把pending归因于修改门禁，也不能外推此刻global空闲。原run资源和source/env绑定均与新计划相同；本轮未修改准入。来源仍为上述first-driver-tick-read收据内run对象（原run.json SHA 07db370de8085a8eee7f077c1b631c3a17df7d19146f3813e4d5207129a8811c）。

### 温度交叉熵与DPO完整runner三配置真实preflight已验收

两批原始stdout均preflight_ok，三配置均固定86785训练样本/2520validation/120415物品，depth3/vocab771，runtime real/cuda:0、单线程、deterministic algorithms、TF32关闭；原始source/config/input/SID绑定一致，test未读。R2 old_checkpoint_read=false；只读旧A1 control manifest指标不读旧CP。温度source7a06178/DPO277a1f1，两批已接续各自唯一CUDA工程job；当前CUDA均pending、无rejected，实测训练显存尚UNKNOWN。mean自动verify三个已由原driver发布，其中s314159已claimed、其余pending；尚未读取raw verified。叠加真实PF已claimed，未验收。

| 来源stdout | SHA256 | 实际读取UTC |
|---|---|---|
| main05/r1-pf-event-result-20261016T203400Z.stdout | 9151b0d4cccfb01b326c8d02b7feac6d1c5a03f6405b2d409164d4647bb8a07d | 2026-10-16T20:33:03.025820+00:00 |
| main05/r2-full-pf-event-result-20261016T203410Z.stdout | bbc2b584ccc867db8c3bfc678164281d997c7ae5c18b6f5ad8b594997a21f94e | 2026-10-16T20:33:19.035643+00:00 |
| main05/cuda-admission-read-20261016T203550Z.stdout | e32b7b5116f986e37142c15f92f0511b857f08beefc4b01fb7dfd1ddb24ca2fc | 2026-10-16T20:34:05.263900+00:00 |

### 序列用户塔+组合叠加自身真实路径检查通过

p-seq-combo-stack-real-preflight-v1 succeeded，原始preflight_ok，三seed全部engineering=False；各自L1/L2/L3/L4/L5/L8/L9及历史反馈反向对照pass，L6/L7/L10为STATIC_REVIEWED。每条原始source/config/binding/data_hashes、512train+2520validation、120415候选、非零recency/beta/open gate已机械核对，未读checkpoint/test，未训练。该证据独立于旧组合84b补证。唯一叠加CUDA已接续提交；独立回读：R1 CUDA claimed，DPO/叠加CUDA pending，均无rejected；未伪称实测显存或科学训练已验收。

| 来源stdout | SHA256 | 实际读取UTC |
|---|---|---|
| main05/stack-preflight-event-result-20261016T203830Z.stdout | 6d0b83c298b3b1210efd3c4437bf9bd01e0c30c42f9a03c5870f34b5ce9995d6 | 2026-10-16T20:38:34.883408+00:00 |
| main05/all-cuda-admission-read-20261016T204015Z.stdout | d9e7984ee6718ab9f166ea2587b44575ad62e51af95c105c82bb2d827d1e92a4 | 2026-10-16T20:39:27.525978+00:00 |

## 2026-10-17 后继三批：CUDA实测、正式dry及唯一提交

| 批次 | 科学source | 本PID保守实测MiB | 乘1.15后向上取整256的声明MiB | CUDA核对 |
|---|---|---|---|---|
| 温度交叉熵（R1） | 7a061788780333970748238cff792eca383052d5 | 5692 | 6656 | 两synthetic effective step、自身state重载beam100/b1相等 |
| 自生成难负样本DPO（R2） | 277a1f14f33a82c0dd1eecf5330b4bb6ebaa0715 | 13332 | 15360 | 两SFT+两DPO step，actor/ref各4710403参数，ref冻结/optimizer仅actor、自身重载解码相等 |
| 序列用户塔+组合叠加 | a7536e55e99bf7d126ae983176f205b19ef30be1 | 888 | 1024 | 三真实shape step、catalog120415/users2520/H50，自身synthetic verify直接verified |

均包含本PID CUDA context与allocator peak，非global free差值；PF/CUDA source/data/runtime/model绑定逐字段核对，未读旧CP/test。R2工程预约12288MiB不是训练声明，实测较大，因此科学训练按15360申报，不沿用A1。

R1首个正式dry被scheduler pin schema拦截，无科学训练发布；仅新增标准scheduler-pinned-v1.json与请求v2、两批补真实只读PF-v2保持entry.job严格等值，未修改/伪造旧PF/CUDA，代码/config/data/seed/检查项/原deadline均不变。原launcher对新pin与metadata大小实际通过（26460/27362 bytes）；R1唯一正式dry重验dry_run_ok、DPO首次正式dry_run_ok、叠加dry_ok。均已事前告知Eddy后各发布三seed，原始submit只认driver_released_not_accepted，随后独立只读确认：叠加三claimed；R1 s42 claimed、s2024/s2025 pending；DPO三pending；所有新train/verify均无rejected。不能把pending称claimed。verify各由唯一版本化owner自动补，不依赖模型醒来。

四owner最新实际PPID=1/alive：双向确认2934201、叠加2939421、温度交叉熵2942456、DPO2947540；均poll900秒、不调用模型、原48h截止不重置（精确epoch见各原driver记录及下表raw）。有限队列，不承诺无限训练。

| 来源stdout | SHA256 | 实际读取UTC |
|---|---|---|
| main05/r1-cuda-event-result-20261016T204200Z.stdout | 3d0867d14ff5d79eb659d62105a59851a4386e6a838067cb99ea0a0ccf61f73c | 2026-10-16T20:41:15.406534+00:00 |
| main05/r2-full-cuda-event-result-20261016T204720Z.stdout | 5d1c77ad2125f754ac10f9c1d0d9520a7493b678a3f58024e4591fa417bb9626 | 2026-10-16T20:46:54.844656+00:00 |
| main05/stack-cuda-event-result-20261016T204730Z.stdout | 34171678413591eb2aaa9b11f4c69021c9ef90460dffd9e6c7ed4f25f21cb101 | 2026-10-16T20:46:58.322319+00:00 |
| main05/r1-formal-dry-20261016T204250Z.stdout | c0affab21a34f9ed915d4aea0baeff378b37b5e2b0b4daedcee9bbdf79af605d | 2026-10-16T20:41:32.450452+00:00 |
| main05/r1-formal-dry-v2-20261016T205110Z.stdout | 89a41285581435823ebc194f3f8e2482537dd54313cc57e97fd991427b6de61f | 2026-10-16T20:49:37.039087+00:00 |
| main05/r2-full-formal-dry-20261016T205430Z.stdout | c08ac4e1cd9162ee6f532b643db6c4a793e7889809c366634b2949b34d1f4d7e | 2026-10-16T20:52:51.220418+00:00 |
| main05/stack-formal-dry-20261016T204900Z.stdout | 8e46cf08f635bb56f3b60bbbca12c289614c4fa10d193e9a2af0f7dbb1b07aa5 | 2026-10-16T20:48:15.591716+00:00 |
| main05/formal-all-admission-20261016T205730Z.stdout | dcdd7de9efe84fc4cc2b2732c9ec6adfa8295f0aaff69435268f525ef8f1ccce | 2026-10-16T20:55:32.946316+00:00 |

## 2026-10-17 双向序列模型确认：六原始verify及best数值验收

每个训练succeeded、每个verify succeeded且原stdout status=verified；verify run.argv的manifest路径/SHA与launch input SHA及本次实际训练manifest SHA逐一相等。训练manifest.binding canonical SHA与真实PF expected_binding_sha256六项全部一致，涵盖源码/config/data/seed/初始化。非仅exit0/driver摘要。selection.best_metric顺序R@50→N@50；hits按固定2520分母恢复，尚未另做逐query配对。

| 新seed | 臂 | hits@50 | R@50 | N@50精确值 | best / stop epoch | 原始verify |
|---|---|---|---|---|---|---|
| 314159 | mean | 482 | 0.19126984126984126 | 0.05548624053716333 | 9 / 12 | verified |
| 314159 | bi | 479 | 0.19007936507936507 | 0.05591994429082736 | 6 / 9 | verified |
| 271828 | mean | 467 | 0.18531746031746033 | 0.05391783441617653 | 6 / 9 | verified |
| 271828 | bi | 489 | 0.19404761904761905 | 0.056776717149511956 | 9 / 12 | verified |
| 161803 | mean | 462 | 0.18333333333333332 | 0.054241793040909585 | 6 / 9 | verified |
| 161803 | bi | 478 | 0.18968253968253967 | 0.055794849909078056 | 10 / 13 | verified |

同新seed bi−mean Δhits为−3 / +22 / +16，均值+11.666666666666666，至少+6的seed为2。逐seed exact McNemar、p_pool、bootstrap、延迟未补，不能完整套§8.1行2或判转正；只报告已核事实，anchor仍M1-mean，交Opus解读。未读test/旧checkpoint，无重跑。

### 原协同向量SID：训练成功，验收计划config SHA空值阻塞

两路train42原始terminal均succeeded/exit0，stderr空，stdout complete；wall C1′7461.91189289093s、C2 8183.0320699214935s。驱动2931675已ALERT退出，不能写成训练失败或仍自动补seed。原capacity plan b14630f9b2be83b12e4d21503b7a5d9ce04f44c1b5b277ce50ff20d7a3d5753d的两train42 proof.expected.binding.config_sha256均null，而raw receipt为各自冻结config42 SHA（C1′d6e4ebba1971788b593e17433d06c20c85eea570cd0ec0a7146d5da53d45818a；C2 788248e0d71dec364b7218b3237d3e8f1702824a26d7e92598b794e55a217277），实际值亦各自等binding.data.config42。summary exception为raw proof/frozen identity mismatch。不得改科学门禁去认null，后续仅有版本化新验收计划/新owner恢复提案，不改/清旧plan/STOP，不重跑两训练；当前verify与条件补seed仍未推进，召回指标未验收、不摘取raw_proof指标。

| 来源stdout | SHA256 | 实际读取UTC |
|---|---|---|
| main05/bi-six-manifest-read-20261016T205850Z.stdout | a69d9a18e9bfa6b796bf40ced8ee929d680084100f765d7963f9a1fd434c2871 | 2026-10-16T20:56:49.542648+00:00 |
| main05/formal-all-admission-20261016T205730Z.stdout | dcdd7de9efe84fc4cc2b2732c9ec6adfa8295f0aaff69435268f525ef8f1ccce | 2026-10-16T20:55:32.946316+00:00 |
| main05/s1-alert-r1-admission-20261016T205330Z.stdout | 4930cc5d8c75dbb1c19788ea07234f534b5a4dcd6b18d926c2325d0c7b174401 | 2026-10-16T20:51:23.938500+00:00 |
| main05/s1-receipts-readonly-20261016T205650Z.stdout | 8692041fcbbfae57dd7907f29245f8990cb448db49443a3380df32a008c4b3ea | 2026-10-16T20:55:04.940818+00:00 |

交付节点更新（后续协同向量SID恢复另行进行）：叠加三训练均succeeded，原driver已自动补其中两verify pending，seed42终态晚于当次tick、待原900秒检查，不重复补；温度交叉熵三训练均claimed；DPO s42 claimed、2024/2025 pending；所有新train/verify无rejected。三个后继driver实际PPID1/alive；双向确认driver已退出，六raw verified和manifest绑定验收已独立取得，不因PID退出推断失败。来源main05/delivery-queue-snapshot-20261016T210430Z.stdout，SHA256 ae5cb5ea0689d779421dfb302b989d447afe6a7f482433514086c6364843f2b8，实际UTC 2026-10-16T21:04:29.833865+00:00。


## 2026-10-17 05:50+08：协同向量 SID（S1）获批恢复并接续唯一 verify

Eddy明确批准“按提案恢复”。恢复工具84ab6d9a7c21c010e96c93d51fec656224bc0008已push并新部署，脚本SHA92f89b0c9180facbb115e01c2a70edf4c0a27e5be16023f0489e7834447af6a4；一次独立code review对象023d0b1为APPROVED，后续普通uv.lock白名单及历史容量修复见证兼容按fresh RED/GREEN与76项回归收口，未重复设计review。真实Linux ready/release与独立ACK均已通过，不再保留“仅本机模拟”的验收缺口。

首次metadata inventory因根uv.lock白名单失败，未写S1计划/控制目录、未发布作业；保留旧失败收据。一次修复后的真实inventory通过：28个既有阶段中18 accepted、6 absent、4 absent_deferred，原8个intent字节保留。两条历史CUDA只按已批准的39312→39168资源修复见证核验：原backup/claimed原字节SHA、唯一差异vram、原job和成功receipt全部核对；不泛放开intent不等，也不改原guard。

新计划pipeline-plan-configsha-recovery-v1.json字节SHA **e35c3543ce63bf572a666a5e2132221c8261029409cd694075b159847af850cd**。与原capacity计划仅十二处null expected config SHA改为既有冻结配置SHA、新control_dir；jobs/data/source/resources/created/deadline不变。离线批准draft与新计划逻辑对象逐字段相等，字节SHA不同仅序列化方式。新控制目录controller-configsha-recovery-v1，旧计划、STOP、summary、intent与原两训练均保留。

新唯一owner **2953005**，实际PPID=1、alive，900秒fallback，保留原绝对deadline **1792334206.75411**。独立ACK实际核/proc/locks三把FLOCK均属同PID；ready/request/新plan/原deadline绑定通过后才release。原driver2931675未重启。两原train42仍succeeded，未重跑/重投；截至 **2026-10-16T21:49:48.528464+00:00**，行为向量配置sid-coll-c1-verify42-v1唯一claimed，物品塔向量配置sid-coll-c2-verify42-v1唯一pending，两者rejected均为0（无.reason）；后者尚不能称claimed，两个verify均未科学验收。原2024/2025训练均尚未发布；由同owner在本配置verify42原始verified且整数hits≥260后条件补齐，各配唯一verify，不加名额、不改门槛、不重置截止。

同次只读快照额外读到序列用户塔+组合叠加三原始verify stdout的status=verified，terminal均succeeded/exit0；s42 best_epoch10/steps2210，s2024 best_epoch6/steps1530，s2025 best_epoch5/steps1360。完整manifest/config/data绑定和配对统计仍需补，不据此改anchor或转正。

原始收据（路径以main05为本机gitignored收据目录；SHA为完整stdout字节）：

| 收据 | SHA256 | 实际读取UTC |
|---|---|---|
| main05/s1-handoff-v2-inventory-20261016T214140Z.stdout | 34aff5dd567dc93b0789d1134a10101939f7f10a7b78a1cbcf31d44f617a4efb | embedded per-file timestamps；本机核读2026-10-16T21:50:50.171052+00:00 |
| main05/s1-handoff-v2-prepare-20261016T214450Z.stdout | 34aff5dd567dc93b0789d1134a10101939f7f10a7b78a1cbcf31d44f617a4efb | embedded per-file timestamps；本机核读2026-10-16T21:50:50.171052+00:00 |
| main05/s1-handoff-v2-start-20261016T214918Z.stdout | 8640ce7c1e3361ee53bc167f3b8db1965005c9eb24e45d0bc933634fad0462db | 2026-10-16T21:49:24.604593+00:00；本机核读2026-10-16T21:50:50.171052+00:00 |
| main05/s1-handoff-v2-ack-20261016T214931Z.stdout | 333c41a1dec84569082b94d988e16d3352cc70ad1db756ec7b1c9d43d7415e6b | 2026-10-16T21:49:32.695111+00:00；本机核读2026-10-16T21:50:50.171052+00:00 |
| main05/s1-restored-queue-20261016T214947Z.stdout | f676817da99d5a293db4702a8b2e65607b1f4ebbbb6daaa2675a00921fcb0238 | 2026-10-16T21:49:48.528464+00:00；本机核读2026-10-16T21:50:50.171052+00:00 |

未读test、未读任何checkpoint/向量/SID map；未重启其它batch driver，不将queue pending称运行。P/G anchor不变，科学解读仍交Kiro/Opus。


## 2026-10-17 12:49+08：本轮训练与verify收口；协同向量SID自动补seed完成

范围：只读队列/terminal/stdout、JSON receipt/manifest/verification与全局vkgr-status；无test/任何checkpoint/向量/逐query预测读取，无提交/重启/重投。context UNKNOWN。原始stdout均直接读到verified，不以exit0或driver摘要替代。

### 协同向量SID（S1）

两s42原训练成功后，新owner2953005按原门槛自动接续。行为向量配置（C1′）s42为328 hits、物品塔向量配置（C2）s42为261 hits，均原始verified且≥260，因此两个配置均自动补2024/2025。六train与六verify全部唯一claimed，无pending/rejected，terminal均succeeded/exit0、stderr空。独立核12个receipt与新plan冻结expected的九身份字段（生产者动态manifest/receipt引用解析后逐项比对），六manifest原字节SHA/selection与verify.proofs、verification文件SHA全部相等；六raw_proof.status=verified。scientific source11d67681e0cbde09983f6bf482f55ab8a7191bdb。

新summary-final SHA769798b3d013024d71b6d629c408ca494911cd3234a6f283fc8f5c2117b5b230，于UTC2026-10-17T01:55:55.942268（北京时间09:55:55）收口：status=complete、两支complete/screening=SUPPLEMENT、unaccepted=[]、plan_digest=e35c3543ce63bf572a666a5e2132221c8261029409cd694075b159847af850cd、原deadline不变。driver现已退出；与完整终态一致，无需重启。旧STOP与失败工程attempt均保留。

| 配置 | seed | hits | R@50 | N@50 | best/stop epoch | 对T3同seedΔhits | verify receipt SHA256 |
|---|---|---|---|---|---|---|---|
| 行为向量（C1′） | 42 | 328 | 0.13015873015873017 | 0.03939072828605707 | 16/19 | +74 | 1f26279876004ad492ffb287425658c31afaac4e130d0a535b59301a7996300f |
| 行为向量（C1′） | 2024 | 342 | 0.1357142857142857 | 0.04116689053773432 | 17/20 | +68 | e0d49691485e9b4c519cd7ac12e0d511d1b8582b8178d83247c10c494d93a07d |
| 行为向量（C1′） | 2025 | 333 | 0.13214285714285715 | 0.03863006653243303 | 14/17 | +58 | 0dcdc1929e0f485559455697c808028e9b2caabc24bfe7c51aeec94ae44e962d |
| 物品塔向量（C2） | 42 | 261 | 0.10357142857142858 | 0.03462631988097493 | 18/21 | +7 | 7f0f365919be69218358ea06d0f4a2a02f172961f70524669b43cd6c9c47b71b |
| 物品塔向量（C2） | 2024 | 268 | 0.10634920634920635 | 0.03284575371730338 | 15/18 | -6 | f2598088f1172985480b0173634a97584832c3af4286fe1e123f6928de9ea874 |
| 物品塔向量（C2） | 2025 | 261 | 0.10357142857142858 | 0.03230282255986304 | 17/20 | -14 | bfaa2478ebeb3d04469448076fcefc5cb7c93dad3d75046c0800541ce158fb9f |

严格按FINAL卡§6：只用2024/2025主判，行为向量配置Δ=+68/+58，机械状态UP；物品塔向量配置Δ=−6/−14，机械状态DOWN（−6含边界）。seed42仅筛选；三seed均值不作主判据。该结果不等于授权启动可学习SID或新SID A1确认，不自动更改G anchor，科学解释与后继冻结交Kiro/Opus。S0 A–D新SID诊断的报告收口仍需核完整证据，不把训练verify齐了当所有诊断报告完成。

### 其余三批本次新验收

九train+九原始verify均succeeded，逐一唯一claimed/no rejected，verify argv的manifest路径/--manifest-sha与本次读取原字节SHA相等，manifest完整且seed身份相符。双向序列模型确认此前六train+六verify已独立验收，本次未重复执行。

| 任务 | seed | hits | R@50 | N@50 | best/stop epoch | manifest SHA256 |
|---|---|---|---|---|---|---|
| 序列用户塔+组合叠加 | 42 | 498 | 0.1976190476190476 | 0.060354009246902075 | 10/13 | 6ac360e42f0919f4fb0abdd7c3772a19be750ea242c77b8a8c84a4d4346e7f03 |
| 序列用户塔+组合叠加 | 2024 | 504 | 0.2 | 0.057730775965677945 | 6/9 | 6ac48de98db9ca88f782c39ec92228c814942adace26af73553b26795aa87c30 |
| 序列用户塔+组合叠加 | 2025 | 496 | 0.19682539682539682 | 0.05915274327788308 | 5/8 | b6bdfdf816d694249f36f752a0ea037aab63695b8b6ead405935bd29a055af2b |
| 温度交叉熵（R1） | 42 | 321 | 0.12738095238095237 | 0.03967299125512014 | 18/21 | c20afc74eae2e7d61f290df01e8cd04d517ffc990b673e37fa3adea381ae0639 |
| 温度交叉熵（R1） | 2024 | 318 | 0.1261904761904762 | 0.03778555081737306 | 23/26 | d111e2e914fe7a98fee744b18c76260252006a1d2e904ee0f2d9d3f1718ed342 |
| 温度交叉熵（R1） | 2025 | 283 | 0.1123015873015873 | 0.03461196187433329 | 8/11 | dd21553c2fe155536a2577cca456d501d7d5b90e1571266d711ca5c6d423fb6c |
| 自生成难负样本DPO（R2） | 42 | 256 | 0.10158730158730159 | 0.030855299455734814 | 1/3 | be7f0a4e67c388120abffedb88155becb1000b05818ea5e58890a45d7a4a2148 |
| 自生成难负样本DPO（R2） | 2024 | 316 | 0.1253968253968254 | 0.038787858342256065 | 1/3 | 56f5ff53821c48f91b7d023b21a7e2678471ed3f0938b55c8211af04465465f2 |
| 自生成难负样本DPO（R2） | 2025 | 305 | 0.12103174603174603 | 0.03723540091193772 | 1/3 | 021c465697e0ee0b9b6b0fc4a11f13ed57a9b6bfe222ae83f225dc189f4d7ba5 |

叠加对M1-mean（466/481/471）逐seedΔhits=+32/+23/+25，均值+26.6667；p_pool、逐seed exact McNemar/bootstrap与延迟未补，不写有效/转正结论。温度交叉熵对A1+T3（289/328/310）Δ=+32/−10/−27、均值−2.3333，≥+6 seed数1，卡§3有效条件不满足。DPO Δ=−33/−12/−5、均值−16.6667，≥+6 seed数0，同样不满足；科学解释不自裁。DPO三seed best均为DPO epoch1、stop3；epoch0 R/N对已登记A1+T3差均为0，跳过0/20000（均0%）；verify只验已归档beam/sample/highest-eligible合法性，未第二次完整重做挖掘，不隐去限制。

### 当前全局与失败分类

北京时间12:47全局vkgr-status仅sched-serve-02 running，pending(0)，GPU49139/49140MiB free、CPU idle100%；本轮最新terminal为DPO s2025 verify，北京时间12:38:24.044931。可确认目前无训练/verify运行或排队；以最后已知终态估算约9分钟空档，尚不宣称已连续空闲超过1小时。没有已冻结且工程就绪的下一批自动队列，不为填空新增实验。

本轮新科学训练/verify失败0；旧T1/T2拟合、组合泄漏IO v1、两SID preflight v1及辅助诊断等failed为历史保留，不能因全局failed列表误报本轮训练失败。物品塔向量SID的DOWN是科学结果，不是程序失败。原config SHA验收ALERT已按批准恢复并最终收口。

### 来源收据

| main05 stdout | 完整SHA256 | 实际读取UTC |
|---|---|---|
| main05/progress-check-20261017T044512Z.stdout | 2efa7a6f83b49e7bd0e91867051fc28f2637a2f5e41be2ea9985cdb66d4a0cbd | 2026-10-17T04:45:28.256755+00:00（本机复核2026-10-17T04:48:57.149780+00:00） |
| main05/s1-final-metadata-20261017T044559Z.stdout | 9d3533e66b28089d2b0f546b6071d4e76a7b28a6daa79a6e479a486fe78bbd18 | 2026-10-17T04:46:00.282436+00:00（本机复核2026-10-17T04:48:57.149780+00:00） |
| main05/progress-manifest-bindings-20261017T044650Z.stdout | cb1980e3a70ef21eec06e5e3792c76c10eb025418a8fb2655d1038af1a7daa40 | 2026-10-17T04:46:51.557251+00:00（本机复核2026-10-17T04:48:57.149780+00:00） |
| main05/global-progress-20261017T044729Z.stdout | 532767fce72362e194c60537b48bfd7bbfaaac3a9103164c88220d9e71b79dbb | 2026-10-17T04:48:57.149780+00:00（本机复核2026-10-17T04:48:57.149780+00:00） |

本机逐字段验收清单cache/codex-morning-execution-20261017/progress-accepted-20261017.json，来源索引progress-receipts-20261017.json；原始JSON仍gitignored，不公开完整queue/raw数据。


## 2026-10-17 13:25+08：叠加与双向确认配对补证；用户塔延迟计时启动

依据decision_log午间/午后裁决。工具7308447d0f83a96a0cdd99ea9980d2a4e380a02a，独立review与两MEDIUM fresh RED/GREEN关闭见docs/superpowers/reviews/2026-10-17-postbatch-analysis-code-review.md；31项synthetic回归通过。全新Git部署109文件SHA核过，真实已验收validation归档的metadata preflight（分析入口不实例化训练loader，以synthetic=False等绑定排除工程夹具）先核12个原验收manifest、唯一claimed job/config/seed/deployment、原binding(source/config/data)完整digest与历史/query/catalog/support，raw stdout verified；preflight不读预测。正式1核配对作业才读取已存validation逐query预测，复用原冻结phase2 load_archive/paired逐字函数（source SHA548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8），没有读取checkpoint/test、没有重跑模型。

统计run_id p-postbatch-paired-v1，唯一claimed/no rejected，terminal succeeded/exit0、stderr空，wall_seconds2.645493268966675。原始stdout直接paired_complete，聚合report SHA **d11dd832753031b4eaa77f313e3c346497169522a2b6e24c05a4d5a40be4c224** 与服务器report原字节相等；统计结果在此记账，不由driver摘要推断。

### 逐query配对：exact双侧McNemar与bootstrap

所有query2520，OOV623结构性0、in-catalog1897；query顺序/history/catalog/support/hash与原same-seed对照一致。bootstrap沿原工具：query为采样单位、10000次、NumPy default_rng(0)、percentile95%区间；每seed独立计算。p_pool为三seed discordant b/c分别求和后exact双侧二项；同批query跨seed不独立，p_pool偏乐观，是必要门槛，不代表独立重复证据。

| 比较 | seed | 新/对照hits | Δhits | 仅新/仅对照 | exact p | ΔR@50 95%CI | ΔN@50 95%CI |
|---|---|---|---|---|---|---|---|
| 叠加 vs M1-mean | 42 | 498/466 | +32 | 94/62 | 0.012813143553546107 | [0.002777777777777778, 0.02261904761904762] | [0.0026866943913888366, 0.00868913160442641] |
| 叠加 vs M1-mean | 2024 | 504/481 | +23 | 93/70 | 0.08454305818434507 | [-0.0003968253968253968, 0.01904761904761905] | [-0.0012625562346606156, 0.0048426978267984404] |
| 叠加 vs M1-mean | 2025 | 496/471 | +25 | 90/65 | 0.05353757584043604 | [0.0003968253968253968, 0.019444444444444445] | [-0.0004502965834197796, 0.006023212417910332] |
| bi vs 同新seed mean | 314159 | 479/482 | -3 | 52/55 | 0.8468020751891949 | [-0.009126984126984128, 0.006746031746031746] | [-0.002144172714194776, 0.003065112890353657] |
| bi vs 同新seed mean | 271828 | 489/467 | +22 | 86/64 | 0.08607110318597075 | [-0.0007936507936507937, 0.018253968253968255] | [0.00027411412841798745, 0.005481608180853326] |
| bi vs 同新seed mean | 161803 | 478/462 | +16 | 86/70 | 0.22965784665432648 | [-0.0031746031746031746, 0.015873015873015872] | [-0.0012360553448230048, 0.004417586740168332] |

### 冻结规则机械核对（不替Kiro转正裁决）

| 项 | 序列用户塔+组合叠加 | 双向序列模型确认 |
|---|---|---|
| mean Δhits | 26.666666666666668 | 11.666666666666666 |
| ≥6 seed数 | 3/3 | 2/3 |
| 合并b/c | 277/197 | 224/189 |
| p_pool | 0.00027644228278844257 | 0.09420166081553626 |
| §8.1行2（mean≥6、k≥2、p_pool<.05） | 满足 | 不满足，行3（均值过线但合并检验不显著） |
| 至少2seed exact p<.05 | **不满足，仅1/3** | 不满足，0/3 |
| 独立泄漏review | 新E pooling接合点APPROVED | 既有实现review覆盖，见原报告 |
| 延迟 | 用户塔计时作业已提交，结果待验收 | 同次用户塔计时作业，结果待验收 |

叠加即使延迟补齐，也不能绕过“至少2seed p<.05”门槛；bootstrap区间与离散McNemar在临界值不完全一致不能用来替代已冻结检验（例如seed2025）。交Kiro/Opus解释，P anchor保持M1-mean，不自动改阈值/加seed/合并bi。叠加相对旧组合同seed描述差为+14/+2/+6，按午后要求保留“主要来自组合特征”的归因限制，不声称序列塔与组合特征可加；D-1与23:2x统计偏离照旧保留。

接合点review：docs/superpowers/reviews/2026-10-17-stack-e-pooling-leakage-review.md，SHA d38372295c99c7162219cd44b86a698869f585423ac87e26d60ac857ae06268d；独立reviewer核真实三seedPF/非零E/g/branch隔离、62源码pins与3配置、7项synthetic与直接断言，未读checkpoint/test。C/D/E组件不重复审，不把代码review当科学有效性。

### 用户塔计时（尚无最终数值）

Eddy直接澄清“沿原用户塔口径补报”，因此只计model.users，不称端到端；原10-16下午§1边界保留，不读任何checkpoint。新工具复用已通过的seqm1_latency_bench.measure/export_pass/length_schedule源码，bench b089b8f2da3b72d115f3df743f1e5461392270f6，SHA ed0daa90dc09d531238d2c4bbe2fa37a0f81f5653b4c9da07d53ad3a04c72c6f；mean/bi/stack用相同已部署stack a7536e5中相同M1模型与新stack.users源码。真实validation长度分布2520，侧值与权重初始化合成；CPU1/A6000、50warmup/1000timed、b1/b512、另2520导出pass、配对输入SHA，均值/p50/p95/p99和同次mean比值。不能外推真实权重/服务端到端成本。

run_id p-postbatch-users-latency-v1；dry_ok后唯一提交，早期pending不称claimed，后续全局running/queue唯一claimed/no rejected已核。1CPU/9216MiBRAM/1024MiBVRAM/2hcap，不调用模型。显存依据同源码三用户塔在相同N120415/L50/b512的已验收fullshape训练+导出probe，峰值max888MiB→ceil256(888×1.15)=1024；只是先前实测覆盖的更保守包络，非新计时的实测显存。两个原工程proof作为job inputs绑定；不伪造新实测。统计先结束再启动计时，避免配对计算并发；同次mean对照，保留调度/SSH控制与共享机器的微基准局限。

收据索引将在延迟终态后同节补齐；现有主来源：main05/postbatch-paired-preflight-20261017T052150Z.stdout、postbatch-users-preflight-20261017T052209Z.stdout、postbatch-paired-admission-20261017T052312Z.stdout、postbatch-paired-report-20261017T052413Z.stdout、postbatch-users-claimed-20261017T052531Z.stdout。曲线诊断结论只见issue I-2026-10-17-g-earlystop-censoring，本节不重复。

本批截至计时运行时的完整来源索引（后续终态另补，不以pending称claimed）：

| main05 stdout | SHA256 | 实际读取UTC |
|---|---|---|
| main05/postbatch-deploy-inventory-20261017T052048Z.stdout | 63395c4e56335a9e2b744b86fc74d2ccc84bf615e9654db0479ea9ebb184c583 | 2026-10-17T05:20:49.703251+00:00 |
| main05/postbatch-deploy-apply-20261017T052113Z.stdout | 1cc6becf69d169630b404dc9950677ca1b9ebc8ed2476f00f8f8ef42c617b612 | 2026-10-17T05:21:14.255830+00:00 |
| main05/postbatch-paired-preflight-20261017T052150Z.stdout | 5a2be955618847c4f3ecd1f5563594480a1df05e8054d5b4a49f7f4b596279a6 | 2026-10-17T05:21:51.085756+00:00 |
| main05/postbatch-users-preflight-20261017T052209Z.stdout | 18d9783a144551b9a49743b887f75523795f938979cb24e51456b40a7f4405d4 | 2026-10-17T05:22:12.282048+00:00 |
| main05/postbatch-paired-dry-20261017T052244Z.stdout | dd71ffe6383632e5cbd9e44d7f7d057760ea54010efd0f72ce35a7156fdb12e7 | 2026-10-17T05:22:44.829858+00:00 |
| main05/postbatch-paired-submit-20261017T052301Z.stdout | a9632842356a005df1742f051bc206f34da19a0e5ea293ae8bbcdc3928d112d5 | 2026-10-17T05:23:02.631755+00:00 |
| main05/postbatch-paired-admission-20261017T052312Z.stdout | 53baeed9266d6d25ca2a5e953c29c41bede57c5ba5e9b985be0a528472333498 | 2026-10-17T05:23:13.201270+00:00 |
| main05/postbatch-users-dry-20261017T052330Z.stdout | dfecfab5d50bf1a725cec83cd0acdb87f6769cb21f0c8fa6939c6159de43ba83 | 2026-10-17T05:23:30.684242+00:00 |
| main05/postbatch-users-submit-20261017T052344Z.stdout | 8386fb144684af44d46d38fe2d57040d597c54a0b2fa05cb2f05e0e2750cdd8e | 2026-10-17T05:23:45.787421+00:00 |
| main05/postbatch-users-admission-20261017T052402Z.stdout | 51f41f67d9dcc3d953cc23e4d94381ad764b6365da24284026a08834b2b5201a | 2026-10-17T05:24:03.111126+00:00 |
| main05/postbatch-paired-report-20261017T052413Z.stdout | 96cb6c16ad56b9997e4bfd3a366fd697bb638bbfce089e891f57f614aa5b6ec7 | 2026-10-17T05:24:13.909085+00:00 |
| main05/postbatch-global-admission-20261017T052450Z.stdout | 17aee251b2a8c42c499ef64cfb5abb7ff02b68a3e93177ef6afb40a978f5dd1f | 2026-10-17T05:28:55.313101+00:00（本机核读；producer无UTC字段） |
| main05/postbatch-users-claimed-20261017T052531Z.stdout | 544834a3a49a525b1da5d390d77535789884f3d69dad64e21595ce2014ee26d3 | 2026-10-17T05:25:31.793588+00:00 |

统计独立算术核查（validate-data标准范围）：本机SciPy未安装，未新增依赖；用标准库Fraction逐项递推二项PMF并按概率排序求双侧尾和，与原math.comb下尾倍增实现独立。六逐seed exact p及两个p_pool全部相符（abs tolerance1e-13），阈值条件计数不变。bootstrap沿原冻结实现，不重复换算法或用区间替代p门槛。审计JSON SHA fd57e6303a4e1571ef1f5374f2cb6369d88a45f3fc1db5caf1c7320a376dc9ba，实际本机核查UTC2026-10-17T05:37:00+00:00；cache/codex-morning-execution-20261017/postbatch-independent-p-check.json。


### 2026-10-17 14:01+08：用户塔延迟最终验收（原作业13:37完成，通知运输失效后补读）

p-postbatch-users-latency-v1 原始terminal succeeded/exit0、空stderr；北京时间13:24:11.960361启动，13:37:07.331448结束，wall_seconds775.3710868358612。原stdout latency_measured、report SHA **2abac68d7ffd5a1d587eaffa25c1b2da9ee9bffe2b4181eb38985355864726cf**，与补读report原字节一致。没有重跑/重投。独立逐项核CPU/GPU各3模型、各模式1000timed/50warmup/b1或b512、n1000、单位ms/query；所有raw times有限正数，均值与nearest-rank分位数重算相等；六组输入SHA按模式跨模型/设备完全相同；另每模型5batch合计2520query导出pass。源与真实长度SHA同原PF，三处M1模型源（原M1 benchmark、bi确认、stack）文件字节均相同，SHA fc2629da1825393818e1defbf3756cdf3fa4967e715decd96cc12c9fa02cef5a。

**范围仅model.users()，不是端到端**。实际validation长度分布、合成侧值/初始化权重；不读任何checkpoint。排除特征准备/输入传输/特征存储/物品塔/检索/网络，不将GPU batch1慢于CPU解读为服务整体表现。同次mean对照，不拿昨日不同负载原始延迟直接比。计时前配对统计已经结束，原训练均收口；共享机器/调度与控制访问的残余影响仍需披露。

| 设备 | 模型 | 模式 | mean ms/query | p50 | p95 | p99 | 对同次mean比值 | 2520 export pass ms/query |
|---|---|---|---|---|---|---|---|---|
| cpu | mean | offline（batch512） | 0.340543086 | 0.339246664 | 0.350273871 | 0.358471705 | 1.000000 | 0.335868582 |
| cpu | mean | online（batch1） | 1.728128799 | 1.754062017 | 1.786743058 | 1.809752081 | 1.000000 | 0.335868582 |
| cpu | bi | offline（batch512） | 0.261999676 | 0.261070224 | 0.269780090 | 0.275946225 | 0.769358 | 0.272892572 |
| cpu | bi | online（batch1） | 1.690146524 | 1.718560932 | 1.742131077 | 1.779092010 | 0.978021 | 0.272892572 |
| cpu | stack | offline（batch512） | 0.359871043 | 0.349069994 | 0.406028283 | 0.413169551 | 1.056756 | 0.344441487 |
| cpu | stack | online（batch1） | 1.966505770 | 1.997402869 | 2.017083112 | 2.044413937 | 1.137939 | 0.344441487 |
| cuda | mean | offline（batch512） | 0.012866127 | 0.012861998 | 0.013026920 | 0.013643701 | 1.000000 | 0.013214776 |
| cuda | mean | online（batch1） | 5.880027668 | 5.842637969 | 6.204881007 | 6.443410879 | 1.000000 | 0.013214776 |
| cuda | bi | offline（batch512） | 0.012629214 | 0.012608947 | 0.012896861 | 0.013416863 | 0.981586 | 0.012791277 |
| cuda | bi | online（batch1） | 5.718311546 | 5.725086899 | 5.758136977 | 5.796937970 | 0.972497 | 0.012791277 |
| cuda | stack | offline（batch512） | 0.014519290 | 0.014496012 | 0.014893457 | 0.015930338 | 1.128490 | 0.014314708 |
| cuda | stack | online（batch1） | 6.623047144 | 6.632863078 | 6.917774910 | 7.284866879 | 1.126363 | 0.014314708 |

机械转正表补充：延迟报告现已附齐。叠加行2/接合点review/延迟三项满足，但至少2seed exact p<.05仍不满足；bi行3与显著seed条件不满足不变。这里只报规则核对，anchor裁决仍交Kiro。

通知限制：单次inotify SSH观察在elapsed1754.3573215秒后exit255，stdout空；其后一次report读取也因旧master Broken pipe失败。原计时早于断线已成功完成，root曾将“无通知”说成“仍运行”，已按原终态向Eddy更正；见issue I-2026-10-17-latency-observer-transport。不能把连接/通知状态当作业状态，原结果不受影响。

| 新收据 | stdout SHA256 | 实际读取UTC |
|---|---|---|
| main05/postbatch-after-event-transport-20261017T055522Z.stdout | 4478ab3f3501a387c2d0c0b19dc55eaea949ca3834068c6e151359b7f0c65d84 | 2026-10-17T05:55:24.969171+00:00 |
| main05/postbatch-users-report-reconnect-20261017T060046Z.stdout | 0e373b5bfa7183465c57c6a6fa487bbebdb1ae025259c9ddeebfd2c4ed5e9147 | 2026-10-17T06:00:48.898888+00:00 |


## 2026-10-17 14:10+08：两协同SID的S0 A–D聚合报告收口

复用原两个sid_verify的mapping-diagnostics.json（A/D/D′）与六verify的prediction-diagnostics.json（B/C/C_delta），没有大文件重算、重拟合或重跑模型。八文件由原已验收receipt/summary绑定SHA；有限reader a6a3fd1890f9970b7f6a3e107702be541160094b经六synthetic tests及固定SHA新Git部署112文件核验，首次真实intake直接verified（只指聚合来源/schema，不替代训练verify）。未读预测明细/完整映射/向量/checkpoint/test。

本机再将六C_delta逐档对原result.C.T3同seed重算，并核总差值等于本配置item hits减T3；全部PASS。原S0 result SHA a2e93d69bfb79af2c94e344aa5368e7604825777acc3ed6f2f6cde37c5db1924；热度slice SHA6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26，median1/p9011。原五档分母high963/mid548/low155/unsupported231/OOV623保持。B分母in-catalog1897，OOV623独立排除；C合计2520且OOV命中0。B rate不能当全query R@50。

### A：结构

| 配置 | 层 | distinct digits /256 | utilization | normalized entropy | parents/nodes | avg children | avg items/node |
|---|---|---|---|---|---|---|---|
| C1 | 1 | 256/256 | 1.0 | 0.9999999049037053 | 1/256 | 256.0 | 470.37109375 |
| C1 | 2 | 256/256 | 1.0 | 0.9958010613171234 | 256/65536 | 256.0 | 1.8373870849609375 |
| C1 | 3 | 2/256 | 0.0078125 | 0.12429281723841393 | 65536/120415 | 1.8373870849609375 | 1.0 |
| C2 | 1 | 256/256 | 1.0 | 0.9999999049037042 | 1/256 | 256.0 | 470.37109375 |
| C2 | 2 | 256/256 | 1.0 | 0.9958010613171234 | 256/65536 | 256.0 | 1.8373870849609375 |
| C2 | 3 | 2/256 | 0.0078125 | 0.12429281723841393 | 65536/120415 | 1.8373870849609375 | 1.0 |

两个配置各n_items120415、semantic_depth3，按原定义effective_layers3；semantic/full_code leaves各120415、avg/max_items1、collision_items/pairs/collided_leaves均0。effective_layers是avg_children>1统计定义，不等于证明三个层次学到了语义。

### B：前缀与item命中（原始计数，分母1897）

| 配置 | seed | prefix1 top1/top50 | prefix2 top1/top50 | prefix3 top1/top50 | exact item top1/top50 | full SID top1/top50 |
|---|---|---|---|---|---|---|
| C1 | 42 | 299/1239 | 16/371 | 15/328 | 15/328 | 15/328 |
| C1 | 2024 | 315/1236 | 16/397 | 15/342 | 15/342 | 15/342 |
| C1 | 2025 | 293/1221 | 14/371 | 13/333 | 13/333 | 13/333 |
| C2 | 42 | 201/939 | 33/300 | 21/261 | 21/261 | 21/261 |
| C2 | 2024 | 219/947 | 28/310 | 15/268 | 15/268 | 15/268 |
| C2 | 2025 | 192/930 | 24/303 | 17/261 | 17/261 | 17/261 |

原聚合保留完整rate；表中计数可重算。full_sid_is_item_equivalent原True与无碰撞结构相符，不泛用于其它SID。

### C：原G4五档与对T3同seed差

| 配置 | seed | high hits (Δ) | mid hits (Δ) | low hits (Δ) | unsupported hits (Δ) | OOV hits (Δ) | 总Δhits |
|---|---|---|---|---|---|---|---|
| C1 | 42 | 298 (+57) | 28 (+16) | 1 (+0) | 1 (+1) | 0 (+0) | +74 |
| C1 | 2024 | 306 (+48) | 34 (+19) | 1 (+0) | 1 (+1) | 0 (+0) | +68 |
| C1 | 2025 | 301 (+48) | 29 (+7) | 2 (+2) | 1 (+1) | 0 (+0) | +58 |
| C2 | 42 | 235 (-6) | 24 (+12) | 2 (+1) | 0 (+0) | 0 (+0) | +7 |
| C2 | 2024 | 242 (-16) | 25 (+10) | 1 (+0) | 0 (+0) | 0 (+0) | -6 |
| C2 | 2025 | 231 (-22) | 29 (+7) | 1 (+1) | 0 (+0) | 0 (+0) | -14 |

算术描述：C1′三seed总增量200，其中high153（76.5%）、mid42、low2、unsupported3、OOV0；只提交Kiro解读，不自行写“学到协同结构”或跨架构超过A1 anchor。full-train D-1及既有fallback限制保留。

### D与D′：原定义并报

各配置cooccurrence n_pairs3630957，随机对同数量、seed0；原D随机有放回可含self-pair，D′对内两个不同视频、对间可重复，结构无定义记STRUCT_NA。原D不替换，不作新门槛。

| 配置 | 指标 | prefix | cooccurrence matches/n | random matches/n | lift | status |
|---|---|---|---|---|---|---|
| C1 | D | 1 | 434005/3630957 | 14083/3630957 | 30.817652488816304 | 原D无status字段 |
| C1 | D | 2 | 15230/3630957 | 61/3630957 | 249.672131147541 | 原D无status字段 |
| C1 | D | 3 | 0/3630957 | 26/3630957 | 0.0 | 原D无status字段 |
| C1 | Dprime | 1 | 434005/3630957 | 14043/3630957 | 30.905433311970377 | OK |
| C1 | Dprime | 2 | 15230/3630957 | 32/3630957 | 475.9375 | OK |
| C1 | Dprime | 3 | 0/3630957 | 0/3630957 | STRUCT_NA | STRUCT_NA |
| C2 | D | 1 | 126967/3630957 | 14045/3630957 | 9.04001423994304 | 原D无status字段 |
| C2 | D | 2 | 3245/3630957 | 58/3630957 | 55.94827586206897 | 原D无status字段 |
| C2 | D | 3 | 0/3630957 | 26/3630957 | 0.0 | 原D无status字段 |
| C2 | Dprime | 1 | 126967/3630957 | 13984/3630957 | 9.079447940503432 | OK |
| C2 | Dprime | 2 | 3245/3630957 | 37/3630957 | 87.70270270270271 | OK |
| C2 | Dprime | 3 | 0/3630957 | 0/3630957 | STRUCT_NA | STRUCT_NA |

D′第三前缀两配置均STRUCT_NA，不读成lift0；原D该层self-pair引入随机分母，照报但不作机理解释。零分母仍应UNKNOWN，本次未出现。事实齐证不代表科学解释自动通过。

| 来源 | SHA256 | 实际读取UTC |
|---|---|---|
| main05/coll-sid-eight-aggregate-20261017T060935Z.stdout | bd48af1c00b7a747c7cab84f52c79695d3c328e5a32573d8ef897583d7f5c083 | 2026-10-17T06:09:36.493814+00:00 |
| 私有本机交叉验收JSON | adb9f3138822610642cabad9be38e44aa198d8044e2d3c7d4b87c7fd34609dc2 | 2026-10-17T06:10:27.780671+00:00 |

八独立文件的path/SHA和原receipt/summary绑定详见cache/codex-morning-execution-20261017/coll-sid-diagnostics-accepted.json；方法/限制见docs/superpowers/reviews/2026-10-17-coll-sid-diagnostics-evidence.md。不据本节换anchor、启动新实验或改门槛。


## 2026-10-17 14:30+08：新SID A1确认冻结、review修复、固定SHA部署与真实PF接纳

新增无人值守授权§1预确认午间§5/午后§1内容，FINAL卡同名draft文件现已冻结，原字节SHA da30eebb8caba93f19bfb0b4fc017ffe8f88d634eb75c0ae3f611e7f1b29e2ff。对原已验收A1+T3三个config逐字段核，只有schema/task_id/sid不同，数值与类型全同；只复用SID c53、input725、vectorsbf840，不重新拟合。原C1 sid_verify原receipt与stdout里的raw verified+mapping_replay/原ref均直接读过，绑定一致。

实现源a5da4d6d0746ef41ca1ed93028971360dcb89547已push，12文件全新增不改旧入口。一次独立完整review对象34a2000只发现两工程MEDIUM（source identity与启动真实JSON尺寸），无科学设计偏离；五fresh RED后修复，98相关tests含4mutants/真实detached故障/47103允许47104拒绝边界通过。主owner核真实日志与修复diff，没有重复整套review。完整suite未完成、旧G6 fixture input hash mismatch按原限制报告，不能声称全库通过，真实共同数据在下一PF独立核。

新Git DEP/OPS-v1经只读inventory核8个新ID及目录都clear，fetch指定SHA/new worktree部署，313文件逐SHA、无data overlay；scheduler pin为39实际消费文件、base=target=a5da/overrides为空。私有三template JSON来自原同seedA1+T3 job，原资源逐一核1CPU/18432RAM/6656VRAM/24h/4GiB/no_admission=false，数据prepared/history/content的path和SHA三seed全同；新训练声明仍等本卡CUDA实测。

首个g-coll-a1-c1-preflight-v1：dry_ok，启动metadata7025bytes；STATE已写14:26工程入队前记录。只发布一次，独立readback唯一claimed、pending/rejected均0，尚未读到terminal，不以未收到通知推断仍running。工程包络1CPU/18432MiBRAM/6656MiBVRAM/2GiB/1h；GPU临时预留参照同A1/T3同vocab771/H50/micro256/effective512已实测fullshape的保守包络，并非本卡新正式训练实测。PF只读原输入、允许新OPS scratch；成功后才发布CUDA（新2步synthetic+own memory checkpoint replay、ownPID/context显存），所有科学train/verify尚未发布。新plan固定deadline_epoch1792390818.4130461，后续不得因部署/工程修复重置。

| main05来源 | 完整stdout SHA256 | 实际读取UTC |
|---|---|---|
| s5-coll-contract-20261017T060541Z.stdout | df17fad4c4098e4ecd0d735e0b9d02ef84f3887bb2dc6d7de193d425f385c92e | 2026-10-17T06:05:42.484425+00:00 |
| s5-fit-verify-raw-20261017T061016Z.stdout | 3ae3e100a554321a6dbfcb5bb61642d7378e62f54f2596369506c023f17794af | 2026-10-17T06:10:16.894165+00:00 |
| s5-fit-verify-receipt-20261017T061205Z.stdout | 1d3435a1d09d6ab85df92610bc4c2b007be192a191d85a81d1249607ddd7f2f6 | 2026-10-17T06:12:05.787506+00:00 |
| s5-deploy-inventory-20261017T062040Z.stdout | f3e8612488e3083429925582a05ba3ad41a9cb8f5f596f25662c61f306d75722 | 2026-10-17T06:20:41.199435+00:00 |
| s5-deploy-apply-20261017T062213Z.stdout | 2951cbca86f8cf47bb1a521304d05d846fbe48c5374d33ddf0c26555b799d834 | 2026-10-17T06:22:15.328767+00:00 |
| s5-pf-dry-20261017T062515Z.stdout | 7127e0e789854506373dfa5c63548e5a957cc7fb3365ee00ad5ab7acf7aaff95 | 2026-10-17T06:25:17.134195+00:00 |
| s5-pf-submit-20261017T062616Z.stdout | 8733707dbbcb14ad4aeb1c32986b12bb69455a7c76d78365ab47f1919f7753bf | 2026-10-17T06:26:18.153577+00:00 |
| s5-pf-admission-20261017T062643Z.stdout | 1c195c83cabbf5678e9c5acd6447c22a6b09047d1501ca725a5b9ac71f75e341 | 2026-10-17T06:26:43.971227+00:00 |


## 2026-10-17 15:03+08：新 SID 的 A1 确认（S5）三 seed 已接纳，有限 verify 驱动已独立核验

覆盖此前14:20–14:45工程过程状态。Kiro无人值守预确认已转FINAL，源a5da4d6d0746ef41ca1ed93028971360dcb89547已push并固定部署；原C1′ SID不重拟合、三个科学配置数值与旧同seed A1+T3一致，仅按卡替换SID及新入口。一次独立review的两个工程MEDIUM经fresh RED/GREEN关闭；没有第二份post-fix APPROVED，也不声称旧全suite通过。

三配置正式路径engineering=False真实preflight已通过（原输入86785训练/2520 validation/120415物品，源/数据/配置绑定逐项核）；工程wall143.649958秒。CUDA full A1 shape两更新、beam100/batch1、自己新建memory checkpoint replay通过，wall145.362801秒；含ownPID/context峰5692.0 MiB，ceil256(5692×1.15)=6656 MiB。CUDA为engineering_only，server_full_runner_train_verify=False，不代替科学verify；读取正式context用于shape，不宣称未读取validation，但没有产出科学validation指标。

正式dry三项通过，实际spec+prepared包络检查与canonical/raw config SHA分开核对。北京时间14:58:30.899261已在STATE写入队前告知；随后仅一次submit，初始driver_released_not_accepted不作接纳。独立读取UTC2026-10-17T07:00:11.396909Z：g-coll-a1-c1-s42、g-coll-a1-c1-s2024、g-coll-a1-c1-s2025各claimed=1、pending=0、rejected=0；尚无终态，不能称科学结果已通过。各声明CPU1/RAM18432MiB/VRAM6656MiB/24h/4GiB，三项合计CPU3/VRAM19968MiB；这是本批资源声明，不是全局实时占用预测。

唯一服务器driver PID2969937，actual PPID1/alive、SID2969937，poll900秒，不调用模型；原48h截止epoch1792390818.4130461保留。07:03:07.362954Z再次独立核验plan/request/ready/release：plan canonical SHA9240c43a73280c54de6008d3009e939e54c9c3d004d1064c07096a206d0b403c与ready/release相等，ready原字节SHA3070db121510d5244cfb96cbf81a1194d0a1104427442e18e68e12c1ba66eb1f与release一致，submitted-request等于原工程请求，三seed与源/截止均未变。plan文件raw SHA7ee0d47c160877f458049e7f40b539c994be43ce304eef2b19560ffe810b9378与canonical不同是序列化口径，不混用。

每个train成功后仅由现有driver补对应唯一-verify，失败/拒绝/不明状态停支ALERT、不重跑；当前三个verify尚未验收，科学hits/R/N/best/stop为UNKNOWN，G anchor仍A1+T3。不要重复submit/start/redeploy或重置截止。P统计/用户塔计时、接合点review、一次曲线、两协同SID A–D、指定两论文转述已完成；其余后继不提前跑。完整私有验收索引cache/codex-morning-execution-20261017/s5-final-admission-receipts.json。

| main05原始来源 | stdout SHA256 | 实际读取UTC（无producer时间时为本次本地重读） |
|---|---|---|
| s5-pf-final-20261017T064414Z.stdout | 435dbfd5f873d102aa9b618368c0b71961134f644afb43e90d86c77bb86110c3 | 2026-10-17T06:44:14.819515+00:00 (producer_read) |
| s5-cuda-stage-acceptance-20261017T065152Z.stdout | 26759b716273ebbfa73755f47244abbe16b6e04480e9144ac4f43f758bc0e386 | 2026-10-17T06:51:53.410483+00:00 (producer_read) |
| s5-formal-dry-20261017T065519Z.stdout | d0e5c89a0956d911137ed63fe4ebee0b0170b315e7e88e84e351a8cba68f8b70 | 2026-10-17T07:06:50.640941+00:00 (local_reread) |
| s5-formal-submit-20261017T065834Z.stdout | 55a70fd4d4724968cdd3fe617d4dbfa90982df140b9c1c7c2bc8f1340662fbc3 | 2026-10-17T07:06:50.640941+00:00 (local_reread) |
| s5-scientific-admission-20261017T070010Z.stdout | 8bb113dc5edda293e8c1e7d49f448595f6cf9311face705e4f0e7ff642780958 | 2026-10-17T07:00:11.396909+00:00 (producer_read) |
| s5-owner-seal-correct-path-20261017T070306Z.stdout | fd83fddd4fe97e4813a24e2cf28c200e50fd0a2ba5364aa3c886474d610184d5 | 2026-10-17T07:03:07.362954+00:00 (producer_read) |


### 无人值守五项交付复核（UTC 2026-10-17T07:08:33.756907+00:00；不新增测量）

完整重读decision_log最后三节后核对既有交付，没有新增训练、重复诊断或延迟重测。原实施/补证提交54fdbce、20be14e；科学接纳提交a239cfe。本次不声称补做了逐项顺序提交，原提交次序与打包保留。科学verify仍未验收，接纳状态以15:03节原UTC为准，不冒称重新采样服务器。

| 已有交付 | 本地复核文件 | SHA256 |
|---|---|---|
| 新SID A1确认 | cache/codex-morning-execution-20261017/s5-final-admission-receipts.json | 25a25f0e2d949009e93e063603edb1651a45e1650a5f4f8c9fcc76215b3a2747 |
| 叠加与双向配对 | cache/codex-morning-execution-20261017/postbatch-paired-accepted.json | cb3e65f5c5294d97a2e7a9e7e9f8cb75ca33c0e335eea39f18fc6694d225a1e6 |
| 用户塔延迟 | cache/codex-morning-execution-20261017/postbatch-users-accepted.json | 0c3eb0e1efa21eb6bd913f719fe185dcccf57c18369ed60b08b6a50e5c05e5d0 |
| 早停只读诊断 | cache/codex-morning-execution-20261017/epoch-curve-diagnosis-20261017.json | 6d9b233cbca375fa6ba8ce2dd06a0b79e9141ff1da8e83258e90609c1bc2a7a4 |
| 协同SID A–D | cache/codex-morning-execution-20261017/coll-sid-diagnostics-accepted.json | adb9f3138822610642cabad9be38e44aa198d8044e2d3c7d4b87c7fd34609dc2 |
| 指定论文摘要 | docs/superpowers/research/2026-10-17-sid-depth-accuracy-notes.md | fea86a42b573d2610e24138b811e7f0ebf887b61a1c25ce1ba9a1a9ac25ee52a |

表内文件均在上述UTC本地读取并重算SHA；完整stdout来源/原始实际读取时间见相应指标节与索引。独立接合点review原APPROVED保留；早停判读只在issue_log，转正/anchor/机制解释留Kiro。五项准备完成后按§3停止新增模型工作，既有服务器driver继续。


## 2026-10-17 傍晚裁决同步：叠加、双向确认均不转正

决策原文唯一来源：docs/decision_log.md「2026-10-17 傍晚」§1–3、§6–7。仅同步状态，不新增测量，不改原始数字/收据/门槛。

| 已验收批次 | 当前状态 | 原指标与收据入口 |
|---|---|---|
| 序列用户塔+组合叠加 | 不转正 | 本日志13:25配对补证、14:01用户塔延迟最终验收；接合点独立review APPROVED |
| 双向序列模型确认 | 不转正、收口 | 同上配对与用户塔计时，bi对同新seed mean |

叠加逐seed显著性条件不满足，不用p_pool/bootstrap替代，不追加seed。双向确认按裁决记“在新seed上bi与mean无可检出差异”，计时差仅描述。P anchor维持M1-mean，延迟限model.users()，不称端到端。G anchor仍A1+T3；新SID A1确认（S5）继续由原驱动接续，科学verify收口后交Kiro；本次没有重新采样或操作服务器，不启动用户画像特征（F7）/双塔与生成式融合（G5）。

共现SID机制措辞与早停解释更正以该裁决§4–5为准；不沿用“差值主要反映早停而非温度”的强解释，原曲线和指标收据不变，已知限制仍见issue_log。


## 2026-10-17 15:52+08：新 SID 的 A1 确认（S5）阶段进度，只读

三个训练均claimed且无pending/rejected、无terminal；progress中的实际训练PID均alive。三个verify在queue各目录/runs中尚无终态或有效产物，尚未发布，不能写“自动verify已通过”。当前训练未成功结束，原driver等待训练成功后再发布verify，未观察到应触发却未触发的证据。

| seed | train run_id | 当前epoch | 当前stage | verify |
|---|---|---|---|---|
| 42 | g-coll-a1-c1-s42 | 6 | evaluate | 未发布/未验收 |
| 2024 | g-coll-a1-c1-s2024 | 6 | evaluate | 未发布/未验收 |
| 2025 | g-coll-a1-c1-s2025 | 6 | train | 未发布/未验收 |

s42评估进度512/2520仅为读取时瞬时值；其余没有行级分子，不外推完成百分比。三个进程分别2971222/2971318/2971416。原driver2969937实际PPID1/alive、poll900秒，原截止1792390818.4130461未变。stdout/stderr在此次快照均空，不用空stdout推断无进展；progress明确显示formal进度。源码/配置的run.json仍绑定固定a5da4d6及各seed原SHA。

按FINAL卡§4机械登记：条件1（三train succeeded+三raw verified）尚未满足验收；条件2（三seed mean Δhits≥6）与条件3（至少两个seed Δhits≥6）均未验收，不能用中途epoch或进度补算。最终hits/R/N/best/stop继续UNKNOWN。待原驱动收口后核原始verified及完整绑定，再与A1+T3同seed289/328/310套规则、交Kiro/Opus，不改anchor，不追加作业。训练早停，剩余墙钟目前UNKNOWN。

| main05来源 | stdout SHA256 | 实际读取UTC |
|---|---|---|
| s5-progress-20261017T075146Z.stdout | f110498ee7172a5cd256d5222f709fa6373dadce02031eb8cd7b745da6831274 | 2026-10-17T07:51:47.398504+00:00 |
| s5-epoch-progress-20261017T075215Z.stdout | a26e5016977784cba2eda8b5c275b8e8f0dfcc0ae2bdf01e5cbc20d8a00c8363 | 2026-10-17T07:52:16.104813+00:00 |


## 2026-10-17 16:23+08：新 SID 的 A1 确认（S5）复查，尚未收口

| seed | epoch | stage | train终态 | 自动verify |
|---|---|---|---|---|
| 42 | 10 | train | 未结束 | 尚未触发/未验收 |
| 2024 | 9 | evaluate | 未结束 | 尚未触发/未验收 |
| 2025 | 9 | evaluate | 未结束 | 尚未触发/未验收 |

三个训练各唯一claimed、无pending/rejected，实际训练PID均alive。s2024与s2025的评估均为瞬时1920/2520，不能当全训练完成率。原driver2969937实际PPID1/alive、poll900、原截止未变；训练均尚未成功终态，尚未到自动verify触发点。没有重投/重启，不将stdout空视为停滞。

FINAL卡§4：三train succeeded+三raw verified尚未齐证；mean Δhits≥6、至少两个seed Δhits≥6均未验收。最终指标/best/stop仍UNKNOWN，当前不足以生成最终Kiro验收结论。待原驱动补verify并收口后，再核原始verified/source/config/data/SID绑定及热度/前缀报告，交Kiro/Opus；anchor不变，无新增实验。

| main05来源 | stdout SHA256 | 实际读取UTC |
|---|---|---|
| s5-progress-20261017T082243Z.stdout | 261d3484d0f9643d3efe5083fc85c3ed7f96964561672d49d83e29a78eb02157 | 2026-10-17T08:22:46.149729+00:00 |
| s5-epoch-progress-20261017T082304Z.stdout | 19f3fcf8de453452cf1873e2e259f5ca36e4270081188fd215057934398deca4 | 2026-10-17T08:23:04.798112+00:00 |


## 2026-10-17 17:49+08：新 SID 的 A1 确认（S5）两训练成功，自动verify已接上

| seed | train | 进度/选择信息（待科学verify） | 自动verify |
|---|---|---|---|
| 42 | 尚未结束、实际进程alive | epoch20 evaluate，瞬时960/2520 | 尚未触发 |
| 2024 | succeeded/exit0 | best14/stop17（未验收） | 已claimed，无rejected，尚无terminal/verified |
| 2025 | succeeded/exit0 | best14/stop17（未验收） | 已claimed，无rejected，尚无terminal/verified |

原driver2969937实际PPID1/alive，poll900、原截止不变；没有手动补verify、重投或重启。两个完成训练wall分别9367.942631959915/9333.023693799973秒。训练及已发布verify均无rejected；不因s42尚在推进判其异常，最终剩余时长UNKNOWN。

两个train stdout complete所指manifest SHA与本次读取相等；config与binding config一致；按既有canonical_bytes（含末尾换行）的binding摘要与sealed plan同seed expected相等。此为初核，不替代原始verified；尚不发布其selection为已验收效果。三条预注册条件：①三train succeeded+三raw verified尚未齐证；②mean Δhits≥6未验收；③至少两个seed Δhits≥6未验收。最终三seed指标/均值/有效性与Kiro最终复查提示词待齐证后整理；对照A1+T3 289/328/310不变，P/G anchor不变。

| main05来源 | stdout SHA256 | 实际读取UTC |
|---|---|---|
| s5-progress-20261017T094904Z.stdout | 88f612766a6edd77b9409f6444a9a01575a9f3be0964209da8f34493ce2cb0a1 | 2026-10-17T09:49:05.108708+00:00 |
| s5-partial-manifests-20261017T094927Z.stdout | b06894a8423ca4ab73ef819a85780afdce6115e96b9d06d981b9adb7d3b90aed | 2026-10-17T09:49:28.044587+00:00 |

训练manifest原SHA s2024：09a0b790c6b027613ef15bb6c9855d7b1adee79d9bf6ea093c16d09a1a5b8d36。

训练manifest原SHA s2025：86767e4d30ababc8087898f6f960197021ce54fd09cc6d775da4c373b242c712。


## 2026-10-17 19:00+08：新 SID 的 A1 确认（S5）三seed与自动verify最终验收

本节覆盖前述进度快照。只读独立验收：三训练succeeded、三自动verify均succeeded，原stdout直接读到status=verified并核best/replayed/epochs，不能只凭exit0。六个claimed唯一且无pending/rejected，train对sealed plan逐字段相等、verify对从原job派生的唯一manifestSHA/argv/inputs/resources逐字段相等；run源码均固定a5da4d6d0746ef41ca1ed93028971360dcb89547。三个manifest原SHA等train stdout所指，config/raw SHA/canonical SHA、binding摘要（末尾换行）、prepared/history/context、SID bundle与原plan一致。独立重算每epoch R→N→最早epoch选择，与selection.best_metric/best一致；driver summary数值和verify stdout SHA逐一相等。没有读checkpoint、test或逐query预测；服务器verify只验自己的新产物。

| seed | hits | A1+T3 | Δhits | R@50 | N@50 | best/stop | train/verify |
|---|---|---|---|---|---|---|---|
| 42 | 388 | 289 | +99 | 0.153968253968 | 0.046901287771 | 17/20 | succeeded/原始verified |
| 2024 | 376 | 328 | +48 | 0.149206349206 | 0.047984393382 | 14/17 | succeeded/原始verified |
| 2025 | 384 | 310 | +74 | 0.152380952381 | 0.049904570187 | 14/17 | succeeded/原始verified |

分母均2520，其中in_catalog1897、OOV623且OOV命中全0。均值hits382.666666666667，对照309，mean Δhits=73.666666666667；mean R@50=0.15185185185185188，mean N@50=0.04826341711322748，R绝对提升0.02923280423280424（约2.9233个百分点）。冻结卡§4机械三条件：①三训练succeeded+三原始verified并核绑定，满足；②mean Δ≥6，满足；③至少2seed Δ≥6，实际3/3，满足。因此只记“满足冻结有效条件”，不自行换G anchor。P= M1-mean、G=A1+T3不变，交Kiro/Opus裁决。

三个verify由原driver自动补，无手动抢投：s2024/2025已在17:49快照直接观察claimed，s42在本次核到唯一claimed与verified。最后verify于18:06:41.931590+08结束；driver于18:14:17.058786+08写summary-final status=verified/publication_alert=false，现实际PID2969937已退出，属于全部收口后的结束，不重启。原plan/deadline未变；只观察本批，不据此声称全服务器空闲。

### 热度分档：同已归档A1+T3 best相减（只读本地原manifest，不重跑诊断）

同seed两侧prepared/history SHA及slice_statistics_manifest_sha256=6e96e5bc3d21e22b71fcc64293cc2844be85ca3c100c7d28e910e2360470628d逐一相等。五档分母依次为头部963/中段548/尾部155/unsupported231/OOV623。

| seed | 头部 hits（Δ） | 中段 hits（Δ） | 尾部 hits（Δ） | unsupported | OOV |
|---|---|---|---|---|---|
| 42 | 344（+67） | 42（+30） | 2（+2） | 0（0） | 0（0） |
| 2024 | 338（+51） | 38（−3） | 0（0） | 0（0） | 0（0） |
| 2025 | 344（+63） | 39（+10） | 1（+1） | 0（0） | 0（0） |

头部合计Δ181/总Δ221=81.9004524887%，符合卡中“总增量>0且头部≥50%”的机械描述条件；机制解释仍留Kiro，不能写“学到了协同结构”。**报告未验收项：A1新SID的前1/2/3位前缀准确率及同对照变化尚未独立计算，记UNKNOWN**；不能用core03的B指标或全SID召回替代，不声称完整机制报告已齐。该项属于补充报告，未新增到冻结§4三个有效门槛。

### 原始身份与收据

| seed | config原字节SHA | manifest原字节SHA | verify stdout原SHA |
|---|---|---|---|
| 42 | e8f07f67b9ff1d5a650caaf47bce1aeb4a2db5b8e4804150f012ad584a74ef75 | a431a5b1734930671cf7ca092e469a6da16e3c130ffb30836b9aa57aaca8b36b | 9a04887ff44ee514fffe569a51360184e9c4dab314247493e49585634fb66aac |
| 2024 | 263f5ec1c883ee89a90627a1f58c48f7f0f2aca73f52252c688b8be1820e887a | 09a0b790c6b027613ef15bb6c9855d7b1adee79d9bf6ea093c16d09a1a5b8d36 | 48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67 |
| 2025 | 2e28801ce898dfdb79664d976771d44827d9be9111548a489afb5e6d583e2c45 | 86767e4d30ababc8087898f6f960197021ce54fd09cc6d775da4c373b242c712 | 48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67 |

共同SID manifest c53b81ddfcf8d45fd55903b30f8538372110c8e1df7376156719509994625187；prepared a55df0a3b285ce4cf623126fe53df53378bf5d7c0645c6484f0f405641bddd82；history 8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43；context d81c7012bf2adc2ecc7d71cb19ec4b26ab17c61607148e930d31b26eda752189。driver summary-final原SHA e6a6eb41cc5a2c85aa016bc5ddb42a137ef0446db9a9d4866a18f80da5e8ec52。

| main05来源 | stdout SHA256 | 实际读取UTC |
|---|---|---|
| s5-progress-20261017T105949Z.stdout | cbbad455852938ccc22302349753d9e02aab0f2e45d14cc86d63d49a4ecc6df0 | 2026-10-17T10:59:50.210674+00:00 |
| s5-final-manifests-20261017T110010Z.stdout | f93f1d01b26d40364b29e2889c20c880bcb598431b7f0bca1e426b2b00adb59c | 2026-10-17T11:00:11.014234+00:00 |
| epoch-curve-once-20261017T051548Z.stdout | bcc11573de62def002d08c0a99126ec293f8bdddc653d2f6e61587dde68126e5 | 2026-10-17T05:15:49.304036+00:00 |

完整独立验收缓存cache/codex-morning-execution-20261017/s5-final-scientific-acceptance-20261017.json（SHA f1dce775429bf335d7ecd5abf2dfdbe4889bf07b954118813e3f4d2dde5b2c87）；公开提示词docs/superpowers/reviews/2026-10-17-coll-sid-a1-result-review-prompt.md。全部失败旧记录保留，不新增实验、seed或自动后继。


## 2026-10-17 晚间裁决同步：G anchor更新

决策唯一来源decision_log“2026-10-17 晚间”§1–3。Kiro接受本日志19:00最终验收的三个有效条件，G anchor从A1+T3改为A1+共现向量SID（C1′，COLL_C1_B256D3_S42）；P anchor仍M1-mean。旧A1+T3仅转为历史对照，原指标/原产物不删改。没有新增测量，指标沿19:00表及其原SHA/UTC。限制随anchor继承：共现含训练自身target与之后train事件、validation第25周不进入共现输入；约0.2%视频内容回退、623个OOV结构性零命中、早停右删失。前缀报告补证不影响本次anchor裁决，仍待一次只读补齐。

融合设计交付登记：docs/superpowers/specs/2026-10-17-g5-anchor-fusion-draft.md已起草，仅设计、不新增测量。明确离线合并不需训练与原200候选协议所需P top100/旧checkpoint导出对拍授权缺口，不用旧oracle数值替代新anchor融合结果，不降低配额。交Kiro冻结，当前未入队。

可学习SID设计交付登记：docs/superpowers/specs/2026-10-17-learnable-coll-sid-draft.md已在G5之后起草。无新测量；LETTER/ETEGRec仅候选，不构成两臂或名额批准。作者repo根目录/README与试探LICENSE地址未确认兼容许可证，许可初审UNKNOWN，不能复制代码；论文来源及边界见卡。时间成本为引用既有A1实测/授权参考的推算，tokenizer拟合等UNKNOWN，不把GPU小时当并行墙钟。未冻结/实现/入队。


## 2026-10-17 19:24+08：新 SID 的 A1 确认（S5）一次只读前缀补证完成

授权decision_log“晚间”§4，非训练、不占训练名额。新适配入口d7c6551ac9f8626ebd86d9d2b99e0a95a47c24c9已扫描/push固定新部署116文件，原S0 B函数analysis/sid_diag_metrics.py SHA ad966940d682aa0f4e576eeacfece8cef92b2f9597fec4a41522b7e535e55dc4逐字节保持，配套原sid_diag/archive loader两文件也硬钉。仅三个新G anchor run自己的validation最佳epoch归档及本地已有A1+T3 S0 B聚合；没有重跑模型、读checkpoint/test、读取任何额外训练预测或导出向量。

新入口fresh RED→GREEN；一次独立review最初CHANGES_REQUESTED的两个工程MEDIUM（legacy直接日志路径绕过、辅助JSON代替原stdout verified）在新适配层修复，未改冻结函数；14相关测试与3变异全杀。首次symlink变异被另一门禁掩盖后已强化synthetic有效日志夹具再全杀；一次相关discover选错目录为0测试不计验证，已改正确模块运行14项。不称全suite通过，旧G6 fixture mismatch仍OPEN。

独立真实只读preflight job sid-a1-prefix-preflight-v1已claimed/no rejected并succeeded，原preflight_ok；只读小metadata/query JSON且未读预测分片或mapping数组。随后唯一正式诊断sid-a1-prefix-report-v1，经dry和queue，先pending（如实保留收据），之后事件终态与最终只读确认claimed唯一、无pending/rejected，succeeded/exit0/空stderr，实际wall2.714296579秒。两项各CPU1/RAM4096MiB/VRAM0/1GiB/1h cap；没有后台模型轮询或新driver。

正式原stdout status=prefix_aggregate_complete、原函数聚合计数，与三run原始verified和manifestSHA链/epoch分片computation校验一致；输入SHA143项记录在私有原收据，不把mapping/预测放Git。三seed完整命中与先前已验收388/376/384逐一相等，in_catalog1897，OOV623不进入前缀分母，主表全2520口径保持。OOV没有目标SID，前缀准确率不定义，不伪造为0；其item级命中仍结构性0。

### B：每格为命中计数/1897（top1与top50）

| 配置/seed | 前1 top1 | 前1 top50 | 前2 top1 | 前2 top50 | 前3 top1 | 前3 top50 | 完整SID top1 | 完整SID top50 |
|---|---|---|---|---|---|---|---|---|
| A1+共现向量SID/42 | 317/1897 | 1344/1897 | 25/1897 | 446/1897 | 19/1897 | 388/1897 | 19/1897 | 388/1897 |
| A1+共现向量SID/2024 | 295/1897 | 1326/1897 | 23/1897 | 425/1897 | 22/1897 | 376/1897 | 22/1897 | 376/1897 |
| A1+共现向量SID/2025 | 310/1897 | 1307/1897 | 29/1897 | 435/1897 | 28/1897 | 384/1897 | 28/1897 | 384/1897 |
| 历史A1+T3/42 | 89/1897 | 977/1897 | 13/1897 | 293/1897 | 13/1897 | 289/1897 | 13/1897 | 289/1897 |
| 历史A1+T3/2024 | 90/1897 | 1016/1897 | 19/1897 | 339/1897 | 19/1897 | 328/1897 | 19/1897 | 328/1897 |
| 历史A1+T3/2025 | 85/1897 | 1000/1897 | 19/1897 | 322/1897 | 19/1897 | 310/1897 | 19/1897 | 310/1897 |

三seed均值准确率（先逐seed按1897计算，再均值；不是把三seed当独立query）：

| 配置 | 前1 top1/top50 | 前2 top1/top50 | 前3 top1/top50 | 完整SID top1/top50 |
|---|---|---|---|---|
| A1+共现向量SID | 0.16201019/0.69882270 | 0.01353014/0.22948515 | 0.01212441/0.20172202 | 0.01212441/0.20172202 |
| 历史A1+T3 | 0.04638904/0.52591812 | 0.00896152/0.16763310 | 0.00896152/0.16288877 | 0.00896152/0.16288877 |

完整SID与exact_item在本bundle无碰撞时逐项相同；此补证仅机制描述，不改anchor/冻结有效条件、不把首位前缀变好自动解释成学到协同结构。原S0历史B沿用已存聚合，本轮不重新解码或重读旧逐query预测。

| main05原始来源 | stdout SHA256 | 实际读取UTC |
|---|---|---|
| s5-prefix-mapping-20261017T111129Z.stdout | a966f2c3d10154ce75af84ab477791a5865e2c43bafe8f49492577f2025fb3e1 | 2026-10-17T11:11:30.418679+00:00 |
| s5-prefix-history-20261017T111651Z.stdout | 506985e0d4135912a027ef6c4e480fff7270ed0473716621a4a12f39e61dc390 | 2026-10-17T11:16:52.245575+00:00 |
| s5-prefix-deploy-inventory-20261017T111744Z.stdout | 3a3f28abb13771126b20c2ba3ac6e5c5acd56f2d3715167277f2d3b847e23f6e | 2026-10-17T11:17:45.566056+00:00 |
| s5-prefix-deploy-apply-20261017T111758Z.stdout | 24abc6c507e8b832b656a000c39b0d11d83c2ac4fecf0c03b575509da884ea41 | 2026-10-17T11:17:59.996217+00:00 |
| s5-prefix-pf-dry-20261017T112111Z.stdout | b32fd6d5ac4c6976de941665b9e2a532c0241fceb64035f2aa7b8e454e335fa2 | 2026-10-17T11:21:14.043731+00:00 |
| s5-prefix-pf-submit-20261017T112230Z.stdout | 1e1256ca5543bdb1e1ede3d9be4e8a6efeeb20cbb85929189d0e321bba24b8a9 | 2026-10-17T11:22:31.542202+00:00 |
| s5-prefix-pf-accept-20261017T112243Z.stdout | a4b1c6976426e30bbf6b40762c1a9703cb74085bb180cb1acf63a710f43b1347 | 2026-10-17T11:22:43.843568+00:00 |
| s5-prefix-report-dry-20261017T112301Z.stdout | b26b4f9c9f5d124786067213e6a2feed9b509a5a60ea236ba21197c0bccbfcb6 | 2026-10-17T11:23:02.148578+00:00 |
| s5-prefix-report-submit-20261017T112313Z.stdout | 25478c54b563c02703a6d6fee311ed1b2312e8a39ef769807ea142564ce79829 | 2026-10-17T11:23:14.157959+00:00 |
| s5-prefix-report-accept-20261017T112330Z.stdout | 75abcddb918089b038b634bae862fc35b22e7392d67fdf66acf87bfa8f7ca4f5 | 2026-10-17T11:23:31.373974+00:00 |
| s5-prefix-report-event-20261017T112410Z.stdout | 4176436cc301439455c6ca261cc9d410033dc979c086af8039e6273810bba454 | 2026-10-17T11:24:11.108334+00:00 |
| s5-prefix-report-final-20261017T112426Z.stdout | c9d8d2856589cc84ab8c28a95e8ed72081fad2ce31a3d6d2ef0b00fb6dd531d3 | 2026-10-17T11:24:26.775064+00:00 |
| sidfix-s0-report-artifacts-retry1-20261016T101107Z.stdout | bee0fd2b10718a3f8f5d7bb41fabf85c2e9faf46fef11ef71d4c80e3e3ce91f2 | 2026-10-16T10:11:10.373407+00:00 |

原正式诊断stdout路径/workspace/vkgr-infra01-root/runs/sid-a1-prefix-report-v1/logs/stdout.log，SHA ce837fc39bccdab357459fc2c062369533589f150ce29665d9a743d74eb15f3f；其内实际执行读取UTC 2026-10-17T11:23:41.698590+00:00。原S0 result.json SHA a2e93d69bfb79af2c94e344aa5368e7604825777acc3ed6f2f6cde37c5db1924；mapping.npz SHA7423ece4be062a86fa9b1734d07649b1ba9f2999e72eccb2dbb1f4bc07facfac、validation-l50原SHA c703bbcc3e3004a58a2a14c2c7c2420b51b3b7ae0c47f6b284d587e67819ab1a。完整索引cache/codex-morning-execution-20261017/s5-prefix-final-acceptance.json。

最终执行身份独立核验：两个诊断run.json均为已push d7c6551、Git clean、argv request SHA0f46d6e101ab8566a36f7d3160f9f11ac3ce06a887b397adaa6278a4210b5464，与已部署请求相等；唯一claimed/succeeded/no rejected。来源main05/s5-prefix-final-identities-20261017T112636Z.stdout，stdout SHA5855d66b929bcdb4d71a702c11033dbe565ab6af059d4ccafd66334dd93a79a9，实际读取UTC 2026-10-17T11:26:36.957214+00:00。


## 2026-10-17 用户画像特征：冻结与对照身份登记

FINAL卡保留2026-10-17-user-profile-draft.md原名，科学内容逐字取自decision_log晚间用户画像第1–6项/执行授权；第4项补记已明确Eddy批准两臂六训练及各verify。尚无本批测量或训练。对照身份来自本日志M1-R2既有验收，原stdout main05/codex-morning-m1-fresh-20261016T102000.stdout，冻结身份表已填原manifest/config SHA，无checkpoint读取。


## 2026-10-17 19:40+08：用户画像特征（F7）like字段只读审计通过

沿FINAL卡/既有M1原inputs a12b9800，history pin8b929d10、side pin3f4ee94a；已核HIST_V2_SIDE_V1 feedback_mask bit0=like，原有合同可用，未扩展数据字段。训练原25个Parquet仅解析like列，U1/test/checkpoint均未读。真实metadata preflight p-profile-like-preflight-v1和真实audit p-profile-like-audit-v1均唯一claimed/no rejected且succeeded，原stdout分别like_audit_preflight_ok/like_audit_complete；不是以written/exit0替代。

原始训练事件总数3,720,328；True105,700、False3,614,628，missing0（缺失率0）、invalid0，schema bool。范围是训练00–24全部原始事件，不是86785监督examples，不用此前审计里的其它p统计当like率。输入SHA共28项（原inputs/history/side加25文件）见私有收据，不输出raw rows。audit真实wall2.429064512秒，CPU1/RAM4096MiB/VRAM0/1GiB/1hcap；源码3df6d9f已push、244文件新部署，旧M1原入口未改。

一次audit工具review唯一MEDIUM（异常域丢聚合回执）已fresh RED/GREEN修复，5tests/3mutants通过，无科学规则改变、不追加真实attempt；不能把这次audit工具review冒称后续模型泄漏review已完成。step2 gate通过，进入实现；尚无用户画像训练/verify入队，显存仍UNKNOWN。

| main05来源 | stdout SHA256 | 实际读取UTC |
|---|---|---|
| f7-control-source-20261017T113128Z.stdout | 68aa5b45a182f10c12a416f4a64a53b0a4f8403f15d396058fe779752762140f | 2026-10-17T11:31:28.814717+00:00 |
| f7-control-inputs-20261017T113144Z.stdout | e1a122b19041f511682c2593a976a5f6c954ad01334d1ff62d91632052f6a201 | 2026-10-17T11:31:45.289260+00:00 |
| f7-side-contract-20261017T113203Z.stdout | f764a61a205f77a0bc26975f9c56ae4794a7aa01c6bb57caa8222ed776350623 | 2026-10-17T11:32:04.077687+00:00 |
| f7-audit-deploy-inventory-20261017T113706Z.stdout | a12c316ee9b0ffbea753123127ebf57546a59f6c27cb17822750a82cbd737dba | 2026-10-17T11:37:07.262476+00:00 |
| f7-audit-deploy-apply-20261017T113723Z.stdout | 5c9efa61d5cdf59c8ecb14ecc92e7307e1c91d055bd91fe1af50ce58b007c125 | 2026-10-17T11:37:24.550487+00:00 |
| f7-like-pf-final-20261017T113858Z.stdout | 841463f21d91e4f05860c3ce54635c7f55bfcfa5e380416c22274197c35f58c0 | 2026-10-17T11:38:59.245122+00:00 |
| f7-like-audit-final-20261017T114007Z.stdout | 9b3e2d794070085c34964086413d973f4ca67d75be92a12e96238b75a4910dde | 2026-10-17T11:40:08.251150+00:00 |

原audit stdout路径/workspace/vkgr-infra01-root/runs/p-profile-like-audit-v1/logs/stdout.log，SHA d7f56d999cc2f25a3100056d3b72c412c4f264aa9b3a4d7858c0cd0f216847d5。完整缓存cache/codex-morning-execution-20261017/f7-like-audit-accepted.json。


## 2026-10-17 用户画像特征：新增实现及独立泄漏review通过

隔离分支codex/f7-user-profiles相对原producer0ca61bc仅21文件新增，旧M1源码全未改；六配置仅schema/task/arm/seed必要身份变化，projected到原mean后三seed旧配置逐字段相等（f7-config-control-parity.json）。实现严格snapshot-before-update、U2完整前缀/U3 H50、零/undefined分离、train-only线性分位数/去重、完整sidecar顺序绑定，零表在共享参数draw完后创建，不耗RNG；不要求训练轨迹相等。

完整独立review对象27344539686953a5cea3fd8f747d59c633849c40，APPROVED，HIGH/MEDIUM0；reviewer独立61/61PASS，原test_seqm1_model有tensor scalar warning，不称全suite通过。此前核心5个/审计3个有意义变异已杀，detached fake队列真实PPID1/ready-release前零发布测试通过（首次检出latch未初始化已修复）；新verify stdout含verify run_id/train_run_id/manifest SHA。另补一项完整synthetic CLI在runtime IO guard下通过，只新增测试、生产字节与review对象相同。

review docs/superpowers/reviews/2026-10-17-f7-independent-leakage-review.md。Git新增历史扫描通过并push工作分支；真实profile fit/modelPF/CUDA/VRAM/dry仍未完成，无科学训练/verify入队，资源声明待本卡实测。执行计划检查点见2026-10-17-user-profile-execution.md。

用户画像特征步骤4完成：独立APPROVED2734453后仅追加一个CLI集成测试（生产字节不变），最终a6dd81cd6a9744a750b22e1d267d941cd2eb3d47已扫描/push；新DEP/OPS=/workspace/vkgr-profile-p-v1、/workspace/vkgr-profile-ops-v1，263文件逐SHA验证、data_overlay=false。来源main05/f7-model-deploy-apply-20261017T122101Z.stdout，stdout SHA780810d3eb38799253e82fbae2bcd6531ef09169a5cf41347a944c9c9d10478c，实际读取UTC 2026-10-17T12:21:02.588312+00:00。scheduler固定62消费文件，全部16个新工程/科学身份部署前查无重复；真实fit/PF/CUDA/dry待执行，无科学入队。


## 2026-10-17 用户画像特征：train-only画像与分位数拟合验收

p-profile-prepare-preflight-v1与p-profile-prepare-v1均唯一claimed/no rejected、succeeded；后者原stdout为profile_prepare_complete，86785训练样本/2520 validation画像已严格绑定。仅训练正比例85663条拟合，七边界[0.07692307692307693, 0.13333333333333333, 0.2, 0.2631578947368421, 0.3409090909090909, 0.46, 0.6]，实际8个正比例区间/8个占用桶；零与无定义另列。manifest SHA bd25a72823a444131e73efe369d6ad3c67f1589e6a3274bcd2eacdcc639aec84，fit_input SHA f7be700bd10ba292179d22004bb0f0d3001e184be86b29df2a63a7ad38ff0c2d，fit_script SHA 7adb4729851a1dbfeb8a0b8aaa876080383519d3abfb2054e730c9d10a8b918c；profiles/quantiles SHA分别2290682f56778300e580f9bde14d80bf854c8c0e8e26a3aad3aa10381be30583 / 3c95b2774213adbd8d8914f48f3cd64ce43084f2ad88e6d709eb5b61bdcbac3f。test/U1/checkpoint均未读，无科学召回指标。

来源main05/f7-prepare-fit-accept-20261017T123340Z.stdout；stdout SHA 587c14cf8d2bf7939d5bf4649fd4ac9a1bf2b772dc7b326b7aee24077fe9f9fe；实际读取UTC 2026-10-17T12:35:25.798015+00:00。原作业stdout /workspace/vkgr-infra01-root/runs/p-profile-prepare-v1/logs/stdout.log，SHA 9c3e021ca441ac3c36b7d895277cd29a5ab9bcb754a42629fd89e83261b0965d；wall 64.75704669952393秒。下一步六配置正式路径preflight，科学训练尚未入队。


用户画像正式PF首次工程失败：p-profile-real-preflight-v1在PyTorch依赖源码test_operators路径被误挡，模型科学检查尚未产出；原终态/空stdout保留，问题与一次工程重验依据见issue I-2026-10-17-f7-framework-path。仅profile_io路径识别修复后fresh RED→GREEN、63相关回归PASS；科学模型/配置/fit producer未变，已验收画像/分位数复用、不重拟合。新DEP/OPS=/workspace/vkgr-profile-p-v2、/workspace/vkgr-profile-ops-v2，fixed e96b1178c22e7bb545a7cd4196da619a5fab3627已扫描/push，263文件逐SHA。来源main05/f7-retry-deploy-apply-20261017T124300Z.stdout，stdout SHAf316a3f6b171cc5bf885e2063a8b58c4fe3e3f74f8e62f05778cd3cf2cf246b4，实际读取UTC2026-10-17T12:41:46.849405+00:00。唯一retry1待执行，科学训练未提交。


用户画像六配置真实preflight重验p-profile-real-preflight-retry1已succeeded、claimed/no rejected，原stdout preflight_ok、engineering=False。每配置3032条（512 train+2520 validation）共享参数/零表forward逐位一致、非零p target扰动不变全部通过，profile manifest bd25a728强绑定；test/既有checkpoint未读。真实shape=batch512/H50/catalog120415/content_rows120416/supported21496/query2520。来源main05/f7-real-pf-retry-accept-20261017T124530Z.stdout；stdout SHA8f66b47cc22b1d5dade065dd3b6153199c0745e55390feb0faf7ca896bcb055a，实际读取UTC2026-10-17T12:44:53.490832+00:00；原stdout SHA09c25820be6097cac6b865b492c689577bf37753ebdd7a8ead9bc58273b51ba4，wall100.99626564979553秒。

CUDA首次提交前核对发现synthetic fixture误设catalog+2，真实M1合同为catalog+1（仅PAD）；尚未执行CUDA，无CUDA失败attempt。仅修工程夹具的映射/target偏移，fresh RED→GREEN、64相关回归PASS，模型/六配置/fit producer未改；新固定源码部署后补同检查的新revision preflight，之前成功检查保留，不追加科学名额/不重拟合。


用户画像最终源99e9b1371729caddd9f17860c92930f6d087c82a真实PF p-profile-real-preflight-shape-v3完整通过，与先前成功PF同六配置/同3032样本/同画像pin；unique claimed/no rejected、engineering=False、zero/shared/nonzero target三检查各6/6，实际shape120416物品行。原stdout /workspace/vkgr-infra01-root/runs/p-profile-real-preflight-shape-v3/logs/stdout.log，SHA47e8b577647915c5e137ac545738816273d805aa78d25100a9f76534ad26bab1；来源main05/f7-v3-pf-state-20261017T125040Z.stdout，外层stdout SHA7c383a31904ba4424234a6151c18111c5de96a59bacca2c52860e31dba3f9895，读取UTC2026-10-17T12:50:20.113849+00:00。wall100.2401168346405秒；新科学训练未提交，下一步首次CUDA工程实测。


用户画像CUDA p-profile-cuda-v1首次工程失败、claimed/no rejected，NameError根因与唯一工程重验见issue I-2026-10-17-f7-cuda-fixture-query；没有可验收CUDA峰值，不声明训练显存。新65回归仅工程修复已过，真实六训练仍未入队。来源main05/f7-cuda-accept-20261017T125345Z.stdout，SHA2cb5436741b573bdd7febd2710452b10232534727ac06f1425fe48960fd936f9，UTC2026-10-17T12:52:23.778321+00:00。


用户画像CUDA接合修复最终源b2039497d78a7185739d12a66afe232dd8056c71已push，新DEP/OPS /workspace/vkgr-profile-p-v4、/workspace/vkgr-profile-ops-v4，263文件SHA通过，原模型/prepare/六配置均与review对象生产字节相同（仅guard工程修复除外）；既有fit复用。来源main05/f7-v4-deploy-apply-20261017T125625Z.stdout，SHA57e155c78f9cf8acca58b4256f512e028401977d161f96a1839329c8a7647bbe，读取UTC2026-10-17T12:55:00.018078+00:00。新PF p-profile-real-preflight-final-v4同六配置同检查已唯一claimed/no rejected，原始来源main05/f7-v4-pf-admission-20261017T125750Z.stdout，SHA2848cf06cc2afd5ea866892ed3dacc35fe16b5960ed8defe2cbb0b4c46d91799，读取UTC2026-10-17T12:56:20.216322+00:00；终态待核，无科学入队。


用户画像最终b203949同源真实PF p-profile-real-preflight-final-v4已succeeded、唯一claimed/no rejected；原stdout preflight_ok/engineering=False，六配置各3032样本三检查全True。来源main05/f7-v4-pf-accept-20261017T125825Z.stdout；外层SHA4a4cba1fa7b3d35dcdc8ace515762bc305bcd612dec6f141f6b402f148855fbf，UTC2026-10-17T12:58:11.876910+00:00；原stdout SHA1a6a459d6e2114f8e88038799538c3bc545987604d715739698322e5470ad5f6。进入唯一CUDA retry1，正式VRAM与科学训练仍未验收。


## 2026-10-17 用户画像特征：CUDA唯一工程重验与显存验收

p-profile-cuda-retry1原终态succeeded、unique claimed/no rejected，原stdout cuda_engineering_verified。U2/U3各自新synthetic产物verify直接verified，replay best分别epoch3/2；完整真实shape(batch512/H50/catalog120415/content_rows120416/support21496/query2520)、三步有限梯度、初始化逐位一致、history oracle、全目录exact与g0均通过。两臂实测context-inclusive自身进程峰值各886MiB，正式声明各ceil256(886×1.15)=1024MiB；数值与旧声明相同但本次是新实测，未沿用。85.381034374秒工程耗时，不当作科学训练耗时/效果。test/既有checkpoint/真实训练样本均未读；只验证自己新synthetic产物。

来源main05/f7-cuda-retry-accept-20261017T130125Z.stdout，外层SHA72a66ad70a9a5d0548d86a4e8b3bfe4cec52d3b12b3b6d6e4085aeece32bab66，实际读取UTC2026-10-17T13:01:10.945064+00:00；原stdout /workspace/vkgr-infra01-root/runs/p-profile-cuda-retry1/logs/stdout.log，SHAba79682d1965c8a3d37a0a4d96be239e3b7b91f048981e6699f7650cebaa1437。原cuda-v1失败保留；科学六训练仍未提交，下一步不可变48h计划与正式dry。


用户画像正式计划/dry验收：最终b203949，六train/六唯一verify、各CPU1/RAM9216MiB/VRAM1024MiB/8hwallcap；原M1数值训练条件不变。raw plan SHA ae561f64bb3ebe9a494a459934a31d41938b35cb9beef7b316e9feb6cd158a09，canonical98d7690ffb2c45ec624e28ade7586b125d59dd61a37a386d3f978bea86e5b1ca，poll900、created1792242138.868892/deadline1792414938.868892。来源main05/f7-scientific-dry-20261017T130420Z.stdout，外层SHA41e52748ec04e06566d1902f1e7b058a0f4ca59cf10bd94728cc52e1305533ab，实际读取UTC2026-10-17T13:03:30.220247+00:00。全部launcher.prepare包络/config/source/数据身份通过，无同名run/queue/intent，下一步唯一submit，尚未以写入宣称接纳。


用户画像唯一科学submit后driver2984239 actual PPID1/alive/SID=PID，ready/release/launcher2984238、raw/canonical plan与原deadline独立核对一致；六个队列job全部与冻结plan逐字段相等。首次只读是6 pending/0 claimed/0 rejected，不冒称已接纳。来源main05/f7-scientific-admission-20261017T130530Z.stdout，SHAd22215ebf8684163429e155710194befdcd0ae4b08519751d6a6264fea0f36d3，读取UTC2026-10-17T13:04:35.255918+00:00。各verify未触发属正常，原服务器driver负责唯一接续，禁止手工抢提。


用户画像接纳窗口第二次独立只读：U2 s42/s2024各unique claimed、其余四pending，所有train/verify无rejected，原driver2984239 actual PPID1/alive。尚不能称六个都已claimed；不重投、不改资源。来源main05/f7-scientific-claim-20261017T130620Z.stdout，SHA626f62fc25fbd3f5642372e4f3fdd95bde889b63fa5d4c6aec50f2b683756ba5，实际读取UTC2026-10-17T13:05:29.605977+00:00。


用户画像接纳期实际负载只读：vkgr-status已见U2三seed和U3 s42为running，其余U3两seed pending；GPU测到三科学进程各884MiB、其余仍初始化，GPU总2672/49140MiB（本时点采样）。CPU quota10.24核，准入仍按约8.19核，未改门禁；本批每job CPU1，scheduler逐批接纳，未为排队调整声明。来源main05/f7-admission-load-20261017T130720Z.stdout，SHAf4eaf37f8eb2f3fc34127a76a1e9886996caf75dbc39c34df4f4ea01e2275b1b；stdout读数时钟13:06:42，本机实际读取UTC2026-10-17T13:07:49.923882+00:00。此为阶段负载，不据此外推最终训练峰值/耗时，正式声明只用CUDA验收公式。


## 2026-10-17 21:08+08 用户画像特征：科学队列阶段接收

已唯一发布6训练，四个unique claimed（U2三seed、U3 s42），U3 s2024/s2025各唯一pending；全部train/verify无rejected，队列job与冻结plan逐字段相等。pending未当作claimed/验收成功，不改资源、不重提；原scheduler按准入接纳即可。所有训练尚无终态，verify未触发，科学结果未验收。原driver2984239 actual PPID1/alive/SID同PID，原poll900/deadline1792414938.868892不变，只有driver补各唯一verify。

来源main05/f7-scientific-final-admission-20261017T130900Z.stdout；SHA78e3bcc793c7aeac3b83826aa5faab55a62bd5d2d83f8918258610565a2f38db；实际读取UTC2026-10-17T13:08:05.573199+00:00。四running另由main05/f7-admission-load-20261017T130720Z.stdout佐证，原负载数值见上节。剩余逐query McNemar/p_pool/bootstrap、用户塔model.users()延迟与H50<50/=50 Δhits待原raw verified齐后只读完成；p_pool同query跨seed相关的偏乐观限制保留，不判转正、不改anchor。


## 2026-10-17 21:22:29+08 用户画像特征：六训练均已接上，自动verify已触发，尚未收口

先前两个pending（U3 s2024/s2025）均已unique claimed；六训练配置/job逐字段与原不可变plan相等，所有train/verify无rejected。四训练succeeded（U2三个seed、U3 s42），U3 s2024/s2025仍running，真实scheduler/进程另有load只读佐证。

| 臂 | seed | 训练终态/阶段 | 自动verify | 科学验收 |
|---|---|---|---|---|
| U2 | 42 | succeeded | running | 未验收 |
| U2 | 2024 | succeeded | running | 未验收 |
| U2 | 2025 | succeeded | not_triggered | 未验收 |
| U3 | 42 | succeeded | pending | 未验收 |
| U3 | 2024 | running (live scheduler corroboration) | not_triggered | 未验收 |
| U3 | 2025 | running (live scheduler corroboration) | not_triggered | 未验收 |

三份verify队列绑定与原driver的verify-intent manifest SHA逐条相等，source/config/inputs/seed均由同一冻结job派生，证明是原driver自动接续；不是人工补提。当前尚无任何原stdout verified，不能以训练complete/exit0或verify claimed判验收成功。U2 s2025的verify未触发：训练完成晚于上一轮驱动检查，原900秒节奏接续，不手工抢提、不重启。原driver2984239 actual PPID1/alive/SID同PID、原deadline1792414938.868892未变，summary-final不存在；失败0/结果未验收6。

来源main05/f7-progress-final-20261017T132400Z.stdout，stdout SHA256 7ef2bbdcf0ec8a434ac2b3af68dfb87adb529001a7605da0bd9b2f6c7ce6d7d0，实际读取UTC 2026-10-17T13:22:29.353551+00:00。
来源main05/f7-auto-verify-intents-20261017T132500Z.stdout，stdout SHA256 b310930543c5dd57ebb1c91ed74c7054298e1a009fc1658d22c435ce54a8d6f4，实际读取UTC 2026-10-17T13:22:58.317235+00:00。
运行/进程佐证main05/f7-progress-load-20261017T132300Z.stdout，SHA c761b49c104e6490aa1cc6735c8ce2f96520abfcfad9bf47eae7768b87f9265b；stdout时钟与本机实际读取UTC 2026-10-17T13:24:06.226326+00:00关联，本次仅作运行佐证，不据此外推剩余时间。

逐query exact McNemar/p_pool/bootstrap、model.users()延迟、H50分档与四行机械判定待原始verified齐；本次不读未验收效果指标、不出最终Kiro裁决材料。P/G anchor维持M1-mean/A1+共现向量SID，无重提/重跑/规则变更/test或checkpoint读取。


## 2026-10-17 21:48+08 用户画像特征：六训练/六原始verify全部通过

六训练与六verify均unique claimed/no rejected、终态succeeded；每个verify原stdout都有status=verified、准确verify run_id/train_run_id/manifestSHA，与训练complete及冻结队列job逐项绑定。原驱动仍PPID1/alive、summary-final暂缺，不能冒称驱动已结束；独立原始verify已齐，不等待模型轮询才能验收。科学指标待只读归档统计，尚不作有效/转正解读。来源main05/f7-recheck-20261017T134000Z.stdout，SHAe20acc0cdceafe36837b653f04e8a0a501ece2ab616dc91e1f5afc81f8e05aca，实际读取UTC2026-10-17T13:48:52.725105+00:00。

只读验收工具727e167已review闭环/扫描push，新DEP/OPS=/workspace/vkgr-profile-report-tools-v1、/workspace/vkgr-profile-report-ops-v1，118文件逐SHA。新增7tests、原统计计时13tests通过；review唯一MEDIUM导入顺序已fresh RED/GREEN，并加实际模块源拒绝，报告docs/superpowers/reviews/2026-10-17-f7-readout-review.md；不伪称二次review APPROVED。来源main05/f7-report-deploy-20261017T140050Z.stdout，SHA035bf41c21d5d341d8be627064a7ae7d2fb7605fcf8dacf60e945687d661ea9f，实际读取UTC2026-10-17T13:58:32.661774+00:00。只读配对/PF/用户塔计时job待运行，不是新增训练/实验。


用户画像只读验收两个真实PF均succeeded、unique claimed/no rejected：配对preflight检查九run/原始verified/精确源配置数据与冻结plan；计时preflight检查固定F7/bench源与2520真实长度，尚未执行计时。新7tests、原13tests及3有意义mutants均通过。
来源main05/f7-paired-pf-status-20261017T140400Z.stdout；SHA87dfec9fca8303c2a1af247a134f6d2f61de705520c7c123db5dde16c9f123f6；实际读取UTC2026-10-17T14:00:25.983956+00:00，原stdout SHAef754ef3576fb0534e8d53268c8f3b5687f02323cbe4f824dbf83be86c8395fe。
来源main05/f7-users-pf-status-20261017T140425Z.stdout；SHA0ba980fac05404bdac7f5e8213225b0b28d2d1af389a988ee01e738a3b4be0fb；实际读取UTC2026-10-17T14:00:39.125771+00:00，原stdout SHA0eb2b6775c671091ac1e960b56d33acb35dc2f47dca1516bcfaf42a3510664a3。


## 2026-10-17 用户画像特征：六原始verified与validation配对最终验收（延迟另报）

六训练与六verify均succeeded/unique claimed/no rejected；精确verify/train run_id+manifestSHA已读到，原driver2984239写status=verified的最终汇总后退出。最终summary SHA fc98171fbb4bec400fa9a1c06dadfd1ac621530836362f63d130b0859fc330b1，各run的源/配置/数据/初始化binding与冻结plan及原验证job均核对；原PID已不存活，不重启。以下数字只来自已验收validation的best归档，2520 query（含623结构OOV）、catalog120415；九归档自行重算R/N与selection.best_metric严格一致，均PASS，79项输入SHA存私有已验收报告。

| 配置 | seed | hits | R@50 | N@50 | best/stop epoch | Δhits vs M1-mean |
|---|---|---|---|---|---|---|
| p-seqm1-mean | 42 | 466 | 0.18492063492063493 | 0.054675733733997554 | 12/15 | 0 |
| p-seqm1-mean | 2024 | 481 | 0.19087301587301586 | 0.055926352399775565 | 7/10 | 0 |
| p-seqm1-mean | 2025 | 471 | 0.18690476190476191 | 0.056395259920009738 | 11/14 | 0 |
| p-profile-u2 | 42 | 467 | 0.18531746031746033 | 0.054493774268351662 | 10/13 | 1 |
| p-profile-u2 | 2024 | 481 | 0.19087301587301586 | 0.056758091430069144 | 7/10 | 0 |
| p-profile-u2 | 2025 | 472 | 0.1873015873015873 | 0.055752560871738466 | 12/15 | 1 |
| p-profile-u3 | 42 | 466 | 0.18492063492063493 | 0.05488979508462468 | 9/12 | 0 |
| p-profile-u3 | 2024 | 481 | 0.19087301587301586 | 0.055344354218301153 | 9/12 | 0 |
| p-profile-u3 | 2025 | 478 | 0.18968253968253967 | 0.056032639806617719 | 14/17 | 7 |

| 臂 | mean Δhits | k+（Δ≥6） | pooled仅新/仅旧 | p_pool | 逐seed p<0.05数 | 冻结四行表 |
|---|---|---|---|---|---|---|
| U2 | 0.6666666666666666 | 0 | 158/156 | 0.9550086046795542 | 0 | 第4行：未检出与M1-mean的差异 |
| U3 | 2.3333333333333335 | 1 | 171/164 | 0.7431068317811952 | 0 | 第4行：未检出与M1-mean的差异 |

机械套用：两臂D均在(-6,6)，不进入第2行；至少2个逐seed p<0.05亦均不满足；独立科学泄漏review APPROVED（2734453对象，后续仅工程修复）；用户塔延迟待本批只读计时验收。**不满足冻结转正统计条件**是机械事实，不代替Kiro的收口/anchor裁决；不写等价、不写特征没有价值，不追加seed或改门槛。p_pool重复使用同一批query，跨seed不独立，偏乐观且仅为必要条件。

| 臂/seed | 仅新/仅旧 | exact McNemar p | ΔR@50 bootstrap95%CI | ΔN@50 bootstrap95%CI |
|---|---|---|---|---|
| U2/42 | 60/59 | 1.0 | [-0.007936507936507936, 0.00873015873015873] | [-0.0027092565142595817, 0.0023272779394750394] |
| U2/2024 | 38/38 | 1.0 | [-0.006746031746031746, 0.006746031746031746] | [-0.001091118187836899, 0.0027506327876479866] |
| U2/2025 | 60/59 | 1.0 | [-0.007936507936507936, 0.00873015873015873] | [-0.003163168767605439, 0.001819505656035917] |
| U3/42 | 64/64 | 1.0 | [-0.00873015873015873, 0.00873015873015873] | [-0.0021738070592074083, 0.002628490977654541] |
| U3/2024 | 49/49 | 1.0 | [-0.00753968253968254, 0.00753968253968254] | [-0.0028910447609577705, 0.0016662437221344252] |
| U3/2025 | 58/51 | 0.5657001139736338 | [-0.005555555555555556, 0.010714285714285714] | [-0.0027619461720007157, 0.00203355810437001] |

bootstrap沿原phase2 reader：10000次、default_rng(0)、query为配对重采样单位，每seed固定同一方法，不择优区间。

| 臂/seed | H50未满 n940：新/旧/Δhits | H50已满 n1580：新/旧/Δhits |
|---|---|---|
| U2/42 | 247/243/4 | 220/223/-3 |
| U2/2024 | 253/251/2 | 228/230/-2 |
| U2/2025 | 247/239/8 | 225/232/-7 |
| U3/42 | 239/243/-4 | 227/223/4 |
| U3/2024 | 247/251/-4 | 234/230/4 |
| U3/2025 | 246/239/7 | 232/232/0 |

H50已满含恰好50条与超过50后截断者，不能只凭这两档验证“增益集中于严格>50”机制；本卡只描述分档，不改判定。validation无训练事件用户0，like率无定义0（不把零比例当无定义）。未读test/任何checkpoint，未导出用户向量。P=M1-mean、G=A1+共现向量SID均保持。

来源main05/f7-paired-report-status-20261017T140710Z.stdout；SHA256 e0e5b418509c50a7b131ad78f729b68ad0061f8fee06c0717a65dd6f03708c1f；实际读取UTC 2026-10-17T14:02:15.841713+00:00。
来源main05/f7-final-owner-20261017T140730Z.stdout；SHA256 4fd9ea269ca8278731a11ccd90b3994d3609ebc185fce353ac6d9e13615a6d76；实际读取UTC 2026-10-17T14:02:30.257473+00:00。
原配对job stdout /workspace/vkgr-infra01-root/runs/p-profile-report-paired-v1/logs/stdout.log，SHA 6118eae94120e14336eff4965f00f8613d8a8eb4cc4a190f19681517ae1b97f3；原report.json SHA 70445cb8f0010b21a35efb3eb60af51547869a6023af5a9a54b02d70ce3c7051。完整79输入SHA/六新及三旧manifest/config/verify绑定见cache/codex-morning-execution-20261017/f7-paired-final-accepted.json。


## 2026-10-17 用户画像特征：用户塔延迟最终验收与完整交付

p-profile-report-users-v1唯一claimed/no rejected/succeeded，原stdout latency_measured；实跑1063.5955038070679秒。source固定727e167工具、b203949模型、b089b8f2/ed0daa90旧v2 bench，原query长度SHA c703bbcc3e3004a58a2a14c2c7c2420b51b3b7ae0c47f6b284d587e67819ab1a。CPU1线程及CUDA各mean/U2/U3，offline batch512/online batch1；每模型每模式50warm/1000 timed，完整raw时间1000条的mean/p50/p95/p99已独立重算；所有模型/设备的配对输入SHA完全相等，完整2520长度导出pass各5calls。参数数mean7761857，U2多2176、U3多640，与冻结表尺寸相符。

单位ms/query；offline已按batch512摊销，online单query。

| device | 模型 | offline mean/p50/p95/p99 | online mean/p50/p95/p99 | vs mean：offline/online |
|---|---|---|---|---|
| cpu | mean | 0.488617809/0.485805479/0.515339168/0.534720502 | 1.77824419/1.79027999/1.97046204/2.05527199 | 1/1 |
| cpu | u2 | 0.492692209/0.489273289/0.521420617/0.54550377 | 1.86822702/1.88887096/1.93562103/2.02476187 | 1.00833862/1.05060206 |
| cpu | u3 | 0.493213113/0.489226559/0.523441139/0.542776699 | 1.88163854/1.87845086/1.97628094/2.25603301 | 1.0094047/1.05814406 |
| cuda | mean | 0.0132595328/0.0130956619/0.0143116854/0.0161149201 | 6.23191972/5.89060504/7.803455/9.02193296 | 1/1 |
| cuda | u2 | 0.0138702556/0.0136654708/0.0150535839/0.0177519005 | 6.08505744/6.06965483/6.18168595/6.23045606 | 1.04605915/0.976433861 |
| cuda | u3 | 0.0136231562/0.0135922087/0.0139779318/0.0157002091 | 6.19931907/6.12274511/6.33779704/7.55769387 | 1.02742354/0.994768763 |

范围：仅model.users()，初始化权重与合成特征值、真实历史长度分布；profile IDs在timer外预打包，mean适配器在users内剥后三列，其开销包含在计时中。不是读取训练checkpoint的计时、不含特征取数/物品塔/检索/ANN/网络，不称端到端或线上延迟。CUDA online略低只是本次描述，不能据此称加特征使模型更快；未锁GPU时钟，顺序/噪声等限制保留，不重跑择优。

完整机械条件：U2与U3均【第2行不满足；至少2seed exact p<0.05不满足；独立科学泄漏review APPROVED满足；用户塔限定口径延迟报告已附满足】。因此本轮统计转正条件未满足，仍交Kiro/Opus正式解读与收口，不自动写关键decision_log/不换anchor/不增加臂seed。六训练/六verify/配对/H50/计时全部验收，无运行中或待验收科学job，原driver退出；历史两个工程失败与原收据保留，并不写成“项目从未失败”。旧G6 fixture/历史verify身份问题仍OPEN。

来源main05/f7-users-report-accept-20261017T142125Z.stdout；stdout SHA256 8714c9008a09e509c8c218a4e5286fd5b1fe81083fc4d0065b94e54a1e40cee4；实际读取UTC 2026-10-17T14:21:05.467825+00:00。
来源main05/f7-users-physical-report-20261017T142150Z.stdout；stdout SHA256 09b5d4d6dc4ce9cd4e838f230cf291533f3a224296a4a3689292658125d66f38；实际读取UTC 2026-10-17T14:21:19.864721+00:00。
原计时stdout /workspace/vkgr-infra01-root/runs/p-profile-report-users-v1/logs/stdout.log，SHA c4e95c94789cf1f8c963a5fc2c661cbefc1f24664387cf74711a6c7e5a6d6c39；原report.json SHA 43fc4bef8d6db91d799686957dade151352c433adf83d6a85532de54e1f9aca5。完整raw时间/同输入SHA/源pin/统计重算检查见cache/codex-morning-execution-20261017/f7-users-final-accepted.json。

两只读分析job实际run.json的Git commit均727e167679d694ca249d0f207ad12f0f7ca703a7、dirty=false，原claimed job唯一/无rejected，与已部署代码/请求身份匹配。来源main05/f7-readout-identities-20261017T141850Z.stdout；SHA 40b27e98307fd4f194f118a689cf182903611ef7607ca17e541628d21ea3e944；实际读取UTC 2026-10-17T14:19:14.259748+00:00（该身份读取时计时尚在运行，其后成功终态以最终计时收据为准）。完整交付收据索引cache/codex-morning-execution-20261017/f7-final-delivery-receipts.json；Kiro完整提示词docs/superpowers/reviews/2026-10-17-user-profile-result-review-prompt.md。


## 2026-10-17 深夜裁决同步：用户画像特征两臂不转正、收口

Kiro/Opus已接受完整性、两臂均第4行，四转正条件前两项不满足、后两项满足，裁决两臂均不转正、F7收口；P anchor维持M1-mean。原测量/verify/参数/延迟数字均不改，结果不解释为等价或画像无价值；不新增F7后继/seed/字段。原判定措辞按其第4条在一次只读诊断前暂定，诊断只细化措辞或揭示需回Kiro的问题，不自动推翻裁决。

来源docs/decision_log.md最后“2026-10-17 深夜：用户画像特征（F7）结果裁决”原文，当前完整文件SHA 9feefcb88971cf2d9e8da816b76a49338b9600cfa96f6d228cfa3b50f8469692，实际本机读取UTC 2026-10-17T14:37:15.928719+00:00。decision_log原字节提交，不由Codex修改科学裁决。


## 2026-10-17 深夜：用户画像收口后一次只读排序与档位诊断

授权decision_log深夜§4；非训练、不占新名额、未改不转正/收口裁决。源码494351ec4fba9ab89f59ecb83598c7ee46aefb66已review APPROVED/扫描push，新DEP/OPS /workspace/vkgr-profile-diag-tools-v1、/workspace/vkgr-profile-diag-ops-v1，120文件SHA；真实preflight与唯一正式p-profile-ranking-diagnostic-v1均succeeded/unique claimed/no rejected，实际Git494351e且clean。正式wall2.4136528968811035秒，CPU1/RAM8192MiB/VRAM0。无checkpoint/test/模型导入/重拟合/新推理或导出。15个已验收run、best归档身份与R/N自校验，validation侧车/分位数/原纯生产脚本SHA、ordinal/user/history/positions身份与顺序均核；127输入SHA保留accepted报告，报告物理SHA/JSON与原stdout完全相等。

F7每seed2520同序query，U2/U3与M1-mean同42/2024/2025；bi参照用独立预注册314159/271828/161803，与其同seed mean比较。bi参照与F7 seed集合不同，以下跨组对照仅描述，不做新的跨实验配对检验。

| 比较 | seed | 平均重合/50 | p05/p50/p95（nearest rank） | min/max | 相同集合/完全同序query |
|---|---|---|---|---|---|
| u2_vs_mean | 42 | 36.5357142857 | 27/37/43 | 3/47 | 0/0 |
| u2_vs_mean | 2024 | 39.65 | 30/41/46 | 3/48 | 0/0 |
| u2_vs_mean | 2025 | 37.4416666667 | 28/38/44 | 4/47 | 0/0 |
| u3_vs_mean | 42 | 37.4051587302 | 29/38/44 | 10/47 | 0/0 |
| u3_vs_mean | 2024 | 39.2174603175 | 32/40/44 | 17/47 | 0/0 |
| u3_vs_mean | 2025 | 37.9821428571 | 29/39/44 | 12/47 | 0/0 |
| bi_vs_mean_reference | 161803 | 33.4531746032 | 24/34/40 | 10/47 | 0/0 |
| bi_vs_mean_reference | 271828 | 34.5531746032 | 25/35/42 | 13/46 | 0/0 |
| bi_vs_mean_reference | 314159 | 37.0511904762 | 28/38/43 | 13/48 | 0/0 |

完整top50重合整数频数（键=集合交集大小，值=query数，未列键为0；不是Jaccard）：
- bi_vs_mean_reference/161803：{"10": 3, "12": 2, "13": 3, "14": 3, "15": 1, "16": 4, "17": 3, "18": 8, "19": 15, "20": 8, "21": 16, "22": 17, "23": 20, "24": 32, "25": 27, "26": 52, "27": 59, "28": 83, "29": 106, "30": 145, "31": 161, "32": 188, "33": 168, "34": 221, "35": 241, "36": 204, "37": 182, "38": 199, "39": 131, "40": 111, "41": 64, "42": 28, "43": 11, "44": 2, "45": 1, "47": 1}
- bi_vs_mean_reference/271828：{"13": 2, "14": 1, "15": 4, "17": 6, "18": 5, "19": 3, "20": 10, "21": 13, "22": 12, "23": 21, "24": 22, "25": 31, "26": 38, "27": 38, "28": 67, "29": 94, "30": 112, "31": 109, "32": 147, "33": 190, "34": 199, "35": 232, "36": 214, "37": 203, "38": 203, "39": 183, "40": 115, "41": 105, "42": 78, "43": 42, "44": 12, "45": 8, "46": 1}
- bi_vs_mean_reference/314159：{"13": 1, "14": 2, "17": 2, "18": 2, "19": 3, "20": 6, "21": 7, "22": 8, "23": 4, "24": 5, "25": 21, "26": 20, "27": 29, "28": 32, "29": 43, "30": 37, "31": 60, "32": 78, "33": 132, "34": 127, "35": 151, "36": 198, "37": 234, "38": 238, "39": 236, "40": 268, "41": 197, "42": 151, "43": 114, "44": 73, "45": 27, "46": 11, "47": 2, "48": 1}
- u2_vs_mean/42：{"13": 2, "14": 1, "15": 2, "16": 2, "17": 6, "18": 4, "19": 4, "20": 5, "21": 5, "22": 9, "23": 12, "24": 11, "25": 19, "26": 26, "27": 33, "28": 49, "29": 52, "3": 1, "30": 54, "31": 94, "32": 96, "33": 128, "34": 132, "35": 159, "36": 193, "37": 159, "38": 231, "39": 204, "40": 213, "41": 177, "42": 181, "43": 125, "44": 71, "45": 40, "46": 12, "47": 3, "5": 2, "6": 1, "7": 1, "8": 1}
- u2_vs_mean/2024：{"10": 1, "12": 1, "17": 1, "18": 2, "19": 3, "20": 2, "21": 3, "22": 4, "23": 3, "24": 7, "25": 9, "26": 15, "27": 18, "28": 7, "29": 24, "3": 2, "30": 29, "31": 30, "32": 38, "33": 68, "34": 57, "35": 80, "36": 115, "37": 128, "38": 150, "39": 195, "40": 231, "41": 248, "42": 253, "43": 261, "44": 232, "45": 163, "46": 90, "47": 40, "48": 7, "7": 1, "8": 1, "9": 1}
- u2_vs_mean/2025：{"10": 2, "13": 1, "14": 4, "15": 2, "16": 2, "17": 3, "18": 4, "19": 3, "20": 3, "21": 11, "22": 7, "23": 5, "24": 12, "25": 14, "26": 16, "27": 24, "28": 25, "29": 32, "30": 47, "31": 60, "32": 73, "33": 104, "34": 121, "35": 155, "36": 165, "37": 158, "38": 224, "39": 234, "4": 1, "40": 247, "41": 252, "42": 192, "43": 146, "44": 109, "45": 37, "46": 22, "47": 2, "6": 1}
- u3_vs_mean/42：{"10": 1, "15": 1, "16": 3, "17": 2, "18": 1, "19": 4, "20": 2, "21": 8, "22": 5, "23": 3, "24": 7, "25": 10, "26": 16, "27": 22, "28": 33, "29": 44, "30": 43, "31": 65, "32": 60, "33": 105, "34": 123, "35": 150, "36": 178, "37": 223, "38": 249, "39": 256, "40": 239, "41": 218, "42": 179, "43": 140, "44": 82, "45": 35, "46": 12, "47": 1}
- u3_vs_mean/2024：{"17": 1, "18": 1, "19": 1, "21": 3, "23": 1, "24": 4, "25": 7, "26": 8, "27": 6, "28": 11, "29": 15, "30": 17, "31": 23, "32": 41, "33": 50, "34": 71, "35": 110, "36": 118, "37": 181, "38": 223, "39": 284, "40": 282, "41": 306, "42": 308, "43": 224, "44": 144, "45": 54, "46": 22, "47": 4}
- u3_vs_mean/2025：{"12": 1, "15": 1, "18": 1, "19": 3, "20": 4, "21": 6, "22": 8, "23": 3, "24": 3, "25": 18, "26": 24, "27": 19, "28": 20, "29": 22, "30": 39, "31": 43, "32": 55, "33": 85, "34": 117, "35": 126, "36": 189, "37": 182, "38": 225, "39": 263, "40": 254, "41": 257, "42": 208, "43": 166, "44": 103, "45": 55, "46": 14, "47": 6}

| 三seed合并描述 | 重复query观测数 | 平均重合/50 | discordant总数 | 命中侧45–50数/占比 |
|---|---|---|---|---|
| bi_vs_mean_reference | 7560 | 35.0191798942 | 413 | 84/0.203389830508 |
| u2_vs_mean | 7560 | 37.8757936508 | 314 | 102/0.324840764331 |
| u3_vs_mean | 7560 | 38.2015873016 | 335 | 116/0.346268656716 |

合并7560是三seed重复观测同2520query，不是独立用户数或新的p检验。F7所有query都不是完全相同的top50集合；与参照相比更重合但不能写“几乎不变”。

Discordant target名次：只命中一侧的target仅在该侧可观察精确1–50名。另一侧仅知>50，**精确名次、51–55计数、>55计数均UNKNOWN**，未填51，未读更深候选/CP来补。下表压缩展示命中侧直方图，不是新分档门槛；完整每seed双方向1–50频数在accepted报告。

| 组别/命中侧 | 1–10 | 11–20 | 21–30 | 31–40 | 41–44 | 45–50 |
|---|---|---|---|---|---|---|
| bi_vs_mean_reference/only_new | 1 | 33 | 42 | 79 | 31 | 38 |
| bi_vs_mean_reference/only_control | 5 | 23 | 40 | 54 | 21 | 46 |
| u2_vs_mean/only_new | 2 | 11 | 29 | 48 | 26 | 42 |
| u2_vs_mean/only_control | 2 | 5 | 25 | 36 | 28 | 60 |
| u3_vs_mean/only_new | 2 | 10 | 23 | 52 | 29 | 55 |
| u3_vs_mean/only_control | 2 | 15 | 15 | 40 | 31 | 61 |

only_new=仅新臂命中、其命中侧为新臂；only_control=仅对照命中、其命中侧为mean。F7 b/c逐seed与已验收配对报告精确一致，bi三个delta仍−3/+22/+16。可见命中侧多数在1–44名；不能据此判断两侧完整45–55分布，因为另一侧右删失。

validation档位：2520query/2520去重user，各用户固定训练末快照一致，query数与人数本次完全相同；只对已验收profiles.validation按原producer transform，不重拟合分位数、不读取原始训练事件。

| 字段 | 实际非空档 | 每档人数（未列=0） | 常量 |
|---|---|---|---|
| H50_like_rate | [0, 1, 2, 3, 4, 5, 6, 7, 8] | {"0": 100, "1": 538, "2": 361, "3": 380, "4": 223, "5": 275, "6": 265, "7": 199, "8": 179} | False |
| N_exp | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] | {"1": 4, "10": 9, "2": 19, "3": 55, "4": 182, "5": 345, "6": 573, "7": 699, "8": 477, "9": 157} | False |
| N_like | [1, 2, 3, 4, 5, 6, 7, 8] | {"1": 227, "2": 385, "3": 483, "4": 534, "5": 458, "6": 294, "7": 121, "8": 18} | False |

H50 like率桶0=100人，桶9（无定义）=0人，二者未混用。侧车的validation档位均非常量，并与原F7 manifest/fit/source绑定一致；这不是读取训练后向量或测量实际p贡献的证据。

结论/待复查仅见issue_log I-2026-10-17-f7-postclose-diagnostic：不对(a)/(b)/(c)作唯一因果判定，不自动添加“画像通道影响很小”的确定性句；完整模型的预测变化还包含不同训练轨迹影响（FINAL已不要求轨迹相等）。不改变F7不转正/收口和P anchor。

来源main05/f7-diag-run-state-20261017144821Z.stdout；stdout SHA256 8882ddb6a1cb42e367c8e37aa041ab50a13a4b7b698b9713deb29feb26bbc9d3；实际读取UTC 2026-10-17T14:48:24.789130+00:00。
来源main05/f7-diag-physical-20261017144908Z.stdout；stdout SHA256 3db1e3817373f2159e7247f4762cb1e9863e9a43dffd75eb3bf10f94537342ba；实际读取UTC 2026-10-17T14:49:11.840480+00:00。
来源main05/f7-diag-pf-state-20261017144623Z.stdout；stdout SHA256 d13e663295680880aabc3cda36da1b8ecd1a5846400f3fac62051d8dab576076；实际读取UTC 2026-10-17T14:46:26.013387+00:00。
来源main05/f7-diag-identities-20261017145044Z.stdout；stdout SHA256 6e015b9b82405506f00162b2e9b13fd07f2b080f3da01ed27ea8ed710b712f91；实际读取UTC 2026-10-17T14:50:48.114691+00:00。
原正式stdout /workspace/vkgr-infra01-root/runs/p-profile-ranking-diagnostic-v1/logs/stdout.log，SHA 3ae3e91c2d6b004101258f2df185ad6eb4a9d304915ca64a2be4fbb01585559f；原report.json SHA 6441969a51ad825910cd24e1aed859e061a190655506a51b2753cfccd492a0aa。完整127输入SHA/原验证绑定/完整直方图：cache/codex-morning-execution-20261017/f7-postclose-diagnostic-accepted.json。
