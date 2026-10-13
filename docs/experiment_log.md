密级: 公开

# Experiment Log

## 当前状态

尚无实验。

本轮未下载数据、未训练模型、未验证效果。

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
