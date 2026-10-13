密级: 公开

# Architecture

## 当前边界

本文件与根 `tasks.md` 使用同一组 P0～P13 编号。

本文件只定义接口、数据流和artifact boundary，不包含实现代码。

P2已完成MVP下载和隔离环境验收；P3进行中，尚未训练模型或产生模型质量指标。当前状态以STATE.md为准，阶段证据见对应执行worktree。

## 总体数据流

```text
public source manifest
→ raw server-side partitions
→ schema/order audit
→ frozen label/history/split/catalog contract
→ traditional baselines
→ SID tokenizer
→ TIGER smoke
→ unified item-level evaluator
→ flagship streaming/ANN
→ dense fallback
→ aligned SID refresh
→ optional multi-route
→ release audit
```

每阶段只消费前一阶段的versioned artifact，并输出manifest和hash。

Public Git只保存代码、配置、文档、synthetic fixture和小摘要；大产物留服务器。

## P0｜Bootstrap与治理

输入：冻结研究方案。

输出：根治理文件、设计文档、draft configs、Codex handoff和server runbook。

Gate：`AGENTS.md`必须正式落位；规则、阶段和事实源无冲突。

## P1｜基础代码与可审计运行框架

输入：P0文档。

输出：最小Python package、CLI、config schema、logging、run manifest、deterministic seed、synthetic fixture与targeted tests。

Gate：synthetic dry-run可复现；manifest能定位config、输入指纹、seed与资源。

## P2｜服务器验收、数据下载与完整性检查

输入：P1可复现框架、官方VK-LSVD source manifest，以及用户对计费实例的明确批准。

输出：当前已确认平台A6000不超过1小时的acceptance probe报告、服务器本地raw partitions、checksum inventory、许可快照和失败恢复记录。

Gate：按`docs/server_runbook.md`验证SSH reconnect、CUDA/PyTorch、FAISS、RAM/disk、Hugging Face吞吐、disk I/O与经批准的平台持久化/恢复验证；只取得批准的subset；原始数据不进入Git；磁盘余量满足runbook。

## P3｜数据探查与合同冻结

输入：MVP raw partitions。

输出：schema/order evidence、feedback coverage、sequence length、duplicate、catalog/first-seen/OOV报告，以及冻结的label/history/split/catalog policy。

Gate：只用train定label；若`like`覆盖不足才定义composite；看valid/test前冻结。

## P4｜传统召回基线

输入：冻结queries、catalog和64D content embedding。

输出：Popularity、content-centroid + `IndexFlatIP`、chunked exact evaluator和固定query audit set。

Gate：toy brute-force、chunked exact与IndexFlat结果一致；不物化完整score matrix。

两天后再补trained dense sequential baseline；ItemCF/SASRec不属于两天硬承诺。

## P5｜SID tokenizer与约束结构

输入：train-observed catalog和content embedding。

输出：balanced hierarchical k-means codebook、item→SID、collision suffix、compact prefix constraint和诊断。

Gate：所有train-observed item可逆映射；collision、utilization、entropy/Gini与prefix occupancy可计算。

首版只做一个tokenizer；RQ-VAE/collaborative tokenizer不属于两天硬承诺。

## P6｜TIGER smoke

输入：target-prefix histories、SID mapping和constraint。

输出：small TIGER one-seed checkpoint、训练manifest、catalog-constrained decode和固定query smoke结果。

Gate：合法SID能映射回item；invalid/dead-end/duplicate与资源指标完整；不要求优于baseline。

## P7｜MVP vertical slice

输入：P4与P6同一冻结合同下的candidates。

输出：固定query audit set上的Popularity、content baseline与TIGER统一item-level对照。

Gate：共享history/catalog/query/features；记录item Recall/NDCG、collision、latency、RAM/VRAM与artifact hash。

若两天不足，保留固定audit set；不得伪装全test benchmark。

## P8｜Flagship streaming与基础设施

输入：`ur0.01`单week pilot。

输出：streaming ETL、out-of-core catalog/SID、compact constraint、ANN pilot、scale re-validation、snapshot/restore、冻结的旗舰validation/final holdout manifest与成本manifest。

Gate：A6000/RAM/disk保留至少20%余量；final holdout不参与调参；预算获批。

## P9｜Flagship pure generative retrieval

输入：P8基础设施与冻结旗舰合同。

输出：trained dense sequential control、pure TIGER、ANN approximate full-catalog与固定query blockwise exact audit。

Gate：报告ANN recall、exact delta、item quality、reachability、latency与memory。

## P10｜Hybrid dense fallback

输入：pure GR failure slices与dense candidates。

输出：固定总候选预算下的GR/dense union、fusion和fallback reason。

Gate：增益不能来自扩大总候选或不公平算力；必须定位到low-support/OOV/dead-end slices。

## P11｜Dynamic SID refresh

输入：连续week observed-catalog snapshots、旧checkpoint与新SID。

输出：stale、fresh-unaligned warm-start、aligned refresh、fresh-scratch，以及dense fallback组合。

Gate：多seed比较SID churn、old/new reachability、item quality、refresh wall-clock和GPU cost。

## P12｜Multi-route retrieval

输入：P10/P11稳定单路结果。

输出：目标条件或互补route、固定总预算fusion和route-level attribution。

Gate：新增route必须贡献独立命中；不能用扩大候选量解释收益。

MaskGR仅在profiling证明AR decode为瓶颈后作为stretch；RPG/HSTU/LLM/RL defer。

## P13｜Release与公开复核

输入：拟公开的代码、配置、manifests和摘要。

输出：可复现README、frozen configs、synthetic tests、THIRD_PARTY_NOTICES、secret/data/license scan和release evidence。

Gate：小数据链路可从干净环境复现；所有指标指向`docs/experiment_log.md`；禁止未授权数据与第三方代码。

## Artifact boundaries

### Public Git

原创代码、配置、文档、接口schema、source/split/hash manifest、synthetic fixture、小型指标摘要。

### Server only

Raw/processed parquet、完整embedding、query dataset、checkpoint、optimizer state、ANN index、codebook、constraint结构、完整SID map、逐query预测和resource trace。

### Secret

API key、SSH key、cloud token、signed URL、account identifier和完整连接信息不进入任何manifest。

## 公开来源

- VK-LSVD: https://huggingface.co/datasets/deepvk/VK-LSVD
- TIGER: https://arxiv.org/abs/2305.05065
- LIGER: https://arxiv.org/abs/2411.18814
- SID staleness: https://arxiv.org/abs/2604.13273
- Collision-faithful evaluation: https://arxiv.org/abs/2605.25330
