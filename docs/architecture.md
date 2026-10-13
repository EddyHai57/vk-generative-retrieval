密级: 公开

# Architecture

## 当前边界

本文件与根 `tasks.md` 使用同一组 P0～P13 编号。

本文件只定义接口、数据流和artifact boundary，不包含实现代码。

P2已按批准范围完成固定revision服务器下载、隔离依赖与资源验收。P3已实现数据审计与prefix生成并冻结合同；P4已实现未训练的Popularity/content baseline及固定audit exact evaluator，实际质量/resource摘要只读experiment_log，整体验收状态只读STATE。

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

Gate：按`docs/server_runbook.md`验证SSH reconnect、CUDA/PyTorch、FAISS、RAM/disk、固定revision dataset传输吞吐、disk I/O与经批准的平台持久化/恢复验证；只取得批准的subset；原始数据不进入Git；磁盘余量满足runbook。

## P3｜数据探查与合同冻结

输入：P2固定revision/hash的MVP raw partitions，仅服务器只读。

输出：schema/order evidence、feedback coverage、sequence length、duplicate、catalog/first-seen/OOV报告，以及版本化label/history/split/catalog manifest。生产模块src/vkgr/data_audit.py提供train→freeze→holdout CLI；实际producer/source/lock/精确依赖与policy由train audit绑定。完整queries/catalog/training examples保留server-only，Public Git只保存脱敏冻结manifest。

Gate：只用train定label；若`like`覆盖不足才定义composite；看valid/test前冻结。

## P4｜传统召回基线

输入：冻结queries、catalog和64D content embedding。

输出：Popularity、normalized-content centroid、固定float64 pair reduction的blockwise exact evaluator与固定query audit set；IndexFlatIP只作canonical tie后的toy reference。原创src/vkgr/baselines.py与baseline_run.py消费冻结静态P3输入，run manifest verify重算抽样、两baseline预测与stored-prediction指标。真实score矩阵只物化bounded query/item块，完整query/预测只留服务器。

Gate：toy brute-force、chunked exact与IndexFlat结果一致；不物化完整score matrix。

两天后再补trained dense sequential baseline；ItemCF/SASRec不属于两天硬承诺。

## P5｜SID tokenizer与约束结构

输入：train-observed catalog和content embedding。

输出：balanced hierarchical k-means codebook、item→SID、collision suffix、compact prefix constraint和诊断。

Gate：所有train-observed item可逆映射；collision、utilization、entropy/Gini与prefix occupancy可计算。

首版只做一个tokenizer；RQ-VAE/collaborative tokenizer不属于两天硬承诺。

P5生产接口见sid.py/sid_run.py。SID固定3语义digits+1确定性collision suffix，per-position token namespace独立。Constraint用sorted uint64 mixed-radix SID keys+uint32 item permutation查询prefix range；empty prefix为root，完整SID为terminal。static reader要求调用方pin manifest SHA并校验map/codebook graph/capacities/constraint/features/audit/producer/resource证据；producer replay另重读固定官方features并refit，不能把static checksum检查冒充真实fit复算。完整map/codebook/constraint/normalized cache仅server-only，Public只保存小aggregate/hash。算法/gate细节读p5_validation。

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

## P6实际接口补充

原创tiger.py提供SIDTokens/TigerModel/constrained_beam/decode_history；四digit namespace与learned absolute位置编码共享history/target。Encoder padding与decoder causal/cross-attention padding masks明确，固定4输出无需EOS；每query memory只编码一次再复用beam。Training.py提供ShuffledCursor、finite train_step与caller-pinned自有checkpoint（weights_only）恢复model/optimizer/global RNG/data cursor。tiger_run.py消费静态固定P3examples/P4audit/P5SID，通过train→实际退出→resume→decode→verify独立CLI阶段；冻结config/source/lock/input绑定，最后资源gate与artifact hashes之后独占发布successmanifest。所有checkpoint/optimizer/完整query与prediction仅服务器，公开只脱敏摘要。参数与失败/验证证据唯一入口p6_validation，指标数字只experiment_log。


## P7实际统一比较边界

P7不增加生产模块。只消费已验收P4/P6 audit与metric artifact；ignored cache中的独立stdlib审计逐字比较query/history/target/OOV、重建抽样成员并手写item指标。Public artifacts/p7-comparison-manifest.json用role/hash绑定P3合同、P4baseline、P5SID、P6run与同源feature/catalog/query，不放完整candidates或私有路径。统一质量与系统成本摘要只读experiment_log；latency边界不同，调参次数相同不等于计算预算相同。当前验收状态只读STATE，P8之后的数据流仍是计划。

## P4B traditional consumer boundary

`retrieval_data`负责冻结hash/catalog/PAD、严格prefix/fullpast证据、按split reader和server-only prepared artifacts；不调用同时加载holdouts的旧baseline reader。`retrieval_training`负责独立epoch cursor、distinct-target/mask/FP32 loss和可恢复checkpoint状态，不复用旧TIGER跨epoch补batch的cursor。`retrieval`组合来源/资源audit、真实train-only prepare及bounded synthetic train/verify；test gate关闭。Formal architectures/validation选择、baseline/index/export/fusion/ANN仍由后续owner实现。

## P4C traditional baseline boundary

retrieval_baselines负责all-train-like Pop/binary incidence、逐row sparse cosine top200、history事件sum/Popfill、shared exact content评价及input-to-output replay；retrieval组合validation-only pinned输入、bounded graph/resource pilot、exclusive realrun/verify与legacy regression。P4B prepared producer identity保持原版本，P4C consumer另绑定source/transport/dependencies。Graph/arrays/predictions不进public Git，test gate closed；后续architectures仍由P4D负责。

## P4D formal core consumer boundary

retrieval_models提供四个固定TwoTower表示；retrieval_experiment提供finiteconfig/runtime gates、validation-best session/epochcheckpointchain、preparedvalidation-only loader、actualCUDAresourcepilot、float32 export与逐checkpoint输入到输出verify、completefixedsearchselection。它复用既有P4B sampler/mask/FP32step/checkpoint及P4 exact scorer，不改producer或旧TIGER接口。FormalCLI单独模块，旧syntheticCLI回归保留；实体vectors/checkpoints/predictions只server。当前状态/实际实验只读STATE/experiment_log。


P4D v2 stats serialization compatibility使用独立scripts/p4d_v2_verifier_compat.py：训练producer文件和fingerprint保持原v2，完整原transport固定pin；只在隔离namespace中替换verify_session/verify_draw_provenance的两处stats表示比较。其余函数保持相同codeobject；epochs键严格同类int或canonicalstr，非法key/数值漂移拒绝。train/resume/verify/select每次调用均显式进入adapter并在模型创建前保存producer＋adapter/proof双身份sidecar；验证结果另有receipt，旧manifest/checkpoint不重stamp。长GPU oracle为same-process独立构造恢复，先前真实bounded不同进程恢复证据另列。
