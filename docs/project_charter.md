密级: 公开

# Project Charter

## 项目结论

本项目研究 VK-LSVD 的 week-level distribution shift 下，SID staleness 与 item reachability 如何变化，以及 aligned SID refresh + dense fallback 能否用低于 fresh-scratch retraining 的成本恢复 item-level retrieval quality。

当前阶段只是本地 scaffold。

本轮未下载数据、未训练模型、未验证效果。

公司内部信息、数据、schema、代码与实验结果不得进入本项目。

## 核心可证伪问题

在相同 history、target、catalog、item feature 与 serving budget 下，比较 stale SID、fresh-scratch SID、aligned-refresh SID，以及它们各自结合 dense fallback 的结果。

若 aligned refresh 不能以更低 refresh cost 接近 fresh-scratch 的 item Recall/NDCG，核心假设被否证。

若 dense 或 hybrid 在 quality、latency、memory 三项上稳定支配 pure generative retrieval，则停止扩大 pure generative model。

若 collision-corrected item-level 指标消除 SID-level 增益，则先修 tokenizer 或 evaluator，不增加模型复杂度。

若 new-item path 不可达而 content ANN 可达，则优先采用 dense fallback，不用训练目标掩盖 support 缺口。

## 项目层级

### MVP：两天可审计 vertical slice

数据层使用 VK-LSVD `ur0.01_ir0.01`。

任务层先冻结 label、history、target、split 与 train-observed catalog 合同。

基线只做 Popularity 与无训练 content-centroid + `IndexFlatIP`。

SID 只做 balanced hierarchical k-means + collision suffix。

生成模型只做 small TIGER one-seed。

评估只承诺固定 query audit set，而非全量 test queries。

评估必须报告 item-level Recall/NDCG、invalid、dead-end、collision、latency、RAM 与 VRAM。

MVP 的目标是证明接口、约束与评估闭环，不是证明算法优越。

### Flagship：Advanced Generative Retrieval

数据层升级到 `ur0.01` 做 million-item / multi-million-item catalog scale audit。

基线补一个 trained dense sequential model，确保不是只与无训练 baseline 比较。

模型比较 pure TIGER、dense retrieval 与 LIGER-style hybrid。

动态实验按 week 比较 stale、fresh-scratch、aligned refresh，并拆分 old/new item reachability。

规模评估采用 ANN approximate full-catalog，并在固定 query block 上做 blockwise exact audit。

最终报告 multi-seed quality、collision、cold/first-seen、latency、memory 与 refresh cost。

MaskGR 只作为 stretch，不属于核心完成条件。

## Success

数据合同在查看 validation/test 指标前冻结并可追溯。

同一比较中的方法共享 split、catalog、history、item feature、query set 与调参预算。

SID-level 与 collision-corrected item-level 指标分开报告。

exact evaluator 在 toy data 上通过 brute-force、chunked exact 与 `IndexFlatIP` 一致性检查。

所有 run 保存 config、commit、dataset manifest/hash、artifact hash 与资源记录。

负结果和 stop gate 与正结果使用相同留档标准。

公开仓库只保存代码、配置、文档、manifest 与小型指标摘要。

## Non-goals

不预测 next exposed item 作为用户偏好主任务。

不声称存在精确 timestamp、真实 listing time、普通 click 或 CTR label。

不训练 raw multimodal encoder；官方 64D embedding 只能称 content-aware 或 multimodal-derived embedding。

不做线上 A/B，不推断 online lift。

不把两天 vertical slice 包装成完整 benchmark 或研究结论。

不在当前主线堆叠 LETTER-lite、MaskGR、RPG、HSTU、LLM 或 RL。

不重分发 raw parquet、content embedding、checkpoint、index 或完整 SID map。

## 完成度命名门槛

### Scaffold

仅完成治理文档、合同、接口、配置与目录约定。

没有数据验证、训练或效果结论时只能称 scaffold。

### TIGER Benchmark

冻结数据合同，并在同一固定 query audit set 跑通 Popularity、content baseline、SID tokenizer 与 small TIGER。

保存 item-level quality、collision、合法性、latency、memory 与 artifact evidence 后，才可称 TIGER benchmark。

One-seed 结果只表示复现闭环，不表示稳定收益。

### Advanced GR

完成 trained dense control、pure TIGER、LIGER-style hybrid 的公平比较。

完成 stale、fresh-scratch、aligned refresh 的 week-level multi-seed 实验。

完成 million-item scale 下的 ANN 与 exact audit、collision、first-seen/OOV、latency、memory 与 refresh-cost 审计。

上述证据齐全后，才可称 Advanced Generative Retrieval。

## 公开来源

VK-LSVD data card: https://huggingface.co/datasets/deepvk/VK-LSVD

TIGER: https://arxiv.org/abs/2305.05065

LIGER: https://arxiv.org/abs/2411.18814

SID staleness: https://arxiv.org/abs/2604.13273

Collision-faithful evaluation: https://arxiv.org/abs/2605.25330
