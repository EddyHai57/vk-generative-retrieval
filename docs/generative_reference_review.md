# 生成式召回参考项目调研

密级: 公开

调研日期：2026-10-07。Eddy授权Sol短期只读subagent调研，Astra核查关键原始页面并整合。没有下载/运行第三方代码或权重，没有证明其在本项目环境可复现。阶段计划唯一入口为主目录tasks.md“生成式召回完成路线（提案）”，本文只记录来源与设计依据，不是实验结果或执行授权。

## 推荐阅读顺序与证据

|项目|已核事实与来源|本项目借鉴及边界|
|---|---|---|
|[RQ-VAE Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)|Edoardo Botta社区实现，非TIGER原作者官方实现。[README](https://raw.githubusercontent.com/EdoardoBotta/RQ-VAE-Recommender/main/README.md)包含RQ-VAE、collision处理、prefix约束、T5及CPU tests；[LICENSE](https://raw.githubusercontent.com/EdoardoBotta/RQ-VAE-Recommender/main/LICENSE)实际为MIT。|优先学习tokenizer与retriever分阶段冻结、collision与接口测试。Amazon配置及自动下载不能替代VK合同；本轮不引入代码或依赖。|
|[Meta LIGER](https://github.com/facebookresearch/liger)|论文官方仓库。[README](https://github.com/facebookresearch/liger/blob/main/README.md)列出dense/generative/hybrid实验；根[LICENSE](https://raw.githubusercontent.com/facebookresearch/liger/main/LICENSE)为CC-BY-NC4.0。README提及rqvae另有Apache2，但本轮未核其具体许可文件。|优先学习候选预算、纯生成与dense的独立对照和hybrid增量。按[论文](https://arxiv.org/abs/2411.18814)独立实现设计，不能将全库称Apache2，不能把简单RRF融合称为LIGER复现。|
|[Kuaishou OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)|官方团队后续Foundation/RecIF-Bench项目，与[2025 OneRec论文](https://arxiv.org/abs/2502.18965)不是同一版实验配方。[tokenizer说明](https://raw.githubusercontent.com/Kuaishou-OneRec/OpenOneRec/main/tokenizer/README.md)要求新数据接foundation模型采用对应Qwen3-8B-Embedding流程；主README有1.7B/8B及多阶段训练。|学习模块边界和远期路线；不能把VK官方64D embedding直接接上其预训练token体系。代码README声明Apache2，实际根许可文件本轮未核，不能视为完成引入审计。OneRec/RL不进入本轮完成范围。|
|[HF rqvae-amazon-beauty](https://huggingface.co/edobotta/rqvae-amazon-beauty)|[model card](https://huggingface.co/edobotta/rqvae-amazon-beauty/blob/main/README.md)是Botta实现产物，非Google官方权重；[config](https://huggingface.co/edobotta/rqvae-amazon-beauty/blob/main/config.json)输入768D，3层256码本。card未标明确license，本轮未取得权重许可文件。|学习config/model card发布形式。Beauty文本向量与VK视频向量不同，不直接使用权重；GitHub MIT不能自动推定独立HF权重许可。|
|[KakaoBrain rq-vae-transformer](https://github.com/kakaobrain/rq-vae-transformer)|RQ-VAE原作者图像生成代码；[LICENSE](https://raw.githubusercontent.com/kakaobrain/rq-vae-transformer/main/LICENSE)区分源码Apache2和stage2 weights CC-BY-NC-SA4.0。|补充residual quantization原理；不是next-item推荐工程，不搬图像encoder/checkpoint与训练预算。|
|[Meta Generative Recommenders](https://github.com/meta-recsys/generative-recommenders)|官方HSTU/M-FALCON研究代码；[README](https://raw.githubusercontent.com/meta-recsys/generative-recommenders/main/README.md)及[LICENSE](https://raw.githubusercontent.com/meta-recsys/generative-recommenders/main/LICENSE)可核，源码Apache2。|学习序列control和系统测量；HSTU不等于SID自回归召回。本轮不新增HSTU、TorchRec或专用CUDA kernel依赖。|

Kaggle：Sol以generative retrieval recommendation、TIGER semantic、RQ-VAE recommendation检索code页面，本轮未找到足够可信可核验的SID召回项目。这不是断言平台没有相关项目；不为凑平台数量推荐图像生成notebook。

网页main是本次读取快照，尚未冻结外部commit。将来如拟引入源码/依赖，必须先固定revision并逐文件核provenance、license和依赖；本次只允许学习设计。上述许可信息是观察到的文件/声明，不代表已批准复制或完成完整license审计。

## 对现有项目的判断

本地来源为G0工作区的docs/g0_validation.md、src/vkgr/tiger.py、sid.py、generative_foundation.py及P4D工作区docs/data_contract.md。Astra另外读取了实际接口，结论如下：

- 已有SIDTokens.batch的teacher forcing、TigerModel的padding/causal mask、合法SID路径解码和item指标；不用再造这些基础组件。
- G0仅synthetic两步CPU训练，尚未正式验收。缺新协议真实runner、保存/恢复、full eligible validation、有限调参、多seed及融合证据。
- 现有balanced hierarchical k-means是原创工程baseline，不是RQ-VAE；旧P5未全部收敛。可逆性和合法性不等于语义质量。应先核占用、suffix负担、收敛与train-only内容结构，再决定是否更换tokenizer。
- SID token CE不是双塔in-batch contrastive loss。共同输入/评估必须一致，但不能机械照搬negative mask或temperature。
- 当前beam只保证合法候选；不能冒称full-catalog exact最优排名。模型质量与search截断损失应分开；beam宽度敏感性仅提供可观察诊断，没有exhaustive评分oracle不能完整归因。
- test26历史audit已被读取；本轮冻结后一次test只说明新方案未再用它调参，不能恢复pristine holdout资格。

## 取舍建议（尚未批准）

优先完成小型TIGER-style + 现有SID的可复现实验及固定预算融合。增加随机SID分配control，帮助判断语义地址是否有价值；它是额外消融预算，应显式列出。RQ-VAE作为后续有界改进候选，不与baseline同时大规模搜索。先不做OneRec的session生成、reward model、DPO/RL、扩数据、SID动态更新或线上部署。

“完成”指工程、实验和证据闭环，不要求生成式必须赢Popularity。若单路和融合均没有可信收益，应交付负结果及失败定位，不无上限追加试验。
