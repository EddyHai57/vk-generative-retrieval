# VK 项目「站在巨人肩膀上」调研清单（2026-11-04）

目的：重新规划 VK 前，先弄清楚别人已经做了什么、哪些代码能直接拿来用。星数、协议和最近更新时间都是 2026-11-04 从 GitHub API 现查的；论文里的数字标明了出处，没有逐个复核。

## 一、和我们用同一份数据的工作（最该先读）

| 工作 | 要点 | 代码 / 协议 |
|---|---|---|
| VK-LSVD 数据集论文 [arXiv 2602.04567](https://arxiv.org/abs/2602.04567)（WWW'26） | 官方基线只有 Random / Popular / Conversion / iALS（观看超过 10 秒算正例），指标是 NDCG@20、Coverage、AUC，用 ur0.01_ir0.01 子集。每条记录有 `timespent`（封顶 255 秒）、like / dislike / share / bookmark / 点作者 / 开评论；物品有 `duration`、作者、64 维内容向量；没有文本，没有逐条时间戳 | 无官方代码，只有 HF 数据集（Apache-2.0） |
| 可变长度 SID [arXiv 2602.16375](https://arxiv.org/abs/2602.16375)（Khrylchenko） | **直接在 VK-LSVD 上做生成式推荐**：10% 用户、只用 like、约 90 万物品，decoder-only 8 层，Recall@100。据 summary，dVAE 可变长度 SID 比 R-KMeans 高 26.5%（数字待读 PDF 核对） | [varlen_semantic_ids](https://github.com/KhrylchenkoKirill/varlen_semantic_ids)，MIT，23★ |

结论：VK 上已经有现成的生成式推荐参照（第二行），协议是 MIT，代码可以直接拿来用。我们得和它对齐，或者讲清楚为什么不对齐。

## 二、生成式推荐的代码骨架

| 项目 | 用途 | 协议 / 活跃度 |
|---|---|---|
| [GRID](https://github.com/snap-research/GRID)（Snap） | 物品向量 → RQ-KMeans / RQ-VAE → TIGER；Hydra 配置，适合大量做对比实验 | **仅限非商用研究**；764★；2025-10 后未更新 |
| [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) | TIGER 的 PyTorch 实现 | MIT，855★，活跃 |
| [torch-rechub](https://github.com/datawhalechina/torch-rechub) | TIGER 复现，有中文资料 | MIT，1.2k★，活跃 |
| [generative-recommenders](https://github.com/meta-recsys/generative-recommenders)（Meta HSTU） | 长序列用户编码器；我们的历史中位数约 4126 条，现在只用 50 条 | Apache-2.0，2k★，活跃 |
| [ETEGRec](https://github.com/RUCAIBox/ETEGRec) | tokenizer 和推荐模型联合训练，输入是协同向量（不需要文本） | 无协议声明，38★ |
| [LETTER](https://github.com/HonghuiBao2000/LETTER) | SID 加入协同正则 | 无协议声明，195★ |
| [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)（快手） | 基础模型 + RecIF-Bench；依赖 LLM 和文本，留给第二个项目参考 | 仓库无协议声明，模型卡是 Apache-2.0 |
| [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) | SID + SFT + GRPO；第二个项目用 | Apache-2.0，1.9k★ |

没有协议声明的仓库只能参考思路，不能把代码拷进公开仓库（AGENTS §7）。

## 三、观看时长 / 时长偏差

| 工作 | 要点 | 代码 |
|---|---|---|
| D2Q（快手 KDD'22） | 按时长分组 → 组内分位数当标签 | 教材版：RecSys-Industrial-Book `ch23/ch04_classic_methods/d2q.py`（CC BY-NC-SA，只参考） |
| CWM（KDD'24） | 反事实观看时长，去除时长偏差 | [hyz20/cwm](https://github.com/hyz20/cwm) |
| EGMN、DADF | 时长分布建模 / 去偏 | [EGMN](https://github.com/BestActionNow/EGMN)、[DADF](https://github.com/liuzhao09/DADF) |

VK 数据有 `timespent` 和 `duration`（都封顶 255 秒），D2Q 式标签可以直接做。要注意：timespent 是多次观看累加的，可能超过 duration。

## 四、综述与教材

- [What Makes a Good Semantic ID…（arXiv 2609.24430）](https://arxiv.org/abs/2609.24430)：固定 TIGER，只换 SID 方案。结论是没有哪种 SID 方案始终最好。
- [GRID 论文 / Practitioner's Handbook（arXiv 2507.22224）](https://arxiv.org/abs/2507.22224)
- [Awesome-Generative-Recommendation](https://github.com/uestc-huangyw/Awesome-Generative-Recommendation)（MIT，偏工业界）
- [RecSys-Industrial-Book](https://github.com/jhljx/RecSys-Industrial-Book)：第 9 章生成式推荐，第 23 章时长预估（CC BY-NC-SA，代码是教学模拟版）

## 五、建议的阅读顺序

1. 可变长度 SID 论文 + 它的代码：同一份数据，最直接的参照。
2. VK-LSVD 数据集论文：搞清楚字段和官方口径。
3. GRID Handbook + 2609.24430：SID 设计的全局图景。
4. TIGER 原文对照 RQ-VAE-Recommender 的代码。
5. D2Q + CWM：时长去偏。
6. HSTU：长序列，作为后续方向。
