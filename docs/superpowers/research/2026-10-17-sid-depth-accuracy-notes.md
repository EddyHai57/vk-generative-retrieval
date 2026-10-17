密级：公开

# SID 深层准确率：两篇公开论文阅读摘要（S3）

阅读日期：2026-10-17；context：UNKNOWN。依据 decision_log「午后：无人值守授权」§2.5，仅转述指定论文；下述均为论文作者的发现，不是本项目已证实的结果，不构成适用性、anchor 或训练决策。

| 核实的论文身份（均为 v1） | 作者与首次提交日期 | 主要支持位置 |
|---|---|---|
| [LLMs Need Encoders for Semantic IDs Too](https://arxiv.org/abs/2606.00324v1) | Xiangyi Chen、Zelun Wang、Xinyi Li、Yi-Ping Hsu、Jaewon Yang、Jiajing Xu；2026-05-29 | [全文](https://arxiv.org/html/2606.00324v1)：§3–6，表2–4、7–8 |
| [From a Static Multi-Level Small Semantic Codebook to a Dynamic Single-Level Large Semantic Codebook for Generative Recommendation](https://arxiv.org/abs/2608.21012v1) | Tianlu Xie、Xin Ku、Mingjie Sun、Yunhao Sha、Lixiang Wang、Peng Wang、Yiyu Wang、Wenjin Wu、Zhaojie Liu、Peng Jiang、Wenwu Ou；2026-08-21 | [全文](https://arxiv.org/html/2608.21012v1)：§1、3–4，图1，表1–4，附录B |

**前缀记忆。** 第一篇在 SID token 输入处加入 PrefixMem：哈希前缀，查表后投影并加到 embedding。设置为 RQ-VAE 的五层、每层2048码；约1000万训练子序列、每条至多32次交互，约2.4亿 item occurrences，训练与评估按时间和用户隔离。默认 Qwen3 1.7B，训练50K步，编码器用于 L4/L5；2M 是哈希表行数，不是 SID 码本宽度。[§3、§4.1–4.2](https://arxiv.org/html/2606.00324v1#S4)

分母为10万 held-out examples，每例随机选一个目标位置。TF-L5 给定正确 L1–L4，只查第五码 top1：37.6%→54.8%（2M随机初始化，约+46%相对），L4 为33.3%→42.6%。增大哈希表至5M，TF-L5为57.2%；分类预训练5M为64.0%。这些数不表示完整 SID 命中。[§4.3、§5.1–5.2，表2–3](https://arxiv.org/html/2606.00324v1#S5)

Full-SID Recall 用自回归 beam，完整五码须同时匹配；beam100时9.5%→11.3%（2M）或11.6%（5M）。论文明确 SID 可被多个 item 共享；碰撞修正后的 item recall 为 UNKNOWN。表7的“难例”按 TF 下 L1–L4 是否全部进入各层top10划分，并非实际 beam 可达性；表8按 L4 前缀频次划分，不能直接称 item 长尾。作者在§6称收益依赖前缀结构及足够深度；本项目适用性未判。[§4.3、§5.3–5.4、§6，表4、7–8](https://arxiv.org/html/2606.00324v1#S5.S3)

**减少语义层。** 第二篇的“单层”保留消歧 token，实际从三个 token 缩到两个。公开数据对比为[256,256,256]→[1024,512]，末层均消歧，层数与词表宽度同时改变。图1的2.48%是每个活跃 SID1 下出现的 SID2 种类占 SID2 词表的平均比例，不是预测准确率。[§1图1、§3.1–3.2、§4.1](https://arxiv.org/html/2608.21012v1#S4.S1)

作者在 Beauty、KuaiRec 上按时间排序、leave-two-out，三seed {7,42,2024}配对；报告 Recall/NDCG@5/10。Beauty 的 TIGER Recall@10 反而由0.0520降至0.0463，不能概括为所有架构都受益。动态更新另用固定日期划分，不能与静态表混算。逐层 TF 准确率、公开评估的固定 beam 宽度、碰撞后 item 映射规则及过滤后的准确评估例数均 UNKNOWN；表1用户总数不能直接代替分母。[§4.1–4.3，表1–4；附录B](https://arxiv.org/html/2608.21012v1#S4)

第一篇的“轻量”指计算开销：默认稀疏哈希表约20亿参数，表及优化器状态增加约10GB峰值显存。表中相同联合训练步数不等于预训练成本为零；分类预训练最高准确率与最佳检索召回也来自不同配置。[§5.2表3、§6开销分析](https://arxiv.org/html/2606.00324v1#S6)

第二篇用曝光加权聚类，动态部分再加入时间衰减、中心移动平均及改码惩罚，消歧码保持固定。其碰撞诊断分母有两种：已占用完整SID数，或评估物品总数；二者都不同于推荐命中率。重建余弦、码本利用率和聚类负载只用于筛查，作者明确仍需下游推荐实验确认效果。[§3.3–3.5，公式18](https://arxiv.org/html/2608.21012v1#S3.S5)

范围限制：已核 arXiv 身份页及上述官方全文位置；未读作者代码、未复现，未访问本项目服务器、数据、checkpoint 或 test。未据论文提出新卡、训练或门槛。
