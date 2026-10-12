# P/G 召回研究路线 — 从可解释对照到联合模型

密级：公开。日期：2026-10-12。

Eddy已认可聊天中的研究路线，并要求本地落盘、Git保存、随后在Plan模式逐阶段细化和执行。本文件保存研究设计，不代表新增GPU预算或已实现模型。第一阶段见[执行计划](../plans/2026-10-12-pg-phase01-baseline-reception.md)。当前运行状态仍以项目STATE.md为准；现有冻结任务仍按tasks.md及原协议执行。

## 目标与成功标准

研究哪些用户历史信息、物品表示、序列架构和生成方式改善召回，以及收益与成本的来源。项目不以必胜或完成旧阶段清单为目标；可复核负结果也是成果。探索性分支可以替换后续优先级，但不能据此追改已有科学合同或重置支出。

HSTU是可训练的序列编码架构，不是embedding查表或单纯数据预处理。可解释性通过明确对照、消融、分组及案例支持；attention权重不是因果解释，复杂程度也不严格等于解释难度。

## 已核事实与授权边界

- 实际interaction没有timestamp；顺序来自week与原文件物理行号。行号差不是时间差或单用户行为次数。可以构造用户行为距离、历史周间隔，不能声称小时、真实session或精确事件间隔。
- timespent是观看秒数，uint8上限255；duration在item metadata。历史反馈可作新特征候选，但先验证单位、缺失、异常及可用时点。
- 当前P4H为ID-only SASRec-style：64维、1层、4heads、FF256、dropout0.1、最近50行为、causal、last-valid pooling；共享可训练item表，非原版SASRec完整方法复现。P新训练尚未准入。
- 当前G是balanced hierarchical k-means SID加small encoder-decoder，非完整论文TIGER。原四个seed42配置中core00已有指标摘要；core01/02/03有2026-10-12 07:50:42北京时间ALL_TERMINAL收据，exit0且owned groups关闭，但模型成果仍待接收验收。终态不等于质量验收或新资源permit。
- P原pilot 2/2attempt保守spent，原deadline已过；最新63项本地测试与UID只读probe不等于完整publisher准入。新增attempt必须取得明确有限包络，保留旧spent。Formal未启动，不继承过期pilot窗口。
- 本轮授权本地计划文件及其commit；不push。生产/服务器执行仍由STATE登记的唯一owner负责。规划对话不并行改生产、monitor或STATE/handoff；owner变更先做安全交接。
- 本次context实时计数不可用，记UNKNOWN，不估算比例或复用旧样本。

## 阶段与研究问题

| 阶段ID | P线 | G线 | 要回答的问题 / 进入下一阶段的证据 |
|---|---|---|---|
| PG-R01 | 对账既有结果、P失败ledger与有限恢复包络，原基线恢复按原任务执行 | 接收原四配置产物；按冻结规则selection；补seed按原授权与准入处理 | 基线身份、分母、指标、成本和缺口可核；缺口不伪称完成 |
| PG-R02 | 等权历史聚合vs固定行为距离衰减；Transformer位置-only vs距离/周间隔 | SID与decoder固定，比较历史距离/周间隔特征 | 新近程度是否有价值；固定规则vs学习关系。固定衰减强度在本阶段预注册，不现场追加搜索 |
| PG-R03 | like history→observed ID history→历史反馈→观看强度，分别添加 | 使用同版历史变体，保持目标和query不变 | 收益来自历史更多还是行为质量；like-only里新增like标记无信息 |
| PG-R04-P | ID/content融合；近期与长期分支分别对照 | 保留稳定G参照 | 哪些支持度/历史长度用户受益；不一次合并两种改动 |
| PG-R04-G | 保留稳定P参照 | 论文方法对齐TIGER：RQ-VAE、配置差异表；当前SID/RQ-VAE/随机SID对照 | 编号语义、可达性、碰撞、suffix负担与召回贡献；VK方法复现不冒称原论文数值复现 |
| PG-R05 | 同输入/输出/目标下SASRec-style vs HSTU | 同SID/decoder/目标下Transformer encoder vs HSTU encoder | 架构贡献与成本；首轮不加特征、不延长历史。混合模型不称原版HSTU复现 |
| PG-R06 | 作为向量参照与潜在teacher，不自动同时升级 | 根据瓶颈选择LETTER、ETEGRec或COBRA中的一个具体问题 | 协同tokenizer、联合优化或稀疏/稠密融合是否解决已观察问题 |
| PG-R07 | 与G共享新增研究设计 | 自研有依据的变体 | 文献缺口、机制假设、强基线与消融共同支持；组合已有模块不自动构成创新 |

阶段是研究优先级，不是全部必跑清单。PG-R04-P/G分别立具体任务；每次只执行一个tasks.md登记的task ID。G原random control/beam诊断可提供现成证据，登记来源后复用，不重复训练。G融合与全量扩展不因旧路线自动抢占当前研究优先级；改变具体已冻结执行次序需唯一owner记录。

## 公平比较和特征时点

1. 不重写原label/history/catalog/query/split；新特征与新history用独立版本。主label仍like，train00–24/validation25/test26，保留all-eligible含OOV分母。
2. 每组架构比较同输入；新增特征组双方可用性明确。P ID-only与G内容SID不是同features，分别声明；补充matched-feature对照才讨论架构贡献。
3. 历史年龄锚定预测时已知cutoff或最后已观察事件；训练/推理定义一致。不使用未来target精确位置/隐藏时间/反馈、不从timespent拼出伪timestamp。
4. watched历史新增反馈来自已完成历史事件；不使用待预测item反馈。全train popularity/rank不能冒充早期prefix可用特征。
5. ratio需duration>0及缺失标记、train确定变换；255上限作为限制。主label不随观看阈值重选。
6. RQ-VAE、协同tokenizer及teacher只用允许的train信息；映射/版本冻结后再评估，不借未来内容可用性扩原catalog。

## 实验卡与推进标准

每个具体实验开始前冻结：研究问题、对照、唯一改变量、数据/特征版本、配置枚举、seed、输入SHA、选择规则、成本预测、允许attempt/绝对deadline/disk cap与失败处理。seed42用于预注册筛查；主张稳定收益的候选确认42/2024/2025全值及mean/sample std，不挑best seed。不同方法调参次数明确；等次数不称等compute。

记录item Recall/NDCG@10/20/50、all/in-catalog/OOV、train-defined热门/长尾/支持度/历史长度分组；召回覆盖、候选数/非法候选、train/decode/verify wall time、P50/P95及吞吐、RAM/VRAM/disk。至少给出改善、退化和无变化的匿名案例；逐query原数据留ignored服务器目录。

固定衰减是透明参照，不保证胜出；HSTU长序列加速不能外推当前50物品历史。负结果记录机制与成本，不追加无界HPO。只有结果、机制或成本改善值得投入时进入更复杂阶段。

validation支持阶段选择，但反复适配会过拟合，累计实验全部登记。test26已有暴露范围，不能重新声称pristine holdout；后续可信最终验证在具体阶段先设计额外数据/时间窗口或外部benchmark，不未经授权下载。

## 实现与运行纪律

原代码和科学包不覆盖；新增生产代码fresh RED→GREEN→相关回归，完整交付一次fresh-context review，接受过的旧review/全suite不重复。唯一job owner与终态receiver；服务器确定性guard负责实时安全，模型仅任务节点/原轻量兜底，不新增长期agent或timer。

原支出不重置；服务器资源余量至少20%。新阶段预算在实测后提出有限数字，human grant与runtime pin先于submit。无完整validation/verifier成本就不准入。规划认可不等于新付费资源、生命周期操作、训练重试、全量下载、push或发布授权。

## 文献入口

- [TIGER](https://arxiv.org/abs/2305.05065)：RQ-VAE SID＋自回归生成。
- [TiSASRec作者代码](https://github.com/JiachengLi1995/TiSASRec)：位置和时间关系，周粒度改编须声明。
- [HSTU论文](https://arxiv.org/abs/2402.17152) / [官方实现](https://github.com/meta-recsys/generative-recommenders)：行为序列架构，非单纯SID生成。
- [LETTER](https://arxiv.org/abs/2405.07314) / [作者代码](https://github.com/HonghuiBao2000/LETTER)：语义、协同与编号多样性。
- [ETEGRec](https://arxiv.org/abs/2409.05546)：tokenizer与推荐模型联合优化。
- [COBRA](https://arxiv.org/abs/2503.02453)：离散SID与连续向量级联。

本文引用用于研究方向，不构成仓库license审计；使用第三方代码前核许可，不复制无兼容许可实现。
