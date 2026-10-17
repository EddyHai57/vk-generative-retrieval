# 可学习 SID（S2）立卡草稿：固定共现信号，学习离散 tokenizer

密级：公开。状态：DRAFT / DESIGN_REVIEW_REQUIRED；**不冻结、不实现、不拟合、不训练、不入队**。在双塔与生成式融合（G5）草稿之后起草。授权源decision_log「2026-10-17 晚间：新SID的A1确认」§5.2；旧pipeline §4只作路线，不当作本轮完整冻结规则或新增训练名额批准。

## 1. 问题与固定对照

问题：将已验证有用的C1′共现信号用于可学习tokenizer，是否比当前固定BHKM SID更适合A1生成模型？当前G anchor固定A1+共现向量SID，bundle COLL_C1_B256D3_S42；三seed完整已验收身份及指标见experiment_log 19:00最终验收与晚间anchor更新，不以旧A1+T3作为主对照，不重跑对照。

唯一允许变化的模型部件是**item→SID的tokenizer及其必要输入身份登记**。下游A1的history/features/catalog/query/split、初始化、优化器/预算/selection、beam100与温度1、evaluation/verify均沿当前anchor逐字段固定；不混入R1温度损失、DPO、多兴趣、特征、attention改动或新seed。tokenizer方法本身会同时改变量化和正则，不声称分离出某一项损失的因果贡献。

共现监督固定复用已验收C1′向量/内容回退mask及其SHA，不再训练SGNS，不读P checkpoint、不导出用户向量。已验收内容向量只作同源内容输入，不拼接新P向量，不变更C2。训练仅0–24周，validation第25周不进入tokenizer目标、负采样或checkpoint选择；test第26周不读。

## 2. 候选方法与建议（不是两臂批准）

| 候选 | 项目内拟议适配 | 取舍 |
|---|---|---|
| LETTER式离线学习tokenizer（建议优先） | 在已修RQ-VAE L2基础上保留内容重建/量化，用固定C1′向量作协同对齐目标，另加码多样性项；拟合后冻结SID，再从头训练相同A1 | 新训练只集中tokenizer模块，主对照易定义；仍无法把量化器改变与正则各自贡献分开。C1′是监督来源，不宣称原样复现作者LightGCN输入 |
| ETEGRec式端到端或交替学习 | 将tokenization与推荐目标协同训练，候选ID变化时同步更新历史token与trie及其验证 | 与“仅替换冻结SID”的A1合同相距更大；训练阶段与缓存失效更复杂、成本未测。若采用需另行明确设计，不能沿本草稿直接实现 |

请求Kiro选**一个**主方案；另一方案留候选清单，不自动新增臂/名额。建议先LETTER式离线方案，避免在同一实验同时改变A1优化协议。该建议仅设计判断，不是预测收益。

## 3. 文献来源与许可证初审（2026-10-17实际只读网页）

- [LETTER论文](https://arxiv.org/html/2405.07314v3)描述RQ-VAE、协同对齐及多样性正则；[作者仓库](https://github.com/HonghuiBao2000/LETTER)公开了实现。当前浏览的根目录/README未发现明确代码许可证；master/LICENSE返回404。**未核递归全部子目录、未钉仓库commit，许可证兼容性UNKNOWN，禁止复制源码/配置/权重。** 论文页面的传播许可不能替代软件许可证。
- [ETEGRec论文](https://arxiv.org/abs/2409.05546)将tokenization与生成推荐联训；[作者仓库](https://github.com/BishopLiu/ETEGRec)根目录/README未见明确LICENSE，所试main/LICENSE返回404；默认分支与逐文件许可尚未完整核查，兼容性UNKNOWN，同样不复制。

这是**初审而非“已通过license gate”**，404不等于证明全库无许可证。冻结前补精确commit/tree与逐文件来源审计；若没有明确兼容许可，仅基于论文公式在本项目原创实现、保留引用，不逐行翻写官方代码。引用论文不意味着使用原仓库数据/实现。

## 4. 待Kiro写死的科学合同

以下缺一不可，不能由执行者在看结果后补定：
1. 选LETTER离线还是ETEGRec；LETTER适配使用CONTENT_V1作重建输入、C1′64维作协同目标，还是直接量化C1′，请明确择一。本稿建议前者贴近旧pipeline §4，同时承认相对BHKM-C1′属于完整tokenizer替换，不是单一对齐损失ablation。
2. 语义层3、每层256和确定性suffix的合同是否保留；输入/latent维度、encoder/decoder、初始化、死码重置是否逐项沿L2已修版。全部写入新schema/版本，不冒用旧bundle。
3. 重建/commitment/协同对比/多样性损失的精确定义、各权重、归一化、负例来源、温度；仅一个先验取值，无扫描。不得拿R1结果为理由重做温度训练。
4. tokenizer fit seed、optimizer/LR/batch、固定epoch或train-only停止规则与峰值/墙钟cap；不足时STOP，不追加attempt或按validation重选fit。
5. 数据hash、C1′内容回退项如何参与对齐/是否mask；零向量/非有限的处理。不得把回退内容当成原生共现监督；继承D-1偏离。
6. H1–H6/R1–R4按旧GTOK-01原文门槛，及新输入族H5a/H5b作用对象由Kiro在冻结前明确；不私自放宽门禁，若确要改已登记门槛需Eddy批准。fit有自身verify、同seed replay；下游三配置真实PF/CUDA实测/dry/verify不得省略。
7. G seed集合、是否三seed无条件运行或s42筛选再补seed、精确对照和UP/DOWN/FLAT/有效门槛。**当前未冻结这些数值规则**；不把旧T3的254、S1的260筛选阈值移用到新anchor，也不擅自写388+6为门槛。
8. 新SID必须同原S0函数重算A–D/D′、热度、in_catalog/OOV和prefix报告；不能用SID级准确率替代item召回。所有verify stdout含run_id+manifest_sha256，旧fixture mismatch仍OPEN，不能靠改测试合同绕过真实PF。

## 5. 训练名额与成本：请求，不是已获本轮批准

建议预算供Kiro/Eddy决策：**一个tokenizer方法配置 + 一个拟合/replay审计 + 最多三个A1训练及各verify**，不另训新对照、不做两个候选方法同时比较。具体是否先s42和补seed必须按§4.7冻结，未冻结前不建正式run_id或驱动。

旧pipeline曾登记S2一个core03名额；本轮目标换为对当前A1 anchor的确认设计，不能把旧“一个名额”自行解释成多个A1或端到端训练许可，必须Eddy明确确认本轮准确范围。

成本粗估：A1一个train按授权参考1.5–3小时，三train串行4.5–9小时；本次已验收S5实测2.59–2.89小时/seed，可作参考但不是新tokenizer的承诺。verify另加约7–8分钟/seed；tokenizer fit/审计、工程前置、端到端额外轮数与并发效率均UNKNOWN。总GPU小时不是并行墙钟；不承诺一晚全部完成。显存声明须各新入口真实shape实测×1.15 ceil256，不继承6656。若选ETEGRec，以上仅下游A1部分，完整预算重新估算后再请批。

## 6. 交Kiro的设计评审问题与停止点

请按§4逐项冻结或明确退回原因，设计评审最多两轮；先完成G5卡的裁决，再评审本卡。请同时确认原路线/名额的适用范围，列出需要Eddy批准的具体训练数量。未拿到Kiro定稿和Eddy名额前仅保持DRAFT，不实现、不拟合、不运行健康检查/吞吐探针、不读新checkpoint、不入队。出现设计偏离、共同合同失效或NaN/Inf按AGENTS停止；没有为了填满服务器自行加实验的后继。
