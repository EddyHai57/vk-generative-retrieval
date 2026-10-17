密级：公开

# 项目地图（一页看全；Kiro 2026-10-17 初版，之后由 Codex 每次更新 STATE 时同步维护）

> 维护规则：任务状态变化（开始、入队、完成、失败、裁决）时更新本页对应行；只写状态和指针，**不写指标数字**（数字只在 `docs/experiment_log.md`），决定只在 `docs/decision_log.md`。任务一律用描述性名称，原编号放括号（对照表见 decision_log「2026-10-17：任务改用描述性名称」）。状态四种：✅ 已完成　▶ 服务器上在跑　📝 已安排、本地在做　⏳ 未开始。

## 项目只回答三个问题
1. **双塔（P 线）能做到多好？** 当前 anchor：序列用户塔（M1-mean）。
2. **生成式召回（G 线）能做到多好？** 当前 anchor：A1 + 共现向量 SID（C1′，COLL_C1_B256D3_S42）；A1+T3保留历史对照。
3. **两者合起来能不能更好？** 双塔与生成式融合（G5），两边anchor已定，当前冻结卡待Kiro。
之后的收尾：读一次 test → 搬到全量数据 → 报告与发布。

## 问题 1：双塔
| 任务 | 状态 | 说明 / 指针 |
|---|---|---|
| 负样本 logQ、观看时长/比例特征、新近性特征、温度 0.10 | ✅ | 已进 anchor |
| 温度 0.05 / 0.07 | ✅ | 更差，不采用 |
| 新近性池化、作者+内容只改历史侧、物品热度、物品质量、行为权重 | ✅ | 均未转正 |
| 序列用户塔（M1） | ✅ | 转正为 P anchor（decision_log 2026-10-16 下午） |
| 物品热度+质量+行为权重组合（FEAT-04 组合） | ✅ | 定点review及真实补证已APPROVED；组件可用于叠加，anchor不自动变更，见P1闭环review |
| 多兴趣召回（M2） | ✅ | Kiro上午已裁决收口；完整延迟UNKNOWN不再补；见decision_log |
| 双向序列模型确认（bi 另卡） | ✅ | 不转正、收口；见Kiro傍晚裁决，新seed上bi与mean无可检出差异 |
| 用户画像特征（F7：活跃度、H50 like率；U1不做） | ✅ | Kiro深夜裁决两臂第4行、不转正、收口，P anchor=M1-mean；一次只读排序/档位诊断已验收，机制简报见issue与review简报待Kiro，不新增后继 |
| 序列用户塔 + 组合叠加 | ✅ | 不转正；见Kiro傍晚裁决，P anchor维持M1-mean |
| 特征组合（F8）、请求上下文（F9）、P 线 HSTU | ⏳ | F9 早前审计未通过 |

## 问题 2：生成式召回
| 任务 | 状态 | 说明 / 指针 |
|---|---|---|
| 小模型 TIGER、随机 SID 对照、beam 宽度诊断 | ✅ | SID 语义有用；beam 不是瓶颈 |
| 论文规模 A1、G 线调参、G 线特征 | ✅ | A1 更好；调参与特征在 A1 上无增益 |
| RQ-VAE 塌缩修复、256 叉 BHKM（T3） | ✅ | A1+T3为历史G anchor/对照，现已由共现向量SID取代 |
| SID 分辨率扫描、1024 叉 A1 确认 | ✅ | 小模型上 1024 叉最好；A1 上未复现 |
| SID 结构诊断（S0、D′） | ✅ | 见 experiment_log「SID-FIX S0」 |
| **SID 修复计划**（卡：`docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md`） | | |
| 　协同向量 SID（共现向量 SID、内容加 P 向量 SID）（S1） | 📝 | Kiro已裁决共现向量UP、内容加P向量DOWN；后继先A1确认，可学习SID暂缓，见STATE |
| 　可学习 SID（LETTER 式）（S2） | 📝 | 立卡DRAFT已起草，见2026-10-17-learnable-coll-sid-draft.md；等Kiro设计评审和Eddy名额，不实现/入队 |
| 　SID 深层准确率调研（S3） | ✅ | 两篇指定公开论文已转述成一页，见research/2026-10-17-sid-depth-accuracy-notes.md；不据此起卡 |
| 　SID 全量规模基准（S4） | 📝 | 本机首轮有界合成探针完成；完整拟合成本仍缺证据 |
| 　新 SID 的 A1 确认（S5） | ✅ | 三训练/三原始verify验收通过；Kiro晚间已转为G anchor，前缀补证已齐，机制解读交Kiro |
| 　GTOK-02 报告补证（B256D2 四列 B1） | ✅ | 只读实测已补，保留既有解释性偏离，见experiment_log |
| **G 线模型侧增强**（不在 SID 修复计划内） | | |
| 　温度交叉熵（R1） | 📝 | Kiro已判无效收口，不追加配置；见decision_log午后 |
| 　SFT + 自生成难负样本 DPO（R2） | 📝 | Kiro已判无效收口，不改β/LR/采样；挖掘验收限制保留 |

## 问题 3：融合与收尾
| 任务 | 状态 | 说明 / 指针 |
|---|---|---|
| 双塔与生成式融合（G5） | 📝 | 命中重叠预检已做；新冻结卡DRAFT已起草；无需训练，但P top100/旧checkpoint导出权限待裁决，本轮不入队 |
| 读一次 test | ⏳ | 两条线 anchor 与 G5 都冻结后，需 Eddy 批准 |
| 搬到全量数据（换大盘机器，属于迁移） | ⏳ | 子集阶段收口、读完 test 之后 |
| 报告与发布 | ⏳ | — |

## 治理准备

| 任务 | 状态 | 说明 / 指针 |
|---|---|---|
| AGENTS.md 整理 | ✅ | Kiro有效AGENTS已完整重读；两治理文档原字节发布codex/governance-20261017-public@d528238，远端已核；无新规则冲突，三草稿仍仅参考 |

## 当前执行更新（2026-10-17 15:03+08）

新SID A1三seed已独立确认claimed且无rejected，唯一服务器有限driver存活与绑定已核，自动补各自verify，结果尚未验收。P统计/接合点review/用户塔计时、曲线诊断、两协同SID A–D、两论文摘要均已交付。科学解读留Kiro、anchor不变；不提前跑其它后继。指标和原收据只见experiment_log最终接纳节。

无人值守五项逐项复核完成：既有入队交付与科学结果验收分开，后者仍待原驱动收据。P补证、早停、协同SID分档、两论文均有证据入口，STATE复查表逐项列齐；未重复执行任务或新增实验。

傍晚裁决已同步：叠加、双向确认均不转正，P anchor维持M1-mean；新SID A1由原驱动接续，verify收口后交Kiro。

15:52+08只读进度已同步：新SID A1确认三个seed仍训练中，尚未收口，无失败终态/rejected；原驱动等待成功后补verify。未改anchor、未新增作业。数值与时间/SHA只见experiment_log本次阶段进度节。

16:23+08只读复查：新SID A1确认仍训练中、原driver存活，三个自动verify尚未触发，无失败终态/rejected；不重投、不重启，待原始verify收口后按卡验收交Kiro/Opus。当前数字与SHA/UTC统一见experiment_log最新节。

17:49+08更新：新SID A1自动verify已在两个已完训练上直接确认接续；剩余训练未结束，整批仍未收口，不改anchor。数字与收据只见experiment_log最新节。

19:00+08最终验收：新SID A1确认已收口，原driver verified终态后退出；最终提示词见reviews/2026-10-17-coll-sid-a1-result-review-prompt.md。P/G anchor未变，不启动新实验。

19:24+08：新G anchor三seed前缀聚合已补齐。融合冻结卡和可学习SID立卡草稿均已交付待Kiro，均未冻结/实现/入队；verify身份与旧fixture风险仍OPEN。

## 交接更新（2026-10-18；Kiro 收束，之后由 Claude Code 维护本页）

执行者由 Codex 交接给 Claude Code（Eddy 批准，见 decision_log「2026-10-18：文档收束与交接给 Claude Code」）。交接时服务器只有调度服务在跑、pending 0、无悬空作业。

- 问题 1（双塔）：anchor = 序列用户塔 M1-mean。叠加、双向确认、用户画像特征均已收口不转正；P 线下一步未裁决。
- 问题 2（生成式）：anchor = A1 + 共现向量 SID。排序导向训练两配置收口无效；可学习 SID 停在 DRAFT，等 Kiro 定稿与 Eddy 名额。
- 问题 3（融合）：G5 草稿待冻结；已授权一次只读盘点 P top-100 归档，结果决定是否需要 Eddy 批 checkpoint 读取。
- 治理：AGENTS §1 执行者名已更新；`handoff.md` 已重写为当前快照（原文在 git 历史）；一次性接手说明 `.agents/tasks/claude-code-takeover-20261018.md`。
- 待办风险：工作分支约 71 个提交未推送，公开仓库推送前须按 AGENTS §3 第 3 条扫描全部新增历史。

### 更正（2026-10-18，同日）：Claude Code 未实际接手

Claude Code 额度同样耗尽，交接未发生。现由 **Kiro（Opus / Sonnet）单人兼任执行者与裁决者**，属降级状态，见 decision_log「2026-10-18：Kiro 单人兼任执行者与裁决者」。上一节关于 Claude Code 的执行者指定作废，其余内容（anchor、各线状态、未推送风险）不变。降级期间不读 test、不迁移全量、不改已登记门禁、不新增名额、不删除或覆盖任何产物。
