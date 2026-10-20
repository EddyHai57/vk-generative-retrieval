# ur0.01 全量数据合同细则卡草稿

**状态：DRAFT，待 Kiro 冻结；不能用于派生或训练准入。** 依据 942a5b0 第4条、012643c；沿用子集规则只替换数据源/规模。下文发现 history 文本与已实现子集语义存在差异，明确列为待裁决，不自行选择。

这里的“全量”指 1% 用户的完整交互 `ur0.01`，不是全平台数据。数字/原始收据唯一入口：experiment_log「A二次裁决恢复」「官方week26有界准备」「夜间本地准备：SGNS算术与分叉核算」。本卡不修改 data_contract 或任何冻结卡。

## 数据源与身份替换

|角色|新数据源|约束|
|---|---|---|
|train|固定 revision 的 subsamples/ur0.01/train/week_00–24.parquet|仅此窗口确定 label、统计、catalog、supported 与阈值|
|validation|同 revision 的 subsamples/ur0.01/validation/week_25.parquet|不读 test；共享 P/G query 身份|
|test 资产|官方 interactions/test/week_26.parquet，按 A 授权窗口用户集合过滤的独立 holdout|目前仅下载/过滤/四项 ID 计数授权；**不在阶段2构造 test query/history**|
|metadata|固定 users/items metadata 与 item_embeddings.npz|逐文件 size/LFS/SHA；catalog 按 train-observed，而非 metadata 所有物品|

完整文件身份见 `configs/ur001-preparation/hf-files.json`、`analysis/ur001_week26_bounded.py` 的固定对象与用户集合身份。新实例重做必须与原四项计数逐项相等；不相等 STOP，不修改预期。

## Query：沿用 p3-like-v1 文本

子集事实源 `0ca61bc:docs/data_contract.md`：

> 各evaluation split每user仅一个query：target是该week中首个真正`like=True`的item。不能先过滤OOV再选target，也不能换成首个可达like。

> 没有positive target的train users、target-positive但history不足的users与train未观测target users分别报告。history不足者不进入主eligible query集合；有history的首target即使OOV也保留并计入all-eligible分母。

依 942a5b0 第4条，validation/test 分别取该周每合格用户首个 like，**合格条件为 train-only like 历史 ≥1**；target 身份先确定再标 in-catalog/OOV。HIST_V2_OBS overlay 不改变 query 的 user/target/target_position/oov。候选计数只用 A 已验收结构表，不在本夜生成 query。

## History：已冻结原则与需要 Kiro 确认的差异

既有 `HIST_V2_OBS` 定义（`0ca61bc:src/vkgr/history_v2.py` 模块说明）：

> Same event definition as P4E `retrieval_watch.build_watch` (every train interaction row,

train 样本采用 strict-past observed 最近50条，保留原顺序；先输出该 target 的历史，再 append 该事件；不去重、不 collapse 连续重复。query 身份沿用原 like query。

942a5b0 第4条要求：

> history：HIST_V2_OBS，strict-past observed 最近 50 条，不改变 query 身份。test query 的 history 允许包含 week 00–25 及 week 26 内该 target 之前的事件；validation 同理到 week 25 内。

**待 Kiro 决定（阻断合同冻结，不阻断本夜工程准备）：** 子集 `0ca61bc:docs/data_contract.md` 写“history均固定为train-only最近50个like items…test不加入validation反馈、不rolling追加target week的反馈”；后续 `history_v2.build_observed` 确实把 like 历史替换为 observed，但 validation 仍直接使用 train 结束时的 `watches`，并未遍历 validation 周前缀。942a5b0 的“逐条沿用子集”与“允许同周 target 前事件/跨 validation 的 test history”不能仅靠换数据源同时实现。请明确后者是否构成显式覆盖；在裁决前不编写/运行这部分派生，不把两种历史混称已等价。

## Train 样本：沿用文本与 overlay

子集 `0ca61bc:docs/data_contract.md`：

> 每个train positive（有至少一个更早positive时）形成一个prefix→next-positive训练例；prefix只含该target之前的positive，截断为最近50个，按原序排列。

> 不去重positive history，且不做consecutive-repeat collapse；若将来增加消融，须另立version而不改当前结果。

按 HIST_V2_OBS 后续版本，**target 集合不变**：train 每个 like、每用户首个 like 除外；history 用该 target 前 observed 最近50条。每条样本须绑定 example_id/user/target/target_position/history_positions/past_likes；保留原 like prefix 的身份核对后再 overlay，不能把每次曝光变成训练目标。实际样本数由正式派生守恒核对，不能把 Kiro 粗估当准入实测。

## Catalog / label / supported

- Catalog = train-observed week00–24 全部物品，排序与 ID/PAD/OOV 映射沿子集原实现；validation/test 都用同一 catalog。模型不在 validation 上重训后再测 test。
- Label = like；`click_on_author` 不解释成普通 click，曝光不解释成正反馈，不报告 CTR。
- P supported = full-train likes 中出现的物品（原 `retrieval_experiment` 的 `id_support='full-train likes; unsupported output fixed zero'`）；unsupported 不删除，OOV 单列；不按观察历史扩大有训练 ID 向量的集合。
- 分档、阈值、supported、任何看后筛选都只用 train；P/G 共享 history/features/catalog/query/split 与调参预算。

## S0 热度分档：原定义在新 train 重算

原 `generative_evaluation` 协议文本：`nearest_rank_positive_item_event_counts_median_p90`、`train_like_counts_only_no_holdout_boundaries`。

对 train catalog 统计 like 事件数，仅对正计数物品的 count 排序，median/p90 分别 `sorted[ceil(q*n)-1]`；物品档沿 `unsupported0`、`positive_le_median`、`positive_gt_median_le_p90`、`positive_gt_p90`，OOV 独立。用户 bucket 边界沿 `[1,5,20]`。不沿用子集阈值数值，不改为 np.quantile 线性插值。

## 观看正例阈值：保留定义，独立实验才启用

沿 `2026-10-20-p-watch-positive-card.md` 原文：

> train 周 00–24 的 observed 事件，同时满足
> (a) 不是 like 事件（like 已是目标）；
> (b) 不是 dislike 事件；
> (c) watch ratio ≥ `t_b`，`t_b` 为该事件所在既有时长桶内、全部 train observed 事件 ratio 的 80 分位（只用 train 计算）；
> (d) 目标物品在 P 当前 supported 集合内。

`t_b = sorted_b[ceil(0.8*n_b)-1]`，在(a)–(d)筛选前用全部 train observed 事件、一事件一个值、保留重复；ratio/dtype 沿旧特征代码、不加 epsilon，ties 全收，空桶不产阈值/样本。非有限 ratio 或时长≤0 不入统计/池，超过 train observed 的1% STOP。沿已修订空 observed history 排除并报告的规则。**不把观看正例加进首批 like anchor**；实际重算与混合/采样/logQ 按后续独立实验卡，不因本草稿自动启用。

## 顺序、重复与待裁决清单

已知物理顺序 proxy 为 `(week, physical_row_ordinal)`；A 的重复 pair/null 检查通过是已验收事实，不等于逐事件 timestamp。history 严格早于 target；无顺序证据 STOP。

|待 Kiro 决定|原因|冻结前要求|
|---|---|---|
|validation/test history 是否显式覆盖子集 train-only 行为|上文列出实际实现与新裁决的差异|明确覆盖原条款及独立版本身份；未决定不生成|
|同周多个文件/row-group 跨文件 physical ordinal|现子集每周单文件；新并行读取不能暗改顺序|冻结文件遍历与ordinal定义；不能以 worker 完成序排序|
|同周多个 like 的首条/完全相同位置如何判定|现无timestamp，仅物理位置|沿物理proxy是否足够，异常一律 STOP，不自写 tie-break|
|新数据源 manifest DAG / 版本命名 / new mapping 绑定|不可冒用子集 schema/source身份|只新版本；字段、完整输入闭包由卡冻结|
|全量 SID branching|下表仅算术，未选择|Kiro 写死 K 后才能起 BHKM 实现卡|

## 附：同利用率规则的分叉算术（只报数，不选 K）

来源：A 已验收 train catalog `N=12,012,895`；原 48³ 卡 `k=min(branching,len)` 与原 `sid.py::_assign` capacity `n//k`，前 `n%k` 个 child 加1。逐层传播这两个整数 child size，occupied/full slots 为利用率（同原 audit 的 prefix occupancy 规则）。`N^(1/3)=229.0248252105078`。

|K（depth=3）|K³|第1层|第2层|第3层|N/K³（非截断）|
|---|---:|---:|---:|---:|---:|
|228|11,852,352|100%|100%|100%|约1.01355|
|229|12,008,989|100%|100%|100%|约1.00033|
|230|12,167,000|100%|100%|98.7334182625%|0.9873341826251335|
|256|16,777,216|100%|100%|71.6024339199%|0.7160243391990662|

原 ≥80% 门禁保持，算术不替代真实 H/R fit/replay 门禁；不据此自动选229或把通过利用率当 anchor。完整可复算结果与资源下界见 experiment_log 的夜间算术节。
