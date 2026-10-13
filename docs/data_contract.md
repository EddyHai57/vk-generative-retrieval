密级: 公开

# Data Contract

## 当前边界

本合同是 Draft，必须在只查看 train 数据后冻结 label 与 task policy。

P2已完成服务器端MVP下载与完整性验收；P3正在核验schema、规模与语义。本合同仍为Draft，未训练模型、未验证效果。当前执行状态以STATE.md为准。

公司内部信息、数据、schema、代码与实验结果不得进入本项目。

## 证据分层

### A. 官方公开事实

VK-LSVD data card 描述 27 周、按 week 组织的公开序列数据。

公开数据提供多反馈字段，而不是统一的普通 click label。

`click_on_author` 表示进入作者页，不解释为点击视频或普通 click。

可用反馈包括 `like`、`dislike`、`share`、`click_on_author` 与观看相关信号；实际列名、dtype、缺失率必须下载后复核。

公开 item content 表示为 64D float16 embedding。

该 embedding 是官方预计算表示，不等同于本项目训练的 raw multimodal encoder。

公开记录确认week partition；文件内顺序、排序键与稳定性必须下载后复核。

当前没有足够依据声称逐事件精确timestamp。

数据卡公开的全量量级约为 10M users、19.63M videos、40.77B exposures。

### B. 容量推算

`ur0.01_ir0.01` 的调研记录为 90,178 users、125,018 items、4,044,900 interactions，拟作为 MVP；下载后必须用 manifest 重算。

`ur0.01` 的调研记录为 100,000 users、12,467,302 items、407,854,360 interactions，拟作为 flagship scale audit；下载后必须用 manifest 重算。

MVP 全 user-item float32 score matrix 约 42 GiB，因此禁止整体物化。

Flagship 全 user-item float32 score matrix 约 4.54 TiB，因此必须使用 query/item blocking、running top-k 或 ANN。

Flagship 12.47M × 64D content embedding 约占 1.49 GiB float16 或 2.97 GiB float32，未计 index、metadata 与运行时副本。

规模数字是规划输入，不是本轮实测结果。

### C. 待数据探查确认

确认公开文件的准确 schema、dtype、partition、排序键与空值约定。

确认 `ur0.01_ir0.01` 与 `ur0.01` 的实际行数、去重后 user/item 数与 week 覆盖。

确认 `ir0.01` 的生成规则与是否改变 item popularity、feedback 或 catalog coverage。

确认每个反馈字段的取值、共现、覆盖率与跨 week 稳定性。

确认 `timespent` 的聚合粒度、单位、截断、缺失值与时序语义。

确认 `duration` 的单位、零值、异常值与版本稳定性。

确认同一 user-item-week 的重复记录表示重复 exposure、聚合结果还是数据重复。

确认 `ur0.01` 是否没有独立官方 test split。

## Event 语义合同

每行先保留官方 event/feedback 原义，不统一改名为 click。

`like` 只表示 like。

`dislike` 只表示 dislike。

`share` 只表示 share。

`click_on_author` 只表示进入作者页。

`timespent` 是观看相关观测值，不自动视为正反馈。

`duration` 是内容时长或相关属性，不自动视为用户反馈。

没有官方逐事件 timestamp 时，只使用 week 与已验证的文件内相对顺序。

first-seen week 只能称 observed-item proxy，不称真实 publish 或 listing time。

本项目不构造或报告 CTR。

## Label 合同（Draft）

第一步只读取 train，统计 `like` 的 user coverage、item coverage、event count、每 user 序列长度与 week 稳定性。

若 `like` 覆盖足够，主正反馈固定为 `like`。

若 `like` 覆盖不足，才定义 strong-engagement composite。

strong-engagement composite 的字段、阈值、归一化、优先级与冲突处理必须只用 train 决定。

label policy 必须在查看 validation/test 指标前冻结并写入版本化 manifest。

`next exposed item` 禁止作为用户偏好主任务，因为它会主要复刻 logging recommender policy。

负反馈可用于辅助分析或后续消融，不得在未定义因果口径时解释为 preference probability。

## History 与 Target

MVP history 仅包含发生在 target 之前、满足冻结主正反馈规则的 item。

主 target 是 validation/test week 中下一个满足同一正反馈规则的 item。

MVP 每个 user 生成一个 query，避免长用户贡献不成比例的 queries。

history 必须保持已验证的 week 与文件内相对顺序。

跨 week 同 item 是否保留重复，按“重复处理”规则执行。

没有符合 target 的 user 不进入主 query 集，但必须报告过滤数量与原因。

## Split

所有 split 按 week 前后顺序构造，不做随机 interaction split。

MVP 优先使用 `ur0.01_ir0.01` 的官方 train/test 结构，并在下载后验证。

若需要 validation，只能从 train 尾部 week 划出，不能用 test 调参。

Flagship `ur0.01` 若无官方独立 test，则从 train 锁 validation，并将最终后续 week 作为一次性 holdout。

同一 holdout 不得既用于调参又用于 final 报告。

## Catalog

主候选域为 train-observed catalog。

每个 catalog snapshot 必须记录 cutoff week、item count、feature coverage、SID version 与 hash。

validation/test 中 first-seen target 单独标记为 OOV/first-seen，不静默剔除。

主指标报告 train-observed target slice。

first-seen/OOV slice 单独报告 target count、reachable rate 与 retrieval quality。

first-seen 不等于真实上新，所有结论只称 week-level observed-item dynamics。

## 重复处理

完全相同记录先计数并保留原始计数证据，再决定是否去重。

重复 exposure 不自动去重，因为它可能代表多次展示。

正反馈 history 默认保留按时间顺序发生的重复 item，但同时提供 consecutive-repeat collapse 消融。

同一 target query 只保留一个 target item。

任何去重规则必须在 validation/test 评估前冻结。

## `timespent` / `duration`

先检查 `timespent` 与 `duration` 的单位、缺失、零值与异常值。

如使用观看强度，优先考虑 `timespent / duration`，并在 train 上确定 clipping。

`duration <= 0` 或缺失时不得直接做除法，必须单独标记。

归一化、截断与 missing indicator 必须进入 feature registry。

MVP 主 history 不依赖观看阈值，除非 `like` 覆盖不足且 composite policy 已冻结。

## Feature Registry（Draft）

| Feature | Entity | Source | Availability | Transform | Missing/OOV | MVP role |
|---|---|---|---|---|---|---|
| `item_content_embedding` | item | 官方 content artifact | catalog snapshot | float16 保留；检索前 L2 normalize | missing flag；不得补成真实内容 | SID + content baseline |
| `week` | event | 官方 partition/order | event time proxy | ordinal index | invalid row quarantine | split + dynamics |
| `file_order` | event | 待 schema 验证 | event order proxy | stable ordinal | 无法验证则停用 | within-week ordering |
| `like` | user-item event | 官方 feedback | train/valid/test | binary，待 dtype 核验 | missing 不等于 0，先审计 | Draft primary label |
| `dislike` | user-item event | 官方 feedback | train/valid/test | binary，待 dtype 核验 | missing 单列 | audit/auxiliary |
| `share` | user-item event | 官方 feedback | train/valid/test | binary，待 dtype 核验 | missing 单列 | composite candidate |
| `click_on_author` | user-author event | 官方 feedback | train/valid/test | binary，待 dtype 核验 | missing 单列 | composite candidate，不称 click |
| `timespent` | user-item event | 官方 watch signal | train/valid/test | unit audit + clip | missing indicator | composite candidate/ablation |
| `duration` | item/event | 官方字段，粒度待核验 | catalog/event | unit audit + clip | missing/zero indicator | timespent normalization |
| `item_first_seen_week` | item | 本项目派生 | snapshot-relative | minimum observed week | unknown flag | OOV proxy only |
| `history_length` | query | 本项目派生 | target-prefix only | count | 0-history excluded/reported | slice metric |

## 公开来源

VK-LSVD data card: https://huggingface.co/datasets/deepvk/VK-LSVD
