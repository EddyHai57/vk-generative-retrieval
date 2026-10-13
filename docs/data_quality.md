密级: 公开

# P3-C：现有数据质量与清洗决策

现有 train 数据可在既有 official-order assumption 下继续设计 like 召回。没有发现需要删除原始记录的证据；建议保留全部曝光、低活跃用户、长尾 items、显式反馈冲突与时长异常 flag。没有实施过滤或改 label。正式新训练评估合同由 P3-D 单独冻结，旧 `p3-like-v1` 保留。

例如，一个 item 在 train 中被曝光却从未被 like：它仍属于 all-train-exposure catalog，而不是脏数据。只学习 item ID 的模型可能缺少该 item 的正例支持，content route 仍可提供表示；这需要单独报告支持分层，而非提前删掉 item。这里是解释，不宣称新模型已经训练。

实测数字、清洗前后基数和资源唯一人工摘要见 [experiment_log 的 P3-C 节](experiment_log.md#p3-c--现有数据质量与清洗决策已正式验收)。机器证据见 [aggregate manifest](../artifacts/p3c-data-quality-manifest.json)，真实命令/hash/失败/review ledger 见 [p3c_validation](p3c_validation.md)。真实 ID、rows 和完整实体映射未回传 Mac。

## 已有证据与这次补查

| 对象 | 已做，按固定 hash 复用 | 这次补做 |
|---|---|---|
| 官方 interaction partitions | P2 全部官方 train/validation/test 文件完整性；P3 schema、duplicates、顺序两次 digest、train 基数 | 仅 train 文件重核 SHA/bytes、schema、每列 null、每 week row/like 对账；不读 holdout values |
| 反馈、时长与 history | P3/补充 EDA 六反馈、255/ratio、候选支持、duration/context/week、user history/贡献 | 复算 like/exposure user hist、prefix 总量；把时长 flag 定位到 train-like 支持分层 |
| Item metadata | duration 覆盖/null/histogram | 所有字段 train-selected null/唯一 join；author_id 分组后 item/曝光/like 支持与长尾 |
| User metadata | schema、train 覆盖/null | age/gender/geo 的匿名码范围、distinct 与 zero-code；不解释业务编码 |
| Content embedding | P3 完整 catalog ID join、dtype/dimensions、finite/zero-norm 检查 | 重核整个既有文件 SHA/bytes，复用相同字节的质量证据；不 load/copy 完整 embedding |
| Item support | train catalog/liked-item 基数 | catalog 全人口曝光/like 直方分箱、nearest-rank 分位、头部份额、zero-like |
| 清洗 | frozen preserve policy | 逐项解释保留/STOP，before/after 对账；实际删除为零 |

完整文件 hash 确认的是字节一致性，不认证上游日志真实性；也不重新证明未提供的 timestamp。P3-C 输入列表只有 train、metadata 与 embedding。官方 holdout 完整性来自既有验收，不拿其 values 学习清洗或 feature 规则。

## 字段是什么、能做什么

Interaction grain 是一条物理记录；没有 event_id 或 timestamp。官方 source types 与 train 全量 null/schema 对账均在 manifest。

| 字段 | 实际来源/type | 用途与可用性 |
|---|---|---|
| user_id / item_id | interaction，uint32 | join 与匿名实体键；可映射稳定内部索引，不能从数值推用户/item业务意义 |
| like | interaction，bool | 主 label；仅 target 前的 like 进入 history，target 自身反馈只能当监督 |
| dislike / share / bookmark | interaction，bool | 各自显式反馈；本轮审计；不覆盖 like，不把未反馈当明确负例 |
| click_on_author / open_comments | interaction，bool | 打开作者页/评论区；没有普通 click，不报告 CTR |
| place / platform / agent | interaction，uint8 | 匿名 context；码与业务含义未确认，core 不使用，若研究只能取 target 前事件 |
| timespent | interaction，uint8 秒编码 | 观看画像/异常 flag；255 机制未知，core 不作 label/history阈值或权重 |
| week / physical_row_ordinal | partition/物理读取衍生 | 顺序位置，非精确时间；prefix 必须严格小于 target，不制造 session |
| item_id / author_id | item metadata，uint32 | 按 item_id 唯一 join，author 仅 train-only 质量分层；无作者语义或实时 snapshot claim |
| duration | item metadata，uint8 秒编码 | 观看 ratio 审计；静态 metadata 采集时点未知，core 不使用 |
| train_interactions_rank（item） | item metadata，uint32 | 全 train 排名；包含早期 prefix 之后的互动，不能输入早期训练样本 |
| user_id | user metadata，uint32 | 按 train user ID 唯一 join；不因表中覆盖未来用户而扩主样本 |
| age / gender / geo | user metadata，uint8 | 只有匿名码 profile 与 coverage；映射/采集时点/zero sentinel 未证实，core 排除 |
| train_interactions_rank（user） | user metadata，uint32 | 全 train 排名，同样有 prefix 未来信息，排除 core |
| item_content_embedding | 固定官方 NPZ，float16 content vector | 通过 item_id join；官方称 content-only，沿用旧 content feature；无独立生成时点证据 |

User age 的码值范围不等于已经验证真实年龄；gender 数值不能猜男女；geo code zero 不能擅自称缺失或映射地区。dtype 的整数范围检查不是语义编码验证。所有 required selected metadata null 和 train join 缺失都必须 STOP，不能静默 fill。

## Point-in-time registry（本轮建议，未改冻结合同）

| 候选信息 | 早期 train prefix | validation/test query | 当前处理 |
|---|---|---|---|
| Past like item history | 只取严格早于 target 的事件 | train-only 最近 like，target week 不追加 | 主线保留；截断不改变时间界限 |
| Watch history 消融 | 同一 target 前曝光，保留事件原序 | 同一 train-only 观测历史；与主线共 targets/queries/catalog | 后续一次独立消融，未在本轮实施 |
| Target like/其他反馈/timespent | outcome，不能当 target 的输入 | 同样不可见，不能用于选择 query | 不作为 query feature |
| Past context/反馈/watch | 过去事件可观测，但编码机制仍未知 | 保守 train-only | 只审计，core 排除 |
| History ordinal/length | 由 target 前事件计算 | 由固定 train history 计算 | 可用；ordinal recency 不是秒级 recency |
| Full-train item/user rank、popularity、author count | 截止末 train 才有，放入早期 prefix 会泄漏未来 | end-train 模型/评估切片可用 full-train 统计 | metadata ranks 排除 core；全 train Pop 是离线拟合基线，不是早期 prefix 特征 |
| Train catalog/support/ID mapping | offline train corpus 的共同域；不是每个 prefix 的动态 catalog | cutoff train 结束，共同候选域 | 保留共同域，未称真实 dynamic catalog；未因 feature 覆盖加入未来 item |
| Item content embedding | 静态官方 content artifact，content-only 声明；生成 snapshot unknown | 同一 pinned artifact，按 train catalog join | 沿用明确静态 benchmark assumption，不能称历史时点 independently verified |
| Author/duration metadata | snapshot/effective time unknown | 同样没有时点凭据 | 审计可用，core 排除；未来 feature 需另立合同 |
| User age/gender/geo | snapshot/effective time 与编码 Unknown | 同样 Unknown | core 排除，不从 null=0 推断 point-in-time 可用 |
| Observed first-seen | 只能相对 prefix cutoff 求此前观测；不是 publish time | train 内最早观测 week，供 proxy 切片 | 只称 observed-item proxy；不称真实上新 |

Like label 支持与严格先后是两个独立问题。支持门已通过，不证明隐藏时间完全正确；现有顺序依赖固定 README 的 official-order assumption、稳定 physical bytes/digest，而不是独立 timestamp/session 验证。若后续协调不接受该 assumption，停止训练。

## 质量发现与最小处理

| 发现 | 证据/信心 | 对 like 训练的影响 | 清洗决策 |
|---|---|---|---|
| Item/author 长尾与 zero-like | 完整 train 聚合，高信心 | ID-only 正例支持不均，不能把 catalog 缩成 liked items 后声称同一候选任务 | 保留主 catalog/尾部，报告 train-defined 支持分层 |
| 少数 users 贡献较多 prefixes | 完整 train 聚合，高信心；异常成因未知 | 按 example 等权会强调高贡献 users，规模不等于独立 user 数 | 保留主样本；user-balance 后续独立消融，不称 bot/刷量 |
| ratio>1、255、like&dislike | 复用 pinned EDA，用户分层补核，高信心；机制未知 | 主线不依赖这些字段，不能据此换 label 或剪观看值 | 保留并 flag，显式冲突不改 like |
| metadata snapshot/rank | 字段与 train rank 来源明确；采集时点未知 | 全 train rank 放入早期 prefix 会泄漏，匿名属性还缺语义 | 排除 core；不做填零或自动删除 |
| Required null/ID/schema/hash/join | 本轮 gate 未触发；完整 train/selected metadata 对账 | 若发生将破坏样本/feature 绑定 | STOP，保留证据，禁止静默 repair |
| exact duplicates 与重复 user-item | 复用 pinned 完整 audit；计数/实现语义明确 | 没有 event ID，不可由两项零计数推断 upstream 事件恰好唯一 | 保留 frozen duplicate policy，不新增删除规则 |

P3 的 `duplicate_user_item` 是跨 train 全局 `(user_id,item_id)` 重复额外行；`exact_duplicate_rows` 的记录 hash 包含 week，实际意为同 week 全字段完全相同记录的额外行。跨 week 同字段不属于后一口径；前一项全局 pair 证据更强，但仍不能识别隐藏日志重放、近重复事件、session 或上游过滤机制。禁止把每个重复 pair 默认当脏数据，也不能用“无重复”加强独立时间证据。

## 拟训练样本与清洗前后

[experiment_log 的基数表](experiment_log.md#p3-c--现有数据质量与清洗决策已正式验收)逐项列出原始曝光、全部 train users/items、like positives、支持 users/items 与 prefixes 的 before/after。清洗没有改变任何一项，也没有把无 like user 删除；按 like task 的 prefix 定义，无历史或只有一次 like 的 user 自然没有训练例，这与清洗删除 user 不同。

低活跃/长尾仍进入主样本的源域和 catalog；不实行 5-core。后续 full query 的 eligibility/OOV 分母由新合同规定，本轮不读取 holdout 或重新挑 query。确定性 ID 映射、PAD 与 dtype 转换属于后续工程，不是删除数据。本轮没有生成新的 filtered data 或训练输入替代件。

## 完成 gate

Source/synthetic、真实 train aggregate、schema/基数/reuse hashes、公开边界与受保护文件 hash 已通过独立review和Astra正式验收。Astra独立重算完整item/author profile逐字段一致；独立final source/docs/public review PASS，P3-C Complete。作者 self-check 不替代这些验收证据。本轮未训练、未新增 label、未改 data_contract、未读 holdout values、未下载或 commit/push，不自动进入 P3-D。
