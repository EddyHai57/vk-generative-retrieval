密级: 公开

# Data Contract

## 当前边界

P3 policy已于真实train audit之后、任何validation/test内容读取之前实际冻结，版本`p3-like-v1`，配置`configs/p3.yaml`。P3已正式验收，P4/P6已消费该冻结合同完成固定audit评估；当前阶段状态只读`../STATE.md`，历史验证见`p3_validation.md`，指标只读`experiment_log.md`。

P2已完成服务器固定revision `084a18386e149a454dd6dbe0123478e75c6b09ce`的MVP下载与SHA校验；实际数据统计唯一人工摘要见`experiment_log.md` P3。内部公司数据、代码、schema与指标不得进入本项目。

## Source与schema

使用`deepvk/VK-LSVD`的`ur0.01_ir0.01`，不重采样、不下载parent全量。固定官方README及P2 inventory记录在`p3_validation.md`。`irX`的生成含义在该README未明确，不能凭名字宣称随机item采样；此subset结果不外推完整平台分布。

Interaction实际schema：`user_id/item_id:uint32`；`place/platform/agent/timespent:uint8`；`like/dislike/share/bookmark/click_on_author/open_comments:bool`。没有普通`click`、timestamp或interaction内`duration`。固定README将timespent描述为秒，取值0–255；duration在item metadata中为uint8秒。实际null、范围、覆盖、cofeedback与重复证据见train audit。

`click_on_author`仅表示打开作者页；`open_comments`仅表示打开评论区。任何字段不得改名为普通click，也不报告CTR。

官方content artifact通过`item_id`匹配，`embedding:float16[64]`，不能按metadata行号匹配。README称其只由内容训练，无collaborative信号；它不是本项目的raw multimodal encoder。metadata/content可以覆盖未来观测item，但主catalog只从train互动确定，未来item不得借feature可用性进入候选。

## Ordering evidence与限制

顺序键固定为`(week, physical_row_ordinal)`。ordinal在每个parquet内从零开始，按照row-group、batch、row物理读取顺序生成，不能按user/item排序后重新生成。

证据由固定README对week内时间顺序的官方声明、subset保序过滤示例、固定文件hash与两次读取的完整row-sequence digest组成。实际parquet无timestamp，因此不声称独立验证隐藏timestamp单调、逐事件精确时间或全量parent到subset逐行保序。当前协议依赖明确的official-order assumption，协调review若不接受此证据强度则停止后续训练。

History必须严格先于target位置；同week仅允许较小ordinal作为train prefix。验证与test使用更早train weeks，不包含同target week的早期反馈，因此避免对hidden timestamp ties作更强claim。

## Label与工程support门

主positive固定为`like is True`；不存在composite阈值，不把exposure当positive。所有feedback保留原义用于审计，dislike与like共现也不事后改label。

预注册门：仅用train验证至少1000位user有不少于两个like positives；minhistory为1。这是小型工程smoke的支持门，不证明统计效力、代表性或模型质量。train支持门通过才冻结；失败时按tasks预注册composite，仍不足则停止TIGER。不得查看holdout后改变label或support门。

Null不是负反馈；required字段出现null/schema drift/invalid ID时停止并保留证据，不静默填零。真实train的schema与每个文件逐项校验。

## History与训练Target

只收录like positives，按上述顺序保留。每个train positive（有至少一个更早positive时）形成一个prefix→next-positive训练例；prefix只含该target之前的positive，截断为最近50个，按原序排列。单positive user没有训练例；长user可贡献多个训练例，这不同于evaluation每user一个query。P3阶段不训练模型。

不去重positive history，且不做consecutive-repeat collapse；若将来增加消融，须另立version而不改当前结果。完全相同记录与重复user-item计数分别报告；不把重复当独立顺序证据。

## Split与Evaluation Target

官方train为week00–24，validation为week25，test为week26。固定README与P2 inventory确认已有独立validation，因此取代Draft中从train尾week划validation的proposal；不再缩减train，不随机interaction split。

各evaluation split每user仅一个query：target是该week中首个真正`like=True`的item。不能先过滤OOV再选target，也不能换成首个可达like。history均固定为train-only最近50个like items，minhistory1；test不加入validation反馈、不rolling追加target week的反馈。这是保守MVP协议，未使用全部test前可观察历史。

没有positive target的train users、target-positive但history不足的users与train未观测target users分别报告。history不足者不进入主eligible query集合；有history的首target即使OOV也保留并计入all-eligible分母。主in-catalog slice与OOV slice必须并列报告，不能只给静默缩减后的分母。

P3保存query全集和hash。P4只可用固定seed用户抽样冻结audit set，不能按target可达性或模型命中挑选；抽到的OOV保留分母。audit set不代表全test。

## Candidate Catalog、first-seen与OOV

主候选为train所有exposure中出现过的item，不限于liked items，cutoff为week24。每个snapshot保存排序后的item ID数组hash、item count、feature coverage、source/policy版本；完整ID集合只在服务器。后续所有方法共享完全相同catalog、features、histories、queries、split与调参预算。

`item_first_seen_week`只取此次数据中最早观测week，是observed-item proxy，不是publish/listing或真实上新。validation/test报告相对train的OOV与相对此前weeks的新增observed-item统计，不能称真实dynamic catalog。

`seen_item_masking=false`，所有方法保留同一完整train候选集。同score时`item_id`升序确定tie；evaluator在P4实现，当前只冻结规则。OOV item不凭metadata/embedding可用性加入候选。

## Feature Registry

| Feature | Entity | Source | Availability | Transform | Missing/OOV | MVP role |
|---|---|---|---|---|---|---|
| item_content_embedding | item | 固定官方NPZ | train-observed catalog ID join | 保留float16，检索前在P4做float32 L2 normalize | missing/nonfinite停止；zero单列 | content baseline + SID |
| week / physical_row_ordinal | event | partition与物理读取 | official-order assumption | 有序位置对 | invalid/order不稳定停止 | split + prefix |
| like | event | bool feedback | 只取target前prefix | `is True` | null停止 | 固定主label |
| other feedback | event | 官方bool字段 | audit only | count/coverage/cofeedback | null停止 | 审计，不作主feature |
| timespent | event | uint8秒 | audit only | range/histogram | null停止 | 不作label/history阈值或模型feature |
| duration | item | uint8秒 metadata | catalog metadata join | range/histogram | missing/null/zero单列 | 审计，不用于timespent归一化 |
| observed first-seen week | item | 最早观测partition | snapshot-relative | min week | unknown/OOV单列 | proxy only |
| history length | query | target-prefix | train-only | count after截断 | zero-history过滤并报告 | query slices |

Producer兼容规则固定为CPython3.11，允许记录中的Mac/Linux patch不同；source、uv.lock、Git基点与PyYAML/numpy/pyarrow精确版本必须相同。旧producer audit、policy漂移或嵌套bool在任何holdout读取前拒绝。

MVP不使用timespent/duration ratio、clipping、观看composite、user年龄/性别/地区或popularity rank作为模型feature。未来新增feature须另行注册并确保query-time可用性。

## Frozen artifact

冻结manifest SHA256为`3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f`；policy SHA256为`3be46f445676d302c4b4f69cebfff0e4af35c040f09473f1c08d54dfe7507a01`。复现命令、producer/source/lock及artifact hash只读`p3_validation.md`；机器manifest只含aggregate support与文件hash，完整queries/catalog/examples保留在服务器。

## 公开来源

[固定revision官方Data Card](https://huggingface.co/datasets/deepvk/VK-LSVD/blob/084a18386e149a454dd6dbe0123478e75c6b09ce/README.md)。本合同基于官方声明与本项目真实审计，二者的证据强度不同。


## traditional-like-v1

本节是 P3-D 冻结的传统召回补充合同，版本 `traditional-like-v1`；机器镜像为 [configs/traditional.yaml](../configs/traditional.yaml)，二者 byte SHA 由 [P3-D manifest](../artifacts/p3d-retrieval-contract-manifest.json) 单向绑定。所有 label/history/target/catalog/feature/预算与评估语义仍以本文件为唯一合同；YAML 是可机读镜像而非第二套规则。若二者不同或历史 producer 来源不匹配，STOP 并交协调裁决。当前验收状态只读 STATE；本节冻结不意味着任何新训练/评估已执行。

上文 `p3-like-v1` 逐字节保留；旧 `configs/p3.yaml`、P3 冻结 manifest、producer audit/source/lock 及 P4/P7 artifacts 都不重写。新版本不改变 like、官方 split 或旧 audit 的成员。此前 test 的固定 audit 指标已被读取，test26 不是 pristine final holdout；不能通过新版本重新声称未用 holdout。

### 共同样本与评估

例如某 user 在 train 内有依次三个 likes：第二、第三个分别是两个训练 targets；第二个 target 的 history 只能来自第一个 like。当前 target 即使以前也被 like，仍是有效监督；它的 candidate 列不能被 past-positive mask 屏蔽。最近50限制只缩短模型输入，不能缩短 loss 的全部过去正例排除集合。

- Label 仅 `like is True`，train00–24，validation25，test26；顺序键沿用 `(week, physical_row_ordinal)` 与 official-order assumption，非真实 timestamp/session。保留所有原始 rows、重复语义、低活跃与长尾，不做 5-core、未知异常删除或 ratio clipping。
- Train examples 恰好是每个已有 train next-like prefix target（至少一个严格更早 like）；输入为 target 前最近50 like 事件，保留重复和原序。每 example 等权，epoch shuffle without replacement；尾 batch 不跨 epoch 补齐。
- 每个 evaluation split 每 user 的首个 like 为 target，先选 target 再标 OOV。Eligibility 仅要求至少一个 train like；所有 eligible queries 都进入主评估，保留 OOV；history 是 train-only 最近50 likes，test 不追加 validation 或 target-week 事件。不把 full eligible 改成可达/命中/活跃子集。
- Candidate 是全部 train exposure items，cutoff week24，seen masking=false，同分 item_id 升序；所有 methods 共 targets、history、query members、catalog、固定官方 features、split 与批准调参预算。OOV 不因 metadata/content 可用而加入候选。
- 全 eligible / in-catalog / OOV 各报告 item Recall 与 NDCG@10/20/50 及各自分母；单目标 Recall 为 hit，NDCG 为命中 rank 的 `1/log2(rank+1)`，OOV 贡献0。空 slice 记 null。主 selection 是 validation all-eligible Recall@50，次 NDCG@50。
- 旧 P4 audit 仍用 seed42 的 SHA256(`42:split:user_id`)排序、每 split 最多512，抽样前不看 target/OOV/hit。旧成员与 history/target/hash、@10/20 指标 regression 必须复核；@50 是新增补充，不能伪称旧 artifact 已有。旧 audit 仅 regression，不替代 full eligible 选择指标。
- Existing deterministic float64 fixed pair reduction（float32 vectors输入）与 item-ID tie 的 blockwise exact 是 authoritative；不得物化完整 user-item score matrix。ANN 只提供近似与系统成本补充。

### Feature 与 point-in-time

Core 仅 item ID + 固定官方64D content；past ordinal position/length 是已过去 history 的结构，非额外 user/context feature。ID维度64；content按 item_id join，float16 source 转 float32 L2 normalize，zero保持零且不产生NaN，missing/nonfinite STOP。官方声称 content-only，但生成 snapshot 时点 Unknown；明确依赖 static-content benchmark assumption，不称 independently verified historical availability。所有 models 用同一 pinned content artifact，不把未来 catalog items引入。

Item branch可训练；从未有 train positive 的 catalog item，ID branch固定为0，content仍可用；单列 train-defined unsupported-item metrics。Core排除 user ID embedding、author/duration、age/gender/geo、item/user train_interactions_rank、timespent/context/feedback embeddings及target outcome。全train metadata rank用于早期prefix会泄漏；不入features。全train Pop是offline拟合baseline，不伪称早期prefix可用feature。first-seen仅 observed-item proxy。

四个 core architectures 固定为：ID valid-masked mean；ID+projected frozen content相加后的valid-masked mean；相同content输入的ordinal weighted mean（weight=`0.8^distance`，最近位置distance0）；相同content输入的Transformer（1layer、d64、4heads、FF256、dropout0.1、learned ordinal position、valid-masked mean后linear user projection）。Transformer可bidirectionally attend已过去prefix，必须pad masked；position不是秒级时间。

### Loss、采样与恢复

In-batch candidate columns只保留distinct target item，保留每条query。每row屏蔽该query严格过去的所有like items，不限最近50；当前target列永远有效。不得用future positives、validation/test来排negative。L2 normalized dot/temperature CE；无negative的row skip并报告，分母为每epoch及累计all drawn queries（含skipped），任一skip fraction >1% STOP。保持历史所有positives的可核验证据，不能用截断history冒充全部mask。

FP32 AdamW，weight decay1e-4，gradient clip norm1。Pilot依次batch512/256/128，要求四模型全部满足RAM/VRAM/disk至少20%余量，之后冻结共同batch；三档都不通过则STOP，不自行加资源或降门。输出目录exclusive，保留best/last；checkpoint/resume绑定model、optimizer、global RNG、epoch/data cursor、source/config/input hashes，不能因resume重置预算或漏/重复样本。nonfinite/hash/resource/ordering/fairness conflict STOP，失败ledger保留。

### 有限预算与选择

每模型4 configs：LR按 `[1e-3,3e-4]`、temperature按 `[0.07,0.15]` 的nested固定枚举；model顺序 ID mean、ID+content mean、ordinal content mean、content Transformer。Tuning seed42，最多30epochs，每epoch full validation，patience3；strict improvement按Recall50再NDCG50 lexicographic，平分epoch取更早；config平分取固定枚举先者。Best core architecture按各模型seed42 selected config的best checkpoint validation Recall50→NDCG50比较，平分取固定model顺序先者，不按test或best seed。部署/融合代表checkpoint固定seed42，其余两seed仅独立稳定性报告。每模型selected config再跑seed2024/2025，报告42/2024/2025全部结果与失败，不挑best seed，不追加搜索。旧TIGER仅保留不同预算smoke，不能称matched-budget comparison。

Pop按train like事件计数；ItemCF是binary-user-like cosine cooccurrence，每item top200 neighbors，historysum，sparse blocked无fullmatrix，shortlist用Pop去重补全并报告fillrate；content baseline为normalized64D history mean exact。

Best core architecture/config冻结后做三项**独立**消融，各seed42/2024/2025、不额外调参、不混合赢家：in-batch + 每batch128 distinct uniform catalog negatives（排existing columns，沿用同row past-like mask/target override；无hardnegative/logQ）；有放回uniform eligible train user再有放回uniform prefix（每epoch draws=原example count）；watch history只替换输入为target前最近50 observed items，evaluation仍train-only。Watch保留原trainingtargets/evalquery members/targets/catalog，不混入无like的新users、不用timespent权重或feedback embeddings。Watch的loss mask仍是全部strict-past like positives。Train-defined popularity/history/support/OOV slices只作报告，不由holdout挑边界，不对微小组作强结论。

Fusion主线使用best core而非消融赢家；routes按ItemCF / best-core TT / content / Pop，每route top100、final50，weighted RRF k60。Weights `{0,1,2}^4`排全0与成比例重复，保留lexicographically first integer tuple。Validation Recall50→NDCG50→更少nonzero routes→固定lex tuple；只用validation选择。报告single/equal weights/best、leave-one-route的original quota及equal-total quota（400slots，剩余routes等分，余数按route顺序）、overlap/union，不把更多候选当方法收益。

ANN固定existing FAISS HNSW M32/efConstruction200，efSearch64/128/256，same pinned export vectors；复用旧P4已冻结validation audit（seed42/max512，不重新抽样），query hash取P4 manifest的audit_set_hashes.validation；在这组fixed validation queries验top50 overlap门0.98，取最小passing ef；无passing时报告ANN gate unmet并保留exact，不降门。Overlap、成对exact/ANN model metrics delta、P50/P95 latency、memory、index build cost均限这组audit并分开，不冒充full validation metric delta；模型质量/调参仍以exact full eligible validation为准。Offline export/index/infer需shard resume无query loss/duplicates，绑定checkpoint/catalog/idmap/index；unknown IDs计数，empty history Pop fallback另列cohort，不扩主eligible分母。大产物仅server。

Core configs、固定消融、fusion和ANN选择完成冻结后，新协议test只执行一次预注册pass（其中含全部core seeds及预注册消融），不再依test调参。历史已经用过test这一限制持续有效。合同冻结本身不读取任何新的holdout values或指标。

### Producer 与 consumer binding

P3/P3-C既有producer identities、source/config/lock与文件hash从已验收manifest逐项保留验证；P3-D manifest另绑定P3/P3-C/P4 public文件hash、catalog/fullquery/audit hashes与入场旧合同hash。旧P3 producer lock是其manifest记录的历史版本，当前overlay已在P6加入torch并使用另一个lock；两者分别绑定，P3 replay必须原历史producer/source/lock环境，不能把当前lock追写旧产物。旧producer依赖版本/CPython3.11兼容门照旧，不允许因新consumer源码增加而重写或弱化旧校验。未来consumer在自己的run manifest独立记录实际Git基点、dirty/source fingerprint、config/lock/精确依赖、输入与生成target/history/catalog/query/ID-map pins；尚未实现的consumer fingerprint不得伪造为旧producer fingerprint。新/旧fingerprint职责不同，hash验证一致性而非上游真实性。

旧 `baseline_run.load_static` 会同时加载validation/test；新consumer必须按split选择读取，不能直接复用它而提前读取新的test metrics。旧 `training.ShuffledCursor` 跨epoch补batch；新协议tail不得跨epoch补齐，后续新consumer需独立实现该行为，不修改旧TIGER cursor或其回归语义。prepare可绑定既有test query artifact hash但不能由它选配置；test读取与计算新metrics必须等本协议freeze gate。

P3-D只冻结声明：机器config与本合同完整bytes由manifest绑定，manifest不包含自身hash，本合同不嵌manifest/config hash，避免自引用。后续任何新文件改变需要新版本或明确协调裁决，不以文档编辑静默覆盖已运行的协议。
