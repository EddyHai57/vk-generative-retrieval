密级: 公开

# Decision Log

## 当前边界

本文件记录已确认 Decision 与尚待验证 Assumption，不记录虚构实验结果。

P2已完成批准范围的服务器下载验收；P3真实数据审计与合同冻结完成，尚无模型训练或效果评估。当前整体gate以STATE为准。

公司内部信息、数据、schema、代码与实验结果不得进入本项目。

## 决策

| ID | Type | Status | Decision / Assumption | 原因 | 复核触发条件 |
|---|---|---|---|---|---|
| D-001 | Decision | Confirmed | 主数据从 Amazon 路线更换为 VK-LSVD。 | VK-LSVD 同时提供 week-level 顺序、多反馈、官方 content embedding、大 catalog 与多级 subset，更贴合 dynamic SID 审计。 | 数据下载后发现关键 schema 或许可与 data card 不符。 |
| D-002 | Decision | Confirmed | 项目核心 claim 是 week-level SID staleness、aligned refresh 与 dense fallback。 | 该问题可被 stale/fresh/aligned 三组实验否证，且比堆叠模型更可审计。 | 数据无法构造可信 week snapshot 或新增 observed-item slice。 |
| D-003 | Decision | Confirmed | MVP 使用 `ur0.01_ir0.01`，flagship 使用 `ur0.01` 做 scale audit。 | MVP 需要约 125K item 的 exact audit；flagship 才承担约 12.47M item 的 ANN 与 streaming 复杂度。 | 下载后的 manifest 规模或 split 结构不符。 |
| D-004 | Decision | Confirmed | 默认 GPU 为 Thunder Compute RTX A6000 48GB。 | 48GB VRAM 与 Ampere 兼容性适合大 catalog prototype。 | ≤1h acceptance probe 未通过库存、SSH、CUDA、FAISS、吞吐、磁盘或 restore gate。 |
| D-005 | Decision | Confirmed | Thunder 失败时才切 RunPod Secure Cloud RTX 4090 + persistent network volume。 | fallback 需要成熟 CUDA 环境与独立持久化。 | Thunder gate 失败或成本/可用性显著变化。 |
| D-006 | Decision | Confirmed | Phase-1 总预算不超过 AUD 100。 | 限制个人项目试错成本。 | 用户另行调整预算。 |
| D-007 | Decision | Confirmed | GPU 目标不超过 30 小时，hard ceiling 48 小时，首次充值不超过 US$50。 | 为失败 probe、存储、汇率与重跑留余量。 | 进入 flagship 前重新审批。 |
| D-008 | Decision | Confirmed | 仓库初始设为 public。 | 目标是形成可审计 portfolio。 | 数据、许可或安全审计不通过时转 private。 |
| D-009 | Decision | Confirmed | Public Git 只保存代码、配置、文档、manifest 与小型指标摘要。 | 避免分发数据、大 artifact、credential 与完整 SID mapping。 | 无。 |
| D-010 | Decision | Confirmed | Raw data、processed data、embedding、checkpoint、index、trie 与完整 SID map 只留本地或服务器。 | 控制许可、仓库体积与可逆识别风险。 | artifact governance 改版。 |
| D-011 | Decision | Confirmed | 两天目标是可审计 engineering vertical slice，不是完整 benchmark。 | 数据合同、约束解码与 evaluator 正确性优先于方法数量。 | MVP 提前完成且所有 gates 已通过。 |
| D-012 | Decision | Confirmed | 两天只做 Popularity、content-centroid + `IndexFlatIP`、balanced hierarchical k-means + collision suffix、small TIGER one-seed。 | 这是最小端到端闭环。 | 不因剩余时间临时扩 scope。 |
| D-013 | Decision | Confirmed | 两天只承诺固定 query audit set。 | 全 test wide-beam 的 wall-clock 尚未验证。 | 实测资源允许时可扩大，但必须保留固定 audit set。 |
| D-014 | Decision | Confirmed | 主正反馈 label 为 Draft：先只用 train 检查 `like` 覆盖。 | VK-LSVD 没有统一普通 click，label 必须先于模型冻结。 | train coverage audit 完成。 |
| D-015 | Decision | Confirmed | `like` 覆盖不足时才定义 strong-engagement composite，并在看 validation/test 前冻结。 | 防止用 holdout 指标反向挑 label。 | label policy version 变更需新实验系列。 |
| D-016 | Decision | Confirmed | 禁止用 next exposed item 作为用户偏好主任务。 | 它主要学习 logging recommender policy，不等同于正偏好。 | 仅可作为单独 policy-modeling auxiliary task，不能替换主任务。 |
| D-017 | Decision | Confirmed | 主 catalog 为 train-observed items；first-seen/OOV 单独报告。 | 没有真实 listing time，不能静默删除不可达 target。 | 获得更可靠 catalog availability 信号。 |
| D-018 | Decision | Confirmed | MVP exact full-catalog 面向约 125K items；flagship 采用 ANN full-catalog + 固定 query exact audit。 | 全 score matrix 在旗舰规模不可物化。 | exact evaluator 或 ANN audit 不能通过。 |
| D-019 | Decision | Confirmed | 评估以 item Recall/NDCG 为主，并同时报告 SID collision、invalid、dead-end、duplicate、latency、RAM、VRAM。 | SID 命中不等于 item 可达，quality 必须与系统成本一起解释。 | 新 claim 需要增加但不能替代这些指标。 |
| D-020 | Decision | Confirmed | Keep：TIGER、trained dense control、LIGER-style hybrid、stale/fresh/aligned refresh。 | 这些模块直接回答核心可证伪问题。 | stop gate 触发。 |
| D-021 | Decision | Confirmed | Stretch：MaskGR。 | 只有核心 claim 完成后才值得研究 decoding efficiency。 | flagship gates 全通过且预算剩余。 |
| D-022 | Decision | Confirmed | Defer：RPG、HSTU、LLM、RL。 | 它们需要不同 SID、长序列条件或不同训练目标，会分散当前主线。 | 独立项目或新阶段审批。 |
| D-023 | Decision | Confirmed | 当前不添加 repository LICENSE。 | 需先完成 third-party code/data provenance 审计；public 不等于允许复制受限实现。 | 首次发布可复用代码前。 |
| D-024 | Decision | Confirmed | 对 LIGER 与其他受限/不明许可工作只按论文独立实现概念，不复制代码。 | 研究引用不自动授予代码复用权。 | provenance 审计定位明确兼容许可。 |
| D-025 | Decision | Confirmed | 效果数字只进入 `experiment_log.md` 摘要，原始 metric artifact 以 hash 引用。 | 避免同一指标散落后漂移。 | 无。 |
| A-001 | Assumption | Open | 公开 week partition 足以作为跨周切分依据，且下载产物包含可验证的文件内顺序。 | 本轮尚未下载数据验证排序键与稳定性。 | P3 schema/order audit。 |
| A-002 | Assumption | Open | `like` 覆盖可支持 MVP 主任务。 | 仅是预注册优先项，不是已证实事实。 | train-only label coverage audit。 |
| A-003 | Assumption | Open | A6000 48GB 可在两天内完成 small TIGER vertical slice。 | wall-clock、数据吞吐与 checkpoint 时间尚未实测。 | Thunder acceptance probe 与 smoke run。 |
| A-004 | Assumption | Open | aligned refresh 可低成本恢复 fresh-scratch quality。 | 这是核心研究假设，不是历史实验结论。 | flagship multi-seed experiment。 |

## 命名边界

仅文档与接口完成时称 scaffold。

完成同合同下的 Popularity、content、SID 与 TIGER item-level 审计后，才称 TIGER benchmark。

完成 trained dense、hybrid、dynamic refresh、multi-seed 与 million-item quality/latency audit 后，才称 Advanced Generative Retrieval。

## 公开来源

VK-LSVD data card: https://huggingface.co/datasets/deepvk/VK-LSVD

TIGER: https://arxiv.org/abs/2305.05065

LIGER: https://arxiv.org/abs/2411.18814

SID staleness: https://arxiv.org/abs/2604.13273

Collision-faithful evaluation: https://arxiv.org/abs/2605.25330

## Bootstrap补充决策

| ID | Type | Status | Decision / Assumption | 原因 | 复核触发条件 |
|---|---|---|---|---|---|
| D-026 | Decision | Confirmed | 在workspace权限允许正式创建`AGENTS.md`前，使用`AGENTS.draft.md`保存完整规则；P1保持Blocked。 | Kiro permissions明确拒绝agent写`**/AGENTS.md`，不得绕过访问控制。 | 用户允许并将草案落位后，复核所有规则链接。 |
| D-027 | Decision | Confirmed | `tasks.md`是P0～P13阶段编号唯一事实源，`docs/architecture.md`必须使用同一编号。 | 防止Codex因阶段编号冲突执行错误任务。 | 阶段变更时同步复核两文件。 |
| D-028 | Decision | Confirmed | 真实VK-LSVD只允许由服务器直接下载；本机仅使用synthetic fixture。 | 避免本机/公开仓库承载大数据，统一artifact治理。 | 用户明确批准新的数据环境。 |
| D-029 | Decision | Confirmed | P1只使用synthetic fixture；真实VK-LSVD从P2开始由服务器下载。 | 防止Codex越过数据许可、服务器验收和P3合同冻结。 | 数据执行环境经用户重新批准。 |
| D-030 | Decision | Confirmed | Thunder A6000 acceptance probe归入P2，并要求用户先批准计费实例。 | P6/P7依赖真实数据与GPU，服务器验收不能拖到P8。 | 服务器工作流改版。 |
| D-031 | Decision | Confirmed | 正式P1 branch/worktree必须以用户批准的bootstrap commit为基点。 | unborn main无法创建可复核worktree；agent无权自行commit。 | 用户明确要求创建bootstrap commit。 |

| D-032 | Decision | Confirmed | P2负责服务器验收与下载完整性，P3负责schema/order审计与合同冻结。 | 消除下载和顺序证据的owner冲突。 | 数据pipeline阶段调整。 |
| D-033 | Decision | Confirmed | P4负责首次实现exact evaluator；P1只负责通用可审计运行框架。 | 避免为后续阶段预建接口。 | evaluator职责调整。 |
| D-034 | Decision | Confirmed | P8冻结旗舰split/final holdout，P9实现trained dense control与pure GR。 | 消除P9输入/输出循环依赖。 | 旗舰阶段调整。 |
| D-035 | Decision | Confirmed | `outputs/`不提交指标表或图表；人工摘要只进入`docs/experiment_log.md`。 | 与default-deny `.gitignore`保持一致，减少误提交风险。 | artifact governance改版。 |

| D-036 | Decision | Confirmed | `AGENTS.md`保持thin，只保存稳定行动规范、对话风格、安全边界和事实源指针。 | 减少每轮context占用并避免易变事实漂移。 | 项目治理结构调整。 |
| D-037 | Decision | Confirmed | `STATE.md`是当前快照唯一源；根`handoff.md`是跨对话恢复入口；`tasks.md`是路线与验收唯一源。 | 三者分别回答现在是什么、如何接手、以后做什么，避免重复。 | 状态治理调整。 |
| D-038 | Decision | Confirmed | `CLAUDE.md`仅保留指向`AGENTS.md`的兼容指针。 | 当前不使用Claude Code，不维护重复规则。 | 工具链变化。 |
| D-039 | Decision | Confirmed | GitHub remote不阻塞P1；bootstrap commit是P1 branch/worktree的前置条件。 | 本地Git即可开发；remote可在首次发布前创建。 | Git工作流变化。 |

| D-040 | Decision | Confirmed | Thin `AGENTS.md`已正式落位；P0关闭，下一gate只以`STATE.md`为准。 | 稳定规则与STATE/handoff/tasks职责已分离。 | Harness或事实源结构变化。 |
| D-041 | Decision | Confirmed | 旧`AGENTS.draft.md`本地保留并由`.gitignore`排除，不阻塞P0。 | 不执行未经确认的删除，同时避免误提交。 | Eddy决定删除或归档。 |

| D-042 | Decision | Confirmed | Eddy已明确批准创建一个P0 bootstrap commit；该commit不包含remote、push或P1实现。 | 为后续P1 branch/worktree提供可复核基点。 | Bootstrap commit创建并验证。 |

## P1执行决策（2026-10-07）

| ID | Type | Status | Decision / Assumption | 原因 | 复核触发条件 |
|---|---|---|---|---|---|
| D-043 | Decision | Confirmed | 用户已批准P1独立worktree与首次联网准备；Python固定>=3.11,<3.12、.python-version=3.11、PyYAML==6.0.2、标准库unittest；不commit/push。 | 显式执行授权承接P0后的scope gate。 | 用户改变scope或工具链。 |
| D-044 | Decision | Confirmed | P1 schema v1只使用人工fixture与local_cpu/no network/no GPU/model none；固定fixture路径、seed范围[0,2**32)。 | 防止smoke配置携带真实数据或跨阶段行为。 | 后续已批准task需要新的schema version。 |
| D-045 | Decision | Confirmed | Manifest限定授权的19字段；用git_commit+git_dirty+source_fingerprint区分基点与未提交代码；hash不是真实性认证。 | 避免把bootstrap commit冒充实际源码；具体字段优先于旧handoff的通用pending/command模板。 | 协调Agent验收或后续manifest设计。 |
| D-046 | Decision | Confirmed | 测试目录可在唯一outputs/p1-test-*内嵌独立子目录；manifest/result绑定同目录。 | 满足repo-relative引用同时隔离与清理测试产物；未改变persistent P1输出边界。 | 产物路径治理变更。 |

依赖provenance：仅从官方PyPI安装PyYAML 6.0.2，未复制第三方实现。官方metadata为MIT： https://pypi.org/project/PyYAML/6.0.2/ 。已在当前.venv复核distribution version与license；lock由uv 0.12.23生成并保存artifact hashes。这里只审计P1新增依赖，I-007未来数据/代码发布审计仍Open；不据此添加repository LICENSE。

| D-047 | Decision | Confirmed | 每个part按做事顺序、成果、问题与处理、下一步进行简要汇报；Sol可起草，协调Agent核对。 | Eddy要求理解实现过程并积累面试复盘素材。 | 用户调整汇报要求。 |
| D-048 | Decision | Confirmed | Eddy已批准P1本地commit并合并main；未授权本次创建remote/push或自动租用服务器。 | 保存已通过协调验收的P1成果。 | 本地合并及验证完成。 |


## 平台与SSH授权更新（2026-10-07）

| ID | Type | Status | Decision / Assumption | 原因 | 复核触发条件 |
|---|---|---|---|---|---|
| D-049 | Decision | Confirmed | 当前服务器平台改为Eddy手动租用的Vast.ai标称RTX A6000 48GB；取代D-004/D-005的平台选择，D-030的P2验收归属保持。 | Eddy表示原平台需要排队，并已手动租用Vast；不能沿用Thunder特有恢复流程。 | P2硬件、环境、预算或持久化/恢复验收失败，或Eddy更换平台。 |
| D-050 | Decision | Confirmed | Eddy授权整个Codex通过本机SSH配置连接服务器；方法唯一源为`server_runbook.md`。 | 跨对话复用已配置的连接，不传播实际地址、端口或私钥。授权不代表各对话已连接，也不替代任务、高风险操作及计费审批。 | 主机、密钥、别名、权限或用户授权改变。 |

D-004/D-005与D-030中的Thunder描述保留为历史决策；当前平台按D-049执行。A-003仍Open，其复核改为当前平台的acceptance probe与smoke run。


## P2续执行授权（2026-10-07）

| ID | Type | Status | Decision / Assumption | 原因 | 复核触发条件 |
|---|---|---|---|---|---|
| D-051 | Decision | Confirmed | Eddy明确撤销费率gate及原US$1首轮费用门，授权现有实例资源验收、最小隔离环境搭建、固定revision30文件MVP下载与SHA256验证；费用保持Unknown。 | 用户要求实际推进，不重复询问已批准常规安装下载。 | 资源硬门槛失败、数据/许可冲突，或需改变计费资源/危险恢复操作。 |

不训练、不P3、不commit/push；恢复危险操作仍具体审批，未确认恢复不阻塞可安全重下的官方数据。30分钟为有界检查目标，不作为等待费率理由。


| D-052 | Decision | Confirmed | P2 GPU FAISS验收使用官方pytorch/nvidia/conda-forge渠道精确FAISS1.11.0 py3.11/cuda12.1 build及独立cache conda环境；48包显式列表固定，不改变driver/torch/P1环境。 | 官方1.15.1与所需Py3.11/CUDA12.1解算不兼容；受信兼容版本实际GPU查询及Astra复核通过。 | 后续任务需要不同FAISS功能或环境时重新锁定/验收。 |

恢复小bundle仅为可审查准备，不构成用户已批准backup/restore或缩小完整P2恢复gate的决策。


| D-053 | Decision | Confirmed | Astra转达Eddy“完全批准”46文件279765bytes/86365bytes gzip公开与synthetic bundle备份到Mac，再传回服务器唯一新目录恢复、重建P1环境及验证；无stop/recycle/destroy/new volume、训练/P3、commit/push或重下raw授权。 | 验证最小目录备份与恢复链路；原目录与数据保留。 | Astrafresh复核完成，完整P2 gate覆盖差异由Eddy/Astra裁决。 |


| D-054 | Decision | Confirmed | Astra独立恢复验收通过，P2按Eddy明确批准的目录级恢复方案完成tasks的恢复门；停在commit前与P3范围门。 | 不把验收扩充为未批准的stop/host-loss操作，也不伪称全灾备。 | P8再验或Eddy批准更强恢复范围；quota/host-loss/GPU环境/raw重获风险保留。 |

## P3过夜授权与policy预注册（2026-10-07）

| ID | Type | Status | Decision / Assumption | 原因 | 复核触发条件 |
|---|---|---|---|---|---|
| D-055 | Decision | Confirmed | Eddy批准最多8小时过夜goal，P3→P6逐阶段由Astra协调验收；Sol当前仅执行P3，P7 stretch，截止UTC2026-10-07 02:46:21。 | 保持阶段gate与现有实例，允许必要隔离依赖/官方查证，不新建资源或commit/push。 | deadline、stop gate或用户steering。 |
| D-056 | Decision | Confirmed | P3用只读固定revision train审计，预注册至少1000位like长度>=2用户的工程support门；minhistory1、maxhistory50。 | 门只证明小型工程smoke可行性，不能声称统计效力或benchmark代表性。 | train支持门失败则按tasks停止/预注册composite。 |
| D-057 | Decision | Confirmed | Astra批准最终P3 policy，合同完整规则只在data_contract维护；official validation25/test26取代draft train-tail proposal。 | 固定README与P2inventory确认已有官方三split，无需挪动train尾部。 | source/order证据变更。 |
| D-058 | Decision | Confirmed | P3采用官方ordering assertion + 固定README hash + physical row ordinal与两次读取digest；不声称独立验证隐藏timestamp。 | parquet没有timestamp；subsample官方示例保序，但无全量parent逐行比对证据。 | 原作者纠正顺序声明、schema改变或稳定读取失败。 |


## P4预注册执行语义

D-059：Astra确认P4只消费P3静态冻结artifacts，原producer/source/lock验证守卫不变；P4新增模块自然产生新总fingerprint。Popularity按train like事件计数，含catalog零计数item。Content每个catalog embedding float32 L2 normalize，history按冻结顺序含重复求均值，再L2 normalize；zero item/centroid保留零分并按item ID升序tie，计数但不造feature。固定seed42按SHA256(`42:split:user_id`)排名，每split最多512 query，抽样不读取target/OOV/hit；Recall/NDCG@10/20给all/train-observed/OOV与分母，空slice null。query block32/item block4096，无全矩阵；不调参。完整policy唯一源仍data_contract，P4不改合同。

D-060：Astra批准P4 fix1 score arithmetic为float32 normalized feature输入上的固定float64 non-BLAS einsum reduction（optimize=False），brute/blockwise必须同算术且跨block invariant。不用epsilon聚桶改tie；FAISS仅canonical tie的toy reference，不声称任意近分数float32 FAISS与float64 evaluator逐位相同。完整原因与v1 invalid/fix1证据只读p4_validation/I-032。

## P5预注册确认（Astra，2026-10-07）

只实现单一capacity-constrained greedy balanced hierarchical kmeans：16分支、3语义层、seed42、每node最多10iter；mean Euclidean centers，固定逐维float64 distance，对边distance/item ID/child ID稳定排序填floor/ceil容量，不称全局最优。每leaf按item ID升序加1-position suffix，OOV静态encode拒绝。Compact mixed-radix sorted-key constraint与pinned manifest static reader独立于P5 producer replay；P6改source时不放松P3/P5历史产物的pin/hash/结构检查。完整参数/gates/接口/资源观测限制只读p5_validation；当前阶段验收只读STATE。

## P6预注册接口与训练预算（Astra授权scope）

D-061：在读取任何P6模型指标前冻结configs/p6.json；单seed small encoder-decoder Transformer、训练/optimizer/beam参数与deadline唯一详细定义见p6_validation，不按validation/test调参。PAD/BOS与4个disjoint position namespace、teacher forcing shift、fixed4tokens/noEOS、full-vocabulary logsoftmax后合法prefix约束、global beam裁剪明确。实际step100自有checkpoint保存并训练process退出，新processresume，model/optimizer/RNG/epoch/cursor/source/config/input hashes全部绑定。Budget从原首次started_at起不重置；1000step与45min双上界，不足step目标明确partial_time_bound。只消费P3/P4/P5固定artifact；同P4audit/OOV/item evaluator，不称TIGER忠实复现或方法优势。


## P7收口与状态指针

D-062：在P6正式Complete后，按Eddy过夜stretch授权只复用固定P4/P6审计结果完成P7统一比較与MVP交接；不新增模型/fit/train/tuning/P8，不commit/push或资源生命周期。公平性定义为同一冻结label/history/target/catalog/query与同源features，独立item-level公式和共同slice分母；模型表示及计算成本天然不同，不宣称相等compute预算或直接latency排序。

D-063：协调明确批准docs/data_contract.md仅两处旧阶段状态句改为正式验收/当前状态与历史证据指针，以及“P3阶段不训练模型”；保留首句实际train-only freeze/版本/config及全部label/history/split/catalog规则和冻结artifact/config/source hashes。不重新生成合同，不把状态更正解释为新policy。


## P3-D传统召回合同预注册

D-064：Eddy已批准完整传统召回goal，当前仅P3-D合同冻结，不训练、不读取新holdout指标。新版本 `traditional-like-v1` 追加在data_contract；旧p3-like-v1及producer验证不改。YAML仅机器镜像，manifest单向绑定合同/config及P3/P3-C/P4来源；实际规则唯一指针 [data_contract的traditional-like-v1节](data_contract.md#traditional-like-v1)，不在本日志复制预算或label规则。旧test已用、static content snapshot时点Unknown、official-order非timestamp验证的限制持续保留。

D-065：新consumer必须记录自己实际source/config/lock/input pins；不能要求新源码假装旧producer，也不能弱化旧artifact的producer校验。当前尚无传统consumer实现或训练结果。独立review及Astra正式验收状态只读STATE。

## P4B implementation interpretation — consumed sampling denominator

按协调明确裁决，no-negative门在每次record后、optimizer step之前检查已drawn epoch累计和run累计（含skip），不得借未来draws稀释当前分母，也不以单batch分母替代。epoch开头偶发skip可能早停，即使后续完整epoch比例能降到1%以下；这个tradeoff保留。仅说明冻结合同实现，不修改data_contract/config/manifest。Synthetic best pointer按训练loss用于恢复框架测试；正式best选择仍归P4D validation合同。

## P4C ItemCF 实现裁决

协调明确裁决：cooccurrence neighborhood排除diagonal/self edges；最终candidate保持 frozen seen_item_masking=false，因此seen item仍可通过其他邻居贡献或Popfill出现。Graph从全部train likes的binary user-item关系构建，包括不产生prefix的single-like用户；不得由最近history/model prefixes反推。仅记录本阶段实现解释，不修改冻结contract/config。

## P4D implementation choices before metrics

按协调裁决，content projection与Transformer最终user projection均bias=False；mean模型无隐含projection。初始化/标准Transformer内部/FP32 deterministic attention细节绑定于architecture_spec并在p4d_validation解释。Sharedbatch须实际四模型allpass后freeze，正式selection前不读新的test。来源规则仍唯一data_contract，不复制冻结预算成为第二事实源。


P4D首完整run的verifier发现stats int/JSONstr key表示差异后，协调批准最小独立兼容adapter，原v2 producer与产物保留，不因表示问题重训、不重跑资源pilot。source/operation fresh review通过后先CUDA长epoch恢复与仅原产物只读重验；未取得实际formalverify通过及协调接受前，固定24slot队列保持停止。恢复队列仍同注册配置/batch/预算，执行身份明确记录producer v2与adapter/proof，不能冒称纯v2执行或任意source兼容。

P4D core selection正式冻结：所有预注册seed42配置经过完整verifier后，协调独立重算逐epoch strict选择、各架构config与整体tie规则和共享binding，正式接受固定selection。Best core为ID mean，代表checkpoint保持预注册seed；配置和selection hash唯一见experiment_log。后续仅固定additionalseeds，无追加HPO、无test选择、无因弱结果改recipe；P4E尚未开始。
