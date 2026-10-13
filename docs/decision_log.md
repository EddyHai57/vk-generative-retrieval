密级: 公开

# Decision Log

## 当前边界

本文件记录已确认 Decision 与尚待验证 Assumption，不记录虚构实验结果。

本轮未下载数据、未训练模型、未验证效果。

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
| D-051 | Decision | Confirmed | G3/G4一次性8-slot aggregate草案已否决并保持全部G slot暂停；先读取首slot真实指标、完成既定P4F，再提出不突破D-007 hard ceiling的缩减分阶段方案。 | 原草案超出已确认GPU上限，且提案时尚无首slot指标证据。 | Eddy另行调整D-007或批准新的有限分阶段预算。 |

D-004/D-005与D-030中的Thunder描述保留为历史决策；当前平台按D-049执行。A-003仍Open，其复核改为当前平台的acceptance probe与smoke run。


## P4E / G2 共享分层统计接口决策（2026-10-08）

Astra批准，两位执行Sol均已确认：传统与生成式评估共用完整train-like item frequency与user like count及冻结分层边界。不能用ordered example target frequency或截断history长度替代。原G1合同保持冻结；独立slice-statistics artifact作为新consumer显式输入，绑定train producer/raw身份、schema和外部预期SHA。

唯一producer由传统Sol负责，在P4E既有raw train扫描中同时产出，不追加扫描或实验；生成式Sol先用synthetic统计验证consumer，真实统计须经Astra验收才用于G3。统计缺失或身份不符拒绝，不能把未知值当0。分层边界保持已批准P4E定义；无validation/test定阈值。此决定不授权提前运行server prepare，也不开放G3训练。


### 当前P4E/G3窗口维持串行（2026-10-08）

唯一scheduler用已有证据只读评估，未新采样或GPU试验：原G3 pilot可在设计上承担一次并发探测，但现有global 20%门与P4E门相同，无法保证竞争时只停止G3；缺P4E不同阶段CPU/IO和资源增量证据，当前显存低不能证明净吞吐收益。Astra决定本窗口维持既有串行，不新增并发guard/sentinel框架、不改冻结launcher/release，不追加pilot或训练。此为现有证据下的调度选择，不撤销用户长期双线/资源pilot授权。P4E自然终态及验收后安排原G3 pilot，既有原预算不变。


### P4E实验产物验收与owner退出记录限制（2026-10-09）

Astra接受P4E实验产物Complete并允许P4F规划：九run实际train/verify及summary成功收据、source/input/config/checkpoint绑定、完整指标聚合与预算均通过实际核验和独立review。原detached owner自身wait exit没有历史证据，永久保留null；不以dispatcher或owner-result代替，不声称全部编排进程有成功wait。此运行记录缺口作为accepted limitation保留，不重复训练或第三次replay来补造历史。


### G3一次补充窗口批准（2026-10-09）

Eddy在生成式执行对话01a11625-74a4-7af3-824d-d1d8523b8fd0的human turn 01a11c2d-6cd7-7753-a116-95fc2f4f770b明确回复“完全批准”，新Astra已独立读取核实。落实为一次最多1800s补充wall，仅冻结候选pilot→acceptance，首次新server写入前开始；总200steps/128decode/1resume、旧新累计2GiB不变，保留旧失败及历史预算。唯一传统Sol调度，禁止重复前五stage或盲重启。authorization及批准证据见主仓cache/g3-coordinator-supplemental-v1/，实际运行状态只看STATE/handoff及真实receipt。


### 生成式协调优先级（2026-10-09）

Eddy在新主协调对话直接要求“现在主要协调生成式召回，进度落下太多了”。当前优先G3原pilot终态、正式预算核定和核心训练；暂不派新的传统P4F任务，保留既有验收成果。固定实验次数、单一server scheduler、资源和test gates不变。


### 双线持续推进与自动协调澄清（2026-10-09，覆盖上一条暂缓）

Eddy纠正为生成式更多关注、传统继续，并明确要求两Sol自动与新主协调交流和派工。落实为生成式优先处理、传统不闲置，worker事件回报加主协调10分钟heartbeat兜底。保持唯一server scheduler及阶段验收/预算gate，不扩大运行权限。


Eddy调度修订（2026-10-09）：协调heartbeat vk已由工具update为ACTIVE每5分钟，保留原目标/安全边界；两Sol均已发送持续要求：完成、失败、阻塞必须在结束turn前主动向主协调回报一次READY/BLOCKED/URGENT与task/证据/hash/验证/限制/下一步，不得只在自己final交付；发送被拒则记未送达、不绕过，由heartbeat兜底。生成式当前已交extractor等待真实aggregate，不是未回报。Astra已read_thread核实Eddy在P1 human message01a11c62-e18b-77b0-ba6b-a36ef9d6e0f2直接说“同意，审批通过继续”，P1当前正在按正常审批处理原一次取证；先前拒绝证据保留，执行结果尚待回报，不重复派工或SSH。


P4F依赖metadata验收与artifact审计派工（2026-10-09）：Astra已读报告/evidence并fresh核两交付与5份保存的primary source hashes。接受faiss-cpu1.8.0.post1历史社区PyPI wheel为审计候选，不称Meta官方binary或安装通过。已派传统Sol只在新cache/p4f-wheel-audit-v1做一次精确hash wheel下载（120s/28MiB），ZIP目录与有界license/metadata读取（声明解压总量256MiB门、文本每项1MiB/总4MiB）；不安装/import/执行binary、不SSH、不改源码/固定环境。下一需真实wheel内部许可与成员size证据；缺失/审批拒绝则STOP，不自动retry。生成式G3 formal-cost-review-v2仍active，未重复派工，正式训练仍待成本gate。


P4F native来源补证收口（2026-10-09）：Astra读review并fresh核review.json SHA92a89a2f24c103c11194171801e896a19a1ef54d78c69495c45515cf245f6d6d及报告/sources hashes。保留已证release/gitlink许可与未绑定native build限制，停止legacy考古。范围裁决：AGENTS要求provenance/license审计，未硬性规定完整binary attestation；不可把衍生最高证据标准当永久阻塞，缺attestation不能单独判禁止使用，组件许可不清仍需处理。派传统Sol仅最多两个当前维护upstream CPU版本的metadata兼容性比较及一页风险/选择建议，cache/p4f-maintained-dependency-decision-v1，不下载binary/SSH/安装/改源码。生成式G3成本v2仍active，未重复派工。


P4F维护发行候选裁决（2026-10-09）：Astra读decision并fresh核decision.json SHA8bb3227c8522614d7c76bb68e5542870ab301556a5a34f5ff428c886d53da3fa及md/sources hashes，选择upstream faiss-cpu1.15.1继续artifact审计。仅派传统Sol新cache/p4f-upstream-wheel-audit-v1，一次精确wheel下载120s/20MiB、hash f2c31b7f2f6647eb76829a5cfe3c398fb9346df9f26b1d4db35269c91eb58c33，再有界ZIP/许可文本读取；不安装/import/SSH/修改numpy或torch。runtime声明兼容仍不等于actual ABI/platform通过。生成式G3 cost-audit候选本地实现active，真实运行预算未批准。


Scheduler权限澄清及生成式接续（2026-10-09）：生成式Sol转述Eddy授权两Sol均可承担scheduler；按具备资格与实际负责人分开记录，当前实际server owner仍P1，任何切换须明确交接，不增加monitor。此澄清不是新增90分钟审计预算批准。Astra read_thread发现生成式原本地cost-audit任务在回答中途权限问题后结束turn，未有候选交付或技术stop gate，已明确派原Sol继续同一已授权本地任务；中途问题不替代原工作，不重复创建任务或server操作。


P4F upstream wheel许可范围裁决（2026-10-09）：Astra已读audit并fresh核audit.json SHA53d6faf302ee9669131cec99349ff6bc0b84bb8b57bb1025ea9332f56aa2beee及md/receipt。基于官方upstream精确artifact、Faiss MIT/OpenBLAS BSD notices与已读GCC GPL/LGPL许可，本项目仅内部运行未修改依赖、不再分发binary/第三方源码的范围接受；原native构建/notice对应限制保留，不称全部MIT或exception逐文件已证。GPLv3第2节非convey运行权限见https://gcc.gnu.org/onlinedocs/libgomp/Copying.html，GNU FAQ https://www.gnu.org/licenses/gpl-faq.en.html#InternalDistribution；未来分发/用途变化另审。关闭仅内部使用的许可阻塞，停止无限考古。下一已派唯一实际scheduler传统Sol一次有界platform/dependency只读probe及本地安装gate提案，cache/p4f-install-gate-v1；30s脚本/60s运输/32KiB输出，不importGPU、不安装/写server，审批拒绝不retry。安装仍未授权，生成式本地候选继续。


G3 cost-audit接口gate裁决（2026-10-09）：Astra读interface-review.md SHA668c5e85e7b2f19c7b169975d1f7c2849a5c494480cc5d8816e481010922d65a并fresh核review.json SHA90b243888ecf25380e0edead993473db1650da976198e6717063b386e74c0501。现有evaluation直接写入且无已证aggregate allocated quota，polling/RLIMIT_FSIZE不能冒充1GiB硬cap。批准原Sol继续同目录conditional fail-closed本地candidate，只实现独立于存储的预算/零update/阶段计时/摘要与synthetic拒绝测试，实际storage prerequisite unavailable时部署分支禁用；不可伪造quota凭证或声称完整可执行交付。真实storage门、human新预算与Astra exact-start仍未满足；生产源码/旧pins不改，无server运行。


P4F隔离依赖安装放行（2026-10-09）：Astra全文审install-supplement SHA0a5d13386c99e843da234fbccf9ea331a28abc5aac9f65dd93d748cde2215f9a并fresh核packaging audit/md hashes，选择普通pip有限两wheel+观察STOP，不额外自建writer。明确512MiB allocated/2GiB owned RAM是采样观察停止阈值，接受采样间overshoot可能性，不称硬quota，且不改变G3硬cap。派P1唯一实际scheduler一次安装到exclusive /workspace/vkgr-p4f-isolated-dependency-v1，240s全局含运输/安装/tag/import/收尾，先核工具/root/资源20%门；packaging先tags后faiss，fixedenv不改、无依赖solver/新包/重试，CPU能力检查不构建index或训练。local receipt新cache/p4f-isolated-install-v1，成功尚未回报；自动审批拒绝则准确保留，不绕过。


G3 storage只读核查验收（2026-10-09）：Astra fresh核result SHA2e4e4becb46a5e23dd79746bd5a888c99dbcc293975f83f34472d61eb18bad00、receipt SHAa8155dff5c4798fd4bdd74a72789e9996260025db49425b5bd9e0f7de31455a7及probe hash，读取实际字段。visible overlay无quota选项、相关配置/工具未可用；exclusive配额/1GiB assignment/all-writer逻辑界仍unknown，不推断不存在quota。硬cap未证，停止重复storage探查。下一向Eddy提出明确选择：接受1GiB观察STOP（采样间可能overshoot）及一次90分钟/5040decode/0训练更新的成本审计预算，或保留硬cap另审filesystem方案。尚无批准，conditional入口保持禁用；P1按既定范围接P4F只读诊断。


Eddy新增运行授权与全量准备（2026-10-09）：human明确批准新增90分钟真实成本审计，时间成本不作为阻碍；既提案90min/5040decode/0update/0resume/新增1GiB observation STOP及采样间overshoot授权已获，不再待用户。G3 cost-measure仍待本地READY及Astra fresh/exact-start，随后P1唯一scheduler执行，不跳gate或重启pilot。两Sol已收到授权更新。Eddy同时要求尽早检查全量下载容量；P1完成当前P4F安全交付节点后优先S0只读盘点，cache/s0-capacity-readiness-v1，60s probe/90s transport/64KiB摘要，实测磁盘及inode、官方全量inventory、完整中间产物容量账和20%余量。S0可提前，S1需容量/来源/费用gate及精确计划，S2以后合同/规模pilot不省略；不新增付费资源、不删成果、不commit/push。详见tasks.md现有S0–S6路线；无需因Eddy离线重复申请已授权范围。


## COORD-01 协作流程简化（2026-10-09）
Eddy认可分工并授权征询Sol、写规范后恢复。采用完整任务自主执行、异步READY队列、一次独立验收、15分钟兜底；详细唯一规则见docs/codex_handoff.md（本docs内为codex_handoff.md），落实计划docs/superpowers/plans/2026-10-09-agent-workflow.md。两Sol沿原task恢复，不新建框架、不改变科学/资源预算，不commit/push。


## 2026-10-10 COORD-02：仅更换Astra、职责端到端
Eddy在审阅独立研究后明确认可仅重开Astra，要求STATE/handoff、接手提示词和三方职责持久化。G/P原Sol保留并各自负责本线获批实现/执行/诊断/监控/交付，Astra共同合同/资源准入/实质升级/一次验收；覆盖历史P1唯一调度新作业的安排，旧job不得无确认切owner。两线和定时器仍暂停。本轮只交接，不放行G3新attempt或P4F diagnostic。旧文件逐byte归档，规则唯一源codex_handoff.md。

COORD-02接收完成：新Astra 01a12202-362a-7cb2-80e1-43bcdea63ff7及两原Sol均已ACK；新Astra接任治理，旧Astra退出。vk已转新协调仍PAUSED；实验恢复仍待Eddy明确指令。

## 2026-10-10 Eddy明确恢复：生成式优先，传统收尾

Eddy在新Astra对话直接要求“来我们开始吧，依然重点生成式，辅助传统的收尾”。解除协调及两线授权范围内工作暂停，沿既有两个Sol端到端推进，不重建worker。Astra派G3固定候选一次fresh差异review，G Sol并行整理owner接收与完整执行条件；不将恢复指令解释为新增真实attempt或正式训练预算批准。P Sol获原固定diagnostic脚本一次只读执行准入（server30s/local60s/8KiB，原metadata/header/resource边界），失败不retry，后续范围内本地诊断修复与真实ANN候选准备自主完成；真实ANN预算仍null。旧job先凭真实身份明确交接；无并发证据则资源密集作业串行、生成式优先。恢复原vk 15分钟协调兜底，旧p4d继续暂停；不新增资源、不删除、不commit/push。


## 2026-10-11 — P4H operational-integrity 信任边界

Eddy批准P4H-03B只提供operational integrity：防误操作、错job、重复writer、stale、rollback、fork、partial write与普通篡改。SHA用于内容和operation identity绑定，不是外部身份认证。拥有same Unix UID全部写权限的恶意进程若整体重写所有证据文件并重算全部SHA，可构造语义自洽的新世界；该攻击明确为`OUT_OF_SCOPE_REQUIRES_EXTERNAL_TRUST_ROOT`，不能声称本地自校验可拒绝。若未来要覆盖，必须另引只读介质、不同UID、签名服务或remote transparency log等外部trust root；本轮不新增dependency或remote机制。P4H remote decisions保持BLOCKED，P4H-04/common test/SSH关闭。

2026-10-10｜G3新单attempt明确human批准：v3一次review阻塞关闭后，Eddy对具体执行提问回答“批准这1次有界成本测量”。范围为已审v3固定代码包上传既有A6000独立新目录，G owner先核旧job/输入/至少20%资源余量后自主现场绑定、一次启动、监控、收尾；总90分钟含部署清理、最多5040decode、0updates/0resume、新增1GiB observation STOP可overshoot，失败不重试、不新增付费资源、不formal训练。Astra已发完整条件准入：旧pilot责任接收闭合、最多2次30s/60s轻量只读preflight，原固定source/六成员/输入/新root不变，最终同参数hash现场存证，无逐stage审批。当前尚无实际启动证据；原failed spent不退款。P4F仅本地补证，不与G并发资源密集任务。

2026-10-10｜Eddy睡前授权6小时goal持续管理：03:33:38—09:33:38北京时间，优先生成式、辅助传统，尽量减少Astra token并争取一晚完成。采用原两Sol完整task自主执行、事件回报、原vk唯一15min兜底；Astra只必要准入/一次差异验收/升级，不重复轮询或旧suite。窗口结束停止新派工并汇总，已有作业按已批准包络安全收尾，vk设PAUSED。此授权不保证实验完成、不扩大实际attempt/正式训练预算/资源权限。GPU UUID不一致已定位，Eddy明确未做实例/目标变更；先本地调查来源与可信目标，不修改旧pin绕过。

2026-10-10｜新增一次G3 UUID表示只读诊断human明确批准：先前自动审批拒绝后，Eddy对具体问题回答“批准这一次只读诊断”。固定原可信连接、旧origin<=64KiB及原SHA、GPU name/UUID查询、仅canonical/raw比较bool与表示enum、server30s/local60s/8KiB、无写入/model/训练、失败不retry；本次已工具确认派工送达，不是退还旧2次额度。canonical相等才允许原G在新v4本地最小表示修复/TDD，不改旧origin或科学合同；actual提交仍需v4差异接受，原1次90min成本测量授权未实际消耗。


## 2026-10-10 — 无限时goal的双线并行恢复

Eddy确认现有subset两项目按已讨论路线并排推进，当前资源足够，不默认分批执行；未要求上全量数据。覆盖旧暂停与资源密集默认串行规则，两原Sol各自端到端执行。无需新goal或定时器。并发准入已有human授权，现场联合资源观测仍须真实；原attempt/消耗/科学合同不变，正式训练与ANN具体预算仍需完整候选及实测依据。普通范围内修复不逐步审批，完整task一次独立验收，必要预算或权限升级合并提出。


## 2026-10-10 — 固定v4只读启动检查直接授权

Eddy明确批准一次固定v4 script/input发送至原既有可信A6000并只读执行，具体SHA及30s/60s/64KiB限制见handoff。解决此前auto-review所指出的直接human外发范围授权缺口，原拒绝原因保留，失败不retry。通过后仍在原已批准单次成本attempt内推进，未增加正式训练预算。


## 2026-10-10 — G3首固定slot仅本地执行候选

主协调选择operational-spend-v1方案A作为本地实现范围，未批准48h/20GiB实际投入。方案B六core与G4不提前实现。候选须维持冻结科学配置和完整replay，覆盖formal日志规模的资源监督、实际wait/cleanup、最终封存顺序及pending许可拒绝。完备候选一次验收后由Eddy决定实际运行；CPU artifact审计取消，不依赖缺失旧checkpoint，不新增取证probe。


## 2026-10-10 — 取消逐次human常规审批，Astra在既定包络内决策

Eddy原话：“为什么都要我来批准，不需要我批准你觉得可以就行”。当前两subset项目的常规修复/只读诊断/候选验收及既定预算包络内的新attempt由Astra具体批准，覆盖此前每attempt直接human最后许可要求。资源/科学/范围及spent约束不取消，不自动无限retry/加预算/开全量/新付费资源/删除。主据此批准已审P4F第二次固定attempt，不再索重复human确认；真实工具权限拒绝不绕过，必要时如实升级。

## PG-R01-P-RECOVER｜自主继续授权（2026-10-13）

Eddy直接回复“不用管限制，一直做下去”，覆盖此前attempt3/no-retry/旧06:00:55截止作为继续阻塞；普通诊断、修复和接续不再逐步问确认。原话固定Main main05/direct-human-autonomous-continue.json SHA f0d48747…；旧queue、登记、全部spent和失败产物不覆盖/退款。主执行选择下一attempt04有限4h、同charged root旧＋新20GiB、0resume/no新资源/no正式训练；每个新作业仍first-write UTC+monotonic freeze、fresh shared/headroom/identity/hash guard与真实wait+CLOSED。失败先诊断修复，不盲目重复提交。科学数据合同、test gate、同资源、一次完整delta review与真实验收继续遵守。部署与最新状态见STATE及main05/attempt04-stage-acceptance.json。

## P/G 两线独立真并行（2026-10-13，Eddy直接决定）
P与G是两个独立推进的项目，唯一真正融合点是后续协同召回；此前不设互相等待的汇合点，只有某线完全收口才等待另一线。硬件允许同时运行，故按原计划分别推进：G补seed（core03 d128/LR0.0003，seed2024→2025）与P4H-05恢复/S06同时运行。约束：P已冻结gate把未登记GPU进程视为unknown并STOP、已登记G异常终态也STOP，故顺序为先登记并启动G run，再启动P recovery01并把该live G run写入P的G名单；科学合同、test gate、各run有限预算、single owner（Kiro）、不新建认证连接扰动live gate不变。此条修正旧“P4G/G5共同冻结为汇合点”的表述。

## G seed2025 与 seed2024 同机并行（2026-10-13 19:5x，Eddy直接决定）
Eddy批准在同一A6000上立即提交 selected-seed2025，与运行中的 seed2024 并行（原话 main05/direct-human-seed2025-parallel-20261013.json）。理由：G为单核CPU瓶颈（实测train进程98.5% CPU、GPU 0–1%），64核/48GB下并行不互相拖慢；两seed与seed42同硬件，避免换机后的硬件可比性问题。caps不变（每run 108000s/20GiB/1 attempt/0 resume）。替换旧“先2024后2025”的顺序。admission最小改动：只放行已登记 seed2024 owner 的后代进程，其余vkgr进程仍STOP。
