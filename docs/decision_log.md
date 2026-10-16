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

## 工作流规范化与服务器迁移（2026-10-13 20:xx，Eddy直接确认）
Eddy确认按工业界流程统一工作流，写入AGENTS.md顶部“工作流硬规则”（覆盖旧“未经要求不commit/push”等条款）：Git唯一代码来源与按SHA部署、public repo push前扫描与路径占位符（方案B）、作业detached与SSH解耦、gate只管自身进程树与资源、两线独立并行、单owner、review仅在不可逆节点前。服务器迁移：subset阶段换3090/≥16vCPU/≥64GB RAM/500GB；备份走服务器直传，代码走GitHub、数据重下；P4D/P4E仅保留P4G冻结引用的best checkpoint（调查报告 cache/migration-checkpoint-keep-v1/report.md）；旧机在G seed2024/2025收口且传输核对后由Eddy销毁。计划见STATE迁移进度节。

## INFRA-01 D0：G 冻结依赖文件的交付方式（2026-10-13 23:4x，Eddy“按照建议”）
背景（design.md R0）：G1 合同钉死 git_base=0376d9a；公开脱敏改动了被 P3-D 合同按 sha 绑定的 docs/p3c_validation.md（c89b6466→5449378c，原字节在 cache/pre-sanitize-backup-20261013.tar 保留）；provenance/g0-v1 与 cache/g3-formal-invoke-v2.py 未入 Git。决定：①g3-formal-invoke-v2.py（2ac33aee…，扫描干净）commit 进 Git；②原版 p3c 文档与 provenance/g0-v1 作为 sha 锁定输入走数据通道，不进 public Git，launcher 按 sha 校验；③现有 G 合同不重冻，现行 G 实验用 frozen-base overlay（base 0376d9a + 目标 SHA 覆盖 + 3 份锁定文件）部署；新 G 实验（G-TOK-01 起）另立新版本合同，锁定代码内容 hash 而非 git HEAD。教训：脱敏前必须先排除被合同/sha 绑定的文件。

## 实验卡 Batch1 决定 + history 版本升级（2026-10-14 00:xx，Eddy直接决定）
1. **history 改用观看/observed 历史**：Eddy 判断 like-only 历史不符合业务逻辑（数据合同 data_contract.md:107/167 的 MVP 选择是为免定观看阈值）。作为新版本 history-v2（旧 like-history 合同与结果冻结不改），P/G 两线同步；后续 Batch1 的 HP/NEG/TOK 实验均在 v2 上进行，避免先在 v1 调参再重做。v2 的“观看”定义优先复用 P4E `watch_history` 已验证的口径，冻结前核对并写入实验卡。
2. **max_epochs 提高到 100**（两线同时，patience3 不变，早停防浪费），属于新预算版本，旧结果不重算。
3. **history 长度做对比实验**（例如 50 vs 100，可含 20），有效就拉长；需各长度重做 prepare，两线同步。
4. 其余拍板点全部按卡内推荐：G4 beam 诊断用全 2520；G4 不等补seed终态；logQ 只 α=1.0；N0 不一致改用 N0 对照；每线 8 名额对等口径；RQ-VAE 深度 3→4；G-TOK 2×2；RQ-VAE 训练长度按论文核实值预注册；内容向量用官方 64D 并在报告注明偏离；G-TOK-02 等 G-TOK-01 结果再定。

## 实验卡 Batch1 冻结（2026-10-14 01:xx，Eddy“可以，开始”）
§8 六条全部按推荐：新 P run 只留 best+last；HIST-LEN 两线联合规则；L100 的 D0 门槛（>50 占比<10% 则取消）；新代码只在 v2 跑 core03 三 seed；加不计数 P-REPLAY；G 对等名额用 TIGER 论文规模、在 INFRA-02（GPU 批量解码）就绪后提交。服务器 CPU 配额 10.24 核为准。新增 INFRA-02：GPU 批量 beam 解码（结果逐 query 等价）。

## 夜间自主推进授权（2026-10-14 01:xx，Eddy原话“如果我睡着了还没来review你直接进行后续的训练”）
Eddy睡眠期间，Kiro在INFRA-01与Batch1代码通过自动独立review和服务器对拍/断线演练后，可直接按冻结实验卡提交训练，不等Eddy人工review。边界不变：不读test、不删数据/不销毁实例、不force push/不进main、caps与名额按冻结卡；任一验证不通过即停并记录。
- 补充（2026-10-14 01:5x，Eddy）：验证未通过时，Kiro 可按自己的修复提案自行审批并尽量修复后继续；确实无法修复才停下等 Eddy。不可逆边界（读test、删数据、销毁实例、force push/进main）仍需 Eddy。

## Batch1 review 自审批（2026-10-14 03:xx，Kiro，按 Eddy 夜间授权）
review（.agents/tasks/batch1-review.md）APPROVED，2 MEDIUM 处理：①G 撞 16h cap 风险 → G 的 W1 run wall 上限改为 30h（=既有授权的每 G run 108000s），并且训练时就开启分片验证（W=4）。g-hist-v2-l50-s42 同时作为 R3 对拍对象：它训练完后，用同一个 best checkpoint 分别跑 w1 和 w4，对拍 §2.6 projection；对拍不一致就作废这批 G 结果、改回串行重跑。②verify 只重评 best → 属于 Eddy 8.1 已接受的口径；作为低成本加固，后续对 last.pt 加一次重评，不阻塞本批。

## 特征工程分批推进（2026-10-14，Eddy“可行，分批测试”）
采纳 docs/superpowers/specs/2026-10-14-feature-plan.md：先只读审计（place/platform/agent、train_interactions_rank 来源、author 覆盖、用户属性边界），再按 F1→F9 分批、一次一个变量、seed42 筛选、有效补 3 seed。召回只做塔内特征与组合，用户×物品交叉留给精排。G 同款特征在 INFRA-02 启用后按对等名额做，特征定义两线共用一份。

## Batch1 选择点（2026-10-14 11:3x，按预注册规则机械执行）
L* = 50（HIST-LEN 联合规则：L20 两线增量均 <0.002，L100 P 变差）。P anchor = p-neg-n1-logq（id_mean + in-batch logQ，R@50 0.1472），FEAT-01 F1–F3 以此为对照；G anchor 暂为 core03 v2 L50（G-ARCH 未跑）。

## 2026-10-14 下午：anchor、INFRA-02 启用口径、下一批并行任务（Kiro 自审批，按 Eddy 夜间/持续授权）
1. **P anchor = F3（N1 logQ + 观看时长/比例分桶 embedding，R@50 0.15556）**，依预注册 ≥0.002 规则；F4 起叠加其上。补 seed 与 F3-masked 对照已排（feat01c-seeds、FEAT-02）。
2. **C0 口径**：只对 unsupported 历史事件加同一个 1×64 常数向量（supported 事件不变，side 值一律不读），不是“所有事件加常数”；后者等价于给用户向量加偏置，是另一个变体，未跑。
3. **INFRA-02 启用口径（自审批）**：GPU 批量 beam 解码仅在 `VKGR_DECODE_BATCH=1` 且 L≤50 时启用（逐位相等已在 best/last 两个 checkpoint 的全部 2520 query 上对拍）；batch>1 不相等，**不开任何容差**（改变评估口径需单独的科学决定，不在授权内自行批准）。批量入口 `vkgr.generative_b1_batched`；用它训练的 run 必须再用逐条解码 verify 一次。每个 G 作业按 1 核声明（此前 W=4 分片声明 5 核导致只能并行 1–2 个），CPU 配额 10.24 核、调度器留 20% → 同时约 8 个 1 核作业。
4. 并行启动（均含 review 门禁，再部署提交 seed42）：FEAT-02（P：F3-masked、F4 新近性、F5 场景、F6 作者+内容）、GFEAT-01（G：F2/F3 同款特征，含 none 回归对照）、GTOK-01（RQ-VAE T1–T3，论文超参已对照 arXiv 核实；新 SID 契约版本隔离旧契约）、GARCH-01（论文规模 A1/A2；U9 已解：T5 风格 6 heads×64 独立 head 维度，inner dim 384≠d_model）、feat01c-seeds（F3/N1 补 seed）。HP 类实验（P-HP/G-HP）推迟到特征与 tokenizer 定型之后，避免预算名额先花在会变的 anchor 上。
5. 特征落地后若 P 的 F6 让物品塔对 unsupported item 产生非零向量，要在 target 切片上单独报告；winner 采用前做一次针对泄漏的 review（采用 anchor 前的不可逆节点）。

## 2026-10-14 15:4x：FEAT-02 runner 重写事件（更正 + 处理，Kiro）
1. 更正：此前把 feat02-p 工作树里未提交的 runner 改动称为“外来改动”，不准确。FEAT-002 coder 的自述是：15:25 它用 fs_write 覆盖了已提交的 a14a3d0（覆盖前没重新检查文件是否存在），两份功能等价；没有第三方写入。来源仅为 coder 自述，我没有独立取证。
2. 处理沿用裁决 B：重写版单独提交为 b383476（refactor，常量改名 PIN_PAIRS / REQUIRED_PINS），不丢弃任何改动；coder 同时补回重写版漏掉的 top50 的 p[:50] 截取并加了断言；91 tests green。
3. 影响面：只在 feat02/p 本地分支（未 push、未部署），没有任何 run 用过这两版；部署前的 review 门禁会覆盖 b383476。

## 2026-10-14 16:3x：GTOK-01 RQ-VAE 的执行口径（Kiro 自审批，回应 FEAT-002 findings）
1. 预注册的 Adagrad lr 0.4 不改。coder 在玩具数据上看到塌缩：三层只用了 1–2 个码。真实内容上会不会塌缩是 UNKNOWN。fit 在第 2,000 步和最终步记录每层利用率，只做诊断，不提前终止；成功与否仍按卡上 ≥80% 判定。失败就记为负结果，要重试，得先在这里写决策，并换新 attempt id。
2. 依据：TIGER 论文已用第一个 batch 的 k-means 初始化码本来防塌缩，计划已照做（plan §2.1）。如果在训练中途发现塌缩就换超参，等于事后调参，破坏 P/G 调参预算对等。
3. gtok01/g 上有两条不是 coder 主动提交的记录：4cf8195 和它的 Revert 3a5f2f5，净效果为零。保留，不改写历史（分支尚未 push）。

## 2026-10-14 19:0x：F4 泄漏专项 review 结论与采用为 anchor 的门槛（Kiro）
1. 结论：feat02/p 94eb0bc 上的 F4（位置桶 + 周间隔）泄漏专项 review 为 **APPROVED**（0 HIGH/MEDIUM，5 LOW；.agents/tasks/feat02-f4-leakage-review.{md,json}）。依据：周间隔锚点是历史最后一个事件，query 时刻和 target 行都不进特征；week 整体平移后用户向量逐位不变；分桶边界只来自 train（2,959,906 个 train 事件）；独立 oracle 在 3,499 个训练例子上与 build_observed 零差异。
2. 解读限制（reviewer LOW，采纳）：+0.0127 更像“历史长度/时间跨度先验”，而不是近期加权。validation 里 61.9% 的 query 历史超过 50 条，训练例子是 43.0%，两边历史长度分布不同。
3. **F4 采用为 P anchor 之前必须满足**：(a) seed 2024/2025 补完且 verify（feat02c-f4-seeds 已入队）；(b) 按历史长度分层的 ΔR@50 与配对检验（已排只读分析任务）。(a)(b) 之一不支持，就不采用，F3 继续作 anchor。
4. 其余 LOW，不阻塞但记账：评估路径缺端到端不变性测试；行序假设 I-012 仍 Open；真实数据上的 oracle 复核未做；F4 seed42 的 verify 在 review 时未完成（现已 verified）。

## 2026-10-14 19:3x：P anchor 由 F3 改为 F4（Kiro 自审批，依 16:3x/19:0x 两条已记录的采用门槛）
1. **决定：P 线 anchor = F4**（id_mean + N1 + F3 + 位置桶 + 周间隔；代码 feat02/p 94eb0bc，配置 configs/feat02/p-feat02-f4-recency-s42.json）。依据：三个门槛都满足。(a) 3/3 seed 都比 F3 高 0.010–0.016，均过 ≥0.002；(b) 逐 query 精确 McNemar 3/3 seed 都 p<0.01（Bonferroni 后仍 <0.05），bootstrap 区间都不含 0；(c) 泄漏专项 review APPROVED（0 HIGH/MEDIUM）；历史长度分层不支持 reviewer 的“增益来自 >50 条历史”的担心。
2. **仍须保留的措辞**：F4 的机制没有归因，对外只称“历史长度 + 时间跨度先验”，不称“近期加权”。F4 s2025 的 verify 未完成时该决定为暂定，verify 通过后转正；verify 若失败则撤销，F3 继续作 anchor。**19:4x 更新：feat02c 的 2 个 verify 均 verified（s2024 / s2025），决定转正，不再是暂定。**
3. **后续**（P 线）：(i) 不计名额的机制拆分对照 position-only / gap-only，把“长度先验 / 活动跨度 / 近期加权”分开；(ii) 真正检验新近性需要另一个变量（可学习的按新近性分桶的池化权重），单独立卡；(iii) 用户级活动汇总特征（特征计划的用户静态一类）看是否能覆盖 F4 的信号。HP 类实验（P-HP）仍按 16:3x 的决定推迟到特征定型之后，不因 anchor 换成 F4 提前开。
4. 记账：FEAT-02 的 F4/F5/F6 三个名额已用完；F3-masked 与 F4 补 seed 不计名额。

## 2026-10-14 20:3x：调参与训练并行；tokenizer 先在小模型上筛（Eddy 批准“按你的建议来，训练是训练，调参是调参，并行开始”）
1. **GTOK-01（选项 A，偏离冻结卡，记录在此）**：T1/T2/T3 的 tokenizer 拟合与审计照常先跑（与生成模型无关）；拟合通过的 tokenizer **先在小模型 core03 上训练 G**，作为“tokenizer 筛选”。卡 4 原文是“T1–T3 的生成模型 = 卡 0c 选出的 G anchor 模型”；如果 A1 成为 G anchor，赢的 tokenizer 再在 A1 上补跑一次，那一次记为“确认 run”，不计名额。代价：tokenizer 与模型规模可能有交互，core03 上的排序不一定能搬到 A1。T1–T3 仍计 3 个名额。
2. **P-HP-01 现在开，anchor = F4**（不等 FEAT-03）。理由：温度、维度、学习率这几个旋钮与特征大体独立；P 作业实测 10–15 分钟一个，6 个可以同波。撤回 16:3x“HP 推迟到特征定型之后”中关于 P 的部分。若 FEAT-03 改了 anchor，就拿 P-HP 最优的 1–2 个旋钮在新 anchor 上复核（不计名额）。名额：6 个单旋钮 + 1 个条件组合 = 7 个，取值按卡 3A。
3. **G-HP-01 现在写代码，anchor 预定为 A1**。入队门槛：A1 seed42 训练终态 + verify 通过，且 R@50 ≥ core03 anchor 0.0754 + 0.002（卡 0c 的预注册选择规则，按 seed42 判定）。不满足就停，以 core03 为 anchor 重新规划。旋钮按卡 3B 的“改动方向不变”规则换算到 A1 上：G4 改为 2+2 层（往下探）；A1 的 dropout 已是 0.1，所以 G5 改为 dropout 0；G6（SID 深度 +1）依赖 RQ-VAE 深度 4 的拟合，推迟到 GTOK-01 之后，仍在 7 个名额内。撤回 16:3x 中关于 G-HP 推迟的部分。
4. **显存声明**：以后新入队的 core03 规模 G 作业声明 3072 MiB（实测 2006–2046 MiB）；正在跑和已入队的不改。A1 规模的作业仍声明 6656 MiB（实测 5690 MiB）。
5. 模型使用：平时 Sonnet 5.5，关键节点由 Kiro 提醒切 Opus 5.5；Eddy 在家时全程用 Opus 5.5。

## 2026-10-14 20:5x：FEAT-03 设计稿的 4 个待决项（Kiro 自审批，依 Eddy 20:3x“按你的建议来”）
1. **名额**：批准 FEAT-03 新增 2 个 P 线特征计数名额（Q2 新近性池化 `f4_recency_pool`、Q3 用户级活动 `f3_user_activity`）。Q1a/Q1b（只加位置 / 只加周间隔）为不计数诊断。P 线特征计数累计 3 + 3 + 2 = 8。**对等**：G 线特征名额同样保留到 8，已用 2（GFEAT-01 的 F2、F3），剩下 6 个留给以后把 P 线有效的特征移植到 G（包括 A1 上的重测）。
2. **Q3 去掉“最近一次事件距查询时刻的周数”字段：接受。** 这条更正我自己的任务书：任务书里说“查询时刻服务时本来就知道”，但训练样本的查询时刻只能取 target 行的 week，等于把“离下一条事件多近”喂给模型，和 label 贴得很近；在 validation 上它又是“离截止周多远”，含义不同；它还会打开 F4 泄漏 review 刻意关掉的日历时间通道。这个字段推迟到请求时刻口径审计（F9/C2）通过后另立卡。
3. Q1a/Q1b 无条件补 seed 2024/2025（4 个不计数 run），归因只用三 seed 均值：接受。
4. Q2 的空结果接受写成“在 lr=1e-3 与 best-epoch 预算内未检出”，同时记 logit 轨迹旗标；不额外预注册 logit 增益变体。

## 2026-10-14 21:2x：两个小裁决与一处口径更正（Kiro）
1. **GTOK-01 公开历史卫生（review L1）**：gtok01/g 未 push 的提交 24f3306 里，测试词表直接写了 `HelloBike`、`/Users/` 等字面量；HEAD 树已经是干净的。不改写本地历史：本地分支 gtok01/g 原样保留（含 4cf8195 / 3a5f2f5 / 24f3306，作为过程证据）。push 时另建分支 `gtok01-pub/g`，从 2c8882b 起只放 1 个提交，内容等于当前 HEAD 树，commit message 列出被压平的本地提交 SHA。服务器部署、run binding 都用这个公开分支的 SHA。push 前的扫描范围改成 `git log -p <base>..HEAD`，覆盖要推送的全部历史，不再只扫树。这条撤回 16:3x“4cf8195 及其 Revert 原样推送”的做法（该裁决早于 24f3306）。
2. **P-HP-01 的 H7 组合配置**：由脚本生成后，必须先提交进 Git（php01/p 上一个只加配置的小提交），重新 freeze `--write/--check`，再入队。不允许只用 sha 钉住却不进 Git（工作流硬规则 1：Git 是唯一代码来源）。
3. **口径更正**：之前多处写“0.002 门槛 ≈ 5 个命中”，不准确。n=2520 时 5 个命中 = 0.001984 < 0.002，**门槛实际是 ≥6 个命中**（6/2520 = 0.002381）。已有判定都按 ΔR@50 ≥ 0.002 计算，没有一个落在 5 个命中上，所以以往结论都不变。

## 2026-10-14 22:0x：排程体系 + 四个提前做的代码项目（Eddy 批准“做一个可维护的排程表，后面全量也能用；同时开始四个项目代码”）
1. **SCHED-01（排程体系）**：一份可维护的排程表，加一个只读检查脚本（容量、剩余活量、预计空闲时刻、资源占比，剩余活量低于门槛就提醒），再定一条流水线规则：一批入队的同时开始写下一批的代码；预先登记不计名额的“填空作业”。脚本进 Git（ops/tools），排程表进 docs/schedule.md（新事实源：排程与资源，Kiro 是唯一 writer），全量阶段沿用。
2. **G5-PREP（两线融合代码）**：沿用 tasks.md #12 与研究路线 G5 的预注册骨架：固定总候选预算、RRF k60、final50、等权对照，并报告 coverage、overlap、新增命中、成本。具体用哪几路，按 v2 的现状重新预注册，冻结后才读融合结果。本批只写代码、做合成测试、写预注册卡，不跑真实融合。
3. **GFEAT-02（F2/F3 搬到 A1 上）**：计 2 个 G 线特征名额（G 线特征名额 8，已用 2，再用 2 后剩 4）。入队门槛与 G-HP 相同：A1 seed42 verify 通过且 ≥196 hits。
4. **G6-PREP（语义 ID 深度 +1）**：变深度 codec + 当前 anchor SID 的深度 4 版本（BHKM 16 叉 4 层）+ RQ-VAE 4 层配置。review 通过后只允许入队 tokenizer 拟合与审计这类不读 validation 的作业，当作填空作业；G6 的训练等 G-HP 前 5 个旋钮和 GTOK-01 的结果出来后再入队。
5. **TEST-01（读 test 的工具，闸门默认关闭）**：只写代码和合成测试，**不在真实 test 上运行**。闸门要求 permit 文件里有 Eddy 的原话批准、冻结候选清单的 hash；只允许读一次，读后上锁，第二次拒绝。

## 2026-10-14 22:4x：G5-PREP 设计稿的两个待决项 + A1 seed42 verify 补入队（Kiro）
1. **G5 融合总预算 400 → 200：批准。** 理由：v2 现在最多 3 路（P anchor、G anchor、Pop），G 的冻结 beam=100、每 query 至多 100 个合法 item；总 400 时两路要各 200，G 填不满，P+G 与 P+Pop 的池子不一样大。总 200：两路各 100、三路各 66（剩 2 个空位不分配）。这是对 tasks.md #12 原“总 400 slots”的偏离，原因是路数变了，已记录；k=60、final=50、等权、不做权重或 k 搜索都不变。
2. **SASRec-v2 不进入本卡：批准。** 它比 P anchor 低约 0.06，且没有现成导出器；留作 backlog，另立卡。
3. 设计稿里的 H1/H2 门槛按整数命中算（≥6 hits，Bonferroni 后 p<0.025），与 21:2x 的口径更正一致。
4. **A1 seed42 的 verify 没有入队：发现并补上。** A1 seed42 训练 21:30 前后就 succeeded，但 GARCH-01 workflow 的轮询卡在 `watch:wait`，一直没有入队 verify；这会同时挡住 G-HP-01、GFEAT-02 的入队门槛和 A1 补 seed 的 verify。22:4x 我自己补入队：`g-garch01-a1-lr3e-4-s42-v1-verify`，由 A1 训练作业派生，逐字段与 GFEAT-01 的“训练作业 → verify 作业”差异集合做了交叉断言（argv、config、disk_gib、env、inputs、run_id、wall_hours 七个字段，其余全部不变），串行解码、1 核、wall cap 2h。脚本 main05/kiro_b_enqueue_a1verify.py（先 dry-run 再写）。

## 2026-10-14 22:5x：G6-PREP 的未决项 U4（Kiro）
G6 的 RQ-VAE 4 层版本只对应 T1（RQ-VAE 3×256 → 4×256，即 `SID_RQVAE_B256D4_S42`）。若 T1 的拟合没过利用率 ≥80% 门槛，就不拟合、不入队 RQ-VAE 版本的 G6，在报告里记“无”；T2 赢而 T1 输时也不另起 `SID_RQVAE_B16D4_S42`（那是新的变量，需要新名额）。G6 保底只有 BHKM 16 叉 4 层这一版，它在 G-HP-01 的 7 个名额内。

## 2026-10-14 22:5x：FEAT-03 补 seed 规则 + TEST-01 五个开放问题（Kiro）
1. **FEAT-03**：Q1a/Q1b、Q2、Q3 四个变体在 seed42 之后**一律无条件补 seed 2024/2025**（共 8 个 run，各约 15 分钟，不计名额）。理由：F3 的增益本身对 seed 敏感（seed2024 上 F3−N1 不显著），单 seed 的 ±5 个命中不能当结论；Q2 还有“预算内未检出”的假阴性风险。解读阈值（16、6、0.4、0.8、0.405、0.1）在任何 run 出结果之前已写死，补 seed 不改它们。
2. **TEST-01 的 Q1–Q5**：
   - Q1 候选清单的 query 侧文件：另立 TEST-00，本工具只算指标，不并入；TEST-00 本身要打开含 target 的文件，所以也需要 Eddy 的批准，现在不做。
   - Q2 permit 防伪：不上 SSH/GPG 签名；改用“读 test 的命令必须在交互式终端里，由人输入 permit hash 前 8 位确认”（agent 非交互运行时过不了）；permit 仍须提交进 Git，且由 Eddy 本人 push。T27 记为残余风险，写进文档。
   - Q3 服务器文件权限：由我在读 test 之前的合适时点给出命令、你来执行；本工具不实现，文档里写明。
   - Q4 账本位置：服务器数据盘和本地仓库 `docs/test_reads/` 各一份（只记 hash，公开安全）。
   - Q5 切片常量：工具里切片定义作为输入参数传入并记录在输出中，最终常量等 G5 卡冻结后再定，不在工具里写死。
3. **A1 seed42 的 verify 已通过**（`verified`，只重放了 best epoch 14，共 17 个 epoch）。G-HP-01 与 GFEAT-02 的入队门槛（verify 通过且 ≥196 hits）因此满足：R@50 0.099206（250 hits）。

## 2026-10-14 23:2x：TEST-01 对设计稿的三处收窄 + 第二轮 review 的处理（Kiro）
1. **接受 docs/test_once.md §7 的三处收窄**：(a) T3 不再把 test 文件的 sha 与 Git 里冻结的 P3D query pin 交叉核对，改为只核对 permit 里的 `test_sha256`，由 Eddy 写 permit 时对照 P3D 记录；(b) T12 不再用 `git cat-file -e` 核对候选的 selection evidence / checkpoint / config 是否在 Git 指定 commit 里（checkpoint 本来就不进 Git），manifest 里的 `checkpoint_sha256` / `config_sha256` 只原样记进结果；(c) run 根不由 permit 给出，`--out` 仍由命令行给，防重读由固定账本承担。理由：它们依赖的“已冻结进 Git 的文件”在当前公开仓库里不存在，硬接不存在的路径只会让工具不可用。T4（拒绝等于 validation pin 的输入）和 T14 的 `force` 已实现。残余风险：删账本才能重读；读 test 前后的账本、锁和 `docs/test_reads/` 记录三处互证。
2. **第二轮 review 新增的 2 个 HIGH，按 review 的修复建议执行**：(a) 伪装的 `--test-file` 可以让硬链接到真 test 的 `--permit` 绕过别名守卫：对所有非 test 输入拒绝 `st_nlink != 1`（lstat 与 fstat 两处检查），补一组把伪装 `--test-file` 与硬链接的 permit、catalog、slices、候选文件、账本配对的测试；(b) 输出粒度允许从单次读重建逐 query 的 target：对每个报告的切片桶设最小组大小常量（建议 ≥50 个 query），读之前能判定的就在读之前拒绝，读之后才知道的把指标置空，并补单例桶的测试。
3. **“路线预注册”要恢复**：设计稿原有的“manifest 里的 routes ⊆ 冻结卡里的 routes 列表”被丢掉了，丢掉后一次读 test 可以评分多个变体并事后挑最好的，同时每个结果都写着 `read_index == 1`。恢复为：permit 里的候选 manifest 的 run 必须是 sha 钉死的切片卡（G5 卡）里 `routes` 的子集；卡未冻结时工具拒绝运行。

## 2026-10-14 23:5x：省 token 的流程收紧（Eddy 指出 subagent 太多，Kiro 同意）
1. **暂停两条今晚没有价值的开发线**：G5-PREP（两线融合）要等 P、G 两条线的 anchor 都定了才能冻结，现在写设计和代码会反复返工；TEST-01（读 test 工具）要到最后阶段才用。两者都不涉及服务器，可随时恢复。暂停时状态：G5-PREP 设计稿 rev2 已通过一轮审阅、还剩 8 个 MEDIUM；TEST-01 代码已提交、第二轮 review 的 2 个 HIGH 的修复方案已写在 23:2x 的记录里，待实现。
2. **轮询**：模型轮询的间隔统一 ≥30 分钟（作业动辄 2–3 小时）。之后新的 workflow 的轮询改用 watch 的 command 处理器（不调用模型，状态变化才唤醒）。
3. **补 seed、补 verify 这类只改配置的小动作**，不再开“实现 + review + 部署 + 报告”四步的 workflow，由 Kiro 用 dry-run 再写的脚本直接做（A1 seed42 的 verify 就是这么补的）。
4. **设计审阅最多 2 轮**：第 2 轮仍有 MEDIUM 时由 Kiro 直接裁决，不再开第 3 轮；不再因为 NIT 开新一轮。
5. **并发开发线上限约 5 条**；新开一条前先看有没有可以暂停或合并的。
6. 部署、轮询这类机械步骤，新开 workflow 时把 effortLevel 设为 low。

## 2026-10-14 23:5x：G anchor 由 core03 改为 A1（lr 3e-4）（Kiro 自审批，卡 0c 预注册选择规则）
1. **决定：G 线 anchor = A1（TIGER 论文规模，LR 3e-4）。** 依据：卡 0c 的选择规则按 seed42 判定，A1 seed42 R@50 0.099206（250 hits）已 verified，比 core03 多 60 个命中，远过门槛；补 seed 也支持：seed2024 +25、seed2025 +26 个命中，三个 seed 方向一致。
2. **补 seed 的 verify（s2024、s2025）运行中，未验收。** 若任一失败，该 seed 不进统计，anchor 决定仍以 seed42（已 verified）为准，同时在 issue_log 记一条。
3. 后续：G-HP-01（A1 上 5 个旋钮）、GFEAT-02（A1 上 F2/F3）本来就以 A1 为 anchor，不用改；G5 融合的占位符 G_ANCHOR = A1 的 seed42/2024/2025；GTOK-01 的 tokenizer 筛选是在小模型 core03 上做的，赢家要在 A1 上补一次确认 run（不计名额，22:0x 第 1 条已记）。
4. 称呼：对外说“论文规模的 TIGER 结构、我们自己的语义 ID（k-means）与 64 维内容向量”，不说“复现 TIGER 论文数值”。

## 2026-10-15 00:0x：P-HP-01 收尾，P anchor 暂不换（Kiro）
1. **P anchor 仍是 F4**。P-HP-01 的 winner H1（温度 0.10）过了预注册门槛（+15 hits），但与 F4 的逐 query 配对检验不显著（McNemar p=0.203，CI 跨 0），且只有 seed42。F4 当年转正的标准是 3/3 seed 过门槛加配对显著加泄漏 review，H1 现在一条都没满足。数字见 experiment_log 的 P-HP-01 节。
2. **H7 组合 void**：只有温度一个旋钮有赢家，条件组合未触发，不入队，不占名额。P 线 HP 名额按 6 个单旋钮计（7 个里的组合名额作废）。
3. **FEAT-03 保持在 T=0.15 上跑**。理由：FEAT-03 的 4 个变体是相对 F4/F3 的归因对比，同一温度才可比。若 H1 之后转正，按 20:3x 第 2 条在新 anchor 上用 T=0.10 复核（不计名额）。
4. **H1 补 seed 2024/2025 待 Eddy 点头**。冲突点：卡 1.4 写“入选者补 3 seed 移到全量数据阶段做”，而 F4、A1 转正时都补了 seed。补 seed 不计名额；成本是 2 个训练 + 2 个 verify，P 作业约 22 分钟、1 核、GPU 约 1%，但需要一个只加配置的 Git 提交、重新 freeze 和部署。
5. **遗留**：本机 `/private/tmp` 下 php01_baseline.log、php01_r1.log、php01_r2.log、php01_r3.log、php01_rdone 五个文件不是本步写的，未删除，并入最终清理清单给 Eddy。

## 2026-10-15 00:2x：G anchor 正式确认为 A1（Kiro）
A1 三个 seed 的 verify 全部通过（数字见 experiment_log“A1 三个 seed 验收完成”）。G 线 anchor = A1（TIGER 论文规模，4+4 层、6×64 heads、d128、FF1024、dropout 0.1、LR 3e-4）。此前“若某 seed 的 verify 失败就丢该 seed”的条件未触发。提醒：此前在 core03 上测的 G 特征（F2 稳定、F3 不稳定）需要在 A1 上重测，由 GFEAT-02 承担。

## 2026-10-15 01:4x：显存声明按实测重算（Kiro）
1. **现象**：G-HP-01 的 G1–G5 五个作业同时在跑，声明 VRAM 合计 38144 MiB，准入线 39312 MiB（总量 49140 × 0.8），只剩 1168 MiB。P 的 2 个 H1 补 seed verify、2 个 FEAT-03 cudasmoke、GFEAT-02 预检、GTOK-01 T3 训练共 6 个作业因此排队。同一时刻实测 VRAM 合计约 31 GiB，GPU 实际还有约 18 GiB，GPU 利用率均值 95%。
2. **原因（我的口径偏保守）**：P 作业我声明 3072 MiB，而实测峰值约 640 MiB。`ops/SCHEDULING.md` §5 第 4 条的规则是“声明≈实测×1.15，向上取整到 256”。
3. **裁决**：以后新入队的 P 作业（训练和 verify）声明 1024 MiB（实测 640，留余量给 Q2/Q3 的新增表）。FEAT-03 的 12 个真实 run 先用这个值。已入队的作业一律不改、不撤回（队列没有改声明或撤回的工具，手改 pending JSON 风险不明）。
4. **代价与风险**：只剩 1168 MiB 时一次仍只能放进 1 个 1024 的作业，所以这个改动在 G-HP 作业结束前只能让 P 作业串行跑；作业结束后才会并行。若实测超出声明，会让账本少算，GPU 实际余量约 18 GiB，OOM 风险低但没有消除。
5. **不做**：不调 HEADROOM（20%）；不改 G-HP 已在跑的作业；不为抢位撤回 G-HP 作业。

## 2026-10-15 02:0x：GTOK-01 的 RQ-VAE 重试（Eddy 批准“按工业化成熟的方案做”；Kiro 记录）
1. **背景**：T1（RQ-VAE 256×3）和 T2（RQ-VAE 16×3）拟合跑满 300000 步仍 gate_failed（R1/R3/R4），逐层只用了 1、1、2 个码，损失停在约 0.811，与 K 无关。读法是码本塌缩（推断，原因未验证）。数字见 experiment_log“GTOK-01 T1/T2（RQ-VAE）拟合 gate_failed”。
2. **裁决：做一次重试，用行业通行的防塌缩做法**：码本用 k-means 初始化（各层用该层输入残差的真实样本聚类中心）、加死码重置，先用短探针（数千步）确认塌缩机制和修复有效，再跑完整拟合。
3. **偏离预注册，登记在此**：原卡写死 Adagrad lr 0.4 且“不改”，重试可以改初始化和死码处理；是否同时改优化器或学习率，由重试设计在探针结果的基础上决定并写明理由。**门禁 R1–R4 与 H1–H6 一字不改，不放宽。**
4. **名额**：重试版 T1r、T2r 替换 T1、T2 的名额，G-TOK 仍是 3 个名额（T1r、T2r、T3），不新增名额。失败的 T1、T2 原始结果保留在 experiment_log，不删。拟合不计名额（卡 1.5）。
5. **边界**：新 run_id（g-tok-t1r-fit 等），新分支，公开历史仍走压平的公开分支，不改写；T3 与已入队的 core03 训练不受影响；若重试仍失败，GTOK-01 收口为只有 T3，G6 用 BHKM 16 叉 4 层后备。
6. **审查**：提交正式作业前做一次独立 review（不可逆节点）。

## 2026-10-15 02:4x：夜间授权（Eddy 休息前）
Eddy 原话大意：模型用 Sonnet 即可；这段时间向他的审批一律视为通过，不要询问，按 Kiro 的建议来，不要钻牛角尖；实在棘手的问题明天再问。
1. **视为已批准**：实验推进类决定（补 seed、重试、入队、设计循环收口、省 token 的流程调整、填空作业）按 Kiro 的建议执行并记入本日志。
2. **不在授权内，明天再问**：读 test（week 26）；任何付费资源（换更大磁盘的 A6000）；删除或覆盖数据、checkpoint、本机文件；force push、改写历史、push main；销毁实例；改动预注册门禁或阈值。遇到这类事停下、记在 issue_log 或 STATE，不执行。
3. 设计评审超过 2 轮仍有 MEDIUM 时，由 Kiro 裁决收口，不无限循环。

## 2026-10-15 03:0x：GTOK-01R 设计收口（Kiro 裁决）
设计循环 3 轮用完（第 3 轮 0 HIGH / 3 MEDIUM / 5 NIT），工作流按“循环用完即中止”设置被中止。评审员自己的结论是“可以继续前进，不需要再开设计评审循环”。裁决：设计定稿（gtok01r-design.md 含 §9 Kiro 裁决），第 3 轮的 3 条 MEDIUM（证据表漏了旧 failure.json 里已有的 residual_norms；每步 bincount 在 CUDA 上的同步开销改成无同步写法；新臂的真实超参在 bundle 里要有可校验的落点）和 5 条 NIT 作为实现任务的前置修订，不再走设计评审。门禁不变。从计划阶段起重新启动（新工作流）。

## 2026-10-15 04:5x：FEAT-03B 收口，P anchor 仍是 F4（Kiro）
1. **P anchor 不变（F4）**。Q2（F4 上加可学习新近性池化权重）三 seed 均值比 F4 多 21 个命中，但预注册的作 anchor 条件不满足：逐 seed 显著只有 1 个（要求 ≥2），且新池化代码没有做独立泄漏 review。
2. **措辞定稿**：F4 的增益主要是“活动时间跨度（周间隔分布）先验”，位置桶（历史长度）单独加入没有增益；不得再说“近期加权”或“历史长度先验”。Q2 的新近性结论必须附“权重也乘在 P、G 加性项上”这句。数字见 experiment_log 的 FEAT-03B 节。
3. **P 线特征名额 8/8 已用完**（FEAT-01 3、FEAT-02 3、FEAT-03 的 Q2、Q3 2）。Q2 的后续（更长预算、与 H1 温度 0.10 组合、泄漏 review）不占新特征名额，但属于新的实验设计，留给明天（需要 Opus 解读）。
4. **机制故事的当前状态**：周间隔直方图是 F4 增益的主体（Q1b 拿走约 90%）；可学习新近性在此之上还有增益（Q2）；用户级活动桶（Q3）有一部分。三者的关系没有被分开，不要写成已归因。

## 2026-10-15 05:1x：G-HP-01 收口，G anchor 仍是 A1（Kiro）
1. **G anchor 不变（A1）**。G1（宽度 ×2）过了预注册门槛（+7 hits），但：单 seed；A1 自己三个 seed 的散布约 22 hits（sd），+7 远在噪声内；N@50 反而更低；参数量约 4 倍。与 P 线 H1 的处理口径不同：H1 便宜（P 作业约 22 分钟）且 +15 hits，所以补了 seed；G1 补两个 seed 要花约 7 小时 GPU 去验证一个在噪声内的 +7，不值，**不补**。
2. G2–G5 都明显差于 A1（−31 到 −49 hits），A1 的 LR 3e-4、4+4 层、dropout 0.1 在这一轮单旋钮范围内是局部最优。
3. G7 组合不触发。G6（SID 深度 +1）仍待 GTOK-01；届时需先扩展 ghp01_combo.py 才能做第二次判定。
4. G 线 HP 名额按 6 个单旋钮中已用 5 个计，G6 保留。

## 2026-10-15 07:0x：GFEAT-02 收口（Kiro）
1. **G 线不采用 F2、F3**：在 A1 上 F2 记 inconclusive（−1 hit）、F3 记有害（−20 hits），闸门通过说明入口无偏差。G anchor 仍是 A1，不带这两个特征。
2. **不补 seed**：F2 的 −1 hit 在噪声内，补 seed 只能证明“没效果”，而 F3 已明确变差；省下的 GPU 时间留给 GTOK-01R 与后续的 G5 融合。
3. **口径**：对外说“F2/F3 在 core03 小模型上有增益，但在论文规模的 A1 上没有”，不要说“G 线特征无效”。

## 2026-10-15 08:xx：GTOK-01R 重试成功，T1r、T2r 通过拟合门禁（Kiro）
1. 预注册门禁（H1–H6、R1–R4）一字未改，T1r、T2r 在 L2 臂（Adagrad lr 0.1 + 死码重置）上全部通过。T1r/T2r 替换 T1/T2 的名额，G-TOK 仍是 3 个名额（T1r、T2r、T3）。
2. 偏离预注册的部分登记：原卡写死的 Adagrad lr 0.4 在本次重试中改为 0.1，由探针按预先写死的规则选出（见 experiment_log）。汇报时要说明：RQ-VAE 在照搬论文设置下塌缩，调低学习率并加死码重置后才拟合成功。
3. 接下来按原计划：-verify 审计，再在 core03 规模训练 G 并 verify。G6 的“RQ-VAE 4 层”依赖的 T1 现在有了基础，是否改为 RQ-VAE 4 层由 G6 评审时再定，不在今晚做。

## 2026-10-15 10:4x：P 线超参 anchor 采用温度 0.10（F4 + T0.10）；G 线下一步（Kiro）
1. **P 线超参 anchor = F4 + 温度 0.10**（简称 F4-T010）。依据：H1 在三个 seed 上都比 F4 高且都过 ≥6 hits 门槛（+15/+31/+15），seed 2024/2025 没参与选择仍然复现。温度是纯优化超参，没有泄漏风险，所以用比“特征转正”宽的标准：三 seed 方向一致且过预注册门槛即采用；统计上只有 seed 2024 逐 seed 显著，汇报时如实说明。F4 的特征集合不变。
2. **范围**：F4-T010 用于后续 P 线实验、G5 融合和全量阶段；FEAT-03 已经在 T0.15 下完成，结论不重做。Q2（可学习新近性）与温度 0.10 的组合是否更好，是新的实验设计，留给 Eddy 起床后决定。
3. **G 线**：core03 上 tokenizer 筛选的赢家是 T3（BHKM 256×3，254 hits）。按 20:3x 第 1 条的‘确认 run’规则，已排 A1+T3（seed42）确认 run（不计名额）和 T3 core03 的 seed 2024/2025 补跑（补 seed 不计数）。G anchor 目前仍是 A1+BHKM16×3；如果 A1+T3 明显高于 A1，则 G anchor 候选变为 A1+T3，需要补 seed 后再定。
4. **G6（SID 深度+1）**：256 叉 3 位已经追平 A1，深度 +1 的价值要重估；G6 的评审继续暂停。

## 2026-10-15 12:0x：P 线 FEAT-04 与序列模型备选（Eddy 批准“三个都做”，并要求把“换模型”列为备选、先做详细计划）
1. 批准 FEAT-04 五个实验：A Q2 在 F4-T010 上复核（不计）、B F6 只改历史侧（诊断，不计）、C 物品热度/新鲜度（1）、D 物品被观看质量（1）、E Q2 加行为类型权重（1，Eddy 提出的“浏览与点赞收藏一起用”）。P 线特征名额 8 → 11。
2. 换序列模型列为备选：先 M0 诊断上次 SASRec 输给 id_mean 的原因，再 M1 SASRec-v3（1 名额），M2 多兴趣有条件（1 名额）。
3. 计划草案：docs/superpowers/plans/2026-10-15-p-feat04-and-seqmodel-plan.md。判定规则、条件组合名额、M2 是否列入，待 Eddy 确认计划后冻结进卡。

## 2026-10-15 12:3x：FEAT-04 与序列模型计划确认（Eddy）
1. Eddy 确认 2026-10-15 计划草案：判定规则（均值 Δhits ≥6、≥2 个 seed 过线、转正再要 ≥2 个 seed 逐 seed p<0.05 和泄漏 review）、名额（P 线特征 8→11，条件组合 +1，序列模型 +2）、M2 多兴趣列入（有条件：M1 至少追平 F4-T010 才做）。
2. 方向补充：这是研究性项目，希望在复现论文的基础上做出有独到见解的创新（不要求到发 paper 的程度）；服务器经费充足，鼓励多做实验。因此计划里的结论要同时写“发现了什么机制”：目前已有的独到发现有——F4 的增益是周间隔先验而不是近期加权；可学习新近性池化有增益；特征效果随模型规模变化（core03 有、A1 无）；SID 分辨率（256 叉）比聚类方法影响更大；RQ-VAE 在论文设置下塌缩、降低学习率加死码重置后修复。后续实验尽量围绕这些机制设计，而不是堆特征。
3. 启动：FEAT-04（设计→冻结→部署→入队）与序列模型 M0 诊断 + M1 设计/实现（本地，暂不上服务器）两个工作流并行。

## 2026-10-15 13:xx：G 线 anchor 候选更新（Kiro）
1. T3（BHKM 256×3）在 core03 三 seed 均 +64/+93/+67 hits（均值 +74.67），且均值（267.67）高于 A1（230）。G 线最强配置的候选从“A1 + BHKM16×3”改为“T3 SID”；是 core03 + T3 还是 A1 + T3 还没定，取决于 A1+T3 确认 run（GTOK-01B）的三 seed 结果。
2. 在 A1+T3 结果出来之前，G anchor 名义上仍是 A1+BHKM16×3，G5 融合暂停不恢复；P 线 anchor 是 F4-T010。
3. 研究含义（写进最终报告）：这份数据上 SID 分辨率比聚类方法、模型规模都更关键；下一步值得做的拆分是 512/1024 叉（分辨率继续加是否还涨）、以及叶子碰撞率与召回的关系。作为新实验（名额：G-TOK 的扩展，不占已有 3 个）排在 A1+T3 之后。
## 2026-10-15 15:2x：FEAT-04 拆分为三个独立工作流；P 线温度精扫预注册（Eddy 指示，Kiro 记录）
1. **拆分**（schedule.md 4b 教训第 1 条）：旧 FEAT-04（A–E 绑在一起，P 线自凌晨 5 点没有作业）已中止（wf_e4d2ec0e2d0994dd 与更早的 wf_0e558697353b5369，设计稿 rev0 与第 1 轮评审保留在 feat04-p worktree 的 .agents/tasks/）。改为三个互不阻塞的工作流：
   - **FEAT-04a**（wf_1f586db76f039a87，分支 feat04a/p）：A = Q2-T010（在 F4-T010 上复核 Q2）、B = F6-hist（只改历史侧），3 seed 各，不计名额；设计从简，今天入队。
   - **温度精扫 php02**（wf_8b5f7ba0b7e3e2c6，分支 php02/p）：F4 特征集上温度 0.05 与 0.07，3 seed 各，共 6 训练 + 6 verify，今天入队。
   - **FEAT-04b**（wf_652ea62860f0a965，沿用 feat04/p worktree）：C 物品热度/新鲜度、D 物品被观看质量、E Q2 + 行为类型权重，3 seed 各；名额 C、D、E 各 1（P 线特征 8→11 不变）；设计在 rev0 基础上最多再 1 轮评审，之后我裁决。
   - 序列模型 M1（seq/m1，本地已 review APPROVED）的服务器步骤排在 FEAT-04a 与温度精扫之后。
2. **温度精扫的名额与预注册（看结果前写死）**：
   - 名额：使用 P-HP-01 作废的第 7 个名额（H7 组合 void，见 2026-10-15 00:0x 第 2 条），不新增名额。取值只有 0.05、0.07，不加第三个值，不做其它旋钮。
   - 对照：同 seed 的 F4-T010（温度 0.10，hits 439/434/444）。
   - 判定：有效 = 三 seed 均值 Δhits ≥ 6 且至少 2 个 seed Δhits ≥ 6；有害 = 均值 Δhits ≤ −6；其余 inconclusive。
   - 采用为新温度：温度是纯优化超参，没有泄漏风险，沿用 10:4x 的宽标准（三 seed 方向一致且过 ≥6 hits 门槛即采用；McNemar 逐 seed 只作描述，不作门槛）。0.05 与 0.07 都有效时取三 seed 均值 hits 更高的一个；差 ≤6 hits 时取更接近 0.10 的（更保守）。
   - 限制（汇报时必须写）：这是在 validation 上从 {0.05, 0.07, 0.10} 里选，存在赢者诅咒；FEAT-03 与 FEAT-04a/b 的 anchor 对照仍在各自的温度上比，温度改变后其它特征结论是否保持未测。
3. **FEAT-04a/b 的判定规则**：沿用 2026-10-15 12:3x 已确认的规则（有效 = 均值 Δhits ≥ 6 且 ≥ 2 个 seed 过线；转正 = 有效 + ≥ 2 个 seed 逐 seed McNemar p < 0.05 + 独立泄漏 review），工作流只做机械套用，转正结论由我在 Opus 下解读后写。
4. **GTOK-02 的 B256D2（深度 2 的新门禁 B1D2）仍未获 Eddy 批准**：其余 5 个深度 3 配置不依赖它，照常推进；B256D2 批准前不拟合、不入队。
## 2026-10-15 15:3x：GTOK-02 冻结前的两项预注册（P1、P2；Kiro）
GTOK-02 第 2 轮设计（gtok02-design.md §6/§8）要求编排者在冻结前登记 P1、P2。**第 2 轮设计评审结束前这两条是草案，评审通过后生效。**
1. **P1 选 seed 候选池**：8 个扫描点一起排（已有的 BHKM 16 叉与 T3 256 叉，加新增的 32/64/128/512/1024 叉深度 3、256 叉深度 2）。取 R@50 最高的 2 个，再加与最高值相差 ≤10 hits 的，补 seed 2024/2025（补 seed 不计名额）。
2. **P2 B256D2 无条件补 seed**：256 叉深度 2 是唯一“强制叶子碰撞”的对照点，它的 seed 间方差决定能不能做碰撞归因，所以无论排名都补 seed 2024/2025（2 个 core03 训练 + 2 个 verify，不计名额）。
3. **一处需要 Eddy 知道的口径问题（未解决，坏消息先说）**：设计第 2 轮撤回了新增门禁 B1D2，改为让预注册的 B1 检查作用在补零前的 3 位序列上，阈值与其它门禁一字未改。我不确定这算不算“改预注册门禁”：阈值没改，但检查的对象换了。第 2 轮评审要先核这一点；在 Eddy 明确说“不算”之前，B256D2 不拟合、不入队，其余 5 个深度 3 配置不受影响。如果 Eddy 认为算，B256D2 退出 GTOK-02，名额不用。
## 2026-10-15 15:5x：B256D2 对预注册 B1 的解释性偏离获 Eddy 批准（附三个条件）；GTOK-02 设计收口（Kiro 记录）
1. **登记为“对预注册 B1 的解释性偏离”**。B1 的规则与阈值（子节点大小差 ≤1）一字未改；仅对 B256D2 把 B1 作用于补零前的 [第1层, 第2层, 序号] 三列。理由：补零后的第 3 列全是 0，不是聚类产物，B1 在这一列必然判违反；B256D2 的前两层继承 T3 的映射，T3 已通过完整 B1；不这样处理，该变体在构造上永远过不了 B1。H1–H6 及其它门禁不变，没有新增门禁（B1D2 已撤回）。**Eddy 已批准**，B256D2 的“暂缓/HELD”解除，按 GTOK-02 原计划拟合、审计、训练。
2. **条件 2（透明）**：报告里并列给出补零后四列的 B1 检查结果（预期违反，256 处），与三列结果并排，不隐藏。
3. **条件 3（措辞与推断检验）**：B256D2 与 T3 的对比在报告里写成“位置对照”，不写成碰撞率因果证据。同时检验这条推断：T3 第 3 层 256 个取值只用了 2 个，是因为前两层的 65,536 个格子平均只有约 1.8 个视频（120,415 / 65,536 ≈ 1.84，由审计的 mean leaf size 复核），第 3 层只需区分 1 到 2 个视频。**判据（看结果前写死，操作化由 Kiro 定，Eddy 可改）**：B256D2 三 seed（42/2024/2025）R@50 均值 hits 与 T3 三 seed 均值（267.67）之差的绝对值 ≤ 12 hits（T3 三 seed 254/274/275 的样本标准差 11.85 取整）→ 推断得到支持；超出 → 推断未获支持，如实写“未获支持”。无论哪种，结论都标为推断，不升级为因果。
4. **P1、P2 生效**（见上一条 15:3x）：GTOK-02 第 2 轮设计评审仍是 CHANGES_REQUESTED（0 HIGH、2 MEDIUM、7 NIT）。第 1 个 MEDIUM（设计写“B256D2 不需 Eddy 点头”与暂缓裁定冲突、驱动没有 HELD 开关）因本批准而消失，B256D2 不需要 HELD 开关；第 2 个 MEDIUM（驱动未定义 fit、fit-verify、compare 的提交者，compare 在有配置失败时可能永不触发）已被实现计划吸收。设计循环按 2 轮上限收口，由 Kiro 裁决，不再开第 3 轮。B256D2 无条件补 seed 2024/2025（P2）。
5. **温度精扫“采用”规则澄清（php02 设计评审第 2 轮 MEDIUM：规则歧义、阈值未冻结）**：15:2x 第 2 条里的两层口径明确如下，阈值都是 6 hits（ΔR@50 ≥ 0.002 ≈ 6/2520）。
   - **有效**：三 seed 均值 Δhits ≥ 6 且至少 2 个 seed Δhits ≥ 6（对 F4-T010 同 seed）。
   - **采用为新温度**：三个 seed 的 Δhits 都 ≥ 6（沿用 10:4x 对 H1 的标准：三 seed 方向一致且都过门槛）。
   - 只“有效”但不满足“采用”：记为线索，不换 anchor 温度，报告里写明哪个 seed 没过线。
   - 两个温度都满足采用时：取三 seed 均值 hits 更高的；差 ≤ 6 hits 时取更接近 0.10 的。
   - 仓库内冻结卡里的规则文本是副本；以本条为准，不为此改已 review 的代码。
## 2026-10-15 15:5x：A1+T3 补 seed 2024/2025 提前入队，不再等 seed42 结果（Kiro，按 Eddy 授权）
1. **改了什么**：10:4x 第 3 条与 schedule §4a 第 3 批写的是“A1+T3 seed42 比 T3-core03 多 ≥6 hits 且方向一致才补 seed”。现改为**无条件入队**：g-gtok01b-t3-b256d3-s2024、g-gtok01b-t3-b256d3-s2025（A1 规模，各 1 训练 + 1 verify，6656 MiB、1 核、RAM 18432、wall 声明 24h、batched b=1）。不计名额。
2. **理由**：①seed42 还要约 1 小时以上才出，T1r 两个 seed 也在 16:45 前后结束，之后到第一个 P 作业入队（预计 18:00–20:00）之间 GPU 会空 1–2 小时；②无论 seed42 好坏，A1+T3 的三 seed 均值都是 G anchor 选择需要的（A1+BHKM16 三 seed 均值 230 hits 是对照）；③无条件补 seed 比“看了 seed42 再决定补不补”更干净，不引入对 seed42 的条件选择。
3. **代价与风险（坏消息先说）**：①如果 seed42 很差，这 2 个 A1 作业各约 1.5 小时以上的 GPU 时间等于白花（但 GPU 本来是空的）；②墙钟无实测，preflight 粗估 5.2–5.5 分钟/epoch，并发下更长；③实测 3 个 G 作业合计用了 4.35 核（超过声明的 3 核），5 个 G 作业 + 之后的 P 作业可能让 CPU 配额被限速，P 作业会变慢；④G6 路径 verify 的真实运行、2 小时封顶够不够，仍无证据；⑤驱动在真实终态上的分支没实测。
4. **解读规则不变**：A1+T3 是否替代 A1+BHKM16 当 G anchor，仍按 seed42 加两个补 seed 的三 seed 均值、门槛 ≥6 hits 且方向一致判定，结果解读等 Opus。
## 2026-10-15 16:0x：FEAT-04a 设计收口；FEAT-04b 冻结前三项登记（Kiro，Opus）
1. **FEAT-04a 设计收口**（2 轮用完，第 2 轮 0 HIGH / 2 MEDIUM / 6 NIT，科学设计无阻塞）。两条 MEDIUM 按评审给的修法采纳，写进 feat04a-design.md 末尾裁决节，作为实现前置修订，不再走设计评审：
   - F1：闸门按变体拆开。A 的 preflight 与 CUDA 预演通过即可入队 A，不等 B；B 的 CUDA 反向确定性若不过，只挡 B。同一 SHA 能同时给出 A、B 时用同一 SHA；B 落后时先冻结、部署 A，B 用后续 SHA 跟上。
   - F2：preflight 的执行载体用远端暂存目录 /workspace/vkgr-feat04a-preflight/<prefix>/（唯一写入点，不在 runs/、queue/ 下），命令外包 `timeout 1800`，结果由只读脚本取回到 MAIN05；超 30 分钟无 exit 文件即失败、禁止入队。
   - 6 条 NIT 作为实现前置修订一并处理。
2. **D-1（FEAT-04b C/D，以及 04a 的 B 作者词表）：登记为对 common-brief §4 第一句（“特征只能来自 target 之前”）的已知偏离**。C/D 的物品统计用 train 第 0–24 周全部行聚合，对 86,785 个训练样本包含它自己的 target 行和之后的行；04a 的 B 的作者词表（supported 物品的作者，supported 由 full-train likes 决定）同类。依据：Eddy 确认过的计划 §1.1 允许“在卡里写明并给出理由”。validation 行从不进统计，validation 指标本身没有用到 validation 的标签。**读法限制**：C/D 若“有效”，必须说明增益里可能含训练侧的自身 target 信息；对外不能说“严格时间隔离的物品特征有效”。
3. **C9（D 的转正附加条件）：我裁决接受，作为冻结前新增的预注册规则，不需要另请 Eddy 点头**。理由：它在看任何结果之前写入，只收紧 D 的转正门槛（不改 §1.2 的任何数值，不影响有效/无效/有害的分类），属于实验推进类决定。规则：p_like（扣掉自己那一行后 like_bucket 发生变化的训练样本占比）> 0.30 时，D 不得转正，除非 seed42 的 loo1 对照 run 存在、verify 通过、且相对 anchor 的 seed42 ΔH ≥ 6。30% 这个数字是设计评审给的，没有更强的依据，汇报时如实写。loo1 对照是条件触发的诊断 run，不计名额。Eddy 可以否决这一条。
4. **D-2（条件组合的动作）**：触发条件 = 计划 §1.2 原文（C/D/E 中 ≥2 个“有效”，unstable-positive 不计入）。触发后不自动入队；由我在 decision_log 登记后再入队。名额已在 12:3x 第 1 条批准，属于实验推进类，不再单独请 Eddy 批准，但入队前会在汇报里告诉他。
5. **D-3（04b 的组织方式）**：新文件一律带 b 后缀，避免与 04a 重名；E 的 git 基底 = 合入 04a 冻结 SHA（merge --no-ff，必要时 cherry-pick -x，不改写历史），04a 冻结前 C、D 先行；两阶段冻结，bucket_spec_sha256 按变体。
## 2026-10-15 16:3x：更正 16:0x 第 3 条——C9 改为报告项，不当转正门槛（Kiro）
1. **更正**：16:0x 第 3 条把 C9 定为“D 的转正附加条件，不需要另请 Eddy 点头”，这是越权。给转正加条件就是改 Eddy 已确认的预注册判定（12:3x），属于“改预注册门禁须问 Eddy”的范围，不论是收紧还是放宽。FEAT-04b 第 2 轮设计评审（F2）指出了这一点。
2. **现行口径（取代 16:0x 第 3 条）**：C9 只是报告项。p_like > 0.30 时，D 的报告必须写明“增益里训练侧自身点赞信息的占比可能大”，并建议先跑 seed42 的 loo1 对照（诊断 run，不计名额）再采用；D 的转正完全按计划 §1.2 字面判定，不加条件。“预注册”一词只用于 §1.2 的四条规则，C9 叫“预设报告提醒”。
3. 同理，C2（显著为负的 seed 高亮）也只是报告项。
4. 如果 Eddy 希望 C9 恢复为门禁，需要他明确批准，并在阶段 1 冻结前完成。
## 2026-10-15 16:3x：下一阶段路线（Eddy 确认“都做”）；向 Codex 交接的方式（Kiro 记录）
1. **批准做**：(a) 协同信号进 SID（G-COLL）；(b) 多通路融合（G5，P + G + Pop，之后加协同 SID 版 G 作第三路）；(c) P 线 F7 用户静态特征，先做 U2 活跃度和 U3 口味画像，U1（年龄/性别/地域，敏感属性）单独决定；(d) F8 特征组合，等 FEAT-04a/b 与 F7 的结果再定组合哪些。路线、依赖、风险见 docs/superpowers/plans/2026-10-15-next-phase-roadmap.md。
2. **名额**：F7 的 U2、U3 暂按各 1 个新名额（P 线特征 11→13），具体数字写进冻结卡时定；G-COLL 的两个输入版本（只用塔向量、塔向量拼内容向量）按 G-TOK 扩展计，各 1 个，不占原有 3 个。看结果之前先登记，不事后补。
3. **P、G 的落库 anchor 都未定**：P 暂为 F4-T010，G 暂为 A1 + BHKM16×3，候选 T3。后续还会升级，路线图按“anchor 会变”设计：融合只认冻结的 anchor，变了就追加新版本并列报告。
4. **向 Codex 交接**：Eddy 打算先让 Codex 只读了解情况再接手；以后重要的事由 Kiro 断（裁决），实现交 Codex。接手清单在路线图 §3。正在跑的 4 个工作流归当前 Kiro 会话所有，新会话不能恢复，只能按重启包重启；服务器上的作业和驱动不依赖任何模型会话。
## 2026-10-15 16:3x：温度精扫 php02 入队完成；两处偏离的裁决（Kiro）
1. **状态（16:30 只读复核）**：6 个训练（p-php02-t005 / t007 × s42 / s2024 / s2025）都进过 claimed，无 rejected；t007-s2025 在评估，t005-s42-verify 在跑，另有 3 个 verify 在 pending，t007-s2024 与 s2025 的 verify 由驱动（pid 32766，PPID 1，日志 main05/php02_pollloop.log，每 900 秒轮询，不调用模型）在训练结束后补。公开分支 php02-pub/p = 1cc70b5f（父 ab53a8d），部署目录 /workspace/vkgr-php02-p，树 digest 与本地一致（241 文件）。真实输入 preflight 与 CUDA 预演通过。
2. **偏离 1：服务器侧 test_php02_untouched skipped=2 → 接受**。原因是公开压平分支里没有 BASE 对象，那两条测试需要对象才能比较。替代证据：服务器树 digest 与本地逐文件一致 + 本地同一棵树上该测试 skipped=0。理由：这两条测试守的是“相对 BASE 只新增、不改旧文件”，本地已经在含 BASE 对象的 worktree 上证明过，服务器树又与它逐文件相等。限制：服务器上没有独立跑过这两条，记入 issue_log。
3. **偏离 2：push 前按原命令扫 git log -p 命中 2 处 HelloBike 字面量 → 接受，不改写本地历史**。命中在未推送的中间提交里；推送的是压平后的单个提交（php02-pub/p），对压平后整体 diff 重扫 0 命中。与 2026-10-14 21:2x 对 gtok01/g 的做法一致：本地过程历史原样保留，公开分支只放压平提交。以后写测试词表时避免直接写 HelloBike、/Users/ 字面量。
4. **读结果的规则不变**：温度精扫的采用规则见 15:5x 第 5 条（三个 seed 的 Δhits 都 ≥ 6 才“采用”）。结果由 Codex 读数并记入 experiment_log，解读和 anchor 温度的裁决回到 Kiro。
## 2026-10-15 16:5x：复查 Codex 阶段一清单；两项结果判读规则在看三 seed 结果前预注册（Kiro）
1. **G anchor 是否换成 A1 + T3（预注册）**：对照 = A1 + BHKM16×3 同 seed（250 / 206 / 234 hits）。换 anchor 的条件，三条同时满足：(a) A1+T3 的三个 seed 训练都 succeeded 且对应 verify 都 verified；(b) 三 seed 均值 Δhits ≥ 6；(c) 至少 2 个 seed 的 Δhits ≥ 6。SID 是无标签泄漏的数据无关变换，不要求 McNemar 和泄漏 review（与温度同类，沿用 10:4x 的宽标准，但这里采用“有效”的口径而非“三 seed 全过”，因为 A1 自己 seed 间散布约 22 hits）。不满足则 G anchor 保持 A1 + BHKM16×3，A1+T3 作为并列报告。**读 s42 单个数不下结论**；Codex 记录的 A1+T3 s42 = 289 hits（R@50 0.1147），只是一个 seed、verify 未完成。
2. **T1r 与 T3 的比较（预注册，只回答“BHKM 与 RQ-VAE 谁更好”）**：T3 三 seed 均值 267.67 hits（254 / 274 / 275，样本标准差 11.85），T1r 三 seed 均值与它之差的绝对值 ≤ 12 hits 记“不能区分”；超过 12 hits 才写方向，仍标为推断（单一数据集、core03 规模、聚类方法与 SID 分辨率相同）。前提：T1r 三个 seed 的 verify 都 verified。
3. **驱动与铁律差异（Codex issue，复查结论）**：旧驱动 35118 / 68450 的轮询间隔是 600 秒，低于 Eddy 要求的 ≥15 分钟，是我在 GTOK-01B 的 brief 里写了“例如 10 分钟”造成的，错在我。它们是只读轮询、不调用模型，对服务器和额度没有实质影响，**不重启**（重启本身有风险，现有作业优先）。新驱动一律 ≥900 秒。四个驱动及其辅助脚本在 gitignored 的 main05，不满足“运维脚本进版本管理”：登记为已知偏差，新驱动启动前必须版本化；旧驱动脚本在对应作业收口后补进版本管理。
4. **工作流实时状态（Codex 无 inspect_workflow 能力，由我补）**：温度精扫工作流已完成（php02/p，入队 6 个训练）；FEAT-04a code review APPROVED，进入冻结 / 部署 / preflight / 入队；FEAT-04b 计划完成、在实现循环；GTOK-02 code review 刚 APPROVED，进入冻结 / push / 部署 / preflight。这些仍归 Kiro 会话所有，Codex 不接管、不重建。
5. **issue 编号冲突**：issue_log 里出现两个 I-2026-10-15-g（Kiro 的 FEAT-04a TDD 偏差，Codex 的驱动差异）。请 Codex 把自己的那条改成 I-2026-10-15-h（它是后写的）。
## 2026-10-15 17:0x：FEAT-04a 的 B（F6-hist）OOV 规则登记（冻结前置条件；Kiro）
**B 的 OOV 沿用 F6：零贡献、不可训练；不含可学习 OOV 向量。** 作者表行布局 0 = PAD、1 = OOV、2+rank = 词表作者（词表 11133 个，共 11135 行）；词表外的作者查到行 1（不越界），但经 `(rows >= 2)` 掩码贡献恰为 0 且收不到梯度；64D 内容投影对所有事件（含 OOV 作者）照常生效。计划表 B 行写的“含 OOV 桶”按此条读，对外称“OOV 行（零贡献）”。若要“OOV 有一个可学习的共享向量”，那是另一个变量，需要 Eddy 明确改写，本卡不做。依据：FEAT-04a 冻结卡 §1.6 A1（经 2 轮设计评审、code review APPROVED）。
## 2026-10-15 23:2x：FEAT-04b 阶段 1（C+D）冻结后的事实更正与连接中断记录（Kiro）
1. **阶段 1 = C + D**，E 不在阶段 1（04a 冻结 SHA 3170417 未合入，E4 未在该 SHA 上重跑）。冻结 X1 = e8bc9dc，记录 feat04b-freeze-stage1.json（sha256 27f19f1e…63a2），公开分支 feat04b-pub/p = 51da61ff，部署 /workspace/vkgr-feat04b-p-x1，真实输入 preflight 与 CUDA 预演通过。第 1 波 C-s42、D-s42 及其 verify 已 succeeded，指标未读。
2. **更正（C/D 的审计事实，取代冻结卡里的推断）**：卡 §8 风险 1 写的“Σlikes ≈ 训练样本数”在总量上不成立：审计回执 sum_likes = 105,700，n_train_examples = 86,785，差 18,915（+21.8%）。likes_i 等于 target_freq_i 的物品占 93.0%，说明 D 的点赞率分子与 logQ 目标频次高度重合，D-1 的读法限制更要写足。p_like = 0.2115，未超过 0.30，C9 的预设报告提醒不触发。冻结卡已冻结，不改文字；以本条和 feat04b 报告为准。
3. **§7.2 C9 的引用修正接受**：卡里 loo1 那句指向“16:0x 第 3 条”改为“16:3x 第 2 条”，规则和数值不变（该条原文含“诊断 run，不计名额”）。
4. **连接中断记录**：约 19:49 起本机到服务器的 mux 反复断开，FEAT-04b 阶段 1 的第 2 波（c-s2024、c-s2025、d-s2024、d-s2025）因此推迟约 3 小时；同时 GTOK-02 驱动在 19:30–23:10 之间没有轮询（驱动日志的空档）。空档的原因我没有确认（疑似本机休眠或断网，pmset 日志里没有直接的睡眠记录，Amphetamine 在运行）。后果：这段时间 P 线没有作业，服务器大部分时间空闲超过 1 小时（违反“空闲超过 1 小时要报告”的目标，主因是没人知道连接断了）。
5. **教训**：本机一旦睡眠或断网，工作流步骤、驱动、我的会话都会同时停；服务器侧的驱动（如 FEAT-04a、FEAT-04b 的 verify 驱动）不受影响，本机驱动（GTOK-02、A1+T3）会停。以后新驱动尽量放服务器侧，并且连接断开超过 30 分钟要有告警。
## 2026-10-16 01:xx：FEAT-04b 阶段 2（E）入队完成；服务器侧 freeze 测试失败的接受方式（Kiro）
1. **状态**：04a 冻结 SHA 3170417 合入 feat04/p（d3ca347）；E4 在合入后的 HEAD 上通过，K10 记录已写；review #2 = APPROVED（0 阻断，3 条非阻断）；阶段 2 冻结 `--write/--check --stage 2` 通过，阶段 1 记录与 §7 哈希未变；feat04b-pub/p 快进到 a91bcd38（树 335aadce… = d3ca347 的树）；部署 /workspace/vkgr-feat04b-p-x2（270 个文件 sha256 逐行一致）；真实输入 preflight 与 CUDA 预演通过（实测 354 MiB，声明 1024）；3 个 E 训练 `p-feat04b-e-q2-behavior-s42 / -s2024 / -s2025` 已确认进 claimed、不在 rejected；服务器侧 verify 驱动 pid 2914285（PPID 1，900 秒，日志 /workspace/vkgr-feat04b-driver/x2-e/driver.log）。s42、s2024 训练已 succeeded，其 verify 在 pending，s2025 仍在训练。指标未读。
2. **服务器侧 04b 测试 197 ok、16 FAIL、1 skip：接受**。16 个 FAIL 全在 freeze 测试，原因与 php02 的 skipped=2 同类：服务器上的公开分支是压平历史，基准提交 3463846 不可达，freeze 与 untouched 测试需要 git 对象。替代证据：服务器树与本地 270 个文件逐行 sha256 一致 + 本地同样的 39 个 freeze+untouched 测试全 OK。限制：服务器上没有独立跑过这些检查。记入 issue_log（I-2026-10-16-a）。
3. **E 的读法**：相对 anchor 判定（计划 §1.2），另有 E−A 描述性对照（设计 §7.4，需要 04a 的 A 三 seed 已验收，A 已 verified）。β 符号与大小要进报告。不得把增益直接说成“行为权重有效”：先看 E−A。
4. **报告缺口**：feat04b-report.md 是阶段 2 新建的，阶段 1（C、D）因旧工作流中止没有写报告节；读完 C、D 的结果后由 Codex 补阶段 1 一节（含 23:2x 的审计事实更正）。
5. **本地提交**：feat04/p 在 d3ca347 之上多了一个 verify 驱动的提交 5d51228（未 push）；驱动脚本应进版本管理，由 Codex 决定何时压平推送。
## 2026-10-16 02:xx：序列模型 M1 真实训练前的冻结裁决（F3、F4、ablation_g0、0.05/3 阈值）（Kiro；起草时模型为 Sonnet，**M1 入队前需在 Opus 下复核一次，改动只能发生在看到任何 M1 结果之前**）
依据：seq-report.md §2 F3/F4、§6 第 5 条；seq-m1-design.md §3.6、§8.1–§8.5、附录 A。M1 仍无任何真实数据结果，所以本条是登记，不是事后修改。
1. **F3（M2 触发吃 4 选 1 赢者臂）：采纳评审修法。** M2 的自动触发只看主臂 `mean`。非 `mean` 的代表臂不能自动触发 M2：必须同时满足 D ≥ 0、k₊ ≥ 1 和 §8.2 条件 2（分支启用），并且报告里并列写出 `mean` 的 §8.1 行；是否触发 M2 由 Kiro 单独裁决（M2 本来就需要编排者点头）。非 `mean` 代表臂的 p_pool 按 Bonferroni ×4 另报（上限 1）。非 `mean` 代表臂仍不能直接转正（设计 §3.6 第 6 条）。`pure` 不参与代表臂选择和 M2 触发。
2. **F4 的阈值 0.05 / 3：按设计 §8.5 原文冻结，不再改。** 0.05 与 3 是 `branch_ratio_median = ‖g·r‖/‖m‖` 的相对阈值（`< 0.05` 分支未启用，`> 3` 分支压过 anchor 路径），本身已经对 ‖m‖ 做了相对化，所以不需要在看结果后按数据重新标定。SEQM1-000 量 F4-T010 best checkpoint 的 ‖m‖ 只作**前置核对**，不是调阈值：在 validation query 上量 ‖m‖ 的中位数与 p10/p90，设计 D2 的尺度假设是“‖m‖ 约 0.1–0.5（未测）”。若测得中位数落在 [0.05, 5] 之外，阈值仍不变，但**停止并报告 Kiro**，因为 LN_f 初值 0.1 的尺度论证不再成立，需要重新评估设计再决定。量到的数字写进 experiment_log。
3. **`ablation_g0`：不写进 §8.2 条件 2，只作必报项。** 理由：给 M2 触发再加一个门槛，等于改 Eddy 在 12:3x 确认过的“M1 至少追平才做 M2”的含义，改预注册门禁要问 Eddy（C9 的教训）。处理：报告必须逐 seed 列出 `ablation_g0` 的 ΔH（带分支 − g→0），并给一条“分支有效用”标记（ΔH > 0 的 seed 数）。若 §8.2 两个条件都满足但“分支有效用”的 seed 数 < 2，报告必须写“分支启用但无正贡献”，并且 Kiro 在裁决 M2 时把它作为反对理由之一。Eddy 如果希望把它升为门槛，需要明确批准。
4. **其余沿用设计原文**：§8.1 主结论只认 `mean` 对 F4-T010（hits 439/434/444）；§8.3 机制比较是描述性，不得改写 §8.1；`pure` 只做归因；转正需 ≥2 个 seed 逐 seed McNemar p < 0.05 + 新模型代码的独立泄漏 review + 延迟报告，通过后另写 decision_log。D2（按读出方式拆分的尺度估算）不做设计改动。
5. **入队顺序与名额**：M1 占序列模型名额 1 个（计划 §3），5 臂 × 3 seed = 15 个训练 + 15 个 verify 都在这 1 个名额内。先入 `mean` 的 3 个 seed（主结论），再入其余 4 臂；VRAM 先量（设计估算 3072 MiB 未实测）；每个 run 多一次完整评估（`ablation_g0`），墙钟估算要加上。真实训练前的前置：SEQM1-000（只读核对：F4-T010 三 seed 逐 query predictions 可读、seed 42 与 2024/2025 模型代码一致、sha256 覆盖本分支 php01 文件、‖m‖ 前置核对）→ SEQM1-009 CUDA 预演 → 真实输入 preflight。
## 2026-10-16 夜间：Opus 复核 02:xx 的 M1 冻结裁决（通过）；M1 夜间批次；已结束批次的确认；FEAT-04 条件组合内容登记（Kiro，**本条在 Claude Opus 5.5 下完成**）
依据（逐份读过）：seq-m1-design.md §2 A3、§3.3、§3.6、§8.1–§8.6、附录 A；seq-report.md §2、§6；experiment_log「2026-10-16 Codex训练安排」§1–§5、M2 节、「Codex全权接收」§1–§5；STATE 末节待复查表；docs/superpowers/plans/2026-10-16-overnight-queue-proposal.md。M1 尚无任何真实训练结果，本条全部属于“看结果前”。

### 1. M1：Opus 复核结论 = 通过
**Opus 已复核 02:xx 裁决，允许 Codex 在完成正式提交检查后执行 M1 全部五臂×三 seed，先 mean 三 seed，再其余十二训练，各配 verify。** 这不允许跳过正式 dry-run、提交前 review、真实输入检查或 claimed/rejected 验收。逐项确认：
1. **M2 触发只看主臂 mean**：确认。理由：Eddy 12:3x 的门槛原文是“M1 至少追平 F4-T010 才做 M2”，M1 的主结论预先指定为 mean（§3.6），只看 mean 是对该原文最忠实的读法，不是新增门槛。“触发”只表示允许开始写 M2 设计卡；M2 本身从设计、实现、冻结到入队仍走完整流程（附录 A：本稿不实现），没有任何自动入队。
2. **非 mean 代表臂**：按 §3.6 规则 1–4 报告（须比 mean 高 ≥6 hits 才取代、并列报告、×4 Bonferroni、注明赢者诅咒），不能自动触发 M2，满足 02:xx 第 1 条的条件时交 Kiro 单独裁决；非 mean 代表臂不能直接转正（§3.6 第 6 条）。`pure` 不参与代表臂选择与 M2 触发，只用于 §8.3(B) 归因。确认。
3. **branch_ratio_median 阈值 0.05 / 3**：按 §8.5 原文冻结，不再标定。确认。
4. **SEQM1-000 的原始 m 范数**：只作固定区间 [0.05, 5] 的尺度核对，越界停下报告，不改阈值。确认；实测三 seed 中位数 0.1435 / 0.1315 / 0.1498，通过。
5. **ablation_g0**：不进 §8.2 条件 2，只作必报项（逐 seed ΔH，以及“分支有效用”= ΔH > 0 的 seed 数）。确认。
6. **主结论只认 mean**，其它臂按 §8.3 诊断口径报告，不事后挑赢家改写 §8.1。确认。
7. **转正与 anchor**：仍须 §8.1 行 2 + 至少 2 个 seed 逐 seed exact McNemar p < 0.05 + 对新模型代码的独立泄漏 review + 延迟报告（§3.11），另写 decision_log，不自动发生。确认。
8. **A3（名额口径）**：接受。M1 占序列模型 1 个名额，`last / attn / bi / pure` 四个诊断臂不另计名额（FEAT-03 Q1a/Q1b 先例）；15 个训练都在这 1 个名额内。失败 attempt 不退名额（§8.6）。

**复核中发现、需要写明的读法限制（不阻塞，不改阈值）**：按实测 ‖m‖ 中位数约 0.14、设计 D2 对初始 ‖r‖ 约 0.8 的估算（未实测），ratio ≈ 5.7·|g|，0.05 对应 |g| ≈ 0.009，低于设计自己估的“一个 epoch 随机游走约 0.013”。所以 §8.2 条件 2（分支启用）**可能只靠 g 的噪声就成立**，是弱证据。处理：(a) 报告里 §8.5 的判定必须与 ablation_g0 并列写；(b) M2 若满足触发条件，Kiro 在冻结 M2 设计卡时必须读 ablation_g0，若“分支有效用”的 seed 数 < 2，M2 设计卡里要写明“分支启用但无正贡献”的风险——这是 M2 卡的评审内容，不是新增触发门槛。若 Eddy 想把 ablation_g0 升为触发门槛，须 Eddy 明确批准，并且要在 M1 任何结果出来前决定。

### 2. M1 夜间批次（有限、预先登记）
- 范围：mean × {42, 2024, 2025} 先入；然后 last、attn、bi、pure 各 × {42, 2024, 2025}，共 15 个训练 + 15 个 verify。不得根据 mean 的结果增删诊断臂或挑 seed。
- 前置（由 Codex 做，全部要有收据）：正式 dry-run；**对 12 个诊断臂配置也做与 mean 相同的服务器真实输入只读 preflight**（此前真实输入 preflight 只核了 3 个 mean 配置，诊断臂只有合成 CUDA verify）；提交工具与接续驱动的一次提交前 review；入队后只读确认 claimed 且无 rejected/.reason。
- 接续方式（工程金丝雀，不是看结果）：12 个诊断臂训练在“至少 1 个 mean 训练 terminal succeeded 且其 verify 直接读到 verified”之后由服务器侧驱动自动入队；只看收据状态，不看 R@50。若 Codex 今晚交付不了经过测试的接续驱动，备选是 15 个一次性入队（mean 在队列前面），由 scheduler 按资源放行——真实 preflight 与五臂 CUDA verify 已通过，风险可接受。
- 停止条件：任一 mean 训练失败或其 verify 不是 verified → 不入队诊断臂，驱动写 ALERT 并停；任一作业进 rejected → 读 .reason，不重投；任一失败 attempt 不重跑、不覆盖（§8.6）。
- 驱动：先入 Git、push、固定 SHA 部署；PPID=1、轮询 ≥900 秒、不调用模型、有有限截止（建议 48 小时）；终态写机器可读汇总（每个 run 的 train/verify 状态、run_id、commit/config/data hash、best/stop epoch、R@50、N@50、ablation_g0、branch_ratio_median、异常、未验收项）。
- 资源：每个声明 1024 MiB、1 核；GTOK-02 六个补 seed 占约 6 核期间，scheduler 会限制并发，属正常。真实墙钟 UNKNOWN，不承诺明早全部完成。

### 3. 已结束批次：Kiro 确认机械结论（按预注册规则）
1. **G anchor = A1 + T3（BHKM 256×3）**。16:5x 第 1 条三个条件全部满足（三 seed succeeded 且 verified；Δhits +39 / +122 / +76，均值 +79；3/3 ≥6）。限制：seed 间散布大（289 / 328 / 310，best epoch 8 / 20 / 16），+122 那个 seed 拉高均值；三 seed 评同一批 query。对外说“256 叉 SID 在 A1 规模上三 seed 都更好”，不说“A1 和 256 叉的收益可以相加”。G5 融合的 G 路从此用 A1 + T3。
2. **P anchor 仍是 F4-T010**：温度 0.05 / 0.07 机械有害，不采用；FEAT-04a A 有效但逐 seed 显著 1/3，不转正；B inconclusive；FEAT-04b C、D、E 均机械有效，但逐 seed 显著分别 0/3、0/3、1/3，都不满足转正统计条件。E−A 为 −1 / +2 / +2（均值 +1）：**E 相对 anchor 的增益基本来自 Q2（A），行为类型权重本身没有可检出的增量**；β 的符号（点赞、收藏、点作者为正，点踩为负）符合直觉但只是描述。
3. **T1r 对 T3**：T3 高 18.33 hits，超过 ±12，方向写“在 core03、256×3 设置下 BHKM 高于 RQ-VAE（推断）”，不推广为方法优劣。

### 4. FEAT-04 条件组合：内容登记（不在今晚队列）
- 触发已满足（C、D、E 均“有效”，≥2）。名额：12:3x 已批的 1 个条件名额。
- **内容 = F4-T010 + C + D + E**（E 内含 Q2），三 seed。理由：计划 §1.2 的字面是“C/D/E 中有效者的组合”；若因为看到 E−A≈0 而把 E 换成 A，就是根据结果改组合内容，不允许。E−A 作为描述性对照继续报告。
- 判定：同计划 §1.2（对 F4-T010；有效 / 转正 / inconclusive / 有害），另报组合相对最佳单项（E）的描述性差值，不作门槛。C、D 的 D-1 读法限制与 23:2x 审计更正照抄。
- 状态：需要实现（C/D 物品塔 + E 用户塔同时启用的配置与可能的接线）、测试、review、冻结、真实 preflight；Codex 完成后入队前在汇报里告诉 Eddy（16:0x 第 4 条），不需要再请批准名额。

### 5. 其余候选的状态（今晚都不进队列）
- GTOK 新 SID 的 A1 确认 run：需先等 B1024 / B512 / B256D2 补 seed 三 seed verified 并按 gtok02 设计的 Q1 规则判读；A1（G6 runner）是否接受 sid_v2s 的新 SID 版本未验证，需要配置、冻结与真实 preflight；入选名单交 Kiro 裁决。
- G-COLL 两种输入：草稿的源 seed、零向量回退、64/128 维接口未定，需设计评审与实现。
- F7 U2/U3：前缀 sidecar 与窗口/分桶未冻结；U1 不在授权范围。
- M2：受 §8.2 与本条第 1 节约束，M1 结果出来后由 Kiro 裁决是否开始写设计卡。
- G5：G 路已定为 A1 + T3，P 路仍是 F4-T010；完整框架的重新冻结需要先更新 g5prep-design 的 anchor 绑定，不算填空训练。
## 2026-10-16 晨间：M1 与 GTOK-02 结果裁决；P anchor 的转正前置；M2 设计卡；B1024 的 A1 确认 run 预注册（Kiro，**Claude Opus 5.5**）
依据（逐份读过并复算关键数字）：experiment_log「2026-10-16 晨间 M1 / GTOK-02 验收（最终）」§1–§5；STATE 晨间快照与待复查表；seq-m1-design.md §3.6、§3.11、§8、附录 A；seq-m1-code-review.md；seq-report.md §3、§6；gtok02-design.md §3.8；本文件 02:xx、夜间 §1–§5。复算：mean Δ +27/+47/+27；bi Δ +43/+50/+42，bi−mean +16/+3/+15；E_start = (58+72+80)/3 = 70；E_cond = 408.67−259 = 149.67；B1024−T3（2024/2025）+11/+8，B512 −1/−8，B1024−B512 +12/+16，B256D2 +6/−13，B256D2 三 seed 均值与 T3 差 −9.33。全部与 Codex 记录一致。

### 1. M1 主结论：确认机械“有效”，转正前置还差一项
1. **mean 有效（§8.1 行 2）确认**：D = +33.67，k₊ = 3，p_pool = 2.9e−6；逐 seed exact McNemar p 0.038 / 0.00015 / 0.036，3/3 < 0.05。p_pool 跨 seed 重复同一批 query，偏乐观，报告必须写。
2. **独立泄漏 review：认可已有的 seq-m1-code-review.md（APPROVED）**。理由：它是 fresh-context 的独立 review，评审对象是 seq/m1@04d4ede，正文有专门的泄漏核对（用户塔签名只有 (histories, side)、翻转 target 字段后 5 个臂用户向量逐位不变、会读 target 的变异体 M13 被抓、因果 mask 方向、test 关闭），而部署的 0ca61bc 与 04d4ede 树相同（experiment_log D1）。已知限制写进报告：泄漏测试晚于实现（变异体补偿），评审者只逐字段比对了 mean-s42 一个配置。不需要新增 review。
3. **还缺：§3.11 延迟报告（需补证，阻塞 P anchor 切换，不阻塞其它批次）**。现在只有本机 CPU、合成输入的量级参考（seq-report §3：在线口径约 2.3–3.4 倍），不算 §3.11 的口径。Codex 用 `scripts/seqm1_latency_bench.py` 在服务器上按 §3.11 跑：用户塔、batch=512 离线与 batch=1 在线、CPU 单线程与 A6000、validation 长度分布、预热 50 次、计时 1000 次、mean/p50/p95/p99、anchor 与 M1-mean 同脚本对比；数字写 experiment_log。§3.11 没有通过阈值，只报告。
4. **预注册的 anchor 切换（看延迟数字之前写死）**：延迟报告按上条口径进入 experiment_log、并由 Kiro 核对口径（不评判数值好坏）之后，**P anchor 由 F4-T010 改为 M1-mean**（p-seqm1-mean-s42/s2024/s2025，466/481/471）。在此之前 P anchor 仍是 F4-T010。切换不追溯：已登记的 FEAT-04 条件组合仍对 F4-T010 判定。
5. **g0 消融的读法限制（必须写进报告）**：mean 的 g0 Δ = +109/+97/+107，说明训练后分支在模型内部有正贡献；但 g→0 后模型只剩约 357/384/364 hits，**远低于 F4-T010**，即训练后的 m 路径已和分支共同适应，不再等于 anchor。所以 g0 Δ 不能读成“序列分支带来约 100 个命中”；序列建模相对 anchor 的净收益只看 §8.1 的 +33.67。

### 2. M2：批准开始写设计卡（只到设计评审）
- §8.2 两个条件机械满足（mean D > −6、k₀ = 3；分支启用 3/3），g0 正贡献 3/3，没有出现“分支启用但无正贡献”的风险。**批准开始写 M2 设计卡**，范围仅限设计与设计评审（最多 2 轮，之后由 Kiro 裁决）；实现、冻结、训练另行裁决，不追加臂或 seed。名额：序列模型第 2 个（12:3x 已批）。
- 设计卡的约束（看 M2 任何结果之前写死）：①骨干用 mean 臂的**因果 mask**（预先指定的主臂），只把读出从均值换成 K=4 的 ComiRec-SA 多兴趣读出，单变量对比；②**主对照改为 M1-mean 同 seed**（附录 A 原文写“M1 代表臂”，代表臂现为探索性的 bi，用它当对照会把赢者诅咒带进 M2；在 M2 出任何结果前改写，属于设计期修改），F4-T010 对照照 §8.1 报告；③附录 A 的 N10 风险（hard argmax 下头 1–3 拿不到梯度）必须有守卫与处理；④延迟口径加“K 路检索 + 合并”；⑤沿用 §8 的阈值与统计，不新增门槛。

### 3. bi 与 pure 的对外措辞
- **bi**：“四个锚定臂中，双向 mask 的三 seed 均值最高（对 F4-T010 +45，比 mean 高 11.33，×4 校正后 p_pool 2.2e−8）；但 bi 对 mean 的逐 seed 配对 p 为 0.20 / 0.87 / 0.20，都不显著，且是四选一，有赢者诅咒。不能说双向优于因果，也不能据此转正。”对推理的说明：bi 只在 target 之前的历史上做双向注意力，没有读 target，泄漏 review 已覆盖 5 个臂。
- **是否另卡验证 bi**：今天不立卡。要做需要新的序列模型名额（现有 2 个已分给 M1、M2），**须 Eddy 批准名额**；作为候选登记。
- **pure 归因（§8.3(B)）**：E_start = 70、E_cond = 149.67，机械落“两者都有贡献；二者不可加，不拆比例”。必须同时写：S1 只有 seed42（259 hits），E_cond 是 F3/F4/N1/T0.10/pre-LN 等一整包条件，不能说哪一项起作用；pure 仍比 F4-T010 低 30.33，说明 anchor 的均值路径本身很重要。对外一句话：“上次 SASRec 输给均值模型，主要是没吃到相同的特征和训练条件、起点也不在 anchor 上；补齐之后，在 anchor 上加一条序列分支能再提升约 34 个命中。”

### 4. GTOK-02：Q1 结论与 B1024 的 A1 确认 run
1. **Q1 确认**：B1024 = UP（+11/+8）、B512 = FLAT（−1/−8）、B1024−B512 = UP（+12/+16），按 §3.8 优先级 1 输出“还涨，所测范围内峰值在 1024 侧”。措辞必须带 §0 第 1 条：B1024 在第 2、3 层的利用率只有 11.5% 和 0.1%，实际是“少一层语义 + 更大编号”，不是“更细语义”；不写“分辨率越高越好”，不外推到 1024 以上。
2. **P2（B256D2）**：位置对照 FLAT，两个 seed 一升一降（+6/−13），分歧单独注明；三 seed 均值与 T3 差 −9.33，绝对值 ≤ 12，按 15:5x 条件 3 落“推断得到支持”，仍只是推断，不是碰撞率因果证据。
3. **A1 确认 run 名单：只确认 B1024D3**（B512 FLAT、B256D2 FLAT，不进）。三个 seed（42/2024/2025）。性质：确认 run，不计名额（2026-10-14 20:3x 第 1 条）。对照：A1 + T3 同 seed（289 / 328 / 310）。
4. **判定（预注册，沿用 16:5x 第 1 条的三个条件，不另立阈值）**：三个训练都 succeeded 且 verify 都直接读到 verified；三 seed 均值 Δhits ≥ 6；至少 2 个 seed Δhits ≥ 6。全部满足 → G anchor 由 A1+T3 改为 A1+B1024；否则保持 A1+T3，B1024 只并列报告。限制：A1+T3 自己 seed 间散布很大（289–328），门槛 6 相对噪声偏小，报告写明。
5. **入队前的前置（Codex，都要有收据）**：核对 A1/G6 入口是否接受 sid_v2s 的 B1024 版本（GTOK-01A 的 B1–B5 教训：只用 engineering 测试不算数）；配置只改 SID 绑定与 run_id；冻结；服务器真实输入 engineering=False 只读 preflight；CUDA 预演与**真实 shape 的显存实测**（B1024 解码每步展开 1024 个分支，6656 不能直接沿用，按实测 ×1.15 取整到 256）；墙钟没有样本，不承诺。
6. **GTOK-02 报告缺口（需补证，只阻塞 GTOK-02 最终报告，不阻塞 B1024 确认 run）**：B256D2 补零后四列的 B1 实测是 Eddy 15:5x 条件 2 的要求，报告收口前必须补（只读审计脚本即可）；新叉数真实显存峰值记为 UNKNOWN 并保留 issue；常零位置 logp(0) 诊断按设计可写 UNKNOWN。

### 5. 后续批次与接续
- **可以按条件接续的就绪批次**：(a) FEAT-04 条件组合（F4-T010 + C + D + E，三 seed，内容见夜间 §4），工程闸门通过后入队，入队前告知 Eddy；(b) B1024 的 A1 确认三 seed，前置齐全后入队；(c) M1 的 §3.11 延迟测量（小作业，只为报告）。三者互相独立，可以并行，按 8.19 核与显存准入放行。驱动规则同夜间 §2（版本化、PPID=1、≥900 秒、不调用模型、48 小时截止、终态汇总；失败或 verify 非 verified 写 ALERT；rejected 读 .reason 不重投；失败 attempt 不重跑不覆盖）。
- **不在接续里**：M2（只写设计卡）；G5（P anchor 可能因延迟报告而改、G anchor 可能因 B1024 确认而改；设计规定“只认第一个冻结版本”，所以等两个 anchor 定下再冻结，期间可以更新设计里的绑定但不冻结）；G-COLL、F7（设计未定；U1 不做）；bi 另卡（需名额）。
- **关于空档**：昨夜约 6 小时空档的原因是下一批没准备好，不是驱动问题。规则 §3 第 1 条（一批入队时下一批必须已在写）今天要落实：上面三个就绪批次的工程前置现在就做，不要等结果。没有就绪批次时如实报告空闲，不为填满机器加实验。
## 2026-10-16 午间：Eddy 批准 SID 修复 pipeline、C1′ 替换、新名额与几项授权（Kiro 记录）
Eddy 原话大意：“可以替换，新的名额批准，pipeline 可以使用”；上一轮：“bi 可以另开，U1 要用，可以清理临时文件”。
1. **SID 修复 pipeline（S0–S5）批准**，顺序与内容见 Kiro 2026-10-16 的说明，Codex 写入路线图：
   - S0 诊断（不训练）：对 T3、B1024 做逐层准确率、按热门/长尾分层的增益、每层利用率、码分布熵、碰撞率、每叶平均视频数；做成可复用的 SID 诊断工具，指标要与规模无关，以便换算到全量（约 1247 万视频）。
   - S1 G-COLL：两个配置，core03 seed42 筛选，入选补 seed，对照 T3 与 B1024（core03）。
   - S2 LETTER 式可学习 tokenizer：在已修好的 RQ-VAE（L2 臂）上加协同对齐（对比损失）与多样性损失，协同目标取 S1 中赢的那种向量；只在 S1 显示协同信号有用后才做。用开源代码前先做许可证审计，不兼容就按论文独立实现。
   - S3 深层准确率：先通读两篇论文（Pinterest 最深层 SID 准确率；单层大码本），再决定是否立卡。
   - S4 规模准备：BHKM / RQ-VAE 在 1247 万 × 64 上的内存与时间、增量或分批 k-means、suffix 规模、新视频的 SID 分配（路线图 P11）；与 S0–S3 并行，不依赖结果。
   - S5 确认：S1/S2 赢家在 A1 规模确认三 seed（确认 run 不计名额），再进 G5。
   - 每一阶段都要独立的冻结卡、看结果前写死的判定规则，沿用 ≥6 hits 口径；每个新 SID 都要重新训练 G 模型。
2. **G-COLL 的 C1 改为 C1′**：C1 原定只用 P 线物品向量，但 P 物品向量只对 supported 视频非零（21,496 / 120,415 ≈ 18%），82% 的视频会挤在零向量上，做不成。C1′ 改用训练期（第 0–24 周）观看历史训练的共现向量（item2vec 一类），覆盖全部 train-observed 视频。C2（内容向量拼 P 物品向量，零向量时退回内容）保留。名额不变，仍为 2 个。这是在任何结果出来前的设计修改。已知偏离：共现向量用了训练样本自身 target 及之后的事件（与 D-1 同类），validation 第 25 周不进；报告必须写明。观看比例加权等细节在设计卡里定。
3. **S2 新名额批准**：G-TOK 扩展再加 1 个名额（LETTER 式）。
4. **bi 另卡批准**：序列模型第 3 个名额。bi 对 mean 正面比较，用新的 seed；具体 seed 取值、判定规则在设计卡里看结果前写死；排在 M2 设计卡之后，不影响 P anchor 的既定切换条件。
5. **U1（年龄/性别/地域）批准使用**：并入 F7 设计，P 线特征名额 +1。报告只报整体，不做分人群结论，写明公平性与使用边界。
6. **服务器临时文件清理批准**：只限 `docs/superpowers/plans/2026-10-15-server-tmp-residue-list.md` 列出的路径；清理前逐个确认不是任何运行中作业的工作目录或打开文件，清理后记录删除清单。**不包括**公开 Git 历史里本机路径的清理（I-2026-10-15-i，需要 force push，仍须单独批准）。
## 2026-10-16 下午：M1 延迟口径通过 → P anchor 改为 M1-mean；M2 设计五项裁决与执行授权（Kiro，**Claude Opus 5.5**）
依据（逐份读过）：docs/superpowers/plans/2026-10-16-m2-multi-interest-draft.md 全文；experiment_log「2026-10-16 SUCCESSOR」§1–§3；cache/codex-successor-engineering-20261016/latency-audit.json；seq-m1-design.md §3.3、§3.8、§3.11、§8、附录 A；本文件「2026-10-16 晨间」§1、§2。

### 1. M1 延迟报告：口径通过；P anchor 自本条起为 M1-mean
- 对照 §3.11 逐项核：只测 `model.users()`；离线 batch512（每 query 摊销）加一次完整 2520 条导出遍历另报；在线 batch1；CPU 单线程与 A6000；50 次预热、1000 次计时、CUDA 前后同步；mean/p50/p95/p99；anchor 与 M1-mean 同脚本、同输入 SHA；历史长度取 validation 真实分布（L50 1580 条，最大余数法分配）。特征取值合成、权重为初始化值：§3.11 原文就允许合成取值（用户塔成本与取值无关，属推断），权重取值同理，不影响口径。以上全部符合。
- 读法（写进报告，不设门槛）：M1-mean 的用户塔在线单 query 约为 F4 的 2.9 倍（CPU 1.74 对 0.59 ms），A6000 离线摊销约 3.1 倍；绝对值都在毫秒以内。A6000 batch1 比 CPU 慢是小 batch 的 kernel 启动与同步开销，不代表服务性能。不含特征取数、物品塔、ANN、网络，不能称端到端延迟。
- **按晨间 §1.4 的预注册，P anchor 自本条起由 F4-T010 改为 M1-mean**（p-seqm1-mean-s42 / s2024 / s2025，466 / 481 / 471 hits）。不追溯：FEAT-04 条件组合仍按登记对 F4-T010 判定；M2 主对照本来就是 M1-mean。
- **需要知情的一点（不改上面的决定）**：FEAT-04 条件组合（F4-T010 + C + D + E）三 seed 为 484 / 502 / 490，均值 492，比 M1-mean 的 472.67 还高约 19；但它的逐 query McNemar 与 bootstrap 还没算，C、D 带 D-1 已知偏离，转正条件未验收。所以 P anchor 可能很快再变。由此：G5 的冻结要等组合的转正判定出来再做；“M1 与组合叠加”是值得登记的后续候选（未立卡、无名额）。

### 2. M2 设计五项裁决（本条即设计评审第 1 轮；按以下修改后设计定稿，不再开第 2 轮）
1. **N10 训练法：采纳草稿 §3 的唯一方案**——前 100 个 optimizer step 用 detach 的 target 软权重（温度 0.10）对逐头 N1 loss 加权求和（不是先平均四个 u），从第 101 步起固定 hard argmax（平手取最小 j）。100 步约为 0.59 个 epoch（86,785 / 512 ≈ 170 步/epoch），包含在原预算内。**必须的措辞限制**：M2 相对 M1-mean 改了两处——读出（K=4 多兴趣）与训练时的兴趣分配规则——结论只能写“K=4 多兴趣读出及其分配训练规则”整体的效果，不能归因到读出单独一项。
2. **守卫与停止条件：给定唯一替代——hard 窗口的覆盖检查改为诊断，不是停止条件。** 理由：附录 A 原守卫的作用是“不满足就改成 soft 起步”，这个补救现在已经内置；在它之后再把“某头没被选中”设成停止条件，会造成缺 seed，并带来重跑的压力，而某头饿死本身不是数据或泄漏错误，是一种需要报告的结果（有效兴趣数 < 4）。所以：第 101–200 步每头被选比例照常记录；为 0 的头在报告里标“该 seed 有效兴趣数 < 4”，训练继续。**唯一的停止条件是数值异常**（loss、梯度或用户向量出现 NaN/Inf）：停止该 attempt、保留证据、交 Kiro，不重试、不延长 warmup、不换 seed。这不改任何 Eddy 确认过的门禁，也不新增性能门槛。
3. **检索与公平性：采纳 full-catalog exact max top50 为唯一主评估。** 分路 exact（每头 top50 → 去重 → 按四头最大分重排 → 全局 50）只能作为实现优化，前提是与 full-catalog max 在同一全序 tie-break 下逐项相同，并有 synthetic oracle（重叠、平手、各头独有项）。报告必须公开：最终 50 个位置与训练预算相同，但打分是 4 路、分路中间候选最多 200（M1 为 50）。**本卡不做 ANN**，不另报 ANN 口径，也不立 ANN 调参臂。这是多兴趣召回的通行评估方式（每个物品取多兴趣最大分），不是给 M2 额外优待，但“多了 3 个查询向量”的事实要写明。
4. **分支诊断：采纳**每 query 取 `max_j ‖g·r_j‖ / ‖m‖`、再跨 query 取中位数，best epoch，沿 0.05 / 3 原阈值与 §8.5 原措辞；另报每头的 ratio 中位数、serve 阶段 top1 获胜头计数、g→0 同模型消融。这些对 M2 全是描述性诊断：M2 之后没有被它触发的实验，不构成门禁，不需 Eddy 批准。必须写明：max 过线不能证明四头都被用上；g→0 的差值不能读成相对 M1-mean 的净收益（与 M1 的共同适应限制相同）。pure / S1 归因及其 12、259 阈值不适用。
5. **冻结前实现细节（定稿要求）**：
   - 共享参数（与 M1-mean 完全相同的那部分）按 M1-mean 的同一抽取顺序与同一 seed 先抽；W1、b1、W2 之后再抽。测试：同 seed 下共享参数的初始 state_dict 与 M1-mean 逐位相等。
   - W1、b1 沿用 M1 attn 臂里 `W_a`、`b_a` 的初始化代码（冻结卡里写出源码位置），W2 用 std 0.02 的正态、无 bias。
   - 零起点：第 0 步四个 u_j 都与 M1-mean 的 `m` 逐位相等（train / eval 两种模式）；第 0 步的 L_soft 与 M1-mean 第 0 步 loss 的差 ≤ 1e-6（四个相同值求均值的浮点误差，实测值写进收据）；分支参数梯度为 0、g 的梯度非零。
   - 必须的测试：逐头 soft / hard 的手算 oracle；g ≠ 0 时 soft 阶段每头 W2 行梯度有限且非零（synthetic 非退化夹具）；翻转 target 字段后四个 u 逐位不变；因果扰动（只改未来位置，较早 z 不变）；padding 宽度不变性；运行期 test 读取拦截及其变异体；全 unsupported 历史、n=1、n=50。
   - M1-mean 三 seed 的逐 query 预测必须可读、可与 M2 配对（同 SEQM1-000 的做法核对一次），代码与数据绑定写进 manifest。
   - 统计沿草稿 §6（即 M1 §8.1 的表；p_pool 偏乐观的限制照写；不对单主臂做 ×4）。转正要求不变。

### 3. 执行授权（逐项）
- **允许立即实现**：Codex 在隔离 worktree 按 fresh RED → GREEN 实现模型、runner、三个配置（只差 seed 与 run_id）、verify、诊断与驱动；M1 原冻结代码不动。不等当前 G 训练结束。
- **允许冻结**：完整产物做一次独立 code review（提交正式作业前的不可逆节点，规则 8）APPROVED 后冻结。内部修复不重复 review。
- **允许条件提交**：冻结后，扫描 → 工作分支 push → 固定 SHA 新目录部署 → 逐文件 SHA → 三个配置的真实输入 engineering=False 只读 preflight → CUDA 预演（soft→hard 切换、batch512×L50×K4、优化器状态、K4×全 catalog 分块 max 评估的最坏 shape、verify 重放与 g0 消融）→ 真实 shape 显存实测（声明 = 实测 × 1.15 向上取整 256，不沿用 M1 的 1024）→ dry-run（含最终启动记录大小与 supervisor 可读性，B1024 的教训）。**全部通过后，允许一次性提交三个 seed 并由服务器侧驱动自动补 verify，不需要再回 Kiro**；入队前在汇报里告诉 Eddy。三个 seed 一律跑完，不按 seed42 的结果决定后两个。
- **允许与当前 G 并行**：按 8.19 核与 39,312 MiB 准入（保留余量）由 scheduler 放行；资源不够就预排，不侵占余量，不伪称 claimed。延迟测量若与其它作业混跑，要标注不可比。
- **需要回到 Kiro 的情况**：实现中发现设计需要改（任何偏离本条第 2 节）；任何 attempt 出现 NaN/Inf；独立 code review 给出 HIGH/MEDIUM 而修复会改设计。
- **不变**：不增加臂、seed 或名额；不读 test；不改 U1/C9 的决定；不读旧 checkpoint 或导出完整 embedding；不删/覆盖、不付费、不 force push、不 push main、不改历史。M2 结束后只出结果、诊断和完整延迟（用户塔、四路检索、去重重排、合计，写明串/并行与 exact），解读与 anchor 交 Kiro。不自动接 M3、bi 另卡、G-COLL、F7 或 G5。
## 2026-10-16 傍晚：SID-FIX 卡登记，S0 冻结（Kiro 记录，Eddy 确认“尝试你的方案，写成卡”）
1. 卡：`docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md`。S0（诊断，不训练、不占名额）按卡 §2 冻结，可由 Codex 实现；S1–S5 是路线，各自另立冻结卡后才能实现。
2. **S0 的读法（看数之前写死）**：B1024 − T3 只用补 seed 2024/2025，按热度档分解；总增益 > 0 且头部档占 ≥ 50% → “更接近记忆”，否则 → “支持语义变好”；A1+B1024 三 seed 齐后对 A1+T3 另报。只描述，不是门槛，不改 GTOK-02 的 Q1 结论。50% 无更强依据，Eddy 可在结果出来前改。
3. **S1 的 C2 需要读取 M1-mean 的 best checkpoint 导出物品向量**，属于“读 checkpoint”，须 Eddy 单独授权；未授权前 S1 只做 C1′。
4. S1 两个配置都用 T3 的分叉（BHKM 256×3），只换输入，对照 T3 同 seed，补 seed 规则为 s42 ≥ T3 + 6 hits，判定只用 2024/2025、±6 口径（沿 GTOK-02 §3.8）。
## 2026-10-16 傍晚（续）：S1 的 C2 读取 checkpoint 获 Eddy 授权（Kiro 记录）
1. **授权**：Eddy 批准 S1 的 C2 读取 P 线 checkpoint 导出物品向量。范围：只读加载 M1-mean seed42 的 best checkpoint（`weights_only=True`），导出 train-observed catalog 120,415 个视频的 64 维物品塔向量（supported mask 照原样），存服务器、不进 Git；不导出用户向量，不读其它 checkpoint；checkpoint、脚本、导出文件均记 sha256。
2. **来源固定为 M1-mean seed42**，之后 P anchor 改变也不换（避免反复重冻）。FEAT-04 组合模型虽然三 seed 均值更高（492 对 472.67），但转正条件尚未验收，且物品塔含带 D-1 偏离的 C/D 统计特征，不作为 SID 来源。
3. **S1 冻结卡由 Codex 起草、Kiro 复查**，可与 S0 实现并行；冻结前不拟合、不训练。卡里必须写死：C1′ 的 item2vec 超参（窗口、负采样数、维度 64、epoch、seed）、是否按观看比例加权、C2 的 128 维输入合同与归一化方式、两个配置的分叉（BHKM 256×3）、补 seed 与判定规则（卡 §3）。
## 2026-10-16 夜间：SID-FIX S0 结果裁决；S1 设计定稿（含一处必须修正）；S1 执行与自动接续授权（Kiro，**Claude Opus 5.5**）
依据（逐份读过）：docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md；docs/superpowers/specs/2026-10-16-sid-fix-s1-freeze-draft.md 全文；experiment_log「SID-FIX S0 最终诊断」「S0 Markdown 报告实际读回」「A1+B1024 三 raw verify 收口」；issue_log sidfix-legacy-verify / -lift-readout / -control-label / -s1-beam；.agents/tasks/gtok02-design.md §3.5、§3.8。本条是 S1 设计评审的第 1 轮，也是最后一轮。

### 1. S0 结果的措辞（规则不改，只定怎么说）
1. **B1024 − T3（core03，补 seed 2024/2025）**：合计 +19，其中头部 +5、中段 +11、长尾 +2、unsupported +1，头部占 26.3% < 50%。按看数前冻结的读法，写：“B1024 相对 T3 的增益分散到中段及以后，不支持‘更接近记忆’，按预注册读法支持‘语义变好’。”必须同时写三条限制：①总量只有 19 hits、两个 seed，每个 seed 的门槛是 6；②50% 分界没有更强依据；③结构上 B1024 只有约 2 层有效分叉（第 2、3 位只用 118/1024 与 1/1024 个码），所以“语义变好”更准确的含义是“**第一层分得更细（1024 个大类，每类约 118 个视频）帮助了中段视频**”，不是“更深的语义层次”。
2. **A1 确认 run**：B1024 在 A1 上三 seed 为 313 / 330 / 210，对 A1+T3 同 seed 为 +24 / +2 / −100，均值 −24.67；晨间第 4 节的三个条件只满足第一个，**G anchor 保持 A1+T3**。写：“core03 上的 1024 叉优势在 A1 规模上没有复现；结论受 seed2025 一个低值（210）影响很大，A1 规模本身 seed 间散布就大（A1+T3 289–328，A1+BHKM16 206–250）。”**不得**事后剔除 seed2025 再算。请 Codex 只读补报三个 seed 的 best/stop epoch 与逐 epoch R@50（描述性，看 seed2025 是不是早停在差的 epoch），不重跑、不补 seed。
3. **B 组前缀准确率**：只描述；不同叉数的“前 k 位”粒度不同，不比较不同配置的同一 k 的绝对比例来下机制结论。
4. 以上都不改 GTOK-02 的 Q1 结论、不改任何阈值。

### 2. D 组（lift）的口径
1. **原结果保留，口径标注为“随机对含 self-pair 的原实现”**。卡 §2.2 对共现对写了“item 不同”，对随机对没写；冻结意图是同类比较，即随机对也应是两个不同的视频。self-pair 的概率约 1/120,415，对 k=1、2 的 lift 影响可忽略，但在每个节点只剩 1 个视频的层，它会成为随机匹配的全部来源，造成 lift = 0 的假象。
2. **登记补充诊断 D′（在看 D′ 数字之前写死）**：随机对改为不放回地抽两个**不同**视频（Python Random(0)，数量同共现对）；当某一层每个节点最多只有 1 个视频时，该层 lift 记 “STRUCT_NA（结构上无定义）”，不记 0，也不记 UNKNOWN。D 与 D′ 并列报告，原 D 不改、不删。
3. D 与 D′ **只描述，不是 S1 的门槛**；D′ 的补跑与 S1 工程互相独立，不阻塞 S1。S1 的新 SID 同时报 D 与 D′。

### 3. 四个旧归档（g-hist-v2-l50-s2025、g-g4-rand-s42/-s2024/-s2025）
- 允许 Codex 做一次纯只读的旧收据与别名核查（不读 checkpoint、不重启旧驱动、不重投、不改旧目录）。
- 仍找不到就保留 UNKNOWN，**只影响历史报告的完整性**（B16 三 seed 均值与随机 SID 对照的 B/C 列），不是 S1 的门槛。S0 的主差分（T3、B1024、A1 两配置）都已直接 verified，不受影响。
- 不建议为此读旧 checkpoint 补 verify；如果将来要做，须 Eddy 另行授权，不能借用 C2 的单 checkpoint 许可。

### 4. S1 唯一设计（草稿 §3–§6 按以下定稿）
1. **C1′（SGNS）——采纳草稿的取值，但必须修正一处会让训练几乎不动的问题**：草稿写“batch 内按正对取均值、lr 0.025、batch 4096”。word2vec 的 lr 0.025 是**每个正对**一步；对 4096 个正对取均值再用 0.025，相当于把每个正对的步长缩小约 4096 倍，5 个 epoch 下向量会基本停在随机初始化附近，C1′ 就变成了近似随机 SID。**定稿为**：batch 内对正对**求和**（不取均值），SGD 初始 lr 0.025，按已处理正对数在 5 个 epoch 内线性衰减到 0.025 × 1e-4（word2vec 的标准做法）；其余照草稿：64 维、对称固定窗口 5、不跨用户、同 item 跳过、合法重复保留不去重、每正对 5 个负样本（第 0–24 周频次^0.75，拒绝 center 与本对 positive）、无 subsampling、无观看比例与距离加权、无早停、seed 42、input 表均匀初始化 ±0.5/64、output 表全零、只导出 center 表并逐行 L2 归一化。事件序列来源与 S0 D 相同（第 0–24 周同用户完整观看序列），无法恢复唯一事件序列就 STOP。
2. **新增一条健康检查（工程正确性，不是科学门槛）**：训练后，在 S0 的共现对与 D′ 的随机不同视频对上各算一次平均余弦；**共现对平均余弦必须严格大于随机对**，否则视为训练失败（实现错误），STOP 回 Kiro，不调超参重训。差值本身写进报告。
3. **无合法正对的视频用同源内容单位向量补全：采纳**，记 mask 与数量。限制：两种向量不在同一空间，补全的视频在 BHKM 里的位置没有协同含义；报告写“协同为主、无正对项用内容补全”，并给出补全比例。NaN/Inf 或“有正对却导出零向量”STOP。
4. **C2：采纳草稿 §4 的合同**——来源永久固定 M1-mean seed42 的 best（先从已有 manifest 锁定路径与 SHA，不按文件名猜，不换 last 或其它 seed）；`weights_only=True`；只导出 120,415 个视频的物品塔输出，supported mask 原样；`[P, content]` 两块各自单位化后整体单位化；P 为 0 或 unsupported 时为 `[0, 单位 content]`；supported 但为 0 单列计数；非有限 P STOP；内容缺失、非有限或零范数 STOP。报告必须写两条：①只有约 18% 的视频（supported）带协同块，其余 82% 实际只有内容；②带协同块的视频与不带的视频在 128 维空间里会被系统性分开，BHKM 第一层很可能先按“有没有协同块”（接近“是否被点过赞”）分，这是本设计的已知性质，不是 bug。
5. **H5 合同：两部分，均为本卡自己的门禁登记，不改 GTOK-02 已登记的门禁**。GTOK-02 §3.5 的 H1–H6 是对它那 6 个 SID 登记的；S1 是新卡、新输入族，它的门禁现在（看任何 S1 结果之前）登记如下：
   - H5a（原文保留）：输入里用到的内容来源，其 digest 必须等于 CONTENT_V1。
   - H5b（新输入身份）：变换后的 64 维（C1′）/ 128 维（C2）输入数组，其 digest 必须等于对应导出或 SGNS 作业的已验收 manifest 中登记的 digest；该 manifest 必须同时绑定其来源（CONTENT_V1 digest、第 0–24 周事件序列 digest、C2 的 checkpoint SHA、脚本 SHA 与 commit）。
   - H1–H4、H6、B1 对两个 SID 原样生效（BHKM 256×3，B1 作用于完整映射前三列，tolerance 1）；R1–R4 不适用。旧 loader 与旧白名单不改，新版本名、新 schema 显式登记维度与变换。
   - **需要 Eddy 知情**：我判断这不是“修改已登记门禁”，因为 S1 是新卡；但它和 B256D2 那次一样涉及“门禁作用对象的定义”，如果 Eddy 认为算，告诉 Kiro，S1 在拟合前暂停即可（工程实现可以继续）。
6. **G 训练与判读：采纳草稿 §6**——逐字段继承 T3 同 seed 的 core03 配置（beam = 100，其余不漂移）；三个 G seed 共用同一个已验收 SID bundle；筛选：s42 的 train 与 verify 都直接 verified 后，hits ≥ 260（T3 的 254 + 6）才补 2024/2025，259 不补，不把名额挪给第三个配置；判定只用 2024/2025 对 T3 的 274 / 275，UP / DOWN / FLAT 按 ±6，分歧单列；缺 seed 只能写暂定状态；s42 与三 seed 均值只并列。S1 不改 G anchor，不外推 A1，不自动启动 S2 或 S5。

### 5. 执行授权（逐项；Codex 按此推进，普通步骤不再等 Eddy 或 Kiro）
- **阶段 A，立即开始**：隔离 worktree；fresh RED → GREEN 实现 SGNS（含求和与线性衰减、健康检查）、C2 精确导出、新 SID 合同与 loader、6 份 G 配置（两配置 × 42/2024/2025）与 verify 模板、S0 的 A–D 与 D′ 重算、调度脚本；一次完整独立 code review；只有需要改科学设计的 HIGH/MEDIUM 才回 Kiro；全部进 Git，扫描、push 工作分支、固定 SHA 新目录部署、逐文件 SHA；动手前先只读核当前队列与资源，不动 M2、G、S0 已有作业。
- **阶段 B，各配置自己的前置全过就推进，不互相等待**：C1′ 拟合（第 0–24 周）与健康检查；C2 导出与审计；各自 BHKM 拟合 → fit-verify 直接 verified → 同 seed replay 一致；三个 seed 配置的服务器真实输入 engineering=False 只读 preflight；CUDA 工程预演；真实 shape 显存实测（实测 × 1.15 向上取整 256，不沿用 3072）；正式 dry-run（含启动记录大小与 supervisor 可读性）。全部通过后**允许提交该配置的 s42 训练与 verify 接续**，入队前告诉 Eddy；queue JSON 必须带 schema；提交后独立只读确认 claimed 且不在 rejected。
- **阶段 C，服务器驱动自动接续（允许）**：驱动先进 Git、push、固定 SHA 部署；detached、PPID=1、轮询 ≥900 秒、不调用模型、**固定 48 小时截止**。train 成功自动补唯一 verify，只认原始 verified 加 job/manifest/config/commit/data/prediction 绑定。**s42 verified 且 hits ≥ 260 → 自动提交同配置 2024/2025 及其 verify 接续**；不满足则该配置结束。终态写机器可读汇总（每个 fit/train/verify 的 run_id、状态、各 hash、best/stop epoch、R@50、N@50、hits、筛选动作、A–D 与 D′、异常与 UNKNOWN）。
- **停止条件**：某配置任一步失败、verify 非 verified、健康检查不过、NaN/Inf → 停止该配置后续步骤、写 ALERT、保留证据，另一配置照常；共同数据或合同失效 → 两个都停、回 Kiro；rejected 读 .reason 不重投；失败 attempt 不换 run_id 重跑、不覆盖。设计偏离立即回 Kiro。
- **范围上限**：两个配置，最多 6 个 core03 训练与各自 verify，加上必要的向量拟合、导出、SID 拟合与审计；不增加臂、seed、名额。
- **仍须 Eddy 单独批准的事**：为四个旧归档读旧 checkpoint；若 Eddy 认为 H5b 属于改门禁则暂停拟合；以及既有清单（读 test、付费、删除/覆盖、force/main push、改历史、销毁实例）。C9 仍是报告项；C2 的许可不扩到任何其它 checkpoint。
## 2026-10-17：S1 入队后的并行本地任务安排（Kiro；Eddy 要求“等训练时本地做其它任务”）
1. **触发**：S1 两个配置各自的 s42 训练都已独立确认 claimed 且不在 rejected（或某配置已按停止条件写 ALERT 结束），且接续驱动存活。之后 Codex 在等待训练期间做下列本地任务；不碰 S1 任何作业与驱动。
2. **优先级 1：给 Kiro 的两份复查材料（只读，不下结论）**
   - FEAT-04 组合转正：三 seed 逐 seed exact McNemar、p_pool、bootstrap；对 F4-T010 与对 M1-mean（描述性）两组；已有泄漏证据清单（C、D、E 各自的泄漏测试与 code review 出处）；D-1 与 23:2x 审计事实；转正四个条件逐条“满足 / 不满足 / 缺证据”。
   - M2：按 2026-10-16 下午第 2 节核对完整统计（对 M1-mean 的 §8.1 表、逐 seed p、p_pool、bootstrap）、每头诊断（获胜头计数、hard 窗口选头比例、两两 cos、max-ratio、g0）、延迟（用户塔 + 四路 exact 检索 + 去重重排 + 合计，写明串/并行）。缺的只读补，不重跑训练。
3. **优先级 2：下一批服务器作业的准备**
   - **bi 另卡**：起草冻结卡（问题：双向 mask 是否优于因果 mask；臂 = bi 对 mean；用 3 个从未用过的新 seed，seed 值在卡里写死；判定沿 M1 §8.1 的表，对照是同新 seed 的 M1-mean，所以两臂都要用新 seed 重训，共 6 个训练；名额为已批的序列模型第 3 个）。只起草，交 Kiro 冻结；代码可以先按 M1 的 runner 准备配置，不入队。
   - **S4 规模准备**：按 SID-FIX 卡 §6 实现合成数据基准（N = 125 万与 1247 万 × 64 维，测 BHKM 与 RQ-VAE 拟合的峰值内存、墙钟、suffix 规模，并比较分批或流式 k-means 的可行性；新视频 SID 分配只出设计）。**它只用合成数据、不做任何科学判定，所以不需要单独冻结卡**：一次独立 code review 后即可在服务器以 1 核作业运行（按准入放行，不挤 S1），或在本机运行；结果写 experiment_log。
4. **优先级 3：补证与草稿**
   - GTOK-02 报告缺口：B256D2 补零后四列的 B1 实测（只读审计脚本），满足 Eddy 15:5x 条件 2。
   - F7（U1 年龄性别地域、U2 活跃度、U3 口味画像）设计草稿：先核 U3 与 F3/F4 的信息重叠；对照 anchor 写成占位，等 FEAT-04 组合裁决后再填；只起草不冻结。
5. **不做**：G5 重新冻结（等 P anchor）；S2、S5（等 S1 结果）；S3（Kiro 读论文）；任何新训练入队（bi 等 Kiro 冻结）。
## 2026-10-17：任务改用描述性名称（Eddy 要求“用具体任务来命名”，Kiro 记录）
S0–S5、C1′/C2 等编号是 Kiro 在 SID-FIX 卡（2026-10-16）里起的。以后对话、汇报、STATE、schedule 一律用下面的任务名；编号只放括号里方便对照，文件名与已有引用不改。
| 任务名 | 原编号 |
|---|---|
| SID 结构诊断 | SID-FIX S0（含补充诊断 D′） |
| 协同向量 SID | SID-FIX S1；两个配置：**共现向量 SID**（C1′）、**内容加 P 向量 SID**（C2） |
| 可学习 SID（LETTER 式） | SID-FIX S2 |
| SID 深层准确率调研 | SID-FIX S3 |
| SID 全量规模基准 | SID-FIX S4 |
| 新 SID 的 A1 确认 | SID-FIX S5 |
| 多兴趣召回 | M2 |
| 双向序列模型确认 | bi 另卡 |
| 序列用户塔 | M1（主臂 mean） |
| 物品热度+质量+行为权重组合 | FEAT-04 条件组合 |
| 用户画像特征 | F7（U1 年龄/性别/地域、U2 活跃度、U3 口味画像） |
| 双塔与生成式融合 | G5 |
## 2026-10-17：项目地图（Eddy 批准，Kiro 记录）
新增 `docs/project_map.md`：一页列出三个问题下每个任务的状态（已完成 / 在跑 / 本地在做 / 未开始）。**Codex 每次更新 STATE 时同步维护**；只写状态与指针，不写指标数字，不写决定内容。
## 2026-10-17：G 线后续方向；协作约定写进 AGENTS.md（Kiro）
1. **协作约定写进项目 AGENTS.md**（Eddy 指出提示词只在对话里会被遗忘）：新增「当前协作约定（2026-10-17）」一节，内容为角色分工、待 Kiro 复查表、描述性命名、project_map 维护、排程纪律；启动顺序与事实源表加入 project_map。AGENTS.md 里叠加的旧 owner 条款没有删，新节写明优先；整理旧条款是另一项工作，未做。
2. **G 线差距的位置**：SID 结构诊断的热度分档显示，生成式（A1+T3）比双塔（M1-mean）少的命中约四分之三在头部视频，长尾两边都接近 0；前缀诊断显示模型多半能猜对第 1 位、很少猜对前 2 位。读法：G 线主要输在“热门视频里挑不准具体哪一个”，不是长尾。
3. **方向排序**：①协同向量 SID（在跑）先出结果；②**生成式排序导向训练**（卡：`docs/superpowers/specs/2026-10-17-g-rank-objective-card.md`，Kiro 起草，DRAFT）——温度交叉熵与 SFT + 自生成难负样本 DPO 两个配置，须 Eddy 批准 2 个新名额；③双塔与生成式融合在 P anchor 稳定后重新冻结；④HSTU 编码器放到全量阶段（A1 上容量不是瓶颈）；⑤生成式 + 稠密向量混合召回暂不做（它主要补长尾，与差距位置不符，且会模糊 P 对 G 的比较）。
4. 排序导向训练卡冻结前还要：Eddy 批准名额；Codex 核对 R1 的温度取值出处；R2 的挖掘成本实测（超过 4 小时回 Kiro）。
## 2026-10-17 上午：组合转正、多兴趣召回解读、双向确认冻结、排序导向训练授权、AGENTS 整理（Kiro，**Claude Opus 5.5**）
依据（逐份读过）：docs/superpowers/reviews/2026-10-17-feature-combination-evidence.md、2026-10-17-multi-interest-evidence.md（含 02:40 服务器补数）；docs/superpowers/specs/2026-10-17-bidirectional-confirmation-draft.md；docs/superpowers/specs/2026-10-17-g-rank-objective-card.md；STATE 02:40 快照与待复查表。

### 1. 物品热度+质量+行为权重组合（FEAT-04 组合）
1. **前三个转正条件满足**（对注册对照 F4-T010）：均值 +53 hits；3/3 seed ≥ +6；3/3 seed exact McNemar p < 0.05（0.0014 / 3.5e−6 / 0.0010）。
2. **第四条（独立泄漏 review）：现有记录不够，需补一次定点 review。** 计划 §1.3 第 5 条写的是“泄漏 review 只对‘有效’的变体做”，即结果出来后的专项 review；已有的 #1、#2 是结果前的代码 review，组合接线那次 review 的原文在本地找不到。要求：一次 fresh-context 的独立泄漏 review，对象是部署的组合 commit `84b160c`（与本地 `c4e4eb8` 同树），范围按 FEAT-04b 冻结卡 §7.6 全部条目（`_tensors4b` / `fr._tensors2`、E 池化与桥接、C/D 统计 builder 与候选、L1–L10 在真实 engineering=False 路径上运行的证据、STAT_CUTOFF_WEEK 与 binding、训练与评估候选同源、§4.2 缺口 / AC16 / C9）。只读，不重跑训练。APPROVED 即第四条满足。
3. **满足后怎么用（看到组合对 M1-mean 的数字后，这里规定的是如何把组合并入 anchor，不改任何阈值）**：计划 §1.2 的“转正”是“成为 anchor 的一部分”。当前 anchor 已是序列用户塔（M1-mean），组合是建在 F4-T010 上的；组合对 M1-mean 均值 +19.33，但逐 seed p 为 0.20 / 0.13 / 0.19，都不显著。两者改的是不同部分（组合改池化权重与物品塔，序列用户塔加的是序列分支），能否相加不能假定。所以：**泄漏 review 通过后，组合的 C、D、E 成为“已验证可并入 anchor 的组件”，但 P anchor 暂时仍是 M1-mean；用一次“序列用户塔 + 组合”叠加实验来决定新 anchor。**
4. **叠加实验（预注册，看结果前写死）**：模型 = M1-mean 的结构，均值路径 m 换成 E 的池化（Q2 + 行为权重 β，初值与 E 相同），物品塔加 C、D 的四张统计桶表（supported 规则不变）；其余逐字段沿 M1-mean。三个 seed 42 / 2024 / 2025，对照为已有的 M1-mean 同 seed（466 / 481 / 471，不重跑）。判定沿 seq-m1-design §8.1 的表（对象换成“叠加”、对照换成“M1-mean”）；转正：行 2 成立 + 至少 2 个 seed 逐 seed exact McNemar p < 0.05 + 一次独立泄漏 review + 延迟报告。满足则 P anchor 改为叠加模型；否则 P anchor 保持 M1-mean。报告必须保留 D-1 与 23:2x 的限制（C/D 训练侧统计含自身 target）。**名额：需要 Eddy 批准 1 个**（原“条件组合”名额已用于 FEAT-04 组合）。
5. **双塔与生成式融合（G5）**：等叠加实验的判定出来再冻结（P 作业单个约 10–15 分钟，预计不会拖很久）；若叠加实验未获批准，则用 M1-mean 冻结。

### 2. 多兴趣召回（M2）：结论定稿，收口
1. 对固定主对照 M1-mean：Δhits −25 / −34 / −28，D = −29，k₊ = 0，逐 seed p 0.089 / 0.0045 / 0.040，p_pool 1.3e−4。**§8.1 行 1：“K=4 多兴趣读出及其分配训练规则（整体）在本设置下劣于 M1-mean。”** 不得写“多兴趣无效”，也不得把原因归给读出单独一项。对 F4-T010 附报为行 4（未检出差异）。
2. 诊断（描述）：hard 窗口四个头都被选中（有效兴趣数 4，没有饿死）；但头之间用户向量两两余弦偏高（例如 seed2025 头 2、3 之间 0.99，接近塌缩），serve 阶段 top1 获胜头分布很偏。可能的解释（推断，未验证）：训练时按 target 选头、服务时取四头最大分的不一致，加上多出的 3 个查询向量把头部的位置分给了较弱的兴趣。g→0 的差值（+113 / +157 / +124）只说明分支在模型内部有用，不是相对 M1-mean 的净收益。
3. **完整延迟不再补**：M2 劣于对照、不会转正，延迟只影响转正判定，所以记为 UNKNOWN 并写明原因；不为此申请读取 M2 checkpoint。
4. 不立后续多兴趣卡（M3 等）。

### 3. 双向序列模型确认（bi 另卡）：冻结
- 采纳草稿全文：bi 对 mean，两臂都用 M1 的均值读出与 `u = m + g·r`，唯一差异是 attention mask；三个新 seed 314159 / 271828 / 161803，两臂都重训，共 6 个训练 + 6 个 verify；判定与转正沿 §8.1 原文（对象 bi、对照同新 seed 的 M1-mean）；延迟按 §3.11 报告（转正需要）。
- **冻结前置（Codex 做，有一项不满足就停在草稿回 Kiro）**：在服务器 queue/runs 历史与本地全部分支、experiment_log 里只读检索这三个 seed 值，确认未被任何既有 run 使用；发现碰撞则不自行换值。
- 实现：新入口只新增文件，显式白名单只含这三个新 seed；原 M1 入口与其 42/2024/2025 白名单逐位不变；bi 用历史内双向敏感性 oracle，不套因果扰动断言。
- **授权**：前置通过后允许实现 → 一次独立 code review → 冻结 → 部署 → 真实输入 preflight → CUDA 预演 → 真实 shape 显存实测 → dry-run → 一次性提交 6 个训练并由服务器驱动自动补 verify（入队前告知 Eddy）。三个 seed 一律跑完。

### 4. 生成式排序导向训练：Eddy 批准 2 个新 G 线名额（R1 温度交叉熵、R2 SFT + 自生成难负样本 DPO）
- 卡 `docs/superpowers/specs/2026-10-17-g-rank-objective-card.md` 的设计采纳。冻结条件与授权：
  - **R1**：Codex 核对 τ 在 LETTER 论文或官方代码中的出处；出处唯一 → 写入卡并冻结；不唯一或找不到 → 回 Kiro。冻结后允许实现 → 一次独立 code review → 部署 → 真实 preflight → CUDA 预演 → 真实 shape 显存实测 → dry-run → 一次性提交三个 seed（42 / 2024 / 2025）并由服务器驱动自动补 verify（入队前告知 Eddy）。
  - **R2**：先实现并实测负样本挖掘吞吐，按实测估算 20,000 条、beam 10 的整批时间；**≤ 4 小时**则冻结并按与 R1 相同的流程推进（DPO 阶段的显存要实测，含参考模型）；超过 4 小时回 Kiro 缩小样本，不自行改 beam 或样本数。
  - R1、R2 互不等待，与协同向量 SID 并行，按准入放行。
- 对照：A1+T3 同 seed（289 / 328 / 310）；判定按卡 §3。

### 5. AGENTS.md 整理（Eddy 批准）
- 由 Codex 起草，**只整理，不改任何规则的含义**：项目 AGENTS.md 只保留当前有效的规则（以 2026-10-17「当前协作约定」与“工作流硬规则”为核心）；被取代的历史 owner / 模型 / 交接条款**原文**移到 `docs/history/agents-md-history-2026-10.md`，不删。
- 同时交一张对照表：原文每一节 → 保留 / 合并到哪一节 / 移入历史，并标出任何你认为含义可能变化的地方。Kiro 只审这张表与新文件，通过后再提交。整理期间旧文件继续有效。

### 6. 其余
- SID 全量规模基准首轮（流式原型，完整拟合按资源跳过）：不需裁决；全量路线决策时再看。
- 用户画像特征（F7）草稿：等叠加实验判定出 P anchor 后，按新 anchor 冻结。
## 2026-10-17 上午（续）：叠加实验名额获批；两条线并行的执行安排（Kiro 记录）
1. **Eddy 批准“序列用户塔 + 组合”叠加实验的 1 个名额**。设计按本文件「2026-10-17 上午」§1.4，由 Codex 按该节写成冻结卡（不再回 Kiro 审设计，除非实现中需要偏离）。**入队前置**：组合的定点独立泄漏 review（§1.2）APPROVED；叠加模型自己的实现照常 fresh RED → GREEN、一次独立 code review（须覆盖泄漏）、真实 preflight、CUDA 预演、真实 shape 显存、dry-run。实现可以和组合 review 并行，入队要等 review 通过。三个 seed 一律跑完；对照用已有 M1-mean 同 seed，不重跑。
2. **两条线并行的安排**（谁先就绪谁先入队，按准入放行，不为排队顺序人为等待；scheduler 按 8.19 核与当前显存上限 39,168 MiB 放行）：
   - P 线（短作业，单个约 10–15 分钟）：双向序列模型确认 6 个训练；叠加实验 3 个训练（组合 review 通过后）。
   - G 线（长作业，单个约 1.5–3 小时）：协同向量 SID 的自动补 seed（已在驱动里）；排序导向训练 R1 三个 seed；R2 三个 seed（挖掘成本 ≤ 4 小时才推进）。
   - 本地并行：组合泄漏 review、AGENTS.md 整理草稿与对照表、各卡实现与 review。
3. 服务器空闲超过 1 小时照常报告 Eddy；没有就绪批次时如实报告，不为填满机器加实验。
## 2026-10-17：排序导向 R1 的温度取值裁决（Kiro，**Claude Opus 5.5**）
事实（Codex 核实）：LETTER 官方代码唯一默认 τ = 1.0，等于普通交叉熵；论文只给了 0.6–1.2 的扫描，没有唯一的非 1 默认。
1. **不用 τ = 1**：τ = 1 时 R1 与 A1+T3 是同一个配置、同一批 seed，等于把对照重跑一遍，回答不了“排序导向有没有用”，白花 3 个 A1 训练。
2. **预注册 τ = 0.8，只取这一个值，不扫描、不按结果改**。理由：τ < 1 才是“更强调难区分的负样本”这一机制；在论文扫描的小于 1 的区间（0.6–1.0）取中点，不把结果押在扫描端点上。这个值没有更强的依据，报告写明“一次性先验取值，不是调参所得”。
3. **更正卡 §2 R1 的一处（冻结前、看任何 R1 结果前）**：原写“推理不变（beam 用模型原始 logits）”。训练时模型学的是 `softmax(logits / τ)`，推理也应用同一分布，否则跨位置累加的对数概率会和训练时不一致。**改为：beam 解码也用 `logits / τ`**。测试要求：τ = 1 时训练损失与解码结果都与原 A1+T3 入口逐位相等。
4. 其余按卡与本文件「2026-10-17 上午」§4 执行：冻结后实现 → 一次独立 code review → 部署 → 真实 preflight → CUDA 预演 → 真实 shape 显存 → dry-run → 一次性提交三 seed（42 / 2024 / 2025）并自动补 verify；入队前告知 Eddy。对照 A1+T3 同 seed（289 / 328 / 310），判定按卡 §3。
## 2026-10-17：组合泄漏 review 的补证方式（Kiro，**Claude Opus 5.5**）
依据：docs/superpowers/reviews/2026-10-17-feature-combination-leakage-review.md（CHANGES_REQUESTED，唯一阻塞项 P1：缺“A/C/D/E 四变体 L1–L10 在真实 engineering=False 路径完整运行”的逐项回执；其余 §7.6 条目已核通过）。
1. **不放宽，补一次真实路径检查**。FEAT-04b 冻结卡 §7.6① 写明要真实路径证据；reviewer 已逐行核过代码路径干净，所以补证的目的是排除“synthetic 夹具与真实数据形态不同”这一类风险，成本很低，不是重新怀疑代码。
2. **补证作业（只读，1 核，不训练）**：在服务器上用组合部署 `84b160c` 的正式 loader（engineering=False）读真实输入，取与组合 preflight 相同的 512 条训练样本 + 全部 2520 条 validation query；对 A、C、D、E 四个变体分别实例化模型（**不读任何 checkpoint**；用固定 seed 的初始化，并把 E 的 recency logits 与 β 设成固定的非零值，确保新路径被真正走到），逐项运行设计 L1–L10 中**运行时**适用的检查：L1 翻转 target 相关字段后用户向量逐位不变、L2 导出隔离、L3 历史侧反向对照（改历史事件自己的字段，用户向量必须变）、L4 周平移不变性、L5 物品 ID 一致、L8 week 25 输入被拒、L9（C/D）训练候选与导出候选逐位相等。L6、L7、L10 属于静态或构造器约束，reviewer 已按固定 commit 源码核过，在回执里标“静态，已由 review 核对”，不虚构运行。测试编号与设计编号的错位按 review 的对照写清。
3. **回执要求**：执行 commit 与源码 SHA、真实输入与侧表的 digest、每个变体 × 每项检查的 pass/fail 与样本数、运行路径（engineering=False）、test 未读声明。任一项 fail → STOP 回 Kiro，不修代码后静默重跑。
4. **关闭方式**：全部 pass 后，由原 reviewer（或另一 fresh-context reviewer）只读核对回执是否覆盖 P1，出 APPROVED 或仍 CHANGES_REQUESTED；不重做整份 review。APPROVED 即组合第四条件满足，叠加实验按「2026-10-17 上午（续）」第 1 条放行入队。
5. **附带记录**：review 发现组合最终回归原日志是 128 tests，而工程摘要写 130；以原日志为准，摘要更正，不推断另两项是否跑过。
6. **教训（写进以后的冻结卡）**：凡是冻结卡要求“真实路径运行”的泄漏检查，实现阶段就要做成一个服务器只读作业并产出逐项回执，不要等结果后 review 才发现缺口。叠加实验的冻结卡照此办理。
## 2026-10-17：组合真实路径泄漏补证的一次工程重验（Kiro 同意；Eddy 转交即视为确认）
1. **同意一次新工程 attempt**。失败点在回执落盘（父目录不存在），属于运载层，不是科学结果；该检查是确定性的通过 / 不通过检查（同一输入、同一 seed、同一初始化），重跑不存在“多试几次挑好的”风险，与“训练失败不重跑”的理由不同。
2. **范围**：只修新输出目录的创建与失败时 stdout 回执保留；重新部署固定 SHA 后运行一次。科学代码仍是 `84b160c`；输入（同 512 训练 + 2520 validation）、四变体、seed、非零初始化值、检查项全部不变；旧失败 run 的目录与日志完整保留；不改门禁，不重跑训练。
3. **一致性要求**：如果旧失败 run 的 stdout 里已经打印了任何检查结果，新 run 对应项必须逐项相同；不同就停下回 Kiro。
4. 这次再失败，停下回 Kiro，不再追加 attempt。
## 2026-10-17：非训练作业工程失败的常设一次重验授权（Eddy 批准，Kiro 记录）
写入项目 AGENTS.md「当前协作约定」第 6 条：只读检查、preflight、诊断、CUDA 工程预演、dry-run 若在产出科学结果前因运载层问题失败，允许修复后重验一次，无需逐次请批；科学代码、输入、seed、检查项、门禁不变，旧记录保留，旧 stdout 已有结果须逐项一致，再失败交 Kiro；训练与 verify 失败不在此列，照旧不重跑。每次使用记 issue_log。
## 2026-10-17：排序导向 R2（SFT + 自生成难负样本 DPO）两项补定（Kiro）
1. beam 10 内没有合格负例（非 target、且不在该样本完整 past_likes 中）的样本：跳过，不补抽；20,000 条抽样集合固定不变；报告跳过条数与比例。
2. DPO 阶段 min_epochs = 1（max 5、patience 2 不变）；best 只在 DPO 的第 1–5 个 epoch 中选，**不含阶段 1 的 SFT best（epoch 0）**，以免在 validation 上把 R2 构造成“不差于 SFT”。另报 epoch 0 的指标作复现性检查（应与 A1+T3 同 seed 一致，不一致如实报告，不是门槛）。
3. 卡 `docs/superpowers/specs/2026-10-17-g-rank-objective-card.md` §2 R2 已同步。挖掘吞吐实测后整批 ≤ 4 小时即可冻结并按既定流程推进。
## 2026-10-17：项目 AGENTS.md 整理（Eddy 要求 Kiro 来写，Kiro 执行）
1. 新 AGENTS.md 由 Kiro 重写，约为原文一半长；整理前全文原样备份在 `docs/history/AGENTS-2026-10-17-before-cleanup.md`（cp 原字节）。Codex 起草的历史归档 `docs/history/agents-md-history-2026-10.md` 与对照表 `docs/superpowers/reviews/2026-10-17-agents-section-map.md` 保留作参考。
2. **整理中裁决的冲突**（对照表“潜在语义点”）：
   - 旧硬规则第 7 条“Kiro 统一调度、唯一 writer”与 10-17 协作约定矛盾 → 以 10-17 为准：Codex 唯一执行者并维护 STATE / handoff / schedule / experiment_log / project_map，Kiro 只写 decision_log。
   - 事实源表 schedule 的 writer 由 Kiro 改为 Codex（与实际分工一致）。
   - review 触发点统一为“不可逆节点前”：正式提交前一次独立 code review、push main 前一次、删除或销毁前一次。
   - “每次只执行 tasks.md 的一个 task ID”删除：已不符合多批并行的实际；改为“并行任务用独立 branch / worktree”。
   - “含糊任务先复述等 Eddy 确认”与“已授权任务自主推进”合并：有 decision_log 书面授权的按授权推进，没有的先复述确认。
   - 旧 Astra / Sol / Claude Code / 2026-10-13 Kiro 接管 / 单 Codex 对话 / 15 分钟轻量兜底等条款不再生效，原文已归档；其中仍有效的约束（不伪称模型切换、接手先核原 job 与预算、automations 保持暂停不新增 timer、单一长期执行者）写进新文件 §1、§4。
   - 上下文条款去掉 Codex 特定的 50 万 / 65 万 / 70 万数值，保留“UNKNOWN 不猜、压缩后核事实源、按需换对话”，细则指向 `docs/context_session_policy.md`。
3. 新增两节，把散在各轮提示词里的铁律固定下来：§2「必须由 Eddy 单独批准的事」、§4「服务器作业纪律」（真实 preflight、claimed / rejected 确认、verify 原始 verified、显存声明规则、驱动规则、失败处理、非训练作业一次工程重验）。没有新增或放宽任何规则；§4 的内容都来自 decision_log 已有裁决与 Eddy 已批准的授权。
