密级: 公开

# Decision Log

## 当前边界

本文件记录已确认 Decision 与尚待验证 Assumption，不记录虚构实验结果。

本轮未下载数据、未训练模型、未验证效果。

哈啰内部信息、数据、schema、代码与实验结果不得进入本项目。

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
