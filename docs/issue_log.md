密级: 公开

监控schema观察问题关闭：已审固定monitor在下一既有heartbeat真实单次SSH snapshot成功，无解析异常；latest manifest数值排序及缺收据zombie拒绝修复一并经Astra本地scoped接受。真实证据`cache/p4e-monitor-recovery-20261007t2358.json`，不将此关闭称为实验最终通过。继续原队列监控，未修改运行端。

监控观察schema错误：临时inline读取`exit_code`而冻结helper.invoke写`exit`，导致只读收据检查提前退出。该错误属于观察器，不构成训练失败。固定ignored监控脚本及最小synthetic测试已本地修复，缺失字段不补成功；仍待下一次既有heartbeat真实观察关闭。未改训练/dispatcher/helper/source/budget或追加服务器操作。

# Issue Log

Outer真实invoke接口缺口：Astra发现局部SimpleNamespace未保留真实helper所需STDOUT/run，假Helper测试漏过。真实helper integration freshRED已复现child启动前AttributeError，改为保留原module属性的局部proxy并fresh scopedGREEN，原helper不改；awaiting coordinator source/tests acceptance。原review候选包保留，修订另立v2，不能把此接口失败当服务器训练失败。

P4E outer候选测试失败保留：最初全局替换helper.subprocess.Popen影响Mac ps子进程，fresh scoped测试明确失败；改为只替换helper的subprocess引用，原共享模块保持。sandbox随后禁止ps导致测试错误；明确escalation后只执行获批CPU短进程/自建group scoped测试，不接触server。独立OS终态检查先于test finally兜底，不能仅用作者cleanup_complete字段证明成功。Linux真实operation尚未执行，fixed包待fresh source/tests gate。

## 当前边界

P2已完成批准范围的服务器下载验收；P3真实数据审计与合同冻结完成，尚无模型训练或效果评估。当前整体gate以STATE为准。

下列项目都是开放风险，不代表问题已经发生或已经解决。

公司内部信息、数据、schema、代码与实验结果不得进入本项目。

## 状态定义

`Open`：风险已登记，尚无关闭证据。

`Investigating`：已有 owner 与检查动作，但结论未冻结。

`Blocked`：阻止后续 gate。

`Closed`：关闭证据已保存并可复核。

## Open Issues

| ID | Risk | Status | Impact | Next evidence | Closure criteria | Closure evidence |
|---|---|---|---|---|---|---|
| I-001 | `like` label 覆盖已train-only审计并冻结。 | Closed（工程smoke范围） | 可能无法形成足够 history/target，迫使定义 composite。 | 仅用 train 统计 user/item/event coverage、序列长度与 week 稳定性。 | 冻结 `like` 或 composite policy，并保存 coverage report 与 policy hash。 | docs/p3_validation.md v3 train/freeze SHA；artifacts/p3-data-contract-manifest.json。 |
| I-002 | `ir0.01` 的含义与 sampling 机制待查。 | Open | 可能改变 popularity、catalog coverage 或反馈分布，影响 MVP 外推。 | 核对 data card、subset manifest 与下载后分布。 | 明确生成规则，并在 data card 中写清可比性边界。 | 待补。 |
| I-003 | `timespent` 可能是聚合值，时序语义未知。 | Open | 可能把未来或聚合信息泄漏进 target-prefix feature。 | 核对 schema、粒度、单位、生成逻辑与同 user-item-week 重复。 | 证明 query-time 可用；否则从主任务移除，只保留审计。 | 待补。 |
| I-004 | `duration` 的粒度、单位、缺失与版本未知。 | Open | `timespent / duration` 可能失真或除零。 | 做 dtype/range/null/zero 与跨 week 稳定性审计。 | 冻结 clipping、missing/zero policy 并保存 feature manifest。 | 待补。 |
| I-005 | Flagship `ur0.01` 公开树可能没有独立 official test。 | Open | 调参与 final holdout 可能混用。 | 下载前核对公开结构；下载后保存 partition manifest。 | 冻结 train-tail validation 与一次性 final holdout，或采用明确的官方 split。 | 待补。 |
| I-006 | Thunder A6000 实时库存与 snapshot restore 未验证。 | Open | 两天窗口可能因实例不可用或 artifact 丢失失败。 | 执行 ≤1h acceptance probe：stock、SSH、CUDA、FAISS、吞吐、磁盘、reconnect、snapshot restore。 | 全部 gate 通过并保存无 secret 的 probe report；否则切 fallback。 | 待补。 |
| I-007 | Dataset、论文参考实现与未来依赖的 license/provenance 未完成。 | Open | Public repository 可能错误复制或分发受限内容。 | 建立 third-party provenance 清单，逐项记录 source、license、使用方式。 | 所有发布内容 provenance 明确；首次发布前再决定 repository LICENSE。 | 待补。 |
| I-008 | Full-catalog score 与 embedding/index 内存可能超限。 | Open | MVP 或 flagship 可能 OOM，或被迫缩小候选后误称 full-catalog。 | 用 manifest 规模做 storage estimate；运行 blocked top-k 与 ANN smoke。 | 峰值 RAM/VRAM/disk 留 ≥20% 余量，且 evaluator 不物化全 score matrix。 | 待补。 |
| I-009 | Exact evaluator已通过P4批准MVP验收。 | Closed（批准MVP范围） | 未来collision correction与旗舰算术仍须对应阶段验证。 | 后续SID/item与旗舰exact audit。 | P4 toy brute/blockwise/canonical-tie IndexFlat与真实replay一致。 | docs/p4_validation.md v2 source/run/verify hashes及Astra正式关闭；见文末I-009。 |
| I-010 | 大 catalog trie representation 未定。 | Open | Python nested dict 可能导致 RAM 爆炸与 decode latency 不可控。 | 比较 compact array/CSR/automaton 等 representation 的 build time、size 与 lookup。 | 选定 representation，在 flagship pilot 中满足 memory/latency gate。 | 待补。 |
| I-011 | 两天 MVP 时间风险高。 | Open | 数据清洗或 TIGER 训练可能挤占 evaluator 正确性工作。 | 先完成合同、toy evaluator、Popularity/content baseline，再启动 small TIGER。 | 两天内 P1～P7 固定 audit query 闭环；否则只交付 scaffold，不升级为 TIGER benchmark。 | 待补。 |
| I-012 | official-order假设下prefix已冻结并独立核对；隐藏timestamp残余见I-030。 | Closed（批准MVP范围） | 可能产生 future leakage 或无法定义唯一 next positive。 | 审计 partition/order keys、稳定排序与重复记录。 | 保存 order proof；无法证明则阻塞主任务。 | docs/p3_validation.md固定README/physical digest/Astra逐query oracle；残余I-030。 |
| I-013 | SID collision suffix 对 vocabulary 与 decode path 的影响未知。 | Open | SID-level hit 可能不能恢复唯一 item，或 suffix 增加稀疏 token。 | 报告 collision group size、suffix length、reachable rate 与 item-level correction。 | item mapping 可逆，指标分离，且无静默丢 item。 | 待补。 |
| I-014 | 固定 query audit set 可能不代表完整 test 分布。 | Open | 两天结果存在 sampling bias。 | 在 train/validation 可见统计上按 history length、popularity、first-seen 状态分层抽样并冻结 hash。 | audit set 构造可复现，覆盖预注册 slices，结论明确限定为 audit set。 | 待补。 |
| I-015 | `AGENTS.md`未落位。 | Closed | Codex自动发现规则与P1启动条件不成立。 | Eddy手动创建`AGENTS.md`。 | `AGENTS.md`存在并已读取。 | 本轮已读取受保护的`AGENTS.md`。 |
| I-016 | `AGENTS.md`曾混入易变事实。 | Closed | 每次对话占用过多context并产生漂移风险。 | Eddy手动落位thin规则；Kiro拆分STATE/handoff/tasks并复核。 | AGENTS只含稳定规则和指针；当前/历史/路线各有唯一源。 | Thin规则已落位；STATE/handoff/tasks完成二次职责收敛。 |

## 关闭纪律

关闭 issue 必须填写具体 artifact、hash、run ID 或公开文档链接。

“命令成功”不能单独作为关闭证据。

效果不佳不是关闭风险的理由；负结果同样需要 artifact evidence。

涉及 label、split、catalog 与 evaluator 的 issue 未关闭前，不发布效果 claim。

## 公开来源

VK-LSVD data card: https://huggingface.co/datasets/deepvk/VK-LSVD

TIGER: https://arxiv.org/abs/2305.05065

LIGER: https://arxiv.org/abs/2411.18814

SID staleness: https://arxiv.org/abs/2604.13273

Collision-faithful evaluation: https://arxiv.org/abs/2605.25330

## P1环境与实现复盘（2026-10-07）

### I-017 — Python/uv环境前置条件（Closed）

- 预期：P1有Python 3.11与uv可用，项目测试使用当前worktree环境。
- 实际：主协调接力记录系统Python 3.9.6，最初缺uv/Python3.11；主协调已按用户授权安装。执行Agent没有冒充自行安装。
- 影响：直接用系统Python无法满足requires-python，不能把缺依赖当TDD RED。
- 复现证据：本轮fresh运行已批准Python路径`--version`为3.11.17，`python3 -m uv --version`为0.12.23；当前`.venv/bin/python`复核3.11.17/PyYAML6.0.2。详见[p1_validation.md](p1_validation.md)环境准备。
- 排查与排除：不是项目逻辑失败；.local/bin不在PATH，使用明确路径；系统Python只启动uv。
- 已证实原因/未知：初始环境版本与工具缺失来自接力记录；本轮确认安装后版本。没有重演或伪造初始安装输出。
- 处理tradeoff：锁定3.11范围，不改shell配置，不再次安装Python/uv。首次依赖安装有网络，测试/CLI直接离线执行。
- 验证/学习：当前worktree独立.venv与实际uv.lock是关闭证据；面试时区分环境失败、无效RED与功能失败。

### I-018 — worktree metadata/cache/DNS的sandbox边界（Closed）

- 预期：已授权branch切换、必要依赖安装和真实lock生成可执行。
- 实际：git switch默认exit128（共享metadata的index.lock不可写）；uv默认cache不可写exit2；独立tmp cache解决写权限后，PyPI DNS仍失败exit2。
- 影响：阻塞环境准备，不代表config/runtime问题；未算TDD RED。
- 复现证据：validation环境记录和本对话实际工具输出；git switch经escalation exit0；uv lock/sync在独立tmp cache与获准联网escalation后exit0，installed pyyaml==6.0.2，uv.lock实际产生。
- 排查与排除：基点/branch冲突已排除，两者均df32a1a；未改Git配置、main文件或shell配置。cache写权限与DNS分别出现，不能混为缺包。
- 原因/未知：默认sandbox写入范围与网络限制；未调查OS网络全局状态，因为本次escalation已成功。
- 处理tradeoff：只为获准动作escalate；cache留独立tmp目录，运行与测试不调用uv，不借用其他worktree依赖。
- 验证/学习：fresh版本、lock与当前.venv元信息复核；worktree代码隔离不代表Git metadata物理上也在worktree内部。

### I-019 — 拒绝应发生在写输出之前（Closed，限定preflight）

- 预期：不合规output路径、已有目录、Git provenance失败在写入前拒绝。
- 实际：R05旧实现会覆盖已有result；B01虽最终报错却已在不合规目录写文件；B02 Git失败会先留下result。
- 影响：exit非0也可能产生副作用或误导性半成品。
- 复现证据：[p1_validation.md](p1_validation.md) R05/B01/B02 RED→GREEN及对应tests；测试只损及自身临时fixture，没有覆盖用户产物。
- 排查与排除：不是manifest hash或Git基点问题；原因是mkdir/result写入排在路径/provenance校验之前。
- 原因/未知：验证时序已证实。运行中输入并发变化、突然I/O失败仍可能留下本次新目录的部分产物；P1不实现transaction/自动回滚。
- 处理tradeoff：路径、fixture、Git与source hash preflight前移，mkdir exclusive、文件x mode；不自动删除任何失败证据，重跑选择新ID。
- 验证/学习：测试断言原字节不变及output不存在；面试时解释“报错正确”与“没有危险副作用”是两个验收条件。

### I-020 — fixture symlink与测试误通过（Closed）

- 预期：批准的fixture filename不能绕过repo-local人工输入边界。
- 实际：runtime最初未对固定fixture路径执行symlink校验。B03第一次GREEN实际因macOS /var临时路径别名提前拒绝，未触达目标行为。
- 影响：未经修正的测试会隐藏fixture可引用外部文件的问题。
- 复现证据：[p1_validation.md](p1_validation.md) B03意外GREEN说明、canonical temp root后的真实RED、修复GREEN；tests/test_boundaries.py。
- 排查与排除：使用内容合法的外部人工fixture，排除schema错误；canonical ROOT后仍触达Git sentinel，证明遗漏fixture confinement。
- 原因/未知：config路径与output有safe_path，fixture只拼接ROOT；ROOT别名使第一版测试提前退出。并发symlink替换不在P1单进程静态文件的保证范围。
- 处理tradeoff：runtime和build_manifest固定fixture路径也调用safe_path；不改变原fixture或扩大真实数据访问。
- 验证/学习：修正后先有assertion RED再GREEN，output不存在；不能把“测试通过”直接等同于“覆盖了预期分支”。

### I-021 — 人工oracle算术错误（Closed）

- 预期：synthetic加权和使用独立手算期望。
- 实际：R03最初误写229；正确是3×2+7×1+11×5+19×4=144。
- 影响：若未纠正，会把正确实现误判为错误。
- 复现证据：[p1_validation.md](p1_validation.md) R03手算修正、重新RED与GREEN；当时生产seeded计算尚未实现。
- 排查与排除：权重与输入人工复算，不用生产helper构造oracle；并非随机数版本或实现bug。
- 原因/未知：测试编写算术错误已确认，无模型或真实数据影响。
- 处理tradeoff：先纠正oracle再fresh RED，未调整生产代码迎合错误数字。
- 验证/学习：literal144与实际权重匹配，换seed会变；测试oracle同样需要审计，不能以测试权威代替推算。


## P2首轮问题（2026-10-07）

### I-022 — 当前费率无法在无凭据接口中核实（Open）

- 预期：首轮30分钟/新增计算费用US$1上限可在操作前预测。
- 实际：按服务器指南读capabilities并仅筛选费率allowlist，返回rate_fields空；当前计算、存储、下载流量费均Unknown。Exit0只是查询成功，非预算门Passed。
- 证据：`p2_validation.md`费用门命令与脱敏JSON。
- 处理：未读取/使用计费API credential；立即停止GPU、clone、下载等后续远程动作，完成本机输入和公开元数据核查。未停止实例，空转仍可能计费；实际扣费未知。
- 关闭条件：Astra核实当前控制台费率及具体动作成本，确认批准时限和US$1上限内可控；补来源、UTC与费用推算，不照搬旧截图价格。

### I-023 — 无独立volume时持久化/恢复尚未闭环（Open）

- 实际：capabilities本次报告workspace_is_volume=false；guide说明无volume的容器文件会在recycle/destroy后丢失。
- 证据：`p2_validation.md`存储属性输出及指南读取记录；另fresh核对官方 https://docs.vast.ai/guides/instances/storage/types 。Container storage在stop时保留，destroy时丢失，stop仍计storage费；volume绑定同一host且独立计费，不是跨host容灾。这是属性/文档证据，不是恢复实测。
- 处理：未创建marker、备份、snapshot或执行stop/destroy/recycle；原有恢复gate保留。无工作盘free容量/quota结论。
- 关闭条件：具体持久化与服务器外备份方案经用户批准并实测恢复、hash一致，记录保留/不保留内容、费用与耗时；不能凭SSH成功关闭。


### I-022续执行 — Unknown但非阻塞

Eddy撤销费率gate及原US$1费用门，D-051为明确授权证据。本轮继续已批准搭建与下载；费用仍Unknown，不能伪造费率或扣费，也不再用此未知项阻塞。历史暂停记录保留，不当作当前gate。

### I-024 — 现有Python不满足P1与FAISS GPU未提供（部分解决）

现有torch环境Python3.12.14、PyYAML6.0.3，不满足P1范围。使用uv建立项目独立Python3.11.16/PyYAML6.0.2，31tests与smoke/verify通过；未污染现有环境。FAISS原缺失，独立cache环境装CPU1.15.1，IndexFlatIP toy通过；GPU数0，原runbook GPU可见门尚未通过，不冒充GPU FAISS。证据见p2_validation续执行节；GPU验收口径/后续方案由Astra裁决。

### I-025 — 短probe不能替代quota/长任务/完整恢复验收（Open）

工作盘statvfs显示足够空间，但quota工具未提供；64MiB direct-I/O通过不保证持续吞吐，reconnect文件保持不保证后台长任务resume。无volume的stop/start文档保留不等于本轮已恢复实测。Astra审查具体后续方案与预算，危险恢复动作仍明确审批；本輪未自动stop/destroy/new volume。


### I-024续执行 — GPU FAISS环境问题Closed（限定验收）

官方pytorch GPU FAISS1.11.0精确py3.11/cuda12.1 build与libfaiss在独立conda cache环境安装；GPU count1，8×8 synthetic IndexFlatIP CPU/GPU及独立oracle相符。Sol与Astra各自fresh验证exit0，artifact/UTC见p2_validation。GPU wheel1.15.1/Python3.11初始解算失败及中间未安装的大依赖方案保留；没有改driver/torch或以CPU通过替代GPU。

### I-026 — tmux首版未生成日志（Closed：第二版限定probe通过）

首版tmux启动exit0但无日志，后续FileNotFoundError exit1；stderr未保留，根因不能单凭猜测定为引号。第二版为明确synthetic文件，先compile/py_compile再运行、保留stderr、隔离socket/config。首SSH退出后8步/DONE完整、stderr0；Sol和Astra各自复核hash一致。证据见p2_validation；只关闭这个probe失败，不保证任意训练resume或实例恢复。

### I-025续执行 — tmux已覆盖，quota/恢复仍Open

小型synthetic断线保护通过，但独立quota工具不可用、overlay无可见quota flags不能证明无隐藏限额。46文件冻结bundle只准备，不代表backup或恢复已执行；实例/host-loss gate继续Open，须Astra向Eddy审查。


### I-023/I-025目录恢复覆盖更新 — 完整gate仍Open

批准的server→Mac→server小bundle在全新checkout恢复46/46hash并独立重建P1环境，31tests及新旧manifest通过；此有限恢复项Passed，完整实例/host-loss/GPU环境/原始数据重获未演练。不可把partial覆盖写成完整Closed，待Astra复核裁决。

### I-027 — 恢复成功后本机元数据记录KeyError（Closed：记录补正）

local validation使用exec覆盖expected变量，远程恢复已完成但本机末尾记录archive_sha256发生KeyError，首轮exit1。补正本机记录、不重跑恢复；后续46hash、fresh环境/31tests/smoke/旧manifest通过，报告与日志保留。临时验收orchestration未进入生产代码，失败不隐藏。


### I-023/I-025 Astra裁决 — 已批准P2恢复项通过，剩余风险保留

Astra独立Mac包/服务器46hash、HEAD、31tests及新旧manifest复核通过。Eddy批准的是目录级backup roundtrip/P1环境重建，故本次P2恢复验收项Closed（限定范围）；旧条目更强的实例/host-loss条件不自动扩充为本次授权。无volume及host-loss/stop-start/GPU环境/raw重获未测风险、隐藏quota未知仍Open，P8获准后再验。不能将验收通过解释为风险已消失；证据读p2_validation最新Astra验收节。

## P3复盘与剩余限制

### I-028 — 新服务器源码包环境漏项（Closed，限定P3测试环境）

- 预期：新唯一P3目录能fresh跑P1与P3 synthetic regression。
- 实际：首次包未带outputs容器和.python-version，连续暴露FileNotFoundError；tar另有macOS xattr warning。所有失败logs保留在P3 cache，train/holdout尚未执行。
- 原因：原创源码包清单未包括P1运行fingerprint的公开环境文件及test临时目录容器，不是数据或P3行为失败。
- 处理：补公开.python-version/.gitignore与新outputs目录；Git对象只读使用P2 object context。后续code-v2采用--no-xattrs独立快照并fresh完整regression。
- 证据：docs/p3_validation.md、server cache/tests-environment-fixed.log与tests-code-v2.log；无覆盖旧运行产物或改Git配置。
- 学习：目录级代码包也要覆盖运行metadata，不把environment error记成RED。

### I-029 — `irX`公开生成定义不完整（Open）

- 固定revision README列出ur0.01_ir0.01，但定义段仅解释urX/ipX；不能凭名字宣称ir是随机item采样。
- 主train/holdout可按固定文件审计；结果仅适用于此subset，不外推全量分布、公平随机抽样或无偏popularity。
- 下一证据：作者官方生成代码/明确文档；未取得前保持未知，不复制未审计第三方代码。

### I-030 — 顺序证据仅官方声明与physical读取（Open，已限定协议）

- 官方固定README声明week内timestamp顺序并给保序subset join示例；实际12列无timestamp。
- P3验证physical顺序两次digest一致、schema/week集合与无重复pair；不能独立证明隐藏timestamp单调或此subset逐行保留全量parent顺序。
- 当前policy显式使用official-order assumption；如协调review不接受该证据强度则阻塞后续训练，不能伪造timestamp补门。
- 证据与hash见p3_validation及frozen manifest；后续需作者澄清或公开事件顺序键才可关闭全部残余风险。

### I-031 — 独立review发现policy与producer绑定缺口（Closed，限定实现与复查）

- 原min/max validator只检查范围，freeze能接受不同于train硬编码的history规则；train weeks的bool在Python equality下可误通过。
- 原train audit没有实际producer信息，freeze记当前hash与硬编码Git身份，holdout没有提前验证源/lock/库版本。旧audit因此可能被新代码消费。
- 独立Sol review由Astra安排；已fresh RED复现并最小修复，详见p3_validation fix round1。此前v1/v2 audit保留且明确不可freeze；未读取holdout或训练模型。
- 历史fix1时点：46tests通过；独立Sol scoped re-review三项addressed，独立拒绝复现与source fingerprint见p3_validation。当时实现风险关闭，v3真实audit与整体P3验收尚未完成。
- 后续闭环：v3真实train audit、freeze及holdout均exit0；Astra独立producer/freeze hash核验及逐query target/history/OOV核对通过，详见p3_validation的v3与holdout章节。最终独立Sol review public manifest/hash/producer亦通过，无source/spec新阻塞；整体P3最后文档验收仍由协调者裁决。
- 兼容规则只允许CPython3.11 patch跨Mac/Linux差异，源/lock/Git基点与精确库版本保持一致。

## P3数据风险收口证据

I-001按批准工程smoke范围关闭：v3 train audit与frozen policy manifest/hash见p3_validation，支持门在读取holdout之前通过；未声称统计效力。I-012仅在批准official-order assumption下关闭：固定README hash、physical两次digest及Astra独立逐query prefix/target位置核对通过，残余hidden timestamp风险仍由I-030记录。

I-003/I-004：timespent/duration只做官方单位与dtype/null/range审计，不进入当前label/history或模型feature，主任务不存在这两字段引入的feature leakage/除零；未来使用时须重新审计注册。I-002/I-029抽样生成定义不完整保持Open，不猜ir含义。I-007完整发布provenance仍留P13。


### I-032 — P4 GEMM block shape导致相同向量tie失真（Blocked）

独立review在NumPy1.26.4以seed42生成2x64 query、1x64 item归一化，再重复item三次、IDs[3,2,1]、k1/qblock2/iblock2，brute输出[[1],[1]]、block输出[[1],[2]]。float32第二query score存在不同最后位；不是candidate或policy改变。简单2D literal fixture未覆盖BLAS reduction shape差异。停止v1效果比较，保留v1源码与run为invalid candidate；下一步固定每pair reduction、fresh反例RED→GREEN及跨block regression，然后新v2真实run与独立验收，不用epsilon修改tie定义。


### I-032 fix1 closure（Closed，限定source/scoped review）

改为float32 feature输入、固定float64 einsum dimension reduction，反例fresh RED→GREEN，多block regression通过；独立Sol额外20组mixed64D/repeat测试与spec/quality通过。v1 raw计算仅停止作者已核实专用任务，原件aborted_invalid保留；该source/scoped-review时点v2真实runtime尚未关闭整阶段gate；后续正式关闭见I-009。证据docs/p4_validation.md与v2 source fingerprint。

### I-033 — P4 relative输入证据与absolute replay不一致（Closed，限定source/scoped review）

独立review发现run写relative evidence，verify从manifest absolute roots重新load导致binding mismatch；fresh relative run→verify ERROR复现，run/load_static入口统一canonical roots后GREEN，独立scoped review通过。Public只脱敏manifest，不公开私有路径/连接参数。

### I-034 — P4文档状态补记脚本编码错误（Closed，限定补记）

真实run成功后一次本机临时STATE补记脚本报Non-UTF-8 SyntaxError，未执行状态替换；同一shell后续public manifest检查通过，所以末尾exit0不能冒充前面脚本成功。使用明确.venv/bin/python和UTF-8 coding重新执行，STATE已补正；不重跑实验/不修改生产source。编码来源未独立定位，不能猜为数据或baseline失败。证据docs/p4_validation补记与实际工具输出。

### I-009 P4 MVP evaluator证据更新（Closed，限定批准MVP范围）

P4 v2 literal oracle、跨block64D identical-vector反例、brute/blockwise与canonical-tie IndexFlat toy均通过；真实两baseline run和完整prediction/metric replay通过。独立Sol fix1 source/spec/quality通过，Astra另独立Popularity/抽样/hash/metric核查通过。I-032/I-033 source问题已闭环，v1 invalid保留；Astra正式关闭P4 gate，独立Sol最终spec/quality与文档/public manifest review通过，无新增阻塞；I-009按批准MVP evaluator范围Closed，不以单一exit0或作者自检代替。

I-008 MVP已证明bounded score blocks与实测process峰值（唯一数字experiment_log），不关闭旗舰scale/OOM风险。I-014固定seed hash-rank audit用户可复现并保留OOV，但representativeness风险仍Open，不把512 audit当全test或无偏平台评估。

## P5 review资源门修复

### I-035 — 最后资源gate前发布success manifest（Closed，限定P5发布门）

v1 run在write_artifacts内发布success manifest，最后一次resource sample低于20%才报错；static reader不要求producer/resource，失败目录可能仍可被后续消费。独立Sol指出后，fresh assertion RED用high/high/low控制序列复现run报错但manifest已存在；另缺producer/resource的重hash manifest被旧reader接受。fix1将data bundle与成功发布分离：exclusive写data，stop/join采样线程、最后sample及全部gate、strict success schema后最后xb写run_manifest；失败保留部分数据，无success manifest。Static reader拒绝缺producer/resource/sample/timing证据。真实v1保留且不验收，验证入口p5_validation与cache/fix1-red/green/regression。仍不保证进程突然crash的filesystem transaction或并发改动。

### I-036 — monitor采样异常未传播主线程（Closed，限定P5采样失败传播）

v1 monitor遇OSError可线程退出而主线程仍成功；独立Sol review指出，fresh assertion RED以background-only OSError+慢loader复现未抛错且产生manifest。fix1捕获采样/资源异常，fit node/iteration与阶段boundary检查异常和观测样本，finally stop/join，发布前确认无pending worker error。GREEN测试在主线程仍可运行时明确拒绝成功且manifest不存在。Resource声明为250ms可见host采样加阶段边界，不保证瞬时硬限制、hidden quota或其他租户影响；source/真run/review证据只读p5_validation。

I-035/I-036关闭证据：独立Sol fix1 scoped re-review通过，Astra fresh local全suite与server v2全suite通过，source c61ffbc1651dd0b8bd508f498cdde265de886fd4df45aeede6f7ebc1e2e65729；fresh assertion RED/GREEN分别复现和修复失败success发布/monitor OSError传播/static缺证据。v2 manifest669029504ab18df5f6a6109820d3d4a36a243f6ab0d3e8c46fc8ce17b1b6bbe9及完整producer replay通过。关闭这两项代码问题不代表P5整体gate已正式关闭或资源瞬时硬限制成立。

## P6实现与验证复盘

### I-037 — 两条训练路径一致仍可能两个都没训练（Closed，限定synthetic验证覆盖）

独立Sol指出checkpoint equality test只比较不中断与resume两条路径；如果optimizer.step被no-op，两条都不训练也能equal。dropout0不消耗global RNG且rebuild重设seed，删除RNG恢复也可能误过。增加至少一个参数相对初始值确有变化；保存前推进torch/numpy/Python RNG，rebuild后扰乱再load检查下一值。Global beam原toy4items/width20也没有pruning，补81item/width5独立逐层global oracle；causal从仅position0补到所有有future的位置。独立临时副本取消global裁剪/RNG恢复各目标test exit1 FAIL，真实source GREEN。证据p6_validation。只关闭测试洞，不以synthetic等价冒充真实GPU resume。

### I-038 — fresh-process verify先比较环境导致deterministic误拒绝（Closed，限定P6初始化）

训练seed_all使deterministic_algorithms=True写入manifest，新process默认False；旧verify在seed_all前比较environment，正确run也会拒绝。作者自检与独立Sol同时识别，正式GPU训练前修正为CUBLAS_WORKSPACE_CONFIG及seed_all先于env比较；fresh subprocess测试确认已越过environment gate推进到预期missing-config。临时独立副本移除初始化时test exit1 FAIL；真实source GREEN。环境不是随意忽略差异，而是初始化后仍精确比较Python/torch/CUDA/依赖/lock/backend证据。

### I-039 — server源码快照遗漏既有测试上下文（Closed，限定P6测试环境准备）

首次上传只src/tests/config/base files，忘记read-only Git对象上下文、P3/P4/P5脱敏public synthetic fixtures、outputs父目录，三次fresh全suite分别失败，不把shell退出或partial tests当通过。补齐合法小文件及GIT_OPTIONAL_LOCKS=0/GIT_DIR/GIT_WORK_TREE后fresh server全suite通过；旧失败log保留，具体命令与结果p6_validation。没有改旧tests放宽assertions，torch核心tests实际执行。以后按完整允许文件清单与测试依赖检查snapshot，避免逐文件trial-and-error。


## P7比较与接力风险

### I-040 — 历史阶段状态误当当前状态（Closed，限定文档指针）

P6已正式关闭，但experiment顶部仍写尚无模型、data_contract当前段仍写P3 awaiting；P7核对发现后将允许文档改当前状态指针，合同两句经协调明确授权修正，历史run/失败/review过程保留。冻结规则、输入、config、source与artifact不变。旧public P5 manifest的awaiting字符串是原快照hash绑定的一部分，P7不修改历史manifest，阶段正式验收以STATE和p5_validation为准。

I-014 audit代表性、I-029 irX语义、I-030 hidden timestamp顺序风险继续Open；one-seed无统计胜负证据、共同catalog的OOV不可达、P5有限迭代未收敛/greedy objective非单调均继续披露。P7独立核算通过只解决本固定audit比较输入/公式一致性，不关闭这些研究限制。

P7只读脚本最初用P6“config”字段，实际为effective_config，KeyError exit1后修正并fresh exit0；初schema探查用P4 predictions通用字段也KeyError，实际为分方法字段。均为审计适配失败，原tool记录保留，不把失败或shell后续成功冒充验收；生产及产物未改。

本地fingerprint复算首次错误地对base+src全列表排序而与producer的base先行顺序不同，assertion exit1；按manifest.py定义保留base顺序后fresh PASS。另process检查临时inline shell quoting导致SyntaxError exit1，改独立脚本stdin后fresh PASS。仅验证脚本修正，不改production/hash或资源。


## P3-C 数据质量补充风险

### I-041 — Metadata snapshot与匿名编码未提供（Open，core排除）

完整train-selected metadata schema/null/join与anonymous code profile通过，数值只读experiment_log的P3-C节；age/gender/geo没有业务reverse mapping或effective time证据，geo zero不能直接当missing。Item/user train_interactions_rank为全train统计，把它输入早期prefix会包含未来互动。Author/duration/content snapshot也没有独立历史时点凭据；旧content feature只沿用固定content-only benchmark assumption。

处理：不fill、不删、不推编码，core排除user属性与metadata ranks/author/duration；content静态来源/时点限制显式注册。数据null=0只说明覆盖，不证明point-in-time。正式新合同P3-D承接；不能由本轮gate通过关闭上游snapshot未知。证据见data_quality point-in-time registry与p3c_validation。

### I-042 — Catalog zero-like与head贡献集中（Open，保留主任务分层）

完整train item/author支持和user prefix分层显示长尾/zero-like与贡献集中，数字唯一experiment_log；这不是脏数据或bot成因证据。Alltrain catalog需要保留无正例支持items；ID-only表示可能缺少正例支持，content仍提供表示，不能缩catalog后称公平提升。

处理：raw immutable，主样本保留低活跃/长尾，不5-core；unknown ratio/255/cofeedback异常保留flag。后续模型支持分层/user balance消融由已批准阶段独立实施，本轮不训练或提前变更label/权重。Data质量审计通过不代表ID-only检索性能已证实，也不证明head users异常。

## P4B implementation issues (all evidence preserved)

- Train-positive support起初由eligible examples反推，会漏掉single-like用户独占item。协调review指出后补fresh assertion RED，改为全train like events显式support artifact/ID-map绑定；singleton synthetic GREEN。真实prepare后Astra与作者分别独立全raw oracle一致，support问题已验证修复；正式task仍待协调关闭。
- 第一次source-upload ledger误用system Python缺PyYAML，尚未上传source；空目录与失败日志保留，借已有批准Python env生成ledger并创建新exclusive目录。没有安装依赖或复用失败目录。
- 第一次real audit请求错误旧producer文件名，exit2。fresh RED/GREEN改为既有frozen-contract-manifest.json；新source/new目录audit exit0。原失败证据保留，冻结producer未改。
- 在原prepare进程terminal前执行verify，manifest尚未生成，exit2。这是操作顺序错误；原prepare继续且不重启，原prepare已terminal exit0后用新日志名verify-prepare-final验证exit0；旧失败证据仍保留。不能以缺manifest快照猜测prepare失败、覆盖exit或重复process。

## P4C verifier review finding（已修复，独立复核通过）

v1 verifier仅在persisted outputs之间重算metrics/ItemCF；错误content或错误graph与自身metrics/hash同步后仍success。协调独立复现后freshRED四assertions，v2要求pinned raw train重新建graph、pinned features重算content与三route全部predictions，并核final/initial artifact hashes不冲突。Fresh GREEN11与全regression123通过。当时独立fix review/真实full replay尚未完成、仅v1pilot运行，这是历史记录。后续final source/scopedreview通过，v2 pilot/full/verify实际terminal success，Astra独立全raw/pair图/全部三route ranks、metrics/coverage/fill/旧audit均PASS；v1 source/pilot原样保留。Task正式关闭仍需docs/public review与协调裁决，不能凭作者tests关闭。

P4C接力消息中曾误给未经读取确认的runtime manifest hash；立即撤回并fresh读取ignored aggregate更正。仓库/服务器文件没有写入该错误值，experiment_log由实际aggregate自动写入。Closing public manifest将programmatic提取实际hash并fresh核bytes，不凭手写聊天值生成证据。

## P4C coordinator content oracle归约差异（oracle修正，全量裁决待定）

独立coordinator最初对单个float32 mean使用1D np.linalg.norm，冻结旧P4/新P4C使用2D rows axis=1 norm。归约产生last-bit centroid差异，近tie的top ranks交换；诊断确认匹配冻结norm轴后global lexsort严格一致，fixed-pair d,id与qd,id scores逐值相等。Production/容差/tie rule均未改，失败log和diagnostic保留。此问题限制“任意数值实现排序相同”的声明，不代表数据/contract漂移；Astra修正版全量独立oracle已exit0/PASS，未改production/容差；正式task仍待docs/public review与协调裁决。

## P4D source-stage findings（真实run前修复）

- 初次入场suite默认sandbox无worktree临时output写权限，产生PermissionError；正确权限rerun通过，原log保留。不是模型或数据失败。
- STATE启动脚本按main格式replace而父acceptedSTATE格式不同，未实际改snapshot；协调fresh指出后定点重写并实际核读。命令意图不等于文件变更。
- Best更新初版发生在checkpoint保存后，manifest best可与checkpoint budget不同；fresh tamper RED证明改写manifest metric不被拒绝。修正logicalbest存储时序与verify_session selection/role/stat/epochchain核验，freshGREEN通过。
- 存储vectors到rank/metric自洽不证明模型输入链；fresh同向缩放vector并rehash可保持rank。新增每epochcheckpoint重导出与逐float核验，篡改RED→GREEN，formal verifier组合全部epoch。此时没有真实训练产物被更改或重stamp。

- Source-ready最后组合cursor核验时一次字符串replace同时匹配function声明，完整regression捕获SyntaxError；定点修声明后fresh rerun，旧失败log保留。未部署或启动真实run。

P4D独立v1 source review拒绝三个gate：CLIstr参数reader崩溃、混合sharedbatch/pilot可被选择、pilot缺ranking live context。协调另复现nonfinite被当capacity降batch。全部freshRED24–29修复后GREEN；实际source v2仍待scoped review，未真实运行。自检RED30再加固定content/support checkpoint buffer一致门。失败ledger与旧source package保留，未经通过不启动realpilot。

P4D source gate PASS后上传被automatic approval review拒绝：审查未识别既有human授权及payload/target归属。实际公开105文件小tar已逐hash核；请求被工具pre-execution拒绝，server未写入。补原始human授权/manifest证据前不重试或间接绕过，当前不是CUDA/resource或模型失败。

P4D上传审批在补可信原始human授权与105public文件逐member证据后，同原通道透明唯一重试获准，actual exit0。随后launcher生成器用全局PY/CODE替换污染ENV名称；sh语法检查无法发现bad variable name，remote第3行立即停，PID不存在且无stage产物。实际本地shell export RED复现，literal三ENV行GREEN通过；保留原launcher与日志，使用exclusive v2b修正脚本。失败发生在GPU/metrics前，不消耗正式调参run；生产source冻结。

v2b前四stage实际通过后，pilot因操作脚本把prepared_manifest文件传到需要目录的--prepared接口而退出；实际NotADirectoryError复现，manifest正确hash通过，resource-pilot目录不存在，零batch测量。修正为prepared_manifest.parent，仅独占pilot-v2c日志再次调用，旧exit2保留；不重复GPU恢复、不额外调参。后续pilot/train/verify CLI目录与file参数已逐接口核对，记录ignored path-contract。

首真实bounded操作在初始resource_snapshot(output.parent)时因runs父目录尚未存在而FileNotFound退出，实际只读复现；零model/optimizer/checkpoint/output，slot1预算未消费。v2b操作脚本完整preflight补exclusive parent创建、既有runtime/plan一致、目录/file/symlink/logs、CPU prepared/query与pilotbinding检查；local临时fixture执行实际脚本语句4scenario PASS。原failed plan/log/exit保留，生产source不改；新操作脚本独立review后续同slot1步骤。

Astra首真实checkpoint审计的第一次oracle因checkpoint best_metric tuple与JSON list直接比较产生exit1；诊断后仅做JSON序列化标准化，fresh独立oracle通过。生产source、容差和产物未修改，这属于验证表示差异，不是恢复或模型失败。

正式队列live poll曾在最新manifest open/write窗口解析空JSON而exit1，训练PID仍正常；Astra随后独立读取同run完整4900checkpoint状态通过。只读监控改为明确列出最新未可解析文件与上一个完整JSON；这只用于进度展示，不用于resume、不回避终态corruption。正式pipeline只在child.wait之后读取，因此其最终严格校验与生产source均未变。

首配置train终态exit0，但formal verify.exit2，queue实际停止、slot2未启动。actual CPU trace落verify_session:177；独立逐子条件best/selection/records/evaluations一致，仅stats epochs整数key与JSON字符串key超过十个epoch后排序表示不同，规范化后全部值一致。verify_draw_provenance有同类缺口。旧producer/产物不改；协调批准独立双指纹兼容adapter，先TDD/review，不能宣称run01已验收。

Formal verifier可观测性限制：其正式验收会重放全部epoch/checkpoint、完整向量与exact ranks/metrics，耗时较长；冻结实现没有逐阶段/逐epoch进度日志。外部只读ps、CPU/I/O、GPU和可见打开文件只能确认活动，不能可靠定位当前epoch或给出ETA。持续计算且尚无terminal不等于失败/卡死；不为监控改冻结source或注入正在运行进程，最终耗时仅在真实终态后补记。

首run全epoch兼容formal verifier actual0，双身份receipt/gates hashes经Astra fresh独立核，首run正式accepted（不是全P4D完成）。恢复24slot的操作v1在slot1receiptreuse read_json(str)接口错误处退出，未训练slot2。新v2仅两处Path包裹，实际原reader/operation body的gate-reuse、existingreceipt-reuse与selectionreuse三路径fresh RED1→GREEN0，禁止subprocess/train/verify。旧v1及失败log保留，producer/adapter/pilot/预算与已accepted产物不改、不重复5100verify；v2 scopedreview后继续原队列。


## I-AGENT-MONITOR-001 — 正常等待期间重复监控消耗

- 状态：规则已修订，实际交接 pending。
- 事实：Astra与Sol在正常训练/验证等待中重复轮询、每分钟重复汇报，造成不必要token消耗；无训练故障，具体节省量未测量。
- 用户要求：单监控者、长期可见对话、只在阶段验收/失败需要时升级协调。
- 处理：原服务器队列继续，不kill/restart/重复提交；本P4D子agent交接ready后停止周期轮询和文件写入，由已存在独立Sol对话接管。主目录AGENTS已修订，P4D冻结source及远程AGENTS不修改、不重新上传。接管确认前不写已完成。
- 证据：cache/p4d-visible-thread-handoff.md及原queue实际存活快照；接管确认待协调记录。


### P4D可见监控接管权限限制

独立可见Sol对话已实际接管，但自动审批分别拒绝向Astra发接管确认和创建每5分钟persistent heartbeat，理由是未识别直接人类跨线程消息/recurring授权。授权选项已直接提交Eddy，未绕过拒绝、未创建持续自动化；正常当前运行内有界只读检查不受影响，服务器既有队列持续自主运行。跨对话确认保留本对话/STATE/handoff供协调读取，避免声称Astra已收到消息。


### P4D监控收口及观察错误保留

单监控对话实际接管并完成作者交付，heartbeat最终实际PAUSED，未测token节省。旧权限拒绝保留为历史。最终checker误将run01 exit定位后续attempt导致FileNotFound，按历史独立attempt及已accepted链修正，无重训/改产物。mean/std exact float相等因运算顺序舍入失败，fresh reltol1e-14/abstol1e-16通过，正式数字未变。审批一次因模型capacity未完成待办写入，后续只读snapshot实际成功，无绕过。暂停前本机缺tomllib读取失败，以ast读取同TOML字段后purpose-built tool实际PAUSED；未直接改automation文件。文档脚本一次stdin编码失败，补显式UTF-8后执行；不冒称失败命令写入成功。

### P4D质量限制（开放，非执行stop gate）

内容及Transformer结果弱，部分固定seed best validation无命中，全部保留于experiment_log。预算/finite/artifact/rank一致性/资源门通过不证明质量可用；OOV未显示检索能力。无额外优化、因果归因或subset外推parent/线上。后续baseline使用与阶段放行由Astra fresh验收裁定。
# P4E source review recovery notes (2026-10-08)

Formal dispatcher scoped review Important：首候选未绑定operation引用的production plan与segment.binding.plan及watch，可能将其他watch的pilot提供给formal consumer；旧fixture遗漏真实字段，scoped测试通过未覆盖此风险。作者fresh RED复现并仅ignored代码修复，外部生产plan/watch pins、所有segment binding及dispatch prechild watch一致性补齐，局部rehash negative tests通过。原reviewer定点复核待定，不能据作者GREEN声称review关闭；无原GPUproof/生产源码变更或formal启动。

P4E v2收口观察曾将outer handle按child schema读取process_identity，产生KeyError；这是只读观察脚本错误，不是运行runner失败。按实际outer birth_ticks与child process_identity分别核原身份后观察通过。未修改生产、运行runner、plan或budget，未重启或重试。旧P4D heartbeat已暂停的终态不覆盖当前P4E监控职责。

- Astra first source review阻塞：prepare-watch/verify-watch没有persistent operation gate，允许每invocation重置预算和越界输出。新CLI测试fresh复现deadline/disk/path三个RED，修复后全拒绝且before real input read；当前交修订固定包等待Astra重新验收。
- CPU/CUDA重导出差异首轮为未实测风险。现绑定producer execution device/FP32/deterministic/TF32，verify同device且禁止CPU fallback，精确ranking/metric比较保留。真实CUDAproof仍Open，不能冒称通过。

- 测试generator语法失败为无效RED；恢复后fresh目标assertion RED及GREEN已实际执行，未补造历史。
- 首次新ROOT完整suite的旧P1 outputs受sandbox限制；正常自动审核放行本地测试后此权限错误关闭，原失败证据保留。
- 新ROOT旧P4D兼容身份测试拒绝新增consumer source：有意保留历史producer pin，不改fixture或旧源码。冻结legacy snapshot独立完整回归通过；该新ROOT失败单列，不称全绿。
- 开放operation gate：真实CUDA独立进程恢复、实际CLI/exit/receipt/completion封装及实测资源预算尚未核；本地synthetic和进程内pilot不替代这些证明。Astra source及operation审查前不SSH、不真实运行。


### P4E runner 最小修复与开放 gate（2026-10-08）

新增函数返回 receipt 的预期 RED 已由最小 ignored runner 修改关闭，本地回归与独立CLIverify通过。旧 SyntaxError/sandbox ps诊断保留；inspect无法确认身份时不据此声称进程停止，不重试。production source未变。真实CUDA恢复、真实watch产物及formal预算仍未测/未批准；preflight仅knownmanifest pins与瞬时容量，raw全量hash/cgroup祖先将在实际获批guard检查，不能把preflight称完整资源验收。


### P4E真实prepare终态与协调验收入口

固定exclusive部署、prepare-watch与独立verify-watch已完成，实际阶段与outer退出通过；shared slices正文留server。small hash-only交付入口cache/p4e-prepare-hash-only-delivery.json，只读验收提案cache/p4e-prepare-readonly-acceptance.py已交Astra先审再fresh执行。当前awaiting coordinator prepare acceptance，不称P4E Complete；GPUpilot/formal未放行。此节覆盖此前运行中/未执行记录。自动审批拒绝完整plan/binding/slice manifest导出，未执行被拒命令，随后仅hash/bytes/exit/budget摘要的安全只读核对获准并完成。历史resource minimum未记录，不虚构。固定历史包保持原字节，own状态文档的操作后更新另记录。


### P4E真实GPU pilot首次attempt STOP

原worker及outer实际非零并自然退出，原root/plan/budget/receipt保留，无继续队列或重试。安全只读根因：worker输出父目录未创建，production resource_guard对output.parent做disk检查抛FileNotFoundError；在训练前失败，没有train-return或checkpoint。CPU synthetic guard未经过此真实资源路径，验证覆盖不足如实记录。只向Astra提出ignored runner parent orchestration最小目录修复；生产source/runner均尚未修改，实际新attempt另gate。诊断证据cache/p4e-pilot-stop-diagnostic-hash-only.json。自动审批拒绝日志正文导出，改路径存在/已知错误分类/异常类型/hash-only调查成功，正文未返回。
# P4E最新operation gate

Watch绑定Important已由原独立reviewerClosed/PASS并获Astra fresh验收。本轮静态部署及derived/plan完成，formal未启动。下一启动gate需明确outer log/receipt在static sibling而dispatcher要求formal root不存在的联合累计预算安排；内部预算不能替代outer绝对起点及全部产物计费。保留现有静态产物，不覆盖、不盲重试。G3既有invoke缺draft要求的PGID/全部descendant inventory，需最小scoped launcher审查后执行，当前只提案。
## P4E detached owner exit evidence gap — awaiting coordinator decision

P4E bounded只读诊断UTC 2026-10-07T22:35:42Z：首slot train actualexit成功，verify原PID/birth匹配且两次短采样CPU/I/O增加，所查train/verify/owner/dispatcher日志无错误。没有verify receipt或最终completion；资源/原deadline未触门。不能据空日志、低GPU或等待状态判卡死，尚未定位具体性能瓶颈，也未证明语义核验循环进度。原verify仅成功返回时输出状态，期间日志不持续刷新。当前无kill/restart/预算干预依据，仍原监控；是否存在对话/automation UI错误未有具体错误文本，不能归为训练失败。

本地源码链dispatch_real_run→real_data/load_watch_artifacts→verify_run：加载并hash核输入与watch，逐record重放cursor/负采样/skip，逐epoch checkpoint hash/load/model→CUDA export_vectors→NumPy全catalog exact ranking/metrics，最后核best selection及last cursor/RNG/stats。没有run_session/optimizer step完整训练重跑，也没有额外训练子worker；resource checks会短暂调用nvidia-smi，source_identity调用git，属辅助子进程。CUDA导出与CPU sampler/ranking并存，GPU瞬时低不等于失败。Pilot短segment未完成完整epoch，不能将其verify函数计时按比例当本次full verifier ETA；既有P4D wall代理source不同也不可作故障阈值。Owner历史actualexit缺口保留，与当前live verify无因果结论。诊断摘要在ignored cache，数值唯一experiment_log。

最新G3准备gate覆盖下方“待scoped review”：Astra及独立Sol已接受固定parent与实际上传/启动链，候选文件冻结；本地准备风险审查关闭，未服务器执行。Parent外部中断限制及P4E原owner实际exit缺口仍保留；真实启动只在P4E自然全终态证据及独立exact-start后，不扩大框架或重复测试。

G3后续最小原parent候选已按协调scope仅本地实现，未改P4E原owner证据；cache/g3-exact-start-proposal.md包含实际固定上传/启动代码，待合并scoped review。原parent birth/handle OSError下仍wait的targeted回归已关闭本地遗弃短子进程问题，实际失败子进程自然结束，无kill。候选未覆盖外部SIGTERM/parent crash后的跨session cleanup；冻结outer负责正常stage失败，异常parent中断需STOP/协调裁决，不能以missing process填actualexit。CPU fixture mock Linux proc/resource仅测试传输和bootstrap逻辑，不称server-ready或实际G3运行。

原once-bootstrap detached Popen只短暂poll，写actual-start后父进程退出，无持久wait。原owner_exit_at_observation null不可改成功；owner-result是dispatcher结果。未来原birth zombie退出码如仍可读则保存真实证据；PID1回收后不可恢复，不以进程消失/观察者exit替代。终态采集器分别核训练/verify/summary/dispatcher与owner限制；后者缺失awaiting_coordinator_decision，由Astra裁决。当前不重启/改运行代码。G3后续父进程须持久wait，但本轮不实施。Collector fixture祖先symlink与输出cap本地失败已修，详见experiment_log；仍待review。


### G3 permit deadline精度误拒绝（待协调恢复裁决）

真实prepare返回成功、permit已生成，但冻结launcher用ISO→timestamp与原float绝对deadline精确比较，微秒序列化丢失亚微秒部分，触发conditional original runtime/budget permit binding。只读检查binding/runtime一致，当前disk谓词通过；历史失败点disk未独立记录，不替代为全历史证明。outer实际失败，pilot未started，parent记录收束且fresh owned groups无活成员。固定launcher未改，旧预算/产物保留，恢复仅提案、不自动retry。证据入口cache/g3-concurrent-admission/actual-v4-terminal-review-v1.json；P4E ACTIVE，仅G3 gate暂停。


### P4E终态owner退出证据缺口保留

一次固定collector实际成功，九run/summary/dispatcher/checkpoint hash及预算核通过，原owned无活成员；detached owner实际wait未记录，null保留待Astra裁决，不补0、不重启取证。入口cache/p4e-final-delivery-v1/review-package.json。此缺口不改原成功run/dispatcher收据，也不冒称coordinator已验收。


### P4E协调正式验收与accepted limitation

最新协调最终裁决：Astra正式接受P4E实验产物Complete；owner_actual_exit永久null，接受原detached owner自身actualwait不可追溯的历史运行记录限制，不称全部编排wait成功。九run/verify/summary实际成功与产物核验已通过，无第三次模型replay。原交付包保持bytes，裁决增量证据cache/p4e-final-delivery-v1/coordinator-acceptance-v1.json。既有p4d已按Astra明确批准经native更新PAUSED并fresh核保存配置；服务器未作生命周期操作。下一步仅P4F只读一页设计，实施/实际执行另待范围与启动gate，test仍关闭。

### P4F local regression legacy identity fixture — awaiting coordinator ruling

全套regression剩余旧P4D fixture source mismatch；fixture复制当前source glob，但历史adapter要求冻结P4D完整身份。此拒绝不允许改旧pin或保护producer使其通过。原P4E source bytes独立hash保持，新增consumer scoped review通过；全套仍记录失败。Sandbox outputs错误在获准fixture writes后消失。具体数字与证据入口见experiment_log及cache/p4f-local-review-v1.json。真实FAISS本地缺失保持明确pending，不install或SSH验证；下一步Astra对旧fixture兼容性裁决及独立server synthetic gate。

### P4F historical fixture问题本地关闭（v2）

Astra授权最小fixture修复后，改为历史精确成员及hash，写入前重建并核冻结fingerprint/实际bytes；既有synthetic ledger override仅用于原fixture，不改变生产adapter、source或真实ledger pins。真实CLI已验证正确producer可达invocation，历史bytes篡改与producer新增module均被拒绝，当前仓未来module不污染fixture。Fresh scoped/full regression及独立diff review通过，关闭本地fixture blocker；v1失败证据保持。真实FAISS/server验证仍pending，由独立gate推进。数字唯一experiment_log，证据cache/p4f-local-review-v2.json。


### G3 supplemental terminal handoff notification blocked

运行端pilot/acceptance与worker actualwait成功，终态证据已保存；新Astra跨对话通知被auto-review拒绝，具体理由见cache/g3-supplemental-scheduler-v1/notification-blocked.json。已读取newmain真实用户协调要求及AGENTS交接规则后仍被拒绝，不绕过/重试。待Eddy明确授权新目的地的脱敏终态与本地证据路径；正式训练与test仍关闭。


P4F consumer deployment Git blocker：本地显式release修复 READY awaiting Astra acceptance，证据 docs/p4f_consumer_release_validation.md 与 cache/p4f-consumer-release-v3/review.json。Closed local review findings:额外bytecode闭包、测试依赖cache/个人绝对路径；实际runtime metadata/批准release SHA及Linux CPU oracle仍pending，不能称真实ANN gate已通过。


## S0全量容量与成本门 BLOCKED
S0只读容量核查已实际完成，但全量下载gate BLOCKED awaiting Astra decision；cache/s0-capacity-readiness-v1/review.md、capacity-ledger.json与receipt/snapshot为证据，数字唯一docs/experiment_log.md。原固定官方parent逐文件metadata/bytes/LFS SHA核通过；当前磁盘扣20%余量后不能容纳全量raw，后续processed/checkpoint/index/temp及完整cost上界未知。仅一次有界只读SSH，无下载/server写入/训练/删除/新增资源/生命周期；不retry、不恢复p4d。下一仅Astra裁决reader/容量pilot或容量方案，S1仍关闭；subset按各自gate继续。
目录盘点为bounded scope，partial根不当总量；quota/持久化与future writer未凭空证明。原P4F/G3验收或预算不因此改写。


### P4F exact-A direct authorization gate
Open：已审exact-A命令被自动审批在CreateProcess前拒绝，非实验失败/transport超时。原因：Astra permit不足以替代用户对具体payload、目的地与remote写入副作用直接授权。证据cache/p4f-runtime-start-blocked-v1/rejection.json；未启动/未重试，无server lifecycle动作。等待Eddy明确该范围。


### G3 actual cost audit direct authorization gate
Open：固定payload/argv/有效permit已本地核，但进程创建前auto-review拒绝具体payload、远程目的地、计费及server root副作用授权。cache/g3-cost-entry-start-blocked-v1/review.json保存证据；NOT_STARTED/no retry，不称实验失败。需Eddy直接授权及协调确认尚有效exact-start。


### G3真实入口 generic worker failure
Open awaiting coordinator：真实单attempt worker非零，reviewed wrapper只保存generic worker_gate_or_execution_failure，failed_phase null；准确根因unknown。cleanup与fresh session空已核，原attempt不retry，不猜runtime/hash/resource具体失败。证据cache/g3-cost-entry-execution-v1/review.json。下一仅协调批准有界read-only诊断或新scope，不自行改实验/追加训练。

P4F已审单次metadata/header/resource只读取证actual transport exit1、STOP bounded_readonly_closed，非timeout；未返回成功scope证据，失败阶段unknown，未retry。cache/p4f-readonly-actual-evidence-v1/review.json为BLOCKED_ON_EVIDENCE；真实input/payload fresh绑定与资源reserve未闭合，预算null，不启动ANN。新local diagnostic候选只加有界enum/line/role错误telemetry，输入/读取/caps不变，待新exact-start；原包/A/B保留。

P4F恢复diagnostic定位metadata/baselines line27长度或hash拒绝；local accepted P4C文档与closing package确认final SHA误绑intermediate文件。新stdin仅路径修正，RED1fail→GREEN2pass；远端唯一根因/修正验证未完成，不retry，真实执行预算null。入口cache/p4f-resume-diagnostic-v1/review.json。

P4F corrected只读actual exit1、metadata/core_run line25 regular/per-file/cumulative cap门失败，原因未细分；不raise cap/retry。header/resource未成功，真实预算null。新packet cache/p4f-corrected-readonly-execution-v1/review.json；stat-only后续仅候选未执行。旧G三job责任转移已确认，真正pilot补件待G明确接收。


### P4F fixed enum diagnostic legacy_manifest mismatch

Open/BLOCKED_ON_EVIDENCE：直接human批准的一次固定enum诊断实际STOP metadata/legacy_manifest/length_or_hash/ValueError。长度或hash分支unknown，保留receipt，不改pins、不retry、不follow-on SSH。原失败与固定包保持；下一Astra合并裁决输入证据缺口，真实ANN gate关闭。证据cache/p4f-enum-diagnostic-execution-v1/review.json；消耗见experiment_log。


### P4F legacy raw/public identity mix-up — LOCAL_FIX_REMOTE_OPEN

已追溯旧runtime template误将public sanitized SHA配到server run_manifest path；path recovery沿用旧pin。已验收P4/P7证据明确两个身份，refs-only新候选TDD最小纠正到历史server SHA，原包保持。既有consumer仍要求public artifact，不能用raw ref替代；未来context必须raw/public分别绑定，0科学source/consumer改动。远端identity/short-read或漂移未证明，新scope尚未准入；不retry旧诊断或零散probe。Astra已local接收原通知拒绝，按wait/local交付不重发。完整trace/候选/合并缺口见cache/p4f-legacy-manifest-mapping-v1/review.json。


### P4F separated entry — LOCAL_CANDIDATE_READY_REMOTE_OPEN

已本地闭合 raw/public 两身份、consumer-relative pinned public 部署候选与 pair/audit guard，metadata 尺寸在 streaming hash 前核验。原错误诊断及失败证据保留；不 repin/retry。完整 single entry packet cache/p4f-separated-entry-v1/review.json 等待 Astra 一次差异验收；human 全 scope/server-write/native-budget 与 fresh exact permit 仍缺。当前远端 identity/resource 与 Linux actual wait/group closure 未证明；本地测试不替代。新 root 部署 failure 可能保留 partial files，全部计预算，未授权删除。transport timeout 仅 unknown、不手动 kill/重新 launch。


### P4F foreground I1/I2 — LOCAL_REPAIR_AWAITING_REVIEW

原identity异常漏wait与resource只记日志问题由真实child反例确认，已定点补原Popen bounded wait/fallback和verified supervisor资源STOP；accepted pipeline独立stage cleanup经本地fixture检查。unknown identity不signal，actual wait超时/STOP请求失败不冒称closure，待原reviewer一次差异核。原v1、v4及source/science保持；新entry仍PENDING/预算未批，不SSH/retry旧诊断。证据cache/p4f-separated-entry-v2/review.json。


### P4F actual full entry generic STOP — OPEN_UNKNOWN_PHASE

唯一已审实际entry返回generic foreground STOP及owner_actual_exit null，原localtransport非零wait/reap完成不能识别服务器preflight/native具体phase或groupclosure。保留original finalstart/permit/allpins/launch/receipt/foregroundhash；不猜根因、repin/改code预算/retry，不手动kill/删除/newresources。partialownedfiles留server计原预算。证据cache/p4f-actual-entry-v1/review.json；等待Astra原receipt bounded读scope，未追加SSH。


### P4F actual STOP — PREFLIGHT_LINEAGE_LOCATED_DETAIL_OPEN

原generic unknownphase已由固定foreground receipt定位为preflight lineage/ValueError。实际wait/reap/历史groupclosed已核，owner null且按固定gate顺序七stage/model/ANN未called。具体lineage条件unknown，header/resource成功和freshglobalclosure不声明。原budget/attempt/partialfiles保持；待Astra本地定位scope或停止，不probe/retry/repin/改实验。


### P4F lineage STOP: diagnostic loss fixed locally, real cause unresolved

The wrapper discarded exact predicate on lineage exceptions. Locally fixed with unchanged scientific validation and bounded output. Public summaries cannot establish full raw shared/checkpoint/selection bindings or ordered audit membership. One merged gap remains; no source mismatch or serializer root cause inferred. Original attempt/root stays spent; candidate awaits Astra review, no retry or remote probes. Evidence cache/p4f-lineage-local-revision-v1/review.json.


### P4F ordered membership serialization mismatch

Precise readonly result confirms failure at legacy membership SHA after producer lineage and basic membership gates. Local source proves producer/frozen consumer use newline canonical, while three helpers omit LF. Minimal audit-only serializer fix is locally verified and awaits independent acceptance. Keep original failures/consumption; corrected real outcome and whole pipeline remain unverified. No repin/retry or additional remote action. Patch cache/p4f-legacy-membership-serialization-v1/review.json.


### P4F second attempt STOP, diagnosis pending

The newly authorized complete attempt returned foreground STOP before any non-null owner exit. Preserve original pins, wait receipt and server result SHA; root cause and fresh server closure unknown. No retry or code adjustment. Coordinator to review fixed hash-bound foreground receipt scope. Evidence cache/p4f-fresh-attempt-actual-v1/review.json SHA 28bf419db97743d829a996c81814dbbb3b7de49d75f7a8bdd0298d4b1aadeadf.


### P4F pair validator producer/public schema mismatch

Second attempt failed preflight with KeyError. Local frozen producer schema reproduces audit_set_hashes missing because the public extension was incorrectly required in raw. Targeted minimal helper revision moves membership lookup to public and retains all shared-field/artifact/lineage pins. Local proof only; exact runtime key not reported, real corrected integration not rerun. Evidence cache/p4f-legacy-pair-schema-revision-v1/review.json SHA ffda1e4a3d91600902a471c82ad1acd37bf3dad133c6e226f35881373f8b5ae3.

### P4F third attempt 与 artifact/science acceptance — READY_FOR_FRESH_REVIEW

第三 attempt 已真实完成七stage并fresh核无active process；前两次失败和第三次spent全部保留。独立只读 acceptance 重核所有固定 input/producer/artifact hashes，逐query重算65个wRRF候选、公平quota和2520-query inference，使用saved exact top50+serialized HNSW重算三档ANN overlap/model delta；结果ACCEPT，efSearch128最小通过0.98门，exact保留。证据 `cache/p4f-artifact-acceptance-v1/review.json` SHA `c227d3ab6c75bf632b5ebde550b51c3397fc75510494f4ce5c3c95e77885d437`；指标只读experiment_log最新P4F节。Fresh review verdict APPROVE，report SHA `92459e7429291d50fb7bb87472c1ea6380fac604b5e255a62cdbbd0e771dd9f3`；P4F closed。无pipeline rerun、remote write、retry/resume或budget reset。P4G必须保留review的六项限制且test继续关闭至共同freeze。

### P4G pre-test freeze — WAITING_SHARED_G_SELECTION

Machine freeze `cache/p4g-pretest-freeze-v1/freeze.json` SHA `c0bf8ddc9cdbee1a5902f364c3a84717f2c4e17662bcb3995eec4146743c3f87`，review SHA `c8fd0ffa76f2a884bddb8e8b7607c41f3fd77731dbeff8f501c51786c4700e8d`。P4D 24/24（16 seed42 search + 8 additional seeds）与P4E 9/9（三消融×三seed）齐全，missing/duplicate=0；冻结best core id_mean/LR.001/T.15/seed42、wRRF `[2,1,1,2]`、exact authority/HNSW128 audit。六条remote paid/operational failure ledger保留spent与UNKNOWN；local TDD失败继续留issue history。

P lane已freeze但test input未读、metrics未算。仅在G3 selection+补seed、G4、G5 joint fusion和single finite test identity全部冻结后允许一次共同pass；禁止根据test回调。当前不新增P实验/search/remote动作。
