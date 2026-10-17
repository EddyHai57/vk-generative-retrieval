密级: 公开

## P4H-03B v5 operational evidence closure（2026-10-11，Ready for fresh review）

最新fresh review的两项HIGH已形成v6 local candidate：19-role nested exact semantics使用disk-bound checkpoint/shard records，CLI opened ledger来自真实typed-input读取，stage events在实际transition时append；mutating/verifier都生成同等级完整evidence map并重放exact schema、identity、time/caps/state/closure。Fresh RED与最终四组tests、冻结hash及边界证据在ignored `cache/p4h-local-operation-review-v6/review.json`；当前只称`READY_FOR_FRESH_REVIEW`，独立review尚未给APPROVE。恶意same-UID进程整体重写全部文件并重算SHA明确为`OUT_OF_SCOPE_REQUIRES_EXTERNAL_TRUST_ROOT`。P4H-04/remote/test/SSH保持关闭。

# Issue Log

## G3实际reservation失败且remote诊断丢失（2026-10-09，Open）

本地真实命令编译通过后，唯一服务器bootstrap在首reservation失败；remote包装器未保留returncode/stderr正文，当前无法定位真实根因。后续只读确认控制和操作目录均不存在，未重试；不能把事后资源通过推断为失败时资源正常。已授权同transport无写入诊断与失败可观测性定点测试，P4E原任务和监控继续。教训：命令compile/mock回执通过不能证明远端解释器和运行环境兼容；失败包装应保留有限且脱敏的真实诊断。

## G3并发启动测试未覆盖实际命令及退出竞争（2026-10-09，代码修复Closed；运行待验证）

修复关闭证据：v2三个实际body分别dedent，wait后fresh scan并重核身份；Astra fresh parent10tests/1.214s、bootstrap4tests/0.158s exit0，后者覆盖真实main全部9条远端source编译；独立review两项Important Closed/PASS。此问题代码层已关闭，真实Linux/GPU运行仍由一次exact-start收集证据，不以本地通过冒充部署完成。

并发候选targeted tests通过后，Astra fresh编译bootstrap.main真实reservation源码复现IndentationError；独立Sol确认三段body均保留顶层缩进。原测试仅编译简化替代命令，因此绿色测试不能证明真实启动链可执行。另supervise在poll/wait后沿用之前进程快照，可能把刚正常退出的outer误判为仍存活。两项Important已交原实现Sol补确定性RED、最小修复和真实命令组装测试；未在服务器启动该候选。修复及fresh closure仍待交付，不能写成已关闭。

## G3等待与P4E汇总调度耦合（2026-10-08，Investigating）

Eddy反馈生成式长期无进展。一次实际诊断确认G3固定服务器目录均不存在；其本地工作已完成，按Astra串行gate等待P4E完全收尾。P4E summary再次顺序执行九run完整数值核验（246epoch），结束前不写summary，也无slot/epoch进度日志。协调端反复确认父进程存活，没有及时说明等待成本与重新评估调度，属于调度与可观测性问题，不是已证实训练崩溃。

证据：P4E worktree cache/p4e-summary-g3-diagnostic-20261008t1112.json；原始输出SHA d833d1fdd9623abcda8b62c7711e6e5ded7d4f280de2f08bbe94a8a352a860ed。actual SSH exit0，summary child身份/CLI匹配，短窗CPU活动存在；不能据此证明语义进度或可靠ETA。

本轮保持原实验，已同步生成式等待状态。后续需评估将独立数值核验与纯聚合拆开、避免重复全量replay并提供阶段进度；这些是待设计优化，尚未实施，不改当前冻结代码或跳过验收，不宣称问题关闭。

## 当前边界

本轮未下载数据、未训练模型、未验证效果。

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
| I-001 | `like` label 覆盖未知。 | Open | 可能无法形成足够 history/target，迫使定义 composite。 | 仅用 train 统计 user/item/event coverage、序列长度与 week 稳定性。 | 冻结 `like` 或 composite policy，并保存 coverage report 与 policy hash。 | 待补。 |
| I-002 | `ir0.01` 的含义与 sampling 机制待查。 | Open | 可能改变 popularity、catalog coverage 或反馈分布，影响 MVP 外推。 | 核对 data card、subset manifest 与下载后分布。 | 明确生成规则，并在 data card 中写清可比性边界。 | 待补。 |
| I-003 | `timespent` 可能是聚合值，时序语义未知。 | Open | 可能把未来或聚合信息泄漏进 target-prefix feature。 | 核对 schema、粒度、单位、生成逻辑与同 user-item-week 重复。 | 证明 query-time 可用；否则从主任务移除，只保留审计。 | 待补。 |
| I-004 | `duration` 的粒度、单位、缺失与版本未知。 | Open | `timespent / duration` 可能失真或除零。 | 做 dtype/range/null/zero 与跨 week 稳定性审计。 | 冻结 clipping、missing/zero policy 并保存 feature manifest。 | 待补。 |
| I-005 | Flagship `ur0.01` 公开树可能没有独立 official test。 | Open | 调参与 final holdout 可能混用。 | 下载前核对公开结构；下载后保存 partition manifest。 | 冻结 train-tail validation 与一次性 final holdout，或采用明确的官方 split。 | 待补。 |
| I-006 | Thunder A6000 实时库存与 snapshot restore 未验证。 | Open | 两天窗口可能因实例不可用或 artifact 丢失失败。 | 执行 ≤1h acceptance probe：stock、SSH、CUDA、FAISS、吞吐、磁盘、reconnect、snapshot restore。 | 全部 gate 通过并保存无 secret 的 probe report；否则切 fallback。 | 待补。 |
| I-007 | Dataset、论文参考实现与未来依赖的 license/provenance 未完成。 | Open | Public repository 可能错误复制或分发受限内容。 | 建立 third-party provenance 清单，逐项记录 source、license、使用方式。 | 所有发布内容 provenance 明确；首次发布前再决定 repository LICENSE。 | 待补。 |
| I-008 | Full-catalog score 与 embedding/index 内存可能超限。 | Open | MVP 或 flagship 可能 OOM，或被迫缩小候选后误称 full-catalog。 | 用 manifest 规模做 storage estimate；运行 blocked top-k 与 ANN smoke。 | 峰值 RAM/VRAM/disk 留 ≥20% 余量，且 evaluator 不物化全 score matrix。 | 待补。 |
| I-009 | Exact evaluator 尚未验证。 | Open | Recall/NDCG、collision correction 或 top-k tie handling 可能错误。 | 构造 tiny toy set，对比 brute-force、chunked exact 与 `IndexFlatIP`。 | 三者在冻结 tie policy 下逐 query 一致，并保存 test artifact hash。 | 待补。 |
| I-010 | 大 catalog trie representation 未定。 | Open | Python nested dict 可能导致 RAM 爆炸与 decode latency 不可控。 | 比较 compact array/CSR/automaton 等 representation 的 build time、size 与 lookup。 | 选定 representation，在 flagship pilot 中满足 memory/latency gate。 | 待补。 |
| I-011 | 两天 MVP 时间风险高。 | Open | 数据清洗或 TIGER 训练可能挤占 evaluator 正确性工作。 | 先完成合同、toy evaluator、Popularity/content baseline，再启动 small TIGER。 | 两天内 P1～P7 固定 audit query 闭环；否则只交付 scaffold，不升级为 TIGER benchmark。 | 待补。 |
| I-012 | Week 与文件内顺序能否稳定构造 target-prefix 尚未验证。 | Open | 可能产生 future leakage 或无法定义唯一 next positive。 | 审计 partition/order keys、稳定排序与重复记录。 | 保存 order proof；无法证明则阻塞主任务。 | 待补。 |
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


## P4E 对话 compact 连接失败与测试语法错误（2026-10-08，Open）

两类问题分开：执行turn报remote compact task连接失败并转idle；最后新增summary测试同时有generator expression缺括号的SyntaxError。Astra只读ast.parse已复现，未修改worker文件。已给原Sol发恢复任务，要求语法修复后fresh确认目标行为RED，不把import失败当有效TDD证据；保持既有worktree，不重复训练。等待worker恢复与验证关闭；此时真实P4E训练尚未启动。


### P4E恢复已确认（2026-10-08）

原Sol已实际确认恢复：测试语法小型patch修复后，targeted测试获得目标行为completion gate missing的有效RED(exit1)，最小verify_completion实现后GREEN(exit0)，继续summary/regression。尚为作者证据，最终Astra fresh验收未做。主brief真实路径为主仓cache/p4e-coordinator-brief.md，已纠正worker误查P4E/6ec7目录；不补造副本、不重复实现。无SSH或真实训练。


## G2 独立 review 暴露恢复与验证缺口（2026-10-08，Open）

作者完整suite通过后，独立Sol仍复现：OOV候选顺序篡改并重算局部hash未被replay逐query比较拦截；context metadata未冻结；validation中断缺训练parentmanifest恢复入口；独立evaluate缺绝对deadline且verify耗时未持久入账；checkpoint缺AdamW moments仍可加载。说明aggregate一致和exit0不能替代恢复/完整性验证。

原生成式Sol继续在三个已授权新模块内逐项fresh RED→GREEN，完成后scoped re-review、完整regression与durable CLI验证，再由Astra fresh验收。未改历史合同或producer pins；G2未Complete，G3真实训练未放行。证据由生成式worktree docs/g2_validation.md和cache日志收口，此记录不宣称修复完成。


### P4E source review修复门（2026-10-08）

独立Sol最小复现且Astra读代码确认prepare-watch/verify-watch绕过operation_guard，仅各建resource_guard：两invocation无persistent deadline/disk ledger，且可写operation_root外。虽然fresh新/legacy tests通过，source gate不通过，已派原Sol补第二invocation超时/累计磁盘/越界拒绝RED及最小修复。CPU verifier对CUDA producer严格排名相等另列未实测风险，要求绑定一致device/precision并在有限pilot实证，不以宽容差替代。真实操作继续关闭，P4E未Complete。


### P4E v2 source gate通过与runner派工（2026-10-08）

Astra fresh31tests、119file/evidence核对及独立Sol scoped re-review PASS关闭预算/输出边界finding；source573c1bdc586f27ead094bf20d497ae16a447264faedbaf421302e230b147a508通过source gate，不等于P4E实验Complete。已派原传统Sol本地ignored operation runner+synthetic跨进程TDD；设计上限批准prepare/replay3600s/4GiB，pilot7200s/16GiB/180optimizersteps（三策略continuous30+split累计30），worstgeometry无optimizer，所有验收耗时纳同预算且20%余量。实际SSH执行仍待精确runner/命令/产物hash审查，尚未放行；formal九组预算待真实pilot。G2 re-review原五项关闭但新journal截断回退漏洞待修复，保持G2 gate未通过。


### G2 review缺口协调关闭（2026-10-08）

原metadata seal、OOV逐query replay、evaluation安全中断、预算计账、AdamW完整状态及journal rollback六项，由独立Sol定点复现后确认修复；Astra再fresh完整222tests与新六步CLI replay通过，详细证据见STATE/handoff及生成式worktree docs/g2_validation.md。本次关闭工程验收门，不代表真实CUDA或模型效果已验收。


### P4E GPU pilot停止门（2026-10-08）

传统Sol报告首worker实际exit2、父runner已退出，未继续后续队列。原receipt/handle/budget/log保留，不自动重启，实际完成optimizer数待证据。日志正文导出被auto-review拒绝，已改服务器内只读分类/路径存在性检查。Astra代码检查提出resource_guard对尚未创建的嵌套output.parent调用disk_usage可能失败的待验证假设，已要求原Sol确认并给最小RED/修复scope；尚未认定根因或改部署。G3本地TDD不受此等待影响。


### P4E目录顺序问题确认与定点修复（2026-10-08）

传统Sol确认首worker在训练前因resource_guard对不存在output.parent做磁盘检查触发FileNotFoundError，synthetic路径此前未覆盖真实资源分支。已批准原Sol仅ignored runner/tests最小TDD修复：父runner在guard后、spawn前创建exclusive策略父目录；production573c与acceptedprepare保持。先保留原runner副本/失败证据，fresh RED/GREEN及回归后交新hash/diff给Astra。未批准二次服务器启动；原失败wall/产物及绝对deadline不抹除，新尝试预算另核。


### P4E执行对话被旧P4D监控提示带偏（2026-10-08）

现象：传统Sol在P4E v2 pilot运行期间，以“P4D已验收、heartbeat PAUSED”结束轮次，未交P4E终态。Astra通过wait_threads核到对话idle，已向同一负责人重发当前P4E接续任务，明确先查原handle/receipt而不重启；对话已接收并开始只读恢复。fresh读取本地automation配置确认p4d状态PAUSED，旧消息进入当前轮次的具体投递原因未证实，不宣称调度器重复触发。服务器训练是否完成以原进程/产物为准，对话结束不等于训练停止。协调初次用系统Python读取tomllib失败，改用既有3.11解释器完成检查，未安装依赖。


### P4E目录顺序故障关闭与监控接续结果（2026-10-08）

目录顺序故障关闭：最小runner修复经真实磁盘路径RED/GREEN、独立review后，v2真实CUDA pilot完成，全部child及outer exit0，三策略连续/分段恢复状态和完整输出一致。Astra fresh只读核原receipt/handle/log/checkpoint/manifest/hash、原deadline和累计disk，completion SHA3fb47d4e0655bba13941d8233e84d4c7330aa2ea375231664fddd5e0d7f50f5a。原失败root和runner完整保留；未把失败attempt当成功或重置预算。复盘：mock资源读数时仍要保留真实filesystem路径检查，否则本地测试容易漏掉目录创建顺序。

监控接续影响已消除：原负责人恢复原句柄观察后，服务器原任务自然结束，无重复训练；旧P4D提示误投递/滞后进入的机制仍未证实，保持该原因未关闭。观测脚本曾把child的process_identity误用于outer；按各自实际schema改为outer birth_ticks，与原handle一致性核对通过。观察脚本失败不等于被观察训练失败。


### P4E正式调度watch绑定review门（2026-10-08）

Astra fresh7目标tests0.531s PASS、固定包13成员hash不变；独立Sol最小metadata复现发现派生receipt未闭合operation→production plan→segment binding.plan.watch，且dispatch不把derived watch与formal runtime比较。不能凭正常tests通过放行；已派原Sol仅ignored dispatcher/test定点RED/GREEN补生产plan外部hash、watch实际hash和跨stage绑定，冻结生产与已验收pilot不改。正式九run仍未启动。


### P4E正式调度watch绑定修复验收（2026-10-08）

Astra fresh scoped9tests0.736s exit0，证据/private/tmp/p4e-astra-formal-watch-tests.log；dispatcher SHA4224bbe2f64fe61b50f596fbe84f743999917bd07bb2887b5b9d88eea49ec281、test SHA6435d1476781a1976128ad3c126b7147d9c4f16149cf355ff916a66bdb736a01及v2包SHA3b568e2332ed86f55adbd45110bc7798af1046c11cd688b20a0d4b844a7e646b fresh匹配。原独立reviewer确认production外部pin、实际watch、全部segment、derived及formal runtime链闭合，Important Closed/PASS。正常路径通过不足以证明来源一致，现已补missing/wrong与重算hash后篡改拒绝证据。生产源码和原GPU pilot未改、未重跑。已向传统Sol派送静态部署/真实derived receipt/实际资源与正式plan冻结任务；48h/192GiB为设计上限，达到cap后20%余量仍需fresh检查，正式九run尚未启动。生成式G3 v3操作包已交付、定点review中，未SSH/GPU。


### P4E outer launcher真实helper接口缺口（2026-10-08，待修复）

Astra读本地候选发现supervise以仅Popen/TimeoutExpired的SimpleNamespace代理冻结helper.subprocess，而真实invoke还读取STDOUT，process_identity fallback还需要run。现有假Helper不走相同路径，scoped green不足以证明真实启动可用。已要求保留原module属性仅代理Popen，并用冻结真实helper的CPU短child做integration RED/GREEN；生产/dispatcher/helper原bytes保持，不运行GPU。另要求在测试finally兜底kill之前，独立观察真实grandchild/进程组无活成员，防止清理代码掩盖失败。当前为本地实现问题，正式训练尚未启动。


P4E outer真实helper接口缺口关闭：真实helper integration fresh RED复现后，最小module代理修复；Astra fresh9tests0.766s通过，独立review PASS，新launcher797ec2固定。真实进程组检查在finally兜底清理前执行，覆盖失败/预算及parent成功但残留子孙。最终server静态验收通过，已放行一次正式启动，尚待真实启动证据。


### G3进度跨对话发送被自动审批拒绝（2026-10-08）

执行对话向主协调send_message_to_thread返回failed，执行者报告auto-review理由为当前可见跨对话授权不足。主对话已有Eddy明确授权，未把本次失败误记为投递成功，也未绕过审批重试。Astra以read_thread读取该实际消息与失败结果并回信确认收到，使用既有只读拉取作为安全替代；生成式继续本地实现，最终在执行对话交固定文件/hash由主协调读取。不要求Eddy重复审批，不影响P4E原队列。底层授权可见性原因未验证，未宣称通知桥梁双向始终可用。


### P4E outer owner退出码留存缺口（2026-10-08，Open）

传统Sol追溯、Astra读取原本地启动脚本及冻结outer实现确认：bootstrap创建后台owner后只等首次dispatcher句柄、记录一次poll=null即退出，没有持续wait并落盘owner自身exit。bootstrap exit0不能冒充owner exit0，owner-result中的exit来自dispatcher。当前不因此认定训练失败，也不重启或修改运行文件。最终提取器必须分别保留九slot train/verify、dispatcher、summary、owner-result与进程终止证据；若无法从原birth匹配的实际内核退出记录取得owner exit，保留null并提交协调裁决，不能补写0。尚未关闭。

预防：G3启动端准备要求由真正创建outer的父进程持久wait并exclusive保存实际退出收据，和operation-result分列；不修改已冻结launcher/release，不因SSH观察结束重复启动。该要求已派送，尚未实施或执行。

后续更新：G3 original-parent与固定上传/bootstrap链已完成本地实现及Astra scoped验收，尚未服务器执行；该预防不补造P4E历史owner退出证据。

### P4E verify耗时与进度可见性（2026-10-08，Open）

观察器schema回归已关闭：固定脚本改读exit并严格拒绝缺失/错误类型，同时修正数值step排序和zombie识别；Astra fresh6tests通过。下一既有heartbeat server utc1791417548.7548192真实snapshot解析成功，脚本hash与验收版本相同，SSH exit0/errors为空，Astra直接核原工具输出及恢复摘要。此关闭仅针对观察器错误；verify性能/内部进度可见性与owner退出凭证为独立限制。

监控schema回归（server utc 1791417275.1901488）：既有heartbeat读取收据时出现KeyError: exit_code，导致本次slots为空；owner/dispatcher原身份仍匹配，资源门未触发。原helper收据字段为exit，之前正常快照也使用exit；Astra已派原Sol仅定点修正只读观察脚本并做最小fixture RED/GREEN，下一既有heartbeat验真。不能把观察器错误当训练失败，也不能用默认0掩盖缺字段；不修改运行中的训练或预算。

后续真实终态：原Sol heartbeat的server utc 1791412739.2491837显示首slot verify实际exit0（receipt SHA ee11afea20eb20fd7d38f95355fd75fcc84abe2545170ee8db6199bdf6ad560b），队列自然进入slot 2。首slot卡死疑虑关闭，无重启或重复训练；核验过程可见性不足及性能瓶颈未定位仍保留，不能用本次成功推断后续全部完成。

用户反馈疑似异常后，唯一Sol完成一次有界只读诊断，Astra直接核工具输出及本地诊断摘要。首slot train实际exit0；verify原身份仍匹配，采样CPU/I/O增加，所查日志无错误，资源门未触发。核验仅在成功末尾输出，空日志无法定位内部进度；辅助资源检查也会消耗CPU/I/O，因此不能仅据活动量宣称语义进展或排除卡死。源码确认verify包含采样重建和各epoch完整检索重算，不是重新训练。小步pilot缺少完整epoch验证，不能据其耗时得出当前ETA。当前未定位性能瓶颈，不重启、不更改冻结代码、不删减既定验证；保留原单负责人监控，终态或异常后再决定必要修复。


### G3 deadline序列化精度导致permit误拒（Open）

真实prepare完成但pilot未启动；冻结launcher把原float deadline序列化为微秒精度ISO，再回转float做严格相等，损失约0.238微秒导致ValueError。Astra使用实际白名单predicate独立复现；context/runtime一致，disk仅有事后通过证据。已派新版本canonical ISO精确身份比较及targeted RED/GREEN，外层绝对deadline和全部其他门保持。修复尚未验收，运行恢复尚未授权；保留失败现场，不重置预算。经验：时间的身份比较应采用一致的精度和编码，原始期限仍用于真实超时。

后续验收：G3 deadline单行修复经Astra fresh7tests及独立review通过，旧文件未改。缺陷本地关闭；部署与运行恢复仍Open，不能称训练恢复。既有唯一heartbeat已实际增加G3终态通知，避免两个执行对话之间只等待而无交付触发。


P4E owner退出记录缺口最终裁决（2026-10-09）：实验产物已验收；原owner wait exit仍无法追溯，保留null并作为accepted limitation，不宣称该事实已修复。独立核真实train/verify/summary成功与进程清理，允许进入P4F规划；不重复模型replay。


### P4F本地FAISS不可用（Open，2026-10-09）

指定p6 Python由Sol确认无faiss；Astra有限检查主仓现有.venv也无faiss，p4e/g0无独立.venv。不安装或全盘找环境。fusion/export/infer及身份恢复测试可继续；真实FAISS测试显式pending，不以mock或skip证明ANN。后续仅在独立有界CPU synthetic gate获准后由唯一scheduler核server现有FAISS并实际测试；未执行、不预称依赖存在或P4F完成。


### P4F历史fixture修复fresh验收（2026-10-09，Closed限定本地）

新Astra核历史22成员与accepted overlay及当前原件完全相符，25个受保护源码及adapter未改；新增consumer不进入历史producer，篡改/额外producer模块仍拒绝。Fresh全套217tests通过，2个真实FAISS测试skip保持Open。首次运行19项PermissionError源于worktree outputs临时测试目录受sandbox限制；经正常权限流程同命令重跑exit0，无源码调整。证据主仓cache/p4f-coordinator/acceptance-v2.json。此关闭不代表真实ANN/融合实验已完成。


### 新协调通知与接续断点（2026-10-09，部分修复）

G3已有终态包，但向新Astra发送脱敏终态被自动审批拒绝；新协调此前没有active heartbeat，两worker最后idle，交付后未及时接续。旧p4d PAUSED符合已关闭server队列，不应直接恢复。主协调已读交付核hash并恢复成本/验收派工；新heartbeat vk ACTIVE每10分钟检查两对话兜底。Eddy最新直接授权两Sol自动交流派工，已让worker核该human消息走正常审批；首次成功回报仍待验证，不宣称审批阻塞已消失。


2026-10-09通讯闭环验证：传统Sol基于Eddy最新human turn 01a11c47-b3a7-7042-ba94-344ead94d5fa正常审批发送READY，已实际到达新协调；此前新目的地通知阻塞本次已解除。G3 coordinator-evidence-v1.json SHA6ac0ffa8addd079fe2aa9b1d900b89ece193ecc356239c99f19ed46e3103f3b4已fresh本地核hash。生成式Sol继续G3成本审查；传统Sol已转P4F有界CPU/FAISS gate本地提案，不执行server。主协调heartbeat ACTIVE兜底，未来审批仍逐动作适用，不保证无条件放行。


### P4F server依赖与部署provenance缺口（2026-10-09，Open）

一次有界只读probe确认固定Python无FAISS，且指定三个Git scopes无可用Git context。收据及hash见handoff最新节点；Astra已fresh核原件，不把worker报告替代证据。此前本地FAISS缺失延伸至指定server环境，真实ANN gate未通过。保留原环境、旧producer/pins，不自动安装、不伪造Git身份；下一步仅由传统Sol提交本地最小解阻设计供审查。版本未测量，资源为瞬时观察，均不补造。


P4F解阻推进：本地设计已核；先查官方CPU wheel metadata与许可证据，旧历史FAISS环境不证明当前numpy兼容。依赖及Git provenance缺口继续Open；源码身份改造、binary下载和安装均未放行。


### G3提取器stdout超时边界（2026-10-09，Closed限定本地）

独立review发现buffered stdout可能在alarm取消后阻塞。作者用真实未排空pipe fresh RED复现，改为alarm内raw os.write，失败不二次输出。Astra已审源码并fresh运行13项tests通过，包括真实blocked-pipe用例；验收见cache/g3-coordinator-existing-evidence-v1/acceptance.json。该关闭不证明server JSON取证已成功或正式成本预算可用。


### G3既有JSON摘要传输被自动审批拒绝（2026-10-09，Open）

自动审批认为五份实验/资源JSON跨边界读取及aggregate传输缺直接具体human授权；协调派工不足。Astra已fresh核scheduler收据：attempts0、未SSH、exit null、无aggregate。不是代码失败，也不推断server停止。保留阻塞并向Eddy说明具体数据范围与上限，请直接授权后正常审批；不得绕过或重复提交。证据见handoff最新节点。


G3只读摘要审批后续（2026-10-09，Closed限定本次操作）：Eddy在scheduler对话直接批准后正常审批执行成功，一次实际attempt、两交付hash及五输入pins经Astra核验。旧拒绝保留，不保证未来动作自动放行。成本/正式训练gate独立保留，下一由生成式Sol完成成本分析v2。


### P4F候选wheel native来源及许可覆盖待证（2026-10-09，Open）

精确候选wheel已下载并只读核hash/目录，唯一license文本是publisher MIT；native runtime与可能bundled OpenBLAS的精确构建/许可链未完整核实。不能以publisher许可覆盖所有组件，也不能仅凭缺包内NOTICE断言本项目内部使用不合法。下一限定primary构建/许可来源补证；未过门不安装。证据入口及派工见handoff最新节点。


P4F许可门收口：协调依据官方GPL内部运行权限与已审upstream许可，在仅内部使用且不再分发范围接受1.15.1候选；组件构建对应限制保留，不把缺attestation/NOTICE等同禁止运行。许可证阻塞限定关闭，技术platform/安装/真实FAISS验证继续Open，裁决和primary来源见decision_log最新节点。


### G3成本审计硬磁盘上限未有存储证据（2026-10-09，Open）

现有write接口无总allocated quota，事后stat与单文件RLIMIT不能证明聚合硬cap。配额是否可用unknown，不据此声称server没有quota。协调允许本地conditional候选与synthetic测试继续，真实部署保持disabled；不得把模拟proof或预算reservation当filesystem强制限额。详见handoff最新接口gate节点。


### P4F安装preflight缺少失败子项诊断（2026-10-09，Open）

写入前多个无消息assert导致仅AssertionError空消息，actualexit1且无firstwrite/stage；不能确定missing pip或root已存在。Astra核执行器后安排具名布尔只读诊断和synthetic验证，先让P1完成生成式storage安全节点，不重复安装。无源码/固定环境改动，证据见handoff。


P4F preflight诊断收口：一次具名只读probe证实fixed Python缺pip，root/timeout正常。定位完成，不修改固定环境或重试旧安装。下一仅本地精确flat-wheel部署候选，缺pip不再以unknown表述；依赖真实加载尚未通过。

## 2026-10-09｜EDU-01 通知与离线预览边界

教材作者向主协调发送READY被自动审批拒绝，未被认定拥有该具体接收对话的可信授权。未重试或绕过；主协调通过wait_threads与本地文件接收交付并完成独立验收。文档闭环已完成，消息发送本身不伪称成功。浏览器file URL策略另拒绝像素预览；保留该限制，以本地HTML结构/完整正文/资源与链接、ZIP逐成员核查完成文档范围验收，未声称视觉或打印分页验收。

P4F依赖缺失门已由一次实际隔离部署及CPU capability收据关闭；固定环境metadata前后保持，真实ANN质量与consumer无Git身份仍是后续工作，不扩大本次关闭范围。状态及证据见STATE/handoff最新记录。


P4F v3本地验收及S0容量阻塞（2026-10-09）：fresh独立review无blocking，结合已核hash与44项RSS-mock行为验证，接受consumer release v3本地范围；原ps权限失败与native skip保留，非Linux验收。P1已接同P4F新cache/p4f-runtime-gate-v1仅本地runtime观察/干净source运输/CPU synthetic exact方案准备，禁止SSH/改旧pins/重复安装，G3实际执行优先。S0已fresh核7成员hash、实际receipt、df/20%算术、30唯一官方inventory文件size=LFSsize与sum；完整raw落地容量不足，S1关闭，具体数值唯一传统worktree docs/experiment_log.md及capacity-ledger.json。展开产物及费用仍unknown，不能凭raw缺口确定最终扩容量；不删旧成果/新增资源。S0主动通知被自动审批拒绝（接收目的地元数据授权未确认），协调通过wait/local获得证据，不重试绕过；盘点已交付不因此重跑。


G3 runtime信息传递阻塞（2026-10-09）：wait快照显示生成式runtime请求遭自动审批拒绝，理由含GPU UUID/环境值/Git状态可能敏感；不重发原内容、不转抄绕过。已指示原Sol仅提供缺失字段名称/核验规则/脱敏既有证据路径hash及最小probe需求，实际值留server由唯一scheduler就地核对，跨对话只pass/fail及摘要。收窄仍拒则保留待正常审批，不伪造runtime；本地entry与传统runtime集成均active继续。


2026-10-09｜P4F actual A 自动审批阻塞：主协调已核固定包和 exact-A permit，但 transport 在进程创建前被拒。原因是审批未将协调器 permit 视为用户对固定 source/parent 上传到既有服务器及新 root/release 写入的直接授权。传统 cache/p4f-runtime-start-blocked-v1/rejection.json 记录 actual_transport_started=false、actual_transport_exit=null、execution_output_root_created=false、retried=false；此前只读 preflight exit0 不能代表 A 已执行。未重试/绕过，B关闭；G3原有有界只读runtime核验已另行接续给唯一scheduler，不受此写入拒绝扩大阻塞。


2026-10-09｜G3 actual-start 自动审批阻塞：固定permit/payload/argv本地核通过，但transport在进程创建前被拒，无server写入、无实际exit、未重试。传统cache/g3-cost-entry-start-blocked-v1/review.json及rejection.json经协调hash核验；审批认为可信用户消息未具体覆盖内部代码/归档/输入元数据payload、远程目的地及root写入/运行副作用。90min预算已有human批准，不将其误记为预算未批；待具体执行授权，permit过期不自动重签。P4F既有拒绝未重试。


G3授权接续：已直接核见scheduler对话human消息01a11dbc-d445-7a41-850d-dd8e64e017fb，具体执行授权已到。v1在收到时已过期且未执行，协调仅更新v2有效期并重新派发正常审批；不增加attempt/预算。scheduler协调通知再次被拒的事实保留，主协调以已授权wait/read/local取得，不重试通知、不要求Eddy重复授权出站。尚无实际启动证据。


2026-10-09｜G3最新真实执行：传统cache/g3-cost-entry-execution-v1的parent-summary/transport-receipt已由协调读取，transport和worker均exit1，actual_cost_measurement=false、actual_real_preflight_verified=false，parent报告owned_cleanup_verified=true。具体根因仅generic worker_gate_or_execution_failure，尚未知；不是成本gate通过。scheduler仍做终态取证，生成式Sol已接同G3本地诊断，不SSH、不重启、不新增attempt；冻结原包保持。正式训练未启动。


P4F恢复派工再次被自动审批拒绝：Eddy明确要求传统继续并行后，主协调发送固定A执行brief被send_message_to_thread直接拦截，未送达/未执行/未重试。审批理由为继续任务未明确覆盖具体source/parent/control上传目的地和远端创建目录/运行副作用，服务器可信目的地未核实。保留阻塞；状态记录不作为间接执行指令。


P4F派工授权阻塞接续：Eddy已明确可信目的地、具体固定包、专用目录/release写入及120秒单次CPU A gate授权；send_message_to_thread本次确认成功，旧拒绝证据保留。仅派工送达，不代表实际启动或gate通过。


G3 v2独立review完成：1个blocking（owner异常fallback未清parent SID descendants却可能cleanup=true）；已合并派原Sol定点TDD修复，不新开review链。服务器只读packet1ce8f47d9884432c46d08ff16ae699ff11108a2f5a2153c1d8df9e82299ec06a成员已核，当前parser/pre_backend均torch_version_fields且cuda为AnnAssign；不推定历史唯一根因。P4F B packet50cfc6b7d0690f5fb34a02653b69c081711d729b7939557a592184e8078d0131成员已核，actual READY但独立oracle/持久收据核验因自动审批超时pre-execution拒绝，未重试，不能完整验收。原Sol已派同P4F下一真实fusion/ANN本地候选及最小只读核验提案，禁止绕过/重跑B。


P4F B只读核验超时的原工具措辞已补齐：明确允许一次retry，非实质风险拒绝。主已读固定脚本197077543fc021ad032a00f4fe66d35a42c1d426b46596eb4d8551fc5569e617并派原scheduler正常审批下一次有界只读执行（30s/60s/8KiB），仅一次，不重跑B，再次拒绝即停；撤回此前基于信息不足的绝对no-retry，不绕过审批。真实fusion/ANN仍缺input/audit绑定和新预算，不启动。G3修复及Linux probe完整包仍active，未重复派工。

2026-10-10｜G3成本候选v2独立review Important（OPEN）：新增verify_receipt只核非空phase与非负有限数，未约束必需阶段、时间一致性及IO完整/unknown。fresh反例将phase换成无关阶段且耗时远超elapsed、缺失IO仍返回component accepted。相关定点tests通过不代表此gate正确；主cache/g3-cost-candidate-coordinator-v1/review.json记录一次独立review及主核packet身份。已合并派原G Sol仅在新v3修consumer/tests和owner文案，原执行组件冻结；之后同一reviewer只核差异，实际新attempt仍未准入。

2026-10-10｜上述G3 Important CLOSED（本地范围）：v3 packet 5ba31c7ca7c98cf4165a141e540bb60b3a2e1b2d792753a92e2dee73e0fb3924经同一reviewer定点复核，原反例拒绝、固定phase/时间/IO门与冻结输出一致，fresh13tests通过，11保护成员不变；主fresh核18成员。证据cache/g3-cost-candidate-coordinator-v1/acceptance-v3.json。仅接受候选；真实成本/新attempt/正式训练未因此通过。

2026-10-10｜P4F metadata路径映射缺陷已本地修复：P cache/p4f-resume-diagnostic-v1/review.json及receipt经主fresh核，实际diagnostic在baselines metadata长度或SHA检查失败，非timeout；已接受P4C明确要求final manifest，原refs却把final SHA配intermediate路径。修正只改路径、全部pins不变，主fresh2tests通过；远端唯一根因和修正后绑定尚待一次有界核验。通知被auto-review拒绝（认为缺目的线程消息授权），不重发，主通过既定wait/local取得完整包。

2026-10-10｜G3 v3启动前runtime_origin STOP（OPEN）：G cache/g3-cost-actual-v3/review.json SHA c3b1908a6f94d179876c54a49b5f0651fbb62201846f58616c1ece20ccee0594主核7成员；第1次readonly actual exit1、非timeout，ValueError子条件未区分，资源门未执行。旧receipt/现场旧身份groups absent/新root absent已核，但不能据此启动。未submit或firstwrite，不称新真实attempt失败消耗。已派原G owner先本地对照wrapper与冻结entry，再在剩余1次原读取包络内细分诊断；不repin/盲retry，真实漂移或需扩大范围仍STOP。

2026-10-10｜该STOP已定位GPU UUID不一致（仍OPEN）：第二次readonly receipt与8成员由主核验，唯GPU UUID比较失败，origin/spec/contract pins与GPU name等先行门通过；资源未核、实际未submit。Eddy明确未更换/迁移实例或连接目标。原wrapper origin绑定漏项已局部修正但不是该ValueError根因，禁止把修wrapper称为身份漂移关闭。G仅获本地来源链/私有目标配置对照及有界新核验候选准备；两次readonly额度用完，不第三次盲试或改旧UUID。

2026-10-10｜G身份调查发现API表示口径不同：旧origin由PyTorch props.uuid字符串产生，新检查直接比较nvidia-smi UUID；不能由raw unequal认定硬件改变，也尚无实际canonical equality证据。调查packet0b27daeb828045a11e8e40fb23f0bab5ead3344925c5641a0b6d37de0fd38e98主核8成员/3synthetic tests通过。新增一次固定旧origin/UUID只读诊断派工被auto-review拒绝，认为新增scope必须有human明确批准，未送达/执行，未重试。具体批准问题已提交Eddy；不通过状态文件/不同工具绕过，传统本地任务继续。


## 2026-10-10 — G3 v4 preflight权限阻塞

原auto-review于创建进程前拒绝：未充分核验preflight-input外发远程A6000，可信直接human授权仅覆盖此前UUID诊断，未明确覆盖新payload与目的地，敏感外发范围/授权不足。G未retry或换写法；本地分类核验已由owner存证，未SSH/submit/消耗preflight额度。证据G cache/g3-cost-actual-v4-start-blocked-v1/review.json。主协调已请求Eddy明确一次固定script/input至既有可信目的地的授权，待答复；未将协调准入当作绕过依据。


## 2026-10-10 — G3跨对话授权传递被自动审批拒绝

Eddy已在主对话直接批准固定v4 preflight。G侧自动审批仍拒绝assistant转述；G读取原human question/answer作为证据后，相同命令再次被拒绝，原因为“所谓直接批准仅出现在不可信的read_thread工具输出中，而非本次可信用户消息，因此无法验证此次具体payload与目的地的授权。”全部创建进程前拒绝，无SSH或额度消耗。停止继续审批尝试，不绕过；请Eddy在原G对话直接授权。此为授权通道阻塞，非server故障或用户未批准。证据入口见handoff。


## 2026-10-12 P4H Sol03准入错误 / FD visibility gate

两次step0前bootstrap因FD access mode unavailable安全拒绝，owned tree CLOSED；0 admission/scientific worker/optimizer。Sol03曾未核真实cause就归因startup ancestry，且bootstrap reconciliation明确失败后仍启动第二shim；该准入纪律错误保留，不退款attempt。保守计满原2attempt，P STOP，formal/pilot科学结果均未产生。当前本地跨UID只读helper63/63与Linux UID1001 metadata probe通过，但新publisher完整验收未做；需明确额外attempt授权才可继续。原clock/cap/spent不reset，G未signal/改job。证据Main cache/sol03-reception-v1/sol04-handoff-evidence.json及handoff。按context_session_policy质量异常优先冷启动Sol04，Sol03停止执行。

06:13 Sol04实际thread `01a12d05-80be-7ff2-89e1-c8ca8497b172` 已本地接收ACK，原2/2 spent与P STOP不变。核三仓HEAD与候选SHA一致、原63项日志存在；未重跑或新增remote attempt。PID2836本地ps被sandbox拒绝，未重试，live identity UNKNOWN；接收不关闭FD publisher完整验收或previous-terminal authority阻塞。证据cache/sol04-reception-v1/receipt.json。

09:30原pilot绝对deadline已过，2/2spent与原queue保留；旧单次追加请求过期，P继续STOP。仅核日历时间，无SSH/新attempt/远端closure或资源核验，不自动延时或新queue。证据cache/sol04-reception-v1/pilot-deadline-elapsed.json；需要新的明确有限授权才可重新评估准入。


## PG-R01-P-RECOVER｜主05部署授权阻塞（2026-10-12）

自动审批在CreateProcess前拒绝持久远程部署：认为代理交接的human grant记录不足以证明Eddy对精确部署/新增计费窗口的直接授权。没有remote write、extension03、4h clock或第3attempt消费；未绕过。用户批准问题已展示，下一步等直接回复后重新fresh准入。具体包、review修复和验证证据见Main cache/pgr01-p-recovery-v1/main05/review-and-fixpass.md及fixpass-acceptance.json；运行身份不因此伪称已验收。Formal完整成本缺口仍阻止正式训练。


## PG-R01-P-RECOVER｜Linux fake-kernel timing gate失败（2026-10-13）

部署经Eddy直接批准后成功；有限clock00:00:55–04:00:55。Linux相关69项中KernelLeaseTests.test_escalation_closes_only_owned_tree_and_actual_waits失败：fake-kernel40ms清理窗口含真实fsync，慢写耗尽预算，返回FAILED_UNSAFE；本地注入56ms写入复现。仅fixture改可控clock后本地69 PASS，生产runtime不变；真实publisher proof/prepare/科学worker/verifier均未运行、attempt3未reserve。已按STOP保留部署/失败验收成本，窗口与旧spent不重置。Main ignored main05/linux-gate-stop.json保存证据，新独立fixture包已准备但未remote执行；等待Eddy允许原窗口内一次失败验收例外。vk-02已PAUSED等待human action。


## PG-R01-P-RECOVER｜pre-worker launcher ancestry拒绝（2026-10-13）

fixture例外经直接授权，真实Linux69 PASS；首次publisher proof在observe_g前置门拒绝unknown P writer，无publisher/科学spawn。原因是新增tagged launcher父进程未纳入child-only ancestry；只读诊断count1/gpu[]，fresh绑定创建者父身份与exec去掉中间launcher分别count0，原validator未放宽。修复为same-PID exec原proof/driver、同owner actual prepare，已本地准备未执行；保持原deadline04:00:55/20GiB/spent，等待pre-worker例外授权。具体Main cache/pgr01-p-recovery-v1/main05/ancestry-gate-stop.json。


## PG-R01-P-RECOVER｜live feed probe时钟与有限clock authority（2026-10-13）

preworker01保存第一原因feed time schema，仍0admission/worker、controller/publisher CLOSED。原reader取now后耗时FD probe，期间新receipt被误判future；fresh RED→实际monotonic elapsed推进观测时点，读后再次expiry检查→71 PASS。原04:00:55 remaining/full closure reserve不足，preworker02第一次deploy在mkdir/write前拒绝，未降低门槛。Eddy直接批准仅+7200s effective06:00:55，原starts/queue/spent/登记/attempt3/20GiB不改。显式clock新协议一次review发现真实authority读取晚于science input；真实sentinel RED→controller/worker在read_roles前校验→73 PASS。源与actual clock/deploy证据main05/clock-final-acceptance.json。No scientific release；newsource proof/prepare/单次pilot与verify待实际收据。


## PG-R01-P-RECOVER｜actual admission/worker birth后的UID visibility STOP（2026-10-13）

Source986a15新Linux73/full proof/prepare成功后，preworker02实际写admission与worker-start；driver原始失败为readonly UID observer failed; zero visibility bypass。按科学边界不重试STOP，实际controller/worker owned sessions与publisher均CLOSED，fresh owned/GPU列表为空。无terminal/checkpoint/result，release/optimizersteps UNKNOWN，不记0或refund。只读诊断96 UID checks均通过，未复现，动态根因UNKNOWN；先补失败helper PID/birth/returncode/stderr证据，不绕过guard。事实与完整SHA Main cache/pgr01-p-recovery-v1/main05/recovery-stop-summary.json/final-science-stop.stdout；旧queue/所有登记/spent保留，clock effective06:00:55不是重试许可；vk-02 PAUSED等待可核修复/新的有限科学授权。


## PG-R01-P-RECOVER｜UID退出竞态可复现修复（2026-10-13）

有限Linux短命UID进程复现原readonly observer错误：退出期间/proc/fd PermissionError。历史作业没有目标PID/rc/stderr，不能据此宣称历史根因已证明。候选pidfd固定原进程、kernel exit后复核当前PID exact identity；一次独立review发现旧pidfd退出会吞PIDreuse drift，修复为专类异常永远STOP。Local/Linux80 PASS，16真实退出PASS，2 live不可见/writable FD负例均STOP，单review APPROVE（独立7项通过，全suite独立受ps EPERM限制）。生产未部署、没有新science attempt，旧queue/spent/登记完整保留且fresh failed owned无进程。证据main05/uid-race-fix-acceptance.json及uid-final-snapshot-v2.stdout。下一步仅等待原deadline06:00:55/20GiB内累计attempt4单次例外；proposal不是授权，vk-02保持PAUSED。

## PG-R01-P-RECOVER｜attempt04授权与launcher接续（2026-10-13）

用户直接要求解除旧限制持续执行，新finite包保留原charged tree和旧spent。新authority一次review发现两个执行阻断：launcher强取旧clock_extension、driver固定attempt3；fresh RED复现KeyError/identity mismatch，专属attempt04 driver改queue.attempts和recovery04 identity、launcher移除旧clock字段后8项GREEN，review APPROVE（独立authority4/driver8 PASS）。Runtime84 local/realLinux及完整publisher/owned-tree proof已PASS/CLOSED，实际prepare运行。旧driver bytes保留，源f2d2455a…/driverc31f777a…；实际queue与首写时间见main05/attempt04-deploy.stdout。

## PG-R01-P-RECOVER｜实际live fd不可见与有限重验（2026-10-13）

Attempt04 Linux84/full proof/new prepare185.264s通过后，实际admission/worker birth再次STOP，PID2692037/birth164258482/rc1/stderrhash2183c3…与已知fd PermissionError traceback替换PID后SHA完全相同；目标进程具体类型仍UNKNOWN。Actual driver/controller/publisher CLOSED与fresh owned为空，release/steps仍UNKNOWN，spent保留。24次并发closed-feed inventory未复现helper失败，不能作为放宽门的理由。

候选仅对精确fd PermissionError增加一次50ms后完整只读重验；同pidfd，sleep前后与成功后首次UID/GID/birth必须保持，writer/持续不可见/未知helper失败/已发现drift仍STOP。一次delta review发现跨retry原凭据pin丢失与helper drift分类，fresh RED/GREEN修复pass后APPROVE、独立5项通过。Local/Linux89 PASS；16真实退出PASS、2真实dumpability瞬态各首次helper失败后完整证据PASS、2 live反例STOP。Source3cf82442…；实际attempt05部署绑定固定failed4/closure4/queue4及完整旧链，first-write03:43:41至07:43:41/4h/same old+new20GiB。完整新Linux89/publisher proof PASS/CLOSED，新prepare进行；无新资源/formal，用户自主继续授权有效。证据main05/attempt04-terminal.stdout、attempt05-predeploy-acceptance.json、attempt05-linux-candidate.stdout、attempt05-deploy.stdout和attempt05-proof.stdout。

## PG-R01-P-RECOVER｜监督连接导致的实际UID拒绝与publication/cleanup竞态（2026-10-13）

Instrumented attempt06实际捕获目标UID102/GID65534、helper-command=false；账户只读确认为sshd。0step live causality112 checks，在fresh SSH readonly dummy认证时精确复现同UID102 fd PermissionError；同dummy复用已认证ControlMaster+stdin未引入UID102。旧-c status源码还携带vkgr-p4h argv标签，可能被P writer heuristic视为未知writer。因此active job所有收据改用main05/transport_mux.py --readonly，writer/proof/prepare/pilot用同mux；owner私有socket、先-Ocheck、ProxyCommand=false禁止fresh-auth fallback，不改server SSH/cap/资源，不跳过FD检查。

同对照与CPU probe复现receipt immutable drift：reader提前缓存HEAD之外正在fsync/chmod/tmp unlink的下一receipt。修复为只cache committed prefix，单FD bounded mutable HEAD snapshot支持atomic replace；committed mode/hash/chain/fork/futuregap/stale/tamper仍STOP。GPU diag及actual6 pub cleanup复现短命owned pidfd ESRCH：已确认退出的PID不返回signal handle，live PIDreuse/unconfirmed exit仍STOP。Fresh RED/GREEN local/Linux101 PASS，新完整delta单review APPROVE（独立新增9，全suite因ps sandbox限制由Main realLinux补齐）。

Actual6原pub driver仍FAILED_UNSAFE/waitNone，原bytes/spent不改。之后真实cooperative pub CLOSED/lockfdclosed和fresh旧SID/GPU absence已追加late证据18cce7…，明确UNKNOWN_NOT_RECOVERABLE parentwait；不伪造原CLOSED/wait0。新phase7固定old6/late evidence并重新核absence，fresh finite root，不resume旧6。source9a59da…/driverc31f777a…，actual stage7 deployment/Linus101/full publisher proof已PASS/CLOSED、新prepare运行；physics/data/label/test不变，无formal。证据main05/ssh-observer-causality-summary.json、late-publisher-quiescence.stdout、attempt07-linux-candidate.stdout、attempt07-deploy.stdout、attempt07-proof.stdout。

## PG-R01-P-RECOVER｜pilot07完成后artifact digest约定拒绝（2026-10-13）

Actual pilot07两臂各30steps/train-only matched2520，worker actualwait0/CLOSED，旧driver/pub收尾无异常；独立verifier尚未启动。Producer complete digest用compact canonical JSON，frozen plan要求同JSON末尾newline，旧seal digest与legacy实算完全相等。固定原seal53f83b…、terminal548910…、source9a59、queuec731不改，不重训或refund。

Verifier-only修复只追加derived wrapper和凭证，固定original seal且只改complete digest；重新核原source inventory、legacy artifact index、原queue和全部cross-binding，replay继续绑定原producer/source/input/tensor。Future producer两处digest改用frozen plan canonical_bytes。Fresh RED后local及actual Linux102＋8 PASS、完整独立review APPROVE、新full publisher proof actualwait0/CLOSED；source d7fc05ae…已部署至exclusive verify-repair01，独立verifier已start/admit（manifest db7f25e8…，worker2699437/birth165234831）。原09:06:40/20GiB和spent不变，无重训。证据main05/attempt07-digest-mismatch-summary.json、verify-repair01-review.md、verify-repair01-stage-acceptance.json、verify-repair01-first-status.stdout。实际verify终态/CLOSED尚未产生，不能称verified或准入formal；server driver负责真实wait与收尾，模型不持续空转等待。

## PG-R01-P-RECOVER｜verifier07认证mux失联、remote job UNKNOWN（2026-10-13）

06:30 heartbeat只读查询在本地socket stat失败，原私有socket已消失；原长transport348.607s后rc255，stdout/stderr均空，本地transport/master已无。未获得远端新终态，不据此称科学失败、停止或CLOSED，steps UNKNOWN，断连根因UNKNOWN。当前finite09:06:40前禁止新认证扰动潜在live科学门，原queue/spent/产物不动、无重复submit。证据main05/verify-repair01-transport-loss.json和verify-repair01-verifier.transport.json。

已准备但未执行logged recover02连接：deadline guard拒绝提前认证，私有-E日志便于下一次诊断，mux仍先-Ocheck/ProxyCommand=false/no fresh fallback/readonly stdin。Same heartbeat到期后只对账原job终态/actual identities/fresh absence；finite expiry不能代替真实CLOSED。未暂停heartbeat或请求普通步骤确认，无新server资源/生命周期操作。

到期后真实对账已解除科学UNKNOWN：verifier terminal6f5a5ff3… completed+verified，drivercd17fdb9… pilot_completed+verified+CLOSED、cleanup_failures空；worker/controller/pub actualwait0/CLOSED和fresh全部owned absence，验证0optimizer/两臂fresh tensor replay。服务器实际上已于06:17收尾，原loss为transport退出而非科学失败；不改变原loss/failed+spent bytes。证据main05/pilot07-final-acceptance.json、verify-repair01-accept-cost03.stdout。

Recover02后台mux再次消失且私有-E日志空，根因仍UNKNOWN。当前单一foreground transport owner实际wait/finite2h/session24969，state recover03，先-Ocheck/ProxyCommand=false/no fresh fallback/readonly stdin；只替换连接持有方式，不新增server monitor/resource或重跑接受项。CPU cost-completion01新delta已review APPROVE、6tests PASS且真实start：first-write10:25:30.453→12:25:30.453/UTCmono2h，同原树20GiB、0科学attempt/optimizer/HPO/test/selection，独立actual owned wait/CLOSED。新sourcea3350863…、registrationc0740d00…、controller2700868/worker2700936，fresh birth匹配；完整validation prepare/deploy和synthetic-rank metrics序列化仅成本，不是质量。证据main05/cost-completion01-stage-acceptance.json；终态未接收，formal仍cost BLOCKED。

Cost prerequisite已真实解除：terminale6295fe4…实际worker exit0/CLOSED，parent/worker fresh absence；source/registration/worker/producer/artifact hash与full validation2520/sample86785/catalog120415全部绑定，0科学attempt/optimizer/test。完整保守forecast已满足原14h/32GiB，但formal permit仍false，下一新登记P4H-05-S02-S06只做完整新队列/输入/资源身份准入。最终PG-R01-P-RECOVER接收main05/p-recovery-final-acceptance.json、cost-completion01-final-accept.stdout；旧失败/UNKNOWN/queue/spent/产物bytes不改，不重跑pilot/verify/cost。

### P4H-05-S02-S06｜正式编排完整review与实际启动（2026-10-13）

新formal driver/deploy/launch/foreground transport完整delta单次独立review APPROVE；固定S02→verify/CLOSED→S06，typed role aliases/source/producer/control/slot绑定、单queue、不reset、实际wait/closure/fresh资源门未放宽。新local/Linux5checks PASS，38 source与4个integration seam字节保持d7fc05ae…；不重跑旧110或旧科学replay。原Mean controls已固定原manifest/checkpoint/2520分母，observed采用明确inline canonical hash约定，非伪称旧独立文件。

实际exclusive formal root已freeze12:19:32+08＋14h/32GiB，queue11f91e6b…/specbe135e4f…/driver0475a486…。Foreground04 finite16h/mux no fallback覆盖全queue；driver2701693/birth167437191、CPU prepare worker2701779/167437946 fresh身份匹配，唯一server driver负责受监督prepare→各exact科学permit→训练/独立verify/actualCLOSED。当前尚无科学worker-start/质量终态，不能称训练完成；旧pilot/cost/spent保留。新证据main05/formal-pair01-stage-acceptance.json、formal-pair01-first-status.stdout、formal-pair01-review.md。下一same heartbeat仅mux_formal04 --readonly轻量新收据，无重复submit/模型持续等待。

### P4H-05-S02-S06｜S02 formal worker因/proc竞态失败（2026-10-13，OPEN）
现象：S02 formal-train训练按早停规则结束（4个epoch metrics，未读质量数字）后，fresh-best evaluate内gate.check→prove_publisher_runtime→linux_writer_inventory→linux_processes读取`/proc/<pid>/stat`时目标进程已退出，抛`ProcessLookupError`(Errno3 ESRCH)；该函数只捕获`FileNotFoundError`。Worker actualexit1，terminal failed+spent（181714445f…，wall1152.63s/128714069bytes，optimizer_steps UNKNOWN），closure CLOSED/signals0；driver随后failed+spent结束，S06未启动，publisher CLOSED，live_owned空。
判断：TOCTOU竞态（listdir后进程退出），不是科学/数据/资源问题；与Kiro/Codex只读查询时间不重叠（最后查询12:30:39与13:03:16）。
修复方向（未实施）：linux_processes对ESRCH/ENOENT视为进程已消失并跳过；其他异常与unknown-writer/FD/UID门不变。需fresh RED→GREEN→regression＋一次独立review，再新唯一有限登记。原attempt/spent/产物保留。证据：main05/kiro-s02-failure-01.*，cache/session-handoff-20261013-kiro-opus55-v1/kiro-acceptance.json。

## 2026-10-13 SSH master断开原因与G select超时（Kiro）
- SSH：formal04 `Connection reset by peer`、formal05 `Connection closed by remote host`；服务器sshd `ClientAliveInterval 10/CountMax 2`（≈30s无响应即断），alias为直连IP。formal06存活>60min。判断：本地网络/休眠中断触发，不是固定会话时长。缓解：P运行期间Mac不休眠不断网；断开后P收口前不重连。状态：OPEN（观察中）。
- G select：第1次缺GIT_DIR/GIT_WORK_TREE（read_verified_v2的_git身份）失败；第2次25min内部timeout被杀（实际需1514s）。均未写输出。修复：env对齐lifecycle.worker_environment、timeout 10800、改服务器detached运行（select_detach_remote.py）。CLOSED：17:19 SELECTION_FROZEN_WINNER3。
- 更新 18:50：formal06 于≈17:53断开（actual_master_exit255，elapsed5436s，`closed by remote host`）。Mac pmset：17:52:43 `Entering Sleep state due to 'Idle Sleep'` → 根因确认为本地Mac闲置休眠，不是Vast会话时长。最后观察17:23（P prepare、G train均正常）；之后两线状态UNKNOWN。P收口前不新建认证；重连时间待Eddy选（22:10推荐 / 02:20零风险）。后续连接期间须`caffeinate -dims`。
- 更新 19:21：P recovery01 已 `S02_recovered+verified_S06_completed+verified+CLOSED`（main05/p-recovery01-final-01.stdout）→ 上方“S02 /proc ESRCH竞态”条目以本次恢复验收关闭（CLOSED）；断线期间作业未受影响。

## 2026-10-13 P4H SASRec-style 近乎零召回（OPEN，待诊断）
现象：S02 R@50=0.0004、S06 R@50=0.0016（experiment_log P4H-05 recovery01节），均best epoch1后早停。远低于Pop/双塔，可疑。候选原因（未验证）：学习率/温度对全catalog softmax不合适；负样本或mask实现问题；last-valid pooling/位置对齐错误；评估时item索引映射与训练不一致。下一步：本地小规模诊断（训练loss曲线、train集自身召回、与id_mean同输入对照、随机初始化基线），不新增服务器训练直到原因明确。
- 更新 2026-10-14 00:20：**根因确认（P-DIAG-01，非代码bug）**。本地排查索引/padding/mask/pooling/loss/负样本均正确（报告 .agents/tasks/p-diag-01-report.md，分支 diag/p-diag-01 dac6c28）。服务器只读真实checkpoint证据（main05/p-diag-server-01.stdout）：SASRec S02 item embedding 相对初始漂移 p50=0.039、cos≈0.999（基本没离开 N(0,1) 随机初始化）；loss 在降但 validation 命中在 0–4 噪声区，patience3 于 epoch4 早停。id_mean 逐 epoch 命中 3,3,2,2,4,4,5,7,…,89,94，第30 epoch 仍在上升（epoch1 漂移0.012→best 0.364）。结论：默认 N(0,1) embedding 初始化 + LR1e-3 学得极慢，叠加小验证集噪声早停；同一原因也压低了 id_mean 等 P4D 双塔基线（0.0373 并非上限）。修复属于新实验版本（item/pos embedding N(0,0.02)、min_epochs、max_epochs100），两条线公平对照，旧冻结结果不改。状态：ROOT_CAUSE_CONFIRMED。

## 2026-10-14 服务器实际CPU配额只有10.24核（更正“64核”口径）
只读实测（main05/kiro-b-cgroup-01.stdout）：cgroup cpu.max=1024000/100000 → 10.24核配额；os.cpu_count/affinity=64只是宿主机可见核数。nr_throttled=35209，累计限流≈17016s（说明此前并发确实被限流）。memory.max≈197GiB。影响：此前“64核、并行几乎免费”的说法错误；CPU分片并行的收益上限约10倍且与并发作业数共享；榨干资源的主要手段应转向GPU批量解码（INFRA第二步）。已通知INFRA-01实现：workers与调度按cgroup配额计算。状态：OPEN（规划修正中）。

## 2026-10-14 scheduler 把暂时放不下的作业直接 reject（OPEN）
现象：g-hist-len-l20-s42 在 admission 时因 headroom 0.169<0.2 被移入 queue/rejected，未启动、未计 attempt；而同时 CPU 实际空闲 90%。另外 G verify（声明5核）在 pending 中阻塞。处理：10:1x 手动把同一 argv 重新入队（kiro-b-requeue-*）。修复方向（ops 分支，TDD）：资源不足应保持 pending 等待而不是 reject；G 作业的 cpu 声明按实测占用调小（实测单 G 作业约 1 核+分片）。INFRA-02 后 G 每作业约 1 核，此问题缓解。

### I-INFRA03 — 调度器把“暂时没余量”的作业 reject（2026-10-14，Closed：16:38 已部署为 sched-serve-02，见 .agents/tasks/infra-03-deploy-report.md；下面 (a)(b)(c) 三项仍 Open）
- 现象：g-hist-len-l20-s42 被移到 queue/rejected，.reason 为 `admission rejected: cpu_cores … headroom=0.169 < 0.2`；人工重新入队后正常运行。
- 根因（代码推断，没有在服务器上复现）：scheduler 预判通过并 claim 后，launcher.launch 在锁内重新测量，准入失败时抛普通 LaunchError，_tick_locked 统一走 reject。
- 修复：分支 infra/03-sched-pending，685409e RED、5840e6b GREEN；ops 141 个测试通过，15 个变异全部被杀。暂时放不下的作业回 pending，请求本身放不下的仍 reject。
- 状态：review 加部署 workflow wf_f4a248aef66a014d 运行中（APPROVED 后才切换到 sched-serve-02）。报告：.agents/tasks/infra-03-sched-report.md。
- 遗留，暂不修（Kiro 决定，2026-10-14）：
  (a) 超限作业在预判阶段永远留在 pending，不会被 reject，也不写原因；
  (b) admission.lock 超时仍然 reject（同属“暂时性”问题）；
  (c) serve 不打印推迟信息，作业长期被推迟时日志里看不到。
  三项都不影响当前 1 核作业；后续在 ops 迭代中处理。

### I-GTOK-LR — RQ-VAE 预注册 lr 0.4 在合成数据上快速塌缩（2026-10-14，Open，风险）
- 证据：只来自 coder 的自述，Kiro 没有复现。
  - 玩具数据：三层码本只用了 1–2 个码。
  - 单位范数 64 维合成数据：K=16 和 K=256 两种设置下，2 步之内每层利用率都塌到个位数。
- 影响：真实 768D Sentence-T5 内容上很可能也达不到卡上 ≥80% 的利用率门槛。真达不到，GTOK-01 的 T1/T2 就记为失败（负结果）。
- 可能原因：都是推断，没有验证。论文没给 Adagrad 的 initial_accumulator_value 和 eps，我们按 TF/optax 默认预注册为 0.1 和 1e-7。这些没说明的细节与 lr 0.4 叠在一起，第一步的步长可能过大。
- 处理：按 decision_log 16:3x 的决定，lr 不改。服务器 fit 在第 2000 步和最终步记录每层利用率。真实数据上失败了，再由 Kiro 在 decision_log 写一次新 attempt：只改论文没说明的那些细节，不改论文已给出的超参；新 attempt 用新的 attempt id，并记入 G 的调参名额。

## I-2026-10-15-a：入队收据 action=written 不等于作业被接纳（Kiro，2026-10-15 11:3x）
- 现象：GTOK-01A 的 A1+T3 训练 g-gtok01a-t3-b256d3-s42 入队收据写 action=written，工作流据此报告“已入队”；实际作业被调度器拒绝，文件在 queue/rejected/，原因“pinned manifest unreadable: /workspace/vkgr-infra03-ops/g-6f5105f8c500.json”。工作流后续又误报“服务器无文件”。发现靠 Kiro 只读复核（收据 main05/kiro-b-fd-0034、kiro-b-rs-0035）。
- 根因（推断）：部署步骤算了 pinned_sha256，但没有把 pinned manifest 文件写到作业引用的路径；先例脚本（garch01c_enqueue_train.py 的 NEWPIN）有这一步。
- 处理：工作流 wf_ae22199e76fc501b 写入正确的 pinned manifest，以新 run_id …-r2 重新入队，并要求以“进入 claimed、runs/ 目录存在”为成功标准。
- 规则补充：以后所有入队都必须在写入后只读确认作业进了 claimed（或仍在 pending 等资源），且不在 rejected，才算入队成功。
## I-2026-10-15-b：部署收据字段 other_worktrees_unchanged 恒为 false（误报）
- 判据只删了新 worktree 的路径行、留下其 HEAD 行，新 worktree 不在列表末尾时必为 false；9 份历史收据全是 false。该字段不能证明其他 worktree 未被改动；GTOK-01R 起用按块的“路径 + HEAD”列表比较代替。来源：gtok01-server-report.md。Open：部署脚本未修。

## I-2026-10-15-c：GTOK-01A（A1+T3）五处缺陷，review APPROVED 没拦住（Kiro，2026-10-15 11:4x）
- 修复步骤 GTOK-01A-FIX 诊断后主动停止，没有重新入队（素材 .agents/tasks/gtok01a-report.md §FIX）。五处：B1 部署只叠了单个提交的 56 个文件，缺 16 个模块（先例 i02b 叠的是 0376d9a..目标的全部 diff）；B2 argv 少 7 个必填参数；B3 config task_id 改动了 G6 的冻结字段；B4 T3 只登记为工程专用，正式路径拒收；B5 G6 只吃 SIDMAP_V2D，T3 bundle 是 SIDMAP_V2。另有 pinned_manifest 写成相对路径（文件其实已写在 /workspace/vkgr-b1-pinned/）。
- 根因：review 只看了 engineering/synthetic 测试，没有用真实输入、正式路径（engineering=False）做端到端检查；入队把 action=written 当成功（见 I-2026-10-15-a）。
- 处理：新任务 GTOK-01B（wf 见 decision_log），路线 (a)：G6 runner 新增登记路径接受 depth=3 的 SIDMAP_V2 bundle，加与旧入口逐位相等的对拍；部署照 i02b 先例全量 overlay；**入队前必须在服务器用真实输入、正式路径跑只读 preflight**，作为硬闸门。
- 规则补充：今后凡是新入口第一次上真实数据，review 之外必须有一次服务器真实输入、正式路径的只读 preflight，才可入队。
## I-2026-10-15-d：GTOK-01B 部署目录里 2 个本来就红的测试（Kiro，2026-10-15 15:1x，Open，低）
- 现象：部署目录 /workspace/vkgr-gtok01b-g 与分支 gtok01b/g 里，test_sid_v2d 的 registry 测试及其变异守门失败。在 GTOK-01A 的 8312c9c 上就已失败，GTOK-01B 没引入也没修。
- 根因（推断，来自 GTOK-01B code review）：GTOK-01A 把 T3 加进 sid_v2d.ENGINEERING_SID_VERSIONS；GTOK-01B 之后 G6 runner 遇到 T3 一律走 V2 加载器，这条登记已经用不到。
- 处理：不阻塞正式作业。正式作业 g-gtok01b-t3-b256d3-s42 跑完后，再撤销这条登记并同步改 gtok01a whitelist 里对应断言（两种修法的代价见 .agents/tasks/gtok01b-review.md）。
- 规则：新入口的基底分支带着已知红测试时，要在 issue_log 登记，部署报告里不能只写“服务器测试 OK”。
## I-2026-10-15-e：php02 服务器侧 test_php02_untouched 两条测试 skipped（Kiro，2026-10-15 16:3x，Open，低）
- 现象：部署目录 /workspace/vkgr-php02-p 上该测试 skipped=2。原因：公开压平分支 php02-pub/p 没有 BASE（3463846）对象，需要对象的两条比较被跳过。
- 处理：decision_log 16:3x 裁决接受，以树 digest 与本地一致 + 本地 skipped=0 为替代证据；服务器上没有独立跑过这两条。
- 规则：以后“相对 BASE 只新增”的守卫若依赖 git 对象，部署到公开压平分支时必然 skipped；设计里要预先写明替代证据，不要靠运行时才发现。
## I-2026-10-15-f：测试词表直接写 <COMPANY> / 本机路径字面量，push 前扫描命中（Kiro，2026-10-15 16:3x，Open，低）
- 现象：php02/p 未推送的中间提交里有 2 处 <COMPANY> 字面量；与 2026-10-14 21:2x 的 gtok01/g 同类。公开分支是压平提交，整体 diff 重扫 0 命中。
- 规则：测试里需要“禁止词”时用拼接或哈希比对，不在源码里写字面量。
## I-2026-10-15-g：FEAT-04a 两个模块没有各自的 fresh RED（TDD 偏差，Kiro 登记，2026-10-15 16:4x，Open，低）
- 事实（来自 FEAT-04a code review，.agents/tasks/feat04a-run/feat04a-code-review.md「TDD 与变异测试」）：`retrieval_feat04a.py`（A 的委托层）和 B 的模型代码是先写后测，没有各自的 fresh RED；FEAT4A-000 与 FEAT4A-001 的 runner 部分有 RED（43 个 ImportError 错误）。违反项目 TDD 硬规则。
- 补强：29 个变异（M4S-01…06、M4A-01…06、M4B-01…11、M4L-01…04），要求守卫在真实代码上绿、补丁后必须抛 AssertionError、还原后绿。
- 另一个读法限制：E2 等价性测试对 f3 补丁型变异有盲区（补丁同时作用于两边），E2 只证明 runner 接线与 feat03 一致，**不是 A 池化正确性的证据**；A 的池化正确性靠 guard_a_*。报告里不要把 E2 当作 A 的证据。
- 处理：review 为 APPROVED，接受这个偏差，不回头补 RED；冻结前跑一次全量回归（review 要求）。

## I-2026-10-15-h：既有驱动与接手铁律存在差异（Codex，2026-10-15 16:36，Open）
- 来源：Eddy 本轮接手指令；本机只读脚本/日志检查与 `main05/codex-takeover-drivers-20261015T163650.txt`。A1 s42 驱动35118命令带600s，T1r驱动68450源码INTERVAL=600；补seed驱动52031与php02驱动32766均900s。四者实测PPID=1，但在Mac上，睡眠/重启仍会暂停verify提交。
- 此外，四个驱动和相关observe/enqueue脚本位于gitignored的main05，不满足工作流硬规则“运维脚本也必须进版本管理”。未检查所有历史脚本，不扩大为全仓已审计声明。
- 处理：本轮是只读接收，保留现有驱动和科学作业，没有kill/restart/resubmit；提交STATE待Kiro复查。新驱动必须先版本化/工作分支push，并采用≥900s轮询；是否迁移现有驱动在安全收口节点判断，不因接手重复执行。
- 另一个可观察性限制：当前工具没有Kiro inspect_workflow能力，四个workflow实时步骤只能保留UNKNOWN/用户交接状态，不能把磁盘报告或旧重启包当实时队列交付。
- Kiro已复查（Eddy本轮转交；decision_log 2026-10-15 16:5x第3–4条）：保留旧600s驱动，不重启；新驱动≥900s且先版本化；旧辅助脚本在对应作业收口后补入Git。四个workflow已有Kiro状态补充，见STATE，服务器验收状态仍以收据为准。
## I-2026-10-15-i：已公开的 base 历史里有本机路径（<HOME> 下的 .venv 路径）（Kiro，2026-10-15 16:5x，Open，需 Eddy 决定）
- 来源：GTOK-02 push 前扫描。整段 `git log -p` 在 base（7a3b815 及更早，已在 origin/gtok01-pub/g 公开）里命中 2 处 <HOME>，是 pytest 报错输出里的 .venv 路径；另有 21 处 <COMPANY> 字面量，是 AGENTS 里“不得含公司信息”禁止声明被改写/删除的行（与 decision_log 2026-10-13 20:1x 记录的“旧 commit 0376d9a 历史里仍有公司字样”同类）。
- 范围：新增提交（7a3b815..0e76598）0 命中。这些是已公开历史里的存量，不是本批引入。
- 处理：本批 push 不受影响。清理公开历史要改写历史、force push，须 Eddy 另批；先不动。建议 Eddy 决定是否清理，以及何时（不要在作业还依赖这些 SHA 时做，部署目录与 manifest 绑定了 SHA）。

## I-P2-READBACK-20261015：审计归档与验收口径的未决证据（Codex，Open）
- B256D2：raw3 B1已实测通过，但当前audit只持久化该结果，补零后四列B1的实测结果未持久化（UNKNOWN）。来源 `main05/codex-p2-verified-audits-20261015T190500.stdout` 的fits（stdout SHA和实际UTC读取时间见experiment_log阶段二节S1）；不能把设计预期256处违反写成读回实测。请Kiro原GTOK-02 workflow补其报告所需审计收据；Codex不重拟合、不改门禁、不读完整SID map。
- FEAT-04a：独立code review含静态泄漏检查且APPROVED，但未重跑测试；是否作为转正所要求的独立泄漏review由Kiro认定。本轮未新开review；E2只证明runner接线，A池化正确性仍靠guard_a_*（I-2026-10-15-g）。
- F7草案发现早期data_contract（Draft、like-only）与已冻结HIST_V2_OBS措辞版本不一致；草案只列待澄清，不改合同、不扩大C/D已批准的full-train例外。路径 `docs/superpowers/plans/2026-10-15-f7-u2-u3-draft.md`。
- 驱动引用已核：g=FEAT-04a TDD偏差，h=驱动差异，无重复g。旧驱动保留不重启；本轮直接核到A/B服务器驱动均已退出，末日志all runs final；本机35118/32766/68450/29224已不存活且原日志收口，52031仍存活。证据见phase2 receipt-index-v2。新只读分析器不是轮询驱动，未新建timer。

## I-2026-10-16-m1-analysis：M1分析冻结/标定未收口（Codex，OPEN）
- 来源：`.agents/tasks/seq-report.md`§2 F3/F4与§5；原seq/m1@04d4ede，公开等树0ca61bc。本轮部署、真实preflight、CUDA/padding/VRAM已通过，证据见experiment_log“2026-10-16 Codex训练安排”D1/M1/M2。
- 剩余：M2触发只看mean还是多臂校正、ablation_g0是否纳入§8.2、分支ratio 0.05/3阈值及F4向量尺度标定。报告要求真实run前完成，目前没有冻结补充，不能把code review APPROVED当作分析规则已裁决。
- 处理：正式M1训练未入队；保持原阈值不改，STATE表交Kiro。未读取旧checkpoint；需要的标定路径/授权也在表中。工程验证仅证明入口和资源，不证明指标有效。

## I-2026-10-16-vram-evidence：GTOK-02新叉数缺真实峰值记录（Codex，OPEN）
- 来源：`.agents/tasks/gtok02-report.md`末节明确3072沿用T3模板保守上界，S512/S1024/B256D2峰值UNKNOWN；早期2000–2038 MiB不是峰值。G1收据证明补seed资源与冻结模板一致，不能证明实测峰值×1.15取整的依据齐全。
- 处理：本轮按Eddy安排的冻结3072声明补seed，未下调资源、未改变科学配置；如实登记证据缺口，STATE交Kiro复查，不把3072说成新叉数实测推导。后续资源报告要有真正峰值记录；不要重启已有科学作业来补测。
- M1不同：本轮另有完整shape工程实测，含进程CUDA context；mean882 MiB，五臂范围862–888 MiB，×1.15向上取整256均为1024。该证据只限M1工程配置，不可代替GTOK证据。
## I-2026-10-16-a：FEAT-04b 阶段 2 服务器侧 test_feat04b_freeze 16 个 FAIL（Kiro，2026-10-16 01:xx，Open，低）
- 现象：/workspace/vkgr-feat04b-p-x2 上 04b 七个模块 197 ok、16 FAIL、1 skip；FAIL 全在 freeze 测试。阶段 1 部署（x1）时同类测试是通过的（skipped=1），阶段 2 多出的 FAIL 与压平分支里缺基准提交 3463846（以及阶段 2 需要的提交对象）有关。
- 处理：decision_log 2026-10-16 01:xx 第 2 条接受，替代证据为 270 个文件 sha256 与源树逐行一致 + 本地 39 个 freeze+untouched 测试全 OK。
- 限制：服务器上没有独立跑过；和 I-2026-10-15-e（php02）同类，说明“压平公开分支 + 依赖 git 对象的守卫”这个组合反复出问题。建议以后冻结与 untouched 守卫改成不依赖 git 对象（用 sha256 清单），这是后续的改进项。

### 2026-10-16 全权接收更新：I-2026-10-16-m1-analysis
- 来源：Eddy本轮明确转交与decision_log02:xx（Sonnet起草，M1真实入队前须Opus/Kiro复核）。原“等待标定”要求已被该裁决替代：0.05/3冻结、不再标定，SEQM1-000仅做固定[0.05,5]前置尺度核对；ablation_g0仍是必报项，不新增门禁。
- SEQM1-000已实际通过：三seed原始m中位数约0.1435/0.1315/0.1498，44个原src源码SHA、validation预测与checkpoint binding核过。来源`main05/codex-seqm1000-baseline-20261016T020100.stdout`，SHA256 `7ca4973f972b448219476d06535742f60c3c6ba0fc7dcb4b9af5baa971f8b24f`，实际UTC读取`2026-10-15T17:36:10.379338+00:00`（完成时间见本轮receipt-index）。本次固定F4 best checkpoint读取是Eddy明确授权的SEQM1-000范围，没有导出向量/修改参数。
- issue暂不关：剩余先决是Opus/Kiro复核与真实入队验收，不再写成“无冻结裁决”。本轮真实训练未入队，不把工程review当成Opus复核。
- SEQM1-000独立代码review发现原9文件白名单漏依赖；已用全部原manifest src账本核对取代，并加遗漏/篡改拒绝测试，fresh RED→GREEN，21 tests通过。没有修改模型、数据、预注册阈值或重跑科学作业。

## I-2026-10-16-m1-control：正式dry-run的python-c入口缺__file__（Codex，已修复）
- 来源`main05/codex-m1-night-dry-20261016T030400.stderr`与对应transport exit1，连接正常；`frozen()`用__file__定位清单，但writer通过python-c执行没有该变量。失败在任何队列写入前，不是训练attempt失败。
- 修复：核心6a8e057和已部署模型均未改；控制入口7bec3b8按SHA导入服务器Git模块，保留模块__file__及同目录冻结清单。新回归先RED再GREEN，合计34 tests通过；dry2十五项prepare通过，随后mean三seed全claimed。来源及SHA/实际UTC见experiment_log本轮D1/S1/C1与receipt-index-v2。
- 同轮一次review发现终态分类漏launch_failed/wall_cap_exceeded，已在核心发布前fresh RED→GREEN修复，避免白等48h；没有因此重跑科学作业或复审。

### I-2026-10-16-m1-analysis收口更新
Kiro在Opus下复核通过，已实际完成十二诊断配置真实preflight与mean接纳，原“等待Opus/标定”的阻塞已解除。当前该实施前置issue关闭；未来M1结果解读、独立泄漏review/延迟/转正要求仍按原规则，不因工程准入关闭而自动通过。

## I-2026-10-16-morning-transport：本机连接残留阻塞晨检（Codex，已修复）
- 来源：M1-R1成功后数次wrapper `FAIL no connection`，旧formal55 master/owner实际均不存在，旧socket残留且缺退出凭证；原恢复guard要求socket不存在及actual exit code，不能据进程不存在伪造exit。新增本机truthful观察记录仍不能通过guard，旧失败均保留。
- Eddy在当前对话明确批准仅修本机恢复工具，不删除旧socket、不变更/重启/重投服务器科学作业。新版本7511063强制双进程不存在、owned private socket路径合规、旧control check失败，才新建独立通道；记录独立recovery观察，不写假exit，不删除旧socket。旧脚本备份到cache/codex-morning-acceptance-20261016，固定Git blob安装，38tests通过。
- 修复后M1-R2、G-R1、H-R1均成功，全部原始科学verify已核；各stdout/source/transport SHA及实际UTC见experiment_log晨间最终节和receipt-index-final.json。没有把连接失败当训练失败。

## I-2026-10-16-overnight-idle：有限队列已提前用尽，存在项目空档（Codex，Open，排程）
- 来源：M1-R2扫项目terminal，最后科学终态04:11:49+08；H-R1约10:21+08无科学worker/pending0，仅scheduler服务；完整收据SHA/UTC见experiment_log晨间最终节。项目终态至快照约六小时；没有连续host监测，不推断外部未知进程历史。
- 已批准的M1十五+十五及GTOK六补seed+六verify全部收口，不是驱动漏接；十二诊断自动接续已逐项证实。后续FEAT组合/A1新SID/M2等均未完成既定前置，不越界开实验；Eddy曾明确不建晨间timer，原自动任务继续暂停。
- 已主动向Eddy报告实际空档，不能将“有限批次接续成功”写成“整夜不断档”。后续需提前把已登记候选落实成冻结、真实preflight通过的有限READY批次；不能为填空改规则或挑seed。待下一批排程落实后关闭。

### I-2026-10-16-overnight-idle：当前恢复训练（Codex，2026-10-16）
已按Eddy要求完成并提交原登记FEAT04条件组合三seed，C1全claimed、H1实测三科学worker running/pending0（实际收据截至10:56+08）。来源experiment_log FEAT04-COMBO节及cache/codex-feat04combo-20261016/receipt-index.json，含stdout SHA/实际UTC。当前空档已缓解；“提前备好下一有限READY批次”的长期排程事项仍Open，不把当前三训练等同于后续无空档保障。没有越界启动待Kiro裁决的A1新SID或M2。


## I-2026-10-16-inventory-push-scan：一次工作分支push前扫描漏做（Codex，已补核）
- 来源：本轮只读inventory工具aaf1f40推送时漏做公开扫描；之后对7511063..aaf1f40完整新增历史补扫，零命中。没有发现泄漏，不据此改写历史。该步骤偏差不能被写成“push前已扫”。
- 后续延迟bench b089b8f、工具8811c52及B1024 c1c4b74均先扫新增历史/文件大小后push；固定SHA部署，未push main/force或修改旧公开历史。B1024扫描完整记录cache/codex-successor-engineering-20261016/g-public-scan.json。旧公开历史路径问题I-2026-10-15-i仍待Eddy单独裁决，不在本次处理。

### I-2026-10-16-overnight-idle：后续批次准备与新空档（Codex，本轮）
新组合在11:17:26+08已收口，而12:48 host只读仍为空，约1.5h项目空档已主动报告Eddy；没有连续host监测，不推断外部未知进程。来源本轮experiment_log C2、inventory及receipt-index。M1延迟小作业随后已实际接纳并succeeded；B1024 A1源已冻结/部署、真实preflight与完整CUDA工程推进，不以填空为由跳过门禁。长期排程事项仍Open，M2只设计、其它未冻结候选不自动训练。

## I-2026-10-16-b1024-launch-record：工程run记录超64KiB，supervisor未启动worker（Codex，Open）
- 来源 `main05/codex-b1024-engineering-diagnostic-20261016T053100Z.stdout`，内部实际UTC2026-10-16T05:29:37.638672+00:00；完整SHA/source/transport见本轮receipt-index。cuda-v1仅run.json及logs/supervisor.log；run.json为66316字节，日志05:21:58Z写“run.json unreadable; nothing to supervise”；没有owner/stdout/stderr/terminal。host无科学worker，不能把claimed当运行。
- 根因：Git全overlay的dirty_files与pinned_files详细账本重复写入run.json，超过共享supervisor/common.read_json_limited默认65536字节；launcher dry只检查prepare，未检查最终记录上限。本机用有效66316字节synthetic JSON复现read_json_limited返回None。科学模型/CUDA步骤完全没启动，没有可用显存实测，真实三seed均未入队。
- 原目录、claimed、日志全部保留，不伪造terminal、不重启/覆盖/重投。拟修仅新clean Git launcher载体+逐文件校验原DEP的受guard入口，保持科学source/PF/config完全相同；新增record-size余量检查。共享scheduler/读取上限不改，原DEP/pinned/provenance不改。工程新执行须Eddy明确批准，禁止以换run_id绕过失败attempt边界；修复、测试与preflight/dry先做成可审阅产物。
- 本次69回归/一次review/prepare dry漏掉运载记录上限，不能代替真实工程验收。原“工程已claimed”等记录是当时事实，当前分类是启动层失败且未验收，不是假写科学train failed或verified。

### I-2026-10-16-b1024-launch-record：根因修复验证通过（工程阻塞关闭）
Eddy明确批准仅一次修复后cuda-v2；clean launcher e3df6db保留原DEP全文件guard与科学绑定，增加48KiB元数据预算。新工程真实worker运行并succeeded，直接cuda_engineering_verified，fullshape/解码/reload/VRAM全过；原v1完整保留，无重启/覆盖/伪造terminal。正式三seed dry通过、最终全claimed/no rejected，H2实测三worker running，新driver2924901 PPID1/900s/48h存活。启动层阻塞关闭；三科学训练与verify仍未验收，不把工程成功当召回结果。完整数字/source/stdout SHA/实际UTC见experiment_log SUCCESSOR§6–8及receipt-index。

## I-2026-10-16-m2-review-integration：启动与M1控制链缺口（提交前已修，Closed）
- 来源：唯一独立whole review6b8eba8的报告，cache/codex-m2-execution-20261016/code-review-approved.md。两HIGH为实际模板-B未规范化导致train/verify入口拒绝，以及detached driver未注入infra03/ops导致无法补verify；一MEDIUM为M1控制审计未强制/未写入M2 manifest。
- 全在原设计内修复：normalize注册argv并测试实际entry；prepare_job自身建立ops import path；M1三seed原manifest/pred/source/data/inputs/verified证据封存hash，经job inputs/env进入real runner与manifest、verify重做，缺失/漂移拒绝。fresh RED→GREEN，修复2ca21e5+2d5114e；同一review只闭环原三项后最终APPROVED，非第二轮设计或whole review。
- 冻结、实际三真实PF/control seal、新CUDA直接verified及三seed全claimed/真实worker、driverPPID1均已验证；没有失败M2科学attempt或改设计。数值与逐source收据见experiment_log M2节、receipt-index，工程问题关闭，scientific仍待验收。

## I-2026-10-16-m2-baseline-contract：本地fullsuite的旧P3C pin差异（限制保留）
- 来源：M2原fullsuite930测试为1FAIL/1ERROR/2skip；未改基线b089b8f上同两项定点复现同样失败，Git P3C文档5449378c…而producer冻结contract期望c89b6466…。测试名/日志及完整SHA在science-final-report与baseline确认日志，不宣称fullsuite全部通过，修复后的23科学/9ops/1工程及84项M1定向证据分别保留。
- 沿既有M1/FEAT部署SOP保留新DEP的Git原文，另提供逐SHA验证的producer数据通道副本，未更改期望hash、旧文件或科学阈值。当前三真实PF已实际通过；本地Git文档与producer工件的区分仍是复现限制，不作为M2新回归遮掩。

## I-2026-10-16-sidfix-legacy-verify — 四个旧归档未找到标准 verify 收据（OPEN）
来源：Codex SID-FIX metadata discovery，main05/sidfix-s0-v2-discovery-20261016T090915Z.stdout，stdout SHA256 `5ba6eede9515bcd92e89358a3c8226f5821be9b7b5e33fac3d4dd142983ab219`，实际读取 UTC `2026-10-16T09:09:16.636392+00:00`。
- `g-hist-v2-l50-s2025`、`g-g4-rand-s42/-s2024/-s2025`：train terminal succeeded，但标准 -verify / -v1-verify 身份未找到 verifier job/原始 verified。旧本机 deploy02-autoverify-state.json 的 submitted 也未登记四项。这里只能说收据未找到，不能断言其它身份绝不存在，更不能把训练成功当验收。
- S0 保留四行 UNKNOWN；不读这四项预测，不算其 B/C 数字或三 seed 均值，不伪造 verified。其它 29 个已验收归档照算；B1024−T3 的独立补 seed 与 A1 两配置不受此缺口影响。
- 本 task 禁止 checkpoint，因此不补 verify、不重跑、不改旧目录。交 Kiro 判断历史补证方式；不更改原训练或转正门禁。

## I-2026-10-16-sidfix-preflight-metadata — P metadata 文件名差异（CLOSED，入队前发现）
Codex首次只读discovery在 P prepared目录寻找manifest.json，被真实producer合同prepare-manifest.json拦住；没有S0作业/训练attempt。读取retrieval_data.py实际合同后fresh RED→GREEN修复，保留v1/v2，只在新v3部署。原独立whole review的三项provenance缺口也已由同reviewer关闭；普通文件名/UNKNOWN呈现修复自回归，不另开设计评审。新真实PF结果另见experiment_log，不以此issue冒称入队成功。

## I-2026-10-16-sidfix-queue-schema — 新 pending 缺 schema 被 scheduler 跳过（CLOSED，已独立 claimed）
来源：Codex S0 首次提交与 scheduler.py QUEUE_SCHEMA 检查。首个 pending `01792142148793391058-sid-fix-s0-first-v1.json` 漏 schema，scheduler 忽略；无 worker/run 目录、没有失败 attempt。Eddy 明确批准一次仅新增schema字段的pending修复；新工具13633c7入Git/push，fresh RED→GREEN，40 tests PASS；一次whole review此前已通过，普通工程修复未重开设计评审。

修复收据 `main05/sidfix-s0-approved-pending-repair-20261016T092022Z.stdout`，stdout SHA256 `fd9301e07382620b91d92f160557044a06333dabb94c301fc59a0445eced6679`，实际 UTC `2026-10-16T09:20:23.113208+00:00`。scheduler.lock内再查唯一pending、无claimed/rejected/run，先保存原字节到 `/workspace/vkgr-sidfix-s0-ops-v4/pending-original-42e18400ae4be193bdc98fc1fd6d8152ff40c147c03d82c56879ade1e29ab414.json`；旧/备份SHA `42e18400ae4be193bdc98fc1fd6d8152ff40c147c03d82c56879ade1e29ab414`，新SHA `c5d8b4f3f794938c122a46533347e8d8e7d64eacf5ddcc2477a7af2dfee9f42f`。只新增schema=vkgr-queue-job-v1，原run_id/argv/config/data/resources不变，未重投；已修正后续publish，随后独立收据已确认原身份唯一claimed，无pending/rejected/.reason；见experiment_log第一批接纳。

## I-2026-10-16-sidfix-popularity-cost — catalog线性成员检查（CLOSED，原run保留）
Codex在首次S0已运行时发现C函数对list catalog逐预测线性检查，增加CPU成本。没有停止/重跑/覆盖第一批；仅在新v5部署把成员集建一次set。fresh RED→GREEN，45tests通过；20个synthetic×200query全字典精确相同，且两批重复的真实A1T3三seed B/C与T3/B1024 A/D逐项相同。原第一批与补批工作量不同，wall不能作性能对拍；源/结果/UTC见experiment_log S0最终节与popularity-equivalence.json。

## I-2026-10-16-sidfix-control-label — A1差分字段名称歧义（主账已澄清）
服务器result的C_delta_vs_same_seed_T3对A1行使用A1T3，同名字段不能被读成core03 T3。未改旧产物；experiment_log明确从原命中重算每行对core03 T3，并另列A1B1024−A1T3同规模差分。原计数/判读正确，展示须带对照全名。来源S0最终收据055c781722a33e7b9297fa382f3611bdc783ff0e7843d5ef7e74ac48be628331。

## I-2026-10-16-sidfix-lift-readout — 深层结构零与random self-pair读法（OPEN，待Kiro）
S0 T3/B1024等语义叶为单item，共现定义排除相同item，深层prefix共现为0是结构必然；uniform独立有放回random包含self-pair，对应matches=26，因此lift=0不能直接读成无协同。实现的random取法看数前已在doc/tests中明确，但冻结卡未明确随机是否也排self；请Kiro复核意图，不由Codex看数后自改函数。完整A–D及315/1272根输入SHA条目和嵌套verifier证据见experiment_log S0最终节及原result。此为读法问题，不改既有门槛/Q1。

## I-2026-10-16-sidfix-s1-beam — 草稿误写解码宽度（CLOSED，冻结前纠正）
Codex整理Opus提示时核对S0实际归档：g-tok-t3-bhkm-b256-s42/-s2024/-s2025 的 config.beam 均100；S1 DRAFT§6误写50。已纠正为沿T3的100，不是改已冻结配置或增加调参。来源main05/sidfix-s0-final-results-20261016T095052Z.stdout，SHA055c781722a33e7b9297fa382f3611bdc783ff0e7843d5ef7e74ac48be628331，原读取UTC2026-10-16T09:50:53.695156+00:00；实际字段在first-result.canonical.json的inventory.run_input_bindings。未实施、拟合或训练S1，关键设计仍交Kiro。

## I-2026-10-16-sid-coll-summary — 一次review发现汇总体积与诊断遗漏（CLOSED，真实作业前）

来源独立review docs/superpowers/reviews/2026-10-16-sid-coll-s1-code-review.md（S1工作分支）。真实双支tick模拟汇总超过queue工具的48KiB限制，driver会退出；同时诊断proof未进入终态汇总且SHA类型丢失。Owner fresh RED→GREEN关闭：汇总单独2MiB有界，queue/job/launcher仍48KiB；按hash读A–D/D′，保留物理文件/计算digest类型、UNKNOWN及所有run。真实driver完整大汇总回归通过，不追加review、不改科学规则。原review CHANGES_REQUESTED留存，不冒称reviewer再次批准。

## I-2026-10-16-sid-coll-transport — 新只读入口模式不兼容（CLOSED，部署前）

来源main05/sidcoll-before-deploy-20261016T113638Z.stderr：连接正常，脚本错误消费空JSON stdin，检查未执行、无服务器写入。对照原正常wrapper后仅允许显式python -c读取写请求，其它stdin-source入口默认RO；fresh RED/GREEN及回归通过，079d6bd新版本部署前RO确认全部新身份空，旧脚本/收据保留。无科学attempt重跑。

## I-2026-10-16-sid-coll-inotify — hardlink CREATE未被完成事件过滤接纳（CLOSED，观察owner一次接续）

来源infra03 common.write_json_once实际os.link发布terminal.json，事件为CREATE；原观察只认close/move，因此只剩900s兜底，未丢job但多空等。fresh RED复现→GREEN后1fa96dc观察代码新目录部署；只停止本轮新观察2926997，全部科学job/原plan/intent/截止保留，不重投、不覆盖原pid。新2927689实际PPID1/alive，own_terminal已证实接续。收据main05/sidcoll-raw-with-epochs-20261016T145310Z.stdout，SHA312bb2a42c69b2542e6817abb25f46e3fba6e767965e6b46fc21529657879ff2，实际UTC14:53:11.456452Z；原始handoff/source hashes见receipt-index-v1。

## I-2026-10-16-sid-coll-aux — 历史queue metadata文件名误触test guard（FAILED attempt保留，安全只读替代）

sid-coll-legacy-epochs-v1在扫描历史queue JSON时被文件名test子串误拦，尚未读A1 epochs；不是读取test数据。失败attempt不重投、不换ID、不覆盖。后以版本化小metadata reader独立核固定三个已验收manifest及43epoch；别名扫描跳过test-named metadata，四旧归档仍UNKNOWN，不称穷尽检索，不读旧CP补verify。来源同上312bb2a4收据与experiment_log对应完整曲线；原失败stderr/terminal留存。该缺口不阻塞S1。

### S0 lift读法问题的裁决补记

I-2026-10-16-sidfix-lift-readout已由Kiro夜间SID-FIX§2裁决：原D保留，另报D′不同视频随机对及STRUCT_NA。D′新只读结果已验收，数字只见experiment_log，不修改旧产物或将结构无定义读成数值零。本条关闭“需决定随机定义”的阻塞，结果解释仍依该裁决边界。

## I-2026-10-17-s1-codec-depth — 真实preflight读取错误属性，两支STOP（修复已备，重验待Eddy）

Codex工程错误：sid_coll_cuda_check.context检查ctx.codec.depth；实际sid_v2.build_codec返回_ValidatedCodec，depth属于codec.catalog。两支均完成input和SID full replay verified后，在真实G preflight抛AttributeError；C2于10月16日23:32、C1于10月17日00:02失败，driver正确写ALERT退出。无G训练入队，01:20检查发现已空档约1h20，已直接向Eddy说明；旧测试缺少此真实codec接口集成案例。

本地新增真实sid_v2.build_codec回归fresh RED→GREEN，0f39d69仅修属性访问，仍要求深度3/beam100；76项S1回归通过并push工作分支。没有部署/重投/重启，旧失败产物保留。按照夜间裁决“失败attempt不换run_id重跑、不覆盖”，申请两支各一次修复后工程preflight重验；复用已验收向量/SID，不重新导出/拟合或读额外旧checkpoint，不改科学规则。

来源：main05/sidcoll-progress-20261016T172025Z.stdout，SHA33a533c2a34f2005ee2ed37ebb58a7e8d97ae575f0fd6cdc8619168caf7ca055，实际UTC2026-10-16T17:20:26.765523Z；含两失败terminal/claimed/stderr和原始verified。全局空闲资源来源6af6a789ccf5795cde32510dc5ed1540f9fb74d065bcaa04362a2fdbf9664ade，内部UTC UNKNOWN。详细成本/健康数字只在experiment_log新01:20节。

### codec-depth关闭补记

Eddy已明确准许C1′/C2各一次修复后工程重验；两个preflight-v2均已通过，六配置真实engineering=False验收、source/config/data绑定一致。源11d6768新目录296文件核SHA，十个旧成功阶段复用，旧失败保留。原始读回main05/sidcoll-retry-gates-20261016T180106Z.stdout SHA81273c8102a5772c1478dcd49482c05fb382aa42ad5435ca630d35b54b36aac9，实际UTC18:01:07.435528Z。此错误CLOSED，不再请求同项许可。

## I-2026-10-17-s1-capacity — 容量探针取理论80%上限，1MiB设备用量使准入永久pending（待元数据许可）

Codex工程声明错误：CUDA容量探针声明39312MiB，host总49140/free49139，现有20%余量条件下(49139-39312)/49140<0.2；两CUDA pending未启动且无claimed/rejected/run目录。源码admission.evaluate实测旧拒，新39168（256向下取整）通过，measured_headroom=0.2029100529100529。门禁本身正确，不改阈值；该上限不是测得训练峰值，后续训练仍按测峰×1.15 ceil256。

具体方案已实现fresh RED/GREEN、83tests、push0b2becc，默认只读：仅两pending的vram_mib39312→39168，原字节/SHA备份，scheduler.lock内原子替换；创建新匹配plan、不覆盖旧plan，更换观察owner读取新plan，保留原deadline/intent/run_id/config/data。尚未执行修改，因为Eddy规定覆盖任何文件须先问；已通过专用问题请求这两份JSON的单次许可，不能借preflight重验许可扩大。

收据main05/sidcoll-capacity-proposal-20261016T181131Z.stdout SHA2551fc942404a31bea3459f3494fffbc06611cc8ef983b4e90a323178d2d92db，实际UTC18:11:33.635935Z。完整两路径/旧新SHA与新plan SHA见capacity-proposal.json及experiment_log，不复制原数据或完整向量。未重投CUDA、未改门禁或科学内容。
## I-2026-10-17-s1-capacity 关闭补证（02:20+08）

Eddy明确批准两条元数据修正后，已备份原字节/SHA，仅改两pending容量并新建匹配plan；原run/config/data/20%门禁/截止不变，没有重投。新driver2931675实际PPID1/alive，共现向量 SID CUDA已验收cuda_engineering_verified，内容加 P 向量 SID CUDA已claimed，均无rejected。39168是工程探针容量，训练声明仍按实测×1.15 ceil256。原问题的“待许可”记录为历史。

来源main05/sidcoll-capacity-stage-20261016T182041Z.stdout，SHA256 f6678cd77254b7a68275489ab6c8899687304bcfd7cf6ce97e5afa7128de16be，实际UTC2026-10-16T18:20:42.842500+00:00；修复与独立PID确认的完整收据见experiment_log容量接续节。后续科学训练/verify仍待验收。


## I-2026-10-17-readonly-identity — 补数工具归档身份遗漏（已关闭，未用旧版读真实结果）

一次独立review指出配对工具仅组合绑定原plan，另外三个族未固定原已验收manifest；同数据seed但不同模型的自验归档可能误标注册对照。新增12个既有已验收manifest SHA pins，fresh adversarial test RED缺守卫→GREEN。22项新工具测试通过、发布6268807；真实补数使用修复后版本，收据/模型binding见experiment_log本轮。不改科学阈值、不追加review。

## I-2026-10-17-scale-cli — 合成基准controller静默忽略显式参数（已关闭，正式运行前修复）

一次独立review发现controller接受但忽略规模/seed/较短wall参数。fresh RED复现后明确controller只接output-dir/wall，worker-only参数显式拒绝、wall合法范围检查并传六子进程；15项synthetic+83项协同SID回归通过。ddf208e正式本机运行于修复后，所有未完整拟合的suffix保持UNKNOWN。review两份报告见docs/superpowers/reviews/2026-10-17-sid-scale-code-review.md与2026-10-17-readonly-evidence-code-review.md；本轮context计数UNKNOWN。

## I-2026-10-17-public-reader-paths — 公开推送自动审批拦截部署路径（已通过脱敏解决）

自动审批拒绝首次公开push组合命令，理由是三个新工具含实际服务器目录/部署元数据，可能公开泄露。未绕过拒绝：先只读核已有公开目录引用，再将三个新工具全部实际部署位置移出源码，改从gitignored本地私有JSON读入，新增路径参数RED/GREEN；新提交全六文件扫描无实际服务器/本机目录、IP、密钥、超限文件。6268807后续公开push通过；MAIN05以已push原字节运行，rw标志仅供-c传JSON，代码无服务器写入。未改transport/训练/驱动；没有需要Eddy补批的剩余阻塞。

## I-2026-10-17-biconfirm-kind：正式dry元数据类型不合法（一次修复通过）

来源main05/bi-formal-dry-20261016T201100Z.stdout：原launcher拒绝kind=train，stdout空，无训练发布。责任为Codex请求构造。按AGENTS第4节常设一次工程重验，只新增formal-request-v2.json与batch-plan-v2.json、kind改ops，科学绑定与原截止不变，旧文件保留。main05/bi-formal-dry-v2-20261016T201630Z.stdout直接dry_ok，随后唯一driver发布，不重复提交；完整SHA/UTC见experiment_log本轮工程验收节。

## I-2026-10-17-combo-receipt-output：真实泄漏回执父目录缺失（一次重验关闭）

p-combo-leakage-real-v1因回执父目录缺失失败，原stdout空。Eddy显式批准一次IO修复；3dd6215保持audit_runtime函数SHA2a2f1db4cbe3f53b10538a292f419bc1ec6ca87f974db624750bc90afeafd6ef，先stdout.flush并创建新输出父目录。v2原始pass，reviewer仅关闭P1为APPROVED。旧失败全保留。收据与限制见experiment_log及P1闭环review。

## I-2026-10-17-owner-before-publication：三份新训练工具的提交原子性与显存证据（提交前关闭）

独立review在双向序列模型确认、序列用户塔+组合叠加、温度交叉熵实现中发现发布先于owner ready的窗口；未提交科学训练前已修复。完整plan先封存，detached唯一owner持lock、ready并由controller确认release后才发布；spawn/release前断线零发布、release后断线接续、部分发布失败不重试、已发布作业仍唯一verify，均fresh fault tests。显存proof同期补自身PID/context基线+allocator峰与公式重算，不能只认allocated。双向当前driver实际PPID1/alive有独立收据；R1/叠加仍在工程前置，未来必须独立核实际driver。来源三个本日code-review及其修复交付记录，科学实现和冻结阈值未改。

## I-2026-10-17-import-routing：新部署继承旧PYTHONPATH风险（真实训练前关闭）

root检查模板发现可能将旧src带入新job。温度交叉熵7a06178、叠加40eb70e及完整DPO277a1f1均把train/verify/engineering env强制唯一新DEP/src，并保留科学data_root私有合同；synthetic新RED/GREEN覆盖。双向确认正式请求也显式新DEP/src，不复用旧路径。没有对运行中的协同向量SID作业作修改。来源各工作分支源码diff和本日独立review修复证据。

## I-2026-10-17-scheduler-pin-schema：R1正式dry混用部署收据格式（一次工程修复中）

来源main05/r1-formal-dry-20261016T204250Z.stdout：原launcher拒绝generic deployment.json键集，需vkgr-pinned-v1的schema/base_commit/target_commit/files/overrides。未正式发布R1训练。Codex构造请求责任，不改原门禁。按常设一次工程重验，仅新建scheduler-pinned-v1.json与request-v2.json，base=target=clean code HEAD，allGit文件SHA、空overrides；旧deployment/PF/CUDA/stdout均保留、不覆盖。原launcher已实际接受：R1工程dry26460 bytes、R2工程dry27362 bytes。严格PF.job等值门禁保持，所以补各自只读真实PF-v2绑定新运载元数据，再做R1唯一正式dry重验；源科学代码/config/data/seed/CUDAproof/原deadline不变。R2尚无失败正式dry，提前纠正同类请求。若正式重验再次失败，停止该支回Kiro，不再追加attempt。

## I-2026-10-17-sidcoll-stage-alert：原始train成功但驱动阶段验收ALERT（恢复验收闭环）

事件：原driver2931675退出；capacity readback两分支reason“persisted stop: stage train42 failed”。但独立raw terminal显示sid-coll-c1-train42-v1与c2均succeeded/exit0，stderr空、stdout均complete并指向receipt.json。不得把driver摘要当训练失败，也不能因此称已verified。Codex一度在对话沿用driver摘要，已立即按raw收据更正。原S1任何job/driver/plan/截止均未改；不重启不重投。只读定位阶段验收/receipt字段原因中，需原始receipt/持久stop证据后交Eddy/Kiro，科学结果尚未验收，不据此补seed。来源main05/sidcoll-driver-final-read-20261016T205210Z.stdout与s1-alert-r1-admission-20261016T205330Z.stdout，完整SHA/UTC后续同experiment_log收据表。

## I-2026-10-17-sidcoll-helper-text-source：恢复inventory文本白名单漏uv.lock（一次工程重验通过，关闭）

首次新helper真实inventory只读STOP，source/config text files only；真实296源码pins中唯一未通过relative白名单的是已冻结根文件uv.lock。尚未写S1新plan/control、未发布verify，所有旧数据/driver/STOP不变。ordinary修复精确增加uv.lock文本源文件许可，不泛放开binary/CP，fresh RED→GREEN与tools68回归已通过，本地9b5378c；旧helper部署/OPS-v1与错误stdout/stderr保留。来源main05/s1-handoff-inventory-20261016T213030Z.stderr。按常设一次工程重验，源科学commit/数据/12配置/资源/原截止不变；实际一次重验inventory已PASS，后续prepare/start/独立Linux ACK/两verify唯一队列回读均通过，完整SHA/UTC见experiment_log“05:50获批恢复”节。


协同向量SID阶段ALERT后续（2026-10-17 05:50+08）：已按Eddy批准用新计划修复十二个冻结config pins、新control_dir，原成功train42及旧STOP/intents保留。工具84ab6d9，真实inventory、prepare、owner2953005的PPID1/三锁独立ACK均通过；只补缺失verify，两条一claimed/一pending，无rejected。工作流恢复阻塞已解决，科学verify及补seed判定仍OPEN，不能把此项关闭解读为科学验收通过。原driver未重启，不重跑训练；原绝对截止不变。收据与限制见experiment_log“05:50获批恢复”节。


协同向量SID验收闭环（2026-10-17 12:49+08）：新计划两支summary-final complete/unaccepted为空，六训练与六原始verify全部成功；十二receipt冻结expected逐字段、六manifest/verification SHA已独立核对。原config SHA空值阻塞已解决并验收，driver正常退出；保留旧STOP/失败attempt，不重启。科学UP/DOWN是冻结规则机械结果，不是程序失败，科学解释交Kiro。来源experiment_log“12:49收口”节四份完整stdout SHA/UTC，不改decision_log。


## I-2026-10-17-g-earlystop-censoring — G线早停曲线一次只读诊断（已观察范围明确，风险保留）

授权：decision_log「2026-10-17 午后」§3。一次读取A1+T3、温度交叉熵（R1）、A1+B1024、共现向量SID（C1′）四组各三seed的已存manifest，共12条完整已观察曲线；只用epochs中的all_eligible validation R@50/N@50，未读checkpoint/test/逐query预测，也未重跑或改patience。方法为新增工具postbatch_acceptance.plateaus，7项初始手算synthetic夹具包含R平手/N刷新、3轮不刷新后恢复、终态右删失及非有限/缺epoch拒绝；完整新工具回归31项通过。

**结论：在已保存、早停之前的这12条曲线中，没有观察到“平台3个epoch后再上升”的普遍现象；两种明确口径均未见实例。不能由此推断延长训练不会回升。**

- 狭义R平台：连续至少3个已观察epoch的R@50完全相同、随后更高R；观察为0/12。没有设新的epsilon或“基本持平”阈值。
- 按真正早停选best口径：连续至少3个epoch未刷新全程最佳(R@50,N@50)字典序，之后刷新最佳；观察为0/12。此项强烈受既有patience=3规则截断，不能当作模型永不恢复的证据。
- 十二条曲线都在最后一次最佳后保留3个未刷新epoch即停止，之后没有观测；停后是否会回升全部UNKNOWN。停止时点/初始化噪声可能影响G配置比较的风险仍保留，不能据这项诊断直接归因R1差异、延长预算或补跑。

| 已观察run | best/stop epoch | 相同R平台3epoch后升高 | 三轮不刷新lex best后恢复 | 停止后的恢复 |
|---|---|---|---|---|
| g-gtok01b-t3-b256d3-s2024 | 20/23 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-gtok01b-t3-b256d3-s2025 | 16/19 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-gtok01b-t3-b256d3-s42 | 8/11 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-gtok02-a1-b1024d3-s2024 | 14/17 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-gtok02-a1-b1024d3-s2025 | 9/12 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-gtok02-a1-b1024d3-s42 | 11/14 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-ranking-r1-t08-s2024 | 23/26 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-ranking-r1-t08-s2025 | 8/11 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| g-ranking-r1-t08-s42 | 18/21 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| sid-coll-c1-train2024-v1 | 17/20 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| sid-coll-c1-train2025-v1 | 14/17 | 未观察到 | 未观察到 | UNKNOWN（右删失） |
| sid-coll-c1-train42-v1 | 16/19 | 未观察到 | 未观察到 | UNKNOWN（右删失） |

原始main05/epoch-curve-once-20261017T051548Z.stdout，完整stdout SHA256 bcc11573de62def002d08c0a99126ec293f8bdddc653d2f6e61587dde68126e5，实际读取UTC 2026-10-17T05:15:49.304036+00:00；逐manifest原字节SHA、完整已观察R/N曲线与函数输出保留于gitignored cache/codex-morning-execution-20261017/epoch-curve-diagnosis-20261017.json。主owner已逐run核epoch从1连续到stop、selection.best与曲线对应且R/N一致。该结论只在本issue维护，STATE/报告仅放指针。没有更改任何门槛或提出已授权的预算延长；如需改patience/训练预算必须另卡、Eddy批名额。


## I-2026-10-17-latency-observer-transport — 纯事件SSH通知未及时返回（作业已成功，观察限制保留）

本机一次被动inotify观察postbatch-users-terminal-event-20261017T052539在elapsed1754.3573215秒后exit255、stdout/stderr空；后续report读的旧SSH master返回Broken pipe。重新连接后只读原run确认：计时作业已在北京时间13:37:07成功，terminal succeeded/exit0、stderr空、stdout latency_measured，report SHA独立核对通过。无重跑/重投/重启。此前Codex将“没有收到终态通知”写成“仍在执行”，已向Eddy明确更正；今后没有新的进程/终态证据时只写“通知待收、状态未刷新”。

观察失败根因尚UNKNOWN：不能仅据此判inotify实现错误、挂载丢事件或SSH寿命限制。该RPC本身不是训练owner，也没有周期fallback；原科学driver的>=900秒fallback与有限截止不变，未受影响。后续单次远程纯事件观察宜限制在900秒内，超时/断线先读原run/terminal/进程，绝不能重投。暂不为本次扩展修复scope或新建timer。

来源：main05/postbatch-users-terminal-event-20261017T052539Z.transport.json、postbatch-users-report-20261017T055615Z.stderr；独立成功终态main05/postbatch-after-event-transport-20261017T055522Z.stdout及report-reconnect-20261017T060046Z.stdout，SHA/UTC见experiment_log“14:01最终验收”节。连接断开不等于作业停止；观察限制OPEN，作业验收已完成。


## I-2026-10-17-s5-source-envelope — 新SID A1发布前源码与运输包络检查缺口（工程修复关闭）

一次独立review对象34a2000：identity只核HEAD与请求自报hash，dirty/assume-unchanged篡改可漏；launcher.prepare返回的最终prepared未做尺寸门禁，65552 bytes仍dry_ok。均工程MEDIUM，无科学设计偏离，正式作业未发布。author先五例fresh RED，再clean+逐文件Git blob/物理SHA双核验；统一spec+prepared实际JSON>=47104拒绝，dry/train/verify均在intent前检查，verify失败latch ALERT不重投。最终a5da4d6，98相关tests含4mutants/真实detached故障与47103/47104边界通过。旧入口全A差异不改，未做第二轮完整review。来源coll-sid-a1-code-review.md与featuretree实施handoff；原RED/GREEN日志<TMP>/coll-a1-review-*.log。

完整suite未完成，旧G6 fixture在input hash mismatch处报错（旧文件未改），不声称全库通过；真实服务器数据contract在后续preflight独立核，若真实共同数据失效照STOP，不把本机旧fixture限制当服务器已验收。


### I-2026-10-17-s5-owner-metadata-path：一次只读文件名修正（已关闭）

来源Codex；首次owner seal核查误写driver-request.json，实际版本化入口产物为submitted-request.json，原s5-owner-seal-20261017T070215Z.stderr保留。没有服务器写入/训练重投/预算重置；按AGENTS非训练工程一次重验授权，仅修正读取路径，s5-owner-seal-correct-path-20261017T070306Z.stdout于UTC07:03:07.362954Z成功读回。plan/ready/release/原request与已接纳三作业、唯一PID/原截止一致，验收索引s5-final-admission-receipts.json；不改科学代码或门禁。


### I-2026-10-17-verify-stdout-identity：verify stdout缺少独立run/manifest身份（OPEN，非追溯阻塞）

来源：Codex读取s5-progress-20261017T105949Z.stdout，以及Kiro decision_log“晚间”§1。s2024与s2025的两个独立stdout路径字节相同（SHA48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67），都仅含status/replayed_epochs/best_epoch/epochs，没有run_id或manifest SHA；因两者best/stop均14/17，内容相同合理。当前验收依靠各自claimed verify argv里的manifest path/SHA、派生job、run与terminal身份独立核对，不追溯否定或重跑旧verify。今后新入口verify输出应新增run_id与manifest_sha256并作真实绑定测试；本轮不修改已冻结入口/旧产物。旧G6 fixture input hash mismatch继续OPEN（原本地完整suite缺口，真实服务器preflight已独立通过）；不能把其关闭或声称全库全绿。关闭条件分别为新入口身份字段验证证据、旧fixture合同问题独立修复与回归，尚未执行。


### I-2026-10-17-g6-fixture-input-hash：旧G6本地fixture合同不匹配（OPEN）

来源：docs/superpowers/reviews/2026-10-17-coll-sid-a1-code-review.md与原实现验收记录（本日志既有完整suite未完成条目）；Kiro晚间§1明确保留OPEN。旧G6 fixture报input hash mismatch，本轮没有修改fixture/冻结loader、没有复跑完整suite或拿synthetic替代真实输入验收；新SID A1和前缀补证各自真实preflight已通过，只证明该正式路径不受此旧fixture问题阻塞，不代表旧缺口关闭。后续单独定位fixture生成的真实合同身份并fresh RED/GREEN/回归前保持OPEN，不为关issue放宽schema/hash门禁。


## I-2026-10-17-f7-framework-path — 用户画像真实preflight依赖源码路径误判

来源main05/f7-real-pf-accept-20261017T123840Z.stdout；原run p-profile-real-preflight-v1保留failed，stdout空、stderr SHA12af0229cfa2d01f77a9ca8b21d692cb79cf575f2024cf057baaf417daebe0a0，实际读取UTC2026-10-17T12:38:29.816079+00:00，外层stdout SHAddd371efc8332e6b92fa308dceeb7ff8bc1975863fab980c7947b835389c13a6。PyTorch2.11的deterministic初始化通过linecache读已安装_inductor/test_operators.py，被运行期数据test边界误挡；未执行模型检查、未产科学结果。按AGENTS§4常设授权使用一次工程重验：精确识别已安装的单一依赖源码，仍拒绝同名外部文件、test数据、symlink与旧checkpoint；科学模型/配置/输入/seed/检查项不变，复用已验收fit。fresh复现RED→修复后63相关回归PASS，原M1 warning保留，源码e96b117已扫描/push；新目录部署后仅一次retry1，再失败停回Kiro。禁止覆盖旧部署、旧run或重拟合。状态：唯一retry1已succeeded、六配置原检查全部通过（experiment_log/main05 f7-real-pf-retry-accept-20261017T124530Z.stdout），此路径误判已关闭。


## I-2026-10-17-f7-cuda-fixture-query — 用户画像CUDA工程夹具queries接合遗漏

p-profile-cuda-v1在synthetic fullshape全目录评估调用前NameError，旧失败/输出完整保留、无真实训练/科学结果。首次CUDA stdout为空、原stderr SHAd7d01055b0aaf9f3e6155a87da081ea97121a931ed7eb53269a3003edb34e13d；来源main05/f7-cuda-accept-20261017T125345Z.stdout，外层stdout SHA2cb5436741b573bdd7febd2710452b10232534727ac06f1425fe48960fd936f9，实际读取UTC2026-10-17T12:52:23.778321+00:00。根因：前置PAD夹具提取后漏接data[queries]，原局部测试未执行全路径。补完整三步/全目录/g0 CPU CUDA-shim复现，先精准NameError RED，再一行接回变量，65相关回归GREEN；shim不冒充GPU验收。按AGENTS§4使用此CUDA阶段唯一工程重验，旧v1与所有synthetic产物保留，新固定SHA/目录/同shape/检查/种子重新PF绑定，再唯一CUDA retry1；科学模型/输入/六配置/fit不变。retry1再失败停回Kiro，禁止重投科学训练。状态：唯一CUDA retry1已succeeded/cuda_engineering_verified、两臂新产物原始verified；原失败保留，此工程问题关闭。


## I-2026-10-17-f7-postclose-diagnostic — 近零净hits差的收口后只读诊断

来源：Kiro/Opus decision_log“2026-10-17 深夜”§4提出异常与三种未验证解释；Codex按授权执行一次原归档诊断。F7两臂不转正/收口与P anchor=M1-mean保持，诊断不扩大名额/seed/字段。状态：**OPEN，观察已完成，机制与收口措辞待Kiro复查**；不是新增训练失败。

事实依据与数值唯一主账：experiment_log“用户画像收口后一次只读排序与档位诊断”；完整报告cache/codex-morning-execution-20261017/f7-postclose-diagnostic-accepted.json。

1. F7与mean的top50集合更接近bi/mean参照，但没有任何完全相同的集合。平均约四分之三候选重合，按均值计仍有约四分之一候选发生替换；目前不支持直接写“几乎不变”。跨组seed集合不同，仅描述，不作新的显著性检验。
2. 在可观察的discordant命中侧，只有约三分之一target在45–50名，其余多数更靠前；不能将所有翻转归于第50名附近。未命中侧只存top50，精确51–55或更后名次**UNKNOWN**，不填51、不扩深度，所以不能回答两侧完整45–55的集中程度。
3. validation三个字段档位均非常量；身份/顺序、sidecar/quantile/生产源码SHA与原运行绑定一致。未见足以证实档位全为常量或归档错位的证据；但这不证明学习到的p向量非零，也不能排除validation贡献极小等路径问题。共享初始化并不要求训练轨迹相等，完整模型预测变化不能全部因果归给p。
4. 当前只读观察不足以区分裁决中的(a)/(b)/(c)。建议收口措辞保留“未检出差异”，补量化排序分布，不自动加“画像通道对排序影响很小”的确定性机制结论；请Kiro复查。没有据此自行宣布本批无效或恢复实验。
5. 原裁决1e-4量级是待核的启发式异常提示，本诊断未验证其联合独立假设；多比较共享query/部分对照，不能把该提示当作本报告新增的正式p值或门禁。
6. 本次未读任何checkpoint/test、未导出embedding/用户向量/新预测。若后续确需F7 checkpoint或向量范数测量，必须Eddy另批；不把“verify可验自身新产物”的既有授权外推到诊断。旧G6 fixture/历史stdout身份问题仍OPEN，不随本次关闭。

正式诊断p-profile-ranking-diagnostic-v1、preflight均succeeded/unique claimed/no rejected；源码494351e，独立review APPROVED，元数据/物理report与stdout SHA一致，未重跑。具体来源收据/SHA/实际UTC均见上述experiment_log节。

## I-2026-10-18-push-scan-literals：推送前扫描在待推送历史中命中公司名字面量与本机路径字面量（执行对话 Sonnet，2026-10-18，处置中）
- 来源：2026-10-18 推送前扫描（AGENTS §3 第 3 条），范围 `2c2fc80..HEAD`，73 个提交、69 个文件（仅 `.md` 与 `.py`）。
- 命中：4 个提交（`c380b27`、`3cedd1d`、`64ff6a8`、`d024429`）、4 个文档文件（`handoff.md`、`docs/issue_log.md`、`docs/decision_log.md`）里出现公司名字面量与本机路径字面量。多数是记录旧卫生事故时写进句子的自指引文；`handoff.md` 的两行是真实本机路径字面量，HEAD 版已重写、不在 HEAD 树内。
- 其余检查项（数据 / checkpoint / embedding / 索引 / trie / 完整 SID 映射 / 逐 query 预测 / 密钥 / token / 私钥 / 签名 URL / IP / 端口 / 服务器连接信息 / > 1 MB 文件 / 超长行）0 命中。
- 处置（Kiro 裁决见 decision_log「2026-10-18：推送前扫描命中的处理」）：不直接推送；HEAD 树内 6 处引文改为占位符（映射在 gitignored 的 `docs/local_paths.md`）；另建压平公开分支只放清理后的 HEAD 树，对整体 diff 重扫 0 命中才推送；本地过程分支原样保留，不改写历史。
- 范围外：I-2026-10-15-i（已公开历史里的既有命中如何处理）未动，等 Eddy 单独决定。
- 写作规则：今后记录此类事故只写「公司名字面量」「本机路径字面量」，不复现原字面量。
