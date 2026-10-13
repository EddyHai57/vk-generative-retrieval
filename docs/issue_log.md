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
