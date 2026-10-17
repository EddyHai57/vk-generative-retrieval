# 给 Kiro／Opus：M2 设计裁决、M1 延迟口径及下一批接续

请在实际 Opus 模型下复查。Eddy希望在当前G训练结束前把下一批准备好。Codex仍是唯一实现、服务器与事实源执行者；你负责设计复查和关键裁决，不恢复旧workflow、不重复监控或提交作业。

这次请完成两件事：核对M1延迟报告的口径；对M2给出可执行的唯一设计结论，并明确通过后是否允许Codex立即实现、冻结、满足工程门禁后提交三seed。**当前M2授权只到设计/评审；下面接续方案是待你裁决的提案，不是已获训练授权。**

## 1. 先读事实源，不沿用早期交接状态

1. 顶层与项目AGENTS.md；STATE.md、handoff.md最新节。旧owner条款按Eddy当前直接授权：Codex执行，你复查。
2. docs/decision_log.md「2026-10-16 晨间」§1、§2、§4、§5。
3. docs/superpowers/plans/2026-10-16-m2-multi-interest-draft.md全文，重点§3/§4/§6/§8；这是DRAFT，没有通过、冻结或实现。
4. .agents/tasks/seq-m1-design.md §3.11、§8和附录A；已认可的seq-m1-code-review.md。不要再加一轮M1泄漏review。
5. docs/experiment_log.md「SUCCESSOR」§2（正式延迟）、§6–9（当前G执行）；docs/schedule.md §4/§4a/§9。
6. cache/codex-successor-engineering-20261016/latency-audit.json、receipt-index.json；必要时读其中的原始stdout，不读取test、既有真实checkpoint或完整embedding。

截至2026-10-16 15:07+08的已核快照：当前服务器是G线A1+B1024三个seed，不是M1。三worker均running/evaluate，尚无train终态或verify；driver2924901存活/PPID1、900秒、固定48小时。此后进度由Codex据新收据核对，不把此快照当实时状态，不据当前一轮query进度估计总完成时间。

## 2. M1延迟报告：请只核口径，不新增数值门槛

M1十五训练与十五verify已验收。你已认可mean有效、独立泄漏review和原统计限制。按晨间§1.4，看延迟数字前已预注册：报告符合口径并由你核对后，P anchor切M1-mean；此前仍F4。FEAT组合保持原F4对照，不追溯更换。请核对后把执行结论写进decision_log，而不是重新按延迟快慢选门槛。

正式作业p-seqm1-latency-report-v1已succeeded。以下数字引用experiment_log SUCCESSOR§2，单位ms/query；anchor=F4，mean=M1-mean。

| device | model | mode | mean | p50 | p95 | p99 |
|---|---|---|---:|---:|---:|---:|
| CPU单线程 | F4 | offline512 | 0.024146869 | 0.023963217 | 0.025876217 | 0.027719998 |
| CPU单线程 | M1-mean | offline512 | 0.412940403 | 0.413399336 | 0.423254725 | 0.428272799 |
| CPU单线程 | F4 | online1 | 0.592868637 | 0.593383098 | 0.604384113 | 0.613934128 |
| CPU单线程 | M1-mean | online1 | 1.744423641 | 1.761041116 | 1.853301190 | 1.875081100 |
| A6000 | F4 | offline512 | 0.004220154 | 0.004222838 | 0.004317312 | 0.004508525 |
| A6000 | M1-mean | offline512 | 0.012925273 | 0.012898867 | 0.013242521 | 0.014133289 |
| A6000 | F4 | online1 | 2.117529798 | 2.092363080 | 2.229033969 | 2.295193961 |
| A6000 | M1-mean | online1 | 5.853951691 | 5.843315972 | 5.962647032 | 6.087996997 |

方法及限制必须一起核：

- 同机器/同脚本，只计model.users()；CPU单线程后A6000顺序执行。各模型各模式50次预热、1000次计时，CUDA前后同步，控制与M1输入SHA相同。
- 来自2520条validation真实历史长度，L50共1580条；最大余数法分配计时长度并固定seed打乱，不再用均匀长度近似。离线每次batch512，耗时除512；在线batch1按长度裁剪。
- 另做完整2520长度的单次导出遍历，五batch为512/512/512/512/472，独立报告；不把五次冒充1000次。
- 特征取值合成、权重初始化；没有读训练checkpoint。不含数据生成、H2D、特征取数、item tower、ANN或网络，无KV缓存，不能称真实服务端到端延迟。真实特征取值引起的cache行为未测。
- 已从全部raw samples复算mean/nearest-rank分位数，核1000/50/输入一致性，8行均通过。GPU batch1慢于CPU不构成完整服务性能结论。

收据：main05/codex-latency-result-20261016T132000.stdout；实际读取UTC2026-10-16T05:11:16.233790+00:00；stdout SHA256=1a40d6226dbadb5804e6d3782e5ef552b76c5e4a93840fc7f8d1446a9253261e。服务器该run/out/report.json SHA256=249f55574045d46c3087d88760034b187a77f98e1f3763ec0d401b34493624db。bench commit=b089b8f2da3b72d115f3df743f1e5461392270f6；模型src/config不变。

g0读法沿原裁决：分支内部贡献不等于净增益；m与分支共同适应，净提升只能对同seed主对照计算。这里不重开M1结论。

## 3. M2固定范围与必须裁决的设计项

拟仅一个配置m2-sa-k4-causal，seed42/2024/2025，各一次训练+对应verify，合计3+3；K=4不是四组实验。不扫描K、warmup、温度或mask，不追加bi或诊断训练臂。

沿M1-mean因果骨干、相同输入/数据/训练预算，从共同初始化重新训练，不加载M1 checkpoint。主对照固定M1-mean同seed，F4仅附报，不能换成探索性bi或当前FEAT组合。四兴趣残差共享零初始化g；serve对各物品取四头cosine最大值，最终仍50个唯一物品。详细参数、oracle、泄漏不变量和§8统计见设计卡。

请逐项写“采纳/给定唯一替代/暂缓”，不要留给执行者看结果选择：

1. **N10梯度饥饿与单变量解释**：草稿提议固定前100个optimizer steps用detach目标软权重的逐头N1加权loss，沿温度0.10；step101起固定hard assignment。这改变训练起步/loss路由，不能只写“仅改读出、训练完全相同”。请裁决唯一方案和必要限制；不增加探索臂选方案。
2. **守卫与停止条件**：g=0时分支梯度为0是预期，g更新后才检查每头梯度。草稿提议首次hard100步（101–200）每头有选择覆盖；请核它是原N10的操作化还是新增门禁。若涉及改预注册门禁/阈值，标明需Eddy批准，不能自行放宽或另加性能门槛。明确失败后停下、保留证据、不重试；不能靠延长warmup或换seed救回。
3. **四路检索公平性**：主指标full-catalog max exact top50。每头top50→去重/重排→最终50的最多200中间项，只有相同全序tie-break下才等价；需oracle。最终槽位/训练预算相同，但计算与中间候选预算不同，必须公开。请明确是否采纳；ANN不能静默替代exact主评估，也不追加ANN调参臂。
4. **多头branch_ratio与g0诊断**：草稿提议query内max_j||g r_j||/||m||再跨query median，另报每头ratio/获胜计数/g0；沿0.05/3原阈值，但max过线不能证明四头都有效。请固定口径并核是否属于门禁变更，不移植pure/S1归因阈值、不新增结果门槛。
5. **冻结前实现细节**：给定W1/b1初始化约定、共享参数抽取顺序、零起点等价/梯度oracle及target翻转/因果/padding/运行时test拦截要求。沿§8的主比较、逐seed配对及pooled重复query限制；不根据M2结果补设计。

## 4. 下一批准备与接续预案（请明确授权层级）

Eddy要求尽量在当前G训练结束前就绪。**M2本身不依赖这批G结果**；设计通过并获得实现授权后，Codex立即做本地实现，不等G收口。请明确下面哪些阶段获准，尤其“设计通过”是否同时允许实现/冻结/条件提交；不要让模糊状态造成新的空档。

1. 裁决写入decision_log，设计卡补唯一方案；设计评审最多2轮，余项由你收口，不无限循环。
2. 若获准：Codex隔离工作树，以fresh RED→GREEN实现模型/runner/三seed配置/verify与诊断/运维；原冻结M1代码保持，固定数据/主对照和统计，不重跑对照挑数。完整可验收产物做一次独立review，不为每个内部修复重复review。
3. 公开扫描→工作分支push→固定SHA新目录部署→逐文件SHA。真实输入只读preflight覆盖三配置、engineering=False/test关闭/数据与source绑定；CUDA预演核soft→hard、完整batch512×L50×K4、optimizer状态、exact max/评估最坏shape、重放及g0。没有新实测前VRAM/墙钟UNKNOWN，不照抄M1声明。继承B1024教训：dry须检查最终启动记录大小与supervisor可读性，不能只看prepare返回0。
4. 显存按实测×1.15向上取整256。当前G三train声明3核、19968MiB VRAM；总预算约8.19核/39312MiB且保留余量，实时资源再核。M2全前置通过且获条件训练授权时，可与G并行入队；不必等三G完成。资源不够则将已READY的固定三seed预排，由scheduler放行，不能侵占余量或伪写claimed。工程测量避开干扰既有作业；比较延迟需避免混跑争用，否则如实标注并不冒充可比结果。
5. 正式三job dry→一次提交→独立ro核唯一claimed且不在rejected/.reason；所有训练配verify。服务器版本化driver PPID1、≥900秒、no LLM、固定48小时；成功train补verify，仅认原始verified+job/manifest绑定，失败/rejected ALERT、不重试、不覆盖，继续其它成功seed验收。不得按seed42效果决定是否跑其余seed。
6. M2完成后只产出结果/诊断/完整延迟，尤其用户塔、四路检索、去重重排及合计，串/并行与exact/ANN范围明确。结果解读/anchor由你在Opus复查；不自动接M3、bi、G-COLL/F7或未冻结G5。

如果设计仍需修改，请一次列出最小阻塞清单，优先解决N10与检索/诊断定义；Codex可先整理接口依赖和测试清单，**未获实现授权不写M2生产代码**。如果G先结束而M2尚未READY，如实报告空档，空闲超过1小时主动告知Eddy，不为填空新增训练、改规则或恢复timer。本预案不承诺必然无空档，也没有创建自动任务。

## 5. 请交回的裁决

请在decision_log新增一节，区分：

- M1延迟口径：通过/具体缺证；如通过，沿原预注册处理P anchor，不追溯FEAT组合。
- M2五项设计取舍的唯一结论、设计review轮次和剩余阻塞；不得把本草案提议直接记成既定事实。
- 权限逐项明确：允许实现？允许冻结？允许全部工程门禁通过后一次性提交三seed及自动verify？是否可与当前G并行？若暂缓，明确停在哪一阶段以及由谁解除。
- 不增加实验臂/seed/名额；不读test、不用U1、不恢复C9门槛、不读旧checkpoint或导出完整embedding；不删/覆盖旧数据、付费、force/main push或改历史。任何触及Eddy保留审批边界的变更单列，不由Kiro或Codex代批。

裁决回到本对话后由Codex执行与维护STATE/schedule/experiment数字及收据；Kiro不另起训练workflow，不形成两个调度owner。
