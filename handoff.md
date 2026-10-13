# KIRO_CONTINUES / CODEX_PRELOAD_ONLY — G3-R2 replacement monitor 恢复入口（2026-10-11）

Eddy最新明确Kiro继续作为G/P两线唯一协调与执行owner；Codex只可只读预加载，不得启动writer、monitor或job，除非Eddy以后明确切换。原monitor PID27709已不存在，退出原因UNKNOWN。只读核验报告在主仓`.agents/tasks/g-monitor-readonly-recovery-check.md`；replacement packet为`cache/g3-monitor-recovery-v1/`，packet SHA `6910de1cd4f9ed46fd039edf0284f393ad311b7e885eaec72c9cd15ff092804b`，validation内immutable launch-member SHA `22773ecaecc4c9cff87e1f4ef54329584a8d52f3f08240b699741fe222f81910`。

Kiro terminal7持有唯一replacement monitor runtime owner PID2836，input SHA `8e801a49e9b0824e3e7e32adce3eaa5f59e84620b9a5b698f993f045e0ba950`，started_at epoch `1791734602.037766`，只读且禁止submit/restart。首receipt transport exit0/no timeout；observation epoch `1791734605.107748`，core01/core02/core03为TERMINAL/RUNNING/RUNNING。core01 terminal complete、worker exit0、elapsed `21241.7523078979s`、terminal SHA `bca51b1a2acd54231d3cf886ec59affae3204bce18cf08432dbb892ab76e3b6d`、owned groups closed；core02/core03保持原start SHA `3780c19d0a67082af28869e3e567f2b00a7b79d6649e851058cb5fabe401c154`/`5c84b86141ef7227d91c27f28353a1aa8f3b0d44a019e024a5a0582154a0f269`且owner live。首观察之后没有state-change receipt，不推测或改写状态。

下一步只由PID2836按退避接收core02/core03 terminal，并且只有state-change receipt才更新；不启动重复monitor，不追加SSH，不submit/restart/resume。旧失败与三个v2 attempt的spent、caps均不重置；三个v2未全部terminal前不收口，四seed42配置未齐前不select/winner seeds/G4/test。

# G3/G4当前恢复入口（2026-10-11）

首slot终态与指标读取完成；指标只看`docs/experiment_log.md`，读取证据`cache/g3-metrics-read-v1/`。原104h/36GiB提案已拒绝/superseded，超过48 GPU-hours hard ceiling；全部剩余G slot暂停。下一步在P4F完成后，基于已读指标提出缩减分阶段预算，不启动。

# Claude Code已接管：G3首slot终态入口（2026-10-11）

02:48已审同job observer返回`TERMINAL/complete`且fresh owned groups closed；formal/replay各14 epochs，train/best/verify actual exit0，consumer seal closed。0resume/1attempt，原deadline/caps/spent不变，无重提或timer恢复。train/best/verification/invocation/terminal五个真实artifact SHA已独立重算吻合，摘要见`cache/g3-formal-A-actual-v1/status-024-takeover-summary.json`。下一只冻结首slot指标/选择证据，并基于约10.21h实测wall汇总剩余5core与G4有限预算；未获批准前不启动。

2026-10-10 19:07:05：Eddy要求当前进度，单次已验收progress只读status-022返回RUNNING/train已进入/workeridentity alive。epoch/update/currentactive/replay unknown，未见best/verify。此为单次状态询问，不恢复heartbeat或其他暂停工作；原预算与attempt保持。

# G 对话暂停恢复入口（2026-10-10）

Eddy经main最新明确暂停所有对话，停止模型推进/SSH/诊断/测试/review/新实验。原G heartbeat g3-core-00-seed42已实际设PAUSED；等待明确恢复。原远端job/产物/预算/唯一owner保持，不kill/restart/delete，对话暂停不代表训练停止。暂停前最后实际status-021：RUNNING、latest_stage_entered=train、worker_identity_live=true；epoch/update/currentactive/replay进度unknown，best/verify missing。进度读取已在暂停指令到达前完成；此前新progress heartbeat升级脚本因本地配置文件读取失败未更新automation，无远端动作。恢复后先核原job，不重提，原48h/20GiB/0resume/1attempt预算不reset。

G3最小samejob progress observer本地READY_FOR_SOL_MAIN：cache/g3-progress-observer-v1/review.json SHAd292d004a161e73007f9c7a26e1e20088b6863260979aa7b5f151a9c623e495a。main已授权本地候选；source确定8固定文件，不glob/全tree/checkpoint，safe FD/hash/bytes/count/time；原observer-v1所有门保持，新增externally pinned stdlib script/input只读输出stage entered/真实progress epoch/workeridentity/mtime。fresh6 synthetic边界GREEN，countsupdates/currentactive未知；pump buffered stderr可能尚未落盘，空/截断不当0或卡住。尚未remote，原5min observer继续，原job/预算不变。

Eddy于2026-10-10直接授权约5分钟只读检查及跑完删除。已创建ACTIVE本G heartbeat：g3-core-00-seed42，名称“G3 core-00-seed42 五分钟状态检查”，每5分钟；仅accepted observer-v1原job，一次/唤醒，普通RUNNING安静，完成/失败/实质异常通知Eddy与main。可信终态、owner不活且fresh_owned_groups_closed后使用automation_update删除id g3-core-00-seed42，保留作业产物/对话；UNKNOWN不当停止。此新指令覆盖本G历史不建automation规则，旧vk/p4d仍PAUSED。

最新按Eddy请求只读核查：2026-10-10 16:26:57（Asia/Shanghai），status-004返回RUNNING，原owner identity仍活着；具体stage/epoch/metrics未知，未取得终态。原作业/预算不变，无重提/resume。收据cache/g3-formal-A-actual-v1/status-004-receipt.json。

# 最新原job RUNNING；原attempt已spent（2026-10-10）

已审readonly observer-v1首次status-003实际RUNNING，原owner alive；当前stage/epoch/metrics unknown。唯一submit/no retry/0resume，原fixed包/服务器bytes/预算不变。下一正常有界退避同jobobserver监控，terminal/失败/资源异常升级main，禁止新submit。原status读取缺陷已由main定点接受只读内存修正，不再用坏status。全部入口见STATE和cache/g3-formal-A-actual-v1/review.json。

# G3监控缺陷已定位，最小只读差异送main（2026-10-10）

status-002仍UNKNOWN。本地固定lifecycle70414bytes第一次262144核pin通过，status326第二次默认65536触发regular_bounded并吞为UNKNOWN。原job存活/阶段/终态仍unknown，禁止重复坏poll/重提/改服务器。最小内存readonly status差异cache/g3-formal-A-status-read-v1/review.json SHA409603d7eec3b7c604291afb4361663845258547952fa429bb99484cce285986已送main定点验收，尚未运行。原30s/60s/64KiB/三pins保持。

# G3 core-00-seed42 实际已提交，初次状态 UNKNOWN（2026-10-10）

Eddy在本G chat直接批准fixed formal-A-v2（875f096ee1855381c9f9b3091cb24f61895add660720f602a8adbee92df8caa2）单次运行。预算48h含全部部署/训练/full validation/replay/收尾，新增20GiB观察STOP可能overshoot，联合20%余量，10200updates/153720decode/0resume/失败不retry，保留持久产物；无其他配置/G4/test/新资源。

按已审确定性方法仅绑定approved与真实600s窗口，approval SHA `4e8ca5c77b788e710c26f4bdb0a96b6d69a138cc320befc32aebf07daff75b0e`，payload SHA `3b1a5d0d11fcd18ed5f9dbe9e9bc68ef62ba42a79ce792eb6202df9cf6281ca9`。唯一submit实际返回SUBMITTED，start SHA `1c430779d3a5933ac2b862286ca39aa9d85773e2903e218f9a0141b4a3dfa022`。因此原attempt已spent，不重submit/重启或改现场代码。

随后已审三pin readonly status-001返回UNKNOWN，transport实际exit0/non-timeout，不是owner退出或训练完成证据。原因和当前训练阶段未知；G仍为唯一监控owner，先本地追溯和原固定status有界核查，必要向main升级最小同job诊断。完整证据入口cache/g3-formal-A-actual-v1/，旧fixed包保持。下一只处理此原job状态，不进入其他task。预算数字和观察记录见docs/experiment_log.md；风险见docs/issue_log.md。

以下历史记录不作为当前pending许可：

# 最新 G3 formal-A-v2已ACCEPT，实际许可pending（2026-10-10）

main原reviewer报告cache/g3-cost-candidate-coordinator-v1/formal-A-v2-review.json：ACCEPT_CANDIDATE_PENDING_HUMAN_SPEND_APPROVAL，唯一Important Closed无新增阻塞。accepted packet仍875f096ee1855381c9f9b3091cb24f61895add660720f602a8adbee92df8caa2，旧包不变。不重做suite/review/preflight/probe；0新server/真实训练，48h/20GiB仍未获本G chat directhuman批准。

确定性materialization无需改代码：accepted transport.build固定六members→只改approval status与真实not_before/expires≤600s→canonical controller.raw/hash→controller.decode核exact pins/spec/caps/livewindow→transport.command(actual approval SHA,固定controller SHA)。原reviewer已fresh运行test_transport的approved drift/caps rejection，方法已可执行/核验。现场root/controller/input/science/caps不变，private runtime由原pinned origin仍只在server绑定。获得直接具体human批准后新actual exclusive证据目录保存完整canonical authorization/envelope/payload/argv/fullpins与actual收据，不把当前pending/测试fixture当实际许可。无B/G4/select/test/newresources/delete/retry/resume；P并行，真实余量不足STOP。

以下为历史恢复记录：

# 最新 G3 status v2 READY_FOR_SAME_REVIEWER（2026-10-10）

main v1 CHANGES_REQUIRED仅1 Important：status执行code先于外部SHA。按原G3范围已完成本地最小修复；新cache/g3-formal-A-candidate-v2/review.json SHA `875f096ee1855381c9f9b3091cb24f61895add660720f602a8adbee92df8caa2`，README/diff/validation为入口。固定stdlib bootstrap先nofollow/regular/size/外部controller SHA，再exec verified bytes，不以文件路径重载执行；只status三pins，同30s deadline/64KiB输出。source44/consumer/helper/lifecycle/server六核心bytes/input/approval/payload及旧v1全部保持；0新server操作/训练，预算pending。下一原reviewer只核此差异，不重跑原30+7query、不新reviewer。后续实际许可仍由Eddy在G对话最后批准。

readonly验包：python3 -B cache/g3-formal-A-candidate-v2/verify_local.py。以下为v1历史基线，不能重启动cost或以旧READY绕过v2差异验收。

# G3 A首slot完整候选恢复入口（2026-10-10）

main派工后的本地实现已完成READY_FOR_ASTRA；不再停在cost/design或IMPLEMENTING。唯一新包cache/g3-formal-A-candidate-v1/review.json SHA `0c40c1e9c79fef77c773d57cee18901b7a6c14d24c3f60a29e89bfe193dfcbe0`，入口README/validation/input/command-template。当前approval.template和payload.template pending，0新server/真实训练；下一main一次fresh差异review，然后Eddy直接批准实际scope。source44/原v4 lifecycle/helper保护不改；实际预算仅见operational-spend方案及experiment_log，账单unknown。

关键实现：专用detached owner+stdin identity latch/external approval SHA；old terminal/good identity/shared headroom prewrite门；context/runtime/生成formal permit全计原clock；train→nested best→原helper full replay→原consumer final seal；独立bounded scanner/实时global guard/实际wait与Linux adoption闭组；全部checkpoint/shard/journal/ledger留存。0resume为更保守候选选择，main须明确验收；worker limits结构1、实际pool0。Linux actual运行未发生，portable controlled tests不能冒充现场。loader TemporaryDirectory原语义保留。

readonly本地验包：python3 -B cache/g3-formal-A-candidate-v1/verify_local.py；source44/pins/templates/parse通过，具体测试见validation。勿再执行原v4检查/submit、旧CPU审计、B/G4/select/test；失联也不重submit。旧事实保留如下历史，不作为当前派工。

## 当前实现恢复入口

main已派原G3首slot方案A本地完整实现；cache/g3-formal-A-candidate-v1/为唯一新WIP，lifecycle.py逐byte复用accepted v4底层，test_safety.py已有8项fresh RED，controller/scanner尚未实施。正式48h/20GiB未批准，0SSH/部署/真实训练。问答不取消本实现任务；不重做旧cost或CPU audit。

# G3 operational spend 恢复（2026-10-10）

先STATE、cache/g3-operational-spend-v1/review.json、decision.md、execution-design.md、estimate.json、validation.json。main范围裁决已取消缺checkpointpin的CPU audit；旧prototype仅保留未完成标注，不再实施/SSH补pin，也不建议新碎probe。

本轮不依赖旧checkpoint，不加载模型：冻结source/config推导tensor参数/FP32+AdamW、cursor/RNG/metadata模板容量，所有实际大小/运行耗时风险仍明确。首slot和六core分别给有限spend上限，超限STOP可失败，非ETA/严格upper，账单rate未知，不代表已批准。

关键最小接线：复用v4 CODE/prepared原pin只读，新增exclusive formal root/owner；旧4096node diskobserver不能直接用于正式大量journals。固定stage train→best-eval(在原train tree)→mandatory fullverify/replay最终封口，封口后不再写pool；需要新相关TDD验证而不是改科学规则。samepool/deadline最多一次安全中断恢复，failedspent不refund、全checkpoint/shard/journal保留；pidbirth/SID/actualwait/闭组/不可延长clock/有界scanner还必须完整实现审查。

下一main一次决定完整最小实现part，之后G自主实施/定点验收，Eddy最后批准固定SHA运行。当前无新server/训练/CPU audit许可，G3未Complete/G4/test关闭；旧actual-v4 source/pilot/failed证据包全保留。
