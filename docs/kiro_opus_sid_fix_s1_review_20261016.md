密级：公开

# 给 Kiro / Opus：SID-FIX S0 复查、S1 冻结与条件训练接续

你是 VK Generative Retrieval 项目的顾问和复查者，Codex 是唯一实现、修改、服务器和作业执行者。
请在实际 Opus 下复查。不要开启你自己的实现/训练 workflow，也不要重启、恢复或重复提交现有作业。

Eddy 的要求：尽快完成必要裁决，让下一批准备和训练接上；不因空档跳过冻结、真实 preflight、verify 或资源门禁。
本提示里的“执行预案”是请你裁决的方案，尚不是 S1 已获执行授权的声明。请一次写清最终设计、允许执行阶段和条件接续边界，减少之后逐步等待。

一、先读这些文件，以文件与原始收据为准

1. AGENTS.md、STATE.md 当前快照及“待 Kiro 复查 / 待裁决”表、handoff.md 末节。
2. docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md 全文。
3. docs/decision_log.md 的“2026-10-16 午间”“傍晚：SID-FIX 卡登记”“傍晚（续）：S1 的 C2 读取 checkpoint 获 Eddy 授权”。
4. docs/superpowers/specs/2026-10-16-sid-fix-s1-freeze-draft.md，重点 §3–§6、§9。
5. docs/experiment_log.md 的“2026-10-16 SID-FIX S0 最终诊断”及“Markdown报告实际读回”，包括完整 A–D、B三seed均值、C的两个明确对照。
6. docs/issue_log.md：sidfix-legacy-verify、sidfix-lift-readout、sidfix-control-label、sidfix-s1-beam。
7. .agents/tasks/gtok02-design.md §3.2、§3.5、§3.8；docs/schedule.md §4、§11。

不要将旧状态表里的 running 当作当前运行状态。本提示没有新的实时资源快照；Codex 获准执行后先只读核队列和资源。

二、S0 已完成的事实与收据

两批诊断均 succeeded/exit0、唯一 claimed、无 pending/rejected/.reason：
- sid-fix-s0-first-v1：第一批33个配置-seed归档，29个直接确认原始 verified，4个旧归档未验收。
- sid-fix-s0-a1b1024-v1：A1+B1024 与 A1+T3 各三seed，共6个直接 verified 归档；其中3个A1+T3参照与第一批重复。

A–D 已入 experiment_log，数字、query/prediction/manifest/verify/input SHA 均有收据；不是只看 supervisor exit0。
S0没有读test、checkpoint、内容向量或完整embedding，没有训练新模型。原pending漏schema已获Eddy单独批准，仅补schema、保留原字节/SHA，未重投。

最终结果收据：
- main05/sidfix-s0-final-results-20261016T095052Z.stdout
- SHA256：055c781722a33e7b9297fa382f3611bdc783ff0e7843d5ef7e74ac48be628331
- 实际读取UTC：2026-10-16T09:50:53.695156+00:00（北京时间17:50）。
- 第一批result SHA：a2e93d69bfb79af2c94e344aa5368e7604825777acc3ed6f2f6cde37c5db1924
- 补批result SHA：ce587ba5513109652945bf656eba5fcef156f7471daadba79b9ee50be10beb41

两份实际report.md也已读回，A–D章节齐全，18:11北京时间补证：
- main05/sidfix-s0-report-artifacts-retry1-20261016T101107Z.stdout
- SHA256：bee0fd2b10718a3f8f5d7bb41fabf85c2e9faf46fef11ef71d4c80e3e3ce91f2

私有本机证据目录 cache/codex-sid-fix-20261016/：
- first-result.canonical.json、a1b1024-result.canonical.json
- final-acceptance.json、tables-acceptance.json、report-acceptance.json
- receipt-index-delivery.json、delivery-manifest.json
不要把完整映射、预测、向量或这些大体量原始产物加入Git。

三、S0 结果摘要，请裁决措辞，不改看数前冻结的规则

A. 结构：
- catalog均120,415；全部完整SID碰撞为0。
- B16/random：4096个语义叶，平均29.3982个视频，最大叶30。
- T3：120415叶，单item；第三位只使用2/256个码。
- B1024：120415叶，单item；第二/三位使用118/1024、1/1024个码。
- B512也单item；B64/B128也单item；T1r为117105叶、平均1.0283、最大17；B256D2为65536叶、最大2。
- S0按“占用父节点平均孩子数>1”计的有效层数：B512/B1024/B256D2为2，其余为3。这是分叉结构指标，不自动等于有语义的聚类层数，不据此改GTOK02 Q1。

B. 前缀准确率：
in-catalog分母1897，OOV623单列排除。以下是三seed均值，前1/2/3位比例，六位小数展示：
配置       top1前缀                     top50前缀
T3         .051133/.007029/.007029       .477948/.144966/.141100
B1024      .027939/.007556/.007556       .319979/.149007/.149007
A1+T3      .046389/.008962/.008962       .525918/.167633/.162889
A1+B1024   .017747/.008083/.008083       .309787/.149886/.149886
逐seed、其它配置、完整SID与item命中分开计算的结果见主账。不同叉数对应不同前缀粒度，请勿把前缀比例差直接当作语义机制优劣。

C. 热度档沿原G4，不重估边界：
train-like nearest-rank median=1、p90=11。
OOV先单列；count=0为unsupported；0<count≤1为low；1<count≤11为mid；count>11为head。
query数：head963、mid548、low155、unsupported231、OOV623。
train-slices SHA：
6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26。

按冻结§2.3，只用core03独立补seed2024/2025：
B1024−T3：head +5、mid +11、low +2、unsupported +1，合计+19；
头部份额5/19=26.3158%<50%。
这只机械落“正增益、头部不到50%”分支。Codex没有自行下“支持语义变好”的机制结论，请你裁决报告措辞，保留50%无更强依据、只描述的限制。s42不参与该分支。

A1确认三seed均可用：
seed42：313−289=+24；
seed2024：330−328=+2；
seed2025：210−310=−100。
合计−74，均值Δ=−24.6667；按热度差分head−43、mid−33、low+2、unsupported0。
机械为“无可分解的正增益”，不能计算有意义的正增益头部份额。
原换G anchor条件：三train/verify通过满足，但均值Δ≥6、至少2个seedΔ≥6均未满足；当前G仍A1+T3。请复查seed散布和结论，不从core03赢家外推A1。

服务器JSON的C_delta_vs_same_seed_T3字段对A1行实际用A1T3；主账已明确重算对core03 T3，并另列同规模A1T3差分，别混淆两种对照。原run产物未改。

四、D指标口径与必须复查的结构性限制

实际方法：
- 共现对：第0–24周同用户完整观看序列中相邻的不同item，不用重复/截断的query history拼接。
- 本批共现对3,630,957个，相邻同item剔除数0。
- 随机：相同catalog升序，Python Random(0)，独立有放回抽两个端点，允许self-pair；数量也是3,630,957。
- lift_k=共现前缀匹配比例/随机匹配比例；随机分母0才UNKNOWN，不补值。
- B256D2的k3按实际存储的常零padding比较，A的有效层数不算padding。

lift1 / lift2 / lift3：
B16       2.213963 / 4.946381 / 10.700688
random    1.002929 / 1.015559 / 1.007795
T3        5.843396 / 16.880000 / 0
T1r       6.237623 / 16.983871 / 4.678571
B32       2.767892 / 8.169374 / 18.631579
B64       3.614084 / 12.926230 / 0
B128      4.653191 / 16.502283 / 0
B512      7.431601 / 0 / 0
B1024     9.448699 / 0 / 0
B256D2    5.843396 / 16.880000 / 16.880000

关键限制：
T3/B1024等的语义叶已经单item。共现明确排相同item，完整语义前缀匹配必为0；随机含self-pair，所以这些位置仍有26个随机匹配，lift显示0。
不能把这种结构零直接读成“没有协同”，也不能用它否定S1。

随机含self的技术定义在看数前已经写进工具doc/tests，但原冻结卡没有明确随机是否也应排self，请裁决：
1. 原口径是否符合冻结意图，可以保留并附上述限制？
2. 若需distinct-item随机基线，是否另登记补充诊断、保留原结果？Codex不会看数后静默改D定义。
3. 是否可明确D只描述、不作S1训练门槛，口径补证与S1工程独立推进？若必须阻塞，请写明阻塞的具体步骤和理由。

五、四个旧归档的验收缺口

以下train均succeeded，但未找到标准-verify/-v1-verify身份的原始verified：
- g-hist-v2-l50-s2025
- g-g4-rand-s42
- g-g4-rand-s2024
- g-g4-rand-s2025

这不等于断言其它命名下绝没有verify；只是本轮标准收据未找到。
S0没有读这四项预测，对应B/C及不完整三seed均值均UNKNOWN，不补零、不引用旧“通过”声明代替原始证据。random的A/D只依映射和训练期事件，仍可计算。
T3主对照三个seed均直接verified，B1024独立补seed及A1确认也已verified，上面的主差分不依赖这四个缺口。

请裁决：
- 是否让Codex另做纯只读旧收据/别名核查？
- 若仍找不到，是否保留UNKNOWN、仅阻历史报告完整性，不作为S1新门槛？
- 若需要新verify而须读旧checkpoint，必须另请Eddy授权；不能借C2的单checkpoint许可补这些verify。
不要为了补历史收据重启旧driver、重投训练或覆盖旧目录。

六、S1请一次裁清并冻结的唯一方案

当前只DRAFT，尚未导出、拟合或训练。已批名额仍仅C1′、C2两个配置。
草稿§6曾误写beam=50，Codex已按T3三个已验收manifest全部beam=100纠正并记issue；卡仍DRAFT。这是纠正抄写错误，不是新增调参。
请以T3实际config逐字段继承，正式beam=100，其它超参不漂移。

C1′提案：
- SGNS，两张表，只导出center/input表。
- 维度64，对称固定window5，不跨用户，不随机缩窗。
- 窗口内不同item有向正对，同item跳过，合法重复事件保留，不按pair去重。
- 每正对5 negatives；train事件频次^0.75分布；拒绝center和本对positive，负样本彼此可重复。
- 固定5完整epochs、seed42；无观看比例/距离权重、无subsampling、无early stopping。
- SGD，lr0.025、batch4096，无momentum/weight decay；初始化、事件顺序、连续RNG按草稿固定。
- 导出float32 center向量逐行L2 normalize；无合法正对者拟用同源content单位向量，单列fallback mask/count；不把随机初始化当协同学习。
- NaN/Inf、有正对却导出零向量STOP，不能用fallback掩盖。
请批准或在任何S1结果之前一次修改这些选择；之后不调参。

C2提案：
- 永久固定M1-mean seed42 best；以后P anchor改变也不换，不用组合模型。
- Eddy已授权weights_only=True只读这一checkpoint，只导出120415视频的64维物品塔输出，supported mask原样；不导出用户向量、不读其它旧checkpoint，服务器保存不进Git，checkpoint/脚本/导出文件均SHA256。
- 冻结后先凭已有manifest锁定真正best路径与SHA，再读，不从名字猜。
- 128维块顺序固定[P,content]：P、content分别单位化后拼接，再整体单位化；P有限零/unsupported时回退[0,单位content]。非有限不fallback。
- supported零向量单列；supported/unsupported内容块幅度不同的限制写进报告。
请裁决此归一化与回退合同。

两项SID：
BHKM branching256×depth3、fit seed42、max_iter10；只换输入；新64/128维合同显式绑定原CONTENT_V1来源和变换后array digest。
H1–H6/B1沿既有门禁，不冒用旧content digest、不绕H5。请明确H5“原内容来源核验+新输入身份绑定”的合法合同；如确需改已登记门禁，列给Eddy单独批准，不能把本提示当授权。
每个输入拟合/导出、SID fit/fit-verify、G train/verify的验收职责须写清，不能只看exit0。

G筛选与判读沿已登记规则：
- 两配置各先core03 seed42，只有train succeeded且对应verify直接verified后读数。
- s42≥T3同seed254+6，即hits≥260，才补2024/2025；259不补，不挪名额给第三配置。
- 正式UP/DOWN/FLAT只用2024/2025，对T3的274/275。
- UP=两Δ均≥+6；DOWN=两Δ均≤−6；其它FLAT，分歧单列；缺seed只按原tentative规则，不写确定胜负。
- s42/三seed均值只并列描述，不混入正式主判。
- 每新SID重算S0 A–D；D只描述，不加阈值。S1不自动换G anchor。

七、请批准或明确修改以下“冻结后自动接续”执行预案

目的：卡一旦冻结，Codex能在已有名额与门禁内自主推进，不为每个普通步骤再等Eddy/Kiro。

阶段A：允许立即实施工程
- Codex隔离worktree，fresh RED→GREEN，实现SGNS、C2精确导出、新SID合同/loader、config、fit/train verify、S0诊断及调度脚本。
- 只做一次完整独立code review；普通不改设计的修复回归后继续；需改科学设计的HIGH/MEDIUM回Kiro。
- 科学代码/运维/driver全部Git版本管理；扫描、push工作分支、固定SHA新目录部署、逐文件SHA。
- 先只读核当前队列/原run_id/资源，不重启M2/G/S0，不重复提交。
- 预先冻结两配置各42/2024/2025共6份G配置与所有verify模板；SID拟合完成后再封存实际bundle输入SHA。不能等s42入选才开始准备补seed代码。
- 两配置独立准备：先ready的先进入自己的下一阶段，不人为等另一项或M2完整报告/组合转正；前提是不绕过真实数据/身份/泄漏问题。

阶段B：允许在各自前置全过后进行拟合与训练
- C1′严格0–24周事件拟合和验收；C2按单一已授权M1 checkpoint导出及审计。
- 各自BHKM fit→fit-verify直接verified、必要replay一致。
- 对各配置实际三seed做服务器真实输入engineering=False只读preflight；不能只用synthetic证明loader可用。
- CUDA工程预演、真实shape显存实测、正式dry-run；显存=实测×1.15向上取整256，不直接照抄旧3072。
- 准入沿可用约8.19核、保留≥20%余量；资源不足由scheduler排队，不强插、不伪称claimed。
- 全过后允许Codex提交该配置s42及服务器侧verify接续，入队前告知Eddy，不再为已批名额重复请准。
- queue JSON必须有正确schema；提交后独立只读确认claimed且无rejected/.reason，written不是成功。

阶段C：有限服务器驱动自动接续
- driver先Git/push/固定SHA部署，服务器detached、PPID1、轮询≥900秒、无模型调用、固定48h截止，不建app timer。
- 成功train自动补唯一verify，直接读verified并核job/manifest/config/commit/data/prediction绑定。
- s42 verified且hits≥260：自动把同配置的2024/2025及对应verify接上；不再等待模型醒来决定。
- 不满足260：不补seed；不改超参、不挑seed、不借空档加第三配置。
- 一项失败只停该配置后续并写ALERT，独立合规的另一项继续；共同数据/合同身份失效则停相关推进回Kiro。
- rejected读.reason不重投；失败attempt不换run_id绕过、不重跑、不覆盖；NaN/Inf、设计偏离立即回报。
- 终态产机器可读汇总：每个fit/train/verify身份、状态、输入/配置/commit hash、best/stop epoch、R@50/N@50/item hits、筛选/补seed动作、S0 A–D、异常及UNKNOWN。
- Codex维护STATE/schedule/experiment_log与待Kiro表；结果解释/anchor仍由你裁决。

S2/S5仅保留原条件路线。即使S1某项UP，也不自动实现/训练LETTER或A1确认；仍须下一卡冻结。G5/bi/F7不混进这批。
不承诺无限无空档；本预案只覆盖固定两配置、最多6个core03训练及各verify，另含必要的输入拟合、导出和审计。

八、请在decision_log明确写出的裁决

请尽量一次给出：
1. S0结果允许使用的描述措辞与限制；不改GTOK02 Q1或既有阈值。
2. D的self-pair/结构零口径怎样处理，是否可与S1工程独立补证。
3. 四旧归档如何补证，是否只阻历史报告，不新增S1训练门槛。
4. S1唯一设计是否通过；SGNS/fallback、C2归一化、H5合同及beam100逐项定稿。
5. 明确允许Codex实现、review后冻结代码、部署、真实preflight、拟合/导出、工程预演、门禁全过后的条件提交，以及s42达到260后driver自动补两seed。
6. 固定driver截止、失败停止与任何仍需Eddy单独批准的事项。

裁决写回docs/decision_log.md，给Codex同步FINAL卡和执行的清晰依据；不由你开新的训练workflow。
设计最多2轮，不开启无界评审循环。若某个未决确实阻塞，请具体指出；其余独立且已授权部分允许Codex继续。

不变边界：不读test week26；不改已登记门禁/阈值；不增加臂/seed/名额；不删除/覆盖旧数据/run/bundle；不付费、销毁实例、force/main push或改历史。C9仍报告项；C2许可不扩为读取其它旧checkpoint。需要Eddy单独批准的事仍单独问。
