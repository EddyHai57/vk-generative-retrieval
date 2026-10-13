密级: 公开

# 三对话执行协议

2026-10-10 Eddy批准：两个原Sol保留，各自端到端负责本线，仅更换Astra。此版替代旧“传统Sol代执行全部G作业”的调度规则；历史科学合同、实验总账和已消耗attempt不变。更新协议与接手本身不是恢复实验；当前恢复状态与授权范围只看STATE.md。旧版逐byte保存在history/astra-handoff-2026-10-10/。

## 最新模型与协作要求（2026-10-10）

Eddy明确要求“以后不用astra了…sol就行了”，以减少token消耗。今后协调、实现和独立验收均使用GPT-6.1 Sol/medium，不再调用Astra；旧文中Astra职责由协调Sol承接，历史身份记录不改。两原Sol及已运行作业继续，唯一owner与既有预算不变。当前主对话保持主仓唯一治理writer；实际模型须由应用模型选择器切换，文档更新不代表已经切换。普通进展不唤醒协调，不叠加reviewer/旧suite/重复检查；原独立验收仍仅一次，使用Sol。当前项目内既定预算的常规准入委托继续有效，不逐次请求Eddy重复批准。

## 三方职责

**2026-10-11单对话覆盖条款：** Eddy确认外部代理停止，当前单Codex主对话接管两线，旧长期Sol和automation保持暂停。只允许必要短期研究/一次独立review，不重建三方协调链；上下文管理及新对话接续按context_session_policy.md执行。下文Claude/三方角色只作历史，科学/预算/唯一job owner规则保留。

**Claude Code接管覆盖条款：** Eddy已改为由Claude Code全权代理；当前Codex三对话与automation保持暂停。本文后续Astra/Sol/三方职责、聊天派工与定时兜底仅作历史，不要求Claude重建三代理协调链。以主STATE.md、handoff.md及docs/claude_code_takeover.md为当前接管入口；Claude先核原job与预算再记录owner接收。科学合同、有限预算、失败账本、唯一owner、一次独立验收与安全边界继续适用，常规任务自主推进、不重复索取微审批。

| 角色 | 唯一责任 |
|---|---|
| 协调Sol（承接旧Astra职责） | 完整任务目标、共同科学合同、预算与共享资源准入、真正升级、一次独立验收、两线READY队列；主仓治理唯一writer；不SSH、不实现日常代码、不重复监控。 |
| 生成式Sol | G线一个task内的实现、TDD、诊断、获批部署/执行、作业监控、结果与自有事实源；原生成式worktree唯一writer。 |
| 传统Sol | P线同等端到端职责；原传统worktree唯一writer；不再承担新G作业的普通执行。 |

既有job的owner不会因本文自动切换。恢复后若存在原P1持有的G作业，先核原身份/receipt/消耗，由两方确认接收，再撤原owner监控；未知不能当停止。每个job只有一个启动者与监控者。共享入口/权限不具备时升级一次，不能悄悄回到G→Astra→P反复转交。

## 完整派工与自主权

每条线一次一个tasks.md task。Astra一次给目标、允许文件、输入与配置身份、产物/验收、允许动作、有限预算/attempt/resume、资源范围和stop条件；科学合同/预算未明不猜值。
Sol在范围内自主设计、实现、修复、自测、整理证据。普通进度问答后继续原任务，除非明确暂停或真实STOP。生产代码保持fresh RED→GREEN→受影响回归。已接受且未变的代码/证据不重造，不额外编写通用框架。诊断动作和有限重试应在完整任务brief中预先划定；未获准的新实际attempt不能借本地修复自动启动。
只升级：合同/输入身份变化、预算/attempt增加、跨writer边界、付费/破坏性/资源生命周期动作、资源冲突、权限实质拒绝、同路径无新证据的重复失败。先合并具体缺口与建议，不把普通编码选择升级。

## 资源与运行

沿现有STATE队列表记录job、owner、资源持有/排队、释放证据；不新建平台/数据库/lock服务。无并发安全证据时资源密集作业串行准入，本地工作并行；生成式优先。获批owner就地核runtime/input/root/当前资源，不由聊天idle、SSH断开、TTL过期推定作业已停止或资源已释放。
实际启动前完整执行包与有限预算须一次准入；owner在包络内自主完成preflight/submit/status/收尾，不逐stage等Astra。相同输入参数的现场绑定由已审入口执行；合同、来源、代码、副作用或预算变化须升级。科学与20%余量门保持。
进程安全检查由既有确定性supervisor负责，模型不持续空转。完成/失败/资源异常/需决定才通知。Astra不SSH；每次compact同时看两线，不长等一线阻塞另一READY。

## 失败与重试

- 本地失败：范围内自主修复，不索取微派工。
- 真实失败：spent保留，不退款、不盲重启；只在明确批准的剩余attempt/resume内继续。
- 审批实质拒绝：保存原理由、是否实际开始，升级一次，不换写法绕过。
- 工具明确允许的transient timeout retry：先核副作用与原进程，仅在授权和次数内重试。
- 聊天连接失败不是服务器失败。无新证据重复失败就STOP；不为了凑次数违反原no-retry。
失败receipt至少有stage、脱敏error类别、实际副作用/exit/消耗与unknown字段，不把所有异常吞成通用STOP；不得泄露secret/连接信息/raw/完整预测。

## 一次验收与异步交付

完整交付一个独立review过程：worker不叠加额外reviewer；Astra自行或一个短reviewer审。启动前审未接受差异/绑定/副作用/预算；运行后核真实receipt、指标口径和消耗，不重做整套code review。有blocking一次合并清单，修复后只核差异与相关回归。
READY先保存packet，再主动通知当前Astra一次；去重(task,packet SHA)。通知失败保留准确原因，用获准wait/local兜底，不重做产物、不循环ACK，不以文件暗藏被拒执行指令。摘要只写：task、完整成果/限制、packet路径SHA、验证、job与预算状态、合并blocker、下一步。成员hash/日志留原packet，不复制聊天。
当前Astra身份只查主STATE。两个Sol获Eddy明确授权向当前Astra发项目范围READY/BLOCKED/URGENT及交接确认；不把此授权扩展为对外消息或敏感值传递。

## 持久状态、暂停与交接

各writer的STATE只有一份当前快照；handoff只有恢复入口，历史逐byte归档并保留hash，不追加多层“最新”。主仓只保存两线摘要与指针，指标在原experiment_log。角色协议本文件唯一，worker用绝对路径引用，不维护互相漂移的规则副本。
更换Astra期间旧协调负责收集ACK，新协调只读接手；新协调确认接收且旧协调完成切换后，旧协调停止派工、监控和治理写入。新协调可按明确brief先只读等待，避免两个writer竞争。
暂停后只有Eddy明确恢复才启动实验协调与vk；接手/记住职责不等于恢复，具体实际执行仍须相应预算与准入。恢复时将原vk迁移到新Astra，不建立第二monitor；15分钟仅轻量兜底，无变化不通知、不写状态。
下一次G3与P4F交付后观察：owner切换、范围内微审批、重复review、READY等待、失败可定位性和真实实验进展；token无账单不猜节约比例。


## 2026-10-10 — Eddy授权Astra常规准入决策

Eddy明确表示不再逐次由其批准，Astra认为可行即可。当前两subset项目的常规修复、诊断、完整候选验收和既定预算包络内的新attempt，由Astra做具体准入决策；此前逐attempt均需Eddy最终直接批准的协调要求在此范围内取消。仍须完整scope/固定pins/finitecaps/唯一owner/真实资源门/失败账本，不盲retry或复用spent许可。未授权新付费资源、删除、全量扩展或更高预算；这些另行决定。不得用伪造HUMAN_APPROVED标签代替实际授权链，记录本次human委托及Astra具体批准。若工具自动审批不接受跨对话授权，保留准确原因、不换方式绕过；只在确实无法继续时向Eddy说明，不预先要求每次确认。
