# AGENTS.md — VK Generative Retrieval

密级: 公开

> 本文件只保存稳定行动规范与事实源指针。当前状态见`STATE.md`，跨对话接力见`handoff.md`，阶段计划见`tasks.md`。

## 工作流硬规则（2026-10-13，Eddy确认“按工业界流程来”，最高优先，覆盖下文所有冲突条款）

1. **Git是唯一代码来源**：每个task结束即commit；服务器只运行已push的commit。每个run的manifest记录commit SHA、数据hash、config、seed。服务器上不直接改代码；运维脚本（launcher/admission/observe/driver）也必须进版本管理，不得只放在`cache/`。
2. **部署**：服务器用只读deploy key `git fetch`，`git worktree`检出指定SHA；部署时仍逐文件校验sha256。旧“tar包+stdin上传”只作历史记录。
3. **push边界**：可直接push到工作分支；合并到`main`前做一次review；force push、改写历史、删除远端分支需Eddy单独批准。repo为public：每次push前扫描数据/checkpoint/密钥/IP端口/本机绝对路径/公司名，>1MB文件拒绝。文档中的本机路径一律用占位符（映射见gitignored的`docs/local_paths.md`）。
4. **作业与SSH解耦**：所有作业detached运行（setsid/nohup，日志写run目录）。任何人随时SSH登录、断线、重连不得导致作业停止；连接断开≠作业停止。
5. **gate只管自己**：只检查本作业进程树与GPU/RAM/磁盘余量（≥20%）。不因外部未知进程、sshd认证进程或另一条线的终态而STOP。（现有旧gate代码按此在迁移task E1中改造。）
6. **两线独立**：G与P各自run目录、预算、终态，互不等待；唯一融合点是协同召回。资源够时尽量并行，避免GPU或CPU闲置；新作业前先估算瓶颈（CPU单核/GPU/IO）。
7. **单owner**：Kiro统一调度两线，并是事实源文件唯一writer；subagent只做边界明确的短任务（实现、调查、review），不长期占有作业。
8. **review只放在不可逆节点前**：提交正式作业、push到main/public、销毁或删除数据之前各一次。普通修复自己fresh RED→GREEN→回归后直接继续，不复审。
9. **不可逆操作**：删除数据/checkpoint、销毁实例由Eddy执行或单独点头；先给出逐路径清单与可恢复性。
10. **事实源**：指标只写experiment_log；风险写issue_log；决策写decision_log；STATE只写当前状态。

## 最新直接授权：Kiro / Opus 5.5 接管（2026-10-13，优先）

Eddy明确要求暂停定时检测并让Kiro的Opus 5.5全权接手。Codex Main05仅完成STATE/handoff/交接提示词后停止模型推进、轮询和写入；现有服务器driver/publisher/训练与有限transport保留，不kill/restart/resubmit。Kiro必须先核Main原项目事实源、原job/source/queue/spent/deadline和新收据，记录实际ACK后成为唯一模型执行者；未ACK不得声称接收完成，实际模型选择未核验不得伪称已切换。此条覆盖后文单Codex/旧Kiro停止/仅Sol模型的历史要求，不覆盖科学、数据、test、身份、资源与禁止push等边界。项目automations保持PAUSED，不自行恢复或新增timer。当前快照与具体步骤只见STATE/handoff和docs/kiro_opus55_takeover_prompt.md，不扫描/继承全部历史。

## 上下文提醒与交接（2026-10-11，优先执行）

自动压缩是Codex配置（目标70万）；主动提醒与交接是agent指令。新task/part结束/真实job前核带时间的context计数，长part约20次工具调用无新样本时补查，不持续轮询。50万主动提醒一次，65万整理STATE/handoff准备交接；有效window W较小则分别取min(500000,60%W)、min(650000,80%W)。计数不可用/过期报UNKNOWN，不猜百分比。compact后核事实源，能准确恢复就继续；持续合同/预算/owner混淆或Eddy要求时才安全换对话，不因到70万强制新建。一次只留一个长期执行者，原job/spent不重置。全局默认在`~/.codex/AGENTS.md`，项目细则在`docs/context_session_policy.md`；无变化不刷通知/状态，不扫描全部历史。

## 最新：单Codex主对话（2026-10-11）

Eddy已确认Kiro/外部代理停止，改由当前一个Codex对话接管。只有一个长期主执行者负责两线、治理及作业接收，不再派旧G/P Sol或外部代理、不使用Astra；必要短期研究或一次独立review可用sub-agent，不继承全部历史。本条覆盖后文Claude/Astra/Sol多对话历史要求。作业实时owner与预算需据新证据对账，不因聊天接手宣称服务器已核。上下文提醒、压缩与换对话遵守`docs/context_session_policy.md`；每次只更新当前快照与证据指针，不累积多层“最新”。当前COORD-03只处理制度和配置，未启动实验。

## 最新模型与协作要求（2026-10-10）

**最新接管指令优先：Eddy现要求Claude Code全权代理后续任务。** Codex对话和automation保持暂停；Claude启动并核对原job后接收协调、实现、准入与验收。旧“必须Astra/Sol/三独立对话”及普通步骤等待Eddy确认的要求，在既有subset授权范围内不再强制适用。一次明确完整task后自主执行，不逐步询问；原科学合同、finite caps、spent、独立验收一次、唯一job owner与高风险边界保持。Claude入口见`STATE.md`、`handoff.md`及`docs/claude_code_takeover.md`。本条覆盖后文历史模型/调度要求，不代表Claude已完成接收或Codex恢复。

Eddy明确要求“以后不用astra了…sol就行了”，以减少token消耗。今后协调、实现和独立验收均使用GPT-6.1 Sol/medium，不再调用Astra；旧文中Astra职责由协调Sol承接，历史身份记录不改。两原Sol及已运行作业继续，唯一owner与既有预算不变。当前主对话保持主仓唯一治理writer；实际模型须由应用模型选择器切换，文档更新不代表已经切换。普通进展不唤醒协调，不叠加reviewer/旧suite/重复检查；原独立验收仍仅一次，使用Sol。当前项目内既定预算的常规准入委托继续有效，不逐次请求Eddy重复批准。

## 启动顺序

每个新对话先读：

1. `AGENTS.md`
2. `STATE.md`
3. `handoff.md`
4. `tasks.md`
5. 当前任务涉及的`docs/`文件

文件冲突时停止执行，列出路径与冲突内容，由Eddy裁决。

## 协作风格

- 中文回复，technical terms保留英文。
- 结论和风险先行，例子在抽象解释之前。
- 默认Eddy不熟工程细节；一次只推进一个可消化步骤。
- 不猜数据、schema、label、指标、资源或实验结果。
- 多文件或含糊任务动手前，先用大白话复述目标、范围和验收，等Eddy确认。
- 坏消息、tradeoff、失败实验和不确定性不得隐藏。

## 执行纪律

- 每次只执行`tasks.md`中的一个task ID，不合并跨阶段任务。
- 开始前明确：允许修改文件、输入、输出、验收命令、stop gate。
- 生产代码使用TDD：先fresh RED，再最小GREEN，最后fresh regression。
- 同一文件不得由多个agent并行修改；fan-out任务使用独立branch/worktree。
- 完成前必须运行相关验证；exit code 0不等于目标达成。
- 完整可验收交付完成后进行一次独立fresh-context review；不对每个内部步骤重复加review。
- commit/push按顶部“工作流硬规则”第1、3条执行；不修改Git配置，不创建新remote。
- 删除、覆盖、丢弃改动、计费资源和其他高风险操作必须先列清单并等待确认。

## Agent 调度与训练监控

2026-10-10 Eddy批准端到端分工及仅更换Astra；详细协作规范以`docs/codex_handoff.md`为准：完整任务自主执行、异步READY队列、一次独立验收、15分钟轻量兜底。以下稳定规则与其共同适用。

1. Astra负责范围规划、任务拆分、接口和合同、关键决策、独立验收及交接；主要实现和训练编排交给GPT-6.1 Sol，reasoning effort为medium。Sol在完整task边界内自主实现、修复、自测；Astra仅处理真正升级及最终验收，不逐步审批。
2. 长期实现与训练任务优先使用Eddy可直接查看的独立Sol对话。先查找并复用合适的已有执行对话；确需新建时，Eddy已授权为本项目创建对应执行对话并发送任务提示词。主协调须提供实际对话名称和入口。
3. Subagent用于边界明确的短期工作，如只读探索、独立code review和验证；不得悄悄替代已约定的长期执行对话。每次派工说明执行载体、模型、任务和文件边界。
4. Worker prompt必须自包含，写明task ID、目标、允许修改文件、禁止范围、输入及事实源、输出、验收命令、stop gates和证据要求。同一文件只能有一个实现负责人。
5. 已运行任务由原负责人完成当前安全交接节点，再转交独立执行对话。交接包含worktree、版本及未提交变更、现有进程/任务标识、产物位置、验证状态和下一步；不得中断、重启或重复提交同一训练。Secret和连接信息遵循既有安全规则。交接未确认接收前不得写成已完成；新负责人接收后原负责人停止监控与写入。
6. 每个作业的执行与训练监控只设一个负责人；生成式Sol负责G线，传统Sol负责P线，Astra统一共享资源准入。旧作业完成身份/监控交接前不改变owner；当前暂停不因交接解除。正常等待期间Astra不与Sol重复轮询，不每分钟重复汇报；使用有界等待、退避或可用的完成通知，完成、失败、资源异常、需要决策时再升级给Astra。失联或观察超时不等于任务停止，先检查原进程。
7. 通过精简交接、减少重复日志、避免无必要继承完整历史和重复检查节省token，不得削弱TDD、实验预算、数据合同或验收。服务器自主训练与模型调用分开管理。
8. Worker的“通过”声明不能替代Astra fresh验收。Astra在验收节点运行必要定点检查，核对真实产物、配置、指标口径及证据，不只看exit code；默认只做一次独立review，不叠加worker reviewer与多个协调review或重复全suite。
9. 每完成一个part，依次向Eddy说明：做了什么；成果及入口；问题及处理；验证及限制；当前阶段及下一步。用易懂中文，保留必要technical terms，支持理解和面试复盘。

具体阶段计划保留在`tasks.md`；当前负责人、执行对话和交接状态记录在`STATE.md`、`handoff.md`；问题及解决证据记录在`docs/issue_log.md`。遵循既有事实源划分，避免重复维护。

## 数据与实验底线

- Label/history/split/catalog的完整规则只在`docs/data_contract.md`维护。
- Label policy只用train确定，并在读取validation/test指标前冻结。
- History必须严格早于target；顺序证据不足时停止训练。
- 传统与生成式方法必须共享history、features、catalog、query set、split和调参预算。
- SID-level指标不能替代collision-corrected item-level指标。
- First-seen只能称observed-item proxy，不称真实上新。
- 不声称普通CTR、真实dynamic catalog、线上A/B、因果lift、OneRec或AI Search全栈，除非事实源已有对应证据。

## 公开与安全边界

- Public Git只允许原创代码、配置、文档、synthetic fixture、脱敏manifest和小型指标摘要。
- Raw/processed data、完整embedding、checkpoint、optimizer state、index、trie、完整SID map和逐query预测不得进Git。
- Secret、token、私钥、signed URL、服务器连接信息不得写入仓库或聊天。
- 任何公司的内部数据、schema、代码、文档、截图、日志、指标和业务信息不得进入本项目。
- 第三方代码先做provenance/license审计；没有明确兼容许可时只按论文独立实现，不复制代码。

## 事实源

| 信息 | 唯一事实源 |
|---|---|
| 当前阶段、状态、阻塞、下一步 | `STATE.md` |
| 新对话恢复与上一轮证据 | `handoff.md` |
| P0～P13路线、依赖、验收 | `tasks.md` |
| 数据与label合同 | `docs/data_contract.md` |
| 模块边界与数据流 | `docs/architecture.md` |
| 已确认决策 | `docs/decision_log.md` |
| 实验与指标摘要 | `docs/experiment_log.md` |
| 风险与关闭证据 | `docs/issue_log.md` |
| A6000操作 | `docs/server_runbook.md` |
| Codex执行协议 | `docs/codex_handoff.md` |

副本只放指针；指标数字只在`docs/experiment_log.md`维护。

## 状态维护

任务开始、完成、阻塞、gate失败、commit/branch变化、实验冻结或对话交接时：

1. 更新`STATE.md`当前快照。
2. 更新`handoff.md`上一轮证据和下一task。
3. 按类别更新decision/experiment/issue事实源。

Hook只能提醒更新，不得自动编造状态或实验结论。
