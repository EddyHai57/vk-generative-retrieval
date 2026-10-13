# AGENTS.md — VK Generative Retrieval

密级: 公开

> 本文件只保存稳定行动规范与事实源指针。当前状态见`STATE.md`，跨对话接力见`handoff.md`，阶段计划见`tasks.md`。

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
- 修改完成后进行fresh-context review。
- 未经明确要求不commit、push、创建remote或修改Git配置。
- 删除、覆盖、丢弃改动、计费资源和其他高风险操作必须先列清单并等待确认。

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


## 最新工作流指针（2026-10-09 Eddy批准）
三对话职责、完整task自主实施、一次最终独立验收、既有worktree复用及通知去重，按主协调 `<MAIN_REPO>/docs/codex_handoff.md` 与 `docs/superpowers/plans/2026-10-09-agent-workflow.md`。覆盖本文件旧逐微步骤审批、多重review要求；不覆盖科学合同、预算、公开边界或实际exact-start。默认不自派reviewer，Astra完成一次验收；runtime敏感值server就地绑定，仅整体证据hash回传。每文件一个writer，保留既有worktree，不commit/push。
