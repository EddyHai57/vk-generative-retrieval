# 三对话异步协作实施计划

**Goal:** 两个Sol并行完成完整任务，Astra集中统筹、异常与最终review，降低重复和等待。
**Architecture:** 复用三个可见对话、既有worktree和packet；消息通知加15分钟compact快照兜底，不建设新调度系统。
**Tech Stack:** 现有Codex对话工具、Markdown、现有server监督程序。
**Spec:** `../../codex_handoff.md`。Eddy已批准前述分工并明确要求讨论、写规范后恢复；执行方式沿用既有两个Sol，主协调只修改治理文件。不因通用skill模板新增subagent实施、commit或重复许可。

## COORD-01 文件与职责

- 修改AGENTS.md：稳定角色与协议指针。
- 修改docs/codex_handoff.md：唯一详细工作流。
- 归档STATE/handoff旧快照到docs/history/workflow-2026-10-09/并收敛当前内容。
- tasks.md仅登记行政任务，不改变P/G/S科学路线；decision_log记流程决策。
- 更新既有vk heartbeat为15分钟，不建第二monitor；旧p4d保持暂停。

## 讨论与取舍

Eddy：协调少介入日常决定、Sol并行主动汇报，任何READY不被另一个长实验阻塞。
生成式Sol：完整entry交付一次派工；一次独立review；packet为单入口、状态只指针；runtime server就地绑定；训练等待不让模型空转。
Astra采纳以上五点。保留资源实际启动唯一owner和exact-start，避免把流程简化误成放宽科学/预算gate。暂不引入LangGraph/额外平台或第三个长期worker。

## 落实步骤

- [x] 请生成式Sol一次只读评议，获得五项具体建议；不重复召集reviewer。
- [x] 归档旧记录并写协议，审查与AGENTS的角色/验收/通知规则一致。
- [x] 通知两Sol采用新协议并恢复原task，保留既有实现和已验收packet。
- [x] 将vk改15分钟ACTIVE；指向简短当前快照，不复制完整项目历史。
- [x] 确认恢复派工已送达，验证文档链接与归档完整性。
- [ ] 前三次完整交付后复盘等待和重复成本（运行后动作，不阻塞本次恢复）。

## 当前恢复范围

生成式G3：恢复cache/g3-component-cost-entry-v1，先修测试helper异常finally wait，再完成ApprovedAdapter、owned parent/worker、bounded transport/exact argv与receipt整体交付。旧measure/accounting已验收，不重做；真实90min预算已批准，实际启动前必要gate不跳。
传统P4F：恢复cache/p4f-runtime-gate-implementation-v1，修已记录receipt/单次执行防护问题，完成A/B整体交付；现有依赖不重装。G3启动就绪时scheduler在安全节点优先处理。
两者只恢复已授本地实现；真实执行由完整packet验收及exact-start接续，不把恢复消息当所有未知命令获批。S1容量不足仍BLOCKED，不下载/删除/迁机/新增计费资源。

## 验证场景

1. A RUNNING、B READY：马上处理B，不等待A答复。
2. 重复同SHA通知：只记录一次验收；新SHA才增量审。
3. READY通知被拒：记录未送达，正常权限wait/local兜底，不绕过。
4. 两线争GPU/磁盘：scheduler排实际运行，本地实现仍独立。
5. 聊天idle/server运行：保持RUNNING，不误派重启。
6. 修复后交付：一次独立review+必要定点复验，无重复全套审查链。

验收采用文档一致性和上述场景逐项检查；无产品代码，不新增镜像实现的测试。预算/合同/安全规则不变。
