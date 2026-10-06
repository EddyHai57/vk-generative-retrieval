密级: 公开

# Codex Handoff

## 启动读取顺序

Codex 必须按以下顺序读取并遵守：

1. `AGENTS.md`
2. `STATE.md`
3. `handoff.md`
4. `tasks.md`
5. `docs/data_contract.md`
6. `docs/architecture.md`
7. `docs/server_runbook.md`

若文件之间冲突，停止实现，列出冲突及文件位置，由用户裁决；不得静默选择。

## Bootstrap commit 与 worktree gate

当前branch、commit和remote状态只以`STATE.md`及fresh Git检查为准。

如果`STATE.md`显示没有bootstrap commit，必须先由用户明确批准创建。

Bootstrap commit存在前，不得创建P1 branch/worktree，也不得开始实现。

后续每项任务从最新已验证主分支commit创建独立branch和worktree。

## 第一项任务边界

第一项任务只执行 `tasks.md` 中 P1 基础代码，并严格使用 TDD。

不得把 P1 到 P13 合并执行，也不得预先实现后续阶段的接口、模型或基础设施。

开始 P1 前先把验收标准改写成可运行的最小测试；先保存 failing test 的 red 证据，再写满足该测试的最小实现，最后保存 fresh green 证据。

## 每项任务的工作协议

1. 从最新主分支创建该 task ID 独立的 branch 和 worktree。
2. 只修改该任务明确指定的文件；发现范围外问题只记录，不顺手修复。
3. 测试必须在当前 worktree fresh 运行，不复用其他分支的成功结论。
4. 生成或更新 manifest，至少记录 task ID、Git commit、配置、数据版本/hash、产物 hash、命令、时间与资源摘要；未知项明确写 `pending`。
5. 完成后写 handoff：改了什么、没改什么、验证证据、已知限制、产物位置/hash、下一任务入口。
6. 接受 fresh-context review 后，才进入下一项任务。

## 安全与确认门槛

以下操作必须先列出具体动作、影响、费用或恢复方式，并等待用户确认：

- 删除文件、覆盖不可恢复产物、丢弃未提交改动。
- 创建、扩容、续租或恢复计费资源；调整账单上限。
- 调用外部服务、上传数据、创建公开发布物。
- 任何可能暴露 credentials、数据、模型产物或个人信息的操作。

不得上传 HelloBike 或任何其他公司的内部数据、schema、代码、文档、截图、日志与业务信息。

不得把 secret、token、私钥、原始数据、checkpoint、index、完整 SID map 或服务器连接信息提交到 GitHub。

## 每次交付的最小证据

- task ID 与严格 scope。
- red/green 测试命令、exit code 和关键输出。
- 数据与 claim 边界：使用了什么、没有证明什么。
- CPU/GPU/RAM/disk/wall-clock 影响。
- artifact/manifest hash 与服务器位置的脱敏引用。
- 已更新文档及未解决问题。
