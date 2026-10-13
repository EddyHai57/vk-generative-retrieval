# Claude Code 项目入口

**历史入口：2026-10-11 Eddy已停止外部代理并改回单Codex主对话。** 本文件不再授权Claude/Kiro自动恢复；最新执行方式只看AGENTS.md、STATE.md及docs/context_session_policy.md。原下文接管提示只作历史。

Eddy要求Claude Code全权接管现有subset传统和生成式召回后续任务。先读AGENTS.md、STATE.md、handoff.md，再读docs/claude_code_takeover.md；只读取当前task相关计划与合同，不默认加载历史全篇。

Codex所有对话和automation保持暂停。旧Astra/Sol固定模型与三方调度已由Claude接管要求覆盖；不重新唤醒旧agent。Claude实际启动后先核原G job，接收唯一owner，不重复submit、不重置clock/spent。具体路线、预算、证据与权限边界以STATE/handoff为准。

当前只授权subset完整交付，不自动下载全量或扩预算；常规实现/诊断/验收自主执行，一次完整task后继续，无需逐步询问。保留未提交与ignored产物，无commit/push/删除/新付费资源。任何工具拒绝不得绕过。
