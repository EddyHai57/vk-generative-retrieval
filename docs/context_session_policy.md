# 单主对话与上下文交接制度

密级：公开。2026-10-11，COORD-03。适用于本项目的Codex长期执行；本机context配置为全局设置。一个长期主对话负责G/P两线，必要短期只读研究或一次独立review才用sub-agent，不加载全部主历史，不重建多层审批链。

## 已核配置与能力边界

- 本机配置与最后已记录turn：GPT-6.1 Sol/medium。2026-10-11模型cache（client_version 0.162.0）默认context272000、max_context_window872000、有效比例95%；全部当前可选模型的本机max均872000，不为窗口切换Astra。
- 官方GPT-6.1 Sol API规格为1050000 tokens；这不是本机Codex已生效容量。Codex公开源码限制配置不超过模型max；本机当前可配置最大872000，预期有效828400，但最终以实际会话token_count/app显示为准。
- Eddy已确认长窗口方案：本机`~/.codex/config.toml`现为`model_context_window = 872000`、`model_auto_compact_token_limit = 700000`。本次仅将compact从200000改到700000，其他TOML字段完全不变，修改前备份`~/.codex/config.toml.context-backup-20261011T063612Z`；最初未加context设置的备份仍保留。未改模型、认证、权限或models_cache。
- 本机全局`~/.codex/AGENTS.md`已建立精简提醒/交接规则，项目顶层AGENTS.md给相同触发及细则指针。官方AGENTS发现通常在新run/session启动时建立指令链，项目/更深目录规则可覆盖全局；不保证其他已开会话热加载或每次精确及时执行，不为验证调用额外模型。
- **配置已写入，运行时扩容尚未证实。** 当前旧会话最后已观察有效窗口258400。新会话/客户端重新加载后再核`model_context_window`；没有该证据就不声称当前已扩容。不能恢复压缩时已丢失的历史。不为验证发送超长测试prompt或重启训练；无需为本轮配置强制马上换对话。
- context容量不等于账号额度；扩大窗口/换对话均不增加账号剩余tokens或credits。长历史会增加每次输入，cached tokens仍属于上下文；API长输入费率不可直接当ChatGPT订阅扣额公式。默认保持medium，任务需要才加reasoning，不启Ultra自动fan-out。

## 如何判断，不依靠“感觉”或历史总tokens

优先用Codex可见context计数或本会话rollout的最后`event_msg/token_count`：同时记录timestamp、模型、`last_token_usage.input_tokens`和`info.model_context_window`。input是最后请求的输入样本，不是此刻实时精确占用，须预留output/reasoning空间。`total_token_usage`是累计请求计数，不可当当前window占用或收费金额；账号usage/reset也不是context百分比。

仅定点读本会话末尾有界日志与安全metadata，不读/打印全部对话、其他会话、secret、tool payload或原始数据。计数时间落后、模型不同、字段缺失则标UNKNOWN，不推测百分比。至少在开启新task、结束part、准备真实job前检查；一个长part连续约20次工具调用且没有新样本时补查一次。不是每回合轮询，不设新的定时模型唤醒。

最新有界样本：2026-10-11 14:37:15 Asia/Shanghai，last input187328，有效window仍258400，约72.5%；新配置尚未被本旧会话加载。按小window提醒阈值155040已进入提醒区，准备阈值206720尚未达到；样本不是实时数据。无持续质量异常或新建需要，不强制创建新主对话。

## 触发与处理

下面50万/65万/70万是Eddy确认的工作规则，不是行业标准或模型质量实验结论；覆盖之前未获其认可的15万/18万/20万保守方案。提醒=min(500000,60%实际有效W)，交接准备=min(650000,80%W)，自动compact配置目标700000，实际门仍受客户端/模型限制。当前设置未加载时不能按大窗口假设等到50万；计数只用带时间样本，承认采样滞后。

1. **提醒阈值：** 主动简短告诉Eddy“上下文已较长”，给时间/样本及来源；更新当前task、证据和下一步，不再追加无关历史或重复日志。相同阈值不重复提醒。
2. **交接准备阈值：** 核STATE/handoff能否恢复并保存最新证据，必要时准备新对话提示词；保留checkpoint、job身份和不可重置预算。可以继续当前task，不因准备交接阻断正常工作。
3. **压缩阈值：** 平台目标700000自动compact是配置行为，提醒/文件更新依赖agent执行，不是已安装的token报警hook，不保证刚好在阈值前提醒。compact后核task/合同/授权/owner/预算与下一步；可准确恢复则继续同一主对话。只有质量异常持续、阶段结束后大量历史无关或Eddy要求时才安全冷启动轮换，不因到70万强制新建。不要fork全部旧历史。
4. **质量异常优先：** 即使tokens少，出现已证实的owner/预算/合同混淆、反复重做接受项、引用错误产物或遗忘暂停，立即停相关动作、对照事实源纠正。一次严重授权/身份错误，或核对后同类混乱再次发生，提前交接换新对话；不得把每个普通bug都归咎于上下文。
5. **无计数：** 明说无法测量；依task/part节点维护两文件，按上述可观察质量异常换对话，不说“已经70%”。人工说“换对话”时立即完成安全交接。

## STATE与handoff，事件更新而非聊天累积

开始/完成task、真实job提交或终态、预算/合同变更、阻塞或暂停、owner交接、压缩/轮换前更新。STATE只保留当前阶段/owner/阻塞/唯一下一步；handoff只保留恢复顺序及原证据指针。旧历史确需保留时归档并hash，不继续堆叠“最新”章节；不因无变化反复写文件。

交接必备：task ID与验收目标、最新human授权和禁止动作、允许文件、准确worktree/branch/未提交与ignored产物、唯一job owner及start/permit/root、真实状态时间与UNKNOWN、预算spent/剩余/deadline/no-retry/resume、packet/产物路径SHA、已完成验证和局限、stop gates、下一条已获准动作。不复制长日志/原始数据/secret。指标仍只写experiment_log，不能把旧RUNNING或作者通过变成新验收。

冷启动检查只做必要身份/证据核对，不重跑已接受suite。远端job是否仍活须依据新收据；新聊天不重提、重启、退款或重置clock。运行在监督器内的job可保留；只有内存中的本地进程句柄而无可接管持久身份时，先到安全节点或标BLOCKED，不能假装交接。

## 单对话轮换与标题

当前主执行01：01a12202-362a-7cb2-80e1-43bcdea63ff7。后续标题格式`VK｜主执行02｜<当前task>`，之后序号递增；STATE记录当前序号/真实thread ID，handoff记录上一会话及接收状态。

Eddy本次请求允许达到交接条件后自动准备并创建下一主对话；无需逐次微审批。只在条件满足时调用app create_thread：先查本项目实际project ID，沿用同一local项目与现有worktree路径，不新建checkout、不自动archive/delete旧聊天。创建前停止旧对话执行与写入，保存HANDOFF_PENDING；新对话确认证据后写ACK与唯一owner。创建失败准确报告并给手动入口，不循环创建、双主推进或通知旧长期worker。工具若不提供自动创建则让Eddy手动新建并粘贴下列冷启动提示。

> 接续VK项目单主执行。先读主仓AGENTS.md、STATE.md、handoff.md、docs/context_session_policy.md，再定点读当前task事实源。旧主会话已停止执行；核最新交接、worktree、授权、budget与job身份后记录接收，不能重复启动原job或重跑接受项。保持GPT-6.1 Sol/medium、一个长期主执行者，旧Sol/外部代理与timer不唤醒。先核实际context配置和计数，不假定百万已生效；按制度提醒与维护快照。只推进STATE给出的下一task，常规授权内自主完成，真正预算/科学/高风险变化才合并询问Eddy。

## 研究来源与适用限制

- [OpenAI官方配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)：context及自动compact参数；配置存在不等于当前会话后端已接受。
- [GPT-6.1 Sol官方规格](https://developers.openai.com/api/docs/models/gpt-6.1-sol)：1050000是API规格；不替代本机max metadata或ChatGPT计费规则。
- [Codex GitHub覆盖与最大值实现](https://github.com/openai/codex/blob/main/codex-rs/models-manager/src/model_info.rs#L16)、[有效window/compact计算](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/openai_models.rs#L487)：公开main源码会变，安装版与实际runtime需分别核对。
- [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：compaction、结构化笔记与按需检索；不提供通用百分比质量保证。
- [Claude Code context及成本](https://code.claude.com/docs/en/costs)：说明无关历史与上下文成本，产品命令不能直接照搬Codex。
- [Codex官方usage说明](https://learn.chatgpt.com/docs/pricing)：model/context/reasoning/tool use/caching共同影响额度，减少无关context及MCP注入；新对话不恢复账户额度。

COORD-03研究使用一个短期Sol/medium sub-agent，未启旧G/P线程、无新实验/SSH。TOML解析及其他字段不变核验通过；文档验收不重复项目训练suite。

## 额外token开销

全局指令本次802字符；字符不是token数，未声称精确token开销。规则有少量输入成本，节点计数读取/简短通知/事件写文件也有开销；不新增长期agent、定时轮询或全历史扫描。比规则自身更大的变量是将历史保留到70万，可能增加每次请求输入、延迟与账户消耗，不能保证净节省。cached历史也不是零成本；不以累计token数计算账单。若费用压力明显，可先删无关context/减少重复工具输出，或另与Eddy调整压缩阈值，不静默改回20万。

[官方AGENTS加载与覆盖顺序](https://learn.chatgpt.com/docs/agent-configuration/agents-md)说明全局规则怎样进入新会话；不是系统强制实时告警机制。
