# PG-R01 — 基线接收与恢复准入阶段计划

日期：2026-10-12。密级：公开。

**Goal:** 收齐已有P/G基线证据，给出下一真实实验的可审核入口，先不增加模型或重置预算。

**Design:** [两线研究路线](../specs/2026-10-12-pg-research-roadmap.md)。这是Plan模式第一阶段的接收/准入计划，不是后续模型的代码实施计划。

**执行载体:** STATE登记的唯一主执行者。当前规划对话只编写本文件；不向旧owner发送任务，不并行修改facts/生产。阶段实际执行前由唯一owner将PG-R01登记到tasks.md，更新STATE/handoff；若用户选择本对话执行，先完成旧owner安全交接，保留原job/spent。

## 输入、输出与边界

输入：AGENTS→STATE→handoff→tasks；P/G worktree的experiment_log及冻结合同；G原终态收据与accepted core00结果；P原queue/driver闭包、失败原因、source/test/probe证据。

允许写入：唯一owner登记的tasks.md、STATE/handoff、对应worktree事实源；新接收包放ignored cache；本阶段公开摘要建议docs/research/pg-r01-baseline-reception.md（只放状态与证据指针，指标只放各experiment_log）。

允许操作：本地只读核验；需要时定点只读服务器产物读取，限定JSON/schema/hash与小型摘要，timeout60s/每文件不超过1MiB/每次不超过8MiB。大manifest超界先记录缺口，再申请具体有界读取设计，不无界下载。checkpoint只绑定必要身份，不做全量下载/新模型replay。只读数据接收不等于新的GPU训练。

禁止：重跑已接受训练、第三P attempt、G submit/restart/resume/signal、新monitor、新模型/HPO、test读取、新资源/生命周期、删除现场、reset/stash/discard、push；数值验收若需要新的实际计算，先核原验收授权和剩余有限预算。

当前context计数UNKNOWN。开始/完成/真实job前按项目规则核，不为此加monitor。

## Task 1 — G原四配置接收

- [x] 核core00既有accepted指标入口；核core01/02/03 observation与receipt SHA一致、run/start/terminal身份、worker exit及group closure。ALL_TERMINAL只计工程终态。
- [x] 定点读取原train/best-evaluation/verifier/seal摘要，核config、source/input/SID/catalog/query、2520完整分母、all/in-catalog/OOV、best选择、原spent与失败ledger。缺证标UNKNOWN，不填指标。
- [x] 指标更新到G docs/experiment_log.md，不复制到STATE/roadmap。产物证据和验证限制保存新ignored接收包。
- [x] 全部四配置通过真实接收后，按原Recall@50→NDCG@50→枚举tie rule生成selection；没有通过不冻结winner。
- [ ] 2024/2025补seed与random/beam诊断复用原任务；先核原authorizations/finite caps及spent。若剩余包络不足，提出明确数字后等待新grant，不因本计划自动submit。

验收：四配置分别为accepted或具体缺证；selection状态与证据一致；不把terminal receipt当效果通过，不重跑core00。

## Task 2 — P失败与候选恢复包络

- [x] 保留2/2已spent ledger及原deadline；核两driver CLOSED、0admission/worker/optimizer证据和FD PermissionError原因。第二attempt的错误准入如实记录。
- [x] 核最新source/test SHA与63项历史日志；UID只读probe只证明该局部检查，不作为完整publisher批准。新source不得重pin旧producer冒充原输入身份。
- [x] 盘点最新候选完整publisher/bootstrap/prepare/controller/verifier闭包缺口；使用已有实测提出下一次有限验证的步骤与成本，不先启动第三attempt。
- [x] 给Eddy一张新的包络卡：attempt上限、绝对deadline、总disk cap、包含的部署/prepare/pilot/verify/cleanup成本、保留旧spent、失败后动作。任何数值须说明实测依据及不确定性。
- [ ] human grant后另按原P任务执行完整publisher proof→train-only pilot→独立verify→完整成本→正式S02/S06。PG-R01本身不包含训练授权。

验收：有明确的原因、待验证项和有限包络；若无grant，状态是BLOCKED_BY_NEW_ENVELOPE，不是完成P基线。

## Task 3 — 建立基线表与阶段出口

- [x] P既有accepted传统结果与G已接受结果按同query/catalog/分母核可比性；ID-only/content-SID差异明确，不误称同features。
- [x] 基线表索引记录配置、seed、数据/SID版本、验收状态、成本边界及experiment_log入口；表不重复指标数字。P4H未开跑明确缺席。
- [x] 分类记录：完成训练、产物接受、效果结论、未运行实验。匿名case分析在对应产物许可内执行，不输出raw。
- [x] 本阶段汇报包含“已核、仍缺、下一步”；唯一owner更新STATE/handoff。只有实际基线齐备才称PG-R01基线就绪。
- [ ] 进入PG-R02前，在Plan模式冻结具体feature定义、实验矩阵、TDD范围、配置数、seed及finite预算；PG-R04-G论文对齐另立计划，不在本阶段顺手实现。

## 定点核验与stop gates

本地收据可用stdlib读取字节并重算SHA；文档使用git diff --check。生产没有改动就不跑旧full suite。真实结果核验需逐个产物检查schema/身份/分母及外部seal，不只看exit0。若需新增验证代码，先按TDD定点测试缺件、错hash、错run、错query分母、未完成verifier必须拒绝。

STOP：source/data/hash漂移、owner不明确、history/query/label变化、预算未知、需要新attempt、远端状态UNKNOWN需要推断、需要改旧科学包、意外test读取。UNKNOWN不等于进程停止。

计划文件保存和本地commit不更新旧运行许可，不恢复Go或改变heartbeat。当前应用模式仍Default；用户在应用切到Plan后，先细化本阶段实际接收设计再开始执行。

## 本阶段执行结果（2026-10-12）

接收报告与selection已一次fresh review APPROVE；G四配置已接收，P恢复具体条件与proposal已给出。G2024/2025、P真实pilot/formal未运行，baseline尚非全部就绪。下一先确认有限预算与最小修复设计，未勾选的真实执行/后续阶段条目不假报完成。报告docs/research/pg-r01-baseline-reception.md，下一计划docs/superpowers/plans/2026-10-12-pg-baseline-next-execution.md。
