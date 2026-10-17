# Handoff — 当前快照（2026-10-18；Kiro 单人兼任执行者与裁决者）

密级：公开。本文件只写**恢复当前对话所需的最小快照与证据指针**，不累积多层「最新」。数字一律看 `docs/experiment_log.md`。
新对话按 AGENTS §10 走启动顺序。`.agents/tasks/claude-code-takeover-20261018.md` 里的项目背景、踩坑清单与协作约定仍然适用，**但其 §1 的角色分工已被本文件第 1 节取代**（Claude Code 未实际接手）。

> 重写说明：本文件在 2026-10-18 由 Kiro 重写（Eddy 批准）。此前内容是 2026-10-13「Kiro 全权接管」的快照，含已失效的 API timer 暂停说明与 2026-10-14 到期的连接信息，与当前分工冲突。原文可从 git 历史取回（分支 `codex/takeover-20261015-stage1`，本次重写提交之前的任一提交）。

## 1. 执行者与分工

- **Kiro（Opus / Sonnet）同时是唯一执行者与裁决者**，2026-10-18 起（Codex 与 Claude Code 额度均耗尽）。详见 decision_log「2026-10-18：Kiro 单人兼任执行者与裁决者」。
- **这是降级状态**：独立复核消失，自评自己的风险上升。本期间不读 test、不迁移全量、不改已登记门禁、不新增名额、不删除或覆盖任何 run / bundle / checkpoint。
- 补偿措施：执行与裁决分两次动作、写两处文件；review 与结果复核一律派独立 subagent 以干净 context 做，不自审；Sonnet 做实现与只读核对，Opus 做冻结与裁决，decision_log 标注模型。
- Codex 额度恢复后交还执行权，AGENTS §1 改回双人分工，交接前核原 job / 预算 / 截止收据。

## 2. 服务器状态（交接时刻）

只读核对收据 `cache/pgr01-p-recovery-v1/main05/kiro-status-handover-20261018.stdout`：

- 仅 `sched-serve-02` running；**pending 0**；本项目没有训练或 verify 在跑或排队。
- GPU 49139/49140 MiB free；CPU quota 10.24 cores、idle 100%；磁盘 203.8/500.0 GiB free。
- 新 SID 的 A1 确认（S5）与用户画像特征（F7）的驱动均已正常写终态后退出。**没有悬空作业，不需要恢复、不要重启任何驱动。**
- status 列表里的历史 `failed` 条目是早前的工程失败记录，按规则保留；不能把它们误读成本轮训练失败。

接手后第一步仍应自己跑一次只读核对（命令见一次性说明 §2.2），与上面不符就停下报告。

## 3. 两个 anchor

| 线 | anchor | 来源 |
|---|---|---|
| 双塔（P） | 序列用户塔 M1-mean（`p-seqm1-mean-s42/-s2024/-s2025`） | decision_log 2026-10-16 下午转正 |
| 生成式（G） | A1 + 共现向量 SID（`g-coll-a1-c1-s42/-s2024/-s2025`，bundle `COLL_C1_B256D3_S42`） | decision_log 2026-10-17 晚间；A1+T3 降为历史对照 |

两个 anchor 都已固定，这是 G5 可以进入冻结讨论的前提。

## 4. 阻塞与下一步

1. **未推送（最高优先）**：工作分支 `codex/takeover-20261015-stage1` 无 upstream，约 71 个提交只在本机；远端是公开仓库。按 AGENTS §3 第 3 条扫描要推送的全部新增历史后 push 工作分支；不 push main、不 force push。
2. **双塔与生成式融合（G5）**：草稿 `docs/superpowers/specs/2026-10-17-g5-anchor-fusion-draft.md`，未冻结。已授权**一次只读盘点** P top-100 预测归档是否存在及验收状态（decision_log 2026-10-18 §6）。有归档则交 Kiro 冻结；没有则列最小 checkpoint 读取清单交 Eddy 逐项批准。不得自行改成纯 top-50 版本。
3. **可学习 SID（S2）**：草稿 `docs/superpowers/specs/2026-10-17-learnable-coll-sid-draft.md`，DESIGN_REVIEW_REQUIRED。等 Kiro 写死 §4 的 8 项合同 + Eddy 批名额。许可证 UNKNOWN，不复制任何官方实现。
4. **P 线下一步**：未裁决。候选为 G5 的 P 侧部分，或全量规模评估。
5. 收尾顺序：G5 冻结 → 读一次 test（Eddy 批）→ 全量迁移（Eddy 批）→ 报告发布。

## 5. 上一轮证据指针

- 本轮全部裁决：`docs/decision_log.md`，从「2026-10-17 午间」读到文件末尾。
- 指标与原始收据：`docs/experiment_log.md`，2026-10-17 各节（S5 最终验收、F7 配对与延迟、协同 SID 诊断收口）。
- 风险：`docs/issue_log.md`，近期为早停右删失、延迟观察运输失效、F7 收口后诊断、verify stdout 身份缺口。
- 一页状态图：`docs/project_map.md`。

## 6. 已知 OPEN 项（接手后不要当成已关闭）

- 旧 G6 fixture 的 input hash mismatch：仍 OPEN，不靠改测试合同绕过真实 preflight。
- verify stdout 不含 run_id / manifest SHA：新入口请补上。S5 有两个 seed 的 verify 收据字节相同（best/stop 恰好一致），身份靠 manifest SHA 绑定。
- 四个旧归档缺少 verify 别名：UNKNOWN，只影响历史报告完整性。
- 公开历史清理项 `I-2026-10-15-i`：需 Eddy 单独批准，未执行。
