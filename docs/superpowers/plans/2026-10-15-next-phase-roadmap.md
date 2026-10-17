密级: 公开

# 下一阶段路线图与接手清单（2026-10-15 16:3x，Kiro；草案，随结果更新）

指标只在 `docs/experiment_log.md`；决策在 `docs/decision_log.md`；本文只放路线和顺序。实时 anchor 与作业状态以 STATE/schedule 为准；下方原 10-15 背景是历史，R1 SID-FIX 为 10-16 更新。

## 0. 现在的位置（16:21 实测）

- P 线：anchor = F4-T010（hits 439/434/444）。温度精扫 p-php02-t007-s42 / s2024 已在跑，A、B（FEAT-04a）在写代码，C/D/E（FEAT-04b）在写代码。
- G 线：名义 anchor = A1 + BHKM16×3（250/206/234）；候选 = T3（BHKM 256×3）：core03 上 254/274/275，A1 规模的 seed42（今晚出）和 2024/2025（已在跑）决定是否换。
- 差距：P 约 0.174，G 最好约 0.109（core03 + T3）。多通路融合、协同 SID 是缩小差距的两条主路。

## 1. 路线（按依赖和性价比排序）

### R1 SID-FIX：语义 ID 修复 pipeline（2026-10-16 登记）
唯一冻结内容见 `docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md`；S1 另卡草稿 `docs/superpowers/specs/2026-10-16-sid-fix-s1-freeze-draft.md`。此节替代原“只用 P 塔向量”路线；C2 来源固定 M1-mean seed42，未来 anchor 改变也不换来源。

| 阶段 | 工作与名额 | 前置 | 当前状态 |
|---|---|---|---|
| SID-FIX/S0 | SID 结构、逐位准确率、G4 热度分层与训练期邻接共现 lift；不训练、不占名额 | §2 已冻结；只读工具、一次 review、1 核 queue | 两批诊断完成；四旧归档B/C未验收，读法待Kiro |
| SID-FIX/S1 G-COLL | C1′ item2vec / C2 内容+固定 P 物品塔；2 配置、core03 s42 筛选后补 seed | S0工具可用；夜间Opus S1已FINAL | 工程实现/一次whole review中，未提交真实拟合/导出/训练 |
| SID-FIX/S2 LETTER 式 | 协同对齐与多样性；已批 1 名额 | S1 至少一项 UP、license 审计、独立冻结卡 | 未立卡 |
| SID-FIX/S3 深层准确率 | Kiro 先读两篇论文，名额视方案 | Kiro 判定适用性并另立卡 | 未立卡 |
| SID-FIX/S4 规模准备 | 合成 125 万/1247 万 ×64 拟合成本、suffix、新 item 分配设计；不占名额 | 独立冻结卡；可与 S0–S3 并行 | 未立卡 |
| SID-FIX/S5 A1 确认 | S1/S2 赢家 A1 三 seed，确认不计名额 | S1/S2 赢家与独立冻结卡 | 未立卡 |

每个新 SID 都须重新训练 G；checkpoint 读取许可严格限 S1 的固定 C2 导出，S0 不适用。指标与运行证据只见 experiment_log，当前执行状态以 STATE/schedule 为准。

### R2 多通路融合（G5，设计稿 .agents/tasks/g5prep-design.md rev3）
- 定义：P、G、Pop 三路（RRF，k=60，总预算 200，不调权重），问题 = 「G 作为第二路是否比 Pop 或另一个 P seed 作为第二路更好」。
- 先做一个零成本预检：用已存档的逐 query 预测，算 P 与 G 的命中并集和重叠（上限估计），再决定值不值得建完整框架。
- 阻塞：设计稿规定「只认冻结的 anchor」，要等 G anchor 的正式决定（A1+T3 三 seed 出来后），P anchor 也可能因 FEAT-04/温度精扫改变；anchor 冻结后又变则只能叫复测。
- 后续扩展：把 R1 的协同 SID 版 G 作为第三路加进去。

### R3 P 线特征（继续）
- FEAT-04a：A = Q2-T010，B = F6-hist。FEAT-04b：C 物品热度/新鲜度、D 物品被观看质量、E Q2 + 行为权重。温度精扫 0.05 / 0.07。
- F7 用户静态（Eddy 2026-10-15 批准）：先做 U2 活跃度（截至 target 前的曝光/点赞累计分桶）和 U3 口味画像（历史平均观看比例、点赞率、主要场景/平台）；U1 年龄/性别/地域涉及敏感属性，单独决定，不在本轮。名额暂按 U2、U3 各 1，写进冻结卡。设计时必须先查 U3 与 F3/F4 的信息重叠。
- F8 特征组合：等 FEAT-04a/b 与 F7 的结果出来，只组合有效的特征，不做用户×物品交叉。
- 序列模型：M1（seq/m1，本地 review 已 APPROVED，服务器步骤排在 04a 与温度精扫之后）；M2 多兴趣有条件。

### R4 G 线其它
- GTOK-02：SID 分辨率扫描（含 B256D2，已获批准）。
- A1+T3 三 seed：决定 G anchor。G6（SID 深度+1）价值要重估，仍暂停。

### R5 需要 Eddy 的节点
- 读一次 test（week 26）：TEST-00 + 文件权限 + 明确批准。
- 全量阶段换更大磁盘的 A6000：付费。
- 年龄/性别/地域（U1）是否使用。
- C9 是否恢复为转正门槛（decision_log 16:3x）。
- `/tmp` 残留文件的手动清理（见各 *-report.md「需要 Eddy 手动清理」）。

## 2. 作业顺序（以 docs/schedule.md §4a 为准，这里只列依赖）

1. 正在跑：A1+T3 s42 / s2024 / s2025，T1r s2024 / s2025，温度精扫（部分已入队）。
2. 结果出来后先做：T1r vs T3 对比、A1+T3 三 seed → G anchor 判定（关键节点，需要强模型解读）、温度精扫判定。
3. 然后：FEAT-04a A、B 入队 → GTOK-02 → FEAT-04b → M1 → R1 → R2 → F7/F8。

## 3. 接手清单（给 Codex）

读的顺序：AGENTS.md → STATE.md（文末）→ handoff.md（文末）→ 本文 → docs/schedule.md §4、§4a → docs/decision_log.md 末尾约 15 节 → .agents/tasks/common-brief-20261014b.md → .agents/tasks/relaunch-pack-20261015.md。

必须守的规则：不读 test；入队后只读确认进 claimed 且不在 rejected（`action=written` 不算成功）；新入口第一次上真实数据前做服务器真实输入的只读 preflight；每个训练配 verify；显存声明 = 实测×1.15 取整 256（P 作业 1024，core03 G 2560–3072，A1 规模 6656）；设计评审最多 2 轮；脱离会话的驱动，不用模型轮询；服务器只经 `main05/kiro_b_run.sh <脚本裸文件名.py> <新前缀> ro|rw`（脚本名不带路径）。

正在跑的驱动（本机，Mac 睡眠会停）：gtok01b_driver pid 35118（A1+T3 s42）、gtok01bc_driver pid 52031（A1+T3 s2024/s2025）、gtok01rc-driver pid 68450（T1r）；pid 文件与日志在 main05/。php02 的驱动由其工作流启动，pid 见 .agents/tasks/php02-report.md。

进行中的工作流（归 Kiro 当前会话所有，新会话不能恢复，只能按 relaunch-pack 的方式重启）：
- FEAT-04a `wf_1f586db76f039a87`（分支 feat04a/p，设计卡 .agents/tasks/feat04a-run/feat04a-design.md 末尾有裁决节）
- 温度精扫 `wf_8b5f7ba0b7e3e2c6`（分支 php02/p，报告 .agents/tasks/php02-report.md）
- FEAT-04b `wf_652ea62860f0a965`（沿用 feat04/p worktree，设计 feat04-design.md 在该 worktree 的 .agents/tasks/，计划 feat04b-plan.md）
- GTOK-02 `wf_5b153fe76f6a356f`（分支 gtok02/g，设计 gtok02-design.md §9、计划 gtok02-plan.md 末尾有裁决）

谁写什么（需 Eddy 确认）：建议 Codex 写 STATE、schedule、experiment_log 里的数字（带来源收据）；Kiro 只写 decision_log 的关键决定与裁决意见；issue_log 谁发现谁写，但标注来源。
