密级: 公开

# 物品热度+质量+行为权重组合（FEAT-04 条件组合）：Kiro 复查材料

2026-10-17，本地只读证据整理；context 计数 UNKNOWN。本文是既有指标与收据的复查索引，指标唯一主账仍是 `docs/experiment_log.md`，不作转正或 anchor 裁决。本轮没有服务器连接、训练、checkpoint/test 读取或事实源修改。

## 范围与冻结规则

组合固定为 F4-T010 + C + D + E，E 已包含 Q2；用户塔完整复用 E，候选塔添加 C 两表与 D 两表，仅对 supported 物品生效。不能根据 E−A 结果把 E 换成 A。主对照永久保持同 seed 的 F4-T010；序列用户塔 M1-mean 仅描述性，不能追溯替换主对照。

依据：`docs/decision_log.md`「2026-10-16 夜间：Opus 复核」§4、2026-10-16 下午 anchor 条目、2026-10-17 并行本地任务；`docs/superpowers/plans/2026-10-15-p-feat04-and-seqmodel-plan.md` §1.2；`docs/superpowers/specs/2026-10-15-feat04b-card.md` §7。组合三 seed 的工程/终态/配对证据分别见 experiment_log「FEAT组合工程」「SUCCESSOR」§1与「下午：P anchor执行记账与FEAT组合配对统计」。

## 配对统计：已有真实数字

每个 seed 都评相同 2520 个 validation query，623 个 OOV；catalog 120415。b=只组合命中，c=只对照命中。exact McNemar 是双侧精确二项检验；不追加方向条件或 Bonferroni 规则。下面数字是主账与原摘要的引文。

| 对照 | seed | 组合/对照 hits | Δhits | b/c | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| F4-T010（冻结主对照） | 42 | 484/439 | +45 | 118/73 | 0.0013902601073185607 |
| F4-T010（冻结主对照） | 2024 | 502/434 | +68 | 140/72 | 3.493232865141978e-06 |
| F4-T010（冻结主对照） | 2025 | 490/444 | +46 | 118/72 | 0.001042653290765216 |
| 序列用户塔 M1-mean（描述性） | 42 | 484/466 | +18 | UNKNOWN | UNKNOWN |
| 序列用户塔 M1-mean（描述性） | 2024 | 502/481 | +21 | UNKNOWN | UNKNOWN |
| 序列用户塔 M1-mean（描述性） | 2025 | 490/471 | +19 | UNKNOWN | UNKNOWN |

F4 的平均差值为 +53 hits，3/3 seed Δhits≥6，3/3 exact p<.05；M1 的平均差值为 +19.333333333333332 hits，仅由既有同 seed 总数相减，不能据此恢复 b/c 或 p。原 `combo-paired-summary.json` 的第二组实际是 **E，不是 M1**，不得改标签替用。组合−E 的 +24/+36/+29（平均 +29.666666666666668）沿原登记仅描述性，不是组合相对 M1 的结果。

| 对照 | seed | ΔR@50 bootstrap CI95 | ΔN@50 bootstrap CI95 |
|---|---:|---|---|
| F4-T010 | 42 | [0.007142857142857143, 0.02857142857142857] | [0.0014255026029736917, 0.007752843133269786] |
| F4-T010 | 2024 | [0.015873015873015872, 0.038492063492063494] | [0.003868239613604296, 0.01056913831910389] |
| F4-T010 | 2025 | [0.00753968253968254, 0.02896825396825397] | [0.0013509820033960449, 0.00800426840972733] |
| 序列用户塔 M1-mean（描述性） | 42 | UNKNOWN | UNKNOWN |
| 序列用户塔 M1-mean（描述性） | 2024 | UNKNOWN | UNKNOWN |
| 序列用户塔 M1-mean（描述性） | 2025 | UNKNOWN | UNKNOWN |

bootstrap 固定复用 phase2 原算法：10000 次按 query 配对重采样，`numpy.default_rng(0)`，10×1000 分块，同一采样索引计算 ΔR/ΔN，percentile 95% CI。三个 seed 重复同一批 query，用户也可能相关；逐 seed query CI 不代表 seed、用户簇或总体的不确定性。

**p_pool**：原摘要没有存该项。依2026-10-17报告请求，可从已验收的 F4 b/c 本地派生：Σb=376，Σc=217，双侧 exact 二项 p=**6.721183655121359e-11**。这是本轮新计算的补充描述项，没有服务器新收据；它不是 FEAT-04 §1.2 的新增转正门槛。M1 p_pool=UNKNOWN，必须读取已有逐 query 配对归档。合并跨 seed 的 discordants 不独立，p_pool 偏乐观，不能当成独立证据强度。

## 真实来源、SHA 与 UTC

| 证据 | 真实读取时间（UTC） | SHA256 |
|---|---|---|
| `cache/pgr01-p-recovery-v1/main05/codex-successor-combo-read-20261016T112000.stdout`：三 train/verify 原始接收 | 2026-10-16T04:43:12.800296+00:00 | `7888e2fd00f3b4f056c3f13c3b4b299484903eac2dbdab561f0d440fe14ecf8f` |
| `cache/pgr01-p-recovery-v1/main05/codex-combo-paired-afternoon-20261016T080000Z.stdout`：组合/F4/E共9run的配对原收据 | 开始2026-10-16T07:26:59.452774+00:00；结束07:27:01.851367+00:00 | `9b8f02faaccab8bf84cf2d266cd354dfddba751a6bf446f14621c0095cad9a5b` |
| 同配对 transport | UTC见原 transport；不按文件名前缀猜 | `8ec57286625e951da60bda143c4ec0fcd623e30e64e7f9209f17dc5d24bfca39` |
| `cache/codex-m2-execution-20261016/combo-paired-summary.json`：上述配对的本地摘要 | 继承上行来源，独立生成UTC UNKNOWN | `10336361bf7486184ec10ea1377b26bdaa115e1e07800e4716a7a7855102c613` |

配对 reader 来源 commit=`05fd95e71d0f8221c0daffce7f691b1a6de423c7`，源码 SHA=`6774225098f5b5cb13cb8c5bc37ab5b01c81762d50581aac77e530371179fa5d`。固定 phase2 依赖 SHA=`548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8`。组合科学公开部署 commit=`84b160cb0eeb3c53b877f6cf6474276ba06a6ba5`，与本地`c4e4eb8c2a9d87d42303e2c0021e168942fda33a`等树；冻结 SHA=`04ae4dcc0fb1ec56ea50c4ae692606dd002198c8aa82f9080c722175bd0da90c`。

原配对收据核对 raw verified、job/manifest/config/data/source、predictions SHA与顺序。逐run明细和完整 `read_artifacts_sha256` 在原 stdout；本报告不复制完整服务器清单或逐query预测。文件名前缀不是实际执行时间。

## C/D/E 泄漏证据清单与限制

以下 worktree 统一以 `<WORKTREES>` 表示；映射沿项目 gitignored 的 `docs/local_paths.md`。本轮只读这些文件，没有重新运行旧生产路径或读取旧 checkpoint。

| 对象 | 测试/实现出处 | 已有独立 review 出处及范围 | 可确认与缺口 |
|---|---|---|---|
| C 物品热度/新鲜度 | `<WORKTREES>/feat04-p/vk-generative-retrieval/tests/test_feat04b_run.py`：L1 target翻转、L2导出隔离、L3候选反向对照、L5物品ID一致、L6 week25拒绝、L7源码守卫、L9训练/导出同源；`test_feat04b_stats.py`：截止周/绑定、L10构造器无query输入；`test_feat04b_model.py`：候选统计不进history、unsupported零贡献 | 同worktree `.agents/tasks/feat04b-review.md`，fresh-context #1，APPROVED，review commit=`e8bc9dcb50db403a19a12989396abc02022b4628`；freeze stage1记review文件SHA=`3c8691252bcf548453362ca7a4666bb3e7e47d53f638eb5414e97a4b468a89fc` | review读diff/验证记录，未重跑测试；有定点pins/常量核对。review早于结果，是否满足“有效后”专项review由Kiro认定 |
| D 物品观看质量 | 与C相同，另 `stats_self_inclusion` / `p_like` / `logq_redundancy` 审计及候选桶约束 | 同 #1；`.agents/tasks/feat04b-audit-receipt.json`与`feat04b-freeze-stage1.json`保留审计绑定 | D-1偏离已登记，不得用review的“没有发现泄漏”概括成严格时间隔离；`stats_self_inclusion`仅审计回执、未进runner诊断的LOW仍须披露 |
| E Q2+行为权重 | `test_feat04b_run.py` 的L1/L2/L4/L5/L6逐变体含E，L4翻历史事件自己的feedback位；`test_feat04b_model.py` 的β仅进logit/初始零/不耗RNG/19标量/位序/PAD守卫；E4与04a冻结SHA桥接 | 同 #1；同worktree `.agents/tasks/feat04b-review2.md`，APPROVED，范围`e8bc9dc..d3ca347`，review commit=`d3ca3473b8d6efce840d493a617d61ee1e9beb16`；主仓副本`.agents/tasks/feat04b-review2-for-freeze.json` SHA=`03315b1e04314174af329bbc421569a922284843765cd6f39918819043dc7d77` | #2未重跑/spot-check，E4初始是coder自述，后freeze-stage2实际e4={ok:true,ran:2,skipped:false}且stage1行为pins重算相同；不是结果后专项泄漏认可 |
| 组合接线 | experiment_log「范围、实现与验证」：旧04b基线236、相关回归130（新增15）；真实preflight每seed查512训练+全部2520验证，翻target/position/oov不改user，候选仅supported | 同节记录“一次独立Sol review未发现组合公式或新增泄漏问题”；组合cache保存regression与review修复RED/GREEN日志 | 独立review原文/明确review对象SHA在本地combo cache未找到，UNKNOWN；现有记录不能替代Kiro对§7.6全范围的认可 |

冻结卡 §7.6 的专项范围仍需逐项认可：`_tensors4b`及`fr._tensors2`、E池化/桥接、C/D统计builder/候选；L1–L10四变体存在并在真实`engineering=False`路径运行；STAT_CUTOFF_WEEK与binding；训练/评估候选同源；§4.2缺口/AC16/C9齐全；有条件loo1（本次未触发）。现有preflight说明不能证明L1–L10全部在真实路径运行，**完整范围认可缺证据**。旧review没有重跑测试也必须保留。

历史工程限制：服务器压平树缺基准Git对象导致16个freeze测试FAIL；decision_log 2026-10-16 01:xx已接受替代证据（270文件SHA一致与本地39测试通过，issue I-2026-10-16-a），不能写成服务器该组独立通过。本轮不重复review、不重跑这些旧任务。

## D-1 与 23:2x 审计事实（必须随结论保留）

decision_log「2026-10-15 16:0x」D-1：C/D聚合train第0–24周全部行，包含训练样本自己的target行及其之后的train行；validation不进入统计。增益可能含训练侧自身target信息；不能称“严格时间隔离的物品特征有效”，不能由validation隔离推断训练侧无偏差。

decision_log「2026-10-15 23:2x」取代冻结卡的推断：sum_likes=105700，训练样本86785，差18915（约+21.8%），所以“Σlikes≈训练样本数”总量不成立；93.0%物品likes_i=target_freq_i，D点赞率分子与logQ目标频次高度重合。p_like=.2115≤.30，C9预设提醒不触发。**C9仅报告项，不是转正门槛，也不能证明无偏差**；.30没有更强依据。组合不能分离C/D贡献，不能声称交互或因果效果。

## 冻结转正四条件逐条核对（不下转正结论）

为便于核对，将计划“有效”的两个并列项拆开，保持原规则不变；不把p_pool、bootstrap、C9或延迟加成FEAT门槛。

| 冻结条件 | 状态 | 证据/剩余工作 |
|---|---|---|
| 三seed平均Δhits≥6 | 满足 | +53，对原F4-T010 |
| 至少2 seed Δhits≥6 | 满足 | +45/+68/+46，3/3 |
| 至少2 seed逐seed exact McNemar p<.05 | 满足 | 3/3；全部Δ为正，按字面计数与正向计数一致 |
| 新池化/新增数据通过一次独立泄漏review | 缺证据 | 既有#1/#2与组合review记录可供认可；§7.6全范围与有效后认可尚未见正式裁决，不由本报告代判 |

三训练及对应science verify均已接收，是数值有效性的前提；不代替第4项。未发现需写为“不满足”的前三项；第4项是认定/证据缺口，不能擅自写通过或失败。当前P anchor沿既有decision仍是序列用户塔M1-mean。

## 主执行者下一动作与补数入口

1. 用新增`analysis/acceptance_paired_read_20261017.py`及synthetic unittest完成本地检查、扫描、固定SHA发布后，由主owner经既有MAIN05只读通道运行一次。它沿固定phase2 `load_archive`读取已verified的组合/F4/M1/M2验证归档，统一出三组逐seed exact/CI/p_pool，核对共享query/history/catalog/support及组合plan绑定；无CP/model/test/queue写入。新stdout SHA、内部UTC、source SHA/commit尚UNKNOWN。
2. 把combo−M1的b/c/p/CI与两组p_pool新收据补入experiment_log，由主owner更新本文；不要用总hits或E配对推算M1检验。F4本地派生pool应与新工具交叉一致。
3. 将既有泄漏review全文、freeze/E4/真实preflight与§7.6缺口交Kiro确认能否认可；不要默认追加第二轮review，先让Kiro按已有材料决定。仅Kiro写转正/anchor裁决，G5绑定继续等该裁决。

工具本地验证边界：初次fresh RED为reader缺失的6 failures；实现后6 tests GREEN，再补消费者catalog/support/nonfinite切片测试共7 tests GREEN。它验证synthetic与fail-closed边界，不冒充服务器真实归档验收。原phase2按query bootstrap不改变；服务器真实新数仍UNKNOWN。


## 02:40服务器只读补数（当前，覆盖前文M1配对UNKNOWN）

来源main05/acceptance-paired-20261016T183959Z.stdout；SHA256 4a283900c081c0ee488a6fbfda16b16715eea487397559237eb10f3a6f023b2f；实际读取UTC 2026-10-16T18:40:00.703469+00:00，结束UTC 2026-10-16T18:40:04.197577+00:00。

| seed | 新/对照 hits | Δhits | b/c | exact McNemar p | ΔR@50 95% CI | ΔN@50 95% CI |
|---|---|---|---|---|---|---|
| 42 | 484/466 | +18 | 98/80 | 0.20246043236353428 | [-0.0031746031746031746, 0.01746031746031746] | [-0.0016637427087852572, 0.004429912977361445] |
| 2024 | 502/481 | +21 | 99/78 | 0.13253419223810542 | [-0.001984126984126984, 0.01865079365079365] | [-0.0009708181302984046, 0.005276939333005028] |
| 2025 | 490/471 | +19 | 102/83 | 0.18555890492457608 | [-0.0031746031746031746, 0.018253968253968255] | [-0.0022255414407312017, 0.004590511029794268] |

p_pool=0.014096472751947788（Σb/Σc=299/241）；mean Δhits=19.333333333333332。

F4的三seed统计与原表完全一致，p_pool=6.721183655121359e-11。两组完整R/N CI及两组原始收据索引见experiment_log“训练等待期只读复查补数”。对M1三个逐seed p均≥.05，p_pool不作为组合转正门槛；不改F4注册对照。第四条件仍缺完整范围认可，不下转正结论。
