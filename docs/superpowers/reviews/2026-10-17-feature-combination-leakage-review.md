密级: 公开

# 物品热度、质量与行为权重组合：结果后独立泄漏 review

日期：2026-10-17。Verdict：**CHANGES_REQUESTED**。

依据 decision_log「2026-10-17 上午」§1.2，对冻结卡 `docs/superpowers/specs/2026-10-15-feat04b-card.md` §7.6 做一次 fresh-context 定点 review。存在一项阻塞性证据缺口：尚未找到 A/C/D/E 的完整 L1–L10 在真实 `engineering=False` 路径运行的逐项证据。不能据测试总数、真实 preflight 或既有 coder/reviewer 自述宣称该条满足。本结论不等于已发现新的 validation 标签泄漏；也不构成转正或 anchor 裁决。

本轮仅执行 Git 对象读取、代码/既有文本收据阅读及 SHA256/JSON 比对；没有运行任何测试、模型、训练、preflight 或服务器命令，没有读取 checkpoint、test 或真实数据。唯一新文件是本文；未修改实现、STATE 或事实源，未 commit。context 计数 UNKNOWN。下文 `<WORKTREES>` 指本机 worktree 根的占位符，映射沿 gitignored `docs/local_paths.md`。

## 固定对象与身份

- 直接读取 `git show 84b160c:<path>`，部署 commit 全称 `84b160cb0eeb3c53b877f6cf6474276ba06a6ba5`。
- 实测 `git rev-parse 84b160c^{tree}` 与 `c4e4eb8^{tree}` 均为 `acf222d2c34c6d5485d9184940018910d0596498`；本地 commit 全称 `c4e4eb8c2a9d87d42303e2c0021e168942fda33a`。
- 旧依赖位于 `<WORKTREES>/feat04-p/vk-generative-retrieval`。读取该处源码时，另与固定部署 Git blob 逐字节比较：item_stats、retrieval_feat02_run、retrieval_feat03、retrieval_feat04b、test_feat04b_run、test_feat04b_stats 六文件全部相等。未借用其他 checkout 的 HEAD 代替固定对象。

固定 commit 中所读关键文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `src/vkgr/item_stats.py` | `306ff83bf13332079a819497507f193d8ce9e53b5ed02dcf3a04150d762e73a1` |
| `src/vkgr/retrieval_feat02_run.py` | `1e66de6586a183565bc9817ec971247ae140f762646c24411b56950e108984fd` |
| `src/vkgr/retrieval_feat03.py` | `f0ff468901ec74ff6421823be8b92aea976087d5e2e12a3568eb92cd95c3581c` |
| `src/vkgr/retrieval_feat04b.py` | `7313d585fa3c468f4f5fe1c3c279e4a6e89be540c10af6ae943e8f79717147fd` |
| `src/vkgr/retrieval_feat04combo.py` | `4fc1e38f389914c8bfecb48c4ee251df83687814b281d0bd3dc3db60f921a6fb` |
| `src/vkgr/retrieval_feat04combo_run.py` | `2a4f9d1a093ffe570f4d3f7e9a8448e45dd75eb2425d85011d7c5f9db97be3b8` |
| `tests/test_feat04b_run.py` | `c8ee902b7dfa2d608d02c1023ab01e0ed7ecdf09614780c53bad011f62d04865` |
| `tests/test_feat04b_stats.py` | `383c6136b7b72b22da67a15f20ea16f137da050d4e8a5dde7f45eefebdc4ebdb` |

## §7.6 逐项结论

| 条目 | 结论与证据 |
|---|---|
| `_tensors4b` / `fr._tensors2` | 已核代码。组合 `_tensors4b` 固定委托 `_tensors2(..., 'f3_f4_recency')`；后者仅取 `history`、每历史事件自己的 `side` 与 `history_positions` 周。place/platform 保持零，未使用 target 周、target 标签、past_likes 或 oov 装配 user 输入。target 在训练 loss 的标签/候选列中的合法用途不等于进入 user 特征。 |
| E recency / behavior / pooling | 已核代码。组合的 `variant` 由旧 E 构造器设置，继承 E `users/history_items`；recency 是位置与历史内周差 logits，behavior 是历史事件 mask 各 bit 与 β 的内积，只加进 pooling logit；PAD 经 valid 与 pooling mask 屏蔽。`pooled_weights` 做减最大值的稳定指数，`weighted_mean` 汇总；不是 target-time 权重。β 初值零且不耗 RNG。 |
| E4 桥接 | 有既有机器冻结记录，非本轮重跑：`.agents/tasks/feat04b-freeze-stage2.json` SHA `a39a474cb9fc90b9c2b9f24445d72736ffbd8d8d776e85f658b1c59fd1720ede`，commit `d3ca3473b8d6efce840d493a617d61ee1e9beb16`，e4=`ok:true, ran:2, skipped:false`；K10 指向 04a 冻结 `31704176843134c467d3100f39f2700089a5a346`，stage1 行为 pins 重算相等。测试 `guard_s_bridge_e4` 经两 runner 的 synthetic CLI 比 manifest 排除项之外内容及模型 state 的逐位值。桥接证据来自 synthetic 路径，不能替代真实 L1–L10。 |
| C/D builder / candidate | 已核代码。`build_item_stats(partitions,catalog,durations)` 经 `aggregate_rows` 聚合 n/first_week/likes/ratio_pm_sum，拒绝非递增或 week≥25 分区；无 query 参数。旧 `candidate_stat_terms` 只按 item index 取桶并乘 supported。组合 `candidate_items` 顺序叠加 C 两表+D 两表，乘 supported；继承的 history_items 不读统计表。 |
| ① L1–L10 四变体存在且真实路径跑过 | **未满足证据要求，见下方唯一 finding。** 测试存在及相应真实 preflight 有证据，但完整真实运行矩阵缺失。 |
| ② 截止周与 binding | 已核代码及收据。`STAT_CUTOFF_WEEK=24`；loader 校验 cutoff、metadata/history/catalog/support、artifact/content digest、预审计 stats digest。组合 `apply_feat04b` 必须有 stats pin，按相同 idmap 生成四 buffer；binding 包含 stats_manifest SHA、stats_sha256、stat_cutoff_week、stat_bucket_ids SHA、bucket spec。verify 重算完整 binding 并比较，load_state_dict 对四 buffer 不可变性校验。 |
| ③ 训练/评估候选同源 | 已核代码：组合 `train_step` loss 候选是 `model.candidate_items(columns)`；export 全目录亦是 `model.candidate_items(catalog_ids)`，同模型同静态 buffer，无按 target 或 query 分支。旧 C/D 的 `guard_l9_train_and_export_vectors_equal` 是相应逐位测试；其真实路径执行覆盖仍受 finding 限制。 |
| ④ §4.2、AC16、C9 | 已核 §4.2 原文及后续更正；AC16 receipt.audit 与固定 Git 中 `configs/feat04b/stats-bucket-constants.json` 解析后完全相等，后者 SHA `c48a5908fa2e655bc108be77b54054d919f3fa44d95126955827b10145851ae2`。数字和限制见下节。runner 确有 target_bucket_hist/logq_redundancy，stats_self_inclusion 仅在审计中，保持既有 LOW 披露。 |
| ⑤ loo1 | 条件未触发：p_like=`0.21152272858212826`≤.30，审计 `p_like_exceeds_threshold=false`。因此本次不要求 loo1 实现/结果，不将未运行写成失败，不新增门槛。 |

## 唯一阻塞 finding

**[P1] §7.6① 的完整真实路径运行证据缺失。**

具体源码：固定 commit `tests/test_feat04b_run.py:442` 起的 L1/L2/反向对照/周平移/week25 守卫均从 `syn()` 取得 `synthetic_data4b()`；`test_l1...` 等遍历 RUNNABLE（含 A/C/D/E）。C/D 的 L3、L5、L9 候选替代守卫也用 synthetic fixture。`tests/test_feat04b_stats.py` 有 cutoff、增量、签名与 binding 夹具。测试编号有历史错位：设计 L3 的历史反向对照名为 guard_l4_reverse_controls，设计 L4 周平移名为 guard_l5_week_shift_invariance，设计 L8 的 week25 拒绝名为 test_l6_week_25_rejected；因此按语义核覆盖，不能只查函数编号。L6/L9/L10 等按设计有 C/D 或 builder 的适用范围，不要求不适用分支虚构测试。

已找到的实际执行文本：

- `cache/codex-feat04combo-20261016/vkgr-feat04combo-baseline-20261016.log`：236 tests / OK；SHA `549ef4cbb155b6ebf15526ba1e1a988db0e264a8a3b4a2ce1fc19cafe5cb41cd`。无逐项测试身份、真实输入 binding 或四变体真实执行矩阵。
- 同目录 `vkgr-feat04combo-final-regression-20261016.log`：原文 **128** tests / OK（不是 engineering summary 写的 130）；SHA `166ca4d0f2742599240990f6527c469daa69f81d5c0794f6100fce3ce4fc477a`。此差异不能用摘要覆盖原日志，也不据此推断额外两项是否跑过。
- `cache/pgr01-p-recovery-v1/main05/codex-combo-preflight-20261016T105600.stdout`：SHA `4be5665a07ef861c227d0eb884dcda92f487ef5a0e7db08bb2a45941993d2a9e`，三 seed 各检查512 train+2520 validation，均 preflight_ok；它运行的是组合一个变体。按固定 commit 的 preflight_check 源码，覆盖初态 user/control 相等、候选 supported 隔离、有限性及翻 target/target_position/oov 不变，没有在非零 E logits 下遍历四变体、未执行完整 L1–L10。
- `.agents/tasks/feat04b-preflight-cuda-stage2.md`（SHA `84b17ef2583d67e4900c6b4b19ce93fad5ad3fb96342b47d5f32314f20666929`）记录 E 真实 preflight 和 synthetic CUDA；同样不能补齐完整矩阵。源码中的 shared production functions 被 synthetic 测试调用，不自动等同于冻结卡明确要求的真实 `engineering=False` 运行证据。

关闭条件：先查找并提供**已有**逐项运行回执，绑定执行 commit/source、真实输入/侧表摘要、A/C/D/E 适用测试项、实际运行路径及结果。如既有回执不存在，由主owner将缺口交 Kiro 裁定如何补齐；本 review 的只读授权不允许自行重跑、降格条款或将 preflight 解释为全矩阵。当前不能出 APPROVED。

## §4.2 与审计读法（随结论保留）

审计原收据 `.agents/tasks/feat04b-audit-receipt.json` SHA `e093969a42e07c75ccdba0350701ec160051f3b28e0edf09d82a5d59b40d85f5`。冻结 stage1 收据 SHA `27f19f1e7caa290bfa2c184e24216b4ff8f17b378f10a7b571752fbc5e3b63a2`，记录 AC16 齐全、receipt/constants/code 相等、阈值 .30，threshold/stats commit `856d36873359b08a02737861f04bed36e6a7a9a9`，audit script SHA `37b1e978cc6b4cd61875a47f5d6e71c019dbcea1b3771aa68604b5d33c0ba033`。本轮直接确认 receipt.audit 与固定 Git JSON 相等；未重建真实统计。

AC16 数值沿唯一指标主账 `docs/experiment_log.md` 与上述收据引用：sum_likes=105700，train_examples=86785，差18915；likes_i=target_freq_i 占比0.9298426275796204，Σ绝对差18915；rows=3720328、catalog=120415、supported=21496；timespent=1..255、零行0，duration=5..180。self-inclusion：p_expo=.02630638935299879、p_ratio=.06917093967851587、p_like=.21152272858212826、p_first=.015348274471394826、p_first_is_own_row=.03825545889266578；n≤3占比=.03115745808607478、n=1占比=.010681569395632886。expo/like 与 logQ 的 Spearman 分别 .5349395568646098/.11170960284082157；描述性相关不能解释因果。

**D-1 继续成立**：C/D 聚合 train 0–24 周所有行，包含训练样本自己的 target 行及其之后的 train 行。validation 不进入统计，不能由此推断训练侧无泄漏或无偏差。增益可能含训练侧自身 target 信息，不能称严格时间隔离的物品特征有效；训练/服务偏差对 validation 指标方向未知。组合不能分离 C/D 增量或证明交互/因果。

**23:2x 更正优先于旧推断**：105700比86785多18915（约21.8%），不能继续写“Σlikes≈训练样本数”总量成立；约93%的物品相等与总量接近不是同一结论。p_like≤.30仅意味着 C9 预设报告提醒未触发；C9 是报告项，不是转正门槛，.30无更强统计依据，也不能证明无 self-inclusion。

独立 review 只判断本次 §7.6 证据是否完备。McNemar 等结果条件仍沿既有冻结计划与指标主账，由 Kiro 作最终组件/anchor 裁决；当前 verdict 不能被当成第四条件已满足。
