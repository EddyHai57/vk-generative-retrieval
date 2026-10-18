# 双塔与生成式融合（G5）冻结卡（FINAL）

密级：公开。状态：**FINAL（冻结）**。冻结依据：decision_log「2026-10-18：双塔与生成式融合（G5）冻结」（Kiro，Opus 5）与「2026-10-18：批准落账、L100 事实更正、下一批优先级重排」第 5 条；Eddy 已逐项批准读取 `p-seqm1-mean-s42/-s2024/-s2025` 的 best checkpoint。文件名保留 draft 以维护已有引用。本卡由执行者按上述裁决改写，只抄不增；§2–§4 原文保留生效，§1、§5 末条、§6 按裁决替换，新增 §7 身份表。尚未实施、未入队、未读取任何融合结果。授权：decision_log「2026-10-17 晚间：新SID的A1确认」§5.1。指标事实源仅experiment_log。

## 1. 是离线合并，还是需要新训练？（已裁决）

**本卡不需要新训练：两边模型已经训练并验收。** 计算本身是固定候选列表的离线合并，不更新任何模型参数；零新训练名额。

只读盘点结论（experiment_log「2026-10-18 双塔与生成式融合（G5）：P anchor top-100 预测归档的一次只读盘点」）：**不存在 P top-100 归档**。按裁决走下面的路径：

1. **保留 rev3 协议与 200 预算**：两路各 100、三路各 66、剩 2 不分配。不改成纯 top-50 版本。
2. **P 侧 top-100 来源**：对三个 `p-seqm1-mean-*` run 的已验收 best checkpoint 做只读离线推理导出（非训练，不占名额）。约束逐条生效：
   - `weights_only=True` 只读加载；只前向推理导出候选；
   - 不导出用户向量或完整 embedding；不读 `last.pt`；不动任何已有 run 目录；不读 test；
   - 产物存服务器、不进 Git；checkpoint、导出脚本、导出文件均记 sha256；
   - **正确性闸门**：新导出 top-100 的前 50 位必须与原 `predictions-epoch-NNN.json`（各 run 已验收 best 对应的那一个）逐位相等；不相等即 STOP 回 Kiro，不放宽、不改评估口径；
   - 该导出按 AGENTS §4 走闸门：fresh RED/GREEN 与变异、一次独立 subagent review、扫描 push、固定 SHA 部署、真实只读 preflight、涉及 CUDA 则预演与实测 × 1.15 ceil256、dry-run、claimed/no rejected、自身 verify。
3. **取消 G 侧前 64 query 的二次解码对拍**：直接复用 G 侧已归档候选。理由：对拍只验证解码可复现，三个 `g-coll-a1-c1-*` 的 verify 已直接 verified，收益不足以换取三个额外 G checkpoint 的读取面。
   - **已知限制（必须写进最终报告）：G 侧候选未做二次解码对拍。**
   - G 侧候选深度若不足配额（已知 G 短列表现象），按 §3 原规则：不补、不把空位转给 P。
4. 如另提需要新训练的融合器，另卡、另请 Eddy 批名额，不在本卡。

## 2. 冻结模型与问题

P固定M1-mean：p-seqm1-mean-s42/-s2024/-s2025，已验收best，不用叠加、bi或第二次挑选。G固定A1+共现向量SID：g-coll-a1-c1-s42/-s2024/-s2025，共享COLL_C1_B256D3_S42 bundle。A1+T3只作历史对照，不进入替代G路选型。对应选择/manifest/config/source的原SHA以experiment_log已验收条目为准，执行前写入不可变anchor receipt，任何UNKNOWN必填项拒绝冻结。

主问题：在相同候选配额下，G作为P第二路是否优于Pop或另一P seed？“在P+Pop上再加G”另报描述性比较，不替换主问题。旧oracle重叠预检用的是旧anchor，不能迁移其数值到本卡；新anchor的并集只作上限，不能当融合效果。

Pop固定同源train-only item_like_counts，排序(-count,item_id)，不按validation拟合；不读原始test。三方使用同一catalog120415、同一2520 validation query、HIST_V2_OBS L50；623 OOV留在主分母并单列结构性0。

## 3. 路线与确定性算法（沿既有rev3，等待Kiro冻结）

主表seed42：C1=P top50；C2=G top50；C3=Pop top50；C4=P+Pop；C5=P+G；C6=P+G+Pop；C7=P(seed42)+P(seed2024)。2024/2025各复现C1/C4/C5，仅稳定性描述，不挑seed或顶替主表；是否扩展主判至三seed属于另行设计，本文不做。

总预算200，n路配额q=200//n；两路100、三路66，剩2不分配。G短列表不补，不把空位转给P；P/Pop不足所需配额则输入闸门失败，不按G短路规则绕过。单路直接top50。跨路等权RRF，k=60、rank从1起，精确Fraction(1,60+rank)求和；跨路按item_id去重、同路重复致命；按(-score,item_id)稳定排序，final50。不做权重/k/预算搜索、ANN或候选补全训练。

## 4. 验收、配对检验与解释

保留decision_log 10-14 22:4x的整数门槛与Bonferroni口径，旧设计稿rev3作技术来源：
- H1：C5−C1≥6 hits且exact双侧McNemar p<0.025。
- H2：C5−C4≥6且p<0.025，并且C5−C7≥6且p<0.025，intersection-union；两族各α=.025，族内不再校正。
- D1=C6−C4只作描述，不占α，不替换H1/H2。
- 上述四比较均报b/c、整数Δ、N@50差与bootstrap95%区间。沿rev3 random.Random(20261014)、10000次query重采样、最近秩分位，四比较共用下标；不能改用其它已用过的bootstrap实现当成相同协议。

主表全2520及in_catalog/OOV分开报；同train-only热度、support、用户历史、历史长度与G短列表切片。每多路行相对C1/C4都报新增/丢失/净增、路级独立命中、交并/重叠、pool大小、每路实际填充、coverage。低支持定位沿rev3：非OOV中unsupported或低热度或G短列表的query集合只算一次；净增≥6且占全量净增≥50%，全量净增≤0则false；这是继承待冻结字段，不因新anchor结果调阈值。

只过H1不足以把收益归给G；H1/H2与G独立命中/定位按rev3分开报告。OOV无收益不能略去。沿新G anchor的D-1同类偏离、约0.2%内容回退及早停右删失；不称线上A/B或因果收益。

## 5. 输入合同、工程闸门与成本

必须逐query核ordinal、user、target、严格历史item/side字段、target时间/OOV标记、query/history/catalog hash；不只比目标序列。源预测经原verify且完整SHA链到best manifest。P top100前50须与原归档逐位相等（§1第2条闸门）；G候选直接用原存档，**不做前64query二次解码对拍**（§1第3条，已知限制）；checkpoint读取仅限§1第2条已批准的三个P best checkpoint，不外推到其它checkpoint。实际新入口需新verify stdout含run_id和manifest SHA。

P/G仍用各自原解释器、固定source与binding，输出写新目录。离线推理/export/fuse都是只读输入且新写输出的非训练作业，各走AGENTS§4：fresh RED/GREEN与变异、一次独立review、扫描push固定SHA部署、真实只读preflight、涉及CUDA则预演/实测×1.15 ceil256、dry、claimed/no rejected、各自verify。不能从用户塔延迟外推检索或端到端；P导出/检索与G解码、载入、融合wall分开，存档amortized b1延迟不与串行逐query混称同口径。

冻结版本只认首个实际产出融合结果的primary；后续anchor变更为复测，旧结果保留，不删失败、不重跑挑最好。

## 6. 冻结事项（decision_log「2026-10-18：双塔与生成式融合（G5）冻结」逐条）

1. 保留 rev3 协议与 200 预算，不改纯 top-50（见 §1 第 1 条）。
2. P 侧 top-100 按已批准的只读离线推理导出（见 §1 第 2 条），含正确性闸门。
3. 取消 G 侧前 64 query 二次解码对拍；「G 侧候选未做二次解码对拍」列为已知限制，写入最终报告（见 §1 第 3 条）。
4. 草稿原 §6 第 2 项：新 anchor 身份表（§7）、主表 seed42 + 两个稳定性复现、H1/H2/D1 与低支持定位规则**逐项承接生效**；rev3 末尾 23:0x 的裁决**确认具有正式效力**。身份表任一必填字段为 UNKNOWN 即拒绝冻结，身份只从 experiment_log 已验收条目填，不凭记忆。
5. 本卡零新训练名额；资源与耗时按工程实测填，不预填训练式 ETA。
6. 判定与报告按 §4 原文：H1 与 H2 各族 α = .025，D1 只作描述；只过 H1 不足以把收益归给 G。限制继承：D-1 同类偏离、约 0.2% 内容回退、早停右删失，另加 §1 第 3 条的解码对拍缺口。不称线上 A/B 或因果收益。
7. 冻结后执行顺序：P top-100 导出（含正确性闸门）→ 融合计算 → 配对检验与报告 → 交 Kiro 在 Opus 下解读。**不自动下结论、不改 anchor。** 用户画像特征不进入本卡。

## 7. 身份表（只来自 experiment_log 已验收条目；无 UNKNOWN 必填项）

### P：序列用户塔 M1-mean（best 已验收，三 seed）

模型源码固定 `codex/seq-m1-pub@0ca61bc35ad68bd79b86103b54e470445cbab07f`；运维核心 `codex/phase2-readonly-tools@6a8e057bb021ed50dbaacc92d75c7a048b5bf45a`（experiment_log「2026-10-16 晨间 M1 / GTOK-02 验收」）。

| run_id | seed | config SHA256 | manifest SHA256 | best/stop | verify stdout SHA256 | 出处 |
|---|---:|---|---|---|---|---|
| p-seqm1-mean-s42 | 42 | `e88dc757ff4085af020efc1f070ebba8df54876e85adf68120dff730f983b621` | `bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e` | 12/15 | `12ef936b63e42a289b3c1d61a04499c375cf55c3a4d65db3b6a534cc9b6bb18c` | 配置：M1 PF1 表；其余：2026-10-18 G5 盘点节 |
| p-seqm1-mean-s2024 | 2024 | `3e1e40f200c3a09a39e15048df33657bbb54080a7da33d604d92c41518a5dde5` | `8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8` | 7/10 | `0422a9b5d324ff79b3d37c51fbfbff09240ed2bea9b98589dc3294f211aaf591` | 同上 |
| p-seqm1-mean-s2025 | 2025 | `e4f8f612991ee960b496067b905cb57bb46d7d2b9e4c4af9026004a53e8713f3` | `783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb` | 11/14 | `cd0eeb35e2f28038dc9374e9648a9d74174ca360f3c85f5ac07a3ecb7a8583ec` | 同上 |

每个 run 的 verify 回执 `--manifest-sha` 与上表 manifest SHA 相等（盘点节已核）。best 对应的原预测归档为各 run `out/predictions-epoch-{best:03d}.json`（best=12/7/11），其 `predictions_sha256` 在各自 manifest 的 evaluations 中，preflight 读取并核对，不在此处复制。

### G：A1 + 共现向量 SID（COLL_C1_B256D3_S42）

run 源码固定 `a5da4d6d0746ef41ca1ed93028971360dcb89547`（experiment_log「2026-10-17 19:00+08」）。

| run_id | seed | config SHA256 | manifest SHA256 | best/stop | verify stdout SHA256 |
|---|---:|---|---|---|---|
| g-coll-a1-c1-s42 | 42 | `e8f07f67b9ff1d5a650caaf47bce1aeb4a2db5b8e4804150f012ad584a74ef75` | `a431a5b1734930671cf7ca092e469a6da16e3c130ffb30836b9aa57aaca8b36b` | 17/20 | `9a04887ff44ee514fffe569a51360184e9c4dab314247493e49585634fb66aac` |
| g-coll-a1-c1-s2024 | 2024 | `263f5ec1c883ee89a90627a1f58c48f7f0f2aca73f52252c688b8be1820e887a` | `09a0b790c6b027613ef15bb6c9855d7b1adee79d9bf6ea093c16d09a1a5b8d36` | 14/17 | `48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67` |
| g-coll-a1-c1-s2025 | 2025 | `2e28801ce898dfdb79664d976771d44827d9be9111548a489afb5e6d583e2c45` | `86767e4d30ababc8087898f6f960197021ce54fd09cc6d775da4c373b242c712` | 14/17 | `48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67` |

s2024 与 s2025 的 verify stdout SHA 相同：两个不同路径的文件内容都只含 status/replayed_epochs/best_epoch/epochs，且 best/stop 恰好相同（decision_log 2026-10-17 晚间已核），身份靠 manifest SHA 绑定。共同 SID manifest `c53b81ddfcf8d45fd55903b30f8538372110c8e1df7376156719509994625187`；prepared `a55df0a3b285ce4cf623126fe53df53378bf5d7c0645c6484f0f405641bddd82`；history `8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43`；context `d81c7012bf2adc2ecc7d71cb19ec4b26ab17c61607148e930d31b26eda752189`。

### 共同数据身份

query/history/catalog 的逐项 hash 不在冻结时抄录：由各 manifest 的 binding 在真实只读 preflight 中读出，P 与 G 两侧逐 query 核 ordinal、user、target、历史与 catalog 一致（§5），不一致即 STOP。catalog 120415，validation 2520 query，其中 in_catalog 1897、OOV 623。

## 8. 导出入口的源码身份：固定增量（decision_log「2026-10-18：G5 导出 binding 失配的裁决…」第 1–4 条）

**源码身份用固定增量形式；导出正确性由 top-50 逐位相等承担。** 两者分工不得混淆：身份检查只证明「源码树多了几个不参与计算的文件」，真正证明推理链被完整复现的是 §1 第 2 条的前 50 位逐位闸门（不放宽、不开容差，不等即 STOP）。

- 断言：`实际源码账本 == 训练源码账本 ∪ {下表逐条钉住的新增路径及其 sha256}`；python 版本、依赖、`lock_sha` 严格相等。
- fail-closed：任何预先存在文件的字节差异、任何未钉住的新增路径、任何已存在于训练账本的钉住路径，一律失败。
- `source` 之外的全部绑定字段（inputs、config、init、side 及其数据 hash、execution、init_report 等）仍严格相等，不得放宽。
- commit SHA 差异不是通过条件；导出 manifest 同时保留训练 commit 与导出 commit。
- 钉住的新增路径由作业 argv 的 `--pin-new-path PATH=SHA256` 传入（模块无法包含自身 hash），该值与本表逐字相同，由入队工具核对。
- 测试 `test_pinned_module_is_not_imported_by_model_or_data_construction` 证明被钉住的模块不被模型构造与数据构造路径导入，且 `src/` 下其它文件不引用它，因此不可能改变数值。

| 新增路径 | sha256 |
|---|---|
| `src/vkgr/retrieval_seqm1_export100.py` | `462a69a16aed0b73f0205b60e538a1dfc16a6fb37d3379f98f251ec49b76d811` |

账本范围说明：源码账本只覆盖 `.python-version`、`pyproject.toml`、`uv.lock` 与 `src/vkgr/*.py`（非递归）；账本之外的字节（`docs/`、`artifacts/`、子目录等）不在固定增量覆盖内，数据合同文件由 `real_inputs` 内的 `validate_contract` 另行核对。钉住的 sha256 必须等于最终部署的模块字节；模块任何修改都会使该 pin 失效、需重新钉住并重新部署。导出与 verify 必须在同一份部署上运行（verify 会重算并比较导出 commit）。

训练源码：`codex/seq-m1-pub@0ca61bc35ad68bd79b86103b54e470445cbab07f`。导出源码分支 `codex/g5-p-export100`，导出提交在部署时写入 experiment_log。除上表一个文件外，账本中其余全部文件必须与训练账本逐字节相同。

## 9. 授权与运行纪律（decision_log 同节第 4–6 条）

- 本导出重做是 Kiro 明确授权的最后一次尝试，仅限 G5 的 P top-100 导出；第三次失败停该支、写 ALERT、回 Kiro。
- 全套闸门：独立 review（须覆盖 binding 改动、preflight 子命令、stdout 显存峰值输出）→ 扫描 push → 固定 SHA 新部署 → 真实只读 preflight → CUDA 实测 → dry-run → 新 run_id 入队。今后一律先 push、再部署、再运行，包括一次性脚本。
- 正式导出作业的显存用本次 stdout `cuda_peak_mib` 实测值 ×1.15 向上取整到 256 MiB 重新声明，不得沿用 1024。

## 10. 融合核心的实现钉住与报告措辞（decision_log「2026-10-18：G5 融合核心的两项实现选择裁决；变异测试有效性风险；两处追问」第 1–3 条）

- **bootstrap 实现钉住**：`src/vkgr_g5/fusion_v2.py` 内的 bootstrap 实现按下表 SHA256 钉住；该哈希同时写入每次融合输出的 manifest（`bootstrap.fusion_v2_script_sha256`）。抽样方式：`random.Random(20261014)`，10000 次重采样，每次依次 `rng.randrange(n)` 取 n 个下标，四个比较共用同一组下标，最近秩分位。模块任何修改都会使该 pin 失效、需重新钉住。

| 脚本 | sha256 |
|---|---|
| `src/vkgr_g5/fusion_v2.py` | `46c562257c372a46b550ca442ab417dc8551345ddf4c80f3d9f8a18fc1fe9537`（以最终部署字节为准，复审若改动会更新本行） |

- **不可比声明（须写入报告）**：G5 用 `random.Random(20261014)`（来自 rev3 原协议），与 P 线此前各批（叠加、双向确认、用户画像特征）使用的 NumPy `default_rng(0)` 配对 bootstrap **不是同一协议**，区间不得跨批直接比较。
- **阈值措辞（须按此写）**：净增 ≥ 7 是 p < .025 的必要条件，不是充分条件；真正的约束是 discordant 高度集中在一侧（b/c = 7/0 过；6/0、8/1、10/3 均不过）。预注册的「≥ 6 hits」原样保留，只是在 α = .025 下被 p 值条件覆盖，不是阈值变更。不要写成「实际要求净增至少 7」。
- **α 说明**：G5 因两族 Bonferroni 用 α = .025，严于 P 线各卡的 p < 0.05；G5 若判不成立，不能与 P 线的「不转正」等价看待。
- **变异测试隔离**：本卡相关的所有变异结论均在隔离缓存下取得（见 `docs/superpowers/reviews/2026-10-18-g5-fusion-mutation-ledger.md`）；未注明隔离的变异结论不采信（issue `I-2026-10-18-pyc-false-survivor`）。

## 11. catalog 表示规范化（工程附录，分支1；不改科学门禁）

依据decision_log「G5 catalog 摘要不一致的诊断授权」分支1；真实诊断及完整SHA只见experiment_log最新内容诊断节。两侧实际有序整数item_id列表集合/顺序完全相同，G原摘要比P多末尾LF。统一规范对象：item_id升序、统一整数语义、compact UTF-8 JSON、无末尾LF；规范摘要双方均 `abe2e7a3291f45eb7d535cac574c3ffae8192c79ff726fc25cd6349543152cf5`。不取交集、不补齐、不缩小catalog、不改任何科学阈值或统计脚本。

原始诊断stdout SHA `42959f7c75501e940662b3b6e50db270ee4ef86f93ba60403d39cfc7be28b521`；诊断源码SHA `f5a255cb9dc1e0f04d8f5997d537c86a2da758b14918929d824c8446a00708eb`。只重建G s42实际context；其它两个G的原catalog摘要相同，后续仍须逐seed核原文件SHA及摘要，不声称重复重建。

实现必须保留全部原输入与失败记录，在本新run目录生成规范化输入和manifest，逐个绑定原文件SHA/原摘要/规范化口径/新摘要/诊断收据SHA；所有非表示字段保持相同。C档独立复核后继续原融合闸门，旧融合核心与统计脚本SHA不变。尚未完成规范化实现/复核，不将本附录当成融合已通过。

表示规范化接续入口固定为 `analysis/g5_catalog_normalised_fusion.py`，SHA256 `48a2499d22f48b7b0cb74071bb8044bbc41213e00deab981106887d40d96cfe3`；原 `fuse.py` SHA `015f30acb630226b5a787170b1dc1d038b7c3fd7ac60d2cce00f095fc3432cd7`、统计核心 SHA `46c562257c372a46b550ca442ab417dc8551345ddf4c80f3d9f8a18fc1fe9537` 不变。独立 C 档初审仅有测试缺口，B 档逐条补测后全部变异杀死；报告见2026-10-18-g5-catalog-normalisation-review.md。接续为一个有界CPU非训练队列作业：规范化真实只读preflight→只写本新run输入与manifest→原融合真实check-only→唯一primary→原统计全量确定性replay→verified及完整绑定。不用模型监控，不新增训练/arm/seed；任何失败停止，不重跑或覆盖。规范化口径与新摘要进输入provenance、规范化manifest及验收manifest。尚未部署或执行本接续，不能称融合结果已验收。
