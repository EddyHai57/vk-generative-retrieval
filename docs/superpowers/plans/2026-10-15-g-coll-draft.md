密级: 公开

# G-COLL 实验卡草案（PHASE2-STEP1-GCOLL，2026-10-15）

**状态：DRAFT，供 Kiro 复查；不是冻结卡，不是运行授权。** 本次只新增本文；未实现、未导出、未读取 checkpoint / 完整 embedding / test，未连接服务器、未入队、未 push。本文没有新实验结果。

目的：检验在同一 catalog、history、query set 与 core03 训练设置下，把 P 学到的协同物品表示用于 SID，是否比 T3 的纯内容 SID 更有用。两条路线分别回答“纯 P item tower 表示”和“P 表示与内容拼接”是否值得继续。G 会使用 P 的训练产物，对外必须称“使用协同信号的生成式召回”，不能继续称纯内容方法。

## 1. 已有授权与本卡待裁决的边界

| 项目 | 状态与唯一依据 |
|---|---|
| 做 G-COLL 两个输入版本 | 已批准，见 `docs/decision_log.md` 2026-10-15 16:3x“下一阶段路线”第 1–2 条；路线细节见 `docs/superpowers/plans/2026-10-15-next-phase-roadmap.md` R1 |
| 配置名额 | 两版本按 G-TOK 扩展各 1 个，不占原有 3 个；seed42 筛选、入选补 seed；补 seed 的具体选择规则尚待本卡裁决 |
| 架构与 tokenizer 路线 | R1 指定 core03、BHKM 256×3；依赖 GTOK-02 / sid_tok2 的拟合、审计工具及 P 导出 |
| P / G anchor 身份 | 均不能据草案冻结；P 暂为 F4-T010，G anchor 仍待正式裁决。G-COLL 的主对照是 core03 + T3，不因 G 正式 anchor 变化自动改成 A1 |
| 阈值、采用、fallback、混合权重与 seed 绑定 | 本文均为 DRAFT 建议，不属于已经批准的门禁；Kiro 先复查。若涉及改已确认的预注册门禁，须 Eddy 明确批准 |
| 实施与真实数据动作 | 当前未授权；不得把路线批准或本文完成当成冻结、读 checkpoint、导出或入队凭证 |

指标的事实源仅为 `docs/experiment_log.md`，主参照是其中“GTOK-01C：T3 core03 补 seed 2024/2025”节：T3 三 seed 均值 **267.67 hits**。这只是既有参照数字；完整逐 seed 数、分母、verify 限制以该节为准。本卡不写新数、不判优劣。

## 2. 源码证据与接口限制（只读）

下表的路径均相对各自 Git worktree；不是当前主仓已合并接口，也不代表服务器已核实。

| Git 版本 / 文件 | 已读事实 | 对设计的影响 |
|---|---|---|
| `feat02/p` `94eb0bc`，`src/vkgr/retrieval_models.py`，`TwoTower.items` | 64D ID embedding 乘 `supported` 掩码；PAD/unsupported 的 ID 部分为零 | 不可导出未掩码的随机 embedding 表作为“纯 P 向量” |
| 同版本，`src/vkgr/retrieval_feat02.py`，`Feat02IDMean.items` | F4 不改 item tower；F6 才增加作者与内容投影；作者 PAD/OOV 经 `rows >= 2` 掩码 | F4 的 unsupported 仍为零；不能把 F6 的内容投影默认为 F4 已有能力 |
| 同版本，`src/vkgr/retrieval_feat02_run.py:168`，`export_vectors_side2` | 按 catalog/idmap 调 `model.items`，返回归一化 user/item 向量及 item 原始 norm | 可作为共享导出核心；原始 norm 与 support 必须保留 |
| 同版本，`src/vkgr/retrieval_experiment.py:503`，`normalize_export` | float32 行 L2 normalize，零行保持零 | 零行可显式保留，不得静默赋随机方向 |
| `php01/p` `587198e`，`src/vkgr/retrieval_php01_run.py:29,93,99` 与 `retrieval_php01.py` | F4-T010 属 `p_php01_train` / H1；模型由 `F4IDMeanD` 构造，绑定函数为 `binding_for3`；H1 dim=64，继承 F4 item 路径 | 不能拿旧 `p_feat02_train` adapter 直接载 F4-T010；须按真实 run kind/schema 分派 |
| `feat02/p` `94eb0bc`，`src/vkgr/retrieval_ann.py:66` | 旧 `export_vectors` 只认历史 checkpoint schema/binding、seed42 等固定条件 | 它是模式参考，不是已支持新 anchor 的通用导出器 |
| 同版本，`src/vkgr/retrieval_ablation_run.py:47`，`source_identity` | source 文件账本、Git dirty、依赖与 lock 都参与身份 | 导出输出与 scratch 放部署树之外，不改原部署字节 |
| `gtok02/g` `0e76598`，`src/vkgr/sid.py:94`、`sid_tok2_run.py` | `sid.fit` 固定输入 `(N,64)`，允许零行或单位 norm；sid_tok2 只接受 CONTENT_V1 hash、dim64 与白名单配置 | P64 已不等于 CONTENT_V1；拼接后 128D 更不能直接传入。需要独立 G-COLL 输入契约/adapter 与测试，旧入口保持原样 |

G5 的 `.agents/tasks/g5prep-design.md` rev3 仅作为既有设计参考：冻结 anchor、原部署重建 binding、best checkpoint sha、top50 逐位复核等要求可复用。该设计含旧部署特例，真实执行仍须遵守当前 Git 部署规则；本卡不复制旧 overlay 方法。其内容不等于全部裁决已落入 decision_log。

## 3. 两版本与同空间 fallback（待裁决）

定义 `C` 为冻结 train-observed catalog，`S` 为 P checkpoint 的 immutable supported buffer 映射回 item_id 的集合。定义 `p_i = normalize(model.items(idmap[i]))`；`c_i` 为同 catalog 顺序的冻结 CONTENT_V1 归一化向量。两者坐标语义不同；同为 64D 不代表可互换。PAD 行不进入 catalog，不进入聚类。

### 3.1 推荐的最小两版本（DRAFT-A）

| 版本 | 输入及维度 | unsupported 的实现 | 取舍 |
|---|---|---|---|
| `C1-PONLY` | `x_i = p_i`，64D | 保持零，记录 zero/support 标记；不读未掩码 ID 表 | 纯 P 定义清楚，但 unsupported 没有内容区分度，SID 仍需覆盖整个 C，可能出现结构性聚集；不能删掉这些 item 来改善指标 |
| `C2-PCONTENT` | `x_i = normalize(concat(p_i, c_i))`，128D，块顺序 P→content | `p_i=0` 时为 `[0,c_i]` 再归一化，和其它行处于相同 128D 空间 | 是可实现的内容 fallback；supported 时两个单位块等权，不搜索权重。总归一化后，unsupported 的 content 块幅度与 supported 不同，需如实解释 |

上述块权重与归一化顺序也待 Kiro 裁决，不能在看结果后改成“效果更好”的比例。content 零行/缺失只记真实 flag；双块都零保留零并报告，不补成真实内容。missing 的处理若与既有 CONTENT_V1 合同不同，先停下来裁决。

路线 R1 的“新视频回退到内容”在 C2 内可直接实现；C1 保零是待裁决的适用范围澄清，**不能声称已经履行内容 fallback**。不应默默用 content 的第 j 维替代 P 的第 j 维。推荐先保留纯 P 的识别性、由 C2 承担内容回退；Kiro 必须确认这种拆分是否符合 R1 原意。

### 3.2 若要求 C1 也使用内容 fallback（DRAFT-B，备选）

可在 S 上拟合内容到 P 的线性映射 `W`：输入为 `c_i`，目标为同一冻结 P 导出的 `p_i`；仅用 train-defined S。unsupported 使用 `normalize(c_i W)`，supported 仍用 `p_i`。所有行处于该 P 64D 坐标空间；在 C2 中若使用映射，也必须把替代值放 P 块、保留 content 块，不能临时改维度。

该方案可实现，但多了蒸馏/投影模型和拟合规则：损失、是否带 bias、求解器、正则或伪逆截断、数值 rank、精度、零输出规则、拟合顺序、软件版本均须在训练前冻结并给 hash。映射误差只用 train 内划分诊断，不能用 validation target 来拟合或挑超参。映射退化时不能临时退回原始 c；保零还是整版失败需先裁决。

**不推荐在当前两个名额里同时试 A 与 B 再挑结果。** B 会使“纯 P”变成“P + 内容投影 fallback”，不能再标纯 P；若当额外对照，先登记新配置名额。直接把原始 c 填入 P 坐标、静默截断 128D 到 64D、为适配旧代码临时 PCA 都不是本卡默认方案。

### 3.3 support 与 OOV 的分界

- `in_catalog & supported`、`in_catalog & unsupported`、`out_of_catalog target` 分开；unsupported 不等于新视频，train-observed 不等于 train-like-supported。
- P buffer、prepared support、train-only `item_like_counts>0` 逐 item 交叉核对，记录定义和 hash；不假定三者天然一致。有差异先定位来源，不用 validation 标签重建 support。
- out-of-catalog target 仍计原 query 分母并单列；不因 fallback 扩展 catalog，不读取 week26。对 catalog 外新 item 在线分配 SID 是另一个合同，本卡不实现。
- 17:0x 的作者 OOV“零贡献、不可训练”保持原义；不新增可学习 OOV 行，也不据此宣称 item OOV 已解决。未来若 P anchor 换为 F6 类非零 unsupported 模型，必须新版本和独立泄漏复查，不能把它当成同一 C1 复现。

## 4. 单次 P 导出、版本与 seed 绑定（DRAFT）

建议把“共用一次”定义为：**每个冻结 P run/checkpoint/source 身份只做一次 item 向量导出，C1、C2 与未来 G5 复用该不可变 bundle。** 不是不同 anchor、不同 seed 共用同一份权重。G5 仍需其独立的 user/query/top100 导出与前缀复核；本卡不建 G5 框架，不把 G5 的 validation 预测文件喂给 SID 拟合。

建议本轮固定一个 P source seed42、一个 tokenizer fit seed42，两个 C 版本共用该 P bundle；后续 G 的 seed42/2024/2025 都使用各版本的同一 SID map，仅改变 G 训练 seed。这回答“给定 P 产物时的 G 训练稳定性”，不能估计 P 训练随机性。若改为 P/G 同 seed 配对，则每个 seed 重新导出/拟合，问题与成本都变了；须先裁决，不能混报三 seed 均值。

待冻结身份必须填全：P anchor version 与 decision_ref、P run_id/seed、run manifest sha、best epoch/step/checkpoint sha、kind/schema、source 文件账本与 commit、runtime/lock；catalog/idmap/support/content/prepared/history/label-policy hashes；export adapter commit/config/hash。由导出 receipt 绑定两个 SID bundle，再由 SID manifest 绑定 G 配置、core03 模型版本、train/eval contract、G seed。任何一项变化新建版本，不覆盖旧结果。

anchor 变化处理建议：冻结前若 P 改变，由 Kiro 先决定用哪个版本；冻结后不静默跟进温度或特征升级。为新 anchor 重导出/重拟合属于新版本，保留旧版并列报告，是否算新名额先登记。G-COLL 不自动改变正式 G anchor；未来协同 SID 路作为 G5 新路也必须重新冻结收据。

## 5. 拟合、训练与防泄漏边界

拟合建议：两版本都 BHKM branching=256、depth=3，fit seed=42、max_iter=10，沿用 T3 的算法和 SID 后缀/合法性规则。新增 feature schema 接受明确的 64D/128D 与输入 lineage；不把 P hash 冒充 CONTENT_V1，不更改旧 GTOK-02 白名单。128D 距离计算需要独立 adapter，并在 64D synthetic fixture 上与旧算法对拍；模型 decoder/词表按对应 SID manifest 构建，不硬编码相同有效 vocab 大小。

泄漏说明必须分两层：

1. P 权重、support、任何 fallback 映射仅来自 train；SID fit 不打开 validation/test query、标签或逐 query 预测。P 的源特征、label/history/target 契约必须追溯至冻结证据；本卡不重定义 label policy。
2. P 的 best checkpoint 和 anchor 是经 validation 选择的，因此该 SID 虽不直接拟合 validation 标签，也**不是数据无关变换**。不能套用 decision_log 16:5x 对纯 T3 的“无标签泄漏、免 review”理由。validation 上的多层选择与赢者诅咒必须在报告写明；不得据验证集胜出宣称 test 泛化。

P 权重可能利用 train 全期 target 信息，而 G 的训练样本是较早前缀；这是 train 内的跨样本监督迁移，不能称“每个训练例子严格只用自身 target 前信息”。是否接受这种偏离及如何在报告写，由 Kiro 在冻结前明确。validation history 仍必须严格早于 target，validation 行不能进入 P 训练、support、统计、映射或 SID fit。

G 训练控制：仅替换 SID 输入/映射，core03、L50、训练样本、validation query、catalog、负反馈与 label policy、优化器、batch、early-stop、beam、collision-corrected item-level 评估均与 T3 对照一致。来源不同或无法证明一致则暂不执行。SID-level 命中不能替代 item-level 命中。

## 6. 筛选、补 seed 与采用：仅为待裁决建议

已有授权只确定“各 1 配置名额、seed42 筛选、入选补 seed”。下列数值**没有因写入本卡而冻结**；不能把温度、T1r 或 A1+T3 的专属判据直接搬作既有授权。

- 建议 seed42 入选线：相对同 seed 的 core03+T3，item-level `Δhits >= 6`；两版本各自满足就各自补 seed2024/2025，均未满足则不补。不是只挑两个里最高的；相等照同一规则处理。
- 建议三 seed“有增益线索”：同 seed 配对后均值 `Δhits >= 6` 且至少 2 个 seed 的 `Δhits >= 6`；均值 `<= -6` 标有害，其余 inconclusive。这里只是新规则提案，不已生效。不要把 T1r 的 ±12 hits 推断带用于本实验。
- 两个版本都补 seed 时两个都报告，不以较好 seed/较好版本替代整体。两个配置加 seed42 条件选择带来的多重选择限制如实写；三 seed 不是三份独立 query 数据集。
- 逐 query 的 new/lost、配对 McNemar、bootstrap CI 建议作为描述性证据；若要把显著性或校正后的 α 加成采用门槛，必须先裁决并冻结，不能事后按 p 值决定。
- “入选补 seed”“结果有增益线索”“采用正式 anchor”“纳入 G5”是四件事。采用前核对全部训练终态及科学 verify、来源/泄漏 review，再由 Kiro 判断；不因新提案满足就自动换 anchor。

## 7. 目录、manifest 与可复现证据（建议 schema）

以下都是路径占位符，尚未创建：

```text
<ARTIFACT_ROOT>/p-item-export/<P_VERSION>/<P_RUN_ID>/<EXPORT_VERSION>/
  manifest.json, ids.npy, items.npy, item_raw_norms.npy, support.npy
<ARTIFACT_ROOT>/g-coll/<CARD_VERSION>/<C1_OR_C2>/sid-fit-s42/
  feature_manifest.json, sid_manifest.json, audit.json, verify.json
<RUN_ROOT>/g-coll/<CARD_VERSION>/<C1_OR_C2>/g-s<SEED>/
  manifest.json, logs/, evaluation/, verify/
```

P bundle 保存排序唯一的 catalog item_id 与准确 idmap 对应、float32 `(N,64)`、零行/原始 norm、support mask/hash；raw embedding 表不导出。C2 feature bundle 额外记录 CONTENT_V1 sha、块顺序/权重/归一化与 128D shape。fallback B 若获选，单独保存映射权重/拟合支持集/solver/config/hash，不藏进普通 normalize 函数。

每个 manifest 应记录 schema/task/card version、parent manifests 的 sha 链、所有文件 sha256、shape/dtype、配置、seed、source commit 与逐文件 sha、runtime、创建时间、train cutoff、`split=validation`（评估部分）/`fit_scope=train_only`（拟合部分）、test gate closed、support/fallback 定义和计数。计数与 norm 诊断从真实产物生成，不预填猜测。脱敏公开摘要只放原创配置、hash、聚合审计与小型指标；完整导出、SID map、index/trie、逐 query 预测、checkpoint 不进 Git。

所有输出使用新目录与 exclusive creation；已有目标/partial 拒绝覆盖；完成后原子写 receipt。共享导出以 receipt hash 引用，不复制修改。运行目录位于部署 worktree 之外；部署依已 push commit + 逐文件 sha 校验，不写本机路径、连接信息或凭证进公开 manifest。

## 8. 实施前验证与科学 verify（本次均未运行）

### 8.1 本地工程测试建议

- fresh RED→GREEN：错误 checkpoint schema、run kind、source/config/hash、support/idmap/PAD、不同 catalog 顺序必须被拒绝；从 adapter 误走旧 P4F 入口的变异要被测试拦住。
- 手算 synthetic：C1 supported/unsupported 恰为预期，零行 normalize 不产生 NaN；C2 `[p,c]` 与 `[0,c]` 的 shape、块次序、norm 可逐元素核验；不能用 shape 正确替代语义检查。
- 64D adapter 与旧 `sid.fit` 同输入同 seed 逐位相等；128D synthetic 证明所有维度参与距离/聚类，不截断后半块；fit/verify 重放映射一致；篡改一个父 hash 必须失败。
- 固定 P export 被两个 G seed 引用仍为同 hash；P source seed、G seed、fit seed 三字段不能混淆。只改 anchor/内容/support/映射配置任一字段必须导致新版本或拒绝复用。
- test 字段、week26 路径、错误 validation week、空 history、query/catalog/history 不一致都拒绝；不通过真实 test 文件测试这条拒绝规则。

### 8.2 未来真实输入 preflight

新入口第一次上真实数据前，由主执行者按冻结授权做服务器只读 preflight：核现有 job owner/队列以免重复提交；核磁盘/RAM/GPU 至少 20% 余量；只检查本作业进程树；确认 input pins、shape/dtype/finite、catalog 完整唯一、support/零行/missing 诊断与 adapter 分派。读取真实 checkpoint/完整向量需有下一阶段授权，本次不得执行。

测量 P 导出、128D 输入的 BHKM 的 CPU/RAM/IO 瓶颈与 wall，再依 schedule 定资源；不照搬 64D fit 成本作为 128D 实测。训练、verify 都 detached，日志写各自 run；新增运维脚本先版本化，driver 间隔按现行不少于 900 秒，不新增模型 timer。正式入队前完成规定的一次 review；claimed 且无 rejected/.reason 才算接纳。

### 8.3 科学 verify 清单

1. 源 P 训练与 verify 真正通过；重建源 binding，best checkpoint sha/epoch/step 与 manifest 相符，导出正确的模型 `items()`；P exact top50 与归档预测逐位相等、指标重算相等。共用 G5 的导出验证可复用收据，但不能用旧 schema 的成功冒充新 anchor 成功。
2. 两版本逐 item catalog 一一对应；核 support 定义差异，零行/fallback 切片完整。SID fit 独立重放，审计完整性、合法 token/trie、collision/suffix、B1 等适用门禁。沿用门禁需确认其检查对象仍相同；CONTENT_V1 专属 hash 检查改绑新 feature lineage 是新契约，不能偷偷放宽旧 gate。
3. 各 G run 对应独立 verify，重算 collision-corrected item-level 指标并核 query/history/catalog hash、原始 denominator、OOV 切片。若用 batched b=1 训练，仍用串行解码 verify 且逐位相等，不新增容差。
4. 报告全量、in-catalog supported/unsupported、out-of-catalog target、零向量/fallback 覆盖、合法可达率；空切片输出 n=0 与指标不可用，不填零冒充效果。三 seed 使用同一 query 集的限制写明。

验证命令须在实现计划中绑定**实际存在的**新增测试模块与运行入口；本文不虚构可执行 CLI，也不把上面清单标为已通过。验收交付需包含命令、exit code、输出 hash、原始收据与未核限制；supervisor succeeded/exit0 不等于科学 `verified`。

## 9. Stop gates 与 Kiro 复查清单

现行硬边界：本次只写草案；任何服务器动作、checkpoint/完整 embedding/test 读取、实现、冻结或入队需求都停止当前 task。身份、数据来源、label/history/split/catalog/预算冲突时停下交主执行者；不重启、不重复 submit 原作业、不重置 spent/deadline。不删数据、不付费、不改远端历史。

下一阶段冻结前必须裁决的事项：

1. **C1 对 unsupported 保零、C2 同空间内容 fallback，还是加映射？** 若选映射，如何命名、是否占额外名额、solver/零输出策略均先定；不把新规则写成原授权。
2. **P 来源版本与 seed 绑定。** 用冻结 F4-T010 还是后续 anchor？建议固定 P seed42、fit seed42，G 补 seed 只改训练 seed；是否接受该方差范围限制？
3. **128D 新 adapter 与输入契约。** 如何独立复用 BHKM 算法、保留旧入口及门禁？哪些门禁的对象/来源需新卡明确，哪些变更必须问 Eddy？
4. **筛选和结果判定。** §6 的筛选线、三 seed 判定、采用/统计角色由谁批准并何时生效；现有各专属预注册规则保持原义。
5. **泄漏与报告。** 接受 P 经 validation 选 checkpoint/anchor、以及 full-train 协同产物相对 G 训练前缀的时间边界吗？泄漏 review 的范围与正式采用节点是什么？
6. **导出复用范围。** 每个 P 身份一次 item 导出与未来 G5 共用，G5 的 query/top100 验证另算；anchor 改变时新版本、旧结果并列与名额如何登记？

完成上述裁决并由事实源登记后，才进入独立实现计划；本草案保留 DRAFT，不主动冻结任何阈值、SID 或 anchor。
