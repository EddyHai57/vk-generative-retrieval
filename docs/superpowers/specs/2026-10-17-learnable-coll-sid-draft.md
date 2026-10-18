# 可学习 SID（S2）立卡：固定共现信号，学习离散 tokenizer（FINAL）

密级：公开。状态：**FINAL（冻结）**。冻结依据：decision_log「2026-10-18：可学习 SID（S2）冻结」（Kiro，Opus 5）逐条，以及「2026-10-18：批准落账、L100 事实更正、下一批优先级重排」第 4 条（名额）；本轮按「2026-10-18 深夜：S2 四项与 S4 两项裁决」纠正四处配方，未改 R2。文件名保留 draft 以维护已有引用。本卡只抄既有裁决；历史§6的设计/名额前置已满足，不恢复为DRAFT。仅有torch-free合同前置实现，未拟合、未入队。授权源 decision_log「2026-10-17 晚间：新SID的A1确认」§5.2。

## 1. 问题与固定对照

问题：将已验证有用的C1′共现信号用于可学习tokenizer，是否比当前固定BHKM SID更适合A1生成模型？当前G anchor固定A1+共现向量SID，bundle COLL_C1_B256D3_S42；三seed完整已验收身份及指标见experiment_log 19:00最终验收与晚间anchor更新，不以旧A1+T3作为主对照，不重跑对照。

唯一允许变化的模型部件是**item→SID的tokenizer及其必要输入身份登记**。下游A1的history/features/catalog/query/split、初始化、优化器/预算/selection、beam100与温度1、evaluation/verify均沿当前anchor逐字段固定；不混入R1温度损失、DPO、多兴趣、特征、attention改动或新seed。tokenizer方法本身会同时改变量化和正则，不声称分离出某一项损失的因果贡献。

共现监督固定复用已验收C1′向量/内容回退mask及其SHA，不再训练SGNS，不读P checkpoint、不导出用户向量。已验收内容向量只作同源内容输入，不拼接新P向量，不变更C2。训练仅0–24周，validation第25周不进入tokenizer目标、负采样或checkpoint选择；test第26周不读。

## 2. 方法（已裁决）

选 **LETTER 式离线学习 tokenizer**。ETEGRec 式端到端不做，留候选清单，不新增臂或名额。

## 3. 文献来源与许可证初审（2026-10-17实际只读网页）

- [LETTER论文](https://arxiv.org/html/2405.07314v3)描述RQ-VAE、协同对齐及多样性正则；[作者仓库](https://github.com/HonghuiBao2000/LETTER)公开了实现。当前浏览的根目录/README未发现明确代码许可证；master/LICENSE返回404。**未核递归全部子目录、未钉仓库commit，许可证兼容性UNKNOWN，禁止复制源码/配置/权重。** 论文页面的传播许可不能替代软件许可证。
- [ETEGRec论文](https://arxiv.org/abs/2409.05546)将tokenization与生成推荐联训；[作者仓库](https://github.com/BishopLiu/ETEGRec)根目录/README未见明确LICENSE，所试main/LICENSE返回404；默认分支与逐文件许可尚未完整核查，兼容性UNKNOWN，同样不复制。

这是**初审而非“已通过license gate”**，404不等于证明全库无许可证。冻结前补精确commit/tree与逐文件来源审计；若没有明确兼容许可，仅基于论文公式在本项目原创实现、保留引用，不逐行翻写官方代码。引用论文不意味着使用原仓库数据/实现。

## 4. 科学合同（decision_log「2026-10-18：可学习 SID（S2）冻结」逐条；不得由执行者在看结果后补定）

1. **方法：LETTER 式离线学习 tokenizer。**
   - 输入分工：**CONTENT_V1（64 维）作重建输入；已验收的 C1′ 向量仅在 120415 个 train-observed 行上拟合固定 PCA，取 32 主成分后作协同对齐目标**。主成分矩阵记 SHA256，报告解释方差比例；投影冻结，不随训练更新，不加可学习线性头。latent 侧与导出路径沿 L2 不变。不直接量化 C1′。
   - 如实写明：相对 BHKM-C1′，本实验是**整个 tokenizer 的替换**，不是单一对齐损失的 ablation；不声称分离出量化改变与正则各自的因果贡献。
   - 许可证 UNKNOWN，只照论文公式原创实现，保留引用；**不复制任何官方源码、配置或权重**。
2. **SID 结构合同**：语义层 3、每层 256、确定性 suffix **全部保留**，与当前 anchor 可比。输入 / latent 维度、encoder / decoder 结构、初始化、死码重置**逐项沿已修 RQ-VAE L2 版本**。全部写入新 schema / 新版本名，不冒用旧 bundle 身份。
3. **损失与权重**：
   `L = L2_total + α·L_align + γ·L_div`
   - `L2_total = recon + Σcodebook + Σ(β·commit)`，包括 codebook 项，逐字沿用已修 L2；`β=0.25`不改。
   - `L_align`：straight-through 后的量化 latent → 同视频的固定 PCA32 C1′ target，单向 InfoNCE，两侧 L2 归一化，温度 0.07，批内负例；回退视频既不作 anchor 也不作负例。`α = 0.1`。
   - `L_div`：码本多样性正则，对每层码使用频率的归一化熵做惩罚（鼓励均匀），`γ = 0.01`。
   - **四个数值（0.07 / 0.1 / 0.01 与 β 沿用）都是一次写死的先验取值，不是调优结果**，本卡与报告必须写明。**不扫描、不按结果改**。论文虽扫过范围，本项目无对等调参预算，且 P/G 调参预算必须对等。
4. **拟合配方**：fit seed = 42；**Adagrad LR 0.1 + 原 L2 死码重置，batch 1024，固定 300000 步**（无早停、不按 validation 选 fit）。R2 原 `steps == 300000` 不改；AdamW/20 epoch 作废。Adagrad initial_accumulator_value=0.1、eps=1e-7，原 L2 reset_check_every=100、dead_window=500、reset_until=270000、rng_reset_offset=7919逐字沿固定 L2 源码。墙钟上限 4 小时、显存按真实 shape 实测 × 1.15 ceil256。超限即 STOP，不追加 attempt。新增两项损失未针对其重调优化器，报告必须声明。
   - 诊断（只记录、不提前终止）：每层码利用率在第 2000 步与最终步各记一次（与 RQ-VAE 塌缩修复时同一做法）。
5. **C1′ 内容回退项的处理**：约 0.2% 无共现正对的视频，其 C1′ 向量是内容回退值。**这些视频在 `L_align` 中 mask 掉，既不作 anchor 也不作批内负例**；它们仍参与重建与量化，照常分配 SID。理由是回退值不含协同信息，当成协同监督会污染目标。回退数量与比例从原审计报告实报。NaN / Inf 不补值，出现即 STOP。D-1 同类偏离继承。
6. **门禁**：H1–H6 / R1–R4 按旧 GTOK-01 原文门槛，不放宽。新输入族合同 H5a（内容来源摘要）/ H5b（新输入身份绑定）的作用对象为**本卡的新 tokenizer 输入族**，即 CONTENT_V1 + C1′ 的组合身份。fit 配自身 verify 与同 seed replay；下游三个配置的真实 preflight、CUDA 实测、dry-run、各自 verify 一律不省。
7. **G 训练与判定**：**三个 seed 42/2024/2025 无条件全跑**，不做 s42 筛选。
   - 对照：**A1 + 共现向量 SID 同 seed**（现 G anchor，388/376/384），不重训对照，不用 A1+T3。
   - 有效条件沿 decision_log 2026-10-15 16:5x 第 1 条三条：三训练 succeeded 且三 verify 直接 verified；三 seed 均值 Δhits ≥ 6；至少 2 个 seed Δhits ≥ 6。
   - **不把旧阈值移用过来**：T3 的 254、S1 的 260 都不适用；也不自行写「388 + 6」作为门槛，门槛是上面三条，对照值从 experiment_log 取。
   - 满足三条只提交 Kiro 裁决是否换 G anchor，不自动换。
8. **诊断与报告**：新 SID 必须用原 S0 函数重算 A–D 与 D′、热度分档、in_catalog / OOV 分列与前 1/2/3 位前缀报告，分母沿原口径（in-catalog 1897、OOV 623 单列）。SID 级准确率不得替代 item 级召回。所有新 verify stdout 必须含 run_id 与 manifest_sha256（修补已知缺口）。旧 G6 fixture mismatch 仍 OPEN，不靠改测试合同绕过真实 preflight。**本卡保留256叉三层，也继承第三层近乎空置的结构缺陷；若无效，不能排除「缺陷在结构而非可学习性」这一替代解释。**
9. **执行顺序与并发**：S2 的 tokenizer 拟合可与 G5 的 P top-100 导出**并行准备**（两者无共享输入），但三个 A1 训练必须在 fit-verify 直接 verified 之后才入队。GPU 与 CPU 配额允许并发，按 AGENTS §4 各自声明资源、保留余量。

## 5. 训练名额（已获批）

Eddy 已批：**1 个 tokenizer 拟合（含其 replay 审计）+ 最多 3 个 A1 训练及各 verify**（decision_log「2026-10-18：批准落账、L100 事实更正、下一批优先级重排」第 4 条）。不另训新对照、不做两个候选方法同时比较、不增加臂、seed 或重跑。显存声明须各新入口真实 shape 实测×1.15 ceil256，不继承 6656。成本参考：已验收 S5 实测 2.59–2.89 小时/seed，verify 另加约 7–8 分钟/seed，仅作参考；tokenizer 拟合墙钟以 §4 第 4 条上限 4 小时为界。总 GPU 小时不是并行墙钟。

## 6. 交Kiro的设计评审问题与停止点

请按§4逐项冻结或明确退回原因，设计评审最多两轮；先完成G5卡的裁决，再评审本卡。请同时确认原路线/名额的适用范围，列出需要Eddy批准的具体训练数量。未拿到Kiro定稿和Eddy名额前仅保持DRAFT，不实现、不拟合、不运行健康检查/吞吐探针、不读新checkpoint、不入队。出现设计偏离、共同合同失效或NaN/Inf按AGENTS停止；没有为了填满服务器自行加实验的后继。

## 7. 身份表（只来自 experiment_log 已验收条目）

**固定对照（G anchor：A1 + 共现向量 SID）**：run_id、config / manifest / verify stdout SHA、best/stop 与源码 `a5da4d6d0746ef41ca1ed93028971360dcb89547` 全部见 `docs/superpowers/specs/2026-10-17-g5-anchor-fusion-draft.md` §7，不在此复制。对照 hits：388 / 376 / 384（seed 42 / 2024 / 2025），出处 experiment_log「2026-10-17 19:00+08」。

**C1′ 协同向量输入（已验收，复用，不重新拟合）**：producer `sid-coll-c1-input-v1`，input_verify `sid-coll-c1-input_verify-v1`。
- 输入 manifest SHA256 `725fb211a342c518770623be81d490feaaec38a285c954b5b6f51f50a388e9a4`；向量文件 `vectors.npy` SHA256 `bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7`；producer manifest SHA256 `c0d20a24137711144c198626d6e70ac0b37b59a1b94b90683bfa65c98e60414c`；audit.json SHA256 `886a38d436d1b974e4be123eb93f99c97e4e342bafeb7f0291b9a4f47a157902`；input-sources.json SHA256 `2b9dbf7e414906c61091a8350d62b7feb28b595618db8c012ce15f4646626087`（出处：`main05/s1-final-metadata-20261017T044559Z.stdout` 的 input / verify artifacts，及 FINAL 卡 coll-sid-a1-confirmation-draft §1）。
- 内容回退：251 / 120415 = 0.0020844579163725447（experiment_log「2026-10-17 01:20 S1 progress」）。**回退集合改为算子式定义**（decision_log「2026-10-18：G5 导出 binding 失配的裁决；三处偏离处理；S2 回退集合定义」第 8 条），不依赖任何已存的逐 item 标记文件：回退集合 = 在第 0–24 周训练期观看历史上，按 C1′ 已冻结的共现对构造规则（window 5、同用户内、跨用户不连接）**不存在任何合法共现正对的 train-observed 视频**。该集合由已钉 SHA 的训练历史与已冻结的对构造规则确定性重算得出，不需要存储标记。闸门：重算出的数量必须等于已登记的 251（分母 120415）；不等即 STOP 回 Kiro，不取近似、不按比例凑。

**CONTENT_V1 重建输入**：array digest `766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84`（experiment_log「C2 export/input verification audit」）。

**SID 旧 bundle（仅作固定对照来源，不改、不补 fit）**：SID manifest `c53b81ddfcf8d45fd55903b30f8538372110c8e1df7376156719509994625187`，sid_version `COLL_C1_B256D3_S42`。

**已修 L2 版本来源**：RQ-VAE L2 臂（Adagrad lr 0.1 加死码重置），代码 `gtok01r/g`，公开分支 `gtok01r-pub/g@80f7eed`（experiment_log 第 524 行附近）。`L_recon`、`L_commit`、`β`、维度、结构、初始化与死码重置的具体取值一律从该固定提交读取，不凭记忆。

**S0 诊断函数**：`analysis/sid_diag_metrics.py` SHA256 `ad966940d682aa0f4e576eeacfece8cef92b2f9597fec4a41522b7e535e55dc4`（experiment_log 前缀补证节登记）。


## 8. L_div精确冻结定义（以下整节原文）

## 2026-10-19：S2 的 `L_div` 精确可微定义（Kiro，**Claude Opus 5**）

执行者正确指出：对硬 argmin 得到的码使用频率求熵，梯度几乎处处为零，无法训练。该项影响科学目标，按 AGENTS §4A 回 Kiro。本节给出冻结定义；执行者照此落卡，只抄不增。**不改任何门禁。**

### 0. 前提：`L_div` 是正则项，不是门禁

利用率 ≥80% 的 R 系门禁用的是**拟合后的硬 argmin 统计**，本裁决不触及它，因此不构成 AGENTS §2 的门禁变更。`L_div` 不需要在数值上等于门禁统计量，只需把梯度推向「使用分散」的方向。这一点须写进卡，避免把正则项的数值误读成利用率。

### 1. 采用软分配，不采用 straight-through

- **采用**：对负平方距离做 softmax 得软分配，熵作用于**批内平均软分配**。
- **不采用 straight-through**：它让前向数值等于硬统计量，看似更贴门禁，但那是在一个**计数统计量**上做直通估计，梯度偏差难以论证；而第 0 节已说明不需要前向数值与门禁一致。软分配是无偏优化它自己的目标，行为可预测。

### 2. 概率定义（冻结）

对第 ℓ 层（ℓ = 1..L，L = 3）、批内样本 i、码 j：

```
d_{i,j}^{(ℓ)} = || r_i^{(ℓ)} - e_j^{(ℓ)} ||²                （与量化器所用的同一平方距离）
p_{i,j}^{(ℓ)} = softmax_j( - d_{i,j}^{(ℓ)} / (D · τ_div) )
```

- `r_i^{(ℓ)}` **必须是前向过程中该层实际用于量化的残差**，即前序层按 L2 原样**硬量化**后得到的残差。**禁止**另建一条软量化链——那等于引入第二个模型。
- `D = 32`（latent 维度）。除以 `D` 使 softmax 的参数量级与维度无关。
- **`τ_div = 1.0`**，一次写死的先验值，**不扫描、不按结果改**。
- **诊断（只记录，不授权调参）**：在第 2000 步与最终步记录各层 `d/D` 的均值、p50、p90，用于事后判断 τ 是否严重失配。若失配，**如实报告**，任何 τ 变更须另立卡。

### 3. 熵、归一化与跨层聚合（冻结）

```
q_j^{(ℓ)} = (1/B) Σ_i p_{i,j}^{(ℓ)}                          （批内平均使用率）
H^{(ℓ)}   = - Σ_j q_j^{(ℓ)} log q_j^{(ℓ)}
L_div     = - (1/L) Σ_ℓ H^{(ℓ)} / log K                      （K = 该层码数）
```

- **逐层熵除以 `log K`**，再**对层取均值（mean，不是 sum）**。两点理由：① 除以 `log K` 后 `L_div ∈ [−1, 0]`，与码数无关；② 对层取均值使 `γ` 的含义与深度无关。合起来保证同一个 `γ = 0.01` 在 256×3 与结构匹配实验的 48×48×48 上含义相同，**两卡的 `L_div` 可直接对照**。若改成 sum，`γ` 的有效强度会随深度变化，跨卡不可比。
- 符号为负熵，最小化 `L_div` 即最大化使用熵。
- 数值稳定：`q_j` 加 `eps = 1e-12` 后取 log；`q` 不重新归一化（softmax 已保证 `Σ_j q_j = 1`）。

### 4. 梯度对象（冻结）

- 梯度**同时流向编码器（经 `r`）与码本（经 `e`），两侧均不 detach**。
- 理由：塌缩是联合现象，且死码本身是**码本侧**现象；若 detach 码本，正则项就失去了对其核心病灶的作用。
- 已考虑并否决的替代：detach 码本以减少对 L2 既有码本动力学的扰动。否决原因是会显著削弱该项的作用；且 `α·L_align` 已在扰动编码器，再做半边 detach 并不能换回「L2 原样」这一性质。该取舍如实写进卡。
- 残差链本身仍按 L2 原样（前序层硬量化、既有 straight-through 不变），`L_div` 不新增任何直通路径。

### 5. 统计口径与其它细节（冻结）

- **逐批计算，不用 EMA**。batch = 1024、K = 256 时每码期望约 4 个样本，但软分配使 `q` 为平滑平均，噪声可接受；不引入 EMA 以免增加状态与第二个超参。
- **不做掩码：全部物品都进 `L_div`**，包括约 251 个内容回退视频。理由是码使用的分散性关乎整个 catalog，回退视频同样需要 SID。**与 `L_align` 的处理不对称**（后者 mask 掉回退视频），这一不对称须在卡与报告中明确写出及其理由。
- **死码重置行为不变**，逐字沿 L2。重置触发时对 `L_div` 不做任何特殊处理；重置依据的是硬使用统计，与本项无关。

### 6. 实现验收（必须覆盖）

1. **`γ = 0` 时总损失与「L2_total + α·L_align」逐位相等**，证明 `L_div` 未污染既有项。
2. **手算小例**：固定小张量，手算 `p`、`q`、`H`、`L_div` 并断言数值；至少含「使用完全均匀 → `L_div = −1`」与「全部质量集中单码 → `L_div → 0`」两个端点。
3. **梯度存在性**：断言 `L_div` 对编码器参数与码本参数的梯度均非零（这正是硬频率熵做不到的）。
4. **深度与码数不变性**：对 (L, K) = (3, 256) 与 (3, 48) 构造等价使用分布，断言 `L_div` 相等，证明第 3 节的归一化成立。
5. **导出路径不受影响**：改变 `τ_div` 或 `γ` 不改变 SID 分配结果（SID 由硬 argmin 决定）。
6. 变异测试按 AGENTS §4A 隔离缓存；至少覆盖 sum/mean 互换、漏除 `log K`、符号取反、`q` 对错轴求平均。

### 7. 边界

- 本节只补齐卡中未写明的公式，`α = 0.1`、`γ = 0.01`、`L_align` 温度 0.07、Adagrad lr 0.1、300000 步等已冻结值**一律不动**。
- `τ_div = 1.0` 与上述三者同性质：一次写死的先验，报告须写明「四个数值均未针对本实验调优」。
- 若实现中发现本定义与 L2 的残差计算存在不可调和的冲突（例如该层实际残差在前向中不可得），**停止 S2 该支、回 Kiro**，不自行改残差链。
