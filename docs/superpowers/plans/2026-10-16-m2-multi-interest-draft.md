# M2: K=4 causal multi-interest - FINAL design

Design finalized by Kiro / Claude Opus 5.5 on 2026-10-16 afternoon, decision_log latest section. Design review round 1 closes design; no second design review. Code freeze still requires one independent code review. Implementation and conditional three-seed admission are explicitly authorized by Eddy and the decision. Codex is the sole execution owner.

This file replaces the earlier DRAFT; historical proposals are superseded by the verbatim ruling below. Only one arm, seeds 42/2024/2025, each train plus verify. Main control: M1-mean same seed; F4 remains secondary. No old checkpoint, test or ANN.

## Architecture. 固定部分与多兴趣向量

继承 M1-mean 的事件输入、初始化与训练条件：F3/F4 的 T/R/P/G、supported 处理、L50、back-index O、1 层 pre-LN Transformer（dim64、4 attention heads、FF256、ReLU、dropout0.1）、末端 LN（weight 初值0.1）、严格因果 mask、物品塔与 N1 in-batch + logQ α=1、T0.10、batch512、LR1e-3、AdamW wd1e-4、clip1.0、预算100/5/3、相同 seed 与数据绑定。额外参数的抽取放在所有共享参数初始化之后，防止改变共享表与骨干同 seed 的抽样。模型从共同初始化重新训练，**不加载 M1 最佳 checkpoint**；M2 的 m 路径随后也会训练，不能称全程等于 M1 或 F4。

对有效历史事件 `e_t`：`m = mean_t(e_t)` 仍直接复用 anchor 均值路径；因果骨干得到 `z_t`。只替换 readout：

```text
S[j,t] = W2[j,:] tanh(W1 z_t + b1)
A[j,:] = softmax_over_valid_history_positions(S[j,:])
r_j = Σ_t A[j,t] z_t                  j=0..3
u_j = m + g r_j                       同一个标量 g，精确零初始化
score(q,i) = max_j cosine(u_j, v_i)   j=0..3
```

W1 为64×64、b1 为64、W2 为4×64（无 b2），额外参数4,416；W2 使用固定 std0.02 非零初始化，W1/b1 沿 M1 attn 的初始化约定，具体代码与抽取顺序须在冻结前绑定。softmax 沿历史位置，不能沿兴趣维；padding 在 softmax 前 mask，读出先 `where(valid,z,0)`，屏蔽 NaN，所有有效向量必须 finite。每条合法 query 至少一个有效事件，空历史沿现有合同拒绝，不造新 fallback。

第0步四个 `u_j` 都必须与 F4 的 `m` 逐位相等，eval/train 模式均验证；有限 `r_j` 下成立。此时 `g` 可有梯度，所有分支/readout 参数的任务梯度精确为0是预期；不能为“每头有梯度”把 g 初值改为非零。第1次 g 更新后才检查 readout 梯度。原始 attention 可非对称，但第0步用户向量必然重合。只有一个 g，不为每头增加 gate。


## Authoritative five rulings (verbatim from decision_log)

### 2. M2 设计五项裁决（本条即设计评审第 1 轮；按以下修改后设计定稿，不再开第 2 轮）
1. **N10 训练法：采纳草稿 §3 的唯一方案**——前 100 个 optimizer step 用 detach 的 target 软权重（温度 0.10）对逐头 N1 loss 加权求和（不是先平均四个 u），从第 101 步起固定 hard argmax（平手取最小 j）。100 步约为 0.59 个 epoch（86,785 / 512 ≈ 170 步/epoch），包含在原预算内。**必须的措辞限制**：M2 相对 M1-mean 改了两处——读出（K=4 多兴趣）与训练时的兴趣分配规则——结论只能写“K=4 多兴趣读出及其分配训练规则”整体的效果，不能归因到读出单独一项。
2. **守卫与停止条件：给定唯一替代——hard 窗口的覆盖检查改为诊断，不是停止条件。** 理由：附录 A 原守卫的作用是“不满足就改成 soft 起步”，这个补救现在已经内置；在它之后再把“某头没被选中”设成停止条件，会造成缺 seed，并带来重跑的压力，而某头饿死本身不是数据或泄漏错误，是一种需要报告的结果（有效兴趣数 < 4）。所以：第 101–200 步每头被选比例照常记录；为 0 的头在报告里标“该 seed 有效兴趣数 < 4”，训练继续。**唯一的停止条件是数值异常**（loss、梯度或用户向量出现 NaN/Inf）：停止该 attempt、保留证据、交 Kiro，不重试、不延长 warmup、不换 seed。这不改任何 Eddy 确认过的门禁，也不新增性能门槛。
3. **检索与公平性：采纳 full-catalog exact max top50 为唯一主评估。** 分路 exact（每头 top50 → 去重 → 按四头最大分重排 → 全局 50）只能作为实现优化，前提是与 full-catalog max 在同一全序 tie-break 下逐项相同，并有 synthetic oracle（重叠、平手、各头独有项）。报告必须公开：最终 50 个位置与训练预算相同，但打分是 4 路、分路中间候选最多 200（M1 为 50）。**本卡不做 ANN**，不另报 ANN 口径，也不立 ANN 调参臂。这是多兴趣召回的通行评估方式（每个物品取多兴趣最大分），不是给 M2 额外优待，但“多了 3 个查询向量”的事实要写明。
4. **分支诊断：采纳**每 query 取 `max_j ‖g·r_j‖ / ‖m‖`、再跨 query 取中位数，best epoch，沿 0.05 / 3 原阈值与 §8.5 原措辞；另报每头的 ratio 中位数、serve 阶段 top1 获胜头计数、g→0 同模型消融。这些对 M2 全是描述性诊断：M2 之后没有被它触发的实验，不构成门禁，不需 Eddy 批准。必须写明：max 过线不能证明四头都被用上；g→0 的差值不能读成相对 M1-mean 的净收益（与 M1 的共同适应限制相同）。pure / S1 归因及其 12、259 阈值不适用。
5. **冻结前实现细节（定稿要求）**：
   - 共享参数（与 M1-mean 完全相同的那部分）按 M1-mean 的同一抽取顺序与同一 seed 先抽；W1、b1、W2 之后再抽。测试：同 seed 下共享参数的初始 state_dict 与 M1-mean 逐位相等。
   - W1、b1 沿用 M1 attn 臂里 `W_a`、`b_a` 的初始化代码（冻结卡里写出源码位置），W2 用 std 0.02 的正态、无 bias。
   - 零起点：第 0 步四个 u_j 都与 M1-mean 的 `m` 逐位相等（train / eval 两种模式）；第 0 步的 L_soft 与 M1-mean 第 0 步 loss 的差 ≤ 1e-6（四个相同值求均值的浮点误差，实测值写进收据）；分支参数梯度为 0、g 的梯度非零。
   - 必须的测试：逐头 soft / hard 的手算 oracle；g ≠ 0 时 soft 阶段每头 W2 行梯度有限且非零（synthetic 非退化夹具）；翻转 target 字段后四个 u 逐位不变；因果扰动（只改未来位置，较早 z 不变）；padding 宽度不变性；运行期 test 读取拦截及其变异体；全 unsupported 历史、n=1、n=50。
   - M1-mean 三 seed 的逐 query 预测必须可读、可与 M2 配对（同 SEQM1-000 的做法核对一次），代码与数据绑定写进 manifest。
   - 统计沿草稿 §6（即 M1 §8.1 的表；p_pool 偏乐观的限制照写；不对单主臂做 ×4）。转正要求不变。


## Leakage and OOV. 泄漏、OOV与不变量

继承 M1 L1–L9及逐事件oracle：历史严格早于target，只读历史自己行的item/timespent/duration/week；P/G/O锚于最后历史事件；支持集、logQ、label策略只由train冻结；validation仅用于既有早停/评估，test仍 `closed_hash_only`。不改变数据合同或宣称关闭I-012 target-prefix顺序证据风险。因果测试固定n/week，只扰动未来位置的item/timespent/duration，较早z不变；改最后week导致早位置G变化属于既有语义。

unsupported历史的id分量仍为0，T/R/P/G继续参与，不新增UNK token；item塔维持supported mask与原catalog规则，不用validation/test扩充embedding或频次。覆盖n=1、n=50、全unsupported、不同padding宽度、无效id/side输入和候选塔不依赖用户的测试。只读训练assignment允许target；用户塔签名仅 `(histories, side)`，训练和serve接口分离，test读取运行时拦截并配变异体。


## Statistics. 判定、统计与诊断（沿 M1 §8，不加门槛）

M2主比较令 `B(s)=M1-mean(s)`，`H=round(R@50×2520)`、`Δ_s=H(M2,s)−B(s)`、`D=mean_s Δ_s`、`k₊=#(Δ_s≥6)`、`k₀=#(Δ_s≥−5)`。主结论依次：D≤−6 →“劣于M1-mean”；D≥6且k₊≥2且p_pool<0.05 →“对齐条件下优于M1-mean”；D≥6但未满足后两项 →“均值过线但增益不稳”；−6<D<6 →“未检出差异”，不能称等价或多兴趣无价值。F4对照单独用原A(s)执行同表，不替换主结论。

p_pool为三seed逐query配对discordant b/c相加的exact双侧二项检验；每seed另报exact McNemar p。三个seed重复评相同2520 queries，p_pool偏乐观，是必要条件；不宣称独立重复样本，不对单主臂追加×4校正（M1探索性四臂的×4仍仅属于其历史比较）。三seed一律完成，不按seed42挑补seed。门槛6小于噪声的限制仍写入报告。

若将来提出转正，继承§8.1的额外要求：上述有效行、至少2seed exact McNemar p<0.05、一次独立泄漏review、延迟报告，再由Kiro单独记决策；本卡不授权自动切anchor。§8.2 的描述性追平仍为D>−6且k₀≥2、分支启用seed≥2，不作为新实验接续授权。


## Frozen implementation references

W1/b1 copy the initialization behavior of `src/vkgr/retrieval_seqm1.py:SeqM1IDMean.add_branch`, attn branch `self.attn_proj = torch.nn.Linear(DIM, DIM)` (weight+bias PyTorch Linear defaults); W2 has shape[4,64], no bias, normal std0.02. Shared parameters and their draw order first exactly match `make_seqm1_model(..., arm=mean)`; additional W1/b1/W2 draws follow. See the verbatim ruling for zero-step loss tolerance, finite checks and the diagnostic-only hard window.

The numerical/performance source is docs/experiment_log.md. No new result is claimed in this card. M2 conclusions concern K4 readout PLUS its assignment training rule, not readout alone. Branch ratio/g0/head coverage are descriptive diagnostics, not gates. Exact only; no ANN. Full-shape VRAM/wall time UNKNOWN until measured. Future latency includes users, four exact retrieval routes, dedup/rerank and total, with shared machine/input methodology and contention disclosure.

## Execution authority and stop gates (verbatim)

### 3. 执行授权（逐项）
- **允许立即实现**：Codex 在隔离 worktree 按 fresh RED → GREEN 实现模型、runner、三个配置（只差 seed 与 run_id）、verify、诊断与驱动；M1 原冻结代码不动。不等当前 G 训练结束。
- **允许冻结**：完整产物做一次独立 code review（提交正式作业前的不可逆节点，规则 8）APPROVED 后冻结。内部修复不重复 review。
- **允许条件提交**：冻结后，扫描 → 工作分支 push → 固定 SHA 新目录部署 → 逐文件 SHA → 三个配置的真实输入 engineering=False 只读 preflight → CUDA 预演（soft→hard 切换、batch512×L50×K4、优化器状态、K4×全 catalog 分块 max 评估的最坏 shape、verify 重放与 g0 消融）→ 真实 shape 显存实测（声明 = 实测 × 1.15 向上取整 256，不沿用 M1 的 1024）→ dry-run（含最终启动记录大小与 supervisor 可读性，B1024 的教训）。**全部通过后，允许一次性提交三个 seed 并由服务器侧驱动自动补 verify，不需要再回 Kiro**；入队前在汇报里告诉 Eddy。三个 seed 一律跑完，不按 seed42 的结果决定后两个。
- **允许与当前 G 并行**：按 8.19 核与 39,312 MiB 准入（保留余量）由 scheduler 放行；资源不够就预排，不侵占余量，不伪称 claimed。延迟测量若与其它作业混跑，要标注不可比。
- **需要回到 Kiro 的情况**：实现中发现设计需要改（任何偏离本条第 2 节）；任何 attempt 出现 NaN/Inf；独立 code review 给出 HIGH/MEDIUM 而修复会改设计。
- **不变**：不增加臂、seed 或名额；不读 test；不改 U1/C9 的决定；不读旧 checkpoint 或导出完整 embedding；不删/覆盖、不付费、不 force push、不 push main、不改历史。M2 结束后只出结果、诊断和完整延迟（用户塔、四路检索、去重重排、合计，写明串/并行与 exact），解读与 anchor 交 Kiro。不自动接 M3、bi 另卡、G-COLL、F7 或 G5。
