密级：公开（路径使用占位符；不含真实数据、向量、checkpoint 或连接信息）

# SID-FIX-S1：G-COLL FINAL 冻结卡（2026-10-16）

**状态：FINAL。** 科学裁决与执行授权唯一依据为decision_log「2026-10-16 夜间：SID-FIX S0结果裁决；S1设计定稿；S1执行与自动接续授权」§4/§5；不再设计评审。S0原产物不改。当前可见 context 计数 UNKNOWN，未估算百分比。

## 1. 问题、范围与唯一来源

问题是：保持 core03 与 T3 的 BHKM 256×3 不变，只将 SID 的输入改为协同向量或协同与内容的拼接，能否改善 item-level validation recall？不是比较新 G architecture，也不按 S0 结果选择超参。

授权与科学边界按以下原文，冲突交 Kiro，不自行解释为放宽：

- `docs/decision_log.md`「2026-10-16 午间」§1–§3：pipeline、C1′ 替换、S1 两名额与 D-1 偏离。
- 同文件「2026-10-16 傍晚」§1–§4及「2026-10-16 傍晚（续）」§1–§3：S0 已冻结；S1 另卡；C2 精确读取授权；Codex 起草、Kiro 复查。
- `docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md` 全文，尤其 §2、§3、§8。
- `.agents/tasks/gtok02-design.md` §3.2（BHKM）、§3.5（门禁）、§3.8（独立 seed 判读）。
- `docs/data_contract.md`：catalog、history、split、content 合同；`.agents/tasks/seq-m1-design.md` §3.10：M1 物品塔不含用户字段与 supported 语义。

只登记 C1′、C2 两个配置，占已批 S1 两个名额。拟合不占名额；同一配置的 2024/2025 补 seed 不增加配置名额。没有第三种输入、混合权重搜索、观看比例 ablation 或额外 tokenizer。S2/S5 不包含在 S1 执行范围。

## 2. 输入、时间边界与泄漏说明

catalog 固定为原 train-observed catalog，120,415 个 item，按原 item_id 升序逐行对齐；content 仍取 T3 的同一 CONTENT_V1 来源（64 维）。不能通过丢弃缺向量 item、扩大 catalog 或换 query set 改善结果。history 仅第 0–24 周，沿原数据合同的用户内顺序；不跨用户、不把多条 query 的重复 history 当成重复事件采样。原始事件唯一性及顺序凭据在实现 preflight 中核对；无法恢复唯一事件序列或原 history 顺序合同失效则 STOP，不自行拼接去重。

C1′ 在**全部训练期**历史上一次拟合，包含某条训练样本自己的 target 及其之后的训练事件。它不是每条 query 的严格 prefix-only SID，属于已批准、必须披露的 D-1 同类 full-train 偏离。validation 第 25 周不得进入共现、频率、负采样、watch-ratio 决策或输入补全；test 第 26 周一律不读。C2 的 M1 物品参数同样学习过 train target，best checkpoint 又经过原 M1 validation selection，因此它是固定的监督迁移来源，不能称为完全不接触 validation 选型的无监督 tokenizer。不得更换 best、P seed 或来源来补救 S1 结果。

S1 的 validation 仅依冻结评估流程读取，不能据 target 分布重加权碰撞率、输入或 S0 热度边界。不得声称线上因果提升、真实上新、全量结论或消除了 D-1 风险。

## 3. C1′：冻结 item2vec，不搜索

| 项 | 冻结的唯一值 |
|---|---|
| 方法 | skip-gram with negative sampling（SGNS），两张 item 表，输出只用 input/center 表，不求和、不平均两表 |
| 维度 | 64 |
| 窗口 | 对称固定 window=5，中心两侧各最多 5 条事件；不随机缩窗，不跨用户 |
| 正样本 | 窗口内不同 item 的有向 center/context 对；相同 item 对跳过，重复事件产生的合法不同 item 对保留，不按 item pair 去重 |
| negative 数 | 每正对 5 个；分布为第 0–24 周事件 item 频次的 0.75 次方归一化 |
| 负样本碰撞 | 拒绝 center 与本对 positive context，按同一 RNG 继续抽；允许负样本之间重复，不拒绝其它已看 item |
| epochs / seed | 5 个完整 epoch；唯一 seed=42（向量与 BHKM seed 不随 G 补 seed 改变） |
| 权重 | 每个正对权重 1；不按观看比例或距离加权；不做 subsampling、min-count 截断、early stopping |
| loss | 每对 `-logsigmoid(v_center·u_context) - sum(logsigmoid(-v_center·u_negative))`，batch 内对正对求和（不取均值） |
| optimizer | SGD，无 momentum/weight decay，初始 lr=0.025，按已处理正对数跨5个epoch线性衰减至0.025×1e-4，batch=4096 正对，最后不足一批保留 |
| 初始化 / 顺序 | input 均匀 `[-0.5/64, 0.5/64]`，output 全零；用户升序、用户内按已冻结事件顺序、center 升序、context 从旧到新；每 epoch 同顺序；单 worker，负采样 RNG 连续推进 |
| 导出 | float32，center 表逐行 L2 normalize；只保留 catalog 顺序的离线 item 向量 |

理由：window=5 给局部共同观看信号；5 negatives、5 epochs 是一次有限的 SGNS 工程取值，不是依据本项目 validation 调优。64 维保持 T3 输入维度，seed42 保证一次拟合可复推。不加观看比例权重，避免再引入 watch-ratio 定义、裁剪和强度三项自由度；没有证据宣称这些取值最优。学习率与 batch 也一次固定，不能因效果差修改。

**覆盖与冷项**：词表预先含全部 train-observed item，`min_count=1`，但“有一行初始化”不等于“学到协同向量”。导出同时记录每个 item 的合法正对参与计数；无合法正对者不使用随机初始化冒充协同。定稿：该 item 使用同源 content 的单位向量，标 `content_fallback_no_pairs`；其余使用 SGNS center 单位向量。训练中任意 NaN/Inf 直接失败，不 fallback 掩盖；有正对却导出零范数也 STOP。fallback 数量与 mask/hash 必须披露，所以 C1′ 的准确描述是“协同为主、无正对项用内容补全”，不能称所有 item 都学到协同。catalog 之外的新 item 不在本卡分配 SID，validation OOV 单列，不临时训练/扩表。

该content fallback已获夜间裁决采纳。限制：协同与内容向量不在同一空间，补全视频的BHKM位置没有协同含义；须报mask、数量与比例，不称所有item都学到协同。新增工程健康检查：训练后在S0相邻不同item共现对、同数量D′随机不同视频对上各算平均余弦；共现均值必须严格大于随机均值，报告差值；不通过视为训练失败，STOP回Kiro，不改参重训。

## 4. C2：128 维输入与精确 checkpoint 授权

**来源永久固定：M1-mean seed42 的 best checkpoint。** P anchor 后续变化也不换；不使用 M1-bi、其它 seed、last 或 FEAT-04 组合。授权原文范围是：以 `weights_only=True` **只读加载这一个 checkpoint**，导出 train-observed catalog 120,415 个视频的 **64 维物品塔向量，supported mask 照原样**，**存服务器、不进 Git；不导出用户向量，不读其它 checkpoint；checkpoint、脚本、导出文件均记 sha256**。工程前置未通过前不执行该读取。禁止不安全加载后再声称等价于 `weights_only=True`；加载失败即 STOP。

物品塔必须走原 `items()` 的 item-id → embedding×supported 语义，不拼用户塔、history 或事件桶特征。导出 manifest 绑定已有 M1 best 选择证据、checkpoint SHA、catalog/id-order SHA、supported mask SHA、原 M1 版本及导出脚本 SHA。具体 best 文件路径、run/manifest 身份与 SHA 此时 UNKNOWN；导出前由唯一执行 owner 依据收据补齐，不能从文件名猜 best，不能以探索其它 checkpoint 寻找来源。

冻结的数值合同（块次序不可互换）：

1. `P_i` 为带原 supported mask 的 64 维 float32 物品塔输出；`C_i` 为同一 item 的 64 维 CONTENT_V1 内容。
2. 有限非零 `C_i`：`c_i = C_i / ||C_i||2`；原 catalog 有缺失 content、非有限或零范数则 STOP，沿原 content 缺失合同裁决，不构造假内容。
3. supported 且 `||P_i||2 > 0`：`p_i = P_i / ||P_i||2`，`x_i = concat(p_i,c_i) / ||concat(p_i,c_i)||2`。即先分别单位化，再整体单位化，两个非零块各占一半平方范数，没有可调权重。
4. unsupported 或有限零 `P_i`：`p_i = zeros(64)`，`x_i = [P0,c_i] = [zeros(64),c_i]`；supported 但零向量单独计数，mask 仍保留原值，不伪造 supported。非有限 P 不 fallback，STOP。
5. 输出 `float32[N,128]`，每行 norm≈1，列 `0:64=P`、`64:128=content`；item_id、supported 与 fallback mask 行序严格一致。原 mask 中 unsupported 对应的 P 块必须逐位为零。

分块归一化已获夜间裁决采纳：若直接拼原始 P 范数与 content，隐含权重会受 P 训练尺度影响；此处固定两块权重，便于解释。代价是 supported/unsupported 的内容块幅度不同（前者 1/√2，后者 1），要在报告说明，不再试另一种权重。

## 5. SID 合同、S1门禁与失败处理

两个配置均为 **BHKM branching=256、depth=3、fit seed=42、max_iter=10**，复用原冻结 BHKM 算法；完整 SID 仍为三位语义码加规范 suffix 的 `[N,4]`。C1′ 64 维、C2 128 维分别用新版本名/新 manifest schema 明确登记，不冒用 CONTENT_V1 的 64 维 feature array 身份，也不改旧入口白名单让旧 bundle 被误认。新的 128 维合同只扩展输入维度，不放宽数值、目录、碰撞或解码门禁。

**继承精确指针：`.agents/tasks/gtok02-design.md` §3.5（旧输入族门禁不改；本卡H5按下述H5a/H5b登记）与其引用的 `sid_tok_audit.gates(audit,'bhkm')`；BHKM 算法引用同文件 §3.2 首段。** 主仓当前未提供该模块，实施前须在对应受控 worktree 核对真实函数及基底，不能用设计卡代替代码核验。

- H1–H4、H6、B1对两个SID原样生效：完整SID碰撞0；roundtrip/reachable=N；valid-prefix deadends0；ids升序/digits范围/suffix规范；全数值有限；B1完整映射前三列每节点min(b,n)非空、大小差≤1、tolerance1。R1–R4不适用。
- H5a：输入用到的原内容来源digest必须等于CONTENT_V1。
- H5b：变换后的64维C1′/128维C2输入digest必须等于其已验收SGNS/导出manifest登记digest；manifest同时绑定CONTENT_V1、第0–24周事件序列digest、C2唯一checkpoint SHA、脚本SHA与commit。
- H5a/H5b是S1新卡/新输入族看结果前登记，不改GTOK-02六SID的既有门禁。旧loader与白名单不改，新版本/schema显式登记维度/变换。Eddy已转交“按此执行”的裁决；若Eddy另表示认为属于改门禁，拟合前暂停，工程可继续。

fit 后独立 `fit-verify` 必须重载、重算审计、核 config/input/catalog/版本绑定；再做同 seed replay 验证映射一致。仅 exit0 不准训练。若沿用 GTOK02 expected capacity 检查，按同 N、256×3 的容量推演逐整数验证，不能读取结果后改 expected 使其通过。

门禁失败是该配置 FIT_FAILED，不训练、不调参、不放宽、不占训练名额；fit-verify/replay 不一致则隔离目录、NEEDS_ATTENTION，不删除；失败 attempt 不自动重跑或覆盖。原 GTOK02 §3.5 的基础设施自动重提分支**不复制**到本卡，遵守 pipeline §8 的“不重跑”。异常交唯一 owner/Kiro 按现有授权处理，不能重新开始预算。

## 6. G 训练、筛选与最终判读（冻结）

先完成 S0 工具验收；C1′/C2 各只拟合一个 SID，三 G seed 共享该 bundle，不为补 seed 再拟合向量。core03 的 architecture、history/features、catalog、query/split、训练预算、selection、beam=100 与验证口径逐字段沿用 T3 同 seed；除新 SID/schema/task/run 身份必要字段外不改超参。B1024 仅描述性参照，不参与筛选或胜负。

每个配置流程严格为：输入与导出审计 → SID fit → **fit-verify verified** → 真实输入 `engineering=False` loader/codec/config preflight → CUDA 与真实 shape 显存测量 → core03 seed42 train → **对应 train-verify verified** → 筛选 → 如入选，core03 seed2024/2025 train，各自 verify → 最终判读。不能靠 engineering fixture 代替正式 loader 可用性证明。

筛选：只对 verified 的 seed42，用 item-level 整数 hits；相对同 seed T3 **≥+6** 才补两个 seed，即 **s42 hits≥260**。未达者不补、不挪名额给第三配置。T3 的已登记参照身份及数字回核 experiment_log；本卡数值是继承的决策阈值，不是新实验结果。

最终状态逐配置**精确沿 `.agents/tasks/gtok02-design.md` §3.8 的状态定义**：

- `Δ_s = hits(X,s) - hits(T3,s)`，只用独立 seed **2024/2025**，T3 对应参照 274/275。
- `UP`：两个 Δ 都≥+6；`DOWN`：两个 Δ 都≤−6；`FLAT`：其它全部，包括一升一平、一升一降，分歧单列。
- 缺任一独立 seed（未补或未 verified）：仅 `T_UP/T_DOWN/T_FLAT`，按 seed42 相对 254 的同一 ±6 死区，不能写确定胜负。
- 含 seed42 三 seed 均值 Δ 是次要描述，不参与最终状态；seed42 已参与筛选，不能混进主判据。

不搬用 GTOK02 的 Q1 分辨率分支表到本卡输入对比。两配置各独立报告；至少一个 UP 只是 S2 的必要前置，不自动授权 S2。若两者均 UP，不由本卡按看过的结果临时挑 S2 目标或 A1 赢家，交 Kiro 在下一卡冻结。没有 UP 则暂停 S2 返回 Kiro。S1 不自动改变 G anchor、不外推 A1。

## 7. S0 A–D 必须对每个新 SID 重算

复用 pipeline §2.2 同一套 S0 函数与冻结输入 SHA：A 结构/利用率/归一化熵/孩子数/叶规模/有效层数；B in-catalog 逐位 top1/top50 前缀准确率及完整命中，OOV 单列排除；C 原 G4 train-only 热度分档、每 seed 对 T3 的差；D 第0–24周相邻不同 item 共现对相对 catalog 均匀随机对（seed0、等数量）的 lift1/2/3，随机分母0为 UNKNOWN。C1′ window5 与 S0 D 的相邻对定义不同，诊断函数不随之改窗口。

A/D 在 fit-verify 后可出，B/C 需对应 G train-verify 后才能出。lift、有效层数、覆盖率仅描述，不新设协同阈值，不决定入选。S0 的 50% 热度读法仅继承原卡适用对象；不能变成 S1 新通过门槛。机器可读 JSON、输入 SHA 与收据保留服务器；指标只写 experiment_log，风险只写 issue_log，由唯一 owner 更新，本文件不抄实验数字。

## 8. 实施前验收与 stop gates

实现范围在执行计划逐路径列明：新增 SGNS/只读 C2 exporter、新 SID 合同与 runner/config/tests/ops，以及必要文档；不改旧 run、bundle 或冻结入口。生产实现 fresh RED→最小 GREEN→fresh regression。synthetic 验收至少覆盖：窗口/重复/跨用户边界、negative 拒绝规则与固定 seed、手算 SGNS loss、零正对 fallback、NaN/Inf 不补、C2 `[P,content]` 顺序与三类 mask/zero、128 维拒错、目录与 sha 篡改拒收、门禁负向、正式 `engineering=False` write→load→codec、fit-verify→train 调度依赖及筛选临界259/260、独立双 seed ±6 与分歧/缺证据。

正式提交前按项目要求一次独立完整 code review、冻结版本、public 扫描、工作分支 push、服务器只读 deploy key fetch 与指定 SHA worktree、逐文件 sha256 校验。manifest 记录 commit、全部数据/输入 hash、config、seed；完整向量/checkpoint/用户数据不进 Git。GPU 声明按实测×1.15 向上取整256，CPU/GPU/IO瓶颈与≥20%余量先核；数值和墙钟此时 UNKNOWN，不沿用旧显存假装实测。

所有运行 queue、detached，训练各配 verify；入队核 claimed 且不在 rejected。新 driver 版本化、PPID1、轮询≥900秒、不调用模型；不新增模型 timer。只管本作业进程树与资源，另一线不阻塞本线。停止：未冻结、输入/身份/SHA不符、test触碰、loader非正式可用、checkpoint授权越界、NaN/Inf、门禁/verify/replay失败、工程修复需改科学设计。停止不等于 kill 其它原作业，不删除、不覆盖、不重提旧 job。

## 9. 执行与自动接续授权（夜间§5）
Codex隔离实现→一次完整独立code review→Git扫描/push/固定SHA新目录部署/逐文件SHA。每配置独立：向量拟合或唯一M1导出/审计（C1含健康检查）→SID fit→fit-verify原始verified+同seed replay→三个seed真实engineering=False只读preflight→CUDA/真实shape VRAM（实测×1.15 ceil256）→启动记录大小及supervisor可读性dry。全过后可提交本配置s42，入队前告知Eddy，确认claimed/无rejected。
服务器driver先入Git/push/固定SHA部署，detached PPID1，poll≥900，无模型，创建计划起固定48h。成功train补唯一verify；s42原始verified且hits≥260自动补2024/2025与各verify；不满足结束。不加臂/seed/名额，最多6core03训练。每阶段run_id/hash/状态/epoch/R/N/hits/筛选/诊断/异常UNKNOWN机器汇总。
配置任一步失败/非verified/健康失败/NaNInf→该配置ALERT停止后续，另一项继续；共同数据合同失效两项都停回Kiro。rejected读reason不重投；失败attempt不换run_id重跑/不覆盖；设计偏离或修HIGH/MEDIUM需改科学设计回Kiro。G anchor仍A1T3；不自动S2/S5。

草稿事实纠正：原§6的beam=50为抄写错误，已按T3三个已验收manifest/config全部beam=100纠正；并非新增调参。来源为experiment_log S0最终收据及cache/codex-sid-fix-20261016/first-result.canonical.json的inventory.run_input_bindings。此项已获夜间Opus裁决，FINAL不再待设计评审。

## FINAL附记：C2报告与D′
C2必须报告：约18% supported视频含协同块，其余约82%仅内容；两类在128维空间系统性分开，BHKM首层可能先按有无协同块（近似是否被点赞）划分，这是已知设计性质不是bug。实际比例按导出审计实报，不把约数当测量。
D原实现含self-pair保留不改；D′用Random(0)，每对抽两个不同视频（对内不放回、对间可重复），数量同共现对；某前缀层每节点最多1item时lift=STRUCT_NA，其它随机分母0=UNKNOWN。D/D′并报，只描述，不作S1门槛。D′/历史别名核查/A1 epoch描述与S1工程独立，不阻塞。

## Checkpoint授权澄清（Eddy直接答复，2026-10-16）
既有checkpoint只允许上述固定M1-mean s42 best；本轮新训练与CUDA作业的verify可读取自己新产生的checkpoint。Eddy明确答复“是，允许新作业验证自己的新产物”。此项不授权补验四个旧归档、不扩大旧checkpoint范围。
