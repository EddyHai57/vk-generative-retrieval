# 双向序列模型确认（bi 另卡）设计草稿

密级：公开。日期：2026-10-17。状态：**DRAFT，待 Kiro 冻结；不授权入队。**

## 1. 目标与范围

在相同的新 seed 上比较双向历史 attention（bi）与因果历史 attention（mean），确认探索中出现的 mask 差异是否复现。两臂都使用序列用户塔（M1）的均值读出、`u=m+g·r`，唯一模型差异是 attention mask。这里的“双向”仅在严格 target 前的历史内部双向，不允许看 target 或未来事件。

依据 `docs/decision_log.md`「2026-10-17：S1 入队后的并行本地任务安排」§3及「任务改用描述性名称」。本卡占已批的序列模型第 3 个名额，不新增名额：**2 臂 × 3 新 seed = 6 个训练，各自配一个 verify**。三个 seed 无条件跑完，不用首 seed 筛选；不复用旧 seed 的 mean 结果作为主对照。当前仅交草稿，不冻结、不部署、不入队、不读取 checkpoint 或 test。

## 2. 固定 seed 与证据边界

草稿固定 seed：**314159、271828、161803**。每值同时用于 bi 与 mean；不得因结果不佳换值或补值。

本地于 2026-10-17 用 `rg --hidden -n '314159|271828|161803'` 搜索 `configs/`、`docs/`、`.agents/tasks/`、`src/` 中 json/yaml/md/py，未匹配；另用 `git grep -n -E '314159|271828|161803'` 搜索全部本地 `refs/heads` 下的 `configs/`、`docs/`、`.agents/tasks/`（该次 Git 检索排除 experiment_log，工作区 experiment_log 已由前次 rg 覆盖），未匹配。原序列用户塔固定配置的 seed 为 42/2024/2025。

这只证明**所查本地文件与当前本地分支 tip 内未见使用**，没有穷尽所有历史 commit、删除文件、远端未 fetch 分支或服务器旧作业，不能宣称全局从未使用。Kiro 冻结前须核历史 seed ledger/既有收据；若发现碰撞，停在草稿，由 Kiro 在任何结果出现前重新登记三值，不自行换值。

## 3. 模型与训练合同

代码基准：本地 `seq/m1` tip `04d4edeb84826d4d56387f95dc820fda1dce10b2` 的 `src/vkgr/retrieval_seqm1.py`、`src/vkgr/retrieval_seqm1_run.py`、`configs/seqm1/p-seqm1-mean-s42.json`；正式冻结须绑定最后经审阅的执行 SHA，不以浮动 branch 代替。

- `ARMS` 中 mean=`(causal,mean,True)`，bi=`(bi,mean,True)`；相同参数数与初始化抽取顺序，g 初值 0；共享参数在同 seed 下逐位相等。两臂第 0 步均等于对应 seed 的 F4-T010 用户塔。
- 保留 dim64、4 heads、FFN256、dropout0.1、原序列层数/位置表/末层归一化。保留 F3 时长/比例桶、F4 位置/周间隔桶，物品塔及 supported 规则不变，不加入多兴趣或新画像。
- 逐字段继承原配置：HIST_V2_OBS L50、HIST_V2_SIDE_V1、N1 in-batch + logQ α=1、extra_uniform=0、T0.10、batch512、LR0.001、AdamW/weight_decay0.0001、gradient_clip1、fp32、small 初始化、max/min/patience=100/5/3、best_last、closed_hash_only。保留相同 train/history/catalog/query/split hashes，validation n=2520。
- 原 runner 将配置投影给旧 php 校验器，旧 seed 白名单仅含42/2024/2025。**只改配置不能运行本卡**。后续需独立入口/显式新 seed 校验与 binding，不能放宽原冻结入口白名单；当前不实现。run/config 命名和 schema、最终 commit/hash 由冻结实现材料补齐，不允许用现有 run_id 覆盖。

## 4. 判定：原序列用户塔 §8.1 原文及本卡绑定

来源：`.agents/tasks/seq-m1-design.md` §8、§8.1。下面四行条件与措辞逐字保留，避免改阈值；它们是原文引用，**不表示本卡比较 F4-T010**：

| 行 | 条件 | 措辞 |
|---|---|---|
| 1 | `D ≤ −6` | 「M1-mean 劣于 F4-T010」 |
| 2 | `D ≥ 6`，`k₊ ≥ 2`，`p_pool < 0.05` | 「M1-mean 在对齐条件下优于 F4-T010（有效）」 |
| 3 | `D ≥ 6`，不满足行 2 | 「均值过线但 seed 间不一致或合并检验不显著，增益不稳」 |
| 4 | `−6 < D < 6` | 「未检出与 F4-T010 的差异」；**不得写成"序列建模没有价值"**，也不得写成"等价" |

本卡仅将实验对象绑定为“bi”、对照对象绑定为“同新 seed 的 M1-mean”；阈值、判定顺序、统计定义与限制不变。`H(X,s)=round(R@50×2520)`，`ΔH(s)=H(bi,s)−H(mean,s)`，D 为三个差值均值，k₊为差值≥6的seed数。报告对应对象措辞为「bi 劣于 M1-mean」「bi 在对齐条件下优于 M1-mean（有效）」「均值过线但 seed 间不一致或合并检验不显著，增益不稳」「未检出与 M1-mean 的差异」；不得写等价或序列建模没有价值。

p_pool：三个 seed 的逐 query 配对 discordant b/c 分别相加，做 exact 双侧二项检验。三个 seed 评同一批 query，不独立，p_pool 偏乐观，是必要条件，不是证据强度；单一预注册主比较不做旧探索臂的×4校正。逐 seed 报 b/c、exact McNemar p、Δhits、R@50、N@50，另报配对 bootstrap（方法沿原统计工具，冻结时绑定脚本与重采样 seed，不按区间择优）。单 seed 只给方向，缺 seed 或非 verified 不出完整结论。

转正逐字继承原要求的条件结构：行2成立，至少2seed逐seed exact McNemar p<0.05，新模型代码一次独立泄漏review，延迟报告已附；通过后另写decision_log，不自动切anchor。对象替换及本节是否完整沿用由Kiro冻结确认；当前不作转正裁决。

## 5. 验收、限制与停止

冻结后实现需 fresh RED→GREEN→回归，保留 target/未来字段隔离、PAD宽度不变、候选侧独立、同seed初始化相等、零起点、test运行期读取拦截与mutation、train/export/verify一致和hash绑定。因果扰动守卫只用于mean；bi应有历史内双向敏感性oracle，不能拿“不许较晚历史影响较早位置”的因果断言拒绝bi。两者都不能读取target之后事件。

按原§8.5报告best epoch分支ratio及g，沿0.05/3描述性措辞，不新设性能门槛；按原§3.11同输入hash报告CPU单线程/A6000用户塔batch1与离线摊销延迟，比较bi与mean，注明非端到端成本、混跑限制。保留I-012文件内顺序证据限制，不声称真实线上时序已证。

正式入队前仍需一次独立code review、冻结、public扫描、push工作分支、固定SHA新目录部署/逐文件SHA、真实engineering=False preflight、CUDA预演、真实shape显存实测与准入/dry-run。预算、资源和manifest最终SHA未冻结，不能把旧声明直接搬为新实测。失败attempt保留，不自动重跑/换seed，不碰原作业；共同数据/hash/顺序失效、NaN/Inf、verify失败均STOP回Kiro。不存在额外训练、额外臂、test、旧checkpoint或自动衔接其它实验授权。

待Kiro：核新seed历史唯一性；冻结对象绑定与完整统计/延迟合同；核实现入口及最终config/commit/hash、finite资源包络。**本文到草稿为止。**
