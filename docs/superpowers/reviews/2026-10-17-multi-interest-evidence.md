# 多兴趣召回（M2）复查证据包

本地核对 UTC：2026-10-16T18:30:39Z。当前会话 context 计数：UNKNOWN。只读本地 cache、raw 收据与冻结源码；未访问服务器、checkpoint、test、完整 embedding，未训练或提交作业。唯一新增文件为本文，不改事实源、不作 anchor 裁决。本文是复查材料，指标主账仍为 `docs/experiment_log.md`。

## 1. 来源与边界

依据 `docs/decision_log.md` 的 2026-10-16 下午 §2、2026-10-17 本地任务安排及描述性命名；设计卡 `docs/superpowers/plans/2026-10-16-m2-multi-interest-draft.md`；统计原规则 `.agents/tasks/seq-m1-design.md` §8.1。

| 来源 | SHA256 | 原始读取 UTC |
|---|---|---|
| `cache/pgr01-p-recovery-v1/main05/sidfix-m2-resource-state-20261016T090151Z.stdout`（下称 R） | `ec8e1a6184e4ed633ca9925f1ce9569e6c8fa591cea608217236f2c6443a99da` | `2026-10-16T09:01:51.913535+00:00` |
| `cache/pgr01-p-recovery-v1/main05/codex-m2-preflight-20261016T080629Z.stdout`（下称 C） | `688b45738dc87bb651d48106821300e0970afedf6d010e5f2958b8ef188694bd` | `2026-10-16T08:07:22.229739+00:00` |

R 的 transport 为同名前缀 `.transport.json`，记录 reader SHA `6d23ba1ecd732b9a435e93d3644b1820a59e2b0c05280b9d4c539409b41f04ca`。科学 commit 为 `8688f09b9f58fc5f3b3ef3f396099403222cc206`。R 三 seed 均训练/verify succeeded、job_bound=true、raw records.status=verified、exceptions=[]。这是已归档验收状态，不代表统计和延迟完整。

结论对象仅能是“K=4 多兴趣读出及其分配训练规则”整体，不独立归因于读出。三个原 job 不重投，原预算不重置。

## 2. 固定 M1-mean 对照与 §8.1 表

M2 hits 由 R 的 `round(best_metric[0]*2520)` 重算；M1 控制来自 C 的封存控制完整副本。每行数值来源及读取 UTC 均为 R/C，未从总 hits 反推配对统计。

| seed | M2 best/stop epoch | M2 R@50 | M2 hits | M1-mean hits | Δhits | exact McNemar p | bootstrap |
|---|---|---|---|---|---|---|---|
| 42 | 10/13 | 0.175 | 441 | 466 | −25 | 0.088630250906162958 | ΔR CI [-0.021031746031746033, 0.0011904761904761906]；ΔN CI [-0.00425835978211307, 0.0033317351099699933] |
| 2024 | 7/10 | 0.1773809523809524 | 447 | 481 | −34 | 0.0044812516752525244 | ΔR CI [-0.02261904761904762, -0.004365079365079365]；ΔN CI [-0.004160194240319254, 0.00210922413294579] |
| 2025 | 6/9 | 0.1757936507936508 | 443 | 471 | −28 | 0.040363102046086119 | ΔR CI [-0.02142857142857143, -0.0007936507936507937]；ΔN CI [-0.00599688172160281, 0.0011137390625876006] |

机械重算 `D=−29`、`k₊=0`、`k₀=0`、mean ΔR@50=`−29/2520=−0.0115079365079365`。`p_pool=0.00013322720452796184`、pooled b/c=211/298；逐seed R/N bootstrap已补（下方新收据）。未新增联合bootstrap口径。

| §8.1 顺序 | 固定条件 | 机械核对 |
|---|---|---|
| 1 | D≤−6 | 满足（−29），落入相对 M1-mean 的行1；科学解读交 Kiro |
| 2 | D≥6、k₊≥2、p_pool<0.05 | 前两条件不满足，p_pool 已补，D/k₊条件仍不满足 |
| 3 | D≥6 且不满足行2 | D 条件不满足 |
| 4 | −6<D<6 | 不满足 |

F4 附报：固定 F4 hits=439/434/444（`docs/experiment_log.md`、M1设计§8），机械差为 +2/+13/−1、D=14/3、k₊=1、k₀=3，处于行4。本包未重新读取 F4 原始逐query文件，逐seed p/p_pool/bootstrap仍 UNKNOWN。附报不替换 M1-mean 主对照。

p_pool 是三 seed 相同2520 queries的 discordant b/c相加再做 exact 双侧二项检验；不是7560个独立query，偏乐观。每seed exact McNemar与bootstrap均已由02:40服务器只读补齐，不对单主臂追加×4。6 hits阈值小于已记录epoch波动，不能据此挑seed或改阈值。

## 3. 每头诊断与 g0

本节数值全部来自 R 的 best epoch diagnostics，来源 SHA/读取 UTC见§1，manifest SHA见后附表。头序0/1/2/3。serve wins是top1获胜头计数，不是top50所有位置的计数；argmax平手取最小头。

| seed | serve top1 wins（sum=2520） | hard 101–200 counts | hard fractions |
|---|---|---|---|
| 42 | [261,1244,684,331] | [8353,31040,7233,4319] | [0.16396113455687505,0.6092845225242909,0.14197664147610167,0.08477770144273236] |
| 2024 | [584,125,332,1479] | [14553,9357,16288,10747] | [0.28566100696829916,0.18366866228285406,0.31971734223181864,0.21095298851702818] |
| 2025 | [1559,796,70,95] | [17458,13362,7929,12196] | [0.3426832858965551,0.26228285405829815,0.15563843360486798,0.23939542644027872] |

各 hard 窗口实际观测100 steps，counts合计50945，四头均非零；这只给出该窗口有效兴趣数4，不证明语义互异，不新增停止条件。

| seed | median_query(max_head ratio) | 每头 ratio median | gate g |
|---|---|---|---|
| 42 | 0.8501678109169006 | [0.8233532309532166,0.7244329452514648,0.7388767004013062,0.6139976978302002] | −0.13069623708724976 |
| 2024 | 0.9572662711143494 | [0.9470769166946411,0.5645675659179688,0.5457805395126343,0.7056023478507996] | −0.1099584698677063 |
| 2025 | 0.7297130823135376 | [0.6937918066978455,0.602497935295105,0.5045404434204102,0.5004937648773193] | −0.11360976099967957 |

ratio=`‖g·r_j‖/‖m‖`，先逐query取max再跨query取median，不是先四个median再max。三seed max-ratio均在原[0.05,3]内，按原措辞“分支被使用”；max过线不能证明四头都被用上。

两两cos顺序由源码固定为(0,1)/(0,2)/(0,3)/(1,2)/(1,3)/(2,3)，是最终用户向量u的逐query cosine后跨query mean，不是r之间cos。

| seed | cos01 | cos02 | cos03 | cos12 | cos13 | cos23 |
|---|---|---|---|---|---|---|
| 42 | 0.6519098281860352 | 0.28504928946495056 | 0.7694423794746399 | 0.7190313339233398 | 0.667732834815979 | 0.7410542964935303 |
| 2024 | 0.7784329652786255 | 0.697506844997406 | 0.27267178893089294 | 0.8728426694869995 | 0.8009108901023865 | 0.7435307502746582 |
| 2025 | 0.5515298247337341 | 0.8602949976921082 | 0.9114090204238892 | 0.8920838236808777 | 0.8352240920066833 | 0.990629255771637 |

g→0 三seed R@50/hits/Δhits已在本文末节补齐。初始本地缺口原因：`seqm2_ops.snapshot`仅摘best_metric和diagnostics，没有摘`manifest.ablation_g0`。这不说明服务器未算g0；源码在原训练完成后计算并归档，verify也核对。CUDA synthetic g0不能替代科学g0。只读manifest的小metadata即可补，不需CP。

g0是同一best模型归零g；训练后m与分支共同适应，差值不能读为相对M1-mean净收益。pure/S1归因及12/259阈值不适用。两两cos只做描述，不另设collapse门槛。

## 4. 完整exact延迟

| 项 | 状态 |
|---|---|
| 用户塔 | UNKNOWN |
| 四路exact检索（各路/合计） | UNKNOWN |
| 去重重排 | UNKNOWN；当前主评估并非四头各top50后合并 |
| 合计、单query/批量、p50/p95 | UNKNOWN |
| 实测串/并行、线程、同机竞争 | UNKNOWN |

已核生产口径：`retrieval_seqm2_run.exact_max_topk`在CPU NumPy用float64 `einsum(qhd,id->qhi,optimize=False)`打四头分，先对头max，再按`(-score,item_id)`全序取50；query块32、catalog逐item_block，源码循环顺序执行。不是四个并行ANN请求。实际计时线程、竞争不能由源码推测。`export_vectors`包括用户/物品分batch生成和搬CPU，也不是单独用户塔计时证据。

最终50位置、训练预算相同，但每物品4路打分，比M1多3个查询向量。分路优化若启用，四头各top50→最多200候选→去重→按四头最大分重排取50，必须在同tie-break下与full-catalog max逐项一致，并保留重叠/平手/各头独有synthetic oracle。没有ANN口径。不能拿M1 user-only延迟或CUDA engineering wall替代M2完整延迟。

## 5. 可只读补的工程步骤

1. 唯一server owner通过既有只读transport读取后附三个固定SHA的M2 `out/manifest.json`，核source/config/data/control；从best evaluation取得预测ref与SHA。由best epoch推导的期望文件名分别为`predictions-epoch-010.json`/`007`/`006`，实际SHA目前UNKNOWN，必须从manifest取得。读取`ablation_g0`小metadata补g0，不读CP。
2. 在服务器端SHA核对三份M2 best预测、封存M1三份预测、共同validation query，核query身份、目标和catalog；配对0/1 hits，输出b/c/共同中/共同未中、每seed exact双侧McNemar、pooled b/c与p。沿既有`analysis/feat04combo_paired_read.py`归档绑定模式准备独立只读工具，不覆盖旧文件、不把逐query预测进Git。F4附表另核其原始同seed预测。
3. bootstrap须记录resample单位、重复数、RNG seed和CI算法。逐seed配对query bootstrap不估seed或总体不确定性；若给三seed均值的query bootstrap，同一抽样query索引同时用于三seed以保留依赖，不把7560行当独立样本。bootstrap不新增门槛。
4. 延迟先只读盘点OPS summary、各run manifest与既有timing小文件。无记录继续UNKNOWN。实际forward计时通常需要模型载体，本次没有旧M2 checkpoint读取许可；“只读补证”不能自行扩成重开旧CP或完整embedding/用户向量导出。需要测量时列精确CP路径/范围，由主owner取得授权，再版本化工程与小作业准入，不重跑训练。
5. 测量必须统一M1/M2硬件、threads、batch、真实history长度分布、50/1000/512/1口径、warmup/同步/重复数与竞争；分别记录用户塔、exact打分/选取、合并、总wall。阶段p50之和不冒充总p50。串行/并行分别实测；full-max无独立dedup阶段应写不适用并给全流程时间，不填0伪造测量。

扫描范围为本机`cache/codex-m2-execution-20261016`、`cache/codex-sid-fix-20261016`、`cache/pgr01-p-recovery-v1/main05`中匹配M2 run身份的JSON/stdout/报告，以及冻结M2源码。没有声称穷尽服务器归档。剩余为证据缺口，不是训练缺口。

## 6. 归档身份与源码追溯

M2归档根为`/workspace/vkgr-infra01-root/runs/p-seqm2-k4-s<seed>/out/manifest.json`；各verify stdout为相应`-verify/logs/stdout.log`。下列SHA均来自R（读取UTC见§1）。

| seed | M2 manifest SHA256 | verify stdout SHA256 |
|---|---|---|
| 42 | `a07faf79e00983a450750cde78c22934ea0dbb46516beeb4006b925755a144d8` | `2106f66715e9e6313d9cc9cfc5d86efa211e3b082bc399e2f884c2df006b48a9` |
| 2024 | `31ac1df62299e4543b167a46b4596797f1b127cf05219c83ab10ebecf7d4d0d9` | `0422a9b5d324ff79b3d37c51fbfbff09240ed2bea9b98589dc3294f211aaf591` |
| 2025 | `b41e32a451a64030eff0fdc8da0714d92c3118ad4f0aab0a489b3638eb21fea5` | `2d3da44e217c19f10a3cb36a597840b83025d0d6d3b8ece0f23c41497e12aa23` |

封存控制在`/workspace/vkgr-seqm2-ops-v1/m1-controls-v1.json`，SHA `9dc09a9b71026ffdccbb8ff6a57c43b09c8b478136cc71af16a65a516fc7472e`；完整副本在C，包含source/data/input/history/verifier绑定。M1 best预测根为`/workspace/vkgr-infra01-root/runs/p-seqm1-mean-s<seed>/out/`。

| seed | 预测文件 | M1 manifest SHA256 | prediction SHA256 |
|---|---|---|---|
| 42 | `predictions-epoch-012.json` | `bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e` | `1ddd6e8eb49e9e743d6a0ded9b32de65b0ef95f0b6bedd6dbf2b3b3ed26be6fd` |
| 2024 | `predictions-epoch-007.json` | `8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8` | `c9f1c6552856f613840759295bd3fc0b8dc44e26f00e53dc38a48caf4ef48f3d` |
| 2025 | `predictions-epoch-011.json` | `783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb` | `392a1ded600b869e7609acd75381b129d88b67b8c590570570715c4e286e1b90` |

C三控制逻辑query SHA均`c703bbcc3e3004a58a2a14c2c7c2420b51b3b7ae0c47f6b284d587e67819ab1a`，不是输入物理文件SHA，不混用。

本地冻结源码读取UTC见文首；路径前缀`<WORKTREES>/seqm2-causal-k4/vk-generative-retrieval/`。

| 源码 | SHA256 |
|---|---|
| `src/vkgr/retrieval_seqm2.py` | `eb2a33b22baf13039a5282e88f89544b1db951ac0c5d4af03f53a8f4fcfba0fc` |
| `src/vkgr/retrieval_seqm2_run.py` | `4e3672e4adb0519faf1a189b252b26a4017a5acf82521ea5c1b009ab20892dd0` |


## 02:40服务器只读补数（当前，覆盖前文配对与g0 UNKNOWN）

来源main05/acceptance-paired-20261016T183959Z.stdout；SHA256 4a283900c081c0ee488a6fbfda16b16715eea487397559237eb10f3a6f023b2f；实际读取UTC 2026-10-16T18:40:00.703469+00:00，结束UTC 2026-10-16T18:40:04.197577+00:00。

| seed | 新/对照 hits | Δhits | b/c | exact McNemar p | ΔR@50 95% CI | ΔN@50 95% CI |
|---|---|---|---|---|---|---|
| 42 | 441/466 | -25 | 87/112 | 0.088630250906162958 | [-0.021031746031746033, 0.0011904761904761906] | [-0.00425835978211307, 0.0033317351099699933] |
| 2024 | 447/481 | -34 | 51/85 | 0.0044812516752525244 | [-0.02261904761904762, -0.004365079365079365] | [-0.004160194240319254, 0.00210922413294579] |
| 2025 | 443/471 | -28 | 73/101 | 0.040363102046086119 | [-0.02142857142857143, -0.0007936507936507937] | [-0.00599688172160281, 0.0011137390625876006] |

p_pool=0.00013322720452796184（Σb/Σc=211/298）；mean Δhits=-29。

原表逐seed p/CI与p_pool缺口已补；原D=-29、k+=0及行1机械位置不变。g0和完整延迟盘点见experiment_log同轮metadata节；g0数值已补，完整延迟仍UNKNOWN。


### 原始g0补数与延迟盘点

| seed | best hits | g0 hits（R×2520） | best−g0 hits | g0 R@50 | g0 N@50 |
|---|---|---|---|---|---|
| 42 | 441 | 328 | +113 | 0.13015873015873017 | 0.037281042642595297 |
| 2024 | 447 | 290 | +157 | 0.11507936507936507 | 0.032299619022277411 |
| 2025 | 443 | 319 | +124 | 0.12658730158730158 | 0.034305321732547495 |

来源main05/multi-interest-meta-20261016T184013Z.stdout，SHA256 251a2fd69af94b4f5605bf0266f0e798564d7e158dcd5a68757987f84df3c26b，实际读取UTC 2026-10-16T18:40:14.237881+00:00。g0缺口已补；完整用户塔+四路exact+去重重排+合计延迟未找到，只读有界直层盘点非穷尽；保持UNKNOWN，未打开checkpoint。该限制须交Kiro，不用实现顺序冒充实测串/并行。
