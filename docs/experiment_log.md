密级: 公开

固定monitor恢复真实验收：UTC epoch 1791417548.7548192，单次SSH exit0，errors[]，script SHA `9fafb51d0bc831f8c783d5a3bb003703734dccb75e1da490e18bd714e2ac1c11`。slot01实际train/verify exit0，slot02 train exit0、原verifier1014729身份匹配state R；最新manifest均5100 steps/30 evaluations。原owner660147/dispatcher660156匹配存活非Z，final completion/summary/dispatch均未出现。瞬时剩余RAM .2600622273632446、VRAM .9877492877492877、disk .652936767578125；剩余原deadline 161526.53572773933s、联合bytes31396753349。仅瞬时余量，不称全历史minimum/peak。证据cache/p4e-monitor-recovery-20261007t2358.json，未追加SSH/replay或运行修改。

监控端schema修复验证（本地，非实验run）：冻结`cache/p4e-operation-runner.py` SHA `479d4afff029ca068d18a6091dd86b098211d4985d360fb2f840d83475b42f7a`，invoke写`exit`。target命令：`<LOCAL_VENV_WORKTREE>/.venv/bin/python -B cache/test_p4e_readonly_monitor.py`。fresh RED：4 tests，exit 1，actual exit fixture复现KeyError `exit_code`；fresh GREEN：4 tests，exit 0。覆盖actual exit=0、非零=7保留、missing及None/bool/string拒绝。固定观察脚本保存原只读snapshot逻辑，恢复原watch pin读取；未启动SSH或任何新run。真实恢复验收待下一次既有heartbeat。

# Experiment Log

## 当前状态

P0～P7已正式验收；P7统一固定audit比较已核算并通过独立review与协调验收，当前状态只读`../STATE.md`。P3数据审计、P4传统baseline、P5 tokenizer fit和P6真实small TIGER-style smoke分别记录如下；历史awaiting口径只代表当时状态。

公司内部信息、数据、schema、代码与实验结果不得进入本项目。

本文件是人工维护的实验与指标唯一摘要源。

原始 config、metric JSON、日志与大 artifact 不复制到本文，只记录路径别名与 hash。

## Run Summary

| Run ID | Status | Dataset | Split | Catalog | SID | Model | Seed | Hardware | Metrics | Resources | Artifact hashes | Claim boundary |
|---|---|---|---|---|---|---|---:|---|---|---|---|---|

下方分阶段记录保留来源与历史过程；P7统一表为当前固定audit质量比较入口，不预填未来实验或收益。

## Run Record Template

### Identity

- Run ID:
- Date:
- Status: `planned | running | completed | failed | invalidated`
- Git commit:
- Config path:
- Parent run:
- Purpose:

### Dataset

- Dataset/subset:
- Source manifest hash:
- Processed dataset hash:
- Schema version:
- Row count:
- User count:
- Item count:
- Week range:
- Feedback policy version:
- Duplicate policy version:

### Split

- Split manifest hash:
- Train weeks:
- Validation weeks:
- Test/holdout weeks:
- Query set hash:
- Query count:
- History rule:
- Target rule:
- Holdout usage boundary:

### Catalog

- Catalog snapshot ID:
- Catalog cutoff:
- Train-observed item count:
- First-seen/OOV target count:
- Feature coverage:
- Catalog hash:

### SID

- Tokenizer name/version:
- Codebook config:
- SID depth/vocabulary:
- Collision suffix rule:
- SID map hash:
- Collision diagnostics artifact hash:
- Train/freeze week:

### Model

- Model family:
- Parameter count:
- Input features:
- History length policy:
- Objective:
- Decode policy:
- Constraint version:
- Checkpoint hash:
- Baseline fairness notes:

### Reproducibility

- Seed:
- Determinism settings:
- Dependency lock hash:
- Command manifest hash:
- Retry/resume notes:

### Hardware

- Provider:
- GPU model/count:
- GPU memory:
- CPU:
- RAM:
- Disk:
- CUDA/runtime:
- FAISS/backend:

### Metrics

- Item Recall@K:
- Item NDCG@K:
- SID-level metric:
- Collision-corrected item metric:
- Invalid rate:
- Dead-end rate:
- Duplicate rate:
- Effective valid K:
- Train-observed slice:
- First-seen/OOV slice:
- Warm/cold slices:
- ANN recall against exact audit:
- Statistical summary / seeds:
- Metric JSON hash:

### Resources

- Data preparation wall-clock:
- Training wall-clock:
- Evaluation wall-clock:
- Refresh wall-clock:
- Peak RAM:
- Peak VRAM:
- Disk footprint:
- Index/trie footprint:
- Latency p50/p95:
- Cost:
- Resource trace hash:

### Artifact Hashes

- Source manifest:
- Dataset manifest:
- Split manifest:
- Catalog manifest:
- Feature manifest:
- SID/codebook:
- Checkpoint:
- Index/trie:
- Predictions:
- Metrics:
- Logs:

### Claim Boundary

- This run supports:
- This run does not support:
- Known confounders:
- Failed gates:
- Follow-up required:
- Result classification: `positive | negative | inconclusive | invalid`

## 填写规则

没有 artifact hash 的指标不得作为正式结果写入 summary。

One-seed 结果必须明确写 one-seed，不外推稳定性。

SID-level 指标不得替代 collision-corrected item-level 指标。

ANN 指标必须附固定 query exact audit。

`first-seen` 只能解释为 observed-item proxy，不解释为真实 listing 或 cold-start ground truth。

不报告 CTR、online lift 或 A/B 结果。

## P3 train-only历史审计（非freeze依据）

Run ID：p3-train-v2-20261006t1902；Status completed，后因producer绑定review修复而不能作为freeze输入。Source快照与历史hash见p3_validation；原artifact保持server-only。

| 审计项 | v2实测 |
|---|---:|
| Train interactions | 3,720,328 |
| Train users / exposure items | 89,371 / 120,415 |
| Like events / users / items | 105,700 / 18,915 / 21,496 |
| Users with >=2 like positives | 10,863 |
| Prefix training examples（min1/max50） | 86,785 |
| Duplicate user-item / exact rows | 0 / 0 |
| Catalog metadata / embedding matched | 120,415 / 120,415 |
| Missing metadata / embedding / nonfinite / zero norm | 0 / 0 / 0 / 0 |
| Duration null / zero | 0 / 0 |
| Timespent observed seconds range | 1–255 |
| Like同时dislike / share / bookmark / author-page / comments | 45 / 3,501 / 1,742 / 593 / 5,278 |
| Wall-clock / peak RSS | 418.900825s / 3,483,340 KiB |

这些是实际v2 train audit，不是未来v3或holdout的猜测。Astra独立pyarrow/Counter核对train行数、like events/users、>=2支持与catalog匹配；方法与UTC见p3_validation。历史v1壁钟与内存不拿来拼接此v2结果。

## P3最终审计

Run ID：p3-train-v3-20261006t1920 / p3-holdout-v3-20261006t1919；Status completed；P3最终协调验收通过，历史过程与关闭证据见p3_validation。Producer、policy及完整commands/hash见p3_validation。v3 train audit/feature统计与上方v2表逐字段相同，training-examples hash一致；只有v3具备实际producer绑定并获freeze授权。

Final train audit SHA256 `3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe`；holdout audit SHA256 `f9e44a97d410488b7c132abaa27da66076b27917474bcc1adaf0875f46c6f85d`。Source/policy/freeze hash与artifact明细统一见p3_validation。

| 审计项 | Validation week25 | Test week26 |
|---|---:|---:|
| Interactions | 170,111 | 154,461 |
| Observed users / items | 43,370 / 13,126 | 42,199 / 11,814 |
| Like events / target-positive users | 4,946 / 2,922 | 4,292 / 2,562 |
| Eligible queries（train history>=1） | 2,520 | 2,186 |
| Train-observed target queries | 1,897 | 1,445 |
| 首target OOV queries | 623 | 741 |
| Eligible OOV比例 | 24.7222% | 33.8975% |
| 结构性可达比例（catalog membership，非模型效果） | 75.2778% | 66.1025% |
| No-history target users过滤 | 402 | 376 |
| Target-positive users未在train曝光出现 | 13 | 25 |
| Train users没有此split positive target | 86,462 | 86,834 |
| 本week items未在train出现 | 3,328 | 2,621 |
| 相对此前weeks首次observed items | 3,328 | 1,275 |
| Split内duplicate user-item / exact row | 0 / 0 | 0 / 0 |

Freeze后独立projected ID pair-set全27文件审计：4,044,900 interactions、90,178 users、125,018 items，跨split重复user-item为0，匹配固定官方MVP总规模；不是随机抽样无偏性证明。Aggregate hash `4e1f0709c385a73c883c988d31eaa2edab2bf9fdf2f827fc8b39c0ec23d4ffb9`。

v3 train wall-clock169.788374s，peak RSS3,482,008KiB；holdout64.657610s，peak RSS299,080KiB。P3使用CPU审计，没有模型/GPU训练；上述为各进程ru_maxrss，不冒充全实例峰值。所有大产物只在服务器；容量与artifact bytes见validation。

Astra独立不import作者audit地重建所有train positives与holdout首like，逐query匹配targets/positions、train-only最近50 history/positions、unique user与OOV标记，并复核query hashes；UTC见validation。主policy未在查看holdout后调整。

结果为工程数据可行性已得到支持，模型质量未测。OOV比例高且相对train的OOV不同于相对此前weeks的首次observed：后续full-eligible与train-observed slice必须并报，不静默删OOV；所有first-seen仅observed-item proxy。无Recall/NDCG模型指标、TIGER复现、线上A/B或因果结论。


## P4传统baseline固定audit — 已正式验收
只消费P3冻结p3-like-v1，无训练/调参；seed42 SHA256用户排名每split最多512，抽样不读target/OOV/hit。此表仅固定audit、不代表全test；逐query预测/完整selected IDs仅服务器。v1 GEMM tie失败任务aborted_invalid，不使用其数字。下面小摘要来自v2真实metric artifact，真实run与完整source/config/input/query/prediction/metric replay均exit0、stderr空，Astra另独立核查hash/抽样/Popularity/全部指标通过；最终gate状态只读STATE。
Run started UTC 2026-10-06T20:11:52.141315+00:00；catalog 120415。质量小数保留9位，完整精度只在server metrics.json。
质量表已合并到本文P7统一固定audit比较；此处保留P4运行与资源证据。

OOV仍保留于all分母、两个baseline同一query集合，OOV必须miss。当前audit中zero embedding/centroid均0；toy验证zero情形而不造替代feature。Observed Recall@20分子：validation Popularity15/content5；test两者各5；不同split/小audit不能推导方法总体胜负或统计效力。

| Resource / timing | Actual v2 value | Boundary |
|---|---:|---|
| Total elapsed | 334.008680 s | 包括input hash/load、全feature解包/join、train counts、两个split检索与写产物 |
| Process peak RSS | 2742840 KiB (2.615776 GiB) | Linux ru_maxrss；不是独立quota或全系统峰值 |
| Validation content | 7.141671 s / 13.948576 ms per query | 包含centroid与exact blocked topk；CPU1 thread |
| Test content | 7.179947 s / 14.023334 ms per query | 同上 |
| Validation Popularity replication | 0.000038511 s | 不含一次catalog rank/train count，不能与content作公平online latency比较 |
| Test Popularity replication | 0.000124780 s | 同上 |
| Maximum score block | 32×4096 float64 / 1MiB | 无完整user×catalog score matrix；合并topk另有bounded scratch |

OPENBLAS_NUM_THREADS=1/OMP_NUM_THREADS=1，CPython3.11.16、PyYAML6.0.2/NumPy1.26.4/PyArrow19.0.1。全部content输入float32 normalize、pair score固定float64非BLAS reduction；没有GPU/VRAM benchmark。固定query hash与环境/source/lock/input/artifact hashes见public脱敏manifest及p4_validation；完整audit server-only。

Artifact evidence：server `code-v2/outputs/p4-baselines-v2-20261006t2023`，run manifest SHA256 `65d90cc9fb4df88824786a794349e3e2a6724ff0a8e08e4f95b573c699bdb801`；metrics SHA256 `b0e43673c7b0a68aba03ce1807c13065a6ffa5a8d8c030b49a82c78f5b795b1c`。No model win、全test benchmark、线上A/B或causal claim。

## P5 — 真实SID tokenizer审计（已正式验收）

这是unsupervised tokenizer fit与映射/约束审计，不是TIGER模型训练或检索质量评估。只消费P3 train-observed catalog与与P4相同的官方normalized64D features，validation/test不参与fit；未改变P4 query audit set/evaluator。最终source/run/input/artifact hashes与命令唯一复现入口为p5_validation；当前阶段gate只读STATE。

v2：branching16、semanticdepth3、seed42、maxiter10；120,415 train catalog items。完整4-position SID collision0；120,415/120,415 item roundtrip和reachable，valid nonterminal prefix4,369、deadend0；independent array oracle另检查2,465非法next edges无匹配。全部273nodes child容量差<=1。语义leaf4,096，leaf利用率1.0，prefix occupancy16/256/4096；2,465leaf含29items、1,631leaf含30items，max occupancy30，达到ceil(120415/4096)预注册上限。语义collision group4,096、涉及120,415items、额外同prefix members116,319；一个suffix position、vocabulary30将fullSID collision降为0，不能把semantic hit称item hit。Per-position token vocabulary总计78（3×16+30），不含未来模型special tokens。

Leaf entropy11.999800439833193bits，leaf Gini0.008151365119456688；这主要反映强制平衡约束，不据此称语义聚类质量优越。273nodes中245达到10iter仍未converged，完整node objective轨迹/初始化/centers只留服务器；有限迭代与greedy capacity assignment没有全局最优保证，不追加迭代或调参以掩盖限制。

实际start2026-10-06T20:35:39.411984+00:00，elapsed108.09765407582745s（含官方input校验/load/fit/artifact/audit，不是单独在线encode latency）。Peak process RSS2,692,476KiB。424资源样本，最小visible host RAM available fraction0.5169435873092724、disk available fraction0.9865269165039062。250ms采样加fit node/iteration及阶段边界，没有瞬时硬限制、独立quota或未来训练VRAM保证；本阶段CPU/NumPy，无GPU模型训练。

Compact constraint原始arrays1,444,980bytes；NPZ1,445,484bytes。Mapping NPZ2,408,800bytes，codebook JSON5,960,077bytes，normalized feature NPY30,826,368bytes，audit JSON1,186bytes；全部server-only ignored。Public只回传role/hash/size manifest，未回传ID/SID/centers或features。v1真实候选因I-035/I-036 review gate失败保留不验收，不能拿v1 exit0称阶段通过。

Objective补充：273nodes中140至少一次post-mean objective增加；28nodes在labels unchanged条件converged。Greedy固定容量的Lloyd步骤没有单调下降保证，本次如实记录，未把达到maxiter解释成converged。


## P6 — original small TIGER-style smoke（已正式验收）

固定configs/p6.json、seed42、单配置无调参；输入P3冻结prefix/catalog、P5已验收SID与相同官方content feature来源，evaluation逐字使用P4固定audit set。不是RQ-VAE tokenizer，不称TIGER忠实复现、全test、online/causal或方法优势。完整runtime/hash/TDD只读p6_validation；当前阶段状态只读STATE。

### 实际训练与恢复

- 冻结训练例86785；本轮1000个optimizer steps、batch64，共64000个采样example位置，epoch0/cursor64000，未完整训练一个epoch。实际step100 checkpoint保存，first process退出，新process从100继续，训练log连续1～100和101～1000。
- 第一checkpoint optimizer step100/cursor6400，final optimizer step1000/cursor64000；作者独立torch/stdlib核checkpoint hashes、绑定一致、所有modelweights与loss/gradient finite，resume后68个parameter tensors相对step100改变。独立协调验收另记录，不把作者自检当其替代。
- 原首次train started_at2026-10-06T21:00:32.884804+00:00，resume ended_at21:02:26.909064+00:00，原开始到训练最终结束114.024260秒，包含100step进程与resume之间的load间隔，未重置45min预算。训练状态step_limit / complete_1000_steps。
- 1000行loss与gradient均finite；first loss4.4285931587，final2.6608951092，min2.4635210037，max4.4285931587。Loss不是item检索效果或baseline胜负证据，不保证单调。

### 固定audit item-level结果

全部完整4digit SID映射到唯一item；既有P4item Recall/NDCG evaluator，未使用SID hit替代item hit。OOV原样留all分母，OOV必miss，不过滤或替换query；train_observed slice单独报告。

质量表已合并到本文P7统一固定audit比较；此处保留P6训练、合法性与资源证据。

每split512个query，beam20，共1024query全部完成；两split invalid/deadend/full-SID item collision均0。successmanifest保存的是合法catalog item预测，不以零invalid替代质量验证。整体all Recall@20对应validation命中7个、test命中6个；信号弱，OOV候选域限制与fixed audit代表性仍保留。这里只报告P6结果，P7公平对照另gate，不据此宣称模型优于或劣于baseline。

### 时间与资源

| Phase | 指标 | 实测 |
|---|---|---:|
| Initial train100 | Torch peak allocated VRAM bytes | 332696576 |
| Resumed train900 | Torch peak allocated VRAM bytes | 332696576 |
| Decode | Torch peak allocated / reserved VRAM bytes | 54770176 / 71303168 |
| Decode process | peak RSS KiB (Linux) | 1470992 |
| Validation decode | seconds / mean ms per query | 22.8996065247 / 44.7257939936 |
| Test decode | seconds / mean ms per query | 22.5790973757 / 44.0997995620 |

Visible资源250ms采样+每step/query边界；初训/续训/decode采样数131/1121/1212。最小RAM available fraction分别0.5112017473/0.4998295283/0.5209221053；disk available fraction0.9731370163/0.9731135178/0.9731122437；visible whole-GPU VRAM available fraction0.9829192327/0.9829192327/0.9914068264。观测全部超过预注册20%余量。Allocator peak与wholeGPU free不是同一口径；不保证瞬时hardquota/hidden cgroup。

Decode为逐query tokenize+encoder+constrained beam+item mapping且CUDA同步，不含input load。P4latency边界不同，不能直接对这几个ms值排序。Train时间含process间load，不报告纯GPU吞吐优势。所有checkpoint/optimizer/fullqueries/predictions/resource证据仅server ignored。

### 运行证据

四stage train/resume/decode/verify均exit0且stderr为空，pipelinecompleted；完整verify重新核source/env/config/all inputs、checkpoint/actual steps、固定P4 queries、预测与item指标。Server/public manifest hashes统一见p6_validation与public machine manifest；没有在holdout后调整模型/config、没有额外seed/tuning。


## P7 — 统一固定audit比较

Status：Complete / independent review and coordinator acceptance PASS。只复用已验收P4/P6产物，不增加训练、tokenizer fit或调参。来源绑定见`../artifacts/p7-comparison-manifest.json`；核验命令与hash见`p7_validation.md`。所有表值由stdlib独立读取现有预测手写核算，与原metric artifact逐项一致，质量值保留12位小数。

Recall@K为单target是否在前K的平均值；NDCG@K命中rank r时贡献1/log2(r+1)，未命中为0。all保留OOV分母；train_observed与oov按共同target catalog membership互斥划分。Popularity直接使用train like counts，content直接使用官方normalized features，small TIGER-style使用同源features拟合的P5 SID；共享来源与catalog不代表三者输入表示相同。

| Split | Method | Slice | n | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---:|---:|---:|---:|---:|
| validation | popularity | all | 512 | 0.019531250000 | 0.029296875000 | 0.007393515594 | 0.009732565747 |
| validation | popularity | train_observed | 383 | 0.026109660574 | 0.039164490862 | 0.009883759749 | 0.013010636194 |
| validation | popularity | oov | 129 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| validation | content | all | 512 | 0.003906250000 | 0.009765625000 | 0.001429114363 | 0.002982898004 |
| validation | content | train_observed | 383 | 0.005221932115 | 0.013054830287 | 0.001910460976 | 0.003987581666 |
| validation | content | oov | 129 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| validation | small_tiger | all | 512 | 0.011718750000 | 0.013671875000 | 0.005457000640 | 0.005934833730 |
| validation | small_tiger | train_observed | 383 | 0.015665796345 | 0.018276762402 | 0.007294998245 | 0.007933772506 |
| validation | small_tiger | oov | 129 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| test | popularity | all | 512 | 0.003906250000 | 0.009765625000 | 0.001536882315 | 0.002973227849 |
| test | popularity | train_observed | 353 | 0.005665722380 | 0.014164305949 | 0.002229132423 | 0.004312443792 |
| test | popularity | oov | 159 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| test | content | all | 512 | 0.005859375000 | 0.009765625000 | 0.002575805774 | 0.003527634364 |
| test | content | train_observed | 353 | 0.008498583569 | 0.014164305949 | 0.003736012908 | 0.005116568823 |
| test | content | oov | 159 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| test | small_tiger | all | 512 | 0.009765625000 | 0.011718750000 | 0.006545148231 | 0.007004931265 |
| test | small_tiger | train_observed | 353 | 0.014164305949 | 0.016997167139 | 0.009493246159 | 0.010160126933 |
| test | small_tiger | oov | 159 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Validation的all Recall@20：Popularity15/512、content5/512、small TIGER-style7/512；test分别5/512、5/512、6/512。这里只描述本固定audit实际命中，不宣称总体优劣或显著性。排序随split/指标不同，未进行多seed、置信区间或paired significance检验。

| Sampling diagnostic | Validation | Test |
|---|---:|---:|
| Eligible queries | 2520 | 2186 |
| Audit queries | 512 | 512 |
| Eligible OOV | 623 | 741 |
| Audit OOV | 129 | 159 |
| Audit覆盖eligible比例 | 20.317460% | 23.421775% |
| Eligible OOV比例 | 24.722222% | 33.897530% |
| Audit OOV比例 | 25.195312% | 31.054688% |

抽样只依赖seed/split/user SHA256，未按target、OOV或hit挑query；以上构成差异仍不能证明平台代表性。OOV在三种方法共同train catalog外，全部Recall/NDCG为0是结构性不可达，不是decode invalid。P6两split invalid/deadend/full-SID collision均0，且每条预测为20个唯一catalog item；P4同样每条为20个唯一catalog item。P4无生成路径，invalid/deadend不是同义的模型指标，记不适用；P5语义prefix collision与完整SID零collision需分别理解，见P5记录。

| Method / phase | 已测量系统成本 | 可比较边界 |
|---|---|---|
| Popularity | P4总进程RSS与总elapsed、两split热门榜复制时间见P4资源表 | rank/count预处理不在复制时间内；不作为完整在线latency |
| Content | P4 CPU单thread centroid+exact topk时间、bounded score block、RSS见P4资源表 | 无训练；包含centroid与检索，不包含完整input load；无GPU VRAM测试 |
| SID fit | P5 fit/load/hash/output wall-clock、RSS与map/constraint/codebook bytes见P5记录 | offline一次fit；不当在线encode时间 |
| Small TIGER-style train | P6真实train/resume wall-clock、optimizer steps与allocator VRAM见P6记录 | 从首次start计预算，含process间load；与无训练baseline天然不同 |
| Small TIGER-style decode | P6 tokenize+encoder+beam+item mapping/CUDA同步时间、RSS/VRAM见P6资源表 | input load不在query latency；与P4边界不等，不能直接排名 |

所有方法仅一个预注册配置，没有hyperparameter search或holdout后改配置；相同的是调参次数，不是相等GPU/CPU计算预算。旧invalid候选是正确性修复后重跑，不是按质量选run。仅一个seed、小audit，输出合法与loss下降不代表模型质量强；P5有限迭代多数node未收敛、greedy objective非单调，不掩盖负结果。

Label仍like-only，history仍train-only conservative，官方physical-order assumption无法独立验证hidden timestamp；irX定义未明确。First-seen只为observed-item proxy；完整数据、features、SID、checkpoint与逐query预测仅server ignored。结果不支持真实dynamic catalog、普通CTR、线上A/B、causal lift、忠实TIGER复现、OneRec或AI Search全栈。


## P3 train-only补充EDA — 已正式验收

Run ID：p3-label-eda-20261007t0056-final。只读固定revision train00–24及既有metadata；冻结p3-like-v1不变。修正源码fresh scan已完成，schema/null/join与既有基数核对通过；不是新label冻结或新模型实验。通俗解释见[p3_label_eda.md](p3_label_eda.md)，source/commands/hash/历史失败与独立复核见[p3_label_eda_validation.md](p3_label_eda_validation.md)。所有本轮人工数字摘要只在本节维护；机器aggregate见`../artifacts/p3-label-eda-manifest.json`。

### 原始train画像与join

| 项目 | 实测 |
|---|---:|
| interactions / users / exposure items | 3,720,328 / 89,371 / 120,415 |
| interaction列数 / 各列null | 12 / 全部0 |
| train catalog item metadata matched / duration null / duration zero | 120,415 / 0 / 0 |
| train users metadata matched / 已读字段null | 89,371 / 全部0 |
| timespent秒 P50/P90/P95/P99；min/max | 9 / 58 / 72 / 135；1 / 255 |
| catalog duration秒 P50/P90/P95/P99；min/max | 25 / 70 / 90 / 161；5 / 180 |
| timespent=255上限编码 | 3,078（0.0827%） |
| timespent/duration >1 / =1 | 654,484（17.5921%） / 516,180（13.8746%） |
| exposure事件中duration=0 / =255 | 0 / 0 |

timespent/duration两者均uint8秒编码；固定README未说明255是截断/饱和、重复播放机制或日志累计口径。所以上限编码占比可测，真正的饱和比例不可识别；ratio>1不裁剪，也不能确定成因。duration分位数按catalog items统计，timespent分位数按events统计，不能互换分母。用户metadata全量表与item metadata全量表只做train实体筛选，不扩候选域；用户属性未作为label/feature。

### history长度与头部贡献

| train user层面 | Exposure count | Like count（含零like用户） |
|---|---:|---:|
| P50/P90/P95/P99 | 18 / 111 / 162 / 290 | 0 / 2 / 5 / 24 |
| zero用户 | 0 | 70,456 |
| 完整count>50的用户 | 23,235 | 239 |
| 如只保留完整末尾50，快照被截事件 | 1,584,791 | 7,113 |
| 最高贡献1% train用户的event share | 8.8455% | 38.4248% |
| 最高贡献10% train用户的event share | 44.8623% | 88.7398% |

Like训练prefix数按每用户max(like_count−1,0)汇总为86,785；最高贡献1% / 10% train用户占训练prefix的45.7694% / 97.7819%。这是例数贡献，不是loss权重。所有top用户份额都以全体train曝光用户为排名人口，向上取整人数，按贡献降序取；不是只在活跃like用户中取百分比。末尾50快照截断不是每个训练prefix实际丢弃比例；生产训练仍只取target前最近50like。曝光长度不会自动变成冻结like history。

### 六反馈与候选label覆盖

| Feedback原义 | True events | 占全部曝光 |
|---|---:|---:|
| like | 105,700 | 2.8411% |
| dislike | 1,091 | 0.0293% |
| share | 23,744 | 0.6382% |
| bookmark | 3,971 | 0.1067% |
| click_on_author | 7,545 | 0.2028% |
| open_comments | 43,886 | 1.1796% |

click_on_author是打开作者页，open_comments是打开评论区；没有普通click或CTR。以下候选grid在扫描前声明，均保留explicit dislike冲突；ratio仅对duration>0定义，不裁剪>1。它们只比较支持度及train关联，不是已选positive规则。

| 探索候选 | Events | 曝光覆盖 | Users | Items | Users≥2 | Like关联P(like\|候选) | 覆盖全部like | Dislike冲突events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| like | 105,700 | 2.8411% | 18,915 | 21,496 | 10,863 | 100.0000% | 100.0000% | 45 |
| like_or_share_or_bookmark | 128,003 | 3.4406% | 25,113 | 24,182 | 14,824 | 82.5762% | 100.0000% | 69 |
| seconds_ge_5 | 2,475,822 | 66.5485% | 84,569 | 96,333 | 75,975 | 3.8244% | 89.5790% | 788 |
| seconds_ge_10 | 1,829,022 | 49.1629% | 80,920 | 79,866 | 71,556 | 4.0829% | 70.6509% | 616 |
| seconds_ge_20 | 1,237,947 | 33.2752% | 75,717 | 58,788 | 65,078 | 4.1083% | 48.1154% | 465 |
| ratio_ge_0.5 | 1,813,493 | 48.7455% | 81,023 | 82,571 | 71,552 | 4.7873% | 82.1362% | 544 |
| ratio_ge_0.8 | 1,446,654 | 38.8851% | 78,342 | 73,469 | 68,076 | 5.1324% | 70.2441% | 448 |
| ratio_ge_1.0 | 1,170,664 | 31.4667% | 76,021 | 67,079 | 65,127 | 5.3442% | 59.1892% | 354 |

Like OR share OR bookmark相对like新增：22,303 events、6,198 users、2,686 items、3,961位count≥2 users；explicit dislike冲突从45变成69（新增24），未据此清洗。观看候选带来更密的事件和更多history支持，但不证明这些事件都是满意的正例；更高的like关联也不能证明真实满意度或泛化效果。

| 成对cofeedback | Events |
|---|---:|
| bookmark&click_on_author | 71 |
| bookmark&open_comments | 455 |
| click_on_author&open_comments | 625 |
| dislike&bookmark | 18 |
| dislike&click_on_author | 10 |
| dislike&open_comments | 33 |
| dislike&share | 14 |
| like&bookmark | 1,742 |
| like&click_on_author | 593 |
| like&dislike | 45 |
| like&open_comments | 5,278 |
| like&share | 3,501 |
| share&bookmark | 346 |
| share&click_on_author | 301 |
| share&open_comments | 1,603 |

这些是各对bool同时True的计数；同一事件可属于多对，不能把各行相加当作union。无反馈/未like与明确dislike不同。

### Duration分层：时长规则改变了内容分布

| Duration秒bin | Exposure events | Like rate | Seconds≥10覆盖 | Ratio≥0.8覆盖 | Ratio>1占比 |
|---|---:|---:|---:|---:|---:|
| 1-10 | 562,896 | 4.8245% | 21.8945% | 53.7229% | 26.4889% |
| 11-20 | 681,052 | 3.8558% | 45.5351% | 41.0208% | 18.5482% |
| 21-40 | 782,841 | 2.6266% | 52.5043% | 37.2321% | 15.5277% |
| 41-60 | 946,896 | 1.9531% | 59.6648% | 37.7586% | 16.8456% |
| 61-120 | 597,184 | 1.8735% | 57.1079% | 30.5469% | 13.5034% |
| 121-255 | 149,459 | 1.3643% | 52.6118% | 22.3827% | 11.6079% |

绝对seconds门槛与ratio门槛都明显随duration变化；表只展示已观测关联，不把duration差异归因为用户满意度。bin上界是探索划分，实际最大duration见上表。

### Week与context差异（只列大分母）

| 层级/匿名code | Events | Like rate | Ratio≥0.8覆盖 | Mean watch秒 |
|---|---:|---:|---:|---:|
| place=0 | 1,876,933 | 1.9936% | 37.0325% | 18.7187 |
| place=1 | 1,571,825 | 4.0924% | 43.0507% | 24.6514 |
| platform=0 | 2,523,744 | 2.6587% | 37.5688% | 20.7185 |
| platform=1 | 1,059,139 | 3.1764% | 41.5822% | 20.9893 |
| agent=0 | 2,524,318 | 2.6593% | 37.5689% | 20.7187 |
| agent=1 | 1,090,494 | 3.1776% | 41.6131% | 21.1146 |

Train week00–24的like rate范围2.6571%–3.3465%，ratio≥0.8覆盖范围36.6754%–42.2156%。全部week/context aggregate留在machine manifest，不放真实rows。Context code已匿名，未提供reverse mapping，不能把code0/1猜成feed/search或Android/Web。小context分母未用于优劣结论。

最终fresh scan wall-clock 5.486510s，peak RSS 109,416KiB；可见host/cgroup有效RAM 211,900,432,384bytes、最小余量95.9328%（门为至少20%）。进程峰值不冒充整个实例峰值。基数/原hist/六feedback核对八项全部True。Astra独立Counter/set与float除法oracle复算所有候选events/users/items/users≥2全部匹配；hist分位复算也匹配。不是训练或模型指标。


## P3-C — 现有数据质量与清洗决策（已正式验收）

Train-only、固定既有revision，未训练、改label或读holdout values。复用已验收audit/EDA并hash重核train/metadata/content；本轮补item/author支持、user flags分层与匿名属性。Explanation只读[data_quality](data_quality.md)，实际source/commands/hash/失败/review证据只读[p3c_validation](p3c_validation.md)。机器aggregate `../artifacts/p3c-data-quality-manifest.json`。

### 清洗前后基数

确定性计数没有实施过滤，raw immutable；低活跃与长尾主样本保留、不5-core，不裁ratio、不删除like&dislike共现或上限编码。无like/单like user没有prefix是任务定义，不是清洗删除。

| 项目 | Before | After | 删除/变化 |
|---|---:|---:|---:|
| catalog_items | 120,415 | 120,415 | 0 |
| interactions | 3,720,328 | 3,720,328 | 0 |
| like_events | 105,700 | 105,700 | 0 |
| like_prefixes | 86,785 | 86,785 | 0 |
| like_users | 18,915 | 18,915 | 0 |
| liked_items | 21,496 | 21,496 | 0 |
| train_users | 89,371 | 89,371 | 0 |
| users_ge_2_likes | 10,863 | 10,863 | 0 |

### Item/author 支持与长尾

每个profile的排名人口为该组全体实体（含zero-like），按贡献降序选ceil(人口×比例)个；份额分母是全体实体的贡献总量，整数exact累计。Nearest-rank分位；author由train catalog item的author_id分组，未读holdout行为或全metadata所有作者分布。

| 对象/贡献 | 全人口 | Zero support | P50/P90/P95/P99 | Max | Top1% share | Top10% share |
|---|---:|---:|---|---:|---:|---:|
| item exposures | 120,415 | 0 | 3 / 51 / 116 / 496 | 22,610 | 39.4463% | 80.9196% |
| item likes | 120,415 | 98,919 | 0 / 1 / 3 / 19 | 431 | 46.5184% | 91.0558% |
| author catalog items | 50,952 | 0 | 1 / 5 / 8 / 19 | 77 | 12.0467% | 44.5966% |
| author exposures | 50,952 | 0 | 3 / 103 / 287 / 1463 | 37,184 | 42.0335% | 87.8720% |
| author likes | 50,952 | 39,819 | 0 / 2 / 7 / 42 | 606 | 50.7417% | 93.1230% |

Train-observed authors 50,952；zero-like catalog items 98,919（82.1484%）；zero-like train-observed authors 39,819（78.1500%）。它们保留在共同候选源域；支持稀疏不等于脏数据。

### User 贡献与异常 flag 定位

Exposure max664、like max186、prefix max185；不是bot/刷量证据。复算全user exposure/like hist与既有EDA一致。各行按完整train like数量互斥分层；ratio>1与255是事件flag，不定义清洗条件。

| 完整train like bin | Users | Exposures | Likes | Prefixes | Ratio>1 events | Timespent255 events |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 70,456 | 2,457,117 | 0 | 0 | 403,149 | 1,867 |
| 1 | 8,052 | 459,709 | 8,052 | 0 | 84,635 | 465 |
| 2-5 | 6,500 | 391,682 | 19,072 | 12,572 | 75,605 | 360 |
| 6-10 | 1,997 | 142,631 | 15,195 | 13,198 | 29,009 | 129 |
| 11-50 | 2,127 | 220,153 | 44,318 | 42,191 | 49,758 | 213 |
| 51-100 | 194 | 37,638 | 13,470 | 13,276 | 9,278 | 35 |
| 101+ | 45 | 11,398 | 5,593 | 5,548 | 3,050 | 9 |

有ratio>1 flag的train users 65,966；有255 flag的users 2,749。同一user可属于两flag，不能相加当unique user。全train用户人口Top1%/Top10% prefix share 45.7694%/97.7819%；head集中保留并供后续user balance独立消融。

### Metadata/feature/hash 与资源

| Train-selected user字段 | Code min/max | Distinct codes | Zero-code count | 语义 |
|---|---|---:|---:|---|
| age | 18/70 | 53 | 0 | 编码映射/采集时点Unknown |
| gender | 1/2 | 2 | 0 | 编码映射/采集时点Unknown |
| geo | 0/79 | 80 | 65 | 编码映射/采集时点Unknown |

Train-selected item metadata全部4列/user metadata全部5列null均0，unique join各120,415/89,371匹配；源table footer rows为19,627,601/10,000,000。Interaction全部12列null0、25个train schema matches，rows求和3,720,328。重核28个既有raw文件SHA/bytes（25 train+2 metadata+1 embedding）；metadata按train实体筛选，未扩大catalog。既有embedding float16[64]全部120,415 catalog匹配、missing/nonfinite/zero-norm各0，whole-file SHA相同，复用完整审计而非重新load。Frozenaudit global duplicate user-item0、exact duplicate含week记录0；不能据此称已证明上游event唯一或隐藏timestamp。

本轮12项reconciliation全部True；source真实一次scan exit0/stderr空。Elapsed（截止result构造）4.819094s，peak process RSS 114,528KiB；summary统计554次可见resource samples，effectiveRAM最小211,900,432,384bytes、RAM最小余量95.9219%、disk最小余量97.3111%。还有一次成功pre-publication检查不在summary计数/minima内；不是瞬时hardquota或平台配额保证。CPU only，无GPU模型训练。

限制：metadata rank聚合全train，不能输入早期prefix；snapshot/effective-time unknown；anonymous geo zero不等于missing。Official-order assumption、irX未知、subset代表性与已用旧test继续保留。独立final source/docs/public review PASS；Astra完整item/author独立聚合、真实scan/hash与公开边界验收PASS，P3-C Complete。不是以作者exit0替代验收；本worker停止，P3-D由协调另派。

## P4B pipeline validation — synthetic only, no formal quality comparison

真实train-only prepare对账：86,785 examples、120,415 all-train-exposure catalog items、21,496 train-positive supported items、105,700 like events。每条target/history/位置匹配旧P3；fullpast mask证据、全train support（包括single-like用户）与catalog核验。Astra独立不importproducer重读25train partitions和官方NPZ，全部prefix/fullpast及normalized64D逐float相等，artifact/hash核验PASS。没有新的holdout metrics或正式模型训练。

Bounded CPU synthetic仅5queries、batch3、d64、20steps。连续与独立process停7后resume到20逐batch/loss/model/optimizer/global RNG/cursor/sampler stats完全相同；suite另外覆盖停8(epochboundary)恢复。Server persistent首步loss1.866693139076233，末步0.08886338770389557；累计drawn50/skipped0。数字只说明toy框架和overfit，不是VK model质量或正式调参/validation选择；synthetic best按training loss，正式best仍归P4D validation。

Real-v3可见start headroom：RAM0.5411652519027921、VRAM0.9878103378103378、disk0.9731060638427734；写前RAM0.5242188350783967、VRAM0.9878103378103378、disk0.9731035614013672，均>=20%。sampling不是峰值保证/host承诺，不完成四架构batch pilot，不声称host-loss persistence。来源/source/config/input/output及版本限制见p4b_validation；正式当前状态只读STATE。

## P4C final code-v2 pilot — fixed validation baselines

当前状态唯一STATE；本节仅报告已实际成功pilot，不预告fullrun结果。Source d3f0eb1c644e95430e8eb27703515159ceda7099ca95d54853693aa0d2d99452；test gate closed。

All-train binary user-item edges 105,700；cooccurrence updates including self（Σuserdegree²）2,988,568；max user degree 186，max item distinct-user degree 431，max single-row pair updates including self 7,604。Binary graph不排single-like用户、不从prefix/truncated history反推。

Pilot固定选64 largest-work rows+最多64 evenly-spaced supported IDs，实际128 rows/280,060 pair updates，retained edges 17,267，neighbor计算0.242252s；全train input scan/build binary incidence 6.898972s。Content用前32 validation queries exact fullcatalog，0.456015s；不把这次batch-amortized probe当完整latency或模型胜负。Full eligible validation query count 2,520；后续fullrun/verify实际结果见本节下文。

Pilot visible start/end headroom（RAM/VRAM/disk）分别 {'disk': 0.9729438858032227, 'ram': 0.511939042471628, 'vram': 0.9878103378103378} / {'disk': 0.972939712524414, 'ram': 0.5285367509128596, 'vram': 0.9878103378103378}，每次调用门均>=20%。采样非瞬时hardquota/host承诺；不存在SciPy依赖，不安装。旧v1 pilot保留但不作为final consumer结果。

### P4C full eligible validation（作者run成功，协调验收待定）

共同catalog 120,415 items；queries 2,520。Query/catalog完整identity SHA分别 `9077a1c9495d494bc8ec209c0bf3305fd68db5bfcca24abcadc4a1e552e60a97` / `80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3`。Final runtime manifest SHA `596b2df22449491f9627d2024310275ebce1bca4c898c3f8a33ed08dca25cc64`；完整预测/graph只留server，public只保存hash。Test gate closed。

`train_observed`沿用旧evaluator字段名，在本表严格等于in-catalog（all-train-exposure catalog），不等于train-positive支持slice。各slice分母为该slice所有queries，OOV贡献0，empty slice null。

|Method|Slice|Queries|Recall@10|NDCG@10|Recall@20|NDCG@20|Recall@50|NDCG@50|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|popularity|all|2520|0.023809523810|0.009279009620|0.032936507937|0.011509207152|0.050000000000|0.014877363486|
|popularity|in-catalog|1897|0.031628887717|0.012326359643|0.043753294676|0.015288983670|0.066420664207|0.019763287287|
|popularity|oov|623|0.000000000000|0.000000000000|0.000000000000|0.000000000000|0.000000000000|0.000000000000|
|itemcf|all|2520|0.005158730159|0.002562332712|0.009920634921|0.003759938323|0.032936507937|0.008290343948|
|itemcf|in-catalog|1897|0.006852925672|0.003403836813|0.013178703216|0.004994752016|0.043753294676|0.011013003031|
|itemcf|oov|623|0.000000000000|0.000000000000|0.000000000000|0.000000000000|0.000000000000|0.000000000000|
|content|all|2520|0.003968253968|0.001465749464|0.008333333333|0.002596252807|0.014285714286|0.003770075891|
|content|in-catalog|1897|0.005271481286|0.001947121060|0.011070110701|0.003448896717|0.018977332630|0.005008218896|
|content|oov|623|0.000000000000|0.000000000000|0.000000000000|0.000000000000|0.000000000000|0.000000000000|

|K|ItemCF filled slots|slot rate|queries requiring fill|query rate|Pop coverage|ItemCF coverage|Content coverage|
|---:|---:|---:|---:|---:|---:|---:|---:|
|10|484|0.019206349206|60|0.023809523810|0.000083046132|0.097969522070|0.095519661172|
|20|1145|0.022718253968|69|0.027380952381|0.000166092264|0.123672299963|0.156558568285|
|50|3775|0.029960317460|96|0.038095238095|0.000415230661|0.142083627455|0.278777560935|

Coverage为该route所有full validation topK中distinct catalog items / whole catalog；fill slots只算shortlist不足后添加的Pop项，不把overlap去重误算成fill。

|Route|Measured seconds|P50 ms|P95 ms|Scope|
|---|---:|---:|---:|---|
|popularity|0.002313|0.000709|0.001600|per-query shortlist copy only; train counting and shared Pop sort excluded|
|itemcf|15.444109|5.525911|9.678417|per-query history score sum, sort and Pop fill; graph fit and shared Pop sort excluded|
|content|36.048221|14.267148|14.592418|batch-amortized normalized history mean, fixed-pair exact scoring and ranking; input load excluded|

Content zero centroids 0。Route计时排除input loading、graph fit与resource gate probes；content P50/P95是batch-amortized per-query值，不能与独立single-query latency混为同一口径。Write_run内evaluation+artifact write elapsed 56.482062s，非整个CLI总时长。Process high-water RSS 406,972KiB（Linux），scope为完整CLI进程含audit/load/graph/evaluation；不是逐route峰值。CPU exact baseline，未使用GPU算分；VRAM只是资源门采样。

Run start/before-eval/end visible headroom：{'disk': 0.9729396591186523, 'ram': 0.5422672552411654, 'vram': 0.9878103378103378} / {'disk': 0.9729395523071289, 'ram': 0.5131606051344222, 'vram': 0.9878103378103378} / {'disk': 0.9728559494018555, 'ram': 0.5289084514737967, 'vram': 0.9878103378103378}。所有调用通过>=20%门，但采样不是瞬时hardquota/host承诺。旧validation512 Pop/content top20逐值与@10/20全部slice指标相等，query/history/target membership相等。

本次仅固定baseline定义下的validation探索；不声称线上效果、因果lift、统计胜负、泛化test或后续TT/ANN结果。Full verifier已实际terminal success；Astra独立真实全量oracle已PASS，closing docs/public review及协调裁决仍待完成。

### P4C production verifier实际证据

Verifier完整input audit/raw rescan/graph rebuild/full三route replay/metrics+legacy proof耗时 163.335193s，resource samples 1988，sampled minimum {'disk': 0.972855827331543, 'ram': 0.5036800779749218, 'vram': 0.9878103378103378}，每门>=20%；terminal exit0/stdout verified/stderr空。只描述生产verifier的fresh过程，不能替代协调独立oracle或上游真实性证明。独立raw/graph/Pop/ItemCF核验已通过，content oracle已定位为coordinator未匹配冻结1D vs axis=1 float32 normalization归约差异，单例严格修正通过，Astra全量修正版已实际exit0/PASS；不改production或容差，docs/public review前不写Complete。

Content oracle诊断（coordinator evidence，不是模型效果）：query ordinal698的一维norm0.8976620435714722、冻结axis=1 norm0.8976619243621826，normalized centroid最大逐float差5.960464477539063e-08，导致top1/2交换。仅匹配冻结norm轴后严格排名一致，d,id与qd,id scores逐值相等；无epsilon/容差更改。Astra修正版全量strict oracle已exit0/PASS，全部raw/pair graph/ranks/metrics/coverage/fill/legacy audit通过；正式验收由coordinator收口。

固定baseline定义与本次共同full validation下，ItemCF和content的all-eligible Recall/NDCG在K10/20/50均低于Pop。没有追加调参或隐藏此负结果；不能泛化为线上、test或其他训练/数据条件下的方法优劣。

## P4D source / GPU / resource pilot（非质量实验）

P4D v2 source fingerprint2b8658b06ece14ca703f1b33cf84d8f24a096b9748d1f91ff91a3ce640bd2848，GPU恢复四架构各24steps、epoch8，两次中断后complete model/optimizer/cursor/RNG exact。共同batch512唯一attempt四架构allpass，history50各3optimizersteps、2520user/120415item full exports、32×4096live scoringblock与32query/fullcatalog exactresourcecontext；selection_metrics_read=false。RAM最小约0.5378588、VRAM measured各最小0.9808591（end0.9774318）、disk最小约0.9725844。Transformer最高allocated445976064B、reserved572522496B。pilot manifestSHA2b32a6ab457bc909e288b2ca51d6e0695995ae7e8a8155a2c7b1588c87ba4fa6；author逐binding/measure/12exporthash核一致，Astra独立正式接受resource/restore gate。仅资源测量及synthetic恢复，未报告真实validation质量。

P4D首真实slot1：id_mean LR.001 T.07 seed42 batch512，registered max30/patience3。86785prepared examples，每epoch170steps尾257；absolute cutoffs50→170→220，completedepochs1、第二epoch内50steps，累计drawn112385/skip0。三段actualexit0，正式verify actualexit0；manifest220SHA537baadedd26eb0ac6dd704fa65533d1aefb587cc690fbf151911970d2fb9ec8、checkpoint220SHA7807067905ef777e97fca9404d6417b58c3ea0f94d9d91751d5aa7ab0d40e945。Astra独立fresh接受bounded gate，开始同slot续跑，不额外调参run；质量比较等全部固定预算结果。

P4D slot1 seed42 id_mean首完整train终态：max_epochs停止，30epochs、5100steps、drawn2603550（86785×30）、skipped0；全部resource minima>=.2。final manifestSHA5e015123bf3b4c0ccd7f098d5c98cf9f485830ee87d22d7139b4629dfceca298。Astra独立核预算/selection/checkpoint链通过；full30epoch正式verify尚在执行，不作为最终质量结论。

首配置兼容full-epoch formal verifier已actualexit0并由Astra接受。阶段pid/exit文件mtime差1784秒（29分44秒，秒级stage wall代理，非kernel-only耗时）；不猜重放阶段或逐epoch进度。原manifest/source不变，显式producer v2 + adapter/proof双身份receipt验证。

P4D第三注册配置train.exit0：10epochs/1700steps，stop_reason=patience、bad_epochs3。Astra独立逐epoch strict lexicographic best/earlystop重放通过；terminal manifest SHA2cbf1774f930f28b7ed138d25d748b73e5920fc645d89d11129a591ac7056e8e。Full verifier尚在执行，不作模型质量结论。

P4D第四注册配置train.exit0：9epochs/1530steps，patience3停止；Astra独立完整earlystop重放PASS。terminal manifest SHA866d65120ed32ce987db62a82b54c7b55d8b775b0b150b3be8fe67d221dea765与正式verifier CLI绑定一致。Full verifier运行中，不提前进行整体模型胜负选择。

P4D第五注册配置id_content_mean LR.001/T.07/seed42/shared512 train.exit0：5epochs/850steps，patience3停止；Astra独立earlystop重放PASS。terminal manifest SHA82bde9e8a1c21d6e460eda129b27b64284f588ef2dd38543991ee21fd363b3e7。Full verifier pending；不据单配置早停推断内容特征收益。

P4D第六注册配置train.exit0：5epochs/850steps，patience3；独立earlystop replay PASS。terminal SHA892980bdf20e87b4f984295d43dcec04f9c59f696b7d9f5c797938203ba7b384与actual formal verifier CLI一致。Full verifier pending；不据单配置结果改变预算。

P4D第七注册配置id_content_mean LR.0003/T.07/seed42/shared512 train.exit0：4epochs/680steps、patience。terminal manifest SHAdd780cbcf1880120cc91d680b59828436736a8bd4faef53e1282bbe0a654760a；完整verifier pending，不调整注册预算。

P4D第八注册配置id_content_mean LR.0003/T.15/seed42/shared512 train.exit0：10epochs/1700steps、patience3，独立earlystop replay PASS。terminal manifest SHA87c7d5165302887fb7b13c36f787a2d1049a91e75f21b5bca03b56ac6880eb00；full verifier pending，不调整注册预算。

P4D第九注册配置ordinal_content_mean train.exit0：9epochs/1530steps，patience3；Astra独立strict Recall50/NDCG50选择与earlystop重放通过。terminal SHAc148b885ca2bbf449177a71f56df6fc7ad8efc6ba6f5784cd4e815b2a28982ba与actual verifier CLI绑定一致。Full verifier pending。

P4D第十注册配置ordinal_content_mean train.exit0：9epochs/1530steps、patience3；独立strict选择与earlystop replay PASS。terminal SHAcb5f24f308eaf7b6debbc2ddc65d860b10c3414c7e0399af35db12d48c152402与actual verifier CLI一致，full verifier pending。

P4D第十一注册配置ordinal_content_mean LR.0003/T.07/seed42/shared512 train.exit0：4epochs/680steps、patience3；独立strict选择与earlystop replay PASS。terminal SHAeff60e0ea607ebf5e034ec6d5f99896117c0f2007d431a3b469b1e5936d0b07d；full verifier pending。

P4D第十二注册配置ordinal_content_mean train.exit0：4epochs/680steps、patience3；独立strict选择与earlystop replay PASS。terminal SHA7f1a2b75fc72a36b2e4142ca3fcd0381f6d37e190827ebbb9d5e3827ea8567de与actual full verifier CLI一致，full verifier pending。

P4D第十三注册配置content_transformer LR.001/T.07/seed42/shared512 train.exit0：4epochs/680steps、patience。terminal SHA68132310b10f2372007c909aecf7e62f1fa228b5449601896bbb64650a3b3f7c；full verifier pending，不据单配置推断Transformer收益。

P4D第十四注册配置content_transformer LR.001/T.15/seed42/shared512 train.exit0：7epochs/1190steps、patience。terminal SHA4e334365417123ac04f041dab690a1c3f0a1dc3859650980dc03f609ab7bd21b；full verifier pending。

P4D第十五注册配置content_transformer LR.0003/T.07/seed42/shared512 train.exit0：4epochs/680steps、patience。terminal SHA6ee3adeffed05cccf8391313d8346cc69eeb74931890b11e16d89b09af1f8347；full verifier pending，selection未提前冻结。

P4D第十六注册配置content_transformer LR.0003/T.15/seed42/shared512 train.exit0：7epochs/1190steps、patience3；独立strict选择与earlystop replay PASS。terminal SHA4d1dca62f21e3d8b32d2e199f9b523ab5f228723c83e9ab7a444f7130acc48c8；full verifier pending，selection仍未freeze。

P4D core selection已冻结并获协调独立重算接受：16个seed42 runs全formal verified，Recall@50→NDCG@50与更早checkpoint/固定枚举tie顺序一致。ID mean选LR.001/T.15；ID+content mean选LR.0003/T.15；ordinal weighted mean选LR.001/T.15；content Transformer选LR.0003/T.15。Best core architecture=id_mean，代表seed固定42；selection SHAbc70aae38aa53af8ff70c6e159f0a523cbf362d72b20cbc04039daf44657b6de。随后仅各selectedcfg seeds2024/2025，完整三seed结果留最终汇总，不挑bestseed/追加HPO/test。当前content/Transformer seed42明显较弱，最终报告并仅用现有trainloss/最佳epoch/support-OOV/zero-vector/权重及reexport证据诊断，不改冻结实验、不作无证据因果归因。


### P4D只读瓶颈采样与并行可行性边界

UTC2026-10-07T14:47:35Z，run23 full verifier实际运行，GPU瞬时0%、整卡552/49140MiB；2秒cgroup CPU平均1.002核/10.24核配额，RAM22.005/197.348GiB（余量88.85%）、disk余量71.79%。原train PID已退出，per-child CPU/IO采样未测；cgroup块设备9:0读取增量约4.85MiB/s、写入0，堆叠设备不求和，缓存读取未计入，不能据此判定IO饱和。此前run22 full verifier采样约1.265核、GPU0%/464MiB是另一时间窗口。没有训练GPU/CPU/IO利用率历史时序，也没有完整verifier独立分段耗时/峰值。

Run23 terminal step680、4epochs，每epoch170steps、86785个example indices；逐epoch索引集合精确等于range(86785)，无重复/遗漏，skip0。训练覆盖的是官方deepvk/VK-LSVD ur0.01_ir0.01冻结subset所生成的全部prepared examples，不是parent全量；ir命名的生成机制不作推断。

已有terminal invocation segment记录：run17 train977.922s/evaluation1053.482s，run21 train614.360s/evaluation898.508s，run22 train250.317s/evaluation378.671s，run16 train228.189s/evaluation332.301s。evaluation_seconds是训练逐epoch评价累计，不是额外full verifier walltime。PyTorch allocated/reserved峰值：IDmean236.165/260MiB，IDcontent与ordinal266.228/290MiB，Transformer425.062/546MiB。终态segment VRAM minimum推得整卡采样占用约1.21/1.24/1.49GiB，非整个实验最终峰值。历史cgroup RAM minimum约48.85%～50.64%，对应总cgroup峰值约97～101GiB，非孤立job RSS；两份同量级full job不能据此证明满足20%余量。

后续P4E/G3双job pilot建议（尚未授权执行，G3未ready）：先固定同样有界train+verify工作量分别串行，再两job同时比较初始化及end-to-end walltime、examples/s、verifier queries/s、各自延迟、util分布/峰值/IO wait，pilot不参与正式选择。规划combined RAM≤157.878GiB、整卡VRAM≤38.391GiB（20%余量）；CPU硬配额10.24核，规划≤8.192核，保守单job RAM目标≤70GiB并给共享overhead留空间，实际仍核combined live值。建议收益门aggregate throughput≥20%提升、每job耗时≤串行1.25倍及等价artifact核验/所有stop gates通过，否则串行或只重叠CPU准备。阈值是待阶段brief确认的建议，不是已测收益或当前资源放行。


## P4D固定完整预算交付 — awaiting coordinator acceptance

Actual24completion、24唯一terminal/receipt/result与双身份链闭合，已审只读extractor actual exit0。16seed42注册配置+8additional seeds，共享batch512，每run max30/patience3。best按Recall50→NDCG50 strict lexicographic、tie更早epoch；selected configs仅用seed42冻结，代表seed42保持，无bestseed或追加HPO。以下为validation best checkpoint，单位比例，不是百分数。test closed，仅既有bytes hash。

### 全24run all-eligible指标

| slot | architecture | seed | LR | T | best epoch | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 | ndcg@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | id_mean | 42 | 0.001 | 0.07 | 30 | 0.0031746031746 | 0.00873015873016 | 0.0206349206349 | 0.00133113633882 | 0.00273094885914 | 0.005103420482 |
| 2 | id_mean | 42 | 0.001 | 0.15 | 30 | 0.0103174603175 | 0.0166666666667 | 0.0373015873016 | 0.0040576223256 | 0.00562067857826 | 0.00973890677164 |
| 3 | id_mean | 42 | 0.0003 | 0.07 | 7 | 0.000396825396825 | 0.000793650793651 | 0.00119047619048 | 0.000170903396061 | 0.000261248732845 | 0.000331205651254 |
| 4 | id_mean | 42 | 0.0003 | 0.15 | 6 | 0.000396825396825 | 0.000793650793651 | 0.00119047619048 | 0.000170903396061 | 0.000261248732845 | 0.000331205651254 |
| 5 | id_content_mean | 42 | 0.001 | 0.07 | 2 | 0 | 0 | 0.0015873015873 | 0 | 0 | 0.000303258798786 |
| 6 | id_content_mean | 42 | 0.001 | 0.15 | 2 | 0 | 0.000396825396825 | 0.00119047619048 | 0 | 0.000104226005967 | 0.000262239925399 |
| 7 | id_content_mean | 42 | 0.0003 | 0.07 | 1 | 0 | 0 | 0.000793650793651 | 0 | 0 | 0.000157859346217 |
| 8 | id_content_mean | 42 | 0.0003 | 0.15 | 7 | 0 | 0.000396825396825 | 0.0015873015873 | 0 | 0.0001106916451 | 0.000339949861663 |
| 9 | ordinal_content_mean | 42 | 0.001 | 0.07 | 6 | 0 | 0.000396825396825 | 0.000793650793651 | 0 | 9.51636772096e-05 | 0.00017861999382 |
| 10 | ordinal_content_mean | 42 | 0.001 | 0.15 | 6 | 0.000396825396825 | 0.000793650793651 | 0.00119047619048 | 0.000114708264412 | 0.000208124499875 | 0.000281715382182 |
| 11 | ordinal_content_mean | 42 | 0.0003 | 0.07 | 1 | 0 | 0 | 0.000396825396825 | 0 | 0 | 7.56156398519e-05 |
| 12 | ordinal_content_mean | 42 | 0.0003 | 0.15 | 1 | 0 | 0 | 0.000396825396825 | 0 | 0 | 7.56156398519e-05 |
| 13 | content_transformer | 42 | 0.001 | 0.07 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14 | content_transformer | 42 | 0.001 | 0.15 | 4 | 0 | 0 | 0.000396825396825 | 0 | 0 | 7.18423003095e-05 |
| 15 | content_transformer | 42 | 0.0003 | 0.07 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | content_transformer | 42 | 0.0003 | 0.15 | 4 | 0.000396825396825 | 0.000396825396825 | 0.000396825396825 | 0.000119456347486 | 0.000119456347486 | 0.000119456347486 |
| 17 | id_mean | 2024 | 0.001 | 0.15 | 30 | 0.0107142857143 | 0.0202380952381 | 0.0369047619048 | 0.00394464708807 | 0.00640872238095 | 0.00970963750928 |
| 18 | id_mean | 2025 | 0.001 | 0.15 | 30 | 0.00833333333333 | 0.0166666666667 | 0.0329365079365 | 0.00320142559607 | 0.00530421973696 | 0.00858248415874 |
| 19 | id_content_mean | 2024 | 0.0003 | 0.15 | 6 | 0 | 0.000396825396825 | 0.000793650793651 | 0 | 0.0001106916451 | 0.000184760062258 |
| 20 | id_content_mean | 2025 | 0.0003 | 0.15 | 9 | 0.000396825396825 | 0.000396825396825 | 0.00198412698413 | 0.00025036894983 | 0.00025036894983 | 0.000548479228285 |
| 21 | ordinal_content_mean | 2024 | 0.001 | 0.15 | 16 | 0.000793650793651 | 0.00119047619048 | 0.00198412698413 | 0.000290359743547 | 0.00038377597901 | 0.000529166526491 |
| 22 | ordinal_content_mean | 2025 | 0.001 | 0.15 | 5 | 0 | 0.00119047619048 | 0.00238095238095 | 0 | 0.000300960327943 | 0.000531174019933 |
| 23 | content_transformer | 2024 | 0.0003 | 0.15 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 | content_transformer | 2025 | 0.0003 | 0.15 | 1 | 0 | 0 | 0.000396825396825 | 0 | 0 | 7.93650793651e-05 |

### 固定selected配置：全部seed及分组

All2520=train_observed1897+OOV623；train_observed1897=supported1666+unsupported231。OOV留在all分母，support分组不改变query set。

| architecture | seed | group | count | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 | ndcg@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| content_transformer | 42 | all | 2520 | 0.000396825396825 | 0.000396825396825 | 0.000396825396825 | 0.000119456347486 | 0.000119456347486 | 0.000119456347486 |
| content_transformer | 42 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 42 | train_observed | 1897 | 0.000527148128624 | 0.000527148128624 | 0.000527148128624 | 0.000158687398874 | 0.000158687398874 | 0.000158687398874 |
| content_transformer | 42 | supported | 1666 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 42 | unsupported | 231 | 0.004329004329 | 0.004329004329 | 0.004329004329 | 0.00130316015439 | 0.00130316015439 | 0.00130316015439 |
| content_transformer | 2024 | all | 2520 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 2024 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 2024 | train_observed | 1897 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 2024 | supported | 1666 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 2024 | unsupported | 231 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 2025 | all | 2520 | 0 | 0 | 0.000396825396825 | 0 | 0 | 7.93650793651e-05 |
| content_transformer | 2025 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| content_transformer | 2025 | train_observed | 1897 | 0 | 0 | 0.000527148128624 | 0 | 0 | 0.000105429625725 |
| content_transformer | 2025 | supported | 1666 | 0 | 0 | 0.000600240096038 | 0 | 0 | 0.000120048019208 |
| content_transformer | 2025 | unsupported | 231 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_content_mean | 42 | all | 2520 | 0 | 0.000396825396825 | 0.0015873015873 | 0 | 0.0001106916451 | 0.000339949861663 |
| id_content_mean | 42 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_content_mean | 42 | train_observed | 1897 | 0 | 0.000527148128624 | 0.0021085925145 | 0 | 0.000147044251793 | 0.000451593912172 |
| id_content_mean | 42 | supported | 1666 | 0 | 0.000600240096038 | 0.00240096038415 | 0 | 0.000167432740487 | 0.000514209874784 |
| id_content_mean | 42 | unsupported | 231 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_content_mean | 2024 | all | 2520 | 0 | 0.000396825396825 | 0.000793650793651 | 0 | 0.0001106916451 | 0.000184760062258 |
| id_content_mean | 2024 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_content_mean | 2024 | train_observed | 1897 | 0 | 0.000527148128624 | 0.00105429625725 | 0 | 0.000147044251793 | 0.000245437721081 |
| id_content_mean | 2024 | supported | 1666 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_content_mean | 2024 | unsupported | 231 | 0 | 0.004329004329 | 0.00865800865801 | 0 | 0.00120754521927 | 0.00201556431554 |
| id_content_mean | 2025 | all | 2520 | 0.000396825396825 | 0.000396825396825 | 0.00198412698413 | 0.00025036894983 | 0.00025036894983 | 0.000548479228285 |
| id_content_mean | 2025 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_content_mean | 2025 | train_observed | 1897 | 0.000527148128624 | 0.000527148128624 | 0.00263574064312 | 0.000332593438888 | 0.000332593438888 | 0.000728607092924 |
| id_content_mean | 2025 | supported | 1666 | 0.000600240096038 | 0.000600240096038 | 0.00180072028812 | 0.000378709335877 | 0.000378709335877 | 0.00060634910481 |
| id_content_mean | 2025 | unsupported | 231 | 0 | 0 | 0.00865800865801 | 0 | 0 | 0.00161034652235 |
| id_mean | 42 | all | 2520 | 0.0103174603175 | 0.0166666666667 | 0.0373015873016 | 0.0040576223256 | 0.00562067857826 | 0.00973890677164 |
| id_mean | 42 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_mean | 42 | train_observed | 1897 | 0.0137058513442 | 0.0221402214022 | 0.0495519240907 | 0.00539019939933 | 0.00746658408921 | 0.0129372931284 |
| id_mean | 42 | supported | 1666 | 0.015606242497 | 0.0252100840336 | 0.0564225690276 | 0.00613757998831 | 0.00850186675704 | 0.0147311194865 |
| id_mean | 42 | unsupported | 231 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_mean | 2024 | all | 2520 | 0.0107142857143 | 0.0202380952381 | 0.0369047619048 | 0.00394464708807 | 0.00640872238095 | 0.00970963750928 |
| id_mean | 2024 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_mean | 2024 | train_observed | 1897 | 0.0142329994729 | 0.0268845545598 | 0.049024775962 | 0.00524012159301 | 0.00851343194517 | 0.0128984114514 |
| id_mean | 2024 | supported | 1666 | 0.016206482593 | 0.030612244898 | 0.0558223289316 | 0.00596669307439 | 0.00969386578631 | 0.0146868466527 |
| id_mean | 2024 | unsupported | 231 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_mean | 2025 | all | 2520 | 0.00833333333333 | 0.0166666666667 | 0.0329365079365 | 0.00320142559607 | 0.00530421973696 | 0.00858248415874 |
| id_mean | 2025 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| id_mean | 2025 | train_observed | 1897 | 0.0110701107011 | 0.0221402214022 | 0.0437532946758 | 0.00425281628998 | 0.00704619596054 | 0.0114010859673 |
| id_mean | 2025 | supported | 1666 | 0.0126050420168 | 0.0252100840336 | 0.0498199279712 | 0.00484249249825 | 0.00802318951809 | 0.0129819088115 |
| id_mean | 2025 | unsupported | 231 | 0 | 0 | 0 | 0 | 0 | 0 |
| ordinal_content_mean | 42 | all | 2520 | 0.000396825396825 | 0.000793650793651 | 0.00119047619048 | 0.000114708264412 | 0.000208124499875 | 0.000281715382182 |
| ordinal_content_mean | 42 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| ordinal_content_mean | 42 | train_observed | 1897 | 0.000527148128624 | 0.00105429625725 | 0.00158144438587 | 0.000152379982245 | 0.000276475350387 | 0.000374234456036 |
| ordinal_content_mean | 42 | supported | 1666 | 0 | 0 | 0.000600240096038 | 0 | 0 | 0.000111313939625 |
| ordinal_content_mean | 42 | unsupported | 231 | 0.004329004329 | 0.00865800865801 | 0.00865800865801 | 0.00125136288449 | 0.00227044908954 | 0.00227044908954 |
| ordinal_content_mean | 2024 | all | 2520 | 0.000793650793651 | 0.00119047619048 | 0.00198412698413 | 0.000290359743547 | 0.00038377597901 | 0.000529166526491 |
| ordinal_content_mean | 2024 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| ordinal_content_mean | 2024 | train_observed | 1897 | 0.00105429625725 | 0.00158144438587 | 0.00263574064312 | 0.000385717740505 | 0.000509813108647 | 0.000702951843309 |
| ordinal_content_mean | 2024 | supported | 1666 | 0.000600240096038 | 0.00120048019208 | 0.00120048019208 | 0.000258509338579 | 0.000399811207347 | 0.000399811207347 |
| ordinal_content_mean | 2024 | unsupported | 231 | 0.004329004329 | 0.004329004329 | 0.012987012987 | 0.00130316015439 | 0.00130316015439 | 0.00288923885418 |
| ordinal_content_mean | 2025 | all | 2520 | 0 | 0.00119047619048 | 0.00238095238095 | 0 | 0.000300960327943 | 0.000531174019933 |
| ordinal_content_mean | 2025 | oov | 623 | 0 | 0 | 0 | 0 | 0 | 0 |
| ordinal_content_mean | 2025 | train_observed | 1897 | 0 | 0.00158144438587 | 0.00316288877174 | 0 | 0.000399799697636 | 0.000705618624265 |
| ordinal_content_mean | 2025 | supported | 1666 | 0 | 0.000600240096038 | 0.00120048019208 | 0 | 0.000143945057964 | 0.000266271016602 |
| ordinal_content_mean | 2025 | unsupported | 231 | 0 | 0.00865800865801 | 0.017316017316 | 0 | 0.00224505437163 | 0.00387424682499 |

### 全部42/2024/2025的mean/sample std ddof1

仅描述统计，不是置信区间或显著性/因果证据。

| architecture | metric | mean | sample std |
| --- | --- | --- | --- |
| content_transformer | ndcg@10 | 3.98187824952e-05 | 6.89681543773e-05 |
| content_transformer | ndcg@20 | 3.98187824952e-05 | 6.89681543773e-05 |
| content_transformer | ndcg@50 | 6.62738089503e-05 | 6.07946606164e-05 |
| content_transformer | recall@10 | 0.000132275132275 | 0.000229107249678 |
| content_transformer | recall@20 | 0.000132275132275 | 0.000229107249678 |
| content_transformer | recall@50 | 0.00026455026455 | 0.000229107249678 |
| id_content_mean | ndcg@10 | 8.345631661e-05 | 0.000144550580581 |
| id_content_mean | ndcg@20 | 0.000157250746676 | 8.06427294857e-05 |
| id_content_mean | ndcg@50 | 0.000357729717402 | 0.000182510274742 |
| id_content_mean | recall@10 | 0.000132275132275 | 0.000229107249678 |
| id_content_mean | recall@20 | 0.000396825396825 | 0 |
| id_content_mean | recall@50 | 0.00145502645503 | 0.000606160806211 |
| id_mean | ndcg@10 | 0.00373456500325 | 0.000465154889999 |
| id_mean | ndcg@20 | 0.00577787356539 | 0.000568783105084 |
| id_mean | ndcg@50 | 0.00934367614656 | 0.000659374024484 |
| id_mean | recall@10 | 0.00978835978836 | 0.00127561518003 |
| id_mean | recall@20 | 0.0178571428571 | 0.00206196524711 |
| id_mean | recall@50 | 0.0357142857143 | 0.00241379465488 |
| ordinal_content_mean | ndcg@10 | 0.000135022669319 | 0.000146241927822 |
| ordinal_content_mean | ndcg@20 | 0.000297620268942 | 8.78733607309e-05 |
| ordinal_content_mean | ndcg@50 | 0.000447351976202 | 0.000143449009987 |
| ordinal_content_mean | recall@10 | 0.000396825396825 | 0.000396825396825 |
| ordinal_content_mean | recall@20 | 0.0010582010582 | 0.000229107249678 |
| ordinal_content_mean | recall@50 | 0.00185185185185 | 0.000606160806211 |

### 全24预算、zero与资源

minima/峰值只覆盖terminal invocation segment，不冒称首run历史bounded50→170→220全局minimum；train/epoch_eval累计计时不是外部full verifier walltime。

| slot | epochs | steps | stop | drawn | skip | zero item/user | RAM/VRAM/disk min | allocated/reserved MiB | train/epoch_eval s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 5100 | max_epochs | 2603550 | 0 | 98919/0 | 0.490917222365/0.974786324786/0.945757240295 | 236.165039062/260 | 988.173480657/1051.39098538 |
| 2 | 30 | 5100 | max_epochs | 2603550 | 0 | 98919/0 | 0.486950795187/0.974786324786/0.91907824707 | 236.165039062/260 | 983.113807321/1051.54726651 |
| 3 | 10 | 1700 | patience | 867850 | 0 | 98919/0 | 0.501546858418/0.974786324786/0.910828292847 | 236.165039062/260 | 313.276143121/349.434486919 |
| 4 | 9 | 1530 | patience | 781065 | 0 | 98919/0 | 0.499919739497/0.974786324786/0.903413352966 | 236.165039062/260 | 280.160331996/315.606711864 |
| 5 | 5 | 850 | patience | 433925 | 0 | 0/0 | 0.499844073641/0.974175824176/0.898170310974 | 266.227539062/290 | 154.777848945/236.447745845 |
| 6 | 5 | 850 | patience | 433925 | 0 | 0/0 | 0.506322102684/0.974175824176/0.89293106842 | 266.227539062/290 | 155.206556365/237.262648084 |
| 7 | 4 | 680 | patience | 347140 | 0 | 0/0 | 0.511415207178/0.974175824176/0.888704078674 | 266.227539062/290 | 124.500832186/188.508775064 |
| 8 | 10 | 1700 | patience | 867850 | 0 | 0/0 | 0.495954751697/0.974175824176/0.878065895081 | 266.227539062/290 | 312.385979148/473.206589232 |
| 9 | 9 | 1530 | patience | 781065 | 0 | 0/0 | 0.48812403261/0.974175824176/0.868465263367 | 266.227539062/290 | 282.601469492/424.606801301 |
| 10 | 9 | 1530 | patience | 781065 | 0 | 0/0 | 0.500560740201/0.974175824176/0.858865844727 | 266.227539062/290 | 283.394286418/425.936662175 |
| 11 | 4 | 680 | patience | 347140 | 0 | 0/0 | 0.505523156703/0.974175824176/0.854634811401 | 266.227539062/290 | 123.919454779/189.144111461 |
| 12 | 4 | 680 | patience | 347140 | 0 | 0/0 | 0.492106717189/0.974175824176/0.850408905029 | 266.227539062/290 | 123.636772751/188.70793816 |
| 13 | 4 | 680 | patience | 347140 | 0 | 0/0 | 0.497732369005/0.968925518926/0.846160423279 | 425.062011719/546 | 129.660657262/194.289327801 |
| 14 | 7 | 1190 | patience | 607495 | 0 | 0/0 | 0.501866120147/0.968925518926/0.838839607239 | 425.062011719/546 | 228.067724836/330.917902737 |
| 15 | 4 | 680 | patience | 347140 | 0 | 0/0 | 0.489025250222/0.968925518926/0.834588104248 | 425.062011719/546 | 128.645106497/188.336729546 |
| 16 | 7 | 1190 | patience | 607495 | 0 | 0/0 | 0.494079180894/0.968925518926/0.827267242432 | 425.062011719/546 | 228.189473094/332.300680417 |
| 17 | 30 | 5100 | max_epochs | 2603550 | 0 | 98919/0 | 0.497419864779/0.974786324786/0.800611755371 | 236.165039062/260 | 977.922328584/1053.48241095 |
| 18 | 30 | 5100 | max_epochs | 2603550 | 0 | 98919/0 | 0.492268806361/0.974786324786/0.773933479309 | 236.165039062/260 | 994.949746814/1056.19380971 |
| 19 | 9 | 1530 | patience | 781065 | 0 | 0/0 | 0.500135714454/0.974175824176/0.76431313324 | 266.227539062/290 | 282.872462966/426.203693979 |
| 20 | 12 | 2040 | patience | 1041420 | 0 | 0/0 | 0.506088408219/0.974175824176/0.751562431335 | 266.227539062/290 | 380.786458052/567.983043235 |
| 21 | 19 | 3230 | patience | 1648915 | 0 | 0/0 | 0.488518710197/0.974175824176/0.730725006104 | 266.227539062/290 | 614.360246619/898.508447601 |
| 22 | 8 | 1360 | patience | 694280 | 0 | 0/0 | 0.503797531261/0.974175824176/0.722153617859 | 266.227539062/290 | 250.316622749/378.671229226 |
| 23 | 4 | 680 | patience | 347140 | 0 | 0/0 | 0.503868409289/0.968925518926/0.717900917053 | 425.062011719/546 | 129.874187056/188.852567895 |
| 24 | 4 | 680 | patience | 347140 | 0 | 0/0 | 0.491163621238/0.968925518926/0.713652534485 | 425.062011719/546 | 128.514471596/189.006551888 |

固定selection为IDmean LR.001/T.15，其三seed指标高于本次内容架构；不作统计显著性或外推声明。Transformer seed2024最佳all指标全部0，内容模型命中稀少，如实保留。所有best user zero计数0，内容模型item zero0，不能把弱表现简单归因全零向量。IDmean zero item98919为冻结unsupported规则，不是OOV query数。OOV各模型/seed均0，不能声称解决冷启动。Verifier保证预算与checkpoint重导出/rank/metric一致，不证明优化充分、架构普遍优劣或线上收益；无额外诊断训练或因果猜测。

### 收口磁盘与RAM容量

本次快照disk total536870912000bytes=500GiB，可用383137030144bytes≈356.824GiB；有效cgroup RAM limit211900432384bytes≈197.348GiB，current21898399744bytes≈20.394GiB，headroom190002032640bytes≈176.954GiB。headroom是limit-current且current含cache，不保证可分配量。按磁盘20%余量新增预算275762847744bytes≈275.763GB；协调转达interactions约381GB已超预算，尚未计metadata/processed/checkpoint/index。该网页大小未由此对话独立核，只作协调输入；当前不下载/扩资源。RAM与磁盘不能互换。
# P4E local source verification (2026-10-08; no real ablation metrics)

## Astra reviewfix revision (supersedes current source/regression values below)

Revised consumer source SHA573c1bdc586f27ead094bf20d497ae16a447264faedbaf421302e230b147a508; originalfirstpackageandpublicmanifesthashes remain preserved inignoredcache.

Fresh new suites31tests30.117s exit0; frozen legacy163tests27.096s exit0 skip1; actual newROOT194tests48.762s exit1 only the same historicalproducer identity failure,skip1. Three exact commands are unchanged from first source review and recorded in v2 package; logs cache/p4e-reviewfix-{new,root,legacy}.log. No real operations.

Prepare/verify-watch CLI new RED reproduced three gates returning0 incorrectly: outside_root,shared_deadline,cumulative_disk. All three freshGREEN return2 before real inputs. Device/precision binding missingRED→GREEN; sameproducerdevice required and unavailableCUDA refusesCPUfallback. Initial separate targeted2tests1.424s exit0.

Revised pilot proof proposal requires180optimizersteps total if true continuous30 vs fresh-process split7→20→30 comparison is performed for each of3strategies. Earlier90 ceiling cannot fund both independentarms. This is an explicit proposal increase, NOT authorization or execution; fullvalidation/verifier/worstallocation and receipts remain included in boundedwindow. Fixed formal9run budget remains unapproved.

First-review historical consumer source SHA: 919e51a4066aedba76c5bac04c87658070a2772d2047b64fb0847f3b262f8237. All results below are local validation, not server resource measurements or model-quality experiments.

- Fresh three new suites on actual P4E ROOT: 29 tests, 22.463s, exit0, no skipped tests. Command: `PYTHONPATH=src:tests <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m unittest test_retrieval_ablations test_retrieval_watch test_retrieval_ablation_run -v`.
- Fresh frozen self-contained legacy ROOT suite: 163 tests, 24.621s, exit0, one FAISS environment skip. Same Python, `-B -m unittest discover -s tests -v`, cwd `cache/p4e-legacy-snapshot`.
- Final actual new ROOT complete suite: 192 tests, 41.181s, exit1, one failure and one FAISS skip. Failure: `test_p4d_v2_verifier_compat.CompatibilityTests.test_complete_producer_pin_and_explicit_cli_identity`; old CLI rejects new dynamic producer source. All other tests passed. Log retained at `/tmp/p4e-final-root-20261008.log` and fixed review cache copy. No old source/fixture/pin repair.
- Earlier sandbox run: 189 tests, 39.365s, exit1, 19 PermissionErrors creating old P1 temporary outputs plus the same single producer identity failure. Normal approved escalation removed PermissionErrors: subsequent complete run 189 tests,43.817s,exit1,only identity failure. These are superseded validation observations, not hidden failures.
- Recovery targeted completion RED: missing `verify_completion` assertion, exit1; first GREEN 1test,2.571s,exit0. Summary delivery/dispatcher subsequently obtained fresh RED then GREEN; latest integrated targeted pass 1test,3.833s,exit0. Initial SyntaxError and dispatcher error preserved as invalid/diagnostic failures rather than asserted behavioral RED.
- Source-stage proposal only: prepare plus replay cap3600seconds/4GiB; pilot cap7200seconds/8GiB, three strategies max30 cumulative optimizer steps each/90 total plus worst-allocation backwards without optimizer updates. These are proposed ceilings, not measurements or authorization. Formal runtime cap awaits actual accepted pilot.

No real P4E run, no CUDA resource peak, no actual shared-slices artifact or real paired summary was produced. CPU synthetic scores remain test assertions only. Public manifest contains source pins/protocol/status, not numerical experiment claims.


### P4E fresh-process runner 本地验证及一次只读 preflight（2026-10-08）

最小 stage receipt targeted fresh RED：1test/2.127s/exit1，缺 train_return_receipt；实际 synthetic worker 生成 segment，非服务器训练失败。修复后 fresh full runner regression 6tests/31.156s/exit0。最终独立 CPU synthetic CLI run exit0、verify exit0，3固定策略各 continuous30 对 split7→20→30，合计180optimizersteps、12唯一 child PID、12实际workerexit0、3 exact state/output比较。CUDA RNG 在 CPU fixture 中为空，不能作为 CUDA 恢复证明。真实模型效果未测。final证据 pins/路径见 ignored runner review package。

早先 fresh missingrunner、替换 argv、缺 publication preflight、缺 handle inspection 与伪造 comparison 检查取得 RED后GREEN；2次 SyntaxError 及 ps sandbox PermissionError/retained-child ResourceWarning属于实现诊断，不冒充行为 RED；最新完整回归通过。旧 ROOT suite 仅旧P4D producer兼容测试失败的限制保留，未修改旧pins或源码。

一次批准的只读 server preflight actualexit0：GPU utilization0%，GPU memory used1MiB/free48541MiB/total49140MiB；host MemAvailable290414342144/total540677795840 bytes；visible root cgroup current18648039424/max211900432384 bytes；workspace disk free383134175232/total536870912000 bytes。未发现活跃workspace项目Python或GPUcompute进程，旧P4Dparent不在。是单点瞬时观察，不是峰值/历史minimum，也未完整重验所有cgroup祖先；实际operation gate仍须冻结resource_guard fresh检查。现有Python3.11.16/Torch2.11.0+cu128/CUDA12.8/NumPy1.26.4/PyArrow19.0.1/PyYAML6.0.2，cuda_available true，未执行CUDA模型计算。prepared/core/validation外部pins一致，raw目录存在但此preflight未重新hash全部raw25文件；实际real_data会检查。

协调批准prepare+replay预算设计3600s/4GiB，pilot7200s/16GiB/180steps及zero-update worst probes；这不是实际操作放行。旧90step/8GiB提案仅历史。formal9run仍未批准。


### P4E获批真实prepare及独立replay终态（2026-10-08）

exclusive部署121固定archive成员+2proposal文件，source573c保持。prepareactualexit0、独立verifyactualexit0、outerexit0；阶段wall68.869s/76.094s，prepare开始至verify结束145.034s。共享原3600s/4294967296bytes持久budget，operation输出63711384bytes，optimizersteps0。watch/sharedslices正文留server，只返回smallhash摘要；全部externalpins见cache/p4e-prepare-hash-only-delivery.json。生产resource_guard检查通过，无报错；未记录历史minimum，不声称全程资源实测minimum。当前awaiting coordinator prepare acceptance；GPUpilot及formal9run未执行。只读验收提案300s是独立观察上限，不重置已完成实验原预算，不重新prepare。


P4E prepare阶段协调fresh验收：readonlyscript actualexit0，42.937s。独立raw Counter对账：catalog120415、knownzero98919、positiveitems21496、positiveusers18915、trainlike105700、trainingexamples86785、validationqueries2520，全部checksPASS。Astra正式accepted，不是作者冒充fresh重跑。GPU pilot plan冻结单点检查：currentresource余量RAM.5232177473/VRAM.9878103378/disk.7135219727，无活跃项目Python或GPUcompute。预算设计7200s/16GiB/180steps，未创建pilotbudget或启动GPU；非资源峰值/minimum。


P4E GPU pilot首次attempt失败ledger：第一个uniform_negatives continuous30 workeractualexit2、wall.131380s，outeractualexit2、wall17.140336s；后续11worker未启动，无completion、无trainreceipt/checkpoint，真实训练前resource路径检查失败，optimizersteps0。原operationbudget7200s/16GiB不reset，output观测3586bytes。parent在失败前执行首个worstgeometryzerooptimizer GPU probe；没有完整worstreceipt，不能声称三项probe成功。终态瞬时GPUutil0%、memoryused1MiB/free48541MiB，是观察而非历史峰值。原stage/outer/log/plan/budgethash见ignored诊断/原serverreceipt。无模型质量或吞吐结论；formal预算须等真正acceptedpilot。


P4E ignored runner parent-layout修复TDD：fresh真实资源路径RED1test1.007s exit1（shutil.disk_usage/os.statvfs缺parent）；仅Linux/proc/cgroup/GPU读数mock，磁盘路径不mock。最小parentexclusive mkdir修复后完整runner7tests29.609s exit0，含原6真实CPU进程proof；无新server尝试。原failedrunner/source/prepareconsumer保持，新runnerhash与diff见cache/p4e-runner-parent-layout-v2-review.json。


P4E第二attempt已批准静态准备，未启动模型：oldfailedbytes3586、新diskcap17179865598bytes，累计上限仍17179869184；newledger.started1791401343.3344553/deadline1791408543.3344553/wall7200保持原值，冻结时剩余6358.260s，时间损耗含修复/审查，不reset。oldactualoptimizer0，后续总上限180。旧root/runner保持原hash，新runner独立v2路径，newproduction/opplan/budget/lineage外部pins见cache/p4e-pilot-v2-static-hash-only.json及v2exactproposal，未额外GPU计算。


### P4E v2真实GPU独立进程pilot完整终态（2026-10-08）

Formal scope review watch绑定修复TDD：既有7tests/Astra0.531s通过未覆盖缺口，不抵销Important finding。fixture补真实production plan/runtime watch与12manifest.binding.plan字段；derive目标freshRED1test/9negative cases0.285s exit1，全部ValueError not raised；覆盖缺/错production、缺/错operationwatch、重pin后productionwatch改动、缺/错segment生产plan/watch以及actualwatch bytes变动。相关operation/receipt/handle/segment/comparison局部hash重算后仍应受externalproducer/watch约束。dispatch目标freshRED1test/2cases0.010s exit1，missing/different watch进入resourceguard。最小GREEN后完整scoped9tests0.780s exit0，无GPU或全项目重跑；新fixture只是metadata与微型actualchild，未证明真实模型重训。source573c、runner479d、safe原pilot delivery eefc、旧review包dca3保持原hash。原reviewer定点复核待定。

协调补充：Astra独立metadata fresh SSH actualexit0并accepted bounded pilot，使用已审ad9a脚本；作者后续只增加显式completion/outerreceipt externalpins，最终脚本2aa24a1d6c7efc777b6b6f7792e54d490bf24b6ae4ff9de809456e5e1ccc8b80。未增加数值replay；P4E仍未Complete。

P4E formal候选本地TDD：首次4test freshRED0.001s exit1为模块不存在断言；初GREEN有1error（macOS全局/var别名误作proof内部symlink），仅约束proof内components后4test0.253s通过。追加调度/actualchild观察覆盖6test0.517s通过；新增预算内derived-copy位置目标freshRED0.008s assertion失败，最小copy及argv修复后6test0.562s通过。新增错误seed门第一mock缺receipt导致无效RED，未冒称功能RED；补mock真实metadata后freshRED0.011s明确substituted seed reaches verify，再最小terminal固定slot/pilot校验后7test0.575s通过。只有ignored adapter/dispatcher/tests变更，无真实模型运行。source573c/运行runner479d保持。

预算补证（既有P4D accepted终态）：IDmean四个5100step/30epoch training977.922–994.950s，累计epochvalidation1051.391–1056.194s；六IDmean最大checkpoint102422609bytes、最大manifest16026248bytes，旧首bounded run127checkpoint，其余完整连续cadence126。fullverifier pid/exit mtime wall代理slots2/17/18=1767.650898/1780.517572/1777.129038s，10/9epoch slots3/4=579.199863/522.927139s；首slot不同历史attempt代理未得到，不编造。计时代理非独立调用clock或纯kernel，旧producer与本次P4Esource不同。固定metadata/terminal sourcepins见ignored budgetevidence/proposal。

Formal提案尚未批准：9run固定30epoch/170steps每epoch，actualtrain最多45900steps；每slot独立verify加summary生产verify_completion两遍，drawreplay上限91800steps、epoch重导出verification540次+train270次。rounded .5s/trainstep、.5s/replaystep、45s/fullvalidation及5000s overhead给110300s planning，建议48h/192GiB hardcap只作为有限提案，不是预计必跑时长/承诺。1134checkpoint*128MiB+1134trace-manifest*24MiB≈168.33GiB，剩23.67GiB margin；everymanifest包括累计prediction/trace，best仅指针无额外tensorcopy，验证内存重导出不创建disktemp，summary/actualreceipt/copy/budget也累计计费root。真实fullP4Etrace未执行，上限由实际guard强制，资源at-cap20%free须启动前fresh核；不自动扩容/删除/追加校准/HPO。

原outer实际exit0，12worker实际exit0；三策略连续30与split累计7→20→30完全state/output对齐，实际optimizer更新总数180，三个worst probe各backward1/optimizer0。completion外部SHA3fb47d4e0655bba13941d8233e84d4c7330aa2ea375231664fddd5e0d7f50f5a，outerreceiptSHA6f50cf518f0526ddd515bcda376e4959687891deae011e2043b60ffb1f9e49ad。原7200s绝对budget不reset，secondouterwall1382.041129s，原budgetstart至终态2416.285655s；newroot1127452867bytes+oldfailed3586=1127456453bytes，低于原16GiB累计上限。

| 策略/arm/cutoff | actual worker wall(s) | train函数(s) | verify函数(s) | postverify导出+receipt尾段(s) | incremental training(s)/steps |
|---|---:|---:|---:|---:|---|
| uniform_negatives/continuous/30 | 75.656572 | 13.243085 | 11.072550 | 37.016795 | 11.917006/30 |
| uniform_negatives/split/7 | 60.501883 | 4.745200 | 3.772591 | 37.503566 | 3.324929/7 |
| uniform_negatives/split/20 | 70.136505 | 6.843391 | 7.748990 | 37.311586 | 5.549926/13 |
| uniform_negatives/split/30 | 84.066842 | 5.823735 | 10.901174 | 37.934913 | 4.432945/10 |
| user_balance/continuous/30 | 67.517665 | 5.733188 | 4.160002 | 37.383047 | 4.348549/30 |
| user_balance/split/7 | 57.493421 | 3.273447 | 1.926151 | 37.861356 | 1.842224/7 |
| user_balance/split/20 | 61.391515 | 3.563052 | 3.114495 | 37.999438 | 2.303986/13 |
| user_balance/split/30 | 63.391522 | 3.135979 | 3.927998 | 38.383184 | 1.907512/10 |
| watch_history/continuous/30 | 69.951754 | 9.402210 | 7.254473 | 38.854211 | 6.213294/30 |
| watch_history/split/7 | 62.561212 | 5.168281 | 4.287734 | 38.587420 | 2.010623/7 |
| watch_history/split/20 | 68.746615 | 5.978455 | 5.563809 | 38.447042 | 3.116251/13 |
| watch_history/split/30 | 71.925959 | 5.542514 | 6.982692 | 38.453002 | 2.564550/10 |

所有segment cumulative skip0，连续30每项drawn15360，均未完成真实epoch，epoch evaluation_seconds0；导出使用完整120415catalog/2520validationqueries。尾段包含state加载、fullvectors/rank/metrics与receipt/exit，并非纯evaluate函数时间；函数run_session计时不等于manifest训练kernel时间。parent减actualchildwall残余568.699664s包含inputs、worst、comparison和prepublication globalverify，未独立计时globalverify，不称其纯耗时。prepublication全模型验证由固定原runner实际完成后发布；作者新增脚本仅hash/metadata核，不另重放模型。

bounded worst geometry allocated三项239734784bytes，reserved247463936/251658240/251658240bytes；historywidth50、候选640/512/512、maxpast185。这是synthetic worst probe范围峰值，不是全部训练历史peak。终态瞬时RAM余量.520286、VRAM.987810、disk.711421；观测不足以报告全程minimum。完整receipt/checkpoint/source/CLI/operation/budget/lineage/hash已核，原failedfiles逐hash保持。当前awaiting coordinator pilot acceptance，不称P4E Complete，不含正式模型质量结论。
# P4E静态启动包资源与协调验收证据

真实正式启动：Astra fresh9tests0.766s exit0、原reviewer PASS并FINAL EXACT START GATE。launcher SHA797ec2af5cd3d91bbc5ff45dc869f03f16b9c9b768f3d6865c8e6615218f91ad/v2review SHAf2498964f29dcbda45e5b2864f5aacadfdb07aeb9e64646a73fbf780aa21f353，原start输入包f6d363ad保持。actual once启动owner660147/birth121542991、dispatcher660156/birth121543017。outer预算SHAf79802feb2fce1db5a942dc13acf9462b092dfaa8240077e2c2b353d743fa58e，started1791406275.4360795、deadline1791579075.4360795、172800s/206158430208bytes覆盖static+formal。prestart无活同任务；RAM余量0.5314413561400081/VRAM0.9878103378103378/disk0.7114207916259766，disk满cap后0.32742079162597654，均瞬时观察。UTC1791406312.8753402 fresh只读核owner/dispatcher birth及原group一致，slot01 uniform_negatives seed42 train660171/birth121543044、pgid/sid660156实际live，1个step manifest已发布；innerbudgetSHAa58626d64181d6bf5c037bf5b1105101dda94fa298eda9af25fef00cdf63b8ee更晚起点不能resetouter。尚无actual train/verify完整终态或最终消融结果。已有p4d heartbeat原5min周期更新P4E/ACTIVE，无重复automation，原owner预算不因观察窗口重置。

Outer真实helper接口定点修复：Astra预读指出SimpleNamespace遗漏STDOUT/run；新增真实已审helper.invoke CPU integration，fresh1test0.012s exit1、AttributeError STDOUT（实际child创建前）。仅将helper引用改为完整属性代理/Popen局部覆盖，未改helper源码；真实STDOUT/identity fallback/actualreceipt路径fresh通过，最终9tests0.705s exit0。先前8项仅假Helper不能证明真实invoke接口，这个限制已纠正。固定候选review v1原字节保留，修订v2待协调fresh验收。无server部署或model操作。

Outer targetedTDD：初始scaffold fresh7tests0.006s，6个behavior FAIL及1个missing-status ERROR，日志/private/tmp/p4e-outer-red.log；当时shell末尾cat exit0只属cat，不作unittest退出码。真实源码fresh第一次7tests0.019s exit1（全局Popen影响ps）；改为helper引用隔离后7tests0.027s exit1（sandbox ps PermissionError）；escalated CPU scoped7tests0.305s exit0。加入独立ps在finally前核真实grandchild PID/group终态及success-exit残留后代拒绝，最终fresh8tests0.531s exit0。数值为实际本地CPU测试，Linux/GPU/完整suite未运行，旧生产/dispatcher4224bbe/helper479d保持。v1启动包实际json.loads合法且SHAf6d363ad一致，无格式修复或训练失败。

Astra转达fresh scoped9tests0.736s exit0及原reviewerClosed/PASS，本轮不冒称作者重复测试。真实静态准备actualexit0，UTC epoch1791405543.5063815；formal未启动、无数值replay。dispatcher SHA4224bbe2f64fe61b50f596fbe84f743999917bd07bb2887b5b9d88eea49ec281、helper479d4afff029ca068d18a6091dd86b098211d4985d360fb2f840d83475b42f7a、source573c1bdc586f27ead094bf20d497ae16a447264faedbaf421302e230b147a508保持。

真实derived SHA0e0e8db89ad2eadb873e9ed324b0d085dacd0cdb9f27ad0d6065dfe10ffe97d3（5740bytes）；formal plan SHA8d8e2a657824743f02cb9f7360dfeda9b120b7b26f81a9deec6ac49040257f3d（4437bytes），固定9slots。批准设计ceiling172800s/206158430208bytes；绝对启动时间未创建。静态准备瞬时RAM余量0.5358965811086192、VRAM0.9878103378103378、disk0.7114201736450195，disk total536870912000/free381940797440bytes，减去整个192GiB预算后余量0.3274201736450195。此前同次只读preflight无P4E活句柄，GPU utilization0%、used1MiB/49140MiB，visible cgroup finite max211900432384/current19550302208bytes。这些是瞬时观察，非历史峰值或保证；静态产物及未来outer logs仍须联合累计预算，启动前再次fresh gate。
## P4E metadata/hash-only collector 本地验证（2026-10-08）

### P4E有界诊断实际观测（UTC 2026-10-07T22:35:42.131565Z）

slot01 train actualexit0，ended1791409495.0023491；verify started1791409647.535092，诊断时已约2894.596473s/48.24min，无verify receipt。owner660147/birth121542991、dispatcher660156/birth121543017及verify787369原birth全部匹配。一次SSH内3s双采样verify CPU ticks677380→677499、rchar7204222952→7208179588、wchar349436→351452，state S→R；这些含辅助资源检查，不能据此声称具体record/epoch核验进度。Train log107bytes/mtime21:44:53.492317Z，无错误；verify log0bytes/mtime21:47:27.534505Z，owner/dispatch log均0bytes，无error excerpt；无finalcompletion/summary/dispatch-completion。原预算剩166533.19846057892s，联合static/formal15698406809bytes；瞬时RAM/VRAM/disk余量0.5069837774087498/0.9803622303622304/0.6821790466308594，随后GPU utilization0%、used368MiB/49140MiB，属不同瞬时点，非历史峰值或minimum。Snapshot errors空，不重启或改预算。

既有pilot uniform continuous30的verify函数11.072550s，但只30steps且未完整epoch；当前fullslot5100steps/30epoch需record replay及逐epoch重导出，不能用pilot短计时做同比/ETA。旧P4D fullverifier wall代理1767.650898–1780.517572s，source和实际采样策略不同，不是本次故障阈值。当前结论为慢、尚未定位性能瓶颈，不证明卡死；raw/runtime日志未导出，ignored诊断只small进程/资源/error摘要。

仅ignored cache/p4e-final-readonly-delivery.py及test_p4e_final_readonly_delivery.py。本地命令 `<LOCAL_VENV_WORKTREE>/.venv/bin/python -B cache/test_p4e_final_readonly_delivery.py`。初始stub未实施：9tests/0.294s exit1，9个NotImplementedError。定点关联pin断言收口首轮10tests/0.371s exit1，7FAIL/3ERROR，Mac临时目录祖先symlink；fixture改用resolved路径后10tests/0.493s exit1，2ERROR，默认JSON摘要超60000bytes。限定输出上限改131072bytes后10tests/0.603s exit0；再加shared-input漂移与Linux owned-process absence fixtures，最终fresh12tests/0.611s exit0。失败为本地fixture/collector，不是server训练；未重复production suite或数值replay。Opaque checkpoint/统一cohort-count/aggregate只证明schema拒绝与允许，不证明真实统计或CUDA。真实九run终态尚未采集，owner_actual_exit未取得，不填0；server冻结文件及预算未改。
## G3 minimal original-parent 本地CPU验证（2026-10-08）

协调合并scoped gate转达：Astra fresh parent4tests/0.452s exit0，parent/test pins e2683e1759786510c46b76db6263902fa9731c70637d43e47f457f4b264c1993/4e372905d55463f3224d853b21a2de1124a6596cd3f6571dfb621c7a52abff44匹配；proposal SHAbe7edef70e6de09f5a49255a6231244a72d0b2cc4d38ee3bcda0de22dfd93cc2完整chain及3remote bodies AST compile、7localbytes/hash独立PASS。报告/private/tmp/g3-astra-fixed-chain-static-check.json；独立Sol parent+actualchain PASS，无Critical/Important，起止hash未变。这是协调证据转达，本轮不重复tests、不SSH；三文件冻结。仅准备通过，P4E未终态/G3未启动，真实Linux/GPU证明不补造。

后续明确scope在同一ignored提案补可执行链，未新增脚本或修改parent。实际Python AST compile及七文件本地bytes/hash核exit0；receiver有效输入、错size、错hash、occupied、expired五fixtures均按预期PASS。随后完整七文件exclusive传输fixture与bootstrap固定argv/env、预创建logs/marker及allowed-file race fixture PASS，mock Popen调用一次、实际child零、SSH零。Linux proc birth与资源/admission guards在这些CPU fixtures中mock，未声称真实server/GPU证明；Astra转达parent fresh4tests/0.452s exit0，仅父进程部分accepted测试，不是whole-chain exact-start。

命令 `<LOCAL_VENV_WORKTREE>/.venv/bin/python -B cache/test_g3_original_parent.py`，cwd为own P4E worktree。初始fresh behavioral RED：3tests/0.003s exit1，actual exit为None与duplicate reservation未拒绝；最小实现后3tests/0.350s exit0。新增birth观察失败fixture：fresh4tests/0.362s exit1，OSError及短子进程ResourceWarning；独立ps确认原PID30215自然消失，未kill。修复仅在元数据OSError后仍wait原child并标needs_attention，最终fresh4tests/0.472s exit0。actual短child exit0/7、原wait spy/PID绑定、重复启动拒绝与birth失败仍wait已验证；不证明Linux bootstrap、GPU、完整launcher或真实pilot。parent8620bytes SHAe2683e1759786510c46b76db6263902fa9731c70637d43e47f457f4b264c1993，tests2866bytes SHA4e372905d55463f3224d853b21a2de1124a6596cd3f6571dfb621c7a52abff44。服务器未上传/计时/执行，冻结G3/P4E source与预算均保持。

## P4E collector 协调 scoped gate 转达（2026-10-08）

Astra实际fresh 12tests/0.576s exit0；前后collector SHA19f2c17d9ed2942e075cba447f676613db7b7281c00a6eb4030c3d24ca1839ec、test SHAb8d55cb7fdfa82d5419525553f72fa96f2cc84ceb86c5a32a280eac796351325一致；独立Sol静态兼容性PASS，无Critical/Important。这是协调证据转达，不冒充作者重跑。冻结脚本仅全队列真实终态后一次只读采集，输出上限128KiB；owner实际exit缺口未裁决，P4E未Complete。本轮未SSH或执行collector。


### P4E slot 06 stopping evidence — 2026-10-08 UTC

本次既有 heartbeat 单次只读 snapshot 核实：slot 06 train/verify actual exit 均为 0；terminal manifest 为 `manifest-step-1020.json`，SHA `d26b5775f5770ebad1a619c13d8a69b14698a12c652f9320e882f7727e7fb606`。字段 `status=complete`、`max_epochs=30`、`patience=3`、`selection.epoch=6`、`selection.best_epoch=3`、`selection.bad_epochs=3`、`last.epoch=6`、`best.epoch=3`，evaluation epochs 为 1–6。Manifest 没有显式 `stop_reason` 字段；依据冻结源码 `ValidationSelection.stopped` 的 `bad_epochs>=3 or epoch>=30` 条件及上述实际字段，停止分支为 patience early stopping，与冻结规则一致。并非仅由 exit0 推断；没有追加训练或数值 replay。脱敏证据：`cache/p4e-slot06-stop-evidence-20261008t0454.json`。


### G3 concurrent v4实际prepare与终态诊断

唯一approved实际bootstrap exit0，upload/extract/metadata/assembly/prepare actual exit0；60s checkpoint status passed_continue_same_pilot不代表optimizer pilot通过。outer actual exit1，pilot.started absent，parent actual wait exit未知；parent cleanup_complete=true，独立fresh owned groups live=[]，closure within_original_budget=true。白名单只读diagnostic实际SSH exit0，无模型replay、无阶段重跑。

原预算start1791467906.4949563/deadline1791469706.4949563，1800s/2147483648bytes、200updates/128decode/1resume，不refund/reset。谓词UTC1791468450.1946635剩1256.3002927303314s；permit ISO2026-10-08T14:28:26.494956+00:00解析1791469706.494956，减原float为-2.384185791015625e-07s，精确等号false但canonical ISO相等。binding/runtime完全相同。当前pool disk2076106752bytes≤当前allowed2077118464bytes，combined output4304896bytes，非历史失败点disk证明。源码remaining_pilot_pool line288量化，run_operation line502精确比较。入口cache/g3-concurrent-admission/actual-v4-terminal-review-v1.json。


## P4E固定九run正式终态交付（awaiting coordinator acceptance）

一次已审metadata/hash-only collector真实SSH exit0，脚本SHA与冻结版本一致，wrapper只调用原collect/process_absence。九run全部train/verify实际exit0，summary invocation与dispatcher实际exit0，formal/summary/dispatch/owner-result已发布；原owned processes absent。owner_actual_exit=null：原detached owner没有持久wait证据，owner-result记录的是dispatcher退出，不能补造owner成功。最终验收与该缺口裁决由Astra执行。

冻结三独立消融uniform_negatives/user_balance/watch_history，各seeds42/2024/2025，IDmean、batch512、max30/patience3不变。没有bestseed/HPO/额外模型replay；既定两passes保持。All seed、cohort、mean/sample std与paired delta如下，paired按同seed核心IDmean基线相减，共同query denominator已核。数值为原小aggregate，不是因果lift。

原budget start=1791406275.4360795, deadline=1791579075.4360795, cap=206158430208bytes / 172800s；联合静态+formal观察bytes=125359209385，未reset/refund。总steps=41820，checkpoint/evaluation epochs=246，skipped=0。资源仅瞬时/既有segment证据，不称全历史峰值/minimum。

### 九run选择与全部seed结果

|slot|ablation|seed|steps|epochs|best epoch|stop|drawn|skipped|
|---|---|---|---|---|---|---|---|---|
|1|uniform_negatives|42|5100|30|30|max_epochs|2603550|0|
|2|uniform_negatives|2024|5100|30|30|max_epochs|2603550|0|
|3|uniform_negatives|2025|5100|30|30|max_epochs|2603550|0|
|4|user_balance|42|5100|30|30|max_epochs|2603550|0|
|5|user_balance|2024|5100|30|30|max_epochs|2603550|0|
|6|user_balance|2025|1020|6|3|patience|520710|0|
|7|watch_history|42|5100|30|30|max_epochs|2603550|0|
|8|watch_history|2024|5100|30|30|max_epochs|2603550|0|
|9|watch_history|2025|5100|30|30|max_epochs|2603550|0|

各seed所有cohort；指标顺序R10/R20/R50/N10/N20/N50，null表示该cohort无query，不删除seed或当作零：

|ablation/core|seed|cohort|count|R10|R20|R50|N10|N20|N50|
|---|---|---|---|---|---|---|---|---|---|
|uniform_negatives|42|all|2520|0.00952380952380952|0.019047619047619|0.0400793650793651|0.00386547153063437|0.00621386458598624|0.0104201081181553|
|uniform_negatives|42|history_1|227|0.013215859030837|0.013215859030837|0.0176211453744493|0.00435644404653085|0.00435644404653085|0.00526325828644466|
|uniform_negatives|42|history_2_5|664|0.00753012048192771|0.0180722891566265|0.0316265060240964|0.00278129172170687|0.00543451268252433|0.00819398307154283|
|uniform_negatives|42|history_6_20|916|0.00655021834061135|0.0163755458515284|0.037117903930131|0.00254949603369712|0.00494786410120306|0.00903430720221246|
|uniform_negatives|42|history_gt20|713|0.0140252454417952|0.0252454417952314|0.05890603085554|0.0064094802083537|0.00915745584884243|0.0159154034642319|
|uniform_negatives|42|in_catalog|1897|0.0126515550869794|0.0253031101739589|0.0532419609910385|0.00513494373073201|0.00825458026182674|0.0138422100462579|
|uniform_negatives|42|oov|623|0|0|0|0|0|0|
|uniform_negatives|42|pop_high|963|0.0228452751817238|0.0456905503634476|0.0924195223260644|0.00928344575444926|0.0149520875380733|0.0243350229845905|
|uniform_negatives|42|pop_low|155|0|0.00645161290322581|0.00645161290322581|0|0.00146884031417389|0.00146884031417389|
|uniform_negatives|42|pop_mid|548|0.00364963503649635|0.00547445255474453|0.0200729927007299|0.00146173356872989|0.00188395658544478|0.00473791072060892|
|uniform_negatives|42|supported|1666|0.014405762304922|0.0288115246098439|0.06062424969988|0.00584693172700997|0.00939912290317246|0.0157615080778819|
|uniform_negatives|42|unsupported|231|0|0|0|0|0|0|
|uniform_negatives|2024|all|2520|0.0126984126984127|0.0226190476190476|0.0412698412698413|0.00466436206052143|0.00718066958438117|0.0108566589785951|
|uniform_negatives|2024|history_1|227|0.00440528634361234|0.013215859030837|0.026431718061674|0.00277942622718704|0.00496343720242028|0.00750518635441277|
|uniform_negatives|2024|history_2_5|664|0.0150602409638554|0.0240963855421687|0.0406626506024096|0.00520375881257486|0.00746088066695834|0.0108021645904433|
|uniform_negatives|2024|history_6_20|916|0.00873362445414847|0.0218340611353712|0.0360262008733624|0.00285634167762893|0.0061915833726387|0.00901441670321001|
|uniform_negatives|2024|history_gt20|713|0.0182328190743338|0.0252454417952314|0.0532959326788219|0.0070849338158271|0.00889631413112728|0.0143411785910423|
|uniform_negatives|2024|in_catalog|1897|0.0168687401159726|0.0300474433315762|0.0548234053769109|0.0061962005232019|0.00953889686486059|0.0144221300084658|
|uniform_negatives|2024|oov|623|0|0|0|0|0|0|
|uniform_negatives|2024|pop_high|963|0.0301142263759086|0.0508826583592939|0.0944963655244029|0.0111118098843331|0.016383066883891|0.0249461697405273|
|uniform_negatives|2024|pop_low|155|0|0.00645161290322581|0.00645161290322581|0|0.00157839059431114|0.00157839059431114|
|uniform_negatives|2024|pop_mid|548|0.00547445255474453|0.0127737226277372|0.0218978102189781|0.00192248079179053|0.00378420328710821|0.00564045369309038|
|uniform_negatives|2024|supported|1666|0.0192076830732293|0.0342136854741897|0.0624249699879952|0.00705533757053661|0.0108615170183917|0.0164218371104799|
|uniform_negatives|2024|unsupported|231|0|0|0|0|0|0|
|uniform_negatives|2025|all|2520|0.00952380952380952|0.0178571428571429|0.0353174603174603|0.00359310984980097|0.00568842071950225|0.00918958131259738|
|uniform_negatives|2025|history_1|227|0.013215859030837|0.013215859030837|0.026431718061674|0.00638087717568014|0.00638087717568014|0.00917430232315232|
|uniform_negatives|2025|history_2_5|664|0.00602409638554217|0.0150602409638554|0.0271084337349398|0.00194890093907375|0.00415974907954701|0.00665042966124182|
|uniform_negatives|2025|history_6_20|916|0.00873362445414847|0.0152838427947598|0.0393013100436681|0.00280521630638615|0.00451934966167468|0.00923628410458001|
|uniform_negatives|2025|history_gt20|713|0.0126227208976157|0.0252454417952314|0.0406732117812062|0.00524898926006222|0.00839349707623147|0.0114990978198176|
|uniform_negatives|2025|in_catalog|1897|0.0126515550869794|0.0237216657880865|0.0469161834475488|0.00477313485582416|0.00755657364952328|0.0122075618912733|
|uniform_negatives|2025|oov|623|0|0|0|0|0|0|
|uniform_negatives|2025|pop_high|963|0.0207684319833853|0.0404984423676012|0.0799584631360332|0.00801449201296039|0.0130015253908942|0.0209351724193465|
|uniform_negatives|2025|pop_low|155|0|0.00645161290322581|0.00645161290322581|0|0.00146884031417389|0.00146884031417389|
|uniform_negatives|2025|pop_mid|548|0.0072992700729927|0.00912408759124088|0.0200729927007299|0.00243919892886421|0.00289540330842625|0.00505383872120032|
|uniform_negatives|2025|supported|1666|0.014405762304922|0.0270108043217287|0.053421368547419|0.00543495607532919|0.00860433386143197|0.0139002070274582|
|uniform_negatives|2025|unsupported|231|0|0|0|0|0|0|
|user_balance|42|all|2520|0.00555555555555556|0.0115079365079365|0.0246031746031746|0.00263979462070882|0.00411406166719654|0.00671644738955335|
|user_balance|42|history_1|227|0|0.00881057268722467|0.0176211453744493|0|0.00220264317180617|0.00408847723607956|
|user_balance|42|history_2_5|664|0.00301204819277108|0.00602409638554217|0.0180722891566265|0.00122810975419537|0.0019681202032563|0.00437176805886211|
|user_balance|42|history_6_20|916|0.00545851528384279|0.0109170305676856|0.0251091703056769|0.00176634724124852|0.0030758117558259|0.00590456992013801|
|user_balance|42|history_gt20|713|0.00981767180925666|0.0182328190743338|0.032258064516129|0.00591703154897174|0.00805492288644681|0.010779695723918|
|user_balance|42|in_catalog|1897|0.00738007380073801|0.0152872957301002|0.0326831839746969|0.00350673824153201|0.00546517417044559|0.00892221793446201|
|user_balance|42|oov|623|0|0|0|0|0|0|
|user_balance|42|pop_high|963|0.0134994807892004|0.0259605399792316|0.0539979231568017|0.00638866297423284|0.00950405145880829|0.0150879834270777|
|user_balance|42|pop_low|155|0|0|0|0|0|0|
|user_balance|42|pop_mid|548|0.00182481751824818|0.0072992700729927|0.0182481751824818|0.000912408759124088|0.00221721504836296|0.0043717506959829|
|user_balance|42|supported|1666|0.00840336134453781|0.017406962785114|0.0372148859543818|0.003992966653173|0.00622295042096956|0.010159332185879|
|user_balance|42|unsupported|231|0|0|0|0|0|0|
|user_balance|2024|all|2520|0.00674603174603175|0.0123015873015873|0.021031746031746|0.0026616647657462|0.00404151122674418|0.00579480158026381|
|user_balance|2024|history_1|227|0.00440528634361234|0.00881057268722467|0.013215859030837|0.00156919465686353|0.00258848193950564|0.00339063128078291|
|user_balance|2024|history_2_5|664|0.00602409638554217|0.0135542168674699|0.0225903614457831|0.00191445236411187|0.0037436620915044|0.00549918730338377|
|user_balance|2024|history_6_20|916|0.00545851528384279|0.00982532751091703|0.0196506550218341|0.00204214419545038|0.00313569015286014|0.00517639816809616|
|user_balance|2024|history_gt20|713|0.00981767180925666|0.0154277699859748|0.0238429172510519|0.00450124483838648|0.00594521610427592|0.00762999381501288|
|user_balance|2024|in_catalog|1897|0.00896151818661044|0.0163415919873484|0.0279388508170796|0.00353579083272557|0.00536879720157898|0.00769789139813643|
|user_balance|2024|oov|623|0|0|0|0|0|0|
|user_balance|2024|pop_high|963|0.0145379023883697|0.0207684319833853|0.039460020768432|0.0060371761996458|0.00760732124118127|0.0113551632416402|
|user_balance|2024|pop_low|155|0|0.00645161290322581|0.00645161290322581|0|0.00149276266554683|0.00149276266554683|
|user_balance|2024|pop_mid|548|0.00547445255474453|0.0182481751824818|0.0255474452554745|0.0016306469514991|0.0047944885455803|0.00627098461205384|
|user_balance|2024|supported|1666|0.0102040816326531|0.0186074429771909|0.031812725090036|0.00402604754482618|0.00611321025894077|0.00876524608779401|
|user_balance|2024|unsupported|231|0|0|0|0|0|0|
|user_balance|2025|all|2520|0|0.000793650793650794|0.00277777777777778|0|0.000186980428463449|0.000568488239761452|
|user_balance|2025|history_1|227|0|0.00440528634361234|0.00440528634361234|0|0.00101928728264211|0.00101928728264211|
|user_balance|2025|history_2_5|664|0|0.00150602409638554|0.00451807228915663|0|0.000361163353265258|0.000966989615108199|
|user_balance|2025|history_6_20|916|0|0|0.00218340611353712|0|0|0.000399240876626621|
|user_balance|2025|history_gt20|713|0|0|0.00140252454417952|0|0|0.000271285278565597|
|user_balance|2025|in_catalog|1897|0|0.00105429625724829|0.003690036900369|0|0.000248387285043696|0.000755187329572408|
|user_balance|2025|oov|623|0|0|0|0|0|0|
|user_balance|2025|pop_high|963|0|0.00207684319833853|0.00519210799584631|0|0.000489294579156688|0.00110236699111242|
|user_balance|2025|pop_low|155|0|0|0|0|0|0|
|user_balance|2025|pop_mid|548|0|0|0.00364963503649635|0|0|0.0006770272842292|
|user_balance|2025|supported|1666|0|0.00120048019207683|0.00420168067226891|0|0.000282827538852275|0.000859898177790431|
|user_balance|2025|unsupported|231|0|0|0|0|0|0|
|watch_history|42|all|2520|0.0194444444444444|0.0400793650793651|0.0797619047619048|0.00847858398933836|0.0136807236523893|0.021583302685956|
|watch_history|42|history_1|227|0.0176211453744493|0.0308370044052863|0.0572687224669604|0.00680215854628434|0.0101651803715596|0.0154676306322431|
|watch_history|42|history_2_5|664|0.0225903614457831|0.0481927710843374|0.0873493975903614|0.0101833766068625|0.0166300716566497|0.0244249586154164|
|watch_history|42|history_6_20|916|0.0131004366812227|0.037117903930131|0.0796943231441048|0.00449132028227372|0.0106107454669661|0.0190977328574119|
|watch_history|42|history_gt20|713|0.0252454417952314|0.0392706872370266|0.0799438990182328|0.0125471672056195|0.0159973593715577|0.0240772437546481|
|watch_history|42|in_catalog|1897|0.025830258302583|0.0532419609910385|0.105956773853453|0.0112630636020731|0.0181736550363843|0.0286715460034839|
|watch_history|42|oov|623|0|0|0|0|0|0|
|watch_history|42|pop_high|963|0.0488058151609553|0.0944963655244029|0.186915887850467|0.0214726363969202|0.0330636922046832|0.0514466764882876|
|watch_history|42|pop_low|155|0|0|0|0|0|0|
|watch_history|42|pop_mid|548|0.00364963503649635|0.0182481751824818|0.0383211678832117|0.00125526058923088|0.00480855476443644|0.00884447684377386|
|watch_history|42|supported|1666|0.0294117647058824|0.06062424969988|0.120648259303721|0.0128247488914362|0.0206935315750427|0.0326470124661519|
|watch_history|42|unsupported|231|0|0|0|0|0|0|
|watch_history|2024|all|2520|0.021031746031746|0.0369047619047619|0.0757936507936508|0.0093604886787091|0.0133554516845239|0.0210825399536602|
|watch_history|2024|history_1|227|0.0308370044052863|0.0440528634361234|0.066079295154185|0.0108796266744989|0.0140766693930046|0.018581073646215|
|watch_history|2024|history_2_5|664|0.0256024096385542|0.0481927710843374|0.0843373493975904|0.0129248693960084|0.0185708065422151|0.0256271049101308|
|watch_history|2024|history_6_20|916|0.0141921397379913|0.0283842794759825|0.0665938864628821|0.00582891624364209|0.0094449178559075|0.0170780218395286|
|watch_history|2024|history_gt20|713|0.0224403927068724|0.0350631136044881|0.0827489481065919|0.010094467962286|0.0132928106490127|0.0227913482471218|
|watch_history|2024|in_catalog|1897|0.0279388508170796|0.0490247759620453|0.100685292567211|0.0124345975067722|0.0177415594333159|0.0280063261377036|
|watch_history|2024|oov|623|0|0|0|0|0|0|
|watch_history|2024|pop_high|963|0.0519210799584631|0.0903426791277259|0.179646936656282|0.0227862507340821|0.0324484891094666|0.0501952733240432|
|watch_history|2024|pop_low|155|0|0|0.00645161290322581|0|0|0.00114905544228394|
|watch_history|2024|pop_mid|548|0.00547445255474453|0.0109489051094891|0.031021897810219|0.00300232119238305|0.00439387451201425|0.00841578262521185|
|watch_history|2024|supported|1666|0.031812725090036|0.0558223289315726|0.114645858343337|0.0141587223711566|0.0202015235564227|0.0318895562324272|
|watch_history|2024|unsupported|231|0|0|0|0|0|0|
|watch_history|2025|all|2520|0.0214285714285714|0.0440476190476191|0.0813492063492064|0.00901239168191637|0.0146364656738312|0.0219545970357127|
|watch_history|2025|history_1|227|0.039647577092511|0.052863436123348|0.0881057268722467|0.0165903632402392|0.0198970089095257|0.0271524828858481|
|watch_history|2025|history_2_5|664|0.019578313253012|0.052710843373494|0.088855421686747|0.00977591334749234|0.0180817243533674|0.0250599550996734|
|watch_history|2025|history_6_20|916|0.0174672489082969|0.0360262008733624|0.0786026200873362|0.00654324699794316|0.0111662611036523|0.0195606429718398|
|watch_history|2025|history_gt20|713|0.0224403927068724|0.0434782608695652|0.0757363253856942|0.00906086096780377|0.0142113777475607|0.0204833264607575|
|watch_history|2025|in_catalog|1897|0.0284659989457037|0.0585134422772799|0.108065366367949|0.011972180832066|0.0194432754338717|0.0291647783500243|
|watch_history|2025|oov|623|0|0|0|0|0|0|
|watch_history|2025|pop_high|963|0.0539979231568017|0.107995846313603|0.187954309449637|0.0227644467415486|0.0362060606984491|0.0519243136589309|
|watch_history|2025|pop_low|155|0|0|0|0|0|0|
|watch_history|2025|pop_mid|548|0.00364963503649635|0.0127737226277372|0.0437956204379562|0.00143989931809834|0.00368149095884682|0.00971253736577651|
|watch_history|2025|supported|1666|0.0324129651860744|0.0666266506602641|0.123049219687875|0.013632189098697|0.0221391917755429|0.0332086341716663|
|watch_history|2025|unsupported|231|0|0|0|0|0|0|
|core|2024|all|2520|0.0107142857142857|0.0202380952380952|0.0369047619047619|0.00394464708806978|0.0064087223809498|0.0097096375092829|
|core|2024|history_1|227|0.00440528634361234|0.00440528634361234|0.026431718061674|0.00220264317180617|0.00220264317180617|0.00662649626641008|
|core|2024|history_2_5|664|0.0120481927710843|0.0210843373493976|0.0376506024096386|0.00426890874915861|0.00664803587764756|0.00994946297844921|
|core|2024|history_6_20|916|0.00764192139737991|0.0196506550218341|0.0327510917030568|0.00243210243522209|0.00552367580785381|0.00811284514095571|
|core|2024|history_gt20|713|0.0154277699859748|0.0252454417952314|0.0448807854137447|0.0061404620222035|0.00866198813077338|0.0125193019693017|
|core|2024|in_catalog|1897|0.0142329994728519|0.0268845545598313|0.0490247759620453|0.00524012159300783|0.00851343194517316|0.0128984114514459|
|core|2024|oov|623|0|0|0|0|0|0|
|core|2024|pop_high|963|0.0249221183800623|0.043613707165109|0.083073727933541|0.0092046908974662|0.0140773348299657|0.0218991243463692|
|core|2024|pop_low|155|0|0.00645161290322581|0.00645161290322581|0|0.00157839059431114|0.00157839059431114|
|core|2024|pop_mid|548|0.00547445255474453|0.0145985401459854|0.0218978102189781|0.00196422140086843|0.00428623433689478|0.00572040006518448|
|core|2024|supported|1666|0.0162064825930372|0.0306122448979592|0.0558223289315726|0.00596669307439127|0.00969386578631062|0.0146868466526968|
|core|2024|unsupported|231|0|0|0|0|0|0|
|core|2025|all|2520|0.00833333333333333|0.0166666666666667|0.0329365079365079|0.00320142559606597|0.00530421973696038|0.00858248415874243|
|core|2025|history_1|227|0.013215859030837|0.013215859030837|0.026431718061674|0.00614510856818583|0.00614510856818583|0.00900059292353183|
|core|2025|history_2_5|664|0.00451807228915663|0.0120481927710843|0.0256024096385542|0.00155971538758073|0.00345177627544129|0.00631012267627697|
|core|2025|history_6_20|916|0.00764192139737991|0.0141921397379913|0.0338427947598253|0.00244401147935811|0.00420021601971596|0.00807686523491049|
|core|2025|history_gt20|713|0.0112201963534362|0.0252454417952314|0.0406732117812062|0.00476618138101323|0.00817996742105068|0.0112151409174801|
|core|2025|in_catalog|1897|0.011070110701107|0.022140221402214|0.0437532946758039|0.00425281628997694|0.00704619596053778|0.0114010859673331|
|core|2025|oov|623|0|0|0|0|0|0|
|core|2025|pop_high|963|0.0176531671858775|0.0373831775700935|0.0737279335410177|0.00698952387234545|0.0119899527005356|0.0193365569781784|
|core|2025|pop_low|155|0|0.00645161290322581|0.00645161290322581|0|0.00146884031417389|0.00146884031417389|
|core|2025|pop_mid|548|0.0072992700729927|0.00912408759124088|0.0200729927007299|0.00243919892886421|0.00290627561647336|0.00507132383457702|
|core|2025|supported|1666|0.0126050420168067|0.0252100840336134|0.0498199279711885|0.00484249249825105|0.00802318951809133|0.0129819088115432|
|core|2025|unsupported|231|0|0|0|0|0|0|
|core|42|all|2520|0.0103174603174603|0.0166666666666667|0.0373015873015873|0.00405762232560325|0.00562067857826331|0.00973890677164313|
|core|42|history_1|227|0.013215859030837|0.013215859030837|0.0176211453744493|0.00422144131469597|0.00422144131469597|0.00507354000904518|
|core|42|history_2_5|664|0.00753012048192771|0.016566265060241|0.0316265060240964|0.00274684314674498|0.00499384296710867|0.00798112871748819|
|core|42|history_6_20|916|0.00764192139737991|0.0120087336244541|0.0327510917030568|0.00280416405349291|0.00390203203143391|0.00806521551709862|
|core|42|history_gt20|713|0.0154277699859748|0.0238429172510519|0.0546984572230014|0.00683649783961574|0.00885788186231966|0.0150114230019817|
|core|42|in_catalog|1897|0.0137058513442277|0.022140221402214|0.0495519240906695|0.00539019939932535|0.00746658408920587|0.012937293128382|
|core|42|oov|623|0|0|0|0|0|0|
|core|42|pop_high|963|0.0249221183800623|0.0404984423676012|0.0841121495327103|0.00978627026464819|0.0136224605186307|0.0223587843413004|
|core|42|pop_low|155|0|0|0.00645161290322581|0|0|0.00144673434979081|
|core|42|pop_mid|548|0.00364963503649635|0.00547445255474453|0.0218978102189781|0.00146173356872989|0.00190817616383614|0.00508447430593223|
|core|42|supported|1666|0.0156062424969988|0.0252100840336134|0.056422569027611|0.00613757998830743|0.00850186675703693|0.014731119486519|
|core|42|unsupported|231|0|0|0|0|0|0|

### 三seed聚合与paired摘要

Sample std使用n−1。每项values和paired values按42/2024/2025顺序保留，全部聚合独立stdlib重算，浮点容差1e-12通过，无数值模型replay。

|ablation|cohort|metric|values|mean|sample std|paired values|paired mean|paired sample std|
|---|---|---|---|---|---|---|---|---|
|uniform_negatives|all|recall@10|0.00952380952380952,0.0126984126984127,0.00952380952380952|0.0105820105820106|0.00183285799742738|-0.000793650793650793,0.00198412698412698,0.00119047619047619|0.000793650793650794|0.00143077431566031|
|uniform_negatives|all|recall@20|0.019047619047619,0.0226190476190476,0.0178571428571429|0.0198412698412698|0.00247817380888825|0.00238095238095238,0.00238095238095238,0.00119047619047619|0.00198412698412698|0.000687321749035269|
|uniform_negatives|all|recall@50|0.0400793650793651,0.0412698412698413,0.0353174603174603|0.0388888888888889|0.00314970394174356|0.00277777777777778,0.00436507936507936,0.00238095238095238|0.00317460317460318|0.00104990131391452|
|uniform_negatives|all|ndcg@10|0.00386547153063437,0.00466436206052143,0.00359310984980097|0.00404098114698559|0.000556774679576202|-0.000192150794968877,0.000719714972451646,0.000391684253734994|0.000306416143739254|0.000461874206234813|
|uniform_negatives|all|ndcg@20|0.00621386458598624,0.00718066958438117,0.00568842071950225|0.00636098496328988|0.000756924681026178|0.000593186007722934,0.000771947203431369,0.000384200982541863|0.000583111397898722|0.000194069333682207|
|uniform_negatives|all|ndcg@50|0.0104201081181553,0.0108566589785951,0.00918958131259738|0.0101554494697826|0.000864476799131078|0.000681201346512155,0.00114702146931216,0.000607097153854954|0.000811773323226423|0.000292688140232546|
|uniform_negatives|history_1|recall@10|0.013215859030837,0.00440528634361234,0.013215859030837|0.0102790014684288|0.00508678651268393|0,0,0|0|0|
|uniform_negatives|history_1|recall@20|0.013215859030837,0.013215859030837,0.013215859030837|0.013215859030837|0|0,0.00881057268722467,0|0.00293685756240822|0.00508678651268393|
|uniform_negatives|history_1|recall@50|0.0176211453744493,0.026431718061674,0.026431718061674|0.0234948604992658|0.00508678651268393|0,0,0|0|0|
|uniform_negatives|history_1|ndcg@10|0.00435644404653085,0.00277942622718704,0.00638087717568014|0.00450558248313268|0.00180535147230518|0.000135002731834887,0.00057678305538087,0.00023576860749431|0.000315851464903356|0.000231521946273342|
|uniform_negatives|history_1|ndcg@20|0.00435644404653085,0.00496343720242028,0.00638087717568014|0.00523358614154376|0.00103890216602982|0.000135002731834887,0.00276079403061412,0.00023576860749431|0.00104385512331444|0.00148776605989447|
|uniform_negatives|history_1|ndcg@50|0.00526325828644466,0.00750518635441277,0.00917430232315232|0.00731424898800325|0.00196250074463303|0.000189718277399485,0.000878690088002689,0.000173709399620494|0.000414039255007556|0.000402479028639057|
|uniform_negatives|history_2_5|recall@10|0.00753012048192771,0.0150602409638554,0.00602409638554217|0.00953815261044177|0.00484119014106072|0,0.00301204819277108,0.00150602409638554|0.00150602409638554|0.00150602409638554|
|uniform_negatives|history_2_5|recall@20|0.0180722891566265,0.0240963855421687,0.0150602409638554|0.0190763052208835|0.00460097961340948|0.00150602409638554,0.00301204819277109,0.00301204819277108|0.00251004016064257|0.000869503417454257|
|uniform_negatives|history_2_5|recall@50|0.0316265060240964,0.0406626506024096,0.0271084337349398|0.0331325301204819|0.00690146942011422|0,0.00301204819277108,0.00150602409638554|0.00150602409638554|0.00150602409638554|
|uniform_negatives|history_2_5|ndcg@10|0.00278129172170687,0.00520375881257486,0.00194890093907375|0.00331131715778516|0.00169092287415498|3.44485749618809e-05,0.000934850063416248,0.000389185551493025|0.000452828063290385|0.000453561999161766|
|uniform_negatives|history_2_5|ndcg@20|0.00543451268252433,0.00746088066695834,0.00415974907954701|0.00568504747634322|0.00166476520913209|0.000440669715415653,0.000812844789310783,0.000707972804105718|0.000653829102944051|0.000191904212274833|
|uniform_negatives|history_2_5|ndcg@50|0.00819398307154283,0.0108021645904433,0.00665042966124182|0.00854885910774266|0.00209849433675423|0.000212854354054645,0.000852701611994093,0.000340306984964856|0.000468620983671198|0.000338673131605042|
|uniform_negatives|history_6_20|recall@10|0.00655021834061135,0.00873362445414847,0.00873362445414847|0.00800582241630277|0.00126059010740093|-0.00109170305676856,0.00109170305676856,0.00109170305676856|0.000363901018922853|0.00126059010740093|
|uniform_negatives|history_6_20|recall@20|0.0163755458515284,0.0218340611353712,0.0152838427947598|0.0178311499272198|0.00350933433806148|0.00436681222707424,0.00218340611353712,0.00109170305676856|0.00254730713245997|0.00166760396468553|
|uniform_negatives|history_6_20|recall@50|0.037117903930131,0.0360262008733624,0.0393013100436681|0.0374818049490539|0.00166760396468553|0.00436681222707424,0.00327510917030568,0.00545851528384279|0.00436681222707424|0.00109170305676856|
|uniform_negatives|history_6_20|ndcg@10|0.00254949603369712,0.00285634167762893,0.00280521630638615|0.00273701800590407|0.000164398352915913|-0.000254668019795789,0.000424239242406844,0.000361204827028034|0.000176925349879696|0.000375097269973293|
|uniform_negatives|history_6_20|ndcg@20|0.00494786410120306,0.0061915833726387,0.00451934966167468|0.00521959904517215|0.000868603077274959|0.00104583206976915,0.000667907564784885,0.000319133641958725|0.000677624425504254|0.000363446645716006|
|uniform_negatives|history_6_20|ndcg@50|0.00903430720221246,0.00901441670321001,0.00923628410458001|0.00909500267000083|0.000122756836083393|0.000969091685113839,0.000901571562254304,0.00115941886966951|0.00101002737234589|0.000133709036165057|
|uniform_negatives|history_gt20|recall@10|0.0140252454417952,0.0182328190743338,0.0126227208976157|0.0149602618045816|0.00291958765703525|-0.00140252454417952,0.00280504908835905,0.00140252454417952|0.000935016362786349|0.00214239162924537|
|uniform_negatives|history_gt20|recall@20|0.0252454417952314,0.0252454417952314,0.0252454417952314|0.0252454417952314|0|0.00140252454417952,0,0|0.000467508181393174|0.000809747923127105|
|uniform_negatives|history_gt20|recall@50|0.05890603085554,0.0532959326788219,0.0406732117812062|0.050958391771856|0.00933846860936801|0.00420757363253857,0.00841514726507714,0|0.00420757363253857|0.00420757363253857|
|uniform_negatives|history_gt20|ndcg@10|0.0064094802083537,0.0070849338158271,0.00524898926006222|0.00624780109474767|0.000928589362847215|-0.000427017631262044,0.000944471793623602,0.000482807879048989|0.000333420680470182|0.000697841824461467|
|uniform_negatives|history_gt20|ndcg@20|0.00915745584884243,0.00889631413112728,0.00839349707623147|0.0088157556854004|0.000388298208785154|0.000299573986522772,0.000234326000353904,0.00021352965518079|0.000249143214019156|4.48950899834657e-05|
|uniform_negatives|history_gt20|ndcg@50|0.0159154034642319,0.0143411785910423,0.0114990978198176|0.0139185599583639|0.00223827919789655|0.000903980462250204,0.00182187662174058,0.000283956902337566|0.00100327132877612|0.000773752720811862|
|uniform_negatives|in_catalog|recall@10|0.0126515550869794,0.0168687401159726,0.0126515550869794|0.0140572834299772|0.00243479291171165|-0.00105429625724829,0.00263574064312072,0.00158144438587243|0.00105429625724829|0.00190065960751924|
|uniform_negatives|in_catalog|recall@20|0.0253031101739589,0.0300474433315762,0.0237216657880865|0.0263574064312072|0.00329203900811724|0.00316288877174486,0.00316288877174486,0.00158144438587243|0.00263574064312072|0.000913047341891869|
|uniform_negatives|in_catalog|recall@50|0.0532419609910385,0.0548234053769109,0.0469161834475488|0.0516605166051661|0.00418410855729772|0.003690036900369,0.00579862941486558,0.00316288877174486|0.00421718502899315|0.00139470285243257|
|uniform_negatives|in_catalog|ndcg@10|0.00513494373073201,0.0061962005232019,0.00477313485582416|0.00536809303658602|0.000739626880617832|-0.000255255668593342,0.000956078930194068,0.000520318565847225|0.000407047275815983|0.000613559831160638|
|uniform_negatives|in_catalog|ndcg@20|0.00825458026182674,0.00953889686486059,0.00755657364952328|0.00845001692540354|0.00100550880136319|0.000787996172620872,0.00102546491968743,0.000510377688985502|0.000774612927097934|0.000257804280906254|
|uniform_negatives|in_catalog|ndcg@50|0.0138422100462579,0.0144221300084658,0.0122075618912733|0.013490633981999|0.00114838246379036|0.000904916917875927,0.00152371855701984,0.00080647592394016|0.00107837046627864|0.00038881081359305|
|uniform_negatives|oov|recall@10|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|oov|recall@20|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|oov|recall@50|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|oov|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|oov|ndcg@20|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|oov|ndcg@50|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|pop_high|recall@10|0.0228452751817238,0.0301142263759086,0.0207684319833853|0.0245759778470059|0.00490738901999232|-0.00207684319833852,0.00519210799584631,0.00311526479750779|0.00207684319833853|0.00374408232135409|
|uniform_negatives|pop_high|recall@20|0.0456905503634476,0.0508826583592939,0.0404984423676012|0.0456905503634476|0.00519210799584632|0.00519210799584632,0.00726895119418484,0.00311526479750779|0.00519210799584632|0.00207684319833853|
|uniform_negatives|pop_high|recall@50|0.0924195223260644,0.0944963655244029,0.0799584631360332|0.0889581169955002|0.00786280144797546|0.00830737279335411,0.0114226375908619,0.00623052959501558|0.00865351332641053|0.00261330371591234|
|uniform_negatives|pop_high|ndcg@10|0.00928344575444926,0.0111118098843331,0.00801449201296039|0.00946991588391425|0.00155705582140405|-0.000502824510198932,0.00190711898686691,0.00102496814061494|0.000809754205760972|0.00121930090341924|
|uniform_negatives|pop_high|ndcg@20|0.0149520875380733,0.016383066883891,0.0130015253908942|0.0147788932709528|0.00169741064796863|0.00132962701944269,0.00230573205392534,0.00101157269035858|0.00154897725457553|0.000674387135713978|
|uniform_negatives|pop_high|ndcg@50|0.0243350229845905,0.0249461697405273,0.0209351724193465|0.0234054550481548|0.00216104078836263|0.00197623864329017,0.00304704539415807,0.00159861544116812|0.00220729982620545|0.000751351638626863|
|uniform_negatives|pop_low|recall@10|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|pop_low|recall@20|0.00645161290322581,0.00645161290322581,0.00645161290322581|0.00645161290322581|0|0.00645161290322581,0,0|0.0021505376344086|0.00372484044638468|
|uniform_negatives|pop_low|recall@50|0.00645161290322581,0.00645161290322581,0.00645161290322581|0.00645161290322581|0|0,0,0|0|0|
|uniform_negatives|pop_low|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|pop_low|ndcg@20|0.00146884031417389,0.00157839059431114,0.00146884031417389|0.00150535707421964|6.32488837270376e-05|0.00146884031417389,0,0|0.000489613438057963|0.00084803535078487|
|uniform_negatives|pop_low|ndcg@50|0.00146884031417389,0.00157839059431114,0.00146884031417389|0.00150535707421964|6.32488837270376e-05|2.21059643830809e-05,0,0|7.3686547943603e-06|1.2762884487268e-05|
|uniform_negatives|pop_mid|recall@10|0.00364963503649635,0.00547445255474453,0.0072992700729927|0.00547445255474453|0.00182481751824818|0,0,0|0|0|
|uniform_negatives|pop_mid|recall@20|0.00547445255474453,0.0127737226277372,0.00912408759124088|0.00912408759124088|0.00364963503649635|0,-0.00182481751824817,0|-0.000608272506082725|0.00105355888538253|
|uniform_negatives|pop_mid|recall@50|0.0200729927007299,0.0218978102189781,0.0200729927007299|0.0206812652068127|0.00105355888538253|-0.00182481751824818,0,0|-0.000608272506082726|0.00105355888538253|
|uniform_negatives|pop_mid|ndcg@10|0.00146173356872989,0.00192248079179053,0.00243919892886421|0.00194113776312821|0.000488999687627965|0,-4.17406090778994e-05,0|-1.39135363592998e-05|2.40989518872642e-05|
|uniform_negatives|pop_mid|ndcg@20|0.00188395658544478,0.00378420328710821,0.00289540330842625|0.00285452106032641|0.00095078278300196|-2.42195783913628e-05,-0.000502031049786571,-1.08723080471085e-05|-0.000179040978741681|0.000279797206743738|
|uniform_negatives|pop_mid|ndcg@50|0.00473791072060892,0.00564045369309038,0.00505383872120032|0.00514406771163321|0.000457986798204729|-0.000346563585323307,-7.99463720941012e-05,-1.74851133766911e-05|-0.000147998356931366|0.000174775469259557|
|uniform_negatives|supported|recall@10|0.014405762304922,0.0192076830732293,0.014405762304922|0.0160064025610244|0.00277239024820949|-0.00120048019207683,0.00300120048019208,0.00180072028811525|0.00120048019207683|0.00216419644385594|
|uniform_negatives|supported|recall@20|0.0288115246098439,0.0342136854741897,0.0270108043217287|0.0300120048019208|0.00374849819831837|0.00360144057623049,0.00360144057623049,0.00180072028811525|0.00300120048019208|0.00103964634307856|
|uniform_negatives|supported|recall@50|0.06062424969988,0.0624249699879952,0.053421368547419|0.0588235294117647|0.00476425806314152|0.00420168067226891,0.00660264105642257,0.00360144057623049|0.00480192076830732|0.00158808602104717|
|uniform_negatives|supported|ndcg@10|0.00584693172700997,0.00705533757053661,0.00543495607532919|0.00611240845762526|0.000842180187594254|-0.000290648261297461,0.00108864449614535,0.000592463577078143|0.000463486603975342|0.000698633253128289|
|uniform_negatives|supported|ndcg@20|0.00939912290317246,0.0108615170183917,0.00860433386143197|0.00962165792766537|0.00114492808894716|0.00089725614613553,0.00116765123208106,0.000581144343340634|0.000882017240519076|0.000293550252628549|
|uniform_negatives|supported|ndcg@50|0.0157615080778819,0.0164218371104799,0.0139002070274582|0.01536118407194|0.00130761196507222|0.00103038859136292,0.0017349904577831,0.000918298215915056|0.00122789242168703|0.000442721556654271|
|uniform_negatives|unsupported|recall@10|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|unsupported|recall@20|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|unsupported|recall@50|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|unsupported|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|unsupported|ndcg@20|0,0,0|0|0|0,0,0|0|0|
|uniform_negatives|unsupported|ndcg@50|0,0,0|0|0|0,0,0|0|0|
|user_balance|all|recall@10|0.00555555555555556,0.00674603174603175,0|0.0041005291005291|0.00360070306582439|-0.00476190476190476,-0.00396825396825397,-0.00833333333333333|-0.00568783068783069|0.00232518463376282|
|user_balance|all|recall@20|0.0115079365079365,0.0123015873015873,0.000793650793650794|0.0082010582010582|0.00642726487473881|-0.00515873015873016,-0.00793650793650794,-0.0158730158730159|-0.00965608465608466|0.0055602776605777|
|user_balance|all|recall@50|0.0246031746031746,0.021031746031746,0.00277777777777778|0.0161375661375661|0.0117069096807807|-0.0126984126984127,-0.0158730158730159,-0.0301587301587302|-0.0195767195767196|0.00930073853505129|
|user_balance|all|ndcg@10|0.00263979462070882,0.0026616647657462,0|0.00176715312881834|0.00153043856829679|-0.00141782770489443,-0.00128298232232359,-0.00320142559606597|-0.001967411874428|0.00107081194343044|
|user_balance|all|ndcg@20|0.00411406166719654,0.00404151122674418,0.000186980428463449|0.00278085110746806|0.0022466507775247|-0.00150661691106676,-0.00236721115420562,-0.00511723930849693|-0.00299702245792311|0.00188590700537535|
|user_balance|all|ndcg@50|0.00671644738955335,0.00579480158026381,0.000568488239761452|0.00435991240319287|0.00331564935415885|-0.00302245938208978,-0.00391483592901908,-0.00801399591898098|-0.00498376374336328|0.00266191928014148|
|user_balance|history_1|recall@10|0,0.00440528634361234,0|0.00146842878120411|0.00254339325634196|-0.013215859030837,0,-0.013215859030837|-0.00881057268722467|0.00763017976902589|
|user_balance|history_1|recall@20|0.00881057268722467,0.00881057268722467,0.00440528634361234|0.00734214390602056|0.00254339325634196|-0.00440528634361233,0.00440528634361234,-0.00881057268722467|-0.00293685756240822|0.00672918604251959|
|user_balance|history_1|recall@50|0.0176211453744493,0.013215859030837,0.00440528634361234|0.0117474302496329|0.00672918604251959|0,-0.013215859030837,-0.0220264317180617|-0.0117474302496329|0.0110863941780775|
|user_balance|history_1|ndcg@10|0,0.00156919465686353,0|0.000523064885621178|0.000905974957551084|-0.00422144131469597,-0.000633448514942634,-0.00614510856818583|-0.00366666613260814|0.00279739714066928|
|user_balance|history_1|ndcg@20|0.00220264317180617,0.00258848193950564,0.00101928728264211|0.00193680413131797|0.000817677053834843|-0.0020187981428898,0.000385838767699477,-0.00512582128554372|-0.00225292688691135|0.00276327909144408|
|user_balance|history_1|ndcg@50|0.00408847723607956,0.00339063128078291,0.00101928728264211|0.00283279859983486|0.00160883955617725|-0.00098506277296562,-0.00323586498562717,-0.00798130564088972|-0.00406741113316084|0.00357147802924129|
|user_balance|history_2_5|recall@10|0.00301204819277108,0.00602409638554217,0|0.00301204819277108|0.00301204819277108|-0.00451807228915663,-0.00602409638554217,-0.00451807228915663|-0.00502008032128514|0.000869503417454256|
|user_balance|history_2_5|recall@20|0.00602409638554217,0.0135542168674699,0.00150602409638554|0.0070281124497992|0.00608652392217979|-0.0105421686746988,-0.00753012048192771,-0.0105421686746988|-0.00953815261044177|0.00173900683490851|
|user_balance|history_2_5|recall@50|0.0180722891566265,0.0225903614457831,0.00451807228915663|0.0150602409638554|0.00940511746746747|-0.0135542168674699,-0.0150602409638554,-0.0210843373493976|-0.016566265060241|0.00398456522750691|
|user_balance|history_2_5|ndcg@10|0.00122810975419537,0.00191445236411187,0|0.00104752070610242|0.000969918175328066|-0.00151873339254961,-0.00235445638504674,-0.00155971538758073|-0.00181096838839236|0.000471120242464413|
|user_balance|history_2_5|ndcg@20|0.0019681202032563,0.0037436620915044,0.000361163353265258|0.00202431521600865|0.00169194941949841|-0.00302572276385238,-0.00290437378614316,-0.00309061292217603|-0.00300690315739052|9.45351106103017e-05|
|user_balance|history_2_5|ndcg@50|0.00437176805886211,0.00549918730338377,0.000966989615108199|0.00361264832578469|0.00235953407440182|-0.00360936065862608,-0.00445027567506545,-0.00534313306116877|-0.00446758979828676|0.000867015870732397|
|user_balance|history_6_20|recall@10|0.00545851528384279,0.00545851528384279,0|0.00363901018922853|0.00315147526850232|-0.00218340611353712,-0.00218340611353712,-0.00764192139737991|-0.00400291120815138|0.00315147526850232|
|user_balance|history_6_20|recall@20|0.0109170305676856,0.00982532751091703,0|0.00691411935953421|0.00601263160184072|-0.00109170305676856,-0.00982532751091703,-0.0141921397379913|-0.00836972343522562|0.00667041585874213|
|user_balance|history_6_20|recall@50|0.0251091703056769,0.0196506550218341,0.00218340611353712|0.0156477438136827|0.0119756060203088|-0.00764192139737991,-0.0131004366812227,-0.0316593886462882|-0.0174672489082969|0.0125901338369769|
|user_balance|history_6_20|ndcg@10|0.00176634724124852,0.00204214419545038,0|0.0012694971455663|0.00110803124592111|-0.00103781681224439,-0.00038995823977171,-0.00244401147935811|-0.00129059551045807|0.00105009831942861|
|user_balance|history_6_20|ndcg@20|0.0030758117558259,0.00313569015286014,0|0.00207050063622868|0.00179335607704532|-0.000826220275608011,-0.00238798565499367,-0.00420021601971596|-0.00247147398343921|0.00168854657799381|
|user_balance|history_6_20|ndcg@50|0.00590456992013801,0.00517639816809616,0.000399240876626621|0.00382673632162027|0.00299054381245403|-0.00216064559696061,-0.00293644697285954,-0.00767762435828387|-0.00425823897603467|0.00298657232257421|
|user_balance|history_gt20|recall@10|0.00981767180925666,0.00981767180925666,0|0.00654511453950444|0.00566823546188973|-0.00561009817671809,-0.00561009817671809,-0.0112201963534362|-0.00748013090229079|0.00323899169250842|
|user_balance|history_gt20|recall@20|0.0182328190743338,0.0154277699859748,0|0.0112201963534362|0.00981767180925666|-0.00561009817671809,-0.00981767180925666,-0.0252454417952314|-0.0135577372604021|0.0103381694191192|
|user_balance|history_gt20|recall@50|0.032258064516129,0.0238429172510519,0.00140252454417952|0.0191678354371202|0.0159501843003572|-0.0224403927068724,-0.0210378681626928,-0.0392706872370266|-0.0275829827021973|0.0101461123956104|
|user_balance|history_gt20|ndcg@10|0.00591703154897174,0.00450124483838648,0|0.00347275879578607|0.00308968497467778|-0.000919466290643997,-0.00163921718381702,-0.00476618138101323|-0.00244162161849142|0.00204504129645527|
|user_balance|history_gt20|ndcg@20|0.00805492288644681,0.00594521610427592,0|0.00466671299690758|0.0041768856009369|-0.000802958975872847,-0.00271677202649746,-0.00817996742105068|-0.00389989947447366|0.00382817271275374|
|user_balance|history_gt20|ndcg@50|0.010779695723918,0.00762999381501288,0.000271285278565597|0.00622699160583214|0.0053928641667691|-0.00423172727806371,-0.0048893081542888,-0.0109438556389145|-0.00668829702375566|0.00370005912080922|
|user_balance|in_catalog|recall@10|0.00738007380073801,0.00896151818661044,0|0.00544719732911615|0.00478322178485897|-0.00632577754348972,-0.00527148128624143,-0.011070110701107|-0.00755578984361272|0.00308880615555209|
|user_balance|in_catalog|recall@20|0.0152872957301002,0.0163415919873484,0.00105429625724829|0.0108943946582323|0.00853806404024343|-0.00685292567211386,-0.0105429625724829,-0.0210859251449657|-0.0128272711298542|0.00738634670777849|
|user_balance|in_catalog|recall@50|0.0326831839746969,0.0279388508170796,0.003690036900369|0.0214373572307152|0.0155516143360924|-0.0168687401159726,-0.0210859251449657,-0.0400632577754349|-0.0260059743454577|0.0123552246222084|
|user_balance|in_catalog|ndcg@10|0.00350673824153201,0.00353579083272557,0|0.0023475096914192|0.0020330549246747|-0.00188346115779334,-0.00170433076028225,-0.00425281628997694|-0.00261353606935084|0.0014224808104611|
|user_balance|in_catalog|ndcg@20|0.00546517417044559,0.00536879720157898,0.000248387285043696|0.00369411955235609|0.00298448073767118|-0.00200140991876028,-0.00314463474359418,-0.00679780867549408|-0.00398128444594951|0.00250526391857981|
|user_balance|in_catalog|ndcg@50|0.00892221793446201,0.00769789139813643,0.000755187329572408|0.00579176555405695|0.00440455264759109|-0.00401507519392,-0.00520052005330948,-0.0106458986377607|-0.0066204979616634|0.00353612893302928|
|user_balance|oov|recall@10|0,0,0|0|0|0,0,0|0|0|
|user_balance|oov|recall@20|0,0,0|0|0|0,0,0|0|0|
|user_balance|oov|recall@50|0,0,0|0|0|0,0,0|0|0|
|user_balance|oov|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|user_balance|oov|ndcg@20|0,0,0|0|0|0,0,0|0|0|
|user_balance|oov|ndcg@50|0,0,0|0|0|0,0,0|0|0|
|user_balance|pop_high|recall@10|0.0134994807892004,0.0145379023883697,0|0.00934579439252336|0.0081103319583662|-0.0114226375908619,-0.0103842159916926,-0.0176531671858775|-0.013153340256144|0.00393140072398773|
|user_balance|pop_high|recall@20|0.0259605399792316,0.0207684319833853,0.00207684319833853|0.0162686050536518|0.0125616111537723|-0.0145379023883697,-0.0228452751817238,-0.0353063343717549|-0.0242298373139495|0.0104532148636494|
|user_balance|pop_high|recall@50|0.0539979231568017,0.039460020768432,0.00519210799584631|0.03288335064036|0.0250587577677235|-0.0301142263759086,-0.043613707165109,-0.0685358255451713|-0.0474212530287297|0.0194917386297004|
|user_balance|pop_high|ndcg@10|0.00638866297423284,0.0060371761996458,0|0.00414194639129288|0.00359133341106552|-0.00339760729041535,-0.0031675146978204,-0.00698952387234545|-0.0045182152868604|0.00214330591465868|
|user_balance|pop_high|ndcg@20|0.00950405145880829,0.00760732124118127,0.000489294579156688|0.00586688909304875|0.00475271379808382|-0.00411840905982236,-0.00647001358878439,-0.0115006581213789|-0.00736302692332856|0.00377127364606024|
|user_balance|pop_high|ndcg@50|0.0150879834270777,0.0113551632416402,0.00110236699111242|0.00918183788661011|0.00724167619392071|-0.00727080091422268,-0.010543961104729,-0.018234189987066|-0.0120163173353392|0.00562804136827346|
|user_balance|pop_low|recall@10|0,0,0|0|0|0,0,0|0|0|
|user_balance|pop_low|recall@20|0,0.00645161290322581,0|0.0021505376344086|0.00372484044638468|0,0,-0.00645161290322581|-0.0021505376344086|0.00372484044638468|
|user_balance|pop_low|recall@50|0,0.00645161290322581,0|0.0021505376344086|0.00372484044638468|-0.00645161290322581,0,-0.00645161290322581|-0.0043010752688172|0.00372484044638468|
|user_balance|pop_low|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|user_balance|pop_low|ndcg@20|0,0.00149276266554683,0|0.000497587555182278|0.000861846926789688|0,-8.56279287643025e-05,-0.00146884031417389|-0.000518156080979398|0.000824429147337457|
|user_balance|pop_low|ndcg@50|0,0.00149276266554683,0|0.000497587555182278|0.000861846926789688|-0.00144673434979081,-8.56279287643025e-05,-0.00146884031417389|-0.001000400864243|0.000792293702504672|
|user_balance|pop_mid|recall@10|0.00182481751824818,0.00547445255474453,0|0.0024330900243309|0.00278745480228457|-0.00182481751824818,0,-0.0072992700729927|-0.00304136253041363|0.0037986605829674|
|user_balance|pop_mid|recall@20|0.0072992700729927,0.0182481751824818,0|0.00851581508515815|0.00918471342490359|0.00182481751824817,0.00364963503649635,-0.00912408759124088|-0.00121654501216545|0.00690864762262807|
|user_balance|pop_mid|recall@50|0.0182481751824818,0.0255474452554745,0.00364963503649635|0.0158150851581508|0.0111498192091383|-0.00364963503649635,0.00364963503649635,-0.0164233576642336|-0.00547445255474453|0.0101601539467701|
|user_balance|pop_mid|ndcg@10|0.000912408759124088,0.0016306469514991,0|0.000847685236874394|0.000817247955554267|-0.000549324809605805,-0.000333574449369329,-0.00243919892886421|-0.00110736606261311|0.00115843478116185|
|user_balance|pop_mid|ndcg@20|0.00221721504836296,0.0047944885455803,0|0.00233723453131442|0.00239949653274205|0.000309038884526814,0.000508254208685521,-0.00290627561647336|-0.000696327507753675|0.00191646149681391|
|user_balance|pop_mid|ndcg@50|0.0043717506959829,0.00627098461205384,0.0006770272842292|0.00377325419742198|0.00284459807186763|-0.000712723609949328,0.000550584546869364,-0.00439429655034782|-0.00151881187114259|0.00256910412012225|
|user_balance|supported|recall@10|0.00840336134453781,0.0102040816326531,0|0.00620248099239696|0.00544644161217135|-0.00720288115246098,-0.00600240096038415,-0.0126050420168067|-0.00860344137655062|0.00351708600064965|
|user_balance|supported|recall@20|0.017406962785114,0.0186074429771909,0.00120048019207683|0.0124049619847939|0.00972191325590744|-0.0078031212484994,-0.0120048019207683,-0.0240096038415366|-0.0146058423369348|0.00841050402440324|
|user_balance|supported|recall@50|0.0372148859543818,0.031812725090036,0.00420168067226891|0.0244097639055622|0.0177079306095842|-0.0192076830732293,-0.0240096038415366,-0.0456182472989196|-0.0296118447378952|0.0140683440025986|
|user_balance|supported|ndcg@10|0.003992966653173,0.00402604754482618,0|0.00267300473266639|0.0023149490949027|-0.00214461333513443,-0.00194064552956509,-0.00484249249825105|-0.00297591712098352|0.0016197155446847|
|user_balance|supported|ndcg@20|0.00622295042096956,0.00611321025894077,0.000282827538852275|0.0042063294062542|0.00339829529373484|-0.00227891633606737,-0.00358065552736985,-0.00774036197923906|-0.00453331128089209|0.00285263244510558|
|user_balance|supported|ndcg@50|0.010159332185879,0.00876524608779401,0.000859898177790431|0.00659482548382115|0.00501526793066044|-0.00457178730064,-0.00592160056490281,-0.0121220106337527|-0.00753846616643185|0.00402643252458375|
|user_balance|unsupported|recall@10|0,0,0|0|0|0,0,0|0|0|
|user_balance|unsupported|recall@20|0,0,0|0|0|0,0,0|0|0|
|user_balance|unsupported|recall@50|0,0,0|0|0|0,0,0|0|0|
|user_balance|unsupported|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|user_balance|unsupported|ndcg@20|0,0,0|0|0|0,0,0|0|0|
|user_balance|unsupported|ndcg@50|0,0,0|0|0|0,0,0|0|0|
|watch_history|all|recall@10|0.0194444444444444,0.021031746031746,0.0214285714285714|0.0206349206349206|0.00104990131391452|0.00912698412698413,0.0103174603174603,0.0130952380952381|0.0108465608465608|0.00203634977755829|
|watch_history|all|recall@20|0.0400793650793651,0.0369047619047619,0.0440476190476191|0.0403439153439153|0.00357876964509754|0.0234126984126984,0.0166666666666667,0.0273809523809524|0.0224867724867725|0.00541682413477335|
|watch_history|all|recall@50|0.0797619047619048,0.0757936507936508,0.0813492063492064|0.078968253968254|0.00286154863132063|0.0424603174603175,0.0388888888888889,0.0484126984126984|0.0432539682539683|0.00481125224324689|
|watch_history|all|ndcg@10|0.00847858398933836,0.0093604886787091,0.00901239168191637|0.00895048811665461|0.000444199289539742|0.00442096166373512,0.00541584159063932,0.00581096608585039|0.00521592311340828|0.000716242711326817|
|watch_history|all|ndcg@20|0.0136807236523893,0.0133554516845239,0.0146364656738312|0.0138908803369148|0.000665863037146753|0.008060045074126,0.00694672930357409,0.00933224593687079|0.00811300677152363|0.00119363985689874|
|watch_history|all|ndcg@50|0.021583302685956,0.0210825399536602,0.0219545970357127|0.021540146558443|0.000437627383850605|0.0118443959143128,0.0113729024443773,0.0133721128769703|0.0121964704118868|0.00104507316840909|
|watch_history|history_1|recall@10|0.0176211453744493,0.0308370044052863,0.039647577092511|0.0293685756240822|0.0110863941780775|0.00440528634361234,0.026431718061674,0.026431718061674|0.0190895741556535|0.0127169662817098|
|watch_history|history_1|recall@20|0.0308370044052863,0.0440528634361234,0.052863436123348|0.0425844346549192|0.0110863941780775|0.0176211453744493,0.039647577092511,0.039647577092511|0.0323054331864905|0.0127169662817098|
|watch_history|history_1|recall@50|0.0572687224669604,0.066079295154185,0.0881057268722467|0.0704845814977974|0.0158834857949956|0.039647577092511,0.039647577092511,0.0616740088105727|0.0469897209985316|0.0127169662817098|
|watch_history|history_1|ndcg@10|0.00680215854628434,0.0108796266744989,0.0165903632402392|0.0114240494870075|0.00491676061363321|0.00258071723158837,0.00867698350269273,0.0104452546720534|0.00723431846877816|0.00412597856128443|
|watch_history|history_1|ndcg@20|0.0101651803715596,0.0140766693930046,0.0198970089095257|0.0147129528913633|0.00489701584547218|0.00594373905686365,0.0118740262211984,0.0137519003413399|0.010523221873134|0.00407557970737424|
|watch_history|history_1|ndcg@50|0.0154676306322431,0.018581073646215,0.0271524828858481|0.0204003957214354|0.00605114804456006|0.0103940906231979,0.011954577379805,0.0181518899623163|0.0135001859884397|0.00410335742104206|
|watch_history|history_2_5|recall@10|0.0225903614457831,0.0256024096385542,0.019578313253012|0.0225903614457831|0.00301204819277108|0.0150602409638554,0.0135542168674699,0.0150602409638554|0.0145582329317269|0.000869503417454256|
|watch_history|history_2_5|recall@20|0.0481927710843374,0.0481927710843374,0.052710843373494|0.0496987951807229|0.00260851025236277|0.0316265060240964,0.0271084337349398,0.0406626506024096|0.0331325301204819|0.00690146942011422|
|watch_history|history_2_5|recall@50|0.0873493975903614,0.0843373493975904,0.088855421686747|0.0868473895582329|0.00230048980670474|0.0557228915662651,0.0466867469879518,0.0632530120481928|0.0552208835341365|0.00829453395675618|
|watch_history|history_2_5|ndcg@10|0.0101833766068625,0.0129248693960084,0.00977591334749234|0.0109613864501211|0.00171258738055229|0.00743653346011755,0.00865596064684983,0.00821619795991161|0.00810289735562633|0.000617558447544856|
|watch_history|history_2_5|ndcg@20|0.0166300716566497,0.0185708065422151,0.0180817243533674|0.0177608675174107|0.00100936851838051|0.0116362286895411,0.0119227706645675,0.0146299480779261|0.0127296491440116|0.00165193177962667|
|watch_history|history_2_5|ndcg@50|0.0244249586154164,0.0256271049101308,0.0250599550996734|0.0250373395417402|0.00060139215664104|0.0164438298979282,0.0156776419316816,0.0187498324233964|0.0169571014176687|0.00159911673872746|
|watch_history|history_6_20|recall@10|0.0131004366812227,0.0141921397379913,0.0174672489082969|0.014919941775837|0.00227256113478835|0.00545851528384279,0.00655021834061135,0.00982532751091703|0.00727802037845706|0.00227256113478835|
|watch_history|history_6_20|recall@20|0.037117903930131,0.0283842794759825,0.0360262008733624|0.0338427947598253|0.0047586233008086|0.0251091703056769,0.00873362445414847,0.0218340611353712|0.0185589519650655|0.00866512438121591|
|watch_history|history_6_20|recall@50|0.0796943231441048,0.0665938864628821,0.0786026200873362|0.0749636098981077|0.007268917159912|0.046943231441048,0.0338427947598253,0.0447598253275109|0.0418486171761281|0.00701866867612297|
|watch_history|history_6_20|ndcg@10|0.00449132028227372,0.00582891624364209,0.00654324699794316|0.00562116117461966|0.00104162010260357|0.00168715622878081,0.00339681380842,0.00409923551858505|0.00306106851859529|0.0012405947363282|
|watch_history|history_6_20|ndcg@20|0.0106107454669661,0.0094449178559075,0.0111662611036523|0.0104073081421753|0.000878519039621598|0.00670871343553222,0.00392124204805369,0.0069660450839363|0.0058653335225074|0.00168854186803298|
|watch_history|history_6_20|ndcg@50|0.0190977328574119,0.0170780218395286,0.0195606429718398|0.0185787992229268|0.00132015946362532|0.0110325173403132,0.00896517669857291,0.0114837777369294|0.0104938239252718|0.00134293735675171|
|watch_history|history_gt20|recall@10|0.0252454417952314,0.0224403927068724,0.0224403927068724|0.0233754090696587|0.00161949584625421|0.00981767180925666,0.00701262272089762,0.0112201963534362|0.00935016362786349|0.00214239162924537|
|watch_history|history_gt20|recall@20|0.0392706872370266,0.0350631136044881,0.0434782608695652|0.0392706872370266|0.00420757363253857|0.0154277699859748,0.00981767180925666,0.0182328190743338|0.0144927536231884|0.00428478325849073|
|watch_history|history_gt20|recall@50|0.0799438990182328,0.0827489481065919,0.0757363253856942|0.0794763908368396|0.00352960936665299|0.0252454417952314,0.0378681626928471,0.0350631136044881|0.0327255726975222|0.00662807240708641|
|watch_history|history_gt20|ndcg@10|0.0125471672056195,0.010094467962286,0.00906086096780377|0.0105674987119031|0.00179064272358297|0.00571066936600373,0.00395400594008252,0.00429467958679053|0.00465311829762559|0.000931571430466356|
|watch_history|history_gt20|ndcg@20|0.0159973593715577,0.0132928106490127,0.0142113777475607|0.0145005159227104|0.00137526237911227|0.007139477509238,0.00463082251823933,0.00603141032651006|0.00593390345132913|0.00125716671885486|
|watch_history|history_gt20|ndcg@50|0.0240772437546481,0.0227913482471218,0.0204833264607575|0.0224506394875091|0.00182102230161773|0.00906582075266644,0.0102720462778201,0.00926818554327739|0.0095353508579213|0.000645970574712677|
|watch_history|in_catalog|recall@10|0.025830258302583,0.0279388508170796,0.0284659989457037|0.0274117026884555|0.00139470285243257|0.0121244069583553,0.0137058513442277,0.0173958882445967|0.0144087155157266|0.00270511409564939|
|watch_history|in_catalog|recall@20|0.0532419609910385,0.0490247759620453,0.0585134422772799|0.0535933930767879|0.00475408513739894|0.0311017395888245,0.022140221402214,0.0363732208750659|0.0298717272887015|0.00719578113844431|
|watch_history|in_catalog|recall@50|0.105956773853453,0.100685292567211,0.108065366367949|0.104902477596205|0.00380131921503847|0.0564048497627834,0.0516605166051661,0.0643120716921455|0.0574591460200316|0.00639133139324308|
|watch_history|in_catalog|ndcg@10|0.0112630636020731,0.0124345975067722,0.011972180832066|0.0118899473136371|0.000590080237026962|0.00587286420274776,0.00719447591376441,0.00771936454208908|0.00692890155286708|0.000951466332389867|
|watch_history|in_catalog|ndcg@20|0.0181736550363843,0.0177415594333159,0.0194432754338717|0.0184528299678573|0.000884541303958786|0.0107070709471785,0.0092281274881427,0.0123970794733339|0.0107774259695517|0.00158564704237471|
|watch_history|in_catalog|ndcg@50|0.0286715460034839,0.0280063261377036,0.0291647783500243|0.0286142168304039|0.000581350030207445|0.0157342528751019,0.0151079146862577,0.0177636923826911|0.0162019533146836|0.00138828907980544|
|watch_history|oov|recall@10|0,0,0|0|0|0,0,0|0|0|
|watch_history|oov|recall@20|0,0,0|0|0|0,0,0|0|0|
|watch_history|oov|recall@50|0,0,0|0|0|0,0,0|0|0|
|watch_history|oov|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|watch_history|oov|ndcg@20|0,0,0|0|0|0,0,0|0|0|
|watch_history|oov|ndcg@50|0,0,0|0|0|0,0,0|0|0|
|watch_history|pop_high|recall@10|0.0488058151609553,0.0519210799584631,0.0539979231568017|0.0515749394254067|0.00261330371591234|0.023883696780893,0.0269989615784008,0.0363447559709242|0.0290758047767394|0.00648494080830571|
|watch_history|pop_high|recall@20|0.0944963655244029,0.0903426791277259,0.107995846313603|0.0976116303219107|0.00922969306055616|0.0539979231568017,0.0467289719626168,0.0706126687435099|0.0571131879543094|0.0122428100961076|
|watch_history|pop_high|recall@50|0.186915887850467,0.179646936656282,0.187954309449637|0.184839044652129|0.00452637481156872|0.102803738317757,0.0965732087227414,0.114226375908619|0.104534440983039|0.00895293676453968|
|watch_history|pop_high|ndcg@10|0.0214726363969202,0.0227862507340821,0.0227644467415486|0.0223411112908503|0.000752200328744127|0.011686366132272,0.0135815598366159,0.0157749228692032|0.013680949612697|0.00204608963475905|
|watch_history|pop_high|ndcg@20|0.0330636922046832,0.0324484891094666,0.0362060606984491|0.0339060806708663|0.0020154527549786|0.0194412316860525,0.018371154279501,0.0242161079979135|0.020676164654489|0.00311201953239283|
|watch_history|pop_high|ndcg@50|0.0514466764882876,0.0501952733240432,0.0519243136589309|0.0511887544904206|0.000892909814961711|0.0290878921469872,0.028296148977674,0.0325877566807526|0.0299905992684712|0.00228377631997189|
|watch_history|pop_low|recall@10|0,0,0|0|0|0,0,0|0|0|
|watch_history|pop_low|recall@20|0,0,0|0|0|0,-0.00645161290322581,-0.00645161290322581|-0.0043010752688172|0.00372484044638468|
|watch_history|pop_low|recall@50|0,0.00645161290322581,0|0.0021505376344086|0.00372484044638468|-0.00645161290322581,0,-0.00645161290322581|-0.0043010752688172|0.00372484044638468|
|watch_history|pop_low|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|watch_history|pop_low|ndcg@20|0,0,0|0|0|0,-0.00157839059431114,-0.00146884031417389|-0.00101574363616168|0.000881363527026019|
|watch_history|pop_low|ndcg@50|0,0.00114905544228394,0|0.000383018480761314|0.000663407468916439|-0.00144673434979081,-0.000429335152027193,-0.00146884031417389|-0.00111496993866396|0.000593880007998387|
|watch_history|pop_mid|recall@10|0.00364963503649635,0.00547445255474453,0.00364963503649635|0.00425790754257908|0.00105355888538253|0,0,-0.00364963503649635|-0.00121654501216545|0.00210711777076506|
|watch_history|pop_mid|recall@20|0.0182481751824818,0.0109489051094891,0.0127737226277372|0.0139902676399027|0.00379866058296739|0.0127737226277372,-0.00364963503649635,0.00364963503649635|0.00425790754257908|0.00822855794310747|
|watch_history|pop_mid|recall@50|0.0383211678832117,0.031021897810219,0.0437956204379562|0.037712895377129|0.0064085485114676|0.0164233576642336,0.00912408759124088,0.0237226277372263|0.0164233576642336|0.0072992700729927|
|watch_history|pop_mid|ndcg@10|0.00125526058923088,0.00300232119238305,0.00143989931809834|0.00189916036657075|0.000959815461823049|-0.000206472979499012,0.00103809979151462,-0.000999299610765871|-5.58909329167533e-05|0.00102701280209543|
|watch_history|pop_mid|ndcg@20|0.00480855476443644,0.00439387451201425,0.00368149095884682|0.0042946400784325|0.000570047199864847|0.00290037860060029,0.000107640175119474,0.000775215342373465|0.00126107803936441|0.0014583874308752|
|watch_history|pop_mid|ndcg@50|0.00884447684377386,0.00841578262521185,0.00971253736577651|0.00899093227825407|0.000660666413563473|0.00376000253784163,0.00269538256002737,0.00464121353119949|0.0036988662096895|0.000974355058517227|
|watch_history|supported|recall@10|0.0294117647058824,0.031812725090036,0.0324129651860744|0.0312124849939976|0.00158808602104717|0.0138055222088836,0.0156062424969988,0.0198079231692677|0.01640656262505|0.00308019294084447|
|watch_history|supported|recall@20|0.06062424969988,0.0558223289315726,0.0666266506602641|0.0610244097639056|0.00541326500939123|0.0354141656662665,0.0252100840336134,0.0414165666266507|0.0340136054421769|0.0081935154979765|
|watch_history|supported|recall@50|0.120648259303721,0.114645858343337,0.123049219687875|0.119447779111645|0.00432839288771187|0.0642256902761104,0.0588235294117647,0.0732292917166867|0.0654261704681873|0.0072775244015499|
|watch_history|supported|ndcg@10|0.0128247488914362,0.0141587223711566,0.013632189098697|0.0135385534537633|0.000671898085018096|0.00668716890312875,0.00819202929676536,0.00878969660044597|0.00788963160011336|0.00108339233646073|
|watch_history|supported|ndcg@20|0.0206935315750427,0.0202015235564227,0.0221391917755429|0.0210114156356694|0.0010071877872808|0.0121916648180057,0.0105076577701121,0.0141160022574516|0.0122717749485231|0.00180550566589725|
|watch_history|supported|ndcg@50|0.0326470124661519,0.0318895562324272,0.0332086341716663|0.0325817342900818|0.000661957387337052|0.0179158929796329,0.0172027095797304,0.0202267253601231|0.0184484426398288|0.00158078294381208|
|watch_history|unsupported|recall@10|0,0,0|0|0|0,0,0|0|0|
|watch_history|unsupported|recall@20|0,0,0|0|0|0,0,0|0|0|
|watch_history|unsupported|recall@50|0,0,0|0|0|0,0,0|0|0|
|watch_history|unsupported|ndcg@10|0,0,0|0|0|0,0,0|0|0|
|watch_history|unsupported|ndcg@20|0,0,0|0|0|0,0,0|0|0|
|watch_history|unsupported|ndcg@50|0,0,0|0|0|0,0,0|0|0|

uniform_negatives相对core小幅正向；user_balance明显较弱，seed2025在部分指标接近零且early stop，如实保留；watch_history在该固定离线query set上的paired指标较高，不据此推断线上/因果效果。Raw/prediction/checkpoint/embedding/optimizer/fullbinding/full日志留server。

Hash-only验收入口cache/p4e-final-delivery-v1/review-package.json；small public manifest artifacts/p4e-ablation-manifest-formal-v1.json。历史启动/prepare/pilot固定包不restamp。

### P4F local consumer implementation — awaiting coordinator review

仅synthetic/local工程验证，无P4F真实quality或test结果。Fresh scoped review：21 tests，20 pass，真实FAISS integration 1 skip；无剩余Critical/Important。Related regression actualexit0，108 tests、1 skip（先于最后新增progress-interruption test；后者已在final scoped/full suite覆盖）。Initial full regression actualexit1，215 tests，19 sandbox permission errors、1 frozen identity failure、2 skips；允许fixture writes后full regression actualexit1，215 tests、1 failure、0 errors、2 skips。唯一剩余为test_complete_producer_pin_and_explicit_cli_identity：当前source集合不等于冻结P4D producer identity，正确拒绝；保护adapter/tests不改，待Astra裁决。原P4E source_files 25成员hash全部保持。证据cache/p4f-local-review-v1.json；公开hash artifacts/p4f-source-manifest-v1.json。P4F实验未执行，FAISS/server真实验证仍pending。

### P4F local fixture revision v2 — READY awaiting coordinator acceptance

按Astra定点授权保留v1失败，精确22个历史source成员重建fingerprint等于冻结P4D身份；adapter/生产pins不改。Fresh RED actualexit1：2 tests/2 failures（未来module污染、原CLI identity失败）；GREEN该文件11 tests actualexit0。最终独立scoped reviewer另fresh11 tests actualexit0，无Critical/Important；fresh full regression217 tests actualexit0、2 skips（旧FAISS reference及新真实HNSW integration，本地dependency不可用）。原P4E25 source members及新P4F三个consumer hashes均保持。证据cache/p4f-local-review-v2.json，公开入口artifacts/p4f-source-manifest-v2.json；无P4F真实quality/test结果，真实FAISS仍pending。


### G3 supplemental actual terminal — awaiting coordinator acceptance

- Pilot actual exit0；acceptance actual exit0；original parent实际wait worker exit0，closure/cleanup成功。Supplemental elapsed至worker收束 1084.401599s /1800s；detached parent自身actual exit保持null。
- Frozen pool共 184 optimizer updates /200，128 decode /128，1 resume /1；microbatch 512，fresh-process restore_exact=True。旧+新磁盘fresh snapshot 25198592 bytes /2147483648，包含closure发布后bytes；旧失败不退款。
- 以下仅已冻结pilot工作量的实测计时，不推断formal full-validation/replay成本：
  - capacity512/128: 8 updates，4096 examples，21.347019s；191.877 examples/s。
  - continuous/128: 8 updates，4096 examples，20.917158s；195.820 examples/s。
  - resumed/128: 8 updates，4096 examples，21.962925s；186.496 examples/s。
  - throughput/128: 80 updates，40960 examples，232.868579s；175.893 examples/s。
  - throughput/64: 80 updates，40960 examples，224.355528s；182.567 examples/s。
  - decode continuous/128: 16 queries，3.094955s；0.193435s/query。
  - decode resumed/128: 16 queries，2.883574s；0.180223s/query。
  - decode throughput/128: 48 queries，10.070661s；0.209805s/query。
  - decode throughput/64: 48 queries，9.497518s；0.197865s/query。
- Pilot receipt self maxrss=1524162560 bytes，waited children maxrss=1603756032 bytes，CUDA initialized=true；只对应该receipt进程范围。Final瞬时 RAM available/total=187137953792/211900432384，VRAM free/total=51525976064/51527024640，disk free/total=256551669760/536870912000。不称全历史峰值/minimum，hidden ancestry仍unknown。
- 真实终态入口cache/g3-supplemental-scheduler-v1/terminal-review.json。产物/raw/checkpoint/embedding/逐query prediction/full bindings留server；formal_training_permitted=false，test_gate=closed。


## S0 全量容量 readiness（2026-10-08T19:32:34.906365+00:00）

状态BLOCKED awaiting Astra decision。唯一执行worktree，固定revision `084a18386e149a454dd6dbe0123478e75c6b09ce`；一次只读SSH actualexit0，server≤60s/transport≤90s/output≤64KiB，实际transport 5.869s，stdout 5951bytes。无server写入/下载/训练/资源变更；原包不restamp。

| 容量项 | bytes | 证据类别 |
|---|---:|---|
| 可见总盘 | 536870912000 | statvfs瞬时实测 |
| 当前available | 256465813504 | statvfs瞬时实测 |
| 总盘20%余量 | 107374182400 | 算术 |
| 可新增（未扣未知未来writer） | 149091631104 | 算术上限 |
| 官方30文件parent raw | 383167690036 | 固定revision官方metadata |
| metadata/embedding（已含于raw） | 2590612234 | 官方metadata |
| 全new raw缺口 | 234076058932 | 算术 |
| 即使全部metadata复用的raw缺口 | 231485446698 | 条件估算，未fresh复用验收 |
| 新G3 nominal observation额度 | 1073741824 | 规划预留，非hard quota |
| 扣nominal G3后的可新增 | 148017889280 | 条件算术上限 |

Inodes total 781351296 / available 773801735，只为瞬时可见值。已知项目observed allocated bytes逐目录见snapshot/ledger；P4D 139159511040、P4E 126557241344，扫描完整；P2/P4/P6因depth/entry cap仅partial，50,001个entry observed包含触cap后首个检查entry，不称完整磁盘inventory或连续历史minimum。overlay block accounting与df可能不同。

Raw383.167690036GB=356.852719594GiB；train25、validation1、test1及metadata/embedding3，不含subsamples。expanded/decoded、processed/history/catalog/maps、model-export、checkpoint/optimizer多seed留存、ANN/SID/trie及cache/spill/atomic临时容量全部unknown，不能按零补齐；NPZ/Parquet storage不是解码size。全量训练次数、下载/训练吞吐、ETA和流量费用unknown，subset短测不可直接外推。单个最大week 16104375210bytes只能支持分区输入估算，不能证明整链容量。

固定官方inventory SHA 7bbdeaafb4dcbcf0d5ff5a61e321ff7e14baa02aac54f1ac479f6c76aa98ffe5、API SHA e99eb740e91fd7c852f7dcb189278caee9f46b0710b03f6cfd32b6a2f38b4cd1；30unique文件bytes/LFS SHA逐项匹配API且合计复算通过。Snapshot SHA 8e513293236f42eb7a0098279698e8dceb0dc09352077827d8f30380ecd1cc3b，详情cache/s0-capacity-readiness-v1/capacity-ledger.json与review.md。未重复读取旧raw进行hash/replay；既有metadata候选复用与streaming reader完整性契约另审，边处理边删未经授权。S1下载gate关闭。


## P4F runtime gate v1 local proposal
本地8个proposal/拒绝与独立点积oracle checks实际exit0（python3 -B cache/p4f-runtime-gate-v1/test_proposal.py），不native执行；34source+3parent fresh pins通过，68dependency成员完整等于approved wholewheel+RECORD表。8x8basis/4query/K3的expected IDs只为synthetic预期，实际FAISS尚未测试。proposed资源/输出/进程预算见ignored plan.json，未执行不称实测。v3已获Astra本地验收，真实gate仍关闭。


### 2026-10-09 P4F runtime adapter 本地回归
固定入口 cache/p4f-runtime-gate-implementation-v1/review.json。final-local-regression.log：46 unique tests，0 failures/0 errors，actual local exit0；未执行 Linux/native FAISS。final-pin-check.json 核42 transported source/parent/control files、3 bootstrap headers、68 dependency members及既有v3 pins；wire495839 bytes。A/B各120s、1attempt/0resume，累计execute240s；transport150s observation另有最多2s local cleanup，资源仅128MiB allocated/2GiB RSS观察及20%余量，不称硬quota/历史峰值。


### 2026-10-09 G3真实成本入口单attempt失败
入口cache/g3-cost-entry-execution-v1/review.json。冻结v2 permit执行一次；payload1878783 bytes全量送达。transport实际exit1、worker实际exit1；transport观察61.310034875s，server parent receipt snapshot elapsed1.7368374709039927s（不是完整wall成本）。owned worker PID2611363/birth131584912，PGID/SID同PID；cleanup verified，fresh只读无成员。observed new-root disk peak2895872bytes，仅本段观察；final write后observation通过。真实preflight false/actual cost false，decode reserved/completed/primitive invocation均null，partial timings unavailable，失败阶段null。generic failure code不能说明具体根因。不称完成成本测量，不重试。


### P4F runtime A actual与B本地候选收口
A原transport actual exit0、stage actual exit0，单attempt；server receipt execution_seconds5.963389645097777，transport observation11.739231083s，二者边界不同。一次只读metadata/hash检查actual SSH exit0；31固定本地成员、42server source/parent/control pins、runtime origin hashes与release绑定通过，原groupfresh无活成员。B prepare/render actual local exit0，120s/1attempt/0resume，仅render未执行；入口cache/p4f-runtime-A-closeout-B-candidate-v1/review.json。资源仅receipt snapshot gate通过，无历史峰值/连续minimum/硬quota结论；无真实融合quality或test结果。

P4F B实际单attempt transport exit0/stage exit0，summary READY；server receipt execution3.084159273188561s，transport observation6.843347291000001s，非全量ANN性能。独立server receipt/oracle/reload核验尚未执行，审批超时在CreateProcess前，不把summary替代最终独立验收。入口cache/p4f-runtime-execution-B-v1/review.json。

P4F B独立saved receipt数值/hash收口单次actual SSH exit0；transport observation3.8284589999999996s，stdout784bytes，server alarm30s/local60s/cap8192bytes。Tiny8x8/4queries/K3 FlatIP及HNSW oracle与序列化reload核通过，非全量ANN指标，未native replay。入口cache/p4f-B-readonly-bounded-retry-v1/review.json，资源仅receipt观察门通过。

P4F real-runtime metadata binding scoped11tests actualexit0，旧31fixed成员/42transport files hash保持。RED首1fail/8skip；fixture路径1error及audit分类2error保留；fresh正确RED2fail/0error→GREEN11pass。无真实input构造/模型/ANNquality/server动作。Full project regression本地缺numpy/torch/yaml环境不可执行，未安装。入口cache/p4f-real-runtime-candidate-v1/review.json。

G3 synthetic Linux生命周期probe：5/5 fixed tinyCPU case PASS，preflight/submit/status transport actualexit均0；server terminal elapsed 9.300280467839912秒，fresh disk max_bytes 1396736，8MiB cap内。controller process exit未观察(null)，case actual parent exits及owner wait证据完整。非真实cost、非模型latency、0train/decode/resume；入口cache/g3-linux-lifecycle-execution-v1/review.json。资源仅preflight时点。


### P4F stat后本地完整候选 — 2026-10-10

唯一stat-only actual transport exit0，local观察3.465063916s，stdout867bytes，内容读取0。六metadata总16472067bytes，core_run regular16026248bytes超过原8388608单项门；graph41933150bytes、checkpoint102419921bytes，七payload单pass178377748bytes。Stat不是fresh hash/header/resource reserve，未新SSH或native运行。

新ignored binding角色读取限额core20MiB/graph48MiB/default8MiB，strict JSON同时拒绝溢出浮点Infinity。Fresh正确RED2tests2fails（实际大小producer链拒绝与1e309未拒绝），最小修复后最终binding26executions/15unique cases exit0，最终0.315s前次与fresh final日志见packet；不把重复继承tests称独立新增覆盖。合并readonly候选8tests初1error来自macOS临时目录symlink别名，修正fixture真实路径后fresh8tests exit0；错误证据保留，不称该错误行为RED。Frozen validator AST与原源码相同，固定23members和原binding/refs/source hash保持。Full suite本地缺numpy/torch/yaml未运行，无安装。

合并提案一次读取六metadata、一个有界content NPY header、七payload stat及资源快照；server60s/local90s/stdout8KiB/address-space1GiB/metadata total24MiB/core20MiB，未准入未执行。真实任务有限投入候选7200s、owned aggregate RSS8GiB、新disk4GiB、logs16MiB、1initial attempt/0failure retry/1planned infer resume；这些为待审STOP ceiling，不是预计耗时/峰值或容量证明。Fresh资源20%余量、G共享预约、native解析峰值和真实stage supervisor未闭合；approved预算仍null，真实fusion/ANN未开始。入口cache/p4f-stat-readiness-v1/review.json SHA bc4eadc89e82a25aa713a9ce1bd317c9e76fce6f91e0c3f961538b96dcd597dc，wait/local交付，不重发通知。


### P4F合并只读actual失败与窗口安全收尾 — 2026-10-10

单次已准入合并只读transport exit1、非timeout，local观察4.701013667s，stdout108bytes/stderr仅记录321bytes；sanitized结果STOP/metadata/ValueError。无法区分role或具体predicate；header/resource/lineage未返回成功证据，不retry、不换pins。Local transport控制脚本本身exit0只说明保存receipt，不替代远端script exit1。

Real owned adapter新contract2tests actual RED2fail（旧A/B Runner不接受real plan契约）；拟实施命令被自动审批审查deadline拒绝于CreateProcess前，未运行，未GREEN，未产生adapter子进程。不是代码执行失败或unsafe判定；工具允许once retry，但随后协调明确6h管理窗口结束，故不retry。仅保留原A/B复制scaffold、RED/tests及草稿未创建的显式记录；不是完整adapter。当前SAFE_HANDOFF_WIP_WINDOW_ENDED。数字无模型quality含义。入口cache/p4f-merged-readonly-execution-v1/review.json，actual预算仍未approved，真实ANN未开始，wait/local收尾。


### P4F complete candidate v2 本地执行链 — 2026-10-10

新 ignored runner/stage_driver/binding_v2/pipeline 已实现固定七阶段、原绝对 wall alarm、一次 reservation、actual subprocess wait、非零后不推进、post-proof 资源门与有界树/cgroup 读取。Fresh RED→GREEN 新增 export 完整文件/query/shape、完整 shard/计数、部署 validator 原 SHA 和 whole-owner wall 门；当前 related regression actual exit0，52次执行（diagnostic继承重复，不宣称52项独立新增覆盖），2.885s。仅 stdlib synthetic/subprocess 与保存产物核验，不是 Linux closure、真实模型/ANN质量或完整项目 suite。原7保护refs与全部已验收 source members逐byte核通过。

metadata 旧失败仍未定位具体 role/predicate；本地新合并 enum diagnostic proposal 待一次准入，未SSH、不retry。真实投入 ceiling 沿用候选、approved budget仍null；foreground外部 owner actual wait与fresh联合资源/输入证明必须在exact-start包中闭合，不以本地owner-result冒充自身exit。入口 cache/p4f-complete-candidate-v2/review.json；P4F未Complete。


### P4F v3 proof supervision 差异修复 — 2026-10-10

原一次独立review仅1 Important：proof在owner内执行暂停stage cutoff/资源轮询。Fresh定点RED4tests/4fail，最小owned proof child实现后出现自然exit与身份读取race（4tests/1error），修复后4tests exit0；加忽略TERM的bounded KILL与actual wait后最终5tests exit0/0.977s。受影响owned原fixture改跨process真实文件观察后12tests exit0/0.433s；pipeline6tests exit0/0.770s。不重跑旧52项suite，无SSH/native/LINUX结论。超时与proof期间瞬时RSS超限均拒绝，保存proof actual_exit/PID与原wait，不mark stage valid。旧reviewed runner/test exactbytes按原SHA恢复归档，其余原members与历史保护保持。入口cache/p4f-proof-supervision-revision-v3/review.json，awaiting原reviewer一次差异核验；真实预算仍null，new enum diagnostic技术scope已接受但本执行对话remote授权尚未取得。


### P4F v4 proof cleanup race 定点修复 — 2026-10-10

原review发现异常proof natural-exit与identity/kill race绕过最终wait。只修supervised_proof exception cleanup：先关G admission pipe，再在独立finally无条件bounded原Process join并保留真实exit/reaped，真实wait timeout才记unknown。确定性真实child+受控signal race和pre-handshake STOP fixtures fresh RED7tests/3fail（含原即时异常反例），最小修复后仅相关proof7tests actualexit0/1.009s。未重跑owned/pipeline或旧suite，未SSH/native。v3全部22members及科学/历史保护不变；新packet cache/p4f-proof-wait-revision-v4/review.json，awaiting原reviewer只审该race差异。真实预算仍null，诊断technical scope已接受、执行准入仍缺。


### P4F v4 本地差异协调ACCEPT — 2026-10-10

原reviewer fresh proof7全部通过，既有即时异常退出反例10/10保留原actual wait/reap且stage返回PID消失；v4实际加载SHA、14members与v3原22members匹配，唯一Important关闭。此为协调验收接收，owner不重跑；macOS PID/wait证据不替代Linux closure。仅ACCEPT_LOCAL_REVISION，不是P4F真实实验Complete。接收证据cache/p4f-proof-wait-revision-v4/coordinator-acceptance.json；等待本执行对话human批准固定一次enum诊断，当前0SSH、真实预算null。


### P4F enum固定诊断实际STOP — 2026-10-10

Eddy在执行对话直接批准固定script/input后单次SSH诊断：actual transport exit1，local observation26.323272584s，非timeout，stdout162bytes，stderr仅计321bytes不保留正文。sanitized结果STOP/metadata/legacy_manifest/length_or_hash/ValueError；具体长度或SHA分支仍unknown。server60s/local90s/stdout8KiB及原读取/内存门保持。未retry、未追加probe、0serverwrite/native/model/ANN；header/lineage/resources无成功证据，不称fresh reserve已建立。证据cache/p4f-enum-diagnostic-execution-v1/review.json；P4F真实预算仍null、BLOCKED_ON_EVIDENCE。


### P4F legacy manifest本地身份修复候选 — 2026-10-10

已有public副本7250bytes、SHA84cdef...b2f；fixed诊断及旧stat指定server原件8567bytes，预期SHA却误用public副本。P4 accepted final table/public server_manifest字段/P7 stage map共同绑定原件65d90...801与audit726965...e5b。新refs候选只改该预期SHA为历史原件pin，不来源于远端现值。fresh stdlib RED3tests/1fail→GREEN3tests/exit0。旧v4十四成员/v3二十二成员与实际诊断两成员hash核保持，无新SSH/native/whole suite。consumer已接受守卫仍要求public artifact SHA，故raw producer与public audit身份须在未来context分开；candidate仅本地映射正确，不称全链GREEN或远端原件匹配。远端hash、short-read/漂移、legacy audit/header/resources均unknown；真实预算null。入口cache/p4f-legacy-manifest-mapping-v1/review.json。


### P4F separated entry 本地验证 — 2026-10-10

身份尺寸前置新反例：identity-size-red.log 5tests/1fail（实际错误 input hash mismatch，未先拒绝尺寸）；最小修复后仅新 binding/entry suite new-binding-entry-final-v3.log 14tests、0fail、actualexit0、0.247s。此前 13tests 最终通过以及中途失败/RED 日志原样保留，不以重复执行计新增覆盖。完整 framed bootstrap 本地 PENDING validation actualexit1（预期拒绝），stdout78bytes/stderr0；无远端动作。candidate-consistency PASS：34accepted source members、46deployment members，旧 v4 14/v3 22/mapping 7/actual diagnostic 2 members SHA 保持，固定七 stage argv 与 canonical context/plan匹配；runner/pipeline/binding保持v4 exactbytes。新 proposed deployment523331bytes、framed transport783581bytes；不是实际服务器消耗。真实预算null，未产生新模型指标或 ANN 结果。入口 cache/p4f-separated-entry-v1/review.json。


### P4F foreground I1/I2 修复本地验证 — 2026-10-10

真实反例在原v1 fresh RED：lifecycle-integration-red.log 4tests/3fail/1error，包含initial/fast identity失wait与resource后marker继续写。post-spawn helper异常新RED helper-exception-red.log 5tests/1error。最终定点 lifecycle-final-v2-green.log 5tests/0fail/actualexit0/0.606s；仅受影响entry-final-v2-green.log 5tests/0fail/actualexit0/0.069s。最初fixture symlink路径与误替换assertion造成的中间失败保留，不计有效RED或成功。完整bootstrap本地PENDING预期actualexit1/stdout78bytes/stderr0。原v1 42members与不变controls exactbytes保持；framed candidate789165bytes、拟deployment525723bytes，未服务器消耗。无旧proof/full suite/SSH/native/model/ANN，真实预算null，无新增quality指标。入口cache/p4f-separated-entry-v2/review.json。


### P4F exact refreeze method本地TDD — 2026-10-10

refreeze method fresh red.log4tests/4fail（missing implementation）；green.log4tests/0fail/0.046s；final-green.log4tests/0fail，精确时长见原log。覆盖pending写前拒绝、scope漂移、固定同输入/time生成同hash且context/plan/members不变、stale start与existing output拒绝。全部synthetic临时approval/time，0真实permit/远端执行，未重跑已验收v2/proof/native suite。入口cache/p4f-exact-refreeze-method-v1/review.json，真实budget未批。


### P4F 已审完整entry实际STOP — 2026-10-10

Eddy执行对话直接批准“批准开始吧”后唯一1attempt，generator/context/plan/source输入budget所有固定pins核保持。originalstart1791619129.685371/deadline1791626329.685371，absolute7200s，final-start SHA04b1aad39b646321205366d592c957f613621658cc8a4105f6c52b26a290830b。local SSH Popen58154原wait/reap、actualexit1、非timeout，elapsed10.801568984985352s，stdout137bytes/stderr仅计321bytes不保留正文。返回STOP，foreground_result_sha256860f170e8f1c2b47c0aac0cc50d5471ab21ec08e188ffe24d4df4adbb3f2cc5b，owner_actual_exit null。receipt SHA db32e47fddd16a0f53ea672ab59ca88bbb34b27fa3293944df8220ece5c56ca4。不能由localexit或genericSTOP识别preflight/native具体phase/进程closure；未取得完整7stage/ANN验证结果，0retry/新extractor/后续SSH/预算reset/refund。证据cache/p4f-actual-entry-v1/review.json。


### P4F 原foreground收据 bounded只读定位 — 2026-10-10

Astra批准原同job收尾，fixedreader SHA76149eb780748d583bea48d0e3a36240eed09192feb23bf7b4d0836fb5500e20，local3tests通过；原artifact SHA860f170e8f1c2b47c0aac0cc50d5471ab21ec08e188ffe24d4df4adbb3f2cc5b匹配。actual readtransport exit0/reaped/非timeout，2.980137291s，stdout523B/stderr仅计321B。preflight actualexit1/reapedtrue/timeoutfalse/observation_error null/linux_group_closedtrue；nestedSTOP/phase lineage/predicate lineage/role null/errorclassValueError；owner null/owner_actual_exit null/owner_reapedfalse。acceptedforeground先核preflight非零再invokepipeline，故0七stage/model/ANN启动；具体lineage条件unknown。历史preflightgroupclosed不代表freshcurrentglobal状态，header/resource成功无证据。无其他文件读取/新attempt/retry/预算reset/refund。入口cache/p4f-foreground-receipt-read-v1/review.json。


### P4F local lineage telemetry validation

Fresh RED exit1 (missing install/precise telemetry), GREEN 4 tests PASS, final affected regression 6 tests PASS. Atomic producer rejection cases 22; additional core-run role, legacy membership conditions, stripped/frozen AST and main STOP checks included. Complete deployment 46 members, only control/readonly_diagnostic.py changed; local PENDING bootstrap actual exit1/STOP before writes. No real raw-input replay, remote/native execution or ANN metrics. Final packet SHA 042356e23dc0a42a22e1c451503699ff330ef09a500a707272666715b3c6eee6; evidence cache/p4f-lineage-local-revision-v1/review.json.


### P4F merged readonly proposal freeze

Proposed server total60s/local total90s/output8KiB/addressspace1GiB; original metadata total24MiB/core_run20MiB/default8MiB/diagnostic stdin32KiB retained. Input2460B SHA4d4060fd3edadcc3049ab15aa53cd2cb18d4eebc8814a0d8cc72e52b425f678e; accepted diagnostic14277B SHA8b7e8b391def5fa2725cb675cffc48b1da1dc164580c938570f96edd6968af85; bootstrap SHA6d5d2689955bb9dae14d7bc8af4960c067978c64782474120c0b3c7b936a6803. Only3 new bootstrap rejection cases passed (code/input hash, trailing frame), no repeated accepted suite/real input read/SSH/native/ANN. Proposal review SHA 549cc036a4e9612fd5081bea113f511f9220506c4e7f2c4630a5468b2e76aa55.


### P4F approved merged readonly actual STOP

Actual transport exit1/reapedtrue/no timeout; 3.9650934579999997s, stdout165B/stderr count321B. Precise phase lineage/predicate legacy_membership_sha/role legacy_audit/ValueError. Producer checks and preceding membership empty/cap/unique passed; header/resources not reached. Receipt SHA5e723861900c4a9c77ee93e6c897fb5d4ad40081d10e2fcf1a3e7c29d4aee2d7; review SHA427e84f2bf685feaacd74e59372fc5a4c5b6bff23f4249ba6f6049b6c02c4990. No model/ANN/newattempt/serverwrite/retry. Local producer/consumer newline serialization differs from three helper membership checks; corrected real hash not replayed.


### P4F membership serializer TDD patch

Fresh RED:5 tests,3 expected membership ValueError failures. GREEN:5 PASS. Final related regression:7 PASS, including three helper oracle paths, bare hash rejection, order/empty/unique/cap boundaries,512 valid membership,513 rejection, JSON type byte equivalence/nonfinite rejection and other-module AST identity. No old suite replay, SSH, real corrected hash or ANN metrics. Patch review SHA 0c15360cc10c7c7e8825e40230dd7207a6f17b8df3add1bd8a76bc371a1c38dd; actual receipt/review remain5e723861900c4a9c77ee93e6c897fb5d4ad40081d10e2fcf1a3e7c29d4aee2d7/427e84f2bf685feaacd74e59372fc5a4c5b6bff23f4249ba6f6049b6c02c4990.


### P4F fresh-attempt complete local integration

Only4 new integration/binding checks PASS; no old serializer7/lifecycle/full suite replay. Complete46 deployment members:3 accepted serializer controls differ; other5 controls/source/provenance/public manifest preserved. Proposed distinct budget7200s/RSS8GiB/disk4GiB/log16MiB/joint20%/CPU1/0GPU/1attempt0failure retries1planned infer resume; not actual authorization/ETA, overshoot possible. Core SHA 6a8d8df7d392d149879ae32a3daa9cccb1af9d11448926fec247cbce755749fa; refreeze bf45fff521f82fbeee5ad13bcc1cc77f036ab52a571e82f8a84a4d6ee0384e59; full review 46beb908d9777facfe856aa66612495b97fb631731fda2ed976097cb9e17c71e. No SSH/real permit/deploy/ANN/real corrected membership hash replay. First attempt stays spent; prior human budget approval does not cover second attempt.


### P4F second full attempt operational outcome

One exact submission to the new approved root. Actual transport exit 1, reaped; observation_timeout=false; elapsed_seconds=14.202884292. Returned foreground STOP / owner_actual_exit=null. No fusion, ANN or inference result accepted; no model comparison conclusion. Original start/final pins/foreground result SHA and receipt in cache/p4f-fresh-attempt-actual-v1; review SHA 28bf419db97743d829a996c81814dbbb3b7de49d75f7a8bdd0298d4b1aadeadf. Failure retry not performed.


### P4F second attempt stage clarification

Fixed hash-bound receipt read actual transport exit 0, reaped/non-timeout. Server receipt establishes preflight exit 1 / KeyError at raw_public_pair_or_context, reaped, timeout false, linux_group_closed true at the original wait. Pipeline owner absent; no fusion/ANN/inference results. Local schema RED: 5 tests, 4 KeyErrors; minimal helper GREEN: 5 pass. These are synthetic correctness checks, not model metrics or actual full-pipeline validation. Evidence local fix review ffda1e4a3d91600902a471c82ad1acd37bf3dad133c6e226f35881373f8b5ae3.

## P4F 第三 attempt 真实结果与 artifact/science 验收（2026-10-11，READY_FOR_FRESH_REVIEW）

第三 attempt 固定 root-v3；前两次失败及本次 attempt 全部保留 spent。Foreground 为 `OWNER_VERIFIED`，bind/export/fuse/index_audit/infer_first/infer_resume/verify 七 stage actual exit 全0，逐 receipt 的 result/proof/cleanup 与 Linux closure 已核。独立 acceptance 沿固定 receipts 重核7个 payload、4个 producer manifest、scientific binding、validation query、selected core config、export arrays、fusion、ANN 与全部 inference shards；没有重跑 P4F pipeline、训练或 test。证据 `cache/p4f-artifact-acceptance-v1/review.json` SHA256 `c227d3ab6c75bf632b5ebde550b51c3397fc75510494f4ce5c3c95e77885d437`；原 acceptance stdout SHA256 `790cb00d5ded8bd4f1d84b2628dd4ceafd3e33dacdf307ce666f6d2ee7049b02`。

固定 route 顺序：ItemCF / best-core TT / content / Pop。Validation all-eligible n=2520，其中 train-observed=1897、OOV=623；OOV 留在分母且各方法 OOV quality 为0。65个 canonical weight candidates 已逐 query 独立重算；选择 weights=`[2,1,1,2]`。

| 方法 | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| ItemCF 单路 | 0.005158730 | 0.009920635 | 0.032936508 | 0.002562333 | 0.003759938 | 0.008290344 |
| best-core TT 单路 | 0.010317460 | 0.016666667 | 0.037301587 | 0.004057622 | 0.005620679 | 0.009738907 |
| content 单路 | 0.003968254 | 0.008333333 | 0.014285714 | 0.001465749 | 0.002596253 | 0.003770076 |
| Pop 单路 | 0.023809524 | 0.032936508 | 0.050000000 | 0.009279010 | 0.011509207 | 0.014877363 |
| 四路等权 RRF | 0.015079365 | 0.029761905 | 0.060714286 | 0.006412923 | 0.010071381 | 0.016327042 |
| validation 选择 wRRF `[2,1,1,2]` | 0.016269841 | 0.036111111 | 0.063492063 | 0.007313685 | 0.012298012 | 0.017688008 |

Selected wRRF 的 train-observed slice：Recall@10/20/50=`0.021613073/0.047970480/0.084343701`，NDCG@10/20/50=`0.009715596/0.016336842/0.023496985`。Selected final top50 coverage=15,547 items（0.129111822）；等权=25,015（0.207739900）；四路top100 union=58,744（0.487846199）。

公平 quota 消融只列主选择指标；original 为剩余route各100，equal-total固定总400：

| 移除 route | original R@50 / NDCG@50 | equal-total R@50 / NDCG@50 |
|---|---:|---:|
| ItemCF | 0.053968254 / 0.017163823 | 0.055555556 / 0.017277385 |
| best-core TT | 0.055158730 / 0.016204424 | 0.055555556 / 0.016252920 |
| content | 0.061507937 / 0.017783095 | 0.063095238 / 0.018171395 |
| Pop | 0.037301587 / 0.010115986 | 0.040476190 / 0.010781194 |

ANN 仅固定validation audit 512 queries；exact top50 Recall@50=`0.029296875`、NDCG@50=`0.006325746`。Serialized HNSW M32/efConstruction200/index63,606,486 bytes，build22.779254s；独立用同saved exact top50、audit vectors和serialized index重算下表一致。模型效果 delta 与 ANN overlap 分列：

| efSearch | overlap@50 | ΔRecall@50 | ΔNDCG@50 | retrieval P50/P95 (s) | end-to-end P50/P95 (s) |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.949804688 | 0 | +0.000011315 | 0.000144840 / 0.000172396 | 0.000183947 / 0.000216700 |
| 128 | 0.989726562 | 0 | 0 | 0.000250912 / 0.000288641 | 0.000289542 / 0.000328620 |
| 256 | 0.999101562 | 0 | 0 | 0.000495922 / 0.000531360 | 0.000535553 / 0.000571692 |

最小通过0.98门为efSearch128；exact fallback明确保留。Exact per-query latency P50/P95=`0.013550953/0.013942772s`。RSS before build/after build/after index load=`734978048/798519296/788844544` bytes，只是process observation，不是host历史峰值或hard quota。

Offline infer 使用exact cached routes，不把ANN audit冒充infer backend。2520 queries按shard128生成20 shards；infer_first partial、infer_resume与verify均actual exit0；无漏、重、乱序，完整prediction replay与selected fusion metrics一致。Fallback count=0，unknown-ID count=623，result SHA256 `6d8a7fa861067d29b37788c1835e1f0fd917c9455b3d10e7f0543f5de0361b81`；逐query结果仅留server。Test gate仍closed，历史test暴露不称pristine。

验收脚本首轮使用逐项Fraction在180s deadline停止，零远端写；改为LCM(61..194)的等价整数分数，150个本地对照全部与Fraction同序，第二次40.269s完成。Whole-attempt实际wall仍UNKNOWN；仅上述组件自身计时可报告。

Fresh-context review verdict=`APPROVE`；report SHA256 `92459e7429291d50fb7bb87472c1ea6380fac604b5e255a62cdbbd0e771dd9f3` 由父 orchestrator 核验。P4F 正式接受并关闭。限制持续生效：ANN仅512-query validation audit、resources仅process-scoped、test26非pristine、无host-wide syscall trace、上游raw合同仅继承既有P3 hash。下一仅P4G pre-test freeze；test继续关闭，不根据未来test回调。
