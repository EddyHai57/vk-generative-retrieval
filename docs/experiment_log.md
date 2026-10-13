密级: 公开

## G3 core01/02/03 v2 exact admission 与启动（2026-10-11，RUNNING）

Fresh review APPROVE后，admission stdout SHA `5179b88352c846fa4351ac81a2b265beb7fdb66b926e390a029dd780f1e8a3c4` 闭合v1 terminal/spent、v2 root absence、unknown writer与joint20%资源。三job按 `eddy_delegated_existing_instance_finite_attempt` 各唯一submit，技术approval不冒充direct-human标签；旧v1与新v2 attempts均分别spent。

| run | config | start SHA | start UTC | deadline UTC | caps |
|---|---|---|---|---|---|
| core-01-seed42 | d128/LR0.001 | `dcae5bbf36ee0231b46fed91162c85772ea043e45eac3f091ef5ef448a017210` | 2026-10-11T09:59:02.430634Z | 2026-10-13T09:59:02.430634Z | 48h/20GiB/10200updates/153720decode/1attempt/0resume |
| core-02-seed42 | d64/LR0.0003 | `3780c19d0a67082af28869e3e567f2b00a7b79d6649e851058cb5fabe401c154` | 2026-10-11T09:59:06.912251Z | 2026-10-13T09:59:06.912251Z | 同上 |
| core-03-seed42 | d128/LR0.0003 | `5c84b86141ef7227d91c27f28353a1aa8f3b0d44a019e024a5a0582154a0f269` | 2026-10-11T09:59:11.993801Z | 2026-10-13T09:59:11.993801Z | 同上 |

三次transport exit0，post-submit owner均live；首个deterministic monitor status三者RUNNING。当前无新指标，不冻结winner；普通RUNNING不新增记录。完整approval/input/controller/payload/start/monitor证据见 `cache/g3-seed42-retry-admission-v1/`。

## G3 launcher准备（2026-10-08，无新增真实实验）

仅本地CPU synthetic infrastructure验证；固定outer链及CPU prepare/pilot restore/resource snapshot已核，不产生新增模型质量指标。原fixture仍最长history2、engineering pool6updates/8decode/1resume，不代表真实规定history50/d128/beam100或200reservation/128decode容量实测。实际server metadata、prepare、pilot尚未执行；CUDA资源/吞吐/full-validation与replay成本不得填估测值替代。正式6run/选择与后续3control训练均未开始。当前本地launcher的验证/包pins见docs/g3_validation.md和cache/g3-launcher-review-v1，source44与旧实验结果保持。

## G3 本地 CPU engineering（2026-10-08，非真实模型实验）

最终source44 SHA74e8361155145b5631ed8b61a365bd946a878f14e8d953bc5b341db608b35ca6，ignored cache/g3-final-acceptance-v2/receipt.json SHA82338c00b874d8fb396e7956adc9fee5f8e77e204d4c856fd339d0a67a0bfdc7。实际七CLI全部成功；连续/安全恢复/full replay的state SHA与已记录G2 d4aa162f…相同，item metrics与下方G2 synthetic摘要完全一致，不新增模型质量结论。

train-only CPU pilot实际6updates、8decode、1resume，最长history2；保留6条update timings、8条decode timings及serialized/accounting测量。pilot文件实际12172695bytes，global disk reserved12238421bytes（保守预留不退款），不是GPU50-history容量证明。engineering_selection_no_real_winner，formal_training_permitted=false。真实CUDA、可信完整validation成本、资源窗口及formal6run/9run预算仍待；不读取test。

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

## G2原创synthetic工程验证（2026-10-08，本地通过，待Astra）

固定core config0：d_model64/heads4/encoder-decoder各1层/FF256/dropout0/FP32 AdamW/LR.001/wd1e-4/clip1/effective512/seed42，microbatch2只用于原创小fixture；2epochs、2optimizer steps、单次resume、原60秒deadline。实际六CLI prepare→安全step1中断train→resume→evaluate→verify replay→select全部exit0，验证actual训练及逐query decode，不是模型质量实验。

item-level all_eligible count7，Recall@10/20/50均0.8571428571428571，NDCG@10/20/50均0.5248195074879939；in_catalog count6，Recall均1.0，NDCG均0.6122894254026595；OOV count1，Recall/NDCG均0。OOV保留在总体分母。选择best epoch1，第二epoch平分不换best。只执行一个synthetic config，select明确无core winner。

CPU latency observed P50=0.0013953326269984245s，P95=0.002062937291339039s；RSS observed lifetime high-water288931840 bytes，VRAM not_applicable。这些观察不外推真实数据/GPU吞吐。

公开机器摘要artifacts/g2-synthetic-manifest.json；忽略目录cache/g2-final-acceptance-v1的commands.jsonl/receipts.json保存真实命令和外部pins。run manifest SHAd592e1f6689990946c23b7c6171637db9b46df68f0fcad0e7c8ef36886e1ac95；state SHAd4aa162fa3c959b7eab10d0de73df57e69b2ec6b8bd948f7ef1a27e1a9a94ee9；item computation SHA22cd5bae7787ea5ff6f6063ef09dcee8c80049a3645452a2db0ea338ddee87cb。完整predictions/checkpoint只留ignored cache。真实P4E statistics pin与G3 pilot未放行，test关闭，无真实效果或CUDA一致性声明。
# G3 existing-evidence cost analysis v2 — 2026-10-09

仅已有真实pilot记录的本地成本审查，无新增模型训练/validation/test/prepare。aggregate SHA d5d5688b523a752d05e21c28d6eaa9abafdf9a890b1fb1126ed116713072a97e，actual-receipt SHA c417d53415dfc5e1fc4b351ab27cddfb9fff86d94ee3f483e67b641d56213279；transport actualexit0，读取5个绑定JSON共384269bytes，stdout6126bytes。原184updates/128decode/1resume保持。

N86785来自linked P4B prepared声明。aggregate Q仍null；预算算术引用已经accepted P4E collected.json SHA e800154a2b5562b2ffe6197927af8d43593f88c744c2cde0f07f130e26f76c2a中9run+3core all/count2520，外部review/acceptance及25source/summary/query9077...链fresh核吻合。这是继承的full validation分母，不冒充fresh G3/v2加载。每run最多5100formal+5100full replay updates、153720decode；G3六run61200/922320，G4三run未来规划30600/461160。

真实throughput每model80updates：d64 update mean2.8044441s/p952.85880034s/max2.96686850s；d128 mean2.91085724s/p952.95019753s/max2.96600654s。所有d128 pilot阶段update sample max3.44765337s。Throughput decode各48calls/history50：d64 mean0.19786496s/p950.38816721s/max0.39974826s；d128 mean0.20980543s/p950.39231725s/max0.41630696s。sample max/p95不称确定上界。

checkpoint两样本均6205191bytes，serialization write/fsync/hash最大0.10819121s；decode probe32rows共29216bytes/0.01214514s，不等正式shard schema。local ledger374records：bytes sum2692403/max7264，seconds sum4.39337229/max0.01928948；pool1190records：bytes sum1959916/max1820，seconds sum21.87513236/max0.03478655。accounting总26.26850464s仅诊断；checkpoint timer内部含pool.reserve，未来whole evaluate wall亦含query accounting，不重复加内部组件。

617samples parent GPU peak allocated1207114240/reserved1765801984bytes，parent maxrss1524162560、waited-child maxrss1603756032bytes；sample minima RAM free186007089152、VRAM free48731848704、disk free256552079360bytes。只在原采样/进程范围，不是continuous/future extrema或并发总峰值。closure output25194496bytes仅原closure scope；disk_bytes2147483648是cap，不是实际占用。

按30epoch跑满、继承Q2520且probe性能不变的**kernel+decode组件敏感性**：单run均值d64约16.39476h/d128约17.20612h；固定四seed42加selected两seed的G3总约99.99128h（selected64）或101.61400h（selected128）。不是完整wall/ETA/确定upper；row load/guard、binding增长、full validation写/聚合、context迁移和v2双scan/CPUselect仍unknown。formal wall/disk gate BLOCKED，未冻结runtime/permit。静态eval reservation13008896bytes/调用、v2 parent固定599269376bytes一次；reservation不等实际filesystem占用，不能将pilot最大row宽或25MB占用当formal上限。

唯一待审核成本audit提案详见cache/g3-formal-cost-review-v2/：一次5400s/新增1GiB硬投入cap、0updates/0resume、两model各一次full-Q beam100共5040decode，补端到端validation/guard/metadata/迁移/hash证据。不是获批窗口或完成保证，不沿用未批准3600s/4GiB方案；旧pilot128decode耗尽，新增查询/v2部署prepare须新的明确批准。当前只分析，未创建或运行该audit driver。

G3 v3启动前实际只读preflight（2026-10-10）：1/2次已用，transport exit1、非timeout，STOP runtime_origin/ValueError；未到资源阶段、0 submit、0新root创建、未开始真实组件测量。实际原始时长/bytes及脚本绑定见 cache/g3-cost-actual-v3/preflight-1-receipt.json；不将外层local脚本exit0称gate pass。真实phase/IO/full-validation成本仍unknown，无法据此冻结6run+3control预算。旧spent不退款；具体子gate未知，未retry。

G3实际preflight2（2026-10-10）：已用2/2 readonly，transport exit1/非timeout，gpu_uuid_equal=false；single GPU/name及origin/spec/contract pins、runtime schema通过，resource阶段未到。0submit/firstwrite/模型更新，正式6run+3control预算unknown。receipt/原始时长/bytes与allowlisted checks见 cache/g3-cost-actual-v3/diagnostic-v1/receipt.json。uuid不导出，不repin/retry；旧消耗保留。

G3独立UUID只读诊断（2026-10-10，截止后human单次授权）：actual transport exit0/非timeout；canonical_equal=true、raw_equal=false、single_gpu/name_equal=true，旧origin hash一致。实际receipt时长/bytes见 cache/g3-runtime-identity-diagnostic-v1/receipt.json；无训练/成本submit/firstwrite/serverwrite，formal预算未知。六小时窗口收尾未开v4。

G3 v4成本单attempt实际提交（2026-10-10）：preflight PASS；submit transport exit0，SUBMITTED，cost尚未通过。实际permit SHA 1ed014d2f809a344f8d84358cc9e1106bf54325a8b553d990baa1445fd4e5814，payload SHA ce4969668aae0e521711628bcb225558416bc8954fe49f8fde0ca13a70108840，original start SHA b88b2edaf0d7b1f468b7a68bb7bff97520b5c11f99fe23740107843d5d57cfe8；资源实际snapshot及时间/bytes见 cache/g3-cost-actual-v4/。0updates/0resume预算保持，计数须等终态，非正式训练。

## G3 fixed v4 实际组件成本终态（2026-10-10）

一次actual preflight PASS后唯一成本attempt完成，status-008 bounded raw SHA 57118eb8afe492837335d2b4b0263a036ca6a2b5239886360219357e8e1acd24；固定v4consumer实际exit0。原clock elapsed 2252.2240280141123s /5400s；parent actualexit0、worker actualwait0、owner process exit=null、owner_live=false、fresh owned groups closed=true。两model各2520queries，decode5040/5040，optimizer updates0/resumes0；完整epoch read86785examples/170batches。

| phase | actual seconds | rchar | wchar | read_bytes | write_bytes |
|---|---:|---:|---:|---:|---:|
|cleanup|0.00024331104941666126|129|0|0|0|
|evaluate128|1023.7831070821267|568257154|3161547|0|3260416|
|evaluate64|1032.6840519171674|568347339|3205462|86011904|3420160|
|load128|54.394148534862325|508865331|11254|0|12288|
|load64|55.004237664164975|509023412|11261|4366336|12288|
|prepare|54.68821272905916|508904002|10818|16384|8192|
|read_epoch|25.563938037957996|374514148|94|11157504|0|

Worker elapsed 2251.1557645159774s，unattributed 5.037825239589438s；7phase disjoint total核对通过。fresh final disk allocated 9760768B / logical 9189171B；parent observed peak 9752576B；1GiB observation STOP可能overshoot，不称hardquota。资源观察6697samples、最大gap0.37718156608752906s；没有全程minima导出，不以free snapshot冒充历史峰值。prepare/loads/inits=1/2/2，evaluation manifest/prepared/source/spec/runtime pins在execution_evidence。

固定E30/S170/Q2520下，每run含一次full replay的10200updates/153720decode只是注册工作量算术；G3 six core61200updates/922320decode，9run规划91800updates/1383480decode，不含独立beam diagnostics。

d64 conditional covered-components reference：10200次旧pilot timed update均值 +61次当前full-Q eval +60次当前epoch read = 93132.89326418366s（25.870248128939906h）；乘2为186265.78652836732s。这不是actual fullrun，不是上界、预算或ETA；pilot timer不含rows/guard/accounting，untrained eval不证明trained branch饱和。

d128 conditional covered-components reference：10200次旧pilot timed update均值 +61次当前full-Q eval +60次当前epoch read = 93675.34964165592s（26.020930456015535h）；乘2为187350.69928331184s。这不是actual fullrun，不是上界、预算或ETA；pilot timer不含rows/guard/accounting，untrained eval不证明trained branch饱和。

完整formal wall/disk仍unknown；checkpoint/long accounting/full replay/seal/hash/select/resume/branch/control/diagnostic等缺口见cache/g3-cost-actual-v4/cost-analysis.json。formal_cost_gate_passed=false/formal_training_permitted=false。组件交付READY待Astra fresh验收，未训练/读test/增加HPO，旧失败不退款。

## G3 operational spend 静态估计（2026-10-10，非新增测量）
main取消依赖缺pin checkpoint的CPU audit；不读checkpoint、不加载模型、不SSH。以冻结P4B catalog120415和P5 balanced16/depth3条件源不变量推导suffix≤30/V≤80，正式load仍须核实际codec。P=28d²+236d+(2d+1)V：d64 P140112、model+2AdamW moments1,681,344B；d128 P509520、6,114,240B，FP32、38parameter tensors，step scalar按304B留静态估计。cursor/RNG/ZIP/binding宽度未实际测量。

每checkpoint16MiB估计allowance×124模板；fullQ eval reservation61×13,008,896B；localledger20468×64KiB、pool215584×16KiB估计；parent静态seal allowance599,269,376B一次，child receipt另估，加trajectory/progress。枚举allowance总8981651456B（8.364814758300781GiB），不是完整字节上界或实际写入。旧journal samplemax按模板外推敏感性7894.239859299734s，非wall bound。

有限投入提案A首注册core-00-seed42：48h/原完整5100+5100updates/153720decode，pool12GiB、slotowned16GiB、共享4GiB，总新增20GiB观察STOP。B六core：每slot48h，总300h（6×48+共享12），61200updates/922320decode，总新增104GiB观察STOP（96slots+4shared+4outer）。规划未知风险开销由用户选择接受，不当测得值，不任意乘2，mayovershoot，不保证完成或ETA。钱价rate/null；不包含G4/test，当前proposal未批准，完整执行候选尚待实施/审查。全明细cache/g3-operational-spend-v1。


### G3 A首slot本地执行候选验证（2026-10-10）

review SHA `0c40c1e9c79fef77c773d57cee18901b7a6c14d24c3f60a29e89bfe193dfcbe0`，52成员；30项新候选相关final regression PASS，另已通过且未受后续controller记账改动影响的1项frozen synthetic pipeline，共31不同test。该pipeline为既有7-query CPU工程fixture：train2epoch、nested best、mandatory full numeric replay、final v2 consumer通过，封后best篡改被拒绝；不是真实训练或效果指标。fresh RED/GREEN log、全部argv/解释器及provisional fixture失败见validation/member文件，source44/原v4 lifecycle/helper SHA不变。

实际新SSH/deploy/真实输入训练/CPU audit次数0。首slot仍为core-00-seed42的48h/20GiB finite-spend提案，12GiB pool/16GiB slot/4GiB shared/20GiB total observation STOP、末1800s cleanup、至少20%全局余量；不是ETA/hard quota/保证。选择0自动resume，不超最多1；limits schema1、actual pool0。费率/账单未知，approval及真实现场clock/resource均pending，不因本地fixture通过变成formal许可。


### G3 status-only v2修订验证（2026-10-10）

review SHA `875f096ee1855381c9f9b3091cb24f61895add660720f602a8adbee92df8caa2`，58成员；Python3.11 final affected regression6/6通过：3项new actual status bootstrap反例+3项transport。fresh RED原记录证明wrongSHA仍执行代码/unsafe symlink与oversize执行/FIFO5s timeout/直接文件不调用status接口；GREEN bounded bootstrap先核path/size/SHA后exec buffer，fixed3pins正确，输出与status同30s deadline。旧v1 52成员与server6核心/input/approval/payload SHA重核一致；source44保护不变。原main已fresh通过的30项+7query单项本轮不再运行，0新server/真实训练，48h/20GiB仍未批准、费率unknown。原reviewer差异验收pending。


G3 formal-A-v2独立验收接受（2026-10-10）：main原reviewer确认58members、fixed六成员/source44/旧包保护及status3+transport3 fresh PASS，唯一Important Closed、无新增blocker。未重复任何suite/工程pipeline/验收。实际48h first-write总clock（末1800s仅收尾）/20GiB observation STOP（pool12/slot16/shared4）/10200updates/153720decode/1attempt/0resume无retry的human支出许可仍pending；fee rate unknown，非ETA/账单承诺。joint余量各20%、持久产物保留、B/G4/select/test/newresources/delete关闭。此记录仍是0新server/真实训练，现场clock/resource/envelope SHA待直接human批准后的绑定，不编造。


## G3 formal-A 单次实际投入（2026-10-10）

Eddy本G chat直接批准core-00-seed42：48h总cap含部署/full validation/replay/cleanup；20GiB新disk观察STOP may overshoot；联合20%余量；max10200updates/153720decode/0resume/1attempt无retry。固定v2成员/source44 readonly复核通过。materialization只approved与600s窗口及派生SHA。submit transport实际22.038s/exit0/non-timeout/392stdout bytes，返回SUBMITTED。status-001实际5.642s/exit0/non-timeout/62stdout bytes，UNKNOWN。无可信epoch/update/metric/训练阶段或完成结论，费率/账单unknown；actual receipt为唯一证据，禁止重置预算。


status-002实际9.372s/exit0/non-timeout，UNKNOWN；本地fresh对应固定lifecycle70414bytes默认65536边界复现regular_bounded。此为observer defect证据，不是训练结果或server阶段证据。


status-003使用main已审observer-v1实际6.161s/exit0/non-timeout/62stdout bytes，RUNNING。只确认原owner identity活着，不是当前epoch/updates/指标或completion证据。原48h clock/预算继续，单attempt/0resume保持。


2026-10-10 16:26:57（Asia/Shanghai）status-004 actual3.421s/exit0/non-timeout/62stdout bytes，RUNNING。仅owner alive，非stage/metric/completion证据。

## G3 core-00-seed42 首slot终态指标（2026-10-11）

固定train manifest SHA `9e7bd1a773aed03b1456e2e4dd5d3d35d4a73dc8a1d838228d638852d1bb728e` 与best-evaluation manifest SHA `43601fd3d31d34a80ffc2924a899bd1c57cd1dd23db566af870c8aef82cea9a3`经bounded nofollow读取并重算吻合。配置d_model64、LR0.001、seed42；14 epochs完成，best epoch11，2380steps、0resume。完整数值replay、consumer seal及test gate终态证据见固定terminal receipt；本节只保存指标摘要。

| Slice | n | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all eligible | 2520 | 0.016666666667 | 0.028174603175 | 0.052380952381 | 0.008363156169 | 0.011259949692 | 0.015990650421 |
| in catalog | 1897 | 0.022140221402 | 0.037427517132 | 0.069583552978 | 0.011109727753 | 0.014957866749 | 0.021242192442 |
| OOV | 623 | 0 | 0 | 0 | 0 | 0 | 0 |

每query固定100候选，2520/2520完成，总252000候选，unique4009，catalog coverage0.033293194369。best evaluation观察latency P50=0.092935340130s、P95=0.096316252893s；进程lifetime high-water RSS1267458048bytes；结束时VRAM allocated36301824bytes，不是peak。原始逐query结果、checkpoint与完整artifact仍只留服务器。

这是one-seed、四个预注册配置中的第一个slot，不能冻结core winner或声称统计胜出。OOV全零是train catalog结构性不可达；工程终态不等于指标获胜。读取收据在ignored `cache/g3-metrics-read-v1/receipt.json`，reader SHA `9cebb05f1c55a6b33ed5f1fbda24908e8ebdcf84469db597d830e9f5da09b828`。


## PG-R01 — G四配置真实产物接收（2026-10-12）

原四个seed42作业本次均完成外部终态/train/best/verification/invocation hash、source44/原helper/日志/best checkpoint、分母与selection交叉绑定，并fresh核封存文件集合及全部成员hash。只读stdlib检查，无新训练、数值replay、test读取或服务器写入；原独立full replay成功收据被接收，不重新做一次训练式replay。接收时点与证据在Main ignored `cache/pg-r01-reception-v1/`，single fresh接收review已APPROVE；证据Main cache/pg-r01-reception-v1/fresh-review.md。

共同catalog/examples/SID/query/prepared SHA四配置一致；validation all-eligible2520、in-catalog1897、OOV623。OOV保留分母且各质量指标0。四配置只代表固定seed42 screen，不构成稳定收益或统计胜出。

| Run | d | LR | epochs / best | R@10 | R@20 | R@50 | N@10 | N@20 | N@50 | 总作业wall(h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core-00-seed42 | 64 | 0.001 | 14 / 11 | 0.016666667 | 0.028174603 | 0.052380952 | 0.008363156 | 0.011259950 | 0.015990650 | 10.205917 |
| core-01-seed42 | 128 | 0.001 | 8 / 5 | 0.017857143 | 0.028174603 | 0.049206349 | 0.008788474 | 0.011284676 | 0.015434179 | 5.900487 |
| core-02-seed42 | 64 | 0.0003 | 19 / 16 | 0.016269841 | 0.025396825 | 0.048412698 | 0.008346404 | 0.010620424 | 0.015201899 | 13.731125 |
| core-03-seed42 | 128 | 0.0003 | 14 / 11 | 0.023809524 | 0.035714286 | 0.053968254 | 0.011766157 | 0.014755980 | 0.018366511 | 10.126605 |

| Run | in-catalog R@50 / N@50 | 实际更新 | 纯train(min) | epoch验证(h) | 解码P50/P95(ms) | observed disk peak(bytes) |
|---|---:|---:|---:|---:|---:|---:|
| core-00-seed42 | 0.069583553 / 0.021242192 | 2380 | 20.651036 | 4.345577 | 92.935 / 96.316 | 772947968 |
| core-01-seed42 | 0.065366368 / 0.020502968 | 1360 | 12.422121 | 2.438077 | 93.992 / 99.951 | 591093760 |
| core-02-seed42 | 0.064312072 / 0.020194404 | 3230 | 28.477341 | 5.870102 | 93.367 / 97.186 | 1050525696 |
| core-03-seed42 | 0.071692145 / 0.024398318 | 2380 | 21.956248 | 4.308650 | 93.476 / 97.845 | 1023680512 |

总wall包含原部署/验证/replay/收尾；原core01/02/03并发，不能将三者wall相加当物理GPU账单。decoder latency定义是per-query tokenization+encoder+four-position beam，不是batch吞吐或峰值资源。

按预注册all-eligible R@50→N@50→枚举tie rule，选core03（d128/LR0.0003）固定代表seed42，补充seed2024/2025尚未运行。local selection在Main cache/pg-r01-reception-v1/selection.json；本阶段fresh review已APPROVE；selection仍不构成新submit permit。core03对core00 R@50仅多4个命中，不能据此宣称显著胜出。与已接受P4F单路Pop相比多10个命中，与最佳传统id_mean相比多42个；这些是不同features的validation参考，非等compute因果比较。wRRF的R@50仍高于该G单路，不能声称超越传统融合。

| Run | train SHA | best evaluation SHA | verification SHA | invocation SHA |
|---|---|---|---|---|
| core-00-seed42 | 9e7bd1a773aed03b1456e2e4dd5d3d35d4a73dc8a1d838228d638852d1bb728e | 43601fd3d31d34a80ffc2924a899bd1c57cd1dd23db566af870c8aef82cea9a3 | 5a72eec624a1f24fa1d4c569069cdb9cee0878fad60363d6a73e7c82c0d79fce | 2398c6a0038523bdbdf971c517a9a057e59f94c53accc971a682ac9bc0414b66 |
| core-01-seed42 | d0a35893e3106f95d6ff3065b57366e8ee87e083caef027c48aad379fcaaf36c | 519192ffbe60c14343f39dd3171db8a25b148a4757ee628946252d154f983071 | cebaffd211ebb0d922a35af69110685d1e6f8f099fecbae388c80793166a686e | 184c889302cca8946a3eb4b52d2e4a1a36b970e5d833882764d8f9f192d94622 |
| core-02-seed42 | 7c71dcbe8d524fe08746be045a91c273eabbc791081f4509425254014c22f7c8 | 630bdbb508f1749badb17b2e91c33afbbc3f1f28dfbb81bb19c1594232575767 | d2ca05de092d9dd6d8fa160d5a748552f714f5edc84efe38d7f91743249138f3 | 2aa479b738bf6849775556b66b3ce9c079613634ed11f2bf332e8caa2447f71d |
| core-03-seed42 | 69659242b76d8d58f6379db7bb368bc45978a18d27aec0144bd2e501d610086b | 44f4bc405e2605bf27e3c0ce3e9d10350ba1dc22d42226074f4064654d2c4064 | 806ec46099681f5bf34f1de7c370c8d02ecdcce2a3aef70191ce407534e59277 | f070f75964bad8468d1d0b6cb9d5a3d910ab3726fab67855af717f9018eed540 |

本次local接收不是对全部原轨迹的再次数值验算；原replay已有实际wait/exit0及verified/sealed证据。新补seed、G4与test仍遵守原准入和预算门。
