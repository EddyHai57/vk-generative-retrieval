# 训练排程与资源（唯一事实源：排程与资源）

密级：公开。writer：Codex（2026-10-15 Eddy直接授权；关键决定交Kiro复查）。指标只在 `docs/experiment_log.md`；本文只放容量、成本、规则和当前排程快照。
工具：`ops/tools/server/sched_forecast.py`（分支 infra/04-sched-forecast，口径见 `ops/SCHEDULING.md`）。数字以工具实跑为准，历史成本是 2026-10-14 夜里的实测快照；最新排程见第4/4a节。

## 1. 容量模型（实测）

- CPU：cgroup 配额 10.24 核（不是宿主机的 64 核）。调度器准入留 20% 余量，所以可用 8.19 核，同时约 8 个单核作业。
- 显存 49140 MiB，准入上限 39312 MiB（按**声明**算，不按实测算）；RAM 约 200 GB；磁盘 500 GB。
- 所有作业都是单核 CPU 瓶颈，所以限制吞吐的是**核位数**和**显存声明**，不是 GPU 算力。

## 2. 作业类型与实测成本（单核）

| 类型 | 训练墙钟 | verify 墙钟 | 显存声明 / 实测 | GPU 时间占比（pmon 实测） |
|---|---:|---:|---|---|
| P 线（id_mean，各特征变体） | 10–16 分钟 | 4–6 分钟 | 3072 / 约 640 MiB | 约 1% |
| 小 G（core03 规模） | 约 1.5 小时（最长 2.3 小时） | 约 7 分钟 | 6144（新作业改 3072）/ 约 2050 MiB | 约 5% |
| 论文规模 G（A1） | 约 2.7 小时（seed42，与另一个同跑） | 约 9 分钟 | 6656 / 约 5690 MiB | 训练约 21%，评估约 13–15% |
| tokenizer 拟合 | UNKNOWN（M1 单线程 9 ms/步，服务器未测） | — | 3072（RQ-VAE）/ 0（k-means） | UNKNOWN |

GPU 占用的预测规则：同一时刻 GPU 轮流服务各作业，总占用约等于各作业占比之和，超过约 90% 后互相拖慢。5 个 A1 同跑约 75–105%；8 个小 G 全开约 40%；P 作业几乎不占 GPU。显存声明：5 个 A1 = 33.3 GB，在 39.3 GB 内。

## 3. 流水线规则

1. 一批作业入队的同时，下一批的代码和 review 必须已经开始。入队那一刻没有“下一批在写”就是排程失误。
2. 就绪队列（代码已 review、卡已冻结、随时可入队）加服务器剩余活量，目标 ≥ 8 作业-小时；低于 4 小时时工具首行输出 `BACKLOG_LOW`，先从第 5 节取填空作业。
3. 显存声明 ≈ 实测峰值 × 1.15，向上取整到 256 MiB。**2026-10-15 更新**：A1：6656；小 G（core03）：2560–3072；P 作业：**1024**（实测约 640，旧写法 3072 在 10-15 凌晨让 6 个小作业被 G 线大作业占满的额度挡了 1 个多小时）；tokenizer 拟合：768（实测 536）。声明偏大会白占准入额度，偏小有 OOM 风险；只改新入队的，不改正在跑的。队列没有改声明/撤回的工具，别手改 pending 的 JSON。
4. 每个“训练”作业必须带它的 verify；批量解码训练的 run 必须用逐条解码 verify（INFRA-02）。**入队前核对 verify 是否已经有人入队**：2026-10-14 A1 seed42 的 verify 因轮询步骤卡住漏了一次，挡住了三个后续任务的入队门槛。
5. 填空作业不计名额、不改任何已有结论，且必须先在 decision_log 登记，结果只进“方差/稳定性”一栏。

> **交接（2026-10-15 16:3x）**：本文件之后由 Codex 维护。最新队列以 `kiro_b_vkgr_status_read.py` 实测为准；16:27 的读数见 STATE.md 末节（在跑 5 个 G 作业 + 2 个 P 作业 php02，pending 4，CPU 用 7.03/10.24 核）。下面第 4 节是 15:21 的快照，未再更新。

## 历史状态覆盖说明（2026-10-16 10:56；现已由§4/§11覆盖）
M1十五训练/十五verify、GTOK六补seed训练/六verify已晨间全部验收，原10:21项目空档快照是历史。**当前FEAT04-COMBO三seed全claimed/实际running、pending0、无rejected，服务器恢复科学训练**；服务器verify驱动2921056 alive/当前PPID1，900s/48h/no LLM。最新source为experiment_log FEAT04-COMBO节C1/H1及`cache/codex-feat04combo-20261016/receipt-index.json`（stdout路径/SHA/实际UTC）。三个对应verify仍planned，未验收。下方夜间§4/4a旧表保留过程，不代替当前快照；不声称有限批次可无限覆盖后续空档。

M1本批实测（15样本，M1-R2 terminal）：train最小/中位/最大 8.105/13.366/15.550min；verify 3.779/5.929/6.500min。GTOK补seed六样本（G-R1）：train 97.005/129.650/155.588min，verify11.033/13.494/13.723min。均为本批成本回溯，非下一批ETA；身份/逐项wall秒见experiment_log和收据。

## 4. 当前快照（2026-10-16 23:11+08 S1执行；覆盖下方历史表）

**2026-10-17晚间覆盖**：Kiro深夜已裁决用户画像特征（F7）两臂第4行、不转正、收口，P anchor=M1-mean；原六训练/verify及一次授权只读诊断已全部验收。无F7后继/seed/字段追加。G5冻结卡草稿、S2立卡草稿已交，仍待原约定Kiro冻结/设计与Eddy名额/读取授权，不因本次扩大权限或入队。诊断机制措辞待Kiro，数值/SHA/UTC仅experiment_log，结论见issue。

**10月17日02:24最新实测**：共现向量 SID（C1′）和内容加 P 向量 SID（C2）的s42均claimed且全局running，pending0、无rejected；两路CUDA原始cuda_engineering_verified/dry_run_ok已验收，原观察接续、截止不变。数字与完整收据SHA/实际UTC只见experiment_log“两路正式训练接纳”节；训练结果/verify尚未验收，条件补seed未触发。

**10月17日02:20当前：协同向量 SID 两条容量元数据修正已获Eddy批准并完成。** 共现向量 SID（C1′）CUDA已claimed，内容加 P 向量 SID（C2）CUDA仍pending，均无rejected；新观察2931675实际PPID1/alive，原48h截止不变。两路真实preflight已通过，CUDA/测峰/dry/正式训练尚未验收。容量39168是工程探针预约，训练仍按实测×1.15 ceil256。收据与SHA/实际UTC见experiment_log容量接续节。

**10月17日02:11历史，覆盖01:20阻塞状态：两次获批preflight-v2已通过，六配置真实PF验收完成。** 原向量/SID成功阶段未重训。新2930845驱动存活/实际PPID1，CUDA两个作业已pending但尚未启动：39312MiB容量探针恰在理论80%边界，GPU已用1MiB使现有20%余量准入拒绝。实测下取整39168MiB可准入；不是修改门禁，也不是最终训练显存声明。已备83tests通过的容量元数据修正及只读旧/新SHA，等待Eddy对两条既有queue JSON覆盖许可。

在02:11时无G训练或CUDA实测；该历史时点等待许可期间不更改pending、不另ID重投、不改20%余量。授权后仅两容量字段修正，新增匹配计划并更换观察读取新plan；旧plan/失败记录/原截止保留。原始收据main05/sidcoll-capacity-proposal-20261016T181131Z.stdout，SHA2551fc942404a31bea3459f3494fffbc06611cc8ef983b4e90a323178d2d92db，实际UTC18:11:33.635935Z，详见experiment_log。

**2026-10-17 01:20历史，覆盖本节旧表：当前无科学running/pending，自动可执行BACKLOG=0。** C1/C2 input与SID fit/replay已直接verified；两G preflight均因codec.depth接口错误FAILED，driver按ALERT停止。GPU空闲/CPU近空，距最后失败约1h20已报Eddy。工程修复0f39d69已76tests通过，未部署/重投；须Eddy批准两配置各一次工程重验。复用已验收产物，不重拟合。原48h截止是上限不是可跑活量。

恢复后的历史量级估算：两路s42+verify并行约2–3h；若均达260，四个补seed+verify并行再约2–3h，总约4–6h，均以新VRAM准入允许并行为前提。新SID的实际early-stop时长及工程重验耗时UNKNOWN，不能保证整夜无空档。S2/S5/G5未READY、不得自动补位。原始source/SHA/UTC及实测成本见experiment_log「2026-10-17 01:20 S1 progress」。

当前S1 FINAL已经实现/review/冻结/push，新科学部署079d6bd、观察1fa96dc。C1′SGNS与C2 SID fit已各自claimed，无rejected；前者共现向量拟合、后者已完成导出和input_verify，尚未进入正式G训练。当前CPU前置是瓶颈，GPU仍空闲；不为填空新增S2/S5或改门禁。资源数值与读取UTC/SHA只见experiment_log S1工程节及receipt-index-v1。

| 当前批次 | 实际状态 | 接续 |
|---|---|---|
| C1′ input_preflight | succeeded，原始verified | SGNS input已claimed，后续input_verify→SID fit/replay |
| C2 input_preflight / export / input_verify | succeeded，验收直接读到verified | SID fit进程已succeeded，full replay verify已claimed；最终fit验收仍待 |
| 两输入各自三配置G PF / CUDA / dry | 尚未验收 | 每支独立通过后自动s42；新训练VRAM待各自真实shape实测，不沿用旧值 |
| S1 s42及条件补seed | 尚未提交 | s42 raw verified且hits≥260才补2024/2025；最多六训练，各配唯一verify |
| S0 D′补充 | succeeded/result已只读验收 | 原D不改，结构无定义记STRUCT_NA |
| 四旧别名/A1 epoch原辅助job | FAILED attempt保留、不重投 | 后以固定小manifest只读完成曲线；四旧仍UNKNOWN |

观察2927689当前alive/实际PPID1、事件own_terminal已实证、900s兜底、无模型；原deadline和所有intent保留。CPU阶段声明1核/8192MiB RAM/0VRAM；GPU测峰预演只用准入容量上限暂作独占预演，不能当训练实测或训练声明。最终声明仅取各自测峰×1.15向上取整256。

**以下为先前S0/M1历史快照，不能当当前running：**
**17:50+08最终收据**：M2/G原科学批次raw verified已齐，完整结果解读仍待；S0两个单核诊断均succeeded/exit0，独立claimed/no rejected，A–D/所有来源已进experiment_log，4旧归档的B/C保留UNKNOWN。S1草稿待Kiro冻结，S2–S5未立卡，当前未安排新的已冻结训练。第一批与补批实际身份/成本/所有SHA/UTC见experiment_log S0最终节与cache/codex-sid-fix-20261016/receipt-index-delivery.json。下方M1夜间表与旧容量只保留历史，不能当当前running。

来源：experiment_log“2026-10-16 M1夜间正式批次”DEP1/PF1/D1/S1/C1/H1/T1，及`cache/codex-m1-night-20261016/receipt-index-v2.json`；每个stdout SHA与实际UTC在该节/索引。Opus复核已通过，不再保留“待Opus才能入队”的当前阻塞。

| 批次 | 已核实状态 | owner / 接续 |
|---|---|---|
| M1 mean ×42/2024/2025 | 三个全部claimed/running、无rejected（C1）；尚未science verified | Codex提交；每个1核/9216 RAM/1024 VRAM/8 GiB disk/8h cap |
| M1 last/attn/bi/pure ×3 | 十二个诊断配置真实preflight全通过，未提交（C1） | dispatcher2916804在≥1 mean训练succeeded且verify直接verified后固定入队；任何已观测mean失败/终态未verified写ALERT并停止接续 |
| M1所有对应verify | 固定计划15个，成功训练后自动补 | verifier/reporter2916803独立收口；1核/9216 RAM/1024 VRAM/1 GiB disk/3h cap；不重试失败 |
| GTOK原P1/P2六补seed | 沿原driver，未重提；H1当时仅B256D2-s2025仍训练、s2024-verify运行，B1024两verify pending | 原2910284/P2、2911779/P1；status没列的其它run不据此反推science verified |
| FEAT-04组合 | 内容已由Kiro登记F4-T010+C+D+E、三seed；未实现/未入队 | 下一批，不能填进今晚队列 |
| 已结束P/G批次 | Kiro夜间§3已裁决，G anchor=A1+T3，P仍F4-T010 | 数字只见experiment_log，不新开确认run或改旧结果 |

H1工具实测：GPU空闲43997/49140 MiB、RAM163877/202084 MiB、CPU5.07/10.24核、磁盘213.2/500 GiB；三个mean实际GPU进程观测各884 MiB（不是全程峰值），声明1024保持。科学worker五个（3 M1+1 G训练+1 Gverify），另scheduler服务；pending两个Gverify。该时点后状态以新收据为准。

T1：tok2 train已有11样本、中位1h55、最大2h21；verify8样本、中位12min、最大13min。M1无已结束实测，工具显示wall_cap，**8h/3h是cap而非ETA**。正常CPU可用约8.19核、原admission保留；诊断提交后仍按资源逐项放行，不能保证瞬时无等待或十五训练明早全结束。

## 4a. 接续边界与验收（2026-10-16 S1执行）

**2026-10-17晚间当前覆盖**：Kiro深夜已裁决用户画像特征（F7）两臂第4行、不转正、收口，P anchor=M1-mean；原六训练/verify及一次授权只读诊断已全部验收。无F7后继/seed/字段追加。G5冻结卡草稿、S2立卡草稿已交，仍待原约定Kiro冻结/设计与Eddy名额/读取授权，不因本次扩大权限或入队。诊断机制措辞待Kiro，数值/SHA/UTC仅experiment_log，结论见issue。

**10月17日02:24最新实测**：共现向量 SID（C1′）和内容加 P 向量 SID（C2）的s42均claimed且全局running，pending0、无rejected；两路CUDA原始cuda_engineering_verified/dry_run_ok已验收，原观察接续、截止不变。数字与完整收据SHA/实际UTC只见experiment_log“两路正式训练接纳”节；训练结果/verify尚未验收，条件补seed未触发。

**02:20当前覆盖：元数据修正获Eddy批准并完成，新plan b14630f、新观察2931675已独立RO确认；共现向量 SID CUDA已verified/dry接续，内容加 P 向量 SID CUDA已claimed。后续仍按原冻结顺序与阈值，不重投。实际数字/收据见experiment_log。**

**02:11历史更新**：preflight失败例外已获批、已各消耗一次且通过；当前是never-started CUDA pending容量元数据许可，不是申请第二次preflight。原cuda/dry/train/verify run_id不变，sid/cfg/data hash不变，模型11d6768部署不改。待准许两queue字段后继续同一批，不新增科学臂或seed。

**10月17日01:20历史，已被获批恢复覆盖：接续曾被sticky ALERT阻断。** 不restart原controller、不重投原preflight、不换ID规避失败规则；等待Eddy对一次工程重验的明确许可。后续s42与条件补seed的科学授权仍在，但不解除工程停止条件。

当前固定两支各14阶段，原plan SHA0f5c1029281ec7c65266a4c5ea802425e0ccede8298bb931b020d3ec48c7161a。先真实只读input PF，再向量生产/审计/fit/replay/三配置G PF/CUDA/测峰/dry，之后允许s42；唯一train成功补唯一verify，筛选只看原始verified+绑定和整数hits。任一步失败停止本支并ALERT，另一支继续；共同合同失效停两支；rejected读reason不重投，失败attempt不改ID重跑/不覆盖。没有LLM轮询或新增timer。所有实际身份/48h截止见handoff，不重置预算。

当前已告知Eddy门禁全过后自动入队两个s42，尚未把未来条件作业称入队成功。后继S2/S5不在自动接续范围，G anchor仍A1+T3。

**以下“当前”为先前S0收尾/M1历史记录，已被本节覆盖：**
**当前**：S0两批均终态成功并记账；M2/G原科学train/verify已raw verified，后续完整解读仍待。S1只是DRAFT，须Kiro冻结后才实现与准入；S2–S5未立卡，无无限READY队列。没有新增timer/长期模型轮询。S0终态使用有限Linux文件事件等待，观察超时不等于失败；所有旧job不重提。

**以下为M1夜间历史接续过程**（角色、PID、running均是当时快照，不代表当前）：

采用**主方案A：verified金丝雀**，不是备选十五项一次性入队。服务器`/workspace/vkgr-seqm1-night-20261016/`核心ops6a8e057，模型0ca61bc；两个role均实测alive/PPID1，900s，无LLM，plan创建起48h有限截止。dispatcher只做接续，遇mean失败写ALERT停；verifier继续验收已经启动的成功作业，不漏verify。rejected读.reason、不重投，不重跑或覆盖失败attempt。

| 下一步 | 执行载体 | 验收 / stop gate |
|---|---|---|
| mean训练→verify | 原版本化M1 verifier | 训练succeeded后自动提交对应verify，按claimed/rejected记录接纳 |
| ≥1 mean直接verified→十二诊断固定入队 | dispatcher | 检查全部mean已观测状态，无已知失败才放行，不看R/N选臂/seed |
| 诊断训练→各verify | verifier/reporter | 每个独立记录；失败/拒绝不重跑、不覆盖 |
| 批次终态汇总 | 两角色journal及summary-final/deadline | 每run状态/身份/数据hash/epoch/R/N/ablation/ratio/异常/未验收；不以driver done代替verified |
| Eddy醒来后验收 | 当前Codex对话只读取回汇总 | STATE按已完成/仍运行/失败/未验收更新；用户明确不建晨间timer |

当前没有新automation，自动审批拒绝一次创建请求后，Eddy明确选择不建定时；旧automations仍暂停。其它GTOK/A1确认、G-COLL、F7、M2与G5均按Kiro夜间§4–5保留各自前置，不由夜间driver越界启动。

## 4b. 教训（2026-10-15，写进规则）
1. **不要把互相独立的实验绑进同一个工作流**：FEAT-04 的 A、B（只是合并现有代码）被 C、D、E（要新设计）拖着，导致 P 线整晚没作业。能先跑的先拆出去。
2. **服务器空闲超过 1 小时要主动报告并立刻派填空作业**（第 5 节）。工作流设计/开发的几个小时里，GPU 不该空着。
3. 入队写入后必须确认进了 claimed、不在 rejected；`action=written` 不算成功（issue_log I-2026-10-15-a）。
4. 新入口第一次上真实数据前，必须有服务器真实输入的只读 preflight（issue_log I-2026-10-15-c）。
5. 调度者每次读这份文件后，用 `kiro_b_forecast_read.py` 与 `kiro_b_vkgr_status_read.py` 对一下实际队列，数字以实测为准。
## 5. 填空作业清单（需 decision_log 预登记后才能入队）

| # | 作业 | 价值 | 状态 |
|---|---|---|---|
| F1 | tokenizer 拟合 + 审计（GTOK-01 T1–T3） | 缩短关键路径 | 已完成（T1/T2 首次失败，T1r/T2r/T3 通过） |
| F2 | 深度 4 tokenizer 拟合 + 审计（G6-PREP） | 同上 | 等实现与 review |
| F3 | A1 补 seed 的 verify | 补 seed 转为已验收 | 已完成 |
| F7 | T1r 补 seed 2024/2025（core03） | BHKM vs RQ-VAE 三 seed 对比 | 已入队，在跑（15:21 实测，约 16:45 出结果） |
| F8 | P 线温度精扫 0.05/0.07（用 P-HP-01 作废的第 7 个名额） | 确认最优温度 | 已登记（decision_log 2026-10-15 15:2x），工作流 wf_8b5f7ba0b7e3e2c6 设计中 |
| F4 | F4 anchor 额外 seed 只报方差 | 给 ≥6 命中门槛一个参照 | 未登记，**不跑** |
| F5 | 小 G 的 F3 特征额外 seed | 判断 F3 是否噪声 | 未登记，**不跑** |
| F6 | A1 的 beam 宽度诊断 | 看 beam 是否仍不是瓶颈 | A1 入口不支持该子命令（需另写代码），**暂缓** |

F4、F5 属于“多跑几次挑好的”的高风险项，没有预登记不得入队。

## 6. 工具用法与已知偏差

- 在服务器上只读运行：`python -I <部署目录>/ops/tools/server/sched_forecast.py --root <ops 根目录>`；`--json` 可选。
- 作业类型按“入口 + 子命令”分，**不区分配置**：A1 的 seed42 与 A2 学习率不同，墙钟差一倍，只有 2 个 run 时中位数是两者的平均。估 A1 时间按 A1 seed42 的 2.7 小时。
- 工具的 GPU 利用率采样只有 5 秒，不如 pmon 的 30 秒窗口稳；以后看 GPU 占比用 pmon。

## 7. 全量阶段如何复用

换大盘机器后只需重新读容量（第 1 节三项），成本表从新 run 的 terminal.json 自动归纳，不需要手改；第 3 节规则不变。全量阶段每个作业会大得多，先用一个吞吐 pilot 测出单作业墙钟与显存，再填第 2 节。

## 8. Kiro复查接收（2026-10-15 16:46+08）
来源：Eddy本轮转交与decision_log 16:5x第3–4条。温度精扫workflow完成，后三个workflow仍由Kiro交付；不将workflow完成视作所有verify已验收。旧600s驱动保留不重启，新驱动≥900s且先版本化，旧脚本在作业收口后补入Git（issue_log I-2026-10-15-h）。本轮只更新本地记录，没有追加服务器查询、新作业或新timer。

## 9. 训练安排表（前瞻；Kiro 2026-10-16 00:1x 编写，之后由 Codex 维护）

**2026-10-17晚间当前授权覆盖**：Kiro深夜已裁决用户画像特征（F7）两臂第4行、不转正、收口，P anchor=M1-mean；原六训练/verify及一次授权只读诊断已全部验收。无F7后继/seed/字段追加。G5冻结卡草稿、S2立卡草稿已交，仍待原约定Kiro冻结/设计与Eddy名额/读取授权，不因本次扩大权限或入队。诊断机制措辞待Kiro，数值/SHA/UTC仅experiment_log，结论见issue。

**2026-10-17 03:00+08上午授权覆盖更新**：协同向量 SID 原作业/自动补seed/driver不动；双向序列模型确认新seed无碰撞后实现，正式六训练；序列用户塔+组合叠加三训练实现并行，入队待定点泄漏review APPROVED（当前CHANGES_REQUESTED）。温度交叉熵（R1）官方τ默认退化为原CE，暂停提交等待Kiro澄清；自生成难负样本DPO（R2）仅成本前置工程实现中。批次各自就绪各自入队，不互等；8.19核/39168MiB，queued不冒称claimed。多兴趣召回收口且延迟不再补。新批尚未正式提交，不能称已有连续训练库存；数字/收据见experiment_log。

**10月17日等待期任务更新**：物品热度+质量+行为权重组合（FEAT-04 条件组合）和多兴趣召回（M2）配对/原始metadata已补，只读复查包就绪；双向序列模型确认与用户画像特征草稿待Kiro冻结。SID 全量规模基准（S4）首轮在本机纯合成运行，未占服务器。无新训练入队，原两路协同向量 SID（S1）作业/驱动不变；完整延迟与完整拟合缺口见STATE表。

**10月17日02:24最新实测**：共现向量 SID（C1′）和内容加 P 向量 SID（C2）的s42均claimed且全局running，pending0、无rejected；两路CUDA原始cuda_engineering_verified/dry_run_ok已验收，原观察接续、截止不变。数字与完整收据SHA/实际UTC只见experiment_log“两路正式训练接纳”节；训练结果/verify尚未验收，条件补seed未触发。

**02:20当前覆盖：两CUDA容量修正获批并接续，先各自CUDA/测峰/dry，再两个s42及各verify；仅verified且hits≥260的分支补2024/2025。未满足条件不增加训练。旧01:20无活量记录是历史；不自动S2/S5/G5。**

**02:11历史**：重验已过→CUDA pending容量声明微调待Eddy文件覆盖许可→原CUDA/测峰/dry→两s42→各raw verified且hits≥260自动补seed。现在不是科学失败，也没有GPU训练正在跑；旧01:20状态为历史。

**10月17日01:20更新：自动可执行活量0；两S1路线都在G preflight FAILED后停止，尚无正式G训练。** 恢复申请与条件时长见§4；旧GTOK/M1/M2均不能重复入队填空。

**22:59+08更新**：本节旧GTOK/M1/组合/M2安排已收口，不再入队。当前后继是S1两输入有限流水线，实际状态见§4/§4a/§11；仅C1′向量拟合、C2 SID拟合已接续，核心G训练门禁待验收。没有无限后继队列，不自动S2/S5。
> 前面 §4/§4a 是 Codex 的状态快照与读数分工，**没有“接下来先跑什么”的排序**；本节补上。预计墙钟都是估计，不是实测。
**00:08 实测**：服务器完全空闲（CPU 0.81/10.24 核，GPU 空，pending 0，running 0）。GTOK-02 的 6 个 verify 在 23:42 全部收口（驱动日志 `all six reached terminal`），FEAT-04b 第 2 波也已结束。已空闲约 25 分钟，**00:40 前必须入队新作业，否则触发“空闲超过 1 小时”报告**。

| 序 | 批次 | 作业 | 声明 | 预计墙钟 | 前置条件（谁做） | 状态 |
|---:|---|---|---:|---|---|---|
| 1 | GTOK-02 补 seed（P2，无条件） | b256d2 的 s2024、s2025（core03，2 训练 + 2 verify） | 3072 ×2 | 原估约1.7h训练+13min verify，非本批实测ETA | 配置仅seed不同、冻结pin有效、dry-run通过 | **2训练+2verify全部succeeded/verified，原P2 driver已全终态退出（G-R1）** |
| 2 | GTOK-02 补 seed（P1） | 已机械选出B1024D3、B512D3，各补2024/2025，共4训练+4verify | 3072 ×4 | 同上；新叉数耗时非保证ETA | 六新s42直接verified；八点完整排名/规则见experiment_log、STATE | **4训练+4verify全部succeeded/verified，原P1 driver已全终态退出（G-R1）** |
| 3 | FEAT-04b E | 3训练+3verify | 1024 ×3 | 已结束训练实测12–14min，verify约6min（F1） | Kiro工作流已完成，Codex接收验收 | **3/3 verified；E−A/最佳β已记，原驱动保留** |
| 4 | M1 SASRec-v3 | mean3seed先入；其余4臂×3待verified金丝雀 | 每个1024 | 本批train中位13.366min / verify5.929min（15样本，M1-R2） | Opus通过；15真实preflight、正式dry及一次review均通过 | **十五训练/十五verify全部科学验收，十二诊断自动接续身份已核，双role正常退出（M1-R2）** |
| 4a | FEAT04条件组合（当前） | F4-T010+C+D+E，42/2024/2025；各配verify | 每个1024 | 组合真实成本UNKNOWN；8h/3h是cap | 已实现/130相关回归/一次review/冻结/固定SHA/真实preflight/CUDA/VRAM/正式dry全过，入队前已告知Eddy | **三train+三verify全succeeded/verified，driver2921056正常退出（SUCCESSOR C2）；不重提** |
| 5 | 读数已完成→待Kiro/Opus解读 | A1+T3、T1r、温度、A/B/C/D/E、GTOK | — | — | 科学verified后机械套用；E−A/β及C/D报告已补 | **晨间已机械套用M1/GTOK规则；Kiro待复查表已更新，不新写anchor/转正结论** |
| 6 | G确认原批次 | B1024D3 A1三seed，对照A1+T3 | 6656/个，来源为新实测 | 24h/2h为cap | 原工程/正式接纳通过 | **原driver2924901续verify，最新host仍有s2024-verify运行；完整科学终态待原始verified收据，不重提** |
| 6a | M1延迟补证 | F4与M1-mean同口径CPU/A6000 | 已完成工程 | wall见experiment_log | 原正式报告与raw重算通过 | **Kiro下午已核口径，P anchor=M1-mean；不追溯组合** |
| 6b | M2 K4 causal（当前） | 唯一配置×42/2024/2025，各配verify | **新测932→声明1280/个，1核/个** | 真实训练成本UNKNOWN；8h/3h为cap | 五项裁决定稿、一次独立code review及闭环、冻结8688f09、新部署264files、三真实PF/control seal、新CUDA/shape/verify/g0、正式dry全过 | **三seed全claimed/no rejected，三实际worker运行；driver2925994 PPID1/900s/48h自动补verify，已与G尾部接上** |
| 7 | 后续未冻结 | 午间SID pipeline/C1′、bi、F7含U1；G5 | — | — | 各自设计/冻结与授权；G5等组合判定及固定G绑定 | 不在M2自动接续中 |

最新下午裁决覆盖晨间：M2设计定稿、不再开设计评审，明确允许实现/冻结/全部门禁过后一次性三seed，与G并行；现已执行至真实运行。M2结果与完整exact延迟仍待验收。组合转正交Kiro，G5不冻结；不自动增加实验。

**并发与资源**：可用约 8.19 核。1+2 最多约 2+（2×入选数）个 core03 作业，每个 1 核；加上 3 的 E（3 核，约 15 分钟）；总数控制在 ≤8 个 1 核作业。显存声明合计 ≤ 39312 MiB（准入上限）：core03 每个 3072，8 个 = 24576，充裕。

**BACKLOG（估算）**：序 1 约 2×1.7 = 3.4 作业-小时；序 2 若入选 2–3 个配置 ≈ 7–10 作业-小时；序 3 ≈ 0.7。合计 ≈ 11–14 作业-小时，高于 8 小时目标。M1 的数字待量。

**流水线规则检查**：序 1/2 入队时，序 3（E）的 review 与部署已在工作流里推进；序 4（M1）的部署由 Codex 在序 1/2 入队后立刻开始，避免下一次断档。

**风险**：①本机休眠会停掉本机驱动和工作流（今晚 19:30–23:10 已发生）：入队后优先用服务器侧驱动，离开前开 `caffeinate -dims`；②CPU 实测常超声明，E 与补 seed 同时跑时 P 作业可能被限速；③补 seed 的 verify 要逐个读 `verified`，不要以 supervisor exit 0 代替。

## 10. 夜间连续训练提案（2026-10-16，未新增入队）

**晨间最终执行更新，覆盖下文历史提案**：主方案A自动接续成功，M1十五训练+十五verify及GTOK六补seed训练+六verify全部验收。两个M1 role及P1/P2均收口退出；当前队列空，没有新的就绪训练批次。最后项目终态至10:21快照约六小时空档已报告，不能把有限批次表述为覆盖整夜。下文“尚未入队/仍等Opus/尚未部署”仅为原提案时点描述，不再是当前状态。Eddy明确不建晨间timer。

Eddy要求查看本批之后的训练并尽量无空档、明早验收。完整清单与可交Kiro/Opus的复核提示见`docs/superpowers/plans/2026-10-16-overnight-queue-proposal.md`。

- 近端：当前六G训练→原驱动自动补六verify；之后M1 mean3训练+3verify，再last/attn/bi/pure共12训练+12verify。M1合计15+15，工程前置已通过，**仍等Eddy要求的Opus/Kiro复核，当前尚未入队**。
- 中期候选：新SID在A1规模确认、FEAT-04条件组合、G-COLL两输入、F7 U2/U3、条件M2；各有结果/设计/实现/冻结前置，不能当成今晚已就绪队列。G5是融合评估，不计作填空训练。
- 连续方式提案：复核后预排有限作业，服务器scheduler按资源放行，版本化有限driver补verify和最终汇总；不依赖模型逐个醒来安排。**M1这套提交/接续尚未部署或启动，本节不是执行收据**。
- 最近01:56+08状态：六G running、pending0，两服务器driver存活PPID1。来源路径/SHA/实际UTC见`cache/codex-overnight-plan-20261016/receipt-index.json`；不是本节写入时的重新采样。M1真实时长UNKNOWN，不能承诺十五训练明早全部完成或一定覆盖整夜。

## 11. SID-FIX 排程登记（2026-10-16，新 task）

按 pipeline 卡 §1 / §2 与 decision 午间、傍晚、傍晚续执行；不更改任何原阈值。SID-FIX/S0–S5 与 tasks 原全量 S0–S6 不同命名空间。

| 阶段 | 状态 | 队列 / 依赖 |
|---|---|---|
| SID 结构诊断（S0） | 两批 succeeded，A–D已记账（四旧B/C UNKNOWN） | 新只读诊断，1 CPU，无 GPU/训练；review、固定 SHA、新目录部署、真实 preflight 后经 queue。先除 A1+B1024，其余全配置；A1+B1024 三 raw verified 后另补一次。与原 M2/G 独立，按准入并行 |
| 协同向量 SID（S1） | 两路s42已claimed/running | C1′/C2各自输入→验收→SID拟合/replay→3配置真实PF→CUDA/测峰/dry→s42；raw verified且hits≥260自动补2024/2025；两路工程门禁已通过，正式训练结果/verify待验收。 |
| 可学习 SID（LETTER 式，S2） | 未立卡 | S1 ≥1 UP、license 审计、独立冻结卡；已批1名额 |
| SID 深层准确率调研（S3） | 未立卡 | Kiro 读论文后决定 |
| SID 全量规模基准（S4） | 首轮本机合成有界基准已运行 | 最新decision不需另冻结；一次独立code review后已运行，两规模根流式完成，完整拟合/后缀仍缺证据；不占服务器 |
| 新 SID 的 A1 确认（S5） | 未立卡 | S1/S2 赢家 A1 三 seed 确认；独立冻结卡 |

当前 M2 与 A1+B1024 的 job/driver 身份、预算、截止不变；本登记不代表任何 S1–S5 已就绪或有无限后续训练。

SID-FIX最终：sid-fix-s0-first-v1与sid-fix-s0-a1b1024-v1均独立claimed/no rejected并succeeded；声明各1CPU/12288RAM/0VRAM/1GiB/4hcap。真实wall/输入与stdout SHA/实际UTC只见experiment_log S0最终节与receipt-index-delivery；两批工作量不同不当性能对拍。原pending schema仅一次获Eddy批准、保留原身份；无重投。S1草稿待Kiro冻结，不进入队列。

§9 当前后继补充：SID-FIX S0两诊断已完成；S1 C1′/C2只是DRAFT，先Kiro冻结，之后实现/测试/review/固定SHA部署/真实preflight/fit与verify/core03各verify。S2–S5未立卡，不能为连续训练提前入队。最新状态统一见§11与STATE，旧“补批待提交”过程已被最终两终态覆盖。

18:10+08已向Eddy主动报告GPU训练空档至少1h（17:01已知M2/G全收口，其间S0仅CPU）；S1未冻结、无已冻结后继训练，未为填空越权。报告实际读回SHA/UTC在experiment_log最终补证，观测源码98f2324，科学DEP不变。

夜间Opus执行更新：上文S1草稿待冻结为历史状态；当前FINAL与执行ledger见docs/superpowers/plans/2026-10-16-sid-coll-s1-execution.md。两输入独立、不互等，最多六core03训练，各配唯一verify；finite48h服务器driver，完成事件接续/≥900秒轮询兜底，不新建模型timer。新train显存必须由各自CUDA实测计算；目前UNKNOWN，不沿用3072。已知空闲仍如实记录，不把实现中的pipeline称已入队。

### 2026-10-17 04:15+08 已授权并行批次的工程状态

双向序列模型确认：六配置真实preflight与两臂CUDA通过，正式dry元数据一次修复后已通过，六训练各一次发布，mean三训练succeeded/bi三条claimed，六训练无rejected，verify验收待齐，唯一服务器verify驱动存活、实际PPID1。序列用户塔+组合叠加：组合泄漏APPROVED，自身新部署完成、自身真实泄漏preflight全pass，CUDA已接续排队。温度交叉熵：固定τ=.8，新部署已核，真实preflight已通过、CUDA已排队。自生成难负样本DPO：成本门槛已满足，完整实现review APPROVED，新部署完成，真实preflight已通过、CUDA已排队。每批就绪独立提交，不等待别批；资源不足正常pending，不称claimed。详细收据及数字见experiment_log本轮工程节；不碰原协同向量SID驱动。

### 当前安排（2026-10-17 12:49+08，覆盖上方过程状态）

- 协同向量SID：两个配置s42 verify均通过，原筛选通过后两个配置都由新owner自动补seed2024/2025及唯一verify，六训练+六verify全部收口；原截止/名额不变，driver正常退出，不重启。冻结主判机械UP/DOWN交Kiro；指标与完整收据见experiment_log“12:49收口”节。
- 双向序列模型确认：六训练+六verify收口，统计/延迟待补。
- 序列用户塔+组合叠加、温度交叉熵、自生成难负样本DPO：各三训练及三个原始verify均成功；manifest pin本次已独立核对。配对/解释/延迟材料仍按各卡补齐；程序成功不代替有效/转正裁决。
- 全局vkgr-status北京时间12:47仅调度服务running、pending为0；当前无训练/verify，最后已知终态12:38。约9分钟已知空档，尚未到1小时；详细资源/UTC/SHA只见experiment_log。
- 本轮无新增科学训练/verify失败；全局failed为旧保留attempt，不重跑。不为填空增加实验。后继可学习SID/新SID A1确认/融合均未入队，下一批必须Kiro冻结后走工程前置。P/G anchor不变。


### 2026-10-17 13:25+08：午间/午后授权后的安排

- 新SID的A1确认：三个训练加各自verify名额已批，草稿docs/superpowers/specs/2026-10-17-coll-sid-a1-confirmation-draft.md待Kiro冻结，未实现/入队；复用共现向量SID已验收bundle不重拟合，之后按AGENTS§4工程前置，入队前告知Eddy。可学习SID暂缓。
- 叠加与双向确认配对统计：p-postbatch-paired-v1已唯一claimed并succeeded，原聚合结果/hash验收通过；不重跑训练。叠加接合点独立review APPROVED，完整统计数值只见experiment_log“13:25配对补证”节。
- 用户塔延迟：Eddy明确沿原用户塔口径，p-postbatch-users-latency-v1唯一claimed/running、无rejected，CPU1/有限2h，不读checkpoint，不称端到端。统计先结束才开始计时，避免两项并发；待原始report，不用模型轮询，单次inotify终态事件等待。
- 温度交叉熵与DPO按Kiro午后收口，无追加配置；一次早停曲线读取完成，结论只见issue I-2026-10-17-g-earlystop-censoring，不据此延长训练。当前无训练排队，只有计时作业；不为填空新增实验。

13:39+08已向Eddy主动报告：从最后已知训练verify于12:38结束算，训练空档约1小时；当前仍有用户塔计时，不称全服务器持续空闲。没有新的训练已入队，S5仍等Kiro冻结；没有以填空为由加实验或缩短计时。


### 2026-10-17 14:20+08：无人值守预确认已接收，优先新SID A1

Kiro新增无人值守授权§1预确认同午间/午后科学内容，卡已FINAL，不再等回复。新SID A1实现a5da4d6已push，独立review两工程MEDIUM关闭，98项相关tests通过、旧入口全未改。新DEP/OPS-v1部署与真实PF/CUDA/显存/dry进行中，尚未有科学训练入队；入队前本时段写STATE时间即可，之后必须独立claimed/no rejected与唯一有限driver。

用户塔计时已于13:37成功完成，14:00原报告验收通过，之前“仍运行”过程状态因通知失效已纠正，详见experiment_log最终验收与issue。配对、接合点review、曲线、两协同SID A–D和指定论文摘要均完成；无并行科学训练，当前缺口是新SID A1工程前置，服务器空档已告知Eddy。不加配置填空，R1/R2不追加、S2/G5/F7不提前启动。

### 当前安排（2026-10-17 15:03+08，覆盖上方工程过程状态）

新SID的A1确认（S5）三个seed的工程闸门全部通过，已一次性发布并独立确认各claimed、无rejected；唯一服务器driver PPID1、900秒、原48h截止，负责训练成功后自动补各唯一verify。原训练不重提、不重启。当前科学结果/verify未验收；本批声明、显存实测公式、source/config/data与原始SHA/UTC见experiment_log“15:03接纳”节，不以claimed推断训练具体epoch或剩余耗时。P补证/用户塔计时、曲线、SID A–D、论文摘要已完成；没有另行获批的就绪科学后继，不启动S2/G5/F7，不为填空追加实验。

2026-10-17傍晚裁决同步：叠加与双向确认均不转正、收口，P anchor维持M1-mean。新SID A1确认由原driver2969937接续，不重启/重提/重置截止；原始verify收口后交Kiro。G anchor仍A1+T3，融合（G5）等S5判定再冻结，用户画像（F7）本轮不启动。决策原文见decision_log傍晚，数字与收据见experiment_log。

2026-10-17 15:52+08只读快照：新SID A1确认三个seed训练进程alive、未终态，原driver2969937实际PPID1/alive，verify尚未触发。无重投/重启，保留原截止；原始progress/queue收据SHA/UTC见experiment_log“15:52阶段进度”。尚未收口，三条件未验收，交Kiro材料目前仅阶段状态，最终待原始verified。

16:23+08只读复查：新SID A1确认仍训练中、原driver存活，三个自动verify尚未触发，无失败终态/rejected；不重投、不重启，待原始verify收口后按卡验收交Kiro/Opus。当前数字与SHA/UTC统一见experiment_log最新节。

2026-10-17 17:49+08：新SID A1的s2024/s2025训练succeeded，同名-verify由原driver自动接续且各claimed/no rejected；s42仍epoch20评估，实际PIDalive。原driver2969937 actualPPID1/alive，原截止不变，不重投/重启/手动补verify。整批未收口，三规则未齐证；原manifest/队列/progress SHA/UTC见experiment_log最新节。最终验收及Kiro/Opus结果提示词待原始verify齐，不改anchor。

2026-10-17 19:00+08最终收口：新SID A1确认g-coll-a1-c1-s42/-s2024/-s2025及各-verify全部succeeded、raw verified，独立绑定验收通过，原driver2969937写verified summary后退出，不重启。三个冻结有效条件满足，但P/G anchor仍M1-mean/A1+T3；前缀诊断仍UNKNOWN，完整结果与原SHA/UTC见experiment_log最终节。Kiro/Opus提示词docs/superpowers/reviews/2026-10-17-coll-sid-a1-result-review-prompt.md。现无已授权就绪后继，不自动S2/G5/F7，不重置原截止或重提作业。

2026-10-17 19:24+08晚间交付：G anchor=A1+共现向量SID，A1+T3仅历史对照、P=M1-mean。唯一前缀只读job sid-a1-prefix-report-v1与其preflight均succeeded、claimed/no rejected，原S0 B未改；全部计数及143输入SHA/来源UTC见experiment_log最新节与s5-prefix-final-acceptance.json。DEP/OPS=/workspace/vkgr-s5-prefix-tools-v1、/workspace/vkgr-s5-prefix-ops-v1，source d7c6551，job已结束不重提。G5卡2026-10-17-g5-anchor-fusion-draft.md与S2卡2026-10-17-learnable-coll-sid-draft.md均DRAFT，待Kiro，不冻结/实现/入队。P top100与旧checkpoint导出授权缺口、license UNKNOWN明确保留；没有就绪新训练，不为填空加作业。
