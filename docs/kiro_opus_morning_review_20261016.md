# 给 Kiro / Opus 的复查提示词：晨间结果、现行裁决与后续接续（2026-10-16）

请确认实际使用 Opus。你是顾问/复查者，Codex是唯一执行者；不要恢复旧workflow、直接改代码/队列或重提训练。整理过程中发现你已写入 decision_log「2026-10-16 晨间：M1与GTOK-02结果裁决…」（Claude Opus5.5），**本提示已同步该最新裁决，不请求重复批准已决定的事项**。请只核对后续证据、口径和仍未冻结的设计；关键裁决写decision_log，数字由Codex维护。

## 先读
1. AGENTS.md，STATE.md最后节与当前“待Kiro复查/待裁决”表，handoff.md最后节。
2. docs/decision_log.md的02:xx、夜间§1–5，尤其最新“晨间”§1–5（优先）。
3. docs/experiment_log.md的“2026-10-16 晨间M1/GTOK验收（最终）”及末尾FEAT04-COMBO工程节；schedule当前覆盖说明及§9。
4. seq-m1-design §3.6/§3.11/§8、gtok02-design §3.8、原FEAT04计划§1.2；原卡不在主仓时按STATE指向worktree读。
5. cache/codex-morning-acceptance-20261016/receipt-index-final.json；cache/codex-feat04combo-20261016/receipt-index.json。必要时读对应原始stdout。不要读test、既有真实checkpoint或完整embedding。

## 一、这批已验收结果与原始证据
M1十五训练+十五verify、GTOK六补seed训练+六verify全部succeeded/verified，旧GTOK六s42与六fit也已复核；十二诊断03:16自动接续，发布payload=冻结plan=当前claimed，mean金丝雀原始verified早于放行，无rejected。每项核manifest/job/config/source/data hash与validation归档重算，不凭exit0。

seed顺序均42/2024/2025；validation2520query，623结构性OOV零命中，未读test。

| P配置 | hits三seed | mean hits | mean Δhits vs F4 |
|---|---|---:|---:|
| F4-T010 | 439 / 434 / 444 | 439 | — |
| M1 mean（预注册主臂） | 466 / 481 / 471 | 472.6667 | +33.6667 |
| last | 475 / 485 / 476 | 478.6667 | +39.6667 |
| attn | 468 / 473 / 472 | 471 | +32 |
| bi | 482 / 484 / 486 | 484 | +45 |
| pure（仅归因） | 417 / 413 / 396 | 408.6667 | −30.3333 |

mean逐seedΔ=+27/+47/+27；exact McNemar双侧p=0.0376452399/0.000146283947/0.0364227525，3/3显著；p_pool=2.936836114e−6，k+=3，机械有效。p_pool跨seed重复query偏乐观；完整N@50、CI、best/stop epoch、discordants见experiment_log。最佳ratio=0.573334/0.505181/0.557538；g0Δ=109/97/107，正贡献3/3。**g0后只剩357/384/364 hits，m已与分支共同适应；g0Δ不能读成序列分支带来约100hits，净增益只看对F4的+33.67。**

bi按冻结规则为探索性代表，比mean均值高11.3333，×4校正p_pool=2.157467771e−8；但bi对mean逐seed配对p≈0.198/0.872/0.199，不能说双向优于因果或转正，须披露四选一赢者诅咒。pure：E_start=70、E_cond=149.6667，机械两者均有贡献；不可拆比例，S1仅seed42、条件是一整包，pure仍低于F4约30.33。

| G配置（core03） | hits三seed | mean hits | 补seed2024/2025 Δvs T3 | Q1状态 |
|---|---|---:|---|---|
| T3 | 254 / 274 / 275 | 267.6667 | — | control |
| B1024D3 | 280 / 285 / 283 | 282.6667 | +11 / +8 | UP |
| B512D3 | 263 / 273 / 267 | 267.6667 | −1 / −8 | FLAT |
| B256D2 | 233 / 280 / 262 | 258.3333 | +6 / −13 | FLAT（位置对照） |

Q1只用补seed，B1024−B512=+12/+16，UP，所测范围1024侧较高；不外推全局峰值。B1024第2/3层利用率11.5%/0.1%，实际是少一层语义+更大编号，不能说分辨率越高越好。B256D2±12检验仅支持原推断，不是碰撞因果，两补seed分歧须并列报告。

关键原始收据（相对main05；前缀不是时间）：
| 收据 | stdout | SHA256 | 实际读取UTC |
|---|---|---|---|
| M1原始复核 | codex-morning-m1-fresh-20261016T102000.stdout | 9587030389f099aa3f2609ed31e1604227a4ae0f23e66758bf6b927ad8ac894e | 2026-10-16T02:20:51.870888+00:00 |
| GTOK原始复核 | codex-morning-gtok-repaired-20261016T104000.stdout | 4e156161c0bb608145c20047c09bdb857b61f6bf22f7d95376a015857e00e960 | 2026-10-16T02:20:53.177999+00:00 |
| 当前组合接纳 | codex-combo-final-acceptance-20261016T110600.stdout | 93c4923709c0ef1204bd61da80da339978646628ec54b3d6010c12034ce9e00c | 2026-10-16T02:56:32.549162+00:00 |

## 二、已收到的Opus裁决，不再重复询问
1. M1 mean有效；认可已有seq-m1-code-review，**不新增review**。只缺服务器§3.11延迟报告。报告按口径进入experiment_log并由你核对口径后，P anchor才从F4-T010切至M1-mean；无延迟数值通过阈值，不按好坏临时加门槛。切换不追溯FEAT组合。
2. M2已批准只到设计/设计评审（最多2轮），不实现/冻结/训练。因果mean骨干、K=4多兴趣读出，主对照M1-mean同seed，另报F4；hard-argmax头无梯度风险要有守卫与处理，延迟加K路检索/合并，沿原统计门槛。bi今天不立卡，新名额须Eddy批准。
3. A1确认名单只B1024D3，三seed，对照A1+T3的289/328/310。三训练succeeded且verify直接verified、meanΔ≥6、至少2seedΔ≥6，全满足才按已预注册条件切G anchor；否则保持A1+T3并列报。仍需Codex完成入口兼容/配置/冻结/真实engineering=False preflight/CUDA/实测VRAM，不能直接沿6656。
4. GTOK报告补证只阻塞其最终报告，不阻塞B1024确认：padded4 B1实测要补，新叉数真实峰值保持UNKNOWN并保留issue，常零位置logp(0)可UNKNOWN。C9仍仅报告，U1不做。

## 三、FEAT组合已经推进到实际训练，不再等工程
Eddy直接要求执行，你夜间§4已登记：F4-T010+C+D+E，三seed42/2024/2025；完整E的Q2+六行为权重不换A，候选四表只对supported生效，历史侧不混候选统计。原T0.10/N1/L50/lr/100-5-3、stats sidecar和桶边界不变。

**截至10:56+08**：独立新入口/测试/一次review/冻结/固定SHA部署/三真实只读preflight/CUDA train→verified/显存实测/dry均通过，三真实训练p-feat04combo-cde-s42/-s2024/-s2025全部唯一claimed、实际running、无rejected、pending0。服务器verify驱动2921056 alive/当前PPID1、900s/no LLM、48h截止；每训练成功后自动补verify。本批尚未验收，不写效果或转正。

源码codex/feat04-combo-pub@84b160cb0eeb3c53b877f6cf6474276ba06a6ba5；旧文件未动，236旧基线及最终130相关回归（含15新）PASS；freeze216文件，部署逐文件286项SHA通过。真实PF每seed检查512训练样本、全部2520validation及catalog120415；全shape VRAM预演358MiB，声明1024MiB。全部收据/SHA/实际UTC在cache/codex-feat04combo-20261016/receipt-index.json。

组合仍相对F4按原§1.2，另报组合−E及最佳beta。D-1和23:2x更正必须带上：sum_likes105700对86785、93.0%物品likes==target_freq、p_like0.2115；C9仅报告。不要因M1后续转正而改这项已登记的对照。

## 四、后续接续预案与工程前置，请核口径和证据
**A. 当前组合继续**：三训练→原服务器驱动自动补三verify→终态汇总→Codex验收。失败/拒绝不重跑；不因指标加臂/挑seed。无需再批准已登记名额。

**B. B1024 A1确认**：按你最新明确名单做三seed，前置先于入队：基于实际A1+T3 pinned版本核sid_v2s兼容，只改SID绑定/run_id等允许项，冻结，服务器真实engineering=False只读preflight，CUDA与真实shape显存实测×1.15向上取整256，dry→claimed/rejected→每train对应verify。可与P线并行，按约8.19核和显存准入。未具备这些证据前，不写“已就绪/已入队”，不拿旧6656作实测。

**C. M1延迟小作业**：用户塔、batch512离线/batch1在线、CPU单线程/A6000、真实validation长度分布、预热50/计时1000、mean/p50/p95/p99，anchor与M1-mean同脚本；不读既有真实checkpoint也可做相同结构/输入形状的计时，具体口径按§3.11，最终由你核对。**静态发现：现有seqm1_latency_bench.py仍将非50长度作均匀近似，offline最多只计20次；直接加--iters1000不满足新口径。必须先版本化修正/适配，核真实长度分布及两种batch计时次数后，才能提交合规报告。** 没有把旧合成CPU比值当转正证据。

**D. M2只做设计卡/评审**：按你最新五项约束起草，设计以M1-mean为对照，N10风险不能留空；实现/冻结/训练另行裁决。G5等P/G anchor都定下再冻结，可先更新绑定但不冻结；G-COLL/F7仍有接口/窗口待裁决，U1不做。bi另卡不在现有名额。

三类允许执行的工程批次A/B/C互相独立，尽量预先准备有限READY队列；driver版本化、固定SHA、PPID1、≥900s、48h、机器终态，不依赖模型逐个醒来，不新增timer。任一失败/非verified写ALERT，rejected读.reason不重投，失败attempt不重跑/覆盖；不存在全部前置通过的批次时如实报告，不能为填满机器加实验。昨夜04:11后到10:21有约六小时项目空档，当前已恢复组合训练，不宣称有限批次覆盖无限后续。

## 五、这次/后续需要你返回什么
- 对新补的延迟报告核口径“通过/需补证”，不按数值另立阈值；满足你已冻结条件时写P anchor变更。
- 对B1024真实入口/冻结/preflight/VRAM证据、最终三seed确认结果按已冻结条件复查，缺哪项明确列出；不重复批准名单。
- 对M2设计卡只做设计评审；对组合等三verify齐全再解读，不提前转正。
- 每项写理由、对应文件/收据及是否阻塞。关键裁决写decision_log，再由Eddy转交Codex；不改数字事实源，不直接启动workflow。


## 本轮追加：SUCCESSOR工程与M2设计（请以STATE末表最终状态为准）
请在Opus下复查，不需要重复M1既有泄漏review或已定B1024名单：
1. M1延迟正式结果在experiment_log“SUCCESSOR”§2，原始report SHA249f5557…93624db，全部1000样本/50预热/输入配对/分位数重算已核。真实长度、合成特征值、初始化权重、users()范围及无ANN/取数/网络限制已写。请只核口径；按已有晨间裁决处理P anchor，Codex尚未切换。
2. B1024 A1确认沿原三seed/同seed A1+T3/冻结判据推进；实际工程和claimed/driver状态见STATE最新节和experiment_log后续收据，不把源码review代替server gate或把工程probe当召回结果。三verified齐全后才机械套用条件交您复查。
3. M2设计卡docs/superpowers/plans/2026-10-16-m2-multi-interest-draft.md仅DRAFT，无实现/冻结/训练。请集中裁决§8：固定100-step soft loss warm start→hard、hard覆盖窗口101–200；detach权重和共享温度；exact max/四路各50中间最多200的算力边界；branch_ratio的query内max定义。它们都是建议，尚未变成门禁。最多2轮设计评审，不新增臂选方案。
4. FEAT组合三seed已直接verified，效果量见experiment_log SUCCESSOR§1；尚缺逐query配对统计，不凭均值自主转正。组合对照仍F4，不追溯改成M1。下一步若需要组合后的新实验，应先由您给出唯一内容和冻结规则；当前未增加训练。

### B1024工程阻塞补充（覆盖上一条推进状态）
科学源码c1c4b74/三真实PF均通过，但cuda-v1在worker启动之前被共享supervisor的64KiB run.json读取上限拦住（66316字节）。没有CUDA或显存证明，三真实seed从未入队；原attempt全部保留。issue I-2026-10-16-b1024-launch-record和experiment_log SUCCESSOR§4有根因收据。Codex已实现独立review通过的clean Git launcher运载修复（e3df6db，79文件逐SHA部署），每job仍完整校验原科学DEP且输入/配置/源binding不得变化，共享scheduler/64KiB限制不改；新增48KiB准入元数据预算防复发。先完成新路由真实preflight与工程dry，再请Eddy批准一次重新工程执行，不能因换ID绕过晨间§5失败attempt禁重跑。请将此纳入状态复查，不误认为G模型性能失败或已入确认队列。

### 最新：阻塞已解除，三seed已运行
Eddy已批准仅一次修复后工程，cuda-v2完整证明通过；新显存实测及公式见experiment_log SUCCESSOR§7。三真实A1+B1024 seed全claimed且host实测running，driver2924901存活/PPID1/900s/48h，后续补各verify。science尚未验收，仍按原三条件复查，不提前换G anchor。原v1保留，工程issue关闭；最新来源§8/final-admission-summary。无需重审这三seed启动授权；请复查M1延迟口径、M2草稿§8，新G三verified后再解读。
