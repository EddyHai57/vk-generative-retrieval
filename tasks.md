# tasks.md — Codex 实施阶段

## OWNER-HANDOFF-20261013 — Kiro / Opus 5.5 接管（不新建实验task）

Eddy最新直接要求暂停定时检测并全权交给Kiro。vk-02及旧VK/P/G automations均PAUSED，原服务器P4H-05-S02-S06 driver继续；Codex只完成本次文档交接后停止模型执行/查询/写入。Kiro已于2026-10-13 13:06:09+08保存实际ACK（kiro-acceptance.json）并成为唯一管理owner；S02 failed+spent/CLOSED、S06未启动，详见STATE；不restart/duplicate submit或重置预算。具体恢复见STATE/handoff与docs/kiro_opus55_takeover_prompt.md，本条优先于下文旧owner/model措辞；科学task顺序/合同/预算不因交接改变。

## PG-R01-P-RECOVER — P恢复adapter与一次pilot（2026-10-12）

**COMPLETE_ACTUAL_VERIFIED_COST_READY（2026-10-13）**：实际pilot两臂30steps/matched2520，独立verify/worker/controller/pub CLOSED；补充CPU cost worker实际exit0/CLOSED和fresh owned absence。完整绑定保守S02/S06 forecast15232s/6561625858bytes，成本完整、符合14h/32GiB；formal permit仍false，不能以成本forecast代替新准入。最终接收main05/p-recovery-final-acceptance.json，旧queue/登记/spent/artifacts不改、不退款。原根因/授权历史下文保留，但不重跑接收项。

2026-10-13当前接续范围：pilot07与独立verify已actual CLOSED/APPROVE接收（main05/pilot07-final-acceptance.json）。本task继续补两项cost prerequisite：固定validation2520完整prepare/部署、质量聚合与metrics序列化的CPU成本；只能作成本测量，不训练、不追加optimizer/science attempt/HPO/quality selection/test。零模型向量的CPU排行仅供耗时测量，不能报告为模型质量。新独占cost目录实际first-write冻结finite UTC/mono≤2h，与原树old+new20GiB累计charge、旧spent/queue不改；controller actual wait/CLOSED及fresh资源/identity/unknown-writer门继续。完整实测cost仍不足时不准入formal，足够后才另进原S02/S06登记task。Eddy最新heartbeat明确要求“缺cost主动补齐”，普通实施不再重复确认；不改变已闭合train-only pilot的数据访问。

Eddy在已review有限计划后明确要求一直推进、减少模型等待消耗。当前对话单owner inline执行：只改P retrieval_sasrec_remote.py/tests与新ignored cache、必要Main事实；fresh RED→GREEN→相关回归、新完整delta仅一次review、Linux/full publisher proof、重prepare、一次新增pilot→独立verify，累计attempt3/新增4h/原＋新共20GiB。P正式S02/S06在verified成本准入后才另进入原task；不重审v8/旧suite，不运行新feature/HSTU或test。已过期原queue/spent不变；尚未启动部分按真实first-write冻结有限UTC＋monotonic deadline，不再使用人为晨报日历截止。授权记录cache/pgr01-p-recovery-v1/human-continue.json；状态见STATE。

2026-10-13人类直接追加：允许原04:00:55/20GiB内自主修复worker释放前准入编排，记录main05/direct-human-preworker-exception.json。原attempt3与失败cost不删改；append-only前置接续须同queue/科学/data、原admission/worker不存在、旧driver/controller/publisher真实CLOSED及actual SID fresh absence。科学worker释放后失败仍STOP/no retry/resume，clock/spent不重置。当前source/proof/producer见STATE与main05最新收据，不重跑旧接受项。

2026-10-13直接追加clock grant：只加7200s effective至06:00:55，original queue bytes/00:00:55 UTCmono starts/boot/spent/attempt3/20GiB全部保留，原字段不覆盖。授权main05/direct-human-clock-extension.json固定SHA与append authority证据见STATE/handoff；科学release后失败不retry，正式仍成本BLOCKED。

2026-10-13 UID退出竞态修复已完成：local/Linux80、16真实退出及2 live负例通过，一次review APPROVE；历史failed run根因仍UNKNOWN、无新科学提交。用户要求普通问题主动解决、主要看最终结果。累计attempt4单次有限例外proposal已提出但未授权，原06:00:55/20GiB/0resume及全部spent保持；细节见STATE与main05/uid-race-fix-acceptance.json。

2026-10-13最新用户明确“不用管限制，一直做下去”，覆盖上段attempt3/no-retry/旧截止阻塞，允许自主诊断、修复和连续执行。旧queue/登记/spent不改；主执行选择新attempt04单次4h/同charged root旧＋新20GiB有限包，0resume/no新资源/no formal。新授权原话main05/direct-human-autonomous-continue.json固定SHA f0d48747…；新authority RED/GREEN、runtime84与driver8通过，一次review APPROVE后已真实部署；现场状态见STATE。

## P4H-05-S02-S06 — 已登记下一task，正式队列准入与执行

依赖PG-R01-P-RECOVER实际完成。只S02 like/LR0.001/T0.15/seed42→CLOSED/独立verify后S06 observed同LR/T/seed42；固定模型/source/86785 train/full120415catalog/full2520 validation，每slot≤30epochs、5100steps、1科学attempt、0resume。新formal queue first-write UTC＋monotonic总14h/32GiB，旧pilot/cost spent和全部产物保留。Calendar催促取消，不重置真实first-write包络。完整cost预测与已review1.5余量、原子峰值、prepare/deploy/fresh-best/独立verifier/cleanup均在main05/p-recovery-final-acceptance.json；base per-arm时间需加1104s得到完整每slot上限（uniform7616s），不是直接用旧partial6512/6427。

允许Main ignored新exclusive queue/packet/driver/tests、必要Main/P事实；P retrieval_sasrec_remote.py/tests只有必要普通bugfix且fresh RED/GREEN/相关回归、新完整delta仅一次review。原v8/110已接受suite、pilot/cost job、旧G不重跑。不改科学/label/history/catalog/test/hash/identity/unknown-writer/20%资源门，不加HPO、新资源/生命周期、push/publish。

启动前具体package/source/input/角色/registered slot ledger/mean-control证据绑定、新实际formal prepare、fresh shared资源与G原identity/终态、exclusive absence、外部permit TTL、总budget/atomic/closure余量必须通过。只一个server driver拥有作业，真实process wait、train→fresh best export/exact→独立formal verifier→实际CLOSED；只读收据沿用当前单一认证mux/stdin/no fallback，先保证transport finite覆盖完整formal queue，不能带着即将过期的2h连接启动14h作业。模型不持续等待，same heartbeat接续，只真实变化通知。2026-10-13完整delta review APPROVE、local/Linux新5 PASS，实际12:19:32 first-write＋14h/32GiB freeze，唯一driver/受监督CPU prepare已start；科学S02/S06 worker尚待各自exact permit，不把prepare当训练完成。当前身份/source/transport和证据见STATE/main05/formal-pair01-stage-acceptance.json。

Stop：成本/准入证据不完整或资源/身份/未知writer/data/hash漂移则不启动；运行失败先真实收尾、保留spent和UNKNOWN，再证据诊断，不盲重提旧job，不因指标差调参。质量指标唯一P experiment_log，不把单seed配对或CPU synthetic cost当整体赢家/test结果。G补2024/2025为后续独立task，当前不合并。

## PG-R01 — 基线接收与P恢复预算（2026-10-12）

Eddy在当前对话明确要求开始第一阶段。范围：只读接收G四配置原产物、核对P失败及修复、给出有限执行计划和预算；不提交新训练。执行清单见`docs/superpowers/plans/2026-10-12-pg-phase01-baseline-reception.md`，长期路线见`docs/superpowers/specs/2026-10-12-pg-research-roadmap.md`。当前owner/结果只见STATE；旧spent、合同和test gate保留，新预算proposal不能当grant。

## P4H 过夜最新优先计划（2026-10-12）

Eddy授权新GPT-6.1 Sol/medium对话接收监督；Kiro停止模型推进，旧主移交后停止执行。执行入口`docs/superpowers/plans/2026-10-12-p4h-overnight-s02-s06.md`：03B v8已APPROVED→真实运行层验收→04 pilot→05 S02优先/S06其次；每次一个task，原科学/budget gates保留。下方旧KIRO/v6摘要为历史，不据此重跑接受项。

## TAKEOVER-01 — 双线固定交付计划（2026-10-11）

Owner：Kiro继续作为G/P两线唯一协调与执行owner；Codex仅可只读预加载，不得启动writer、monitor或job，除非Eddy以后明确切换。旧 Codex/Kiro/automation 不恢复，不建立重复 monitor。

| 顺序 | Task | 状态 | 验收 |
|---|---|---|---|
| 1 | TAKEOVER-01：事实源、Git/Python、原 job identity/spent/caps、远端终态对账 | COMPLETE | `cache/server-status-20261011-takeover-v1/stdout.json` hash 与 receipt 一致；无 active `vkgr` process；无 submit/restart/reset |
| 2 | G3-R1：修复 core01/02/03 启动包 config binding | COMPLETE | fresh review APPROVE，candidate/report SHA `2c462070…718de`/`f991911e…e067` |
| 3 | P4F-A1：已有 v3 产物与科学结果验收 | COMPLETE | target review SHA `c227d3…5d437`；fresh review APPROVE，report SHA `92459e…d9f3`；P4F关闭且不重跑 |
| 4 | G3-R2：三个独立新 attempt | RUNNING | admission review SHA `256ab480…7787`；三个v2 unique roots/start/deadline已反绑，attempt均spent；Kiro terminal7唯一replacement monitor PID2836；core01 complete，core02/03最后状态RUNNING，只有state-change receipt才更新 |
| 5 | G3-R3：selection 与 winner 补 seed | PENDING_RUN | 四个 seed42 全部成功后按 all-eligible Recall@50→NDCG@50→既有 tie rule 冻结 winner；只跑 seed2024/2025，不挑 best seed |
| 6 | G4：随机 SID control 与 beam 诊断 | PENDING_RUN | winner 固定；seed42/2024/2025；beam50/200 仅诊断，正式 beam100 不变；不追加搜索 |
| 7 | P4G：传统线冻结与一次 test | PRETEST_FROZEN_WAITING_SHARED_G | freeze/review SHA `c0bf8ddc…3f87`/`c8fd0ffa…0e8d`；P4D24+P4E9=33/33、6条remote failure ledger、传统selection冻结；test未读 |
| 8 | P4H-01：SASRec parallel plan/config/protocol freeze | COMPLETE_LOCAL_FROZEN | 15/15 targeted；5-file manifest SHA `2dfe420b…55c0`；v2→v5与11 local protection pins通过，remote/test BLOCKED/CLOSED |
| 9 | P4H-02：固定 ID-only causal SASRec-style model | COMPLETE_LOCAL_FROZEN | model/test SHA `301b7086…f00c`/`2be2a3aa…c78b`；own10/10 clean、combined23/23，P4H-01与旧P4D/E hashes不变 |
| 10 | P4H-03A：finite typed science runner | COMPLETE_LOCAL_FROZEN | science/test SHA `ce427443…b77a`/`86558a67…bc8c`；own12/12、related60/60；无CLI/manifest/path/process，前序hash不变 |
| 11 | P4H-03B：operation/admission/controller synthetic | KIRO_FRESH_REVIEW_IN_PROGRESS / CODEX_PRELOAD_ONLY | v6 source/test `fd53b42a…82d9`/`00e70e01…b0a`，packet `9bf8c4cf…e6a6`；v5 review两个HIGH仍open；Kiro正在独立fresh review；remote/test/SSH关闭 |
| 12 | G5：固定总候选预算融合与 offline infer | BLOCKED_BY_G4 | k60/final50/总400 slots；四路各100、五路各80；P4H不阻塞G5，G4 cutoff默认原P4D TT |
| 13 | DELIVERY-01：双线报告与全量接入准备 | BLOCKED_BY_G5_AND_P4G | subset 结果/失败/成本/复现/claim boundary/public scan 完整；容量未知写 UNKNOWN；不下载全量、不增资源、不部署、不 commit/push/publish |

### 当前 Task G3-R1 执行合同

- 允许修改：仅 G worktree 新目录 `cache/g3-seed42-retry-candidate-v1/`，以及完成后 G/Main 的 `STATE.md`、`handoff.md`、`docs/issue_log.md`、`docs/g3_validation.md` 当前记录。
- 固定输入：只读 `cache/g3-formal-A-candidate-v2/`、失败包 `cache/g3-seed42-parallel-v1/`、`configs/generative-core-v1.json` 与现有 `src/vkgr/generative_{experiment,real}.py`。
- 预期输出：三个 pending、不可执行的新 packet template；roots 固定为 `...core-01/02/03-seed42-operational-v2`；worker/controller/input/registration/config 全绑定；旧 v1 bytes 不变。
- 验收命令：既有 Python 3.11 执行新 candidate unit tests、`verify_local.py`、`tests/test_generative_real.py`，最后 `git diff --check` 与旧包 hash 保护检查。
- finite caps：本 task 仅本地 CPU 与小文件；0 SSH、0 GPU、0 submit、0 remote write、0 attempt、0 resume。
- stop gates：需要改科学合同/生产模型代码、旧 packet、旧 root/receipt、增加实验或预算，或无法严格拒绝错绑时停止 G3-R1；不以放宽校验换 GREEN。

### 当前 Task P4G pre-test freeze 执行合同

- 允许修改：仅 P worktree 新目录 `cache/p4g-pretest-freeze-v1/`，P/Main 的 `STATE.md`、`handoff.md`、`tasks.md`，以及按事实类别更新 P `docs/{p4f_validation,experiment_log,issue_log}.md`；不改生产代码、冻结 config 或旧 artifact。
- 固定输入：P4D manifest SHA `6ddccee458c3c66bb185d51527c0dc32981ae83949efe420eef97ed45881ac6e`、P4E manifest SHA `b25336929fe6b94d24a5e847343f63cfe4c54b24ae0f9cd9b6f05deab58c58cc`、P4E acceptance、P4F target review SHA `c227d3…5d437` 与 approved report SHA `92459e…d9f3`。
- 预期输出：机读 pre-test freeze，证明 P4D 16个seed42 config+每架构2补seed共24、P4E三消融×三seed共9，无缺项；冻结 `id_mean/LR0.001/T0.15/seed42`、wRRF `[2,1,1,2]`、exact authority/HNSW128 audit，并列真实工程/attempt失败 ledger 与 accepted UNKNOWN。
- 验收命令：既有 Python 3.11 执行 `verify_local.py`，重算全部输入与 packet hashes，运行 `git diff --check` 和 public-boundary scan。
- finite caps：仅本地只读审计与小型 aggregate/hash 输出；0 SSH、0 training、0 pipeline rerun、0 test读取/计算、0新attempt或资源。
- stop gates：任一预注册seed/config/消融缺失、重复、hash或acceptance冲突即阻塞并只补既有计划缺项；不得新增搜索。共同 G/P selection freeze 与 test plan 尚未齐全，因此 test gate 必须保持 `WAITING_SHARED_FREEZE`，不得根据未来 test 回调。

### P4H v5 集成与当前本地 task

- Plan：`.agents/tasks/model-audit/p-sasrec-parallel-plan-v5.md`，SHA `ea3f7566750c4fad13dfba80fd03a5f8fdf7e81c4c68aa8912b09b3e0e732121`；review SHA `3de5ad402b6dd9e033d068567d046071ba4998e888914f5ac49c155c035287a1`，verdict APPROVED。
- Owner：Kiro继续作为Main/P facts与P4H-01 owned files的唯一协调与执行owner；Codex仅可只读预加载，不得启动writer、monitor或job。
- [x] Facts最小集成；P4G freeze/test closed、G优先与G4 cutoff保持。
- [x] P4H-01：fresh RED→minimal GREEN→15项targeted；5-file manifest SHA `2dfe420b…55c0`。
- [x] P4H-02：fixed ID-only model/test；own10/10、combined23/23；SHA `301b7086…f00c`/`2be2a3aa…c78b`。
- [x] P4H-03A：typed science/test；own12/12、related60/60；SHA `ce427443…b77a`/`86558a67…bc8c`，无CLI/manifest/path/process。
- [x] P4H-03B v6 candidate：source/test SHA `fd53b42a…82d9`/`00e70e01…b0a`，packet SHA `9bf8c4cf…e6a6`；review-required11/11、targeted74/74、legacy51/51、full301 passed+2 skipped。
- [ ] P4H-03B v6独立fresh review由Kiro进行中；v5两个HIGH仍open。
- P4H-04 BLOCKED；`D-P4H-REMOTE-01`、`D-P4H-REMOTE-02`、`D-P4H-G5-01` 全部 `BLOCKED`，test `CLOSED`。

## COORD-03 — 单主对话与上下文交接制度（2026-10-11）

状态：COMPLETE_LOCAL。官方来源、独立短研究、当前/max metadata、TOML仅两项差异、制度指针与diff whitespace核验通过。window已配置872000/auto_compact700000；当前运行时重载尚未核实，不声称实际已扩容。无新实验/SSH/新主执行对话。具体制度见docs/context_session_policy.md。

Eddy要求单Codex长期执行者，核当前/最大上下文，短sub-agent研究GitHub与工业实践，制定主动提醒、STATE/handoff维护及冷启动新对话制度。允许文件：AGENTS.md、STATE.md、handoff.md、docs/codex_handoff.md、CLAUDE.md、docs/context_session_policy.md和本节；按用户额度请求核/调整本机Codex context配置，不改模型/认证/权限。验收：官方来源、本机安全metadata/计数口径、TOML仅目标项差异、文档链接与diff whitespace。stop gates：未证明实际生效不称已扩容；不猜实时token数，不启动实验/SSH，不恢复旧对话/定时器，不覆盖外部成果/预算。制度完成后结束本task；后续G3/P4F分别按原task与budget gate推进。

当前阶段与完成状态只在`STATE.md`维护；本文件只定义路线、依赖、产物、验收和stop gate。

执行约定：每次只进入一个阶段；满足验收后才能进入下一阶段；触发 stop gate 时先记录证据，不自动扩大范围。

两天 vertical slice 只覆盖 P1–P7 的最小路径，并允许因 wall-clock 在固定 query audit set 收口。

P8–P13 不属于两天承诺。

## P0 — Bootstrap 与治理

依赖：无。

产物：thin `AGENTS.md`、`STATE.md`、根`handoff.md`、`tasks.md`、`docs/`设计文档、服务器runbook、Codex handoff、draft configs、目录边界说明与本地Git初始化。

验收：AGENTS只含稳定规则和指针；STATE只含当前快照；handoff只含接力证据和恢复方法；tasks只含路线定义；公开边界、必需文件、YAML、本地链接、敏感字段、代码越界和Git状态通过检查。

后续gate：bootstrap commit必须由Eddy单独明确批准，当前状态以`STATE.md`为准。

Stop gate：数据语义、公开边界或第三方许可存在冲突时，停止代码与下载工作。

## P1 — 基础代码与可审计运行框架

依赖：P0正式关闭，且用户批准创建bootstrap commit作为branch/worktree基点。

Planned 产物：最小 Python package、CLI 入口、config schema、run manifest、logging、deterministic seed、synthetic fixture、targeted tests。

验收：单条本地命令使用`configs/smoke.yaml`在synthetic fixture上完成dry run；manifest记录config、版本、seed、输入指纹与资源信息；不依赖网络、GPU或真实数据。

Stop gate：若无法从manifest复原一次运行的输入和配置，停止进入数据下载。

## P2 — 服务器验收、数据下载与完整性检查

依赖：P1；VK-LSVD访问方式、条款与provenance审计完成；用户明确批准创建计费实例。

Planned 产物：当前已确认平台A6000不超过1小时的acceptance probe报告（平台见`docs/decision_log.md`）；仅下载到服务器受忽略目录的脚本、checksum清单、文件inventory、失败可恢复日志；仓库只保留脱敏manifest模板。

验收：SSH reconnect、CUDA/PyTorch、FAISS、RAM/disk、Hugging Face吞吐、disk I/O与经批准的平台持久化/恢复验证按runbook验证；取得`ur0.01_ir0.01`所需文件；checksum与inventory可重复验证；Git status不出现raw data或完整embedding。

Stop gate：服务器验收失败、许可不清、checksum不一致、磁盘余量不足或任何大文件进入Git tracking时停止。

## P3 — 数据探查与合同冻结

依赖：P2 完成 MVP 数据完整性检查。

Planned 产物：train-only label coverage 报告、sequence length 分布、week/file-order 检查、catalog/first-seen/OOV 统计、split proposal、data card。

验收：明确记录没有普通 `click`；保留 `click_on_author` 原语义；冻结主 label、history、target、candidate catalog、validation/test 与 OOV 报告方式。

Stop gate：若主正反馈覆盖不足以形成可靠 history/target，先预注册 composite 规则；仍不足则停止 TIGER，项目收口为数据可行性报告。

## P4 — 传统召回基线

依赖：P3合同冻结。

Planned产物：Popularity、无训练content-centroid/nearest-content baseline、按TDD实现的brute-force与blockwise exact top-k evaluator、固定query audit set、latency/RAM记录。

验收：toy fixture 上 brute-force 与 chunked exact 的 top-k 和指标一致；真实 MVP audit set 有可追溯结果；不物化完整 user-item score matrix。

Stop gate：若 evaluator 无法对齐 toy ground truth，停止全部模型比较；两天内不追加 ItemCF 或 SASRec。

## P5 — SID tokenizer 与约束结构

依赖：P3 catalog 冻结；P4 evaluator 通过。

Planned 产物：balanced hierarchical k-means SID、collision suffix、SID manifest、catalog constraint、dead-end/collision audit、toy trie 或等价紧凑结构。

验收：每个 train-observed item 可映射并可解码；SID-level 与 collision-corrected item-level 指标分开；完整 SID map 不进入公开仓库。

Stop gate：若 collision、不可达 item 或内存开销超过预注册阈值，停止 TIGER 训练并先修 tokenizer / constraint。

## P6 — TIGER smoke

依赖：P5 SID 与 constraint 验收；P1 training manifest 可用。

Planned 产物：small TIGER one-seed config、短训练日志、checkpoint 仅保存在受忽略目录、catalog-constrained decode、固定 query smoke 结果。

验收：模型能从 history 生成合法 SID 路径并映射到 item；结果包含 invalid、dead-end、collision、latency、RAM/VRAM；不要求优于 baseline。

Stop gate：若两天内训练、resume、decode 或 item 映射任一环节未闭环，对外只称 benchmark scaffold，不包装为 TIGER 复现结果。

## P7 — MVP vertical slice

依赖：P4 与 P6 在同一冻结合同和 query set 上可比较。

Planned 产物：Popularity、content baseline、SID tokenizer、small TIGER 的统一 run manifests；item Recall/NDCG 与系统指标表；限制与负结果记录。

验收：固定 query audit set 上完成 collision-corrected item-level 公平对比；保存 config、seed、catalog 指纹、资源统计；README 只引用指标唯一摘要。

Stop gate：wall-clock 不足时保留固定 audit set，不承诺全 test；缺任一公平性条件时不得下模型优劣结论。

## P8 — Flagship streaming 与基础设施

依赖：P7 可复核；当前已确认平台 acceptance probe 通过；预算获准。

Planned产物：`ur0.01`单week pilot、streaming ETL、out-of-core catalog/SID、compact constraint structure、ANN pilot、scale re-validation、snapshot/restore记录、旗舰split/final holdout manifest、成本manifest。

验收：在RTX A6000 48GB上完成scale re-validation；冻结旗舰validation/final holdout且final不参与调参；pilot全程保留至少20% RAM/VRAM/disk余量；预算预测不超过100 AUD。

Stop gate：吞吐、恢复、磁盘、显存或预算任一不可接受时，不启动旗舰全量处理。

## P9 — Flagship pure generative retrieval

依赖：P8通过；旗舰split与final holdout已冻结。

Planned产物：trained dense sequential control、纯generative retrieval baseline、ANN full-catalog evaluation、固定query blockwise exact audit、资源与成本报告。

验收：同时报告 ANN recall、exact-audit quality delta、item-level quality、collision/reachability、latency 与 memory；final holdout 未参与调参。

Stop gate：pure generative retrieval 若无可信 item-level signal，不增加复杂 tokenizer，优先进入 hybrid 或记录停止结论。

## P10 — Hybrid dense fallback

依赖：P9 明确 pure generative failure modes；dense control 与 SID route 均可独立评估。

Planned 产物：dense fallback、score/rank fusion、不可达与 OOV routing、校准报告、pure/dense/hybrid 公平消融。

验收：hybrid 在同一 catalog、query、budget 下报告质量、覆盖、延迟、内存与增量成本；收益能定位到 collision、OOV 或 stale SID 子集。

Stop gate：若收益仅来自额外候选量或不公平算力，不进入 dynamic routing。

## P11 — Dynamic SID refresh 与 checkpoint compatibility

依赖：P10 hybrid 对照成立；至少两个 week 的 observed-catalog proxy 与 stale SID protocol 冻结。

Planned 产物：stale、fresh-scratch、checkpoint-compatible / aligned refresh 三组协议；SID churn、old/new reachability、refresh cost、dense fallback 恢复曲线。

验收：多 seed 比较 collision-corrected item-level quality、刷新 wall-clock/GPU cost、SID churn 与 checkpoint resume；不使用精确 timestamp 或真实 publish time 表述。

Stop gate：aligned refresh 若不能同时保持可达性与 checkpoint compatibility，必须报告失败，不改口为成功迁移。

## P12 — Multi-route retrieval

依赖：P11 已定位单路 SID 与 dense fallback 的稳定边界。

Planned 产物：多个互补 SID / dense route、route selection 或 fusion、route-level coverage 与成本消融。

验收：相同总 retrieval budget 下比较 single-route 与 multi-route；新增 route 的独立贡献、延迟与内存可归因。

Stop gate：若提升来自扩大总候选量、泄漏 holdout 或重复 route，不进入 release claim。

## P13 — Release 与公开复核

依赖：拟公开的阶段结果已通过独立复核；provenance / license 审计完成。

Planned 产物：可复现 README、冻结 configs、synthetic test、manifest 示例、结果限制、THIRD_PARTY_NOTICES、secret/data scan 报告；是否添加 `LICENSE` 由审计结论决定。

验收：从干净环境可复现小数据链路；公开仓库不含 raw data、完整 embedding、checkpoint、index、完整 SID map、密钥或公司内部资料；所有数字指向 `docs/experiment_log.md`。

Stop gate：license、provenance、隐私、数据泄漏或复现检查任一失败时禁止发布。


## 已批准双支线准备（2026-10-07）

Eddy批准当前召回项目由Astra协调、传统Sol与生成式Sol独立执行；另一个生成式推荐项目不在本仓库范围。此前全项目串行要求在此范围内调整为每条支线一次一个task、分别验收。传统P4D及其后续阶段定义仍以当前P4D worktree tasks.md为准，其冻结训练不变；公共数据合同、配置与接口由Astra统一。生成式准备可与传统训练并行，GPU仍由传统执行Sol唯一调度，未批准并发训练或资源变更。AGENTS稳定规则不在本轮改写。

### G0 — 生成式召回基础恢复与合同兼容

依赖：已验收P3-D合同、P4B管线、P4C evaluator及既有P5～P7原创SID/TIGER-style实现；不依赖P4E/F完成。

目标：在独立worktree恢复公开源码overlay，审查旧P6硬编码deadline、512 audit及test读取假设；保留旧代码/证据，建立仅CPU synthetic的新协议兼容入口，证明history→SID训练target→受约束decode→item-level评估可贯通。不能把旧smoke称为当前公平训练结果。

允许产物：新src/vkgr/generative_foundation.py、tests/test_generative_foundation.py、tests/fixtures/g0_synthetic.json、docs/g0_validation.md、artifacts/g0-foundation-manifest.json（小型脱敏），本支线STATE/handoff/issue/decision记录；tasks只带入本节指针。复用现有模型、SID codec与evaluator，不重新造公共实现。只在发现缺失行为时按fresh RED→最小GREEN→regression补实现。

验收：synthetic固定seed可复现；严格过去history与catalog/SID身份检查；合法SID映射item，collision-corrected item-level Recall/NDCG与独立toy oracle一致；非法输入明确拒绝；新入口不自动读取test、不复用过期P6 deadline；命令/exit/关键输出和input/source/artifact hashes可追溯。先审查实际接口再冻结具体函数签名；若必须改公共合同或现有核心模块，停止相关实现并提交最小diff建议。

Stop gates：合同或来源冲突、未来信息、预算或公开边界漂移、需要新依赖/真实数据/服务器/GPU时停止相应操作。G0不得SSH、下载、安装、真实训练、重跑旧test、改传统文件、AGENTS、公共合同/config/evaluator、commit/push或创建新资源。最终独立review及Astra fresh验收通过才进入下一生成式任务；后续训练预算尚未批准。

## 生成式召回完成路线（2026-10-08批准，逐阶段执行）

状态：Eddy于2026-10-08明确批准路线开始并下放Sol。G0定点fixture修复已批准；G1～G5按依赖和独立验收逐阶段派工。pilot窗口与正式GPU总时长/stop deadline仍需按下文确认。不是进入原P8～P13，不覆盖传统已冻结预算。本节是阶段计划唯一来源；外部项目及license证据见docs/generative_reference_review.md。现有可见G0 Sol负责后续实现，Astra统一合同/接口/验收；学习Sol不拥有生产文件，短期subagent仅研究/review。服务器仍由传统执行Sol唯一调度；资源窗口未确认前不得提交生成式GPU任务。

**目标：** 在既有VK subset上交付一条可恢复、可评估的生成式召回路，完成与传统召回的公平单路比较、固定总候选预算融合、离线推理及可复核负结果。完成不等于必须优于Popularity，不因不赢而无限调参。

**方案：** 沿用当前like/strict-past/history/catalog/content/split，复用已有SIDTokens、TigerModel、SIDCatalog及item metrics，新增独立consumer训练/评估入口。保留旧producer及历史smoke，不调用会自动读双split的旧runner；新协议在生成式隔离worktree中版本化，原P3-D合同bytes及旧manifest必须仍可从原producer位置验证，禁止修改旧hash来适配新代码。

**本轮不做：** 新数据下载、真实session推断、OneRec复现、LLM/SFT/DPO/RL、HSTU、多目标label重选、dynamic SID、服务上线、新资源、commit/push。RQ-VAE只在核心结果后另提方案；当前不会复制任何参考仓库代码/权重。

### 建议实验范围与预算

- 核心仅一种small TIGER-style方案，现有balanced SID固定版本/seed42，3语义digit+suffix；先验收输入pins，不能为适配实验静默重算旧producer。SID审计不是重新做tokenizer搜索。
- 建议4个seed42配置：LR固定枚举[1e-3,3e-4]，每个LR内d_model[64,128]；encoder/decoder各1层、4heads、FF=4d，现有dropout0，FP32 AdamW/weight_decay1e-4/clip1。此矩阵已获路线批准，机器配置与输入pins须在G1审查冻结。
- 选择validation all-eligible Recall@50，再NDCG@50；epoch平分取更早、config平分取枚举先者。最多30epochs、patience3，每epoch完整validation；选定config再跑2024/2025，42为固定代表checkpoint。核心合计6runs，不挑best seed。
- 唯一预注册额外消融建议：保持已选config，把catalog items按固定seed42的置换分配到同一SID地址集合，保留地址长度/占用/可逆性；该随机SID control跑42/2024/2025，3runs，无额外调参。它改变语义映射，是消融而非公平同features主模型。总计9runs，失败也进入ledger；无额外tokenizer或architecture搜索。
- 四配置与传统每架构四配置是搜索次数对齐，不是等参数或等FLOPs；显式分别记录tokenizer、训练、validation解码、verify、消融、融合成本。若要宣称等compute比较，需另行冻结双方共同compute协议，不能追改传统结果。
- train-only资源pilot所有capacity/microbatch探测合计最多200 optimizer steps及128个固定train-prefix解码queries，不按配置重置预算；覆盖d128、最长50-item history和beam100的保守负载，不用于模型选择且不续作正式checkpoint。候选microbatch[512,256,128]，不足512时精确按样本权重累积到effective batch512，epoch尾不跨epoch填充；若资源仍不满足，STOP。累积梯度与整batch等价性须toy测试。
- pilot与正式run RAM/VRAM/disk可见余量至少20%；pilot前先批准有界运行窗口，pilot后基于实测吞吐计算9runs与verify的时间/空间上限，交Eddy确认总GPU时长及stop deadline，才启动正式队列。此刻无可信完工时长预测；若pilot不足以预测完整validation及verify成本，停在pilot gate，不启动正式队列。任何超限只保留可恢复产物，不静默降query数量/epoch频率或重置预算。
- 固定beam100输出至多100个去重合法item，主@10/20/50；完整catalog合法路径可选不等于exhaustive exact ranking。最终选中seed42可在原固定validation audit上诊断beam50/200（各一次），不根据该结果改正式beam或重选模型；诊断成本单列。
- test保持关闭，直至生成式、传统融合及全部选择冻结并协调共同test计划。旧test26不是pristine holdout；若传统test已执行，显式记录先后和已暴露范围，不阻塞其原授权队列、不冒称同步封存。

### G0收口 — 基础验收前置gate

已批准允许范围：仅在G0独立worktree定点修改tests/test_p4d_v2_verifier_compat.py的冻结producer fixture成员构造，以及自身验证/状态文档；不得修改adapter/生产源码/fingerprint或重新pin。测试须证明新增consumer不污染模拟旧producer，同时缺失/篡改原成员仍拒绝。若不能限定为fixture问题则STOP并重新review。

验收：fresh复现现有失败，再最小修复、完整regression GREEN、manifest replay及独立scope review。阶段Complete只在Astra fresh复核后写入。现有命令见下方，不把作者通过当验收。

### G1 — 新consumer合同与SID输入冻结

输入：已验收G0、P3-D合同、P4B prepared、P5 SID及producer凭证。产物建议：生成式worktree docs/data_contract.md的版本化补充、configs/generative-core-v1.json、artifacts/generative-contract-manifest.json、src/vkgr/generative_data.py、tests/test_generative_data.py、docs/g1_validation.md。旧producer固定bytes从原版本验证，新consumer独立绑定；设计必须先通过Astra审查，不能沿用指向当前合同文件的旧pin导致失效后再放宽校验。

接口责任：generative_data验证prepared/历史producer、逐example严格过去证据、catalog/content/SID同一性；validation reader只加载指定split，test显式关闭。沿用原train samples，不重选label、不过滤OOV/重复/低活跃。训练loss声明为4位置teacher-forced token CE，full vocabulary归一化，与已有beam评分口径一致；不应用双塔past-positive negative mask，不另加temperature搜索。

TDD顺序：pins错误拒绝→prefix边界及重复语义→SID roundtrip/非法suffix→OOV留分母→禁止隐式test读取→新旧source身份独立。真实SID审计全catalog唯一/可逆/可达、非法路径拒绝；记录各层占用、原始碰撞、suffix长度与负担、收敛和train-only内容距离。收敛不足保留为限制，不凭合法性称语义优良；source/feature/catalog任何不一致STOP。

### G2 — 可恢复训练与验证runner

输入：G1冻结consumer合同。新文件建议src/vkgr/generative_training.py、generative_evaluation.py、generative_experiment.py及各同名tests；保留generative_foundation.py为synthetic入口，复用tiger.py及SID codec，不改旧P6 runner/传统训练代码。产物docs/g2_validation.md和synthetic manifest。

接口责任：training消费冻结config/ordered examples/SID，保存model/optimizer/RNG/epoch cursor/累计预算的best与last；evaluation消费checkpoint/queries/catalog/SID，返回逐query候选和aggregate；experiment负责prepare/train/resume/evaluate/verify/select的入口与stage gate。完整大产物只在server ignored目录。

TDD顺序：手算token CE及真实参数变化→epoch无放回和尾batch→梯度累积等价→新进程中断/恢复与连续运行对齐→checkpoint/config/RNG篡改拒绝→toy exhaustive oracle对照beam→shard恢复不漏不重→完整query/OOV分母→test gate→真实输入到输出独立replay。已有teacher forcing/causal/padding tests保留，不重复实现。Loss下降不替代item metrics；best/last与实际输入必须可复核。

评估输出：full eligible、in-catalog、OOV以及冻结train-defined popularity/history/support slices；item Recall/NDCG@10/20/50、unique候选数、invalid/deadend、coverage、latency P50/P95和RAM/VRAM。Beam分数tie最终按item_id；各层保留前缀tie固定。独立toy oracle是exact正确性证据，真实beam只能称近似search。不得物化完整user-item矩阵。

### G3 — 有界pilot与核心真实训练

依赖：G2代码及synthetic验收、用户批准pilot窗口、唯一服务器调度者确认GPU空档；不抢占传统任务。产物：runtime冻结配置、资源/时间预测、每run manifest/receipt、selection manifest、docs/g3_validation.md，指标仅experiment_log。

先执行上述train-only pilot，再确认总运行上限；批准后按固定4+2顺序执行，不因中途validation不理想追加配置。每个run核best checkpoint选择、sample draw、resume连续性、依赖/source/input绑定及真实验证产物；全部6runs完成后Astra批次fresh验收，等待期间不重复轮询。nonfinite、余量不足、身份漂移、预算越界STOP并保留现场。

### G4 — 语义贡献与搜索失败分析

依赖：G3选择冻结；仅执行上述3run随机SID control及固定beam诊断，不重新调参。建议新增src/vkgr/generative_diagnostics.py、tests/test_generative_diagnostics.py及docs/g4_validation.md。

TDD：置换bijection/地址集合与占用不变→选中config不漂移→成对query与指标差异oracle→beam成本边界。输出报告：预测未命中、beam宽度敏感性及可观察的截断恢复、semantic collision与suffix难度、OOV不可达、热门/长尾差异。更宽beam仍可能遗漏高分路径，不能据此完整区分模型错误与search错误；严格归因需要另行预注册有界query的full-catalog teacher-forced评分oracle并计预算，不属于当前必做项。三seed列全值/mean/sample std；小样本仅诊断，不宣称显著性或因果。Random control未独立HPO，结论限于该消融，不能声称所有随机ID方法都更差。

若semantic SID无收益，可交付负结果并给出RQ-VAE有界提案；不自动增加训练，也不因无收益阻断工程完成。RQ-VAE与新label、多目标均另立批准任务。

### G5 — 固定预算融合、离线推理与收口

依赖：G3/G4完成及传统P4F产物已验收；在此之前生成式可独立完成训练，不必等待传统所有阶段。建议src/vkgr/generative_fusion.py、tests/test_generative_fusion.py、docs/g5_validation.md；生成式CLI新增infer与freeze后test子命令，不修改传统冻结配置。

预注册比较：传统最佳单路、生成式单路、传统四路已选融合、四路等权RRF、加入生成式的五路等权RRF。RRF k60，final50，所有多路总400slots：四路各100，五路各80；两组等权比较隔离新增route价值。传统已选weights作为实际reference，不对新五路偷偷追加weight search。追加计算成本/去重候选数/coverage/overlap/新增命中分别报告，不能只看union Recall。线上效用不由离线表推断。

TDD：固定quota/tie/dedup→无GR回退→catalog/query绑定→shard resume→unknown-ID/empty-history独立cohort→test冻结凭证。Infer消费checkpoint/SID/catalog/版本，输出真实item IDs；full SID map、checkpoint和逐query结果不进Git。测试仅在预注册冻结后一次pass，含已声明seed/control/fusion，禁止随后依据test再调整。

完成标准：全部相关tests与独立review通过；Astrafresh核真实artifact/hash/config/query/指标；保存9run及失败ledger、seed统计、单路及融合收益或负结果、系统成本、claim boundary、离线infer示例；STATE/handoff/issue/experiment按事实更新、公开扫描无越界、worktree文件均可解释。README仅指向指标事实源。停在commit/push之前。指标差不代表可跳过验收，也不代表要不断追加实验。

### 验收命令约定与派工

以下以对应实现worktree为cwd，PYTHON为已核Python3.11解释器（不自动安装）；执行时将其路径写入证据。每项先fresh RED→最小GREEN→相关regression；完整suite不得通过删test/放宽身份规避。G1～G5 CLI尚未实现，以下是拟定验收接口，不是已运行证据；具体函数签名在各阶段实现计划中经Astra审查后固定。

```sh
# 现有G0接口
OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" -B -m unittest discover -s tests -p test_p4d_v2_verifier_compat.py -v
OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" -B -m unittest discover -s tests -p test_generative_foundation.py -v
OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" -B -m vkgr.generative_foundation verify --manifest artifacts/g0-foundation-manifest.json --sha256 fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59
# G1以后拟新增tests，按task只运行所属pattern，最终全suite
OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" -B -m unittest discover -s tests -p 'test_generative_*.py' -v
OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" -B -m unittest discover -s tests -v
git diff --check
```

真实CLI拟采用`python -m vkgr.generative_experiment`，子命令prepare/pilot/train/resume/evaluate/verify/select/diagnose/fuse/infer/test。每个子命令显式config、input manifest与exclusive output；verify显式manifest和预期hash；test额外要求冻结receipt。G2交付必须给出实际`--help`、synthetic完整命令、exit与关键输出，不能在真实训练时临时猜参数。将来的worker prompt必须包含本task允许文件、全部输入pins、真实命令、stop gates与证据要求；长期实施不派隐藏subagent替代可见Sol。


### G1实施范围裁决（2026-10-08）

Astra已审查Sol design-v1，批准将G1原计划的data_contract追加替换为：原docs/data_contract.md bytes不变，新增docs/generative_contract_v1.md仅定义token监督/身份绑定/技术gate，共同数据规则仍只引用唯一合同。另准新增provenance/g0-v1固定37个已pin公开历史文件加inventory，来自G0 consumer_source成员与P3-D bound_files及原fixture/manifest的并集；用于旧G0独立replay，不是活跃维护副本。无raw/venv/cache/git或额外成员。其他G1产物及验收不变，本地TDD已派送可见Sol，真实server审计未授权/未完成，G1不得提前Complete。


### 2026-10-08 B方案调度授权

Eddy选择传统优先完整推进P4D→P4E→P4F→P4G后交其最终验收；其离线或未回复不阻塞生成式已批准G1→G5路线按各阶段gate推进。8h不是任务截止；不改变固定实验预算、阶段依赖和Astra fresh验收要求。双方工程准备并行，传统Sol唯一服务器调度，传统优先GPU，在完整run及verify边界安排生成式窗口。生成式pilot及正式GPU预算依原gate实测审核，不视为无限时长授权。不进入OneRec，不commit/push或改变资源生命周期。


### 双线同步推进授权（2026-10-08，更新B方案调度顺序）

Eddy最新调度修订：传统与生成式同时推进，生成式无需等待传统整体完成或用户最终验收；各支线仍逐task由Astra验收。实际GPU并发须先资源检查与有界pilot，传统Sol仍为唯一服务器调度；不因subset或低VRAM直接启动并发，不扩数据、不变更现有P4D队列。


## 两级总目标与自动推进授权（2026-10-08，最新）

### 2026-10-09 Eddy调度修订（优先于下方旧串行依赖）

Eddy明确要求生成式与传统并发推进，不再等待P4E终态。立即开展G3并发admission及有界pilot启动准备，由传统Sol唯一服务器调度；保留原P4E进程及固定实验，不中断、重启或重复训练。须检查CPU/RAM/IO/GPU联合余量，G3提前止损不得危及P4E既有20%guard；旧串行终态本身不再是启动条件。具体启动实现经必要定点验证后由Astra放行，无需Eddy再次在线。

全量S0只读容量/文件inventory与下载准备提前，不再等待subset完成；S1实际下载仍须完整容量账、来源/费用及20%余量gate通过，不删既有成果、不新增计费资源。S2以后全量训练合同与实验不因本次调度授权自动跳过。固定实验/HPO预算、test冻结及不commit/push边界不变。

Eddy授权：先完成当前官方subset的传统和生成式召回；两支线通过协调独立验收后自动进入全量规模准备、下载和训练，不因用户离线等待重复批准。无8h截止。此授权替代此前“不得扩数据”的限制，仅在下述规模gate通过后生效；不授权新计费资源、资源生命周期、commit/push、OneRec或无界实验。旧阶段状态不因此变为完成。

### 第一级：现有subset完整交付

- 传统P4D→P4E→P4F→P4G，依现有合同固定实验预算执行。
- 生成式G1→G2→G3→G4→G5，依本文件已有接口、TDD及验收执行。G5等待已验收P4F。
- 各支线每次只执行一个task，前task须Astra fresh验收；短期独立review辅助，不替代协调验收。
- 双线并行准备。GPU并发先有界pilot，CPU/RAM/IO/GPU整体余量与end-to-end吞吐均通过才启用；不更改既有P4D队列。传统Sol唯一服务器调度，输出exclusive目录和job身份防止重复提交。
- 正式GPU运行前必须冻结有限次数、maxepochs、恢复预算及基于pilot的wall-clock上限。Eddy授权协调在既定实验范围及现有资源内根据实测审查放行，不要求本人再次在线；无法得出可信上限则停止该队列。
- test只在双方约定选择冻结后一次执行；既有test暴露如实记录，不称pristine。性能差允许负结果交付，不追加搜索。
- 完成验收：TDD/regression与独立review、真实输入/配置/产物hash、完整样本及评估分母、多seed及失败ledger、传统与生成式单路和等候选预算融合、离线infer、资源成本及限制、公开扫描和状态交接均通过。用户未回复不阻碍进入第二级。

### 第二级：全量扩展（依赖第一级完成，不提前下载）

全量指固定revision的官方parent全部交互范围及对应metadata/embedding；不把较大subset称全量。以下S任务为规模扩展路线，与历史P8～P13对应关系在S0明确，禁止两套规模计划同时执行。

| Task | 输入与允许工作 | 输出与验收 | Stop gate |
|---|---|---|---|
| S0 全量可行性与冻结计划 | 官方文件metadata、当前实例实际余量、subset实测；仅只读盘点 | 文件/revision/bytes inventory；raw+解压+processed+checkpoint+index+临时产物容量账；下载及训练时间/费用估算；将本表细化成唯一执行计划，固定全量实验次数和依赖 | 已有磁盘/RAM/吞吐不足；流量费用或文件大小不明；需要新付费资源；不明则不下载 |
| S1 可恢复下载与输入核验 | S0通过后，在既有server exclusive目录下载固定官方文件 | resume/integrity/manifest；全部分区schema/ID/embedding覆盖；无数据回Mac或Git；容量始终保留20%余量 | hash/schema/覆盖失败，无法容纳必需数据；不删已有成果腾空间 |
| S2 全量数据合同与流式管线 | 全量输入；延续like/history规则，重新审计顺序、分布、异常、split及采样边界 | 版本化合同和新manifest；每分区及全局计数、无未来泄漏、流式处理与恢复测试、全量prepared核验 | 原subset假设不适用；不可证明顺序；需改变label规则；不得沿用subset旧hash/SID |
| S3 规模pilot | S2通过后的有界train-only probe | CPU/RAM/VRAM/IO、训练/验证/verify吞吐、断点恢复与整体成本上限；配置和有限run队列冻结 | 无法可信估计完整训练/验证；资源不满足；不以小query评估冒充全量 |
| S4 全量传统基准与训练 | 冻结全量合同/队列 | 先基线，再双塔；每run完整预算/配置/恢复/指标证据，Astra独立验收 | 无限搜索、越预算、资源/身份漂移 |
| S5 全量生成式训练 | 同全量合同及新catalog/SID审计 | 版本化全量SID、生成训练和公平比较；必要时与S4按资源pilot结果重叠 | 直接复用subset SID或checkpoint冒称全量已适配；改变共同数据口径 |
| S6 融合、离线推理、最终交付 | 双路验收与选择冻结 | 共同候选预算融合、holdout报告、可复现推理、成本/限制/失败记录 | 未验收结果或缺失来源；停在commit/push之前 |

S0～S6目前是依赖完整的阶段计划，不是假装已经具备全量容量或实现命令。每task开工前由Astra基于已验收前置结果出自包含Sol brief：精确允许文件、接口、输入pins、RED/GREEN测试、真实验收命令、运行预算和stop gates。缺少实测的资源/预算参数不得凭空填入。稳定数据规则只改对应数据合同，数字只进experiment_log，风险进issue_log。

### 协调与运行边界

实现和训练编排由两位既有可见GPT-6.1 Sol/medium负责，Astra负责范围/合同/放行/fresh验收。完成/失败/资源异常/决策时跨对话通知，正常等待仅单负责人有界监控。断联先核旧PID/job/receipt，不盲重启。允许计划内修复并TDD重验；不允许篡改冻结结果或重置失败预算。某支线阻塞时可继续另一支线不依赖它的任务。严重资源风险安全保存并停止受影响任务，不停止/销毁实例。


## EDU-01｜离线学习教材（2026-10-09 Eddy直接授权）

独立文档任务，不替代P/G/S实验路线。唯一作者为教材对话01a11c9a-0f8c-7ae1-93e3-18b4b1211b17，仅写docs/learning/；主协调逐批审阅并派下一批，无需用户重复批准。输入为治理文件、docs事实源及两既有执行worktree代码/文档，只读；输出为完整易懂的分章及合并Markdown、单文件离线HTML、阅读路线、来源映射和审阅说明。首批五章基础与数据，后续模型/评估/工程及全书整合；每批只执行本task。验收覆盖最终目标pipeline、状态标注明确、算例正确、来源可追溯、HTML无外部渲染依赖、目录/锚点/本地文件可用。Stop gate：实质事实源冲突、缺证却需声称实测结果、secret/raw泄露风险；不得训练/SSH/改生产/Git或扩大实验预算。

EDU-01验收结论（2026-10-09）：22章及离线HTML/Markdown/ZIP完成上述文档范围验收，入口docs/learning/README.md；像素预览/打印分页限制见review-notes。此结论仅关闭EDU-01，不关闭P/G/S实验任务。


## COORD-01 协作流程简化（2026-10-09）
Eddy认可分工并授权征询Sol、写规范后恢复。采用完整任务自主执行、异步READY队列、一次独立验收、15分钟兜底；详细唯一规则见docs/codex_handoff.md（本docs内为codex_handoff.md），落实计划docs/superpowers/plans/2026-10-09-agent-workflow.md。两Sol沿原task恢复，不新建框架、不改变科学/资源预算，不commit/push。


## COORD-02 Astra交接与端到端职责（2026-10-10）
Eddy批准仅更换Astra、保留两Sol，更新STATE/handoff并准备新提示词，三方明确职责。范围：主仓AGENTS/STATE/handoff/tasks、codex_handoff协议、decision_log、astra_restart_prompt，旧治理逐byte归档；两Sol只更新自己的角色/交接快照。验收：旧快照hash一致、当前文件无有效冲突、关键证据入口存在、两Sol职责ACK、新Astra接收ACK、旧协调停止写入、vk仍暂停。禁止恢复实验、SSH/训练、commit/push/删除/资源生命周期变化。报告方案不替代科学gate。

## FEAT04-COMBO — 已登记条件组合执行（2026-10-16，当前task）
Eddy直接要求Codex推进F4-T010+C+D+E三seed；科学内容已由decision_log夜间§4登记，不新增名额/规则。允许新增隔离worktree模型/runner/tests/configs/scripts，维护主STATE/handoff/schedule/experiment/issue及Kiro复查提示；既有冻结文件逐字保留。输入复用原train/history/validation/catalog/support/ITEM_STATS_V1。输出三seed组合及各verify、冻结清单、真实只读preflight/CUDA/VRAM/claimed/rejected收据。验收按原计划§1.2，另报组合−E描述性差值；不转正、不换anchor。stop：身份/hash/旧文件漂移、真实preflight或CUDA失败、重复run、rejected或失败attempt不重试、不读test/不删覆盖数据、不改阈值。执行计划位于隔离worktree docs/superpowers/plans/2026-10-16-feat04-combo-implementation.md；具体身份见STATE。

FEAT04-COMBO本轮工程/准入part已完成：freeze与真实preflight/CUDA/VRAM/dry/claimed均通过，三seed实际运行、服务器2921056自动补verify；science结果仍未验收。当前恢复和有限deadline只见STATE/handoff末节，数字/收据只见experiment_log本轮。Kiro/Opus提示已交付到docs/kiro_opus_morning_review_20261016.md。此状态不代表整个实验已完成，也不自动进入M2/A1新SID。

## SUCCESSOR-ENGINEERING-20261016（当前：用户明确并行授权）
Codex负责B1024 A1确认3seed与M1服务器延迟工程、M2只写设计卡。三者独立短任务、文件边界隔离，主仓事实源由Codex统一写；现FEAT原driver不动。科学规则只按decision_log最新晨间§1/§2/§4/§5。允许新工作树入口适配/配置/测试/运维、固定SHA公开分支发布/新目录部署、真实输入只读preflight及工程scratch、CUDA/真实shape VRAM、dry/入队/claimed验收与有限verify driver。G配置只改SID绑定与run_id相关项，其余A1对照同seed；显存重新实测。延迟真实长度分布、50/1000/512/1、CPU1线程/A6000，原始计时与口径交Kiro，不擅自切P anchor；M2不实现/冻结/训练。禁止test/既有真实checkpoint或完整embedding导出、删除/覆盖旧数据、重提旧job、force/main push/改历史、付费、阈值/U1/C9门槛变更。context计数UNKNOWN。各终态/证据/下一动作只见STATE/handoff最新，数字只experiment_log。

本轮工程交付完成：M1正式延迟报告工程核验、M2 DRAFT已完成；B1024工程gate及Eddy批准的一次cuda-v2通过，三真实seed全claimed/真实running，新服务器verify driver存活。工程/接纳part完成，science仍待固定driver收口；恢复/截止见STATE/handoff，数字只experiment_log SUCCESSOR。原v1记录保留，无额外自动训练。

## M2-REVIEW-PACK-20261016（文档交付完成）
Eddy要求专用Kiro/Opus提示，包含M2未决设计、M1正式延迟和下一批接续。产物docs/kiro_opus_m2_review_20261016.md，允许写该新文档及STATE/handoff/tasks指针；输入为既有设计/裁决/指标与收据，32个延迟值和SHA逐项校验。未更改科学规则/生产代码/decision，未SSH、训练或新建timer。下一动作为Eddy将提示交Kiro裁决唯一方案并明确各阶段授权，回传后Codex立即准备，当前M2仍设计边界。

## M2-EXEC-20261016（当前，Eddy与Kiro下午裁决授权）
唯一科学内容与执行授权见decision_log下午§2/§3、FINAL设计卡；执行计划2026-10-16-m2-execution.md。允许隔离M2新模型/runner/config/tests/ops/冻结部署工程与门禁全过后的固定三seed提交及自动verify；完整产物只做一次独立code review。并行补原FEAT组合validation配对统计，仍以F4判；关键规则不由Codex更改。停止：设计偏离、NaN/Inf attempt、review修复需改设计；头未选中仅诊断继续。不读test/旧checkpoint、不ANN、不增加臂seed名额。父Codex唯一server/fact owner；输入/输出/边界/命令证据与截止记录见计划与STATE/handoff。

本轮工程/接纳part已完成：组合McNemar/bootstrap已记账交Kiro；P anchor按裁决更新M1；M2 FINAL→一次whole review及发现闭环→冻结8688f09/部署264files→三真实PF/M1控制封存→CUDA完整shape/replay/g0/新VRAM→正式dry→入队前汇报→固定三seed全claimed/真实running与driver2925994 PPID1/900s/48h。科学三train/verify及M2完整延迟仍待后续，不能将工程part写成研究已验收或自动接其它实验。恢复与待Kiro表以精简STATE/handoff末节为准。

## SID-FIX-20261016（当前，S0 实现 + S1 草稿并行）
Eddy直接授权；冻结依据pipeline卡§2、decision午间/傍晚/傍晚续。只新增隔离工具与synthetic测试，A–D手算、OOV、0分母、平手/重复；一次独立review→扫描/工作分支push→固定SHA部署→真实只读preflight→1CPUqueue诊断→claimed/rejected/终态验收→来源SHA/UTC记账。先不含A1+B1024；三verify齐再另补。S1仅新建冻结卡草稿，固定超参/输入/归一化/回退/筛选判读/门禁，Kiro冻结前禁止导出拟合训练。S0禁止checkpoint/test/embedding读取或改旧run/bundle；只写自己的新out与事实源；原M2/G不重启重提。S0–S5路线命名空间为SID-FIX，不与旧全量S0–S6混用。数字只experiment，解释交STATE待Kiro表。

SID-FIX本轮执行交付完成：S0两批succeeded、全A–D/差分/均值/输入SHA与实际UTC已记账，四旧归档B/C UNKNOWN；S1草稿/Opus提示与S0–S5路线已登记。S0机械分支与D结构零/self-pair歧义均进STATE待Kiro，不下机制结论。S1冻结仍待Kiro，未导出/拟合/训练，不自动接新实验。

## SID-COLL-S1-EXEC-20261016（当前，夜间Opus裁决授权）
S1 FINAL依decision夜间§4/§5直接执行，不再设计review。允许新增隔离SGNS/C2精确导出/新64、128维SID合同/runner/固定6G配置/tests/版本化运维driver；一次whole review后扫描/push工作分支/固定SHA新部署，两个配置分别完成输入健康与验收/SID fullreplay/三真实PF/CUDA fullshape测峰/dry再提交s42，raw verified且hits≥260自动补2024/25。旧checkpoint只固定M1mean42best，Eddy澄清新作业verify可读自己新产物。补充D′、四旧归档别名、A1epoch只读，不改原D和旧产物。stop gates/允许路径/验收与ledger见执行计划。S2/S5不自动，G anchor仍A1T3。
