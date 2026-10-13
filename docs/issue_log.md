## G3-I014 — 并发 packet 的 worker 仍绑定 core00（Closed，2026-10-11）

三个 v1 remote attempt 各约117秒后真实失败，固定错误为 `fixed registration/config/seed/exclusive output`；fresh status 确认 worker exit1、owned groups closed，attempt/spent 不退款。根因是旧 builder 只替换 controller/input，而复制 worker 继续传 config-index0，并只接受 core00、d64/LR0.001。旧包自带 test fresh assertion exit1 复现同一差异。

最小修复 `cache/g3-seed42-retry-candidate-v1/` 获fresh review APPROVE，candidate/report SHA `2c462070…718de`/`f991911e…e067`。三个v2 packet按预注册config严格绑定，错误run/config/output/d_model/LR/seed继续拒绝。Exact admission `5179b883…a3c4` 闭合v1 spent、v2 roots absence、unknown writer与joint资源；随后三v2 roots各唯一SUBMITTED并反绑start/deadline，现由唯一monitor观察。关闭的是原binding工程缺陷，不代表三训练已完成或指标获胜；新attempt已spent且失败不得auto-retry。

## 当前 G3 BLOCKED — 自动审批不接受跨对话授权（2026-10-10）

v4本地已ACCEPT，固定preflight尚未执行。auto-review分别拒绝delegated human转述和read_thread直接userMessage证据，均在CreateProcess前。已直接核主human消息01a123e6-ae7b-7283-a939-9ecbb154be36批准exact input/script/原可信目标，但审批仍认为工具输出不可信；不再尝试、不换法绕过。无SSH/本地进程/新root/额度消耗/actual submit。所有拒绝及授权来源存cache/g3-cost-actual-v4-start-blocked-v1/独立补充文件，旧2member manifest与旧包保持。下一需要Eddy在本G对话直接确认固定v4检查，无需增加预算；formal训练/test关闭。

## 当前 G3 BLOCKED — v4 fresh preflight自动审批拒绝（2026-10-10）

v4已差异验收接受（23成员/18tests），但fresh preflight exec_command在创建进程前被auto-review以unacceptable risk拒绝；理由是直接human授权未明确覆盖本次新stdin payload及既有A6000目的地，敏感外发范围不足。无local进程/SSH/serverwrite/额度消耗/submit/实际cost，不retry/换写法绕过。固定entry/input hash与schema只读核过，包含accepted stdlib source、历史job PIDbirth、small receipt paths/pins和originSHA，不含key/token/rawdata/connection字段。拒绝及具体payload分类证据 cache/g3-cost-actual-v4-start-blocked-v1/review.json SHA a396051b5bf1f66986f7b8c12ec4b9d771d0f4e255116b6afbc2259424c13397。下一需直接human确认固定新preflight stdin及既有可信目的地；原1cost attempt未开始，formal训练/test仍关闭。

## 当前 G3 v4 READY — UUID比较最小修复（2026-10-10）

Eddy明确恢复subset两线并行，无goal/定时器；G唯一owner。新cache/g3-cost-execution-candidate-v4/review.json SHA 6bcc9b9e399f3737547e3f4fc6baaa40706b76c6b007fa97a94a29e6f3ff35e9，23成员；fresh UUID RED→GREEN，相关UUID5+receipt5+transport8=18 PASS。entry pre_backend及fresh preflight使用同canonical UUID gate，拒绝真实不同/未知/坏格式，保留原PyTorch runtime string及model gate。衍生entry/permit/payload/argv/verifier pins完整更新，7保护文件与v3逐byte相同，source/archive/spec/caps/root不变；v3及所有旧包原bytes保持。等待一次差异验收和1次fresh same-scope preflight准入（旧2次不退款）；已授权单次5400s真实成本attempt未开始，formal预算/test关闭。无本轮SSH/模型/submit；联合资源不足才STOP，不要求P退出。

## 当前 G3 窗口收尾 — UUID表示问题已真实定位（2026-10-10）

截止后的human明确授权单次UUID诊断已完成，实际transport exit0/非timeout；canonical_equal/name_equal/single_gpu均true，raw_equal=false，冻结uuid_text与当前nvidia_gpu_prefixed_text表示不同。旧origin/input/UUID/pins不改，不能将raw mismatch称真实硬件变更。本次只读，无serverwrite/model/submit，实际cost attempt未开始。独立实际packet cache/g3-runtime-identity-diagnostic-v1/review.json SHA 2154a771b099d4e92728ad25558b9eb221b16544df48fe68d1bfa5734cca670c，3成员及source/proposal/script pins完整。诊断进程已actual wait结束；没有本轮仍运行工具。

Astra明确6h窗口已结束，停止新增v4实现/实际成本启动；本Sol未开始v4修改，只封本次收据/快照。下一需明确恢复后本地最小版本化UUID比较修复、保留原PyTorch runtime string并TDD/原reviewer差异验收；accepted v3原raw gate不绕过。formal预算/test关闭，资源余量与完整cost尚无新证据。

## 当前 G3 本地identity调查READY（2026-10-10）

cache/g3-runtime-identity-investigation-v1/review.json SHA 0b27daeb828045a11e8e40fb23f0bab5ead3344925c5641a0b6d37de0fd38e98，8成员。证实原UUID来自PyTorch props.uuid str链，当前entry/preflight来自nvidia-smi CSV raw比较；没有共同canonical表示合同，actual canonical equality未知，不能将raw unequal无条件定性硬件已更换。旧origin全hash及historical predicate/PyTorch pilot证据保留，不判placeholder。私有配置与known-host只读核；同alias但缺历史resolved host/key快照，不证明同host。不第三次SSH，原包不改。新scope候选1次30s/60s/8KiB只读原origin+GPU identity、脱敏canonical equality，synthetic3PASS；仅诊断不自动启用绑定/submit。Astra需一次准入；human1实际attempt未开始，2/2旧readonly耗尽，formal预算unknown。

## 当前 G3 BLOCKED — 已定位GPU UUID漂移（2026-10-10）

已完成获准2/2次只读preflight，第2次actual transport exit1、非timeout，gpu_uuid_equal=false；origin/spec/contract hash、runtime schema、single GPU及name_equal均PASS。当前GPU UUID与冻结runtime origin不符，不导出UUID、不推断更换原因。原wrapper漏origin SHA绑定已本地fresh RED1→GREEN1修复，但不是此次实际ValueError唯一解释，实际失败具体为UUID门。新诊断独立packet cache/g3-cost-actual-v3/diagnostic-v1/review.json SHA 6a635cb2177c6fc0c36764bf1bb0e129572876f6b3ae47be2e103160d32fae2c，8成员，原genericSTOP/v3固定包不改，11保护成员hash保持。资源阶段未到达；0submit/firstwrite/root创建/模型/训练。readonly额度用完，不repin/第三次SSH/重试。单attempt human授权保持但身份门未满足；成本/6+3正式预算unknown。下一Astra决定有界版本化runtime身份复核范围与合同兼容，不能直接改pin绕过。

## 当前 G3 BLOCKED — 已授权成本测量的启动前runtime gate（2026-10-10）

Eddy明确批准1次有界真实成本测量，G唯一owner；所有4项旧job责任均已接收并ACK新Astra，最后pilot补件15a64…c92/5引用source核过，parent exit null保留。已实际执行第1次限定只读preflight，非timeout、transport exit1，STOP runtime_origin/ValueError；具体子gate未知。按控制流，旧receipt pins、当前旧PIDbirth/owned groups absent及新root不存在已过；资源阶段未到达，不能声称20%余量通过。没有实际submit/server root创建/成本模型/updates/正式训练。未第二次preflight，不盲retry/repin；原失败spent不变。真实cost及正式预算仍unknown。完整实际证据 cache/g3-cost-actual-v3/review.json SHA c3b1908a6f94d179876c54a49b5f0651fbb62201846f58616c1ece20ccee0594；下一Astra处理合并缺口，必要窄诊断须明确授权，不绕过runtime gate。

## 当前 G3 v3 READY（2026-10-10）

唯一review的Important收据完整性缺口已修：固定7phase、同clock时间关系、7phase完整typed I/O；缺失/unknown不补0，rel1e-9/abs1e-6秒仅容float误差。新cache/g3-cost-execution-candidate-v3/review.json SHA 5ba31c7ca7c98cf4165a141e540bb60b3a2e1b2d792753a92e2dee73e0fb3924，18成员；fresh RED20失败→GREEN5 receipt，fresh相关receipt5+transport8=13 PASS，diffcheck0。原v2全部17成员及packet SHA不变，v3的11保护成员逐byte复用；生产/entry/transport/measure/accounting不改。G owner文案及同参数绑定说明一并修订，旧job先明确接收。待原同一reviewer只复核blocking差异；新attempt/资源准入/旧job摘要与接收/实际最终bindings仍未闭合，无SSH/实际model/训练/新attempt/退款。角色只引用主协议；本地恢复不代表正式训练获准。

## 当前 READY — G3 新真实成本执行候选（2026-10-10；覆盖下方历史）

Linux synthetic 生命周期 gate 已由 main 接受，实际证据 packet 1fdcb3b7fdb6d95d8498fa19c49d4ad5f167e25f4a3d7ec97a003c73dbcdf3e1；controller 自身 actual exit 仍 null，不称 exit0。新完整本地候选 cache/g3-cost-execution-candidate-v2/review.json SHA d365411ca0520704fafd229e0e8d0b8b30ed1f1c0d5a837f83dfdd1b9eba7a75，17成员，fresh receipt3+transport8=11 tests PASS；entry/transport与已验收 revision3 完全同字节。新增 offline receipt verifier 拒绝 UNKNOWN/非零exit/未闭组/不完整计数，组件验收不自动打开正式训练 gate。

固定真实入口、六文件 payload、exact argv 模板及收据验收齐备；pending payload SHA 8c7874f2e8a4db309d55df62525ccbf8d859e52afdc70fe3bb5a2450a87364a2（1903863bytes），未批准 permit 在 server write 前拒绝。新attempt候选为5400s firstwrite/5040decode/0updates/0resume/新增1GiB observation STOP/1attempt。下一 main 独立验收并放行新 exact attempt，P1 server-local fresh 核 runtime/input/resource/exclusive root 后冻结最终 permit/payload/argv。旧失败attempt已消耗，禁止重启或退款。真实成本尚未通过，正式6run及G4未启动；无本Sol SSH/server/model/训练/commit/push。通知状态见 handoff。

密级: 公开

### G3 launcher问题及关闭证据（2026-10-08）

- G3-I008（Closed，本地）：最早lifecycle测试在sandbox ps不可用时已spawn fixture，留下4个owned members；实际按5个精确已核PGIDs只清理本轮fixture，fresh检查0 live，不触传统/server。随后observer preflight fresh RED（不应spawn却尝试spawn）→GREEN，任何observer不可用在Popen前停止。失败未隐藏；后续lifecycle必须escalated ps并在finally fallback前独立核无live子孙。
- G3-I009（Closed，本地）：failed parent的孙进程继承stdio让launcher等到deadline；fresh RED实测15.620s不小于3s。poll actual exit并在parent结束有live descendants时立即STOP，TERM/KILL仅owned session；GREEN6tests4.072s，四类failure/budget/observer cleanup独立ps通过。
- G3-I010（Closed，本地）：exact env keys漏原v3 GIT_DIR，首target v1四ERROR；保留冻结声明补一致性，作者与Astra fresh PolicyTest通过。prepared/statistics digest无newline、capacity/restore映射误用d64由实际frozen canonical/_config RED3→GREEN6关闭，未改source44。
- G3-I011（Closed，本地）：review实测合法OOM降batch因reserved与completed timing误等被拒。failed仍占预算且无伪timing，原ordered attempts及独立failed ledger核reserved=completed+1/pending；同pilot child固定所有artifact/ledger-tail snapshot，外部resource SHA串接验收重算。独立6tests0.940s exit0及tail删除复现关闭Important。
- G3-I012（Closed，本地）：ambient env继承可能夹带P4E/LD等非固定变量；sentinel真实CPU child fresh RED→显式effective env GREEN，原reviewerfresh2tests1.629s exit0。64MiB reserve舍入依据需要finite<=4KiB fragment，新增machine allocation gate RED→最终28tests13.350s PASS及review1test0.011s，未知/>4KiB STOP。
- G3-I013（桥梁可见性，已用安全替代）：跨chat进度send被auto-review以当前可信上下文缺明确授权拒绝，未送达、未重试或绕过；Astra实际read_thread确认已读取并要求以主动读取/wait为替代，不重复要求Eddy授权。最终本地固定包在本chat交付供读取。

上述只证明CPU工程/指定review scope。parent GPU self peak/child RSS high-water/whole-device sampled VRAM有明确归属限制，未测child isolated GPU peak。最终launcher scope无剩余Important；actual server pipeline/资源/训练预算仍待Astra fresh核，不写G3 Complete。

### G3 review修复进行中（2026-10-08）

- G3-I006：部署提取stdout曾含完整shared statistics binding，Astra指出与先前P4E导出边界冲突；v3改binding hash/knownpins/lineage布尔/runtime白名单，完整binding仅server-local exclusive spec；独立synthetic mock fresh exit0确认无完整文档或private sentinel、各漂移/角色/既存gate通过。Closed（操作草案），未执行server probe。
- G3-I007：新operation root误嵌套旧G1根；v3修为sibling exclusive root，review核通过。历史Macpaths由Astra明确限此private部署允许原bytes带入，不改capsule，不扩大发布许可。Closed（scope/path草案）。

- G3-I003：Closed（本地telemetry缺口）；实际decode/worker update/serialization/accounting测量由原reviewer确认，真实GPU成本审批仍待。
- G3-I004：Closed（本地disk计费）；1-byte genesis RED与byte-sum oracle、self-event/accounted ledger、实际大manifest及checkpoint writer、超cap写前拒绝均通过；原reviewer6tests6.348s定点关闭。
- G3-I005：Closed（代码可移植性）；platform-default exclusive tempdir，owned路径resolve；reviewer用TMPDIR与audit hook实际train验证通过。Linux/CUDA物理环境仍未检查。

# Issue Log

## 当前边界

P2已完成批准范围的服务器下载验收；P3真实数据审计与合同冻结完成，尚无模型训练或效果评估。当前整体gate以STATE为准。

下列项目都是开放风险，不代表问题已经发生或已经解决。

公司内部信息、数据、schema、代码与实验结果不得进入本项目。

## 状态定义

`Open`：风险已登记，尚无关闭证据。

`Investigating`：已有 owner 与检查动作，但结论未冻结。

`Blocked`：阻止后续 gate。

`Closed`：关闭证据已保存并可复核。

## Open Issues

| ID | Risk | Status | Impact | Next evidence | Closure criteria | Closure evidence |
|---|---|---|---|---|---|---|
| I-001 | `like` label 覆盖已train-only审计并冻结。 | Closed（工程smoke范围） | 可能无法形成足够 history/target，迫使定义 composite。 | 仅用 train 统计 user/item/event coverage、序列长度与 week 稳定性。 | 冻结 `like` 或 composite policy，并保存 coverage report 与 policy hash。 | docs/p3_validation.md v3 train/freeze SHA；artifacts/p3-data-contract-manifest.json。 |
| I-002 | `ir0.01` 的含义与 sampling 机制待查。 | Open | 可能改变 popularity、catalog coverage 或反馈分布，影响 MVP 外推。 | 核对 data card、subset manifest 与下载后分布。 | 明确生成规则，并在 data card 中写清可比性边界。 | 待补。 |
| I-003 | `timespent` 可能是聚合值，时序语义未知。 | Open | 可能把未来或聚合信息泄漏进 target-prefix feature。 | 核对 schema、粒度、单位、生成逻辑与同 user-item-week 重复。 | 证明 query-time 可用；否则从主任务移除，只保留审计。 | 待补。 |
| I-004 | `duration` 的粒度、单位、缺失与版本未知。 | Open | `timespent / duration` 可能失真或除零。 | 做 dtype/range/null/zero 与跨 week 稳定性审计。 | 冻结 clipping、missing/zero policy 并保存 feature manifest。 | 待补。 |
| I-005 | Flagship `ur0.01` 公开树可能没有独立 official test。 | Open | 调参与 final holdout 可能混用。 | 下载前核对公开结构；下载后保存 partition manifest。 | 冻结 train-tail validation 与一次性 final holdout，或采用明确的官方 split。 | 待补。 |
| I-006 | Thunder A6000 实时库存与 snapshot restore 未验证。 | Open | 两天窗口可能因实例不可用或 artifact 丢失失败。 | 执行 ≤1h acceptance probe：stock、SSH、CUDA、FAISS、吞吐、磁盘、reconnect、snapshot restore。 | 全部 gate 通过并保存无 secret 的 probe report；否则切 fallback。 | 待补。 |
| I-007 | Dataset、论文参考实现与未来依赖的 license/provenance 未完成。 | Open | Public repository 可能错误复制或分发受限内容。 | 建立 third-party provenance 清单，逐项记录 source、license、使用方式。 | 所有发布内容 provenance 明确；首次发布前再决定 repository LICENSE。 | 待补。 |
| I-008 | Full-catalog score 与 embedding/index 内存可能超限。 | Open | MVP 或 flagship 可能 OOM，或被迫缩小候选后误称 full-catalog。 | 用 manifest 规模做 storage estimate；运行 blocked top-k 与 ANN smoke。 | 峰值 RAM/VRAM/disk 留 ≥20% 余量，且 evaluator 不物化全 score matrix。 | 待补。 |
| I-009 | Exact evaluator已通过P4批准MVP验收。 | Closed（批准MVP范围） | 未来collision correction与旗舰算术仍须对应阶段验证。 | 后续SID/item与旗舰exact audit。 | P4 toy brute/blockwise/canonical-tie IndexFlat与真实replay一致。 | docs/p4_validation.md v2 source/run/verify hashes及Astra正式关闭；见文末I-009。 |
| I-010 | 大 catalog trie representation 未定。 | Open | Python nested dict 可能导致 RAM 爆炸与 decode latency 不可控。 | 比较 compact array/CSR/automaton 等 representation 的 build time、size 与 lookup。 | 选定 representation，在 flagship pilot 中满足 memory/latency gate。 | 待补。 |
| I-011 | 两天 MVP 时间风险高。 | Open | 数据清洗或 TIGER 训练可能挤占 evaluator 正确性工作。 | 先完成合同、toy evaluator、Popularity/content baseline，再启动 small TIGER。 | 两天内 P1～P7 固定 audit query 闭环；否则只交付 scaffold，不升级为 TIGER benchmark。 | 待补。 |
| I-012 | official-order假设下prefix已冻结并独立核对；隐藏timestamp残余见I-030。 | Closed（批准MVP范围） | 可能产生 future leakage 或无法定义唯一 next positive。 | 审计 partition/order keys、稳定排序与重复记录。 | 保存 order proof；无法证明则阻塞主任务。 | docs/p3_validation.md固定README/physical digest/Astra逐query oracle；残余I-030。 |
| I-013 | SID collision suffix 对 vocabulary 与 decode path 的影响未知。 | Open | SID-level hit 可能不能恢复唯一 item，或 suffix 增加稀疏 token。 | 报告 collision group size、suffix length、reachable rate 与 item-level correction。 | item mapping 可逆，指标分离，且无静默丢 item。 | 待补。 |
| I-014 | 固定 query audit set 可能不代表完整 test 分布。 | Open | 两天结果存在 sampling bias。 | 在 train/validation 可见统计上按 history length、popularity、first-seen 状态分层抽样并冻结 hash。 | audit set 构造可复现，覆盖预注册 slices，结论明确限定为 audit set。 | 待补。 |
| I-015 | `AGENTS.md`未落位。 | Closed | Codex自动发现规则与P1启动条件不成立。 | Eddy手动创建`AGENTS.md`。 | `AGENTS.md`存在并已读取。 | 本轮已读取受保护的`AGENTS.md`。 |
| I-016 | `AGENTS.md`曾混入易变事实。 | Closed | 每次对话占用过多context并产生漂移风险。 | Eddy手动落位thin规则；Kiro拆分STATE/handoff/tasks并复核。 | AGENTS只含稳定规则和指针；当前/历史/路线各有唯一源。 | Thin规则已落位；STATE/handoff/tasks完成二次职责收敛。 |

## 关闭纪律

关闭 issue 必须填写具体 artifact、hash、run ID 或公开文档链接。

“命令成功”不能单独作为关闭证据。

效果不佳不是关闭风险的理由；负结果同样需要 artifact evidence。

涉及 label、split、catalog 与 evaluator 的 issue 未关闭前，不发布效果 claim。

## 公开来源

VK-LSVD data card: https://huggingface.co/datasets/deepvk/VK-LSVD

TIGER: https://arxiv.org/abs/2305.05065

LIGER: https://arxiv.org/abs/2411.18814

SID staleness: https://arxiv.org/abs/2604.13273

Collision-faithful evaluation: https://arxiv.org/abs/2605.25330

## P1环境与实现复盘（2026-10-07）

### I-017 — Python/uv环境前置条件（Closed）

- 预期：P1有Python 3.11与uv可用，项目测试使用当前worktree环境。
- 实际：主协调接力记录系统Python 3.9.6，最初缺uv/Python3.11；主协调已按用户授权安装。执行Agent没有冒充自行安装。
- 影响：直接用系统Python无法满足requires-python，不能把缺依赖当TDD RED。
- 复现证据：本轮fresh运行已批准Python路径`--version`为3.11.17，`python3 -m uv --version`为0.12.23；当前`.venv/bin/python`复核3.11.17/PyYAML6.0.2。详见[p1_validation.md](p1_validation.md)环境准备。
- 排查与排除：不是项目逻辑失败；.local/bin不在PATH，使用明确路径；系统Python只启动uv。
- 已证实原因/未知：初始环境版本与工具缺失来自接力记录；本轮确认安装后版本。没有重演或伪造初始安装输出。
- 处理tradeoff：锁定3.11范围，不改shell配置，不再次安装Python/uv。首次依赖安装有网络，测试/CLI直接离线执行。
- 验证/学习：当前worktree独立.venv与实际uv.lock是关闭证据；面试时区分环境失败、无效RED与功能失败。

### I-018 — worktree metadata/cache/DNS的sandbox边界（Closed）

- 预期：已授权branch切换、必要依赖安装和真实lock生成可执行。
- 实际：git switch默认exit128（共享metadata的index.lock不可写）；uv默认cache不可写exit2；独立tmp cache解决写权限后，PyPI DNS仍失败exit2。
- 影响：阻塞环境准备，不代表config/runtime问题；未算TDD RED。
- 复现证据：validation环境记录和本对话实际工具输出；git switch经escalation exit0；uv lock/sync在独立tmp cache与获准联网escalation后exit0，installed pyyaml==6.0.2，uv.lock实际产生。
- 排查与排除：基点/branch冲突已排除，两者均df32a1a；未改Git配置、main文件或shell配置。cache写权限与DNS分别出现，不能混为缺包。
- 原因/未知：默认sandbox写入范围与网络限制；未调查OS网络全局状态，因为本次escalation已成功。
- 处理tradeoff：只为获准动作escalate；cache留独立tmp目录，运行与测试不调用uv，不借用其他worktree依赖。
- 验证/学习：fresh版本、lock与当前.venv元信息复核；worktree代码隔离不代表Git metadata物理上也在worktree内部。

### I-019 — 拒绝应发生在写输出之前（Closed，限定preflight）

- 预期：不合规output路径、已有目录、Git provenance失败在写入前拒绝。
- 实际：R05旧实现会覆盖已有result；B01虽最终报错却已在不合规目录写文件；B02 Git失败会先留下result。
- 影响：exit非0也可能产生副作用或误导性半成品。
- 复现证据：[p1_validation.md](p1_validation.md) R05/B01/B02 RED→GREEN及对应tests；测试只损及自身临时fixture，没有覆盖用户产物。
- 排查与排除：不是manifest hash或Git基点问题；原因是mkdir/result写入排在路径/provenance校验之前。
- 原因/未知：验证时序已证实。运行中输入并发变化、突然I/O失败仍可能留下本次新目录的部分产物；P1不实现transaction/自动回滚。
- 处理tradeoff：路径、fixture、Git与source hash preflight前移，mkdir exclusive、文件x mode；不自动删除任何失败证据，重跑选择新ID。
- 验证/学习：测试断言原字节不变及output不存在；面试时解释“报错正确”与“没有危险副作用”是两个验收条件。

### I-020 — fixture symlink与测试误通过（Closed）

- 预期：批准的fixture filename不能绕过repo-local人工输入边界。
- 实际：runtime最初未对固定fixture路径执行symlink校验。B03第一次GREEN实际因macOS /var临时路径别名提前拒绝，未触达目标行为。
- 影响：未经修正的测试会隐藏fixture可引用外部文件的问题。
- 复现证据：[p1_validation.md](p1_validation.md) B03意外GREEN说明、canonical temp root后的真实RED、修复GREEN；tests/test_boundaries.py。
- 排查与排除：使用内容合法的外部人工fixture，排除schema错误；canonical ROOT后仍触达Git sentinel，证明遗漏fixture confinement。
- 原因/未知：config路径与output有safe_path，fixture只拼接ROOT；ROOT别名使第一版测试提前退出。并发symlink替换不在P1单进程静态文件的保证范围。
- 处理tradeoff：runtime和build_manifest固定fixture路径也调用safe_path；不改变原fixture或扩大真实数据访问。
- 验证/学习：修正后先有assertion RED再GREEN，output不存在；不能把“测试通过”直接等同于“覆盖了预期分支”。

### I-021 — 人工oracle算术错误（Closed）

- 预期：synthetic加权和使用独立手算期望。
- 实际：R03最初误写229；正确是3×2+7×1+11×5+19×4=144。
- 影响：若未纠正，会把正确实现误判为错误。
- 复现证据：[p1_validation.md](p1_validation.md) R03手算修正、重新RED与GREEN；当时生产seeded计算尚未实现。
- 排查与排除：权重与输入人工复算，不用生产helper构造oracle；并非随机数版本或实现bug。
- 原因/未知：测试编写算术错误已确认，无模型或真实数据影响。
- 处理tradeoff：先纠正oracle再fresh RED，未调整生产代码迎合错误数字。
- 验证/学习：literal144与实际权重匹配，换seed会变；测试oracle同样需要审计，不能以测试权威代替推算。


## P2首轮问题（2026-10-07）

### I-022 — 当前费率无法在无凭据接口中核实（Open）

- 预期：首轮30分钟/新增计算费用US$1上限可在操作前预测。
- 实际：按服务器指南读capabilities并仅筛选费率allowlist，返回rate_fields空；当前计算、存储、下载流量费均Unknown。Exit0只是查询成功，非预算门Passed。
- 证据：`p2_validation.md`费用门命令与脱敏JSON。
- 处理：未读取/使用计费API credential；立即停止GPU、clone、下载等后续远程动作，完成本机输入和公开元数据核查。未停止实例，空转仍可能计费；实际扣费未知。
- 关闭条件：Astra核实当前控制台费率及具体动作成本，确认批准时限和US$1上限内可控；补来源、UTC与费用推算，不照搬旧截图价格。

### I-023 — 无独立volume时持久化/恢复尚未闭环（Open）

- 实际：capabilities本次报告workspace_is_volume=false；guide说明无volume的容器文件会在recycle/destroy后丢失。
- 证据：`p2_validation.md`存储属性输出及指南读取记录；另fresh核对官方 https://docs.vast.ai/guides/instances/storage/types 。Container storage在stop时保留，destroy时丢失，stop仍计storage费；volume绑定同一host且独立计费，不是跨host容灾。这是属性/文档证据，不是恢复实测。
- 处理：未创建marker、备份、snapshot或执行stop/destroy/recycle；原有恢复gate保留。无工作盘free容量/quota结论。
- 关闭条件：具体持久化与服务器外备份方案经用户批准并实测恢复、hash一致，记录保留/不保留内容、费用与耗时；不能凭SSH成功关闭。


### I-022续执行 — Unknown但非阻塞

Eddy撤销费率gate及原US$1费用门，D-051为明确授权证据。本轮继续已批准搭建与下载；费用仍Unknown，不能伪造费率或扣费，也不再用此未知项阻塞。历史暂停记录保留，不当作当前gate。

### I-024 — 现有Python不满足P1与FAISS GPU未提供（部分解决）

现有torch环境Python3.12.14、PyYAML6.0.3，不满足P1范围。使用uv建立项目独立Python3.11.16/PyYAML6.0.2，31tests与smoke/verify通过；未污染现有环境。FAISS原缺失，独立cache环境装CPU1.15.1，IndexFlatIP toy通过；GPU数0，原runbook GPU可见门尚未通过，不冒充GPU FAISS。证据见p2_validation续执行节；GPU验收口径/后续方案由Astra裁决。

### I-025 — 短probe不能替代quota/长任务/完整恢复验收（Open）

工作盘statvfs显示足够空间，但quota工具未提供；64MiB direct-I/O通过不保证持续吞吐，reconnect文件保持不保证后台长任务resume。无volume的stop/start文档保留不等于本轮已恢复实测。Astra审查具体后续方案与预算，危险恢复动作仍明确审批；本輪未自动stop/destroy/new volume。


### I-024续执行 — GPU FAISS环境问题Closed（限定验收）

官方pytorch GPU FAISS1.11.0精确py3.11/cuda12.1 build与libfaiss在独立conda cache环境安装；GPU count1，8×8 synthetic IndexFlatIP CPU/GPU及独立oracle相符。Sol与Astra各自fresh验证exit0，artifact/UTC见p2_validation。GPU wheel1.15.1/Python3.11初始解算失败及中间未安装的大依赖方案保留；没有改driver/torch或以CPU通过替代GPU。

### I-026 — tmux首版未生成日志（Closed：第二版限定probe通过）

首版tmux启动exit0但无日志，后续FileNotFoundError exit1；stderr未保留，根因不能单凭猜测定为引号。第二版为明确synthetic文件，先compile/py_compile再运行、保留stderr、隔离socket/config。首SSH退出后8步/DONE完整、stderr0；Sol和Astra各自复核hash一致。证据见p2_validation；只关闭这个probe失败，不保证任意训练resume或实例恢复。

### I-025续执行 — tmux已覆盖，quota/恢复仍Open

小型synthetic断线保护通过，但独立quota工具不可用、overlay无可见quota flags不能证明无隐藏限额。46文件冻结bundle只准备，不代表backup或恢复已执行；实例/host-loss gate继续Open，须Astra向Eddy审查。


### I-023/I-025目录恢复覆盖更新 — 完整gate仍Open

批准的server→Mac→server小bundle在全新checkout恢复46/46hash并独立重建P1环境，31tests及新旧manifest通过；此有限恢复项Passed，完整实例/host-loss/GPU环境/原始数据重获未演练。不可把partial覆盖写成完整Closed，待Astra复核裁决。

### I-027 — 恢复成功后本机元数据记录KeyError（Closed：记录补正）

local validation使用exec覆盖expected变量，远程恢复已完成但本机末尾记录archive_sha256发生KeyError，首轮exit1。补正本机记录、不重跑恢复；后续46hash、fresh环境/31tests/smoke/旧manifest通过，报告与日志保留。临时验收orchestration未进入生产代码，失败不隐藏。


### I-023/I-025 Astra裁决 — 已批准P2恢复项通过，剩余风险保留

Astra独立Mac包/服务器46hash、HEAD、31tests及新旧manifest复核通过。Eddy批准的是目录级backup roundtrip/P1环境重建，故本次P2恢复验收项Closed（限定范围）；旧条目更强的实例/host-loss条件不自动扩充为本次授权。无volume及host-loss/stop-start/GPU环境/raw重获未测风险、隐藏quota未知仍Open，P8获准后再验。不能将验收通过解释为风险已消失；证据读p2_validation最新Astra验收节。

## P3复盘与剩余限制

### I-028 — 新服务器源码包环境漏项（Closed，限定P3测试环境）

- 预期：新唯一P3目录能fresh跑P1与P3 synthetic regression。
- 实际：首次包未带outputs容器和.python-version，连续暴露FileNotFoundError；tar另有macOS xattr warning。所有失败logs保留在P3 cache，train/holdout尚未执行。
- 原因：原创源码包清单未包括P1运行fingerprint的公开环境文件及test临时目录容器，不是数据或P3行为失败。
- 处理：补公开.python-version/.gitignore与新outputs目录；Git对象只读使用P2 object context。后续code-v2采用--no-xattrs独立快照并fresh完整regression。
- 证据：docs/p3_validation.md、server cache/tests-environment-fixed.log与tests-code-v2.log；无覆盖旧运行产物或改Git配置。
- 学习：目录级代码包也要覆盖运行metadata，不把environment error记成RED。

### I-029 — `irX`公开生成定义不完整（Open）

- 固定revision README列出ur0.01_ir0.01，但定义段仅解释urX/ipX；不能凭名字宣称ir是随机item采样。
- 主train/holdout可按固定文件审计；结果仅适用于此subset，不外推全量分布、公平随机抽样或无偏popularity。
- 下一证据：作者官方生成代码/明确文档；未取得前保持未知，不复制未审计第三方代码。

### I-030 — 顺序证据仅官方声明与physical读取（Open，已限定协议）

- 官方固定README声明week内timestamp顺序并给保序subset join示例；实际12列无timestamp。
- P3验证physical顺序两次digest一致、schema/week集合与无重复pair；不能独立证明隐藏timestamp单调或此subset逐行保留全量parent顺序。
- 当前policy显式使用official-order assumption；如协调review不接受该证据强度则阻塞后续训练，不能伪造timestamp补门。
- 证据与hash见p3_validation及frozen manifest；后续需作者澄清或公开事件顺序键才可关闭全部残余风险。

### I-031 — 独立review发现policy与producer绑定缺口（Closed，限定实现与复查）

- 原min/max validator只检查范围，freeze能接受不同于train硬编码的history规则；train weeks的bool在Python equality下可误通过。
- 原train audit没有实际producer信息，freeze记当前hash与硬编码Git身份，holdout没有提前验证源/lock/库版本。旧audit因此可能被新代码消费。
- 独立Sol review由Astra安排；已fresh RED复现并最小修复，详见p3_validation fix round1。此前v1/v2 audit保留且明确不可freeze；未读取holdout或训练模型。
- 历史fix1时点：46tests通过；独立Sol scoped re-review三项addressed，独立拒绝复现与source fingerprint见p3_validation。当时实现风险关闭，v3真实audit与整体P3验收尚未完成。
- 后续闭环：v3真实train audit、freeze及holdout均exit0；Astra独立producer/freeze hash核验及逐query target/history/OOV核对通过，详见p3_validation的v3与holdout章节。最终独立Sol review public manifest/hash/producer亦通过，无source/spec新阻塞；整体P3最后文档验收仍由协调者裁决。
- 兼容规则只允许CPython3.11 patch跨Mac/Linux差异，源/lock/Git基点与精确库版本保持一致。

## P3数据风险收口证据

I-001按批准工程smoke范围关闭：v3 train audit与frozen policy manifest/hash见p3_validation，支持门在读取holdout之前通过；未声称统计效力。I-012仅在批准official-order assumption下关闭：固定README hash、physical两次digest及Astra独立逐query prefix/target位置核对通过，残余hidden timestamp风险仍由I-030记录。

I-003/I-004：timespent/duration只做官方单位与dtype/null/range审计，不进入当前label/history或模型feature，主任务不存在这两字段引入的feature leakage/除零；未来使用时须重新审计注册。I-002/I-029抽样生成定义不完整保持Open，不猜ir含义。I-007完整发布provenance仍留P13。


### I-032 — P4 GEMM block shape导致相同向量tie失真（Blocked）

独立review在NumPy1.26.4以seed42生成2x64 query、1x64 item归一化，再重复item三次、IDs[3,2,1]、k1/qblock2/iblock2，brute输出[[1],[1]]、block输出[[1],[2]]。float32第二query score存在不同最后位；不是candidate或policy改变。简单2D literal fixture未覆盖BLAS reduction shape差异。停止v1效果比较，保留v1源码与run为invalid candidate；下一步固定每pair reduction、fresh反例RED→GREEN及跨block regression，然后新v2真实run与独立验收，不用epsilon修改tie定义。


### I-032 fix1 closure（Closed，限定source/scoped review）

改为float32 feature输入、固定float64 einsum dimension reduction，反例fresh RED→GREEN，多block regression通过；独立Sol额外20组mixed64D/repeat测试与spec/quality通过。v1 raw计算仅停止作者已核实专用任务，原件aborted_invalid保留；该source/scoped-review时点v2真实runtime尚未关闭整阶段gate；后续正式关闭见I-009。证据docs/p4_validation.md与v2 source fingerprint。

### I-033 — P4 relative输入证据与absolute replay不一致（Closed，限定source/scoped review）

独立review发现run写relative evidence，verify从manifest absolute roots重新load导致binding mismatch；fresh relative run→verify ERROR复现，run/load_static入口统一canonical roots后GREEN，独立scoped review通过。Public只脱敏manifest，不公开私有路径/连接参数。

### I-034 — P4文档状态补记脚本编码错误（Closed，限定补记）

真实run成功后一次本机临时STATE补记脚本报Non-UTF-8 SyntaxError，未执行状态替换；同一shell后续public manifest检查通过，所以末尾exit0不能冒充前面脚本成功。使用明确.venv/bin/python和UTF-8 coding重新执行，STATE已补正；不重跑实验/不修改生产source。编码来源未独立定位，不能猜为数据或baseline失败。证据docs/p4_validation补记与实际工具输出。

### I-009 P4 MVP evaluator证据更新（Closed，限定批准MVP范围）

P4 v2 literal oracle、跨block64D identical-vector反例、brute/blockwise与canonical-tie IndexFlat toy均通过；真实两baseline run和完整prediction/metric replay通过。独立Sol fix1 source/spec/quality通过，Astra另独立Popularity/抽样/hash/metric核查通过。I-032/I-033 source问题已闭环，v1 invalid保留；Astra正式关闭P4 gate，独立Sol最终spec/quality与文档/public manifest review通过，无新增阻塞；I-009按批准MVP evaluator范围Closed，不以单一exit0或作者自检代替。

I-008 MVP已证明bounded score blocks与实测process峰值（唯一数字experiment_log），不关闭旗舰scale/OOM风险。I-014固定seed hash-rank audit用户可复现并保留OOV，但representativeness风险仍Open，不把512 audit当全test或无偏平台评估。

## P5 review资源门修复

### I-035 — 最后资源gate前发布success manifest（Closed，限定P5发布门）

v1 run在write_artifacts内发布success manifest，最后一次resource sample低于20%才报错；static reader不要求producer/resource，失败目录可能仍可被后续消费。独立Sol指出后，fresh assertion RED用high/high/low控制序列复现run报错但manifest已存在；另缺producer/resource的重hash manifest被旧reader接受。fix1将data bundle与成功发布分离：exclusive写data，stop/join采样线程、最后sample及全部gate、strict success schema后最后xb写run_manifest；失败保留部分数据，无success manifest。Static reader拒绝缺producer/resource/sample/timing证据。真实v1保留且不验收，验证入口p5_validation与cache/fix1-red/green/regression。仍不保证进程突然crash的filesystem transaction或并发改动。

### I-036 — monitor采样异常未传播主线程（Closed，限定P5采样失败传播）

v1 monitor遇OSError可线程退出而主线程仍成功；独立Sol review指出，fresh assertion RED以background-only OSError+慢loader复现未抛错且产生manifest。fix1捕获采样/资源异常，fit node/iteration与阶段boundary检查异常和观测样本，finally stop/join，发布前确认无pending worker error。GREEN测试在主线程仍可运行时明确拒绝成功且manifest不存在。Resource声明为250ms可见host采样加阶段边界，不保证瞬时硬限制、hidden quota或其他租户影响；source/真run/review证据只读p5_validation。

I-035/I-036关闭证据：独立Sol fix1 scoped re-review通过，Astra fresh local全suite与server v2全suite通过，source c61ffbc1651dd0b8bd508f498cdde265de886fd4df45aeede6f7ebc1e2e65729；fresh assertion RED/GREEN分别复现和修复失败success发布/monitor OSError传播/static缺证据。v2 manifest669029504ab18df5f6a6109820d3d4a36a243f6ab0d3e8c46fc8ce17b1b6bbe9及完整producer replay通过。关闭这两项代码问题不代表P5整体gate已正式关闭或资源瞬时硬限制成立。

## P6实现与验证复盘

### I-037 — 两条训练路径一致仍可能两个都没训练（Closed，限定synthetic验证覆盖）

独立Sol指出checkpoint equality test只比较不中断与resume两条路径；如果optimizer.step被no-op，两条都不训练也能equal。dropout0不消耗global RNG且rebuild重设seed，删除RNG恢复也可能误过。增加至少一个参数相对初始值确有变化；保存前推进torch/numpy/Python RNG，rebuild后扰乱再load检查下一值。Global beam原toy4items/width20也没有pruning，补81item/width5独立逐层global oracle；causal从仅position0补到所有有future的位置。独立临时副本取消global裁剪/RNG恢复各目标test exit1 FAIL，真实source GREEN。证据p6_validation。只关闭测试洞，不以synthetic等价冒充真实GPU resume。

### I-038 — fresh-process verify先比较环境导致deterministic误拒绝（Closed，限定P6初始化）

训练seed_all使deterministic_algorithms=True写入manifest，新process默认False；旧verify在seed_all前比较environment，正确run也会拒绝。作者自检与独立Sol同时识别，正式GPU训练前修正为CUBLAS_WORKSPACE_CONFIG及seed_all先于env比较；fresh subprocess测试确认已越过environment gate推进到预期missing-config。临时独立副本移除初始化时test exit1 FAIL；真实source GREEN。环境不是随意忽略差异，而是初始化后仍精确比较Python/torch/CUDA/依赖/lock/backend证据。

### I-039 — server源码快照遗漏既有测试上下文（Closed，限定P6测试环境准备）

首次上传只src/tests/config/base files，忘记read-only Git对象上下文、P3/P4/P5脱敏public synthetic fixtures、outputs父目录，三次fresh全suite分别失败，不把shell退出或partial tests当通过。补齐合法小文件及GIT_OPTIONAL_LOCKS=0/GIT_DIR/GIT_WORK_TREE后fresh server全suite通过；旧失败log保留，具体命令与结果p6_validation。没有改旧tests放宽assertions，torch核心tests实际执行。以后按完整允许文件清单与测试依赖检查snapshot，避免逐文件trial-and-error。


## P7比较与接力风险

### I-040 — 历史阶段状态误当当前状态（Closed，限定文档指针）

P6已正式关闭，但experiment顶部仍写尚无模型、data_contract当前段仍写P3 awaiting；P7核对发现后将允许文档改当前状态指针，合同两句经协调明确授权修正，历史run/失败/review过程保留。冻结规则、输入、config、source与artifact不变。旧public P5 manifest的awaiting字符串是原快照hash绑定的一部分，P7不修改历史manifest，阶段正式验收以STATE和p5_validation为准。

I-014 audit代表性、I-029 irX语义、I-030 hidden timestamp顺序风险继续Open；one-seed无统计胜负证据、共同catalog的OOV不可达、P5有限迭代未收敛/greedy objective非单调均继续披露。P7独立核算通过只解决本固定audit比较输入/公式一致性，不关闭这些研究限制。

P7只读脚本最初用P6“config”字段，实际为effective_config，KeyError exit1后修正并fresh exit0；初schema探查用P4 predictions通用字段也KeyError，实际为分方法字段。均为审计适配失败，原tool记录保留，不把失败或shell后续成功冒充验收；生产及产物未改。

本地fingerprint复算首次错误地对base+src全列表排序而与producer的base先行顺序不同，assertion exit1；按manifest.py定义保留base顺序后fresh PASS。另process检查临时inline shell quoting导致SyntaxError exit1，改独立脚本stdin后fresh PASS。仅验证脚本修正，不改production/hash或资源。


## P3-C 数据质量补充风险

### I-041 — Metadata snapshot与匿名编码未提供（Open，core排除）

完整train-selected metadata schema/null/join与anonymous code profile通过，数值只读experiment_log的P3-C节；age/gender/geo没有业务reverse mapping或effective time证据，geo zero不能直接当missing。Item/user train_interactions_rank为全train统计，把它输入早期prefix会包含未来互动。Author/duration/content snapshot也没有独立历史时点凭据；旧content feature只沿用固定content-only benchmark assumption。

处理：不fill、不删、不推编码，core排除user属性与metadata ranks/author/duration；content静态来源/时点限制显式注册。数据null=0只说明覆盖，不证明point-in-time。正式新合同P3-D承接；不能由本轮gate通过关闭上游snapshot未知。证据见data_quality point-in-time registry与p3c_validation。

### I-042 — Catalog zero-like与head贡献集中（Open，保留主任务分层）

完整train item/author支持和user prefix分层显示长尾/zero-like与贡献集中，数字唯一experiment_log；这不是脏数据或bot成因证据。Alltrain catalog需要保留无正例支持items；ID-only表示可能缺少正例支持，content仍提供表示，不能缩catalog后称公平提升。

处理：raw immutable，主样本保留低活跃/长尾，不5-core；unknown ratio/255/cofeedback异常保留flag。后续模型支持分层/user balance消融由已批准阶段独立实施，本轮不训练或提前变更label/权重。Data质量审计通过不代表ID-only检索性能已证实，也不证明head users异常。

## P4B implementation issues (all evidence preserved)

- Train-positive support起初由eligible examples反推，会漏掉single-like用户独占item。协调review指出后补fresh assertion RED，改为全train like events显式support artifact/ID-map绑定；singleton synthetic GREEN。真实prepare后Astra与作者分别独立全raw oracle一致，support问题已验证修复；正式task仍待协调关闭。
- 第一次source-upload ledger误用system Python缺PyYAML，尚未上传source；空目录与失败日志保留，借已有批准Python env生成ledger并创建新exclusive目录。没有安装依赖或复用失败目录。
- 第一次real audit请求错误旧producer文件名，exit2。fresh RED/GREEN改为既有frozen-contract-manifest.json；新source/new目录audit exit0。原失败证据保留，冻结producer未改。
- 在原prepare进程terminal前执行verify，manifest尚未生成，exit2。这是操作顺序错误；原prepare继续且不重启，原prepare已terminal exit0后用新日志名verify-prepare-final验证exit0；旧失败证据仍保留。不能以缺manifest快照猜测prepare失败、覆盖exit或重复process。

## P4C verifier review finding（已修复，独立复核通过）

v1 verifier仅在persisted outputs之间重算metrics/ItemCF；错误content或错误graph与自身metrics/hash同步后仍success。协调独立复现后freshRED四assertions，v2要求pinned raw train重新建graph、pinned features重算content与三route全部predictions，并核final/initial artifact hashes不冲突。Fresh GREEN11与全regression123通过。当时独立fix review/真实full replay尚未完成、仅v1pilot运行，这是历史记录。后续final source/scopedreview通过，v2 pilot/full/verify实际terminal success，Astra独立全raw/pair图/全部三route ranks、metrics/coverage/fill/旧audit均PASS；v1 source/pilot原样保留。Task正式关闭仍需docs/public review与协调裁决，不能凭作者tests关闭。

P4C接力消息中曾误给未经读取确认的runtime manifest hash；立即撤回并fresh读取ignored aggregate更正。仓库/服务器文件没有写入该错误值，experiment_log由实际aggregate自动写入。Closing public manifest将programmatic提取实际hash并fresh核bytes，不凭手写聊天值生成证据。

## P4C coordinator content oracle归约差异（oracle修正，全量裁决待定）

独立coordinator最初对单个float32 mean使用1D np.linalg.norm，冻结旧P4/新P4C使用2D rows axis=1 norm。归约产生last-bit centroid差异，近tie的top ranks交换；诊断确认匹配冻结norm轴后global lexsort严格一致，fixed-pair d,id与qd,id scores逐值相等。Production/容差/tie rule均未改，失败log和diagnostic保留。此问题限制“任意数值实现排序相同”的声明，不代表数据/contract漂移；Astra修正版全量独立oracle已exit0/PASS，未改production/容差；正式task仍待docs/public review与协调裁决。

## P4D source-stage findings（真实run前修复）

- 初次入场suite默认sandbox无worktree临时output写权限，产生PermissionError；正确权限rerun通过，原log保留。不是模型或数据失败。
- STATE启动脚本按main格式replace而父acceptedSTATE格式不同，未实际改snapshot；协调fresh指出后定点重写并实际核读。命令意图不等于文件变更。
- Best更新初版发生在checkpoint保存后，manifest best可与checkpoint budget不同；fresh tamper RED证明改写manifest metric不被拒绝。修正logicalbest存储时序与verify_session selection/role/stat/epochchain核验，freshGREEN通过。
- 存储vectors到rank/metric自洽不证明模型输入链；fresh同向缩放vector并rehash可保持rank。新增每epochcheckpoint重导出与逐float核验，篡改RED→GREEN，formal verifier组合全部epoch。此时没有真实训练产物被更改或重stamp。

- Source-ready最后组合cursor核验时一次字符串replace同时匹配function声明，完整regression捕获SyntaxError；定点修声明后fresh rerun，旧失败log保留。未部署或启动真实run。

P4D独立v1 source review拒绝三个gate：CLIstr参数reader崩溃、混合sharedbatch/pilot可被选择、pilot缺ranking live context。协调另复现nonfinite被当capacity降batch。全部freshRED24–29修复后GREEN；实际source v2仍待scoped review，未真实运行。自检RED30再加固定content/support checkpoint buffer一致门。失败ledger与旧source package保留，未经通过不启动realpilot。

P4D source gate PASS后上传被automatic approval review拒绝：审查未识别既有human授权及payload/target归属。实际公开105文件小tar已逐hash核；请求被工具pre-execution拒绝，server未写入。补原始human授权/manifest证据前不重试或间接绕过，当前不是CUDA/resource或模型失败。

P4D上传审批在补可信原始human授权与105public文件逐member证据后，同原通道透明唯一重试获准，actual exit0。随后launcher生成器用全局PY/CODE替换污染ENV名称；sh语法检查无法发现bad variable name，remote第3行立即停，PID不存在且无stage产物。实际本地shell export RED复现，literal三ENV行GREEN通过；保留原launcher与日志，使用exclusive v2b修正脚本。失败发生在GPU/metrics前，不消耗正式调参run；生产source冻结。

v2b前四stage实际通过后，pilot因操作脚本把prepared_manifest文件传到需要目录的--prepared接口而退出；实际NotADirectoryError复现，manifest正确hash通过，resource-pilot目录不存在，零batch测量。修正为prepared_manifest.parent，仅独占pilot-v2c日志再次调用，旧exit2保留；不重复GPU恢复、不额外调参。后续pilot/train/verify CLI目录与file参数已逐接口核对，记录ignored path-contract。

首真实bounded操作在初始resource_snapshot(output.parent)时因runs父目录尚未存在而FileNotFound退出，实际只读复现；零model/optimizer/checkpoint/output，slot1预算未消费。v2b操作脚本完整preflight补exclusive parent创建、既有runtime/plan一致、目录/file/symlink/logs、CPU prepared/query与pilotbinding检查；local临时fixture执行实际脚本语句4scenario PASS。原failed plan/log/exit保留，生产source不改；新操作脚本独立review后续同slot1步骤。

Astra首真实checkpoint审计的第一次oracle因checkpoint best_metric tuple与JSON list直接比较产生exit1；诊断后仅做JSON序列化标准化，fresh独立oracle通过。生产source、容差和产物未修改，这属于验证表示差异，不是恢复或模型失败。

正式队列live poll曾在最新manifest open/write窗口解析空JSON而exit1，训练PID仍正常；Astra随后独立读取同run完整4900checkpoint状态通过。只读监控改为明确列出最新未可解析文件与上一个完整JSON；这只用于进度展示，不用于resume、不回避终态corruption。正式pipeline只在child.wait之后读取，因此其最终严格校验与生产source均未变。

首配置train终态exit0，但formal verify.exit2，queue实际停止、slot2未启动。actual CPU trace落verify_session:177；独立逐子条件best/selection/records/evaluations一致，仅stats epochs整数key与JSON字符串key超过十个epoch后排序表示不同，规范化后全部值一致。verify_draw_provenance有同类缺口。旧producer/产物不改；协调批准独立双指纹兼容adapter，先TDD/review，不能宣称run01已验收。

Formal verifier可观测性限制：其正式验收会重放全部epoch/checkpoint、完整向量与exact ranks/metrics，耗时较长；冻结实现没有逐阶段/逐epoch进度日志。外部只读ps、CPU/I/O、GPU和可见打开文件只能确认活动，不能可靠定位当前epoch或给出ETA。持续计算且尚无terminal不等于失败/卡死；不为监控改冻结source或注入正在运行进程，最终耗时仅在真实终态后补记。

首run全epoch兼容formal verifier actual0，双身份receipt/gates hashes经Astra fresh独立核，首run正式accepted（不是全P4D完成）。恢复24slot的操作v1在slot1receiptreuse read_json(str)接口错误处退出，未训练slot2。新v2仅两处Path包裹，实际原reader/operation body的gate-reuse、existingreceipt-reuse与selectionreuse三路径fresh RED1→GREEN0，禁止subprocess/train/verify。旧v1及失败log保留，producer/adapter/pilot/预算与已accepted产物不改、不重复5100verify；v2 scopedreview后继续原队列。


## I-AGENT-MONITOR-001 — 正常等待期间重复监控消耗

- 状态：规则已修订，实际交接 pending。
- 事实：Astra与Sol在正常训练/验证等待中重复轮询、每分钟重复汇报，造成不必要token消耗；无训练故障，具体节省量未测量。
- 用户要求：单监控者、长期可见对话、只在阶段验收/失败需要时升级协调。
- 处理：原服务器队列继续，不kill/restart/重复提交；本P4D子agent交接ready后停止周期轮询和文件写入，由已存在独立Sol对话接管。主目录AGENTS已修订，P4D冻结source及远程AGENTS不修改、不重新上传。接管确认前不写已完成。
- 证据：cache/p4d-visible-thread-handoff.md及原queue实际存活快照；接管确认待协调记录。


### P4D可见监控接管权限限制

独立可见Sol对话已实际接管，但自动审批分别拒绝向Astra发接管确认和创建每5分钟persistent heartbeat，理由是未识别直接人类跨线程消息/recurring授权。授权选项已直接提交Eddy，未绕过拒绝、未创建持续自动化；正常当前运行内有界只读检查不受影响，服务器既有队列持续自主运行。跨对话确认保留本对话/STATE/handoff供协调读取，避免声称Astra已收到消息。


## G0-I001 — 隔离恢复环境门与行动规范指针

G0本工作区新增记录（来源P4D只读）。首branch/write/fullsuite因sandbox worktree不在默认write roots而失败；已有用户授权下require_escalated成功，fresh163tests PASS/skip1，不拿tail exit0掩盖19 PermissionErrors。恢复断言误要求旧基点AGENTS等于最新主目录AGENTS；差异仅最新调度章节，明确当前规范pointer，保持受保护AGENTS不改；不是合同冲突，也不放松source identity。详细docs/g0_validation.md。

## G0-I002 — 旧P6假设隔离

旧runner过期deadline、512双split与test读取、旧cursor跨epoch填batch不兼容当前支线；保留原实现/证据，G0仅通过新synthetic入口复用model/codec/decode/evaluator，不调用旧runner/cursor。后续真实训练协议与预算仍由Astra冻结，G0不自行选择。


## G0-I003 — 冻结producer fixture扫描新consumer导致full regression gate未关闭

G0 targeted8tests及相关54tests PASS，但fresh完整171tests exit1/1fail/skip1。test_p4d_v2_verifier_compat的cli_fixture扫描所有当前src/vkgr/*.py，新增consumer使旧producer fingerprint正确拒绝。Reviewer独立复算：只排除G0模块则等于冻结PRODUCER_SOURCE。共享生产字节未漂移。当前允许文件不含旧test/adapter/fingerprint，未修、未重pin、未绕过；建议Astra另批test fixture明确旧producer成员选择。G0阶段不Complete；后续STOP直到协调scope裁决/fresh验收，详docs/g0_validation.md。另Minor manifest类型比较已fresh RED→canonical GREEN并review确认，非未解决问题。


### G0-I003 授权修复证据（2026-10-08，awaiting Astra fresh acceptance）

Eddy经主协调明确扩展本worktree旧test fixture范围。Fresh RED仍原source mismatch；原105-member transport ledger bytes与ENTRY_SHA、members digest与LEDGER_SHA匹配，固定19source+3base/config逐hash核查。cli_fixture改明确历史成员allowlist，不glob当前新增consumer、不黑名单排G0。新shadow future-consumer test与删除/篡改/rehashed member拒绝测试GREEN；原固定source身份/adapter/pins不变。11compatibility tests、8G0、完整173tests PASS（仅既有FAISS skip1），原manifest bytes与replay通过。作者修复关闭条件已满足；Astra fresh acceptance尚未发生，不自行阶段Complete。全证据docs/g0_validation.md文末。

## G1-I001 — 历史身份与新consumer解耦（2026-10-08）

Astra设计裁决允许固定37-public-file capsule+inventory及技术协议新文档，旧data_contract/producer字节不动。逐anchor/path/父链/symlink/size/hash/source-members校验与独立进程原G0replay通过；旧公开G0manifest在当前新source下拒绝是预期身份gate，现有动态G0tests保留。新consumer独立source/environment binding，历史static声明不冒称producer重跑。真实SID跨输入审计尚未授权，未关闭该真实gate；docs/g1_validation.md。

## G1-I002 — 同一catalog/content不足以绑定prepared版本

Self-review发现load_sid需直接核prepared context的manifest SHA属于当前合同；fresh RED ValueError not raised，最小GREEN增加直接绑定，独立source review认可。见cache/g1-cross-input-red.log与green.log。

## G1-I003 — batch重复全catalog canonical扫描与readonly布局漏洞

独立Sol初轮P2：sid_targets每batch按leaf重扫全catalog，synthetic成本oracle实际120leaves对应120次wholecatalog scan；改为load/audit一次排序分组及已验证codec O(1)identity。Re-review新P2：readonly ndarray仍可改strides，原gate未冻结布局；freshRED中ids因wrong OOV异常失败，另3数组未拒绝，修复冻结strides/data pointer和immutablebytes backing。四数组layout测试GREEN，scoped finalreview无新发现。保留全部RED/中间GREEN/fullsuite历史，最后fresh193tests通过；不外推真实吞吐，资源仍not_measured。docs/g1_validation.md与cache/g1-review-result.md。

## G1-I004 — 本地验收fixture与shell失败保留

首RED使用zsh readonly变量status导致额外shell错误（unittest新模块缺失RED已有log）；后续变量result_code。第二组temporaryroot漏P3-D bound文件、第五组build_queries错误import均修fixture，未改旧校验/pins；完整记录见G1validation/TDDlogs，不只看最终exit0。

G1公开扫描补证：numeric IPv4正则误报原已pin uv.lock的NVIDIA四段package version与官方wheel filename，共21处；结构化TOML/官方域名分类，未改任何锁字节或放过实际server地址。首次scope assertion失败而后续diffcheck0不能当scope通过，最终set -e重跑scope/public全部PASS。见G1validation与cache/g1-final-scope.json。

## G2-I001 — fresh review发现metadata seal与逐query replay缺口（Closed，本地）

独立只读Sol实际复现：重写context._queries后check仍通过，binding queries SHA未随之改变；另epoch1 OOV query改候选顺序并重算shard/evaluation/run局部hash与computation后，verify_run(replay=True)仍接受。现有suite217及generative52均通过不能代替这两项语义验收。G2不交付、不Complete；下一步分别fresh RED证明可复现，再冻结metadata identity并比较实际重训逐query computation。旧G1/G0身份未受改动，本轮无真实数据/训练或GPU。

G2-I001扩展与关闭证据：同轮review另复现eval中断无parentreceipt、deadline/verify wall缺口、AdamW moments可缺失。逐项作者fresh RED→GREEN，独立scoped6tests14.384s+optimizer0.604s确认原5项Closed。metadata冻结且核canonical hash；actual每epoch query computation比较；SIGTERM/InterruptedError安全pending receipt与续缺shards；逐query deadline及持久phasewall；完整moments/exactstep/非负variance。scope只关闭已复现5项，不声称任意SIGKILL/I/O故障自动恢复。后续journal问题见下条。

## G2-I002 — operation journal尾部截断回滚wall（Closed，本地）

独立Sol实际删除仅临时operation records，原producer ledger/run pin不变，第三次verify接受且累计wall降低。固定deadline未延长，但累计证据不可信。作者freshRED同样删除records后无headverify仍接受；修复genesis started marker+后续mandatory caller external operation head，解析chain后核外部head，缺失/过旧/head不符或截断STOP。CLI正常receipt串接evaluate→verify→select heads，failed body也返回新的failed head。targeted recovery/CLI5 PASS15.104s；独立最后scoped实际有效head前进、旧head拒绝、删尾/删全拒绝、abandoned begin计费均通过，finding Closed，无该scope新Important。最终full222 PASS(skip1)，持久六CLI及actual replay通过。Astra fresh验收仍pending，不自行G2 Complete。

G2最终验收脚本两次调用错误保留：独立scope脚本首跑用无末尾换行的JSON复算source digest，与项目canonical_bytes不一致而assert失败；改用既有canonical_bytes后40成员逐byte及digest一致。首--help调用漏PYTHONPATH=src，ModuleNotFoundError；补批准环境后实际exit0。均是验收调用错误，未改生产代码、pins或扩大gate；随后scope fresh全部通过，不用后续单独exit0掩盖失败。

## G3-I001 — 设计预算轨迹分层（设计Closed，implementation验收pending）

独立设计review发现formal/replay全局累计updates若直接塞入旧checkpoint budget，会与AdamW step==local completed_steps冲突。修正G3 execution pool计global reservations/失败/replay，模型轨迹保留旧local BudgetLedger，caller external heads绑定；scoped design review明确Closed，Astra批准最小实现。不以设计结论代替代码验收。

## G3-I002 — fresh CLI backend恢复及engineering扩预算（Closed，本地）

新进程resume/select首次GREEN实际flags与已pin profile不同，未放宽旧校验；显式configure后再核实际环境/device/flags，真实七CLI局部通过。engineering max_decode+1及deadline延一小时两subcases freshRED均未拒绝，增加派生cap及原60秒硬上限后freshGREEN。pilot测试期待history3是作者测试事实错误，实际prefix最长2；只改期待，未改fixture/gate。

## G3-I003 — pilot成本证据不完整（Closed，本地telemetry）

独立实现review确认原decode timings丢失、serialized writer成本缺失；fresh RED后保留每次decode/worker update/serialization/accounting的实际测量，原reviewer确认Closed。disk进一步发现64KiB大manifest缺口，经actual-byte writer及frozen-schema checkpoint精确预留修复，最后6tests6.348s独立Closed。最终244tests62.551s PASS(skip1)及持久七CLI只证明CPU工程；GPU内存/IO/完整validation成本、正式预算均未放行，详见g3_validation.md。
## G3 component-cost-entry-v1 gate与测试收尾（Closed本地，Astra验收pending，2026-10-09）

### G3 cost真实失败后的parser与可观测性（代码缺陷确认，server root cause pending）

P1 actual attempt已终态exit1/cleanup通过，generic worker failure未记录具体pre-budget stage。冻结parser忽略AnnAssign；旧fixture只覆盖Assign，本地既有version.py AST提供annotated cuda结构证据，synthetic valid annotation fresh RED3errors及mixed duplicate1failure，隔离一函数修复后scoped8 PASS。原server/frozen source未改、无新attempt。诊断证据、最小修复patch及server-local30s/8KiB只读probe需求仅维护cache/g3-component-cost-diagnostic-v1/review.json；历史实际失败原因仍unknown，不把候选bug写成已证实根因。未来新完整entry还需safe stage enum及新版本/attempt gate；不能覆盖旧receipt或为离线窗口绕过formal cost gate。

暂停前RED test helper缺finally wait，短stdlib child产生ResourceWarnings；已按恢复brief优先修finally TERM/bounded wait/KILL+wait/close。原4 PID暂停安全核均not_present，未触碰server。新entry最后fresh RED复现未授权worker会向已有root写failure receipt，修为完整external/source/start/handshake gate及exclusive attempt claim后才允许写，未经claim仅固定非阻塞BLOCKED stdout。其余deadline/log/IO-stub/输入SHA/owned birth与reap边界TDD见cache/g3-component-cost-entry-v1/review.json。无实际runtime/真实成本或server gate通过声明；按最新协议由Astra一次独立验收。

## G3 component-cost-measure-v1 repeated-phase I/O（Closed，本地，2026-10-09）

独立fresh-context Sol review发现同一phase第二次采样无效/异常/计数回退时仍留下第一次subtotal，会把未知成本写成完整成本。17tests fresh RED2fails后修为whole-phase null，并补before/after五种无效采样及KeyboardInterrupt partial计数边界。作者fresh17/0.179s、独立17/0.191s及旧accounting23/0.059s PASS，reviewer确认Important Closed，无剩余Critical/Important。final manifest后disk observation与exclusive loader parent也经各自fresh RED→GREEN。证据cache/g3-component-cost-measure-v1/review.json；仅synthetic protocol验证，实际入口disabled，不冒称真实成本/硬storage quota或已运行。

# G3 v2只读review增量（2026-10-08）

两项Important已由独立g3_v2_readonly_review确认Closed：parent封口扫描缺原deadline guard；invocation输出进入sealed tree，含dotdot alias。分别fresh RED后按授权scope修复为chunk/node guard+原deadline SIGALRM、写前后检查，及reservation前规范路径/禁止dotdot/symlink/所有roots containment。真实child扫描中abort不发布成功invocation，nested/alias拒绝时pool bytes不变。证据/private/tmp/g3-v2-red-review.log、g3-v2-red-alias.log、g3-v2-accepted-helper.log。

初次本地测试沙箱禁止ps/owned-group操作导致PermissionError，按授权scoped本机escalation重跑；未放宽进程收口检查、未SSH或修改服务器。最终author pass只证明CPU engineering增量，等待Astra fresh review；旧pilot包与源pins保持原bytes。不存在“消除select第二轮完整训练”的结论，因为旧select本就没有完整training replay。
# G3 entry-v2（2026-10-09，本地候选）

probe新增审查修订r3（2026-10-10）：entry parent cleanup已主接受CLOSED。probe controller早期client timeout/已有owner异常原仅TERM、不wait/reap问题fresh2tests RED复现，修显式client identity和固定subreaper-owned链有界清理/实际wait/reap/unknown保守；2+3定点PASS。新packet597aafe2e61cec70de977eeeff34b9bd7ca5c583c51f795dda9f787cd33a4b1d，entry hash不改；主probe差异验收pending，真实Linux仍未执行，不把候选称gate通过。

review闭环修订：唯一blocking fallback parent SID sidecar漏cleanup已fresh RED3复现，新增parent全SID/reap/actualwait与worker合取，unknown/reused birth/unreaped zombie保持false。revision2 packet ba62f5e186799b4fba420f4ff0dbbd202a68b215fa0ccee8548a531bcca4629b，31项受影响定点PASS，main闭环复验pending。初版48b1…完整包另存reviewed-v1。P1只读已证明当前固定输入AnnAssign/parser gate可复现，历史唯一原因unknown；具体Linux probe候选齐全但未执行，不夸大离线能力。

问题：v1静态torch parser忽略AnnAssign且早期失败丢stage；Mac SSH拥有parent生命周期，最终状态仅stdout，断联缺完整terminal恢复。修复/证据：cache/g3-component-cost-entry-v2/review.json SHA 48b1e9912c3d8bdec2233e2b6e64b9e8b2fd767734387dad9b2b58e70cc5bfe3，完整本地67tests和冻结pins核验通过，独立owner actualwait、fixed-size durableterminal、只读status/UNKNOWN。历史server失败原因仍unknown，未以本地修复替代历史证明。终态disk采样精确边界/host crash及SSH平台风险见README，真实Linux lifecycle gate未跑；不能宣称已可离线真实训练。无旧attempt退款/重启或新真实run。交Astra一次验收。

## G3 v4 状态观察超时（2026-10-10）
已提交作业的status-006本地/连接60s超时，无stdout，actual exit未知；最近status-005为RUNNING。仅状态通道UNKNOWN，不宣称服务器停止或成本失败，不重启/追加attempt；单owner退避后核同一身份。原始收据cache/g3-cost-actual-v4/status-006-receipt.json。

上述观察通道缺口：status-007实际exit0/non-timeout、RUNNING，恢复同一job观察；仅该连接缺口关闭，不代表成本gate通过。原超时收据保留。

## G3 fixed v4 成本终态与剩余预算门（2026-10-10）
status-008沿原唯一start取得component_cost_complete，固定consumer exit0、fresh owned cleanup通过；status-006连接缺口由007/008恢复，原证据保留。此次成本attempt已花费不重试；组件测量关闭，正式wall/disk/branch/full replay/checkpoint/accounting/control/diagnostic等仍unknown，不用safety factor填空。下一Astra定点验收及完整预算闭合裁决，正式训练/test关闭；所有旧失败/许可拒绝不抹除。

## G3正式预算闭合范围（2026-10-10）
actual-v4组件已被main接受，但审计AuditBudget和正式ExecutionPool开销不同，component guard额外小root inventory而formal query写pool journal；不能直接把full-Q phase当正式upper。checkpoint正式binding/长accounting/两个seal/完整读取核验/resume/branch等缺口见cache/g3-budget-ledger-v1/ledger.json。完整CPU-only既有artifact与metadata提案集中在proposal.md，未批准未实现；不重做已接受GPU测量，不弱化或执行缩减validation/replay，仅报告科学影响供决策。

## G3 CPU审计候选输入stop gate（2026-10-10）
冻结pilot producer manifest仅serialization bytes/seconds，不记录逐checkpoint路径/SHA；有限46份本地真实JSON搜索无G checkpoint绑定，P4F唯一命中排除。source推导路径不证明存在；CPU候选input两个SHA/bytes null，CLI真实BLOCKED，完整driver/owner未实施。证据cache/g3-budget-closure-candidate-v1；不SSH补取/替代synthetic，不索批半成品、不追加probe，由main裁决已有artifact绑定证据。旧成本与ledger包保持，formal仍关闭。

## G3最小formal接线设计约束（2026-10-10）
main取消旧checkpoint依赖CPUaudit，本轮转静态估计+有限operational风险预算，不执行服务器。旧cost disk_observation硬4096nodes，正式shard/journal会超过，不能直接复用cost owner；新bounded scanner/actualwait/原clock/资源/cleanup必须具体实现。源码select要求sealed pool tail不变，verify后best-eval会导致旧receipt拒绝；设计固定train→best-eval→fullreplay verify封口，相关TDD后才能宣称可运行，不改科学规则。checkpoint/全部shards/journals保留，unknown时间/宽度用风险spend cap而非伪造上界；设计入口cache/g3-operational-spend-v1/execution-design.md。


## G3 A本地完整候选的运行与证据边界（2026-10-10）

cache/g3-formal-A-candidate-v1/review.json SHA `0c40c1e9c79fef77c773d57cee18901b7a6c14d24c3f60a29e89bfe193dfcbe0`：已完成fixed owner/approval/context/CLI/full replay/consumer/scanner/wait闭包接线。fresh反例修复：旧4096 ceiling不复用；scanner等待不暂停parent deadline；扫描完成有退避与原完成deadline；final scanner cleanup不丢adopted实际wait；短命guard child不无限积累身份；错误worker approval/start/source或partial full-query/replay counters不能published成功。source44/旧包/冻结helper/consumer保持。

两个provisional owner test错误为历史timestamp fixture过期和datetime微秒往返精度断言，已修test并保留原失败log，执行deadline没有放宽。0resume为候选保守限制（结构limits1、池0），main差异验收；wrapper保留checkpoint/shards/journals/ledger，原CLI loader TemporaryDirectory仍按冻结实现释放scratch，不能声称所有scratch永久保留。Linux现场/subreaper/cgroup/CUDA/真实文件增长及计费未新实测，Mac controlled/table evidence范围明确。预算仍pending，观察STOP may overshoot，不能称成功保证。主一次review后才索Eddy固定实际运行许可；原v4不重复、0新server/真实训练。


## G3 status 代码执行早于SHA门：v2本地修订待原reviewer复核（2026-10-10）

main v1报告唯一Important，无Critical：direct-file status argv先执行可变controller后自检，wrongSHA可执行。fresh本地changed-code marker复现，另symlink parent/leaf及oversize执行、FIFO timeout亦记录RED。v2固定stdlib bootstrap在30s同deadline内先核fixed路径/nosymlink/regular/size/openfd identity/外部SHA，再compile/exec实际已核字节，只status三pins；无重新加载文件执行。six transport/status定点GREEN及readonly packet核通过，旧v1/server六核心/input/approval/payload/source44/helper/lifecycle均不变；review SHA `875f096ee1855381c9f9b3091cb24f61895add660720f602a8adbee92df8caa2`。本地修复不代替原reviewer验收，不重复旧suite/7query。0新SSH/部署/真实训练，实际许可pending。


## G3 status Important独立关闭（2026-10-10）

main原reviewerformal-A-v2-review正式ACCEPT_CANDIDATE_PENDING_HUMAN_SPEND_APPROVAL，唯一Important Closed、无新增阻塞；875f096...8caa2 fixed packet保留。不重复tests/review。runtime spend/实际attempt许可仍pending，绝不把candidate acceptance当human预算批准。approved envelope只按accepted build/raw/decode/command方法改status+真实短启动window，其余fields不变；原reviewer测试已核该可执行方法，无新实现缺口。human批准后才形成实际提交包和现场证据，0新server/训练。


## G3 formal-A actual首次状态UNKNOWN（2026-10-10）

fixed v2 directhuman approved并唯一submit返回SUBMITTED，attempt spent。三pin已审readonly status-001返回UNKNOWN，transport exit0/non-timeout；不能解释为进程停止、训练已启动或完成。当前owner/阶段/terminal闭组未知。不retry/resume/现场修代码，保留全部证据并向main升级；入口cache/g3-formal-A-actual-v1/。


G3实际UNKNOWN已本地定位：controller.status326重复读取lifecycle70414bytes用checked默认65536，fresh regular_bounded；首次显式262144核pinPASS。status-002仍UNKNOWN，不能从此认定训练状态。只读内存最小修订已送main409603d7...5285986，尚未执行，不改server包、不重试attempt。


main已定点ACCEPT observer-v1（409603d7...5285986），原server bytes不变，仅安全外部pin后内存status唯一二次read262144。actual status-003 RUNNING恢复可信owner活性观察。此修订关闭observer固定大小门缺陷，训练成功/当前stage仍未知。原UNKNOWN证据全保留；新v2本地补充测试/derived SHA仅保存未用于server。
