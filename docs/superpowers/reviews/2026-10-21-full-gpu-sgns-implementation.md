# 全量 GPU SGNS 独立入口实现（待独立 C 审）

授权：decision_log `ad1c428`。基线：`173b4d41523390089cacaf5f60c0a65551bfe188`。本项只新增文件；原 `fit_sgns`、`SGNSConfig`、子集入口与 verifier 字节不变。没有 SSH、真实数据读取、真实训练、入队或 push。

## 实现范围

- `analysis.sgns_gpu_full`：同 `python -B -m` 形式的 `dry`、`preflight`、`prepare`、`queue-dry`、`fit`；无 submit 命令。`fit` 首行执行 `require_acceptance`，当前总四门收据生产者/适配尚未冻结，固定 fail closed；手写 bool 和 A1 单项 verified 都不能打开。
- `configs/full-gpu-sgns/fixed-dag.json`：固定已验收 stage2 manifest、原 source、原 runtime/lock、25 个 train 原始文件及 row-group/count、catalog、已验收 counts，另固定基线全部逐路径 sha256。没有机器地址、私有路径或数据 payload。
- `src/vkgr/sgns_full_verify.py`：C1 原函数逐字复制，只替换函数名/全量 binding 参数、子集 catalog 断言与末尾 CONTENT_V1 子集 pin；其余健康、SUM/SGD 合同、五 epoch 进度、learning rate、参与/frequency/fallback、algebra 全部原样。测试逐字对照适配后的函数体。

`fit_sgns(*inputs, device='cuda', engineering=False)` 是唯一训练调用，未覆盖 config、seed 或任一科学参数。当前不可抵达该调用，不把骨架称为已可运行训练。

## 输入与绑定

读取的是 25 周 observed 全 train 原始事件，按 file → row-group → batch → row 原顺序累积每个 user 的完整序列；不使用 UR001 like-target 样本，不截断到 H50。逐周 schema、row-group/row count 与总 user/event/catalog 守卫。固定 inventory 和实际消费 snapshot 同时核验 SHA/size，前后重复 hash。输入读取只打开 train00–24 与官方 content；stage2 只读取 manifest/input manifest/catalog 及既有 job 收据，不打开 validation payload 或 test。

stage2 的固定 manifest SHA 为 `286690e73c27422b3ccc4334071171f90c1a034284f4a9ff4baa7eb6af34ccb3`。原 source 与 runtime 逐字比对固定记录；训练部署须符合固定基线全部路径及固定新增路径集合，source/current commit 同时绑定 `--deployment-sha` 与每份 receipt。新增路径 hash 来自 clean deployment Git commit，写入 source_delta；部署前仍须由执行者/独立复核者固定最终 commit 和逐文件 SHA 清单。未开放“除新增模块外”的无限豁免；合入其它文件后需重新钉住 delta，不会自动接受。

原 C1 content 不能直接套全量。全量 content 按固定 catalog 与官方 embedding 重建，preflight 与 prepare 的 content bytes/feature digest 必须一致；准备出的独占 bundle 再绑定 artifact SHA、序列 SHA、source/runtime、原 preflight job/stdout。fit 骨架读取 bundle、核 prepare job/stdout、原 verifier 检查 persisted vectors/audit，输出后再核 byte hash。

## 验证与限制

本地 synthetic 及结构验证仅代表工程覆盖。真实 A40 runtime/lock/real-input formal preflight、独立 C 审、总四门验收收据适配、首次训练资源实测均未完成；full training 为 STOP。

首次输入准备 launcher 使用保守 RAM 256 GiB / CPU 1 / disk 16 GiB / wall 24h，标记 not_measured。函数内 projected-memory 使用 96 GiB，仍受既有剩余资源守卫约束。没有伪造 peak 或声称通过服务器准入。后续声明应依实测更新，训练 VRAM 声明尚未实现。

真实 launcher 结构链使用原 OPS 固定代码、干净 CODE clone、合法原 lock 路径与 synthetic stage2 manifest，仅经过 launcher.prepare 和相同 module dry，明确不算真实 preflight；SGNS 不读 P3-D，因此 p3c overlay 不适用。

## 本地回执

- fresh RED：入口不存在时 6 项中 1 failure / 5 import error，随后最小 GREEN；新增 consumed JSON 负测先观察缺 helper 的 RED，再实现实际消费字节 hash。
- 相关回归：`python -B -m unittest tests.test_sgns_gpu_full tests.test_sgns_gpu_subset tests.test_sgns_subset_ops`，28 项全部通过。
- 隔离变异：`python -B tests/fixtures/sgns_full_mutations.py`，6 项全部 killed：four-gate bool、旧 source 字节、row-group、消费 JSON hash、job stdout、五 epoch。每项新拷贝、`-B`、`PYTHONDONTWRITEBYTECODE=1` 和唯一 cache prefix。最初尝试移除 metadata 总行数比较仍被最终 ordinal/count 守卫挡住，属判定路径冗余；另加独立 row-group 漂移负测并检验该路径。
- 结构链：在实现提交 `bd147ef4e50102e100b5a50226b6b0ca1d65930c` 的 clean clone 上执行 `tests/fixtures/sgns_full_launcher_chain.py`，回执 `launcher_metadata_dry_verified`；seed=42、submitted=false、payload_read=false、real_preflight=false。结构链内另正向核 exact source delta。最终报告提交后再核同一链，回执交主执行者固定存档。
- `git diff --check` 通过；旧科学文件未改。独立 C review 和服务器正式只读 PF 交回主执行者。

STOP：总四门 receipt 汇聚器/适配未冻结；full fit/submit 不可运行。准备入口仅可在独立 C review 与固定 SHA 部署后运行正式 preflight。完整 suite 另执行 `python -B -m unittest discover -s tests`，出现 OpenMP `Can't open SHM` 与 multiprocessing `worker session owner type mismatch`，运行停滞后仅中断本地测试 session（exit 130）。中断栈位于 `tests/test_throughput_exact.py::test_scores_predictions_ties_and_tail_blocks`；没有完成统计与逐项 failure summary，失败数量及新旧归因 UNKNOWN，不宣称全仓库通过。日志为执行环境临时回执 `<TMP>/full-sgns-full-suite.log`。

新增内容公开卫生扫描：7 文件；本机绝对路径、连接 URL/IP、常见凭据模式与大于 1 MiB 文件命中均为 0。此扫描不代替主执行者 push 前对最终压平 diff 的检查。

## 部署基线更正（主执行者要求；本节取代上文部署口径）

上文原文保留：最初固定基线为 `173b4d41523390089cacaf5f60c0a65551bfe188`，最初结构链由过程主树 clone 验证。该树只用于本地过程记录，不能作为公开父历史。本次按既有最小公开策略改为已公开 `c9a60c5ce6eeb02db862455a5cafbbc861c4effb` 的完整 124 路径账本，加本项固定 7 新路径；不允许其它原文件改字节或其它新增路径。

- `fixed-dag.json` 新 SHA：`a6b3e5d74015fca3a6648a6a6c5edf76e89c5be50ee5b036abe57be602b51543`。仅部署 baseline ledger 及对应 pin 改动；原 SGNS/Config/verifier 科学 body 未改。
- AST 传递 import 核对覆盖 27 个 repo module（含本项两个新 module），所有已有依赖在公开基线中存在且与过程树字节相等。原固定 OPS 继续独立使用 `5840e6b686ec9651314a9e28a1a2c0befcc05158`，不复制 OPS 到 CODE。
- 缺少的旧 fixture 路径只有 `configs/new-instance/anchor-uv.lock`；生产 `--lock` 原本就是外部固定输入，正式 runtime 仍只接受原 lock SHA。无需为结构测试添加第 8 路径。结构 fixture 改为外部 synthetic lock，仅给 launcher 做输入哈希；明确不代表原 runtime/lock 验收。
- fresh RED：新增公开 baseline 断言，旧 `173b4d4` 被拒绝；最小 GREEN 后 29 项相关回归通过，原 6 个隔离变异全部 killed。
- 新结构 fixture 从公开基线 checkout，只逐路径复制本项 7 个已提交文件，临时 commit；断言相对公开基线的 diff 路径恰好等于固定 7 路径，然后执行真实 launcher.prepare → metadata → module dry、exact-source-delta 正向核验，并在合成树重跑 29 项相关回归。此临时树只作验证，不 push、不部署、不读真实数据。

## 下一独立批：四门汇聚器输入与判定草案（尚未实现/冻结）

当前 A1/SID 仍在独立 C 修复；以下仅对齐源码中已有字段，最终生产者 SHA、request SHA、真实输出 SHA/health/collision/hits 均 UNKNOWN，不能用旧有缺陷的 schema closure 作为验收依据。`require_acceptance` 本批继续关闭；下一批应实现实际总 receipt 的严格消费者，缺任一真实闭包拒绝，而不是长期保留无条件 raise。

拟新增独立汇聚入口与测试，不修改 SID/A1 生产文件；其 source delta 另逐路径冻结。汇聚输入只给固定 path+SHA 的 request 与完整原 producer 四件套（run、terminal、stdout、stage-receipt），所有下游引用从该 request/receipt 沿链读取，不接受另传一个可替换的 hits 或通过 bool。

|输入|机械判定/绑定|
|---|---|
|CPU 与 CUDA SGNS 原闭包|`gpu-sgns-subset-stage-v1`、stage=sgns、device 精确对应、authority=ad1c428、engineering=false；原 OPS job/argv/seed/source/runtime/lock/bundle/output 与 stdout=receipt；实际 vectors/audit 的 artifact SHA 与 original_audit_verification 核实。双方 input_manifest/input_producer/source/runtime 一致；CPU 历史 pin 相等与否单列，不作为本四门的额外门槛。|
|门① GPU 原健康|原 SGNS audit 验证链必须完整，原健康 `passed=true`；复用原 checker 机械重算，不自定健康阈值。不得只信 stage verified。|
|CPU 与 CUDA SID 原 request/verify/manifest/input/mapping/audit/tokenizer|`gpu-sgns-subset-sid-stage-v1`、stage=sid-verify、mapping_replay=true，且 run_manifest/request/SGNS receipt hash 相互对应；各阶段原 job/terminal/stdout/manifest 闭包齐全。复用最终 SID `load_sgns_stage`/`load_bundle`/`successful_stage` 等只读验证逻辑，并以最终 C 审通过接口为准。|
|门② 原 SID 门禁与 collision|原 BHKM branching=48/depth=3/seed=42/max_iter=10，原 gates 全过（包含三层利用率≥80%）；原 audit 的同分母整数 `semantic_collision_items` 用原 `collision_gate` 重算 CUDA×10≤CPU×11，receipt 与 manifest 的 gate/comparison 完全一致，不从四舍五入的比例推断。|
|A1 最终 request/preflight/train/verify 与 newfamily input producer|`gpu-sgns-subset-a1-stage-v1`、stage=verify、authority=ad1c428、engineering=false、test_read=false；config/seed/CPU-GPU SGNS/SID/input producer/native manifest、选择 epoch、best/last、evaluation manifest 完整闭包。使用最终 `job_binding`/`receipt_binding` 的同一规则，但汇聚自己的 source/cwd 与 A1 producer 分别验证，不把调用方进程身份冒充原 A1 job。|
|门③ 原 A1 validation|`validation_hits` 必须是整数且≥378，`hits_gate=378`；`native_verification.status=verified`；hits 来自原选中 epoch 的逐 query 原始整数分子，并与评估/训练 manifest 绑定，不接收浮点 recall 反算值。验收训练仅原获批一次，不重训。|
|门④ 全链收据|SGNS→SID→A1 train→verify 的每一条 referenced request/run/stdout/manifest/artifact 逐字 hash 与绑定匹配；各源树按固定增量与原 runtime/lock 校验，terminal 成功且 signal=null、stdout 非空 verified 与落盘 receipt 一致。任一前驱断链、未知字段替换、异常终态或输入/输出漂移均拒绝。|
|汇聚器自己的 request/job/receipt|新的固定 request schema 与完整输入闭包先冻结；独占输出总 receipt，逐门证据指针与 hash，不复制数据；完成后由 full consumer 检查其固定源码 SHA/部署树、job argv/seed、request SHA、terminal、stdout=receipt 与四项结果。总 receipt 的 `verified=true` 只由实际四门核对产生；单项 receipt 的 subset_four_gate_acceptance=false 维持原样。|
|full fit consumer|除上述总 receipt，仍须 full stage2 ready、正式 full 输入 preflight/prepare、full bundle/job/stdout 与当前固定 source/runtime/lock 完整绑定；收据缺失或未知不进入 fit。最终 fit 仅原函数 device=cuda，不改变科学配置；本批不训练。|

待最终 SID/A1 C 审交付后，再确定总 request 的精确键集合、production API 适配和新增路径 hash；字段或 loader 不足时明确 STOP，不伪造成功或为适配放宽已有门禁。本草案提交主执行者确认后进入下一独立批。

## 独立 C 审 producer 绑定修复（单独批；未含四门汇聚实现）

独立复核固定审查 `8965964` 发现：旧 producer 会接受 wrong seed/cwd、空 config/inputs、signal=15 或 wall_cap_exceeded=true 但 status=succeeded/exit_code=0 的前驱；stdout 在解析后被替换时，后续 digest 可能绑定未验证的新字节。这些是 binding 缺陷，不能部署旧版 formal PF。

本次 fresh RED 用旧 API 复现上述异常仍被接受（`ValueError not raised`），另加入各字段变异与 stdout 解析后替换测试，然后最小修复：

- run/terminal/stdout 各读入同一个 immutable bytes snapshot；解析、最终 receipt 比较和 proof SHA 全部基于该 snapshot，返回前重新读三文件并逐字要求不变。没有再对未解析的新 bytes 单独签 hash。
- full preflight/prepare 前驱由固定 argv 与独立参数构造完整 expected launcher 合同：schema/run_id/kind/cwd/argv、seed=42、clean source commit、config path/SHA、精确 input path/SHA/bytes、固定 env、资源声明、原 OPS commit/代码 SHA、overlay/pinned_files/stop_mode。terminal 必须有完整原 schema 字段，signal=null、wall_cap_exceeded=false、exit_code 为整数0且 status=succeeded；未知 run 字段不能代替 expected。
- prepare 的 launcher inputs 现在登记原 preflight run/terminal/stdout 三项，fit 骨架核 prepare job 的同一完整合同。正式 runtime/lock 仍以原严格校验为准，不因本修复放宽。
- stage2 历史部署与 seed=None 元数据保留。主执行者只读重新取得原 stage2 run/terminal/stdout 三个 raw SHA（证据 `<CACHE>/ad1c428-sprint-20261021/full-producer-original-sha-private.log`），逐项固定在 `fixed-dag.json.stage2_job`；三 snapshot 必须先逐字 hash 相等，再核原 source/clean Git/完整 schema/成功终态/stdout 语义，不重编码原 JSON、不重跑原 job。新 `fixed-dag.json` SHA 为 `c3ddc90c2265910bfe2362fe86a21d421cd1fbba4a033bdebf0f00fa26175064`，取代前节同文件旧 SHA。
- 验证：30 项相关回归通过；10 个隔离变异全部 killed（原6项加 expected job 比较、signal、wall cap、snapshot 稳定性）；仍是新 copy + `-B` + `PYTHONDONTWRITEBYTECODE=1` + 唯一 cache prefix。最终提交后重跑公开基线 clean 合成树 launcher 链与30回归。

本修复须新独立 C 复核，不声称已获通过。四门汇聚开发只留一个未提交的独立负测文件，不纳入此提交；full fit 仍 fail closed。没有 SSH、push、真实 PF、真实训练或入队。

## 四门 receipt 汇聚器候选（独立批，须独立 C 审）

本节取代前述「下一独立批尚未实现」状态。只新增四个封闭路径：`analysis/sgns_subset_acceptance.py`、`analysis/sgns_subset_acceptance_a1.py`、`configs/full-gpu-sgns/acceptance-pins.json`、`tests/test_sgns_subset_acceptance.py`。full source 仍严格等于公开 `c9a60c5ce6eeb02db862455a5cafbbc861c4effb` 的124原路径加明确11路径。旧科学实现、`SGNSConfig`、`fit_sgns`、subset verifier 与 full verifier 字节不改；改变 full consumer、绑定比较与接受逻辑，按 **C 档**，不能以负例通过替代独立复核。

### 真实 pins 与输入边界

主执行者确认最终 A1 经独立 C 审 `6b56438` 通过，最小公开 candidate 为 `1ca14e4971df4d40534e28b6a89d2a2d151a1c8f`，等于已公开 SID `41456628c18d7ed6510aaeb5bc468001864d1518` 加固定30路径。汇聚 pins 现已登记该实际 A1 完整158文件 ledger、SID 完整128文件 ledger、最终只读 API 和原 external lock SHA；`acceptance-pins.json` SHA 为 `5d83dcaee024a5746ee8d16ad8bb5967f8a2bfb245b6632db933c2449e08fc59`。此前 null pins 的原文保留，不把过程 source 当最终 source。测试仍验证 pins 未冻结时在读 request 前 STOP。

真实四门 request、A1 原 verify 四件套、全部原 stdout/artifacts、真实 validation hits 与碰撞计数 **UNKNOWN**；本批没有构造冒用这些原件的成功 fixture，没有生成总验收成功 receipt。

request 精确字段为 schema、独立 A1 code root、A1 request(path/SHA)、A1 verify(run/terminal/stdout/receipt 四件套)、本汇聚 ops/output/scratch root、原 lock(path/SHA)、当前 full deployment SHA。所有 CPU/GPU SGNS、SID、input producer、A1 stage 与输出只从该固定原 A1 request 沿链取，不接收另传 hits/通过 bool。单项 `subset_four_gate_acceptance=false` 保持原样。

### 实际实现的机械检查

- 当前 full 源树、独立原 A1/SID/SGNS 源树、原 runtime/lock、原 request 与逐文件实际字节均核对。A1 tree 包括 ignored 文件的完整扫描，只排除 Git 管理命名空间，拒绝 symlink/未知文件。
- `python -I -B` 只运行本部署固定 adapter；stdlib bootstrap 先核固定 pins/request/source，保留原 A1 cwd/ROOT，再固定 analysis namespace 与 vkgr package 来源。导入前后逐 first-party module 检查 origin 和 ledger SHA；不从 request 接受 module/脚本名。
- 重核原 SGNS input preflight→prepare→CPU/CUDA fit 的 job/terminal/stdout、原 bundle 与实际 vectors/audit；复用原 `verify_original_fit`，重新核健康/progress/五 epoch/代数及原 audit proof。终态 signal=null、wall_cap_exceeded=false、整数 exit=0；原 launcher seed/cwd/argv/config/精确 inputs及 bytes/env/resources/OPS source/overlay/stop_mode均构造后比较，不只看成功状态。
- CPU/CUDA SID 各自三阶段原 request/job/receipt 均绑定，读取原 mapping/tokenizer/audit/input bundle，复用原 `load_bundle` 重算原 all gates；不再 fit/replay BHKM。按原 audit 整数同分母重算 `gpu_count * 10 <= cpu_count * 11`，要求与原 manifest/receipt完全一致。
- A1 原 preflight/train/verify 的固定 request、原 input producer、新 family 输入、native manifest、best/last、各 evaluation manifest/rows、原选择 epoch与 EarlyStop state逐项绑定；原 `validation_hits` 从选中 epoch逐 query整数分子重算并核≥378。读取原 verify 已发表 replay rows 并核原 computation；不调用 train、run、execute 或 native.verify重新产科学结果。原 integer hits与原verify回执必须相等。
- `_contexts` 需要独占新 scratch；此检查会写工程临时文件，明确标记 `scratch_written=true`，不称完全无写。原输出和所有已登记证据前后逐字 hash稳定；总 receipt只记录逐门 proof与原证据 hash。
- full `require_acceptance` 已改为有条件严格 consumer：必须显式 receipt SHA和另一个独占 replay scratch；核汇聚自身的原完整 job/terminal/stdout=receipt、当前 source/runtime/lock/request，隔离重核原四门 proof并要求完全相同，结束再核 producer和原字节。缺任意真实闭包则拒绝，不进入 fit；不再永久无条件 raise。
- 新汇聚入口提供 `dry`、`queue-dry`、`verify`，无 submit。`queue-dry`只走原 launcher的运载哈希/metadata，不代表原件已验收；合成 inert bytes仅用于这条链。原 full stage2、full input PF/prepare/bundle检查仍保留。

### 验证与未验证范围

已观察初始缺入口负例 RED；当前40项相关本地测试 GREEN。16项变异全部 killed，使用新 copy + `-B` + `PYTHONDONTWRITEBYTECODE=1` + 唯一 cache prefix；包含未冻结 pins、消耗字节 hash、ignored shadow、module origin、汇聚terminal signal/wallcap，以及原 full10项变异。日志指针 `<TMP>/full-acceptance-negative-green.log`、`<TMP>/full-acceptance-mutations.json`。

最终 A1 code-only隔离 smoke：从实际公开candidate checkout 完整158文件，在 `-I -B` 子进程加载固定 API，45个 first-party modules origin/hash检查与原 native bytes核对通过；payload_read=false、scratch_written=false、training=false、acceptance=false。证据 `<TMP>/full-acceptance-bootstrap-smoke.json`。这只证明最终源码/API可加载，不替代真实数据/四门验收。提交后另跑 clean公开基线+固定11路径的真实 launcher→metadata→同module dry链（full和汇聚两个入口）；最终回执交主执行者，不宣称本批真实 PF或训练已运行。

### 运载阻塞与旧 source 复用边界

主执行者实际回执：原 full input PF保持原 `a1a963c03822b53f30cef5561e714a043b30a538`（c9+7）pending。原声明 RAM262144MiB，原 scheduler当次 total489234MiB/free312414MiB，扣除后余量低于20%，因此未 claimed。不得强插、热改 scheduler、换 run_id重投或把pending写成claimed；本批未碰原队列。

**本批尚不包含旧7→新11 source bridge。** 新候选相对旧7是明确4新增路径，且原7中的full入口、launcher fixture、mutation fixture、此报告共4路径发生字节变化，不能称纯新增。当前同source约束会安全拒绝旧7 PF/prepare原件，不会开放式忽略差异。主执行者已授权下一独立 C 档桥批：冻结旧a1a963c完整131文件/原cwd/argv及科学参数，新候选逐路径固定变化；新prepare严格消费旧PF（包含唯一精确resource override plan），核数据/embedding/stage2/lock/output一致，仅按封闭桥区分cwd/source/deployment。RAM拟改196608MiB（首次not_measured，入口projected96GiB的2倍）的原pending原子改动仅由主执行者按已授权流程处理，且须桥先能严格接受该精确plan；当前队列保持原样。桥不改变科学输入、config、seed、fit或门禁。

本候选没有 SSH、push、部署、真实训练、读test或入队；真实四门闭包和adapter科学读取路径尚未实跑。独立C审、真实只读preflight与原件闭包不足任一项，都不能称full training已获准。

本批 clean chain 首验 fresh RED 揭示 `launcher.prepare` 只返回 cwd/git/config/inputs/env/overlay/pinned_files，还未生成 run schema/resources或注入run环境；错误地把它与完整run合同比较会拒绝全部queue-dry。已按原 launcher 真实接口只比较完整 prepared 字段；正式 `check_job`/consumer 的完整 run 合同不变。没有修改原OPS源码、资源、科学路径或门禁，修复纳入同一尚未review的候选批。

## 新 family input producer transport 闭包修复（独立批）

独立 C 审 `f02217f` 固定审查 `8b21d91`，结论 REQUEST CHANGES，唯一 P1：`accepted.input_producer` 仅经原 A1 `core.input_producer` 验证，未接新汇聚 `producer_records` 和完整 launcher 合同。独立复现证实 wall_cap_exceeded=true、exit_code=false、foreign cwd、删 cwd/env/resources/ops_source 仍被原 checker接受；不是科学结果失败，但不能据旧 helper 的通过签总验收。

fresh RED：新增原 input producer 的完整 transport 测试，先观察新闭包函数缺失而失败；独立复核原件保留上述四种被接受的对照证据。最小 GREEN：只在本汇聚 adapter 新增 `check_input_producer` 并显式接入，原已发布 A1 input producer/源码 pins/科学实现均不改。

- 原 run/terminal/stdout/receipt 四件套经严格 checker，要求完整 terminal schema、signal=null、wallcap=false、整数exit=0；从原 argv消耗实际 derive request字节，不接另传request。
- derive request精确schema/字段、原source/lock/content/UR001/raw_data/GPU SID identity与实际 family manifest逐项绑定；原OPS/root/输出四件套路径与 receipt output/runtime/完整键集合核对。
- 按最终公开 `analysis.sgns_subset_a1_inputs.prepare` 原合同重建 seed/cwd/argv/clean Git/config/精确inputs与实际bytes/env/resources/OPS source/overlay/pinned_files/stop_mode。derive专有环境是 `PYTHONDONTWRITEBYTECODE=1`、PYTHONPATH、OMP、MKL及原launcher注入字段；不借用训练环境，不开放忽略缺字段。
- 原 derive request、UR001/content/lock、family/history与 producer四件套 hash均登记到总proof的前后稳定证据；helper在任何科学context/rows之前运行。

验证：41项相关回归 GREEN。transport-only synthetic正控不声称四门成功；八种负例包括独立审查四种，加 empty inputs/env、wrong RAM、wrong OPS identity，全部拒绝。实际adapter早期调用路径另有负例，移除新hook会失败，且不创建scratch。18项隔离变异全部 killed（前16加 input producer hook删除、完整metadata比较删除），均新copy + `-B` + `PYTHONDONTWRITEBYTECODE=1` + 唯一cache prefix。证据 `<TMP>/full-acceptance-input-transport-red.log`、`<TMP>/full-acceptance-input-transport-green.log`、`<TMP>/full-acceptance-input-transport-mutations.json`；提交后重跑固定11路径clean树的真实launcher双入口链与41回归。

本修复改变接受/binding逻辑，按C档须新独立review；不冒称原四门已齐。旧7→新树 source bridge与唯一RAM override仍分离，队列未改。三个私有plan/before/after原件已只读核SHA匹配，并机械确认before/after仅ram_mib262144→196608；不将私有路径或连接信息纳入公开代码/报告。


## 旧 full PF source bridge 与精确 RAM override（独立工程批）

本批按主执行者转交的 `3c4cd42` 授权实现旧 source PF→新 source prepare 的封闭桥；属于 C 档 source/binding/接受逻辑变化，须新独立 review。仅本地实现、测试和 commit；未 SSH、push、部署、操作原队列、训练或读取 test。原 PF 的 run/source/argv/科学输入/截止保持不变，不换 run_id、不重投。

- `source-bridge-pins.json` 固定原 `a1a963c03822b53f30cef5561e714a043b30a538` 完整131文件 ledger；新公开树为 c9+14（138文件）。与旧131对比是恰好4个旧路径字节变化、7个新增路径：变化路径为 full入口、launcher fixture、mutation fixture、本报告；新增为四门4路径和本桥3路径。所有其他旧路径必须逐字节相同。外置 request 登记当前完整 source ledger 与固定 deployment commit，各变化/新增 SHA逐项写入proof，没有开放式忽略名单。
- 固定 plan/before/after 三原件 SHA，只准原 pending `ram_mib`262144→196608，其他完整JSON字段与原raw bytes均固定。主执行者明确授权新 prepare 首次声明196608MiB；依据入口 projected96GiB 的2倍，标记 `not_measured`，不是实测峰值。原 preflight 生成合同仍262144；唯旧 PF 桥合同消费固定plan after196608。后续声明仍须实测含context峰值×1.15向上取整256MiB。
- 新 prepare 在科学输入读取前核原旧树完整文件、精确after claimed bytes及pending/rejected无同名、原job/terminal/stdout、原schema/read-only身份、runtime/records、完整seed/config/inputs/env/resource/OPS/overlay/argv/cwd合同，并要求数据/embedding/stage2/stage2-run/lock/output一致。科学计算仍为原完整路径，结束时仅按封闭source桥替换来源字段，比较旧PF与新prepare完整科学receipt；写出前重核所有原件。manifest登记旧PF三件套hash与source_bridge；后续read_bundle重新消耗原request、plan、claimed、旧树和原PF，不能只信manifest布尔值。
- 原子 apply CLI仅在严格request/固定plan通过后可到达；锁原scheduler.lock，拒绝已有run/claimed/rejected、原pending bytes变化、已存在backup/temp，O_EXCL保存三原件与temp，fsync文件/backup目录后再次核状态，再atomic replace并fsync队列目录。失败保留旧记录与已写backup/temp，不删除、不重投。低层helper只用于独立临时目录测试；真实apply未执行。

本批fresh RED起点为新module缺失（4测试）；最小GREEN后扩为54项相关回归，覆盖新旧launcher合同、paired flags、source闭集、固定plan/唯一RAM、独立run/claimed/rejected/backup/temp/锁/错误before/二次state重验、double apply、prepare科学输入前hook和bundle重验。首次变异暴露固定SHA负例不独立：伪plan也违反其他检查，原文证据保留，后补“完整有效transport plan但固定SHA错误”负例，以及before/after都有seed的单字段变化负例。所有transport-only synthetic数据仅测运载合同，不代表接受的科学PF。

证据：`<TMP>/full-source-bridge-red.log`、`<TMP>/full-source-bridge-green.log`、`<TMP>/full-bridge-regression.log`；最终隔离变异与clean launcher链收据在提交后补记。变异统一fresh copy、`-B`、`PYTHONDONTWRITEBYTECODE=1`和唯一cache prefix。真实旧PF receipt、真实source bridge科学路径、真实四门闭包均尚未通过，不把本地transport链当成科学验收。

本批24小时仅为原非训练PF/prepare wall，未改任何deadline或训练wall预算。full fit CLI已有conditional consumer实现，但正式训练launcher/资源实测与wall预算尚未就绪；四门未闭及真实只读链未通过，不能称full training ready。若未来训练wall需改变，必须先获相应裁决。


提交后补证（代码候选 `2656932`）：30项隔离变异全部 killed，收据 `<TMP>/full-source-bridge-mutations-final.json`。clean c9+14树 `62986e91c4f0ef22c61fd011621bf407860c581b`（138文件）真实launcher.prepare→metadata→同module dry及54回归通过，含新prepare RAM196608和固定私有plan三原件的bridge transport-only metadata；收据 `<TMP>/full-source-bridge-launcher-chain.json`。最初sandbox回归遭macOS OpenMP SHM环境错误，原stderr保留在 `<TMP>/full-source-bridge-launcher-chain.err`；不改代码/输入/科学项，在本地工程权限重验通过，后续stderr `<TMP>/full-source-bridge-launcher-chain-retry.err`。原队列与服务器未操作。独立C审尚待主执行者安排；公开压平后的整体diff与敏感信息扫描由主执行者在push前执行。
