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
