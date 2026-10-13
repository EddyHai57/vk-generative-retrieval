# G0 validation

密级: 公开

> 当前收口快照见文末“G0授权收口 — 2026-10-08”；此前fullsuite失败与待scope裁决均保留为2026-10-07历史证据。

## 范围与恢复

唯一task G0，已批准CPU synthetic foundation，不进入真实训练或下一生成式阶段。worktree `codex/g0-generative-foundation`，HEAD `0376d9a8119f3500cbf78519675cc7bf18013c1b`；native创建并附着。本worktree不是GitHub完整实现：从P4D当前公开overlay恢复83个modified/untracked非ignored常规UTF-8文件，逐文件检查regular/non-symlink、公开目录白名单与大小，再source重复读取一致、destination逐hash一致。source只读且不删除。inventory时间 `2026-10-07T11:36:32.949442+00:00`，cache/g0-entry-overlay.json SHA256 `a47f5ea5f72844cd9573705cd10be1a082576b2a379e8c67d0189b745b31ee2a`。STATE/handoff/issue是可变时间快照，不冒充固定producer。cache/g0-inherited-STATE.md与handoff保留旧历史，当前G0状态单独写。

最新行动规范是主目录AGENTS.md；基点本地AGENTS缺后加调度章，无相反规则，按明确派工使用外部指针，不修改AGENTS。tasks只带入主目录末节双支线准备/G0，旧路线保留历史。本轮新增仅generative_foundation.py、其test/fixture、本validation、小型public manifest及允许状态/issue/decision。其他git-visible diff均为继承overlay；允许治理文件之外全部继承文件逐hash未变是最终scope gate。

## 环境与实际接口

借用P6工作区`.venv/bin/python`，不是新独立安装：fresh CPython3.11.17、torch2.11.0、numpy1.26.4、pyarrow19.0.1、PyYAML6.0.2。无download/install/network/SSH，未读真实数据/服务器产物。所有命令在G0 worktree，`OMP_NUM_THREADS=1 PYTHONPATH=src`；实际解释器为 `<LOCAL_VENV_WORKTREE>/.venv/bin/python`。

复用retrieval_data.prefix_examples/validate_example/content_join、data_audit.build_queries、sid.SIDCatalog、tiger.SIDTokens/TigerModel/decode_history及baselines.metrics。不改公共接口/合同/config/锁。Fixture是明确synthetic的64D one-hot content、小catalog与hand-defined SID（不是新tokenizer实验）。Target是首next-like，history严格过去、重复保留最近50，OOV首target保留；训练以seed42一epoch无放回draw，batch2尾1不跨epoch，最多2个CPUoptimizer step。Model d16/1encoder+1decoder/ff32，使用既有原创模型。Beam50合法路径映射item，同一semantic leaf的suffix按item ID升序；不是semantic SID命中指标。

源审查：tiger_run.CONFIG保存过期deadline；publish_success要求validation/test各512；evaluate_model/load_inputs/verify_run固定读取两split；baseline_run.load_static同时加载holdouts；training.ShuffledCursor跨epoch补batch。这些仅旧P6证据，不改、不调用。G0新入口只读bundled synthetic fixture；无raw-root/split/test参数，不依赖旧runner/cursor/deadline，test gate closed。预算不与传统matched，不使用旧TIGER结果作新公平比较。

## 实际RED/GREEN ledger

Targeted命令尾缀：`<python> -B -m unittest discover -s tests -p test_generative_foundation.py -v`。下列每轮RED实际观察后才写该行为；不把permission/dependency failure当feature RED。

|Round|fresh RED|GREEN|
|---|---|---|
|R1 prefix/SID/query接口|exit1，缺generative_foundation import|exit0，1test|
|R2 strict boundary|exit1，16变异case中10个ValueError未拒绝，负SID NumPy警告暴露转换前缺gate|exit0，3tests；先Python严格type/range检查再uint转换，零警告|
|R3 model/metrics|exit1，缺run_fixture/evaluate_items，两errors|exit0，5tests；参数hash改变、[2,1]tail、固定seed两次完全相同、独立rank公式及collision miss|
|R4 manifest/CLI|exit1，缺write_manifest且--help stdout为空|exit0，7tests；hash/replay/exclusive/tamper、test CLI拒绝|

首baseline正常沙箱执行163tests，19errors均outputs写入PermissionError（失败不能以tail exit0冒充成功）；授权require_escalated后fresh完整163tests20.825s exit0，既有FAISS skip1。新模型CPUtests实际执行，不用skip掩盖。首分支创建和overlay写入也因sandbox拒绝后授权成功；auto-review未拒绝。恢复脚本在复制及hash成功后误设AGENTS byteequal断言exit1；read-only diff确认仅新增行动规范，改为明确主目录pointer，未改共享源码或生产身份。

## 独立检查及交付gate

固定toy目标20对应SID[0,0,0,1]，history/token namespace由hand literals检查；target999 OOV仍保留分母。Metrics oracle在test中手算命中rank与1/log2(rank+1)，all / train-observed / OOV分母独立检查；semantic sibling10不能算20命中。SID/Tiger/baseline等原tests作为regression，不读取旧真实test指标。

Fresh-context只读Sol reviewer 源码及最终docs/public review均已完成，无Critical/Important新缺陷；Minor代码问题已RED→GREEN复核。Astra独立fresh验收仍pending。最终public manifest/CLI/run/replay/hash及全量regression见后文。当前Implementation ready / awaiting Astra review，full regression gate待scope裁决，不自行宣布下一阶段获准。

## 限制

G0仅工程foundation；toy loss/metrics不证明模型质量、训练收敛、公平预算或真实协议实验。复用既有SID codec不证明新SID质量；输入没有验证hidden timestamp，official-order/static-content assumptions继续适用。没有checkpoint/resume、full eligible真实evaluation、GPU余量测量或资源账单结论；没有新增依赖/资源/真实数据训练/heartbeat/commit/push。


## 最终实现证据与未关闭gate

Fresh-context reviewer独立只读Sol6.1/medium（g0_fresh_review）未发现Critical/Important代码问题；发现manifest dict equality混同bool/int/float类型的Minor。R5先新增rehashed type-drift test，fresh RED exit1 `ValueError not raised`；最小改为canonical_bytes比较，fresh GREEN8tests2.267s exit0。Reviewer随后独立fresh8tests PASS，确认修复。源代码oracle/table checks是reviewer独立读取与验证，不冒称Astra验收。

单条实际synthetic命令（cwd本worktree）：

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m vkgr.generative_foundation run --output artifacts/g0-foundation-manifest.json
```

`/usr/bin/time -l`包裹实测exit0，real1.71s、maximum RSS313950208bytes（约299MiB）、peak memory footprint229968448bytes；只是本机一次synthetic整体进程观察，不是模型inference latency或GPU/服务器成本。Public manifest4763bytes，SHA256 `fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59`。不含逐query预测、真实IDs/SID map、checkpoint、私有路径或连接信息；fixture明确synthetic。输出exclusive，不覆盖既有manifest。Hashes区分input、consumer source、parameter/result computation digest；没有把内存参数digest称为已保存checkpoint。

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m vkgr.generative_foundation verify --manifest artifacts/g0-foundation-manifest.json --sha256 fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59
```

Fresh verify exit0，`G0 source/input/CPU computation/item metrics replay verified; test gate closed`。每次重新读bundled fixture、validate_contract既有P3-D pins、实际consumer源码/锁/config/env、模型小步执行/decode及item metrics；canonical bytes完整比较，不只checksum。`--help`实际exit0；不支持split/test或外部数据入口。后续其他Python/threads/deps/source变化应拒绝旧manifest，须新consumer run，不修改旧producer。

相关regression实际组合discover pattern：test_generative_foundation.py、test_sid.py、test_sid_run.py、test_tiger.py、test_tiger_run.py、test_training.py、test_baselines.py、test_baseline_run.py、test_baselines_faiss.py、test_retrieval_data.py；fresh54tests4.664s exit0，既有FAISS skip1；完整日志cache/g0-related-regression.log。Targeted8tests与其中CPU模型全部实际执行。

完整suite命令`<python> -B -m unittest discover -s tests -v`：首次新增模块后170tests21.044s exit1，review fix后fresh171tests21.971s exit1，两次均只有1fail、既有FAISS skip1。失败名：`test_p4d_v2_verifier_compat.CompatibilityTests.test_complete_producer_pin_and_explicit_cli_identity`，expectedreturn2实际1；实际拒绝`producer source mismatch`。具体根因：旧test cli_fixture把当前src/vkgr/*.py全部复制到模拟冻结P4D producer；新增G0 consumer被纳入旧source_fingerprint，故冻结producer gate正确拒绝。Review独立重跑失败并复算，仅排除generative_foundation.py时fingerprint逐字等于固定PRODUCER_SOURCE，没有别的生产漂移。

该失败不是G0训练/映射失败，**但完整suite未GREEN，阶段不Complete**。当前范围不能改旧test/adapter/公共fingerprint或重pin，不monkeypatch、不偷偷移动G0入口、不弱化gate。最小协调建议：Astra另行批准仅修旧test fixture的source成员选择，复制冻结producer明确成员；adapter/fingerprint/PRODUCER_SOURCE原字节保持。该建议未实施；Astra裁决前stop此项，不进入后续阶段。原P4D/source无写入、不SSH。

Final scope audit：83恢复文件中78个受保护继承文件逐bytehash未变；另5个仅允许STATE/handoff/tasks/issue/decision治理更新。只有5个新增G0公开文件；git-visible88paths，其余diff全继承。AGENTS与基点byteequal；主目录最新AGENTS只作行动规范指针。cache/g0-final-scope.json保存独立范围清单；公开小型常规文件/新文件credential marker检查通过。git status --short / git diff --check fresh exit0。

目前 **Implementation ready / awaiting Astra review，full regression gate待scope裁决**。Astra fresh验收未发生，不称正式G0通过；没有批准后续真实训练预算。本part成果是可审查的CPU工程入口及可复核manifest，问题/处理见G0-I001～I003；下一步只有Astra审查与受保护测试范围裁决。

文档收口脚本首用system `python3`解析中文时报Non-UTF-8 SyntaxError/exit1，未执行任何文档写入；随后以已核P6 CPython3.11.17执行同脚本exit0并fresh读回，原失败保留，不把后续成功掩盖首失败。

最终只读docs/public reviewer实际核manifest bytes/SHA/source全hash、CLI replay、78protected继承文件/88visible scope、两份regression日志与状态一致，无Critical/Important新缺陷。三处Minor旧启动状态/“训练”措辞已定点澄清；Astra正式验收与protected-test scope裁决仍pending。


## G0授权收口 — 2026-10-08（取代此前未关闭regression gate状态）

本次Eddy经主协调明确批准生成式路线并下放Sol；本对话只执行G0收口，不进入G1、GPU pilot或正式训练。扩展允许仅本worktree `tests/test_p4d_v2_verifier_compat.py`的cli_fixture/相关tests及本validation、issue、STATE、handoff；其余文件只读，现有基础manifest不改bytes。主目录暂留提案待批的措辞由本次明确批准取代，本worktreetasks不擅自编辑；未来阶段仍逐task等待Astra fresh验收/prompt。

### 可靠历史成员清单与最小修复

只读原P4D `cache/p4d-transfer-preflight-v2.json`，105个public_files的原ledger bytes SHA `87567fe8c2807cac752237b5da090314a02c012c8ec7c3fc2e4b853f803a7e1e`与adapter固定ENTRY_SHA一致；public_files canonical SHA `10a25620214bef7e94243ca242ac413ea743db0d7319a9a1197a7520e093995d`与固定LEDGER_SHA一致。从这份完整历史transport中选19个src/vkgr/*.py、.python-version/pyproject.toml/uv.lock三基文件与traditional.yaml，逐文件hash与G0当前原件一致。scope摘录在cache/g0-closure-frozen-members.json；实际明确列表固定在cli_fixture，不需要运行时访问其他worktree/cache，不随新consumer增长，也不是仅排除G0的黑名单。

本次diff只将fixture成员构造的泛化glob换为已核显式allowlist，并新增两个边界tests。Synthetic其余105成员仍是原test的public filler，不把fixture称为完整真实transport；原synthetic LEDGER_SHA/ENTRY_SHA替换helper行为不改，实际生产adapter与全部固定pins不改。原PRODUCER_SOURCE `2b8658b06ece14ca703f1b33cf84d8f24a096b9748d1f91ff91a3ce640bd2848`仍通过真实source_fingerprint核验，不mock/重pin/skip/assert放宽。

新边界test通过shadow项目模拟当前G0及任意future_consumer，fixture里都不能出现；独立按生产canonical顺序从复制bytes重算19source+3base fingerprint，与原固定PRODUCER_SOURCE一致。删除原成员、改原bytes以及改bytes后重算synthetic ledger三case，均要求invocation/result不存在且明确transport/source拒绝。保留原test的错误ledger拒绝、原成员篡改、origin篡改与exclusive sidecar。

### Fresh TDD / 验收证据

所有CPU命令cwd本G0worktree。解释器再次fresh核CPython3.11.17、torch2.11.0、numpy1.26.4、pyarrow19.0.1、PyYAML6.0.2，仍借用P6 `.venv`，无install。前缀`OMP_NUM_THREADS=1 PYTHONPATH=src`；单case RED加`:tests`以import test module。

|步骤|实际命令尾缀/exit/关键输出|
|---|---|
|旧case fresh RED|`-B -m unittest test_p4d_v2_verifier_compat.CompatibilityTests.test_complete_producer_pin_and_explicit_cli_identity -v`，exit1，1test/1fail，producer source mismatch|
|新增边界RED|同loader选test_frozen_fixture_ignores_new_consumers_and_pins_original_members及test_missing_or_rehashed_tampered_producer_members_rejected，exit1，consumer意外进入fixture；缺失成员断言起初错误期待FileNotFound，实际gate给transport mismatch，改正确预期后fresh组合3tests exit1/2fail，只有原source与新consumer污染，tamper test PASS；cache/g0-closure-red.log|
|最小GREEN|`-B -m unittest discover -s tests -p test_p4d_v2_verifier_compat.py -v`，exit0，11tests6.899s，全部实际执行无skip；cache/g0-closure-targeted-green.log|
|G0 targeted|`-B -m unittest discover -s tests -p test_generative_foundation.py -v`，exit0，8tests2.285s|
|完整fresh regression|`-B -m unittest discover -s tests -v`，exit0，173tests22.166s，既有FAISS skip1；没有新增skip，cache/g0-closure-full-green.log|
|原manifest replay|`-B -m vkgr.generative_foundation verify --manifest artifacts/g0-foundation-manifest.json --sha256 fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59`，exit0，source/input/CPU computation/item metrics replay verified; test gate closed|
|范围/format|git status --short、git diff --check；scope hashes验证，详情见最终scope；现有manifest原SHA完全不变|

原始RED及之前170/171 fullsuite失败日志保留，不改写为当时已通过。一次只读ledger探索把int类型files当list计算len，TypeError/exit1；随后按实际public_files105清单核pin成功，无文件写入。新增test首轮缺失成员异常名称预期错误已公开修正，未改adapter行为。

改前test SHA `1a66d2a48bedc9ac41fc0ab4173a5de62a29d815b2bb80f0a503192e33bcf981`，改后 `ac1fba8c68090f092c765597841bcbbaf665ee0d38b411c2e4835b5726542ed1`。实际patch保存cache/g0-closure-test.patch；原test bytes保存cache/g0-closure-test-before.py。Before全Git-visible/tracked hash记录cache/g0-closure-before.json，后续所有变化必须仅落五个授权文件；不会把继承overlay当本次修改。

本part：修正fixture混入新consumer这一测试构造问题，成果是完整regression转GREEN并保持篡改拒绝；不是修改模型/数据合同/真实实验。实现状态awaiting Astra fresh acceptance，独立source/scope review另记下文；不自行G0 Complete或进入G1。无SSH/GPU/真实数据/依赖/commit/push/heartbeat，基础manifest与生产source/adapter/config/pins全原bytes。


### G0收口独立review与最终scope

只读fresh-context Sol6.1/medium reviewer `g0_closure_review`独立从原105ledger核ENTRY_SHA/LEDGER_SHA、23选取成员逐hash及PRODUCER_SOURCE，独立targeted11tests7.360s PASS，无Critical/Important/Minor实现问题。Scope记录cache/g0-closure-final-scope.json：全部114tracked/nonignored-visible成员集合不变，只有5授权文件改变，109其他文件bytehash全不变，包括全部production/adapter/合同/config/AGENTS/基础manifest。改前后五文件hash与patch都保留，git diff --check通过。Astra正式fresh验收仍pending，Implementation ready / awaiting Astra fresh acceptance，不自行宣布G0 Complete。


## Astra正式G0验收（2026-10-08）

Astra本轮明确正式验收G0通过，取代此前awaiting acceptance状态：fresh完整173tests22.346s exit0/仅既有FAISSskip1，主机/private/tmp/g0-astra-fresh-tests.log（本对话只读摘要，log SHA26c669830e854eb36d7f4bcd2e150efc0470ebcc5dfed56ee3b701142f382b4d）。Astra原manifest verify exit0、SHA fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59不变；独立读patch并核closure-before114文件仅5授权变化/109不变，test SHA ac1fba8c68090f092c765597841bcbbaf665ee0d38b411c2e4835b5726542ed1，diffcheck0。该formal结果为Astra执行证据，不是本Sol新重跑。旧RED/失败日志保留，production/artifact未动。下一task G1先接口设计，当前不进入真实训练。
