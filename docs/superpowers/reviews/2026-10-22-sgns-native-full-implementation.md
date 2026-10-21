# Native B full SGNS 运载入口实施报告

当前为 **四门协议未冻结、full fit 禁用** 的工程准备。只新增 `analysis/sgns_native_full.py`、`tests/test_sgns_native_full.py` 与本报告；不修改旧 SGNS、B core、原 full producer、原健康门或科学配置。没有 SSH、push、服务器作业、真实 PF、真实 CUDA、科学训练或完整向量本地导出。

## 已实现的独立部分

- CLI 为 `python -B -m analysis.sgns_native_full`，含 `dry`、`plan --stage preflight|fit`、`preflight`、`fit`、`verify --stage preflight|fit`。fit/fit plan/fit verify 的四门入口均明确拒绝；fit 在读取 full 输入、初始化模型或写输出之前拒绝。
- 闭合 request 单次读 raw bytes 再校验 SHA/parse；显式绑定 source、lock、原 full producer、B benchmark、精确源码增量、独占输出、各阶段 run 与整数资源。`acceptance` 当前只允许 null，不能以伪造 `passed=true` 代替协议。
- 原 full 输入 producer 固定 `951c680cc77ceaf17cd9fb91e294f1da193fa447`。先核原 Git、逐文件 ledger 与完整物理源码树，再在**原 deployment cwd** 启动原 `analysis.sgns_gpu_full.read_bundle/full_job_contract/producer`；核原 PF → prepare（含原 source bridge）的闭包。旧 producer source/runtime/manifest 原值传递，不把旧 ledger 改成新 ledger。
- 当前进程再用逐字节相同的原 full `read_bundle` 核同一 bundle；比较前后 manifest、原 prepare run/terminal/stdout hash、runtime/python/dependencies/lock。完整 `source_tree` 拒绝未登记的 ignored 源码或模块遮蔽；source_identity 严格核 clean Git。
- 原 `fullinput-ready-v2` 复用原字节，不重建、不覆盖；实际 bundle/manifest SHA、原 PF/prepare run 路径须由 root 从既有收据填写，本实现不猜测。
- B benchmark 仍由已审 subset adapter 在其原部署 context 调用原 B verify；锁、compiled 选项与 full request 相等，≤12h 判据不变。正式 wall 机械计算为 `ceil(original verified projection_hours × 1.5)`，不是硬编码 3h；request wall 必须相等。
- 新 full 只读 PF 设计为重验原输入、构造同一 `NativeSGNS`、计正对，不调用 epoch/step，不写 vectors；需 root 后续在真实正式路径执行，当前没有实际通过收据。
- 预留 fit 路径只调用同一 `sgns_gpu_native.fit_sgns`；保持原 `SGNSConfig`、seed、五 epoch、batch、负样本分布和正对顺序。audit 明确使用 native-B schema，并调用已审 `subset.verify_audit` 加原 `check_full_binding(..., engineering=False)`；不伪造旧 numpy RNG/audit，原健康检查仍 require_pass。
- observer 只在当前同步 fit 的 context 内包住实例 step/epoch；原 step 参数、返回值、更新与 audit 不变，finally 恢复 class methods。每 64 batch 同步并输出 epoch、batch、已完成正对数、总正对数、elapsed；epoch 结束另报。达到 3h 后机械外推总 wall，超过登记 wall 或零完成进度即抛 STOP。同步增加开销，真实吞吐 UNKNOWN；不是独立外部 timer，也不声明 constructor 卡住时能在 3h 精确抢占（launcher 仍负责总 wall）。
- own job 精确绑定正式 argv、source/cwd/config/seed/resources、完整 recorded env 与有序 input pins；terminal 要求 explicit boolean false `wall_cap_exceeded`、succeeded/exit0/signalNone、stdout 等于落盘 receipt。模型落盘前后重算 B audit/健康与实际 bytes SHA；fit 分支因四门 stub 当前不可达，尚无正式端到端 fit/verify 证明。

## 精确部署增量

主工作树不能直接宣称是符合该合同的部署树。原输入树须保持所有旧文件原字节；native full 部署只接受 `original_951_files ∪ NATIVE_ADDITIONS ∪ OWN_PATHS`，集合及每个 sha 严格相等；任何旧文件字节变化、未登记新增或遗漏均拒绝。

`NATIVE_ADDITIONS` 是公开 B producer `305c415871bd60dc0d7d8a064a7e893830a9c20f` 的八个原字节生产依赖：

|路径|SHA256|
|---|---|
|analysis/p_query_blocks.py|f200cfac3002e00c87ffb9eb0640007f37bf83ee698b43a0aff176e52aae71e5|
|analysis/p_query_sprint_ops.py|a0251a8ef6fc2609c208274ee8a1f31e4722b50e15913fb19f41368728e85d2d|
|analysis/sgns_gpu_native_b.py|d9524bfbd9a6d67fedaf349b120a4005d1a90ba6a6e7998ed3e5150443329b4e|
|analysis/sgns_gpu_native_fit.py|98f11a0c351307bb48b9b151890e877894252eaeabb3e57dd74c90bd16151fe4|
|analysis/sgns_speedup_profile.py|3d46c76cc8328546c5ab21e3a12b999c1a22f4e815ac0880b7c26ec315bb6697|
|src/vkgr/p_query_blocks.py|ee08c4af86fee34d21eb633ff3995aac3253273cdebbcfda0f0aa9762fb520c5|
|src/vkgr/sgns_gpu_native.py|dd2acadd8d38a224410ff21b3d1bc0df3d81b5a7d4ee72d31cedf46125e671c4|
|src/vkgr/sgns_subset_sid.py|6ddc0169619e4d17a5e6749f06a239d5c4ff7217032f903c967e4efdf1a8da21|

`OWN_PATHS` 恰为本次三新增文件，逐 SHA 写在最终固定 request 的 source_delta，且与 clean commit 字节一致。没有开放式“除新增模块外”豁免。主树 profile 比公开 B 更新，首次 launcher fixture 因该差异正确拒绝；原失败记录保留。修正 fixture 为精确 951 树 + 八固定依赖 + 三新增文件后通过，没有修改主树或宽免哈希。

## 四门接线边界与最小接口

现有旧 `sgns_subset_acceptance` 绑定旧 numpy/SID/A1，不能直接证明 B。G 作者拟新增 `analysis.sgns_native_subset_a1.consume(receipt_path, sha256, scratch)`，返回 `{receipt_sha256, request_sha256, producer, proof}`，其中 proof 严格复用原 `check_proof` 七字段及原四门：health、sid_gates、collision、validation_hits、hits_gate、a1_native、evidence。

该模块的最终 SHA、outer receipt schema、source/runtime/input/producer 闭包尚未固定交付，本轮不自造可通过协议。后续接线须在原聚合 deployment context 只读复核并绑定 B subset producer/SID/A1 的原终态；明确固定源码增量和新 outer 身份，原四门阈值不动。未完成前 `require_acceptance` 无条件拒绝全部引用；null/旧 receipt/伪 passed 均不能执行 full fit。

建议四门接线固定且独立 C review 后再跑新 full PF，避免随后更改 request/source 使新 PF 绑定失效；原 951 输入 PF/prepare 保持原样。

## 本地验证与原失败记录

- Fresh RED：新模块未存在时 8 项测试均因缺模块报错；进度实际 stderr 协议补测先出现缺 `emit_progress`，再最小修复 GREEN。不是把 import/syntax failure 当 mutation kill。
- 当前 12 项控制通过（19.372s），含 tiny CPU 原五 epoch、原健康/B audit、formal full binding 拒绝、observer 与无 observer 输出/audit 完全相同、异常恢复、严格增量、wall 取整/3h 边界、own job 环境/argv/resources/有序 pins、正式 dry/非法子命令。
- 真实 pinned OPS `5840e6b686ec9651314a9e28a1a2c0befcc05158` 的非 mock launcher.prepare：951 原树 + 八依赖 + 三新增文件临时 clean commit → metadata → own job → 同 CLI dry；20 input roles，2 个 inherited-env 攻击拒绝。正式 CLI fit 在读 full 输入前因四门 stub 拒绝；queue/output 均未创建。transport fixture 只用 dummy bytes，不能替代真实数据 PF。
- 7/7 隔离 mutations 被预期目标断言 kill：四门拒绝、完整 source delta、wall×1.5、3h STOP、observer finally 恢复、native audit 校验、own job env。每项独立新源码副本、`python -B`、`PYTHONDONTWRITEBYTECODE=1` 与唯一 cache prefix；未接受 import/syntax failure。
- 最终相关回归与最终字节复验回执由下节补记。私有证据指针：`<PRIVATE_VALIDATION>/fresh-red.log`、`progress-red.log`、`launcher-control.log`（首次固定profile差异失败）、`launcher-fixed-layout.log`、`mutations.json`、`final-regression.log`。

## 限制与下一步

该交付不表示 full fit ready，四门接线、独立 C review、最终公开部署卫生/逐文件 manifest、真实 full 只读 PF 与 verify、实际 context-inclusive VRAM/RSS/磁盘/吞吐、有限 deadline 与 queue claimed 均待 root。首次资源只能按保守声明写 `not_measured`，不能把 torch counter 当含 context 峰值。本模块尚无真实资源测量；没有改科学阈值、seed、名额或旧 budget/spent/deadline，也不作 anchor/转正/结果解读。

## 最终相关回归

新增入口、原 full 输入/bridge、B core/CLI 与 subset producer 共 **78 tests PASS / 80.78s**。未运行全仓无关测试或真实 CUDA；本报告不把相关回归通过外推为正式 full fit/verify 通过。


## 四门固定接口接线更正（2026-10-22，保留上文原记录）

上文是原 `b390876` stub 交付的历史状态；其中“四门未冻结”“acceptance 只允许 null”“require_acceptance 无条件拒绝”“fit 分支不可达”与旧测试计数已由本节更正。当前代码已接线，但尚无真实 full preflight/fit/verify 成功收据；不把本地 unit fixture 当作正式四门通过。

- A1 使用独立固定 deployment：public source `96bfb9e6fcb4081168cf4d39563f50c58ee3fbad`，entry `analysis/sgns_native_subset_a1.py`，SHA256 `96c1e851e84d820607ef29cd7708da9fe5bab2aeff8492d86e9aa9cd9a3deb01`。实际 Git tree 为 176 files。先以 `git archive` 枚举所有原始 blobs 并逐 sha 校验，完整 `source_tree` 同时拒绝 dirty/untracked/ignored shadow；不复制 A1/G 依赖进 full runner。
- full runner 的精确增量仍为原 951 源码 ∪ 上表八个固定 B 依赖 ∪ 本次三文件，任何旧文件变动或额外新增均拒绝。A1 source/entry 是消费方固定常量，request 不能另选消费代码。
- acceptance reference 只允许 `root, receipt, receipt_sha256, scratch_root`。完整 outer receipt 固定 B/schema/authority/run/flags，核原 request raw bytes 与原七字段 proof。每个正式 plan/PF/fit/verify 都先消费四门，之后才触及 full 输入、launcher、模型或输出；执行结束与验收再消费并逐字比较稳定结果。
- consumer 子进程在原 A1 cwd，以同一 Python、`-B`、固定 PYTHONPATH、BLAS/OMP=1、CUBLAS 配置执行原 `consume`，复核 source 前后闭包与 module origins。fresh scratch 与代码、OPS、full output、原输入及直接收据严格分离；原 A1 consumer 再核完整上游 evidence/output 分离。返回键恰为 `receipt_sha256,request_sha256,producer,proof`，producer 恰为原 accept run/terminal/stdout 三原始 SHA；proof 保留原七字段与阈值。没有调用模型训练或 native train verify。
- A1 全部源文件、outer/request、原 proof.evidence、accept job 三文件全部进入 own job immutable pins；consume 前后真实重 hash proof.evidence，防止仅比较固定 ledger 而漏掉实际 bytes 漂移。full/A1 runtime 使用 canonical encoded bytes 严格比较，保留 Python/依赖/lock 一致，允许 `TorchVersion` 字符串子类的相同编码。
- 原 benchmark 固定 `5356ac240c9aa58828ad7bdf729a302ebd031289` 完整源码树，原 request/job/result 与所有 source blobs 进入 inputs；既有原 B verifier 不改。full wall 仍严格 `ceil(original verified projection_hours × 1.5)`，不把原 benchmark 自己的 2h wall 改成 full 的 3h。实际 full wall 由 root 按原 verified result 生成 request。
- own terminal 恰为原九字段，typed `succeeded/exit0/signalNone/wall_cap_exceeded=false`，另要求 wall_seconds 为有限非 bool 数、非负且不超过 own registered wall。fit verify 重新核落盘 vectors/audit 的两 SHA、safe npy、原 native B audit/health 与原 full binding，并比较 receipt proof；不只信写入时的一次检查。

### 本次验证与限制

- 四门 fixed consumer 接线有 fresh RED，缺常量时先失败；terminal cap 与 evidence 漂移也各有新的 RED（原实现分别缺 helper、未拒绝变化 bytes），再最小 GREEN。最终新入口 **20 tests PASS**；加原 full 输入/bridge、native B core/CLI/subset producer 的相关回归共 **86 PASS / 55.78s**。
- unit gate fixture 明确合成并 mock 外部 source/consumer，不伪造科学验收。覆盖 exact output/proof/request/producer/runtime、TorchVersion 字符串子类、subprocess cwd/argv/env、失败传播、old/伪造 outer 拒绝、证据漂移、输出/scratch 分离、terminal 数值与类型 cap、存储 audit 重验。
- 真实固定 OPS launcher fixture：原 951 + 八依赖 + 三文件临时 clean deployment → 实际 launcher.prepare metadata → 同 module dry 有非空回执；真实固定 A1 tree 收到 dummy acceptance 后，两个阶段的 plan/PF/fit/verify 均在 full input/queue/output 前拒绝。**这只证明 metadata/dry/拒绝链，不是成功的正式 full plan/PF。**
- ignored `cache/shadow.py` 由真实固定 A1 clone 的 physical-tree 检查拒绝。原 CPU tiny 五 epoch observer 等价、finally 恢复与 original audit 控制继续通过。
- final isolated mutations 与公开 diff 扫描见下方最终补记。没有 SSH、push、部署、真实 full 数据读取、CUDA、full training、完整向量导出、queue 写入或科学结论。独立 C、真实正式路径只读 PF/verify、实测资源与 full 科学作业仍由 root 后续执行。


### 固定交付验证补记

9/9 隔离 mutations 被目标 AssertionError kill：proof 比较、runtime 编码比较、消费先于 full input、精确 source delta、consume 证据漂移、terminal numeric cap、wall×1.5、observer finally、persisted audit 比较。每个 mutant 是独立新 Git archive 源码副本、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一 `PYTHONPYCACHEPREFIX`，未用 import/syntax failure 计 kill。临时 driver 首轮因 wall-factor 匹配字符串写错中止，保留原失败；仅修正 driver 匹配目标后重做九项，生产文件未因此变化。

三授权交付文件全部字节的凭证/IP/连接/签名URL/本机用户绝对路径/公司名字面量/单文件 >1MiB 扫描 **0 命中**，`git diff --check` 通过。root 仍须对最终压平公开分支整体 diff 再扫，不把本扫描当部署成功。私有证据：`<PRIVATE_VALIDATION>/sgns-native-full-wiring-red.log`、`sgns-native-full-cap-closure-red.log`、`sgns-native-full-cap-closure-green.log`、`sgns-native-full-wired-final-regression.log`、`sgns-native-full-wired-mutants-*/summary.json`、`sgns-native-full-wired-public-scan.json`。改动涉及判定/绑定，按 C 档交独立复核；本报告不自签独立 review 结论。


## 独立 C 阻断修复（2026-10-22，原结论保留）

独立复核在 `9662ad7` 实际 `accepted()` 上复现：初次核完 own terminal 后，长四门消费期间若改 terminal cap，原实现仍能返回成功，并把变后 terminal SHA 混进 proof；同时 own job 未约束 `ops_source/overlay/stop_mode/pinned_files`。该四字段与自己的验收闭包之前未闭合，本节更正，未扩大科学授权。

- job 复用原 951 已有 `analysis.sgns_subset_acceptance_a1.transport_job`：包括固定 OPS source/sha、overlay=null、pinned_files={}、stop_mode=signal、完整 metadata 允许键集合、inputs 有序路径/hash/真实 bytes、source/config/env/resources。未扩 full runner 源码增量，也未修改原 helper。
- `accepted()` 开始/结束快照 own run/terminal/stdout/result 原字节及 vectors/audit bytes hash，前后重核 source 完整物理树、951+8+3 精确 delta、原 request SHA/值、runtime/lock、immutable pins 与完整 own job；快照自身也核起止一致性。返回只绑定开始已校验原字节，禁止尾部临时重 hash 拼出混合 proof。
- CLI verify 在首次长 consume 前另取 own evidence/source/request 轻快照，整个 verify 结束比较；这一步不打开 full 输入数据。随后依旧先消费四门再读原 full bundle。成功的内层 accepted 不能掩盖首次 consume 期间的 own evidence 漂移。
- fresh RED：真实 accepted synthetic 负例逐一改变 own 六原件/产物，及四项 launcher metadata，原实现失败。新增 source/input context 前后不等、外层 verify 跨首次消费不等、数据读取顺序控制；修复后新入口 21 控制通过，相关回归 **87 PASS / 60.42s**。最终 CLI 快照内部闭包的小补丁另重跑全 21 控制，回执单列。
- 12/12 独立缓存 mutations 被目标 AssertionError kill，包含新增 own closure、CLI closure、transport metadata 与上轮九项。依旧各用新源码副本、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一 cache prefix；不计 import/syntax failure。

私有证据：`<PRIVATE_VALIDATION>/sgns-native-full-review-closure-red.log`、`sgns-native-full-review-closure-green.log`、`sgns-native-full-review-final-regression.log`、`sgns-native-full-review-final-controls.log`、`sgns-native-full-review-mutants.log` 及指向的独立 `summary.json`。本项为 C 档，固定新 SHA 交原独立复核者闭合；尚未部署、真实 full PF/fit/verify 或运行科学训练。
