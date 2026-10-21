# UR001 full BHKM / trie 运载实施报告

本文件记录限定三新增文件实现与本地 synthetic 验证；尚无正式 full B producer、SID preflight/fit/verify 收据，不表示科学通过。最终实现与验证证据在完成前补记。

## 范围、依据与原科学接口

依据全量合同卡分叉附录、decision_log `6efd105` 与 `863f53a` 的 BHKM / 数据身份条款、阶段 3 的 producer/consumer DAG 条款。只新增本报告、`analysis/sgns_native_full_sid.py`、`tests/test_sgns_native_full_sid.py`。没有修改共享事实源、旧模块或冻结配置；没有 SSH、push、部署、真实 full 数据读取、正式作业或完整向量导出。独立 C 由 root 另派。

- BHKM 调用原 `vkgr.sid.fit`，不是子集 `fit_bhkm_coll`，没有 monkeypatch 科学函数。K 按原整数 floor cube root 规则机械计算，N=12,012,895 时 K=229；depth=3、seed=42、max_iter=10、external_sort=false。原排序、capacity greedy、float64 mean/objective、DFS 与 RNG 调用顺序不变。
- 输入沿原 `validate_input`：严格 ascending uint32 IDs、float32 [N,64]、全部 finite、float64 norm 的 `atol=2e-6,rtol=0` 单位向量；没有借底层 sid.fit 容许零向量而放宽。正式 N 由原 full fixed DAG 约束，子集 120415/schema 不套用全量。
- H1/H2/H3/H4/H6、H5a/H5b 的字段和值/阈值与原子集 adapter 逐项对照。沿原 `au.audit_mapping`、`au.gates(kind='bhkm')` 保留 H1–H6、B1；另外 H5a/H5b 使用原 actual/expected 回执结构。保留的原 H5 与 H5b 执行相同的向量 digest 相等条件，不构成放宽。
- H5a 取已由原 full B verifier 验过的 `input_binding.original.manifest.binding.content_feature_sha256`，再与原 native audit.content_sha256 核对；不接受 caller 自报 expected，不复制旧 CONTENT_V1 literal。H5b 严格原 `au.feature_digest(vectors)`；catalog digest 同原 `sid_v2.catalog_digest`。二者不以 npy 文件 SHA 替换。
- R1 沿原 `usage_ok` 的整数 `used*5 >= K*4`；三层 prefix occupancy 分别 `occupied[d]*5 >= K**d*4`。≥80% 不放宽。节点 centers、初始化、iterations/objectives 等原 nodes 全部保存。

## 调用次数与正式角色

1. `preflight`：真实正式路径 `engineering=False`，先验 full B DAG、catalog/vectors/身份与现场规划，只读、不 fit、不临时更新参数、不写向量。显式 holdout/test/`week_26*`/`week26*` 路径在读原件前拒绝；闭合 request 也不允许新增 holdout 字段。
2. `fit`：原 sid.fit 一次；落盘后的 mapping/nodes/audit/trie 独立读取校验并用原 sid.fit 同 seed replay 一次，逐位比较 IDs/SIDs/完整 nodes/centers；共 **2 次**。fit stdout 的 verified 绑定 fit_verify/same_seed_replay 与最终 manifest SHA。
3. `verify`：独立 run_id、原 launcher job/终态/stdout；先消费已完成 fit 原件，再额外 **1 次**原 sid.fit replay，单独写 `verify/stage-receipt.json`，绑定 fit producer 与同一个 fit manifest。
4. `consume`：只接受上条独立 verify 收据，重新校验自己与 fit 的完整 job/terminal/stdout/source/runtime/request/inputs/产物闭包，重算 H/R/audit/trie，**0 次** sid.fit。不把 fit 的自报 replay 代替独立 verify，也不在每个 G consumer 重复 full fit。消费期间仍会调用固定原 full B readonly verify；该成本与全量 wall UNKNOWN，需 root 实际测量。

正式命令都是 `python -B -m analysis.sgns_native_full_sid`：`dry`、`plan --stage preflight|fit|verify --request <REQUEST> --request-sha256 <SHA>`、`preflight|fit|verify --request <REQUEST> --request-sha256 <SHA> --execute-on-new-instance`。plan 只出真实 launcher metadata，不入队。正式 run_id 用小写 `ur001-native-full-sid-`，符合原 OPS；schema 保留 UR001 新身份。

## 原 full B producer 与自身闭包

request.producer 恰有 root、source、request、receipt、verified_producer。source 是独立部署固定 commit，不能要求等于 SID consumer commit。request/receipt 是原件 path+SHA；receipt 是原 full fit/result.json；verified_producer 恰为 run_id 与 receipt/run/terminal/stdout 四 SHA。

先验证原 full B 完整 Git blobs 与物理树，原入口 SHA256 钉 `b8799ee1c8fcbdfc3dedee9d3b590cfde25c970c70f40d9f20d151fc2873730c`；再在该 deployment cwd、固定环境与同 Python 执行原 full CLI verify(stage=fit)，严格比较原 verified producer。原 SGNS 四门、实际 fit/manifest/audit/原健康检查由该固定已审入口验证，没有自己造一个 passed 替代。前后重 hash 原 request/receipt/run/terminal/stdout、全部 launcher inputs、完整部署 source、原 951 source 与 vectors/audit；原有 producer ledger 不重写。

SID 自身 source_tree 精确匹配请求固定源码；额外钉住原 sid/sid_v2/sid_tok_audit/generative_data/data_audit 原 SHA。runtime/lock 与原 full B 精确相等。own job 复用原 transport_job，严格 argv、完整 env、source/config/seed/resources、inputs 实际 bytes、有序 pins、固定 OPS source/sha、overlay=null、pinned_files={}、stop_mode=signal。terminal 用原九字段并要求 finite/nonnegative wall_seconds 不超过登记 wall。自身原件与所有 fit 产物/上游输入在长核查前后快照；返回 proof 使用开始捕获的已验 SHA，拒绝混合新旧原件。

## RAM 监控

现场读取固定 OPS `5840e6b` 的 common.ram_info 经原 a40_ops.install_cap 的规划口径，必须精确等 `PLAN_MIB=489234`，request 没有放大 planned RAM 的字段；不是物理 free-m 总量或只用 cgroup 上限。阈值按整数 `RSS*5 > (PLAN_MIB*MiB)*3` 立即 STOP。

formal plan/PF/fit/verify 与 consume 全程启动 daemon monitor，每 **1 秒**用原 common.process_table/descendants 采样自身进程树 RSS（含子进程），原 sid.fit on_boundary 再同步检查。超线或 root RSS 不能可靠读取，结构化 STOP 并 SIGTERM 自己；不向外部进程/scheduler 发信号，不自动切外排。采样可能漏掉两个采样点之间的短峰，boundary 不能覆盖大算子内部瞬时峰；不声称精确峰值测量。首次资源仍 not_measured，未报真实 context-inclusive VRAM/RSS/墙钟。

## 给 G backend 的完整 artifact / DAG 接口

`fit/manifest.json` 最后写，schema=`UR001-native-full-SID-trie-v1`，sid_version=`UR001-BHKM-B229D3-S42`（实际 K 机械重算时随 K）。manifest 精确字段：schema、sid_version、engineering=false、test_read=false、seed、branching、depth、catalog_items、catalog_digest、vectors_sha256、suffix_vocabulary、vocab_size、content_feature_sha256、input_manifest_sha256、input_feature_sha256、input_dimension=64、parent、source、runtime、request_sha256、artifacts。

artifacts 恰为下列五键→SHA256 字符串，路径只相对于 manifest 父目录，不接收 caller 另选路径：

|文件|原字节 / 语义|
|---|---|
|mapping.npz|deterministic original writer；ids uint32[N]，sids uint32[N,4]|
|trie.npz|keys uint64[N]、key_items uint32[N]、bases uint64[4]；与原 SIDCatalog / codec 重建逐项相等|
|nodes.json|原 sid.fit 全部 nodes，含 centers/init IDs/objectives/iterations|
|audit.json|恰 mapping audit 与 gates；消费者独立原门重算|
|config.json|UR001 config，K/depth3/seed42/max_iter10/external_sort=false|

catalog_digest 是原 canonical JSON ID list SHA；vectors_sha256 是原 au.feature_digest；input_feature_sha256 才是原 SGNS vectors.npy 文件 SHA。input_manifest_sha256 是原 fullinput-ready bundle manifest SHA，**不是 stage2 manifest**。parent.binding.stage2_manifest_sha256 是从原 full B 已验 binding 原值取出的直接 stage2 SHA，供 G 与自己的 UR001 stage2 manifest 闭合；content_feature_sha256 同样从该已验 binding 取出。parent 还含 reference、source、runtime、binding、input_manifest_sha256、input_feature_sha256、完整 evidence path→SHA ledger。

跨 commit 消费 API 为 `consume(receipt_path, sha256)`，receipt_path 必须为已完成的 `verify/stage-receipt.json`。在固定 SID source tree 的 cwd、相同锁环境执行，不能调用 verify CLI 代替 consume（会再跑 fit）。返回闭合键：

- manifest：原 fit manifest 的 path+sha256。
- producer：独立 verify run_id、receipt_sha256、run_manifest_sha256、terminal_sha256、stdout_sha256。
- source、runtime、parent：上述完整真实身份。
- codec：sid_version、branching、depth、suffix_vocabulary、vocab_size、catalog_digest、kind=bhkm、engineering=false、manifest_sha256。

G 只在固定 source/entry 消费通过后加载 mapping，以原 codec 构造函数与自己的 catalog 建 codec，再逐项核 trie；不修改 A1 科学 body，不伪造旧 CONTENT_V1 或子集字段。本实现 commit 是实现来源；root 最终压平 public source 与入口 SHA 另交 G 钉住，不把工作树直接当部署。

## 验证、失败保留与限制

- 最初 7 控制 fresh RED（模块不存在），最小实现 GREEN；扩到 16 控制，含原 fit IDs/SIDs/centers/nodes/keys/key_items/bases 精确对照、同 seed replay、原 H 回执逐字段阈值、unit/dtype/ascending/catalog/H5a/H5b、R1/prefix整数门、RSS父子树、规划上限、boundary、原 fit 次数、完整 own OPS metadata、own七原件漂移、full B 外部 context/DAG 漂移。
- 原 H5a/H5b 初次写成通用 value/threshold 回执，与原 actual/expected 不同；逐字段测试明确 RED 后更正，门限未变。
- 真实 pinned OPS launcher fixture：b647105 原树 + 本次三文件临时 clean commit → 实际 launcher.prepare（三个 stage 正式 argv）→ metadata → 同 module dry 非空回执。首次因大写 run_id 失败，原记录保留；修正为小写后通过。没有 mock launcher。该 fixture 使用 synthetic metadata；**不是成功的正式 plan/PF/fit/verify**，未创建 queue 或正式输出。
- 10/10 隔离 mutations 目标 AssertionError kill：seed、H5a、R1、prefix、RSS cap、RSS children、fit replay次、文件pin、own closure、parent closure。每项独立新 archive 副本、python -B、PYTHONDONTWRITEBYTECODE=1、唯一 cache prefix；不计 import/syntax failure。
- 新入口与原子集 SID / native SID / 原 BHKM throughput 相关回归 **48 PASS / 51.41s**。一次误写不存在的旧测试名，未收集到测试；保留原日志，改用实际 rg 列出的相关文件后跑上述回归。最后 holdout 文件名前置拒绝补丁再跑全 16 控制。

私有证据：`<PRIVATE_VALIDATION>/sgns-native-full-sid-red.log`、`sgns-native-full-sid-green.log`、`sgns-native-full-sid-transport-controls.log`、`sgns-native-full-sid-boundary-controls.log`、`sgns-native-full-sid-parent-controls.log`、`sgns-native-full-sid-final-regression.log`、`sgns-native-full-sid-final-controls.log`、`sgns-native-full-sid-mutants-*/summary.json`。toy science fixture 与 mock full B trust boundary 不代表任何真实 producer、科学结果或正式通过。

当前阻塞是原 full B 科学 producer 尚未交付及正式路径真实验收，不以 toy 数据伪造。root 后续独立 C、最终公开 source/逐文件部署、真实只读 PF、实际 RAM/CPU/磁盘/耗时与有限预算入队仍需完成。没有 anchor、转正或结果解读。


## 判定精确口径更正（封版前，未运行真实数据）

上文原文保留：曾写“三层 prefix occupancy 分别 occupied[d]*5 >= K**d*4”“UR001-prefix-usage”，并将其计入初次十项变异。root 复核后指出 863f53a 的“三层利用率”必须沿原 `sgns_subset_sid.audit_gates` 的 R1：每位置 distinct/K≥80%；原代码没有 prefix_groups/K**d 门禁。封版已移除新增 prefix 判定，**仅报告原 audit.prefix_groups，绝不作为通过条件**。最终 checks 恰 H1、H2、H3、H4、H5、H6、B1、H5a、H5b、R1；H5 是原 au.gates 中与 H5b 重复的精确相等检查。R1 仍整数 used*5>=K*4，各 H 字段/原阈值不改。本次修正发生在任何真实科学结果之前，纠正未授权新增条件，不是按结果改门。

最终隔离变异去掉不存在的 prefix gate，按实际九项重验；新增反例明确 prefix_groups 改变不能单独使现行门失败。此前10/10仅是先前候选事实，不作为封版门禁证据；最终九项与控制回执另补。

### 封版证据

移除误增 prefix 门后最终 **16 PASS / 5.53s**；最终 **9/9** 隔离 semantic mutations AssertionError kill（不含已移除的 prefix 门），账本 `<PRIVATE_VALIDATION>/sgns-native-full-sid-mutants-pepfusp9/summary.json`。前述48项相关回归属于修正前候选，原科学文件从未修改；封版后的16项全部包含原科学逐位、原门逐字段、R1、prefix只报数、DAG/transport/RSS反例。week26负例初次仅因更早 secure 报错文字不同而失败，已保留日志并改为匹配 holdout 语义；没有放宽拒绝。

入口 SHA256：`3e7a1b1aa06d6e8b71269d0b702ad4e45a0ee1c5ec2f3be432f7db2bacc2876b`。三授权新增文件全字节公开扫描与 `git diff --check` 在封版提交前执行；结果交 root。最终 public source 由 root 压平并全 diff 重扫，独立 C 及真实正式路径验收尚未完成。
