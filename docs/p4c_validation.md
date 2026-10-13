密级: 公开

# P4C baseline validation

当前状态只读 STATE；仅执行 P4C。输入沿用 frozen traditional-like-v1、P3/P3-D/P4B 与旧 P4 validation audit。Test gate closed，仅既有 test bytes hash；不进入 P4D，不commit/push。

## Ownership 与 source

新独立 worktree `codex/p4c-traditional-baselines`，父 P4B accepted Git-visible94文件逐byte继承，ignored ledger `cache/p4c-entry-overlay.json`。新文件 retrieval_baselines.py、test_retrieval_baselines.py；retrieval.py 增加 baseline-pilot/baselines/verify-baselines。冻结合同/config/lock/旧生产模块与测试只读；本阶段治理文档按类别更新。新 consumer独立记source/Git/lock/依赖，P4B prepared manifest/source保持静态原身份，不把新source追写旧产物。

## 行为与边界

Pop从所有train like events计数；ItemCF从全部train likes构建binary user-item graph，包括single-like用户degree。每个item逐row sparse Counter计算cosine，排self edge，保留正cosine top200，raw item ID精确tie；query按原history事件顺序float64累加，保留重复；shortlist用共享Pop去重补齐。无catalog-square或user-item score矩阵；零邻居tail留在catalog。

Content复用已验收normalized official64D features，normalized history mean、fixed-pair float64 reduction on float32 vectors exact retrieval；zero安全，raw item ID tie。三方法共同full validation query/catalog/target/OOV分母，K10/20/50；旧validation512 membership/history/target及Pop/content top20逐值回归，@10/20 metric equality。新test values未读取。

Verifier从pinned raw train重新扫描全部like graph，比较pilot binary incidence、Pop/degrees和重建top200 neighbors；从pinned features重算全部content ranks，重算三方法predictions/metrics及coverage/fill。Final/initial manifest前三个artifact hashes必须逐一相等，final多的legacy proof也逐hash核验。Hash只保证一致性，不证明上游真实性。

## TDD ledger（日志实际保留）

- 01：missing baseline assertion RED；Pop repeat/single-like手算 GREEN。
- 02：missing neighbors两assertions RED；binary cosine/single-like degree/no-self/tie GREEN。
- 03：missing scoring/evaluation两assertions RED；history repetition、float64 sum、Popfill/zero-tail、content brute/block parity、OOV分母 GREEN。
- 04：missing persisted run/static prepared两assertions RED；exclusive outputs/shared pins/query-order/source/tamper GREEN。
- 05：CLI pilot missing command RED；synthetic run/verify与test split拒绝 GREEN。
- 06：missing bounded graph pilot RED；手算binary edges、Σuserdegree²与maxdegree GREEN。
- 07（协调review）：重hash错content、空graph/degree、零Popcounts与final hash冲突四个assertions RED；input-to-output重算与final bindings修复 GREEN。第一次GREEN仍因local v1 transport guard正确拒绝新source；保留v1 origin后fresh targeted11 PASS。此失败不伪装为新增行为RED。

- 08：real verify输入加载早于首resource gate的assertion RED；先resource再加载、重算中采样、最后gate与实际elapsed/minimum scope写stdout GREEN。

日志：`/private/tmp/p4c-red-01.log` 至 `p4c-red-07-review.log`；各GREEN与fresh regression独立保存。入场112tests PASS/skip1；source v1 fresh121tests PASS/skip1；v2 fix fresh123tests PASS/skip1；追加verify首资源门后final fresh124tests PASS/skip1。FAISS skip不冒称实跑。独立协调toy随机oracle与review是外部证据，作者自检不替代fresh-context review。

## Server 与资源

完整server guide/runbook已读。Source v1上传96files逐hash/source fingerprint验证，只有bounded pilot已实际terminal success（exit0/stdout success/stderr空）；预先无其他VKGR任务，resource headroom通过。现有environment缺SciPy，未安装；pilot实测全train graph incidence、Σdegree²和最大work rows，不用总likes推成本。v1原目录/source/manifest保留；source修复后的v2使用新exclusive目录，不复用或重stamp v1。Final source d3f0eb1c644e95430e8eb27703515159ceda7099ca95d54853693aa0d2d99452、97files逐hash/source核验；server12 targeted通过，v2pilot实际terminal成功，full validation实际terminal success、旧validation512 regression通过；verify实际terminal success、stdout verified/stderr空。

所有实体数组/graph/predictions只留server；私密run roots/commands/stage handles/logs/exit paths只在ignored cache/p4c-private-evidence*.json。实际实验数字唯一docs/experiment_log.md；本报告不复制结果数字。

## 验收命令与当前限制

`PYTHONPATH=src <existing-python> -B -m unittest discover -s tests -v`；targeted `-p test_retrieval_baselines.py`。真实CLI分别baseline-pilot、baselines、verify-baselines，显式绑定train/holdout/raw/prepared/legacy-validation；full与verify额外绑定pilot manifest。每stage独占新日志/exit handle，先terminal success才verify，不重复启动。

Full validation/verify均实际terminal成功；input-to-output replay/final bindings及verify首资源门的独立scopedreview均通过，Astra独立124tests PASS/skip1。指标/fill/coverage/latency/RAM实际结果唯一experiment_log；生产verify已实际重扫raw/fullgraph/fullcontent与三route完整predictions/metrics通过；Astra独立全量oracle已通过，closing docs/public review仍待裁决，保持awaiting coordinator acceptance。Latency按route披露scope：Pop每query copy不含shared sorting；ItemCF per-query sum/sort/fill不含fit/Pop sort；content batch-amortized mean+exact scoring+ranking，不用不同口径宣称速度胜负。RAM是全process high-water mark，可见headroom sampling不是瞬时hardquota。Static snapshot时间Unknown、official-order非独立timestamp、旧test历史已用、MVP subset代表性限制持续。

## 可读artifact schema与只读复核入口

真实消费者必须读取 `baseline-final-manifest.json`；`baseline-manifest.json` 是evaluation完成后的中间记录，若后续legacy/resource门失败，只能视为partial evidence。Formal verify拒绝initial filename，final artifacts必须精确包含initial三项+legacy，前三项hash相等。

|Artifact|Schema与职责|
|---|---|
|pilot-manifest.json|kind baseline_pilot；consumer/input/prepared/legacy pins、incidence hash、bounded work与resource evidence|
|incidence.json|catalog/raw IDs排序、全train liked users的unique sorted item sets、Pop event counts；server-only|
|baseline-final-manifest.json|schema1/task P4C/kind baselines/status success；shared input/producer/consumer/pilot绑定、query/catalog identity hashes、ks、artifact hashes、RSS/elapsed/resource scope|
|graph.json|catalog、Pop counts、distinct-user degrees、raw ID keys到sorted positive cosine top200列表；self edge排除，零邻居保留|
|predictions.json|popularity/itemcf/content三个同长度列表，按完整pinned validation query原顺序，每row全catalog的unique top50 raw IDs；server-only|
|metrics.json|三route各all/train_observed/oov count与Recall/NDCG K10/20/50；diagnostics包含zero centroids、coverage、fill、逐route latency及scope|
|legacy-regression.json|membership/history/target equality及Pop/content top20/@10/20 equality booleans；不读取test values|

精确private CLI输入/commands与stage stdout/stderr/exit文件角色在ignored `cache/p4c-private-evidence.json`；历史父证据明确在parent_p4b，top final_source_fingerprint只指P4C final consumer。所有路径留private，不复制到public manifest。

可复用入口 `cache/p4c-readonly-check.py`（ignored）：仅SSH读取既有输入/产物，不import生产vkgr、不下载实体数组到Mac、不写服务器文件；独立重扫全train binary graph与Pop/degrees/neighbors、全query三route排名及指标。此入口是作者准备的arithmetic check，不能替代协调独立oracle或fresh-context review；当前未执行，避免与live verify重叠。协调已经准备自己的独立入口。

证据异常：一次接力聊天误填未经读取确认的runtime manifest SHA，已立即撤回并fresh读取实际aggregate更正；repo/server artifact未改。Experiment数字/hash由实际小aggregate自动生成，closing public manifest同样从真实字段programmatic生成并fresh核bytes。

## Closing package — 历史review快照（正式接受见文末）

Final source与protected inherited bytes保持不变。真实final pilot/full/verify三stage均实际terminal exit0、stdout success/verified、stderr空；完整commands/logs/exit与server roots在private index，不复制到public。Formal verify要求final filename，完整重建输入到outputs及旧validation回归。Run manifest hash从actual bytes/programmatic aggregate读取，public manifest仅含来源/hash与声明，数字仍唯一experiment_log。

Fresh-context source review与修复scopedreview通过；Astra fresh124完整regression PASS/skip1。协调另独立全raw binary graph/Pop/degrees/neighbors与所有Pop/ItemCF query rank已通过；content oracle第一次使用一维norm，与冻结二维axis=1 float32归约不同，在近tie处产生rank交换。改为匹配冻结归约后单例global lexsort严格一致，d,id与qd,id fixed-pair scores逐值相等；未修改production、dtype、tolerance或tie rule。Astra修正版全量oracle已实际exit0，通过全部raw计数/binary degree/独立user-pair全图、全部query三路严格排名、所有slice metrics/coverage/fill及旧audit回归；日志/private/tmp/p4c-astra-oracle-v2.log。最终docs/public review仍待裁决，不写Complete。

不是任意数值实现都能得到相同排名：必须先保证生成的float32 centroid vectors逐float一致，再使用冻结fixed-pair float64 score与raw ID tie。1D norm/BLAS或不同归约实现可能产生最后几个bit差异；不能用epsilon bucket掩盖。失败与诊断脚本私密保留，见issue/experiment证据。

当前收口状态Implementation ready / awaiting coordinator acceptance，Astra独立全量oracle已PASS；只剩closing docs/public review与协调裁决，P4D未开始。未commit/push/main写入、下载、安装依赖、新资源或生命周期操作；失败/历史目录与v1 source/origin/ledger全部保留。

## 正式协调接受与final overlay

Astra正式裁决P4C accepted / Complete。独立source/scoped修复及final docs/public review均PASS；Astra fresh124 regression（skip1）、全raw like计数/binary degrees/独立user-pair全图、全部query三route strict ranks、所有slice metrics/coverage/fill与旧validation512回归均PASS。Coordinator oracle首次norm归约不匹配的失败/诊断保留，修正版全量strict oracle通过；未改production或容差。

本次只同步STATE/handoff/public接受字段和本closing证据，不改source/config/指标；final source与runtime artifacts/hash保持。Public status为accepted_by_coordinator，数字仍唯一experiment_log。P4C完成不表示commit/push/合并main或P4D训练；本worker停止，P4D另派继承最终overlay。
