密级: 公开

# P4F artifact/science acceptance ACCEPTED（2026-10-11）

固定第三 attempt 未重跑。Fresh-context review 对 target review SHA `c227d3ab6c75bf632b5ebde550b51c3397fc75510494f4ce5c3c95e77885d437` 的 verdict 为 `APPROVE`；review report SHA `92459e7429291d50fb7bb87472c1ea6380fac604b5e255a62cdbbd0e771dd9f3` 已由父 orchestrator 独立核验。P4F 正式关闭，下一 task 仅 P4G；不重跑 P4F。

接受范围：input/config/query/producer hashes、export、65个 wRRF candidates、公平 quota、serialized HNSW、20个 inference shards 与 resume/verify。必须持续披露 whole-attempt wall-clock UNKNOWN、ANN仅固定512-query validation audit、resource仅process-scoped、test26非pristine、无host-wide syscall trace、raw strict-before-target/like合同只继承P3 hash。Exact full validation仍authoritative，offline infer仍走exact cached routes。

# P4F 本地 implementation 验证

最新v2覆盖下方v1 regression待裁决状态：Astra已批准历史fixture定点修复，fixed historical members/hash及真实CLI检查通过，full regression与独立新增diff review均通过。交付入口`artifacts/p4f-source-manifest-v2.json`及`cache/p4f-local-review-v2.json`，awaiting coordinator acceptance。真实FAISS仍pending，不称P4F实验Complete。

当前为 local implementation ready / awaiting coordinator review；P4F真实实验尚未执行。Astra implementation-brief-v1允许三个新consumer模块和matching tests，原producer、合同、配置、lock与global CLI保持原bytes。P4E已accepted；owner_actual_exit永久null的accepted limitation保留。旧heartbeat已PAUSED，不再检查已关闭队列。

## 接口与证据

`retrieval_fusion.py`提供固定候选enumeration、exact Fraction weighted RRF、公平quota及validation selection。零权route不贡献候选，同分按item ID；四route为ItemCF、冻结best-core TT、content、Pop。指标与执行数字唯一见`experiment_log.md`。

`retrieval_ann.py`加载显式hash绑定的checkpoint与完整producer binding，沿原producer导出vectors；可缓存四route排名。新consumer identity与历史scientific identity分别记录。FAISS lazy import；真实CPU HNSW及固定old audit query identity核验路径已实现，本地dependency缺失，真实integration明确pending。Index disk bytes、可见process RSS、search-only及mapping/exact-rerank计时边界分列；不称全历史资源峰值。

`retrieval_infer_run.py`独立入口为`python -m vkgr.retrieval_infer_run`，subcommands为export/index/fuse/infer/verify。均需要`--input --input-sha256 --output`；index另需export及SHA，真实index另需audit、exact-predictions及各自SHA；infer/verify另需selection及SHA。infer支持`--shard-size --max-shards --resume`。真实input仅validation，prepared/baseline/core-selection/core-run完整producer链校验；历史producer pins不改为当前consumer。

离线infer当前使用exact缓存route；ANN比较由独立index audit提供，未声称ANN offline backend。Eligible query coverage与selection绑定在创建输出前核实，OOV保留分母；empty history Pop fallback独立报告。Exclusive atomic publication使用fsynced staging与hardlink，owned staging保留并计入未来disk预算。Resume核shard顺序/hash/身份，恢复已发布但progress未发布的orphan；verify仅核cached RRF/prediction/metrics，不重放embedding或训练。

## 本地验证边界

针对新增consumer的fresh TDD、独立read-only scoped review与related regression证据入口为`cache/p4f-local-review-v1.json`；公开hash入口为`artifacts/p4f-source-manifest-v1.json`。Reviewer未发现剩余Critical/Important。全套扩展regression仍须按证据记录实际结果，不能由定点通过替代。

macOS本地验证未证明Linux实际FAISS、断电durability或真实资源成本。无SSH、部署、安装、训练或test metric读取，无commit/push。下一步由Astra fresh验收本地产物，另行冻结small CPU synthetic server gate及有限预算/exact invocation；任何真实P4F evaluation仍待后续gate。


### Stat后本地完整候选（2026-10-10）

入口cache/p4f-stat-readiness-v1/review.json；数字唯一experiment_log。Core路径由accepted P4D public run logical ref及冻结run_session writer证明，实际stat确认regular/尺寸，尚未fresh读取内容/hash。原8MiB单项门导致只读拒绝，新binding仅本地角色限额修复并保持默认门，浮点溢出同步拒绝；原validator AST/pins/source和历史fixed包保持。

全部pending metadata/header/resource合并为一个待审proposal，未远端执行。Real execution candidate包含现有CLI固定顺序、科学合同/输入/audit pins、有限投入proposal、验收/stop gates，以及现有A/B runner不能原样执行真实阶段的具体adapter文件清单。Native memory/serialized artifact growth/throughput/shared G reserve未知有具体证据缺件，不猜容量或声称runtime READY。全部已列旧G jobs由G单owner接收，本线撤销责任，不额外SSH/kill。Astra一次wait/local验收；真实ANN仍关闭。


### P4F raw/public 分离与完整 entry 本地候选 — 2026-10-10

Astra mapping-only 接收后完成分别冻结 raw producer 与 public consumer identity 的 context、paired lineage/audit guard、metadata 尺寸前置，以及合并 receiver/preflight/foreground owner actual wait 入口。科学 source/consumer 保持；接受的 runner/pipeline/binding 原 bytes 复用。新相关 TDD、候选 canonical/argv/member/hash 一致性及 PENDING 完整入口拒绝均有本地证据；数量/时长唯一 experiment_log。入口 cache/p4f-separated-entry-v1/review.json。awaiting coordinator whole difference acceptance；真实预算、fresh permit、服务器写入和 native gate 未放行。macOS fixture 的 resource API stub、render server path stub 不替代 Linux 实际证据；部署失败部分文件保留、timeout 不推断停止，resource 为点/segment 观察非历史峰值。


### P4F foreground 生命周期与资源 STOP 差异修复 — 2026-10-10

原审阅 I1 初始 identity 异常跳过原 wait、I2 resource失败仅记日志均已本地 fresh反例复现并定点修复。任意 post-spawn helper异常也保留独立 original wait兜底；身份unknown不signal，timeout保留unknown。首次resource失败仅对 saved/current identity相同的原supervisor发SIGALRM，进入已验收 owner Stop/finally 以收束独立stage/proof sessions，不直接TERM/KILL owner。真实 accepted pipeline+独立stage fixture核后续marker被阻止、stage actualexit保存及child消失；macOS不替代Linux真实closure。原v1全部members及其余controls/source/science保持，相关hash重冻，PENDING完整bootstrap仍写前拒绝。入口 cache/p4f-separated-entry-v2/review.json；awaiting original reviewer difference acceptance，未SSH/native，数字唯一experiment_log。


### P4F v2 ACCEPT与本地 exact refreeze method — 2026-10-10

Astra原reviewer关闭I1/I2并ACCEPT_LOCAL_I1_I2_REVISION；保存裁决副本于cache/p4f-exact-refreeze-method-v1/coordinator-acceptance.json。尚缺确定性现场permit生成器，已补仅本地allowlist status/start/deadline及派生wire/bootstrap/frame/transporthash refreeze方法，逐个核固定v2 members，审批scope/evidence与fresh原start前置，exclusive/fsynced小产物保存。定点synthetic测试不是human授权或真实permit；无SSH/native，入口cache/p4f-exact-refreeze-method-v1/review.json待Astra一次验收。


### P4F refreeze方法协调ACCEPT — 2026-10-10

主独立验收refreeze-method-v1，保存裁决cache/p4f-exact-refreeze-method-v1/method-coordinator-acceptance.json。本地技术gate已接受，不是真实human approval；授权JSON/evidencehash不能认证human。本执行对话未收到完整部署/preflight/native-budget许可前不生成真实ADMITTED或发送服务器；许可到达后只按已审allowlist现场绑定、持久所有hash、核固定scope后一次执行，无需主重复审批。未重跑suite或新SSH。


### P4F 完整entry单次实际STOP

完整scope直接human授权后按已审method固定refreeze/allpins保存及发送一次；原transport actualwait/reap完成但非零STOP。返回foreground_result hash及owner_actual_exit null，具体failedphase、native是否started和servergroupsclosure未知；不称模型成功/未启动或所有进程停止。原receipt/budget/partialfiles保持，不retry/repin或另probe。入口cache/p4f-actual-entry-v1/review.json，下一Astra裁决原receipt bounded读取。数字唯一experiment_log。


### P4F 原STOP收尾定位

固定foreground receipt SHA只读核通过，preflight lineage/ValueError非零/reaped，owner未called；accepted入口顺序排除七stage/model/ANN已启动。具体lineage条件与freshglobalclosure/资源成功仍unknown。只读原固定文件，无serverwrite/native/实验retry；入口cache/p4f-foreground-receipt-read-v1/review.json。数字唯一experiment_log。


### P4F local precise lineage diagnostic revision

Proved wrapper generic-lineage telemetry loss; exact original mismatch remains unknown without full pinned raw documents. New ignored candidate retains frozen validator literal/AST and predicate semantics, instruments bounded fixed predicates, and preserves all science/pins/other deployment bytes. Complete PENDING transport is a review artifact blocked on spent original root, not remote permission. Full field-source matrix, merged gap and verification evidence: cache/p4f-lineage-local-revision-v1/review.json. Awaiting Astra difference acceptance; no new remote action or real-input replay.


### Accepted local diagnostic; exact merged readonly proposal

Original reviewer accepted local telemetry difference only. Prepared cache/p4f-lineage-readonly-proposal-v1/review.json with accepted diagnostic bytes, original context-derived input and hash-before-exec stdin bootstrap. Original metadata/header/resource scope preserved; no deployed server code execution, no extra probes, no SSH or new ANN deployment. Exact real lineage cause remains unknown; await Astra exact read scope decision.


### P4F precise audit serializer root cause and minimal local patch

Once approved readonly diagnosis yielded legacy_membership_sha/legacy_audit STOP. Producer and frozen consumer canonical_bytes includes LF; diagnostic/binding/stage helper membership hashes omitted it. New local patch changes only those membership byte calls; AST undo proof preserves all other behavior and global wire canonical. No expected pin/raw IDs/source/science changed. Evidence cache/p4f-legacy-membership-serialization-v1/review.json, awaiting original reviewer difference acceptance. Real corrected hash/pipeline not replayed; no new SSH/start bundle.


### P4F distinct new-attempt integration candidate

Accepted three serializer helpers integrated into accepted separated entry/lifecycle/refreeze mechanism. New exclusive root and all dependent code/context/plan/argv/deploy/framing/bootstrap/transport pins complete; historical input/science/consumer untouched. PENDING candidate denies writes, fixed new identity refuses old approval. Entry cache/p4f-fresh-attempt-candidate-v1/review.json awaiting Astra integration review then direct human second-attempt budget decision. Old spent root/permit/partialfiles/consumption preserved; no SSH/real corrected hash/full pipeline proof. G concurrence still requires joint on-site reserve.


### P4F fresh-attempt integration accepted; second human gate pending

Original reviewer accepted local integration with no blocking findings; copy cache/p4f-fresh-attempt-acceptance-v1/coordinator-review.json. Fixed candidate/approval-scope remains proposal awaiting direct Eddy second-attempt approval in P. No real permit/SSH/ANN; first spent approval/consumption not reused or refunded.


### Second full attempt actual STOP

The directly approved accepted candidate was hash-verified, deterministically materialized and submitted once. Original transport wait returned exit 1 / reaped / no observation timeout. Sanitized foreground STOP and null owner exit do not establish a specific preflight cause or current server closure. Review evidence cache/p4f-fresh-attempt-actual-v1/review.json SHA 28bf419db97743d829a996c81814dbbb3b7de49d75f7a8bdd0298d4b1aadeadf. No native success, retry or extra SSH probe.


### Second attempt fixed receipt and local schema diagnosis

Hash-bound reader retained its accepted SHA; only fixed path/artifact override changed. Actual preflight returned KeyError at raw_public_pair_or_context, exit 1 / reaped / no timeout; historical owned group closure recorded. Pipeline not invoked. Exact missing key is not in reader summary. Frozen deployed producer schema lacks audit_set_hashes, supplied by public extension; local fixture reproduces that KeyError. Minimal helper fix preserves shared producer fields, raw audit artifact and public lineage/membership pins, unchanged actual audit checks and all other module AST. Local review ffda1e4a3d91600902a471c82ad1acd37bf3dad133c6e226f35881373f8b5ae3; receipt read review 9d5f7bc1d75571fcb51aa47369ea1d22daff70496f287e1ff9a2482ad0503e35. Awaiting coordinator acceptance, no new attempt.
