密级: 公开

# P4B 数据/训练管线验证

当前验收状态只读 STATE。本实现仅 P4B，不实施 P4C baseline、正式模型比较、export/index/evaluate/fusion/ANN 或 P4E 消融。真实数据只 audit/prepare；train CLI 仅接受显式 `--synthetic`。

## Scope 与输入

独立 worktree `codex/p4b-retrieval-pipeline`，基点 `0376d9a8119f3500cbf78519675cc7bf18013c1b`；P3-D 最终86个 Git-visible 文件完整逐byte带入，入场 `cache/p4b-entry-overlay.json`。P3-D manifest SHA `36fc3a451f1e8bd2dc691b15c382015b6762fec8342d318ce9771bad5adfc972`，全 bound files 与旧 producer evidence 独立校验；不改旧 producer verifier、不把新 consumer fingerprint追写旧产物。

允许新增：`src/vkgr/retrieval.py`、`retrieval_data.py`、`retrieval_training.py`，三份 `tests/test_retrieval*.py`，本报告与小型 public manifest。状态/接力/decision/issue/experiment按类别维护。受保护旧 source/tests/config/lock/冻结合同保持；无 commit/push/main 修改、新下载/资源或生命周期。

Local runtime借现有P6 env只读，`PYTHONPATH=src`指向本worktree。Server借现有Torch env只读；完整 `/etc/vast-agents-guide.md` 和 `/workspace/AGENTS.md` 已读（后者指向前者）。Server无Git repo，consumer如实记Git null；Mac基点/dirty来源在 source transport origin单独记。执行前全部上传文件逐hash核验，实际 source fingerprint/config/lock/依赖独立记录。不会把snapshot来源伪称server实际Git。

## 行为

- Stable sorted uint32 catalog map，PAD=0，item index从1；source item ID0仍可对应index1，PAD不等于source ID0。
- `(week,physical_row_ordinal)`顺序 streaming，保留旧 P3 全prefix/target/user/位置，history严格过去、chronological最近50；full prior-liked set额外保存用于negative mask。保留重复、低活跃和长尾。
- train-positive support从所有train like events收集并单独hash，包括只有一个like、不产生prefix的用户；从未train-positive的item ID branch输出固定0。
- Validation reader只读指定validation artifact；test gate关闭。prepare对既有test bytes计算hash，不decode其values或metrics，不用于选择。
- Epoch permutation without replacement，独立 RNG；尾batch不跨epoch补齐，boundary cursor保留offset=size直到下一次draw。candidate列按target item ID排序去重，保留全部queries，mask全部strictpastlikes，current target override。
- FP32 d64 IDMean小adapter，PAD masked mean，L2 dot/temperature CE、AdamW wd1e-4、clip1及loss/gradient/parameter nonfinite gate。未实现另外三种formal architectures，也未选hyperparameters。
- Skip分母是**当前已经drawn**的epoch累计和全run累计，含skipped；record后、任何optimizer step前check，两者任一>1% STOP，整batch没有negative也STOP。不是单batch的分母，也不提前借未来样本稀释分母。Tradeoff：epoch初期偶发skip可能立即触门，即使完整epoch比例后来可低于1%；按协调明确裁决保留，不静默放宽冻结合同。
- Exclusive output目录；versioned step checkpoints不覆盖，manifest记录best/last pointers。Synthetic best仅训练loss最小，不是正式validation最佳模型。checkpoint绑定model/optimizer、global Python/NumPy/Torch/CUDA RNG、cursor独立RNG、epoch、sampler stats、step/source/config/input和原始max_steps、start/累计elapsed预算。

## 真实 RED/GREEN ledger

日志原件位于 `/private/tmp/p4b-*.log`，本表记录实际顺序，不声称重演已经不存在的中间实现。所有新生产行为先由明确缺行为assertion观察 fresh RED，然后最小实现 fresh targeted GREEN。

|组|RED evidence|GREEN evidence|
|---|---|---|
|01 catalog/PAD/frozen|data module absent assertion|1 test PASS|
|02 prefix/fullpast/parity|prefix stream missing|2 tests PASS，独立旧P3 expected rows及>50手算|
|03 epoch/tail/cursor|training module absent assertion|1 test PASS，无跨epoch填充与restore|
|04 columns/mask/CE/stats|candidate/loss missing，两assertion|3 tests PASS，手算logsumexp oracle|
|05 d64 IDMean/update|adapter missing|4 tests PASS，参数真实变化/PAD/unsupported|
|06 checkpoint/restore/tamper|checkpoint missing|5 tests PASS，optimizer/model/cursor一致、hash/binding拒绝|
|07 split/content|split reader missing|3 data tests PASS，requested split、ID join、3/4/5归一化|
|08 exclusive prepare/hashes|prepared artifact missing|4 data tests PASS，exclusive/tamper拒绝|
|09 gate接线|all-skipped无ValueError assertion|6 training tests PASS，gate在optimizer前|
|10 process/overfit|CLI absent assertion|独立process7→20与uninterrupted逐batch/loss/parameters一致|
|11 audit/resource|audit/resource两缺行为assertion|CLI3tests PASS，missing pins与资源门|
|12 single-like support|显式full-support输入missing assertion|data5tests PASS，singleton独占item仍supported|
|13 producer文件名|明确incorrect legacy filename assertion|CLI4tests PASS，实际existing artifact的hash gate|
|14 joint tail/process|batch长度2,2,2,2与期望3,2,3,2 assertion|CLI4tests PASS；5query/batch3，独立process停7(shorttail前)/8(epochboundary)两位置恢复，逐batch/loss和final model均一致|

真实失败保留：首baseline在sandbox内19个PermissionError，不算行为RED；获授权worktree写入方式重跑旧97tests PASS/skip1。09接线后两个旧toy fixture含无negative row，新增gate正确拒绝；将“参数变化/恢复”fixture改为有实际negative的两rows后fresh PASS（不是放宽gate）。11首次verify传str而read_json要求Path，已有process test失败后修复。首次upload ledger误用system Python缺PyYAML，未计RED，保留空taskdir；改借批准env并新exclusive目录。13首次为FileNotFound error，不计有效assertion RED；调整test将incorrect legacy filename明确assertion后重跑RED。13源码修复后旧source transport fingerprint也拒绝，刷新新版本ledger后GREEN。提前verify真实prepare尚未终端完成，exit2保留；不覆盖、不伪称成功或重复启动prepare。

## 版本与真实验证

Real-v3 source fingerprint `5868c2c5db3df83bcd8f15a0235aba39a31ea4a6d532f31a4a69dfe1699fe6fe`，独立source spec/quality PASS。Final-v4 source fingerprint `6606b1d06cb214ac26960718043cfb695cff71b5d26b61c6ce0c5e6cb9a104f8`，仅synthetic fixture从2query/batch2加固为5query/batch3；audit/prepare/data/training bytes与real-v3相同。不能把real-v3产物冒称final-v4运行，真实大产物留原版本和server-only。

真实 v3 audit、prepare、终端成功后verify-prepare-final均exit0/stdout success或verified/stderr空；上传92 files逐hash通过。Prepare manifest SHA `7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2`，完整旧P3每个target/history/位置、全exposure catalog与全like支持reconciliation全部true。实际server roots/日志/exit路径只在ignored `cache/p4b-private-evidence.json`与private logs保存。Prepared arrays/files留server，public摘要为 `artifacts/p4b-pipeline-manifest.json`；Astra独立不importproducer重读25train partitions和官方NPZ，全部target/history/位置/fullpast同时匹配旧P3与新prepared，catalog/support、全部normalized64D逐float和artifact hashes均PASS（exit0）。作者另不importproduction的完整raw/content/input-hash oracle也exit0/PASS；该自检与Astra独立证据分开。

Final-v4 server synthetic targeted15tests21.342s全部PASS/exit0；包含joint shorttail/boundary fresh-process恢复。Final-v4 local fresh完整regression112tests13.488s PASS/skip1/exit0，FAISS skip不冒称实跑。Astra另fresh完整112tests13.677s PASS/skip1；第一次因sandbox拒绝旧P1 temp目录19errors不计有效RED，正常授权方式重跑源码/tests不变。独立review real-v3 spec/quality PASS；final-v4 scoped synthetic/source/docs/public review PASS，Astra正式验收。

Server final-v4 persistent continuous20、initial7、resume20、verify四stage均exit0/stderr空，versioned manifests/checkpoints保留。作者另不importproduction核所有逐batch/loss、model/optimizer/global RNG/cursor/sampler stats等同；checkpoint bytes因各run start/elapsed预算不同不要求hash相同，分别绑定各自hash。额外toy loss/资源/数据基数只在experiment_log。

## 验收与限制

命令：`PYTHONPATH=src <existing-python> -B -m unittest discover -s tests -v`；targeted使用 `-p 'test_retrieval*.py'`。CLI `audit/prepare`绑定现有train/holdout/raw roots；`verify --manifest`验证prepared/hash/schema/source，synthetic训练参数为 `train --synthetic --max-steps 20 --until-step 7|8|20 --output <new-dir>`，resume指向原目录versioned manifest。Exact server commands以private evidence保存；public文档不含连接信息、真实IDs或大数组。

Hash保证一致性，不证明上游真实性；official order非独立timestamp验证，static content snapshot生成时点Unknown，test26历史已用。这不是formal model comparison、batch pilot或full validation/test评估。Resources是可见sampling headroom，不是host承诺配额，不声称host-loss persistence。准备阶段输出大小仅server；失败目录/日志不删除、不覆盖，不声称I/O transaction。

P4B Complete / Astra正式验收PASS。下一P4C由协调另派，本worker停止。下一task不由本worker启动。

## Final coordinator remote synthetic verification

Astra independent remote verification: three real stages and four synthetic stages each exit0, stdout success/verified, stderr empty. Both runs have identical complete records, model, optimizer, cursor, stats, step, and Python/NumPy/Torch/CUDA RNG states; batches alternate3,2 over10epochs, with actual parameter changes and decreasing loss. The first verification script invocation used an incorrect undated existing-Python path and exited127 before verification ran; using the correct private-index environment then passed. This path typo is not a functional RED. Source and public/server manifests remain unchanged.

## Coordinator acceptance and final overlay

Astra accepted P4B Complete after independent final source/docs/public review, fresh112 regression (skip1), exhaustive raw/content oracle, real stage logs/hashes, and complete remote synthetic state equivalence all passed. Only closing governance docs and the public status were synchronized; source/tests/frozen protocol and runtime prepared manifest remain unchanged. Public manifest status is `accepted_by_coordinator`, SHA256 `d31cc460273c3d94afc48b70f976cd0697aa893af5107d2495f772fd031781f0`. Runtime prepared manifest retains SHA256 `7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2`. Next P4C is dispatched separately; no commit/push or lifecycle action.
