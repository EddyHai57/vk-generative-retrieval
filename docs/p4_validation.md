密级: 公开

# P4 validation

## Scope与冻结输入

仅tasks P4，Eddy过夜授权与Astra任务brief允许独立worktree、TDD、现有服务器真实baseline与阶段review；不commit/push/新资源/P5。原生worktree `codex/p4-baselines`，基点0376d9a8119f3500cbf78519675cc7bf18013c1b。P3/P2已验收21文件overlay逐字节复制，ignored `cache/p4-entry-overlay.json` SHA256 `315ad2af2454e9f4f13aaf1109d25d46c1206962782690a704e34302021fa7a5`；main/P3/P2原件不改。

允许修改：新增src/vkgr/baselines.py、baseline_run.py、tests/test_baselines.py、test_baseline_run.py、test_baselines_faiss.py、configs/p4.json、本validation、必要STATE/handoff/architecture/decision/issue/experiment与脱敏aggregate manifest。不改P3源码、policy、lock或冻结产物。

输入只读消费P3 v3 frozen contract、train audit/catalog、holdout audit/query全集和P2已下载raw。Exact输入路径/hash唯一入口p3_validation最终artifact清单。P4 static loader先验证固定freeze SHA与policy、train audit和producer binding，再验证holdout及query hashes；不调用P3 train/freeze/holdout重新生成，不放松原producer守卫。总source fingerprint增加P4模块自然变化。

输出：新唯一server root `/workspace/vkgr-p4-20261006t2011` 的独立code快照、server-only audit query/IDs/逐query prediction、metric artifact和run manifest。Mac只保存原创源码、synthetic tests、脱敏manifest和小aggregate。Stop gates：oracle不一致、input/hash/label合同冲突、feature missing/nonfinite、leakage、安全/许可或deadline。

## Evaluator定义

原始官方float16[64] content按item ID join，catalog embedding转float32并逐行L2 normalize；zero行保留零。冻结history含repeat，取已normalize embeddings的float32均值，再L2 normalize。Zero centroid保留全零score，按item ID升序tie并计数；不造替代feature，不mask seen item。

最终score arithmetic：float32 feature/centroid输入，NumPy `einsum('qd,id->qi', ..., dtype=float64, optimize=False)`固定每pair dimension reduction。Brute toy与blockwise用同一算术，query/item block shape不改变score；不使用epsilon buckets。实际最大score block为32×4096，float64块最多1MiB；另有bounded top-k merge scratch。绝不物化完整query×catalog score matrix。Brute仅synthetic oracle，真实run不用。

Popularity只数train like=True事件，不去重，所有train exposure catalog item保留包括零计数。Content与Popularity共用同一冻结candidate/history/features/query split。Seed42按SHA256(`42:split:user_id`)升序（hash tie再user ID）抽每split最多512；不读取target/OOV/hit作抽样条件。完整selected IDs仅服务器，公开只保存其canonical ID list hash。

Item Recall/NDCG@10/@20同时给all/train-observed/OOV与denominators；OOV不替换target、不缩分母，必须miss。空slice质量为null/count0。单relevant target的NDCG为命中rank的1/log2(rank+1)。没有训练或调参。

FAISS reference仅toy：现有官方P2隔离FAISS1.11.0环境IndexFlatIP检索全部toy candidate scores，再按冻结item ID tie canonicalize。FAISS原始同分顺序无item ID保证；不声称任意近分数float32 FAISS与本float64 reduction逐位相等，不在真实评估用其ANN/IndexFlat取代exact。

## 环境与provenance

沿用uv真实精确lock：CPython>=3.11,<3.12，PyYAML6.0.2、NumPy1.26.4、PyArrow19.0.1；P4无新增依赖或第三方代码复制。原lock SHA `323c9b2c61a2f2baa4fc89769afd5908562f91d15b92672eda71aa57056e3575`。官方包与MIT/BSD/Apache声明及完整发布残余沿用P3/P2 provenance与I-007；FAISS精确独立环境/官方channels见p2_validation。

Mac Python3.11.17；首次`uv sync --offline --locked`因cache缺PyArrow wheel exit1（非TDD RED）；按已授权联网从lock固定官方distribution补齐，`uv sync --locked` exit0。没有改系统Python/shell/driver/torch。Server新P4 code-v1/.venv Python3.11.16，同精确依赖；code-v2复用此P4独立env，不污染P2/P3。

服务器完整/etc/vast-agents-guide.md与/workspace/AGENTS及P3目标AGENTS已读；只用SSH alias，不记录实际地址/port/identity。Git对象只读复用P2，`GIT_OPTIONAL_LOCKS=0`；不新建remote或改Git配置。实际thread环境为OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。

## 真实TDD ledger

所有命令在P4 worktree、`PYTHONPATH=src .venv/bin/python -B -m unittest ...`。每次缺失API为预期feature RED，没有以依赖失败冒充。历史工具输出为证据，不宣称重新执行已不存在的中间版本。

| Round | Fresh RED | Minimal GREEN |
|---|---|---|
| R01 exact | `discover -s tests -p test_baselines.py -v`：1 test缺brute_topk ERROR；首轮shell末尾cat使shell exit0，但log明确FAILED，不能据此称通过 | 实现brute与bounded blockwise，1 test/exit0；literal不同tie、partial block、k>catalog |
| R02 features/metrics/sample | 同命令4tests：3缺API ERROR，exit1。人工Popularity期望按catalog[3,2,1,4]写错，实施前纠正为[1,0,2,0]，再次fresh RED仍3 ERROR/exit1 | join、normalize、repeat centroid、missing拒绝、Popularity、OOV/空slice、target-independent sample，4 tests/exit0 |
| R03 config/hash | `discover -s tests -p test_baseline_run.py -v`：2缺API ERROR/exit1 | strict config、checksum reader、metrics重算，2tests/exit0 |
| R04 integration | 3tests：evaluate_inputs缺失 ERROR/exit1 | two-baseline literal、zero/OOV、exclusive输出，3tests/exit0 |
| R05 static freeze | 4tests：load_static缺失 ERROR/exit1 | fixed contract/hash先拒绝、冻结input joins/train-only counts，4tests/exit0 |
| R06 replay manifest | 5tests：run缺失 ERROR/exit1 | run→verify、重hash的prediction tamper仍拒绝，5tests/exit0 |
| Review fix1 | `discover -s tests -p 'test_baseline*.py' -v`：12tests，64D identical-vector partial block FAIL、relative root input binding ERROR、FAISS skip1，exit1 | fixed float64 reduction+canonical roots，12tests/exit0(skip1)。fresh full58tests/exit0(skip1) |

首个完整Mac v1 regression：56tests、3.305s、exit0、FAISS1 skipped；55实际executed。Server v1：56tests2.352s、exit0、skip1；P2 FAISS显式1test0.012s/exit0。此GREEN不足以关闭后来发现的I-032。

## 独立review与v1无效证据

Astra安排独立Sol fresh-context source/spec/quality review，发现两项阻塞：I-032 float32 GEMM随block shape改变最后位，使相同64D item tie失真；I-033 loader保留relative input证据但manifest顶层absolute，verify导致inputbinding mismatch。两项均真实复现后最小TDD修复，不用epsilon改变tie。

v1 fingerprint `a76702cdca02450004f22293ac2e35586a1d26e52ca34ccbfc8878a38e801f1c`；原包SHA `d23ad3038384814bac80adc684cf6ac5a57329338564d8d4926b9d71e3050a1f`，server code-v1保留。v1真实tmux任务在预处理期尚未产出最终baseline artifact；核实其专用pane22417/child22419、cwd code-v1后，按Astra明确授权仅TERM该invalid进程。新cache/run-v1.status标`aborted_invalid`，旧stdout/stderr/exit/source不删除，不用于效果比较。不触及实例或其他任务。

独立Sol fix1 scoped re-review通过：原反例全部[[1],[1]]；另20组mixed64D/repeat、多blocks与brute一致；relative路径修复已确认，无新source/spec/quality阻塞。最终整体gate仍由Astra裁决。

## v2源码与fresh验证

最终v2 fingerprint `51db4ea2320b0f8aae0a92d2036ba273bd1984966c6db553b640a5f17956176e`；48文件源码包SHA `63f38173191f9bf6c360dcf8f2e192941e375e9404bbe9d38f33310e53e17252`，server `/workspace/vkgr-p4-20261006t2011/code-v2`；Mac与server fingerprint一致。

Mac `PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`：58tests3.346s、exit0、skip1（57executed）。Server code-v2：58tests2.582s、exit0、skip1；`/workspace/vkgr-p2-20261006t1755/cache/faiss-gpu-env/bin/python -B -m unittest discover -s tests -p test_baselines_faiss.py -v`：1test0.019s、exit0，无skip。

## 真实run与verify命令

以下仅v2有效候选；当前执行状态见STATE，quality数字唯一人工摘要见experiment_log。使用专用tmux `-L vkgr-p4-20261006t2011 -f /dev/null`，日志与exit文件在code-v2/cache，SSH断开不改变原件。

```bash
cd /workspace/vkgr-p4-20261006t2011/code-v2
export GIT_OPTIONAL_LOCKS=0 GIT_DIR=/workspace/vkgr-p2-20261006t1755/.git GIT_WORK_TREE=/workspace/vkgr-p4-20261006t2011/code-v2 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
../code-v1/.venv/bin/python -B -m vkgr.baseline_run run --manifest /workspace/vkgr-p3-20261006t1855/code-v3/outputs/p3-train-v3-20261006t1920/frozen-contract-manifest.json --holdout-directory /workspace/vkgr-p3-20261006t1855/code-v3/outputs/p3-holdout-v3-20261006t1919 --raw-root /workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183 --config configs/p4.json --output outputs/p4-baselines-v2-20261006t2023
../code-v1/.venv/bin/python -B -m vkgr.baseline_run verify --manifest outputs/p4-baselines-v2-20261006t2023/run_manifest.json
```

Verify验证当前P4 source/lock/env、config strict与hash、所有input/artifact hashes；重新根据immutable query全集固定抽样，并从原train/raw feature重算baseline predictions，再从stored predictions/queries重算各slice metrics。重新hash伪造prediction也不能只凭checksum混过。哈希是一致性证据，不认证真实性；不保证并发文件更改或I/O transaction，失败目录保留且不能复用。

Latency边界：content_seconds包括history centroid构建与blocked scoring/topk；popularity_seconds仅复制共享topk给selected queries，不包含一次catalog ranking、train count或raw preprocessing，因此不可直接把两项当公平online latency比较。run elapsed与process peak RSS包括input校验/load/count与两个split评估；峰值RAM不是全系统余量或独立quota证明。

## 验收状态

真实v2 run与完整verify通过，Astra已独立核查hash/query抽样/Popularity/metric；Astra已正式关闭P4 gate；独立Sol最终spec/quality和文档/public manifest review通过，无新增阻塞。当前P4 Complete，下一P5已在过夜授权内，由Astra另行dispatch。No trained model、全test benchmark、方法胜负、线上A/B或causal claim。

### v2 run完成证据

真实run exit0、stderr空；manifest实际started_at_utc为2026-10-06T20:11:52.141315+00:00。Run ID中的t2023为唯一目录label，不作为真实开始时间。Source保持51db4e...；质量/resource数字唯一摘要见experiment_log。

Server-only final artifacts：

| Artifact | SHA256 |
|---|---|
| run_manifest.json | 65d90cc9fb4df88824786a794349e3e2a6724ff0a8e08e4f95b573c699bdb801 |
| metrics.json | b0e43673c7b0a68aba03ce1807c13065a6ffa5a8d8c030b49a82c78f5b795b1c |
| validation-audit.json | 726965d6eae9049e27f8636fe39c0aa96dadb493311dd954f039367584bb2e5b |
| test-audit.json | 0af77fa043c87287d941e27e8f7752cbf85205151a45563eaac7ad273eccf3f8 |

Public sanitized copy `artifacts/p4-baselines-manifest.json` SHA256 `84cdef9a5062c489155c92456b544ffe28901ecc6aeb901ed6d8459228a73b2f`：删除config/input roots等私有路径，input仅role+hash，selected IDs仅set hash，无queries/完整catalog/predictions/embedding。原manifest保留服务器，不冒称public副本与server bytes相同。

Audit user ID canonical list hashes：validation `084290bdc4184487dc1190505d9fbb4ff3d5e938b4189f73912b3109bd3714a2`；test `412bd72b60223a7a1456178917335c54dbcc9265affefbf624f8b5a75e67874c`。两个baseline消费同一selected query，最终verify已重新抽样及重算预测/质量通过。

### Astra独立真实artifact核查

Astra只读核查UTC2026-10-06T20:20:13.175653、exit0，未import P4 baseline生产模块：使用PyArrow独立重数所有train like并排序Popularity top20，逐stored prediction一致；hashlib独立核对server manifest及全部artifact SHA；重算SHA256(seed:split:user_id)排名，stored selected queries逐字相同；stdlib math独立重算两method/全部slice Recall/NDCG@10/@20，与metrics完整一致。指标数字仅experiment_log。

该独立核查没有再次读全NPZ或重算真实content scores；content真实性由作者随后通过的完整verify与已通过的source/scoped review补齐。不能把metric公式独立核对冒充另一次content oracle。

文档补记失败如实保留：一次临时STATE Python脚本报Non-UTF-8 SyntaxError（此前cat独立核验节已写入）；同一shell末尾manifest检查成功使shell exit0，不能掩盖该脚本未执行。明确项目.venv/bin/python+UTF-8 coding重试输出STATE updated；只补正文档，不重跑实验或修改source。Root encoding原因未独立定位，I-034限定补记关闭。

### v2完整manifest reader replay通过

唯一`cache/verify-v2.sh`使用上述verify命令，专用tmux p4-verify-v2。`cache/verify-v2.exit`=0；stdout `P4 manifest/input/query/prediction/metric replay verified`，stderr为空。它重新核source/config/所有输入hash，从全官方NPZ按ID提取特征及train-only like计数，重抽固定query、重算两个baseline predictions，再重算所有stored-prediction metrics；没有只把exit0当成功。v2 final manifest原hash保持65d90...801；未新建实验或改原件。

本机public copy已独立JSON parse/schema task/source/input-role检查及private-root/connection/secret marker定向扫描通过，hash保持84cdef...b2f；`git diff --check`通过，AGENTS/P3源码/config/lock/public freeze逐字节与P3原worktree一致。所有新增大产物与完整ID/预测仍服务器，不进入Mac/Git。该时点交接状态Implementation ready / awaiting coordinator review；其后正式关闭见文末。无commit/push/P5实现。

### Astra正式阶段关闭

Astra正式关闭P4，独立Sol最终spec/quality及文档/public manifest review通过、无新增阻塞；fresh tests/服务器FAISS、独立query抽样/Popularity/指标核验与作者完整content replay构成批准MVP范围验收证据。P4 Complete；下一P5已在过夜授权内，由Astra另行dispatch。本次仅更新STATE/handoff/validation/I-009阶段状态，source与manifest原hash不改；representativeness、ordering、OOV与旗舰风险继续保留。
