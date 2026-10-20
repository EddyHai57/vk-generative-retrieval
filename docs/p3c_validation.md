密级: 公开

# P3-C 执行与验证 ledger

## 授权范围与工作区

Eddy明确批准完整传统召回计划并启动goal；当前仅P3-C。主线like label/like history，后续一次watch history独立消融；保留低活跃与长尾，不做5-core。Astra按批准brief dispatch，本worker不重复问授权、不自派reviewer、不自动下一task。

Native worktree `<WORKTREES>/p3c-data-quality/vk-generative-retrieval`，branch `codex/p3c-data-quality`，HEAD `0376d9a8119f3500cbf78519675cc7bf18013c1b`。从已验收 `codex/p3-label-eda`完整带入Git-visible overlay，78文件逐hash一致；ignored `cache/p3c-entry-overlay.json` SHA256 `7efb1a87296d509d15648f565c50449d99fc22ec18d2b1d4f34b56bd43df2566`。没有仅clone GitHub恢复P1而漏掉P2～P7/EDA。

新增允许：scripts/p3c_data_quality.py、tests/test_p3c_data_quality.py、docs/data_quality.md、本文件与aggregate manifest。更新允许：tasks仅追加批准新task路线，STATE/handoff、experiment_log新P3-C摘要及issue_log按需。AGENTS/data_contract/旧源码/tests/config/lock/旧产物保护；主main与旧worktrees不改。没有commit/push/remote或Git配置修改、训练、label变更、下载、holdout values读取或实例生命周期操作。

输入只读固定revision/subset的train00–24、已有metadata/embedding。requirednull/非法ID/schema/hash/join/order/既有基数不一致和20% RAM/disk余量为STOP；不填删。输出只aggregate与公开原创小文件，不含真实ID/rows/完整mapping。

## 输入与复用绑定

完整读取 `/etc/vast-agents-guide.md`及`/workspace/AGENTS.md`；第一次合并输出截断，中后段及workspace全文补读，操作前已完整。Root guide不授权改变资源。既有环境 CPython3.11、NumPy1.26.4/PyArrow19.0.1，仅CPU/单线程，无新安装。

- 固定 raw root：`/workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183`。
- Train audit：`/workspace/vkgr-p3-20261006t1855/code-v3/outputs/p3-train-v3-20261006t1920/train-audit.json`；SHA256 `3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe`。
- 已验收 EDA：`/workspace/vkgr-p3-label-eda-20261007t0056/train-eda-final-manifest.json`；SHA256 `3ec3a384e3fa3fb8f0444a8ecb792225b2c9b60d51457e0773e7d4c9c8282239`。
- Catalog/users先与audit精确SHA绑定，再验证排序唯一。Raw train/metadata/content文件重核SHA/bytes，重复binding逐entry核对，不以dict覆盖矛盾记录。
- 完整27个官方interaction文件已有P2/P3完整性证据，本轮只重新读取train；holdout完整性复用历史验收，不读取其values。旧embedding完整catalog join/finite/zero-norm审计因whole-file hash相同而复用，没有再次load完整embedding。
- Exact duplicate、global pair duplicate、physical two-pass digest按pinned audit复用，口径限定见data_quality；本轮parquet字节相同并按week物理batch读取，不独立验证hidden timestamp。

## TDD 与真实失败记录

未伪造历史RED：首次4个helper stub返回空dict，4项assertIn直接失败，exit1，均为assertion failures。最小GREEN首次3项通过、user strata失败；人工synthetic prefix总量误写6，实际 max(0−1,0)+max(1−1,0)+(2−1)+(5−1)=5。修正oracle，不改生产计算迎合错误；后续fresh4项通过。

Scan stub fresh第5项assertIn失败、exit1；最小bounded accumulator后5 tests GREEN、exit0。验证items/users计数、原week order不能倒置、未知ID STOP，user flags保留不删。首次full suite忘记`PYTHONPATH=src`，46 tests的import/interface failures/errors、exit1；属于执行环境错误，不称production RED或通过。纠正环境fresh96 tests PASS/skipped1（既有FAISS reference在server独立环境，此处不冒称实跑），exit0。

真实scan前inspect既有binding键，发现P3使用size_bytes、EDA使用bytes，原适配会KeyError。此时未跑真实scan。新增verify_files stub的第6项fresh assertion RED、exit1；最小实现兼容且重复entry hash/size冲突拒绝后6 tests GREEN、exit0；scoped independent reviewer确认无blocker。没有改旧producer/manifests。

最终source fresh regression：97 tests PASS/skipped1，5.519s，exit0；完整真实输出 `cache/p3c-final-regression.log`。前述RED与错误保留原tool调用记录，本文件是准确摘要，未声称存在未捕获的完整旧本地log。Targeted命令如下：

```bash
<LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m unittest discover -s tests -p test_p3c_data_quality.py -v
PYTHONPATH=src <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m unittest discover -s tests -v
git diff --check
```

借用既有Mac env只执行当前worktree源码，无新依赖/污染旧工作区。测试临时写入当前outputs，环境sandbox授权限制以正常escalation执行，未改tests规避。

## 真实一次scan与输出

Server新exclusive目录 `/workspace/vkgr-p3c-data-quality-20261007t0601`，原始/旧产物只读；仅上传原创脚本和已验收 helper。输出已独占创建，无覆盖失败目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/vkgr-p4-20261006t2011/code-v1/.venv/bin/python -B /workspace/vkgr-p3c-data-quality-20261007t0601/p3c_data_quality.py --raw-root /workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183 --train-audit /workspace/vkgr-p3-20261006t1855/code-v3/outputs/p3-train-v3-20261006t1920/train-audit.json --eda /workspace/vkgr-p3-label-eda-20261007t0056/train-eda-final-manifest.json --output /workspace/vkgr-p3c-data-quality-20261007t0601/p3c-data-quality-manifest.json
```

真实 `scan.exit=0`；stdout为 `P3-C aggregate audit complete; all reconciliation gates passed`；stderr空。Server真实stdout/stderr/exit保留，Mac ignored cache只回传同名小log。无额外label grid扫描或holdout读值。

- Final script SHA256：`497f24389ee9aec948ae0d1d406664d5fdbaf7d58d957820ccb00c17c236d9f3`。
- Tests SHA256：`5ed092c6d38ec1089b9fbf2a56e85c2529d6db92b6336ec97321264864cadab8`。
- Read-only helper SHA256：`7d7335a55c70f76b039c1abb0345b347ed5733b98be73b389b5ca9692f1fb669`。
- Final server/public aggregate SHA256：`66d552a1f1387edfc1d40226a93e4ead5cc99f04399dc7b1f959dcf0ba6664e0`；server与Mac逐byte相同。

Train source全部列schema/null按全量batch验，metadata全部字段仅train-selected实体验null与唯一join；per-partition footer row counts、week likes与实际累计对账。Reconciliation全True（rows/users/catalog/likes/like items/prefixes、完整user exposure/like hist、ratio/255、duration hist与每week）。Numeric摘要只写experiment_log。

每个metadata/interaction batch均resource check；另外hash/阶段边界check。Manifest resource summary是构建result前的checks/minima，随后还有一次成功的pre-publication RAM/disk check，不包含在该summary计数/minima。若这最后check失败不会发布success；不是全部瞬时RAM/hidden quota保证。Elapsed也截止result构造，不包含末check和JSON写出。Process ru_maxrss按Linux KiB；visible host/cgroup较严格RAM证据不等于平台承诺配额。

## Fresh-context 与完成 gate

独立reviewer初source review及scoped verify_files review无blocker；fresh targeted PASS，另用独立stdlib排序/求和构造expected值，对照被测函数的随机count vectors、精确大整数author accumulation及batch内duplicate核验通过；review harness会import本模块/旧helper，但expected值未复用生产计算。Astra fresh targeted PASS。作者统计/文档自检不替代fresh-context review。

Astra已用独立stdlib Counter+metadata join重读train与item metadata，没有import本producer；全部item/author曝光、likes和每author itemcounts的entities/total/zero/min/max/P50/P90/P95/P99/top1%/top10%与全部bins逐字段一致，exit0；另独立核server scan.exit/stdout/stderr与aggregate SHA匹配，fresh6tests PASS。独立final docs/public review已PASS，Astra正式验收P3-C Complete；本worker停止，P3-D由协调另派，不自主开始。Metadata snapshot/编码、official-order hidden timestamps、irX机制、subset外推、旧test已用仍明确保留。Source和aggregate无需为非阻塞resource summary截止口径重新扫描；已披露其实际边界。


## Final作者边界检查（非独立review）

Entry overlay除5个授权治理/日志更新文件外73个受保护文件逐hash一致，AGENTS、冻结合同、所有旧源码/tests/config/lock与产物未改。New Git-visible仅5个允许新增文件，UTF-8/text无NUL、每件小于100KiB；aggregate仅匿名汇总、file hashes与source schema，无真实ID/rows/实体映射。Clean before/after相等、全部reconciliation true。git diff --check exit0，staged为空。原始full arrays与metadata仍server。

Fresh Git发现native create_worktree在指定ref时实际建立detached HEAD，而不是自动命名branch；早期文档把worktree name当branch的口径错误现已纠正。按明确批准在同HEAD执行git switch -c codex/p3c-data-quality，exit0，不移动base、不commit或改Git配置；当前branch需fresh命令核。旧worktrees/main文件未写。


## 正式完成裁决

Reviewer最后重新核两处措辞与STATE更新，final PASS；Astra正式验收P3-C Complete。Astra另fresh检查受保护公开文件byte equality/source binding和新public文件边界PASS。指标数字/源代码/manifest hashes不变，只有状态/anchor和完成gate同步；最后git diff --check和新报告本地链接通过。本worker到此停止，保留独立worktree与server证据，P3-D另派新worker；未commit/push、改合同或提前实现未来接口。
