密级: 公开

# P3 train-only补充EDA执行与验收

## Scope与独立工作区

Eddy批准P7后补充P3原始画像/观看时长与label候选比较，并明确裁决“按照最新的结果来修改冲突”。本次定点修正data_card首段及experiment_log当前/P7段旧awaiting；保留历史ledger、冻结p3-like-v1及原实验数字。Task ID仍P3，不进入P8。

Native worktree `<WORKTREES>/p3-label-eda/vk-generative-retrieval`，branch `codex/p3-label-eda`，HEAD `0376d9a8119f3500cbf78519675cc7bf18013c1b`。P7 Git-visible overlay全部逐字节带入，记录`cache/p3-label-eda-entry-overlay.json` SHA256 `8c902396b256789d8c60ea629d3296455acd991307b154c601a6072405de8918`；原P7/main文件未改。

新增范围：`scripts/p3_label_eda.py`、`tests/test_p3_label_eda.py`、`docs/p3_label_eda.md`、本文件、`artifacts/p3-label-eda-manifest.json`。更新范围：STATE/handoff与experiment_log；data_card仅批准的状态句。其余源码/已有tests/configs、AGENTS/lock/pyproject/data_contract不改；无commit/push/新依赖/GPU/模型/训练/下载/holdout。

## 输入、门与来源

先完整读取server guide与workspace AGENTS。固定raw-root `/workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183`，revision `084a18386e149a454dd6dbe0123478e75c6b09ce`，subset `ur0.01_ir0.01`。只读train及既有metadata。Existing P3 final train-audit SHA256 `3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe`；hash绑定catalog/users，再核train文件及item duration histogram。Item/user metadata schema、train join/null、feedback/长度/watch histogram逐项核验，不输出真实ID或raw rows。

Seconds与ratio探索grid、duration bins、dislike冲突保留在源码和manifest，真实scan前已声明。Ratio分母零单列、ratio>1保留，不做clipping；固定README只声明秒编码与范围，未证明上限编码的饱和机制。Python复用既有P4独立env，NumPy/PyArrow版本沿用既有锁，无安装。CPU单线程Arrow batch、BLAS/OMP限线程；每batch检查更严格的host/cgroup可见RAM余量门。输入hash/schema/null/join/基数不一致或RAM门失败即停止。

## RED/GREEN与问题ledger

Mac借用既有P6环境运行synthetic tests，仅两个纯计算行为起步；未给新worktree安装环境。首次stub返回空dict导致KeyError，exit1/errors，属于无效RED，不记成功TDD证据。改成明确assertIn后的fresh RED exit1/two assertion failures；最小candidate_masks/hist_summary实现后GREEN exit0。之后按review增加数值摘要单位及segment/dislike保留的literal synthetic回归，fresh targeted suite exit0。新增回归首次即GREEN，如实记录，不伪造其单独RED。

初次默认SSH被sandbox网络拒绝，exit255；沿既有授权通过正常escalation连接。后续summary-only repair命令有shell heredoc引号错误，exit127/SyntaxError；未产生有效新manifest，改用stdin传入精确Python脚本，exit0。Astra首次读取aggregate的独立脚本也有括号SyntaxError/exit1，修正后exit0。以上错误均不计分析或验收通过证据。

Independent review发现watch/duration histogram错误套用user/history字段。已修正：numeric_summary只含sample_count/sample_unit/value_unit/quantiles/range；last50重命名snapshot_events_beyond_last50。Like每用户count−1的训练prefix贡献另列；其top份额排名人口是全部train用户，不是只取活跃用户。Snapshot last50不等于实际prefix训练被丢弃例数。此修正不改变候选数字或policy。

## 实际命令与artifact lineage

Server新目录 `/workspace/vkgr-p3-label-eda-20261007t0056`，使用exclusive输出，旧源码/manifest保留。V1真实scan exit0；V2 summary-only repair exit0、无raw重读，旧manifest SHA256 `b945665731d8562ebcbbd3102526b10d5fe4b4608e903d61773179c257fd823c`，带producer_scan_script_sha256与summary_repair字段。它仅为历史，Mac副本在ignored cache。

最终采用修正源码fresh同范围scan，资源数据与源码直接对应，不拿V1 elapsed冒充修正后fresh run：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/vkgr-p4-20261006t2011/code-v1/.venv/bin/python -B /workspace/vkgr-p3-label-eda-20261007t0056/p3_label_eda_v2.py --raw-root /workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183 --train-audit /workspace/vkgr-p3-20261006t1855/code-v3/outputs/p3-train-v3-20261006t1920/train-audit.json --output /workspace/vkgr-p3-label-eda-20261007t0056/train-eda-final-manifest.json
```

Exit0，reconciliation每项true；真实数字、wall-clock/RSS/RAM余量只见[experiment_log补充EDA](experiment_log.md)。Final source SHA256 `7d7335a55c70f76b039c1abb0345b347ed5733b98be73b389b5ca9692f1fb669`；final server/public aggregate SHA256 `3ec3a384e3fa3fb8f0444a8ecb792225b2c9b60d51457e0773e7d4c9c8282239`。Manifest中script_sha256匹配当前源码，input_files仅train及metadata。原始records/ID集合、catalog、queries不回Mac；仅小型aggregate公开manifest回传。

```bash
<LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m unittest discover -s tests -p test_p3_label_eda.py -v
git diff --check
```

## Fresh-context证据与stop

Astra独立Python Counter/set及float除法oracle，没有import本脚本，重新只读train与item duration；所有候选event/user/item/支持用户基数匹配，ratio>1匹配。Astra另用既有audit重算nearest-rank分位和反馈计数匹配。独立reviewer穷举uint8组合，以int64有理数oracle核ratio masks通过；最终源码/tests/manifest无blocker，全部segment事件/反馈/候选独立求和匹配。Astrafresh targeted tests exit0、final source/manifest hash匹配；保护文件与P7逐byte相同，input名单没有val/test。

作者自检不冒充fresh-context review；独立reviewer最终docs PASS，Astra fresh复核报告/ledger、Git-visible文件大小与禁止binary、diff检查均通过，无blocker，已正式验收本次P3补充EDA。可见host/cgroup RAM仅为当前可观测证据，不是Vast承诺配额。停在commit前，不自动选新policy、不开始新task。既有validation/test已被使用，不称未触碰新final test；smoke限制和现有模型结论保持。
