# SID 全量规模基准：独立 code review

结论：**CHANGES_REQUESTED**，1项 MEDIUM，未发现 HIGH。只审实现与工程边界，不重开科学设计评审；普通修复由 owner fresh RED→GREEN 后继续，无须重复 review。

范围：`<WORKTREES>/sid-coll-s1/vk-generative-retrieval` 的 `scripts/sid_scale_benchmark.py`、`tests/test_sid_scale_benchmark.py`、`docs/superpowers/plans/2026-10-17-sid-scale-benchmark.md`，并核原 `src/vkgr/sid.py`、`src/vkgr/sid_rqvae.py` 被调用接口。合同为 SID-FIX 卡§6与decision_log 2026-10-17 本地安排：合成64维、两N规模、1CPU、有界工程成本、完整/跳过/截断分开、无真实data/CP/test、fresh输出。

## R1 [MEDIUM] controller CLI忽略已解析参数，可把短冒烟请求执行为全量默认批次

位置：`scripts/sid_scale_benchmark.py:228`–`243`，尤其243行。

`main()`解析并接受`--n`、`--k`、`--block`、`--seed`与`--wall-seconds`，但只有隐藏`--worker`分支使用它们。正常controller分支固定执行`[run_method(method,n) for n in SIZES for method in METHODS]`，未传任何参数；因此用户要求更短wall或tiny synthetic冒烟时，实际仍执行两种全量规模、默认k/seed/block、每个子进程60秒。虽然默认仍不超过合同60秒上界，但显式更短限制被悄悄忽略，可能启动本不打算运行的完整流式工作量。

已做不启动子进程的真实CLI控制流复现：用bundled Python加载脚本，patch `sys.argv`为`benchmark --n 29 --k 3 --seed 17 --block 7 --wall-seconds 0.01`，patch `run_method`为只记录调用的stub。实际六调用为：

```text
('bhkm',1250000), ('rqvae',1250000), ('streaming',1250000),
('bhkm',12470000), ('rqvae',12470000), ('streaming',12470000)
所有kwargs均为空。
```

修复方向：明确controller/worker参数合同。若controller固定两个N，仅允许其有效参数并拒绝worker专属选项；若公开参数本就允许选择规模，则显式传入相应值。无论采用哪种，公开接受的wall限制必须兑现，或在执行前明确拒绝。补一项synthetic CLI测试：传入短wall后断言所有子调用收到该cap，或在任何子调用前被parser拒绝；不得静默回到60秒。现有12 tests只测函数调用，未覆盖此CLI控制流。

## 其余检查与限制

- 默认合成两规模为1,250,000与12,470,000，64维、float32单位向量；没有真实数据、CP、test输入接口。
- 原BHKM直接调用`sid.fit`，原RQ-VAE使用`PREREG.max_steps`与CPU；预估下界不是实测峰值。BHKM根边数组/输入拷贝超cap直接skip，RQ-VAE物化高规模亦skip，未伪造suffix。
- `streaming_fit`是单pass根节点原型，输出balanced=false、suffix=UNKNOWN，未声称完整SID/B1。占用统计与完整拟合suffix分开。
- 子进程独立session，BLAS环境1线程、NumPy OpenBLAS实际set/get核验、Torch intra/inter-op=1。父采样RSS并在wall/RSS限额触发时杀本进程组、communicate回收；完成后再核ru_maxrss与总wall，不把超cap结果标COMPLETE。ps采样有间隔，不能保证瞬时绝不越限，计划已公开此限制。
- 完整拟合才算suffix；RQ-VAE fail_wall/fail_wall_projected归censored，其它失败归FAILED；未完成不外推完整成本。fresh目录与结果文件均排他创建，无删除/覆盖旧结果。
- 本次没有跑全量、没有新建服务器作业，也没有修改实现或S1驱动。运行的唯一复现是mock CLI dispatch，不涉及真实拟合。owner所报12 tests与83回归不作为本review重新实跑结果。
- 隐藏`--worker`是内部子进程入口，资源监控由controller包裹；正式操作必须走controller，不能把worker分支单独当受监控launcher。此处作为调用约束说明，不额外提出设计门槛。

读取/核对UTC：`2026-10-16T18:36:25Z`；worktree HEAD：`0b2becc1914e8aeb7f0afe851ee714173ef2f1c9`。实际文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `scripts/sid_scale_benchmark.py` | `237bec491cda08aaccc9ff8a363dec39aabd2a184ecaa6ce764031d0e07474c3` |
| `tests/test_sid_scale_benchmark.py` | `43b84a0de0ad3adbb5848be87d837cc11921041404e91c01ad42730405eed976` |
| `docs/superpowers/plans/2026-10-17-sid-scale-benchmark.md` | `d2c85a2186c6f286b8df3dba6ced44ddb689d9e606ba0af5389a8a008710b233` |

报告只写本文件；未改STATE/handoff/experiment/decision事实源，未commit。报告针对上述实际读取字节，后续修复无需重复review。


## 作者修复验收（不是独立复审）

作者关闭记录：唯一MEDIUM已fresh RED复现后修复，controller只接受output-dir与wall、worker-only参数显式拒绝、wall传全部六次；15项规模tests和83项协同SID回归通过。ddf208e本机正式有界运行，未修改服务器。没有第二次独立review。
