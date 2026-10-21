# P query 块运输容量修正：实施证据

日期：2026-10-21。实施者：Codex subagent（继承主 agent 模型，未切换）。依据 decision_log `515c58d`、`ad1c428` 第 9 条、`e369424`；本项为 C 档运输与源码绑定改动，待主 owner 另派独立复核。没有 push、部署、SSH、提交作业、信号、读 test 或改原 cost run。旧成功和失败记录均保留；本项不是失败工程重验。

## 问题及最小改动

原 adapter 固定 `max_in_flight_bytes=128 MiB`。原 `throughput_exact` 在 N=12,000,000、k=50、uint32 IDs 时给每块的 result hint 为 **24,008,960 bytes**；原 `ordered_map` 三倍预约为 **72,026,895 bytes**（ordinal 0）到 **72,026,925 bytes**（最后 query block 的起始 ordinal 26,816）。两份预约超过 134,217,728 bytes，所以实际 pool 即使有 2/4/8 workers 也只能保持 1 个尚未按 ordinal 交付的 future。

只把新增 P adapter 的容量改成 `128 MiB × workers`，prefetch 仍固定 1。保留原 result hint、三倍预约、内存/RSS/磁盘守卫、资源测量公式及原 32/4096 float64 kernel。每个 worker 得到原单任务保守容量；支持的 1/2/4/8 workers 对应 128/256/512/1024 MiB，总运输预约上界 **1,073,741,824 bytes**。原 measured declaration 公式中额外 2 GiB 保守余量大于此次最多新增的 896 MiB 运输容量；没有减少任何声明或绕过实时准入。它不是实测 RSS，也不保证资源一定可准入；原真实资源守卫仍有最终拒绝权。

## 机械验证

`ObservedPool` 继承真实 ProcessPoolExecutor，只观察实际 submit 和真实 Future.result 的交付；没有用假的 pool 替代 ordered_map。对 8 个 tiny payload 使用原全量 result hint，旧容量在 workers 2/4/8 时 peak pending 均为 1；新 adapter 经 execute 传出的容量对应 peak pending 分别为 **2/4/8**。全部结果依原 ordinal 0–7 交付。这里的 pending 指 submitted、尚未经 ordered delivery 消费的 futures，不声称等于同一时刻使用 CPU 的进程数。

另用 **256 queries × 4,103 items × 64 dimensions** 的 synthetic mmap，8 个 query blocks、每块两次 item score 调用：workers 2/4/8 均逐块核对原 reference 的 float64 raw-score shape/hash 与 prediction bytes，16 个 score blocks 全等；pending 2/4/8、ordinal 0–7 同时通过。本机实际 8 核，只有这项合成 8-worker kernel 测试以 fixture 模拟 9 核容量，便于验证运输；生产 CPU 守卫未修改，该本地证据不替代服务器准入。

- fresh RED：旧 execute 容量在 workers 2/4/8 分别出现 `1 != 2/4/8`；新增 delta 请求被旧 request contract 拒绝。日志在 `cache/p-query-block-infra/capacity-verification/`。
- GREEN 与相关回归：`PYTHONPATH=src <PROJECT_PYTHON> -B -m unittest discover -s tests -p 'test_p_query*.py'`，**27 tests OK**。固定 OMP/OpenBLAS/MKL threads=1，使用主项目 venv。
- 全 tests：同环境 `-m unittest discover -s tests`，**268 tests，1 failure、3 errors、14 skipped**，不声称全库 GREEN。三项 error 是未改动的 `test_anchor_profile_steps` 中 `test_ambient_vkgr_and_cuda_overrides_are_rejected`、`test_main_dispatches_valid_preflight_and_profile_cli_arguments`、`test_nonfinite_original_step_failure_is_nonzero_and_identifiable` 缺 `ANCHOR_PROFILE_SCHEDULER_CODE`；failure 为同文件 `test_original_g_and_p_train_steps_remain_bitwise_equal_under_hooks` 的子进程找不到 `tests.test_generative_g6_unit`。本项未改这些文件或修复越界问题。

真实 launcher 链使用原 OPS 固定源码与真实 launcher.prepare，经正式模块 argv 执行 prepare→dry；旧 source 真实 metadata 构造的 synthetic producer、固定 P3C 数据通道 overlay、新 source/delta/input pins、原 measured resource 公式全部核对通过，且没有 submit。入口子进程非法命令、缺 execute flag、合法 dry 的原测试保留。证据 `cache/p-query-block-infra/sprint-chain-1792595439255833000/` 与相关回归产生的独立 chain 目录。它只是结构 fixture，**没有实际大数组 preflight、服务器测量或科学结果**。

## 旧 producer 与新 source 的封闭增量

仅接受旧 producer source **43eb5bec6d32a5d37746a033452824fd399f2dc4**。从旧 run.json 的实际 cwd 完整核 `source(旧cwd, 固定SHA, overlay=True)`，再完整核当前树与当前 request 固定 runner SHA；两树都要求固定 D0 overlay，实际 P3C bytes 必须相等。旧 stdout 的 runner ledger 必须逐字段等于重新计算的旧 ledger；原 terminal/stdout/resources/config/argv/input pins 与当前 interpreter 路径比较仍保留。新旧 science/config/lock 等既有文件不允许变化。

请求增加 `source_delta`（源码树外绝对路径）与 `source_delta_sha256`；该 JSON 的 SHA 进入 request，JSON 本身进入 launcher inputs。schema 固定为 `p-query-source-delta-v1`，字段精确为 `schema/old_source/new_source/files`。files 必须精确等于新旧 Git 树实际 diff；每个条目精确为 `old_sha256/new_sha256`，新旧实际文件 bytes 必须对应，新增文件的 old_sha256 为 null。仅 adapter 允许改变旧文件字节，新增路径封闭为本项 test、fixture、实施报告，没有开放式豁免。

允许路径：

1. `analysis/p_query_sprint_ops.py`（旧 SHA256 `729a039d2cd9f3e91547c19147c8e070ab75726f4c9d2981dd0283aaeacfc9e3`；新 SHA256 `057d76aea4fe15ac530e6e9ceefebd1f0f460743857f4a47f46a4551a0ad66de`）
2. `tests/test_p_query_sprint_capacity.py`（新增；SHA256 `39ac004d67a1fdfca886fe2082aba54fee24b40509012c5b2b8840c91a1e78c5`）
3. `tests/fixtures/p_query_sprint_capacity_chain.py`（新增；SHA256 `b5b9f50c2c54d54e603e7f0835e4379e3c80d10615fd23717dfcd627faa4e200`）
4. 本报告（新增；其最终 SHA 在树外 delta 中钉，避免自 hash 循环）。

**发布必须从旧 43eb5be 压平，只加入上述实际需要的封闭路径**。本地开发 worktree 的 ae474743 包含大量未进入旧公开 source 的文件；不能把整个 worktree 压平公开后期望该 delta 验证通过。未来独立 review 报告不在本项运行 source 的封闭列表内，可单独文档提交；若要加入同一运行 source，必须先显式登记新路径并重新复核。当前运行没有重写旧测量 producer、不重跑 q32 来制造新同 source producer、不用新 run_id 绕过失败。

旧记录没有独立 Python version/dependency inventory；继续严格匹配原 argv 中 interpreter 路径，源码中的依赖/lock 字节保持一致，不能据此追认旧 wheel/runtime version 的历史值。

## 隔离变异

每个变异使用独立 Git 拷贝、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一 PYTHONPYCACHEPREFIX；不是共享源码/缓存，且没有把 OMP sandbox 失败算成 killed。

最终七项全被对应行为断言检出：旧 128 MiB cap、跳过 delta SHA、跳过旧 ledger、跳过 accepted cwd、跳过新 ledger、跳过 producer ledger、跳过 delta bytes。证据 `cache/p-query-block-infra/capacity-mutations-1792595662910321000/summary.json` 及逐项日志。

首轮只对 cross-source 用例删除 accepted cwd 条件时存活，原记录保留在 `capacity-mutations-1792595522066114000/`；该分支的 oldroot 本就来自 record.cwd，且随后已按旧固定 SHA 验证其完整 source，因而该比较在此分支冗余。它在普通同 source producer 分支仍有作用；最终改用原同 source producer 篡改测试，检出这个变异，不把首次存活隐去或误称为全局等价。

## 限制与交付

本项没有原单核 cost 成功终态，也没有完成正式全量 cost 对拍；不能据本地 parity 宣称 cost 已验收或给提速倍数。原 a40-exact-run-v1 继续由主 owner 按原身份、预算和截止核查。真实较长 query prefix 的新进程树 RSS 与耗时尚待固定 source 部署后的新工程测量。

撤回只需不部署新 source，既有原 run、旧部署与旧输出不变。独立 commit 的最终 SHA 由主 owner 收据记录；本报告不自引用尚未生成的 commit。

## 主 owner 构造正式 delta/request 的机械步骤

1. 从旧公开 `43eb5bec6d32a5d37746a033452824fd399f2dc4` 建最小新公开 source，只带上文四个路径。独立 C 审报告另作文档提交，不进入运行 source。完成审核、卫生扫描与固定公开 SHA 后，再在新部署目录逐文件核验；旧 producer 的原 cwd 保留，D0 overlay 必须原字节相同。
2. 在新源码仓库运行 `git diff --name-only <OLD_SHA> <NEW_SHA>`，要求实际集合是上文四路径的子集且含 adapter。为集合中每个路径计算实际旧 cwd 的 SHA256（旧路径不存在则为 null）与新 cwd 的 SHA256，构造精确 `files[path]={"old_sha256": ..., "new_sha256": ...}`；完整 JSON 为 `{"schema":"p-query-source-delta-v1","old_source":"43eb5bec6d32a5d37746a033452824fd399f2dc4","new_source":"<NEW_SHA>","files":{...}}`。不含绝对目录映射、payload 或科学数据。
3. 以独占新文件方式把 JSON 写到两棵源码树之外的 `<REQUEST_DIR>/source-delta.json`；计算该文件原字节 SHA256。保留原 measurement request、run.json、terminal.json、stdout、result.json，不重写任何旧记录。
4. 在新请求里写入固定 `runner_source=<NEW_SHA>`、`measurement_request=<原成功q32请求绝对路径>`、`source_delta=<上述JSON绝对路径>`、`source_delta_sha256=<JSON原字节SHA256>`。workers 必须和旧 measurement producer 相同，原 engine/source/data/lock/ops 输入绑定保持不变；目标 query prefix 仍限已授权 64/256，或者等待原 cost 成功后才可进入原 full cost-parity 路径。目标 output 和作业身份须走原碰撞检查，不能重投失败 run 或改原 cost run。
5. 依原固定入口 prepare，再同入口 dry；两步都重核完整旧/新源码与 producer 绑定，JSON 自动列入 launcher inputs。只有独立 C 审及主 owner 的部署准入均通过后才考虑后续工程提交；本项交付不自行执行这些服务器动作。

## 更正：旧固定 synthetic producer 没有 P3C（本轮当前口径）

前文“旧/新两树都必须有 D0 overlay”的要求及将旧 fixture 预先补 overlay 的做法不适用于实际旧部署。主 owner 的真实只读 preflight 发现：旧 `43eb5bec6d32a5d37746a033452824fd399f2dc4` Git 树不含 `docs/p3c_validation.md`，实际旧 cwd 也没有该文件，原成功 producer 的完整 source ledger 因而登记 `p3c_data_channel=false`。旧生产源码/receipt 无需且不得被补写来适配新 adapter。前版 fixture 覆盖了“旧已存在固定 overlay”，遗漏了这个实际历史状态。

前版公开 `23fadb` 与失败 preflight 记录保留；没有 submit、没有科学结果。本轮只作运输兼容修复，保守计为**同入口工程重验第 2 次**（第 1 次为原 q32 cap 运载修复），不重置次数、原预算、spent 或 deadline。新代码完成后仍须新独立 C 审；本分支不 SSH、不改旧部署、不提交服务器作业。四路径公开 closure 保持原样。

最小兼容分支仍先按旧实际 cwd 核完整固定 `43eb5be` ledger，并严格核它等于原成功 producer ledger；只有 `old ledger false`、`old entity 不存在（含不能是 dangling symlink）`、`old 固定 Git 树也无此路径` 三者同时成立，才识别为 legacy absent。新树必须是已登记 overlay，实际字节为固定 `c89b6466…64bb3274b5f`。若旧有 P3C，仍要求旧 ledger 登记该 overlay 且实际字节也是固定 SHA；其余缺失、漂移或历史 receipt 状态不匹配均拒绝。原 terminal/stdout/input/config/argv/resources/full-ledger/delta 绑定全部保留；没有对 source 的开放式忽略。

因此，前文构造步骤第 1 条改为：**旧 cwd 原样保留，不给历史 producer retro-fit P3C；新目录独占补固定 D0 overlay**。其余构造步骤不变：从旧 `43eb5be` 压平四路径、新旧完整 source 与封闭 Git delta、树外 JSON+SHA、原成功 measurement request 及真实 launcher prepare/dry。旧 ledger 仍保持 false，不能伪造为 true。

### 为什么这不改变 synthetic cost 输入

实际读取边界来自旧固定源码，而不是根据文件名推断：

- `analysis/a40_cost.py` 的固定 SHA256 仍为 `a18d2523df8ae5e002d161b416e0a229c994884891b9e65e593a63df93a0a15f`，该文件不含 P3C 路径或读取逻辑。`inputs()` 固定 `data_kind=synthetic`，仅允许 manifest 中 `items.npy/queries.npy/ids.npy` 三个数组；另外核 runtime lock、选定 engineering YAML、冻结 baseline 与原 generate/preflight receipts。
- `analysis/p_query_sprint_ops.original()` 钉上述 cost script SHA、原 source ledger、原 runtime/guard 依赖、synthetic manifest、lock、generate/preflight stdout；诊断只复制原 synthetic queries 前缀。`analysis/p_query_blocks.parity()` 的 oracle 是固定 `throughput_frozen_baselines.py`，按原 32/4096 float64 计算 synthetic arrays。
- P3C 在本 adapter 中参与 source/metadata overlay 身份登记，不作为 synthetic score/label/history 输入。真实 launcher fixture 已在**旧目录没有 P3C**时用原旧 adapter 成功 prepare，随后新 adapter 对 legacy absent producer 走新 overlay 的 prepare/dry。

此兼容仅覆盖上述 synthetic cost 工程链，**不认作原真实 anchor/阶段 1a 的复现，不恢复缺失的真实数据合同或 checkpoint 闭包**。科学 kernel、seed、32/4096、float64、原测量 receipt、资源公式均未改。

### 本轮验证边界

fresh RED 将旧 fixture 改成真实 absent 状态后，前版在 `measurement P3C data channel differs` 处失败；最小 GREEN 已通过。负例覆盖旧未登记的正确/错误 P3C、dangling symlink、新 P3C 缺失/错误、原 producer ledger 的 presence 位或 git dirty 被改写，原全 ledger/science bytes/delta/input/terminal 负例保留；旧已存在固定 overlay 且历史 receipt 一致的正例也保留。最新日志与计数追加于下，旧证据和原文不覆盖。

本轮最终验证：P 定向回归 **27 tests OK**；其中真实 launcher 链明确断言旧 prepare 的 source ledger false、Git clean，且新 prepare 的 source ledger true、pinned_files 为原 D0 SHA，再经同模块 dry 通过。全 tests 再跑 **268 tests，1 failure、3 errors、14 skipped**，仍为前述未改动 anchor profile 的三个缺 scheduler fixture 环境变量及一个测试 import 失败，没有新增全库 GREEN 声明。原 full-suite 证据保留，新日志在 `cache/p-query-block-infra/legacy-p3c-verification/`。

本轮五项隔离变异全被检出：恢复“旧必须已有 overlay”、跳过新 overlay 要求、跳过旧完整 ledger、跳过新完整 ledger、跳过 producer ledger 比较。每项都在独立 Git 拷贝、唯一缓存、`python -B`、`PYTHONDONTWRITEBYTECODE=1` 下运行；逐项日志显示对应 legacy 正例或新文件缺失/源码字节/producer 篡改断言失败，不是 OMP 或 sandbox 异常。证据 `cache/p-query-block-infra/legacy-p3c-mutations-1792597318821398000/summary.json`，本轮 launcher 链结构收据见 `sprint-chain-1792597296402900000/`、`sprint-chain-1792597366615288000/`（同一 cache 父目录）。

当前修复字节（替代前文 e8b2808 版本的新字节表；旧字节和四路径集合不变）：adapter SHA256 `a0251a8ef6fc2609c208274ee8a1f31e4722b50e15913fb19f41368728e85d2d`，test SHA256 `f645dfa67c2f8ff510591eb34ad957858e4128beaf3c28b5b64df15661266c7d`，fixture SHA256 `eed48207f70c57f0b51cf4e835a02a555ac309dfef44ed4fa4c3737e227df290`。本报告最终字节 SHA 仍在树外固定 delta 钉住，避免自 hash 循环。生产 diff 仅 `measurement_source` 的 P3C 兼容判断；science/kernel/32/4096/float64/seed、资源声明公式与旧 producer receipt 均不变。
