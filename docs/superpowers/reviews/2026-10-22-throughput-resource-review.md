# Throughput 工具已知修复定点复核（2026-10-22）

- 复核身份：按委派请求 `gpt-6-luna/high`；当前执行 runtime 没有返回可独立核验的模型身份回执，故不把本报告标成 Luna 执行。
- 范围：只核对原 C 审查中工具 1/2/3/4/7 的指定修复，不做全范围复审。主 tree 只读；本报告是唯一新增文件。
- 边界：未连接服务器、未读取真实数据或 holdout、未运行训练、CUDA、真实 anchor 或完整测试套件；只检查本地源码与既有回执，并计算 A40 YAML/规范化工程配置身份。会话 context 计数不可用：UNKNOWN。
- 原审查依据：`docs/superpowers/reviews/2026-10-22-tool01-luna-c-review.md` 至 `tool04-luna-c-review.md`、`tool07-luna-c-review.md`、`tools01-02-fix-review.md`。

## 逐项结论

### 工具 1：序列构造预算与列类型

**原预算初检前发生 references 转换的阻断：ADDRESSED。** `SequenceStore.from_columns` 先用输入行数和引用数做预算初检（`src/vkgr/throughput_sequences.py:21-25`），再规范化列；完整数组估算也在排序和输出列构造前执行（`:33-41`）。`references.astype(uint64)` 已移动到该估算通过之后（`:35-38`）。side/context、reference、ID、position 的类型与范围检查在转换前完成，拒绝非整数、负引用、越界 ID 与不合规 likes（`:26-32`）。

**生产列式路径的 G/P step parity：ADDRESSED（限现有合成证据）。** `g-synthetic-step-a40-v1.json` 与 `p-synthetic-step-a40-v1.json` 报告 workers 1/2/8 下完整 step state 逐位一致；对应日志为 `cache/throughput-20261022/g-synthetic-step-a40-v1.log` 和 `p-synthetic-step-a40-v1.log`。旧审查指出的 fixture adapter 路径缺口由此次 production `from_columns` 合成运行收据补足。既有 `mutations-01-v2/receipt.json` 报告隔离的 future/order/mask/key 变异均被捕获。

**剩余限制：** `byte_limit` 仍是按行数/引用数与数组字节数计算的保守模型，不是实测的峰值分配上限；Python 输入容器转 ndarray、NumPy workspace 的具体峰值未在大输入测量。此结论仅关闭“引用类型转换发生在预算检查前”的已知缺口，不把它解释为大规模内存承诺。

### 工具 2：worker 输出预留、RSS 与取消

**装配路径输入/输出预留阻断：ADDRESSED（`assemble_rows` 路径）。** `ordered_map` 对每个提交任务预留 `3 × (pickle 输入字节 + 输出 bound)` 并在队列继续提交前检查 cap；worker 返回前按实际 pickle 大小核对输出（`src/vkgr/throughput_workers.py:23-29, 52-60`）。装配器使用 `SequenceStore.rows_bound` 提供输出上界（`:86-105`）；对应定向输出超限测试在 `tests/test_throughput_workers.py:19-26`。`tools01-02-fix-review.md` 记录了 1,200-event 小型 fixture 的多种 history 长度及 bound/实际 pickle 大小，支持该 fixture，不证明任意输入上的普遍紧度。

**worker RSS 只测父进程的阻断：ADDRESSED（有边界）。** `assemble_rows` 按配置内存 cap、父进程进程树 RSS、在途字节预算及 worker 数计算每 worker 子预算（`:95-104`）；worker 在任务前后检查自身 RSS（`:23-29`），交付结果时父进程用 `guard` 检查自身进程树（`:100-106`；`throughput_config.rss_bytes(include_children=True)`）。这是行/任务边界检查与预算估算，不是硬实时峰值限制。

**关闭时不取消 pending：ADDRESSED（取消未开始任务）。** `finally` 对 pending futures 调 `cancel()`，并用 `shutdown(wait=True, cancel_futures=True)`（`:65-67`）。已运行的任务仍会完成后才返回；若任务卡住，关闭仍可能等待，取消延迟没有上界。

**剩余限制：** 任意 `ordered_map` 调用者若提供不真实的 bound，超限只能在结果构造后检测；`assemble_rows` 的 bound 由当前序列结构计算，已覆盖本次指定路径。3 倍系数及 `rows_bound` 没有大规模实测校准；边界检查不能排除任务内部瞬时 RSS 峰值超过 cap。`mutations-02-v2/receipt.json` 的隔离 RNG/order/cursor 变异均被捕获。

### 工具 3：BHKM callback、返回值与磁盘预算

**原 callback 调用位置：ADDRESSED。** `fit` 在进入每个 `visit` 时和每轮迭代开始时分别调用 `on_boundary`（`src/vkgr/throughput_bhkm.py:241-260`），与冻结调用位置一致。定向测试比较原实现与新实现 callback 次数，并核对默认/外部模式都返回 catalog，而不是因开关改变返回形状（`tests/test_throughput_bhkm.py:78-87`）。

**`fit_configured` 返回契约及 trace：ADDRESSED。** 外部功能关闭时原样返回原 `fit` 的 catalog-only 结果；开启时内部可产生 trace，但只通过非空 `trace_sink` 向工程诊断调用者传递，函数仍只返回 catalog（`src/vkgr/throughput_bhkm.py:347-359`）。

**提交前磁盘预留：ADDRESSED（单个作业的整轮保守预检）。** 创建 run root 前估算全部 depth/max_iter 的 edge runs 与文件头，并用 `projected_disk=max_disk` 过 80% 与配置 cap 检查（`src/vkgr/throughput_bhkm.py:219-231`）；每个迭代也检查当前节点完整投影后才提交 block（`:261-281`）。已知的“executor 有并行写入但只为一个 block 检查空间”缺口由整节点/整轮估算覆盖。`03-final-green.log` 显示 10 项 BHKM 定向测试通过；`mutations-03-final-v2/receipt.json` 显示 baseline 及 tie/capacity/dimension/RNG 隔离变异均按预期。

**剩余限制：** 这是预检，不是跨进程原子磁盘租约；其他作业在预检后增加占盘可能与本作业竞争余量。`_guard` 的 `operation_bytes` 参数目前未参与判断（`:110-121`），因此不把动态外部占盘情况下的持续保证判为实测完成。全量 IO、trace 峰值与 run 留存需求仍 UNKNOWN。

### 工具 4：周解析路径拒绝、输出 hint 与内存边界

**PathLike 识别：PARTIALLY ADDRESSED。** `derive_weeks` 递归识别 `str` 与 `os.PathLike`，统一经 `os.fspath` 检查 `holdout`、`week_26`、`/test/` 字符串，并递归检查字典、tuple/list（`src/vkgr/throughput_weekly.py:14-20`）。定向测试用 `Path('/synthetic/holdout/file')` 覆盖 PathLike holdout，`04-final-green.log` 通过；`mutations-04-final-v1/receipt.json` 的 week_order 和 week_holdout 隔离变异均被捕获。但路径名为 `week26`（无下划线）不会命中当前 `week_26` 子串，仍可绕过路径检查，因此不能关闭原审查中 week26 路径输入的全部风险。

**零输出 hint 绕过：ADDRESSED（cap 开启时的拒绝行为）；hint 类型一致性：PARTIALLY ADDRESSED。** 并行路径在提交前要求 hint 为正的内置 int（`src/vkgr/throughput_workers.py:52-58`），且 worker 检查实际结果；周解析测试用 workers=2 覆盖 `result_size_hint=lambda _: 0` 并确认拒绝（`tests/test_throughput_weekly.py:29-31`）。单 worker 路径在正 cap 下也会因 `not bound` 拒绝零值（`throughput_workers.py:35-40`），但它没有复用 `type(bound) is int and bound >= 1` 的显式校验：无效类型/负值的错误路径与多 worker 不统一，需后续统一输入合同。该残余不再允许零值绕过启用的 cap，但会造成模式间行为差异。

**RSS 与周边界 guard：ADDRESSED（边界监测）。** 周解析开始前、每个解析块交付前后及 builder 返回后均调用共享 `guard`（`src/vkgr/throughput_weekly.py:21-34`）；若配置内存 cap 存在，会给 parser worker 分配扣除父进程进程树 RSS 与在途字节预留后的子预算（`:22-27`）。解析输出仍按周有序交给父进程中的同一 builder，跨周 state 保持串行；没有改为 worker 内逐周建 history。

### 工具 7：嵌套 cgroup、A40 planning RAM 与配置身份

**cgroup v1/v2 当前路径及祖先最小限额：ADDRESSED。** `throughput_cgroups.effective_limits` 从 `/proc/self/cgroup` 解析成员路径，遍历 root 至当前路径的祖先，分别取 v2 `memory.max`/`cpu.max` 与 v1 controller limit 的最小有效值（`src/vkgr/throughput_cgroups.py:5-36`）。resource probe 与通用 `guard` 共用此函数（`throughput_probe.py:8,24-34`；`throughput_config.py:85-95`）。`07-final-green.log` 的 3 项测试通过，其中嵌套目录及 root unlimited fixture 验证返回 12 CPU cores、64 GB 的祖先最小内存限额（`tests/test_throughput_probe.py:12-18`）。

**mutation receipt：NOT VALID AS PASS EVIDENCE。** `mutations-07-final-v1/receipt.json` 自身 `pass:false`；baseline 因隔离副本的 `analysis` 不在 import path 而失败（`mutations-07-final-v1/baseline.log`）。这不是产品行为失败，但该 mutation 组不能证明 baseline 或变异判定。本结论依赖通过的嵌套 cgroup 定向测试和源码检查，不引用该 mutation 组为通过证据。

**A40 planning RAM 收紧：ADDRESSED（配置值口径）。** `throughput_config.guard` 将已知 host memory、有效 cgroup memory 与 `planning_memory_bytes` 取最小值，再执行既有 60% RSS 线（`:87-95`）；probe 的 selected manifest 也把实测/已选 planning 值收紧到较小者（`src/vkgr/throughput_probe.py:46-53`）。A40 YAML 当前 `planning_memory_bytes=513000000000`、memory cap `256000000000`、disk cap `400000000000`。本报告独立读取得到 raw YAML SHA-256 `69f8039793ace8dfc67063ab94013a0ae958ad1da75e2f6e369dbc61582c8cc5`；规范化 `Engineering.identity()` SHA-256 `c917256ece3eee6aad65be3fdebb2db909d4a78856f72a8e476b7f9a43f036ad`。两种 hash 分别覆盖 YAML 原始字节与 dataclass 规范化配置值。

## 汇总与限制

- 工具 1：本次点名的预算检查顺序与严格列类型修复 **ADDRESSED**；容量预算仍是估算，不是已测峰值硬上限。
- 工具 2：装配路径输入/输出预留、进程树/worker RSS 预算检查、pending 取消 **ADDRESSED**；已运行任务等待与任务内瞬时峰值仍有限制。
- 工具 3：callback/返回值/trace 契约 **ADDRESSED**；整轮磁盘检查是预检，动态外部占盘仍可能竞争。
- 工具 4：零 hint cap 拒绝、共享 RSS guard **ADDRESSED**；PathLike 识别与 hint 类型合同 **PARTIALLY ADDRESSED**（`week26` 无下划线路径未拒绝；1-worker 与多 worker hint 类型检查不完全一致）。
- 工具 7：祖先 cgroup、probe/guard 共用、planning RAM 与配置身份 **ADDRESSED**；mutation receipt 无效，不能作为通过凭证。
- `cache/throughput-20261022/final-regression-v1.log` 的 39/41 其余通过项不作为工具 1/2/3/4/7 以外功能的结论；其中两项 decode factory/process initializer 环境身份测试因 mismatch 正确拒绝，属于其他工具当前 fixture，不记作本轮目标修复失败。
- 真实 anchor、真实 CUDA 与 context-inclusive 峰值、真实数据、大规模 RSS/IO/磁盘、吞吐收益：**NOT_RUN / UNKNOWN**。合成逐位 parity 和小 fixture 不能替代这些证据。
