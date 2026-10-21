# 工具 4「按周并行解析」独立 C 审查

- 审查角色：独立代码审查；当前 agent runtime 为 GPT-6。未验证模型切换为 Luna，因此不把执行记录标成 Luna。
- 审查范围：`src/vkgr/throughput_weekly.py`、`tests/test_throughput_weekly.py`、`src/vkgr/throughput_workers.py` 接口；对照 `docs/decision_log.md` 2026-10-22 第三部分第 4 项（11249b6）及 `docs/superpowers/specs/2026-10-22-throughput-options.md`。
- 只读限制：未连接服务器、未读真实数据、未训练；未运行完整 suite。项目 context 计数不可见，记为 UNKNOWN。
- 证据：RED 回执因 `vkgr` 未安装/未设源码路径而导入失败；GREEN 回执的工具 4 定向测试 2/2 PASS；`01-05-regression-v3.log` 报 24/24 PASS。另做一个显式合成边界探针，结果见下文。没有据此声称全量 backend 已接通。

## 结论

周结果的排序与串行 builder 的基本形态符合裁决：输入 week 严格递增，ordered_map 按 ordinal 交付，`chain.from_iterable` 让一个原 builder 在父进程中依次消费所有周，因此同一 builder 可保留跨周 state。默认 `week_workers=1`，且实现不挑选 target/label。现有合成测试能证明这个小 fixture 在 workers=1/2/3 一致。

但目前不建议把入口视为满足边界/预算合同：`Path` 类型的数据路径可绕过 holdout/week26 拒绝；并行 worker 的零值输出上界可绕过输出大小核验；而工具 4 没有绑定真实旧 builder 的单遍迭代兼容性，也未自行执行 RSS/磁盘 guard。修复前建议保持工程入口闭锁。

## 发现

### [P1] Path 类型的 holdout/week26 输入绕过拒绝

位置：`src/vkgr/throughput_weekly.py:11-17`。

拒绝逻辑只检查 `str`，递归只遍历 dict/list/tuple；`pathlib.Path`（包括藏在容器里的 Path）没有被规范化成路径字符串。我用 Python 3.11 的小型合成探针传入周号 0 和 `Path('/workspace/holdout/week_26.parquet')`，`derive_weeks` 接受并把路径传给 builder；字符串版本则按预期抛出 `ValueError`。这会使调用者用常见的 Path 输入绕过“week26/holdout path rejected”边界。

建议在入口递归识别 `os.PathLike` 并规范化检查，同时测试顶层 Path、嵌套 Path、`test`/`holdout` 路径以及合法训练路径。不要只靠 week number 检查：路径周号与参数中的逻辑 week 可以不一致。

### [P1/P2] `max_in_flight_bytes` 下零输出 hint 不形成上界

位置：`src/vkgr/throughput_workers.py:43-59`，尤其 `:52-56`；工具 4 从 `throughput_weekly.py:18-19` 透传。

workers>1 时只要求 `result_size_hint` 函数存在，没有校验其返回值为正的有效上界。返回 0 会让预留额按输入长度计算，且 `_task` 第 28 行也因 `if result_bound` 跳过结果大小验证；返回负数还会把预留额压低。于是配置了正 `max_in_flight_bytes` 仍可能接收任意大的解析结果，违背“in-flight cap 限制输出”的直觉与实现计划中显式给输出留预算的要求。workers=1 路径对零值会拒绝，造成 worker 数改变边界行为。

建议统一验证 hint 的类型和正值（允许空输出时可约定最小序列化上界，而不要把 0 当作“无限”），并确保 1/2/N worker 对同一无效 hint 一致拒绝。增加零、负值、非整数和真实超限结果的测试。

### [P2] 解析并行没有 RSS/磁盘保护，且 hint 按 pickle 长度计而非驻留内存

位置：`src/vkgr/throughput_weekly.py:18-19`；`src/vkgr/throughput_workers.py:25-29, 52-59`。

工具 4 没向 `ordered_map` 传 `memory_limit_bytes`，也没有在父进程或 worker 调用 `throughput_config.guard`。配置 `max_in_flight_bytes=0` 时，结果大小可以完全未知；非零 cap 时预留使用 pickle 长度，不能直接代表 Python 解析对象及 worker/父进程副本的 RSS。项目方案要求按周输出量与 prefetch 计入资源预算，并受 60% RSS / 80% disk 硬线约束；当前函数本身未提供这层保护。真实 parser 如把列式数据扩成 Python tuple/dict，pickle byte cap 可能低估驻留内存。

建议在正式接入前明确“in-flight bytes”究竟是序列化传输预算还是内存预算；若要承诺资源上界，应传有效 output bound 并将 worker/父进程 RSS 纳入 guard。保持默认 1 可降低暴露面，但不等于启用预算。

## 语义与覆盖核对

- 顺序：周号必须严格递增且在 0..25，接受稀疏周序；worker 结果由 ordinal 回收，`chain.from_iterable` 按输入周顺序串接。单个 parser 返回结果内部的 physical ordinal/顺序没有验证，依赖 parser 本身保序。
- 跨周 history：工具 4 没有逐周构造 history；交给同一个 builder 串行消费，符合“跨周 stateful builder 在父进程处理”的设计。测试使用了用户跨周重复的 fixture，workers=1/2/3 与直接串行 fixture 相等。
- builder 接口：当前仓库没有真实旧数据 builder 的调用点，只有合成 stateful builder。工具将 `chain` 单遍 iterator 传给 builder；如果原 builder 需要 `len()`、索引、重复遍历或具体 list 类型，语义/兼容性尚未证明。应由后续真实入口适配/测试确认，不应在本次审查中宣称已对拍旧 builder。
- target/label：本模块仅映射 parse 输出并调用 caller 提供的 builder，没有自动选 target/label；全量合同仍未冻结，代码自身没有越权选择。
- week 26：逻辑 week=26 被拒绝；字符串路径能拒绝，但 PathLike 绕过见 P1。
- lazy：`weeks` 在入口立即 `list(weeks)`，因此输入周描述不是 lazy；worker 输出仍按有界 pending 队列预取，但 cap 默认 0 时没有字节上界。串行 builder 的消费是 lazy 单遍。
- 真实接通范围：仓库没有 derive_weeks 的生产调用点，也没有真实 parquet parser/builder。该模块目前是 callback wrapper 和 synthetic test；不得据此声称全量 backend 已接通。

## 建议的后续验收

修复上述两项边界/预算问题后，加入 PathLike holdout/week26、hint 无效值和超限输出测试；再用真实旧 builder 的只读 synthetic rows 验证单遍 iterator、跨周 history、每周 physical ordinal 与 workers=1/2/3 结果相等。正式真实数据 preflight 和 launcher 链按项目规则另行完成；本次未执行。
