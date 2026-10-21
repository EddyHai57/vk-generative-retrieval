# 工具 6：G 逐 query 多进程独立 C 审查

- 日期：2026-10-22
- 复核请求模型：`gpt-6-luna / high`（按委派要求）；本任务没有独立 runtime 型号回执，因此不声称已验证模型切换。
- 范围：`src/vkgr/throughput_decode.py`、`src/vkgr/throughput_workers.py` 的共享执行接口、`tests/test_throughput_decode.py`、工具 6 方案及既有 06 日志。
- 依据：`docs/superpowers/specs/2026-10-22-throughput-options.md` §工具 6；`docs/superpowers/plans/2026-10-22-throughput-implementation.md` 第 8、10 项；本次委派的硬约束。
- 质量边界：只读审查；没有连接服务器、读取真实数据、跑科学训练或重跑完整测试套件。未修改主 tree。真实 CUDA、真实模型、anchor 与吞吐验收均未做。

## 结论

**C 档：`decode_workers=1` 的默认路径符合直调原 decoder 的等价要求；`decode_workers>1` 仍有阻断项，暂不能视作已证明与原 decoder 科学身份等价或已完成资源准入。**

| 项目 | 判定 | 证据与限制 |
|---|---|---|
| 默认单 worker 等价 | 通过（代码路径） | `throughput_decode.py:48` 在 `decode_workers==1` 时直接逐 query 调用 `original_decode`，绕过 worker session、序列化与新解码 wrapper；与“默认关闭”一致。`Engineering` 默认值为 1（`throughput_config.py:25`）。测试用例也包含 n=1 对拍。 |
| `TorchSession` 实际接入 | **未通过 / 阻断** | `TorchSession` 定义了 eval、权重/codec 摘要和 token（`throughput_decode.py:12-31`），但 `decode_queries` 从未构造或检查它；并行执行完全依赖调用方传入任意 `session_factory`（第 46-50 行）。当前测试只用轻量 `Session` / `BeamSession`，未覆盖 `TorchSession`（`test_throughput_decode.py:11-48`）。所以现有 green 证明的是通用 toy factory 接口，不证明实际 torch model/codec 接线正确。 |
| 原始 session 与 worker session 身份绑定 | **阻断** | 调用方提供的 `state_sha` 只校验 SHA 格式；worker initializer 确认 factory identity 等于这个值（`throughput_decode.py:34-37, 49-50`），但函数没有把该值同 `original_decode` 所属的主进程模型/codec/source/runtime 身份关联。若传入与原 decoder 不同但与 worker factory 相符的 state_sha，workers>1 会成功运行另一套 decoder，结果仍按顺序返回。worker 间彼此相同不构成“同原 decoder”。 |
| identity 覆盖面 | **不足以支持完整 runtime/source 身份声明** | `TorchSession.identity()` 哈希 callback 源码、state_dict 张量、catalog ids/sids、beam 与少量 torch flags（`throughput_decode.py:16-25`），但未纳入完整 codec/catalog 结构或实现、依赖函数闭包、Python/Torch 版本等 runtime 标识。相同 ids/sids 不能单独证明相同 codec 行为。这一点与上一项共同限制并行等价声明。 |
| query 内 RNG 与模型状态 | 部分通过 | `_decode` 比较 session token 前后；通用 `ordered_map` 再比较 worker 全局 Python/已加载 NumPy/Torch RNG（`throughput_decode.py:40-43`、`throughput_workers.py:13-20, 23-30`）。但 `TorchSession` 路径没有被测试；全局 RNG 检查不覆盖自建 generator / 外部随机状态。当前 synthetic 仅验证 toy callback 与所用全局 Python RNG。 |
| GPU 显存准入 | **有条件，边界校验缺失** | CUDA 分支对 `None` 的峰值或 live free 值 fail closed，并要求 `workers × per_worker_peak <= live_free × .8`（`throughput_decode.py:51-53`），公式方向符合方案。参数没有要求正数/有限数/整数或来源回执；0、负数或任意小的错误值可以通过。且代码只检查传入值，不证明峰值是 context-inclusive 实测、live free 是本次运行值。真实 CUDA 明确 NOT_RUN。 |
| worker RAM 与 initializer 错误 | 部分通过 | 总预算先扣父进程 RSS 与 in-flight 上限，再均分给 worker；每个任务前后检查 worker 自身 RSS（`throughput_decode.py:54-58`、`throughput_workers.py:23-30`），父进程每个结果后再 `guard`。但 session_factory 在 initializer 中先加载模型，超预算检查到第一个任务才发生（`throughput_decode.py:34-37`），初始化峰值可能先超过配置上限或触发 OOM。initializer 抛错会由 ProcessPool future 传播并使本次调用失败，不会返回成功列表；没有专门 initializer 失败测试或错误归因封装。 |
| result bound | 部分通过 | 并行路径将 `result_bound` 同时作为 in-flight 预留和 pickle 结果上限，实际结果超上限会在 worker 抛 `MemoryError`（`throughput_decode.py:46-47, 60-62`；`throughput_workers.py:28, 52-58`）。默认是固定 8192 bytes；未见依据或按 decoder 输出形态校准。显式传入 0 会让 `_task` 的实际结果长度检查关闭；当 in-flight cap 为 0 时还会绕过正 bound 检查，故接口允许无结果上界运行。默认值路径有界，但调用参数未强制保持这一性质。 |
| ordinal 返回与 toy 对拍 | 通过（合成范围） | `ordered_map` 按提交 FIFO 收回并核对 ordinal（`throughput_workers.py:42-64`）。`test_throughput_decode.py` 覆盖 worker=1/2/3 的顺序、toy identity、全局 RNG、随机回调拒绝、未知 GPU 参数拒绝及合成 catalog 的冻结 `constrained_beam_d` 对拍（第 50-67 行）。未覆盖真实 G loader/model/codec、TorchSession、initializer 错误、结果超 bound、显存参数类型与单位、CUDA。 |

## 阻断项

1. **把并行 session 身份绑定到主进程原 decoder。** 目前 `state_sha` 是未认证的独立字符串参数，只绑定各 worker 与调用方所给 SHA。并行入口应可验证原模型/codec/source/runtime 身份与 factory 加载身份相同；仅验证 worker 之间相同不够。
2. **明确并测试实际 TorchSession 装载路径。** 当前 `TorchSession` 是未接入的类，toy session 测试没有验证 `model.eval()`、真实 state_dict、catalog arrays、callback 参数契约或模型状态 token。需要一条针对可合成 torch model/codec 的测试证明正式 factory 创建的是该接口，并在身份不匹配时失败。
3. **显存参数应 fail closed。** CUDA 并行应拒绝非正数、非有限数、错误类型或未标为实测的 per-worker context-inclusive peak / live free 值；否则仅传 `None` 才能视为 UNKNOWN 的规则不充分。
4. **预算需覆盖 initializer 加载阶段。** RAM 预算需在 session factory 建立前/过程中有可执行的限制或先验峰值门禁。现在任务级 RSS 检查发生在模型加载之后。
5. **强制正结果 bound。** `decode_workers>1` 时应无条件要求有效正整数结果上界；目前 `result_bound=0` 可在无 in-flight cap 时关闭实际结果大小限制。

## 已有验证收据（只读）

- `cache/throughput-20261022/06-red.log`：初始测试导入失败，原因是当时模块不可导入；不能作为行为失败或通过证据。
- `cache/throughput-20261022/06-green-v2.log`：2 项 toy 用例通过，未包含 frozen beam 用例。
- `cache/throughput-20261022/06-final-green.log`：3 项用例通过，包括 n=1/2/3 对拍及合成 catalog 上从 `b5a4d1c:src/vkgr/sid_depth_beam.py` 读取的冻结 `constrained_beam_d` CPU 路径。
- 这些收据支持本地 CPU synthetic 行为，不证明真实模型/loader、真实 catalog、CUDA、context-inclusive 显存峰值或性能收益。按要求本次没有重跑全套测试。

## 限制

未检查或读取真实模型 checkpoint、真实 embedding、完整预测；未运行 CUDA；未测模型初始化峰值、worker RSS、GPU context 或吞吐。没有证据足以评价科学结果、anchor、性能收益或实机准入。建议保留 `decode_workers=1` 默认；并行路径待身份与预算阻断闭合后再做工程复核。
