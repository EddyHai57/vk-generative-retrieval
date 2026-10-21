# Throughput 最终修复与 step harness 定点复核（2026-10-21）

## 范围与身份

- 独立只读复核，仅看本轮指定的 PreparedBatch row 快照、memory-only budget、TorchSession 动态类身份，以及 `analysis/throughput_step_harness.py` 的 G/P worker 对拍与 namespace。
- 未改仓库；未连接服务器、真实数据、GPU 或真实训练；未调用 git，也未派发其他 agent。项目文件及上一轮报告已读取。
- 派发说明为 Luna / high，但本环境没有独立 runtime 型号回执，不能证明实际运行时型号。上下文计数不可见：UNKNOWN。
- 未重做既有七项 C 审；本报告不判断科学有效性、anchor、转正、真实性能或真机显存。

## 结论

三项指定代码修复在源码与定向 RED/GREEN 证据上均关闭了上一轮发现。当前 harness 已补齐六项 SHA namespace（input/SID/config/source/runtime/constructor），并把 frozen 原 G/P CPU step 与生产 `prepared_batches` / cursor 路径接起来；现存 G、P 回执分别覆盖 1/2/8 个 assembly workers，均报告 exact step state parity。release suite v4 为 54 项通过，新增 G 多 constructor call 与 cursor tail 用例也通过。这些证据支持 synthetic CPU 工程对拍，不能替代真实 anchor 或 CUDA 验收。

## 逐项复核

1. **PreparedBatch 防止 row 变更复用 stale tensor：闭合。** `src/vkgr/throughput_tensors.py` 的 `PreparedBatch.__post_init__` 将各 chunk 的 rows 序列化成独立 JSON 字符串摘要；`bind` 每次调用均把实际传入 rows 与该摘要比较。摘要不与可变 row dict 共用引用，修改 `PreparedBatch.rows` 的嵌套值会使绑定构造器拒绝，而非返回旧数组。`closure-findings-red-v2.log` 中定向 mutation 断言按预期失败（原缺陷可复现）；`closure-findings-green-v3.log` 中同一断言通过。`tests/test_throughput_tensors.py` 的负例将 `side` 嵌套值改为 99，并要求 `ValueError`。

2. **仅启用 memory cap 时要求有界输出并使用有界 flight：闭合。** `prepared_batches` 对 `max_in_flight_bytes` 或 `memory_limit_bytes` 任一开启且缺少 `tensor_bytes_bound` 的输入均 fail closed。memory-only 路径计算可用预算并设置非零 `in_flight`，再为 worker 分配剩余预算；有 output bound 时 `ordered_map` 同时接收 `max_in_flight_bytes` 与 `result_size_hint`。RED/GREEN 日志与测试 `test_memory_budget_alone_requires_output_contract` 覆盖该拒绝行为。

3. **TorchSession 动态类身份与闭包来源：本轮支持合同内闭合。** `_hash_torch_module` 遍历自定义 model class 的 MRO 及 class namespace，摘要运行时 class 属性；property、staticmethod、classmethod 和函数分别检查源码、默认值与 closure。`__class__` closure 只在目标 model MRO 内按源摘要；`torch` / `numpy` 模块 closure 要求对应对象仍是 `sys.modules` 中已加载的模块，并纳入版本与模块文件字节。其他 module closure 拒绝。定向 RED/GREEN 覆盖 class 属性变更前后身份不同，release suite v4 中相关测试通过。按给定合同，不支持对已锁定 torch/numpy 运行库做 monkeypatch；未知模型运行属性 fail closed。模块文件摘要不构成对整个 Python 运行环境的隔离保证。

4. **G/P worker step harness：结构与回执一致。** `analysis/throughput_step_harness.py` 从指定 source tree 动态加载 frozen `vkgr`，对实际 G `sid_targets_l` / P `_tensors2` 构造器计算 source SHA 并检查 worker factory 重建出的构造器。namespace 使用生产 `tensor_cache.namespace`，绑定六项 SHA：synthetic input rows、SID/token context、配置、source tree 下 Python 文件账本、Python/NumPy/Torch 版本及 CPU runtime 字段、构造器源码。worker factory 重建后按同一 pins 和实际 SID context 复算 namespace。G validator 比较 token/catalog context；P validator 比较 idmap、feature、padding width。

   Prepared 路径经 `cursor_prepared_batches`、`PreparedBatch.bind` 调用原 train-step code object；每个 step 用 frozen G/P CPU step、CPU model/optimizer 与 synthetic rows 执行。worker=0 作为 baseline，workers=1/2/8 对比 tensor bytes、输入 trace、step 前后 model/optimizer/gradient/loss/cursor 与 Python/NumPy/Torch RNG；任一差异会抛错，不会写成功 JSON。P 的观察 wrapper 保留 step code object，并捕获 labels/mask/negatives 与 log-q 输入。

   `cache/throughput-20261022/step-g-release-v2.json` 和 `step-p-release-v2.json` 均记载 `exact_step_state: true`、`train_step_code_unchanged: true`、`workers: [1, 2, 8]`、`fixture: synthetic-only`，并明确 `real_anchor: NOT_RUN`、`quality_metric: false`。因此 harness 已完成局部 synthetic CPU step 对拍；回执只有布尔结果，没有逐项 hash/差异明细，其可审计性低于完整原始 trace。

## 验证证据与限制

- `closure-findings-red-v2.log`：三个新增定向断言分别复现旧缺陷，3 项失败。
- `closure-findings-green-v3.log`：三个相同断言修复后 3 项通过。
- `tensor-g-multichunk-cursor-v1.log`：新增 G 多 constructor call、cursor tail synthetic 用例通过。
- `release-suite-v4.log`：54 tests、28.987s、OK，含上述定向缺陷测试、动态 class 属性身份测试及 worker 用例。此处只读取现存日志，未重跑测试；不把旧 review 记录的 OpenMP / `ps` 沙箱问题当作本次代码失败。
- 预算是保守估算及边界检查，不是 OS 硬隔离或瞬时 RSS 上限；真实 CUDA、anchor、吞吐和峰值显存均 NOT_RUN。harness parity 只覆盖其 synthetic fixture 和当前 pinned G/P step/source/config/runtime 调用。
