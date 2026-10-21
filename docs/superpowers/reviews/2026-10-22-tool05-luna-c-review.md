# 工具 5（P query 块并行）独立 C 审查

- 日期：2026-10-22
- 派发参数：`gpt-6-luna` / `high`。当前运行环境未提供独立型号回执；这里只记录派发参数，不据此声称实际运行模型型号。
- 范围：`src/vkgr/throughput_exact.py`、`tests/test_throughput_exact.py`、共享的 `throughput_workers.py` 与 `throughput_config.py`；原实现取自 `git show 0ca61bc:src/vkgr/baselines.py`。
- 依据：`11249b6` 的吞吐授权（`docs/decision_log.md` 中工具链条款）、`docs/superpowers/specs/2026-10-22-throughput-options.md` 工具 5、`docs/superpowers/plans/2026-10-22-throughput-implementation.md`。
- 边界：主 tree 只读；只读本地源码和既有日志。没有读真实数据、连接服务器、跑训练或重跑全套测试；没有修改仓库文件。

## 结论

**C 档：有资源预算阻断项，暂不把 `eval_workers > 1` 的 mmap catalog 路径视为满足内存上界。** 固定块计算、float64 `einsum`、item-ID tie-break 与 ordinal 回收符合冻结要求；已留存的 35-query / 4103-item 对拍和 1/2/3 worker 收据支持当前合成样本上的预测字节与分数块 SHA 一致。主要缺口是 catalog 被 dtype 转换成常驻副本时，mmap 分支没有计入该副本，而且守卫在转换和验证扫描之后才执行。

## 发现

1. **[P1] mmap + 非 float32 catalog 会复制完整 catalog，内存投影却将其排除。** `throughput_exact.py:50` 把 `items` 转为 `float32`；如果原输入是非 float32 的 `np.memmap`，该转换分配一份完整 ndarray。`:58` 因 dtype 不同而把转换后的 ndarray 放入 worker initializer，不再传 mmap descriptor；spawn worker 会各自反序列化这份数组。`:59` 在 `config.use_mmap` 为真时仅计入 query 与 ID，不计 catalog 副本，`:60` 的 guard 因而低估 worker 常驻内存。冻结原函数接受并执行 `float32` 转换，因此简单改变转换规则会改接口行为。应在守卫前准确计入该复制路径，或要求并验证预先转换且逐字节固定的 float32 mmap 输入。

2. **[P1] 大数组临时分配和扫描先于资源守卫。** `:50` 的 dtype cast、`:52` 的 `np.isfinite(...).all()` 与 `np.unique(ids)` 均发生在 `:60` 的 guard 之前。cast 可分配完整数组；finite 检查与 unique 会产生与输入规模相关的工作内存。若配置内存上限接近机器余量，guard 不能阻止这些前置峰值触发 OOM。需要让守卫覆盖转换与验证的峰值，或把这些步骤改为有预算的分块检查/转换；修改科学数值路径前须保持对原函数的等价性。

## 逐项核对

| 项目 | 判定 | 证据与限制 |
|---|---|---|
| 32-query / 4096-item 块与算式 | 通过 | `throughput_exact.py:34-45` 固定 query/item 块；分数调用 `np.einsum('qd,id->qi', ..., dtype=np.float64, optimize=False)`，与 `0ca61bc` 原函数一致。无 GPU 公式、`argpartition` 或 ANN。 |
| ties 与预测顺序 | 通过（现有 synthetic） | `np.lexsort((combined_ids,-combined_scores))` 保持 item ID 次序；按 query block ordinal 有序收集。测试含 35×4103 尾块、逆序 ID 和零向量 tie。 |
| 分数块摘要与并行一致 | 通过（现有 synthetic） | 测试对 workers 1/2/3 比较预测 `.tobytes()` 与旧函数，并比较 trace 列表；trace 对每个固定 score block 计算 SHA-256。未对百万级 catalog 实测。 |
| mmap descriptor | 部分通过 | 原 dtype 匹配时传 `.npy` 路径、shape、dtype，并在 worker 只读重开及复核 shape/dtype（`:15-25,57-59`）。未验证文件内容身份；更关键的是 dtype 不匹配时退回整表 ndarray 并触发上面的欠预算。 |
| ndarray 与输入校验 | 部分通过 | 维度、长度、有限值、唯一非负整数 ID、`k` 和空 catalog 均有检查（`:51-54`）；空 query 会得到 `(0,k)` 输出。0 维 ID 可能在 `len(ids)` 处抛 `TypeError`，但不影响冻结的正常输入。 |
| in-flight cap | 通过边界实现，限制有条件 | 每任务为输入与结果 hint 留保守 3 倍 reservation，worker 检查序列化结果不超过 hint；上游 `ordered_map` 在 cap 启用且多 worker 时要求提供正界。工具 5 的 result hint 给出预测字节与 score SHA trace 预算（`:62-64`）。config 默认 `max_in_flight_bytes=0` 表示不启用此项硬 cap，仍由任务数 `workers*prefetch` 限流。 |
| 进程资源与默认值 | 部分通过 | spawn 与 worker initializer 用法清楚；默认 `eval_workers=1`、`prefetch=1`，保持单进程默认。`guard` 检查当前 parent/children RSS 和投影（`:60,65`），但没有传 `memory_limit_bytes` 给 worker 的单独 post-task 检查；该预算缺口与上述投影/时序问题共同构成阻断。 |

## 既有证据

- `cache/throughput-20261022/05-red.log`：初次测试因模块未安装到导入路径失败（`ModuleNotFoundError`），属于运行环境设置问题。
- `cache/throughput-20261022/05-green.log`：工具 5 两个目标测试通过，2 tests / 1.773 s。
- `cache/throughput-20261022/01-05-regression-v3.log`：包含工具 5 测试，整组 24 tests / 14.699 s 全部通过；本次未重跑。
- 现有测试未覆盖非 float32 mmap 的投影值/worker 副本，也未测超内存上限时转换前后峰值。

## 结果与限制

本审查仅给工程与机械一致性结论，不作科学有效性、anchor 或转正判断。真实百万级规模、真实 catalog mmap RSS、CUDA 与性能均 **NOT_RUN**。修复上述预算问题前，不应将 `eval_workers>1` 的非 float32 mmap 路径记作已满足内存 cap。
