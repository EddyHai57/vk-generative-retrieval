# 工具 5 资源边界修复记录

- 执行身份：Luna 实现者；仅在指定隔离 worktree 操作；主 tree 未修改。
- 修改文件：`src/vkgr/throughput_exact.py`、`tests/test_throughput_exact.py`。
- RED/GREEN 日志：`cache/throughput-exact-fix/00-red.log`、`cache/throughput-exact-fix/01-green.log`。

## 修复

- `exact_topk` 在 dtype cast、`isfinite` 与 `unique` 扫描前，先用输入 shape/dtype 估算转换分配和验证临时数组，再调用资源守卫。
- 并行 `use_mmap` catalog 要求 float32；非 float32 在转换前 fail closed，避免 mmap 转成 ndarray 后被每个 worker 复制且漏记投影。
- worker 使用 mmap descriptor 前验证 `.npy` 路径、shape、dtype、strides、offset 和完整 C-contiguous 映射；切片映射会在建 worker pool 前拒绝。worker 仍用 `mmap_mode='r'` 只读打开。
- 转换后的 worker 数据驻留量纳入并行投影；配置了 `memory_limit_bytes` 时将派生的 worker RSS 上限传给既有 `ordered_map`。
- `exact_topk` 参数不变。并行 `eval_workers>1, use_mmap=True` 的工程输入边界收紧为完整 float32 `.npy` mmap；默认单 worker 的有效 float32 算分路径和切片 mmap 输入数值行为、固定块、float64 einsum、tie-break 与 ordinal 收集未改。

## 验证

命令：

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 <WORKSPACE>/.venv/bin/python -B -m unittest tests.test_throughput_exact
```

结果：5 tests / OK。新增边界测试先在旧实现下 RED：非 float32 mmap 观察到 cast 早于 guard；非 float32 mmap 没有被并行 mmap 路径拒绝。另一个默认路径回归测试先暴露了切片 mmap 被错误拒绝；修复后 GREEN。既有 fixture 用 owned `tests/fixtures/throughput_frozen_baselines.py`；测试在 workers 1/2/3 逐字节比较预测与 baseline，并逐项比较 score trace SHA；35 queries、4103 items、尾块、逆序 IDs 与 zero-vector ties 均在原测试内覆盖。

## NOT_RUN / 限制

- 未读真实数据、未连服务器、未运行训练或 CUDA、未做真实 catalog RSS 或性能测量。
- 未运行整个项目测试集；本次按委派只运行工具 5 定向测试。
- 本地 Python 3.11，使用主 tree `.venv`。
- 本地操作未 commit/push；无新增科学计算或结果解释。
