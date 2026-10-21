# 工具 2：多进程装配独立 C 审查

- 日期：2026-10-22
- 复核模型：gpt-6-luna / high（按委派要求登记）
- 范围：`src/vkgr/throughput_workers.py`、`tests/test_throughput_workers.py`、`analysis/throughput_step_harness.py`、`src/vkgr/throughput_config.py` 及 throughput 配置接口。
- 依据：`docs/superpowers/specs/2026-10-22-throughput-options.md` 工具 2（依据 87e5a29、11249b6）与本次委派列出的硬约束。
- 边界：只读本地源码与既有收据；没有连接服务器、读取真实数据、运行训练或重跑完整测试套件；没有改仓库文件。

## 结论

**C 档：有阻断项，当前不建议把 `assembly_workers > 1` 当作已满足资源上限的正式装配路径。** 主进程规划 indices、worker 只装配 rows、step 顺序与 G/P 合成逐位状态收据整体符合设计方向。阻断点是 `max_in_flight_bytes` 只核算提交任务输入的 pickle 字节，不约束更大的 worker 输出和父进程持有的结果，因此配置中的字节上限没有覆盖文档要求的实际在途内存。异常可传播，但提前关闭迭代器或一个 future 失败后，executor 会等待已提交任务完成，缺少主动取消；这是有界但可能延迟停止的行为，需在准入说明中明确或补齐。

| 项目 | 判定 | 证据与限制 |
|---|---|---|
| 主进程固定 indices / 随机抽样 | 通过（实现边界内） | `cursor_rows` 在主进程 deepcopy cursor 后规划 indices；worker 入口 `_rows` 只调用 `SequenceStore.rows`。P 的负例仍由 harness 调用原 `train_step` 时在主进程生成。没有发现 worker 端采样或负例生成。 |
| worker 不消费 RNG | 通过（当前 worker 实际路径） | `_task` 比较 Python、已加载 NumPy、已加载 Torch CPU 与已初始化 CUDA 的全局 RNG 状态；`_rows` 本身没有随机调用。现有随机 worker 变异被拒。通用检测不覆盖任意自建 generator 对象，但当前 `_rows` 不接收或使用这类 generator。 |
| 按 step ordinal 有序返回 | 通过 | `ordered_map` FIFO 等待提交顺序，并检查返回 ordinal；worker 延迟乱序的回归用例覆盖此行为。 |
| prefetch 有界 | 通过（任务数） | pending 数最多为 `workers * prefetch`，workers/prefetch 必须是正整数；workers=1 时顺序执行。字节上界另见阻断项。 |
| live cursor 不被未来预取推进 | 通过（已有覆盖） | 规划使用 `copy.deepcopy(cursor)`；只有对应 rows 交付时才调用真实 cursor，并与独立 expected cursor 及 `example_id` 核对。回归逐批比较 offset 与 cursor 内部 RNG。 |
| 异常与取消 | 有限但未完全闭合 | `future.result()` 会向调用者传播 worker 异常；未发现吞掉异常或自动重试。异常退出/提前关闭时，`ProcessPoolExecutor` 上下文按默认 `shutdown(wait=True)` 等待已经提交的任务，不主动 cancel；每次最多剩余 `workers * prefetch` 个任务，但若其中任务卡住，关闭会一直等待。无专门异常、生成器 close、取消延迟测试。 |
| worker 初始化与进程资源 | 部分通过 | spawn context、每进程 initializer、mmap 路径加载均已接线。非 mmap 多进程时会先估算 store 列副本并调用 `guard`；但估算不是 worker 实测 RSS。mmap worker 初始化会在每个进程完整 SHA 读取所有列再打开 mmap，可能形成启动 I/O 峰值；没有大表或并发启动实测。 |
| 配置 byte cap 生效 | **阻断** | cap 只比较输入 `value` 的 `pickle.dumps` 长度（`throughput_workers.py:43-45`）。`_rows` 输出可能远大于索引输入；父进程收到结果后在 `yield` 生命周期内还持有 rows，而下一轮可能已提交更多输入。`used` 在取出 future 后立刻减去输入大小（`47-49`），不计结果和 worker 中间对象。因而 256 MiB / 1 GiB 配置值不是总 in-flight bytes 或 worker RSS 上限。`guard` 也只读当前父进程 RSS（`throughput_config.py:52-72`）；其 projected memory 只在非 mmap 情况计 store 副本，不计 worker 输出、解释器基线或 mmap worker RSS。 |
| 默认为关闭 | 通过 | `Engineering` 默认 `assembly_workers=1`、`prefetch=1`、`use_mmap=False`；配置接口区分工程 manifest 和科学身份。 |

## 需修复的阻断项

1. **字节 cap 的口径必须覆盖真正被限制的对象。** `throughput_workers.py:43-49` 当前只核算输入 pickle，不能约束 rows 返回量、已完成但尚未消费的 future 结果或 worker 中间驻留。应将计量定义明确为任务输入、返回结果、执行中/排队对象的合计（或用可证明的保守上界），并保证 `max_in_flight_bytes` 超限发生在继续排队前。`throughput_config.py:68-69` 的内存检查应覆盖可获得的 worker RSS / 共享映射影响，或明确将其改为父进程限制并避免对外称作总 RSS cap。需要针对“小输入、大输出”和超限边界补定点验证。

## 非阻断观察

- `throughput_workers.py:37-49` 在 future 异常时及调用方关闭 generator 时等待已有任务；当前有界 pending 避免无限积压，但不是快速取消。若正式 job 要求中断能及时响应，需添加取消/关闭路径和一个有超时的合成验证。若接受等待，应在接口文档把“等待最多在途任务结束”写清。
- `throughput_workers.py:55-60` 的 mmap initializer 通过 `SequenceStore.load` 对所有 `.npy` 逐文件重算 SHA 后 mmap。数据越大、worker 越多，启动阶段重复完整读盘越明显；当前只有小型 synthetic fixture，性能与峰值资源 UNKNOWN。
- `_task` 的 RNG 检查覆盖当前使用的全局 PRNG，不是对任意自建 RNG 对象的形式证明；当前装配任务 `_rows` 不用 RNG，因此这项不阻断现有接线。
- `throughput_workers.py` 存在未使用的 `partial` 导入，属于清理项，不影响行为。

## 既有验证证据

- `cache/throughput-20261022/01-02-regression-v2.log`：9 项回归均 PASS；本次未重跑。
- `cache/throughput-20261022/mutations-02-v1/receipt.json`：baseline 接受；worker RNG、返回顺序、cursor 三个隔离变异均以非零退出被捕获，receipt 标注 isolated copy、unique cache、bytecode disabled。
- `cache/throughput-20261022/g-synthetic-step-v1.json` 与 `p-synthetic-step-v1.json`：G/P 冻结 step 合成 fixture 的 workers 1/2/3 `exact_step_state: true`；包含逐位 step 状态比较路径，P 负例通过原 step 路径生成。二者均注明 `real_anchor: NOT_RUN`、`quality_metric: false`。
- 真实 CUDA、anchor 大规模、真实 RSS / mmap 启动峰值及吞吐收益均 **NOT_RUN / UNKNOWN**；合成收据不证明这些条件。
