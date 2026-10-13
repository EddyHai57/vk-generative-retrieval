# G 验证分片并行（INFRA-01 A）

密级: 公开

## 用法

- `VKGR_EVAL_WORKERS`：验证 decode 子进程数。未设或 `''` = 1，走原串行 `ev.evaluate_queries`，代码路径不变。
- 合法值 `^[1-9][0-9]?$`，≤32；容器有 cgroup v2 `cpu.max` 配额时还必须 ≤ `floor(quota/period)`。非法值在 `main()` 任何输出之前 exit 1（`G3 gate failed: eval workers env…`）。
- 默认值不按核数自动推导，必须显式设置。
- 资源规则：**workers × 同时在跑的作业数 ≤ cgroup CPU 配额**（不是 `os.cpu_count()`；容器里 `os.cpu_count()` 显示的是宿主核数）。
- workers>1 要求 `OMP_NUM_THREADS=1` 且 runtime `threads==1`；新 permit 必须在 `OMP_NUM_THREADS=1` 下签。
- 只对新 run 生效：本模块进入 `consumer.source_members`，旧的 prepared/permit/run 在新源码下不能 load/resume/verify（design R4）。
- `VKGR_PROGRESS_PATH` + `VKGR_RUN_ID`（launcher 注入）：可选的 `vkgr-progress-v1` 进度文件；路径必须是绝对路径、父目录存在、不是 symlink，且不能在 output 或 pool 目录内。写失败只在 stderr 打一行 `progress write disabled: <errno>`，不影响作业。

## 执行模型

- 例子：7 条 query、shard_size=3 → 3 个 shard；workers=2 时 shard 0、2 给 worker 0，shard 1 给 worker 1。
- 每次 evaluation 新起 `min(workers, 剩余 shard 数)` 个 spawn 子进程（至少 1 个），结束即回收。
- worker 只做 `tiger.decode_history`，回 `(ordinal, items, audit, latency_seconds)`；不写盘、不拿 target、不拿 catalog、不调用 `context.check()`。
- 父进程按 shard 升序：校验回包身份 → decode gate（与串行同一条件、同一消息）→ 构造 row → 写 shard / progress → 检查 stop。
- 每个 worker 最多 2 个 in-flight shard；stop 时停派、等已派发的全部回来再写 incomplete manifest，所以 reserve 数 = decode 数 = 已发布行数。
- 大 payload（codec、state_dict、histories）经 Pipe 发送，不放进 `Process` 参数：CPython 把参数写进一条父进程仍持有读端的管道，子进程在 bootstrap 中死掉会让 `start()` 永久阻塞。

## 预算钩子（与串行的差异）

- 串行：每条 query 前后各调一次完整 `_guard`（pool + runtime/environment + context digest + 资源观测）。
- 分片：开始时一次完整 `_guard`；**每个 shard 派发前一次完整 `_guard`**；每条 query 前后各一次 `_query_guard`（`pool.check()` 的 deadline 检查 + 真实模式的 `observe_resources`）。decode reserve 仍是每条 query 一次。
- 原因：本地实测完整 `_guard` 的单次耗时主要来自 `gd._environment()`（读包元数据）；queries/catalog digest 随规模线性增长。数字见 `.agents/tasks/infra-01/impl-a-status.md`（不入库）与 experiment_log（服务器实测后）。
- reserve 最多比真正 decode 早 2×workers 个 shard；pool 从不退款。

## 信号与孤儿

- 父进程在每次 `Process.start()` 期间 block SIGINT/SIGTERM，且先 `resource_tracker.ensure_running()`：tracker 首次启动会在父进程里 SIG_UNBLOCK 这两个信号，导致 mask 失效（本地实测复现）。
- worker target 前 4 步：SIG_IGN → unblock（丢弃挂起信号）→ Linux `PR_SET_PDEATHSIG=SIGKILL` → ppid 校验。`generative_sharded.py` 顶层只 import 标准库。
- Ctrl-C/SIGTERM 只由父进程处理（`main()` 置 stop 标志 → incomplete manifest，可 resume）。worker 被 kill → `sharded evaluation worker died`，不写 manifest。
- 清理：关 Pipe → `join(5)` → `terminate`（worker 忽略 SIGTERM）→ `kill`；返回时无残留子进程。
- 用 Python 脚本直接调用 `evaluate_queries_sharded` 时，脚本必须有 `if __name__=='__main__':` 保护（spawn 会重导入主模块）；生产入口 `python -m vkgr.generative_real` 已满足。

## 等价性口径

`evaluation_projection(dir)`：shard 去掉 `latency_seconds`，progress/manifest 去掉 `observations`、shard 路径只留文件名，其余逐字节比较。允许不同的只有 latency、observations（含 `execution`）、shard 文件 sha256、输出目录绝对路径。

`observations.execution`（仅 workers>1）：`workers`、`start_method`、`assignment`、`definition`，以及父进程分段计时 `parent_seconds{startup,hooks,wait,write}` 与 `worker_decode_seconds`。并发下 latency 偏高，不能和串行 latency 比较。

## 未在本地验证

- 真实 CUDA 下跨进程 decode 是否 bitwise 一致、单 worker 峰值显存：正式启用 workers>1 前须在服务器对拍（design §5.3）。
- Linux 上的 `PR_SET_PDEATHSIG` 路径（本地是 macOS）。
- 真实规模下 `_guard` 各部分的耗时。
