密级: 公开
# ops/SCHEDULING.md — 容量模型、成本表、排程预测（SCHED-01）

目标：服务器别空着。工具 `ops/tools/server/sched_forecast.py` 只读地回答三件事：现在占了多少、手上的活还能撑多久、什么时候会空。

```
python -I ops/tools/server/sched_forecast.py --root R [--json] [--backlog-hours 4] [--gpu-samples 5] [--tick-seconds 30] [--tz-hours 8]
```

- 在服务器上直接跑（不是 vkgr-launch 作业），只用标准库。
- 只读：不写文件、不加锁、不发信号、不碰 `queue/`；子进程只有 nvidia-smi。
- 输出不含路径、pid、argv、env、IP、端口、token、用户 id；只有 run_id、作业类型、资源数字和时间。
- 测试：`PYTHONPATH=ops <python> -B -m unittest discover -s tests -p 'test_ops_sched_forecast.py'`。

## 1. 容量模型

| 资源 | 总量从哪里读 | 可用上限 |
|---|---|---|
| CPU | cgroup v2 `cpu.max` quota/period（无配额时用 `os.cpu_count()`） | total × 0.8 |
| VRAM | `nvidia-smi --query-gpu=memory.total`（多卡求和） | total × 0.8 |
| RAM | `/proc/meminfo` MemTotal 与 cgroup `memory.max` 取小 | total × 0.8 |
| 磁盘 | `statvfs(R)` | total × 0.8 |

- 20% 余量与 `vkgr_ops.admission.HEADROOM` 是同一个常量（直接 import），调度器改了这里跟着变。
- 例：配额 10.24 核 → 上限 8.19 核 → 同时约 8 个 1 核作业。`os.cpu_count()` 看到的 64 不能用。
- 正式准入还有第二条（实测空闲 − 启动不到 600s 的作业声明量 − need ≥ 20%）。预测只用第一条（声明量），所以预测偏乐观，见 §4。

## 2. 作业类型与成本表

- 类型 = argv 里的入口：`python -m <module> <sub>` → `"<module> <sub>"`，如 `vkgr.retrieval_feat02_run train`、`vkgr.retrieval_feat02_run verify`、`vkgr.generative_b1_arch train`。脚本入口用文件名（不带目录）。train 和 verify 自然分成两类。
- 成本 = 所有 `terminal.json.status == succeeded` 的 run 的 `wall_seconds` 的中位数与最大值，另记该类型声明资源的最大值。
- failed / wall_cap_exceeded / stopped / launch_failed 不进成本表（墙钟不代表一次完整作业）。
- `vkgr-sched serve` 是常驻服务，不进成本表，也不算活量。
- 成本表每次运行时从 `runs/` 现算，没有手写数字；新 run 完成后下一次运行就自动更新。

## 3. 预测怎么算

running 作业剩余时间：

| 条件 | 剩余 | 标记 |
|---|---|---|
| 已运行 < 中位数 | 中位数 − 已运行 | `median` |
| 中位数 ≤ 已运行 < 最大值 | 最大值 − 已运行 | `max` |
| 已运行 ≥ 最大值 | 0（随时会结束，或卡住了，去看日志） | `overdue` |
| 类型没有成功历史 | wall cap − 已运行 | `wall_cap` |
| `vkgr-sched serve` | 不结束 | `service` |

pending 作业：按文件名顺序做调度器同款 first-fit 模拟。

- 时钟从 0 开始；每个 tick（默认 30s，= `vkgr-sched serve` 间隔）最多启动 1 个。
- 对每个等待作业，用声明资源判断 `(total − committed − need)/total ≥ 0.20`；第一个放得下的启动，放不下的留 pending，后面的小作业可以先走（first-fit，不是 FIFO）。
- 都放不下 → 跳到下一个作业结束的 tick 再试。
- 持续时间用该类型中位数；没有历史用 `wall_hours`（标记 `wall_cap`）。
- 单个资源在空机上都过不了 20% 规则 → `never_fits`（调度器会把它放进 rejected）。

汇总：

- `backlog_job_hours`：成本表估出来的剩余活量（running 剩余 + 能排上的 pending 时长）之和，单位作业-小时。
- `backlog_job_hours_wall_cap`：没有成本历史、只能按 wall cap 估的部分，单列，不计入告警（wall cap 远大于实际，算进去会掩盖「快没活了」）。
- `idle_at_s`：所有有限作业都结束的时刻（服务进程不算）。
- `cpu_below_half_at_s`：第一次声明 CPU 低于可用上限一半的时刻（例：8.19 核 → 4.1 核）。0 表示现在就低于一半。
- 告警：`backlog_job_hours < --backlog-hours`（默认 4）时输出第一行是 `BACKLOG_LOW`，否则 `BACKLOG_OK`。
- 时间同时给相对值（`+1h30m`）和本地钟点（`--tz-hours`，默认 UTC+8）。

## 4. 假设与局限

- 只用声明量模拟；实测比声明大（observed > declared）时，真实准入会更晚。表里同时给出声明和实测 VRAM，差距大时先改声明（§5）。
- 不模拟 600s settling 规则和实测空闲规则：多个作业刚启动时真实准入会晚几个 tick。
- 成本是历史墙钟，不随负载缩放：CPU 被限流（同时跑太多）时每个作业都会变慢。
- 同一入口不同配置（如小 G 和论文规模 G 都是 `generative_b1_arch train`）会混在一类，中位数居中、最大值偏大。需要分开时给入口加子命令或单独模块，不在工具里加白名单。
- 历史 run 太少（n=1）时中位数就是那一次，可信度低；表里给出 n。
- 不预测还没入队的作业；就绪队列和开发中的作业要人工维护在排程表里（§5）。
- GPU 利用率是采样 N 次（间隔 1s）的均值和最大值，只反映这几秒。

## 5. 流水线规则（排程表维护）

1. 一批作业入队的同时，下一批的代码和 review 必须已经开始。
2. 维护就绪队列：代码已 review、实验卡已冻结、随时可入队。目标：就绪队列估计时长 + 服务器剩余活量 ≥ 8 小时。
3. 预登记填空作业：不计名额、不改结论、只为填满机器（已有 anchor 的额外 seed、只读诊断、复验 verify）。每项写成本、价值、前置条件；`BACKLOG_LOW` 时先从这里取。
4. 显存声明：声明 ≈ 实测峰值 × 1.15，向上取整到 256 MiB。例：实测 5690 → 6656；实测 2046 → 2560。

## 6. 全量阶段怎么复用

- 换机器或改配额：不用改代码，容量每次运行时重读（cgroup、nvidia-smi、meminfo、statvfs）。
- 新作业类型：第一次没有历史，按 wall cap 估并单列；第一次成功后自动进成本表。
- 全量数据作业比子集慢：子集历史不要混进全量类型。全量入口用新的模块或子命令名，成本表自然分开。
- 告警门槛随批次大小调整：`--backlog-hours`。
