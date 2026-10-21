# BHKM exact external-sort independent review

审查对象：`src/vkgr/throughput_bhkm.py`、`tests/test_throughput_bhkm.py`、`docs/superpowers/reviews/2026-10-22-bhkm-implementation-notes.md`、`docs/superpowers/specs/2026-10-22-throughput-options.md`；原实现取 `b5a4d1c:src/vkgr/sid.py`。依据 `docs/decision_log.md` 中 87e5a29 / 11249b6 对应规则。主工作树只读；未连接服务器或真实数据，未运行全套测试。

## Verdict

**CHANGES_REQUESTED：科学顺序保持路径审查通过；外部排序的动态磁盘预留和配置适配接口尚有工程阻断，修复并加定向回归后再接正式入口。** 不据此推断全量资源足够。

## 等价性与覆盖

- **距离路径通过。** `_distances` 以 float64 累加，并按维度 `0..D-1` 顺序逐维执行 `delta * delta`（`throughput_bhkm.py:24-30`），与冻结源码 `_distance` 一致。各块以 `(distance, item_id, child)` 排序（`:37-46`），全局堆归并按相同三键排列（`:130-148`）；唯一 item ID 使边键确定。greedy capacity 只在父进程的 `_capacity_assign` 中按全局顺序修改（`:124-151`），capacity 的余数分配也与原实现一致。
- **更新与随机路径通过。** 每轮仍对完整 `rows[labels == j]` 做一次 `mean(axis=0, dtype=float64)`（`:296-297`）；目标距离按块计算后写入一个完整 N 元素 `selected`，最后只调用一次 `sum(dtype=float64)`（`:301-309`），保留原规约形状。初始化用同一 seed 的 `default_rng`，每节点只调用同一 `rng.choice`，然后按升序 child 深度优先递归（`:229-231`, `:247-248`, `:328-331`）。叶 suffix、SIDCatalog 与 audit 仍由原模块完成（`:333-337`）。
- **trace 与缓存约束通过代码检查。** trace 保留每轮 assignment/objective distance、labels、前后 centers、objective/convergence，并输出最终 SID 和 audit（`:312-337`）；全拟合 trace 上界在建目录前预留，后续以 `trace_committed` 逐轮扣减（`:169-180`, `:222-226`, `:318`）。每次拟合需要新 root，排序 run 保留且设只读，既有 root 不覆盖（`:44-45`, `:188-189`, `:227`）。
- **定向证据可接受但范围有限。** notes 记录 workers=1/2/3、prefetch=1/2、边界 case、完整目录/audit/trace/suffix 与四类 mutation；指定实现检查覆盖了 float64、三键 ties、capacity、DFS RNG、bounded futures、cgroup memory 取最小值与 trace 分配前预算。父任务已给出整合后 24 tests 全部通过的日志指针；本次未重跑。它们都是很小的 synthetic 输入，不验证全量运行时、真实 IO、实际 RSS 峰值或磁盘峰值。

## 阻断项

1. **[P2] 并发写入的累计磁盘预留没有生效。** `_guard` 接受 `operation_bytes` 但完全不读取它（`throughput_bhkm.py:110-121`）；提交 block 时传入 `operation_bytes=bytes_written + submitted_disk`，而实际 `projected_disk` 只含单个 reservation（`:264-275`）。因此 bounded futures 同时在写多个 run 时，守卫不会把其他尚未落盘的 reservation 合计进 80% filesystem-used cap 或配置的 `disk_limit_bytes`。初始 whole-fit 上界能覆盖估算的本作业总量，但不能覆盖启动后外部作业占盘或实际写入大于估算时的并发窗口；notes 所称“每个 submitted block reserves its estimated disk bytes before enqueue”目前只是计数，没有参与拒绝判断。建议让每次检查比较 `usage.used + 当前 reservation + 已提交未完成 reservation`，并加 executor 定向测试证明累计预留超过 cap 时在继续提交前拒绝。配置 cap 目前按“filesystem 总 used 上限”计算（`:120`），这与 root 统一后的口径一致。

2. **[P2] 共用配置适配器与原 fit 调用约定不兼容。** 原 `sid.fit` 接受 `on_boundary` callback（`b5a4d1c:src/vkgr/sid.py`），外部实现 `fit` 没有该参数（`throughput_bhkm.py:183-185`）；`fit_configured` 把 `**scientific` 原样转发，因此正式调用若带原有 callback，会在外部开关开启时以 unexpected keyword 失败（`:341-350`）。此外关闭时返回 `(catalog, None)`，开启时返回 `(catalog, trace_or_none)`，与原 `fit` 的 catalog-only 返回值不同；目前全仓没有 `fit_configured` 调用点或对应测试，故调用端是否正确拆包、是否仍提供原边界回调尚无证据。应明确统一返回契约，兼容/有意封装 boundary callback，并给适配器开启与关闭各加一个真实函数级定向测试，再接正式入口。

## 资源限制与未验证项

- worker future 上限由 `workers * prefetch` 限定（`_bounded_submit`, `:49-64`, 调用 `:285-289`），块内存预算显式计入每 edge 工作集、输入和 worker baseline（`:154-168`）；该 worker RSS 是估算项，守卫读取的是父进程 RSS（`:102-107`, `:112-116`），没有观测运行中的子进程树峰值。
- host/cgroup 内存取已知上限最小值（`:71-99`）。trace 估算是预留模型，不是实测峰值；这次没有大规模输入来验证估算偏差。
- 完整 fit 前估算所有 depth、max_iter 的 edge runs 与 header，并按当前 filesystem used 进行 80% 检查（`:215-226`）；未实测完整 N 的 block/run 数、排序 IO、run 留存量、fit-verify/replay 的累计占盘、真实 worker RSS 或 trace 峰值。
- **真实大规模/IO/峰值：NOT_RUN。** 不能把 tiny-array 等价结果解释为 400 GB 可用空间足以完成全量，也不能把 41 GiB 单轮估算解释为整个 fit/replay 可行承诺。工程接入前应保留 STOP 行为；原 runs 与未验收 runs 不得提前清理。
