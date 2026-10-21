# 可配置逐位等价工具链实施计划

> 执行：superpowers:executing-plans；TDD、隔离变异、每项独立 Luna C 审。Eddy 已授权本地持续执行，不逐步等待确认。

依据：decision_log 87e5a29 / 11249b6 / 2a795cb，及本轮明确七项任务。全程不连接计算实例、不训练真实模型、不读test、不删除旧产物、不改冻结入口/合同；研究只中文转述，不复制第三方代码。context UNKNOWN。

1. 已完成Luna干净context调研，先整合参考笔记，再写 options；实现前核对原源码运算与资源边界。
2. 共用配置 `configs/throughput/*.yaml` 与新 `src/vkgr/throughput_config.py`：默认off/1，工程配置SHA单独入manifest，科学身份对象不改；资源建议不自动覆盖科学配置。
3. 工具1 `throughput_sequences.py`：用户连续事件列/offset + user/target引用；只恢复已冻结target，不自行选query/label。新目录可选mmap。合成fixture与旧行逐字段/tensor/step状态对拍。
4. 工具2 `throughput_workers.py`：主进程固定任务序/随机输入，spawn纯函数worker，bounded prefetch，ordered返回；默认直接原回调，worker禁止PRNG消耗。
5. 工具3 `throughput_bhkm.py`：在1/2之后实现；float64逐维距离、全局distance/item-ID/child序、单线程greedy、原mean/DFS；固定原fit测试全节点trace。临时文件独占保留，fit/replay前不删；多轮保留量进预算。
6. 工具4 `throughput_weekly.py`：只并行每周确定性解析/事件分段，父进程按周串接，再由既有合同builder依次处理跨周状态；不声称周内独立history能自动等价。
7. 工具5 `throughput_exact.py`：按原32条query块调用原4096item-block/float64路径，结果按ordinal回收；原item-ID tie-break不变。
8. 工具6 `throughput_decode.py`：spawn每worker独立只读模型，仍逐query batch1；相同状态身份，query内不消费RNG，GPU内存上界必须覆盖全部worker；CPU synthetic验证、真实CUDA留新实例。
9. 工具7 `throughput_probe.py`：本机read-only硬件/cgroup探测、manifest与配置建议；GPU缺失UNKNOWN；无网络/远端操作。
10. 每项先目标测试fresh RED，最小实现GREEN，workers1/2/3或4回归；独立新拷贝、python -B、PYTHONDONTWRITEBYTECODE、唯一缓存前缀变异；一次Luna C审，普通补证按§4A，必要修复定点核对。
11. 明确记录CPU小样本通过与真实G/P/CUDA大规模NOT_RUN的区别；未证明的科学变更只列待Kiro。完成后同步STATE/handoff/map/log，具体文件commit、公开压平包整体扫描后只push工作分支，交付 `.agents/tasks/throughput-report-20261022.md`。


## 执行账本收口

工具1–7的新工程API与集中配置均已实现；具体初审、修复、54项最终回归、25项隔离变异及原G/P合成step对拍收据见experiment_log「可配置吞吐工具链本地交付与A40配置」。CPU构造补齐为真实worker执行，不再只传row列表；构造参数/六项namespace、prepared row不可变快照及memory-only预算负例均补齐。独立C收口报告2026-10-22-throughput-final-fixes-review.md；初审未通过记录保留。真实资源/anchor/性能NOT_RUN。无科学设计代选，无删除/服务器操作；原合同history与累计外排容量列待Kiro。完成不意味着科学backend可运行。
