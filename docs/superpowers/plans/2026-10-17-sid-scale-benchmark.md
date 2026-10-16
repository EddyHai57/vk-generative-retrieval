# SID 全量规模基准（S4）本地工程计划

2026-10-17；合成数据工程成本，不做科学判定。只新增本计划、scripts/sid_scale_benchmark.py、tests/test_sid_scale_benchmark.py；不改原SID，不碰服务器/S1。实现后交独立review，不自动执行全量基准。

接口：synthetic_blocks(n,block=2048,seed=42)逐批产float32 [B,64]单位向量；allocation_estimate报已知数组保守下界（非实测）；suffix_summary仅统计完整拟合语义码；streaming_fit单pass在线根节点。CLI默认N1250000、12470000，BHKM256×3、RQ-VAE K256原PREREG300000steps、streaming分别独立subprocess，JSON stdout。无真实data/test/checkpoint读取或模型落盘。

CPU单线程：BLAS环境、torch intra/inter-op均1。每方法wall≤60秒、RSS≤4096MiB；父进程ps采样子进程RSS、超限终止进程组。采样峰值与worker ru_maxrss分开，采样间隔内短暂超限无法绝对排除。完整BHKM根N×256边数组与物化RQ-VAE先估显式已知数组下界，超过cap直接SKIPPED_RESOURCE不分配；超时CENSORED_WALL、超RSS CENSORED_RSS。完整fit未完suffix=UNKNOWN，不能用partial耗时外推完整成本。

streaming用先256项初始化、逐批生成/分配/更新累计中心，初始化项计入assigned_items，单pass不保存全量。只报root cluster_counts/finite/assigned_items；非balanced、不产完整SID，suffix恒UNKNOWN，不能替代生产SID。

可选--output-dir只新建目录/独占创建文件，不覆盖删除。验收：5原测试fresh RED缺module；追加worker隔离/线程/cap/skip/censor测试，RED→GREEN，再test_sid_coll*.py回归。测试只用小synthetic；全量运行必须review后owner执行，数字只由owner写experiment_log。

CLI合同修正（独立review MEDIUM）：controller只接受--output-dir与--wall-seconds（0<wall≤60，逐方法传递）；固定两个规模/256叉/block2048/seed42。--n/--k/--block/--seed仅worker内部模式可用，controller显式收到任一项立即parser.error，禁止静默忽略参数后执行全量。测试main时mock六次run_method核wall0.01传递，不实际执行规模任务；逐项拒绝worker-only参数与非法wall。
