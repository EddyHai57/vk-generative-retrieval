# 可配置吞吐优化选项

依据：87e5a29、11249b6；不改科学合同、门禁、精度、batch或采样。研究来源见 `../research/2026-10-22-throughput-references.md`。本表收益均为工程推断；真实墙钟/内存/显存/临时盘需新实例测量，不能把worker数写成已测加速倍数。

## 证据与候选

唯一数表见experiment_log「B两路短剖析全部验收与C资源情景」「夜间本地准备：SGNS算术与分叉核算」。按用户要求在方案中引用关键读数：G tensor_build 523.883912ms/诊断步、P 187.239138ms；G主计时0.9531615324s/步、P0.1911873362s/步。hook嵌套/异步，不可相加或直接换算最大提速。G解码成本与P全量评价墙钟均UNKNOWN；P评价乘加规模约2.1e13，仅算术。BHKM原root下界约82GiB；SGNS整项线性情景约72h，不是真实ETA。

|工具/瓶颈|可借鉴点与收益推断|逐位证明/验收|集中配置键（默认）|资源依赖与最低/推荐建议|
|---|---|---|---|---|
|1 用户连续序列+offset|避免为每个target重复history；存储从O(samples×L)变O(events+samples)，长序列收益可能为数量级，CPU切片收益待测|target引用不变，strict-past切片、past_likes/side/ctx顺序、dtype/bytes相等；cursor/RNG不参与存储|sequence_enabled=false；use_mmap=false|64GB建议mmap、内存byte cap；128GB仍可mmap，不常驻物化历史；默认off调用原rows|
|2 spawn批装配|包含原CPU tensor constructor，保持原G microbatch切分/P动态width；对CPU装配瓶颈，理想并行上界≤workers倍，序列化/IO抵消后可能无收益|主进程先决定indices和随机结果；worker不随机；按step ordinal归还；参数/梯度/optimizer/loss/RNG最终逐位|assembly_workers=1；prefetch=1|最低2worker/prefetch2；推荐4/2；限in-flight bytes及RSS，不盲目吃满核|
|3 BHKM精确外排|降低dense root与排序常驻内存；会增加IO，可能更慢；多worker距离/run-sort理想上界≤workers|三键全局边序、64维升序float64、单一capacity、原mean、DFS RNG与全节点trace|bhkm_external=false；bhkm_workers=1；bhkm_rows=8192；temp_dir=null|最低2worker/2048rows；推荐4/8192；每轮约16NK字节边run，**多轮/节点/replay累计留存**须另计，不能仅用41GiB承诺整项够盘|
|4 按周并行解析|并行列读取/转换，理想上界≤week_workers；跨周history仍串行|周内physical ordinal不变，父进程按周序回收；后续调用原跨周stateful builder|week_workers=1|最低2、推荐4；各周输出量×prefetch计入RSS/磁盘；科学派生仍待合同冻结|
|5 P query块并行|精确检索的独立query块并行，理想上界≤eval_workers；Q×N计算量不变|原32/4096块形状、np.einsum(dtype=float64,optimize=False)、原item-ID ties；逐query分数/预测同序|eval_workers=1|最低2、推荐4；大catalog用共享mmap，禁止每worker复制全表；CPU同依赖/线程环境；GPU公式替换不采用|
|6 G逐query多进程|batch1单query互相独立，理想上界≤decode_workers；GPU争用可能无收益|每worker同只读模型身份、原batch1函数/beam顺序；query内RNG不变；ordinal合并|decode_workers=1|最低1、推荐2（只建议）；多份模型/context实测峰值合计入准入，显存不足保持1，不降batch|
|7 资源探测|按实测核数/cgroup/RAM/盘/GPU给建议，避免机器变化改代码|读取不改科学输入；配置内容与SHA入工程manifest，原科学身份完全保留|memory_limit_bytes=0；disk_limit_bytes=0；max_in_flight_bytes=0|启用项同时受原60%RSS/80%磁盘硬线约束；配置byte cap只能收紧；探测不自动付费或修改配置|

配置示例：`configs/throughput/disabled.yaml`、`minimum.yaml`（12核/64GB/24GB）、`recommended.yaml`（16+核/128GB/48GB）。所有开关同一schema；示例为建议，未测峰前不称已准入。机器不满足推荐值只改YAML；workers>1并非默认。

工程config及SHA放run manifest独立`engineering`字段，**不加入科学config/seed/input身份比较**；优化源码仍须按§4A pinned delta完整核对，不能豁免源码身份。未来科学backend接入须同正式CLI/preflight/真实launcher链，当前工具不解除backend闭锁。

## 不采用（科学变更或未获等价证明）

混合精度/TF32、改effective batch/microbatch、负采样/RNG重排、改变history/query/评价分母、ANN/近似top-k、近似或GPU k-means、GEMM距离替代float64逐维加法、分块partial-sum替代原centroid mean、节点并行改变DFS RNG。HSTU模型/算子、jagged attention、MTGRBoost/TurboGR的模型系统变更不在本轮；只参考数据布局，不复制第三方代码。

## 交 Kiro 裁决 / 限制

- SGNS的向量化负采样或GPU改式可能有大收益，但拒绝补抽顺序/学习率/浮点路径可能改变；仅保留既有数十小时成本情景，不实现。
- BHKM所有轮次run需留到fit-verify与同seed replay通过；单轮41GiB不能代表整项。容量不足时STOP并保留，不提前删旧或本次未验收run；scratch生命周期若需调整另裁决。
- 周并行只能独立解析/无状态转换；任何带跨周历史的科学构造仍按原顺序或显式prefix state处理，合同未冻结不生成。
- 真机CUDA G解码/完整anchor参数梯度对拍、大规模磁盘与性能均NOT_RUN；本地synthetic证据不能声称全量已验收。

## 已选 A40 的初始配置

`configs/throughput/a40-20261022.yaml` 按 Eddy 截图与 decision_log 实测补注设定：CPU 按 cgroup 配额而非宿主线程，RAM 按租用规划上限而非宿主/cgroup 可见大值，磁盘按实例分配而非宿主盘。具体资源数字只见 experiment_log 本轮工具链节。

装配、周解析、P query 块及可选 BHKM 行块各先用 8 worker、prefetch 2；mmap 连续序列开启。G decode 保持 1，未测 context-inclusive 多进程峰值不启并行。BHKM 外排默认关闭：大内存实例可先审原 dense 的保守内存模型与实测 RSS；不够则按同 YAML 开启外排，先核全部轮次保留磁盘。不是已测提速或整项容量通过。

CPU tensor worker 的 factory 必须钉输入/SID/config/source/runtime/constructor namespace，初始化与任务不能消耗 RNG，validate_call 逐次校验原构造器的不可变非 row 参数；PreparedBatch 只替换原 step 私有 globals 内的构造函数，原 code object、负例/mask/logQ/forward/backward/optimizer 仍留主进程。开启预算时须提供保守 tensor output byte bound；已运行任务的瞬时分配不能由前后 RSS 检查硬保证，真机峰值仍为前置。

默认 disabled.yaml 保持 off/1。use_mmap=true 同时关闭连续序列的矛盾组合在新 Toolchain 初始化时拒绝，不生成误导 manifest。所有工具是新工程 API，旧冻结正式入口字节不改；未来 backend 接入仍须按正式 CLI 与真实输入准入。
