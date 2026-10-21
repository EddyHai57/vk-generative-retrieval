# 吞吐工具链外部参考笔记

查阅日期：2026-10-21（本会话可见日期）
范围：公开网页；未连接服务器，未读取数据或 test，未运行训练；无代码、配置或模型实现。以下把来源的原文主张与本项目推断分开。直接引文：0；结论均为中文转述。

## 项目约束与阅读结论

按 `AGENTS.md` §7 与 `docs/decision_log.md` 末尾工具链裁决，本笔记只整理工程候选，不冻结实现。并行只能改变墙钟，`workers=1/2/N` 的输出要逐位相同；随机数仍由主进程按原序消耗，worker 只执行给定下标的确定性计算。BHKM 的边序、float64 累加次序、单一 greedy capacity state、centroid 更新次序与 DFS RNG 请求序均是既定约束。近似或 GPU k-means 不替代 BHKM；HSTU 论文和实现只帮助理解 jagged/offset 表示，不授权修改模型。

| 来源 / 实际访问状态 | 来源主张（一句话） | 可借鉴点与本项目适用性（本项目推断） | 许可证判断：论文 / 文本 / 代码 |
|---|---|---|---|
| [Spheron：更快的数据加载器](https://www.spheron.network/blog/we-thought-we-needed-more-gpus-we-needed-a-faster-dataloader/)；页面可读，文章标注 2026-09-25 发布、2026-10-19 更新 | 作者描述单核 CPU 饱和、GPU 周期性空闲时，瓶颈可能在 CPU 侧 batch 准备，并展示多 worker、预取和 pinned memory 的方案。 | 可借鉴按单核与 GPU 利用率定位装配瓶颈、再调 worker / 预取深度；与本项目已有 profiling 相符。文章面向自身训练栈的建议不能直接作为 worker 数配置，且 pinned memory 影响传输路径，需单独验证逐位一致与内存占用。 | 论文：不适用。文本：页面未见可明确复用的开放许可，UNKNOWN。代码：文章代码段无单独许可声明，UNKNOWN；只转述，不复制。 |
| [PyTorch 论坛：Pytorch Model Results](https://discuss.pytorch.org/t/pytorch-model-results/151454)；页面可读 | 论坛回复建议给 DataLoader 配置 generator 与 `worker_init_fn`，并为其他随机库设置 worker seed，以复现每次运行行为。 | 这是讨论帖，不是规范；官方文档更权威。对项目而言应更进一步，让 worker 不消耗 RNG，只按主进程下发的索引确定性装配，避免依赖 worker 随机流和完成顺序。 | 论文：不适用。文本：论坛页面未查到该帖的明确许可声明，UNKNOWN。代码：帖内示例无单独代码许可，UNKNOWN；不复制示例。 |
| [Wikipedia：External sorting](https://en.wikipedia.org/wiki/External_sorting)；页面可读 | 文章介绍将可装入内存的块分别排序写盘，再对有序 runs 做 k-way merge 的外部归并排序。 | 可借鉴“有限内存生成有序 runs、顺序读回归并”的一般算法结构；与本项目 BHKM 的分块边排序和全局有序合并直接相关。该百科页面只支持算法概念，具体 tie-break、容量状态和浮点次序必须服从项目冻结约束。 | 论文：不适用。文本：页面声明 CC BY-SA 4.0。代码：页面无可引入实现；不复制。 |
| [sortednp 仓库](https://github.com/pombredanne/sauerburger.com-frank-sortednp)；README 与 LICENSE 可读 | README 说明它针对已排序 NumPy 数组做合并 / 交集，支持多路处理和按需加载大型数组。 | 可借鉴流式消费多个有序 run 的接口思路；本项目边记录还需要稳定的 `(distance,item_id,child)` 全序与单一 greedy state，不能将 union/intersection 实现视作 BHKM 分配器。只参考概念，不引入依赖或代码。 | 论文：不适用。文本：README 未发现单独文本许可，UNKNOWN。代码：仓库 `LICENSE` 为 MIT；许可允许使用但当前不引入代码。 |
| [MTGRBoost](https://arxiv.org/html/2505.12663v1)；论文 HTML 可读 | 论文描述分片数据并行读取、预取后续 batch，以及用 copy / dispatch / compute streams 重叠加载与计算；另报告 sequence balancing 在其多 GPU 设置下提高吞吐。 | 可借鉴把读取、装配、传输与计算视作 pipeline 阶段并测量 overlap；与本项目 CPU 装配和预取候选有关。论文的多机多 GPU 序列均衡机制与本项目资源、模型及逐位约束差异很大，不能直接移植或由其性能数值推算本项目收益。 | 论文：arXiv 页面列出 arXiv.org perpetual non-exclusive license（论文发布许可，不是代码许可）。文本：同论文许可信息。代码：此论文页未核验对应实现仓库的代码许可，UNKNOWN。 |
| [TurboGR](https://arxiv.org/html/2605.13433v1)；论文 HTML 可读 | 论文称长尾序列产生 jagged tensor 与 padding 冗余，并提出 jagged fusion、lookup 与动态负载均衡来处理变长输入。 | 可借鉴变长序列用扁平值数组加 offsets / lengths 表达、避免把 padding 物化为冗余；这与项目已授权的“每用户连续序列 + 偏移”存储方向相通。论文结果来自 Ascend 集群，且含算子与模型系统改动；不据此实现新模型或算子。 | 论文：arXiv 页面列出 arXiv.org perpetual non-exclusive license。文本：同论文许可信息。代码：论文页未核验代码仓库许可，UNKNOWN。 |
| [PipeRec / Accelerating Recommender Model ETL](https://arxiv.org/html/2501.12032v3)；论文 HTML 可读 | 论文以 FPGA–GPU streaming dataflow 缓解推荐 ETL 成为训练瓶颈的问题，并报告其硬件方案的吞吐结果。 | 可借鉴“端到端按阶段计时，识别 CPU ETL 与模型计算间空档”的诊断框架；本项目能用作流水化动机。FPGA、P2P DMA 与项目单机 PyTorch 批装配不等价，论文加速倍数不适用本项目。 | 论文：arXiv 页面列出 arXiv.org perpetual non-exclusive license。文本：同论文许可信息。代码：论文页未核验对应实现许可，UNKNOWN。 |
| [flash-kmeans](https://github.com/svg-project/flash-kmeans)；README 与 `LICENSE.txt` 可读 | 仓库说明其为 Triton GPU 上的 IO-aware batched K-means，实现含分块 / 双缓冲数据路径。 | 仅可作 GPU 聚类吞吐与内存管理的对照阅读。它不实现本项目 BHKM 的容量平衡 greedy assignment，而且 GPU / dtype / reduction 路径与冻结的 float64 逐维累加不同，因此不等价、不采用、不借代码。 | 论文：仓库链接论文，但本笔记未审论文许可，UNKNOWN。文本：README 未确认独立文本许可，UNKNOWN。代码：仓库 `LICENSE.txt` 为 Apache-2.0；许可证许可使用不改变算法不适用结论。 |
| [kmcuda](https://github.com/src-d/kmcuda)；README 与 `LICENSE.md` 可读 | 仓库提供 CUDA k-means / k-NN，介绍 Yinyang 加速、低显存路径，以及部分路径的逐位一致测试。 | 可借鉴其把聚类速度、内存占用与近邻剪枝单独报告的工程写法；但算法目标、浮点路径、中心更新和容量约束均不等于 BHKM，GPU k-means 不采用。 | 论文：仓库提到 Yinyang 论文，本笔记未查论文许可，UNKNOWN。文本：README 未确认独立文本许可，UNKNOWN。代码：仓库 `LICENSE.md` 为 Apache-2.0；不引入代码。 |
| [generative-recommenders / HSTU](https://github.com/facebookresearch/generative-recommenders)；跳转到 `meta-recsys`，README 与 LICENSE 可读 | 仓库托管 HSTU 等生成式推荐参考实现，并公开 Apache-2.0 LICENSE。 | 可研究变长序列和 jagged 数据接口如何进入 HSTU 输入；对本项目当前只支持连续序列与 offset 数据布局的工程讨论。改变模型结构、算子或 HSTU 训练方式不在本任务授权内。 | 论文：仓库关联的论文许可未查，UNKNOWN。文本：README 无独立文本许可声明，UNKNOWN。代码：仓库 `LICENSE` 为 Apache-2.0；本次不复制代码。 |
| [semantic-ids](https://github.com/clabrugere/semantic-ids)；仓库页、README 与 GitHub 的 Apache-2.0 许可声明可读；直读 `main/LICENSE` 返回 404，故以仓库页声明为准 | README 描述用 RQ-VAE 将 dense embedding 编成多级 semantic IDs，并在导出时处理前缀碰撞。 | 可借鉴“扁平编码序列 + 层级 / 碰撞消歧”的表达方式作概念参照；它是学习式 residual quantization，不是本项目冻结的容量平衡 BHKM，也不能改写 SID 规则。 | 论文：未见论文许可声明，UNKNOWN。文本：README 未确认独立文本许可，UNKNOWN。代码：GitHub 仓库页声明 Apache-2.0；直读 LICENSE 链接失败（404），故仅记为官方仓库声明，不视作已复核许可证文本；不引入代码。 |

## 补充的官方 API 资料

| 来源 / 实际访问状态 | 官方文档主张（一句话） | 可借鉴点与本项目适用性（本项目推断） | 许可证判断：文本 / 代码 |
|---|---|---|---|
| [PyTorch 2.11 DataLoader 文档](https://docs.pytorch.org/docs/2.11/data.html)；可读 | map-style 数据集的索引由主进程 sampler 生成再发给 worker；worker 有自己的 seed，其他库可能重复 seed，`worker_init_fn` 可初始化；pinned memory 可加快 host→GPU 拷贝。 | 直接支持“主进程决定样本索引，worker 做确定性读取 / 装配”的架构。项目约束比常见 seed-worker 方案更严格：worker 不随机抽样，结果按步号有序归还；预取深度与 pinned memory 的收益需在不改变 batch 数值的前提下核验。 | 文档文本 / 示例代码：PyTorch 源码仓库 LICENSE 是 BSD-style 许可；文档站点内容许可未单独核验，故文本许可 UNKNOWN。只转述 API，不复制代码。 |
| [PyTorch 2.11 Reproducibility 文档](https://docs.pytorch.org/docs/2.11/notes/randomness.html)；可读 | 官方给出的 DataLoader 可复现示例将固定 generator 与 `worker_init_fn` 结合，为 NumPy 与 Python random 设置 worker seed。 | 用于解释论坛建议的官方依据，并提醒 generator 本身会消耗主 RNG；本项目应保持既有 RNG 请求次序，将随机决定留在主进程，worker 只按输入索引计算。 | 文档文本 / 示例代码：PyTorch 源码仓库 LICENSE 是 BSD-style 许可；站点文本许可未单独核验，UNKNOWN；不复制示例代码。 |
| [NumPy `lexsort`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html)；可读 | `lexsort` 是稳定多键间接排序，最后一组 key 为主键，前面的 key 依次打破平局。 | 可用作原全局排序语义的 API 基准与对照；分块 run 必须保留完全相同的键次序与 tie-break，单靠每块内部排序并不能替代全局 merge。 | 文档文本许可未单独核验，UNKNOWN；NumPy 源码仓库 `LICENSE.txt` 为 BSD 3-Clause 类许可。只引用 API 名称，不复制实现。 |
| [NumPy `argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html)；可读 | `argpartition` 只保证第 k 位落在排序后位置，其两侧元素各自未排序；并列项的次序不稳定。 | 对精确 top-k 分块，分块取候选后仍需按完整 `(score, ordinal)` 键排序并归并；必须保留边界分数上的所有并列候选，否则无法保证全局精确、稳定的 top-k。此为由 API 性质推出的工程要求，尚非本项目实现授权。 | 文档文本许可未单独核验，UNKNOWN；NumPy 源码仓库 `LICENSE.txt` 为 BSD 3-Clause 类许可。未引入第三方代码。 |
| [NumPy `einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)；可读 | `einsum` 描述张量收缩，并允许选择 contraction path；文档指出 `optimize` 会影响中间张量的计算路径。 | 可作为分块精确打分的算子候选，但不同收缩路径可能改变舍入与归约次序；项目冻结要求逐 query 分数逐位相同，因此保留既定 float64 调用与块形状，任何改式必须先按冻结规则验收。 | 文档文本许可未单独核验，UNKNOWN；NumPy 源码仓库 `LICENSE.txt` 为 BSD 3-Clause 类许可。没有复制代码。 |
| [Python 3.11 `heapq`](https://docs.python.org/3.11/library/heapq.html)；可读 | `heapq.merge` 从多个已排序输入生成有序迭代器，不把所有输入一次装入内存。 | 可作为 k-way merge 的标准库接口参考；需要把固定 tie-break 键完整写入每个 run 记录，并用回归对拍确认与原 `lexsort` 全序一致。Python 3.11 源码仓库许可证为 PSF License。 | 文档文本许可未单独核验，UNKNOWN；Python 3.11 LICENSE 为 PSF License。只用标准库 API 说明，不复制实现。 |

## 核心结论与限制

- **CPU 装配**：最有用的共识来自 PyTorch 官方 DataLoader 语义与 Spheron 的工程观察：索引 / 采样与样本准备可分开，准备工作可并行预取。项目实现仍以主进程定序、worker 确定性装配、按原步序交付为准；外部文章没有证明本项目的提速比例。
- **外排与 k-way**：External sorting、`heapq.merge` 和 sortednp 都支持分块有序 run 的流式归并思路。sortednp 本身面向 sorted-array union/intersection，不能代替项目的边排序或 capacity assignment。
- **容量平衡聚类**：flash-kmeans 与 kmcuda 可用于理解 GPU 聚类工程优化，但算法或算术路径不同；明确不采用 GPU k-means 替换 BHKM。项目应按既定外排方案保留全局三键排序和单一 greedy capacity state。
- **HSTU / jagged 表示**：MTGRBoost / TurboGR 可支持变长序列减少 padding、将搬运与计算重叠的工程动机；项目可把扁平连续事件与 offsets 作为数据表示候选。任何 HSTU 模型、attention、负采样或算子变化都超出本笔记范围。
- **精确 top-k**：`argpartition` 适合筛候选但不提供稳定次序；精确分块 top-k 必须处理分数边界并列，再以冻结 tie-break 键合并。`einsum` 或其他向量化公式可能更快，但运算次序变化对位级数值的影响必须先由已有合同 / 冻结卡裁定。
- **许可证**：MIT、Apache-2.0、BSD 与 PSF 的判断仅覆盖本页实际查看到的仓库许可证或官方许可声明。论文的 arXiv 上传许可不代表论文代码许可；`UNKNOWN` 不构成复用授权。除标准库 / NumPy / PyTorch 现有依赖外，本轮没有引入任何第三方代码、依赖或文本段落。

## 访问记录与来源计数

指定的 11 个来源均已尝试并可读取页面正文；其中 semantic-ids 的 raw `main/LICENSE` URL 返回 404，但 GitHub 仓库页面显示 Apache-2.0 声明，已在表内注明这一差异。补充查阅 6 个官方 API 页面（PyTorch 2、NumPy 3、Python 1），共 **17 个来源页面 / URL**。页面均在本会话通过公开网页访问；未访问源代码文件内容，仅查许可证文件 / LICENSE 页面，不运行或下载第三方代码。

本项目 exact top-k 的冻结 tie-break 是分数降序、item-ID 升序；query ordinal 只用于并行结果回收，不替代 item-ID tie-break。
