# 本地吞吐工具链交付（2026-10-22）

Codex唯一执行者；裁决者Claude Code（Kiro角色）；context UNKNOWN。范围为Eddy七项本地授权，依据87e5a29/11249b6/2a795cb。实现提交 `d3f37bf`；随后仅更新交付/状态文档。公开压平工作分支与SHA另见本报告「发布回执」。本轮未连接计算实例、不读真实test/holdout、不科学训练、不删旧产物、不改decision_log/冻结卡/门禁，不重置原预算/spent/deadline/名额。

## 交付与状态

- 调研：`docs/superpowers/research/2026-10-22-throughput-references.md`，先查用户给定资源，逐项结论/适用性/许可证；未引入第三方代码。
- 方案：`docs/superpowers/specs/2026-10-22-throughput-options.md`；执行账本：`docs/superpowers/plans/2026-10-22-throughput-implementation.md`。
- 工具1：连续用户事件+offset、user/target位置引用、新目录mmap；完整prefix likes与strict-past顺序保留。列式生产接口/合成适配器均有测试。
- 工具2：spawn有界ordered批装配，原CPU张量constructor在worker执行，原G microbatch/P width保留；原step code object不变。六项namespace、不可变row快照、构造参数、RNG/cursor与预算拒绝均有定向验证。P负例/mask/logQ和模型step仍在主进程。
- 工具3：精确float64逐维距离、三键run排序+k-way merge、单线程greedy/原mean/DFS RNG；原小样本全节点trace、callback/返回契约对拍通过。临时run保留；整项容量未实测。
- 工具4：仅周内纯解析并行，父进程按周拼接并调用原跨周stateful builder；week26/holdout/test组件输入拒绝。
- 工具5：32-query/4096-item原块形状、float64 einsum、原item-ID tie与ordinal；共享float32完整mmap，分数trace/结果逐位对拍。
- 工具6：G逐query、batch1、owner与factory身份匹配；TorchSession绑定模型实例/类属性、权重与buffer/device/拓扑/hooks、codec和runtime；未知状态拒绝。CUDA并行须context-inclusive测峰回执；本轮只CPU合成，不启A40多进程G。
- 工具7：本地cgroup/CPU/RAM/盘/GPU与driver探测，建议不自动应用；实际选用YAML/raw SHA独立进工程manifest。
- 新Toolchain接通统一工程配置；四份YAML：disabled、最低、推荐、A40。默认off/1；科学identity不加入工程配置，源码仍走pinned delta核验。没有修改旧冻结正式入口，科学backend继续闭锁。

## A40 配置

使用 `configs/throughput/a40-20261022.yaml`；资源依据是用户截图和decision_log实测补注，本轮没有远程核规格。CPU按cgroup，RAM按租用规划上限，磁盘按实例分配，具体数字唯一experiment_log本轮工具链节。装配/周处理/P评估与可选BHKM行块先8 workers，prefetch2；sequence+mmap开；G decode1；BHKM外排先关，dense候选须真实RSS前置。若需外排只改配置，但先核累计轮次/节点/replay的全部保留盘量，超限STOP，不能清run。配置是初始建议，不是实测并发准入。

## 验证与review

唯一完整数表/限制：`docs/experiment_log.md`「可配置吞吐工具链本地交付与A40配置」。原始日志均在gitignored `cache/throughput-20261022/`。

- 最终suite `release-suite-v4.log`：54 PASS。
- 最新原G/P合成完整step `step-g-release-v2.json` / `step-p-release-v2.json`：baseline0与workers1/2/8逐位相同，比较tensor、参数、clip后梯度、optimizer state、loss、cursor与所有已初始化RNG；G多chunk/cursor尾批另有测试。harness在内存比较完整trace，成功receipt仅保存布尔摘要，未持久完整逐项trace；不当真实anchor或GPU证明。
- 隔离变异25项通过：序列4、worker3、BHKM4、周2、exact2、decode2、tensor6、probe2；每case全新源码拷贝、python -B、PYTHONDONTWRITEBYTECODE=1、唯一pycache/data cache前缀。每组baseline通过。只认断言或具名语义拒绝，不认syntax/import错误kill。
- 首版tensor RED语法错误排除；修正后模块缺失RED有效。部分测试在沙箱内受OpenMP共享内存/ps限制，只对同一本地合成命令经正常权限流程运行后通过，未改算法/门禁避错。旧失败日志全部保留。
- BHKM预算测试观察点曾错误假定第一个guard包含trace；按B档改为断言SID/trace分配前已预留，生产逻辑不改。变异week锚点更新与tensor_memory语义拒绝分类缺漏已修，旧批次排除。
- 各项独立C初审：`docs/superpowers/reviews/2026-10-22-tool01-luna-c-review.md`至tool07；初审失败结论保留。集中修复收口：`2026-10-22-throughput-final-fixes-review.md`。全部按gpt-6-luna/high、干净context派发；没有独立runtime型号回执，不伪称模型已验证切换。

## 未完成项与待Kiro

1. 真实anchor与CUDA逐位对拍、吞吐、初始化/大输入RSS、context-inclusive显存、临时盘峰值及完整forecast：NOT_RUN；遵守本地任务边界，不给提速倍数或ETA。
2. 全量validation/test history差异：沿原STATE待裁决，不生成科学派生。
3. BHKM累计保留容量/磁盘策略：真实前置不足即STOP；任何删除/迁移/非等价算法另裁决。
4. G解码新实例参照选择：沿既有待裁决项，不新增训练。未知session/构造器状态failclosed，不支持对锁定torch/numpy运行库monkeypatch。
5. 这些新模块是工程API；科学backend集成/正式作业仍须原卡、身份与真实入口前置，本轮不放行科学训练。

## 新实例逐步验收（本轮不执行）

1. 先核已有job/source/queue/spent/deadline；失联先查原job，禁止重投/重置。按主线原收据续办数据重建，不重复A审计或扩大holdout授权。
2. 只读deploy key fetch公开固定SHA，新部署目录逐文件SHA核对；恢复原Python3.11.16/原uv.lock依赖及原p3c数据通道，CUDA小算子smoke。
3. 用正式CLI只读preflight；资源探测示例（OUTPUT必须新路径）：
   `PYTHONPATH=src python -B -m analysis.throughput_probe probe --directory <WORK_ROOT> --config configs/throughput/a40-20261022.yaml --output <NEW_RESOURCE_MANIFEST>`
   dry用 `python -B -m analysis.throughput_probe dry`；此CLI不训练、不推理、不读取数据。
4. 同机同锁新旧原G/P真实anchor工程对拍，tensor/参数/梯度/optimizer/loss/cursor/全部RNG全等；P负例/mask/logQ输入逐项相同。真实路径依正式launcher与数据通道恢复，合成不能替代。
5. 各工具worker1/2/8小规模真实工程对拍；BHKM全节点trace与完整capacity/tie/RNG、P逐分数和ordinal、G逐query/batch1。只用授权输入；科学派生须先冻结合同。
6. 测包含context的GPU峰值、进程树RSS、初始化瞬态、CPU/IO/磁盘累计保留；守原60%RSS/80%盘线与更紧YAML cap。声明按实测×1.15向上ceil256MiB；不够降低工程workers/预取或STOP，不改科学batch/精度/采样。
7. run manifest记录选用工程YAML及SHA。按实测决定dense/外排与并发，完成正式入口端到端/真实launcher链及独立作业前review后，才按获批科学卡入队。未过任一项不能称全量可训练。
