## 2026-10-22：可配置吞吐工具链本地交付与A40配置

授权：Eddy七项本地指令及87e5a29/11249b6/2a795cb；截图和decision_log新实例补注用于配置，无本轮远程核验。Codex唯一执行者，context UNKNOWN；本轮不连接实例、不读真实test/holdout、不科学训练、不删旧产物、不改冻结卡/门禁/decision_log；原预算/spent/deadline/名额不重置。各工具是工程API，科学backend仍闭锁。

### 配置与实现

A40依据口径：48GB显存；CPU cgroup 61.2核；规划RAM上限513000000000 bytes；实例盘按500GB保守规划，不用宿主盘。`configs/throughput/a40-20261022.yaml`：assembly/bhkm/week/eval workers各8、prefetch2、BHKM行块16384、mmap+sequence开、decode_workers1、bhkm_external=false、memory cap256000000000、disk cap400000000000、in-flight cap4294967296 bytes。不是实测峰值或并发准入；原锁Python3.11.16/torch2.11.0+cu128不因driver支持13.2而改。本地测试Python3.11.17/macOS，不替代服务器环境。

七项：连续序列/offset +引用、ordered spawn CPU row与原tensor constructor、精确外排BHKM、周并行纯解析/父进程跨周状态、P32/4096 float64 exact块、G只读逐query batch1 session、cgroup资源探测与manifest。开关集中Engineering，默认off/1；工程配置/raw YAML SHA独立记manifest，不改科学身份；Toolchain新API接线，拒绝mmap无sequence组合。新存储/manifest/临时命名空间独占写入，旧文件保留。CPU构造factory须有六项namespace与参数校验，prepared row独立快照；原P负例/mask/logQ与G/P forward/backward/optimizer留主进程。源码身份仍须pinned delta，不以工程config不进科学identity豁免源码。

### 本地验证与失败保留

- 最终回归 `release-suite-v4.log`：54 tests PASS。命令 `PYTHONPATH=src:tests:. PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p 'test_throughput_*.py' -v`（本地venv）。初版release-suite-v1失败因BHKM预算测试错误地假定第一个guard已包含trace；生产已新增较早的输入预算guard，按B档改为在SID/trace分配前核已预留trace，不改变生产代码；v2/v3均通过当时用例，v4为最终集合。
- 原G/P合成step最新 `step-g-release-v2.json/log`、`step-p-release-v2.json/log`：baseline原路径0 worker对1/2/8，tensor bytes、参数、clip后梯度、optimizer state、loss、cursor、Python/NumPy/Torch及负采样RNG全等；保留原train step code object。输入为6事件/3样本，不读数据/旧checkpoint，不报质量指标；不是实际anchor/新实例验收。G多microbatch调用与cursor尾批独立 `tensor-g-multichunk-cursor-v1.log` PASS。
- 全部原始日志在 `cache/throughput-20261022/`，不入公开Git。fresh RED包括tensor-workers-red-v2的缺模块、pipeline-invalid-config-red-v1、weekly-red-test-components、decode-red-runtime-identity；closure-findings-red-v2三个目标断言失败，green-v3三项通过。最早tensor-workers-red.log为语法错误，不作为有效RED；green-v1/沙箱进程失败不算代码通过或科学失败；旧日志均保留。
- 隔离变异最终账本：sequence-release-v1=4、worker-release-v1=3、bhkm-release-v2=4、week-release-v3=2、exact-release-v1=2、decode-release-v2=2、tensor-release-v3=6、probe-release-v1=2，共25项。每组baseline须PASS，各case全新源码副本、python -B、PYTHONDONTWRITEBYTECODE=1、唯一pycache/data缓存前缀。目标覆盖strict-past/顺序/mask/key、RNG/cursor、BHKM tie/capacity/维序/RNG、week授权、exact tie/precision、decoder序/身份、tensor padding/namespace/args/row快照/memory预算、资源建议。tensor_memory由下层具名MemoryError拒绝，worker RNG/cursor及capacity也接受具名failclosed；不把syntax/import失败当kill。旧bhkm-release-v1基线失败、week-release-v1/v2锚点失配和tensor-release-v2分类缺漏均保留并排除最终通过证据。
- 每项独立Luna C初审见tool01–07报告，早期结论保留；后续资源/精确检索/session身份与本次CPU构造接线集中定点收口见 `2026-10-22-throughput-final-fixes-review.md`。派发gpt-6-luna/high、干净context；没有独立runtime型号回执，不伪称模型验证切换。C档修复依据§4A，未改科学代码或规则。

### 限制与下一步

真实anchor逐位对拍、CUDA context-inclusive显存峰值、初始化/大输入RSS、磁盘并发余量和吞吐倍数均NOT_RUN；墙钟ETA UNKNOWN。小fixture一致不证明真实全量已准入。BHKM精确外排所有轮次/节点/replay run累计保留可能超实例盘；不得提前清理，dense同样须真实RSS准入。未知session属性failclosed，运行库monkeypatch不在支持范围；每query身份hash的成本UNKNOWN。工厂/构造器的保守bound与RSS任务边界不是OS瞬态硬隔离。周解析不自行决定history/label/query；全量history合同仍待Kiro。

新实例下一步：先核原job/队列/预算不重投，固定公开SHA部署逐文件核对/原锁及p3c数据通道恢复，真实CLI只读前置，再同机原/新完整step逐位对拍、各工具worker1/2/8小规模工程测峰、资源manifest入run；逐项满足后才改科学backend/并发。科学卡/名额/正式指标另按原授权，不在本轮启动。
