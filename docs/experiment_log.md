## 2026-10-22：夜间本地准备：SGNS算术与分叉核算

授权依据：942a5b0/edc7f96/ecfcf8c/feb194e 与 Eddy 本轮 goal 指令。今晚未连接任何服务器、未读取任何真实数据/test或checkpoint、未科学训练；旧实例实际Stop尚无回执。本地只有既有A聚合收据的算术与synthetic单测；context UNKNOWN。原预算/spent/deadline不重置。

### SGNS 正对数与条件耗时情景

来源：A汇总 `cache/ur001-preparation-20261020/a2-audit-structure-report-v2.json`，物理SHA `64f7eec73ac04c92b4b1a283e6e6223a8a3b9a2f88e20328b0a0c67e19ced158`；直方图频数和加权行数分别精确等于99,956用户/376,040,151行。只读已授权汇总，不回扫数据。每用户全train曝光序列L，window5保留原user/center/context次序，正对数公式 `2*sum(max(L-d,0), d=1..5)`。本算术不改变SGNS实现。

|项目|值|
|---|---:|
|L=1/2/3/4/5的用户数|9 / 5 / 8 / 3 / 9|
|逐直方图每epoch正对|3,757,403,124|
|假设所有L≥5的简式10R−30U|3,757,402,830|
|短序列修正|+294|
|5 epochs正对|18,787,015,620|
|原batch4096每epoch步数ceil|917,335|
|原batch4096五epoch步数|4,586,675|
|C1子集五epoch正对|173,851,080|
|正对工作量比|108.06384188122387|
|C1原input整项墙钟(s)|2335.1391184329987|
|C1原input_verify整项墙钟(s)|64.40261220932007|
|按正对比线性折算input(h)|70.09558457361224|
|同假设折算input_verify(h)|1.9332204729238172|
|两项合计(h)|72.02880504653605|

C1计数/整项耗时来源本日志「2026-10-17 01:20 S1 progress」；原始收据 main05/sidcoll-progress-20261016T172025Z.stdout SHA `33a533c2a34f2005ee2ed37ebb58a7e8d97ae575f0fd6cdc8619168caf7ca055`。这是包含预扫描、训练、health-check等的**整项成本**，不是纯SGNS训练速度，不把input_verify当梯度/负例逐步full replay。

假设：最终沿全train曝光序列、window5、5epochs、原batch4096、原连续负采样/RNG/学习率；整项耗时严格随正对数线性。真实新机器CPU/GPU、Python枚举/拒绝补抽、全catalog表更新、hash/IO、health-check、内存/缓存变化都可能破坏该假设；**真实ETA=UNKNOWN**，不是排程承诺。情景达到数十小时，按942a5b0回Kiro决定等价提速工程或科学口径调整；本夜不实现、不运行SGNS。

脚本 `analysis/overnight_arithmetic.py`；输出 `cache/overnight-20261022/arithmetic-v1.json`。短L独立枚举单测通过。

### 同原48³规则的分叉算术（不选值）

N=12,012,895，N^(1/3)=229.0248252105078。逐节点k=min(K,len)，capacity商/余数传播；每层occupied/full slots，原≥80%门禁不变。不是fit实测，不能替代H/R门禁。

|K|K³|利用率L1/L2/L3|root dense simultaneously-held lower GiB=32NK/2³⁰|
|---|---:|---|---:|
|228|11,852,352|1 / 1 / 1|81.62677466869354|
|229|12,008,989|1 / 1 / 1|81.98478683829308|
|230|12,167,000|1 / 1 / 0.9873341826251335|82.34279900789261|
|256|16,777,216|1 / 1 / 0.7160243391990662|91.65111541748047|

算术假设原balanced节点；下界不含排序workspace/feature/centroid等，不当实测峰值。K由Kiro冻结，BHKM实现不在本夜范围。合同草稿附同算式供审；无新catalog扫描。

### 本地张量缓存与迁移准备证据

- 新增 `src/vkgr/tensor_cache.py`，G/P共用bounded CPU LRU；原step同code object/private globals绑定，不改旧入口；不缓存P负例/mask/logQ、不改padding/tail/cursor/RNG。真实tensor/参数/梯度/optimizer/loss/RNG parity与短剖析均NOT_RUN。
- 新增测试先确认模块缺失RED：`<LOCAL_TMP>/vkgr-tensor-cache-red-20261022.log`、`<LOCAL_TMP>/vkgr-parity-helper-red.log`、`<LOCAL_TMP>/vkgr-arithmetic-red.log`、`<LOCAL_TMP>/vkgr-migration-red.log`（本机临时日志，公开引用用占位符；仅说明本轮工具已看到RED，文件不入Git）。
- 最终新增suite：18 tests PASS，原week26 worker synthetic回归16 tests PASS；日志 `cache/overnight-20261022/final-local-tests-v1.log` / `week26-regression-v1.log`。这不是科学训练或真实test读取。环境Python3.11.17/macOS，非anchor运行时；不能当新实例验收。
- 隔离变异v1的3项passthrough被Luna发现目标断言未覆盖，不采信为目标变异。v1保留。v2用全新代码副本、python -B、PYTHONDONTWRITEBYTECODE=1、唯一pycache与数据cache前缀，baseline通过、14/14变异由断言杀死；negative/mask/logQ各有明确目标断言。`cache/overnight-20261022/tensor-mutations-v2/receipt.json` 与逐项log保留。
- 一次干净context独立Luna/high C审：`docs/superpowers/reviews/2026-10-22-tensor-cache-luna-c-review.md`。初审CHANGES_REQUIRED，B1已修复并由同一reviewer定点核对收口；parity工具warm closure与P观察proxy缺口也已补synthetic验证和定点复核。生产cache字节未因review修复改变；真实parity/profile仍NOT_RUN，不声称已获科学准入/提速。
- 本机沙箱内最小import torch复现OMP共享内存失败；仅将同一本地synthetic测试通过正常权限流程运行，未更改代码/算法/输入以规避；成功日志已保留。这是本机测试运载，不是服务器入口重验，不计科学名额。
- 迁移CLI `analysis/new_instance_prepare.py`、环境脚本与原anchor lock/pyproject副本已写；只跑dry/拒绝负例和synthetic委托测试，未执行下载、CUDA、过滤或真实capture。原anchor lock SHA `8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063`，当前scaffold lock不替代它。
- 合同草稿发现942a5b0允许同周前缀、但子集history_v2实际validation使用train-only observed。明确记待Kiro，不擅选；冻结前不做科学派生。数据通道原p3c字节/新实例参照闭包仍是明日入口前置。
