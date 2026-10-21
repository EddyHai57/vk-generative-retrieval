# P PF cache 只读复用：实现与复核包

范围：仅新增 `analysis/p_pf_cache_reuse.py`、对应测试与本文。现有 backend、compact 与 detached driver 不变；无 SSH、push、部署、入队、信号、训练、test 或 checkpoint/向量载入。此提交没有切换现行 consumer，也不修改 task passes。

## 证明边界与 API

当前成功 PF 未记录 cache SHA，故禁止事后 hash 冒充原 PF 完成时点的字节证明。新独立 `seal` 是一次当前 cache 的只读语义重验：固定 JSONL 的每行真实 byte boundary/count/SHA、history 物理引用、side item/state/shape、原 train parquet 顺序及原 `_encode` 重算、未引用 slot 的全零字节、闭合文件集和前后漂移。只证明 `current_cache_revalidated_not_historical_PF_bytes`。

原 PF 固定 `18f98f56fc7211d549eaf296015f256d81691c5b`；原 deployment 子进程执行原 `read_transport`、`stage_inputs`、`job_contract`、`close_context` 与 DAG producer 核验，拒绝错误 stdout/terminal/job/source/runtime/p3c/input 绑定。新 reader ledger 必须恰等原 ledger 加本文、模块、测试三个精确 SHA；runtime 逐字相同。seal 有独立 run/job/input/request/manifest/stdout/terminal 闭包；不能用 caller `verified=true` 代替。

`consume(reference, guard)` 返回 `rows` / `queries`（原 `SideRows` / `JsonRows` / `DenseSide` 类型及未改方法，mmap mode=r）、`scientific_binding`（原 PF 原样）、`reader_identity`（新 source/runtime/delta 与两级 producer pins）。二者明确分开。guard 传 consumer 原资源守卫；CLI verify 使用原 PF resource guard。

CLI：`python -B -m analysis.p_pf_cache_reuse dry|plan|seal|verify`。plan 只打印 expected_job 与完整输入列表，不入队；seal 要求 `--execute-readonly`，新回执独占写入。job_contract 沿用原固定环境、资源与 transport 校验；新增 invocation 不读取旧 admission argv 参数，第二输入仍为原 admission 文件并被完整 hash 绑定。

## 成本与局限

语义重验本身增加一次全 train/validation JSONL 解析、一次原 train parquet 全扫描及 side 对照；另有多次原 input/cache SHA 闭包扫描，成本不能当作零。wanted bitmap 为每个 side 物理位置 1 byte，总内存约为 side 文件总字节 / 8，另有原 mmap 与 Python 单行/batch 对象。未实测真实全量时间与资源，资源声明须由 owner 按 AGENTS §4 测量。当前实现偏保守，并不承诺 seal 比剩余旧 PF 更快。

原完整 PF 实证闭包与当前 cache 的真实全量语义封存尚未执行。本地 provenance tests 使用真实 `verify_producer` 配合明确 mock 的 context 边界，仅证明局部拒绝路径；不冒称真实 PF。新 source 固定 delta 不能直接容纳任意 consumer 集成文件；后续 wrapper 应单独冻结其 delta、保留 dual identity。

## 本地验证

- Fresh RED：模块缺失与不闭合 caller 证据测试，缓存日志见 owner 私有交付。
- 89 PASS：新模块 29 项 + 原 compact 14 项 + 原 backend 46 项；含真实固定 launcher.prepare、干净 synthetic Git deployment、独立 native/p3c 数据通道布局、metadata、同调用形式 dry 与非法命令非零。无真实输入 PF、无提交。
- 原 backend SHA `d9c9f038e9d87e07a41c42e818db4483e0b4415187c874df09ca810a9ba836a1`；compact SHA `4953efea67bb59bc5b9903530e0736c90fe5cc1b202a70ea7628b555d432216a` 保持不变。
- 隔离 mutation 结果见本节末补充；每项新 clone、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、独立 pytest temp prefix。原件保留。

## owner 后续有限只读验证

1. 原三 PF 及现行 driver 全部保留；原 PF 真成功后机械生成原 producer 完整 pin。
2. 独立 C 审后固定新 source，另建 seal run（新 run_id、独立 output），按当前 cgroup 与 context-inclusive 实测声明资源，真实 launcher dry 后入队，确认 claimed 且非 rejected。
3. 原 stdout/terminal/job/manifest 机械验收 seal；以新 seal 完整 producer pin 执行只读消费逐位对拍与独立 cache-copy 漂移负例。不得改动原 cache 来做负例。
4. 显式新 wrapper 才能连接 train/verify；原科学绑定保留，新增 reader 绑定单列；独立 C 审与真实只读正式路径 PF 通过前不切换。

最终隔离变异：8/8 killed（source delta、双向 offset 边界、原 side 编码、unused 全零、reader source ledger、完整 own-job、打开后 closure、闭合文件集）。首轮单独删掉 offset 行尾检查仍存活：每个中间边界还由下一行行首检查覆盖，末尾由 `_offsets` 的文件长度检查覆盖，属于冗余等价路径；第二轮同时删掉行首/行尾检查，测试 killed。首轮 own-job 变异暴露负例过早由旧 manifest 的 run SHA 拦截，已把测试改成重新钉住完整 producer/manifest/stdout，确保只剩 own-job 判定，第二轮 killed。首轮与第二轮原始 clone/log 均保留。

## 独立 C 发现后的 mmap 修复（取代首版对应实现）

独立 reviewer 发现首版可把已经映射的旧 inode 重命名，原路径补写相同字节；路径 hash 保持不变，旧 mmap 却可被改写，首版还把补写后的 inode 身份记录成旧 mapping 的身份。该缺口成立，不以旧89PASS声称覆盖。

修复只改本新增模块与测试：在映射前捕获 dev/inode/size/mtime/ctime；打开 rb fd 后 fstat 必须相同；从同一 fd 计算 sealed SHA，再把该 fd 交给 np.memmap，不重新按路径打开；hash和映射之后均核 fstat/path 身份；原对象持有映射前身份，readonly_views 及最终consume闭包再次核对。JSONL 的身份也在 hash 前捕获，hash后及最终闭包核对。原 backend/compact/driver 字节不变。

新增5项 fresh RED（首轮4个映射/JSONL race、第二轮1个late closure race），均先复现后修复。最终94PASS：新34 + compact14 + backend46。新增3/3隔离semantic mutants killed（unsafe path mapping、JSONL late identity、late mmap closure）；每项独立clone/bytecode禁用/唯一缓存与pytest目录，原件保留。原独立攻击仅将注入hook适配为支持file object的 `.name`，相同“映射后detach+补原路径+改旧inode”攻击被 `mmap inode identity changed during mapping` 拒绝；未改独立原始脚本。

同一fd额外SHA读取会增加cache扫描成本，真实资源与时间仍UNKNOWN。此修复仍待独立reviewer复核，不部署、不切换consumer、不修改passes。
