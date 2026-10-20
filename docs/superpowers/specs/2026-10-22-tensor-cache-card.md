# 原样张量构建缓存工程卡（阶段 1a）

状态：依据 3d8a895 第6条与 942a5b0 阶段1a、ecfcf8c/feb194e 夜间授权实施的工程卡；不新增科学规则。今晚只做本地实现/synthetic 验证，真实 parity 与剖析 NOT_RUN。

## 范围与原路径

G：`b5a4d1c:generative_g6_model.train_batch_d` 仅替换其 `sid_targets_l` 构建器。
P：`0ca61bc:retrieval_feat02_run.train_step` 仅替换 `_tensors2`。`_columns`、`_histories`、`_log_q`、负例 RNG、mask、loss、clip、optimizer、同步检查及原调用顺序不动。
用 `types.FunctionType` 复制原 step 的同一 code object，只在新 globals 字典绑定 wrapper；不修改原模块全局，不修改任何旧入口文件。G/P 共用一个 bounded CPU 样本 LRU，不持久化数据，不新建 checkpoint。

## 缓存与等价约束

- namespace 必含 input/SID/config/source/runtime/constructor SHA256；P 的 SID 使用明确的无 SID 身份 hash，不能省略字段。
- row key 包含完整样本 JSON、实际使用的映射值、G token max_len、P feature；不会复用另一输入、配置、源码或运行环境的项。caller 在 setup 固定 namespace，实际映射值逐 batch 进入 key，防止原 map 被改后错用旧项。
- miss 调原 constructor 的 singleton CPU 路径，缓存其确定性张量；hit 只恢复样本 CPU 张量。G source 右侧补零至该 microbatch 的最大长度，decoder/labels 保持原 dtype/shape/bytes；P width 完全采用原 step 参数，不提升到全局最大值。
- batch 行序、tail 长度及 G microbatch 分割均由原 cursor/step 控制；不缓存负例/mask/logQ，不消费 RNG，不改 H2D、精度、stream 或优化器语义。
- 按 tensor storage bytes 限额 LRU；超限 singleton 不缓存，返回新张量；返回数据与内部项不共享可变 storage。总 RSS 还含 key/Python overhead，不能把 tensor byte cap 当 RSS cap。
- constructor 的确定性与 CPU singleton 可分解性只适用于上述钉住的两条路径；通用随机 builder 不在接口合同。不把数学等价当作已验收逐位等价。

## 本地验收

fresh RED→GREEN synthetic 覆盖动态 padding、混合 history 长度、重排 cursor、非整除 tail、G decoder/labels、P side/feature、负例/mask/logQ 原样传递与 RNG 消耗顺序、LRU、返回值污染、key 每个必需 hash/映射漏项。隔离变异使用新代码拷贝、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一缓存前缀；每个变异须被目标断言杀死。

## 新实例逐位对拍（今晚禁止执行）

1. 新实例同一环境重跑旧/新两路，先下载子集参照；不迁移旧 run/checkpoint 或把跨机器旧计时当基准。source/lock/config/data/SID hashes 逐项核，真实 CLI preflight，holdout 误传拒绝；新独占 scratch/输出。
2. 每路从同一初始化、原 cursor 与全部 RNG 状态开始；固定前若干步由验收 request 明确记录，必须覆盖冷 miss、热 hit、不同长度、tail、epoch 边界、G microbatch。若 anchor 短步不自然出现 tail，用独立 synthetic fixture 补，不改变科学样本。
3. 每步捕获 builder 返回 tensor dtype/shape/stride/bytes；forward loss 原值；参数、backward 后/clip 后梯度、optimizer 前后 state；cursor state/indices；Python global 与独立 negative_rng、NumPy global/Generator、torch CPU 与全部 CUDA RNG（独立 generator 必须登记）。G finally 清梯度，须 backward/optimizer hook 在清零前捕获，不能以最终 None 代替。
4. 比较器对上述完整结构逐项逐 byte 比较，float 不用 tolerance，NaN/Inf 按原门禁停止。任一 mismatch=STOP/ALERT，保留两边收据，不改规则/seed/输入追平。
5. 同环境同口径 10 warm + 100 measured + 10 diagnostic 短剖析分别量冷/热缓存及构建时间/吞吐/context-inclusive 显存、RSS；不算召回、不留 checkpoint。未完成这些不能声称提速或真实 parity 通过。
6. 新实例入口/launcher 集成须走 §4/§4A 独立审/真实正式 CLI/部署→metadata→dry 链，当前 standalone 工程工具不等于已获科学 backend 准入。

独立 C 档 review：本地整块完成后一次 Luna（干净 context）；报告须明确模型、源码范围、测试证据与服务器未运行限制。冻结规则与原产物不动。
