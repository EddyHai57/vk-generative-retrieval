# Tensor cache C 档独立审查（Luna）

- 审查模型：**gpt-6-luna / high**
- 审查上下文：独立 reviewer 的干净任务上下文；未派子 agent。
- 日期：2026-10-21（Asia/Shanghai）
- 结论：**CHANGES_REQUIRED；暂不 APPROVE C 档**。静态核查支持缓存实现的主要等价设计，但 mutation receipt 中 3 项关键 passthrough 变异无效，无法据此验收完整 synthetic 证据。真实 anchor parity 按授权未运行，不能据本报告声称通过。

## 审查范围与证据

完整实现/测试包按 `cache/overnight-20261022/tensor-review-package-v1.txt` 中逐文件清单审查；核对了冻结卡 `docs/superpowers/specs/2026-10-22-tensor-cache-card.md`、作者的 `cache/overnight-20261022/tensor-tests-v1.log`、`cache/overnight-20261022/tensor-mutations-v1/receipt.json`，以及冻结源 `b5a4d1c` 的 `generative_g6_model.py` / `generative_data_v2.py`、`0ca61bc` 的 `retrieval_feat02_run.py` 和本地 `analysis/profile_anchor_steps.py`。

没有连接服务器或网络，没有执行真实输入、模型训练或 capture，没有重跑作者的 10 synthetic tests / 14 mutations。只读查看了源码、receipt 和原始路径。旧训练入口在当前工作树没有改动；本报告是本审查唯一新增文件。

## 逐项 verdict

| 冻结卡要求 | Verdict | 证据与限制 |
|---|---|---|
| G/P 共用有界、确定性 CPU 样本 LRU | **PASS（静态）** | `src/vkgr/tensor_cache.py:39-67` 为共享 `TensorCache`，以 tensor storage bytes 淘汰，超额单项不缓存；hit 返回 clone，避免调用者污染内部项。README/docstring 与冻结卡一致，RSS 不等于该 byte cap。 |
| namespace 含 input/SID/config/source/runtime/constructor 身份；P 显式无 SID 身份 | **PASS（静态）** | `tensor_cache.py:19-24,50` 六个 SHA 都强制必填并合入 row key；capture 对 P 提供 `sha('P:no-SID')`（`analysis/tensor_cache_parity.py:101-104`）。row key 还覆盖完整 row 和实际依赖值。 |
| G 样本张量及动态 padding 等价 | **PASS（静态 + 作者 synthetic 收据）** | `tensor_cache.py:69-84` singleton miss 后按该批最大 source width 重建，decoder/labels 按行拼接；动态长度、排序与 tail 有作者测试记录。G 原 constructor 验证规则见冻结源 `generative_data_v2.py:46-52`。 |
| P 样本张量、feature 和 width 等价 | **PASS（静态 + 作者 synthetic 收据）** | `tensor_cache.py:86-102` 按原 step width 拼回；每样本实际映射、feature 和完整 row 入 key。singleton 使用 history 长度，剩余位置为零，与原 `_tensors2` 的零填充形式一致（冻结源 `retrieval_feat02_run.py:118-143`）。 |
| 旧 step code object 与 globals 隔离 | **PASS（静态）** | `tensor_cache.py:27-36` 复用 `original.__code__` / defaults / closure，只复制 globals 并覆写 builder；原模块 globals 不写。 |
| padding/tail/cursor/负例/mask/logQ/RNG 及训练语义未改 | **PASS（静态范围）；测试证据不完整** | 新实现只替代 builder；原 G/P step 仍分别调用相同 stochastic、mask、loss、clip 和 optimizer 路径。作者测试有一个合成随机调用边界用例，但并未运行真实训练；三项 passthrough mutations 无效，见阻断项。 |
| tensor bytes、参数、梯度、optimizer、loss、cursor、所有 RNG 的新实例逐位比较工具 | **PARTIAL** | `analysis/tensor_cache_parity.py:18-60,123-167` 覆盖张量摘要、参数/optimizer 前后、反向钩子梯度、clip 后梯度、loss/result、cursor 与 Python/NumPy/Torch RNG 及 P negative RNG。它按授权未执行。真实 CUDA/source/data/runtime 绑定和跨进程初始化仍待新实例实际验证。 |
| mutation 隔离 | **PARTIAL / BLOCKER** | receipt 的 baseline + 14 项均被标为 accepted、使用 fresh copy / `-B` / `PYTHONDONTWRITEBYTECODE`；但 3 项的 mutation 锚点实际不对应其标签，不能接受为有效的目标变异证据。其余 11 项静态查看与目标断言大体相符，仍以作者收据为依据，未复跑。 |
| 不执行真实 capture、训练、服务器作业 | **PASS** | 本审查未做以上操作；真实 parity 状态保持 `NOT_RUN`。 |

## 阻断项

**B1 — 三个 passthrough mutations 改错位置，造成假阳性杀死。**

- 文件位置：`analysis/tensor_cache_mutations.py:24-26`。三项都以 `('return result\n', ...)` 为替换锚点；该锚点命中的是 `bind_step()` 的 `return result`（实现 `src/vkgr/tensor_cache.py:33-36`），而非 negative、mask 或 logQ 传递路径。
- 可验证原因：被替换后 `bind_step()` 返回 lambda，不再返回同 code object 的 FunctionType。作者测试 `tests/test_tensor_cache.py:128-130` 明确先断言 `wrapped.__code__ is original.__code__`，因此三项会在该处失败，未执行到任何变异后的 stochastic 输出比较。报告 receipt `tensor-mutations-v1/receipt.json` 只记录通用 `AssertionError`，未记录具体断言位置或目标覆盖，无法反证这一点。
- 影响：`negative_passthrough`、`mask_passthrough`、`logq_passthrough` 不能作为其目标逻辑的 mutation kill 证据；“baseline +14 全部被目标断言杀死”不成立。需修正 mutation 使其确实改变对应输出/调用边界，由对应断言杀死，再重新生成隔离 receipt。无需重跑整套 synthetic suite以外的其他流程；三项及 baseline 验证即可，但按冻结卡的 campaign 结论应以完整新 receipt 记录。

目前未发现缓存实现本身已证实的必然字节不等价问题；但因 B1 与真实 parity 未运行，整体 C 档不能批准。

## 质量观察与边界

1. 缓存键把 namespace 汇总摘要、完整 row 与有效 map 值组成 key；G token 映射与 P idmap 在每行计算，能避免 mapping 原地变更后复用旧值。键的 JSON 编码要求 row / map values 是 JSON 兼容值；冻结数据路径符合该预期，通用任意对象不是接口合同。
2. P singleton 缓存使用 `n=len(history)` 构造，再在 caller 指定 width 下零填充；这与旧函数基于 width 分配、逐行写入的结果在常规有效输入上相符。G 同理按每批最大 tokenized history 重建 source。仍须在新实例真实 anchor 上检查逐步构造张量摘要，synthetic 结果不能替代。
3. parity runner 捕获与比较使用 SHA 摘要和序列化文件逐字节比较。它捕获 P `_columns` / `_log_q`，G/P builder 输出，以及训练状态；若正式 capture 发现 anchor 路径使用未登记 RNG 对象，当前 `freeze()` 会明确拒绝未知类型，而不是静默省略。该行为是 fail-closed。
4. 原卡要求的 10 warm + 100 measured + 10 diagnostic profiling 尚未执行；今晚阶段 1a 约束也明确 parity / profile 留至新实例。因此本审查不评提速、不报真实 parity 结论，也不把 synthetic 测试当 anchor 验收。
5. 作者证据显示 10 个 synthetic tests 通过、14 个 mutation cases 连同 baseline 的机械 receipt 为 pass；本报告只因 B1 下调变异结论，没有重跑已完成套件。服务器执行、holdout/test 访问、checkpoint、数据导出均未发生。

## 交给根执行者的最小处理

修正 `analysis/tensor_cache_mutations.py` 中三项替换位置，使 mutant 确实改变 negative、mask、logQ 传递值，并证明相应目标断言触发；用全新隔离副本生成新 receipt。之后本地修复复核按 AGENTS §4A 的 C 档要求由独立 reviewer 覆盖使此前结论失效的改动。真实新实例 parity 与 profiling 仍按冻结卡单独执行，本报告不代替它们。

## 针对性收口补记（B1 与 warm helper）

本节只复核原 B1、P warm-cache helper 的定点修订，不重开全范围审查。

- **B1：CLOSED（synthetic mutation evidence）**。`analysis/tensor_cache_mutations.py:24-26` 现在在 `bind_step` 创建的 private globals 中改写 synthetic `_negative_inputs`；原 step code object 保持不变。对应合成 step 每次仅调用一次 `_negative_inputs`，各变异保留该次 RNG 抽样，再分别改写 negative、mask、logQ。新 receipt `cache/overnight-20261022/tensor-mutations-v2/receipt.json` 明确列出三项 `target_assertion`，且 `target_killed=true`。逐条核日志可见：`negative_passthrough.log` 命中 “negative input passthrough”；`mask_passthrough.log` 命中 “mask input passthrough”；`logq_passthrough.log` 命中 “logq input passthrough”。相应测试断言位于 `tests/test_tensor_cache.py:138-141`。这些失败发生在各自目标断言，不是函数 code object 断言。其余 mutation 结果沿用 v2 receipt；本次未重跑。
- **P warm helper：定点修复通过静态核对**。冻结 profile 的 P `step()` 捕获 `view` 并以 `view` 作为 `feat.train_step` 的 data 参数；闭包没有捕获 `data`。`analysis/tensor_cache_parity.py:69-83` 现在由 `closure['view']['idmap']` 读取映射，与实际 closure 对齐。warmup 对 `closure['cursor']` 做 deepcopy 后推进副本，再从 `closure['rows']` 取 batch；不会推进训练用 cursor。新 synthetic 测试 `test_p_warmup_uses_real_step_closure_view_without_advancing_cursor` 覆盖这两点；作者日志 `cache/overnight-20261022/tensor-tests-v2.log` 记录 11 tests passed。该测试与日志是作者证据，本审查只核对其内容，没有运行。
- **合并 verdict**：B1 原阻断关闭；此次差异没有改变生产实现，`src/vkgr/tensor_cache.py` 未修改。局部工程与 synthetic 证据可接受。真实 anchor parity 和 profiling 仍是 **NOT_RUN**；因此这项定点收口不构成真实 parity / 提速验收，也不能把实际新实例执行批准写成已通过。原报告其他范围与限制保持不变。

### 追加定点复核：P observation proxy 的 callable 绑定问题

新发现的 parity harness 问题已由 `observe_p_inputs` 修正。本次只读核该 helper、synthetic 测试和 v3 日志：

- `analysis/tensor_cache_parity.py:69-84` 从 `step.__globals__['b1']` 读取原模块对象，浅复制其 globals 到 `SimpleNamespace` proxy，并只替换 `_columns` / `_log_q` 为观察 wrapper。每个 wrapper 调原 callback 一次，先 `freeze(value)` 记录，再返回同一个 `value`；没有改写返回数据或原 `b1`。
- helper 用 `FunctionType(step.__code__, globals_copy, ..., step.__closure__)` 重建 step，保留同一 code object、defaults 和 closure，只让该 step 的 private globals 指向 observer proxy。P 原 `train_step` 是模块级函数，以 globals 中的 `b1` 查找这两个函数；此做法解决原先将不可调用的 `SimpleNamespace` 交给 `bind_step` 的校验错误，且不触碰生产缓存模块。
- P 原 profile step 的 `data` 实参来自闭包中的 `view`（`view = {**data, 'rows': rows}`），closure 不捕获原始 `data`；因此前述 warm helper 使用 `closure['view']['idmap']` 的修正仍正确。
- `tests/test_tensor_cache_parity.py:11-22` 验证 synthetic proxy 保持 code object、原 globals 的 `b1` 身份与函数返回值相等，并记录两类观察结果；`cache/overnight-20261022/tensor-tests-v3.log` 记录 12 tests passed。只审阅作者证据，未重跑。

**该定点 verdict：PASS（静态 + synthetic 证据）。** 此结论仅适用于本地 parity harness 工程接线；capture 未执行，真实 anchor parity 仍为 **NOT_RUN**，故不构成真实 parity 批准。此前全范围意见和限制不变。
