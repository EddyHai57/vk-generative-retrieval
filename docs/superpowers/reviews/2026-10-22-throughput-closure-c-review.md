# Throughput closure C 定点复核（2026-10-22）

- 角色：独立 C 档只读复核；委派标注 gpt-6-luna / high。本运行环境没有独立型号回执，不能证明运行时实际型号。
- 范围：throughput_tensors.py / test_throughput_tensors.py、throughput_pipeline.py / 对应测试、刚修的 throughput_weekly.py / 对应测试、throughput_decode.py / 对应测试。未读或重做其他七项的独立 C 审。
- 约束：未改仓库文件；未连接服务器、真实数据、训练或真 GPU；没有实际 step harness/anchor 对拍收据，未据此判断正式科学路径。上下文计数不可见：UNKNOWN。
- 测试：按指定 .venv/bin/python -B 和 PYTHONDONTWRITEBYTECODE=1 尝试四个定点模块；21 项中有 7 项因本机 spawn/OpenMP shared-memory 创建失败而报错，另有 RSS 检查调用系统 ps 时失败。它们不能作为代码 RED 证据，也不能作通过证据。额外用单进程 synthetic fixture 对准备后 row 变更做了定点探针，结果见发现 1。

## 结论

本轮确认 weekly test-path 过滤已修复，pipeline 对 mmap/sequence 矛盾配置已 fail closed，TorchSession 已把模型实例属性、参数/缓冲区设备、子模块拓扑和 hook 注册纳入动态身份。CPU tensor 准备保留指定 code object 的设计也成立。

**仍有一项确定的结果差异阻断：PreparedBatch 对 row 内容的检查是自比较。** chunks 里的 expected 与公开的 PreparedBatch.rows 共用 row dict。准备完成后若调用方改了 row，rows != expected 仍为 false，而已生成 tensor 仍对应旧值；绑定函数静默返回旧 tensor。定点探针将 P row 的 side 值从 1 改为 99，调用绑定 step 后 tensor 仍为 1，且与原构造器基于当前 row 的结果不同。建议对 row 输入做独立不可变快照/摘要并在 bind 时验证，或使准备结果中的 rows 不可变；然后增加该变异的定向 synthetic 覆盖。

另有两项尚不能闭合：冻结卡要求预算开启时有保守 tensor output bound，但当前只在 max_in_flight_bytes 非零时强制；仅启用 memory_limit_bytes 会接受无 output bound 的构造，并没有 in-flight 预留。并且 TorchSession 对模型实例字段及运行时状态做摘要，却未纳入动态 class-level 属性值；类源码哈希不能反映运行期间被改写的类属性。若这些运行时状态允许存在，应将其纳入摘要或对可变 class namespace fail closed。

## 定点状态

| 检查项 | 状态 | 证据与边界 |
|---|---|---|
| CPU builder 在 spawn worker 中构造；workers 不采样；CPU/contiguous/无梯度限制 | 部分闭合 | throughput_tensors.py:18-40 在 worker 初始化/任务中执行构造，并用 RNG guard。多进程用例因本机 OpenMP SHM 失败，不能确认 spawn 路径运行通过。 |
| 原 G microbatch、P 动态 padding、顺序及 constructor 参数验证 | 部分闭合 | G 按传入 microbatch 分 chunk，P 一次构造整个 batch；validator 在 bind 时检查非 row 参数（:31-40,51-64）。测试覆盖 G 的 microbatch=1 和 P builder 输出，但没有 G 多 chunk 实际绑定 step 的调用，也没有完整 cursor 驱动的对拍。须等待 harness/对拍收据确认生产构造参数和实际 chunk 调用相同。 |
| 原 step code object / 私有 globals 替换 | 代码闭合，集成有限 | 使用 bind_step 替换单一构造函数名；测试断言 bound.__code__ is old_step.__code__（测试:45-61）。没有正式 anchor step 对拍。 |
| row 顺序、cursor、RNG | 顺序/RNG局部证据；cursor未验证 | prepared worker 结果按 ordinal 输出，cursor wrapper 对 live cursor IDs 作比较（:105-121）；测试只检查合成 index 顺序和全局 RNG 前后相同，没有直接运行 cursor_prepared_batches。发现 1 说明 row 值完整性检查失效。 |
| constructor namespace + validate_call | 设计闭合，隔离保证不足 | 主进程及 worker init 都验证 namespace 和 validator callable（:22-28,77-82）；bind 逐次复核 identity 和 validate_call（:51-63）。但 rows 快照共享，校验不能发现准备后同一 row 对象被修改。 |
| 装配预算前置/限制 | 部分闭合 | 初始 store duplication 估算、worker RSS cap 和 max-in-flight reservation 均存在（:83-100）。当前只有 max-in-flight cap 开启时才强制 output bound；仅 memory cap 不触发。worker RSS 是任务前后检查，不保证瞬时峰值；保守 bound 也不是实测峰值。 |
| pipeline config-only 接线 / 默认原对象 / mmap 输入约束 | 局部闭合 | 默认 sequence off 返回同一对象（:31-33,50-66）；use_mmap=true 而 sequence off 在 manifest 写入前拒绝（:22-29）。测试提供局部 synthetic store/manifest 覆盖。此模块明确不是科学 backend，不因缺少真实科学 pipeline 接入而判本地工具失败。 |
| weekly test 路径拒绝 | 已修复 | 归一化反斜线并按组件检查 test，也拒绝 test.parquet stem（throughput_weekly.py:14-20）；定向测试列出末尾 test、相对路径、Windows 路径和 test.parquet（测试:35-38）。另有 week26、holdout 与 hint 检查。多进程顺序用例因运行环境限制未通过验证。 |
| TorchSession 属性/device/topology/hooks | 已覆盖指定实例状态；动态类属性有边界 | _hash_torch_module 遍历 _parameters、_buffers、training、hooks、module tree、实例 __dict__ 与 slots；tensor 元数据含 dtype/shape/stride/device/layout/grad（:190-270）。测试覆盖 plain gain、meta device、forward hook 与不支持对象拒绝（测试:216-235）。动态 class-level 属性值未 hash；函数运行时还可能读取可变 module globals。 |
| pipeline / kernel 完整端到端 | NOT_RUN / 不属于本地工具闭合要求 | 没有一次贯通正式构造器、pipeline、real launcher 的真实路径回执；科学 backend 接线仍待后续合同流程。 |
| 真机峰值 / 性能 / 真实 anchor | **NOT_RUN** | 未测真实 RSS、CUDA context-inclusive 峰值、墙钟加速或 anchor 逐位差异。 |

## 发现

**[P1] PreparedBatch 允许准备后 row mutation 导致静默 stale tensor。** _build 的 part 是 rows 的浅切片，保留相同 dict 对象；PreparedBatch.rows 和 chunks 中的 expected 因而共享这些对象（throughput_tensors.py:33-41）。bind 只比较 rows != expected 和 validator（:55-63），同一对象被改写时比较仍相等。我的无服务器、单进程 synthetic 探针把 P row 的 side 值改为 99：原 builder 当前结果为 99，prepared tensor 为 1，未抛错。该路径会改变 tensor 输入，需先修正并钉住。

**[P2] memory_limit_bytes 单独开启时 output bound 不强制。** 冻结卡要求开启预算时提供保守 tensor output byte bound；实现只在 max_in_flight_bytes truthy 时拒绝缺少 bound（:75-76）。仅配置 memory cap 会继续处理，而 ordered_map 不会获得 result_size_hint，无法为排队结果做字节预留。该行为与显式 in-flight cap 的含义不同，建议按冻结卡确认是否把 output bound 约束扩展至 memory-only budget；未作科学规则判断。

**[P2 / 条件性] TorchSession 不摘要可变 class-level 模型属性。** 实例字段和 slots 被枚举（throughput_decode.py:254-263），但类字典值不枚举；模型 forward 可读取类属性。类源字节相同不能证明运行时 class attribute 一致。若 worker 初始化/运行中允许更改此类值，它可能不进身份；可摘要可安全序列化的 class namespace，或对模型类的可变属性 fail closed。函数引用的动态 module globals 也有类似边界。

## 后续需要的证据

- 修复发现 1 后，用 fresh synthetic mutation 验证：改写 prepared rows 必须报错或重新构造，不能返回旧 tensor；同时覆盖 G 多 microbatch 连续调用、P 动态 width 和 cursor live order/RNG。
- 若在本轮声明预算合同闭合，补 memory-only cap/output-bound 负例并检查预留路径。
- 取得分析 agent 的实际 tensor factory / step harness 文件稳定通知及逐位对拍收据后，补看生产 namespace、validate_call 参数绑定和原 step code object 使用形式。
- 真实 CUDA context-inclusive 显存峰值、吞吐/内存实测和 anchor 仍保持 **NOT_RUN**；本报告不据此给出 anchor、转正或有效性判断。
