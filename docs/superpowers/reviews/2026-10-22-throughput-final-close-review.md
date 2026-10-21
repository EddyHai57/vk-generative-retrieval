# Throughput 工具最终定点复核（2026-10-22）

- 角色：独立 C 档定点复核；按委派记录 `gpt-6-luna / high`。运行环境没有独立型号回执，故不声称实际运行型号已切换。
- 范围：只读当前主 tree，核对工具 4/5/6 修复、工具 1/2/3/7 资源复核摘要和新 `throughput_pipeline.py` 接线。未改仓库文件；本报告为唯一写入文件。
- 限制：没有服务器、真实数据、训练、CUDA、真实 anchor、性能测量或测试重跑。上下文计数不可见：UNKNOWN。

## 结论

**工程工具链主要阻断已关闭，但还不宜把所有路径都视为完整闭合。** 工具 5 的前置资源预算修复与冻结 P 算分保持一致；工具 6 原审查列出的 owner/factory、codec alias、初始化预算、VRAM receipt、result bound 阻断均已在代码和既有 synthetic 回执中得到针对性修复。工具 4 的 week26/week_26/holdout 和 hint 正整数检查也已补上。新 pipeline 是 config-only wrapper：默认路径返回原对象，工程配置独立落盘，公开方法都使用同一 `self.config`。

本次仍发现两个实质边界：工具 4 的 `test` 路径识别只匹配 `/test/` 这一种目录片段；工具 6 的 TorchSession 身份摘要覆盖 state_dict 权重，却没有覆盖模型 state_dict 之外的可变运行属性或模型设备放置。pipeline 对矛盾配置 `use_mmap=true, sequence_enabled=false` 也会静默忽略 mmap 开关。当前提交的三份 throughput YAML 均启用了 sequence，因此这个配置组合不影响所列方案。

## 原阻断逐项状态

| 项目 | 当前状态 | 证据与剩余边界 |
|---|---|---|
| 工具 1 序列构造预算顺序、列类型、生产列式 G/P step parity | ADDRESSED | 资源复核记录先估算再 cast/构造；`g-synthetic-step-a40-v1.json` 与 `p-synthetic-step-a40-v1.json` 均为 `exact_step_state: true`、workers 1/2/8。预算是模型估算，非大输入峰值保证。 |
| 工具 2 worker 输入/输出预留、RSS 边界、pending 取消 | ADDRESSED（指定装配路径） | `throughput_workers.py` 在 worker 前后检查 RSS、按输入和输出 hint 预留并取消未开始任务；资源复核中的 1,200-event fixture 仅证明该 fixture。已运行任务仍可能先出现瞬时峰值，关停会等它结束。 |
| 工具 3 BHKM callback/返回契约与整轮磁盘预检 | ADDRESSED | 资源复核记录 `03-final-green.log` 10 项通过、`mutations-03-final-v2` pass；整轮容量是预检，不是对外部并发占盘的原子租约。 |
| 工具 4 PathLike/week26 与 hint | 原阻断 ADDRESSED；`test` 路径仍有缺口 | `throughput_weekly.py:14-20` 现识别 `str`/`PathLike`，包含 `week26`、`week_26`、`holdout`；`tests/test_throughput_weekly.py:29-33` 覆盖 PathLike holdout/week26 与 hint 零值、bool。`throughput_workers.py:38-40,53-55` 对单/多 worker 都要求精确正整数 hint。剩余：weekly 只检查子串 `'/test/'`，例如路径末尾 `/test` 或文件名 `test.parquet` 不匹配，见下方发现。 |
| 工具 5 P exact 的 guard 顺序、float32 mmap 与 worker RSS | ADDRESSED | `throughput_exact.py:61-78` 在转换与扫描前估算/guard；并行 mmap 拒绝非 float32 和非完整 `.npy` 映射；`:83-96` 将 worker array 投影并传 worker RSS cap。修复记录给出 5 tests 通过；既有 `test_throughput_exact.py:15-35` 用 35×4103 synthetic fixture 对比旧预测字节及每个 score trace。固定 32/4096 块、float64 `einsum`、tie-break 和 ordinal 汇集未见变化。资源预算是估算，不是实测峰值。 |
| 工具 6 G decoder 身份/资源准入 | 原列阻断大体 ADDRESSED；仍有模型运行状态身份缺口 | owner 绑定见 `throughput_decode.py:269-281,318-339`；factory type/state SHA 检查在 `:284-304`；TorchSession、真实 tiny model factory、ValidatedCodecD alias/pointer 归一化和 storage floor 在 `:91-135,173-240`；预算及 VRAM receipt 在 `:243-267,284-305,341-357`；正整数 result bound 在 `:323-324`。既有 11 项测试源含模型/codec mismatch、wrong weights、真实 TorchSession factory、alias pointer 和 budget 检查；`mutations-06-final-v2/receipt.json` 为 pass。遗留身份缺口见下方发现。初始化预算仍是 caller estimate，不构成瞬时 RSS 硬限制；CUDA receipt 是调用方声明，CUDA 未运行。 |
| 工具 7 nested cgroup 与 A40 planning | ADDRESSED（工程规划口径） | 资源复核记录 cgroup v1/v2 祖先最小限额路径、probe/guard 共用及 A40 8 workers、256 GB memory、400 GB disk、decode 1、external BHKM false；`mutations-07-final-v2/receipt.json` 为 pass。当前 YAML 原始 SHA 与资源复核记录一致：`69f8039793ace8dfc67063ab94013a0ae958ad1da75e2f6e369dbc61582c8cc5`。这是 planning config，不代表实例测量或自动应用。 |

工具 4 的资源复核文件先前把 `week26` 无下划线路径列为残余；当前源码已包含 `week26` 子串，故该条已解决。报告中的 test 目录覆盖仍限于 `/test/`。

## 新 pipeline 接线与质量

`throughput_pipeline.py:20-26` 深拷贝 science identity，将 config 与其 SHA 放在独立 engineering manifest，并用 `open('x')` 防止覆盖已有 run manifest。`examples` 在默认关闭时直接返回传入的原对象（`:28-29`）；开启 sequence 后再按 mmap 开关创建新 store（`:35-45`）。cursor/assembler、BHKM、week parse、P exact、G decode、resource probe 的公开方法分别位于 `:47-85`，均传递同一个 `self.config`。`sid` 只在 dense BHKM 且有资源 cap 时加入保守 guard callback，预估 `64 * N * min(branching,N) + 4 * features.nbytes`（`:60-74`）；源码已标明这是 planning estimate、不是测峰。BHKM external 开关仍交给原 `fit_configured` 处理。

现有 `pipeline-green.log` 仅含两个成功用例：disabled 时原对象返回、工程 manifest 记录且不覆盖；mmap store 保存后 rows 一致。pipeline-red 是源码未进入 import path 的 setup 失败，不能当作行为失败。individual kernels 有各自 synthetic 与 mutation 收据，但没有端到端 pipeline fixture 一次调用 cursor、BHKM、week、P exact 和 G decode；当前可确认的是静态接线和局部证据，不是全 pipeline 科学路径验收。

**[P2] mmap 开关在不相容配置下被静默忽略。** `examples()` 先检查 `sequence_enabled` 并返回（第 29 行），所以 `use_mmap=true, sequence_enabled=false` 不会使用 mmap，也不会报配置错误。当前 `disabled.yaml` 和另外三个 throughput YAML 的组合彼此一致，不受影响；建议配置构造时拒绝该组合，避免之后产生“配置已启用但实际没生效”的工程记录。

## 仍需处理的发现

**[P1] 工具 4 对 test 路径过滤过窄。** `throughput_weekly.py:15` 当前只把 `'/test/'` 当拒绝片段，不能拒绝常见的末尾 `/test`、文件名 `test.parquet` 或 Windows 风格分隔路径。用户指定的 test 路径拒绝合同尚未完整实现。建议按路径组件做跨平台大小写归一化，检查独立组件 `test`，并保留 week26/week_26/holdout 检查。

**[P1] TorchSession 没有把全部模型运行状态绑定到身份。** `throughput_decode.py:215-219` 只 hash `state_dict()` 的权重名、dtype、shape 与 CPU bytes；模型类源码在 `:199`，但普通模型实例属性（例如影响 callback 输出的 plain attribute / 非持久 buffer）以及参数当前 `device` 没进入摘要。`token()`（`:236-237`）也只复查该身份与 state_dict `_version`。因此原 owner 和 worker factory 可拥有相同 state_dict、不同运行属性或 device，仍得到相同 SHA；若 callback 使用这些状态，workers>1 可能改变输出或资源占用。当前 tiny fixture 只证明 weight/state_dict 和 codec identity 的检测，没有覆盖该状态。建议身份摘要纳入安全、稳定的模型运行属性及每个参数/buffer 的设备，或对未支持的模型状态 fail closed；随后只需定向 synthetic fixture 验证，不需要真实模型。

## 限制与质量裁决

- pipeline 开关接线：列出的配置下均走同一 Engineering config；默认原路径和 mmap sequence path 有两个局部集成用例。矛盾 mmap/sequence 配置会静默 no-op，建议修复。
- 科学语义：已有 exact score trace、G/P synthetic step 与 kernel fixtures 支持被测小样本逐位行为；本次没有真实数据或 anchor 证据，不作有效性/转正判断。
- 资源：A40 YAML 是规划值。初始化/RSS、密集 BHKM、磁盘估算均非实测峰值；CUDA、真实 G 解码峰值和吞吐收益均 **NOT_RUN**。这些不构成本轮必须读真实数据的要求。
