# 序列用户塔 + 组合叠加：独立 code review

密级：公开。结论：**CHANGES_REQUESTED**。时间：2026-10-16 19:19 UTC（北京时间 2026-10-17）。当前请求 context 计数不可用：UNKNOWN。

一次整体 review：叠加 worktree 的 `c4e4eb8c2a9d87d42303e2c0021e168942fda33a` → `4e03caa65cde93807593872812c74af380a5dee4`，共 16 个新增文件。补看 phase2 worktree 的 `analysis/engineering_queue.py`、`analysis/feature_leakage_real.py` 关键路径。未重做组合 `84b160c` 的原专项 review，也未复审前一包双向确认的修复。

依据：decision_log「2026-10-17 上午」§1.4、上午续及「组合泄漏 review 的补证方式」，叠加冻结卡和 implementation-evidence。仅本地 synthetic/只读检查，未访问服务器、真实数据、test 或旧 checkpoint，未修改实现或 commit。以下路径均为各自 worktree 的 repo-relative 路径。

## HIGH 1：训练发布和 detached driver 启动分成两个 action，部分发布后甚至无法 start

位置：`scripts/seq_combo_stack_ops.py:234–242`。

`submit` 写 plan 并同步发布三个训练后直接返回，必须另发 `start` 才启动 detached driver。SSH 在两个 action 之间断开时，训练仍可执行，但没有唯一接续进程来补 verifier。更严重的是部分 publish 失败：`start` 要求三个 seed 都已有 queue 或 terminal 收据，缺一个就拒绝启动；重跑 submit 又会被已有 queue 和独占 plan 拒绝。因此已有训练可能完全失去自动 verify，也没有 driver 的失败 ALERT。这是代码路径确定的问题，不是尚未取得 PPID 收据的普通环境限制。

本地 fault injection：前两次使用真实 publish，第三次抛 `OSError`。得到 `published=2, driver_exists=false, final_alert=false`；随后调用真实 start 的准入逻辑，得到 `registered train has no queue or terminal receipt`，`Popen` 未被调用。仅使用临时 synthetic queue。

需要的 guard：首次发布前已经有 detached、持锁、可确认 ready 的唯一 owner；发布和观察接续使用同一个持久 plan、原 deadline。部分发布失败写明确 ALERT，并能接续已发布的 run，不能靠补发新 attempt 或重复 admission 恢复。补 interruption、部分发布和 driver 启动失败测试；成功 submit 必须返回实际 driver ready 的证据，不能只返回 pending 文件名。

## HIGH 2：叠加自己的真实路径泄漏逐项收据缺失，当前准入也没有要求它

位置：`src/vkgr/retrieval_seq_combo_stack_run.py:377–422`、`scripts/seq_combo_stack_ops.py:121–148`，及叠加冻结卡的工程验收段。

最新裁决明确要求以后的冻结卡、叠加实验在实现阶段提供服务器只读真实路径逐项 receipt。现有 preflight 可以检查共享初值、非零 g/Q2/β 下的 target 翻转、候选有限及 unsupported 为零；返回值只有总体 preflight 状态和样本数。它没有 L2 导出隔离、L3 历史字段反向对照、L4 周平移、L5 item ID、L8 week25 拒绝、L9 训练/导出候选同源的逐项运行矩阵，也没有静态项 L6/L7/L10 与设计编号的映射。实现交付明确尚未运行真实 preflight，没有真实逐项收据。

`train_jobs` 检查的是原组合 review APPROVED、普通 preflight 和 CUDA proof；即使原组合补证通过，它也可能直接放行没有叠加自身真实路径矩阵的训练。原组合模型的收据不能证明新 `SeqComboStack` 的实际路径。本项是泄漏验收与准入缺口，**没有观察到实际泄漏**；synthetic target 测试不能替代真实运行。

需要的 guard：按最新裁决把叠加自身对应的运行时检查接到 engineering=False 正式 loader，512 条 train + 全部 2520 条 validation，固定非零 Q2/β 和开放的 g；静态项如实标记，不伪造 runtime PASS。记录执行 commit/source SHA、输入及侧表 digest、逐项 pass/fail/样本数、test/旧 checkpoint 未读。任一失败 STOP，不静默修复重跑；dry/submit 必须绑定这份叠加收据，并继续单独要求原组合 review APPROVED。当前新增共享 `feature_leakage_real.py` 针对原 A/C/D/E，尚不能关闭叠加这项缺口。

## MEDIUM 3：显存测量漏掉已建立的 CUDA context，准入未验证测量公式

位置：`scripts/seq_combo_stack_vram_smoke.py:81–83,114–119`、`scripts/seq_combo_stack_ops.py:36–43`。

probe 先执行 CUDA allocator 初始化/同步，再取 `free_before`；后续 `mem_get_info` 差值不会包含此前已建立的本进程 CUDA context，`max_memory_reserved` 也只涵盖 PyTorch allocator。docstring 所称“includes the CUDA context”不成立。此外 global free 差值会受并行其它作业释放/分配影响，不能稳定表达本作业总占用。scheduler 按结果并行准入时可能低报进程总显存；当前未运行 CUDA，因此不声称已发生超限。

`check_engineering` 只要求 declared 为整数且 ≥1024，未要求 measured 有限为正，也未复算 ×1.15/向上256。现有 ops test 的“通过”proof甚至完全没有 measured/sample 字段。

需要的 guard：测量本 PID 总显存，并把 allocator peak 与本进程非 allocator/context 占用合并成保守峰值；保留 train/export 阶段测量及峰值证据。准入对同 commit 的 measured/sample/shape 和 `max(1024, ceil(measured*1.15/256)*256)` 复算一致性 fail closed。补缺 measured、非有限值及错误声明的拒绝测试。CPU shape 结果仍是 VRAM UNKNOWN。

## 已核通过的代码边界

- diff 为 16 个 added 文件，未修改 base 原文件；新增原 M1 两模块与 `04d4ede` blobs 的逐字核对测试 PASS。
- 配置只含 42/2024/2025，mean；旧 M1 非 schema/task 配置逐字段一致。构建公共参数与 branch 后再追加 Q2/β、C/D，公共参数 bitwise 初始化保持，g=0 用户塔等于 M1。
- m 使用历史事件的 E 池化；branch 仍用 F3/F4，反馈权重和统计桶不进入 branch；candidate 四统计表仅作用 supported。训练与导出均走 `candidate_items`。Q2/β 手算 oracle、梯度通路及 immutable bucket buffer 拒绝有覆盖。
- 新 schema、best/last hash、source/data/config/stats/side/bucket binding、selection/draw replay、best 重导出及 g→0 重算均存在。私有 JSON 先校 SHA，data root 分离，不覆盖源码文档，不读取旧 checkpoint。D-1 与 23:2x 限制保留，不作转正裁决。
- 原组合 review 门禁明确要求 APPROVED 且 commit 为 `84b160cb0eeb3c53b877f6cf6474276ba06a6ba5`。失败/rejected 不自动重跑，driver 轮询900秒、原48h截止；HIGH 1 发生在 driver 建立之前。

## 本批共享只读工程工具

`engineering_queue.py`：检查 ops/schema/单核/有限显存、原 run 和 pending/claimed/rejected，dry 与 submit 绑定相同 job digest；publish lock 内重查并先独占 intent，再 scheduler submit。中断留下 intent 时拒绝自动重投，适合人工核清原收据后处理；本 review 未要求自动重试。seal 使用独占新文件，status 有读取大小限制。关键路径未发现额外 HIGH/MEDIUM；未实跑真实 launcher/admission。

`feature_leakage_real.py`：原模型 commit/逐文件 SHA、私有参数和 inputs SHA 后走 production loader；运行非零 Q2/β 的 A/C/D/E 矩阵。运行项失败直接抛错，PASS 收据只在整轮完成后独占写入；L6/L7/L10 如实标 STATIC_REVIEWED。C/D 检查 candidate raw/export/训练 columns 一致。未调用 optimizer、训练或 checkpoint loader。synthetic fixture 测试会显式改测试夹具的 synthetic 标志以覆盖生产函数；这只是测试分支覆盖，**不构成真实输入证据**。原组合的最终关闭仍须实际服务器一次运行及原 reviewer 核收据；本 review 不代替它。目前未发现此新工具额外 HIGH/MEDIUM 阻塞。

## Fresh 验证与限制

本地 smoke venv、`-B`、`OMP_NUM_THREADS=1`：

- 叠加 `PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_seq_combo_stack*.py'`：15 tests，PASS。
- `PYTHONPATH=src python -B scripts/seq_combo_stack_freeze.py`：223 files，PASS，freeze SHA `d0a075a31df234c6c35bb280e7a0c059519a9cd06deba7b13506eae05c459655`。
- phase2 的 `analysis.test_feature_leakage_real`，PYTHONPATH 指向原 feat04-combo 生产模块：3 tests，PASS，仅 synthetic。
- HIGH 1 本地故障注入：确认两项已发布、无 driver/ALERT，后续 start 拒绝。

未重复 full suite、CPU fullshape 或 CLI proof；它们仍按实现者证据及其限制记录，不将旧两项合同失败或先前环境错误改写为 fresh PASS。真实工程 receipt、CUDA/总显存、dry/admission、driver PPID=1 与断线存活均未取得。本结论不授权真实训练。
