密级：公开

# 排序导向温度交叉熵：R1 独立 code review

2026-10-17，Codex 独立短任务 reviewer。结论：**CHANGES_REQUESTED**，1 HIGH、1 MEDIUM。修复后由作者 fresh RED → GREEN → regression 自验；本次为完整交付的一次 review，不安排重复整包 review。发现均为工程实现问题，无需重新裁决 τ 或科学设计。

## 范围与依据

- 固定范围：base `fdc988443672dd729a81dde6c0e7ca42a448cd13` → `4cc368f4c2a30cf4f7c26c4609b0072a53b32c95`，11 个新增文件，原源码零修改；本 review 只覆盖 R1，不覆盖 R2。
- 已读主仓 decision_log「排序导向 R1 的温度取值裁决」、主仓排序导向实验卡 §2/§3、该分支 frozen-evidence、ops-handoff、全部新增源码/config/tests及相关旧模型、训练和 beam 路径。
- 只读源码与 synthetic 验证；未登录服务器、未读旧 checkpoint、未读真实/test 数据、未启动真实训练。仅新增本 review 收据。当前会话 context 计数不可用，UNKNOWN。

## HIGH：训练先发布，detached driver 后启动，断线或部分发布失败会失去 verify 接续 owner

位置：`scripts/ranking_r1_ops.py:269–280`、`:361–372`；操作手册也明确要求 `submit → start-driver`。

`submit()` 先独占写 plan，然后在当前前台进程逐个发布三个训练，直到返回才由另一个 CLI 动作启动 driver。此时不存在任何已 ready 的 detached owner。若 SSH 在首次发布后断开，或第二/第三次 `publish()` 因 IO/权限/队列异常失败，已经发布的训练仍可被 scheduler claimed 并运行，但不会自动补 verify。`start_driver()` 也只返回 Popen PID，没有等待 PPID=1、lock、PID receipt 的 ready ACK；子进程启动失败不能由返回值识别。`plan.json` 已存在使简单重跑 submit 失败，手册要求部分发布后 STOP 对账，不能弥补这一自动接续缺口。

本地最小复现：使用临时 synthetic queue，mock `dry_run()` 返回三个 entries，仅令第二次 publication 抛 `OSError`。实际调用原 `submit()` 和第一次原 `publish()`，结果为：首 seed 的 JSON 已在 pending、plan 已存在、driver-pid 与 driver.log 均不存在。没有运行训练或访问服务器。

修复要求：首次 publication 前，唯一 detached driver 必须已核实 PPID=1、持锁并出具 ready ACK；由该 driver 拥有 publication 和后续 observe/verify。部分 publication 失败时保留固定 plan/deadline及已发布身份，记录 ALERT，并继续观察已发布训练、为其补唯一 verify；不得重新提交训练、换 run_id 或重置预算。加入首次发布前断线、第二次 publication 失败、driver ready 失败三个故障路径的 synthetic 测试。

## MEDIUM：显存声明漏计 CUDA process context，可能低估 scheduler 准入成本

位置：`scripts/ranking_r1_ops.py:193–215`，尤其 `:211`；声明计算在 `:53–56`。

CUDA preview 正确执行真实 shape 的两次 effective-batch optimizer step、beam100 b1，以及自身新 state 的保存/重载，但 `measured_mib` 只取 `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`。这两个数覆盖 PyTorch allocator，不能计入 CUDA context、CUDA library 等 allocator 外的该进程显存。之后直接乘 1.15 并 ceil256，可能使准入声明小于该作业真实占用；多作业并行时这会侵蚀资源余量。15% 是约定的测量后余量，不能代替遗漏的 process context。

修复要求：只测本作业 PID 的 GPU 占用，并将 allocator 峰值和进程 context 的组合依据写进 proof，按包含 process context 的总实测值 ×1.15、向上取整256MiB。不得统计其它作业来决定停止。dry-run 校验对应字段与声明推导；无可靠本进程测量则 STOP，不能以 allocator-only proof 放行。加入 allocator 峰值小于进程总量的声明测试。

## 已核通过与边界

- 固定 τ=.8；训练 forward 与 beam decode 都只缩放一次；τ=1 synthetic 权重/logits/loss/梯度/AdamW更新、真实 dropout/microbatch step、serial/batched b1 beam100 parity 测试通过。
- 三 seed 42/2024/2025 一律派生，不依赖 seed42 指标；formal config 锁定原 A1/T3 其余字段；旧 schema 不能进入新 verify。
- 新 data-root helper 显式传给旧 G3 loader，旧 frozen pins 未更改；source 始终由新代码 root 哈希，data-root另记录。verify 重建完整 binding，并做 selection replay、best hash/model digest、重新解码 computation digest 检查。
- PF 要求 engineering=False；CUDA proof 与 PF 核对 data/SID/runtime/model-shape，dry-run 核对三份派生 job/config 并调用 launcher.prepare。拒绝记录会读取 reason；已存在/claimed/rejected不重投；driver 明确要求 PPID=1、单 lock、900秒轮询、原绝对48h deadline。
- 独立 fresh 验证：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests <RUNTIME_PYTHON> -B -m unittest tests.test_ranking_r1 tests.test_ranking_r1_ops tests.test_sid_depth_beam tests.test_generative_g6_unit`，**43 tests / 1.885s / OK**。上述两个发现未被这组测试覆盖。
- 真实 loader/PF、CUDA/process显存、scheduler实际 dry-run、source/runtime部署尚未在本 review 中运行。旧 contract 集成 fixture 的 frozen 文档 pin 漂移仍保留原限制；未更新旧 pin，也未以43项测试声称全 suite通过。

在两个发现修复并完成作者自验前，不应提交正式 R1 训练。
