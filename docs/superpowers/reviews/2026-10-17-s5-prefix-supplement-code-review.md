# 新 SID 的 A1 确认：前缀补证独立 code review

- 日期：2026-10-17
- 结论：**CHANGES_REQUESTED**
- 受审范围：`03ab9c6..d5e39b1`，仅新增 `analysis/sid_a1_prefix_read.py` 与 `analysis/test_sid_a1_prefix_read.py`。
- 对照：decision_log「2026-10-17 晚间：新 SID 的 A1 确认」§4、当前 AGENTS §4/§7；冻结 S0 三文件按新入口的 SHA pin 核验并实际导入。
- 边界：本次只做本地只读代码审查与 synthetic 测试，未连接服务器、未读取真实数据或 checkpoint、未 push。可见 context 计数 UNKNOWN。

## MEDIUM：verify stdout 绕过新增读取守卫

位置：`analysis/sid_a1_prefix_read.py:65–67`、`:81`。

新增入口只替换 `archive.read`，但冻结 `phase2_validation_read.verifier()` 对 `logs/stdout.log` 直接执行 `Path.read_bytes()`；`queue()` 对 rejected reason 也直接 `read_text()`。这些读取不经过 `safe()`，因此“不跟随 symlink、不越界、不读 test/checkpoint”的边界没有覆盖实际全部读取。`inspect_run()` 在 line 82 拒绝 rejected 之前已经调用了 `queue()`。

已在临时 synthetic 目录复现：把授权 run 的 `-verify/logs/stdout.log` 建为指向另一目录 `test/checkpoint.json` 的 symlink，并将 `archive.read` 替换为一经调用就抛错的函数。新 `safe(log, allowed)` 报 `symlink forbidden`，但原冻结 `archive.verifier()` 仍读取该目标并返回其中的 `{"status":"verified","checks":"outside_scope_marker"}`。复现没有读取真实受限文件。

修法：在新增 wrapper 调用 `inspect_run()` 前，对它可能直接打开的 stdout、reason 路径及全部祖先执行同等 scope/symlink/prohibited 检查，或以新增的受限元数据适配层读取这些收据；保持冻结 B 和三份冻结源码逐位不变。添加经过 `execute`/inspect 路径的负例，覆盖 stdout symlink、祖先 symlink 与 rejected reason 越界，要求在任何 payload read 前拒绝。

## MEDIUM：未强制原始 stdout 的 verified 收据

位置：`analysis/sid_a1_prefix_read.py:81`。

这里仅依赖冻结 reader 的 `state['verified']`。冻结 `verifier()` 会将 stdout、`out/verify.json`、`out/manifest.json` 的 JSON 合并到同一个 records 列表；其中任意记录是 `verified` 即可，与 job/manifest 和 terminal 绑定后返回 True。因此这个条件没有落实当前 AGENTS §4 的“科学验收只认原始 stdout 里的 verified 加 job / manifest 绑定”。

已在临时 synthetic 目录复现：terminal 为 succeeded/exit_code=0，claimed job 的 run_id、`--manifest`、`--manifest-sha` 正确，stdout 只有 `no verified result here`，而 `out/verify.json` 放置 `status=verified` 与相同 manifest SHA；原冻结 `verifier()` 返回 `verified=True`。当前 wrapper 没有追加任何 stdout 来源检查，故其 line 81 无法拒绝这类状态。新 6 tests 和既有回归均未覆盖这个场景。

修法：在新增 wrapper 中独立解析通过路径守卫的原始 stdout，要求存在 `status=verified` 记录，并明确绑定同一个派生 verify job 与固定训练 manifest；记录 stdout SHA 和绑定证据。不要仅以 frozen reader 合并后的 records 为依据，也不必修改冻结 B。新增 stdout 缺失、stdout 无 verified 但 verify.json 有 verified、错误 job/manifest 的拒绝测试，以及原已验收格式的通过测试。

## 已确认与验证限制

- `run_ids` 必须按固定顺序恰好包含三个已授权 run，额外 seed、子集被拒绝。
- B 直接调用 SHA 固定的原 `prediction_metrics`；prefix1/2/3、top1/top50、exact item/full SID 的算法未改；1897 in-catalog / 623 OOV 的运行后断言存在。
- 历史 manifest 使用固定 HISTORY_PIN，catalog 文件 SHA 与内容 digest 绑定该历史；query 文件 SHA 来自该固定历史 manifest，数量/OOV 有检查。G 归档继续复用固定 reader 的训练 manifest → best evaluation → shard SHA、完整有序 query 身份、catalog、computation digest 和 item metrics 复核链。
- SID manifest 使用固定 SHA/version，mapping pin 来源于该 manifest，正式模式校验 mapping 的 catalog 顺序；输出 B 是聚合值，没有逐 query 预测或完整 mapping。
- 请求 manifest pin 与真实归档的最终一致性、服务器冻结源目录及真实输入可用性仍需服务器只读 preflight/正式检查，本次没有声称已核验真实文件。
- `--preflight` 不调用 `load_predictions` 或 B；本次测试不能代替正式真实路径 preflight，也不能证明真实归档已通过上述链。

验证结果：

1. 指定 `VKGR_FROZEN_DIAG_ROOT=<S0_WORKTREE> PYTHONPATH=. OMP_NUM_THREADS=1 <PYTHON> -B -m unittest analysis.test_sid_a1_prefix_read -v`：**6/6 PASS**。
2. `PYTHONPATH=. OMP_NUM_THREADS=1 <PYTHON> -B -m unittest analysis.test_phase2_validation_read -v`：**9/9 PASS**。
3. 上述两个本地 synthetic 最小复现均成功，证明遗漏不被现有测试阻止。

两项补齐后按项目规则 fresh RED → GREEN → 回归继续；本报告不授权入队、修改科学函数或扩展读取对象。


## Codex工程修复关闭记录（不声称第二份review APPROVED）

原review两MEDIUM均仅在新增适配层修复：legacy直接日志/辅助回执预先验证父路径无symlink及无rejected，且强制原stdout JSON status=verified；原三个冻结文件SHA未动。新增两项fresh RED，修复后8新tests+6相关聚合tests通过，3内存变异全杀；首个变异夹具曾被另一门禁掩盖，强化为有效verified日志后已检出。最终source d7c6551已扫描push/固定新部署。真实只读preflight与唯一正式聚合均通过、143输入SHA齐，来源experiment_log19:24节。按AGENTS普通工程修复不再重复整份review。旧G6 fixture mismatch与新verify身份字段issue均仍OPEN。
