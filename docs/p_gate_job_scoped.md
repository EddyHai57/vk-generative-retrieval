密级: 公开

# P gate：job-scoped 模式（INFRA-01 子任务 B）

口径截止：2026-10-14。实现：`src/vkgr/retrieval_sasrec_remote.py`；测试：`tests/test_retrieval_sasrec_remote.py`（`JobScoped*`、`JobOwnerTests`、`ObserveJobTests`、`PProgressTests`）。

## 一句话

job-scoped 下 P 只看自己：本作业进程树 + GPU/RAM/磁盘余量（≥20%）。不再因为 G 的 run 状态、未登记的 GPU 进程、全机 cmdline 扫描、别的 UID 的 fd 不可读、外部进程 fd 很多而 STOP。

## 怎么选模式

- operation manifest 顶层可选字段 `gate`：
  - 不写，或 `{"mode":"legacy"}` → legacy，旧行为不变。
  - `{"mode":"job-scoped"}` → 新模式。常量 `NEW_RUN_GATE`。
  - 其他值、多余 key、非 dict → `ValueError('gate mode envelope')`。
- 新 P manifest builder **必须**写 `gate: NEW_RUN_GATE`。absent 不会默认成新模式（absent = 所有已存在的 manifest）。builder 改造属于 P 迁移 task。
- publisher envelope 可多一个 `gate_mode`（`legacy`/`job-scoped`）。job-scoped 时 `runs` 和 `allowed_p` 必须都是 `[]`，否则 `ValueError('publisher pinned envelope')`。

## job-scoped 改了什么

| 位置 | legacy | job-scoped |
|---|---|---|
| publisher 观察 | `observe_g`：读 G 文件、扫 cmdline、`--query-compute-apps` | `observe_job(feed root)`：只跑 `--query-gpu=memory.total,memory.free`，disk 用 feed root 的 `statvfs` |
| payload | 7 个 key | 定死 8 个 key：四个列表都是 `[]`、`p_writer_count=0`、`gate_mode='job-scoped'`、`free_fractions` 多 `cpu` |
| payload 与 mode 不一致 | — | `ValueError('gate mode/payload mismatch')` |
| writer inventory：`/proc/<pid>/fd` 或单个 fd PermissionError | 走 foreign-UID 探针，失败 STOP | 跳过该进程，不调探针 |
| writer inventory：非 publisher 进程 fd >4096 | STOP `per-process fd bound` | 跳过该进程 |
| writer inventory：fd 总数 >32768 | 全机所有 fd 计数 | 只计 readlink 后指向 feed root 的 fd |
| `/proc/<pid>/stat` 超长或解析失败 | STOP | 非本作业 pid 跳过；本作业 pid（publisher、controller、worker 树）仍 STOP |
| worker owner | — | 每次新采样检查 `start.json.controller` 的 `(pid,pgid,sid,birth)`、非 zombie、本进程 ppid = controller，否则 `ValueError('job owner lost')` |

保留的 STOP：ram/vram/disk 任一 <0.2；可读进程（同 UID）以写模式打开 feed root；charged tree 出现 symlink/特殊文件/root inode 漂移；自己进程树 birth/group 漂移与 worker 退出码。

## CPU 只记录

- `cpu` = `/proc/stat` 的 idle+iowait 差分比例（publisher 首次间隔 0.2s，之后用相邻两次观察）。读不了或格式错 → `null`，不 STOP。
- `JOB_SCOPED_CPU_ENFORCED=False`。CPU ≥20% 由 ops 准入检查。要开启：改常量 + 加一个测试。
- `Budget.require_stage` 只接收 ram/vram/disk。`RuntimeGate` 的新采样分支和缓存分支都传 sample 的 `stage_fractions`（只含这三个 key）。

## 账本副作用

job-scoped 的 `worker-resource-ledger.json`、`terminal.json.resources`、`*-resource-ledger` 里每条 sample：
- `free_fractions` 多 `cpu`（可为 `null`）；
- 多一个 `stage_fractions`（ram/vram/disk，送进 `require_stage` 的那份）。

legacy sample 字节不变。seal 只原样存 samples，不校验 key 集合。

## progress.json（P 映射）

- 设了 `VKGR_PROGRESS_PATH` 与 `VKGR_RUN_ID` 时，`run_controller` 写 `vkgr-progress-v1`（契约见 ops README）。`line='P'`；`stage` = manifest stage 小写、非 `[a-z0-9-]` 换成 `-`；`phase ∈ {awaiting-admission, worker-running, sealing}`；`epoch/max_epochs/completed/total` 恒为 `null`。
- 只设 `VKGR_RUN_ID` → 不写。路径非绝对 / 父目录不存在 / 是 symlink / 在 charged tree 内 / 写失败 → stderr 一行 `progress write disabled: …`，之后不再写，**不 STOP**。

## 已知残留（fail-closed，有意保留）

- `/proc` 条目 >8192 → `process inventory bound`（主机级失控，不是另一条线的正常状态）。
- publisher 自己的 fd >4096 → STOP。
- `nvidia-smi --query-gpu` 超时或失败一次 → publisher 跳出循环、发 terminal receipt 并关闭 → 整个 P 作业 STOP。与 legacy 一致，没有重试。G 起多个 eval worker 时 nvidia-smi 变慢的概率会上升。
- publisher 寿命上限 18h（`0<deadline-now≤64800s`）。跑得更久的 job-scoped P 会因 publisher 关闭而 STOP。INFRA-01 不改。
- 同 UID 可读的 rogue writer 仍会被抓；别的 UID 写不了 0700 的 feed root；root 不在防御范围（legacy 也挡不住）。
- writer inventory 在 job-scoped 下，某进程扫到一半遇到 PermissionError 时停止扫它；已经证实的写模式 fd 仍记入（偏保守）。

## 迁移规则

- 服务器上还有 legacy P 在跑时，不启动任何新的 G 或 P 作业（不论模式）。legacy `observe_g` 会把新 P 的进程算成 `p_writers`、把新 G 的 GPU 进程算成未知进程。
- 本文件改了 `retrieval_sasrec_remote.py`，它在 `source_pins` 里并进入模型 binding。legacy 代码 seal 的 P run 只能用 legacy 代码 verify；同一 slot 的 formal-train 与 formal-verify 必须 pin 同一份 `source_pins`；想用 job-scoped 重跑就重新 train。不要拿新代码 verify 旧 sealed run（会白烧 attempt/spent）。
