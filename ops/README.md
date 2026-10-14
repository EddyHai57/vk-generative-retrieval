密级: 公开

# ops/ — vkgr-launch / vkgr-sched / vkgr-status

Standard library only (Python 3.11), never imports `vkgr`. Design: INFRA-01 §4 (local `.agents/tasks/infra-01/design.md`).
Deploy `ops/` from its own clean worktree, separate from the G/P deploy worktrees.

## 先说风险（R1，迁移期）

- 服务器上还有 legacy-mode P 在跑时，不启动任何新的 G 或 P 作业（不论模式）。
- legacy P 会把新 G 进程当 unknown GPU process、把新 P 进程计入 `p_writers` 而 STOP。
- scheduler 不自动识别 legacy P（那要扫全机 cmdline），由 orchestrator 遵守。
- 新代码只给新 run 用；旧 run 用服务器上 pin 住的旧文件跑完。

## 根目录

`--root R` 或 `VKGR_OPS_ROOT`（必须给，不默认 cwd）：

```
R/runs/<run_id>/{run.json, owner.json, terminal.json, progress.json, logs/{supervisor,stdout,stderr}.log}
R/queue/{pending,claimed,rejected}/<UTC ns>-<run_id>.json   (+ rejected/*.reason)
R/queue/STOP              # vkgr-sched serve 看到即退出
R/admission.lock          # launcher 与 scheduler 共用，「准入 → 起作业」串行
R/scheduler.lock          # 同一时刻只有一个 tick
```

N8：作业输出（如 G 的 `--output`）放在 `R/runs/<id>/` 或其它部署 worktree 之外的路径。放在部署 worktree 里会让下一次 launch 的 `git status` 出现未 pin 的 dirty 路径，`real` 直接被拒。

## vkgr-launch

```
ops/bin/vkgr-launch --root R --run-id ID --kind real|smoke|ops --cwd REPO
  [--config P] [--seed N] [--input P]... [--pinned-manifest FILE] [--pinned-file REL=SHA|absent]...
  [--env K=V]... --cpu-cores N --ram-mib N --vram-mib N --disk-gib N --wall-hours H
  [--stop-mode signal|none] [--no-admission] -- <argv...>
```

- 校验 1–5（run_id/资源/env、git、pinned 清单、dirty 规则、input/config sha256）全部在建 run 目录之前、在拿 `admission.lock` 之前做完；任一失败 exit 1，stderr 一行原因。
- `real`/`ops`：dirty 路径必须全部 pin 住且 sha 匹配；`smoke` 允许 dirty（`run.json.git.dirty=true`）。
- porcelain 里已删除的路径只有 pin 为 `"absent"` 才算匹配（N3）。
- 准入拒绝 / 锁 30s 超时 → exit 1；owner.json 10s 内未出现 → exit 2（作业可能仍在启动，看 vkgr-status）。
- `run.json.env` 记 `{key: value}`（N6）：`--env` 记传入值；`OMP_NUM_THREADS, CUBLAS_WORKSPACE_CONFIG, CUDA_VISIBLE_DEVICES, PYTHONPATH, VKGR_*` 记当前环境值（含注入的 `VKGR_RUN_ID`、`VKGR_PROGRESS_PATH`）；不存在的 key 不写。

### pinned 清单（`vkgr-pinned-v1`）：原版 p3c doc 与 provenance/g0-v1 的 sha 校验

D0：原版 `docs/p3c_validation.md` 与 `provenance/g0-v1/**` 不进 Git，走数据通道放进部署 worktree，launcher 在启动前逐文件校验 sha256。

```json
{"schema":"vkgr-pinned-v1","base_commit":"<40hex>","target_commit":"<40hex>",
 "files":{"<rel path>":"<64hex>|\"absent\""},
 "overrides":{"<rel path>":{"sha256":"<64hex>","reason":"<1..200 chars>"}}}
```

- 顶层 key 只能是这 5 个（`overrides` 可缺）；路径必须相对、无 `..`、不是 symlink、resolve 后在 REPO 内；`files` 与 `overrides` 不得重叠。
- `base_commit == HEAD`；`target_commit` 必须在对象库里（服务器先 `git fetch`）。
- `git diff --name-only base target` 的每个路径必须恰好出现在 `files` 或 `overrides` 之一。
- `files`：target 树里有 → 声明 sha == target blob sha == 实际 sha；target 删掉的 → 必须 `"absent"` 且文件不存在；两棵树都没有（`provenance/g0-v1/**`）→ 只校验实际 sha。
- `overrides`：有意不同于 target 的数据通道文件（`docs/p3c_validation.md` 原版 `c89b6466…`），只校验实际 sha。
- `--pinned-file` 与清单重复声明时 sha 必须一致。
- 「运行的 commit」= `run.json.overlay.target_commit`；无清单时 = `run.json.git.commit`。

### supervisor 与停止

- launcher 用 `python -I <ops>/vkgr_ops/supervisor.py <run_dir>` 启动；`-I` 隔离 `PYTHON*` env，supervisor 自己把 `<ops>/` 放进 `sys.path`（M4：3.11 的 `-I` 隐含 `-P`）。`ops/bin/*` 同理，可以 `python -I ops/bin/vkgr-status …`。
- supervisor 自己 double fork（M3）：第一个进程立刻 exit 0 被 launcher 回收，孙进程 `setsid` 后当 supervisor。所以 supervisor 不是 launcher/scheduler 的子进程：scheduler 那个 run 的 RAM/VRAM 不含它拉起的作业，也不积累 zombie。
- SSH 断开不影响作业：supervisor 在自己的 session，忽略 SIGHUP（作业继承）；作业 stdin=/dev/null，只拿到 fd 0/1/2。
- `--stop-mode signal`（G 默认）：wall cap 到期或 supervisor 收到 SIGTERM → `os.kill(job_pid, SIGTERM)`，300s 后 SIGKILL。只发给作业 pid，不 `killpg`。
- `--stop-mode none`（P 必须用）：supervisor 永不向作业发信号；wall cap 只在 terminal.json 记 `wall_cap_exceeded:true`，SIGTERM 只记日志。P 停止走 P 自己的 `control/stop.json`。
- terminal 状态（N7）：supervisor 发过信号时取第一个原因（`wall_cap_exceeded` 或 `stopped`），否则 exit code 0 → `succeeded`，其余 → `failed`；exec 失败 → `launch_failed`。不自动重试。
- disk cap 不在运行中强制，只用于准入。

## 准入（launcher 与 scheduler 共用 `admission.admit`）

对 X ∈ {vram_mib, ram_mib, disk_gib, cpu_cores}：

```
reserved_X = max(declared_X, observed_X)        committed_X = Σ 占用中 run 的 reserved_X
settling_X = Σ 启动不到 600s 的占用中 run 的 declared_X（起点：owner.started_at，否则 run.created_at，N5）
admit iff (total − committed − need)/total ≥ 0.20  and  (free_measured − settling − need)/total ≥ 0.20
```

- 占用中 = `starting`/`running`/`orphaned`；已写 terminal.json 的不算。
- observed：RAM = supervisor 后代的 VmRSS 之和；VRAM = nvidia-smi compute-apps 与后代 pid 的交集（pid-ns 对不上 → 不用 observed）。
- CPU 按核（cgroup v2）：`total = min(os.cpu_count(), cpu.max quota/period)`；有配额用 `cpu.stat usage_usec` 1s 差分，否则 `/proc/stat` 1s 差分。
  - 服务器实测（2026-10-14）：`cpu.max = 1024000 100000` → 只有 10.24 核，`os.cpu_count()` 的 64 不能用。
  - 声明 `--cpu-cores` 超过 total（配额核数）直接拒绝。
- 拒绝分两类（`admission.unsatisfiable(snapshot)`）：任一资源的 `need` 连空闲主机都过不了余量规则，即 `(total − need)/total < 0.20`（含 `cpu_cores > total`），是「不可满足」，scheduler 走 `rejected/`；其余（负载、占用中 run 的 committed/settling、探针故障、nvidia-smi 不可用且 `--vram-mib > 0`）是「暂时不行」，scheduler 回 pending。
- nvidia-smi 不可用 → 只放行 `--vram-mib 0`；其它探针失败 → 本轮拒绝（可恢复，scheduler 下一轮重试）。
- G 的 `--vram-mib` = 父进程峰值 + W × 单 worker 峰值。单 worker 峰值由 §5.3 服务器对拍实测回填；未实测前不得用 W>1 跑正式作业。

## vkgr-sched

```
ops/bin/vkgr-sched submit <同 vkgr-launch 参数>   # 校验 1–5 尽早拒绝，结果不入队；O_EXCL 写 pending
ops/bin/vkgr-sched tick  --root R                 # 一轮；scheduler.lock 被占 → exit 3
ops/bin/vkgr-sched serve --root R                 # 每 30s 一轮，看到 queue/STOP 退出
```

- tick：按文件名 first-fit，只用声明量做无锁预判，每轮最多启动 1 个；`os.rename` 到 claimed 后调用与 CLI 相同的 `launcher.launch()`（重跑 1–5 → 拿锁 → 正式 `admit()`）。
- run 已存在或同 id 已 claimed、校验失败、正式准入拒绝且请求本身放不下（见「准入」）、锁超时 → `rejected/` + `.reason`，不回 pending；run 目录已建后失败 → `launch_failed`，留在 claimed。
- 正式准入只因当前负载、占用中 run 的余量或探针故障拒绝（`launcher.AdmissionDeferred`）→ 同名 `os.rename` 回 pending（保持 first-fit 位置），本轮结束，`note` 为 `deferred (left in pending): …`，下一轮重试；`vkgr-launch` CLI 的报错文本与 exit code 不变。
- 推荐常驻方式：`vkgr-launch --kind ops --vram-mib 0 … -- <python> -I ops/bin/vkgr-sched serve --root R`。

## vkgr-status（只读）

`ops/bin/vkgr-status --root R [--json]`：一行主机余量（GPU、RAM、CPU 配额核数 / 已用核 / 空闲比例 / 被限流次数与秒数、disk），每个 run 一行，再列 pending。不写文件、不加锁、不发信号、不调 git；只调 nvidia-smi（超时 5s）。拿不到的值为 `null`（表格里 `-`），exit 0。

`--json` schema `vkgr-status-v1`：`host` 含 `gpu_free_mib, gpu_total_mib, ram_free_mib, ram_total_mib, cpu_idle_fraction, cpu_total, cpu_used_cores, cpu_nr_throttled, cpu_throttled_seconds, disk_free_gib, disk_total_gib`；`runs[]` 含 `run_id, status, stage, completed, total, progress_age_seconds, wall_used_seconds, wall_cap_seconds, owner_pid, cpu_percent, vram_mib`；`pending[]` 含 `file, run_id`。

status：terminal 的 5 种；`starting`（无 owner.json、创建 <60s）；`running`（supervisor 活着且 birth 一致）；`orphaned`（supervisor 已死但作业 `(pid,birth)` 仍活，准入计入占用）；`lost`（其余）。

## progress.json 契约（`vkgr-progress-v1`，本节是唯一事实源）

```json
{"schema":"vkgr-progress-v1","run_id":"g-core-02","line":"G","stage":"evaluate","phase":"formal",
 "epoch":3,"max_epochs":30,"completed":640,"total":2520,
 "message":null,"updated_at":1760000000.0,"pid":12345}
```

- 必填：`schema`、`run_id`、`line ∈ {G,P,ops}`、`stage`（≤32 字符，`^[a-z0-9-]+$`）、`updated_at`（epoch 秒 float）、`pid`。`phase/epoch/max_epochs/completed/total/message` 可为 null；`message` ≤200 字符；文件 ≤4096 字节。
- 路径由 launcher 注入 `VKGR_PROGRESS_PATH=<run_dir>/progress.json`，`VKGR_RUN_ID=<run_id>`。只设 `VKGR_RUN_ID` 不设路径合法（不写 progress，N4a）。
- 写法：同目录临时文件 → fsync → `os.replace`。作业自己写（G、P 各一份实现），ops 只读。
- status 校验（N4b）：`schema` 不对、`run_id` ≠ run 目录名、超过 4096 字节 → 视为损坏，字段 `null`。
- G 映射：`train`/`verify-replay`（每 optimizer step，节流 30s）、`evaluate`（sharded 每发布一个 shard；串行只在 progress 事件）、`evaluate-best`、`complete`/`interrupted`。
- P 映射：`stage` = manifest stage 名小写、非 `[a-z0-9-]` 换成 `-`；`phase ∈ {awaiting-admission, worker-running, sealing}`；epoch/completed/total 为 null。

## G / P 使用规则

- G real run 用 `--pinned-manifest`（base `0376d9a…` + target = A 合入后的 SHA；`docs/p3c_validation.md` 走 `overrides`，`provenance/g0-v1/**` 走 `files`）。
- G 的 W>1（`VKGR_EVAL_WORKERS>1`）要求 permit 的 `runtime.threads==1`；新 permit 必须在 `OMP_NUM_THREADS=1` 下签。
- 冻结 helper `cache/g3-formal-invoke-v2.py` 用 `start_new_session=True` 起 verify 子进程，`--stop-mode signal` 的 SIGTERM 到不了它：跑 helper 时用 `--stop-mode none`，或接受 verify 子进程自己跑到 G 的 deadline。
- P 必须 `--stop-mode none`；`--wall-hours` ≥ P queue deadline + finish_seconds；P publisher 寿命 ≤18h（D-B7）。

## 测试

```bash
PYTHONPATH=ops <python> -B -m unittest discover -s tests -p 'test_ops_*.py'
```

`/proc`、`/sys`、nvidia-smi 用 fixture（`tests/ops_testlib.py`、`tests/fixtures/ops/fake-nvidia-smi`）注入；launcher/supervisor/scheduler 的端到端测试真实拉起进程（含 `-I`，不 mock Popen argv），Linux/macOS 都能跑。
