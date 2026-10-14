密级: 公开
# ops/tools/ — 部署与运维工具（DEPLOY-01 / 01b）

硬规则 1：服务器上实际跑过的运维脚本都在这里，服务器只用已 push 的 commit。路径一律走配置，不写死本机路径。

## 三类脚本
- `local/`：在本机跑。`--config C`（或 env `VKGR_TOOLS_CONFIG`）给出 ssh Host 别名、收据目录和服务器路径。都有 `--dry-run`。
- `remote/`：经 `local/transport.py` 送到服务器执行（`python -c` 或 stdin），不能带 argv，服务器路径从开头注入的 `CFG` 读（`toolcfg.render`）。`*_read.py` 只读；其余是 writer，只建新路径（`ops_update.py` 例外：把已有 ops checkout fast-forward 到已 push 的 commit）。
- `server/`：在服务器上经 `vkgr-launch` 作为作业跑，路径全是参数，都有 `--dry-run`。

## 配置
复制 `config.example.json` 到仓库外（不入库），填真值。`state_dir` 放收据与 `ssh-control-state-formalNN.json`。

## 常用命令
```
local/hold_mux.py   --config C --n N                    # 一个有限期前台 SSH master（替换 formal{N-1}）
local/transport.py  --config C --n N remote/X_read.py PREFIX --readonly
local/transport.py  --config C --n N remote/launch.py PREFIX --input payload.json
local/deploy_spec.py tree --repo . --commit SHA --branch infra/01-ops --expect-old OLD --out spec.json
local/r3_submit.py  --config C --manifest M --n N --step sign|train|parity --confirm-reviewed
```

## 来源（DEPLOY-01 收据目录里的原始文件 → 这里）
| DEPLOY-01 原文件 | 现在 | 改动 |
|---|---|---|
| transport_mux_n.py | local/transport.py | 配置化、render CFG、--dry-run |
| hold_mux_n.py | local/hold_mux.py | 配置化、--dry-run；去掉只属于旧 P recovery01 的检查 |
| deploy01_r3_submit.py | local/r3_submit.py | manifest v2（W 训练 + 独立对拍作业）、--dry-run |
| （DEPLOY-01 内联生成） | local/deploy_spec.py | 新：tree / g-overlay spec |
| g_permit_tool_v1.py | server/g_permit_tool.py | 逻辑不变；registration 参数化；--dry-run |
| g_parity_tool_v1.py | server/g_parity_tool.py | 改为只解码作业、自己的 pool 和 decode 预算；进程内两次 evaluate |
| deploy01_*_remote.py（detach/g_deploy/launch/upload/ops_deploy/p_deploy/privtmp_shim） | remote/<同名去前后缀>.py | 路径改读 CFG；launch 加 submit 模式（DEPLOY-02） |
| deploy01_*_read.py（probe/r1/status/runlogs/poll/gtest_errors×2/oldg_inventory/oldprep/seed2024_state/seed_terminal） | remote/*_read.py | 路径改读 CFG；r1 进程命令行脱敏；gtest_errors 两个合并 |
| deploy01b_probe_read.py / deploy01b_cpu_read.py | remote/tools_probe_read.py / cpu_read.py | 新 |
| — | remote/ops_update.py | 新：ops checkout fast-forward + 逐文件 sha |
