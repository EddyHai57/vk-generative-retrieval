# 元数据桥一次独立 review

结论：**CHANGES_REQUESTED**。对象 `62b6d801788ad99b8477f78d63cffb455c582c0f` 的 `analysis/ops_metadata.py` 与 `analysis/test_ops_metadata.py`；只检查新增 bridge，不重审已 review 的模型、controller、driver。未连接服务器，未读取真实数据、test 或 checkpoint。context 计数 UNKNOWN。

## HIGH：实际执行程序和导入环境未绑定到 reviewed 入口

位置：`analysis/ops_metadata.py:37–38`。

`check()` 只绑定 deployment HEAD 与入口脚本 SHA，随后实际运行 `r['python']`，且允许请求任意覆盖环境变量。任意 executable 都能忽略 `-B /reviewed/script plan` 并执行其它工作，因而 metadata action 白名单不能保证不在连接内运行 train/preflight/CUDA。`LD_PRELOAD` 等变量也能使 reviewed 脚本启动前执行未绑定代码。

Synthetic 复现：stub `check()` 返回临时 deployment（隔离 Git/部署检查），将 `python` 指向临时 shell executable；它只写一个临时 marker 并输出 JSON。bridge 返回 `metadata_command_ok`，marker 存在，实际未执行 reviewed 脚本。`validate_operation()` 不检查 interpreter/env，真实 HEAD/SHA 检查不会阻断此分支。

建议：解释器使用部署合同固定的可信绝对路径/身份，禁止请求选择任意 executable；环境只放已知必要变量，拒绝注入动态加载与 Python 导入配置。新增拒绝非合同 interpreter、危险 env 的 synthetic 回归。

## MEDIUM：JSON 读取路径检查可越出授权目录

位置：`analysis/ops_metadata.py:27–30`。

`Path.is_relative_to()` 是词法检查；`ops/../outside.json` 被判为 ops 内。`p.is_symlink()` 只检查末级文件，`ops/linked/outside.json` 在 `linked` 为目录 symlink 时同样通过。两种路径均已用临时 synthetic JSON 成功读取边界外值。当前测试 `test_no_path_escape` 只覆盖 script 字段，未覆盖 `read`。

影响：bounded JSON/read basename 过滤不能阻止读取授权 ops/queue/run 目录外的 JSON，也可访问其它 run 元数据。建议先规范化并核验候选实际路径相对于规范化授权根的包含关系，同时拒绝路径链中的 symlink；读取也应维持实际读取字节上限。新增 `..`、父目录 symlink、其它 run 三项拒绝测试。

## 已核与验证限制

- `seal` 用 `open('x')` 创建文件，已有目标不会被覆盖；失败可能留下部分新文件，调用方须保留并停止，不能当成整批原子提交成功。
- `read` 只查询 PID/PPID，未见 kill/终止作业逻辑；`run` 有 55 秒 timeout，会终止该 metadata 子进程。本 review 未证明部署调用全程低于此上限，也未验证 SSH 断线行为；已 review driver 的 ready/release 不在本范围。
- `python3 -m unittest analysis.test_ops_metadata -v`：2 tests PASS。
- 额外临时 synthetic probes：`..` 读取、父目录 symlink 读取、替换 executable 均复现上述行为；未修改 source，未运行正式脚本或提交作业。

本文件只记录一次 review；修复后由执行者 fresh RED → GREEN → 回归验证，不新增整套重复 review。
