密级：公开

# 温度交叉熵：R1 工程交付与 owner 操作入口

## 新增文件

- `src/vkgr/ranking_r1.py`：同参数结构模型，仅 decode logits 温度缩放；温度 CE 数学 oracle。
- `src/vkgr/ranking_r1_run.py`：独立 schemas 的 train/verify；沿用 A1/T3 原训练、早停、best+last、selection replay 和 best 重新解码验证；绑定新 source/config SHA；48h 自身 wall guard。
- `src/vkgr/ranking_r1_data.py`：已验收数据通道与新源码 root 分离。
- `configs/ranking_r1/`：42、2024、2025 三份固定 .8 配置。
- `scripts/ranking_r1_ops.py`：Git/逐文件身份核对、真实只读 PF、真实 shape 合成 CUDA 预演与显存、dry-run、三 seed 发布、detached finite driver 自动 verify。
- `tests/test_ranking_r1*.py`：synthetic 验证；无真实数据或旧 checkpoint。

原源码、config、门禁均零修改。R2 与本 worktree 独立，本实现不读旧 CP，不碰 P/S1 作业，不访问 test。

## 主 owner 执行顺序

一次独立 code review → 本分支 commit/push → 服务器只读 fetch/指定 SHA Git worktree → 逐文件 sha256 → 原准入与资源检查 → `preflight` → `cuda-preview` → `dry-run` → 一次性 `submit` → `start-driver`。正式入队前告知 Eddy。任何 gate 失败 STOP；不重投 rejected，不覆盖失败 attempt，不自动补训练 seed。

工具运行命令（请求文件保留在私有 ops 目录，不进 public Git）：

```sh
<RUNTIME_PYTHON> -B scripts/ranking_r1_ops.py preflight --request <PRIVATE_REQUEST>
<RUNTIME_PYTHON> -B scripts/ranking_r1_ops.py cuda-preview --request <PRIVATE_REQUEST>
<RUNTIME_PYTHON> -B scripts/ranking_r1_ops.py dry-run --request <PRIVATE_REQUEST>
<RUNTIME_PYTHON> -B scripts/ranking_r1_ops.py submit --request <PRIVATE_REQUEST>
<RUNTIME_PYTHON> -B scripts/ranking_r1_ops.py start-driver --request <PRIVATE_REQUEST>
```

前两个动作输出单个 JSON 收据到 stdout；owner 用新文件保存，分别填写请求的 `preflight_receipt`、`cuda_receipt`。PF 内新 scratch 用来重核原 loader，不训练、不读 checkpoint、不重算/查看 test。CUDA 预演只对真实 shape 模型做两个合成 effective-batch 梯度 step，再在内存中保存/重载**自身新工程 state**并核 beam100 b1 解码一致；不读取旧模型。显存用 `max(allocated,reserved)` 的峰值 ×1.15 后向上取整256MiB；proof 必须与 PF 的 source、data binding、SID、runtime、model shape 一致才允许 dry-run。

请求字段：

| 字段 | 由 owner 填入的值 |
|---|---|
| `deployment` | 新代码 clean Git worktree 根目录 |
| `source_commit` | review 后已 push 并部署的完整 SHA |
| `files` | 部署根内相对路径→sha256；必须覆盖 runner.G6_MEMBERS 和新 ops 脚本 |
| `data_root` | 已验收 G 数据通道，沿原 frozen 文件/pins，不覆盖为新源码 |
| `python` | 原 runtime 的 Python |
| `templates` | 字符串 seed→原同 seed A1/T3 claimed-job JSON 路径；只读模板与config，不读旧 checkpoint |
| `pinned_manifest` | 新部署的 scheduler/source pin 文件 |
| `queue`、`runs`、`ops` | 原 scheduler queue、新 R1 runs 根、全新的 R1 私有 ops 目录 |
| `scheduler_ops` | 已版本化并已部署的 `vkgr_ops` 包父目录；只调用原 launcher.prepare，不修改门禁 |
| `preflight_receipt`、`cuda_receipt` | 本次新收据路径 |
| `deadline_epoch` | 绝对 Unix deadline，submit 时必须未来且不超过48h；plan 写入后不得重置 |

`submit` 在三个 run 均不存在时先独占写 plan，再逐个发布。若只发布了一部分时出错，STOP 对账，不重新 submit。没有自动重跑路径。训练 argv 继承原同 seed 所有数据路径和 SHA，仅替换新 runner/config/output/scratch，固定 `VKGR_DATA_ROOT`、batched b1、单 evaluator worker。verify 从已绑定的 R1 训练 manifest 派生，沿同温度模型与 b1，必须直接输出 verified 才验收。

`start-driver` 用 `start_new_session=True`、关闭 stdin、独立日志启动；driver 要求 PPID=1、持有唯一 lock、写 PID receipt，每900秒检查自己的三组 run，最多到既有48h deadline。失败/rejected/非 verified 写 ALERT；读取 rejection reason；不会 kill、restart、retry、改预算或查看外部进程。deadline 到期只写 ALERT，不能推断原作业已停。

## 本地验证与边界

命令：`PYTHONPATH=src:tests <RUNTIME_PYTHON> -m unittest tests.test_ranking_r1 tests.test_ranking_r1_ops tests.test_sid_depth_beam tests.test_generative_g6_unit`。

43 tests GREEN。覆盖 τ1 loss/gradient/AdamW/dropout+microbatch step 与原 CE 逐位相等、serial/batched beam100 一致、τ.8 forward/decode 只除一次、三个 formal configs 锁定、旧 schema 拒绝、新 checkpoint train→verify、data/source root 分离与 G3 路由、固定 b1、模板输入 pins 不变、verify 派生、queue rejection 去重、有限 deadline/显存声明、PF/CUDA 绑定及 detached 启动参数。初始 fresh RED 和后续新增 gate 的 RED 均已运行。

真实 loader、GPU/CUDA、server source/profile、full-shape 显存和 scheduler dry-run 尚未验证。旧集成 fixture 的 P3C 文档 frozen pin 漂移仍是已知限制；没有修改旧 pin 或声称全 suite 通过。新 runner synthetic integration 注入 typed synthetic context，仅验证真实训练/evaluation/checkpoint/verify流程，不能算真实 PF。
