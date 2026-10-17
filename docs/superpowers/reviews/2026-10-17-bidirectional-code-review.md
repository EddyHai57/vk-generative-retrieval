# 双向序列模型确认与共享部署工具：独立 code review

密级：公开。结论：**CHANGES_REQUESTED**。一次独立只读 review；时间 2026-10-16 19:12 UTC（北京时间 2026-10-17）。Context 计数：UNKNOWN；未取得当前请求的可用计数。

## 范围与依据

- 双向确认 worktree：基线 `04d4edeb84826d4d56387f95dc820fda1dce10b2`，新增 runner、六配置、ops、tests 及实现/本地验证说明。原 M1 模型、runner 和配置相对基线 `git diff` 为空。
- 共享运维工具：phase2 worktree 的 `analysis/fresh_deploy.py`、`analysis/test_fresh_deploy.py`。
- 科学合同：主项目 decision_log「2026-10-17 上午」§3、上午续及双向确认卡。新 seed 历史核查 CLEAR 为主执行者提供的前置事实，本 review 不重复访问服务器。
- 未访问服务器、真实数据、旧 checkpoint 或 test；仅运行 synthetic/local tests。未修改实现、科学规则、状态事实源或 Git，未 commit。

## HIGH：先发布训练、后启动 driver，断线可留下无人接续的真实作业

位置：`scripts/biconfirm_ops.py:273–276`（双向确认 worktree）。

`submit` 先创建 state，再同步循环 `publish` 六个训练，最后才 `Popen(..., start_new_session=True)` 启动 driver。发布途中进程被 SSH 断线终止，或任一 `publish` 抛错，已经发布的训练仍可被 scheduler 接纳，但 detached driver 根本未启动。即使六个发布成功，日志打开或 `Popen` 失败也有同样结果。此时没有 driver 的终态 ALERT，成功训练不会自动补唯一 verifier；直接重跑 submit 又被 dry 的已有 run/queue 检查拒绝。该问题影响正式作业的 SSH 解耦和有界自动接续，不涉及模型判定规则。

本地 fault injection（临时 synthetic queue，mock dry/pin 以隔离提交生命周期）：第三次 publish 注入 `OSError`，前两次使用真实 `publish`。结果：`published=2, driver_started=false, final_alert_written=false`。没有发布任何服务器作业。

要求的 guard：使发布与接续由已独立于 SSH 的受控进程负责，首次发布前确认该进程已就绪；持久记录逐项发布状态及失败 ALERT。中途失败的恢复必须复用原 plan/deadline 和已有 run 身份，只接续已发布作业、不能重复 admission 或重跑失败 attempt。单纯在末尾增加异常日志不能覆盖 SSH 进程被终止的窗口。补本地 interruption/partial-publication 和 driver-start-failure 回归，证明已发布训练不会失去唯一 verifier owner。正式 submit 前关闭本项；不要求重做科学设计。

## 已核守卫与共享工具结论

- 新入口只接受三个新 seed 和 mean/bi 两臂，原 M1 白名单未放宽；配置继承旧 H1 的非 seed 合同。同 seed 初始化逐位一致、g=0 对应 anchor、mean 因果与 bi 历史内双向 oracle、target/PAD/candidate 隔离均有 fresh GREEN。
- 新 manifest/checkpoint schema；绑定配置、data、source、init；best/last hash、draw/selection replay、best exact re-export、g→0 重算和 mutation 拒绝有覆盖。私有 data root 走冻结 contract 校验，未覆写源码树合同。
- preflight 真数据规模检查、两臂 full-shape CUDA 路径、显存实测乘 1.15 后向上取整 256、六训练计划、唯一 verifier ID、已有 run 拒绝、900 秒轮询和原 48h deadline 在代码中存在。它们尚未在真实环境执行通过，不能用 CPU GREEN 替代。
- `fresh_deploy` 在 mutation 前重新 inventory；检查 pending/claimed/rejected/run、目标目录及 ops 目录不存在；固定工作分支 commit、新 detached Git worktree、逐文件 SHA、clean Git、独占 deployment 收据。无数据覆盖代码。对此工具未发现额外 HIGH/MEDIUM 阻塞；两个现有 tests 只覆盖 inventory/输入校验，未实跑远程 fetch/deploy。

## Fresh 验证与限制

以项目既有 smoke venv、`-B`、`OMP_NUM_THREADS=1` 执行：

- `PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_biconfirm*.py'`：17 tests，PASS。
- `PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_seqm1*.py'`：84 tests，PASS。
- phase2 worktree `PYTHONPATH=. python -B -m unittest analysis.test_fresh_deploy`：2 tests，PASS。
- 上述 partial-publication fault injection：确认 HIGH 问题，2 个本地 queue 项保留、driver 未启动、无 final ALERT。

未重复 full suite；实现方记录的 full 915 两项冻结合同失败及 untouched baseline 复现属于既有验证限制，本 review 未改 pin，也未将其宣称为本次 fresh 验证结果。真实 preflight/CUDA/显存/dry/admission/PPID=1 收据仍由主执行者在修复并通过工程闸门后取得。
