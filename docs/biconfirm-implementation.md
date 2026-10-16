# 双向序列模型确认：实现交接

密级：公开。基准：`seq/m1@04d4edeb84826d4d56387f95dc820fda1dce10b2`。科学规则仍以 design card 与 decision_log 2026-10-17 上午 §3 为准；本文不替代冻结卡或裁决。

新增 `retrieval_biconfirm_run.py` 复用冻结 M1 模型与训练/verify 算法，仅使用新 task/schema、mean/bi 两臂及 314159/271828/161803。原 M1 文件与原 42/2024/2025 白名单不改。非 seed 配置交原 H1 校验器验证，投影 seed42 仅用于旧 validator 的字段检查；实际初始化、游标、负采样及 manifest 永远使用本卡新 seed。

私有输入通过 `VKGR_DATA_ROOT` 明确指定 contract root，调用原 `validate_contract(root)` 与 `load_training_inputs`，不覆盖源码树的 contract 文档。inputs binding 增加 data root、contract manifest SHA、bound_files；source binding 仍绑定 clean Git 执行代码。data root 不改变特征、label、history 或 split。

`scripts/biconfirm_ops.py` 无固定服务器路径；部署与 scheduler 由主执行者负责。顺序：

1. `preflight --inputs <INPUTS> --inputs-sha <SHA> --data-root <DATA_ROOT> --output <PREFLIGHT>`：真实 engineering=False，核 86785/2520/120415 六配置与 data/source/init binding，先不训练。
2. `cuda-check --preflight <PREFLIGHT> --preflight-sha <SHA> --output <NEW_ENGINEERING_DIR>`：只生成新合成产物，两个 mask 的 train→verify；两臂各测 batch512/history50/catalog120415/query2520 完整训练、exact export/eval 与 g0 峰值。需要真实 CUDA；声明 `ceil(measured*1.15/256)*256`。CPU 测试不构成此项通过。
3. `plan --preflight <PREFLIGHT> --preflight-sha <SHA> --engineering <ENGINEERING_JSON> --engineering-sha <SHA> --request <PRIVATE_REQUEST> --output <PLAN>`：绑定上述原始证据，生成六训练全部无条件运行的固定计划。私有 request 字段为 root、inputs、data_root、runs_root、python、job_template、created_epoch；job_template 使用已审查 scheduler 资源/env/input 结构。created_epoch 只登记一次，48h deadline 不重置。
4. `dry --plan <PLAN> --plan-sha <SHA> --queue <QUEUE> --launcher-dir <LAUNCHER_PACKAGE_PARENT>`：核 clean source/config、原 run 与 verify 均不存在，逐个 `launcher.prepare`。不发布。
5. After the independent review, freeze, public scan, pushed commit, per-file deployment validation and gates above, the owner invokes `submit` with the dry arguments plus `--state-dir <NEW_STATE_DIR>`. The controller seals the entire plan, starts a detached driver, verifies its ready PID/session/plan hash, then atomically records release. Only that driver publishes the six fixed intents. Return status `driver_released_not_accepted` is not scheduler acceptance. Each publication has exclusive intent/result receipts. A partial failure writes `publication-ALERT`; later intents remain `not_attempted`, without retries. Already published trains retain their unique verifier owner. Job polling stays 900 seconds with the original 48h deadline. The at-most-30-second ready/release startup handshake is control synchronization, not job polling. Without release there are no publications.

任何共同数据/source/hash/顺序错误、非有限值、verify 失败都需 ALERT 并交主执行者/Kiro；不存在扩大旧 checkpoint、test 或其它实验的授权。driver 的合成/真实证据读取限本批新 manifest 和本批原始 queue/terminal/stdout 收据。

本地测试：`PYTHONPATH=src OMP_NUM_THREADS=1 <PYTHON> -m unittest discover -s tests -p 'test_biconfirm*.py'`；完整回归使用 `-m unittest discover -s tests`。seed 核查 refs、命令与服务器 ACK 见 `docs/biconfirm-seed-provenance.json`。最终执行 commit SHA 由主执行者在独立 review 后绑定。

Review HIGH lifecycle repair: fresh RED -> GREEN covers actual local detached processes, RPC/controller SIGKILL, an injected third-publication OSError, and driver spawn failure. Local synthetic queue/receipt fixtures simulate the scheduler; no server jobs or checkpoints are used. Partial publication stops new intents while already published trains continue to their unique verifiers.
