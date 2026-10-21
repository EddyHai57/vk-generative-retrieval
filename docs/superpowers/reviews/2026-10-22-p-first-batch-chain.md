# P 首批单 seed 有限运载 driver

依据 decision_log `6f8b56e` 的条件入队边界与 owner 指令。C 档：新增正式队列写入与前后继绑定路径；独立 C review 必须通过后才可采用。本次仅新增 driver、测试与本报告，不修改科学代码、配置、原 consumer、DAG、OPS launcher/scheduler；未连接服务器、启动作业、读取 test 或 checkpoint。

## 固定部署与入口

入口为 `python -B -m ops.tools.remote.p_first_batch_chain dry|run`。`run` 必须传入 `--inputs --request --request-sha256 --ops-code --consumer-code --evidence --deadline-hours --driver-sha256`。一个 driver 只处理原 PF request 的一个已登记 seed（42、2024、2025）；执行者明确启动三实例，driver 不决定是否满足科学入队条件。

采用独立干净 Git 部署放 driver，原已部署 consumer 通过 `--consumer-code` 指定；不重做 PF，不把 driver commit 与 consumer commit 强等。driver 每次轮询检查自身 Git commit、tracked script 原字节与显式 SHA；consumer source/runtime 则独立与原 request 严格核对。import 优先级为 consumer、consumer/src、固定 OPS/ops，cwd 为 consumer。evidence 必须是全新且不与 driver/consumer/native/stage2/output/OPS 重叠的目录。summary 分别记录 driver_source 与 consumer_source。

复用项目现有 `approved_input_chains.py` 的有限链、原终态与只观察已有作业的运载原则；直接调用已登记 OPS（`5840e6b`）的 launcher/parser/prepare 和 scheduler.submit，不复制 scheduler 或科学实现，不新增通用框架。

## 机械链与失败行为

先用原 helper.harvest_pin 与 DAG verify_producer 验证成功 PF 的原 stdout/manifest/job/terminal；完整核对 stage receipt 语义、原 request SHA、own-job contract、输入 bytes/hash、source/runtime 与 binding。train request 只改变 predecessor；已有 run/queue 复用原 request，hash 与内容均须一致。

每阶段 backend.plan → 真实 launcher.prepare → 完整 expected_job 相等后，只有 run 与队列均不存在才原 scheduler.submit。排队 spec 必须与原 spec 完整一致；已有 pending/claimed 只观察，rejected 读原 reason 后停止，丢失队列不重投。写队列不称 claimed；只有实际 claimed 且成功 terminal 才能收割原验证收据。train 成功后才进入 verify，任一失败写 ALERT/no retry。收据结束前再次验证 DAG，防止检查期间输入或 stdout 漂移。

固定轮询 900 秒；须 detached PPID=1；deadline 必须有限、正数且不超过 200 小时；原 train wall 仍为 96 小时。成功 summary 仅表示链完成机械核对，scientific_interpretation=false，不宣称有效、转正或 anchor。

## 验证与限制

- fresh RED：新模块缺失导致原测试失败；随后最小实现 GREEN。作者变异暴露 request hash 掩盖 delta 断言的测试缺口，新增合法 hash 但 config 漂移的反例后修复测试；未改实现。
- 新 driver 16 项测试覆盖：正式 CLI、有限截止、900 秒轮询、原拒绝/重复队列、predecessor-only、已有 claimed 的原 request pin、完整 dry job、source 分离、driver byte pin、成功/拒绝/超时状态机与原 DAG/helper 末端篡改拒绝。
- 真实本地 synthetic 部署测试通过原 backend `-m ... plan`、真实 launcher.prepare、真实 scheduler.submit；只写临时 synthetic pending 队列，核对 claimed=false、无 run 目录，未 tick/launch。
- 相关 driver/backend/helper/reference 回归：106 passed。首次 sandbox 执行因 OpenMP SHM 系统限制失败；相同命令在本地允许环境重跑通过，未更改测试或科学输入。
- 作者隔离变异 6/6 killed：完整 job 比较、rejection、最终 DAG 证明、request delta、轮询下限、driver SHA。每项新源码拷贝、python -B、PYTHONDONTWRITEBYTECODE=1、唯一 PYTHONPYCACHEPREFIX；无等价或存活变异。
- 本地验证不能替代服务器真实 PF、原对拍与条件入队裁决；这些由主执行者按原授权核查。独立 C reviewer 报告另存，不以作者自验代替。

## 独立 C finding 修复

独立 reviewer 复现：初始已观察 pending，真实 prepare 后原 queue/run 都消失时，旧实现会误当新 job 重投。现记录阶段开始时原 queue/run 是否已存在；若曾存在但在写入前消失，立即 STOP，不调用 submit。fresh RED 为 `DID NOT RAISE ValueError`，修复后17项 driver测试通过；消失队列回归还明确断言零 submit、零 sleep、未进入 train。该改动属C档，交原独立 reviewer 复验，不以自验代替。完整相关回归107 passed；作者隔离变异增至7/7 killed，新增“删除曾观察原队列保护”变异。上节16项/106项/6变异为修复前原始记录，保留不覆盖。
