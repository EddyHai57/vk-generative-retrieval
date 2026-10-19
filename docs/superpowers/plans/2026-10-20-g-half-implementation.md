# G 训练用户减半实现计划

状态：实现中。冻结科学口径以 `docs/decision_log.md` 第五节为准；此计划只记录工程接线，不扩写实验合同。

## 目标与边界

新增 STRUCT A1 的用户级训练样本过滤 runner、只读 formal preflight、verify stdout 绑定、三份 seed config 与运维校验。原 A1 runner、旧 ops、SID 构造、validation evaluator、anchor 与项目事实源均保持字节不变。不得运行真实训练、访问 test、改写 checkpoint、push 或部署。

训练用户池定义为实际训练 examples 中至少出现一次的 `user_id` 集合，按整数排序。冻结选择数为 `floor(U/2)`，选择集合使用独立的 `random.Random(20261020).sample(sorted_U, k)`，结果再升序规范化；其SHA采用 compact UTF-8 JSON 数组且无末尾 LF。三个训练 seed共用这一集合。每位入选用户的全部训练 examples 按原顺序保留；不改变 codec、catalog、SID bundle、query/history、selection 或 A1 超参。

## 允许新增路径

- `src/vkgr/g_half_users_a1_run.py`
- `scripts/g_half_users_ops.py`
- `scripts/g_half_users_engineering_queue.py`
- `tests/test_g_half_users.py`
- `tests/test_g_half_users_ops.py`
- `tests/test_g_half_users_engineering_queue.py`
- `configs/g_half_users/g-half-users-s42.json`
- `configs/g_half_users/g-half-users-s2024.json`
- `configs/g_half_users/g-half-users-s2025.json`
- 本计划文件

## 设计与验收

runner 在旧 A1 context 已校验后只替换训练 examples wrapper，保留 validation queries、codec 和全量训练切片统计对象。新 manifest 绑定抽样 seed/method/count rule、用户池与选中用户数、原/选中训练行数及脱敏 SHA；原 evaluator 的完整 best epoch 预测 shard、聚合 metrics、best/stop 证据继续留在原 manifest/输出结构中。

只读 formal preflight 复用旧 request/input/fit-verify gate，检查 anchor seed42 manifest 的原始 SHA 和 `context_binding_sha256`，再以原 full context 为基准确认 sampled context 的 canonical validation query/history 逐字节与 validation source file SHA 完全一致；同时确认 `engineering=False`、full-train STRUCT_C1_B48D3_S42、原 catalog/query/example shape、固定 config 和 source/runtime pinned delta。verify stdout 必须含 `status=verified`、训练 `run_id`、训练 `manifest_sha256` 和 verify job id，并可由 ops 再读原 stdout hash核验。

源码身份证明分两本：科学训练 source 从已接受 anchor manifest 的 41 个 source files 派生，加冻结的 10 项 delta，精确为 51 项；ops/launcher request 从原接受 request 字节（SHA `c6995e313011f69f7acdd3e2af1205ae8f1b71a4fa4e0cd996d1fa77fe07dd55`、source commit `b5a4d1c99ba2d8361a2d90c23bac27af5b71769d`）重读 45 项，再加同一 10 项 delta，精确为 55 项。原 seed42 anchor manifest SHA 固定为 `2f6744f4cafae340666410b87912b8cf1c71d9447c55b740f1280f2c52afefae`。部署 Git tree 还须相对原 source commit 证明旧tracked bytes没有变化、只增加这 10 项路径。

相关测试覆盖用户池/抽样确定性、全量用户样本保留、奇数用户显式取整、context 参数位置、validation anchor binding、精确 source-delta集合和 verify stdout绑定。完成时执行 fresh unit/regression 和隔离变异验证；formal real-data preflight、review、push、部署、入队与真实训练由主执行者安排。

## 冻结抽样规则

Kiro 于 2026-10-19 晚（续二）裁定：用户全集来自训练 examples 用户，`k=floor(U/2)`；使用 `random.Random(20261020).sample(sorted_U,k)`，三个模型 seed 共用同一用户集合；集合SHA为升序整数列表的 compact UTF-8 JSON 字节，不带末尾 LF。
