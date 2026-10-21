# Popularity 有限衔接驱动实现

范围：独立 driver、tests、本报告；不改科学入口、已有 driver、冻结卡、task_list 或状态事实源。授权为 `6f8b56e` 第8–12条已冻结的无训练 Popularity validation 对照与主执行者的有界任务。作者自验不替代独立 C 审。

## 固定科学路径与运载

- consumer 固定 `18f98f56fc7211d549eaf296015f256d81691c5b`；`analysis/p_full_popularity.py` SHA256 固定 `1814f25f1bf08333e5c7a87ea301a619bfd6b6e5d80a3e676078ebef486c6a16`。
- OPS 固定 `5840e6b686ec9651314a9e28a1a2c0befcc05158`。driver 使用独立 clean Git 部署，以 CLI 显式钉住自身 commit/SHA、request SHA、inputs SHA。
- `preflight/evaluate/verify` 对应原 request 的 `preflight/train/verify`。null predecessor；三个不同的 `p-full-popularity-` run IDs；独占 evidence，不修改 request/config/resource。首次资源声明由 owner 按真实余量填写并标记 `not_measured`；此实现没有资源估数。
- 每阶段调用原模块正式 `plan` CLI，然后真实 `launcher.prepare`，把完整 metadata、job argv/environment/source/ordered inputs/resources 与原 expected_job 比较，保存 dry 回执，最后才调原 `vkgr-sched submit`。scheduler stdout 按原始文本独占保存，不误作 JSON。
- 同一阶段最多提交一次；已有 pending/claimed 不重投；rejected、消失、重复 queue ID、失败 terminal、job/spec 不符、原 accepted 拒绝均 ALERT。仅 claimed 且不在 rejected 的成功终态可继续。等待间隔900秒，截止最后一次等待可缩短到剩余时长，醒后立即 deadline 失败；不 tick scheduler、不控制其它作业。
- 每阶段成功后在有剩余 deadline timeout 的独立 subprocess 内调用原 `p.context` 与原 `p.accepted`，按顺序核 PF、evaluate、verify 原始 stdout/terminal/job/receipts/input/DAG；最终 verify 还必须 `scientific_verified is True`。PF 无指标，不等待 M1 side，也不阻塞 M1。
- PPID 必须1；总 deadline 为正、有限且不超过200小时。所有 subprocess（含 Git）均用剩余 deadline timeout；源码扫描/hash每步查 deadline。Git archive 的完整普通文件集合与实际部署逐位核对，实际只排除根 `.git` 管理项；ignored Python/bytecode、symlink file/dir、遍历异常均拒绝。每轮及成功前后重核 driver/consumer/OPS/request/input 闭包。
- 请求内全部绝对路径加入 evidence 不重叠检查；文件用 x/xb，evidence mkdir 不复用；不覆写旧输出。无 SSH、模型、训练、删除、重试、付费、test 入口。

## 验证

fresh RED：12项在 driver 不存在时失败。最终 driver20项、原 Popularity19项、backend/helper71项，共 **110 PASS**（13.24秒）。

- 正式 dry/非法命令子进程、deadline bool/NaN/Inf/范围、实际 subprocess timeout、scheduler 文本stdout、ignored bytecode/symlink、queue重复/rejected/terminal类型、evidence重叠。
- 三阶段成功、pending消失、accepted失败、terminal失败、deadline超时：各阶段仅一次submit，失败不进入evaluate，summary ALERT。
- 真实原launcher synthetic E2E：使用独立原Git/OPS/data-channel合成布局，正式Popularity plan → driver完整prepare/dry比较；故意改expected env失败。无mock launcher、无submit、无训练；不替代真实服务器PF。
- 原accepted调用顺序与最终scientific_verified负例通过。
- **4/4隔离semantic mutants killed**：移除deadline、移除完整源码集合比较、放宽exit_code类型、移除seen/sent消失保护。每项独立新拷贝、python -B、PYTHONDONTWRITEBYTECODE=1、唯一PYTHONPYCACHEPREFIX；无import/语法失败作为kill。

本地日志为临时私有证据，由owner收拢；未接触服务器、未push、未部署、未实际提交/训练，未改passes。独立 C 审与真实资源/运行验收待owner执行。

## 使用

```text
python -B <DRIVER_CODE>/ops/tools/remote/p_popularity_finite_chain.py dry
python -B <DRIVER_CODE>/ops/tools/remote/p_popularity_finite_chain.py run \
  --inputs <INPUTS> --inputs-sha256 <INPUT_SHA> \
  --request <REQUEST> --request-sha256 <REQUEST_SHA> \
  --consumer-code <FIXED_POPULARITY_CODE> --ops-code <FIXED_OPS_CODE> \
  --driver-source <DRIVER_COMMIT> --driver-sha256 <DRIVER_SHA> \
  --evidence <NEW_EXCLUSIVE_EVIDENCE> --deadline-hours <FINITE_HOURS_LE_200>
```

owner使用已有detached部署方式，PPID=1。只读PF可独立开始；不以本报告代替实际原stdout与绑定验收，不给科学结果解读。
