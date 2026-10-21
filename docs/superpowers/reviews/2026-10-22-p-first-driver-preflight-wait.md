# P 原预检等待衔接

依据6f8b56e第5–6、8–12条。只补既有有限driver的运载等待，不改consumer、科学函数、配置、seed、wall、原PF或scheduler。

`run --wait-preflight`可在原PF仍运行时启动。只有该PF已claimed且原run.json存在才等待；900秒轮询，共用原200h总截止，不另开等待预算。每轮重核driver/consumer/runtime/OPS身份、原request完整值和原job合同。原queue失踪/重复/rejected、terminal失败或超时、身份漂移都停止。等待器不提交任何job，不更改task_list或科学资格。

PF自然成功后，仍必须通过原`complete`的stdout/receipt/job/terminal/source/config/runtime/input/DAG闭包，再按原plan→真实launcher metadata/dry→原scheduler submit训练与verify。没有跳过、重投或修改原PF的路径。owner仍负责卡75配置值、子集原参照逐位、p3c、holdout负例、评估准入、实际资源和checkpoint后盘≥20%等既有前置；task状态在真实验收后更新。等待driver运行不等于backend已通过或训练已提交。

Fresh RED：新增等待、顺序和正式argv测试在旧driver明确因缺wait_preflight/未识别参数失败（缓存p-first-wait-red-v2.log）。最小GREEN：tests/test_p_first_batch_wait.py、test_p_first_batch_chain.py、test_p_full_engineering.py、test_p_full_backend.py合计100 passed/17.43s。覆盖原已审driver及真实launcher→metadata→dry/queue路径；不是服务器真实PF成功证明。

独立C复核及新隔离变异另附；通过前不部署或启用本等待增量。旧driver/部署/失败记录保留。新public runner只需独立driver树，不替换consumer，不迫使重做真实PF。

原科学卡及容量限制继续见2026-10-22-p-full-first-batch.md、2026-10-22-p-first-batch-offline-transport.md和mechanical-start-check.md。三seed训练各96h，verify另计；200h包含资源等待，不保证三seed完成。任何运行时资源不满足仍由原准入与守卫拒绝，不改变余量或删除cache。
