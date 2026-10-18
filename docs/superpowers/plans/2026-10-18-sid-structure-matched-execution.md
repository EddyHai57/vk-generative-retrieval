# SID 结构匹配实验：工程执行计划

密级：公开。科学内容仅来自 `2026-10-18-sid-structure-matched-card.md`；不补设计、不改门禁。

## 身份与范围

复用已收口且干净的隔离 worktree，在 `a5da4d6d0746ef41ca1ed93028971360dcb89547` 建立 `codex/sid-structure-matched`。原入口、旧白名单及所有既有文件字节保持不变；只新增模块、测试、配置及版本化运维工具。当前规则以 MAIN 的有效 AGENTS 和 Eddy 本次指令为准，旧基底的历史 owner 文字不恢复。

## 顺序与验收

1. 新增 `sid_structure_v1.py`：新 bundle/schema/version；复用原 C1 输入验证及原审计算法，branching 为冻结值。新身份禁止混入旧 loader。H/B/R 按旧原文的适用对象，不改原函数；本卡显式要求的三层利用率门槛必须另行严格核对，失败即 FIT_FAILED。
2. 新增 `sid_structure_run.py`：调用原 `sid_coll_run.fit_bhkm_coll`，只改已冻结分叉；同输入、seed、max_iter、suffix 不变。先真实只读输入 preflight，再拟合、独立加载与同 seed 全量 replay；清单逐项绑定源码、配置、数据与完整输出 SHA。
3. 新增 `sid_structure_a1_run.py` 与三个配置：沿原 A1 训练/selection/解码实现，只有 SID/schema 身份及必须适应新词表的合同字段变化；正式输入固定原 C1 向量。不得加载旧 checkpoint；新作业只 verify 自己的新产物。
4. TDD：先 fresh RED，最小 GREEN，原 C1/A1 相关回归；覆盖未知身份、错 config/seed/输入 SHA、三层利用率不足、同 seed replay 不同、旧白名单不变、未验收拟合不得下游提交、重复 run/rejected/失败不得重投。变异用新拷贝、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一前缀。
5. 新增运维入口：先只读确认新 run_id 不存在，再 dry-run。正式作业前一次干净 context 独立 code review；C 档变更按 §4A 复核。扫描压平 diff，push 工作分支，固定 SHA 新目录部署，逐文件 SHA。
6. 拟合与 fit-verify 原始 verified/绑定通过后，三配置真实 engineering=False preflight、CUDA 工程预演与真实 shape 测峰。峰值为自身进程加 context，声明乘余量后向上取整；首探针保守预约须标 not_measured。无测量不写已实测。
7. 全闸门通过后告知 Eddy，三个已批 seed 一次性提交，逐个确认 claimed 且无 rejected/.reason。服务器侧版本化 verify 驱动 PPID=1、间隔不低于规则、有限截止，不调用模型；只补唯一成功训练对应 verify，不重复发布，不覆盖失败。
8. 原始 verified 与 job/manifest 绑定齐后，记录 item 指标、best/stop、前缀粒度限制和原热度分档；只作冻结条件机械核对，交 Kiro/Opus，不自动换 anchor。

## 失败与下一批

任何阶段失败保留全部记录，停止该支并 ALERT；共同数据失效、NaN/Inf、设计偏离回 Kiro。不为填空新增名额，不改原 R2。可学习 SID 与规模成本基准按各自已批准顺序准备，原预算和截止不重置。数字只进 experiment_log，状态同步 STATE/project_map/schedule。
