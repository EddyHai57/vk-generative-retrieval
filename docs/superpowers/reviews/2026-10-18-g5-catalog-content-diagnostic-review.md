# 融合 catalog 内容诊断：独立工程 review

密级：公开。依据 decision_log「G5 catalog 摘要不一致的诊断授权」。无科学判读。

独立 reviewer：干净 context subagent `resume_catalog_diagnostic_review`；结论 APPROVED，原代码 SHA256 `f5a255cb9dc1e0f04d8f5997d537c86a2da758b14918929d824c8446a00708eb`。主执行者已复核该 SHA 与原 loader / producer 摘要来源。

发现与关闭：scratch 限定到本新 run 固定子目录并核 run.json 身份；不读完整装配预测，仅读钉 SHA 的 producer stdout；保留原部署、cwd、解释器与正式 loader；实际 catalog 必须复现原摘要、数量匹配冻结合同；未知摘要算法 STOP。完整 catalog 仅在同一作业内存与内部 pipe 使用，不写入结果。报告只含聚合和最多五处差异索引，原收据及私有路径不入 Git。

review 后只新增调用路由测试，生产源码未变，按 AGENTS §4A 为 B 档：补测试钉住复核者发现的缺口，自验逐项重跑，不派第二轮 review。十项测试通过；主执行者在新拷贝、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一缓存前缀下重跑六变异：取消输出上限、颠倒集合判断、允许 bool、取消 LF 原摘要分支、取消 scratch 限定、改回整份 metadata JSON 读取，全部 killed；baseline GREEN，无存活变异。独立 reviewer 亦使用隔离缓存核过其原变异。证据映射保留在本机私有缓存；代码和测试进 Git。

范围限制：本报告不替代服务器真实输入诊断、部署 SHA、队列准入和原始终态收据。尚未产出真实 catalog 比较结论，原失败 fuse-pf 不重投。
