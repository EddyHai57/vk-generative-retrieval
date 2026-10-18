# 融合 catalog 表示规范化：C 档独立 review 与 B 档闭环

密级：公开。依据 decision_log 已登记分支1；原核心、RRF、统计与科学门禁不变。

干净 context 独立 reviewer `resume_fusion_normalisation_review` 已核原代码、授权与规范化范围，结论 CHANGES_REQUESTED，仅四项测试覆盖缺口，未发现确定生产代码缺陷：删除输入 SHA、源码 SHA、result provenance、写后全字段对拍的非等价变异存活。禁止把这份初审称 APPROVED。

按 AGENTS §4A B 档，仅补负向测试，生产控制流未修改；主执行者逐条重跑 reviewer 的全部五变异。最终 baseline 十二测试 GREEN，五变异均 killed：四项上述保护及 preflight 失败仍进入 formal。全部新拷贝、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一缓存前缀，目录保留。未派无必要第二轮复审。源码只将注释中的 independent replay 改为 full replay，并补只读 preflight 状态日志，属 A 档措辞与日志修正，不冒称另一个统计算法。

独立 reviewer 与主执行者分别运行真实原 fuse/core 的 synthetic 集成：只替换 fixture SHA、query count 与 expected hits；原 RRF、bootstrap 10000 次及阈值不变；check-only → primary → full replay 返回 verified。主执行者亦核原 fuse/core 字节 SHA 与冻结卡相等。合并诊断套件共二十二测试 GREEN。此为 synthetic 证据，不替代真实服务器闸门。

实施范围：固定一份已验收的诊断 stdout SHA，逐个固定原输入 SHA；新目录生成四份表示规范化输入及 manifest，只改 catalog_digest 并增加 provenance，全部非表示字段保持原值；原 fuse 真输入 check-only 通过才执行唯一 primary，再完整确定性 replay；终态输出 verified、run_id、run manifest SHA、规范化 manifest SHA、原/新输入 SHA。旧文件、失败记录、旧入口与统计脚本不改。无训练或新实验名额，无 test/checkpoint 读取。

原 G 候选未二次解码、D-1 同类偏离、内容回退、结构 OOV 和早停右删失等限制继续适用。只报告数字与机械条件，不下效果或 anchor 结论。
