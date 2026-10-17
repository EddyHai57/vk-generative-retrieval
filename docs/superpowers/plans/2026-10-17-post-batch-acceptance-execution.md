# 本批后验补证与新SID A1确认 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline；只对用户要求的E pooling接合点做一次fresh独立review。

**Goal:** 交付待Kiro冻结的新SID A1卡、两组完整配对统计/已登记延迟、一次只读早停曲线诊断。
**Architecture:** 主owner维护事实源；新增只读分析工具复用原严格归档/配对实现，参数私有化；延迟只用已批准无checkpoint的合成值与真实长度分布，不读训练权重；大分析经1核队列。
**Tech Stack:** Python、NumPy、PyTorch、Git固定SHA、现有queue/kiro_b_run transport。
**Spec:** decision_log 2026-10-17 午间/午后；seq-m1-design§3.11/§8.1；新SID A1草稿卡。

## Global Constraints
不读test/未授权checkpoint；曲线诊断不读逐query预测；配对统计只读已验收validation预测；不训练/不改门槛；S5等Kiro冻结；R1/R2不加配置。

## Review Focus
归档与verify/manifest绑定；同query顺序和OOV；新seed池化定义不误用旧seed；曲线的R/N tie-break与右删失；延迟scope和初始化合成值不能冒称已训练端到端效果。

- [x] Task 1：读最新裁决；起草 docs/superpowers/specs/2026-10-17-coll-sid-a1-confirmation-draft.md；已按新增无人值守预确认FINAL，后续S5工程进行中。
- [x] Task 2：新增只读 paired/curve 接口与手算synthetic测试，RED→GREEN，原归档/配对函数逐字复用；Git/push固定版本，先metadata preflight核证据，再经1核queue补两组各三seed exact/p_pool/bootstrap。新工具路径 analysis/postbatch_acceptance.py，测试tests/test_postbatch_acceptance.py，隔离read-tools工作分支。
- [x] Task 3：仅E pooling替换m接合点一次独立泄漏review，报告主仓新文件；不改科学代码。
- [x] Task 4：读取四组各run的manifest/原日志epoch指标一次，保留每run曲线与best/stop；是否有3个不刷新best后再刷新分开R-only和R→N；早停后无数据记右删失，不推断延长训练收益；结论只写issue_log。
- [x] Task 5：沿已通过M1延迟脚本，新增mean/bi/stack对比；先合成fixture测试/固定SHA/正式只读PF，再队列测CPU1/GPU空闲且声明显存实测规则。若端到端要求超出已批准口径或必须旧checkpoint，先记UNKNOWN及精确缺口回Kiro，不自行扩大授权。
- [ ] Task 6：更新experiment_log数字带原始收据/UTC/SHA；STATE/schedule/project_map状态；handoff授权和下一步；commit。

14:01用户塔最终报告已验收；Eddy明确沿原users口径，端到端措辞不采用，未读取checkpoint。新增无人值守授权扩展已单独接收：S5优先工程实现/部署/真实PF中，S1 A–D与S3转述已完成，见STATE；不因本清单完成称科学训练已入队。
