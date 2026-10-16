# SID-COLL-S1-EXEC-20261016 执行计划与ledger

科学合同：S1 FINAL卡（文件名保留freeze-draft以维护既有引用）及decision_log夜间SID-FIX§4/§5。唯一执行与服务器owner Codex；Kiro仅科学裁决。此计划不修改阈值/臂/seed。

## 范围与验收

隔离worktree <WORKTREES>/sid-coll-s1/vk-generative-retrieval，branch codex/sid-coll-s1，base c1c4b7413b4ae794894fc78b98828ab26029a487。只新增science modules、6 G/2 fit配置、tests、ops/driver与冻结文档；旧loader/白名单/run/bundle不改。S0补充只新增4个analysis文件在原S0worktree。不删除文件。

1. SGNS64：SUM loss、5epoch、固定窗口/采样/线性LR、fallback/健康检查，fresh RED→GREEN。
2. C2固定teacher source与best SHA，weights_only导出P64＋content64；新H5a/b、64/128 BHKM同seed replay、独立新loader/G入口，synthetic正式合同及负向验收。
3. 运维：新入口、真实只读PF、CUDA/fullshape测峰、dry、有限48h server driver；每配置独立流水，s42 raw verified且hits≥260才补2024/2025。唯一train/verify，失败/拒绝不重投。声明ceil256(peak×1.15)，CPU总准入≤8.19。运维源全部入Git。
4. S0 D′、旧归档别名、A1B1024 epoch补充只读；不读旧CP，不把UNKNOWN补成通过。
5. 一次完整fresh-context code review（science＋ops＋S0补充）；工程finding fresh RED/GREEN/回归，不追加设计评审。改科学设计HIGH/MEDIUM立即停回Kiro。
6. 扫描全部将推送代码/历史、新固定SHA部署、逐文件hash；每配置输入/fit/fitverify→三seed真实PF→CUDA/测峰→dry全过，入队前告知Eddy再提交s42，RO核claimed/no rejected。原始verified与binding核对，机器汇总/事实记账。

## Ledger

- 接收：已读夜间Opus唯一设计与执行授权；context当前计数UNKNOWN；现有Kiro工作流均结束，不恢复旧job。
- 初始RO：main05/sidcoll-initial-queue-20261016T103741Z.stdout；pending0，只有scheduler运行，GPU无训练占用。SHA/实际读取时间见experiment_log后续收据；不据前缀冒充读取时间。
- 授权：Eddy明确允许新作业verify读取自己新产物；旧CP仍仅M1mean42best。H5a/b为新输入族已冻结合同，不改旧GTOK门禁。
- 实现：SGNS/新合同/C2短任务在互不相交新增文件实现；唯一服务器owner不变。独立review尚未开始，未提交S1科学作业。
- 待验收：ops/driver、真实数据source-lock、全套回归、whole review、push/deploy、真实门禁与队列。
