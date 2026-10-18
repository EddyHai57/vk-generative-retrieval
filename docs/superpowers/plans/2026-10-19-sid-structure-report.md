# SID结构匹配只读报告实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复用S0固定算法补结构三seed相对原同seed共现SID的描述性报告，保留全部源码/输入/verify绑定。

**Architecture:** 新增只读runner，复用原字节sid_diag、sid_diag_metrics、phase2_validation_read加载已验收best归档；新版本化队列入口只发布一个preflight和一个报告。部署/原始回执沿既有结构ops，不新增训练/driver/模型timer。

**Tech Stack:** Python、stdlib、已有NumPy；不导入模型或checkpoint。

**Spec:** docs/superpowers/specs/2026-10-18-sid-structure-matched-card.md；用户2026-10-18夜顺序第2项。设计/范围已由该卡和用户明确授权，按要求直接执行，不新增确认环节。

## Global Constraints

- 结构g-structure-a1-s42/s2024/s2025与对照g-coll-a1-c1-s42/s2024/s2025的六个原始verify、accepted manifest和原item hits固定；不重新解码/训练/选best。
- 前缀复用S0 B，热度复用train-slices.json的train-only阈值及counts，不新拟合；不引入折算准确率或跨分叉比较结论。
- 报告主分母2520、in-catalog1897、OOV623，三条16:5x条件机械核对；叶collision surplus与所有碰撞物品分列，不偷换定义。
- 不读test/额外checkpoint/embedding，不导出逐query预测、完整SID或向量；只在服务器内存加载已授权mapping/归档。
- 新代码只新增文件，旧入口逐字节不变；producer/runtime/lock与原训练精确相等，新增执行成员使用卡中逐条固定的路径/SHA。
- 真实readonly formal preflight→dry→唯一入队/claimed，不以exit0代替原始报告与job/source/request绑定；CPU非训练，不占训练名额。

## Review Focus

- 错seed/错对照/未验收或假verified源不能混入。
- Symlink/路径祖先test/checkpoint/越界引用在打开前拒绝。
- 源码新增路径必须固定增量，旧字节/runtime/lock漂移失败。
- 热度档位分母与总Δ一致，OOV独立为0；不将surplus当全部碰撞物品。
- 发布intent/claimed/rejected有状态即不得重投；metadata stdout与实际job/config/argv绑定。

### Task 1: 只读聚合与源合同

**Files:** Create analysis/sid_structure_report_read.py; tests/test_sid_structure_report_read.py。

**Interfaces:** `validate_run_ids(ids)`只认固定六run；`catalog_structure(mapping,branching,metrics)`返回S0结构与明确碰撞计数；`summarize_pair(...)`返回同seedB/C及C逐档差；`execute(req,preflight=False)`元数据检查或完整聚合，stdout仅汇总。

- [ ] fresh RED：缺入口、错run/seed、手算bins差/叶碰撞、人口不符、禁止路径、raw stdout缺失、ledger字节/集合变异。
- [ ] 最小GREEN：复用原固定pure metrics及archive全部原检查；source身份和原始verify绑定先于预测读取。
- [ ] 相关回归与独立缓存变异，具体路径commit。

### Task 2: 有界队列与正式报告

**Files:** Create scripts/sid_structure_report_queue.py; tests/test_sid_structure_report_queue.py；新冻结报告源码增量卡。

**Interfaces:** 请求固定run/config/source/argv，preflight只读stat元数据；report依赖自身原preflight回执，不重投。main实际原run.json绑定，输出含run_id/run_manifest_sha256/request_sha256/source。

- [ ] fresh RED→GREEN：occupied/rejected/intent不重投、错stage/源码/argv拒绝、dry/原始回执身份。
- [ ] 固定源码增量卡与输入request，独立一次whole review，公开压平扫描0→push→固定SHA部署。
- [ ] 真实readonly PF与原始回执→dry→唯一报告；unique claimed/no rejected，完整聚合与输入SHA绑定。
- [ ] 数字与限制仅experiment_log，机械三条件/逐seed及分档差/结构/前缀只出数；结果包交Kiro，STATE/map/handoff/schedule同步，具体路径commit，直接第3项。

## Pre-flight interfaces

Task1 execute(req,preflight)由Task2固定file CLI调用；Task2阶段名preflight/report必须与main的--preflight一致。报告只复用S0 math，不读取S2结果或改变其在机作业；无共享修改文件。
