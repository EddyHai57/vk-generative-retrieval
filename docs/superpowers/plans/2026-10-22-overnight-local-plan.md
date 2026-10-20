# 夜间本地准备 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. 按 Eddy 无人值守授权执行，不等待中间确认；完整板块结束后一次独立 Luna review。

**Goal:** 完成张量缓存工程、SGNS 算术、合同草稿、迁移工具与早间报告。
**Architecture:** 新增 CPU 样本张量 LRU，复用原 constructor；以新函数副本绑定构建器，原 step 字节和全局变量不修改。新实例对拍工具另行显式执行。
**Tech Stack:** Python 3.11、torch、stdlib unittest；今晚只用 synthetic。
**Spec:** `docs/superpowers/specs/2026-10-22-tensor-cache-card.md`；decision_log 3d8a895/942a5b0/edc7f96/ecfcf8c/feb194e；Eddy 本轮五项任务。

## Global Constraints

- 禁止连接服务器、读 test、科学训练、付费、删旧产物或修改冻结规则。
- 新代码/测试为新增文件；旧入口逐位保留；STATE 等仅追加当前指针。
- 本地 synthetic 不替代新实例同环境逐位对拍和短剖析。
- 每步 tensor bytes、参数、梯度、optimizer state、loss、cursor、全部 RNG 逐位相同；动态 padding、tail batch、负例/mask/logQ 不变。
- 工作分支提交；公开前扫描；不 push main。

## Review Focus

- 缓存命中后返回值被下游修改，不能污染缓存。
- map/token 内容变化必须改变 key；hash 字段遗漏必须拒绝。
- batch 重排/尾 batch 不得填满、重排或更改 RNG 消耗。
- P 随机负例必须原调用时点生成，不得缓存。
- 新实例执行仍须真实 CLI、固定源码/lock、独占收据，未运行项标 NOT_RUN。

### Task 1: 共用张量缓存与新实例验收工具

**Files:** 新增 `src/vkgr/tensor_cache.py`、`analysis/tensor_cache_parity.py`、相关 tests 与 mutation 工具。
**Interfaces:** G/P constructor wrapper → 原 step 的新函数副本；parity snapshot 显式捕获全部状态。
- [ ] 写 synthetic 测试，先运行得到缺失模块 RED。
- [ ] 实现 bounded LRU、完整绑定、原样 batch 装配与无全局 patch 的 step 绑定；运行 GREEN。
- [ ] 写新实例对拍脚本与状态比较；今晚只测快照/比较器，不执行真实 step 对拍。
- [ ] 独立新拷贝、唯一缓存前缀、python -B/PYTHONDONTWRITEBYTECODE 隔离变异，逐项记录。
- [ ] 整块完成后独立 Luna C review；修复并按 §4A 说明。

### Task 2: 成本算术与合同草稿

**Interfaces:** 已下载的 A 聚合直方图 → pairs/epoch；C1 原整项收据 → 条件线性情景，不作真实 ETA。
- [ ] 核直方图 SHA/计数守恒，按 window5 精确公式计算。
- [ ] 重读子集规则原文与 48³ 卡，起草全量合同及分叉算术；未覆盖处只列待裁决。
- [ ] 数字写 experiment_log，其他文档只放指针（用户指定卡所需算式除外）。

### Task 3: 迁移 runbook 与未执行脚本

**Interfaces:** 固定 29 文件清单 + week26 size/LFS → 新实例下载；原过滤 worker → 相同四项计数核对。
- [ ] 新增环境身份/CUDA smoke、下载规划/执行、过滤结果验收 CLI 与 synthetic tests，fresh RED→GREEN。
- [ ] runbook 明确独占新路径、deploy key、scheduler/status、holdout 白名单、1a→1b→1c 顺序。
- [ ] 只做本地 dry/负例，今晚绝不执行 download/filter/CUDA/真实对拍。

### Task 4: 收口

- [ ] 同步 STATE/handoff/project_map/schedule 指针与待裁决项。
- [ ] 精确暂存、验证、提交；扫描公开新增 diff 后 push 独立工作分支。
- [ ] `.agents/tasks/overnight-report-20261022.md` 写提交、测试、变异、限制、明日清单；全部完成结束 goal。
