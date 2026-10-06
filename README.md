# VK Generative Retrieval

这是一个尚未实现、尚无实验结果的公开项目scaffold。

项目计划在VK-LSVD上构建可审计的generative retrieval benchmark，并研究week-level SID staleness、aligned refresh与dense fallback。

## 当前状态

当前处于P0治理收口，尚未进入生产实现。

实时状态见[`STATE.md`](./STATE.md)。

新对话接力见[`handoff.md`](./handoff.md)。

实施阶段见[`tasks.md`](./tasks.md)。

项目规则见[`AGENTS.md`](./AGENTS.md)。

## 核心问题

在week-level distribution shift下，SID staleness会造成多大的item reachability与质量损失；aligned SID refresh配合dense fallback，能否以低于fresh-scratch retraining的成本恢复collision-corrected item-level retrieval quality。

完整项目边界见[`docs/project_charter.md`](./docs/project_charter.md)。

## 数据与任务

- 数据：VK-LSVD
- MVP：`ur0.01_ir0.01`
- Flagship：`ur0.01` scale audit
- 主label：尚未冻结，P3只用train审计后决定
- 时间：当前只确认week partition，文件内顺序待下载后验证
- 内容：官方64D content embedding；本项目不声称训练多模态encoder

完整合同见[`docs/data_contract.md`](./docs/data_contract.md)。

## 两天Vertical Slice

Planned范围：

- Synthetic运行框架
- A6000验收与MVP数据完整性
- Train-only label与catalog合同冻结
- Popularity与content baseline
- 单一SID tokenizer
- Small TIGER one-seed
- 固定query audit set上的item-level审计

两天不承诺完整benchmark、全test、多seed、LIGER、SID refresh或算法增益。

完整架构见[`docs/architecture.md`](./docs/architecture.md)。

## Flagship方向

- Trained dense sequential control
- Pure TIGER
- LIGER-style fixed-budget hybrid
- Stale / fresh-scratch / aligned SID refresh
- Million-item ANN与fixed-query exact audit
- Quality–latency–memory–refresh-cost联合分析

达到`tasks.md`的G7/G8证据门槛后，才称Advanced Generative Retrieval。

## 服务器与预算

- Planned GPU：Thunder Compute RTX A6000 48GB
- Acceptance probe：不超过1小时，需用户批准计费操作
- 总预算：不超过100 AUD
- GPU目标：不超过30小时；hard ceiling 48小时

详见[`docs/server_runbook.md`](./docs/server_runbook.md)。

## 公开仓库边界

允许：原创代码、配置、文档、synthetic fixture、脱敏manifest和小型指标摘要。

禁止：raw/processed data、完整embedding、checkpoint、optimizer state、index、trie、完整SID map、逐query预测、secret和任何公司内部信息。

当前不提供LICENSE；首次发布可复用代码前必须完成第三方provenance/license审计。
