# VK Generative Retrieval

这是一个已完成P3数据合同冻结、并取得P6 small TIGER-style真实训练/resume/decode smoke证据的公开项目；阶段验收状态以 STATE.md 为准。

项目计划在VK-LSVD上构建可审计的generative retrieval benchmark，并研究week-level SID staleness、aligned refresh与dense fallback。

## 当前状态

P0～P7已正式验收，P7统一固定audit比较与MVP交接通过独立review及协调验收；当前阶段与下一步只读 STATE.md。统一绑定见[`artifacts/p7-comparison-manifest.json`](./artifacts/p7-comparison-manifest.json)，过程与面试复盘见[`docs/p7_validation.md`](./docs/p7_validation.md)，指标与限制唯一见[`docs/experiment_log.md`](./docs/experiment_log.md)。

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
- 主label：P3已只用train审计并冻结；完整规则只读[`docs/data_contract.md`](./docs/data_contract.md)
- 时间：采用official-order assumption与稳定physical row ordinal；已完成P3顺序读取与prefix核对，但没有timestamp，不称独立验证隐藏时间单调
- 内容：官方64D content embedding；本项目不声称训练多模态encoder

完整合同见[`docs/data_contract.md`](./docs/data_contract.md)。

## 两天Vertical Slice

已实现并逐阶段正式验收的范围：

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

旗舰方向尚未执行；须满足`tasks.md`后续阶段证据门槛，不能用当前small smoke替代。

## 服务器与预算

- GPU平台选择：见`docs/decision_log.md` D-049；SSH方法、验收规则与历史证据见`docs/server_runbook.md`
- Acceptance probe：不超过1小时，需用户批准计费操作
- 总预算：不超过100 AUD
- GPU目标：不超过30小时；hard ceiling 48小时

详见[`docs/server_runbook.md`](./docs/server_runbook.md)。

## 公开仓库边界

允许：原创代码、配置、文档、synthetic fixture、脱敏manifest和小型指标摘要。

禁止：raw/processed data、完整embedding、checkpoint、optimizer state、index、trie、完整SID map、逐query预测、secret和任何公司内部信息。

当前不提供LICENSE；首次发布可复用代码前必须完成第三方provenance/license审计。

## P1 本地 dry-run

使用 Python 3.11（`requires-python >=3.11,<3.12`）与精确依赖 `PyYAML==6.0.2`。`uv.lock`由uv真实生成，包含PyPI distribution hashes；测试使用标准库unittest。首次安装需要网络，运行与验证不需要网络。已有当前worktree `.venv`时直接运行下列命令；新环境用 `uv sync --locked --python /absolute/path/to/python3.11` 准备，不使用系统Python 3.9执行项目。

在仓库根目录执行，两个output目录必须尚不存在；换一个唯一run ID即可重跑：

```bash
PYTHONPATH=src .venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-example-a
PYTHONPATH=src .venv/bin/python -B -m vkgr verify --manifest outputs/p1-example-a/run_manifest.json
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v
```

成功和失败诊断写stderr；成功exit 0，输入/边界错误exit 2。已存在目录会被拒绝，不覆盖产物。smoke生成`result.json`和`run_manifest.json`；result默认被Git忽略，manifest是允许检查的小型脱敏产物。P1已发布；本轮P2～P7 overlay尚未commit/push。

Fixture明确标记为人工整数，不对应VK schema、label/history/split/catalog。独立`random.Random(seed)`生成少量权重，再计算整数加权和。同fixture/config/seed得到相同result字节与SHA-256；manifest的时间、elapsed及output路径允许不同。

Manifest schema v1记录输入/config/result hashes、effective config、Python/dependency版本、Git commit/dirty、source fingerprint与UTC时间。source fingerprint覆盖`.python-version`、`pyproject.toml`、`uv.lock`，以及按文件名排序的`src/vkgr/*.py`，以`repo-relative path + NUL + file SHA-256 + newline`串联后再做SHA-256；fixture和config有各自hash。未提交源码由fingerprint标识，Git commit只表示基点。verify要求当前源码和输入仍匹配，并检查本地commit存在及synthetic result语义。

Hash证明一致性，不认证作者、历史时间或历史dirty状态；需要保存匹配源码与输入才能复核。运行期间不应并行修改输入或源码。P1不记录峰值RAM/CPU百分比，不评估模型质量；唯一claim是`synthetic scaffold execution; model quality not evaluated`。I/O中途失败可留下新目录的部分产物，失败目录不复用，选择新ID重跑；程序不自动删除证据。

实际RED/GREEN、环境与运行证据见[`docs/p1_validation.md`](./docs/p1_validation.md)，问题复盘见[`docs/issue_log.md`](./docs/issue_log.md)。
