密级: 公开

# Issue Log

## 当前边界

本轮未下载数据、未训练模型、未验证效果。

下列项目都是开放风险，不代表问题已经发生或已经解决。

哈啰内部信息、数据、schema、代码与实验结果不得进入本项目。

## 状态定义

`Open`：风险已登记，尚无关闭证据。

`Investigating`：已有 owner 与检查动作，但结论未冻结。

`Blocked`：阻止后续 gate。

`Closed`：关闭证据已保存并可复核。

## Open Issues

| ID | Risk | Status | Impact | Next evidence | Closure criteria | Closure evidence |
|---|---|---|---|---|---|---|
| I-001 | `like` label 覆盖未知。 | Open | 可能无法形成足够 history/target，迫使定义 composite。 | 仅用 train 统计 user/item/event coverage、序列长度与 week 稳定性。 | 冻结 `like` 或 composite policy，并保存 coverage report 与 policy hash。 | 待补。 |
| I-002 | `ir0.01` 的含义与 sampling 机制待查。 | Open | 可能改变 popularity、catalog coverage 或反馈分布，影响 MVP 外推。 | 核对 data card、subset manifest 与下载后分布。 | 明确生成规则，并在 data card 中写清可比性边界。 | 待补。 |
| I-003 | `timespent` 可能是聚合值，时序语义未知。 | Open | 可能把未来或聚合信息泄漏进 target-prefix feature。 | 核对 schema、粒度、单位、生成逻辑与同 user-item-week 重复。 | 证明 query-time 可用；否则从主任务移除，只保留审计。 | 待补。 |
| I-004 | `duration` 的粒度、单位、缺失与版本未知。 | Open | `timespent / duration` 可能失真或除零。 | 做 dtype/range/null/zero 与跨 week 稳定性审计。 | 冻结 clipping、missing/zero policy 并保存 feature manifest。 | 待补。 |
| I-005 | Flagship `ur0.01` 公开树可能没有独立 official test。 | Open | 调参与 final holdout 可能混用。 | 下载前核对公开结构；下载后保存 partition manifest。 | 冻结 train-tail validation 与一次性 final holdout，或采用明确的官方 split。 | 待补。 |
| I-006 | Thunder A6000 实时库存与 snapshot restore 未验证。 | Open | 两天窗口可能因实例不可用或 artifact 丢失失败。 | 执行 ≤1h acceptance probe：stock、SSH、CUDA、FAISS、吞吐、磁盘、reconnect、snapshot restore。 | 全部 gate 通过并保存无 secret 的 probe report；否则切 fallback。 | 待补。 |
| I-007 | Dataset、论文参考实现与未来依赖的 license/provenance 未完成。 | Open | Public repository 可能错误复制或分发受限内容。 | 建立 third-party provenance 清单，逐项记录 source、license、使用方式。 | 所有发布内容 provenance 明确；首次发布前再决定 repository LICENSE。 | 待补。 |
| I-008 | Full-catalog score 与 embedding/index 内存可能超限。 | Open | MVP 或 flagship 可能 OOM，或被迫缩小候选后误称 full-catalog。 | 用 manifest 规模做 storage estimate；运行 blocked top-k 与 ANN smoke。 | 峰值 RAM/VRAM/disk 留 ≥20% 余量，且 evaluator 不物化全 score matrix。 | 待补。 |
| I-009 | Exact evaluator 尚未验证。 | Open | Recall/NDCG、collision correction 或 top-k tie handling 可能错误。 | 构造 tiny toy set，对比 brute-force、chunked exact 与 `IndexFlatIP`。 | 三者在冻结 tie policy 下逐 query 一致，并保存 test artifact hash。 | 待补。 |
| I-010 | 大 catalog trie representation 未定。 | Open | Python nested dict 可能导致 RAM 爆炸与 decode latency 不可控。 | 比较 compact array/CSR/automaton 等 representation 的 build time、size 与 lookup。 | 选定 representation，在 flagship pilot 中满足 memory/latency gate。 | 待补。 |
| I-011 | 两天 MVP 时间风险高。 | Open | 数据清洗或 TIGER 训练可能挤占 evaluator 正确性工作。 | 先完成合同、toy evaluator、Popularity/content baseline，再启动 small TIGER。 | 两天内 P1～P7 固定 audit query 闭环；否则只交付 scaffold，不升级为 TIGER benchmark。 | 待补。 |
| I-012 | Week 与文件内顺序能否稳定构造 target-prefix 尚未验证。 | Open | 可能产生 future leakage 或无法定义唯一 next positive。 | 审计 partition/order keys、稳定排序与重复记录。 | 保存 order proof；无法证明则阻塞主任务。 | 待补。 |
| I-013 | SID collision suffix 对 vocabulary 与 decode path 的影响未知。 | Open | SID-level hit 可能不能恢复唯一 item，或 suffix 增加稀疏 token。 | 报告 collision group size、suffix length、reachable rate 与 item-level correction。 | item mapping 可逆，指标分离，且无静默丢 item。 | 待补。 |
| I-014 | 固定 query audit set 可能不代表完整 test 分布。 | Open | 两天结果存在 sampling bias。 | 在 train/validation 可见统计上按 history length、popularity、first-seen 状态分层抽样并冻结 hash。 | audit set 构造可复现，覆盖预注册 slices，结论明确限定为 audit set。 | 待补。 |
| I-015 | `AGENTS.md`未落位。 | Closed | Codex自动发现规则与P1启动条件不成立。 | Eddy手动创建`AGENTS.md`。 | `AGENTS.md`存在并已读取。 | 本轮已读取受保护的`AGENTS.md`。 |
| I-016 | `AGENTS.md`曾混入易变事实。 | Closed | 每次对话占用过多context并产生漂移风险。 | Eddy手动落位thin规则；Kiro拆分STATE/handoff/tasks并复核。 | AGENTS只含稳定规则和指针；当前/历史/路线各有唯一源。 | Thin规则已落位；STATE/handoff/tasks完成二次职责收敛。 |

## 关闭纪律

关闭 issue 必须填写具体 artifact、hash、run ID 或公开文档链接。

“命令成功”不能单独作为关闭证据。

效果不佳不是关闭风险的理由；负结果同样需要 artifact evidence。

涉及 label、split、catalog 与 evaluator 的 issue 未关闭前，不发布效果 claim。

## 公开来源

VK-LSVD data card: https://huggingface.co/datasets/deepvk/VK-LSVD

TIGER: https://arxiv.org/abs/2305.05065

LIGER: https://arxiv.org/abs/2411.18814

SID staleness: https://arxiv.org/abs/2604.13273

Collision-faithful evaluation: https://arxiv.org/abs/2605.25330

## P1环境与实现复盘（2026-10-07）

### I-017 — Python/uv环境前置条件（Closed）

- 预期：P1有Python 3.11与uv可用，项目测试使用当前worktree环境。
- 实际：主协调接力记录系统Python 3.9.6，最初缺uv/Python3.11；主协调已按用户授权安装。执行Agent没有冒充自行安装。
- 影响：直接用系统Python无法满足requires-python，不能把缺依赖当TDD RED。
- 复现证据：本轮fresh运行已批准Python路径`--version`为3.11.17，`python3 -m uv --version`为0.12.23；当前`.venv/bin/python`复核3.11.17/PyYAML6.0.2。详见[p1_validation.md](p1_validation.md)环境准备。
- 排查与排除：不是项目逻辑失败；.local/bin不在PATH，使用明确路径；系统Python只启动uv。
- 已证实原因/未知：初始环境版本与工具缺失来自接力记录；本轮确认安装后版本。没有重演或伪造初始安装输出。
- 处理tradeoff：锁定3.11范围，不改shell配置，不再次安装Python/uv。首次依赖安装有网络，测试/CLI直接离线执行。
- 验证/学习：当前worktree独立.venv与实际uv.lock是关闭证据；面试时区分环境失败、无效RED与功能失败。

### I-018 — worktree metadata/cache/DNS的sandbox边界（Closed）

- 预期：已授权branch切换、必要依赖安装和真实lock生成可执行。
- 实际：git switch默认exit128（共享metadata的index.lock不可写）；uv默认cache不可写exit2；独立tmp cache解决写权限后，PyPI DNS仍失败exit2。
- 影响：阻塞环境准备，不代表config/runtime问题；未算TDD RED。
- 复现证据：validation环境记录和本对话实际工具输出；git switch经escalation exit0；uv lock/sync在独立tmp cache与获准联网escalation后exit0，installed pyyaml==6.0.2，uv.lock实际产生。
- 排查与排除：基点/branch冲突已排除，两者均df32a1a；未改Git配置、main文件或shell配置。cache写权限与DNS分别出现，不能混为缺包。
- 原因/未知：默认sandbox写入范围与网络限制；未调查OS网络全局状态，因为本次escalation已成功。
- 处理tradeoff：只为获准动作escalate；cache留独立tmp目录，运行与测试不调用uv，不借用其他worktree依赖。
- 验证/学习：fresh版本、lock与当前.venv元信息复核；worktree代码隔离不代表Git metadata物理上也在worktree内部。

### I-019 — 拒绝应发生在写输出之前（Closed，限定preflight）

- 预期：不合规output路径、已有目录、Git provenance失败在写入前拒绝。
- 实际：R05旧实现会覆盖已有result；B01虽最终报错却已在不合规目录写文件；B02 Git失败会先留下result。
- 影响：exit非0也可能产生副作用或误导性半成品。
- 复现证据：[p1_validation.md](p1_validation.md) R05/B01/B02 RED→GREEN及对应tests；测试只损及自身临时fixture，没有覆盖用户产物。
- 排查与排除：不是manifest hash或Git基点问题；原因是mkdir/result写入排在路径/provenance校验之前。
- 原因/未知：验证时序已证实。运行中输入并发变化、突然I/O失败仍可能留下本次新目录的部分产物；P1不实现transaction/自动回滚。
- 处理tradeoff：路径、fixture、Git与source hash preflight前移，mkdir exclusive、文件x mode；不自动删除任何失败证据，重跑选择新ID。
- 验证/学习：测试断言原字节不变及output不存在；面试时解释“报错正确”与“没有危险副作用”是两个验收条件。

### I-020 — fixture symlink与测试误通过（Closed）

- 预期：批准的fixture filename不能绕过repo-local人工输入边界。
- 实际：runtime最初未对固定fixture路径执行symlink校验。B03第一次GREEN实际因macOS /var临时路径别名提前拒绝，未触达目标行为。
- 影响：未经修正的测试会隐藏fixture可引用外部文件的问题。
- 复现证据：[p1_validation.md](p1_validation.md) B03意外GREEN说明、canonical temp root后的真实RED、修复GREEN；tests/test_boundaries.py。
- 排查与排除：使用内容合法的外部人工fixture，排除schema错误；canonical ROOT后仍触达Git sentinel，证明遗漏fixture confinement。
- 原因/未知：config路径与output有safe_path，fixture只拼接ROOT；ROOT别名使第一版测试提前退出。并发symlink替换不在P1单进程静态文件的保证范围。
- 处理tradeoff：runtime和build_manifest固定fixture路径也调用safe_path；不改变原fixture或扩大真实数据访问。
- 验证/学习：修正后先有assertion RED再GREEN，output不存在；不能把“测试通过”直接等同于“覆盖了预期分支”。

### I-021 — 人工oracle算术错误（Closed）

- 预期：synthetic加权和使用独立手算期望。
- 实际：R03最初误写229；正确是3×2+7×1+11×5+19×4=144。
- 影响：若未纠正，会把正确实现误判为错误。
- 复现证据：[p1_validation.md](p1_validation.md) R03手算修正、重新RED与GREEN；当时生产seeded计算尚未实现。
- 排查与排除：权重与输入人工复算，不用生产helper构造oracle；并非随机数版本或实现bug。
- 原因/未知：测试编写算术错误已确认，无模型或真实数据影响。
- 处理tradeoff：先纠正oracle再fresh RED，未调整生产代码迎合错误数字。
- 验证/学习：literal144与实际权重匹配，换seed会变；测试oracle同样需要审计，不能以测试权威代替推算。
