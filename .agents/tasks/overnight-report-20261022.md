# 夜间本地准备交付（2026-10-22，供 Eddy / Kiro 早间复查）

**结论：本轮要求的本地交付已完成；不能据此直接启动全量训练。** 工程缓存真实逐位对拍和性能证据按授权留到新实例；合同history差异、SGNS成本方向与K尚待Kiro。今晚未连接任何服务器、未读真实test、未运行科学训练/下载/过滤/CUDA、未租停销毁实例。context UNKNOWN。旧实例实际Stop尚无回执，旧产物/原预算/spent/deadline不重置。

## 提交与事实源

- 本地实现/文档提交：`db43a1a`（23个具体文件，未修改旧训练入口、decision_log或冻结卡）。
- 停机前盘点提交：`dc47181`，属于今晚本地goal之前已完成的一次只读确认，不再连接复查。
- 本报告与最终交接为后续独立文档提交；其SHA由git log与最终回报定位，不自引用自身SHA。
- 公开部署包将压平到全新工作分支 `codex/overnight-local-public-20261022-v1`；不推主工作分支旧过程历史，不push main、不force、不改写任何旧分支。公开commit与整体文件清单/扫描收据见私有 `cache/overnight-20261022/public-package-v1.json`；无数据或完整ID入Git。
- 科学/工程数表唯一事实源：experiment_log「夜间本地准备：SGNS算术与分叉核算」。本报告不复制科学数表。

## 完成项与效果

1. **张量缓存工程**：工程卡 `docs/superpowers/specs/2026-10-22-tensor-cache-card.md`；G/P共用 `src/vkgr/tensor_cache.py`，仅缓存确定性CPU样本，保留原动态padding/尾batch/样本序/RNG/负例/mask/logQ。新step复用原code object与私有globals，旧文件逐字节未改。它提供可验证候选实现，目前不声称提速。
2. **新实例对拍工具**：`analysis/tensor_cache_parity.py` 已写好dry/capture/compare，按原anchor profile loader读取完整source/config/input/runtime绑定；记录tensor字节摘要、参数、反向及clip后梯度、optimizer、loss、cursor与全部登记RNG。original/cached-cold/cached-warm用独立进程和新输出；任何不一致STOP。今晚未运行capture。
3. **隔离变异与review**：`analysis/tensor_cache_mutations.py` 全新副本、唯一缓存前缀、python -B与PYTHONDONTWRITEBYTECODE。一次独立Luna/high干净context C审，初审B1暴露三项变异杀错断言；已修正，v1保留不用作有效目标证据，v2重新全跑。P warm closure与观察proxy问题也已补synthetic验证，并由同reviewer定点核对，未反复全范围审查。报告 `docs/superpowers/reviews/2026-10-22-tensor-cache-luna-c-review.md` 追加收口，原初审结论不覆盖删除。真实parity/profile仍NOT_RUN。
4. **SGNS与分叉算术**：`analysis/overnight_arithmetic.py` 只读已验收A聚合直方图，SHA/计数守恒；逐短序列精确算pairs，用C1既有整项耗时折算，明确是条件线性情景、真实ETA UNKNOWN。已达数十小时量级，登记待Kiro；没有实现SGNS提速。分叉按原k=min与balanced capacity规则只算利用率/资源下界，没有选K、没有实现BHKM。
5. **全量合同DRAFT**：`docs/superpowers/specs/2026-10-22-ur001-full-contract-draft.md` 逐项写query/history/train/catalog/label/S0/supported/观看阈值及来源；子集未覆盖/有差异之处列待Kiro，不选值。阶段2不构造test query/history，模型不在validation上重训。
6. **迁移runbook与脚本**：`docs/superpowers/plans/2026-10-22-new-instance-runbook.md`；`analysis/new_instance_prepare.py` 复用原固定下载/过滤worker，严格30文件清单、原membership SHA和四项计数；`scripts/new_instance_environment.sh` 只向新ENV_ROOT创建原lock/pyproject副本。原anchor lock从本地Git对象导出到 `configs/new-instance/`，无冻结文件覆盖。所有执行命令写好，今晚只做dry/负例/合成测试。
7. **状态同步**：STATE/handoff/project_map/schedule已加当前入口与待裁决指针；风险登记issue_log，不修改decision_log。

## 验证与收据

|检查|结果|本机私有证据|
|---|---|---|
|新增功能fresh RED|缺失模块ImportError已实际看到；不是伪造预期失败|cache/overnight-20261022/*red*.log|
|新增本地suite|18 tests PASS|final-local-tests-v1.log|
|原week26 worker synthetic回归|16 tests PASS|week26-regression-v1.log|
|隔离变异v2|baseline PASS；14/14目标变异被断言杀死|tensor-mutations-v2/receipt.json与逐项log|
|负例/mask/logQ变异|各命中具名目标断言，未被codeobject断言提前代杀|对应三个log与receipt.target_killed|
|CLI形式|parity/migration dry非空收据、非法subcommand/缺执行flag非零|新增测试日志；未执行真实capture/download|
|环境脚本|bash -n PASS|本轮工具回执|
|源码边界|旧训练/worker/decision/frozen文件不变|git diff/check与local-precommit-check-v1.json|
|本地新增文件敏感扫描|真实敏感项0；原lock内21处四段package版本号已逐项按原SHA/上下文排除|local-precommit-check-v1.json|
|真实新实例环境、入口、launcher链|NOT_RUN|留明天真实前置|
|真实逐位parity、短剖析与提速|NOT_RUN|不得以本地synthetic替代|

表内日志根为 `cache/overnight-20261022/`。本地测试环境Python3.11.17/macOS，不能替代严格3.11.16/cu128新实例。最小import torch曾因沙箱OpenMP SHM失败；经正常权限流程运行同一本地synthetic后通过，不是改算法避错，也不属于服务器入口失败重验。

## 待 Kiro 决断 / 明日潜在阻塞

- **合同history差异（阻断冻结）**：942a5b0说逐条沿子集且允许同周target前事件；子集实际history_v2 validation仅train末尾observed。需明确新条款是否显式覆盖原train-only协议及新版本身份。没有擅选、没有派生。
- **SGNS成本方向**：条件情景已达数十小时，是否先做逐位等价工程提速或另立科学变更，由Kiro定；本轮没有调负例RNG/采样/精度。
- **分叉K**：利用率算式已附卡，Kiro冻结后才能起BHKM外排实现卡。
- **新实例参照与p3c数据通道**：旧checkpoint/run不迁移；原c89数据通道字节必须通过既有获批渠道还原。缺失或重建参照需要额外科学训练时停该入口回Kiro/Eddy，不放宽hash、不拿Git脱敏版替代。
- **具体新实例与首批科学入队**：规格/付费由Eddy最终确认；本地代码就绪不等于科学入队授权。正式test读取仍另批。

## 明天逐步执行清单

- [ ] Eddy确认新Vast实例，核新身份、实际CPU/RAM/盘/显存/driver与余量；旧实例保留，不destroy。
- [ ] 固定公开工作分支SHA部署，逐文件SHA，独立read-only deploy key；恢复原p3c数据通道闭包。
- [ ] 原lock新环境，精确版本检查、CUDA可用性/小算子；部署独立固定scheduler/status，核detached身份与有限截止。
- [ ] 新独占目录重新下载ur0.01固定29文件与全week26，逐文件size/LFS/SHA全相等。
- [ ] 新ID按source-PF→membership→download→filter-PF→filter原顺序重做，membership SHA及四项计数与原值全等，否则STOP。
- [ ] train/eval/verify同正式CLI在真实部署补holdout误传负例；backend仍关闭，不构造test query。
- [ ] 重新下载/建立子集参照并核完整原source/config/input/lock；1a正式CLI真实前置及本地真实launcher链完成后，original/cold/warm逐位对拍与同口径短剖析。未过不进入全量。
- [ ] 按独立工程卡准备并执行1b G固定validation解码/verify计时、1c P synthetic全规模评价成本；不报质量指标。
- [ ] Kiro冻结合同/history与K，裁决SGNS方向；补真实容量/forecast；Eddy确认首批科学入队后才开始模型实践。

## 本轮未做项（按授权留待后续）

真实服务器环境/前置/下载/过滤/对拍/性能测量，1b/1c新入口执行，全量派生、SGNS/BHKM及模型训练、正式test读取；均未执行，不存在可以冒称通过的收据。无活跃服务器job或timer由本轮启动。全部本地任务完成后结束goal，不为等到钟表早晨而空转。
