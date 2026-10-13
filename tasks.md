# tasks.md — Codex 实施阶段

## P4H v5 当前集成（2026-10-11）

Plan：主仓 `.agents/tasks/model-audit/p-sasrec-parallel-plan-v5.md` SHA `ea3f7566750c4fad13dfba80fd03a5f8fdf7e81c4c68aa8912b09b3e0e732121`；review SHA `3de5ad402b6dd9e033d068567d046071ba4998e888914f5ac49c155c035287a1`，APPROVED。当前唯一协调与执行owner为Kiro；Codex仅可只读预加载，不得启动writer、monitor或job，除非Eddy以后明确切换。

- [x] 主/P facts最小集成，P4G freeze/test closed与G优先/cutoff不变。
- [x] P4H-01：五个owned files完成fresh RED→minimal GREEN→15项targeted regression并冻结；manifest SHA `2dfe420b2760327e240e15c12c4702e03ede613e4d47ccbfa87e467d1cb855c0`。
- [x] P4H-02：model/test完成fresh RED→minimal GREEN；own10/10 clean、combined23/23、training6/6 clean；SHA `301b7086…f00c`/`2be2a3aa…c78b`，P4H-01与旧P4D/E bytes不变。
- [x] P4H-03A：typed science/test完成RED→minimal GREEN；own12/12、v5 related60/60；SHA `ce427443…b77a`/`86558a67…bc8c`，无CLI/manifest/path/process，前序bytes不变。
- [x] P4H-03B：operation/test完成fresh RED→minimal GREEN；指定classes10/10、own14/14、full280 total（278 passed+2既有conditional skip）；SHA `c54d7cb3…f238`/`256bf44b…fa8`，review packet SHA `f2decceb…205af`。
- [ ] P4H-03B v6独立fresh review：`KIRO_CONTINUES / CODEX_PRELOAD_ONLY / FRESH_REVIEW_IN_PROGRESS`；source/test `fd53b42a…82d9`/`00e70e01…b0a`，packet `9bf8c4cf…e6a6`；v5 review两个HIGH仍open；same-UID整体自洽重写OUT_OF_SCOPE。
- P4H-01～03A保持冻结；P4H-04 BLOCKED；remote decisions BLOCKED；不SSH/GPU/real data/test/G改动。

当前阶段与完成状态只在`STATE.md`维护；本文件只定义路线、依赖、产物、验收和stop gate。

执行约定：每次只进入一个阶段；满足验收后才能进入下一阶段；触发 stop gate 时先记录证据，不自动扩大范围。

两天 vertical slice 只覆盖 P1–P7 的最小路径，并允许因 wall-clock 在固定 query audit set 收口。

P8–P13 不属于两天承诺。

## P0 — Bootstrap 与治理

依赖：无。

产物：thin `AGENTS.md`、`STATE.md`、根`handoff.md`、`tasks.md`、`docs/`设计文档、服务器runbook、Codex handoff、draft configs、目录边界说明与本地Git初始化。

验收：AGENTS只含稳定规则和指针；STATE只含当前快照；handoff只含接力证据和恢复方法；tasks只含路线定义；公开边界、必需文件、YAML、本地链接、敏感字段、代码越界和Git状态通过检查。

后续gate：bootstrap commit必须由Eddy单独明确批准，当前状态以`STATE.md`为准。

Stop gate：数据语义、公开边界或第三方许可存在冲突时，停止代码与下载工作。

## P1 — 基础代码与可审计运行框架

依赖：P0正式关闭，且用户批准创建bootstrap commit作为branch/worktree基点。

Planned 产物：最小 Python package、CLI 入口、config schema、run manifest、logging、deterministic seed、synthetic fixture、targeted tests。

验收：单条本地命令使用`configs/smoke.yaml`在synthetic fixture上完成dry run；manifest记录config、版本、seed、输入指纹与资源信息；不依赖网络、GPU或真实数据。

Stop gate：若无法从manifest复原一次运行的输入和配置，停止进入数据下载。

## P2 — 服务器验收、数据下载与完整性检查

依赖：P1；VK-LSVD访问方式、条款与provenance审计完成；用户明确批准创建计费实例。

Planned 产物：当前已确认平台A6000不超过1小时的acceptance probe报告（平台见`docs/decision_log.md`）；仅下载到服务器受忽略目录的脚本、checksum清单、文件inventory、失败可恢复日志；仓库只保留脱敏manifest模板。

验收：SSH reconnect、CUDA/PyTorch、FAISS、RAM/disk、固定revision dataset传输吞吐、disk I/O与经批准的平台持久化/恢复验证按runbook验证；取得`ur0.01_ir0.01`所需文件；checksum与inventory可重复验证；Git status不出现raw data或完整embedding。

Stop gate：服务器验收失败、许可不清、checksum不一致、磁盘余量不足或任何大文件进入Git tracking时停止。

## P3 — 数据探查与合同冻结

依赖：P2 完成 MVP 数据完整性检查。

Planned 产物：train-only label coverage 报告、sequence length 分布、week/file-order 检查、catalog/first-seen/OOV 统计、split proposal、data card。

验收：明确记录没有普通 `click`；保留 `click_on_author` 原语义；冻结主 label、history、target、candidate catalog、validation/test 与 OOV 报告方式。

Stop gate：若主正反馈覆盖不足以形成可靠 history/target，先预注册 composite 规则；仍不足则停止 TIGER，项目收口为数据可行性报告。

## P4 — 传统召回基线

依赖：P3合同冻结。

Planned产物：Popularity、无训练content-centroid/nearest-content baseline、按TDD实现的brute-force与blockwise exact top-k evaluator、固定query audit set、latency/RAM记录。

验收：toy fixture 上 brute-force 与 chunked exact 的 top-k 和指标一致；真实 MVP audit set 有可追溯结果；不物化完整 user-item score matrix。

Stop gate：若 evaluator 无法对齐 toy ground truth，停止全部模型比较；两天内不追加 ItemCF 或 SASRec。

## P5 — SID tokenizer 与约束结构

依赖：P3 catalog 冻结；P4 evaluator 通过。

Planned 产物：balanced hierarchical k-means SID、collision suffix、SID manifest、catalog constraint、dead-end/collision audit、toy trie 或等价紧凑结构。

验收：每个 train-observed item 可映射并可解码；SID-level 与 collision-corrected item-level 指标分开；完整 SID map 不进入公开仓库。

Stop gate：若 collision、不可达 item 或内存开销超过预注册阈值，停止 TIGER 训练并先修 tokenizer / constraint。

## P6 — TIGER smoke

依赖：P5 SID 与 constraint 验收；P1 training manifest 可用。

Planned 产物：small TIGER one-seed config、短训练日志、checkpoint 仅保存在受忽略目录、catalog-constrained decode、固定 query smoke 结果。

验收：模型能从 history 生成合法 SID 路径并映射到 item；结果包含 invalid、dead-end、collision、latency、RAM/VRAM；不要求优于 baseline。

Stop gate：若两天内训练、resume、decode 或 item 映射任一环节未闭环，对外只称 benchmark scaffold，不包装为 TIGER 复现结果。

## P7 — MVP vertical slice

依赖：P4 与 P6 在同一冻结合同和 query set 上可比较。

Planned 产物：Popularity、content baseline、SID tokenizer、small TIGER 的统一 run manifests；item Recall/NDCG 与系统指标表；限制与负结果记录。

验收：固定 query audit set 上完成 collision-corrected item-level 公平对比；保存 config、seed、catalog 指纹、资源统计；README 只引用指标唯一摘要。

Stop gate：wall-clock 不足时保留固定 audit set，不承诺全 test；缺任一公平性条件时不得下模型优劣结论。

## P8 — Flagship streaming 与基础设施

依赖：P7 可复核；当前已确认平台 acceptance probe 通过；预算获准。

Planned产物：`ur0.01`单week pilot、streaming ETL、out-of-core catalog/SID、compact constraint structure、ANN pilot、scale re-validation、snapshot/restore记录、旗舰split/final holdout manifest、成本manifest。

验收：在RTX A6000 48GB上完成scale re-validation；冻结旗舰validation/final holdout且final不参与调参；pilot全程保留至少20% RAM/VRAM/disk余量；预算预测不超过100 AUD。

Stop gate：吞吐、恢复、磁盘、显存或预算任一不可接受时，不启动旗舰全量处理。

## P9 — Flagship pure generative retrieval

依赖：P8通过；旗舰split与final holdout已冻结。

Planned产物：trained dense sequential control、纯generative retrieval baseline、ANN full-catalog evaluation、固定query blockwise exact audit、资源与成本报告。

验收：同时报告 ANN recall、exact-audit quality delta、item-level quality、collision/reachability、latency 与 memory；final holdout 未参与调参。

Stop gate：pure generative retrieval 若无可信 item-level signal，不增加复杂 tokenizer，优先进入 hybrid 或记录停止结论。

## P10 — Hybrid dense fallback

依赖：P9 明确 pure generative failure modes；dense control 与 SID route 均可独立评估。

Planned 产物：dense fallback、score/rank fusion、不可达与 OOV routing、校准报告、pure/dense/hybrid 公平消融。

验收：hybrid 在同一 catalog、query、budget 下报告质量、覆盖、延迟、内存与增量成本；收益能定位到 collision、OOV 或 stale SID 子集。

Stop gate：若收益仅来自额外候选量或不公平算力，不进入 dynamic routing。

## P11 — Dynamic SID refresh 与 checkpoint compatibility

依赖：P10 hybrid 对照成立；至少两个 week 的 observed-catalog proxy 与 stale SID protocol 冻结。

Planned 产物：stale、fresh-scratch、checkpoint-compatible / aligned refresh 三组协议；SID churn、old/new reachability、refresh cost、dense fallback 恢复曲线。

验收：多 seed 比较 collision-corrected item-level quality、刷新 wall-clock/GPU cost、SID churn 与 checkpoint resume；不使用精确 timestamp 或真实 publish time 表述。

Stop gate：aligned refresh 若不能同时保持可达性与 checkpoint compatibility，必须报告失败，不改口为成功迁移。

## P12 — Multi-route retrieval

依赖：P11 已定位单路 SID 与 dense fallback 的稳定边界。

Planned 产物：多个互补 SID / dense route、route selection 或 fusion、route-level coverage 与成本消融。

验收：相同总 retrieval budget 下比较 single-route 与 multi-route；新增 route 的独立贡献、延迟与内存可归因。

Stop gate：若提升来自扩大总候选量、泄漏 holdout 或重复 route，不进入 release claim。

## P13 — Release 与公开复核

依赖：拟公开的阶段结果已通过独立复核；provenance / license 审计完成。

Planned 产物：可复现 README、冻结 configs、synthetic test、manifest 示例、结果限制、THIRD_PARTY_NOTICES、secret/data scan 报告；是否添加 `LICENSE` 由审计结论决定。

验收：从干净环境可复现小数据链路；公开仓库不含 raw data、完整 embedding、checkpoint、index、完整 SID map、密钥或公司内部资料；所有数字指向 `docs/experiment_log.md`。

Stop gate：license、provenance、隐私、数据泄漏或复现检查任一失败时禁止发布。


## 已批准传统召回扩展路线（P0～P13历史定义保留）

Eddy在P7与P3补充EDA验收后批准完整传统召回计划。以下task串行推进；每次只执行STATE指定的一个task，每task须独立review和Astra验收，下阶段由协调批准brief，不预建未来接口。不进入P8、不新增数据、commit/push或资源生命周期操作。

### P3-C — 现有数据调查与清洗决策

依赖：已验收P2/P3/P7与train-only补充EDA。产物：完整字段/可用性/point-in-time registry，train item/author长尾与zero-like支持、user贡献/异常flag，清洗前后基数表、原创bounded audit及synthetic tests、脱敏aggregate/hash/reuse与验证ledger。raw immutable，不做5-core、ratio clipping或未知异常自动删除；仅提出决策，不改冻结合同。

验收：已做vs补做清晰；固定train输入/schema/hash/join/null及既有基数对账；aggregate独立复核；feature时点风险可解释；0删除也逐项报告原因；每batch至少20%可见RAM/disk余量；旧源码/tests/合同/产物hash保护与公开边界通过。Stop gate：requirednull/非法ID/schema/hash/join/order不一致、资源或公开边界失败，保留证据停止。

### P3-D — 新训练与评估合同

依赖：P3-C正式验收。规则唯一入口为 `docs/data_contract.md#traditional-like-v1`，机器镜像 `configs/traditional.yaml` 与来源绑定 `artifacts/p3d-retrieval-contract-manifest.json`；当前状态只读STATE。保留旧p3-like-v1；like label、官方split、最近50 train-like history、all-train-exposure catalog、seenmaskfalse、full eligible queries、K10/20/50。旧test已使用，不称pristine final holdout；watch history仅独立消融且targets/queries/catalog相同。验收：补充version/config/feature时点/共同输入与full评估协议冻结，旧512 audit regression可复核。Stop：合同/顺序/来源或公平性冲突。

### P4B — 数据管线与in-batch训练框架

依赖：P3-D。产物：可复核ID/PAD、prefix训练、in-batch去重target列与strict-before-positive mask、损失、resume/manifest框架。验收：synthetic手算/mask/tail/RNG与fresh-process resume、参数真实变化。Stop：泄漏、无negative跳过率超批准门、nonfinite/hash/resource。

### P4C — Pop / ItemCF / content基线

依赖：P4B。产物：train-like Pop、binary-user-like cosine sparse ItemCF、官方content均值exact retrieval。验收：toy独立oracle、共同catalog/query与bounded-memory exact评估。Stop：语义/基数/公平性/资源不符。

### P4D — 四种双塔

依赖：P4C。产物：ID mean、ID+content mean、ordinal recency mean、Transformer四双塔；共享训练预算与按validation选配置。验收：padding/mean permutation/position/overfit/参数变化、统一shared batch/resource门与受限预注册搜索。Stop：未来feature、nonfinite、资源或合同漂移。

### P4E — 独立消融

依赖：P4D。产物：固定best architecture/config的uniform negatives、user balance、watch history三项独立消融，train-defined slices。验收：同targets/queries/catalog，多seed各项独立，不混合赢家或追加调参。Stop：混合任务、更换label/分母或holdout调参。

### P4F — wRRF / ANN / 离线推理

依赖：P4E。产物：四route weighted RRF受限validation选择、公平quota留一路消融、exact authoritative与HNSW质量/latency/memory、export/index/infer与shard resume。验收：共同pins/queries、ANN overlap与model delta分开、无query loss/duplicate、fallback单列。Stop：来源/公平性/资源失败；ANN不达标报告未通过并保留exact交付。

### P4G — 多seed复核与收口

依赖：P4F。产物：所有预注册seed/失败ledger、恢复/可复现/公开边界报告和中文解释；test仅配置/fusion冻结后一次，不以其调参。验收：独立review与Astra fresh复核、指标唯一experiment_log、来源完整、限制透明。Stop：未证实结论或来源/隐私/公开边界失败；停在commit/push前。
