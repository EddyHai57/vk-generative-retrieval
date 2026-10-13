# P4E local source validation

最新运行gate：Astra scoped fresh测试与原独立reviewer通过后明确FINAL EXACT START GATE。本轮只exclusive部署固定launcher/合法输入包，fresh pins/独占/20%门后exactargv一次实际运行；随后核原owner/dispatcher/首slot真实child birth/父组一致并首step manifest发布。尚无完整run/final completion，不称消融效果或P4E Complete。启动证据cache/p4e-formal-actual-start-v1.json；旧review包保留历史。已有heartbeat更新为P4E单监控者，后续仅原句柄观察/终态交付，无额外pilot或数值replay。

Outer v2定点纠正：假Helper不覆盖实际STDOUT及identity fallback，Astra预读发现后加真实固定helper.invoke CPU integration freshRED，改完整属性局部proxy/Popen唯一覆盖后fresh scopedGREEN；不改helper原字节，不改共享subprocess模块。v2固定入口待fresh验收，v1历史保留。

最新outer候选：联合静态与formal所有文件累计预算、exclusive持久绝对deadline、不预建formalroot、原预算拒绝重复启动。复用固定invoke且仅当前dispatcher Popen进入fresh session，不修改共享subprocess模块；预算/资源/子进程失败只收束这次group并保留actual原句柄/终态。scoped测试用真实CPU child/grandchild，独立ps核PID/group死亡在finally兜底kill之前完成，exit零留下sleep后代也明确失败。原生产/dispatcher/helper保持，不部署或启动formal。固定包cache/p4e-formal-outer-review-package-v1.json待协调fresh验收；Mac本地测试不冒称Linux实际训练已验证。

Formal scoped review follow-up：原candidate漏operation/production/segment的watch与生产plan绑定，旧metadata fixture未包含这些真实字段却可派生成功。按真实runner binding结构补fixture，fresh证明局部hash重算不掩盖缺/错production plan、watch或segment.binding.plan；最小修复强制externalproductionSHA与实际acceptedwatch hash，所有segment生产planSHA/watchSHA一致，derived保留watch及production来源。formal dispatch在resource guard及任何child前核pilot.watch_sha==formal.runtime.watch_sha。当前scoped regression通过、awaiting original reviewer定点复核；GPU/全项目不重跑，生产source/原proof保持。固定包v2，旧包历史保留。

最新本地操作候选：ignored derived compatibility receipt明确来源于fresh_process_completion，保留actual updates/invocations事实，兼容kind不伪称原pilot_session执行。Targeted tests覆盖external pin/缺失重复策略/source/非零exit/false comparison/外部manifest/outer身份deadline、exclusive output、actual child失败与观察到期原句柄保留，以及固定slot seed门、预算内派生copy和唯一summary顺序。只使用metadata fixtures与微型真实child，不重跑真实模型。fixed review package为cache/p4e-formal-dispatch-review-package.json，awaiting coordinator scoped review；全部生产/source/原pilot bytes保持，不启动formal。

最新操作gate：真实prepare/shared-slices与v2独立进程CUDA pilot均已获Astra fresh接受。作者metadata-only脚本实际只读exit通过；协调独立同scope复核通过，completion/outerreceipt外部SHA一致。固定交付cache/p4e-pilot-v2-hash-only-delivery.json；最终只读脚本cache/p4e-pilot-v2-readonly-delivery.py在协调已审断言基础上增加显式终态外部pins，不调用模型。完整数值重算为原固定runner prepublication实际执行的事实，不冒称作者或本轮协调另重算。

恢复涉及model/optimizer/cursor/negative RNG/全部随机源/统计/selection/records完整精确对齐与fullcatalog validation vectors/rankings/metrics精确输出；实际CLI、stage函数返回与child exit分开记录，原失败attempt/持久budget/lineage保持。真实pilot未跨完整epoch，不证明epoch选择/earlystop或质量；CPU fixture的epoch证明范围另列。资源历史minimum/全训练峰值未记录，不能从瞬时观察推造。

P4E仍未Complete、formal训练未获准。下一scope已批准仅本地ignored derived兼容receipt与dispatcher targeted TDD、完整epoch容量/成本补证，冻结生产不改。所有数字只见experiment_log，预算提案ignored cache供review。

密级：公开。状态：local implementation awaiting coordinator source review；不是 P4E 实验完成。

本阶段实现三个独立消融 consumer、watch 与 train-only slices、固定计划、训练状态恢复、外部 pin verifier、固定 slot 完成证明及配对 summary。旧 producer、合同、配置、lock 和历史 manifest 保持入场字节；新 source 身份另算。完整输入/产物只留运行环境，公开 manifest 不保存逐 query 预测或 checkpoint。

## 本地验收

使用既有只读 Python 环境，`PYTHONPATH=src:tests`，没有安装或下载。精确环境位置和完整证据保存在 ignored cache review package。验收命令：

```text
python -B -m unittest test_retrieval_ablations test_retrieval_watch test_retrieval_ablation_run -v
python -B -m unittest discover -s tests -v
python -B -m vkgr.retrieval_ablation_run summarize --help
```

新增 tests 在实际 P4E ROOT 运行；冻结 legacy snapshot 在其自身 ROOT 独立执行完整旧 suite。不能通过替换 ROOT、重写旧 fixture 或重算历史 source pin 消除身份差异。新 ROOT 完整 suite 的 `test_p4d_v2_verifier_compat.CompatibilityTests.test_complete_producer_pin_and_explicit_cli_identity` 拒绝新 consumer 身份；失败原样保留，不能称新 ROOT 全绿。最初 sandbox 拒绝旧 P1 临时 outputs；正常审核放行本地回归后权限错误消失。

恢复过程先修测试 generator 的语法错误；该 SyntaxError 不是有效行为 RED。随后 fresh assertion RED 确认完成证明缺失，最小实现后 GREEN。另按 fresh RED→GREEN 修复 bool 冒充数值配置、synthetic watch 非严格 before、summary 缺入口、配对分母漂移及 verifier 缺预算 callback。证据 ledger 保留错误和真实 exit，不能以 tail 命令的 exit 代替 unittest exit。

## 行为证据与边界

- Uniform 候选由独立 seeded sample oracle核对，保留全部过去 like mask/target override；候选不足拒绝。
- User balance 两层 replacement，合法重复保留，独立 RNG 和短 epoch tail 可恢复；建组仅扫描人口一次。
- Watch 按原 physical row 顺序扫描，只替换严格 target 前 observed history；validation 仅 train 历史，query/target/catalog 不变。Raw item zero 与 PAD 区分。
- 三策略在独立本地进程 segment/resume 与连续执行的 model/optimizer/RNG/cursor/selection 一致；这是 CPU synthetic 证明，不能冒充真实 CUDA 恢复证明。
- Verifier 外部 manifest/core/plan/watch pins、独立 draw replay、checkpoint full-catalog 重导出、strict best/earlystop 和 budget callback。Summary 必须固定全部 slot、实际双 stage exit 与终态，读取 accepted core 的 pinned predictions，使用共同 train slices。
- Test 保持 closed_hash_only。真实 prepare、pilot、train、verify、summary 尚未执行；没有真实资源峰值或消融质量结论。

## 后续 operation gate

Astra source review 首轮指出 prepare-watch 与 verify-watch 仅 resource_guard、没有共享 persistent deadline/disk/output root，是阻塞。现两入口在读取真实输入或采样资源前先运行 operation_guard；同一 operation_root、plan SHA 的两 scan 共用已保存起点和 deadline，磁盘统计整个 root。新 CLI tests 实际先复现越界、第二次 invocation 超时、累计磁盘超额的 RED，再验证拒绝且真实输入 loader 未调用。首轮固定包及 manifest 历史副本保留，修订成员差异在 v2 review package。

新 run binding.execution 明确 device、FP32、TF32 禁用及 deterministic；producer 与 verifier 使用同一初始化函数。CUDA 绑定需要 CUDA 可用和 CUBLAS deterministic workspace；verifier 在 producer device 重导出，不能静默 CPU fallback。排名与指标逐项严格相等，未加入 metric tolerance。此修复解决设计中的 CPU/CUDA 跨 device 路径；真实 CUDA 的精确重放仍必须由另批准 pilot 实测，不能从 CPU test 推断成功。

Ignored `cache/p4e-operation-proposal.md` 给出有限 prepare/pilot 提案。Astra 必须先独立 source review，再批准精确部署成员、实际输入路径、外部 plan/source/watch pins 和有限操作窗口。真实 CUDA 独立进程恢复、完整 CLI receipt/exit/completion 封装及资源口径需在 operation plan 闭合；当前进程内 pilot 不足以证明 CUDA fresh-process restore。Formal 队列预算必须根据实测 pilot 再冻结，不能提前启动或追加配置。

资源门只覆盖可见 cgroup 祖先/host、GPU 和 disk；不声称不可见宿主限制或全历史峰值。预算检查有采样间隔，单个底层调用返回前不保证即时中断。失败保留产物，不能盲重启或改冻结配置迎合结果。

最终验收由 Astra fresh-context review 决定。本轮无 SSH、服务器生命周期、真实训练、commit 或 push。


### P4E ignored operation runner 本地收口（2026-10-08）

冻结 v2 production source 保持不变。runner 最小补充 train/verify 函数返回 receipt，区别于父进程记录的实际 worker CLI exit；没有新增训练策略或审计功能。固定交付入口 cache/p4e-operation-runner-review-package.json。CPU synthetic fresh-process 证明仅说明本地实现行为，不证明真实 CUDA 恢复或真实效果。一次只读 server preflight 已获协调批准并完成，private resolved bindings 保存在 ignored cache；未部署、prepare、pilot 或 formal train。真实写入仍待 Astra 最后操作 gate。旧 P4D heartbeat 确认实际 PAUSED，不重查完成队列。


### P4E真实prepare终态与协调验收入口

固定exclusive部署、prepare-watch与独立verify-watch已完成，实际阶段与outer退出通过；shared slices正文留server。small hash-only交付入口cache/p4e-prepare-hash-only-delivery.json，只读验收提案cache/p4e-prepare-readonly-acceptance.py已交Astra先审再fresh执行。当前awaiting coordinator prepare acceptance，不称P4E Complete；GPUpilot/formal未放行。此节覆盖此前运行中/未执行记录。自动审批拒绝完整plan/binding/slice manifest导出，未执行被拒命令，随后仅hash/bytes/exit/budget摘要的安全只读核对获准并完成。历史resource minimum未记录，不虚构。固定历史包保持原字节，own状态文档的操作后更新另记录。


### P4E真实GPU pilot首次attempt STOP

原worker及outer实际非零并自然退出，原root/plan/budget/receipt保留，无继续队列或重试。安全只读根因：worker输出父目录未创建，production resource_guard对output.parent做disk检查抛FileNotFoundError；在训练前失败，没有train-return或checkpoint。CPU synthetic guard未经过此真实资源路径，验证覆盖不足如实记录。只向Astra提出ignored runner parent orchestration最小目录修复；生产source/runner均尚未修改，实际新attempt另gate。诊断证据cache/p4e-pilot-stop-diagnostic-hash-only.json。自动审批拒绝日志正文导出，改路径存在/已知错误分类/异常类型/hash-only调查成功，正文未返回。
# 最新静态准备节点

Astra协调fresh验收watch定点修复、原reviewerClosed/PASS。本轮只在exclusive新静态目录部署已验收dispatcher/helper，使用accepted外部production/watch/completion/outer/continuous pins派生真实兼容receipt，冻结正式plan并实际verify_plan通过。原pilot仅metadata与checkpoint bytes核验，没有数值replay。formal输出root不存在，未启动训练。固定启动包cache/p4e-formal-static-start-package-v1.json供协调最终核；outer launcher须闭合static sibling和formal root联合累计预算，不能提前创建formal root。原source/runner/proof/review快照保持。
## 终态采集器本地准备（等待协调审查）

入口cache/p4e-final-readonly-delivery.md。Stdlib-only metadata/hash读取，checkpoint只流式hash，无模型或生产imports、无额外数值replay。Fixtures验证完整包和缺slot/错seed/非零/文件与handle漂移/预算/指标schema/owner证据缺口/live owned process的具体拒绝；合法关联先一致重建。真实统计、CUDA及历史资源不由fixtures证明。Owner实际exit缺口与训练失败分列，最终acceptance由Astra裁决。测试数字唯一experiment_log，当前未SSH执行采集器。
最新协调gate：Astra fresh targeted验证与独立静态兼容性review PASS，条件允许冻结采集器在实际全队列终态后执行一次。此验收是脚本scope/fixture证据，不是P4E真实实验最终验收；owner实际exit限制仍待终态裁决。


### P4E协调正式验收与accepted limitation

最新协调最终裁决：Astra正式接受P4E实验产物Complete；owner_actual_exit永久null，接受原detached owner自身actualwait不可追溯的历史运行记录限制，不称全部编排wait成功。九run/verify/summary实际成功与产物核验已通过，无第三次模型replay。原交付包保持bytes，裁决增量证据cache/p4e-final-delivery-v1/coordinator-acceptance-v1.json。既有p4d已按Astra明确批准经native更新PAUSED并fresh核保存配置；服务器未作生命周期操作。下一步仅P4F只读一页设计，实施/实际执行另待范围与启动gate，test仍关闭。

