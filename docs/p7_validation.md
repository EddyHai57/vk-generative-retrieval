密级: 公开

# P7 validation

## 范围、输入与stop gate

只执行tasks P7，依赖P4与P6已正式关闭；Eddy过夜goal的P7 stretch已授权，由协调dispatch。deadline UTC2026-10-07 02:46:21，可提前收口。P7只统一现有结果/manifest/docs/交接，不新fit、训练、模型、调参、生产代码或P8，不commit/push/资源生命周期。

允许修改README、STATE、handoff、experiment_log、decision_log、issue_log、architecture；新增本validation与artifacts/p7-comparison-manifest.json。协调随后明确批准data_contract两处阶段状态句修正，label/history/split/catalog/feature规则不改。临时只读验证scripts和aggregate结果放ignored cache；不是新增公共framework或镜像生产tests。

独立managed worktree branch codex/p7-vertical-slice，基点0376d9a8119f3500cbf78519675cc7bf18013c1b。P6最新已验收overlay45文件逐hash带入，cache/p7-entry-overlay.json SHA256 f3a0cbd4f39f660a48e9cfaaaff04511d4c77adde37f1e88fdcf1a3d23be1118。没有只clone GitHub旧P1或使用main旧draft。原P6/main工作区不修改。

输入来自P3 freeze、P4唯一有效v2 baseline、P5唯一有效v2 SID、P6唯一有效train/resume/decode；所有阶段server/public/source/config/seed及catalog/query/feature pins统一由public P7 manifest记录。详细server路径只用既有p3～p6_validation定位，不在新manifest写绝对私有路径。原P5 public manifest的awaiting状态字符串作为历史snapshot原样保留，正式验收以P5 validation/STATE为准，不能因改文字破坏其hash。

任一manifest/input pin、query equality、item指标公式、公开边界或合同语义冲突失败即停止比较并保留证据；不据此跳过独立review或新增训练修数字。One-seed、小audit与latency边界限制在experiment_log与manifest明确披露，不宣称胜负。

## Fresh公平性核验

使用data-analytics:validate-data标准流程，来源是已验收server artifact，不从聊天数字回填结果。先完整读取服务器guide；target AGENTS.md也fresh读取，内容与项目规范一致。仅用已授权SSH alias vast-a6000，不输出连接参数。

独立stdlib脚本cache/p7-audit.py SHA25613256f3ad84624ca6d644374a111eff0e4af07b8b0d8b8f813f621405f259e9d。不import vkgr/生产evaluator，不启动torch或replay模型；读取现有P4/P6 audit预测和metrics。

命令（在P7 worktree）：

```bash
ssh -o BatchMode=yes vast-a6000 'python3 -' < cache/p7-audit.py > cache/p7-audit-result.json
python3 cache/p7-local-verify.py
ssh -o BatchMode=yes vast-a6000 'python3 -' < cache/p7-process-check.py > cache/p7-process-check.log
git diff --check
```

作者fresh fairness命令真实exit0，UTC2026-10-06T21:37:49.095104+00:00，关键output p7_independent_fairness_pass。结果SHA256 b70118e5954325476d127a5402349479e4ed5443e0ba1a2f9f46390ee1007195。完整小摘要只ignored cache，指标唯一人工摘要在experiment_log，不把作者自检冒充独立fresh-context review。

检查范围：

- 核已知P4/P6 server manifest SHA，P6全部输入pins实际重hash，以及本比较JSON产物hash；未重读完整raw inventory或重跑昂贵模型decode。
- P4/P5共同raw/feature/catalog/contract evidence hash逐项相同，P5绑定P4 manifest，P6绑定P5 normalized feature/map/codebook/constraint；P5 config SHA另fresh核对server与本地配置一致。共享feature来源不声称三种方法使用同一输入表示，Popularity本身只用counts。
- 全部P4/P6 audit queries逐字相同，包括history、history_positions、target、target_position、user_id、OOV。从P3完整queries按冻结SHA排名重选sample成员并核每条完整row，无target/OOV/hit条件；唯一user、history非空/maxlen、catalog membership、history严格早于target全量验证。
- 三方法每条预测均为固定长度、唯一catalog item；手写Recall/NDCG逐split与all/train_observed/OOV核共同分母及全部值，absolute tolerance1e-14，零与OOV结构性miss明确。
- P6全部query诊断与metric里的invalid/deadend/full-SID collision相同；SID不是直接item质量替代。

旧命令失败真实保留在tool记录：schema探查对P4用通用predictions字段KeyError exit1；正式audit首次对P6用config而不是effective_config，KeyError exit1。适配已知真实字段后fresh exit0，没有改source/input/产物。local fingerprint首次全路径排序错误导致assertion exit1，改producer base-first拼接顺序后fresh PASS；process检查一次inline quoting SyntaxError exit1，改独立stdin script后fresh PASS。失败只代表审计脚本准备不足，不暗称实验失败或忽略失败。

## 本地hash、schema与公开边界

cache/p7-local-verify.py独立hashlib检查：原P6 overlay源45文件仍逐hash相同；P7中37个受保护overlay文件及所有非允许tracked文件与原P6逐hash相同。base/source/config/lock与P3/P4/P5/P6 public artifacts不改；production source fingerprint仍6472ddb1ca6f103a895ba5cdd5a727b1af7717125d3b6029997e5e83c7ba4d32。

data_contract只替换两处精确批准状态句；逆替换后与P6原件完整字节相同，证明冻结规则未动。configs/p3.yaml、p4.json、p5.json、p6.json逐hash与统一manifest的config pin一致；冻结P3 manifest/hash和policy/hash不变。原source/config/lock/tests未修改，本轮不因纯文档追加无意义full regression，P6正式fresh regression证据仍见p6_validation。

新增manifest重新json.load，检查严格top-level keys/schema/task/stage集合、各public manifest重新hash、server pins、config hashes、seed与验证script/result hash。Manifest只用role/相对public路径/hash/config与边界说明，不含query/prediction实体、完整SID/weights、私有路径、IP/连接/密钥字段；metric数字只experiment_log，manifest仅指针。Hash证明绑定一致性，不认证作者或历史时间。

process只读检查fresh exit0：P5 config pin verified、P6 pipeline completed、active_vkgr_training_processes=0。该检查不stop实例、不改变任务、不宣称账单归零；实例空转费用/未知费率和host-loss恢复限制保留，资源操作由Eddy决定。

最终公开扫描、Git diff --check与untracked逐文件检查见下节已完成证据。只有合法小型文档/manifest新增；cache脚本/aggregate/logs被Git忽略，无数据/checkpoint/index进入Git。

## 做了什么、成果与面试学习点

| Part | 实际成果 | 遇到的问题及处理 | 可讲的学习点 |
|---|---|---|---|
| P1 scaffold | synthetic运行、config/seed/log/manifest与可复核CLI | exclusive输出、symlink/fixture与独立oracle边界详p1_validation | 先把输入、代码和结果绑定，再谈模型 |
| P2 平台与数据 | 受限实例/官方subset完整性、GPU与目录级恢复验收 | 不将目录恢复等同host-loss；未知quota/费用如实保留 | “能连GPU”与可恢复、可审计环境是不同证据 |
| P3 合同冻结 | train-only label/history/target/catalog/split与物理顺序假设 | 真实没有普通click/timestamp，like与official-order限定继续生效 | 数据语义是比较前提，holdout前定policy避免泄漏 |
| P4 传统baseline | Popularity与content exact evaluator/fixed audit | GEMM blockshape使tie不一致，固定pair arithmetic；v1 invalid保留 | 同样的数学式也可能受浮点执行路径影响，要用独立oracle |
| P5 SID | 可逆完整SID与紧凑合法prefix constraint | semantic collision需suffix；多数node未收敛、greedy objective非单调披露 | “合法可达”与“语义好/检索好”是不同验收 |
| P6 模型闭环 | 真实训练、独立process resume、constraint decode与item replay | equality test可能两个都没训练；参数变化/RNG/global beam测试加强 | 工程证据要证明实际学习和恢复，不只看exit0/loss |
| P7 公平比较 | 同query/分母/来源的item-level表与统一绑定 | schema适配错误fresh修正；旧状态句修为指针；系统边界不强行比较 | OOV上限、抽样偏差、compute与latency边界决定能说多强结论 |

面试可以解释从输入冻结到真实resume与统一item评价的完整工程链路，并以issue/validation/hash追溯每次错误如何发现和关闭。不能把本轮称TIGER忠实复现、RQ-VAE、模型优胜、平台全test、真实上新、CTR/online lift/causal、OneRec或AI Search全栈。质量与系统数字只引用experiment_log；下一研究阶段由新授权与tasks gates决定。

## Fresh-context review / 最终验收

作者实现与自检ready之后，独立Sol已完成只读final review、无blocker；Astra fresh公平性/hash/公开边界验证PASS并正式裁决P7 Complete。Review检查本P7相对已验收P6 overlay的实际增量、新untracked文档/manifest与验证脚本；P2～P6继承overlay不解释为本P7生产变更。本轮停止，等待Eddy新指示，不进入P8、不commit/push。


## 最终本机公开与Git检查 / 协调fresh复核

作者最终检查真实exit0：73个Git-visible文件（tracked与非ignored untracked union），最大80519bytes，无大于1MiB、checkpoint/npz/npy/parquet/pickle/binary suffix文件。PEM private-key、sk-key、signed-URL signature值定向regex零匹配；P7 manifest另拒绝private absolute roots、实体ID/query/prediction/weights字段和IP。不是以普通“token”等文档术语误判secret。cache六项脚本/aggregate/log/overlay ledger均git check-ignore通过；staged为空，git diff --check exit0。

本P7相对P6只有十项增量：README、STATE、handoff、architecture、data_contract纯状态句、decision/experiment/issue、p7_validation与p7-comparison-manifest。其余dirty/untracked都是逐hash继承的P2～P6合法overlay，不是本轮扩展生产范围。

当前33个untracked逐文件归属如下（全部已读取、无未解释文件）：

| Files | 来源与用途 |
|---|---|
| artifacts/p3-data-contract-manifest.json | 继承P3公开freeze/hash快照 |
| artifacts/p4-baselines-manifest.json | 继承P4公开baseline/hash快照 |
| artifacts/p5-sid-manifest.json | 继承P5公开SID/hash快照，不改历史status |
| artifacts/p6-smoke-manifest.json | 继承P6公开train/decode/hash快照 |
| artifacts/p7-comparison-manifest.json | 本轮新增统一公开绑定，无metric数字 |
| configs/p3.yaml、p4.json、p5.json、p6.json | 继承四阶段冻结config，逐hash不改 |
| docs/data_card.md | 继承P3数据说明 |
| docs/p2_validation.md、p3_validation.md、p4_validation.md、p5_validation.md、p6_validation.md | 继承各阶段验证与历史失败证据，不改 |
| docs/p7_validation.md | 本轮新增scope/verification/学习点/交接 |
| src/vkgr/data_audit.py | 继承P3审计源码 |
| src/vkgr/baselines.py、baseline_run.py | 继承P4 exact baseline/runner |
| src/vkgr/sid.py、sid_run.py | 继承P5 SID与runner |
| src/vkgr/tiger.py、training.py、tiger_run.py | 继承P6模型/训练/runner |
| tests/test_data_audit.py | 继承P3 tests |
| tests/test_baselines.py、test_baseline_run.py、test_baselines_faiss.py | 继承P4 tests |
| tests/test_sid.py、test_sid_run.py | 继承P5 tests |
| tests/test_tiger.py、test_training.py、test_tiger_run.py | 继承P6 tests |

Astra协调已回报fresh远程执行同只读script，UTC2026-10-06T21:40:22，exit0、fairness_pass；另fresh41个protected production/tests/config/AGENTS/lock全byteequal、当时visible文件公开扫描与diffcheck通过。协调复核与作者结果保持独立归属；新增validation后的73文件最终作者检查如上。独立Sol reviewer已审audit/manifest、统一表、config/source pins、纯metadata例外及最后validation/README措辞，final review无blocker；Astra正式裁决P7 PASS / Complete。


### 最终状态收口

协调明确授权仅STATE/handoff/README/本validation更新P7 Complete与P0～P7已验收；统一public manifest和生产/source/config/lock保持不变。本次没有追加SSH、模型replay或tests，仅文档状态与Git diff --check。统一manifest的implementation_ready字符串保留为当时已hash的产物快照，最终阶段状态只读STATE与本节协调裁决。过夜goal工作已达成，协调核对后关闭goal；本worker停止等待Eddy新指示。
