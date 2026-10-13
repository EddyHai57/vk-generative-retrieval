# G1 validation

密级: 公开

Status：本地已通过Astra fresh验收；Complete — Astra fresh local + actual remote artifacts accepted；**real SID cross-input audit尚未执行，G1未Complete**。

## 做了什么与边界

Astra实际review design-v1后批准本地TDD及两项scope调整。原docs/data_contract.md保持原bytes；新增docs/generative_contract_v1.md仅token监督、consumer/producer身份与技术gate，共同数据规则仍引用唯一原合同。provenance/g0-v1是不可修改历史replay，不是活跃代码。

新增只读模块src/vkgr/generative_data.py复用P4B prepared验证、P5 static SID loader与SIDTokens；不调用旧双split/raw audit、P6runner或跨epoch cursor。没有G1 optimizer/真实训练、SSH/GPU、安装、下载、commit/push；未进入G2、R5/OneRec。fullsuite既有synthetic训练tests照常运行。

## 实际接口与输出

|接口|实际输入/输出|
|---|---|
|validate_legacy_capsule(root, inventory_sha256)|externalpin、37原member/全部父链/path/type/size/hash/源anchor；返回库存，禁止额外成员/目录/symlink/duplicate/..与rehash原bytes|
|build_contract(root)|只读公共声明和实际33-member新consumer identity，返回versioned manifest；不消费server数据、不hash自身|
|load_contract(root, manifest_path, expected_sha256)|先external SHA再strict JSON/typed canonical全部声明与实际root/env对账；返回sealed dict context，后续改字典拒绝|
|load_prepared(manifest_path, contract)|先expected prepare SHA再artifact；验证固定schema/producer/inputbindings/every-prefix/fullpast/最近50/重复/support/idmap；返回readonly catalog、mmap content含PAD0、examples_path/count、manifest与pin；不把全部examples常驻list|
|load_sid(manifest_path, contract, prepared)|prepared context必须对应当前contract prepare SHA；P5静态身份/artifacts/evidence/config校验，再catalog及content[1:] float32 byte digest等同；返回readonly validated SIDTokens codec|
|load_validation(path, contract, catalog)|固定validation role/hash且不同于test声明；test/unknown候选pre-open拒绝；strict typed query、past positions、OOV与原population保留；返回原queries；不能由成品重新证明first-like最早，依赖已验收producer/query hash|
|sid_targets(rows, codec)|只读取history/target并拒非法/OOV/超50；返回CPU source、BOS+前3digit decoder、全部4digit labels，无past-positive mask；loaded codec只做O(1) identity gate|
|audit_sid(codec, prepared)|完整SID合法/可逆/可达与原audit；固定train-support leaf相邻item pairs最多4096的float64 squared L2 aggregate；无N²矩阵，resources=not_measured|

机器config configs/generative-core-v1.json已冻结批准的core_search/pilot/evaluation/selection/diagnostics/test_gate声明。Beam100仅近似ranking；共享item指标，不将SID hit当item hit。Random SID后续需独立版本化mapping/manifest，不放宽现有canonical suffix gate。声明9runs不等于启动授权；pilot/GPU总时长/唯一调度者gate仍有效。

## 实际artifact与历史证据

- 新manifest artifacts/generative-contract-manifest.json SHA `40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d`，actual new consumer SHA `78ce05fdf7ec41959d3595c04832dcc243ca8b1241d78d6e07163a190b42db81`，33 explicit members，无self-reference；real_sid_audit=pending_server_authorization，test_gate=closed。
- capsule精确37原成员，共427387 bytes，加inventory一个文件；inventory SHA `15e4a8d9b7f063f4b2476e95bf75c410249ed9294a885bcb1593841015f13030`。来源逐member写inventory.sources：G0 consumer_source.files、P3-D bound_files或G0 fixture/external manifest pin；外部G0/P3D anchors独立约束，不因重hash库存接受旧内容。
- 原G0公开manifest SHA `fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59`与data_contract SHA `d3a2528731d8f5b9160039b6236e519c761f618d54a4f34939ef4df5ea70efa4`不变；原锁与源码全部保持。完整范围证据cache/g1-implementation-entry.json及cache/g1-final-scope.json。
- fresh独立进程实际__file__指向capsule/src的原G0 verify成功；另实际CLI cwd=capsule exit0：`G0 source/input/CPU computation/item metrics replay verified; test gate closed`。cwd当前根同一旧manifest exit1：`G0 synthetic input/source/output gate failed`，这是新增source后正确历史身份门，未修改旧入口/manifest；现有G0tests动态生成current manifest仍通过。
- 正式新manifest实际load_contract exit0：`G1 consumer/static declarations verified; test=closed; real SID audit=pending_server_authorization`。只核静态历史声明，不称旧P4B/P5 producer replay。

## TDD与fresh验证

解释器实际 `<LOCAL_VENV_WORKTREE>/.venv/bin/python`，仅借已核解释器；Python3.11.17，torch2.11.0、numpy1.26.4、pyarrow19.0.1、PyYAML6.0.2。新consumer的实际环境与旧server producer声明分别绑定；服务器环境不同须显式冻结该部署版本并fresh验收，不放宽历史依赖。

|行为|fresh RED证据（exit1）|最终GREEN证据（exit0）|
|---|---|---|
|capsule外部pins/越界/symlink/独立replay|cache/g1-red-1.log：新模块缺失|cache/g1-green-1.log，3tests|
|strict新contract及历史source独立|cache/g1-red-2.log：build_contract缺失|cache/g1-green-2-final.log，5tests|
|prepared完整prefix/typed/input身份|cache/g1-red-3.log：load_prepared缺失|cache/g1-green-3.log，8tests|
|SID/catalog/content/suffix/内容audit|cache/g1-red-4.log：load_sid/audit_sid缺失|cache/g1-green-4.log，11tests|
|validation-only/OOV/pre-open/typed|cache/g1-red-5.log：load_validation缺失（另一fixture import错误保留）|cache/g1-green-5-final.log，14tests|
|四位置literal teacher forcing|cache/g1-red-6.log：sid_targets缺失|cache/g1-green-6.log，16tests|
|prepared context直接当前合同绑定|cache/g1-cross-input-red.log：ValueError not raised|cache/g1-cross-input-green.log，17tests|
|batch成本边界/readonly fast-path|cache/g1-review-fix-red.log：120次全catalog扫描、hotpath fullaudit触发|cache/g1-review-fix-green.log，19tests|
|readonly ndarray layout mutation|cache/g1-layout-red.log：wrong OOV或未拒绝strides mutation|cache/g1-layout-green.log，20tests/3.085s|

初次GREEN的fixture失败如实保留：第二组临时root漏P3-D bound files，补全fixture；第五组build_queries误从baselines导入，改回data_audit，未放松生产校验。首个RED shell使用zsh readonly变量status导致额外shell报错，unittest RED本身已执行并log为新模块缺失；后续用result_code。不存在抹去失败或伪造成功。

最终fresh完整suite：**193 tests / 24.697s，exit0，既有FAISS skip1**；cache/g1-final-regression-after-review.log SHA `8427ff2400552f415b49bab2e82ac4218d9bf067a03ad20b9cc0e6c02949e1a6`。首轮189及中间192tests是历史snapshot，最后源码已fresh重跑193。独立GPT-6.1Sol/medium只读review发现P2性能及后续P2strides漏洞，均TDD修复并scoped re-review ADDRESSED、无新发现；cache/g1-review-result.md，不冒称reviewer重跑suite或验收真实server。

实际完整命令（cwd本worktree）：

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m unittest discover -s tests -p test_generative_data.py -v
OMP_NUM_THREADS=1 PYTHONPATH=src <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m unittest discover -s tests -v
OMP_NUM_THREADS=1 PYTHONPATH=src <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -c 'from pathlib import Path; from vkgr.generative_data import load_contract; root=Path.cwd(); value=load_contract(root,root/"artifacts/generative-contract-manifest.json","40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d"); print("G1 consumer/static declarations verified; test="+value["test_gate"]+"; real SID audit="+value["real_sid_audit"])'
git diff --check
```

原capsule实际CLI，cwd为本worktree/provenance/g0-v1，使用相同Python/OMP/PYTHONPATH：

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src <LOCAL_VENV_WORKTREE>/.venv/bin/python -B -m vkgr.generative_foundation verify --manifest artifacts/g0-foundation-manifest.json --sha256 fbcb5c6dc41e9144420a79cd1d26926b93c269a84adf7dc837ca4c1217781b59
```

G1没有CLI/runner；build_contract+exclusive写manifest实际命令为 `/private/tmp/g1_freeze.py`，冻结receipt cache/g1-freeze-evidence.json。测试封装synthetic context仅用于隔离loader行为；P5 bundle也标明synthetic资源metadata，不代表正式load_contract接受任意新pin或真实资源测量。

## 未完成项与最小后续只读server审计

1. Astra fresh本地验收仍待执行；author/reviewer声明不代替Astra。
2. 真实prepare manifest、SID run manifest、全catalog mapping/constraint/codebook/normalizedfeatures、完整examples/support/idmap及validation queries未读取；全catalog跨输入same-content、train-only距离与真实收敛结论未完成，不能G1Complete。
3. 需Astra明确授权有界只读server audit，并与唯一传统Sol调度者协调，不中断/重复提交原训练。若服务器尚无本版consumer，公共源码部署也须另明确授权；核实际source与exact环境，显式冻结server consumer版本及external SHA，否则STOP。审计只调用load_contract→load_prepared(PREPARED/prepare-manifest.json)→load_sid(SID/run_manifest.json)→load_validation(HOLDOUT/validation-queries.json)→audit_sid；所有真实路径仅runtime参数，不回传raw/maps/逐query，也不打开/hash实际test文件或重fit producer。
4. 只输出artifact hashes、绑定通过/失败及aggregate；资源未实际监测保持not_measured，有授权才记录实际headroom。任何source/catalog/features/时序漂移、缺证据、implicit test读或需改原sharedsource立即STOP。没有可信pilot/runtime上限，仍不得启动G2训练或G3GPU。

## 最终scope/public扫描

与实施入口库存逐hash比较：只改STATE、handoff、G1validation、issue四个允许文件；111个原文件bytes不变；新增43个文件=五项G1产物+37成员capsule及inventory。原G0manifest/data_contract/lock hashes不变、HEAD不变，git diff --check exit0。cache/g1-final-scope.json逐文件hash可fresh核对。新增/变更47个public文件均为小型UTF-8 text、regular file、无binary/raw/checkpoint/map/index；secret/APIkey/signedURL/connection-address扫描无待处理发现，不声称重做全仓库第三方license审计。

初次numeric-address正则把原已pin uv.lock的四段NVIDIA package versions误报；按TOML package version字段及files.pythonhosted.org官方wheel filename实际分类21处，原锁字节没变，地址门继续保留。首次扫描Python assertion确实失败，随后同shell git diff --check返回0不能作为扫描成功；最终改set -e并独立审计PASS，不隐藏此失败。

## G1 server audit执行准备（2026-10-08，proposal only）

Astra明确确认已fresh193tests/26.566s PASS(skip1)、原capsule replay、新合同load和158文件hash复核；本轮只接收该协调方实际验收证据，没有重跑suite或假造其log路径。原实现/合同/source pins保留。

自包含最小方案cache/g1-server-audit-plan.md；80原public文件+2拟操作草稿+1expected server合同共83库存member，加inventory后84regular-file public tar，逐memberhash见cache/g1-server-transfer-inventory.json。独立bootstrap stdin操作另列hash；完整精确receive/extract/audit命令只在ignored private command文件，路径来源本地P4B/P4D private index与已验收P5文档，不含secret/新连接配置，也未证明server现存。public tar不含private bindings/命令、raw或真实map/features。

Mac合同不改；另预注册server variant仅consumer.python=3.11.16、torch=2.11.0+cu128两处区别，expected SHA4bd3d093d4f86f4a033eda293798b807e46dd9d8ea0af68240481972d3e000a7。现场actual source/Git/env必须完整equal并用审核端externalpin load；不是放宽Mac校验。资源提案1CPU/nice19/idleIO，worker RSS采样2GiB、IO采样4GiB、initial available≥4GiB且host/cgroup headroom≥20%，本audit CPU1500s/wall1800s保护窗口；明确采样overshoot/worker-only观测边界，不称instantaneous kernel guarantee或全局8h截止。任何门失败只停本audit process group，保留产物，不碰传统队列。

本轮只在ignored cache准备公开包/expected variant/草稿、AST parse、shell -n及本地archive83memberhash；未运行草稿、未SSH/上传/部署、未读真实数据/资源、未训练。cache/g1-audit-plan-local-check.json为proposal integrity而非runtime成功。新方案等待Astra核83member+inventory+bootstrap/命令/资源/input scope后放行；真实G1审计验收后才G2，再pilot/固定预算G3。

最新授权唯一指针为主目录tasks.md末尾“两级总目标与自动推进授权”：双线同时推进，生成式不等传统整体P4G；subset传统P4G+生成式G5经Astra验收后自动S0规模准备、通过规模gate后下载/训练，不等Eddy在线。计划内有实测预算且现有资源满足时Astra可放行，不需Eddy再次在线；新资源/未知成本/资源不足仍STOP。本轮仍唯一G1，无提前扩数据。最新用户授权高于旧“Eddy再次确认”操作文案，本轮不改任何冻结config/技术合同byte。

## G1 audit revision v2 — HOLD修复（2026-10-08）

v1不得部署。Astra指出leader退出漏grandchild、异常/TERM无finally及固定cgroup root观察不可靠。fresh RED实际grandchild泄漏及缺安全lifecycle；cache helper改固定start_new_session PGID、所有exit finally收组，TERM→宽限→存活KILL，不依leader是否活；受控TERM/INT/HUP在spawn/cleanup期间阻塞以防重入，恢复handlers/mask。首轮GREEN遇到Mac已消失group的PermissionError，未当通过；改live member观察后测试通过。独立review又发现namespace root重基隐藏祖先，新增fresh RED，加入可信host-initial namespace external inode gate，缺失/default0/不匹配在worker之前STOP。不是以self/PID1现场值信任自己。

最终本地6tests PASS1.318s；独立Sol fresh review6tests PASS1.354s，无新增Important finding，实际Linux有效性仍未证明。测试覆盖leader先退出与ignore TERM grandchild、KeyboardInterrupt/RuntimeError/SystemExit、正常结束+外session隔离、实际重复TERM、membership/mount/祖先限额及未知拒绝、namespace proof缺失/mismatch。INT/HUP共享路径经review，未单独发送真实信号；无法保证supervisor SIGKILL/机器崩溃/worker主动脱session后的收口，当前固定worker无脱session操作。

可信host-initial inode证据尚无，当前v2实际执行受gate阻塞，不能称server audit ready。既有轻量CPU窗口通知不代替scope批准。只改cache helper与3允许文档，原冻结源码/合同/manifest/83库存中82成员不变。原始audit.stderr可能含private paths，只留server cache0700目录，不回传或声称脱敏；诊断仅白名单error type/脱敏原因。

proposal cache/g1-server-audit-plan.md；旧新diff cache/g1-audit-v2-diff.json、g1-supervisor-v1-v2.diff；6test及两组RED日志留cache。新版外部pins见g1-server-plan-meta.json，所有tar84/库存83成员fresh逐hash通过，bootstrap/expected合同不变。v1产物保留并禁止部署，无SSH、真实读取或训练。

## G1 v2资源scope收敛裁决（2026-10-08，覆盖上一版inode门）

Astra明确：本1CPU/RSS2GiB sampled bounded只读audit接受hidden host ancestry unknown，不要求宿主initial namespace身份。已移除新增inode门，未伪造身份。正确映射/proc/self/cgroup与mountinfo、核全部visible ancestors取tightest有效finite limit；无可见有限limit/映射失败/缺controller/读错STOP，禁止host-only fallback。保留原资源上限、testclosed、fixed-owned-PGID全部exit收尾及server原始stderr不回传。这不放行全量/并发训练，也没有扩大infra。

新fresh RED subtree mapping被旧门拒绝；最终local6tests PASS1.345s，独立fresh review6tests PASS1.357s，无actionable findings。此前namespace finding因明确scope裁决转为接受的unknown观测限制；真实Linux gate尚未验证。本轮系统Pythonstdin遇UTF8解析错误，未产出包，改已知Python后fresh重建PASS，不冒用失败exit。

收敛v2 tar SHA1ed6a01a7818ebdddaa98b9e1c8d1baad29deca94242a553530e4d9fa48d094a；inventory SHAb80f58cfe75de5f68431d01511590a759237b9ec80edca7a8d7baf7d87bcc939，84regular/83库存，tar1054720bytes/member983696bytes。只有supervisor member变化，82成员不变；expected合同/bootstrap/冻结源pins不变。新旧diff、local-check、regression与精确command pins留cache。v1禁止部署；所有v2先前hash被本收敛版替代。下一步仅最终包scope review，不SSH/部署/真实读取/训练，G1未Complete。

## G1真实bounded CPU审计（2026-10-08，尚待Astra验收）

Astra已正式批准最终收敛v2固定tar/inventory/expected合同/bootstrap/精确命令一次exclusive部署。已先读runbook/server guide及目标祖先AGENTS；无安装、服务/资源生命周期或GPU操作。fresh preflight实际env exact tuple、Git base、13输入stat size/roles、visible cgroup/RAM/disk及exclusive目标门通过，再执行已pin receive/extract/audit wrapper一次，真实command/worker exit0。仅新exclusive目录写入，旧producer/输入read-only。

聚合入口cache/g1-real-audit-aggregate-evidence.json，包含actual aggregate/supervision小payload及server合同hash。实际server合同与预注册external4bd3...pin完全equal，实际source/p3d/prepare/P5/validation全部保持；本地canonical aggregatehash复算、fullcatalog roundtrip/reachability/无fullSIDcollision/deadend、4非法digit/suffix拒绝、train-only bounded距离群体及资源caps核对通过。对应receipt cache/g1-real-audit-local-verification.json；数字只保留ignored小聚合供Astra核并写主experiment事实源，本轮不另设公开指标副本。

原P5未收敛节点如实保留，不refit或称模型语义质量通过；只能声明static input identity/legality/train-only bounded距离。hidden host ancestry unknown按本audit限定接受；RSS/IO仅worker sampled有overshoot，短Git子进程不计入，gd.audit_sid自身resources仍not_measured。原始audit.stderr可能含private paths，只留server受限目录，从未回传/打开；只取小聚合/receipt及合同hash。test未打开、CUDA未初始化、无fit/训练/预测/逐query或map回Mac。

command与preflight实际exit/UTC/hash、聚合artifacthash都在cache receipts；一次launch marker防重复，失败不retry规则保持。worker通过不代替Astra验收，当前G1尚未Complete，下一步只交真实证据等fresh验收，不G2。冻结生产源码未变，不重跑无关suite；最后范围检查与diffcheck另有fresh记录。

## Astra正式G1验收（2026-10-08）

协调已明确宣布G1 Complete：原fresh193tests、独立scope review及本轮SSH fresh实际aggregate/supervision/server-contract逐byte hash/群体/legality/resource gates匹配。实际证据/private/tmp/g1-astra-remote-verification.json（无真实数据），本worktree cache/g1-coordinator-acceptance-receipt.json记录其hash及三产物身份。Astra最初多拼code-v1导致只读路径失败，改到实际deploy_root/cache核对通过；没有重跑审计或server修改。所有旧pins保持，原未收敛及sampled/hidden ancestry限制仍在，不变成质量证明。下一唯一task G2设计，非直接实现或GPU授权。
