密级: 公开

# P3执行与验收记录

## 批准范围与计划

只执行tasks P3。Astra协调验收，Sol实现；不commit/push，不进入P4。基点0376d9a8119f3500cbf78519675cc7bf18013c1b，原生worktree `codex/p3-data-contract`。P2的11文件未提交overlay逐字节带入，清单保存在ignored `cache/p3-entry-overlay.json`；原P2与main保留。

输入：服务器P2已SHA校验的固定revision MVP train00–24、metadata、source/inventory与固定README。输出：train audit、冻结policy/catalog manifest、holdout统计与data card；大产物只在服务器。

修改范围：src/vkgr/data_audit.py、tests/test_data_audit.py、pyproject.toml/uv.lock、configs/p3.yaml、docs/data_contract.md/data_card.md/p3_validation.md，以及STATE/handoff与必要architecture/decision/issue/experiment记录。AGENTS不改。

- [x] A：synthetic coverage/duplicate/order audit，RED→GREEN。
- [x] B：严格prefix history/首个target/截断/空history与OOV，RED→GREEN。
- [x] C：独占freeze/hash/holdout读取门，RED→GREEN。
- [x] D：server真实train audit，依据train选择like并冻结。
- [x] E：freeze后holdout/catalog feature审计，fresh完整tests与公开边界，交Astra。

接口：audit_rows(partitions)返回aggregate summary；build_queries(history_partitions,target_partition,policy,catalog)返回deterministic server-only queries；freeze_policy(path,policy,train_evidence)返回versioned manifest/hash，validate_frozen_manifest在holdout读取前核验。Parquet adapter只负责固定schema与保序读取；不预建模型。

验收命令：PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v；server python -m vkgr.data_audit train/freeze/holdout；独立SHA复核、git diff --check、ignore/secret/diff检查。

Stop gates：官方ordering声明与固定hash不成立；数据schema/顺序/feature匹配不可解释；train不足1000位like序列长度>=2 users（工程smoke门，不代表统计效力）；无足够history/target且composite不可解释；资源余量不足；任何data/secret tracking。历史最少1个positive，默认最多50个最近positive；train只用长度>=2用户生成prefix/next-positive；每个eval user一个query。所有方法共享同一截断、候选与tie规则。政策细节冻结至data_contract。

## 环境与无效RED

Mac独立tmp uv cache首次offline sync缺wheel，exit1；获准联网sync后PyYAML6.0.2通过。Baseline默认sandbox 19 PermissionError属于worktree临时目录写权限，不算功能RED；escalated fresh baseline通过31 tests。服务器新目录/workspace/vkgr-p3-20261006t1855，独立Python3.11.16/PyYAML6.0.2/numpy1.26.4/pyarrow19.0.1；未改P2或预装环境。

## RED/GREEN ledger

以下保留各行为实际RED/GREEN命令结果与关键输出；最终完整CLI及artifact hashes见v3章节。

A RED: targeted unittest exit1，assertion `P3 audit behavior missing`，1 fail，非environment error。随后最小audit_rows覆盖5行literal oracle、正反馈长度与重复计数。

A GREEN exit0/1 test。A2 RED exit1/5 assertions（invalid IDs/labels、重复或逆序week未拒绝）；missing保持null已GREEN，未伪造其RED。A2最小验证后targeted3 tests GREEN。

B RED exit1/2 missing-behavior assertions；B GREEN exit0/5 targeted。C RED exit1/1 freeze behavior assertion；C GREEN exit0/6 targeted。工程门阈值1000已在真实全train读取前预注册。

D adapter/feature RED exit1/2 assertions；GREEN exit0/8 targeted。Train orchestration RED exit1/1 missing-behavior assertion；GREEN exit0/9 targeted。Holdout首版RED是unimplemented ValueError（未计有效RED），改为期待有效输入返回audit的assertion后fresh RED exit1/1，GREEN exit0/10。CLI/watch/exposure/cofeedback RED exit1/2 assertions；GREEN exit0/12。Freeze完整train文件/strict bool追加RED exit1/ValueError not raised；GREEN exit0/12。

服务器源码包首轮失败：tar有macOS xattr warning；缺outputs容器令19tests环境error，补目录后又暴露缺.python-version；未读train/holdout。补齐公开.python-version/.gitignore后，Git只读复用P2 object context，40tests fresh exit0/1.800s。没有初始化remote或改Git配置。初次logs cache/tests-initial.log、tests-git-context.log、tests-git-context-outputs.log；最终环境log cache/tests-environment-fixed.log。不会把环境error算TDD RED。

Astra在真实train决策前批准最终policy（以train支持门通过为条件）：train每个positive prefix可形成训练例，valid/test每user首真正like target且train-only最近50history，不将validation追加test history；catalog是all train exposures；OOV保留；mask=false；tie item_id ascending。P4采样须固定seed用户抽样，不能按target可达或模型命中挑选，OOV分母保留。

## v2历史source快照（已由v3取代）

当时server source为`/workspace/vkgr-p3-20261006t1855/code-v2`，只解到新目录，旧源码与中间audit保留。code-v2 fresh server43 tests exit0/1.829s；Mac fresh43tests exit0/2.919s。当时将source/lock提交独立Sol reviewer只读核验，review发现问题后进入下述fix round1并生成v3；作者自检不冒充review。

- Source fingerprint：79bda56d718f2572e998261dc210e1fed16acd3e84fa5fa240971f235f97196f。
- audit_module SHA256：951bf91c979ce52cffbb4ad0006ac0481125487573f55702af5dcba1e873e1af。
- uv.lock SHA256：323c9b2c61a2f2baa4fc89769afd5908562f91d15b92672eda71aa57056e3575。
- code-v2.tar SHA256：5870b590e91b6488e27e7a49d8209c6574780d716d3dbda1f7c2a6fa79203473。
- 依赖与容量快照：server cache/p3-final-source-manifest.json。Python3.11.16/PyYAML6.0.2/numpy1.26.4/pyarrow19.0.1；Mac Python3.11.17，同锁依赖。

中间v1 `outputs/p3-train-20261006t1858`不是freeze依据，仅保留执行证据。当时计划使用code-v2 train output冻结，但独立review发现producer/policy绑定缺口后暂停freeze；v2产物仅为历史且禁止freeze。实际最终freeze只使用下述code-v3新train output；新增watch/exposure/cofeedback曾正当化v2的一次fresh完整scan。

Astra独立train复核：UTC2026-10-06T19:04:19.918654，直接pyarrow filter+Counter，不import作者audit。已复核25train文件的行/like/users/长度>=2/catalog统计（数字唯一摘要见experiment_log）；exit0。这是有限协调证据，不代表完整P3验收。

## v2历史执行命令（superseded by v3，不是最终验收依据）

服务器root为 `/workspace/vkgr-p3-20261006t1855`。所有SSH只用已批准alias，真实连接参数不保存；已完整读/etc/vast-agents-guide.md与项目AGENTS。以下仅保存当时code-v2实际执行命令，已由v3替代；当前freeze/holdout命令、exit与hash以“v3真实train与freeze”和“Holdout完成与独立query核对”两节为准：

```bash
cd /workspace/vkgr-p3-20261006t1855/code-v2
GIT_OPTIONAL_LOCKS=0 GIT_DIR=/workspace/vkgr-p2-20261006t1755/.git GIT_WORK_TREE=/workspace/vkgr-p3-20261006t1855/code-v2 PYTHONPATH=src ../.venv/bin/python -B -m unittest discover -s tests -v
PYTHONPATH=src ../.venv/bin/python -B -m vkgr.data_audit train --raw-root /workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183 --inventory /workspace/vkgr-p2-20261006t1755/cache/p2-inventory.json --output outputs/p3-train-final-20261006t1902
```

历史Train stdout/stderr保留`../cache/train-final-cli.log`；v2 train exit0，audit hash `48e30a7c15b4ac583ef4265ef185cdcc6930adbfaca764d1eb5044b809a1e5d2`。v2未执行freeze/holdout，旧audit不能作为freeze依据；最终实际执行记录与hash已在下述v3章节收齐。目录运行Git对象仅提供固定基点，源码fingerprint/hash是实际dirty实现定位；不把基点commit冒充已提交P3。

公共边界：fresh git check-ignore对raw parquet、catalog/examples/query JSON、cache、embedding、checkpoint与index均命中；git ls-files这些目录仅data/README.md，无大产物tracked。未经批准未commit/push/改Git配置/租用资源/训练。

## 独立Sol review — fix round1

Astra安排的独立Sol reviewer只读检查，作者没有自建review。发现两项P1与一项P2：freeze可接收未批准min/max/version；producer/Git/源/lock/环境缺少train→freeze→holdout绑定；train_weeks中bool可误当int。全部属于合同一致性问题，freeze与holdout已暂停，未读取holdout。

Fresh RED：新增policy变化min2/max51/其他version/nestedFalseTrue四个subtest `ValueError not raised`；producer identity缺失一项assertion。R1/R3最小GREEN后R2 producer helper最小GREEN；补旧audit CLI冻结拒绝、实际train policy/producer记录、source/lock/库版本/旧producer/nestedbool在任何reader前拒绝与跨平台CPython3.11 patch兼容测试。完整46tests fresh exit0/3.566s；git diff --check exit0。新bindings集成测试在helper GREEN后追加，真实结果为GREEN，未伪造其单独历史RED。

修复后train记录实际Git base/dirty、完整source fingerprint/module/lock、实际approved policy/hash、PyYAML/numpy/pyarrow精确版本与CPython身份。freeze验证旧artifact与当前producer一致，不给旧产物追写当前身份；holdout在任何holdout inventory/content读取前重复校验。

明确环境兼容规则：CPython major/minor必须3.11，Mac与Linux可使用不同patch并记录各自实际值；source/lock/Git基点及三个库版本必须严格相同。冻结artifact只能由原P3 producer环境重生成；后续P4可校验静态artifact hashes后消费既有queries，不以新的producer静默重生成。

旧v1/v2 audit全部保留并禁止freeze；v2真实train audit已exit0，hash48e30a7c15b4ac583ef4265ef185cdcc6930adbfaca764d1eb5044b809a1e5d2，数字只入experiment_log。须code-v3重新真实audit，不把旧producer记录拼接成新source证据。下一gate为独立review复查及v3 train→freeze→holdout。

Astra另fresh本地43tests exit0/3.041s，是修复前v2有限协调证据，不能替代修复后46tests/re-review。

## 独立Sol scoped re-review fix1

Astra回传独立Sol复查：三项均addressed，source/spec/quality无新阻塞；独立复现policy drift及source/module/lock/dependencies变化均在门前拒绝。修复后source fingerprint `f73c77690a37d0564292e9a1b762349ba574745d6e66c344a26ce9115ccb802d`。获准继续v3真实audit→freeze→holdout；整体P3仍须runtime/文档协调验收，不以review代替数据结果。

Astra在修复后另fresh本地完整46tests exit0/4.063s，命令同标准full unittest；预期拒绝ERROR来自negative test而整套OK。此协调证据对应fix1 source，不代替server runtime与artifact hash核验。

## v3真实train与freeze

v3 server fresh46tests exit0/2.209s后真实train CLI exit0。Final train audit hash `3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe`，其aggregate、features及training-examples hash与v2逐字段一致，新增实际producer/policy绑定；资源/统计数字只见experiment_log。

- Frozen UTC：2026-10-06T19:19:22.701215+00:00，freeze CLI exit0；任何holdout读取发生在此后。
- Frozen manifest SHA256：3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f。
- Policy SHA256：3be46f445676d302c4b4f69cebfff0e4af35c040f09473f1c08d54dfe7507a01。
- Final source fingerprint：f73c77690a37d0564292e9a1b762349ba574745d6e66c344a26ce9115ccb802d。
- Final module SHA256：c7f5f28cdb7e19181fc46f8d69140dba9711ff66f5b8537399692e37e4ded182。
- 版本/lock/Git实际身份在train audit与freeze producer，未给旧产物追加当前producer。
- 本地脱敏manifest副本：artifacts/p3-data-contract-manifest.json；不含任何完整ID集合或逐query数据，与server hash相同。

Astra UTC2026-10-06T19:19:53.060766 fresh只读validate_frozen_manifest/producer通过exit0，独立file_sha256 train/freeze匹配、完整policy核对；这是freeze验收证据，后续holdout hash仍待核。

最终实际命令（固定P3env/Git object context，不变Git配置）：

```bash
cd /workspace/vkgr-p3-20261006t1855/code-v3
export GIT_OPTIONAL_LOCKS=0 GIT_DIR=/workspace/vkgr-p2-20261006t1755/.git GIT_WORK_TREE=/workspace/vkgr-p3-20261006t1855/code-v3 PYTHONPATH=src
../.venv/bin/python -B -m unittest discover -s tests -v
../.venv/bin/python -B -m vkgr.data_audit train --raw-root /workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183 --inventory /workspace/vkgr-p2-20261006t1755/cache/p2-inventory.json --output outputs/p3-train-v3-20261006t1920
../.venv/bin/python -B -m vkgr.data_audit freeze --train-output outputs/p3-train-v3-20261006t1920 --readme ../cache/source-README.md --policy configs/p3.yaml --output outputs/p3-train-v3-20261006t1920/frozen-contract-manifest.json
../.venv/bin/python -B -m vkgr.data_audit holdout --raw-root /workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183 --inventory /workspace/vkgr-p2-20261006t1755/cache/p2-inventory.json --manifest outputs/p3-train-v3-20261006t1920/frozen-contract-manifest.json --manifest-sha256 3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f --output outputs/p3-holdout-v3-20261006t1919
```

对应logs在P3 root cache/tests-code-v3.log、train-v3-cli.log、freeze-v3-cli.log、holdout-v3-cli.log。目录名称只为唯一ID，实际UTC以artifact字段为准。

## Holdout完成与独立query核对

Holdout CLI exit0，audit SHA256 `f9e44a97d410488b7c132abaa27da66076b27917474bcc1adaf0875f46c6f85d`，未修改freeze policy。validation query SHA256 `9077a1c9495d494bc8ec209c0bf3305fd68db5bfcca24abcadc4a1e552e60a97`；test query SHA256 `02b419771860420cb544497518f4ccb0f232c876b4fb14ac3c2fe91d38440b45`。

Astra UTC2026-10-06T19:22:57.997684直接PyArrow重建train positives/history及holdout首like，不import作者audit；所有eligible queries的target/item position是真正首like、history/items positions是固定train最近50、user唯一、OOV标记与query hashes逐条匹配，exit0。具体计数唯一摘要见experiment_log；这不会把source无timestamp升级为独立hidden-timestamp验证。

Freeze后另独立projected uint32 IDs→uint64 pair set扫描全27文件，跨split重复计数与官方subset总规模核对；只输出aggregate，未变source/policy/模型。报告root cache/p3-cross-split-audit.json SHA256 `4e1f0709c385a73c883c988d31eaa2edab2bf9fdf2f827fc8b39c0ec23d4ffb9`。

## 最终artifact清单（server-only，未传Mac）

基准目录 `/workspace/vkgr-p3-20261006t1855/code-v3`：

| Artifact | SHA256 | Bytes |
|---|---|---:|
| outputs/p3-train-v3-20261006t1920/train-audit.json | 3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe | 19512 |
| outputs/p3-train-v3-20261006t1920/frozen-contract-manifest.json | 3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f | 5662 |
| outputs/p3-train-v3-20261006t1920/catalog.json | 80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3 | 941173 |
| outputs/p3-train-v3-20261006t1920/training-examples.json | 9f3fd7c91347d71fe961c9cc52a42436eb0d1613cfbbf6ce9e4fc8ae4a64f0e2 | 31939757 |
| outputs/p3-holdout-v3-20261006t1919/holdout-audit.json | f9e44a97d410488b7c132abaa27da66076b27917474bcc1adaf0875f46c6f85d | 8555 |
| outputs/p3-holdout-v3-20261006t1919/validation-queries.json | 9077a1c9495d494bc8ec209c0bf3305fd68db5bfcca24abcadc4a1e552e60a97 | 1016947 |
| outputs/p3-holdout-v3-20261006t1919/test-queries.json | 02b419771860420cb544497518f4ccb0f232c876b4fb14ac3c2fe91d38440b45 | 889500 |

First-seen/train-users artifact hashes在train audit中绑定；原raw只读复用P2路径，无重下或raw回Mac。本机只有脱敏freeze manifest与临时aggregate摘要，无queries/catalog/examples/embedding。完整ID集合、逐query和大产物全部保持server-only/ignored。

最终code-v3新P1 smoke与verify均exit0，INFO completed/verified；保留outputs/p1-p3-v3-final-smoke，不改任何旧manifest。实际命令：

```bash
../.venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-p3-v3-final-smoke
../.venv/bin/python -B -m vkgr verify --manifest outputs/p1-p3-v3-final-smoke/run_manifest.json
```

最终artifact hashes/size/ignore逐项核对exit0，工作盘free529935724544 bytes。Public大于1MB文件检查无匹配；未commit/push/修改Git配置或计费资源；没有P4代码/模型训练。当时状态为Implementation ready / awaiting coordinator final runtime+documentation acceptance；当前关闭结论见末尾正式验收节。


## 最终公开边界与输入证据检查

固定revision官方README已保存于server `cache/source-README.md`，SHA256 `3b4c3fb2cb2853526cec6145e8e5f20fabb9efbd33ae024f9f96948237f3538e`，freeze绑定同一hash。Primary source：[固定README](https://huggingface.co/datasets/deepvk/VK-LSVD/blob/084a18386e149a454dd6dbe0123478e75c6b09ce/README.md)。其Interactions段给出GTS及文件内timestamp递增assertion；官方subset示例保持left order。真实文件只有合同列出的12列，未暴露timestamp；本项目核验schema及物理读取顺序，不声称独立验证隐藏时间。

最终本地source fingerprint、YAML批准policy与公开manifest逐项匹配；source保持fix1/re-review版本。Secret/private-key/signed-token模式扫描无匹配，trailing whitespace扫描无匹配，`git diff --check` exit0。Inherited P2 overlay中README、mvp/scale配置、server_runbook、tasks及p2_validation仍与P2原worktree字节相同；overlay原清单保留于ignored cache，P2与main未修改。以上只为公开边界和工作区验证，不扩大为完整P3最终验收。


## 最终文档review修订

独立Sol final review确认public manifest/hash/producer通过，source/spec无新阻塞。仅文档修订：P3 issue全局唯一编号改为I-028（打包环境）、I-029（irX未知）、I-030（隐藏timestamp限制）、I-031（policy/producer绑定）；P2原I-024～I-027保留。v2命令与计划明确为historical superseded by v3，I-031保留fix1历史并追加实际v3 runtime闭环。源码与artifact未改，未重复tests或raw扫描；当时整体P3等待协调者最后文档验收；当前结论见下述正式验收节。


## P3正式验收结论 — Complete

Astra已正式关闭P3：独立Sol最终spec/quality及scoped文档review均通过；协调者fresh46tests、独立train统计、freeze/producer、全部query oracle与hash核验均通过。以上分项实际命令、exit、hash及边界见本记录对应章节，不以单项测试替代完整gate。

下一task P4已在Eddy过夜授权范围批准，由新Sol worker复制最新未提交overlay后执行。本次仅更新关闭状态，不改源码/lock/manifest或重复tests/raw扫描；未commit/push。data_card/issue_log中的ordering、sampling、train-only test history、高OOV与observed-firstseen限制仍保留，P3未产生模型效果结论。
