密级: 公开

# 序列用户塔 + 组合叠加：实现者验证证据

2026-10-17。本文是实现者交付索引，**不是独立 review，也不声明 APPROVED、转正或入队**。设计卡 `docs/superpowers/specs/2026-10-17-seq-combo-stack-card.md`；代码 base `c4e4eb8c2a9d87d42303e2c0021e168942fda33a`（与组合部署84b160c同树）。独立 review 由主owner另交 reviewer 一次完成。组合泄漏review尚未APPROVED，submit必须继续阻塞。

## 实现边界

仅新增：原M1依赖两文件、stack模型/runner、三固定config、四个工程脚本、两测试文件、设计卡、工程freeze及本文。原M1来自 `04d4edeb84826d4d56387f95dc820fda1dce10b2`：

| 文件 | 原blob SHA256（已逐字核对） |
|---|---|
| `src/vkgr/retrieval_seqm1.py` | `fc2629da1825393818e1defbf3756cdf3fa4967e715decd96cc12c9fa02cef5a` |
| `src/vkgr/retrieval_seqm1_run.py` | `e81c9183c0cda549a149ed46ae56efc2bfd11e9c3e90f22ff125353e0de559b7` |

原branch、old ROOT、oldrunner、旧freeze及main事实源未改。先构建M1mean共享参数与branch，追加Q2/β零参，最后C/D四表。m的β与Q2位置/周差分别有手算oracle；branch不读feedback/stats，候选only四表与supported mask、初始化共享逐位、train梯度、不可变buffer、target翻转、私有工程参数、manifest/新CP身份与重导出均有定点测试。

新 `real_inputs` 核私有contract文档SHA后对data_root调用原validate_contract，其余load_training_inputs及prepared/query/catalog pins沿原逻辑。source_identity继续来自clean Git source；binding额外记data_root/contract文档与manifest SHA。实际data_root值只存私有ops JSON；job argv仅引用该JSON文件及其SHA，runner进程内读取，不依赖queue worker继承ops环境，不改写公开p3c_validation文档。参数JSON本身与deployment绑定。

## 本地结果与限制

- fresh RED：模型/runner/原M1依赖缺失，6 failures；ops/data-root、私有参数pin与engineering准入分别另有fresh RED。首次sandbox写入被拒导致0 tests的空发现不计RED。
- 最终新增15 tests：GREEN，日志 `cache/seq-combo-stack-local-20261017/seq-stack-final-targeted4.log`。覆盖列于两新增test文件，不用真实数据伪装realflag通过；engineering=False对synthetic输入明确拒绝。
- CPU fullshape synthetic probe：catalog120415、batch512、L50、queries2520、3steps、finite_loss=true；VRAM=UNKNOWN。它验证入口与真实shape，**不是CUDA证明**。收据 `seq-stack-cpu-fullshape.json`。
- synthetic真实CLI train→verify：新产物独立process重导出verified，收据 `seq-stack-cli-proof.json`；仅本轮新synthetic CP被读取。
- 工程freeze：`configs/seq_combo_stack/freeze-v1.json`，223个代码/config/test/card文件pins，SHA `d0a075a31df234c6c35bb280e7a0c059519a9cd06deba7b13506eae05c459655`。生成时逐字核base旧源文件无变化。
- 全仓discover尝试：1192 tests / 305.603s，2 failures、45 errors、3 skipped。该次发现发生在最后3项定点测试添加前；不冒充最终版本全套通过。原47失败项已单独复跑：47 tests / 38.295s，45通过，剩1failure+1error（下表）。最终15定点测试另有fresh GREEN。

| 最终仍失败的原套件项 | 实际限制 |
|---|---|
| `test_retrieval_data.DataTests.test_catalog_pad_and_frozen_identity` | 原validate_contract读取公开checkout合同bound_files时报input hash mismatch；未改旧科学pins或公开文档 |
| `test_retrieval_cli.CLITests.test_synthetic_overfit_and_fresh_process_resume` | 原P4B CLI返回contract/input/state/output gate failed；未更换该旧入口的数据合同 |

初跑其余错误含sandbox禁止worktree outputs临时写入及ps进程探针；取得定点本地测试权限后45项通过。初次复跑工具误把测试method拼接两次，47个loader errors不计实际复跑结果；更正后的rerun2才是上面的47项。完整日志、首次失败名称索引和SHA在gitignored cache，不能将exit0或摘要代替原日志。初跑产生两个未被旧fixture清理的synthetic子进程，按确切PID+完整脚本+PGID核对后仅清理这两个本地测试组；未触碰训练作业。

实际CUDA、新产物CUDA replay、三config真实preflight、GPU显存、dry-run、独立code review、科学结果及延迟均未完成/UNKNOWN。本轮未联网服务器、未push、未submit，未读取真实数据、test或旧checkpoint。公开扫描：新增实现文件未见私钥/credential/IP/本机或服务器绝对路径，单文件均<1MB；不替代后续push前再次扫描。

自动审批曾拒绝“实际data_root直接入job argv”的方案，理由是可能写入作业元数据。该命令未执行；已改为上文私有JSON+SHA方案并获工具审批。没有通过间接方式把data_root值塞入argv。

## 后续工程入口（本轮未运行）

`scripts/seq_combo_stack_ops.py` 从stdin接收显式action和私有paths；PATH_KEYS包含infra_root/deployment/ops/source/python/inputs/data_root/template/ops_library/combo_review，不内置真实路径。主owner在私有通道填值；不得把实际parameters.json放进Git。

流程（按下方修复更新）：inventory→固定工作分支/SHA逐文件deploy→preflight（一次检查3config且输出叠加自身逐项矩阵）→smoke-dry/smoke-submit→status→dry→submit（seal/ready/release原子建立唯一owner）→status；不再发start。CUDA脚本生成本作业新synthetic产物，需fullshape CUDA与commit绑定；CPU证明不被准入。train_jobs还要求combo_review私有收据的SHA、verdict=APPROVED、commit=84b160cb0eeb3c53b877f6cf6474276ba06a6ba5；收据必须来自真实独立review，不得由实现者伪造。

三个train固定seed，一律提交、各唯一verify；无结果筛选、无新增attempt。driver detached、48h截止、900s轮询，raw verified+job/manifest/source/config/data绑定才验收。主owner需按现有runbook核服务器新目录、资源、launcher真实schema与部署权限；尚无真实dry-run证据，不保证环境已接通。


## 2026-10-17 一次 code review 后的普通修复（当前）

对主仓 `2026-10-17-seq-combo-stack-code-review.md` 的HIGH1/HIGH2/MEDIUM3逐项修复，没有再做独立review或改科学设计。原review静态model/builder已核边界引用其SHA `a64c25676d9385e5c5fb37890e7af9b159e195ea12ce760bd32d6fe9a8ffee4e`、reviewed model commit `4e03caa65cde93807593872812c74af380a5dee4`；明确该原review总体为CHANGES_REQUESTED，STATIC_REVIEWED不冒充整体APPROVED。

- HIGH1：完整plan和原deadline先seal；detached持锁owner实际ready后controller release，唯一owner才发布。startup/未release零发布；partial queue发布、intent写入/结果记账失败或link已可见但确认丢失均ALERT；已发布job仍进入唯一verify观察，未发布不重试。submit返回实际ready receipt，不再分开submit/start。
- HIGH2：新增 `src/vkgr/seq_combo_stack_leakage.py`，叠加自身正式loader的512 train+全部2520 validation矩阵，非零Q2/β/g开放。L1/L2/L3/L4/L5/L8/L9各有status/样本数，历史事件/自身反馈反向、branch隔离、还原、week25拒绝、候选training columns/export一致均走自己的helper/model。L6/L7/L10明确STATIC_REVIEWED和源码SHA/原review指针及编号映射。FAIL保留对应item、后续NOT_RUN，runner/ops非零退出且不重跑；submit检查模型身份、完整计数、配置文件SHA、binding/source/data及输入sidecar摘要，并独立要求原组合APPROVED。
- MEDIUM3：新增 `src/vkgr/seq_combo_stack_memory.py`。仅取own PID的nvidia-smi占用（含context），结合torch max allocated/reserved和current reserved，按最大own峰值与max_reserved+最大非allocator样本的较大值保守计量；保留baseline/3train/catalog/user/top50样本。准入验证有限正数、同PID、全部阶段、真实shape，并复算measured及15%/256MiB/floor1024声明；缺样本、NaN、错误公式或global-free method全部拒绝。

修正旁证：原CPU probe把120415误作含PAD总行，实际物品导出120414。此前“fullshape”只证明当时名义shape入口运行，不能继续称实际完整catalog。修复后明确120415物品另加PAD，新CPU收据实际exported=120415、users=2520、history=50、3steps finite，VRAM仍UNKNOWN；旧CPU收据由本更正覆盖，未用它通过CUDA门禁。

fresh RED分批记录：原9个fault/leak/memory guard失败，补intent/result存储、运行计数不足、PAD边界与FAIL exit等均先见RED；修复后原15+新增fault/leakbinding/VRAM测试全部GREEN（当前32项）。`cache/seq-combo-stack-fixes-20261017`保存RED/GREEN/新CPU proof及SHA。fixture显式将synthetic标志改为False仅覆盖production helper分支，不是实际服务器receipt；真实矩阵仍UNKNOWN，禁止拿fixture准入。

本轮按要求仅做定点回归与CPU synthetic probe，未重复全仓suite；前节两项旧合同失败继续保留。未连接服务器、未push/submit、未读真实数据/test/旧CP，未改main事实源或AGENTS三份草稿。真实GPU PID/context样本、真实CUDA、真实矩阵、detached PPID/断线存活、dry-run仍待主执行者在正式通道取得。PASS矩阵也不解除D-1训练侧self-inclusion限制，不作anchor裁决。

本轮更新后的freeze：226个文件，SHA256 `67563d733cd41356191b5c3b4768c2e4ebe72525a42e034f12a48a75aa7cbb8a`，覆盖新增泄漏helper、显存helper和fault tests；此前223-file freeze为历史记录。
