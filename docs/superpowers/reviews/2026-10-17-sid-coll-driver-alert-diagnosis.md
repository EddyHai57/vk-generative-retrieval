密级: 公开

# 协同向量 SID：训练成功后 driver ALERT 的只读诊断

2026-10-17。结论：已确认一处**验收计划的 config SHA 寻址错误**，足以使两支成功的 train42 被 observer 拒绝；当前不是可据此登记的训练失败或已验证科学结果。新原始回执中，两支 expected 各9项，8项相等，唯一差异都是 `binding.config_sha256: null` 对真实冻结 config SHA。C2 原始 summary 的异常精确为 `ValueError: raw proof / frozen identity mismatch`；C1 已进入 persisted stop，原异常文本在最终摘要中被覆盖，但同一唯一矛盾仍明确存在。

本轮未执行修复、服务器连接、driver/tick/start、排队或训练，没有读取 checkpoint、真实数据/test，也未读取或解释 train receipt 的 raw_proof/summary 召回指标。仅 Git show 固定 SHA、现有 metadata JSON 字段比对和本报告新增；未 commit、未改事实源。context 计数 UNKNOWN。

路径用占位符：`<S1_DEP>`、`<S1_OPS>`、`<RUNS>`分别指本次私有请求指定的科学部署、ops-v2、infra runs；实际映射沿 gitignored `cache/codex-sid-coll-s1-20261016/capacity-proposal.json` 与本轮 transport 收据，不把私有服务器路径写入公开报告。

## 硬证据与逐字段比对

| 证据 | SHA256 / 核对结果 |
|---|---|
| `cache/pgr01-p-recovery-v1/main05/s1-alert-r1-admission-20261016T205330Z.stdout` | `4930cc5d8c75dbb1c19788ea07234f534b5a4dcd6b18d926c2325d0c7b174401`；实际读取 UTC20:51:23，两 train42 terminal succeeded/exit0，stderr为空；driver2931675 alive=false |
| `cache/pgr01-p-recovery-v1/main05/s1-receipts-readonly-20261016T205650Z.stdout` | `8692041fcbbfae57dd7907f29245f8990cb448db49443a3380df32a008c4b3ea`；实际读取 UTC2026-10-16T20:55:04.940818+00:00，包含当前capacity plan、两train receipt及final summary |
| `<S1_OPS>/pipeline-plan-capacity-v1.json` | `b14630f9b2be83b12e4d21503b7a5d9ce04f44c1b5b277ce50ff20d7a3d5753d`；created_epoch1792161406.75411，deadline1792334206.75411 |
| `<S1_OPS>/controller/summary-final.json` | `f91375d235909ef7128548bdb576ae2e8f8c3b285067edb83eddf2325da14078`；C2 train42 exception为上述identity mismatch，event_wait=linux_inotify/own_terminal(C2)，fallback_seconds=900，new_submissions=[] |
| C1 train42 receipt | `5fd4559da44103827f0b03ea629195996b79e1ed5a0971abb99d105994a58108`，与原stdout打印的receipt引用一致 |
| C2 train42 receipt | `2e40712e25d9247633f023486b228cc32e9c07d031124134028c7ccf71ffa7ad`，同样与原stdout引用一致 |

两份 receipt 均位于 `<RUNS>/sid-coll-c{1,2}-train42-v1/out/receipt.json`，status=complete。只比较原plan明确要求的下列9项，不碰未验收指标：

| expected字段 | C1p | C2 |
|---|---|---|
| schema、status、stage、variant、run_id | 5项全部相等 | 5项全部相等 |
| binding.commit | 相等，11d67681e0cbde09983f6bf482f55ab8a7191bdb | 相等，同commit |
| binding.spec_sha256 | 相等，13aa0e28c9af1b38c6989bf1cf216d895f48d861bb90543e29b494c530f666a6 | 相等，9b036d5a4f2411533917f05d6abee40c4420d200c0cbbcb88257a925c44ff5a8 |
| binding.data（整个字典） | 完全相等 | 完全相等 |
| **binding.config_sha256** | **expected null；actual d6e4ebba1971788b593e17433d06c20c85eea570cd0ec0a7146d5da53d45818a** | **expected null；actual 788248e0d71dec364b7218b3237d3e8f1702824a26d7e92598b794e55a217277** |

actual config SHA 各自等于 receipt.binding.data.config42，也等于固定科学 commit 的相应配置 Git blob SHA；不能为让验收通过而删除该检查或从未验收分数反推expected。这里是纠正指向既有冻结配置的字面值。

train42 plan.job 与原claimed job的其它字段逐项相等；唯一字面差别是 plan 的动态 `@cuda.raw_proof.declared_vram_mib` 与队列已materialize的2560。应按既有accepted CUDA receipt解析后再比，不能把未解析token直接当作job被改写。本轮未另取该CUDA receipt；source inspect_stage本来在proof比较前就严格检查materialized queued job相等。

## 固定源码给出的因果链

只读 `11d67681e0cbde09983f6bf482f55ab8a7191bdb`、observer `1fa96dc`、`b303f1c` 的 Git 对象：三版本 `scripts/sid_coll_ops.py` 字节相同（SHA `9358f00a57c4355542b0e63c8c0c4911fb61437ec1ef1ebfb1276aa9289e10bc`）；job entry亦字节相同（SHA `383172e46676c417fc190f8ce32d2a97db60cb3888269a8507357ade7d88740a`）。换observer版本本身不会消除错误expected。

1. `scripts/sid_coll_retry.py:25`令 `params=spec['stages']`；primitive `params['train']` 的结构是 `{run_id, parameters:{seed:42,...}}`。第59行却用 `params[generic].get('seed')`，取得None，再找不存在的 `refs['configNone']`，因此写入null。它应按既有spec的parameters.seed定位config；本轮只指出错误，不改代码。
2. `sid_coll_job_entry.execute`使用内层 `params=spec['stages'][stage]['parameters']`，train/verify实际写 `binding.config_sha256=pin('config'+str(params['seed']))`，所以真实训练receipt写了正确非空SHA。
3. `sid_coll_ops.inspect_stage`先核terminal succeeded/exit0、唯一claimed及job身份，再逐项比较proof与frozen expected；任一不等抛 `raw proof / frozen identity mismatch`。
4. `tick.advance`捕获该异常，把具体exception写进当次stage snapshot，再将持久alert.reason统一记为`stage train42 failed`。以后tick先看到alert就短路，结果显示`persisted stop...`，并用UNKNOWN补齐阶段；它不会因后来terminal成功而自动清alert。
5. 两branch均ALERT时driver按设计写final并退出。所以driver退出与raw train成功可以同时成立。C2历史异常已直接确认；C1原首个exception文本当前UNKNOWN，需当次summary才能补历史原文，但当前同一唯一阻断已定。

当前capacity plan两支的train42/verify42/train2024/verify2024/train2025/verify2025共12个expected config SHA都为null，不能只修train42后又卡在verify。六个冻结Git配置摘要如下，各适用于同seed的train和verify；当前plan的binding.data对应config字段亦全部与这些摘要相等：

| 分支 | seed42 | seed2024 | seed2025 |
|---|---|---|---|
| C1p | `d6e4ebba1971788b593e17433d06c20c85eea570cd0ec0a7146d5da53d45818a` | `5afa1e52d07b568d782dead2ccacf3a51c79c254f0b6827272f1d04720446e60` | `e2c3d771e57813f490cc23bf25f14783c7f776e01725fc537515ebdb88fe1bf7` |
| C2 | `788248e0d71dec364b7218b3237d3e8f1702824a26d7e92598b794e55a217277` | `eeb864f52ade96b8e9163b30536d715f8e3a786a354fc9243ed00fb92d51b1dc` | `2ca6ba9e2a4f1eb82bf6c24ac0bf182764d4e1b41185d548f3b8a15a3de18ee9` |

## verify是否未提交，以及还需什么小metadata

源流程在train42拒绝后不进入verify42；final summary中两支verify42均UNKNOWN。**这强烈支持原driver没有发布verify，但不是全局“从未提交”的证明**：本轮新回执没有verify队列/run/intent盘点，不能忽略其它owner或不确定intent。

定因所需四JSON已经取得：capacity plan、两train receipt、summary-final，路径/SHA见上。若要补历史与准备安全恢复，最小再读：

- `<S1_OPS>/controller/alert-C1p.json`、`alert-C2.json`（reason与at_utc）；按该时间定位当次`summary-<time_ns>.json`，只提取对应branch.stages.train42.exception，避免只读末次persisted摘要而丢C1原异常。
- `<S1_OPS>/controller/intent-sid-coll-c1-verify42-v1.json`与C2对应文件，及两verify的`<RUNS>/sid-coll-c{1,2}-verify42-v1/{run.json,terminal.json}`存在性。
- queue pending/claimed/rejected中这两个**确切run ID**的条目及rejected reason；若没有条目只记不存在，不创建文件。claimed条目如果存在则核full job/manifest绑定，不能重复提交。
- 原driver-pid-v3.json、对应进程/持锁状态的fresh identity；原plan与所有已有intent的SHA/存在性盘点。只读metadata，不开CP、预测文件或test。

## 最小安全恢复提案（待 Eddy 单独批准，当前不执行）

需要批准的是越过本轮明确的“不碰S1作业/driver”边界并恢复原已授权工作流；常设非训练运载重验不自动授权清STOP、启动新owner或重投verify。建议先准备可review的逐字段diff与新metadata计划，再向 Eddy 申请以下精确范围：

1. **新版本验收计划**：建议新文件 `<S1_OPS>/pipeline-plan-configsha-recovery-v1.json`。只将上述12处 `branches.<label>.stages.<train|verify><seed>.proof.expected["binding.config_sha256"]` 从null改为对应**既有冻结配置**SHA，保留其余proof expected、科学spec/source/config/data、job对象、run IDs、资源与预算不变；生成器未来修复另用版本化metadata helper，不能在科学DEP直接改代码。
2. **保留原时间与证据**：created_epoch仍1792161406.75411，deadline仍1792334206.75411，不重置spent/attempt；原capacity plan b146、原controller/STOP/所有摘要、失败目录与raw训练产物均只读保留。不是删除alert后重启旧driver。
3. **唯一新owner状态**：建议新control_dir `<S1_OPS>/controller-configsha-recovery-v1`，新pid/ready/summary文件，不覆盖旧driver-pid-v3。独占确认旧owner已退出且原lock释放后才接手，记录实际ACK再声明接收。因为sid_coll_ops按control_dir读取alert/intent，**新目录不能丢旧intent保护**：先固定并核对旧intent清单，对已存在intent保留原字节的guard副本或等效显式冻结引用；有uncertain intent就STOP，不能借新control_dir规避唯一attempt。
   接线注意：原 `start` 明确拒绝已经存在的control_dir；若先写intent guards后再调用原start会被拒绝。真正执行时需获批的、版本化metadata-only handoff launcher负责新目录/guard封存，再以原`driver`入口启动并满足PPID=1、唯一lock和新目录内`driver-pid.json`条件；不能原地改科学DEP或偷偷绕过STOP。此处只指出既有接口限制，不提供或执行启动代码。
4. **新recovery声明建议字段**：`source_plan_ref`（原path+b146 SHA）、`source_control_dir`、`scientific_commit`（11d全称）、`new_plan_ref`、`expected_config_patch`（12字段前后值与配置Git SHA依据）、`original_created_epoch`、`original_deadline_epoch`、`adopted_train_receipt_refs`（两原path/SHA）、`preserved_stop_refs`、`preserved_intent_refs`、`existing_verify_inventory_ref`、`old_owner_identity_ref`、`approval_ref`、`new_owner_ack`（未取得前UNKNOWN）。这只是建议schema，不声称旧driver已支持全部字段。
5. **采纳原train、首次补verify**：重新做严格receipt身份验收，采纳已有complete train作为待verify输入，绝不重跑train42。仅在verify原ID确认无run/queue/rejected/不确定intent时，按原spec发布各自首次verify；若verify曾失败或拒绝，不能按常设一次工程重验自动重投，另交 Eddy/Kiro。
6. **条件补seed保持原规则**：只有原始verified及job/manifest/config/source/data绑定全部通过，且verified summary的整数hits≥260，才按原配置接续2024/2025及各唯一verify；未verified不读分数筛选，不修改260，不新增seed/名额。科学源码/既有job immutable，另一独立线路照常不受影响。

上述恢复目前未批准也未执行；实际新owner、verify发布状态、原科学结果验收均UNKNOWN。本报告只给出了可审计的metadata错误与下一步授权边界。
