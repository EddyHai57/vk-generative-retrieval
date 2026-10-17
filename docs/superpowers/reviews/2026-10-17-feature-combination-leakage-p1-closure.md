密级: 公开

# 原组合泄漏 review：P1 真实路径证据关闭

日期：2026-10-17。**Verdict：APPROVED（仅原 P1 补证覆盖）**。对象：`84b160cb0eeb3c53b877f6cf6474276ba06a6ba5`。

原 review 唯一阻塞的“L1–L10 四变体真实路径运行证据缺失”已由本次获批的工程重验回执覆盖。原 review 与本文合读：§7.6① 的证据缺口关闭，原组合第四项独立泄漏 review 条件已有依据。本文不重新评判原 review 的其他条目、统计显著性或 anchor；最终组件/anchor 裁决仍归 Kiro。叠加模型自己的矩阵仍须独立取得，不能借用本回执。

本轮只读取现有回执与 Git 源码对象、重算 SHA/JSON 身份，没有服务器连接、测试重跑、真实数据/test/checkpoint读取或训练。未修改实现、原 review、事实源或 commit。context 计数 UNKNOWN。

## 可核验的原始来源

| 证据 | SHA256 / 身份 |
|---|---|
| `cache/pgr01-p-recovery-v1/main05/leak-v2-confirm-20261016T194804Z.stdout` | `e1f427e5390e7e32a261088a4482449da9469b43780f27930ea013b4af375cac`；本轮直接重算一致 |
| 内嵌 `stdout.log` | `b9eb5c80b8aea1d21f30e971357f5eeba033163e5730cb129278abd3044d4aa4`；按已核工具的 sort_keys JSON + newline 重建原 stdout，SHA 与收据相等 |
| 内嵌 `terminal.json` | `a5cd29ec8ea77e67bf40bf00e553bdac99c02415b0a4864f62ffb56cfd1bb0e9`；原始 status=succeeded、exit_code=0，wall=56.37766075134277s |
| v2 job | `p-combo-leakage-real-v2`；job SHA `69f5ed4e2ed4d079ab9d09c128e0f001cba5d72d990e52332be0809aefc85c7e` 与现有 dry/submit 收据一致；confirm 为唯一 claimed，pending/rejected 均0 |
| 原 review（未覆盖） | `docs/superpowers/reviews/2026-10-17-feature-combination-leakage-review.md` SHA `fff66fdb38fd407d1bb662207067d6e903ff5d80e65299dd9dfd4e3f7027d9a7` |

回执内实际检查完成时间 UTC `2026-10-16T19:47:49.961958+00:00`，confirm 读取时间 UTC `2026-10-16T19:48:07.617594+00:00`。结论来自原始逐项 status、counts 与身份绑定，不仅来自 exit 0 或“succeeded”。

## 对象、输入与真实路径绑定

- receipt model commit 为 `84b160cb0eeb3c53b877f6cf6474276ba06a6ba5`。53 个 `model_source_files` pins 已逐一与该 commit 的 Git blob 重算，**53/53 相等**；ledger 按 sorted compact JSON 重算 SHA `2b07ec0f0575d9f4033c030d45758db185f3def88cff4030141a0a3e161013d3`。完整相对路径与逐文件 SHA 保留在原 stdout，不在本文复制。
- 工具 execution commit 为 `3dd6215bdea48e46a545260fc39b4bac5e164f11`，`analysis/feature_leakage_real.py` Git blob SHA `dae71adf14d0307be1d71cfc75474b82ca29885f209534e3d2dab90e2b9a0c92`，与 receipt.tool_source_sha256 相等。
- loader 明确为 `retrieval_feat04combo_run.validate_inputs4b/b1.real_inputs/apply_feat04b(engineering=False)`；A/C/D/E 各行 engineering=false；取512条 train + 全部2520条 validation。每个模型固定初始化，不读取 checkpoint；A/E recency 非零，E β 非零。C/D 对应无该参数，false 标志合理，不当作漏跑。
- inputs SHA `08ed500966e506a2599beb2bb2807330165b82f13ec2b8df01286bdd769493d6`，与原组合 `codex-combo-preflight-20261016T105600.stdout` 的 inputs_sha256 直接比对相等；prepared SHA `7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2`；history manifest SHA `8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43`；side manifest SHA `3f4ee94a0cf196d7671743dabc8c8c6b81c0dfbdf64ffbcd3054212cd3d292ee`；stats manifest SHA `eb41bc060c67549a0cb7bf96c627000bb90b73cb5111ca11b21bf572573c036b`；test_gate=closed_hash_only。
- 真实数据摘要：rows `585bfb31802dca1c98fb71e71369e7d8720253f55083c8e5c30785dd8acc3370`，queries `e540e19152fd8a79ea88df537579554313348a5a5cac595143773847ab775f38`，catalog `abe2e7a3291f45eb7d535cac574c3ffae8192c79ff726fc25cd6349543152cf5`。这些是 receipt 的数据 digest，本轮没有重读数据重建它们；source/loader pins 与原始回执是本次核对依据。
- 原始 `test_read=false / checkpoint_read=false / training=false`；所核工具路径只实例化初始模型、装配/导出和做变换对照，无 optimizer、训练或 checkpoint loader 调用。

## P1 矩阵：逐项 status 与计数

A=f4_recency_pool，C=f4_item_popularity，D=f4_item_quality，E=f4_recency_behavior。下面是原始计数，不把静态项伪装成 runtime，也不把批次数写成样本数。

| 设计项 | A | C | D | E | 核对解释 |
|---|---:|---:|---:|---:|---|
| L1 target字段隔离 | pass /3032 | pass /3032 | pass /3032 | pass /3032 | 512 train+2520 validation；张量及user逐位不变 |
| L2 导出隔离 | pass /2520 | pass /2520 | pass /2520 | pass /2520 | validation全量，翻target字段后user/item导出及norm相等 |
| L3 历史反向对照 | pass /3032 | pass /3032 | pass /3032 | pass /3032 | 历史item变化后user向量变化；E另有自身反馈bit反向128条pass |
| L4 周平移 | pass /2791 | pass /2791 | pass /2791 | pass /2791 | 仅可在0–24周内合法整体平移的2791条；不是3032条全都可平移 |
| L5 item ID | pass /3032 | pass /3032 | pass /3032 | pass /3032 | history id映射逐行核，工具另对catalog candidates顺序反转做逐位检查 |
| L8 week25拒绝 | pass /24批 | pass /24批 | pass /24批 | pass /24批 | 每个128行块取一个week25负例；24个负例，**不写成3032个负例** |
| L9 train候选/导出同源 | 不适用 /0 | pass /120415 | pass /120415 | 不适用 /0 | C/D全catalog raw/normalized/norm核对，训练columns对相同item逐位比对 |
| L6 / L7 / L10 | 静态 | 静态 | 静态 | 静态 | STATIC_REVIEWED，引用原固定commit review，不是新增runtime pass |

编号错位按回执映射明确：design L3→旧guard_l4_reverse_controls；design L4→旧guard_l5_week_shift_invariance；design L8→旧test_l6_week_25_rejected。工具实际做的是对应语义检查，不因旧测试函数编号而遗漏。上述矩阵与 Kiro「2026-10-17：组合泄漏 review 的补证方式」的运行时/静态适用范围一致。

## 一次工程重验的边界

Kiro 已单独批准原组合一次新工程 attempt：只修输出父目录及失败时 stdout 保留，科学代码、输入、seed、变体和检查项不变。已核 `ba96b1e..3dd6215` 对工具的 diff：只新增 `emit_receipt`（先print并flush，再mkdir父目录、独占写文件）并替换原末尾输出代码。

两 commit 的 `audit_runtime` AST source segment 原文逐字相等，SHA 均为 `2a2f1db4cbe3f53b10538a292f419bc1ec6ca87f974db624750bc90afeafd6ef`；不含函数后空行。固定seed与非零初值、检查算法没有变化，本轮没有触发第二次重验。

旧失败原收据 `cache/pgr01-p-recovery-v1/main05/leak-job-result-20261016T192949Z.stdout` SHA `896f3ffb5c741692987e341370cf17c992e39c2663a2bc5501d6d7bfe3f048d9`：run `p-combo-leakage-real-v1`，terminal failed/exit1，stderr含FileNotFoundError；stdout payload实为长度0，SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。因此没有旧已打印检查结果需要逐项比对；没有把“未打印结果”当成旧PASS。旧失败term/log收据与原review继续保留，本文不覆盖它们；未登录服务器另验旧目录现状。

## 继续保留的结论限制

本次 APPROVED 只关闭原 P1 的运行证据缺口，不代表训练侧没有self-inclusion。D-1与23:2x更正继续有效：C/D表含训练target自身及之后train行；validation隔离不推出训练侧无偏，方向未知；旧Σlikes≈样本数推断不成立。C9仍是报告项，p_like≤.30不证明无偏，也不是新增门槛。不能由本次工程PASS声称严格时间隔离、分离C/D贡献、交互或因果；不作序列叠加的新模型验收、anchor或转正裁决。
