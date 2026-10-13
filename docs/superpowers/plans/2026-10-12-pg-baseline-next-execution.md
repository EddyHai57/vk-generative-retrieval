# PG-R01 后续基线执行与有限预算提案

密级：公开。日期：2026-10-12。状态：PROPOSED_NOT_GRANTED。本文件不产生submit permit。

输入：[接收报告](../../research/pg-r01-baseline-reception.md)、原P4H过夜/科学合同、Main ignored cache/pg-r01-reception-v1/budget-proposal.json。长期研究路线不变，本轮只恢复基线，不加入新模型/特征/HPO。

## 顺序

1. **本地最小修复与TDD。** P owner只改retrieval_sasrec_remote.py及对应tests：先实现versioned extension-authority adapter（原queue/两失败收据/旧spent＋新的human grant绑定，累计attempt3，新增4h单调clock但旧queue不改）；注册每arm2520条matched train-only长历史capacity cohort；producer和verifier绑定同一选择策略与SHA；补实际stage成本到formal admission的拒绝路径。保留FD helper与科学model/loss/batch512/12slot registry。fresh RED覆盖无grant追加/改旧deadline/重置attempt/重复extension/伪grant来源/旧spent丢失必须拒绝，再覆盖cohort错数量/重复/错arm/holdout lineage/源漂移、虚构或缺成本、budget/closure reserve不足，然后最小GREEN及相关回归。不重跑接受过的v8/旧full suite；新完整delta仅一次独立review。该实现设计需Eddy确认后细化，不在接收任务顺手实现。
2. **新增单次恢复pilot。** 批准后先持久登记累计attempt3/新clock/deadline/disk/spent linkage，再部署最新pin。完整Linux UID/owned-tree/publisher feed闭包通过后reprepare新input origin，启动like/observed各最多30steps的train-only pilot。分别CLOSED，再独立pilot-verify operation CLOSED。任一步失败STOP，不自动再试或退款。
3. **实际成本准入S02。** batch512，86785例→170steps/epoch，最多30epochs→5100steps。成本计算包含30epoch参数训练、每epoch完整2520 validation、fresh-best export/exact、独立verifier、prepare/部署、checkpoint原子峰值及cleanup。pilot计时如果是30step总量，先除实际step数再外推，不错当单step。取较坏arm并拟定1.5安全系数；这是提案，不是已批准门槛。没有完整测量不启动S02。
4. **条件执行S06。** S02结束并独立verify后，以实测剩余总窗口/磁盘核S06完整上限；足够才开。S02/S06只是一对like/observed、LR0.001/T0.15/seed42基线，不运行其他10 slots，不补新features或额外HPO。失败按本次一slot一次提案STOP，无自动retry/resume。
5. **G稳定性确认。** 冻结d128/LR0.0003代表seed42；仅补2024/2025两个seed，不重跑四配置、不挑最好seed。每个完整train→best→原独立replay→consume→cleanup。G4随机SID/beam诊断与论文版TIGER分别另立有限计划，本提案不计入。

真实作业串行安排，避免把并发干扰混进成本估算；保留G优先安全边界。当前建议先P恢复pilot，再基线S02/S06，之后G补seed；若Eddy选择先G，也仍串行，P安全feed保留唯一终态receiver，无重复monitor。

## 有限预算与绝对截止（待人类批准）

| 包络 | 操作范围 | 作业wall上限 | 尝试与磁盘 | 实例时费上限，$0.52/h |
|---|---|---:|---|---:|
| P恢复pilot | 部署/完整publisher与Linux proof/reprepare/两arm≤30steps/独立verify/cleanup | 新增4h | 新增1attempt，总3；原＋新pilot目录合计20GiB | $2.08 |
| P正式，条件项 | 只S02/S06，各最多30epoch，完整validation/fresh-best/独立verify | 合计14h | 每slot1attempt、0resume；formal合计32GiB | $7.28 |
| G补seed，独立项 | winner seed2024/2025，完整训练/验证/replay/收尾 | 每seed30h，合计60h | 每seed1attempt、0resume；每run20GiB，合计新增40GiB | $31.20 |

P pilot deadline=min(first-new-remote-write+4h, **2026-10-13 02:00 Asia/Shanghai**)，latest first write为2026-10-12 22:00；未在此前具备准入条件则该提案到期，不压缩完整验证。P正式deadline=min(formal first-write+14h, **2026-10-13 20:00**)，latest first write为2026-10-13 06:00。G补seed共同stop **2026-10-16 12:00**，每slot另有first-write+30h；不具备两run完整剩余时间则不提交对应run。

旧P pilot原4h窗口已过期；原2次和所有cost保留，追加4h后累计许可上限8h，**不表示已实耗8h**。旧queue不改epoch/attempt字段，不换root假装新项目；新目录与budget grant必须绑定原queue SHA、失败receipts、旧spent和累计attempt3。磁盘20GiB是旧＋新共用总上限，不是额外再加20GiB。旧grant/输入身份不足时先登记显式有限extension authority。

旧formal14h/32GiB尚未freeze，本提案缩小实际执行到S02/S06且仍需完整pilot成本准入。恢复pilot不能自动花满formal预算，也不能因为已花部署费强行开正式训练。closure从各wall中预留至少30min；操作内sample/seal/signal及原子写reserve另按真实成本核，不叠加新预算。

两项P合计上限18h/$9.36；加G两seed全部串行上限78h/$40.56。这是**新增作业预算上限，不是完工ETA或账单保证**。费率为Eddy当前聊天提供的USD0.52/h；原spent、任务外闲置等待、存储、流量和税另计。实例未停止时持续计费，不能把多个旧并发作业wall相加作账单。无新付费资源/生命周期/push授权。

## 估算依据和不确定性

- G winner原14epoch完整wall约10.13h；组件含train stage、best、独立replay及收尾。按相同每epoch成本外推30epoch并加30min收尾为21.70h；再乘1.3约28.21h，因此拟每seed30h硬STOP。原3run并发，未来吞吐可能变化；不同seed早停epoch未知。两seed沿用14epoch时约20.25h/$10.53，仅情景估算，不承诺。
- P旧prepare184.196秒、产物约0.15GiB只是已实测组件；0 scientific steps，无训练/export/exact/verifier实测。旧108286-query capacity每arm约43×formal，不能拿旧4h窗口当吞吐证明。采用新2520/query/arm策略仍需完整实测，不能据传统id_mean性能承诺Transformer时长。
- P stage成本证明必须绑定实际测量index/新source/input origin；若pilot没有覆盖full-catalog exact、完整query规模、fresh-best及独立verifier成本，就保持formal BLOCKED。不改epoch验证频率、不筛去OOV、不削减正式分母。
- approval TTL提议300s；runtime sample5s，资源margin≥20%，真实source/G/资源/absence/unknown-writer身份每次准入fresh核；G早已complete不取消这些门。新grant明确时才能将proposal变为APPROVED，不通过改旧config的BLOCKED字样假装获准。

## stop与汇报

任何hash/data/owner drift、unknown writer、资源不足、cost不全、deadline/attempt/disk不足即STOP，保留产物与spent。每个完整part汇报实际产物、效果/失败、成本与限制；不用模型持续等待，不恢复旧自动化、不创建重复timer。预算获准后先细化步骤1代码设计，再执行一次完整恢复task；本文件本身没有执行新attempt。
