密级：公开

# 自生成难负样本偏好训练（R2）：冻结来源与工程边界

2026-10-17，Codex。科学规则 FROZEN；工程 review / 真实准入尚未通过。仅新增来源证据，不覆盖 Kiro 维护的原卡或 decision_log。

## 冻结来源

- 主仓 `docs/decision_log.md`「排序导向 R2（SFT + 自生成难负样本 DPO）两项补定」与同步实验卡 §2：beam10 无合法负例时跳过、不补抽；固定20,000样本集合；报告skip条数和比例。DPO min_epochs=1/max5/pat2；best只取DPO epoch1–5，SFT best epoch0不参与选择。epoch0与原A1同seed作复现性比较，差异只报告，不作gate。
- 主 owner 成本 ACK：已部署工程源码 `974c6751f49a085680f4cbe75862bbf86b7d9f4e`，真实 train-only PF 与 `g-rank-r2-mining-speed-v1` 直接 succeeded；128前缀、随机新A1权重、beam10/b1/单核，整批加固定开销的工程估算低于4h。收据SHA `089148f65664ff0f7518ff517221ccb07ed0c3a9ad3b7898ec1221be347aee96`。不复制含GPU UUID的原始record进入public Git；具体数值只由主owner写experiment_log。
- 固定20,000 seed0 sample index digest：`bc2a343ddac20bc0a9a7edc664de321bc5b5bb463408d7825e8b43c1bdfb61ca`。该工程估算不是新SFT best的正式挖掘吞吐；正式阶段仍记录实际耗时和skip，统一4h cap，不改beam/样本数。

## 单个新 run 内的三阶段

1. `sft/`：从头按原A1/T3训练，原CE/raw logits、结构/LR/dropout/history/SID/beam100/b1/selection不变；自己产生best+last，自己直接verify。使用原A1配置，只有内部schema/task角色不同。没有R1温度，也不读取既有A1 checkpoint。
2. `mining/records.json`：只用该run已verified的SFT best，对训练样本固定20kseed0索引一次beam10/b1。排除target与该样本完整`past_likes`，不把observed history当like集合。保存输入row hash、beam10、最高合法negative或null，以及skip摘要；不补抽。全部跳过则STOP，不能造负样本。
3. `dpo/`：actor从该run SFT best初始化；reference独立复制/eval/frozen、不进optimizer。DPO beta=.1 + 正样本原next-SID CE(mean)权重1；LR3e-5、AdamW weight decay/clip沿原、effective batch512、microbatch256 pairs。每个pair的chosen/rejected序列logprob按全部SID位置求和。min1/max5/pat2，R@50→N@50→更早epoch选best，候选不含epoch0。

三seed42/2024/2025无条件，分别一个新run+verify；不额外增加名额或seed。对照指标仅从固定SHA的原同seed A1 manifest metadata读取；不打开其CP。完整敏感产物只存新run，不进Git。

## 代码/数据/验证

新代码 root 与 accepted G data root 分离，`VKGR_DATA_ROOT`只传给原G3 loader；不patch旧模块或全局ROOT。固定depth3直接使用原四位置evaluator，无decoder全局替换。新绑定包含新Git SHA、所有阶段源码与三配置SHA、原始数据/context/SID、control metadata SHA、自己SFT manifest/best、mining和reference digest。

总verify先检查同seed SFT metadata，再对自己SFT进行原selection/hash/best重解码验证；核mining sample/row/beam合法性/最高可用negative/skip/source/hash；重放DPO epoch1–5 selection，核best+last与reference绑定，并重新解码DPO best。epoch0读回自己SFT best的已发布evaluation并核对差异报告。**Mining verify核已绑定的beam记录，不额外完整重跑20k解码**；该覆盖边界明确写入verify输出，交本次独立review评估。

判定、机制读法和anchor裁决仍指向原实验卡 §3；代码不自动换anchor，不按seed42筛后续seed。

## 当前验证限制

60相关synthetic tests通过，包括完整两阶段训练/新CP/verify、手算DPO、reference不更新、skip不补抽、epoch0非候选、跨seed提前拒绝、mining篡改、control只读metadata、ready/release真实本地进程fault，以及context-inclusive显存proof规则。typed synthetic context注入不算真实loader PF；尚未执行新完整runner真实PF、CUDA/ref/optimizer全shape显存或scheduler dry-run。旧frozen fixture文档pin漂移没有改。
