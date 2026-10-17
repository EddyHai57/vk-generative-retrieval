# 新 SID 的 A1 确认（S5）FINAL 冻结卡

密级：公开。2026-10-17。状态：**FINAL；Kiro按decision_log「2026-10-17 午后：无人值守授权」§1预确认，按原内容落卡即冻结。工程闸门未过前不得入队。** context UNKNOWN。

## 1. 唯一授权与问题

依据decision_log「2026-10-17 午间」§5及「2026-10-17 午后」§1：Eddy已批3个A1训练及对应3个verify。问题仅为：共现向量SID在core03下的增益能否在A1架构复现？不增加配置、seed或调参。可学习SID暂缓；温度交叉熵与DPO已收口，不并入本模型。P/G anchor在新裁决前保持M1-mean/A1+T3。

## 2. 固定SID输入：只复用，不重拟合

复用协同向量SID（C1′）已验收的同一bundle：producer sid-coll-c1-input-v1，fit sid-coll-c1-sid_fit-v1，fit-verify sid-coll-c1-sid_verify-v1；sid_version COLL_C1_B256D3_S42，BHKM256×3+规范suffix。SID manifest原字节SHA **c53b81ddfcf8d45fd55903b30f8538372110c8e1df7376156719509994625187**，输入manifest SHA **725fb211a342c518770623be81d490feaaec38a285c954b5b6f51f50a388e9a4**。拟合seed42固定供三个G seed共享。

训练和preflight须核原fit/fit-verify raw verified、全部bundle文件SHA、catalog/id-order/CONTENT_V1/input/vectors/codec与H5a/H5b绑定，不能只比版本名。原向量文件SHA bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7。具体服务器路径仅私有部署参数；完整SID map/向量不进Git。不读取任何旧checkpoint，不导出向量，不修改原bundle或补fit。

限制沿S1 FINAL卡：全train共现含训练样本自身target与之后训练事件，D-1同类偏离保留；无合法正对项回退同源内容，实际数量/比例从原审计报告，不将约0.2%写成新实测。不声称纯prefix-only、全item学到协同、线上因果或真实新物品效果。OOV保持结构性零命中，不扩大catalog。

## 3. 唯一模型与配置继承

严格对照已验收A1+T3同seed，科学基准为其冻结配置/manifest：d_model128、heads6、head_dim64、encoder4/decoder4、FF1024、dropout0.1、LR3e-4、effective_batch512、microbatch256、gradient_clip1、AdamW weight_decay1e-4、fp32、small init、history HIST_V2_OBS L50、beam100、batched解码b1、max/min/patience100/5/3、selection R@50→N@50→epoch序、test_gate closed。实现前逐字段核对原A1+T3三个已验收manifest/config，若上述抄写与原值冲突先回Kiro，不私自改参。

唯一科学改动是SID输入由内容T3换为上述固定C1′ SID；task/schema/run身份与新bundle校验所需登记另立入口。新入口仅新增文件，原A1/T3和S1入口逐位不变。相同prepared/history/catalog/query/split/content路径及hash；只允许适配明确的COLL SID合同，不冒用旧CONTENT SID schema，不减弱旧白名单。保持4列SID、suffix词表与原bundle一致，suffix_vocabulary=1、词表771按原审计核对，不重新压缩/重排code。

三个seed **42/2024/2025无条件全部运行**，不得按s42筛选或挪用名额；不重训A1+T3。预定新run_id：g-coll-a1-c1-s42、-s2024、-s2025，各配同名-verify；工程预演另立身份，不占科学名额且不写成召回结果。入队前核所有queue子目录/runs无同名身份。

## 4. 冻结判定

对照A1+T3同seed hits=289/328/310（身份及数字主账experiment_log）。严格沿2026-10-15 16:5x第1条，全部同时满足才记有效：

1. 三seed train均succeeded，三个对应verify直接原始verified并核job/manifest/config/data/source绑定。
2. 三seed mean Δhits≥6。
3. 至少2seed Δhits≥6。

不满足只如实列条件与未验收项；不挑seed、不因早停时点增加epoch。换G anchor仅提交Kiro裁决，不自动换。逐seed必报run_id、source/config/data/bundle hash、R@50/N@50、hits、Δhits、best/stop epoch与verify状态；三seed均值只在齐证后报。

机制描述沿rank-objective卡§3及S0原G4 train-only热度分档：三seed合计增益>0且头部档占≥50%才描述“主要改善头部排序”，否则描述“增益不集中头部”；没有增益或诊断未齐就不作位置归因。另报前1/2/3位前缀准确率变化，OOV与非OOV分开。原S0 A–D报告与本卡并行，不阻塞本卡工程，但机制结论须等该证据；不重设分档或门槛。

## 5. 冻结后的工程闸门与自动验收

原A1+T3三个已验收manifest的全部本卡架构/训练数值已逐字段核对一致。本卡已按预确认冻结，接下来：隔离worktree新增入口/三个配置/测试与运维工具→fresh RED/GREEN/回归/变异→一次独立完整code review→Git public扫描/push工作分支→新目录固定SHA deploy逐文件SHA→三配置真实engineering=False只读preflight→CUDA训练及自有新产物verify→真实shape峰值显存→dry-run→无人值守时入队前在STATE写带时间记录（其它时段告知Eddy）→三个训练一次性发布并只读核claimed/无rejected（有reason必读，不重投）。不得将action=written称接纳。

新SID入口preflight必须实际核loader/codec/digits/suffix/输入合同、A1配置差异白名单、train/history/catalog/query hashes与test未读；synthetic不能替代真实。训练与verify import唯一新DEP/src。显存按实测×1.15 ceil到256MiB，目前UNKNOWN，不直接复用A1旧6656。按约8.19核、39168MiB准入和20%余量。

服务器侧唯一verify驱动先进Git/push/固定SHA部署，实际PPID1、轮询≥900秒、不调用模型、固定48h截止，发布前ready/release确认；每训练成功只补唯一verify。终态机器JSON包含逐run train/verify、commit/config/data/SID hash、best/stop、R/N/hits/Δ、异常与未验收项。训练/verify失败停该支ALERT，不重跑/不换ID/不覆盖；rejected不重投。NaN/Inf、共同输入失效、设计偏离回Kiro。新作业verify可读自己的新产物；不读其它旧checkpoint或test，不新增timer。

## 6. 预确认来源与当前工程状态

Kiro已在无人值守授权§1预确认午间§5/午后§1的内容，Codex照写本卡即视为确认，不再等回复；没有增加科学内容。当前转入实现，尚无本卡真实训练提交；工程门禁照旧。若接入schema必须改变模型或数据，停止本支回Kiro，继续其它已授权只读项。不存在自动启动可学习SID、C2变体、融合或排序目标后继的授权。
