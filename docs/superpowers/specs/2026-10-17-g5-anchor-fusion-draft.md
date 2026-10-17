# 双塔与生成式融合（G5）冻结卡草稿

密级：公开。状态：DRAFT，交Kiro冻结；未实施、未入队、未读取任何融合结果。授权：decision_log「2026-10-17 晚间：新SID的A1确认」§5.1。指标事实源仅experiment_log。

## 1. 是离线合并，还是需要新训练？

**本卡不需要新训练：两边模型已经训练并验收。** 计算本身是固定候选列表的离线合并，不更新任何模型参数。

但**“不需要训练”不等于“现有top50存档足以执行原200候选协议”**：P原存档是`predictions-epoch-NNN.json` top50，G原存档是`evaluation-NNN`，完整合法候选最多100。旧rev3要求两路各100/三路各66，P top50不足，不能补齐、复制候选、改配额或把两路top50的oracle并集冒充正式融合结果。现有top100 P导出是否已存档且已验收尚UNKNOWN；冻结前必须只读盘点。

首选保留已批准的预算200与rev3协议：若P已有同best top100归档，完全复用；若没有，需要**额外离线推理导出，而非新训练**。该导出需读取各P anchor best checkpoint；旧rev3还要求G前64个query重新解码对拍，也涉及三个G best checkpoint。当前授权仅允许本轮写草稿及另一项前缀归档补证，**不包含这些既有checkpoint的读取**，必须由Eddy逐run/路径批准后才能执行。不能借用C2单checkpoint许可或“新作业验自己新产物”的许可。

请Kiro确认保留rev3这套无新训练、但需额外导出/对拍的路径；或明确裁决改成纯存档top50的新版本设计。后者改变已批准的候选深度/配额，必须明确裁决，本文没有将其自动采纳或放宽门禁。本卡在输入缺口和授权未闭合前不冻结、不入队。若另提需要新训练的融合器，另卡、另请Eddy批名额，不在本卡。

## 2. 冻结模型与问题

P固定M1-mean：p-seqm1-mean-s42/-s2024/-s2025，已验收best，不用叠加、bi或第二次挑选。G固定A1+共现向量SID：g-coll-a1-c1-s42/-s2024/-s2025，共享COLL_C1_B256D3_S42 bundle。A1+T3只作历史对照，不进入替代G路选型。对应选择/manifest/config/source的原SHA以experiment_log已验收条目为准，执行前写入不可变anchor receipt，任何UNKNOWN必填项拒绝冻结。

主问题：在相同候选配额下，G作为P第二路是否优于Pop或另一P seed？“在P+Pop上再加G”另报描述性比较，不替换主问题。旧oracle重叠预检用的是旧anchor，不能迁移其数值到本卡；新anchor的并集只作上限，不能当融合效果。

Pop固定同源train-only item_like_counts，排序(-count,item_id)，不按validation拟合；不读原始test。三方使用同一catalog120415、同一2520 validation query、HIST_V2_OBS L50；623 OOV留在主分母并单列结构性0。

## 3. 路线与确定性算法（沿既有rev3，等待Kiro冻结）

主表seed42：C1=P top50；C2=G top50；C3=Pop top50；C4=P+Pop；C5=P+G；C6=P+G+Pop；C7=P(seed42)+P(seed2024)。2024/2025各复现C1/C4/C5，仅稳定性描述，不挑seed或顶替主表；是否扩展主判至三seed属于另行设计，本文不做。

总预算200，n路配额q=200//n；两路100、三路66，剩2不分配。G短列表不补，不把空位转给P；P/Pop不足所需配额则输入闸门失败，不按G短路规则绕过。单路直接top50。跨路等权RRF，k=60、rank从1起，精确Fraction(1,60+rank)求和；跨路按item_id去重、同路重复致命；按(-score,item_id)稳定排序，final50。不做权重/k/预算搜索、ANN或候选补全训练。

## 4. 验收、配对检验与解释

保留decision_log 10-14 22:4x的整数门槛与Bonferroni口径，旧设计稿rev3作技术来源：
- H1：C5−C1≥6 hits且exact双侧McNemar p<0.025。
- H2：C5−C4≥6且p<0.025，并且C5−C7≥6且p<0.025，intersection-union；两族各α=.025，族内不再校正。
- D1=C6−C4只作描述，不占α，不替换H1/H2。
- 上述四比较均报b/c、整数Δ、N@50差与bootstrap95%区间。沿rev3 random.Random(20261014)、10000次query重采样、最近秩分位，四比较共用下标；不能改用其它已用过的bootstrap实现当成相同协议。

主表全2520及in_catalog/OOV分开报；同train-only热度、support、用户历史、历史长度与G短列表切片。每多路行相对C1/C4都报新增/丢失/净增、路级独立命中、交并/重叠、pool大小、每路实际填充、coverage。低支持定位沿rev3：非OOV中unsupported或低热度或G短列表的query集合只算一次；净增≥6且占全量净增≥50%，全量净增≤0则false；这是继承待冻结字段，不因新anchor结果调阈值。

只过H1不足以把收益归给G；H1/H2与G独立命中/定位按rev3分开报告。OOV无收益不能略去。沿新G anchor的D-1同类偏离、约0.2%内容回退及早停右删失；不称线上A/B或因果收益。

## 5. 输入合同、工程闸门与成本

必须逐query核ordinal、user、target、严格历史item/side字段、target时间/OOV标记、query/history/catalog hash；不只比目标序列。源预测经原verify且完整SHA链到best manifest。P top100前50须与原归档逐位相等；G候选用原存档、前64query对拍须与原items/audit相等；检查所需checkpoint授权单列，不默认为已批准。实际新入口需新verify stdout含run_id和manifest SHA。

P/G仍用各自原解释器、固定source与binding，输出写新目录。离线推理/export/fuse都是只读输入且新写输出的非训练作业，各走AGENTS§4：fresh RED/GREEN与变异、一次独立review、扫描push固定SHA部署、真实只读preflight、涉及CUDA则预演/实测×1.15 ceil256、dry、claimed/no rejected、各自verify。不能从用户塔延迟外推检索或端到端；P导出/检索与G解码、载入、融合wall分开，存档amortized b1延迟不与串行逐query混称同口径。

冻结版本只认首个实际产出融合结果的primary；后续anchor变更为复测，旧结果保留，不删失败、不重跑挑最好。

## 6. 待Kiro冻结的明确事项

1. 第1节的P top100缺口与G对拍/旧checkpoint授权：保持rev3或另行纯归档版本，不能带UNKNOWN执行。
2. 核定新anchor身份表、主表seed42+两个稳定性复现、H1/H2/D1与定位规则逐项承接，尤其rev3末尾23:0x只在旧草稿的裁决应由Kiro确认正式效力。
3. 准入资源和耗时均待工程实测，现不填训练式ETA；此卡零新训练名额，额外旧checkpoint读取需Eddy批准。
4. 本稿只交付设计；Kiro冻结及上述授权闭合之前不实现、不入队、不读融合结果。用户画像特征不进入本卡。
