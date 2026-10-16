密级: 公开

# 序列用户塔 + C/D/E 组合叠加

设计冻结：2026-10-17，依据 decision_log 当日上午§1.4及续§1直接授权，无需另轮设计review。工程实现、准入尚未通过；组合独立泄漏review当前CHANGES_REQUESTED，未APPROVED禁止submit。1个配置名额，seed42/2024/2025全部跑完；对照已有同seed M1mean，不重跑。context计数UNKNOWN。

模型沿M1mean `u=m+g*r`：因果1层pre-LN Transformer、mean readout，branch只读F3/F4事件；m换成E的Q2位置/历史内周差+β行为权重；候选加C/D四统计表，supported规则不变。先完整构建M1mean共享anchor/branch，再追加13个Q2 logits与6个β零参，最后C→D四表抽取；g=0，user初态等于M1mean。branch不读反馈或stats。side沿旧6列（mask/timespent/duration/week/零/零），不改变冻结side契约；target/position/past_likes/oov不进user。

原M1两文件从04d4ede逐字新增，旧branch不改；base c4e4eb8与部署84b160c同树。配置除schema/task外逐字段沿M1mean：armmean、feature f3_f4_recency、L50/d64、temperature.1/lr.001、logQα1/inbatch、无extra uniform、batch512、100/5/3、AdamW、best+last、attempt1、无HPO。只新增文件，不全局monkeypatch。stats沿冻结train0–24周，绑定stats/history/side/catalog/support/metadata/source及bucketids，verify重算；禁止旧checkpoint与test读取，允许本作业新checkpoint自验。

判定沿seq-m1-design§8.1对象替换：ΔH_s=round(Rstack×2520)−round(RM1mean×2520)，D=三seed均值，k₊=ΔH≥6数；自上而下：①D≤−6为劣于；②D≥6且k₊≥2且p_pool<.05为有效；③D≥6但不满足②为不稳；④−6<D<6为未检出差异，不称等价或序列无价值。转正另需≥2seed逐seedexact McNemar p<.05、一次独立泄漏review、延迟报告，由Kiro裁决；否则anchor保持M1mean，不自动切换。

本地freshRED→GREEN→回归、独立code review覆盖泄漏后，push扫描/固定SHA部署/逐文件SHA、3config真实engineering=False preflight、synthetic CUDA train+本作业best重导出verify、fullshape batch512/L50/catalog120415/query2520显存、dryrun。入队前组合review必须APPROVED；三train各唯一verify，detached/断SSH不停止，finite48h driver、轮询≥900s，不重试新增attempt。资源gate只管自己，余量≥20%，不恢复automation或新timer。当前仅本地实现，不服务器/不push/不submit。

延迟沿§3.11：只测users，离线batch512每query；在线batch1 CPU单线程/A6000，按validation长度分布合成取值，warmup50/1000次，mean/p50/p95/p99及M1mean比值。不设阈值，未测UNKNOWN；不为本地验证读取真实数据。完整交付一次独立code review，不自评。

D-1保留：C/D train表含自身target及之后train行，validation隔离不证明训练侧无偏，偏差方向未知，不称严格时间隔离有效。23:2x更正保留：Σlikes≈样本数旧推断不成立，C9 p_like≤.30只表示报告提醒未触发，不是无偏证明或转正门槛；loo1未触发。组合不能分离C/D贡献、交互或因果。指标唯一主账experiment_log；决策由Kiro写decision_log。


工程补证要求（2026-10-17 Kiro 最新裁决，落实原验收，不改模型/预算/判定）：叠加自己的engineering=False正式loader取512 train+全部2520 validation，非零Q2/β且g开放，逐项L1/L2/L3/L4/L5/L8/L9；L6/L7/L10如实记录STATIC_REVIEWED与review/source hash及编号映射，不虚构runtime。失败立即STOP；三config receipt各绑定source/config/inputs/side/stats/data/bucket摘要，dry/submit必须核对且单独要求原组合APPROVED，旧组合receipt不能代替叠加。

submit协议：controller先seal完整plan及原48h deadline，再启动持锁detached owner，确认实际ready后release；只有该owner发布训练和唯一verify。部分发布失败写ALERT，已发布训练继续接续唯一verify，未发布intent不重试，不再需要另发start。ready前失败零发布；ready后未release超时零发布。

CUDA显存按本PID nvidia-smi样本（含context）加torch峰值；取own-PID峰值与allocator max_reserved+样本最大非allocator占用的较大值，保留baseline/train/export/top50阶段样本，准入重算measured与max(1024,ceil(measured×1.15/256)×256)。不使用混跑global free delta。fullshape必须实际导出120415物品（另加PAD）、2520用户；CPU仍VRAM UNKNOWN。
