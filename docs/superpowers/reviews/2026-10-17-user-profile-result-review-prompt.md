# 用户画像特征（F7）结果复查提示词

请在 Opus 下复查用户画像特征（F7）本批结果；只做解读与裁决，不启动训练、不补seed、不改anchor。

先按顺序读项目AGENTS、STATE当前快照与F7待复查项、decision_log“2026-10-17 晚间：用户画像特征逐项裁决”第1–6项/冻结与名额补记、FINAL卡 docs/superpowers/specs/2026-10-17-user-profile-draft.md，再读 experiment_log“用户画像特征：六原始verified与validation配对最终验收”及后续用户塔延迟节。下面数字仅引用experiment_log，完整数值与收据以该唯一事实源为准。

身份与边界：U2累计曝光/like严格前缀，U3 H50 like率；u=m+g*r+p、零表在共享随机参数后创建，两臂各42/2024/2025无条件全跑，复用已验收M1-mean同seed。六train+六原始verified全部通过，verify/train run_id+manifestSHA/冻结plan/配置/数据绑定齐；九best validation预测原归档重算R/N全部一致。原driver2984239最终verified汇总后退出，不重启、不重投；本次统计/计时未读test或任何checkpoint，科学verify按既有授权只验证自身新产物。

结果（seed顺序42/2024/2025）：
- M1-mean hits：466/481/471；best/stop：12/15、7/10、11/14。
- U2 hits：467/481/472；Δhits：1/0/1；mean Δ=0.6666666666666666，k+=0，p_pool=0.9550086046795542；逐seed exact p：1.0/1.0/1.0；best/stop：10/13、7/10、12/15。
- U3 hits：466/481/478；Δhits：0/0/7；mean Δ=2.3333333333333335，k+=1，p_pool=0.7431068317811952；逐seed exact p：1.0/1.0/0.5657001139736338；best/stop：9/12、9/12、14/17。

冻结判定：两臂mean Δ都在(-6,6)，机械落第4行“未检出与M1-mean的差异”。第2行条件不满足；至少2个seed exact p<0.05也不满足。不要称等价、不要写画像没有价值，不按U3单seed+7挑seed。逐seed b/c、R/N与10000次query paired bootstrap95%CI（seed0）完整表在experiment_log；p_pool重复同一query跨seed，不独立且偏乐观，不能代替逐seed条件。

H50分档（未满n940/已满n1580，均非新门槛）：U2 Δ未满+4/+2/+8、已满-3/-2/-7；U3 Δ未满-4/-4/+7、已满+4/+4/0。已满含恰好50和>50截断者，只能描述，不能单独证明严格>50机制。无训练事件validation用户0、like率无定义0；0比例与无定义处理仍分开。

工程与来源：科学代码b2039497d78a7185739d12a66afe232dd8056c71，旧M1原入口未改；科学泄漏review对象2734453 APPROVED，之后仅工程修复。只读验收工具727e167，独立review原结论CHANGES_REQUESTED唯一导入来源MEDIUM已fresh RED→GREEN/7新测试+13旧统计计时测试+3mutants闭环，不伪称第二次review APPROVED。原review及闭环 docs/superpowers/reviews/2026-10-17-f7-readout-review.md。旧G6 fixture与历史verify stdout身份问题仍OPEN，本批通过不自动关闭旧风险。

用户塔延迟原报告已验收（ms/query，顺序mean/U2/U3）：
- CPU offline均值：0.4886178/0.4926922/0.4932131；online均值：1.778244/1.868227/1.881639。
- CUDA offline均值：0.01325953/0.01387026/0.01362316；online均值：6.23192/6.085057/6.199319。
每模型/模式50预热+1000正式、offline512/online1，CPU单线程；相同真实长度分布、配对输入SHA相等、均值与p50/p95/p99已从raw时间重算，原物理reportSHA与stdout一致。仅初始化权重/合成特征，profile IDs timer外打包；mean适配器剥3列开销在users内；不含特征取数/物品塔/检索/网络，不称端到端/训练权重/线上延迟。CUDA单query略低只描述本次测量，不能声称新增特征有加速因果效果。

完整四条件：第2行不满足；至少2个seed exact p<0.05不满足；独立科学泄漏review APPROVED满足；限定口径用户塔延迟报告已附满足。因此统计转正条件未齐，最终是否收口由Kiro裁决，Codex不自动写“转正/不转正”决定或换anchor。

请裁决：①接受本批完整性与机械第4行判定，确定最终收口措辞；②逐条复核四个转正条件（第2行、至少2逐seed显著、独立泄漏APPROVED、用户塔延迟报告），不要以总体方向/合并p/分档代替；③复查分档机制与计时范围的报告措辞。P anchor维持M1-mean，G anchor维持A1+共现向量SID。Codex不自行写关键decision_log，结论请由Kiro落盘；不自动启动G5/可学习SID或其它新实验，新增名额/规则变更仍按既有边界申请。


主要复查入口：
- 数字唯一主账：docs/experiment_log.md 最新“用户画像特征”配对最终验收与用户塔延迟最终验收两节（含完整R/N、best/stop、b/c、CI、H50、源收据SHA/实际UTC）。
- 私有可核报告：cache/codex-morning-execution-20261017/f7-paired-final-accepted.json、f7-users-final-accepted.json（仅聚合/原始计时与hash，不含逐query预测正文）。
- 原科学最终收据：main05/f7-final-owner-20261017T140730Z.stdout；原配对：main05/f7-paired-report-status-20261017T140710Z.stdout；原计时：main05/f7-users-report-accept-20261017T142125Z.stdout。main05的完整项目映射见本机gitignored路径表，勿把本机绝对路径发公开Git。
- 科学泄漏review：docs/superpowers/reviews/2026-10-17-f7-independent-leakage-review.md；分析工具review闭环：docs/superpowers/reviews/2026-10-17-f7-readout-review.md。
