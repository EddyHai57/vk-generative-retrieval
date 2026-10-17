# 用户画像收口后一次只读诊断：交Kiro/Opus复查

请读decision_log深夜F7裁决§4、issue_log I-2026-10-17-f7-postclose-diagnostic，以及experiment_log本次“用户画像收口后一次只读排序与档位诊断”。数字以experiment_log为准；原报告/127输入SHA在cache/codex-morning-execution-20261017/f7-postclose-diagnostic-accepted.json。

一次只读作业已验收，原15个已verified run的best validation归档、validation侧车及原纯transform；未读任何checkpoint/test，未拟合/解码/导出。F7原两臂第4行、不转正、收口，P anchor保持M1-mean。

结果摘要（引用主账）：F7每query平均top50重合36.54–39.65/50，bi/mean参照33.45–37.05/50；所有比较均没有集合完全相同的query。三seed合并仅描述，U2平均37.8758、U3 38.2016、bi参照35.0192；参照seed不同，不当跨实验检验。

Discordant可见命中侧45–50占比：U2 102/314=32.48%，U3 116/335=34.63%，bi参照84/413=20.34%。其余命中侧多数名次更靠前；另一侧只知>50，51–55精确计数UNKNOWN，不把未命中填51。因此不能对两侧完整45–55分布作结论。

validation2520query=2520distinct user。N_exp实际占10档（1–10）、N_like占8档（1–8）、H50 like率占9档（0–8），均非常量；桶0的0比例和桶9无定义保持分开，完整人数表在主账。身份/顺序与运行sidecar/fit/producer SHA对拍一致。

请复查：
1. 是否保留“未检出差异”的暂定措辞，并仅添加量化排序变化描述；不要仅据集合更重合就断言(a)或“画像通道没学到东西”。预测变化包含不同训练轨迹，档位非常量不是p贡献非零的动态证据。
2. 这些归档证据尚不能唯一分辨(a)/(b)/(c)，是否还需另行设计有界诊断。若确需读F7 checkpoint或测norm，先列精确范围请Eddy另批，不自行读、不重跑。
3. 原异常概率量级的联合独立假设未在本诊断验证，比较共享query/对照，不将其当已预注册正式p值。

没有发现已证实的归档错位/常量侧车问题，不自行重评本批有效性；是否需要更正科学措辞请Kiro写decision_log。F7不追加后继/seed/字段，不改变anchor。

G5与S2仍按晚间S5§5原授权：G5冻结卡草稿已交，候选深度/导出与对拍权限待裁；S2立卡草稿已交，方法/完整损失与名额待设计评审/批准，不将本次诊断扩权为冻结/实现/入队。入口docs/superpowers/specs/2026-10-17-g5-anchor-fusion-draft.md、2026-10-17-learnable-coll-sid-draft.md。
