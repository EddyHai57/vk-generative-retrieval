# 交Kiro/Opus：新SID A1卡冻结与P线补证裁决

密级：公开。Codex只执行/记账，不写decision_log科学决定。请在Opus下按事实源复核；不要把此文件视为新训练授权。

## 先看

1. decision_log最后“2026-10-17 午间/午后”。
2. docs/superpowers/specs/2026-10-17-coll-sid-a1-confirmation-draft.md：已按你新增无人值守§1预确认转FINAL，工程执行中，不需要本时段回复。
3. experiment_log“2026-10-17 13:25+08：叠加与双向确认配对补证”及其后14:01用户塔计时最终验收补充，已齐证。
4. docs/superpowers/reviews/2026-10-17-stack-e-pooling-leakage-review.md（APPROVED，仅E pooling替换m接合点）。
5. issue I-2026-10-17-g-earlystop-censoring：唯一一次已存曲线诊断，右删失限制保留。

## 请确认/裁决

- **新SID的A1确认**：核草稿是否忠实于午间/午后原规则；同一已验收共现向量SID bundle不重拟合，A1配置除SID身份/schema外逐字段保持，三个seed无条件全跑，对已有A1+T3，原16:5x三条件与逐seed best/stop，机制描述沿原热度分档，anchor不自动切换。无需增加配置/评审轮数。你已在无人值守§1预确认，卡现FINAL，Codex已进入AGENTS§4工程闸门；本时段入队前写STATE时间、之后独立claimed/无rejected。工程未齐前不发布科学训练。
- **序列用户塔+组合叠加**：配对统计齐，原行2满足；但“至少两个seed exact McNemar显著”的独立转正条件不满足。接合点review已通过，延迟独立验收补充现已齐，请回来后按原门槛裁决，不通过bootstrap替代McNemar，不追加seed或bi组合。D-1/23:2x和主要增益来自组合的限制保留。
- **双向确认**：配对统计齐，原行3（均值过线、合并检验未过），转正显著seed条件同样不满足。用户塔延迟现已验收；只作M1结构结论，不与叠加合并。
- **延迟用词**：Eddy已明确选择“沿原用户塔口径补报”。请将午后§5“端到端”同步为model.users口径，遵守10-16下午§1不能称端到端的边界；无checkpoint、真实长度/合成取值/初始化权重，CPU/GPU在线/离线，直接同次mean作比。不要扩大到检索/网络或旧checkpoint。
- **早停风险**：只在已观察曲线内未见注册问题；patience会截断未来，不能推断停后不恢复，不改现有训练预算。是否将其保留为报告风险由你解读，不新立耐心扫描。

温度交叉熵/DPO已按午后收口，无追加配置。可学习SID等新SID A1结果后再定；融合仍等P anchor裁决。所有实际数值/收据/SHA只在experiment_log维护，此处不复制。

两协同SID新A–D报告已收口，C分档逐seed对T3核对齐，算术描述只见experiment_log“14:10收口”；请回来后决定机制措辞，Codex不写学到协同/跨架构胜anchor。两篇指定S3论文转述已在research/2026-10-17-sid-depth-accuracy-notes.md，未据此起卡。
