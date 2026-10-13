密级: 公开

# P3：train原始画像与label候选说明

这次补充画像说明：观看时长能够提供更密的行为信号，但不能直接认定“看够时间＝喜欢”。Like主合同仍然有效；本轮没有选择新的label，也没有新增模型或训练。所有实测数字唯一摘要见[experiment_log：P3补充EDA](experiment_log.md#p3-train-only补充eda--已正式验收)；可检查的机器aggregate见[p3-label-eda-manifest](../artifacts/p3-label-eda-manifest.json)。

## 做了什么

在既有固定subset的train上，先复用P3审计的histogram，再按预声明探索grid补算观看时长与duration联合、六反馈共现、候选支持以及week/context/duration分层。Item metadata按item_id join，user metadata仅核对已下载schema、train覆盖与null；内容embedding沿用已验收覆盖，不重新读取或下载。原始rows与真实ID始终只在服务器。

Interaction实际字段是user_id/item_id、place/platform/agent/timespent和like/dislike/share/bookmark/click_on_author/open_comments。Item metadata含item_id/author_id/duration/train_interactions_rank；user metadata含user_id/age/gender/geo/train_interactions_rank。字段types与null/匹配证据见machine manifest及experiment_log。只有week和physical ordinal，没有精确timestamp、一天内时间或session标记；不能编造session序列或每日时长。

## 为什么要把时长和显式反馈分开看

举一个**synthetic说明，非真实数据样例**：短视频看了一遍，ratio可能高；长视频看了较久却中途离开，seconds可能高而ratio低。相同的观看秒数，对不同duration的含义不同。我们的分层表实际确认了这类分布差异，但不能据此判断具体用户满意还是不满意。

Timespent与duration都是秒编码。上限编码的日志机制没有被固定README解释；ratio超过单位值可能与日志、观看或duration口径有关，成因仍Unknown。没有静默裁剪ratio，也没有把上限编码自动说成真实重复播放。分母为零时ratio不定义，本轮该情况单列核验。

Like OR share OR bookmark拓宽了支持用户和items，观看门槛又显著扩大事件支持；这些规则表达不同意图。Share/bookmark可表达转发或保存，打开作者页/评论区也有各自原义，不能改叫普通click。Dislike与这些反馈的冲突被保留并报告，未偷换成清洗后的like合同。更密、与like相关更强，都不等于新的规则已被证明是满意度。

## History意味着什么

曝光历史可供传统召回设计使用，但必须严格早于target；冻结MVP仍使用target前like history。**Synthetic示例**：历史中的A和B早于target C，C本身及之后的反馈不能放进history。现有ordering依赖official-order assumption，不升级为独立timestamp验证。

少数高贡献用户占了大量like和训练prefix；因此事件规模不能代替独立用户覆盖。末尾截断表计算完整历史快照能保留多少，不能当作每个训练prefix实际截断的比例。当前prefix规则、最近like历史、train曝光catalog和既有query均不改。

## 工业流程中的几种不同角色

观看行为可以作为positive规则，也可以作为连续watchtime目标、训练权重，或与like/share/dislike并列的multi-task目标。这是四种不同设计，不能只加一个ratio门槛就声称实现了多目标排序。绝对时长和完播比例都受duration影响；工业研究会专门处理duration bias，并结合多个互动目标，而不是预设完播就是满意度。参考[duration bias研究](https://arxiv.org/abs/2406.07932)与[Google多任务视频排序](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/)。本项目本轮仅做train关联与可行性画像，没有实现上述模型。

Negative sampling也需区分：未曝光item缺少用户反应；曝光未like表示“未观测到like”；明确dislike表示显式负反馈。三者不能全当成真负例，本轮没有创建新的negative sampler。

## 成果、问题与下一决策

已提供可复核的源码、synthetic oracle、aggregate与hash、基数一致性和独立复核记录。发现并修正旧awaiting状态，以及摘要字段统计对象的命名；保留原扫描和修正历史，坏结果没有删除。

限制仍包括subset的irX机制不明确、匿名context不能映射业务名称、缺timestamp/session、观看编码机制不完整、相关性不证明满意度或因果效果。Validation/test已经用于既有审计与smoke，不是未来选择新label时未触碰的final test。本轮没有读取它们，也不以holdout决定探索阈值。

下一决策是是否另立版本研究watch目标或composite意图，并为新的比较协议安排合适评估；这需要单独批准，不能直接替换已冻结p3-like-v1后复用旧模型比较声称提升。本次P3补充EDA已通过Astra最终验收；本worker停在commit前，等待Eddy下一指示。
