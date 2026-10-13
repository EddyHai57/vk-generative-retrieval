密级: 公开

# MVP Data Card

P3已完成train-only审计并实际冻结合同，冻结后的holdout统计与query核对已完成，P3已正式协调验收通过；审计数据统计见`experiment_log.md` P3，运行证据与hash见`p3_validation.md`，完整policy唯一源为`data_contract.md`。本文件只描述用途与限制。

数据来自公开VK-LSVD固定revision的`ur0.01_ir0.01` subset；许可声明和下载审计沿用P2。Raw与metadata/embedding只在现有服务器，公共仓库不再分发完整数据、用户query或ID集合。所有生产实现为本项目原创；没有复制官方示例或第三方模型代码。

只读train确定like支持门、history、target、split与catalog，先保存freeze hash才读取validation/test统计。没有普通click，不做CTR；作者页/评论区打开保留各自原义。官方已有validation与test partitions，本MVP使用官方split。

官方声明week/file内时间顺序，但parquet没有timestamp。我们核对文件/schema、physical读取稳定性与重复计数，不能独立验证隐藏timestamp或其ties；排序位置只按official-order assumption解释。first-seen是observed-item proxy，不能称真实上新。subset的irX生成机制在固定README未明确，统计不外推全量平台。

主label是like；like稀疏仍可支持小型工程smoke，但工程support门不等于评估代表性或统计效力。长user可贡献多个训练prefix；evaluation每user仅一个首positive target。真实首target为OOV时必须保留，不能改选首个in-catalog positive。

验证与test history都固定train-only，不把validation反馈加入test history。这是保守MVP协议，未使用全部test前可观察历史；结论不能与使用rolling全部历史的benchmark直接对比。候选域仅train-observed items，metadata/embedding对未来item的覆盖不会扩充候选。

后续baseline与生成式方法必须共享冻结history/features/catalog/query/split/budget；P4 audit用户抽样固定seed并保留OOV。P3无模型训练、retrieval指标、线上A/B、causal lift或完整TIGER复现结论。

本轮Python依赖来自官方PyPI distribution：PyYAML6.0.2（MIT）、NumPy1.26.4（BSD声明，wheel另含其上游notice）、PyArrow19.0.1（Apache Software License声明）。版本由真实uv.lock固定；未改driver、torch或GPU FAISS环境。完整发布provenance仍在I-007/P13复核。

实测OOV占比较高，train-observed候选域存在结构性不可达目标；完整eligible分母、in-catalog slice与OOV slice必须同时保留。数据统计及过滤原因只读experiment_log；不得因为该结果改label、换target或扩catalog后沿用本policy版本。
