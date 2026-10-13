# Generative consumer technical contract v1

密级: 公开

共同 label/history/split/catalog/features 规则唯一事实源为 [data_contract.md](data_contract.md#traditional-like-v1)，其原始 bytes 不变；本文件只定义生成式消费者技术协议。批准的阶段预算唯一来源是主协调 tasks.md 生成式路线；机器声明 configs/generative-core-v1.json，不代表训练或资源窗口已获授权。

监督为四个位置 teacher-forced token CE，按所有 example × 四个位置取均值，每位置使用完整 vocabulary 归一化。PAD=0，BOS=1，四个 digit 使用互斥 namespace，无 EOS；decoder 为 BOS 加前三 digit，labels 为全部四 digit。复用 SIDTokens；past-positive negative mask 是双塔规则，不应用于本 token loss，也不额外搜索 temperature。

搜索复用原 constrained beam，先 full-vocabulary log_softmax，再限定合法 next digit，累计四位置分数；prefix 平分按 SID lexicographic，最终 item 平分按 item_id。beam100 是近似 search，不称 full-catalog exact ranking。最终指标复用 baselines.metrics 的 collision-corrected item Recall/NDCG@10/20/50；OOV 保留分母，SID hit 不代替 item hit。本协议不重定义传统合同的 exact vector arithmetic。

provenance/g0-v1/ 是不可修改历史 capsule，仅固定 37 个公开原文件加 inventory.json；不得在里面加 README、cache、git、venv 或真实数据。成员来自原 G0 consumer_source.files、P3-D bound_files 与原 fixture/manifest 的并集，外部 G0/P3-D manifest pins 再约束 inventory，改 inventory 并 rehash 不能替换旧源码。它仅用于历史 replay，不是活跃代码或第二个数据合同。独立进程 cwd 和 PYTHONPATH 必须指向该 capsule；import 的实际 __file__ 须核。当前 source 新增 consumer 后，旧公开 G0 manifest 在当前根拒绝属于正确的身份 gate，不修改旧 manifest 或 validator。现有 G0 tests 仍动态生成当前 manifest。

G1 manifest 的新 consumer source 为显式成员闭合集合、实际 Git base/dirty、lock、Python 与 exact dependencies；与原 producer 各自独立。manifest 不 hash 自身。load_contract 要求 caller 外部 SHA，并与真实当前成员和固定历史声明比较。跨 Mac/Linux 或 Python patch 版本部署需要显式冻结该环境的 consumer manifest 并重新验收，不能偷偷放宽历史 producer 依赖。内部 validated context 仅封装已验证声明，修改其字典后必须拒绝；synthetic 单元测试的 context 只用于隔离 loader 行为，不是正式合同。

prepared 外部 manifest pin 先于 artifacts，固定 schema/producer/input bindings 后逐 example 验 strict-past 与 support/idmap；不重构 raw 或读取 test。validation 仅固定 validation role/hash，文件名必须 validation-queries.json，test/unknown role 在读取前拒绝；验证成品 query 本身不能证明 first-like 最早，真实证据来自已经验收的 producer 及固定 query hash，synthetic 构造语义另有测试。

P5 static loader 校验映射/constraint/codebook/normalized features/artifact hash；G1 额外核 prepared 与 SID catalog、content float32 bytes 同一。静态消费者验证不等于重跑旧 producer。当前 canonical suffix gate 不放宽；后续 random SID control 会改变地址分配，必须另有版本化 mapping/manifest 与相应 gate，G1 只冻结这个要求。

内容距离仅 train-positive support 中每个 semantic leaf 按 item_id 的相邻 pair，按 leaf/item_id 顺序最多 4096 pairs，float64 squared L2，不建 N² 矩阵。报告 count/mean/max 及 unmatched population，不作语义优良的证明。真实 audit 包含全 catalog 唯一/可逆/可达、原碰撞/suffix/各层占用/收敛；未测资源记 not_measured，不能填通过。没有真实 server artifacts 时整体 G1 仍 pending real SID cross-input audit；本地 synthetic/static 通过不能代替。

Stop gates：旧 pins/成员冲突、历史 replay 不一致、source/catalog/features/时序漂移、任何隐式 test 读、需要改共享源码或依赖、资源/预算授权缺失。G1 不启动 optimizer、SSH/GPU 或下一阶段。

已验证 SID codec 的 mapping/constraint arrays 使用 immutable bytes backing，固定对象身份、shape/dtype/bytes/strides/data pointer 与 namespace；batch 入口只做 O(1) identity gate 和本 batch item 查找。完整 canonical suffix gate 在 load/audit 中一次按 semantic leaf 排序分组，避免每 batch 逐 leaf 重扫全 catalog。任何 mapping/layout/namespace 替换先拒绝，不以 readonly 标志单独判断安全。
