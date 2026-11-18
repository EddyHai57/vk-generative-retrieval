# VK-LSVD 共用底座

本包实现第十三轮本地数据与固定查询评估，合同见 `docs/data_contract.md`，本轮结果入口为 `docs/research/foundation_2026-11.md`。

## 环境

本轮实测使用 Python 3.12、DuckDB 1.4.3、NumPy 2.3.5、FAISS CPU 1.15.1。FAISS 仅内容均值检索需要，测试使用标准库 `unittest`。通过 `PYTHONPATH=src` 导入本包。仓库根 `pyproject.toml` 属于旧代码环境，其 Python 版本范围尚未为本包更新。本包使用的 DuckDB hash 与冻结 SQL 绑定，变更 DuckDB 版本前需要重新核对标签。

## 正式产物

显式传入 `cache/round13-foundation/dataset-v3`。其 `foundation.json` 引用 `dataset-v2/master` 内的完整主表，两个目录需要一起保留。首次构建和内存失败产物保留用于追溯，不能把失败目录当作正式入口。

| 文件 | 用途 |
| --- | --- |
| `foundation.json` | 输入 SHA、冻结表达式、DuckDB 版本、主表引用、构建耗时和内存 |
| `sequences/user_block=*/users.parquet` | 每训练用户完整有序列表，以 week、ordinal 排序 |
| `queries.parquet` | 每验证用户固定查询，含空历史用户 |
| `targets.parquet` | 三把尺子正样本并集，按用户和视频唯一，附各分层 |
| `items.parquet` | 全元数据目录，追加训练曝光计数和热度层 |
| `arrays/` | 上述数据的紧凑评估表示 |
| `popularity_diagnostics.csv` | 初始诊断热度层的视频数、训练曝光与验证目标占比 |

`history_items.npy` 按视频 ID 排序，仅供全部训练历史过滤。时序消费读取 `sequences`。训练时先截取严格早于目标的前缀，再取最近 200 条，固定验证查询直接使用训练末历史。九位反馈 mask 的顺序记录在 `foundation.json`。验证行的 `train_weight` 恒为零，主目标标记仍保留。

## 命令

```sh
export PYTHONPATH=src
python -m lsvd --help
python -m unittest discover -s src/lsvd/tests -v

python -m lsvd build \
  --raw-root cache/ur001-raw \
  --frozen-root cache/round4-2026-11/flat-calibrated \
  --output cache/new-foundation --memory-limit 8GB

python -m lsvd verify \
  --foundation cache/round13-foundation/dataset-v3 \
  --output cache/new-verification

python -m lsvd baselines \
  --foundation cache/round13-foundation/dataset-v3 \
  --model global_popularity --output cache/new-global

python -m lsvd evaluate \
  --foundation cache/round13-foundation/dataset-v3 \
  --predictions cache/new-global --model-name global_popularity \
  --output cache/new-global-evaluation

python -m lsvd paired \
  --first cache/first-evaluation --second cache/second-evaluation \
  --output cache/new-paired-evaluation

python -m lsvd.regression \
  --current cache/round13-foundation/evaluations/content_mean \
  --legacy cache/round9-sid/quality/full_catalog_ann \
  --legacy-users cache/round9-sid/inputs/validation_user_ids.npy \
  --output cache/new-content-regression.csv
```

所有输出路径必须尚不存在。`build` 明确列举训练 week00–24 与验证 week25 输入。`baselines` 另支持 `last_author_popularity`、`covisitation` 和 `content_mean`。共现需要 `--graph`，并对复用图执行新序列审计。内容检索需要 `--vector-ids`、`--vectors`、`--centroids`，使用冻结量化中心恢复完整目录索引，无拟合步骤。

内容回归入口明确将旧数组的“用户、目标范围、K”映射到新结果的“slice、K、用户”，逐用户对齐目标数和分数，并通过退出码报告是否一致。旧命中数没有保存时在 CSV 中明确披露，不由宏平均相同推断逐用户一致。

## 评估接口

`EvaluationData.load(dataset)` 与 `evaluate(data, predictions, model_name=...)` 接受按 query 用户顺序对齐的二维整数矩阵，或 `user_id -> 有序视频列表` 的 mapping。每用户输入最多 1000 个视频，`uint32` 最大值仅作 padding。统一执行稳定去重、过滤全部训练曝光，再截到 K。未知目录视频报错，缺失推荐视为空列表。

封顶召回为 `hits / min(K, 该栏目标数)`。短推荐列表仍使用该分母。零目标用户保留在存储数组中，宏平均仅包含该栏有目标用户。三把尺子、place、训练见过与未见、两套时长和热度诊断共形成 495 栏，每栏报告 K=50/200/1000。每用户 Poisson(1) cluster bootstrap 默认 1000 次，种子 2026111301，各模型共享权重生成规则。

`EvaluationResult.save/load` 保存逐用户 counts 和 hits，加载时重建 scores。`paired_bootstrap(first, second)` 校验评估输入身份及双方 foundation 绑定，输出前者减后者。正式模型比较还需核对各自特征和训练配置。

基线从完整排序源调用统一过滤函数，准备最多 1000 个未看候选，评估器再次幂等过滤。此候选供给口径记录在 manifest 中。内容均值仅从最近 200 条全部场景曝光里选主尺子正样本，构造查询向量，使用全目录 IVFFlat、nprobe=64 近似检索。空画像无回退推荐。
