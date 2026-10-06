密级: 公开

# Experiment Log

## 当前状态

尚无实验。

本轮未下载数据、未训练模型、未验证效果。

哈啰内部信息、数据、schema、代码与实验结果不得进入本项目。

本文件是人工维护的实验与指标唯一摘要源。

原始 config、metric JSON、日志与大 artifact 不复制到本文，只记录路径别名与 hash。

## Run Summary

| Run ID | Status | Dataset | Split | Catalog | SID | Model | Seed | Hardware | Metrics | Resources | Artifact hashes | Claim boundary |
|---|---|---|---|---|---|---|---:|---|---|---|---|---|

当前表为空，不预填任何指标或预期收益。

## Run Record Template

### Identity

- Run ID:
- Date:
- Status: `planned | running | completed | failed | invalidated`
- Git commit:
- Config path:
- Parent run:
- Purpose:

### Dataset

- Dataset/subset:
- Source manifest hash:
- Processed dataset hash:
- Schema version:
- Row count:
- User count:
- Item count:
- Week range:
- Feedback policy version:
- Duplicate policy version:

### Split

- Split manifest hash:
- Train weeks:
- Validation weeks:
- Test/holdout weeks:
- Query set hash:
- Query count:
- History rule:
- Target rule:
- Holdout usage boundary:

### Catalog

- Catalog snapshot ID:
- Catalog cutoff:
- Train-observed item count:
- First-seen/OOV target count:
- Feature coverage:
- Catalog hash:

### SID

- Tokenizer name/version:
- Codebook config:
- SID depth/vocabulary:
- Collision suffix rule:
- SID map hash:
- Collision diagnostics artifact hash:
- Train/freeze week:

### Model

- Model family:
- Parameter count:
- Input features:
- History length policy:
- Objective:
- Decode policy:
- Constraint version:
- Checkpoint hash:
- Baseline fairness notes:

### Reproducibility

- Seed:
- Determinism settings:
- Dependency lock hash:
- Command manifest hash:
- Retry/resume notes:

### Hardware

- Provider:
- GPU model/count:
- GPU memory:
- CPU:
- RAM:
- Disk:
- CUDA/runtime:
- FAISS/backend:

### Metrics

- Item Recall@K:
- Item NDCG@K:
- SID-level metric:
- Collision-corrected item metric:
- Invalid rate:
- Dead-end rate:
- Duplicate rate:
- Effective valid K:
- Train-observed slice:
- First-seen/OOV slice:
- Warm/cold slices:
- ANN recall against exact audit:
- Statistical summary / seeds:
- Metric JSON hash:

### Resources

- Data preparation wall-clock:
- Training wall-clock:
- Evaluation wall-clock:
- Refresh wall-clock:
- Peak RAM:
- Peak VRAM:
- Disk footprint:
- Index/trie footprint:
- Latency p50/p95:
- Cost:
- Resource trace hash:

### Artifact Hashes

- Source manifest:
- Dataset manifest:
- Split manifest:
- Catalog manifest:
- Feature manifest:
- SID/codebook:
- Checkpoint:
- Index/trie:
- Predictions:
- Metrics:
- Logs:

### Claim Boundary

- This run supports:
- This run does not support:
- Known confounders:
- Failed gates:
- Follow-up required:
- Result classification: `positive | negative | inconclusive | invalid`

## 填写规则

没有 artifact hash 的指标不得作为正式结果写入 summary。

One-seed 结果必须明确写 one-seed，不外推稳定性。

SID-level 指标不得替代 collision-corrected item-level 指标。

ANN 指标必须附固定 query exact audit。

`first-seen` 只能解释为 observed-item proxy，不解释为真实 listing 或 cold-start ground truth。

不报告 CTR、online lift 或 A/B 结果。
