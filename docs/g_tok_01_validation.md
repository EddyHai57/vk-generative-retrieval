## G-TOK-01 tokenizer 实现说明与预注册（草稿，FEAT-003）

状态：实现草稿。门禁与常量已冻结，指标结果不在本文件（实验结果只写 `docs/experiment_log.md`）。本文件没有任何指标数字。

### 1. 范围与 claim boundary

- 做的事：在官方 64 维内容向量（CONTENT_V1）上，比较三种 Semantic ID tokenizer 对 G 线（TIGER 风格 encoder-decoder）检索效果的影响。
- 这是**方法对齐的 tokenizer 消融**，**不是 TIGER 复现**。输入是 64 维官方向量，不是论文的 Sentence-T5 768 维；数据、模型规模、训练长度都不同。
- tokenizer 只读两个文件：`content.npy`（CONTENT_V1，第 0 行是全零 PAD 行）和 `catalog.json`（升序 item id 列表）。不读任何 label、query、split 或交互文件，代码里没有这些文件名。
- 单 seed 的结果只能说明「在这个设定下」；有效门槛按冻结卡的 ΔR@50 ≥ 0.002 判定，不事后加阈值。

三个 tokenizer（T0 是已有基线，不重新训练）：

| 标签 | sid_version | 方法 | 每位取值上限 |
|---|---|---|---|
| T0 | SID_BHKM_B16D3_S42 | 已冻结的 16 叉 balanced k-means | 16 / 16 / 16 / 后缀 |
| T1 | SID_RQVAE_B256D3_S42 | RQ-VAE，3 层码本，每层 256 | 256 / 256 / 256 / 后缀 |
| T2 | SID_RQVAE_B16D3_S42 | RQ-VAE，3 层码本，每层 16 | 16 / 16 / 16 / 后缀 |
| T3 | SID_BHKM_B256D3_S42 | 冻结的 `sid.fit`，256 叉，深度 3，10 轮，seed 42 | 256 / 256 / 256 / 后缀 |

### 2. 论文与实现的差异表

论文：TIGER，arXiv 2305.05065，§3.1 与 §4.1.3。带 ⚠ 的行是论文没给、我们预先固定的选择，不是论文的做法。

| | 论文 | 本实现（冻结） |
|---|---|---|
| 输入 | Sentence-T5 768 维 | 官方 64 维（CONTENT_V1，L2 归一化 float32，零范数行保持零向量并计数） |
| 编码器 | 中间层 512/256/128，潜在维 32 | 64-512-256-128-32，ReLU，最后一层线性 |
| 量化 | 3 层残差量化，每层独立码本 256×32 | 同；T2 把每层码本改为 16 |
| 损失 | 重建 + Σ(码本项 + β·commitment)，β=0.25 | 同 |
| ⚠ 解码器 | 只说 DNN decoder | 镜像 32-128-256-512-64，ReLU，末层线性 |
| ⚠ 潜在层激活 | 未说 | 线性 |
| ⚠ Adagrad | 只给 lr=0.4 | `initial_accumulator_value=0.1`，`eps=1e-7` |
| ⚠ 残差链 | 未说是否 detach | `r_{d+1} = r_d − sg[e_{c_d}]`：码本只由码本项训练，commitment 只更新编码器 |
| ⚠ 直通 | 未明说 | `ẑ_st = z + sg(ẑ − z)`，解码器吃 `ẑ_st` |
| ⚠ reduction | 未说 | 特征维求和，batch 维取均值 |
| ⚠ 码本初始化 | 第一个 batch 上 k-means | 第一个 epoch 的第一个 1024 batch，逐层：先对 z 做 k-means，再对残差做 k-means；Lloyd 10 轮，初始中心为 seed 42 无放回抽样 |
| ⚠ 权重初始化 | 未说 | PyTorch 默认 Linear 初始化，FP32，关闭 TF32 |
| ⚠ 死码重置 / lr 调度 / 梯度裁剪 | 无 | 不加（对齐原版；加就是新变体） |
| 碰撞处理 | 训练完成后用第 4 码消除，无碰撞补 0 | 同：前三码相同的 item 按 item id 升序编号，唯一时补 0 |
| 训练长度 | 20k epochs，保证码本利用率 ≥80% | 固定 300,000 步（约 2,542 个 epoch，每 epoch 118 步），不早停、不按 loss 选 checkpoint |

### 3. 预注册规则与门禁

训练长度与资源：

- `max_steps = 300000`，batch 1024，发布的模型是第 300,000 步的模型。
- wall cap 7,200 秒，从训练循环开始计时，不含读输入、导出和审计。每 100 步检查一次；超时即失败，不截断使用。
- 第 2,000 步用第 500–2,000 步的速率外推；外推超过 wall cap 时立即失败（`fail_wall_projected`）。
- 正式拟合要求 CUDA；CPU 只在 `--engineering` 下允许，且 bundle 带 `engineering=true`，正式 loader 拒收。

硬门禁（任一不过 = 该 tokenizer FAILED，不产出可加载 bundle，只写 `failure.json`，CLI 退出码 3）：

- H（所有 tokenizer）：H1 完整 SID 无碰撞；H2 每个 item `decode(encode(item)) == item` 且可达；H3 无死路前缀；H4 ids 与 catalog 逐元素相同、数位在范围内、后缀规范；H5 输入内容 digest 等于 CONTENT_V1；H6 数值有限。
- R（仅 RQ-VAE）：R1 每层码本利用率 ≥80%（整数判定 `used × 5 ≥ K × 4`，T1 每层 ≥205/256，T2 每层 ≥13/16，在全量 catalog 的 canonical export 上计）；R2 跑满 300,000 步且在 wall cap 内；R3 重建 R² > 0；R4 逐层平均残差范数严格递减。
- B（仅 balanced k-means）：B1 逐节点子节点大小之差 ≤ 1（从 mapping 重算，不依赖已丢弃的 nodes）。

诊断（不是门禁，不参与任何选择，不触发提前终止）：

- 每层码本利用率（已用码数 / K）在第 2,000 步和最终步各记录一次，写进 bundle 的 `train-log.json`；训练期间每 2,000 步还会记录一次。
- 预注册的 lr 0.4 在小型合成数据上会让码本塌缩（FEAT-001/002 的观察）。这条不改 lr：真实 64 维内容上是否塌缩，由服务器上第 2,000 步和最终步的利用率记录说明；失败就是失败，重试必须先写 decision_log 并使用新的 attempt id。

报告项（没有阈值，只报告）：每个位置的取值数、利用率、困惑度；叶子（前三码相同的组）的数量、最大、中位数、p95、p99、均值、单例数；第 4 码取值数 S；碰撞组数与碰撞 item 数；前缀组数；按前 d 码分组的解释方差；词表大小 `V = 2 + 3b + S`；RQ-VAE 的重建 MSE、R² 与逐层残差范数。比较表把 T0 同口径放在第一列。

浮点复算允许相对误差 1e-9，整数、字符串、布尔必须逐项相等。

### 4. SIDMAP_V2 bundle 与旧契约隔离

- 旧的 P5 加载器写死 16 叉、平衡结构，且被钉在 P5 manifest 的 sha 上，不能复用也不改。新 tokenizer 用独立的 bundle（schema `sidmap-v2`，protocol `generative-like-sid-v2`，task `G-TOK-01`）。
- bundle 目录只含：`mapping.npz`、`tokenizer.json`、`audit.json`、`run_manifest.json`（最后原子发布），RQ-VAE 另有 `weights.npz` 和 `train-log.json`。多一个文件就拒收；失败的目录里只有 `failure.json`。
- `load_bundle` fail-closed：重算 manifest sha、各 artifact sha、content 与 catalog 身份、codec 规则、审计和门禁，并用权重重新导出 mapping 逐元素比对。P5 manifest 被它拒收，v2 manifest 被 P5 加载器拒收。
- 运行观测（设备、torch 版本、墙钟）写在 `train-log.json`（RQ-VAE）或 `tokenizer.json`（k-means），**不**写进 `audit.json`；`audit.json` 里只有门禁 R2 需要的 `elapsed`。

### 5. T3 退化说明

N = 120,415 时，`sid.fit` 每层取 `k = min(256, 节点大小)`：第 1 层 256 个节点各约 470 个 item；第 2 层每个子节点约 120415 / 256² ≈ 1.84 个 item；第 3 层只剩 1 到 2 个 item。因此第 3 位只取 {0, 1}，第 4 位恒为 0。T3 实际是「256 × 256 × 2 加恒 0 后缀」，不是 256 叉三层语义树，解读结果时不能把它当作 256 叉语义层级。算法本身不改。

### 6. 命令（占位符）

```
python -m vkgr.sid_tok_run fit --variant T1|T2|T3 --config <CONFIG> --config-sha <SHA> \
    --content <CONTENT> --content-sha <SHA> --catalog <CATALOG> --catalog-sha <SHA> \
    --output <OUT> [--device cuda] [--engineering --max-steps <N>]
python -m vkgr.sid_tok_run verify --bundle <OUT>/run_manifest.json --sha <SHA> --config <CONFIG> --config-sha <SHA> \
    --content <CONTENT> --content-sha <SHA> --catalog <CATALOG> --catalog-sha <SHA> [--replay-prefix] [--replay-fit]
python -m vkgr.sid_tok_run compare --bundle <T1 manifest> --sha <SHA> [--bundle <T2 manifest> --sha <SHA>] [--bundle <T3 manifest> --sha <SHA>] \
    --baseline-mapping <BASELINE_NPZ> --baseline-sha <SHA> --content <CONTENT> --content-sha <SHA> \
    --catalog <CATALOG> --catalog-sha <SHA> --output <OUT>
```

退出码：0 成功；1 参数、输入或校验错误（此时没有任何输出目录）；3 门禁失败（输出目录里只有 `failure.json`）。`<OUT>` 必须是不存在的新目录。

### 7. UNKNOWN 清单

- U1 T1/T2 实际的第 4 码取值数 S 与叶子分布；U2 N = 120,415 上审计与 codec 构造耗时（本地合成数据已测，服务器未测）；U3 大扇出下每 epoch 的解码墙钟；U4 64 维输入上码本利用率与碰撞率能否过 80%；U5 GPU 训练循环的可复现性（只探测前 2,000 步的权重摘要）；U6 确定性算法下的 GPU 步速；U7 T3 的真实 CPU 墙钟与峰值内存；U9 预注册 Adagrad lr 0.4 与累加器 0.1 在真实 64 维内容上是否稳定。

### 8. 机器可读预注册块

下面一段 JSON 包含所有冻结常量，`fit_config` 必须与 `configs/gtok01/g-tok-01.json` 逐字段相同（FEAT-005 会把它变成测试）。

<!-- gtok01-prereg:begin -->
```json
{
  "schema": "g-tok-01-prereg-v1",
  "task_id": "G-TOK-01",
  "fit_config": {
    "content": {
      "dim": 64,
      "feature_array_sha256": "766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84"
    },
    "rqvae": {
      "batch_size": 1024,
      "beta": 0.25,
      "check_every": 100,
      "content_dim": 64,
      "encoder_hidden": [512, 256, 128],
      "eps": 1e-07,
      "eval_every": 10000,
      "initial_accumulator_value": 0.1,
      "kmeans_iters": 10,
      "latent_dim": 32,
      "levels": 3,
      "log_every": 1000,
      "lr": 0.4,
      "max_steps": 300000,
      "optimizer": "adagrad",
      "probe_from": 500,
      "probe_step": 2000,
      "seed": 42,
      "usage_diag_every": 2000,
      "usage_gate_num_den": [4, 5],
      "wall_cap_seconds": 7200
    },
    "schema": "g-tok-01-fit-v1",
    "task_id": "G-TOK-01",
    "variants": {
      "T1": {"codebook_size": 256, "kind": "rqvae", "sid_version": "SID_RQVAE_B256D3_S42"},
      "T2": {"codebook_size": 16, "kind": "rqvae", "sid_version": "SID_RQVAE_B16D3_S42"},
      "T3": {"branching": 256, "depth": 3, "kind": "bhkm", "max_iter": 10, "seed": 42, "sid_version": "SID_BHKM_B256D3_S42"}
    }
  },
  "gates": {
    "usage_gate_num_den": [4, 5],
    "required_steps": 300000,
    "wall_cap_seconds": 7200,
    "balance_tolerance": 1,
    "float_rel_tol": 1e-09,
    "groups": {"H": ["H1", "H2", "H3", "H4", "H5", "H6"], "R": ["R1", "R2", "R3", "R4"], "B": ["B1"]}
  },
  "baseline": {
    "label": "T0",
    "sid_version": "SID_BHKM_B16D3_S42",
    "branching": 16,
    "mapping_sha256": "9c0c76600dec91957a87119405018793b7216db28f07597556ef79876e4ebfc7"
  },
  "exit_codes": {"success": 0, "argument_or_validation_error": 1, "gate_failure": 3}
}
```
<!-- gtok01-prereg:end -->
