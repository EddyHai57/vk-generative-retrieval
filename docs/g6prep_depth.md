## G6-PREP：语义 ID 深度 +1 的实现说明与预注册
状态：实现说明，门禁与常量已写进代码与 `configs/g6prep/g-g6-tok.json`。本文件没有任何指标数字（实验结果只写 `docs/experiment_log.md`）。

### 1. 范围
- 做的事：把 G 线的语义 ID 从「3 位语义码 + 1 位冲突后缀」改成「4 位语义码 + 1 位冲突后缀」，其余设定（A1 模型、HIST_V2_OBS L=50、beam 100、INIT_EMB_N002、BUDGET_E100_MIN5_P3）不变。
- 这是 G-HP-01 的第 6 个旋钮。它同时改了三样东西，不能说成「只多了一层语义」：语义深度 3→4、历史序列长度 200→250 个 token、词表大小 80→68（16 叉 BHKM 下）。
- 全部是新增文件。旧文件、旧 config、旧 anchor 的 binding 字节不变；深度 3 时新入口与旧入口逐位相等（测试钉死）。

### 2. 新增模块
| 文件 | 作用 |
|---|---|
| `src/vkgr/sid_depth.py` | 深度参数化的 codec 检查、`SIDTokensD`、`ValidatedCodecD`、`SIDTokensLD`、`assign_suffix` |
| `src/vkgr/sid_depth_beam.py` | `depth + 1` 步的受限 beam：串行、同宽分组、批量（batch=1 才与串行逐位相等） |
| `src/vkgr/sid_tok_audit_d.py` | 深度参数化审计与门禁（含 B2），容量规则 `capacity_table` 与冻结块 `PREREG_BHKM_B16D4` |
| `src/vkgr/sid_rqvae_d.py` | 层数可配置的 RQ-VAE（levels=3 与旧模块逐位相等；levels=4 的前 3 个码本初值不变） |
| `src/vkgr/sid_v2d.py` | SIDMAP_V2D bundle 合同：在 SIDMAP_V2 的 17 个 manifest 键上加 `depth`，自己的版本表，与 SIDMAP_V2 互相拒收 |
| `src/vkgr/sid_tok_run_d.py` | `fit` / `verify` 命令行（变体 `B16D4`、`B16D3_REPLAY`、`RQ256D4`） |
| `src/vkgr/generative_g6_model.py` | 位置表为 `(depth+1)×50` 行的 A1 模型；`depth+1` 个位置的损失与训练步 |
| `src/vkgr/generative_g6_data.py`、`generative_g6_run.py` | 上下文构造与训练入口（`train` / `verify`） |
| `configs/g6prep/g-g6-tok.json`、`g-g6-a1-bhkm-b16d4-s42.json` | 拟合 config；训练 config（与 A1 config 只差 `schema`、`task_id`、`sid`） |

版本标签：`SID_BHKM_B16D4_S42`（正式）、`SID_RQVAE_B256D4_S42`（正式，只对应 GTOK-01 的 T1，T1 拟合没过利用率门槛就不拟合）、`SID_V2D_REPLAY_BHKM_B16D3`（工程对拍专用，深度 3，其映射必须与 anchor 映射逐字节相等）。

### 3. BHKM_B16D4 预注册数字（N = 120,415）
容量规则只依赖 n 和 k：一个 n 个 item 的节点分成 `min(branching, n)` 个孩子，孩子大小为 `n // k`，余数 `n % k` 个孩子各多 1。所以这些数字与 embedding 无关，拟合前就能确定。

| 项 | 深度 3（anchor） | 深度 4（预注册） |
|---|---|---|
| 叶子数 / 占用 | 4,096 / 4,096 | 65,536 / 65,536 |
| 每位实际取值数 | 16, 16, 16 | 16, 16, 16, 16 |
| 叶内后缀词表 S | 30 | 2（54,879 个叶子有 2 个 item，10,657 个叶子有 1 个） |
| 词表 V = 2 + depth×16 + S | 80 | 68 |
| 完整 SID 碰撞 | 0 | 0 |
| 语义碰撞组 / 碰撞 item | 4,096 / 120,415 | 54,879 / 109,758 |

为什么不会像 T3（256 叉 3 层）那样退化：决定第 4 位能不能分满 16 叉的是第 3 层节点的大小（29 或 30，都 ≥16）；平均每叶 1.84 个 item 这个小数字只出现在叶子里，体现为后缀只有 {0, 1}。

### 4. 门禁
- 所有 tokenizer：H1 完整 SID 无碰撞；H2 往返与可达；H3 无死路前缀；H4 ids 与结构（升序、位在范围内、后缀规范）；H5 内容 digest 等于 CONTENT_V1；H6 数值有限。
- RQ-VAE：R1 4 层每层利用率 ≥80%（整数判定 `used×5 ≥ K×4`）；R2 跑满固定步数且在 wall cap 内；R3 重建 R² > 0；R4 5 个残差范数（输入 + 4 层）严格递减。
- BHKM：B1 每个节点兄弟大小差 ≤1（深度 0 到 3 全查）；B2 审计重算的 §3 各项与冻结块逐项相等（正式 B16D4 只与冻结字面量比；工程映射与容量规则对其自身大小算出的块比）。
- 任一硬门禁不过：只写 `failure.json`，不产出可加载 bundle，不提交训练，名额不消耗，不重调。

### 5. 披露（坏消息在前）
- **深度 4 不是「深度 3 叶子再切一刀」，而是整棵树重拟合。** `sid.fit` 在递归里用同一个随机数流初始化每个节点的 k-means，第 3 层节点在深度 4 下也要抽一次，所以从第一个子树之后所有节点的初始化都与深度 3 不同。前 3 位与 anchor 的前 3 位大多不会一致；`verify --baseline-mapping` 只报告一致率，不设阈值。
- 位置表变大会改变初始化的随机数流（`nn.Embedding(250)` 比 `nn.Embedding(200)` 多消耗随机数）。深度 3 时表大小仍是 200，构造顺序不变，不受影响。
- 评估器按名字调用冻结的 `tiger.decode_history` 与 `generative_batched.decode_histories_batched`。训练入口在评估调用期间临时换成深度通用版本，结束后恢复（与 `generative_b1_arch` 同一种作用域内替换），没有修改任何冻结文件。因此入口要求 `VKGR_EVAL_WORKERS=1`，批量解码只允许 `VKGR_DECODE_BATCH=1`。
- 估算（未实测）：训练 wall 约 ×1.25–1.6，解码 wall 约 ×1.1–1.4，显存需实测后再声明。

### 6. 复现命令（路径用占位符）
```
python -m vkgr.sid_tok_run_d fit --variant B16D4 --config <g-g6-tok.json> --config-sha <sha> --content <content.npy> --content-sha <sha> --catalog <catalog.json> --catalog-sha <sha> --output <新目录>
python -m vkgr.sid_tok_run_d verify --bundle <run_manifest.json> --sha <sha> --config <g-g6-tok.json> --config-sha <sha> --content <content.npy> --content-sha <sha> --catalog <catalog.json> --catalog-sha <sha> [--replay-fit] [--baseline-mapping <npz> --baseline-sha <sha>]
python -m vkgr.generative_g6_run train --prepared <manifest> --prepared-sha <sha> --history-manifest <manifest> --history-sha <sha> --sid-manifest <run_manifest.json> --sid-sha <sha> --content <content.npy> --content-sha <sha> --config <g-g6-a1-bhkm-b16d4-s42.json> --config-sha <sha> --output <新目录> --scratch <新目录>
```
测试（G 测试只能在冻结基线 harness 里跑）：`OMP_NUM_THREADS=1 python -B -m unittest discover -s tests -p 'test_generative_g6*.py'`；其余新测试可直接 `python -m unittest tests.test_sid_depth tests.test_sid_depth_beam tests.test_sid_tok_audit_d tests.test_g6_bhkm_prereg tests.test_sid_rqvae_d tests.test_sid_v2d tests.test_sid_tok_run_d` 以及各自的 `_mutations` 模块。
