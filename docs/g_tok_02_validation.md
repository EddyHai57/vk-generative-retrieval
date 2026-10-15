## G-TOK-02 SID 分辨率扫描 实现说明与预注册（定稿，已冻结）
状态：定稿。门禁、拟合常量与 18 份训练 config 随首次提交一次性冻结；实验结果不在本文件（只写 `docs/experiment_log.md`）。本文件没有任何实验结果。
### 1. 范围与 claim boundary
- 做的事：在官方 64 维内容向量（CONTENT_V1）上，把 balanced k-means 的叉数从 16 扫到 1024，看召回是否还涨；同时记录叶子碰撞率。训练模型和协议与 G-TOK-01 相同（core03、HIST_V2_OBS L=50、beam 100）。
- 这是**描述性的分辨率曲线**，不是因果实验。不写「分辨率越高越好」，不写「BHKM 优于 RQ-VAE」（本任务全是 BHKM），不外推到别的设定。
- tokenizer 只读两个文件：`content.npy` 与 `catalog.json`。不读任何 label、query、split 或交互文件；评估只读 validation，test 永远关闭。
- 单 seed 的点只画形状。判读只用没有参与选择的 seed 2024 与 2025（见 §6）。
### 2. 扫描点
| 点 | sid_version | 叉数 / 深度 | 来源 |
|---|---|---|---|
| B16 | SID_BHKM_B16D3_S42 | 16 / 3 | 已有 |
| B32 | SID_BHKM_B32D3_S42 | 32 / 3 | 新 |
| B64 | SID_BHKM_B64D3_S42 | 64 / 3 | 新 |
| B128 | SID_BHKM_B128D3_S42 | 128 / 3 | 新 |
| B256D3 | SID_BHKM_B256D3_S42 | 256 / 3 | 已有（GTOK-01 的 T3） |
| B512 | SID_BHKM_B512D3_S42 | 512 / 3 | 新 |
| B1024 | SID_BHKM_B1024D3_S42 | 1024 / 3 | 新 |
| B256D2 | SID_BHKM_B256D2_S42 | 256 / 2，由 B256D3 的映射确定性推出 | 新 |
- 5 个深度 3 的新点直接调用冻结的 `sid.fit(branching=b, depth=3, seed=42, max_iter=10)`，不改算法。
- **叉数大了，后面的位不再有语义。** 节点样本数 n 小于叉数 b 时 `k = min(b, n)`，每个孩子正好一个 item，该位等于「簇内编号」。所以 B64 / B128 / B256D3 只有 2 个语义层，B512 / B1024 只有 1 个。「加叉数」实际是「少一个语义层 + 更大的簇内编号」，不是「更细的语义树」。报告要同时给出每位的 distinct 数。
- **B256D2 是深度 2，叶子数少于 item 数，强制碰撞。** 它保留 B256D3 的 (c0, c1) 分区，组内按 item id 升序编号。冻结的 G 代码要求 4 位 SID，所以存成 `[c0, c1, 0, rank]`：第 3 位恒为 0，`rank` 在第 4 位。因此它是「深度 2 + 1 个恒定位」，不是 3 位 SID。它与 B256D3 只差编号位置、编号规则和恒定位，**不是碰撞对照**，也不是独立样本。
- 6 个新点各占一个新名额（不占原来的名额）；补 seed 不计名额。
### 3. 叶子碰撞率
只从拟合出的映射测，不碰训练、query、target。叶子 = 前三位相同的 item 组。
- **CER**（主指标）= `items_needing_suffix / items` = `1 − leaves.count / items`：靠最后一位才能区分的 item 占比。两种算法互相断言。
- **LCR** = `semantic_collision_items / items`：落在含至少两个 item 的叶子里的 item 占比。
- 辅助（只进表，不入门禁）：平均叶大小、最大叶、各位 distinct 数。
- 读图：碰撞率非零的点只有 B16、B32、B256D2，其余全在 0；它们和叉数、语义层数混在一起。图上标点名，并写明 B256D2 不是碰撞对照。
### 4. 门禁
- H1–H6 与 B1 原样生效，直接调用 `sid_tok_audit.gates(audit, 'bhkm')`，没有新增函数，没有新增阈值。R1–R4 只属于 RQ-VAE，本任务不适用、不调用、不放宽。
  - H1 完整 SID 无碰撞；H2 每个 item `decode(encode(item)) == item` 且可达；H3 无死路前缀；H4 ids 升序、数位在范围内、后缀规范；H5 内容 digest 等于 CONTENT_V1；H6 数值有限。
  - B1 逐节点子节点大小之差 ≤ 1，且每个节点有 `min(b, n)` 个非空孩子（从映射重算）。
- 门禁失败的配置：拟合是确定性的，不重试、不放宽；该点标 FIT_FAILED 并写明失败的检查；名额不消耗；其余配置不受影响。
- `expected` 块（`configs/gtok02/g-tok-02.json` 里每个 variant 的 8 个整数）是「推演与审计一致」的机器检查，**不是门禁**：逐项整数相等、零容差；不符则 `verify` 退出码 4，写 `expected_mismatch.json`，该配置不进入训练。
#### B256D2 的 B1 与裁定前提
- 补零不是树的一层，所以 B256D2 的 B1 作用于**补零前的三列** `[c0, c1, rank]`；函数、tolerance、判定阈值全部不变。补零后的四列上，同一函数会把每个含碰撞的节点判为违规（该节点只有 1 个非空孩子，而应有 `min(b, n)` 个）。
- 这是评审实测过的：同一份合成映射上，三列 `balanced = true、violations = 0`；补零四列 `violations > 0`。
- 预注册原文「子节点大小之差 ≤ 1」在补零映射上同样通过；代码版 B1 多出来的是「每个节点有 `min(b, n)` 个非空孩子」这一条，补零映射只在这一条上失败。所以三列在每个节点上都不弱于原文，不是放宽；但**检查对象换了**，这一点是否算修改门禁，裁定权在项目负责人。
- 在负责人明确说「不算改门禁」之前：B256D2 不拟合、不入队（服务器侧状态 HELD）。W 内的代码、`g-tok-02.json` 的 S256D2 条目和 3 份 B256D2 训练 config 照常冻结，无论怎么裁定都不改任何冻结 sha。HELD 不算「到终态」，缺点不插值；B256D2 退出时 E 规则不触发。
- 补零列全为 0 是 loader 对深度 2 标签的**格式检查**，不是门禁，不进 `checks` 列表；失败即拒收。
### 5. 选 seed 规则（预注册）
输入：8 个点 seed 42 的 verified 结果；`hits = round(R@50 × 2520)`，必须是整数。
1. 排序键：hits 降序 → N@50 降序 → 枚举序（B16 < B32 < B64 < B128 < B256D3 < B512 < B1024 < B256D2）。
2. `top2` = 排序后的前两个。
3. `near` = 与最高 hits 相差不超过 10 的所有点（含等号）。
4. 入选集 = `top2 ∪ near`；其中属于 6 个新配置的各补 seed 2024、2025。B16 与 B256D3 已有三个 seed，不补。
5. **E 规则**：只要 B256D2 的 seed 42 已 verified，无论排名都补 seed 2024、2025（它是位置对照里唯一的新点，只有一个 seed 会让这条对照落在噪声上）；B256D2 缺失、HELD 或未 verified 时不触发。
6. `k` = 需要补 seed 的新配置数（含 E 规则带来的），补 seed 训练数 = `2k`；`k` 可以是 0。
脚本 `scripts/gtok02_select.py` 只输出**提案**，不入队；结果与理由先写进决策记录，才入队。未 verified 的点不进候选池。
### 6. 判读规则（结果出来之前冻结）
符号：对配置 X ∈ {B512, B1024, B256D2}，`Δ_s(X) = hits(X, s) − hits(T3, s)`，s ∈ {2024, 2025}。门槛 6 hits（5 hits 对应的 R@50 差低于 0.002，6 hits 达到）。
- 状态只用 seed 2024 与 2025（seed 42 参与了选择，有赢家诅咒）：`UP` 两个 seed 的 Δ 都 ≥ +6；`DOWN` 都 ≤ −6；其余 `FLAT`（一升一平、一升一降的分歧单独注明）。
- 缺任一个 seed：暂定状态 `T_UP / T_DOWN / T_FLAT`，按 seed 42 的 Δ 用同一个 ±6 死区判，**永远不用来下 Q1 的确定结论**。
- Q1（分辨率继续加是否还涨）按下表自上而下取第一条命中的行（B512 与 B1024 各一个状态，6×6 种组合全覆盖）：
| 优先级 | 条件 | 输出 |
|---|---|---|
| 1 | 任一为 UP | 「还涨」；两点都有 2024 / 2025 时对 B1024 − B512 用同一规则给峰值位置（UP→B1024，DOWN→B512，FLAT→「512–1024 不可分」），否则「峰值位置未决」 |
| 2 | 无 UP，且任一为 T_UP | 「未决」：不下 Q1 结论；追加 seed 的决定必须在读到该配置 2024 / 2025 结果之前写入决策记录 |
| 3 | 两者都是 DOWN | 「256 是峰值」 |
| 4 | 无 UP、无 T_*，且不是两者都 DOWN | 「256 之后不再涨」 |
| 5 | 其余 | 「暂定：256 之后不再涨（含单 seed）」 |
- 低叉数一侧（16→32→64→128→256）只按相邻点的 seed 42 差画符号序列，**仅描述，不下「显著」结论**。
- Q2（碰撞率与召回）：8 个点的 (CER, hits) 散点，可给 Spearman 秩相关（8 个点、3 个非零横坐标，仅描述）。**不下因果。** B256D2 对 B256D3 的状态：FLAT 时只能说「同一 (c0, c1) 分区下，二级组内编号放在第 3 位（k-means 簇号）或第 4 位（item id 序，第 3 位恒定）看不出差别」；UP / DOWN 时只能说「二者只差编号位置、编号规则、恒定位，原因未分离」。
- B256D2 的恒定位：报告要给出 best checkpoint 在该位上的平均 `log p(0 | 前缀)`（只读诊断脚本，不属门禁，来不及就写 UNKNOWN）；恒定位不是严格不改 beam 排序，因为 beam 累加的是全词表 log_softmax。
- 措辞限制：必须同时写明 B512 / B1024 的语义层更少；不外推到别的 SID 组合；不写「B256D2 验证了碰撞有 / 无代价」。
### 7. 可信度边界
- 单 seed 噪声比阈值大：已有三个 seed 的 T3 与 B16 的 seed 间标准差各约 12 与 14 个 hit（设计时的取值，权威数字只在 `experiment_log`），是 6 hits 门槛的两倍左右。
- 没有真实效应时，Q1 的 UP 判定假阳性率粗估为 0.2–0.3（按每个 seed 的差近似独立、标准差约 17 个 hit 估：单 seed 差 ≥ 6 的概率约 0.36，两个 seed 同时约 0.13，B512 与 B1024 两个点任一为 UP 约 0.25）。这只是读法上的提醒，不改任何阈值。
- 赢家诅咒：seed 42 用来选人，所以确定性判读只用 2024 / 2025。
- B256D2 与 B256D3 共用分区，不是独立样本；补零恒定位的成本需要单独诊断。
- 所有点都是 balanced k-means 的 seed 42 拟合，一次拟合一个映射；没有拟合方差的估计。
### 8. 命令与退出码
```
python -m vkgr.sid_tok2_run fit     --variant <S32|S64|S128|S512|S1024|S256D2> --config <CFG> --config-sha <SHA>
                                    --content <CONTENT> --content-sha <SHA> --catalog <CATALOG> --catalog-sha <SHA>
                                    --output <OUT> [--parent-manifest <T3> --parent-sha <SHA>   (只用于 S256D2)]
python -m vkgr.sid_tok2_run verify  --bundle <MANIFEST> --sha <SHA> --config ... --content ... --catalog ... --output <OUT>
                                    [--replay-fit] [--parent-manifest <T3> --parent-sha <SHA>   (S256D2 必须)]
python -m vkgr.sid_tok2_run compare --bundle <MANIFEST> --sha <SHA> (可重复) --baseline-mapping <NPZ> --baseline-sha <SHA>
                                    --content ... --catalog ... --output <OUT>
python -m vkgr.generative_tok2_run train  --prepared ... --history-manifest ... --sid-manifest ... --content ... --config ... --output ... --scratch ...
python -m vkgr.generative_tok2_run verify --prepared ... --history-manifest ... --sid-manifest ... --content ... --manifest ... --output ... --scratch ...
```
退出码：0 成功；1 参数、输入或校验错误（此时没有任何输出）；3 门禁失败（输出目录里只有 `failure.json`）；4 `expected` 不符（`verify`，只写 `expected_mismatch.json`，不动 bundle）。`<OUT>` 必须是不存在的新目录。`fit` 成功时打印含 `manifest_sha256` 的一行 JSON；`verify` 成功时打印 `"status": "verified"` 并写 `verify.json`。
### 9. UNKNOWN 清单
- S512 / S1024 的拟合墙钟与峰值内存；大词表（最大约三千）下的训练、解码墙钟与显存；真实 item 数上的审计耗时；推演表除已有两个点之外各行是否与审计一致（`expected` 块会强制检查）；本机能否出图。
- 本地合成数据和 harness 的通过**不等于**正式路径通过：正式路径（`engineering=False`、真实输入）在服务器上的第一次端到端执行才是证据。
### 10. 机器可读预注册块
下面一段 JSON 包含所有冻结常量。`tests/test_gtok02_prereg.py` 检查五处一致：`sid_tok2_run.expected_config()`、`configs/gtok02/g-tok-02.json`、18 份训练 config、本块、测试里的独立字面量副本。
<!-- gtok02-prereg:begin -->
```json
{
 "exit_codes": {
  "argument_or_validation_error": 1,
  "expected_mismatch": 4,
  "gate_failure": 3,
  "success": 0
 },
 "fit_config": {
  "content": {
   "dim": 64,
   "feature_array_sha256": "766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84"
  },
  "schema": "g-tok-02-fit-v1",
  "task_id": "G-TOK-02",
  "variants": {
   "S1024": {
    "branching": 1024,
    "depth": 3,
    "expected": {
     "items_needing_suffix": 0,
     "leaves_count": 120415,
     "leaves_max": 1,
     "positions_distinct": [
      1024,
      118,
      1,
      1
     ],
     "vocab_size": 3075
    },
    "kind": "bhkm",
    "max_iter": 10,
    "seed": 42,
    "sid_version": "SID_BHKM_B1024D3_S42"
   },
   "S128": {
    "branching": 128,
    "depth": 3,
    "expected": {
     "items_needing_suffix": 0,
     "leaves_count": 120415,
     "leaves_max": 1,
     "positions_distinct": [
      128,
      128,
      8,
      1
     ],
     "vocab_size": 387
    },
    "kind": "bhkm",
    "max_iter": 10,
    "seed": 42,
    "sid_version": "SID_BHKM_B128D3_S42"
   },
   "S256D2": {
    "branching": 256,
    "depth": 2,
    "expected": {
     "items_needing_suffix": 54879,
     "leaves_count": 65536,
     "leaves_max": 2,
     "positions_distinct": [
      256,
      256,
      1,
      2
     ],
     "vocab_size": 772
    },
    "kind": "derive_depth2",
    "parent": {
     "manifest_sha256": "a7af78a3fc04f0f0d174b983987c34d7efa754edd88a05d1350cdadd06fd47b8",
     "sid_version": "SID_BHKM_B256D3_S42",
     "variant": "T3"
    },
    "sid_version": "SID_BHKM_B256D2_S42"
   },
   "S32": {
    "branching": 32,
    "depth": 3,
    "expected": {
     "items_needing_suffix": 87647,
     "leaves_count": 32768,
     "leaves_max": 4,
     "positions_distinct": [
      32,
      32,
      32,
      4
     ],
     "vocab_size": 102
    },
    "kind": "bhkm",
    "max_iter": 10,
    "seed": 42,
    "sid_version": "SID_BHKM_B32D3_S42"
   },
   "S512": {
    "branching": 512,
    "depth": 3,
    "expected": {
     "items_needing_suffix": 0,
     "leaves_count": 120415,
     "leaves_max": 1,
     "positions_distinct": [
      512,
      236,
      1,
      1
     ],
     "vocab_size": 1539
    },
    "kind": "bhkm",
    "max_iter": 10,
    "seed": 42,
    "sid_version": "SID_BHKM_B512D3_S42"
   },
   "S64": {
    "branching": 64,
    "depth": 3,
    "expected": {
     "items_needing_suffix": 0,
     "leaves_count": 120415,
     "leaves_max": 1,
     "positions_distinct": [
      64,
      64,
      30,
      1
     ],
     "vocab_size": 195
    },
    "kind": "bhkm",
    "max_iter": 10,
    "seed": 42,
    "sid_version": "SID_BHKM_B64D3_S42"
   }
  }
 },
 "g_runs": {
  "S1024": {
   "configs": [
    "configs/gtok02/g-tok2-bhkm-b1024d3-s42.json",
    "configs/gtok02/g-tok2-bhkm-b1024d3-s2024.json",
    "configs/gtok02/g-tok2-bhkm-b1024d3-s2025.json"
   ],
   "sid": "SID_BHKM_B1024D3_S42"
  },
  "S128": {
   "configs": [
    "configs/gtok02/g-tok2-bhkm-b128d3-s42.json",
    "configs/gtok02/g-tok2-bhkm-b128d3-s2024.json",
    "configs/gtok02/g-tok2-bhkm-b128d3-s2025.json"
   ],
   "sid": "SID_BHKM_B128D3_S42"
  },
  "S256D2": {
   "configs": [
    "configs/gtok02/g-tok2-bhkm-b256d2-s42.json",
    "configs/gtok02/g-tok2-bhkm-b256d2-s2024.json",
    "configs/gtok02/g-tok2-bhkm-b256d2-s2025.json"
   ],
   "sid": "SID_BHKM_B256D2_S42"
  },
  "S32": {
   "configs": [
    "configs/gtok02/g-tok2-bhkm-b32d3-s42.json",
    "configs/gtok02/g-tok2-bhkm-b32d3-s2024.json",
    "configs/gtok02/g-tok2-bhkm-b32d3-s2025.json"
   ],
   "sid": "SID_BHKM_B32D3_S42"
  },
  "S512": {
   "configs": [
    "configs/gtok02/g-tok2-bhkm-b512d3-s42.json",
    "configs/gtok02/g-tok2-bhkm-b512d3-s2024.json",
    "configs/gtok02/g-tok2-bhkm-b512d3-s2025.json"
   ],
   "sid": "SID_BHKM_B512D3_S42"
  },
  "S64": {
   "configs": [
    "configs/gtok02/g-tok2-bhkm-b64d3-s42.json",
    "configs/gtok02/g-tok2-bhkm-b64d3-s2024.json",
    "configs/gtok02/g-tok2-bhkm-b64d3-s2025.json"
   ],
   "sid": "SID_BHKM_B64D3_S42"
  }
 },
 "gates": {
  "b256d2_balance_basis": "raw3",
  "balance_tolerance": 1,
  "groups": {
   "B": [
    "B1"
   ],
   "H": [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6"
   ]
  },
  "not_applicable": [
   "R1",
   "R2",
   "R3",
   "R4"
  ]
 },
 "judgement": {
  "near_hits": 10,
  "reference": "T3 on the same seed",
  "rule_e": "B256D2 gets seeds 2024 and 2025 whenever its seed-42 run is verified",
  "seeds": [
   42,
   2024,
   2025
  ],
  "threshold_hits": 6,
  "top_k": 2,
  "validation_queries": 2520
 },
 "schema": "g-tok-02-prereg-v1",
 "task_id": "G-TOK-02"
}
```
<!-- gtok02-prereg:end -->
### 11. 结果
本文件不含任何结果。所有指标、run 状态与排名只写在项目的 `experiment_log`；这里不复制数字，以免两处漂移。
### 12. 冻结与部署记录
冻结时间、config 的 sha256 与机器可读冻结块记录在冻结实验卡的附录里；代码 commit 在部署时以追加一行的方式记录，不修改冻结块。
