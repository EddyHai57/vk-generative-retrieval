## G-TOK-01R 修复版 RQ-VAE tokenizer 说明与预注册（定稿）
状态：定稿。本文件只写口径与规则，没有任何指标数字（实验结果只写 `docs/experiment_log.md`）。G-TOK-01 的旧版说明见 `docs/g_tok_01_validation.md`，其门禁与常量一字不改。
### 1. 范围与 claim boundary
- 做的事：GTOK-01 的 T1、T2（RQ-VAE，每层 256 / 16 个码）在 300,000 步拟合后码本塌缩、没过利用率门禁。本任务给 RQ-VAE 拟合加上死码重置、一组预注册的优化器臂和只读诊断，先用 30,000 步探针机械选一个臂，再用同一个臂对 T1r、T2r 做正式拟合；之后训练入口、评估口径与 GTOK-01 相同。
- 这是**方法对齐的 tokenizer 消融**，**不是 TIGER 复现**：输入是官方 64 维内容向量，不是论文的 768 维文本向量；解码器、优化器细节是我们预先固定的选择。
- 单 seed（seed 42）。结果只能说明「在这个设定下」。
- 探针只跑 30,000 步，只能检出早期塌缩；它通过不代表 300,000 步终点一定过门禁。
- 若选中 L2 至 L4 之一，报告必须写明：RQ-VAE 以偏离冻结卡的优化器设置训练。
- 门禁不变：H1 至 H6、R1 至 R4 的定义与阈值（4/5 利用率、R² > 0、残差范数严格递减、300,000 步、7,200 秒、CONTENT_V1 digest）全部沿用，不放宽、不新增。探针专用阈值只用于选臂，不进入任何门禁或 bundle 校验。
- tokenizer 只读 `content.npy` 与 `catalog.json`，各自带 sha256 钉；不读任何 label、query、split 或交互文件。
### 2. 相对冻结卡的偏离清单
1. 新增死码重置（每 100 步检查，500 步未被使用的码重置，270,000 步后停止）。
2. 学习率与优化器可能由 0.4 的 Adagrad 改为 L2、L3 或 L4（Adam）之一，取决于探针。
3. 新增只读诊断（不改训练轨迹，有测试钉死）。
4. 拟合入口是新文件 `sid_tok_run_r`；旧入口不改。
5. 探针长度 30,000 步（比冻结卡里「数千步」的口径长，是为了看 20,000 至 30,000 步的窗口）。
其余（batch、β、编码器与解码器结构、k-means 初始化、seed、门禁）不变。名额不变：T1r、T2r 替换 T1、T2，G-TOK 仍是三个名额；失败的 T1、T2 原始结果保留。
### 3. 选择规则（看结果之前写死）
探针：臂 P0（旧配置，无重置，作对照）与 L1 至 L4，各跑 K=256 与 K=16，共 10 个作业。臂的超参见下面的机器块，不再调。
1. 对照有效性：P0 在两个 K 上都必须完成，并复现旧失败（前三个 1000 步窗口的损失与旧记录相差不超过容差、训练中已用码数逐点不超过旧记录加 1、终点每层不超过上限）。否则 `INVALID_CONTROL`：不选臂，流水线停下，等人决定。
2. L 臂在某个 K 上没有完成（非有限值、超 wall），该臂不合格，也不进 ESCALATE 候选。
3. 某臂合格，当且仅当它在两个 K 上都：(a1) 终点三层都满足 R1（`used × 5 ≥ K × 4`，复用审计里的同一函数）；(b) 终点 R² > 0；(c) 终点逐层平均残差范数严格递减；(d) 第 20,000 至 30,000 步的 6 个记录点训练中模型都满足 R1 判据；(e) 终点 R² 不低于探针专用下限（见机器块）。
4. 在合格臂里按 L1、L2、L3、L4 的固定顺序选第一个，不比数字谁好。两个 K 用同一个臂，所以 T1r 与 T2r 之间唯一的变量仍是 K。
5. 没有合格臂：若某臂在两个 K 上 (b)(c) 成立，且 (a1) 失败的层都满足 (a2)（第 30,000 步严格高于第 20,000 步，并高于 P0 塌缩水平），仅因 (a1)、(d)、(e) 不合格，则 `ESCALATE`；否则 `STOP`（GTOK-01 收口为只有 T3 对照）。
6. ESCALATE 之后只允许一件事：由人在决策记录里写明放行哪一个候选臂（`select --override-arm <arm> --override-source "<记录标题>"`），再按同一流程拟合。不得新增臂、不得改阈值。放行一个探针里没到 80% 的臂，仍可能在正式拟合的门禁上失败。
7. 正式拟合 wall 预检只报告：用所选臂与 P0 的步速比推算拟合耗时，超过预注册上限就停下等人，不改选臂，不是门禁。
已登记的误杀风险：(d) 的 20,000 步起点与 (e) 的下限都是预注册的粗选，可能误杀爬升更慢的臂；这个取舍写死，不事后调整。诊断判据（梯度、更新量、重建一致性、归一化检查）不参与选臂。
### 4. 命令（占位符）
```
python -m vkgr.sid_tok_run_r probe --arm p0|l1|l2|l3|l4 --variant T1|T2 --code-commit <COMMIT> \
    --config <CONFIG> --config-sha <SHA> --content <CONTENT> --content-sha <SHA> \
    --catalog <CATALOG> --catalog-sha <SHA> --output <PROBE_OUT> [--device cuda]
python -m vkgr.sid_tok_run_r select --probe <PROBE_JSON> --probe-sha <SHA> (共 10 组) \
    --config <CONFIG> --config-sha <SHA> --output <DECISION_OUT> [--override-arm <ARM> --override-source "<TITLE>"]
python -m vkgr.sid_tok_run_r fit --variant T1|T2 --decision <DECISION_JSON> --decision-sha <SHA> \
    --config <CONFIG> --config-sha <SHA> --content <CONTENT> --content-sha <SHA> \
    --catalog <CATALOG> --catalog-sha <SHA> --output <OUT> [--device cuda]
python -m vkgr.sid_tok_run_r verify --bundle <OUT>/run_manifest.json --sha <SHA> \
    --decision <DECISION_JSON> --decision-sha <SHA> --config <CONFIG> --config-sha <SHA> \
    --content <CONTENT> --content-sha <SHA> --catalog <CATALOG> --catalog-sha <SHA> [--replay-prefix]
```
一致性测试（在仓库根目录，源码目录加入搜索路径）：`cd <WORKTREE> && PYTHONPATH=src python -m unittest tests.test_gtok01r_prereg`。输出目录必须是不存在的新目录；退出码 0 成功，1 参数或校验错误，3 门禁失败或 STOP / ESCALATE 决定。
### 5. 机器可读预注册块
下面一段 JSON 是预注册常量的第三份副本。`tests/test_gtok01r_prereg.py` 逐字段核对四处：`sid_rqvae_r.PREREG_R` 与 `PREREG_R_SELECT`、`configs/gtok01r/g-tok-01r.json` 的 `prereg_r` 与 `select_r`、本块、测试里的独立字面量；任何一处的任何一个叶子、键或类型不同都会失败。
<!-- gtok01r-prereg:begin -->
```json
{
  "prereg_r": {
    "arm_order": [
      "l1",
      "l2",
      "l3",
      "l4"
    ],
    "arms": {
      "l1": {
        "dead_window": 500,
        "eps": 1e-07,
        "initial_accumulator_value": 0.1,
        "lr": 0.4,
        "optimizer": "adagrad",
        "reset": true,
        "reset_check_every": 100,
        "reset_until": 270000,
        "rng_reset_offset": 7919
      },
      "l2": {
        "dead_window": 500,
        "eps": 1e-07,
        "initial_accumulator_value": 0.1,
        "lr": 0.1,
        "optimizer": "adagrad",
        "reset": true,
        "reset_check_every": 100,
        "reset_until": 270000,
        "rng_reset_offset": 7919
      },
      "l3": {
        "dead_window": 500,
        "eps": 1e-07,
        "initial_accumulator_value": 0.1,
        "lr": 0.025,
        "optimizer": "adagrad",
        "reset": true,
        "reset_check_every": 100,
        "reset_until": 270000,
        "rng_reset_offset": 7919
      },
      "l4": {
        "betas": [
          0.9,
          0.999
        ],
        "dead_window": 500,
        "eps": 1e-08,
        "lr": 0.001,
        "optimizer": "adam",
        "reset": true,
        "reset_check_every": 100,
        "reset_until": 270000,
        "rng_reset_offset": 7919,
        "weight_decay": 0.0
      },
      "p0": {
        "eps": 1e-07,
        "initial_accumulator_value": 0.1,
        "lr": 0.4,
        "optimizer": "adagrad",
        "reset": false
      }
    },
    "diag_every_after": 1000,
    "diag_fine_steps": [
      0,
      1,
      2,
      3,
      5,
      10,
      20,
      50,
      100,
      200,
      300,
      500,
      750,
      1000,
      1250,
      1500,
      1750,
      2000,
      2500,
      3000,
      4000,
      5000
    ],
    "diag_subset_rows": 8192,
    "diag_subset_seed": 42,
    "probe_steps": 30000,
    "probe_wall_cap_seconds": 3000,
    "reset": {
      "dead_window": 500,
      "reset_check_every": 100,
      "reset_until": 270000,
      "rng_reset_offset": 7919
    },
    "usage_diag_every": 2000
  },
  "select_r": {
    "a2_from_step": 20000,
    "a2_to_step": 30000,
    "d_points": [
      20000,
      22000,
      24000,
      26000,
      28000,
      30000
    ],
    "e_r2_min": 0.05,
    "i1_dead_frac": 0.99,
    "i1_max_step": 3000,
    "i1_usage_max": 3,
    "i1_z_std_min": 0.001,
    "i2_rel_tol": 0.001,
    "i2_step": 30000,
    "i4_norm_range": [
      0.99,
      1.01
    ],
    "old_fit_wall_seconds": {
      "T1": 4354,
      "T2": 4302
    },
    "p0_final_usage_max": 3,
    "p0_loss_tol": 0.001,
    "p0_loss_windows": {
      "T1": [
        0.8199,
        1.01,
        0.8115
      ],
      "T2": [
        0.8754,
        0.8116,
        0.8115
      ]
    },
    "p0_usage_ref": [
      1,
      1,
      2
    ],
    "p0_usage_slack": 1,
    "preflight_limit_seconds": 6000,
    "usage_log_steps": [
      2000,
      4000,
      6000,
      8000,
      10000,
      12000,
      14000,
      16000,
      18000,
      20000,
      22000,
      24000,
      26000,
      28000,
      30000
    ],
    "variant_K": {
      "T1": 256,
      "T2": 16
    }
  }
}
```
<!-- gtok01r-prereg:end -->
### 6. 结果
本文件不含任何结果数字。探针、决策和拟合的指标只在 `docs/experiment_log.md` 维护。
