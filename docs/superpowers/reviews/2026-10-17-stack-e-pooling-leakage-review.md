密级：公开

# 序列用户塔 + 组合叠加：E pooling 接合点独立泄漏 review

日期：2026-10-17。**Verdict：APPROVED（仅 E pooling 替换 M1 的 m 路径）**。fresh-context 独立只读 reviewer；context 计数 UNKNOWN。本结论不裁决统计有效性、延迟、转正或 anchor。

对象为公开部署 `a7536e55e99bf7d126ae983176f205b19ef30be1`；本地作者 commit `40eb70ebcb81435b12e441701406a56800c4efd2` 与其同树 `f317188f2416f5dca592001b6099aa0c32164277`。依据 decision_log「2026-10-17 午后」第5条，只核新增接合点；C/D/E 组件沿原 APPROVED 与 `2026-10-17-feature-combination-leakage-p1-closure.md`，不重审全卡。实现行号均相对该固定树，源码根记为 `<STACK>/`，主仓根为 `<REPO>/`。

## 接合点核查

| 要核的边界 | 固定源码证据与结论 |
|---|---|
| users / parts 使用同一 m | `src/vkgr/retrieval_seqm1.py:162` 的 parts 在166行调用 `self.anchor_vector`；169–175行的 users 同样动态调用并 combine。继承不会绕回旧均值 m。`retrieval_seq_combo_stack.py:11–19` 是唯一新 pooling 路径。 |
| m 不读 target / 候选 / 未来字段 | `retrieval_feat02_run.py:118–143` 仅将 history ID、该事件自己的 side 前三列与 history_positions 的周列装入张量；stack 以同一 feature 使用此入口。m 只从这些 history 张量构造 e、历史内 position/week gap 与自身 feedback 权重，不读 query target、target_position、past_likes、oov 或候选列。周差相对最后一个有效历史事件计算（`retrieval_feat02.py:61–64`）。 |
| 候选统计不回流到历史事件 | `retrieval_seq_combo_stack.py:21–24` 将四表限定在 candidate_items；`retrieval_seqm1.py:133–140` 的 event_vectors 调 `self.items`，不是 candidate_items。m 和 branch 共用这些未加权 F3/F4 事件；不会把 C/D candidate 向量用作历史 e。 |
| branch 隔离 | `retrieval_seqm1.py:143–165` 的 branch 和 readout 不调用 E 权重。feedback 仅在新 m 的18行读取；Q2/β不进入 Transformer，原 causal mask/mean readout 保持。参数通过共享 item/F3/F4 表共同训练属于既有模型结构，不等于读取目标字段。 |
| train / export / verify 一致 | `retrieval_seq_combo_stack_run.py:325–326` 使用 combo train_step；`retrieval_feat04combo_run.py:87–94` 分开构造 history 张量和监督 candidates，再调用同一 model.users。108–123行的 export 同样调用 model.users；stack evaluate 在123–125行委托相同 E export；verify 在243–245行重建同一模型核 binding，在295–303行加载本作业 best 并 exact re-export。后者仅静态检查，reviewer 没有读取任何 checkpoint。 |

## 真实 engineering=False 原回执

直接读取 `<REPO>/cache/pgr01-p-recovery-v1/main05/stack-preflight-event-result-20261016T203830Z.stdout`，本轮重算 SHA256 为 `6d0b83c298b3b1210efd3c4437bf9bd01e0c30c42f9a03c5870f34b5ce9995d6`，与 experiment_log「自身真实路径检查通过」一致。confirm read UTC `2026-10-16T20:38:34.883408+00:00`。

- job `p-seq-combo-stack-real-preflight-v1`，job SHA `0dd75d12dd2d101ec16464133c25205941aba631f0daa93d5befcd73e38b8c63`，唯一 claimed、无 pending/rejected；terminal succeeded/exit0。实际验收根据内嵌逐项回执及身份绑定，不只看 exit0。
- 内嵌 stdout 按 sorted JSON + newline 重建，其 SHA `23c1ee90ad6621c22840e7fda31e78893c86f7cd7212329e546aa54cef5ce2fe` 与原收据相等。
- 62个 source.files SHA 全部与公开部署 Git blob、受审本地源码逐一重算相等（62/62）。source.git_commit 等于上述部署 SHA，git_dirty=false。三个 config 文件 SHA 全部重算相等；各 leakage_receipt.source、data_hashes 与对应 entry 相同，receipt.binding_sha256 与 entry.expected_binding_sha256 相同。外层 CPU preflight binding 与 CUDA expected binding 不同属显式设备绑定，不混为同一值（runner:419–430）。
- 三 seed 42/2024/2025 均 status=preflight_ok、engineering=false。每个模型检查512条 train 与全部2520条 validation；gate=0.20000000298023224，recency 与 β 非零（auditor:41–47），避免零初始化掩盖新路径。

三 seed 的各项计数相同，以下是每 seed 原始计数：

| 项 | status / 样本数 | 接合点含义 |
|---|---|---|
| L1 | pass /3032 | 翻 target 字段，输入张量与 users 逐位不变 |
| L2 | pass /3032（512 train +2520 validation） | target 翻转后 user/item exports 与 norms 逐位不变 |
| L3 | pass /3032 | 改历史物品 users 改变，恢复后逐位还原 |
| L3_own_feedback | pass /128 | 改自身历史反馈 users 改变，同时 parts.r 逐位不变 |
| L4 | pass /2791 | 合法历史整体周平移后 users 不变；其余行不适用 |
| L5 | pass /120415 | catalog 候选反序，按 ID 对齐后相同 |
| L8 | pass /3032 | 每行末个 history 周改25；按128行块调用并应拒绝。计数是被置坏的行数，不能称3032次独立拒绝调用 |
| L9 | pass /120415；train 512 | 全 catalog raw/normalized/norm 与 export 一致，训练候选列按 ID 逐位相同 |
| L6/L7/L10 | STATIC_REVIEWED | 静态项；不声称 runtime。L7 的 anchor_vector source SHA 本轮重算为 `eabea405f2066fea2276f317592f360c32f985d287a9cbbe942165b1d9684a18`，与收据相等；本轮直接审了该接合点，不仅依赖旧 review 总体状态 |

真实路径 auditor 在 `seq_combo_stack_leakage.py:25–37,51–107` 实现上述采样/检查/编号映射。输入 pin `08ed500966e506a2599beb2bb2807330165b82f13ec2b8df01286bdd769493d6`，side manifest pin `3f4ee94a0cf196d7671743dabc8c8c6b81c0dfbdf64ffbcd3054212cd3d292ee`，stats manifest pin `eb41bc060c67549a0cb7bf96c627000bb90b73cb5111ca11b21bf572573c036b`。这些输入与数据 digest 是已有回执绑定；本轮没有重读真实数据重新生成 digest。

## 本轮本地验证与限制

在现有 `<WORKTREES>/p6-tiger-smoke/.../.venv/bin/python`、`PYTHONPATH=<STACK>/src:<STACK>/tests`、线程1、`PYTHONDONTWRITEBYTECODE=1` 下运行7个既有 synthetic 子测：共享初始化/zero gate、candidate四表隔离、β手算及branch隔离、target/engineering契约、recency手算、runtime helper矩阵、target泄漏变异体。结果 **7 tests OK**；未运行训练/replay/checkpoint子测。首次选择器误写 LeakageTests 导致2个 AttributeError（其余5项通过），改用实际类 LeakageFixTests 后完整7项通过；没有实现修改。

另以 synthetic、非零 E/g 做一次无 optimizer 的直接断言：train/eval 各自固定相同 RNG 后 `users == combine(parts.m,g,parts.r)` 且 `parts.m == anchor_vector`；将candidate四表全部改为常数后 m/r/users 不变；新增 candidate/future_history/target_feedback/target_week 字段后张量及users不变。全部PASS。上述 synthetic 数据不充当真实路径证据。

本轮只新增本文，没有改代码/事实源、连接服务器、读取真实输入/test/checkpoint、训练、重投或push。APPROVED 表示未发现新增接合点的泄漏阻塞，并有自身真实正式路径证据覆盖。既有 history 构造的严格先后与 C/D 统计表构造沿已批准组件的边界；本轮不重审全数据管道。D-1（train表含自身target及其后train行）、23:2x更正与偏差方向未知继续有效；不能宣称训练侧严格时间隔离、分离 C/D 因果贡献或由此自动转正。统计与延迟须另交Kiro/Opus裁决。
