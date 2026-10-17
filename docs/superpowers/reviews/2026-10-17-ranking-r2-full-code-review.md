# 自生成难负样本偏好训练（R2）完整实现独立 review

结论：**APPROVED（代码 review）**。未发现需要阻止该实现继续工程准入的 HIGH / MEDIUM 问题。对象 `277a1f14f33a82c0dd1eecf5330b4bb6ebaa0715` 相对 `974c6751f49a085680f4cbe75862bbf86b7d9f4e` 新增的 16 个文件；本次完整审查独立于先前 microbenchmark review，未参与 R2 实现或修复。context 计数 UNKNOWN。

本结论不替代新完整 runner 的服务器真实 `engineering=False` preflight、CUDA/ref/optimizer 实测、显存声明与 scheduler dry-run，也不代表已部署、已入队或科学结果 verified。上述真实回执当前尚未执行。

## 合同与科学路径

依据：该实现的 `2026-10-17-r2-full-ops-handoff.md`、`2026-10-17-r2-dpo-frozen-evidence.md`，主仓实验卡 §2–4 与 decision_log 的 R2 两项补定。worktree 旧 AGENTS 中被取代的 owner/workflow 条款按本次直接任务及主仓当前规则处理，未启动旧工作流。

- `ranking_r2_config.py:16–45` 和三份配置锁定 42/2024/2025、原 A1/T3 SFT、普通 CE/raw logits、固定 seed0 的 20,000 样本、beam10/b1，以及 β=.1、CE 权重1、LR3e-5、min1/max5/pat2。`ranking_r2_sft.py:165–245` 沿用原训练数学与选择；与原科学 commit `fdc988443672dd729a81dde6c0e7ca42a448cd13` 对比，所依赖的原 G6 runner/model、batched decoder 和 history 模块未变化。depth3 使用原四位置 evaluator 与该 codec 兼容，不含 R1 温度或全局 decoder 替换。
- `ranking_r2_run.py:93–115` 先训练并直接 verify 自己的 SFT，再读自己的 best；`ranking_r2_engine.py:21–76` 用固定训练索引挖掘，`ranking_r2.py:49–62` 排除 target 与完整 `past_likes`、取最高合法项。history overlay 保留原 row 的 `past_likes`；无负例记 null、跳过、不补抽，全部跳过 STOP。未用 validation/test 挖掘。
- `ranking_r2_engine.py:79–127` 的 actor/reference 独立、reference eval/frozen/no-grad/不进 optimizer，全部 SID 位置 logprob 求和，chosen 原 CE 取 mean，按 pair 数累积 effective512/micro256；clip/AdamW 沿合同。`ranking_r2_run.py:127–175` 仅把 DPO epoch1–5 交 EarlyStop，顺序为 R@50、N@50、较早 epoch；epoch0 只报告同 seed control delta。

## 数据、checkpoint 与 verify

- `ranking_r2_data.py:11–21`、`ranking_r2_sft.py:138–162` 把新代码 root 与 accepted data root 分离，复用正式 loader 和既有 data/history/SID 校验；新 source、配置、runtime/context/control metadata 均绑定。
- `ranking_r2_run.py:24–34` 只读固定 SHA 的原 control manifest 指标 metadata，不打开原 checkpoint。SFT/DPO 的 `torch.load(..., weights_only=True)` 只用于本新 run 的 best；最终 verifier 在进入 SFT checkpoint 验证前拒绝跨 seed/config（`:196–201`）。stage 路径固定，绑定自己的 SFT manifest/best、mining、reference 和最终 best/last。
- `ranking_r2_run.py:178–259` 重验 SFT best、重建 mining sample/row/合法最高 negative/skip/source/hash 绑定，重放 DPO selection、核 best+last，并对最终 DPO best 重新解码。epoch0 的 published evaluation 与自己 SFT best digest/CP 绑定，control 比较不作门槛。
- **明确的 verify 限制**：不重新完整解码 20,000 条 mining；只验证已绑定 beam 记录与最高合法项，不能独立证明每条记录就是 SFT 模型重新生成的排序。生成路径固定由已 verified 的自己 SFT best 执行，source/mining hash 绑定完整，且该限制已在冻结证据与 verify 输出明示；在本次 review 范围内接受此覆盖边界，不把它描述为完整 mining 重放。

## 工程准入与生命周期

- `scripts/ranking_r2_ops.py:108–231` 校验部署 SHA/逐文件 hash，要求完整 scientific source members；三 seed job 从同 seed 原模板推导，保留输入 pins、固定新 data/decode 环境和 control metadata。dry-run 对当前 config/job 与 preflight/CUDA source/data/runtime/model/config 绑定逐项核对（`:303–329`）。
- CUDA preview（`:234–274`）使用真实模型/SID shape、H50、effective512，先两步 SFT，再独立 reference 加 actor optimizer 两步 DPO（chosen/rejected 两倍序列），检查 reference 不变并做 beam100/b1 reload 比较。显存同时记录本 PID 的 context baseline、逐阶段 process 总量与 allocator 峰值；保守取总峰值乘1.15上取整256MiB（`:65–106`）。未沿用旧6656声明；真实 GPU 证据仍须实际运行。
- 提交（`:332–392`、`:440–528`）先 new-only seal plan/request，再由 detached、实际 PPID1/独立 session、持锁 driver ready；controller 只写绑定 ready/plan 的 release，零 publication。driver 唯一发布三 intent、保持原≤48h绝对截止与900秒轮询。
- 发布部分失败时记录 ALERT、不重试剩余 intent；已可见 job 核完整身份并继续唯一 verify（`:348–437`）。验收要求 claimed job/manifest 绑定、verify terminal succeeded 及原始 stdout 的 `status=verified`，没有把 exit0 单独当科学验收。训练/verify failure、rejected 和 deadline 不会触发改 run_id、重投或 kill 原 job。

## 本地验证

使用项目已有 Python 环境，`PYTHONPATH=src:tests`，OMP/MKL/OPENBLAS threads=1：

```sh
<RUNTIME_PYTHON> -m unittest \
 tests.test_ranking_r2_full tests.test_ranking_r2_ops \
 tests.test_ranking_r2_review_guards tests.test_ranking_r2_submission \
 tests.test_ranking_r2 tests.test_sid_depth_beam tests.test_generative_g6_unit
```

实测 **60 tests PASS**，5.202秒；包括完整 synthetic SFT→mining→DPO→verify、CP/跨 seed/配置/mining 篡改、reference/optimizer 隔离、显存 proof，以及本地 detached RPC kill/部分 publication failure/唯一 verify 生命周期。系统 Python 的首次尝试因缺 torch/numpy 失败（并导致两个依赖该环境的 lifecycle 超时），切至已有环境后同组测试全部通过；没有安装依赖或改代码。

未联网、未连接服务器、未读旧 checkpoint/test 数据，未运行真实 preflight/CUDA/训练或提交正式作业；源码与 commit 未修改。本报告只记录一次完整 code review，后续正常工程修复按 fresh RED → GREEN → 回归验证收口。
