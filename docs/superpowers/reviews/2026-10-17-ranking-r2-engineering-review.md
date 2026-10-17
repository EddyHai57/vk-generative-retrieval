密级：公开

# 自生成难负样本 DPO：工程工具独立 review

结论：**CHANGES_REQUESTED**。1 项 MEDIUM，0 项 HIGH。范围仅为损失 primitives、train-only loader 和随机模型吞吐入口；不评价尚未实现的完整 SFT/DPO 训练。

审查版本：`fdc988443672dd729a81dde6c0e7ca42a448cd13` → `cc7f14a19919a4b6645cd9f6166b49905383b24e`，5 个新增文件。依据：排序导向实验卡 R2、decision_log「2026-10-17 上午」以及本次工程 microbenchmark 计划。未修改代码，未访问服务器、真实数据或 checkpoint，未训练。

## 必须修复

### MEDIUM：计时入口没有绑定 A1 的 CUDA runtime profile

位置：`src/vkgr/ranking_r2_microbench.py:94–113`。

CLI 检查 CUDA、设置线程和随机 seed 后直接构建模型并计时，没有调用正式 G6 复用的 `generative_b1_run._runtime(False)` 或等价配置。后者通过 `generative_real.configure_runtime` 明确启用 deterministic algorithms，并将 CUDA matmul 与 cuDNN 的 TF32 关闭。当前 CLI 因而继承进程默认 backend flags；seed 相同并不能固定这些 flags。收据也没有记录实际 deterministic/TF32 状态，只记录 torch/CUDA 版本与 GPU 名称。

这会让同一 commit/config/input/seed 的工程计时在不同 backend 设置下采用不同计算路径，无法核对它是否对应后续 A1/SFT 挖掘的 runtime；尤其启用 TF32 时，吞吐估算不能直接作为该正式 runtime 的工程成本依据。随机权重的既有免责声明不能替代 runtime 绑定。

建议：在模型构建及 CUDA 计时前配置并验证与 G6 相同的真实 CUDA profile，将返回的实际 runtime snapshot 放入收据。补一个 synthetic/mock CLI 测试，证明进入模型构建前该配置已执行、收据包含实际 flags；测试不需要 CUDA 或真实输入。

## 已核对

- 固定 A1/T3 config 实际 SHA256 与代码常量一致；使用正式 `validate_config(..., engineering=False)` 和 `make_model_arch_d`。原入口/config 未改。
- T3 loader 复用正式 SIDMAP_V2 loader，校验 content、catalog、SID bundle hash 与审计，随后适配 depth 3；新入口检查 B256D3 与 config SID 一致。
- 新 loader 只打开 prepare manifest、catalog、examples、history manifest、train-l50；history overlay 保留完整 `past_likes`。全局顺序、逐用户 like prefix、严格早于 target 的 history 与 hash 校验均有实现。没有 validation/test/checkpoint 加载路径。
- seed0 的 20,000 无放回抽样固定，实际计时只取该序列前缀；beam10、decode b1 固定。负样本按 beam 顺序排除 target 和完整 `past_likes`，无合格候选返回 None。
- DPO β=.1 与正样本 CE 权重 1 的手算/梯度方向正确；reference 独立复制、eval、禁梯度，损失里的 reference 项额外 detach；序列 log probability 按位置求和。
- 随机权重工程结果与新 SFT best 正式挖掘吞吐的边界已明确。超过 4 小时只输出标志，代码不自行改 beam/样本数，不自动冻结或入队。
- 入口没有独立 preflight 模式或进程内资源/墙钟 guard。现计划把准入、只读 preflight、运行成本限制交给 owner 外层流程；本 review 不将外层步骤视为已经完成，真实执行前仍须给出相应收据与有界运行命令。此项不是本轮新增阻断，也不能把 `over_four_hours` 当作运行中的 timeout。

## 验证与限制

使用既有 runtime，`PYTHONPATH=src:tests <RUNTIME_PYTHON> -m unittest tests.test_ranking_r2 tests.test_sid_depth_beam tests.test_generative_g6_unit`：独立运行 **35 tests，全部通过**。

已知 G6-v2 集成测试受基线 P3C 文档 pin 漂移影响的限制沿用实现报告；本次没有改 pin、没有重复声称全 suite 通过。没有 CUDA 实测、真实输入 preflight、显存测量或正式成本准入证据。后续完整训练/verify 仍在本次 review 范围之外。
