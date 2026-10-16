密级：公开

# 排序导向温度交叉熵：LETTER 出处核对

2026-10-17，Codex；研究收据，不是冻结卡。R1 暂停真实训练，待 Kiro 裁决。

## 结论

官方代码存在唯一声明默认 **τ=1.0**。按该值执行 `CE(logits / τ, target)` 与原 CE 相同，无法形成预期的新训练目标。论文没有给出唯一的非 1 默认值。不能把扫描图上的某个点自选成默认、不能扫描或按本项目结果选 τ，不能据此启动三个冗余训练。

## Primary sources 与固定版本

- 论文：[arXiv:2405.07314v3](https://arxiv.org/html/2405.07314v3)，版本日期 2025-10-17。
- 官方仓库归属：论文 Introduction 最后一段直接链接 [HonghuiBao2000/LETTER](https://github.com/HonghuiBao2000/LETTER)，README 反向引用同一论文。
- 审计固定 commit：`8d0154e28de37dbb6e24871c508ad8ddb1921cda`，2026-10-17 通过 `git ls-remote HEAD` 与浅 clone 定点核对。不执行第三方代码，不复制实现。

## 定义与数值

| 对象 | 精确位置 | 核对结果 |
|---|---|---|
| 论文定义 | §3.2.1 Eq.(6) | 对每个生成 token，将 logits 除以 τ 后 softmax，再取 target 的负 log probability；对应 `CE(logits / τ, target)` |
| 论文数值讨论 | §4.3.5 Temperature τ、Fig.8 | 分析 0.6–1.2，多值讨论，没有唯一非 1 默认；趋势文字与随后解释有不一致，不能据此补选单值 |
| TIGER CLI 默认 | [LETTER-TIGER/utils.py:75](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-TIGER/utils.py#L75) | 1.0 |
| TIGER 训练脚本 | [LETTER-TIGER/run_train.sh:5,15](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-TIGER/run_train.sh#L15) | Instruments，显式 1.0 |
| TIGER loss | [LETTER-TIGER/modeling_letter.py:31–43](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-TIGER/modeling_letter.py#L31) | 构造默认 1.0；loss 内除温度后 CE |
| LC-Rec CLI 默认 | [LETTER-LC-Rec/utils.py:88](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-LC-Rec/utils.py#L88) | 1.0 |
| LC-Rec 训练脚本 | [LETTER-LC-Rec/run_train.sh:5,23](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-LC-Rec/run_train.sh#L23) | Instruments，显式 1.0 |
| LC-Rec loss | [LETTER-LC-Rec/modeling_letter.py:29–39](https://github.com/HonghuiBao2000/LETTER/blob/8d0154e28de37dbb6e24871c508ad8ddb1921cda/LETTER-LC-Rec/modeling_letter.py#L29) | 构造默认 1.0；loss 内除温度后 CE |

训练参数传递在 `LETTER-TIGER/finetune.py:60` 与 `LETTER-LC-Rec/lora_finetune.py:69`，均调用 `set_hyper(args.temperature)`。对两实现的 Python、shell、README 定点搜索未发现其它训练温度默认；仓库仅有上述两份 `run_train.sh`，没有 Beauty/Yelp 单独训练脚本。此结论限于上述固定 commit，不宣称论文作者实际实验全部使用默认 1.0。

## Provenance / license

`git ls-tree -r --name-only HEAD` 未发现 LICENSE、COPYING 或 NOTICE；README 未声明代码许可。公开可读不等于授权复制。本项目仅依据论文定义独立实现，不引入该仓库代码或资产。

## 下一步

由 Kiro 裁决是否指定一个有记录理由的非 1 工程值，以及相应更新冻结卡。该选择不由本实现者推断。R2 的独立工程预演可以继续，不能将随机权重吞吐写成新 SFT best 的实际吞吐。
