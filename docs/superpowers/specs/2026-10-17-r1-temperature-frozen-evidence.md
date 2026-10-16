密级：公开

# 温度交叉熵：R1 冻结来源与实现边界

2026-10-17，Codex。依据主事实源 `docs/decision_log.md`「排序导向 R1 的温度取值裁决」与 Kiro 已同步的排序导向实验卡 §2。该文件记录来源，不覆盖 Kiro 维护的原卡，不代表工程准入已通过。

- **固定 τ=0.8**，一次性先验值，来自论文扫描中小于 1 区间的中点；没有更强依据，不扫描，不按本项目结果改。
- **训练与推理都用 logits/τ**。新 `TemperatureModel.decode` 只做一次温度缩放；继承的 `forward` 调用同一个 decode，故原 full-vocabulary CE 和每次 beam log-softmax 均自然作用于同一分布。不改原 loss、decoder 或 evaluator 文件。
- **τ=1 不提交真实训练**。仅在 synthetic tests 中证明权重、logits、loss、梯度、AdamW 更新、实际 dropout/microbatch step，以及 serial/batched beam100 结果和原 A1/T3 逐位相等。
- 原 A1/T3 源 commit `fdc988443672dd729a81dde6c0e7ca42a448cd13`；原 config `configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json` SHA256 `9a938cf3ac722e0ad11fd8136d5a037b0f44516cf53e4df85a8188770dca737e`。三个新 config 仅改 schema/task、增加温度，并各用原同 seed 42/2024/2025。其余逐字段沿原，formal validator 逐字段锁定。beam100、batched b1、4+4、LR3e-4、dropout .1、microbatch256、effective batch512、BHKM256×3、L50 与原早停/selection 不变。
- 三 seed 一律跑完，各配 verify；不按 seed42 决定后两个。判定与 anchor 裁决仍指向原卡 §3，Codex 不代 Kiro 裁决。

## 外部出处

[LETTER arXiv v3](https://arxiv.org/html/2405.07314v3) §3.2.1 Eq.(6) 确认训练定义为 logits/τ 后的 token NLL；§4.3.5 分析 0.6–1.2，没有唯一非 1 默认。论文直接链接的[官方仓库](https://github.com/HonghuiBao2000/LETTER)，审计 commit `8d0154e28de37dbb6e24871c508ad8ddb1921cda`，TIGER `utils.py:75`、`run_train.sh:15` 与 LC-Rec `utils.py:88`、`run_train.sh:23` 均默认 1.0。所以 0.8 是 **Kiro 的本项目先验裁决**，不能称 LETTER 的官方默认。完整出处核对见 `docs/superpowers/reviews/2026-10-17-ranking-temperature-source.md`。

官方仓库未发现 LICENSE/COPYING/NOTICE；本实现只按论文定义独立实现，未复制第三方代码。

## 代码 root 与已验收数据通道

旧 G6 不支持 `VKGR_DATA_ROOT`。仅新增 `ranking_r1_data.py` resolver，并在新 R1 runner 内把显式 data root 传给原 `generative_real.load_context`。不改旧 loader、不 patch 全局 ROOT、不改 frozen 文档或 pin。历史输入仍由原 loader 与原 pins 完整验证。

新运行 binding 的 `source` 永远由新代码 checkout 生成，包含其 Git SHA、源码与原/新 config 的 SHA；另记 `data_root` 及 context/原 prepared/history/SID/content 绑定。verify 同样重建并核对两条来源。部署必须是已 push 指定 SHA 的 clean Git worktree，逐文件 sha256 验证，不能把旧 data root 的源码当作新训练源码。

## 验证与尚未完成

本地 synthetic train/eval/save best+last/verify 已验证，context 加载使用显式 synthetic fixture 注入；这不等于真实 loader/preflight 已通过。原 frozen 文档 pin 漂移没有修改；真实只读 PF 必须首次核对指定已验收 G 数据通道。

服务器部署、真实 PF、CUDA/full-shape 预演、显存声明、dry-run、三 seed 训练与 verify 均尚未执行，由主 owner 在一次独立 review 通过后推进。
