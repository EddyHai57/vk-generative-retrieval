# 可学习SID实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在既有S2工作树继续实现已冻结的LETTER式tokenizer，完成真实fit/独立verify后才允许三个A1。

**Architecture:** 复用torch-free输入合同和固定L2核心；新增可微损失、固定PCA、fit/bundle/verify及独立A1入口。运载沿结构匹配已验证的Git部署、队列、真实PF/CUDA/dry/有限驱动做法，科学配置独立，不混用bundle或结果。

**Tech Stack:** Python3.11、现有NumPy/Torch、stdlib运维工具，不加第三方算法实现。

**Spec:** `docs/superpowers/specs/2026-10-17-learnable-coll-sid-draft.md`（含§8整节新L_div裁决）；固定L2原Git提交80f7eed179981c65094c628744099a181a81d907。

## Global Constraints

- 复用codex/s2-learnable-sid@876575e，base a5da4d6。只改该base之后S2自有新增文件；base已有生产入口字节不变。
- CONTENT_V1 64维重建；原C1′固定PCA32 target。PCA仅train-observed行，fit seed42，无可学习头，矩阵SHA/解释方差入manifest。
- 3层×256、latent32、L2初始化/硬量化残差/导出/死码重置沿固定L2。Adagrad lr0.1、accumulator0.1、eps1e-7，固定300000步、batch1024、4h cap，不改R2。
- L=L2_total+0.1L_align+0.01L_div；L_align单向zst→target，双方L2 norm、温度0.07，fallback不作anchor或negative；fallback必须恰251/120415。
- L_div严格按FINAL§8：同一实际硬链残差，softmax(-d/(32×1.0))；q沿batch mean，逐层负熵/logK后层mean，log(q+1e-12)，不重归一化、不EMA、不mask、不detach编码器或当前码本。
- step2000及final记录各层硬利用率，以及实际批d/D均值/p50/p90；仅诊断，不调整τ。
- 不读validation来选择tokenizer；test不读。NaN/Inf/合同不符停止该支，不重跑换ID；三A1须等fit原始verified与job/manifest绑定。
- 不输出完整embedding/PCA targets或用户向量；PCA targets仅内存，持久化授权的主成分/均值及摘要。checkpoint/完整SID只留服务器自己的run，不进Git。
- 所有数值均原先写死、未针对本实验调优；继承第三层近空结构限制，不自动换anchor。

## Review Focus

- fallback同时排除align行/列，但div包含全部物品；手算对比并查梯度。
- div必须用前向q.residuals，不能软链替代；residual仍遵循L2前层embed.detach语义。
- γ0逐位等于原base；uniform与单码端点、K48/K256归一化及编码器/每层码本非零梯度。
- PCA从原C1′/train catalog推导，逐字节replay与源物理pins；未知或自述producer拒绝。
- 原始verified绑定source/runtime/config/job/terminal/claimed与manifest；后封A1配置不能变更fit科学账本。

### Task 1: 冻结配置、PCA与损失核心

**Files:** 更新S2自有`src/vkgr/sid_learn_v1.py`、`tests/test_sid_learn_contract.py`；新增`src/vkgr/sid_learn_loss.py`、`src/vkgr/sid_learn_pca.py`及对应测试；必要的L2 repair reference从固定Git原字节新增，记录来源。

**Interfaces:** `expected_config()/validate_config`严格固定新配方；`fit_pca()/project_pca()`返回冻结mean/components/variance与内存target；`alignment_loss(zst,target,fallback)`及`diversity_loss(q,codebooks)`被fit直接使用。

- [ ] fresh RED→最小GREEN：旧AdamW/20epoch拒绝，Adagrad/300000、PCA32、L2_total及mask字段生效。
- [ ] PCA小样本手算重建、sign固定、序列化SHA与replay，不读真实输入到本机。
- [ ] 按FINAL§8.6六项全部验收；γ0逐位相等、encoder/每层codebook梯度非零不可省。
- [ ] 隔离变异至少sum/mean、logK、符号、q平均轴、detach、残差替换及fallback mask不对称；逐条记录有效kill/等价/无效异常。
- [ ] 相关回归后具体路径commit；不为普通测试修复反复派review。

### Task 2: 真实fit/bundle/verify与A1接续

**Files:** 新增S2 fit/run/bundle/A1模块、配置、版本化运载及窄测试；沿已有代码风格复用算法，不改旧白名单。

**Interfaces:** formal preflight不写结果；fit只返回完成态可用产物；独立verify精确源码/runtime/物理输入/全部artifact及同seed replay；三个A1参数只改SID/schema必要字段。

- [ ] fresh RED→GREEN覆盖错输入pin/producer成员或字节/runtime/fit状态/门禁；无测量显存UNKNOWN，工程预演允许标注保守上限。
- [ ] fit输出封自身新checkpoint与SID、PCA组件；不持久化完整targets；H1–H6/R1–R4原口径，div值不当利用率。
- [ ] 入队前一次fresh-context独立code review；修复按§4A A/B/C分档。
- [ ] 扫描压平push→固定SHA新目录→正式真实输入只读PF→CUDA测峰/dry→唯一fit及verify，之后才后封A1三配置。
- [ ] 三配置真实PF/CUDA/dry→三个无条件seed与唯一自动verify，名额在机≤6，原结构批互不等待、不重置预算/截止。
- [ ] 原始stdout verified+job/manifest绑定齐后记录item hits与固定条件，前缀/热度/结构限制随报告，交Kiro，不自行转正或解释机制。

## 执行安排

既有书面授权已覆盖本计划；本对话inline实现，正式作业前独立review。结构匹配三A1已由有限驱动接续，S2现在只做本地实现准备，不因本计划增加实验名额或改变任何已冻结科学值。
