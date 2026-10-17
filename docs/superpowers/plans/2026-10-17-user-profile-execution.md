# 用户画像特征（F7）Execution Plan

> **For agentic workers:** 执行者用 superpowers:executing-plans 按检查项推进；正式批次前一次独立泄漏review。不得把本文实施选择当作新增科学设计。

**Goal:** 按FINAL卡执行活跃度和H50 like率两臂，六训练/六verify，结果只交Kiro。
**Architecture:** 原M1-mean源码逐位保留；新增profile特征构建/模型/runner/ops，固定原inputs/历史side合同。先只读like审计，合同失效停止，不提前实现依赖数据合同的步骤。
**Tech Stack:** 既有Python/PyArrow/PyTorch/NumPy runtime，不安装依赖。
**Spec:** docs/superpowers/specs/2026-10-17-user-profile-draft.md（FINAL，逐字Kiro裁决）。

## 全局约束与身份

- feature worktree：`<WORKTREES>/user-profile-features/vk-generative-retrieval`；branch codex/f7-user-profiles，base与原三M1 producer相同0ca61bc35ad68bd79b86103b54e470445cbab07f。MAIN事实源由主owner维护。
- 原M1输入p-feat02-inputs-base-v1，SHA a12b98003aa2fbb58df83bc70bb448e426d01c2a4cc2e6357bf1f95778f23e38；历史/side pin均从该inputs读，不猜路径。原prepared数据、合同、入口不改。
- 不读test/U1/既有checkpoint。沿Eddy既有澄清，新训练/CUDA/verify仅可验证自己新产物，不能加载旧M1对照checkpoint。
- 不增臂/seed；原批准两臂×42/2024/2025。所有训练正式提交前必须既有批准记录、真实PF/CUDA实测/dry；不按结果加重跑。

## 任务与文件边界

- [x] 冻结：MAIN原草稿照抄裁决转FINAL、填三对照原manifest/config SHA、同步事实源并commit。
- [x] like只读审计：新增src/vkgr/profile_like_audit.py、tests/test_profile_like_audit.py；固定00–24周like列/原side bit0，先metadata preflight，后1核只读审计。无字段/需改合同、非法domain即STOP；记录缺失率/域/文件SHA/实际UTC，更新事实源并commit。
- [x] 核心特征与模型：新增src/vkgr/retrieval_profile.py及tests/test_retrieval_profile.py。严格target前snapshot、U2累计不截断/U3同H50；侧车绑定example_id/query完整顺序；train-only拟合正like率分位数。M1共享参数先按同seed构造，然后零表，init forward逐位相等而不要求训练轨迹相等。
- [x] 新入口与数据构建：新增src/vkgr/retrieval_profile_prepare.py、src/vkgr/retrieval_profile_run.py及测试；旧runner训练条件/selection/采样/logQ/导出verify原样继承，唯一用户向量增量p与必要schema/binding。加载新sidecar失败即拒绝，不回退旧特征。
- [x] 六配置+版本化ops：新增configs/profile/*、scripts/profile_ops.py及tests，运载/启动metadata先测包络；job/plan/driver全部固定SHA，不重置原48h截止。driver PPID1、900秒、不调用模型，终态逐run机器可读。新verify原stdout附run_id+manifest_sha256。
- [x] 一次fresh-context独立泄漏code review：覆盖用户全部测试清单、真实路径可执行审计与数据/初始化同源，HIGH/MEDIUM设计偏离回Kiro；普通工程修复RED/GREEN后不重复全review。
- [x] 公开扫描→push工作分支→全新固定SHA部署逐文件SHA。先train-only分位数fit/verify，再六配置engineering=False只读PF，再CUDA fullshape峰值×1.15 ceil256，最后dry。每步事实源/commit。
- [x] 入队前写STATE时间并告知Eddy；一次性六train已发布，实际worker/唯一driver PPID1与四claimed/no rejected已核，余两pending不冒称claimed；原scheduler/driver接续，无重提。
- [x] 余两pending已按原准入claimed；三份自动verify intent/job/manifest绑定已核。
- [x] 六训练/六原始verified、冻结plan/实际queue/manifest绑定全部验收，原driver最终verified后退出。
- [x] 逐query exact/p_pool/bootstrap与H50未满/满Δhits已验收，两臂冻结表第4行，非等价/非转正裁决。
- [x] 用户塔model.users()限定口径原报告、raw均值/分位数/配对SHA与物理report全部验收；完整Kiro/Opus提示词已交付。两臂冻结表第4行，不自行转正/换anchor。

## 必须证明的测试/停止点

手算prefix含重复和零/缺失；target/未来扰动；窗口外事件只改变U2；validation扰动不改profile或quantile；zero与undefined分离；sidecar错位拒绝；runtime test读拦截；共享初始化输出逐位一致；有意义变异全杀；真实PF同源完整逐项回执。所有数据合同/顺序证据不足停止，不以修fixture或默认值绕过。完整suite若有旧fixture失败，按原issue保持OPEN，不报全绿。

实施记录：训练分位数使用numpy.quantile(method=linear)、7内部正比例边界排序去重、bisect_left落桶；这是看真实拟合数值之前固定的实现细节，计入script/manifest，独立review未认定设计偏离。数据合同root沿既有P入口/workspace/vkgr-b1-p只读通道，原数据SHA不变，不向部署树复制数据。
