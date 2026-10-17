# 用户画像特征（F7）独立泄漏 code review

密级：公开。结论：**APPROVED**。HIGH：0；MEDIUM：0。

本报告由独立短期 reviewer 在 fresh context 下完成；未修改实现、连接服务器、push 或入队。context 计数 UNKNOWN。审查对象为 `codex/f7-user-profiles` 分支、`<WORKTREES>/user-profile-features/vk-generative-retrieval` 中固定 HEAD `27344539686953a5cea3fd8f747d59c633849c40`，相对原 M1 producer `0ca61bc35ad68bd79b86103b54e470445cbab07f` 的全部 21 个新增文件；`git diff --name-status` 全部为 A，旧入口无改动。

## 依据与范围

已核主工作区 AGENTS、当前 STATE/project_map/handoff/schedule、decision_log 晚间 F7 第 1–6 项及名额补记、FINAL 卡 `docs/superpowers/specs/2026-10-17-user-profile-draft.md`，并读取既有 like 审计接受收据。重点审查 model、prepare、run、ops、metadata transport 与其复用的 M1/FEAT-02/数据合同读取路径。like 审计本身已独立 review，此次复核其输入身份与下游接合。

## 核查结果

- **严格前缀与窗口**：`src/vkgr/profile_features.py:16` 的 producer 按 train week/physical ordinal 遍历；target 身份匹配后先 snapshot，再累计当前曝光/like。U2 不截断；U3 使用同一用户 recent 50 次曝光，逐项校验 history 与 history_positions。第 51 条以前的事件仅改变 U2。validation 在 train 遍历结束后查询固定快照，没有 ingest validation 的路径。
- **train-only 拟合与侧车**：`src/vkgr/profile_features.py:54` 仅从 train snapshot 的正比例拟合；0、undefined 分别落 0/9。linear quantile、边界去重与 bisect_left 为固定实现，未见基于结果改法或改阈值。`src/vkgr/retrieval_profile_prepare.py:51` 核验原审计/输入/history/side/raw pins，并对零缺失原始 like 与既有 side bit0 逐样本对拍；保存 fit input、脚本与产物 hash。`load_profiles`/`attach` 核验 producer 字节与 sidecar ordinal/identity，错序 fail closed。
- **实际模型路径**：`src/vkgr/retrieval_profile.py:6` 先构建原 M1-mean，再用不消耗 RNG 的零表构建 U2 两张 17×64 或 U3 一张 10×64 表；`users()` 实际返回原 `m+g*r` 加 p，p 位于 gate 外。不存在 U1、user_id embedding、MLP 或交叉项。共享参数、RNG 与初始化 forward 的逐位等价有测试；训练轨迹等价没有被当成要求。
- **训练、导出与 verify**：`src/vkgr/retrieval_profile_run.py:113` 的 train_step 与导出均使用新 tensors/model.users 路径；对照原 M1 runner，采样、loss、优化器、早停与验证选择保持原冻结条件。binding 包含 source/init/config/数据/side/profile manifest/fit/实际 profile ids；verify 重放 draw/selection，并对自己新 checkpoint 精确 re-export。g=0 的诊断明确保留 m+p，没有冒充旧 M1 对照。
- **读权限与输入合同**：正式入口保留 validation-only loader，prepared SHA 与验证 query SHA 固定；数据合同从显式 readonly data_root 读取，原合同文件逐 pin 核验。raw producer 只打开精确 train 00–24 pins 的 user_id/item_id/like 三列。`src/vkgr/profile_io.py:10` 为正式 prepare/run/ops CLI 安装 runtime open 拦截，覆盖 builtins/io/os open 及解析后的 symlink 路径；测试证明 test/week26 和旧 checkpoint 路径被拒绝。真实流程未出现 test/U1/旧 checkpoint 读取调用。
- **工程闸门与 driver**：`scripts/profile_ops.py:60` 要求真实 engineering=False preflight，核共享初始化及零表/非零表 target invariance；CUDA 使用 synthetic replay、新产物 verify 与真实 shape，VRAM 由新实测按冻结公式推导。plan 固定两臂三 seed 共六训练，各派生一个 verifier，48h 原始截止与 900s 轮询固定。source/config/plan/claimed manifest 绑定、ready/release 前零发布、PPID1、失败锁存与唯一 verify intent 均有实现。科学验收检查原 stdout `verified`、verify/train run_id 及 manifest SHA，不只依赖 exit 0。
- **metadata bridge**：`scripts/profile_transport.py:11` 将 action 限定为 plan/dry/submit，核固定 clean Git、脚本原字节与现有 runtime，清理 Python/动态库注入环境；真实 fit/PF/CUDA 不通过此桥直接执行。

## 独立验证与限制

独立运行：

```text
PYTHONPATH=src:tests OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <EXISTING_VENV>/bin/python -B -m unittest test_profile_io test_profile_ops test_profile_transport test_profile_prepare_run test_retrieval_profile test_profile_like_audit test_seqm1_model -q
Ran 61 tests — OK
```

首次 sandbox 内运行只因 `ps` 被权限拒绝导致 detached 测试 ERROR；同命令获本地提升后全部通过，包括真实 detached fake queue 的 PPID1/ready 前零发布检查。原 `test_seqm1_model` 有 tensor scalar warning，不影响断言。没有修改测试以规避失败。

本结论是固定源码的独立 code review，通过不等于真实工程验收已完成。真实 train-only fit、engineering=False preflight、CUDA 新实测、正式 dry-run、source 固定部署与实际 claimed/no rejected 回执仍须按既定顺序完成，全部通过后才可入队。未读取真实数据、test、checkpoint；未独立复跑提交者所报 mutants，不把其数量作为本 review 的独立证据。不裁决结果、anchor 或转正。
