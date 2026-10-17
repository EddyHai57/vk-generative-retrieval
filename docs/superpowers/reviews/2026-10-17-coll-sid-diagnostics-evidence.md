密级：公开

# 协同 SID 的结构与分档诊断：证据入口

授权见 `docs/decision_log.md`「2026-10-17 午后：无人值守授权」的诊断收口项；定义沿 `docs/superpowers/specs/2026-10-16-sid-fix-pipeline-card.md` §2 与 S1 FINAL 卡 `docs/superpowers/specs/2026-10-16-sid-fix-s1-freeze-draft.md` §7。本页只登记方法、来源与限制；结果数字统一由唯一执行者维护在 `docs/experiment_log.md` 的协同 SID 诊断收口节。

## 方法与来源

共现向量 SID（C1′）及内容加 P 向量 SID（C2）的 A、D、D′ 来自各自原 sid_verify proof 指向的 mapping aggregate；B、C 与相对 T3 同 seed 的 C 差值来自各原 verify receipt 指向的 prediction aggregate。真实有限 reader 已核验八份小 JSON 的路径角色、原 proof SHA、文件大小及 schema；本地复核再次将来源 pin 与原 receipt / controller summary 绑定。reader 源码与 synthetic 验证见 `analysis/coll_sid_diagnostics_read.py`、`tests/test_coll_sid_diagnostics_read.py`，代码身份见其交接说明。私有验收 JSON 由唯一执行者保存；本页不列私有绝对路径。

A 保留语义层利用率、熵、孩子数、叶规模、有效层数及完整 SID 碰撞口径。B 保留原 in-catalog 前缀与 exact item 分母；OOV 单列排除，SID 命中不替代 item 命中。

C 保留原 G4 train-only 热度边界与 ties 规则，五档为 head、高于 median 且不高于 p90 的 mid、不高于 median 的 positive low、零计数 unsupported，以及 OOV；不能把 unsupported 并入原 low 档后替换原结果。边界来源为原 history manifest 绑定的 train-slices，来源 SHA 与每档 target-count 分母均留在验收 JSON。原 T3 参照取自已验收 S0 result 的同 seed C aggregate，不重估阈值。

独立交叉验收覆盖：原 source pins；B/C 命中与 OOV 分母一致；C 与 T3 的边界、每档 target-count 分母一致；每档命中相减与原 C_delta_vs_T3 一致；五档差之和与 exact item 总命中差一致。验收状态与证据归入结果主账。

D 使用原训练期同用户相邻且 item 不同的事件对，随机对保留 self-pairs；D′ 的随机对排除 self-pairs，不能替换 D。随机分母为零保留 UNKNOWN；D′ 的结构上无可配对前缀保留 STRUCT_NA，不改写成零或验收失败。C1′ 训练窗口不改变 D 的相邻对定义。

## 验证边界

本次只读原小 aggregate 及既有本地 metadata 缓存，不重算诊断，不读取原逐 query 预测、完整 SID 映射、完整向量、checkpoint 或 test，不修改原 run / bundle，也不新增作业。

有限 reader 的 verified 仅表示 aggregate intake 验收；训练验收仍以原始 stdout 的 verified 和 job / manifest 绑定为准。Synthetic 测试用于 schema、手算与拒绝条件，真实 intake 的来源 SHA 与 UTC 单独保留。

本页不据 lift 推断“学到了协同结构”，不跨 core03 与 A1 判赢家，不套用原 B1024 热度读法作为 S1 门槛，不换 anchor 或判转正。结果解读留给 Kiro。context 计数 UNKNOWN。
