# 2026-10-17 readonly evidence tools — 独立 review

范围：隔离 worktree `phase2-readonly-tools`，基线 `05fd95e71d0f8221c0daffce7f691b1a6de423c7`；仅审新增三个 reader 与对应 synthetic tests。未访问服务器，未读取真实预测、checkpoint、embedding 或 test；未修改实现或事实源状态。当前可见 context 计数：UNKNOWN。

## MEDIUM：paired reader 未绑定非 combo 比较臂的已验收 config/source 身份

位置：`analysis/acceptance_paired_read_20261017.py:151-157`。

这里仅对 `p-feat04combo-cde` 调用 `check_combo`。F4、M1、M2 使用当前 run 路径读到的 manifest，并将该文件的即时 SHA 交给 generic verifier；`accepted_archive` 额外只核 seed、real/closed test gate、shared support、catalog 和 slice metrics。被 pin 的 `phase2_validation_read.load_archive` 核 history/query/catalog、best selection 与预测重算，却不核原实验 config/source；其 `inspect_run/verifier` 也不会比较原训练 config/source。

因此一个相同 rid/seed/data、不同 config 或 source 且自身 verifier 成功的归档仍可通过，并在输出里直接被命名为 `m2_vs_m1_registered`（以及另外两个固定方法比较）。输出 identities 能让人工发现差异，但没有实现“只读已验收归档”的 fail-closed 身份绑定。尤其同批 metadata reader 已用固定 M2 manifest `PINS`，paired reader 却没有要求读到同一份已验收 M2 归档。

建议在读取每个非 combo 比较臂时，核其已验收 manifest SHA，或核原 plan/config/source/data binding；M2 至少复用本批 metadata reader 的三 seed manifest pins。保留现有统计定义，不引入新科学门槛。增加一个“seed/data 相同但 config/source 被换”的 synthetic reject case。

## 已核事项

- 三 reader 与对应 tests：18 tests 通过，runtime 使用指定现有 Python，`PYTHONPATH=.`，`python -B -m unittest analysis.test_acceptance_paired_read_20261017 analysis.test_b256d2_four_column_read analysis.test_multi_interest_metadata_read_20261017`。
- phase2 source pin 与本地源码 SHA 一致；paired bootstrap 使用原 `paired`，pooled exact binomial 计算与原定义一致；未发现统计实现 HIGH/MEDIUM。
- B256D2 manifest/mapping pins 与本地冻结 `first_batch-plan-v4` 一致；`balance` 对应原 `sid_tok_audit.balanced_children` 的前三列、tolerance=1 算法；raw3 对应原 `unpad_depth2`。all-four 输出仅作 supplementary diagnostic。
- metadata 的 verifier derivation 与原 `seqm2_ops.derive_verify` 一致；最佳预测及 g0 ref 名与原训练实现一致；固定 M2 config/source/preflight binding、浅层 inventory 与禁止 artifact 名检查已核。无新增 HIGH/MEDIUM。

## 本次审查版本与限制

| reader | SHA256 |
|---|---|
| acceptance_paired_read_20261017.py | dd9cdc65d340a9b56b84d43a2dfc7915087c2a89e061c57e5159c187d23f3085 |
| b256d2_four_column_read.py | d90379869cbf9dd1ec117aff8a9617ab0b374ffb7db87169d7bcb69231d1d85d |
| multi_interest_metadata_read_20261017.py | 2dd810892e7c09642f002a969d9bc91fc17c135c221e46be7128bd6f77e228e8 |

本次只核代码、synthetic tests 与指定已有本地源/收据，没有验证服务器路径当前存在、真实完整 schema、原归档当前内容或最终科学结论。未发现 HIGH；上述一项 MEDIUM 需实现者处理，普通修复后由实现者 fresh regression，不叠加复审。


## 作者修复验收（不是独立复审）

作者关闭记录：唯一MEDIUM已以12个既有验收manifest固定SHA修复，新增替换归档身份失败用例fresh RED→GREEN；后续公开风险整改把部署目录全部移入私有参数并补RED/GREEN，22新tests和18既有回归（合计40）通过。发布6268807，实际服务器只读补数成功；没有第二次独立review。
