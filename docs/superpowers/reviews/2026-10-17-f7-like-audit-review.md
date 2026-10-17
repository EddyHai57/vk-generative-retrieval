# 用户画像特征（F7）训练期 like 审计工具独立 review

结论：**CHANGES_REQUESTED**。1 项 MEDIUM，未发现 HIGH。当前可见 context 计数 UNKNOWN。

受审范围：base `0ca61bc35ad68bd79b86103b54e470445cbab07f` → head `fcdb78c`，仅新增 `src/vkgr/profile_like_audit.py` 与 `tests/test_profile_like_audit.py`。本次只审计只读工具，不替代后续 F7 模型独立泄漏 review；未连接服务器、未读真实 Parquet、未启动作业、未 push。

## MEDIUM：异常取值域导致已算出的审计证据全部丢失

位置：`src/vkgr/profile_like_audit.py:55`，关联 CLI 输出位置 `:59`。

`summarize()` 已计算完整的 missing/true/false/invalid、missing_rate 与 invalid_types，但当 `invalid != 0` 时，紧随其后的 `require()` 直接抛异常；`execute()` 无法返回，CLI 的 JSON 打印也不会执行。此时 stdout 无聚合回执、原始 schema 类型、输入 SHA 或时间，只剩异常。数据取值域异常正是本任务需要审计和留证的分支；保持 STOP 是正确的，但不能丢掉已得到的聚合证据。空输入分支也有同样问题。

复现：使用内存 mock 替代 `read`、`safe` 和 `pyarrow.parquet.ParquetFile`，不访问真实数据；令模拟 like 列为 `[None, 1, 0]`。实际 `execute()` 只抛出 `ValueError: STOP: invalid like domain`。同一列的 `summarize()` 结果应为 `rows=3, missing=1, true=0, false=0, invalid=2, invalid_types={int:2}, missing_rate=1/3`，这些计数均未进入回执。

修复要求：保留现有 STOP 门禁，异常域不得成为成功；在 STOP 时输出机器可读聚合回执并绑定已核的输入 SHA、schema 与时间，CLI 仍返回失败状态。不得输出 raw rows。增加覆盖 `execute()` / CLI 的测试，断言 STOP 聚合仍可解析、缺失值不混作 False、数值 1 不混作 True。

## 已核证据与限制

- 指定命令使用 `PYTHONPATH=src`、既有 p6-tiger-smoke venv、`python -B -m unittest discover -s tests -p test_profile_like_audit.py -v`：4/4 PASS。当前测试覆盖 summary 与 train_records；未覆盖 execute / CLI，故上述缺口未被现有测试发现。
- `safe()` 要求绝对路径、拒绝 `..`、逐级拒绝 symlink，并在读取前拒绝 test/checkpoint 命名路径和常见 CP 后缀。raw 路径的 role 固定为 train/week_00–24，严格匹配顺序与数量；不存在任意追加 week 25/26 的分支。
- inputs/history/side metadata 在解析前核 SHA；25 个 raw 文件先核 size 与全文件 SHA，再打开 Parquet。preflight 只做 metadata / schema，不迭代 like 值；audit 的 Arrow 解码仅选择 `columns=['like']`，无 U1 字段、checkpoint 或 raw rows 导出代码。全文件 SHA 核验会读取原始文件字节，不等于解码其它字段。
- 只读检查私有 metadata 收据 `cache/pgr01-p-recovery-v1/main05/` 下的 `f7-control-source-20261017T113128Z.stdout`、`f7-control-inputs-20261017T113144Z.stdout`、`f7-side-contract-20261017T113203Z.stdout`、`s5-prefix-history-20261017T111651Z.stdout`：M1 原 argv inputs pin 与 inputs 收据一致，history/side pin 一致，side 绑定同一 history，版本为 HIST_V2_SIDE_V1，bit0 为 like，side/history 的 25 个 train raw_records 逐项一致。因此本次已有字段证据成立，未发现扩大数据合同的需要。
- `None` 独立计 missing；只有精确 Python bool 才计 true/false，整数 1、0 不会被强制转换。
- CPU1 / 0VRAM 的 queue 声明、固定 SHA 部署、真实 preflight 与后续 audit 原始 stdout 尚未在本次工具 review 中执行或验收，仍须执行者按原流程核实。本结论不授权新增实验、读取 test/旧 CP/U1 或修改科学门禁。


## Codex工程修复记录

唯一MEDIUM已fresh RED/GREEN修复：非法域或空train仍保留机器可读audit计数/输入SHA，status=STOP且CLI退出2；正常域规则不变。5项审计测试通过，numeric-bool/train-order/STOP-latch三个内存变异全杀。最终3df6d9f已扫描push/new deploy244文件SHA；真实审计未完成前不把metadata/测试当实际缺失率。没有第二份reviewer APPROVED，按AGENTS普通工程修复不重复全review。
