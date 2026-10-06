密级: 公开

# Tests Directory

此目录未来承载 unit、contract、integration 与小规模 evaluator parity tests。

每项实现遵循 TDD：先提交能复现缺失行为的 red test，记录 fresh failing evidence；再写最小实现；最后在当前 branch/worktree fresh 运行并记录 green evidence。

测试默认使用 toy 或 synthetic data，不依赖网络、GPU、真实 VK-LSVD 文件、服务器状态或任何公司内部数据。

涉及大数据、GPU、throughput 与 snapshot 的检查属于显式批准的 integration/acceptance task，不伪装成普通 unit test。

禁止把真实样本、secret、完整 SID map、checkpoint、index 或模型效果常量写入 fixture。

P1使用标准库unittest，分别覆盖config、runtime、manifest、CLI与边界；fixture为明确人工整数。测试自己创建独立temp目录并只清理自身产物；需要持久路径时临时目录位于outputs/p1-test-*。socket阻断和只允许local Git subprocess的检查验证实际调用路径，源码复核补充离线边界证据。测试命令：`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`；每项RED/GREEN见docs/p1_validation.md。
