# 原SGNS前2000batch工程入口：C档独立复核

范围：新analysis/sgns_throughput.py、对应tests/fixture/source ledger/计划与必要原接口。独立review请求gpt-6-luna；实际runtime模型metadata未独立验证。未读全历史、未服务器运行。结论：无blocking finding。

原15文件与079d6bd逐字节相等；C1原spec SHA与accepted原run绑定，device=cpu、SGNSConfig原样。iter代理仅训练遍历起点计时；step代理先执行原更新，再在第2000次后抛边界异常。原fit finally恢复Torch线程/确定性设置；外层finally恢复两个代理。没有到达health或向量输出。固定OPS输入含lock/source pins/A aggregate、seed42；原scheduler、新目录与唯一ID。

独立复核指出恢复路径断言不足（非blocking）。执行者补测试：正常边界与注入负采样异常后，iterator/step、线程数、deterministic及warn-only全部恢复；6项通过。此前联合G/SGNS/清单13项通过，5个隔离新目录变异拒绝（python -B及PYTHONDONTWRITEBYTECODE=1）。小合成前缀参数、梯度、optimizer、Torch RNG、负采样数组和NumPy RNG与原完整实现相同；不是全规模科学逐位验收。

input_verify沿既有overnight_arithmetic口径：原verify墙钟乘全量/子集正对比；纯fit按18787015620/实测吞吐。96小时门槛未动。选择依据e369424：贴近现行算术，可事后核对；若被否决保留原回执再重算工程外推，不改科学run。

服务器正式吞吐、RSS及新任务准入尚未完成；本地runtime不同，不替代服务器原锁正式链。先前全库196项的14个failure/error未宣称清零（原G报告有逐项日志指针）。不报召回或转正结论。
