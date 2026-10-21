# 原SGNS前2000 batch工程测速

依据863f53a §12–14。C1原spec原字节359ad3e1…140与原accepted run绑定，input device=cpu；原SGNSConfig固定dim64/window5/epochs5/negative5/batch4096/seed42/lr0.025/final_ratio1e-4。源079d6bd的fit_sgns与b5a4d1c逐字节相同。

目标：按已验收A历史长度直方图构造synthetic序列，catalog12012895；正式CLI只运行前2000 batch，测吞吐/RSS，不保存向量、不health-check。外推18787015620/实测正对吞吐加input_verify同比例成本，以96h冻结门槛报告；不改变device或任何科学配置。

实现选择（可逆，e369424）：原源码依赖闭包逐字节恢复为新文件。薄新入口调用原fit_sgns；原iter_positive_pairs代理仅在训练遍历开始记时，SGD.step代理先原样执行原step，第2000次更新完成后抛专用边界异常，原finally恢复线程/确定性设置；代理始终finally恢复。不改loss/negative/RNG/LR/梯度，不走health或导出。输入全长与预扫描成本单列，计时不把全量初始化当2000batch吞吐；synthetic频率和真实频率差异随报。

验收：fresh RED→最小GREEN；小synthetic前缀参数/梯度/optimizer与RNG逐位对照、batch上限与无health、恢复代理、CLI非法非零/dry非空；隔离变异；真实launcher部署→metadata→dry链；独立C审后公开扫描、固定SHA新部署、原scheduler准入与原stdout绑定。原P及G解码job不重投不中断。训练入队仍不授权。

[ ] 原device/spec/config输入证据与原源码闭包
[ ] bounded sampler及合成长度/正对外推测试
[ ] 正式新目录CLI/producer绑定/资源与回执
[ ] 本地验证、隔离变异、独立C审
[ ] 固定SHA发布/服务器同CLI工程运行与绑定验收
[ ] experiment_log/任务passes/STATE/handoff/schedule区间增量
