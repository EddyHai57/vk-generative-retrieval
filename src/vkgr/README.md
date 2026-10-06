密级: 公开

# `vkgr` Package Boundary

此目录未来承载可测试、可复用的 VK-LSVD generative retrieval 核心实现。

未来职责按已批准 task 逐步加入：数据合同校验、catalog/history 构建、baseline、SID、受约束 decode、evaluation 和 manifest 接口。

所有行为变更必须先有 failing test，再写通过测试的最小实现；不得一次实现 P1 到 P13，不得为未批准阶段预建抽象。

禁止放置 notebook、下载数据、credentials、服务器专用临时脚本、checkpoint、index、完整 SID map 或任何 HelloBike 内部信息。

当前阶段不创建 `.py` 文件。
