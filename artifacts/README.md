密级: 公开

# Artifacts Directory

`artifacts/` 表示可被后续任务复用的机器产物。

大产物包括 checkpoint、optimizer state、embedding、FAISS index、完整 SID map、trie、缓存和大规模预测文件；它们只保留在服务器持久盘或 snapshot，不进入 GitHub。

Git 中只允许保存本 README，以及脱敏的小型 artifact manifest。Manifest 应记录 task ID、配置版本、数据 hash、artifact hash、大小、格式、生成时间、资源摘要和服务器位置的脱敏引用。

不得提交 secret、signed URL、服务器连接信息、原始数据片段或可还原完整数据集的内容。
