密级: 公开

# Data Directory

`data/` 只定义数据落位规则，不保存数据本体。

## 获取方式

- 数据由计算服务器直接从 VK-LSVD 官方来源下载，不经本地电脑或 GitHub 中转。
- 下载前记录官方 source URL、dataset card/version、获取时间和适用许可。
- 当前许可依据只是VK-LSVD dataset card标注Apache-2.0的调研记录；P2首次下载前必须保存最新许可与使用条款快照，核验前不得作为仓库授权结论。
- 来源、许可或版本无法核实时停止处理，不用镜像或第三方副本替代。

## Git 边界

原始数据、解压数据、处理后数据、embedding 和缓存均不得进入 Git。

目录中未来只允许保留本 README、空目录占位和不含记录内容的公开 schema/manifest 模板。

## Hash manifest

每次下载与派生处理都在受控 manifest 中记录：

- source URL 与 dataset/version 标识；
- 相对路径、文件大小、hash 算法与 hash；
- 下载或生成 UTC 时间；
- 许可标识；
- 生成 task ID、配置版本和父文件 hash；
- 校验状态与失败原因。

Manifest 可提交前必须脱敏：不得含服务器 IP、用户名、绝对路径、signed URL、token、原始样本或内部公司信息。
