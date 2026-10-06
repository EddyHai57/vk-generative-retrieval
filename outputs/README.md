密级: 公开

# Outputs Directory

`outputs/`是本地或服务器运行输出目录，默认不进入Git。

Git中只允许保留本README和文件名明确包含`manifest`的脱敏JSON/YAML/TOML。

人工维护的实验与指标摘要统一写入`docs/experiment_log.md`，不在`outputs/`提交CSV、图表、逐query预测或完整ranking。

原始日志、profile trace、大型metric明细和其他运行输出留在服务器，并通过hash与脱敏路径别名引用。

任何manifest提交前必须检查：无原始样本、secret、服务器连接信息、完整SID map、checkpoint/index位置细节和内部公司信息。

未运行、未冻结数据合同或未完成holdout的结果必须标为draft/pending，不得包装成benchmark结论。
