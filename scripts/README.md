密级: 公开

# Scripts Directory

此目录未来只放薄的、可审计的命令入口，用于调用 `src/vkgr` 中已测试的能力。

脚本不得承载核心业务逻辑，不得绕过配置、manifest、数据合同或测试，不得内嵌真实列名、credentials、服务器地址和本地绝对路径。

新增脚本前先为其调用的核心行为完成 TDD；脚本应支持 dry-run 或等价的低风险检查，并输出 task/config/data/artifact 标识。

禁止自动创建计费资源、上传外部服务、批量删除数据，或在用户未确认时执行 snapshot/restore。

当前阶段不创建 `.py` 或 shell 文件。
