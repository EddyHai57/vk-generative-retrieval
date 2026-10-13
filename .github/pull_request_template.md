密级: 公开

# Pull Request

## Task 与 Scope

- Task ID：
- Branch/worktree：
- 本 PR 允许修改的文件：
- 明确不在本 PR 范围内的内容：

## TDD Evidence

- [ ] 已附 red test 的 fresh 命令、exit code 与关键失败输出
- [ ] 已附 green test 的 fresh 命令、exit code 与关键成功输出
- [ ] 测试未依赖网络、secret、真实数据或其他 worktree 的状态

Red evidence：

Green evidence：

## Data 与 Claim Boundary

- 使用的数据 variant/version/hash：
- Label/history/catalog/split 状态：
- 本 PR 能支持的 claim：
- 本 PR 不能支持的 claim：
- [ ] 未包含 任何公司 或其他公司的内部信息

## Resource Impact

- CPU / RAM：
- GPU / VRAM / GPU-hours：
- Disk：
- Wall-clock：
- 预计或实际费用：
- [ ] 未经确认未创建、扩容或恢复计费资源

## Artifacts 与 Manifests

- Manifest 路径/hash：
- Artifact 类型、大小与 hash：
- 服务器位置的脱敏引用：
- [ ] 大数据、checkpoint、index、完整 SID map 与大输出未提交 Git

## Docs

- [ ] 已更新 `STATE.md` 或说明无需更新
- [ ] 已更新 `tasks.md` 或说明无需更新
- [ ] 已更新 data/architecture/experiment/issue 文档或说明无需更新
- Handoff 路径：

## Security Checklist

- [ ] 无 secret、token、私钥、signed URL、服务器连接信息
- [ ] 无原始数据、真实样本、内部 schema、内部代码或内部日志
- [ ] 无不明许可的第三方代码或数据
- [ ] 已检查 diff 与 artifact manifest
