密级: 公开

# Thunder Compute A6000 验收 Runbook

## 目的与边界

本文件是 Codex 后续执行模板，不代表任何命令已经运行。

默认实例为 Thunder Compute RTX A6000 48GB。

首次租用只做不超过 1 小时的基础验收；任一硬门槛不满足时停止后续训练并记录证据。

不在本地或 GitHub 保存原始数据、checkpoint、index、完整 SID map 等大文件。

## 资源与预算门槛

- Phase 1 总预算不超过 AUD 100。
- GPU 使用目标不超过 30 GPU-hours。
- GPU 使用硬上限为 48 GPU-hours；触及前必须停止并由用户决定。
- RAM 最低 64GB。
- 可用磁盘最低 300GB，目标 500GB。
- 非训练时停止 GPU 实例，避免空转计费。
- 账单达到预算 50%：复核剩余任务和 burn rate。
- 账单达到预算 70%：停止扩展实验，只允许完成已批准的最小闭环。
- 账单达到预算 85%：停止 GPU，向用户报告后再决定是否继续。

## 验收记录

每项必须保存 UTC 时间、实例规格、命令、exit code、关键输出和结论；敏感 hostname、IP、用户名与 token 不得进入 Git。

| 检查项 | 状态 | 证据位置 |
|---|---|---|
| SSH 首次连接与 reconnect | 未执行 | 待填 |
| `nvidia-smi` | 未执行 | 待填 |
| PyTorch CUDA | 未执行 | 待填 |
| FAISS | 未执行 | 待填 |
| Hugging Face 本地模型 throughput | 未执行 | 待填 |
| RAM ≥ 64GB | 未执行 | 待填 |
| 可用 disk ≥ 300GB，目标 500GB | 未执行 | 待填 |
| disk I/O | 未执行 | 待填 |
| snapshot restore | 未执行 | 待填 |

## Codex 待执行模板

### 1. SSH 与 reconnect

本地终端执行；断开后重新连接，确认工作目录和长任务保护策略符合预期。

```bash
ssh <user>@<host>
exit
ssh <user>@<host>
```

不得把私钥、token、完整连接串或未脱敏 IP 写入仓库。

### 2. GPU 与 driver

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

通过条件：GPU 型号为 RTX A6000，显存约 48GB，driver 可正常响应。

### 3. PyTorch CUDA

仅在环境已提供 PyTorch 时执行；本 runbook 不授权安装依赖。

```bash
python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

通过条件：`cuda_available` 为 `True`，device 与实例规格一致。

### 4. FAISS

仅在环境已提供 FAISS 时执行。

```bash
python -c "import faiss; print({'faiss': faiss.__version__, 'gpus': faiss.get_num_gpus()})"
```

记录 FAISS 版本和可见 GPU 数；缺包或 GPU 不可见均记为未通过，不现场擅自安装。

### 5. Hugging Face throughput

只使用服务器上已存在、许可明确的本地模型与本地输入；不得在验收中上传项目数据或默认联网下载。

```bash
HF_MODEL_PATH=<local-model-path> python <approved-throughput-script> --model-path "$HF_MODEL_PATH" --output <server-only-output-path>
```

记录模型标识、dtype、batch size、输入长度、warm-up、样本数、tokens/s、峰值 VRAM 和 wall-clock。具体参数由后续已批准任务冻结，本文件不预填。

### 6. RAM 与磁盘容量

```bash
free -h
df -h <workspace-path>
```

通过条件：RAM ≥ 64GB；目标工作盘可用空间 ≥ 300GB，优先 500GB。

### 7. Disk I/O

优先使用平台已有 `fio`；测试文件必须位于服务器临时目录，测试后只删除该次命令创建的文件。

```bash
fio --name=vkgr-acceptance --filename=<server-temp-path>/vkgr-fio.bin --size=4G --rw=readwrite --bs=1M --direct=1 --iodepth=16 --runtime=60 --time_based --group_reporting
rm <server-temp-path>/vkgr-fio.bin
```

执行删除前核对完整路径；不得使用通配符、递归删除或 `rm -rf`。记录 read/write bandwidth、IOPS 和 latency。

### 8. Snapshot restore

Snapshot 创建、计费和 restore 属于外部服务操作，必须先由用户确认。

1. 在实例中创建一个无敏感信息的 marker 文件并记录 hash。
2. 按 Thunder 控制台流程创建 snapshot。
3. 停止原实例，使用 snapshot 启动恢复实例。
4. 重新执行 SSH、`nvidia-smi`、容量检查，并核对 marker hash。
5. 记录 snapshot 大小、创建/恢复耗时和新增费用。

## 数据与产物边界

- 数据应由服务器直接从许可允许的官方来源下载，不经本地中转。
- 原始数据、处理后大表、embedding、checkpoint、index、完整 SID map 和大规模预测结果保留在服务器持久盘或 snapshot。
- GitHub 只保存代码、配置、脱敏 manifest、小型指标摘要和文档。
- 所有可复现实验引用服务器产物的 hash、生成任务、配置版本和时间，不提交产物本体。
