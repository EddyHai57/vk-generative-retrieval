密级: 公开

# A6000 SSH 与服务器验收 Runbook

## 目的与边界

当前平台选择为 Vast.ai，Eddy已手动租用标称RTX A6000 48GB实例；平台决策见`decision_log.md` D-049。SSH方法与授权见下节，其余命令仍为待执行模板，不能据此声称服务器验收通过。

首次租用只做不超过 1 小时的基础验收；任一硬门槛不满足时停止后续训练并记录证据。

不在本地或 GitHub 保存原始数据、checkpoint、index、完整 SID map 等大文件。

## Mac 与 Codex 的 SSH 连接方法

在已配置SSH的Mac本机终端执行：

```bash
ssh vast-a6000
```

`vast-a6000`是本机`~/.ssh/config`中的Host别名。真实HostName、Port、User和IdentityFile只保存在本机配置中；地址与端口取自同一条连接成功的Vast控制台命令。私钥留在Mac，不复制到服务器、仓库或聊天。公钥由Eddy添加到Vast实例。

Kiro已通过Remote SSH连接同一别名。Kiro连接成功不会自动让Codex获得远程工作目录；本地Codex需显式通过SSH执行远程命令。

Eddy于2026-10-07明确授权整个Codex通过SSH连接该服务器。其他对话应读取本节与D-050；实际可用性取决于其运行主机是否具有同一SSH配置、密钥及网络权限。该授权只覆盖连接，后续操作仍受任务scope、数据边界、预算与高风险操作确认规则约束。

Codex只读连接测试：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 vast-a6000 'printf "CODEX_SSH_OK\n"'
```

Sol执行对话于2026-10-07实际收到`CODEX_SSH_OK`，exit code 0。默认sandbox曾拒绝网络连接；经授权的网络escalation后测试成功。新对话若遇到网络限制，应使用正常权限申请流程，不绕过访问控制；BatchMode下认证失败也不代表服务器不可用，应检查本机agent是否能使用密钥。

连接后、进一步操作前，读取服务器现有`/etc/vast-agents-guide.md`及目标目录的`AGENTS.md`（如存在）。SSH成功只证明连接与认证；GPU/CUDA、容量、持久化和恢复仍须分别验收。

`exit`结束SSH会话；断开终端或关闭Kiro不会停止实例计费。Vast的持久化、停止与恢复流程需要按当前平台确认，不能照搬Thunder snapshot流程。

## P2首轮执行记录（2026-10-07）

本轮30分钟/新增计算费用US$1首轮范围已授权。实际只完成guide读取、两次独立SSH会话及无凭据费率字段查询；capabilities未提供当前费率，因此后续资源/环境验收暂停。workspace_is_volume=false；具体费用与恢复仍未知。全部命令、输入hash、官方文件元数据及未执行项统一见`p2_validation.md`，问题见I-022/I-023；不称P2完整通过。

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
| SSH 首次连接与 reconnect | 两次独立SSH会话通过；正式工作目录/长任务reconnect验收待执行 | `p2_validation.md` |
| `nvidia-smi` | 未执行 | 待填 |
| PyTorch CUDA | 未执行 | 待填 |
| FAISS | 未执行 | 待填 |
| Hugging Face 本地模型 throughput | 未执行 | 待填 |
| RAM ≥ 64GB | 未执行 | 待填 |
| 可用 disk ≥ 300GB，目标 500GB | 未执行 | 待填 |
| disk I/O | 未执行 | 待填 |
| 经批准的平台持久化/恢复验证 | 未执行 | 待填 |

## Codex 待执行模板

### 1. SSH 与 reconnect

本地终端执行；断开后重新连接，确认工作目录和长任务保护策略符合预期。

```bash
ssh vast-a6000
exit
ssh vast-a6000
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

### 5. 固定revision dataset传输吞吐

使用批准MVP文件清单和官方huggingface_hub，固定revision与allow_patterns，local_dir只能是服务器受忽略目录，token=False；不下载额外HF模型。记录实际文件bytes、函数wall-clock、bytes/s、outer/internal重试是否可观测、预期/实测SHA256与文件集合。缓存/Xet dedup未隔离时不得把有效落地吞吐称作纯网络带宽。本轮30文件验收见p2_validation。

### 6. RAM 与磁盘容量

```bash
free -h
df -h <workspace-path>
```

通过条件：RAM ≥ 64GB；目标工作盘可用空间 ≥ 300GB，优先 500GB。

### 7. 有界Disk I/O

只使用已批准的唯一新ignored临时文件，最多256MiB；本轮为64MiB。可用GNU dd的oflag=direct、conv=fsync写及iflag=direct读，记录exit/bytes/elapsed/hash，保留测试文件不删除旧数据。失败保留partial，不能静默降级后称direct-I/O通过。短probe不冒充持续fio/IOPS/latency全面验收；具体证据见p2_validation。

### 8. 持久化与恢复（待平台方案确认）

Snapshot 创建、计费和 restore 属于外部服务操作，必须先由用户确认。

1. 在实例中创建一个无敏感信息的 marker 文件并记录 hash。
2. 先确认Vast实例的存储类型、恢复支持与费用；不得假设Thunder snapshot流程可用。方案未确认时停止本项。
3. 仅按用户批准的Vast持久化/恢复方案操作，不自动停止、销毁或新建实例。
4. 重新执行 SSH、`nvidia-smi`、容量检查，并核对 marker hash。
5. 记录恢复机制、保留与不保留的内容、备份大小、创建/恢复耗时和新增费用；仅在方案支持并获批准时使用snapshot。

## 数据与产物边界

- 数据应由服务器直接从许可允许的官方来源下载，不经本地中转。
- 原始数据、处理后大表、embedding、checkpoint、index、完整 SID map 和大规模预测结果保留在服务器持久盘或 snapshot。
- GitHub 只保存代码、配置、脱敏 manifest、小型指标摘要和文档。
- 所有可复现实验引用服务器产物的 hash、生成任务、配置版本和时间，不提交产物本体。


## P2续执行当前记录（取代前文费用门暂停的下一动作）

Eddy已撤销费率gate并批准搭建下载，D-051记录，费用仍Unknown。资源/CUDA/P1 smoke、CPU FAISS、30文件SHA256/inventory、有界I/O及reconnect文件保持已验证；GPU FAISS、quota/长任务保护限制与完整恢复仍需审查。当前详细证据和矩阵唯一源为p2_validation文末，不用本runbook旧表的未执行记录冒充当前状态。危险恢复操作未执行，完整P2不能称Complete。


## P2 GPU与恢复方案更新

官方GPU FAISS独立环境及实际查询已通过，tmux synthetic断线任务日志已通过；Astra分别独立复核。当前矩阵与精确命令/hash统一见p2_validation文末。GPU/CPU FAISS版本分别使用独立环境，不修改driver或全局torch。

46文件冻结小bundle/逐文件hash及恢复命令已准备，backup回Mac与目录恢复仍待用户确认；目录恢复不替代stop/recycle/host-loss实测，完整恢复gate保持。Quota只限可见容量/挂载证据，不虚构hidden quota。


## 最新恢复证据（2026-10-07）

已批准的46文件小bundle完成Mac备份往返及全新服务器目录恢复，46/46 hash、独立P1环境、31tests和新旧synthetic manifest通过。具体路径、hash、UTC与命令统一读p2_validation最新恢复节；此前“未执行”只为历史状态。实例stop/start/recycle/destroy、不同host、GPU环境与真实数据灾后重获未实测，不能以目录恢复替代整个gate。


## P2批准边界与验收结论（2026-10-07）

Astra fresh复核已通过本次Eddy批准的目录级backup roundtrip、46文件恢复、独立P1环境与manifest验证。按tasks P2“经批准的平台持久化/恢复验证”，本次恢复项Passed；完整证据唯一源为p2_validation最新Astra验收节。未测stop/start、host-loss、GPU环境重建与raw重新获取不扩充为当前任务，保留P8再验；隐藏quota未知且仅有可见容量证据。此结论不代表全灾备，不授权任何实例状态或资源变更。
