# P2 验证记录 — 已批准范围内验收通过

密级: 公开

## 历史首轮结论与scope（已被文末续执行授权取代）

Task ID：P2。首轮只做现有实例的资源/环境验收，最多30分钟、新增计算费用US$1；不训练、不进入P3、不默认安装依赖或下载数据。当前未完成首轮资源验收，也未完成完整P2。

本轮在费率门暂停：本地服务器capabilities未提供可核实费率，未读取API key、token或计费凭据；无法证实费用符合上限。此后没有继续GPU/CUDA运算、服务器clone、依赖探查或文件传输；只完成本机与官方小型公开元数据只读核查。实例未停止、销毁、扩容或创建快照；停止验收动作不会停止已租实例计费。

执行窗口：2026-10-06 17:40:48 UTC开始记录；前置SSH与费率核查发生于17:40:48至17:42:25 UTC之间（工具未给逐命令UTC，不补造精确时刻）。HF首次元数据查询UTC为2026-10-06T17:42:25.918553+00:00，分页复核截至17:42:59 UTC。文档收口时间以最新handoff为准。

输入：fresh本地main、GitHub main、锁定P1代码、主工作区9文件治理overlay、本机已配置SSH alias和现有远程环境。允许输出仅为本worktree的STATE/handoff/server_runbook/issue_log及本验证文档。没有新增生产脚本、没有TDD功能实现；不把环境门失败算RED。

## Worktree与输入provenance

- 原main和旧P1 worktree未修改，四份P1运行证据未删。
- 使用Codex原生create_worktree，以fresh main创建独立worktree，初始Git status clean；随后建立`codex/p2-server-acceptance`。
- 固定基点：`0376d9a8119f3500cbf78519675cc7bf18013c1b`。本轮fresh HTTPS `git ls-remote --exit-code https://github.com/EddyHai57/vk-generative-retrieval.git refs/heads/main`返回该hash，exit0；未fetch/push/改remote。
- 显式读取主工作区9文件diff，以`git apply --check`后apply；前后patch字节与9文件字节均相同，exit0。
- 入场overlay SHA256（`git diff --binary -- <9文件>`原始字节）：`59ead6e485b7768aad6f3ad08f44ca9a9882694d46e58bc37c1167d21712df50`。这是入场快照，后续P2记录会改变部分文件，不混称最终工作区hash。
- 原始patch和hash JSON保留在本机临时目录，仅作接力；独立验证以本节hash及Astra保留的主工作区输入为准。
- 新worktree没有安装独立.venv；未借用别的worktree环境冒充baseline测试。服务器费用门提前停止，因此远程P1 tests/smoke/verify未执行。

| 入场文件 | SHA256 |
|---|---|
| `README.md` | `d0e0aad0b35035494d0572cd1cb361ab12aaa766e9320c69d775e6d6292fd771` |
| `STATE.md` | `f718de8b45b0d577c92ef3be2a70168dc9db2d5b210bc2546c5f5f0f571fd8e5` |
| `configs/mvp.yaml` | `9fd1a23c963e74ba457011ddffdfbd9d48b24e7e6d6ac1bbaf4ed7b079f4ce87` |
| `configs/scale.yaml` | `042ee38cc81896ab53985cdd18142ae27e2d0fdb803689b8187b6d0c3d1d49c2` |
| `docs/architecture.md` | `c21fa732797349451d0e13e7a89b6f1083cb063fa528154d97055b7999c9d0e4` |
| `docs/decision_log.md` | `ac65a8962fc3880b4c57947fd2795b81085614f71172950031b6102d90215509` |
| `docs/server_runbook.md` | `8f1572648101c7676eebacd568aeba6690c3fae0d572fe7e505187bc30a836bb` |
| `handoff.md` | `c2227cff53883c4f42138806a11b8b48f249b1c7c9160acf0371112cc3483560` |
| `tasks.md` | `812668cae13b6949cb5ffca3ff99ee8491d2b58b6244890973d92fe29f2f543d` |

## 实际执行与关键输出

### 1. 服务器指南（Passed：读取前置条件）

脱敏命令：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 vast-a6000 'cat /etc/vast-agents-guide.md'
```

Exit0，读到完整base与PyTorch两个指南。指南说明实例为受限Docker容器、默认环境`/venv/main`、工作目录并不自动持久化。指南提供使用计费API key的方法，但本轮按scope未读取或使用凭据，也未执行其安装/服务控制模板。尚未选择或建立项目目标目录，因此目标目录AGENTS检查待后续；没有对未读规则的目标目录进行操作。

### 2. 费率门与存储属性（费率Blocked；存储属性仅只读事实）

SSH执行system python3，从`http://localhost:11111/capabilities`读取JSON，仅打印allowlist中的数字字段：`dph_total/dph_base/dph/hourly_rate/price_per_hour/storage_cost/inet_down_cost/inet_up_cost`，以及`instance.workspace_is_volume`。递归排除credentials/services/endpoints/environment/env；不打印JSON全体、ID、地址、端口或任何credential。

实际命令（SSH别名保留，查询输出已限定）：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 vast-a6000 "python3 - <<'PYREMOTE'
import json, urllib.request
try:
    with urllib.request.urlopen('http://localhost:11111/capabilities', timeout=10) as r:
        d=json.load(r)
    out=[]
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k.lower() in {'dph_total','dph_base','dph','hourly_rate','price_per_hour','storage_cost','inet_down_cost','inet_up_cost'} and isinstance(v,(int,float)):
                    out.append({'field':k,'value':v})
                elif k.lower() not in {'credentials','services','endpoints','environment','env'}: walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(d)
    print(json.dumps({'rate_fields':out,'workspace_is_volume':d.get('instance',{}).get('workspace_is_volume'),'rate_verified':bool(out)}))
except Exception as e:
    print(json.dumps({'rate_verified':False,'error_type':type(e).__name__}))
PYREMOTE"
```

Exit0，实际脱敏输出：

```json
{"rate_fields": [], "workspace_is_volume": false, "rate_verified": false}
```

这不证明平台没有费率，而是本次无需凭据的只读接口没有给出所需费用证据。当前计算费、存储费、下载流量费与实际扣费均Unknown，未使用旧截图报价推算，不能声称在US$1内。

`workspace_is_volume=false`仅说明本次capabilities报告没有独立volume支撑工作目录；指南称stop/start保留容器文件系统、recycle/destroy会清除无volume的数据。这是指南声明，非本轮恢复实测；不能关闭恢复gate。未创建marker或备份，也不假设mount/quota容量。

补充官方来源：[Vast Storage Types](https://docs.vast.ai/guides/instances/storage/types)，本轮fresh读取。Container storage在stop时保留，destroy时丢失，stop仍收存储费；独立volume也收费且绑定同一物理host，不能当作跨host容灾。无volume不表示停止实例就丢数据。具体恢复方法仍须用户批准并实测，不自动新租volume。

两次独立SSH会话均exit0，但第二次只查询费用门；正式reconnect工作目录一致性与长任务保护仍未验收。

### 3. GitHub与本地输入（Passed）

Fresh固定ref核对、clean起点、branch建立和9文件overlay字节核对均exit0。源码没有修改；未服务器clone，没有服务器checkout或产物目录可报告。

### 4. 官方VK-LSVD小型元数据（部分证据；下载gate未通过）

只读来源：

- [官方repo API](https://huggingface.co/api/datasets/deepvk/VK-LSVD/revision/main)
- [固定revision数据卡](https://huggingface.co/datasets/deepvk/VK-LSVD/raw/084a18386e149a454dd6dbe0123478e75c6b09ce/README.md)
- [MVP目录元数据](https://huggingface.co/api/datasets/deepvk/VK-LSVD/tree/084a18386e149a454dd6dbe0123478e75c6b09ce/subsamples/ur0.01_ir0.01?recursive=true&limit=1000)
- [配套目录元数据](https://huggingface.co/api/datasets/deepvk/VK-LSVD/tree/084a18386e149a454dd6dbe0123478e75c6b09ce/metadata?recursive=true&limit=1000)

使用Python标准库urllib/json；每个响应最多2,000,000 bytes，超过即拒绝。查询及再次分页检查均exit0：两目录无`rel=next`，重复目录文件名一致。只取API JSON和README，不请求任何parquet/npz文件，也不输出signed URL。

- Revision：`084a18386e149a454dd6dbe0123478e75c6b09ce`。
- API cardData及固定revision README均声明license=`apache-2.0`，gated=false；siblings中未发现LICENSE/NOTICE文件，作为审计记录项而非单独阻塞理由。已读官方README，在本轮核查范围内未见相互冲突或额外限制条款；可用于本项目下载研究、原始数据不再分发的有限许可审计结论，不保证未知第三方权利，也不添加项目LICENSE。当前不下载的原因仍是费率及具体下载/容量/恢复scope尚未放行。
- README SHA256：`3b4c3fb2cb2853526cec6145e8e5f20fabb9efbd33ae024f9f96948237f3538e`。
- 本次脱敏JSON元数据记录SHA256：`f034c2ee23cb6249d7a0b03f2e6037ada15db228d84fc7ff02affe0b89398f51`；完整文件清单如下。
- MVP目录实际列出train week_00～24、validation week_25、test week_26，共27文件，合计25,839,960 bytes。这是文件inventory，不是读取后的schema/行数/label验收。数据合同仍Draft；未改变split/label，不读取验证或test内容/指标。
- 配套文件按官方Quick Start列三份全量metadata；交互与配套合计2,616,452,194 bytes（约2.44 GiB）。配套embedding不是MVP大小，下一part需明确允许全量配套下载或另有官方subset方案。
- 表中SHA256来自API的LFS OID，是预期校验值；尚未下载，不称本地checksum验证通过。下载过程的临时文件、解压/加载空间与重试流量需另估。
- 计算费率、Vast下载流量费、可用磁盘与quota、下载工具版本、容量余量与恢复方案均待核。许可声明的有限审计通过不替代文件下载scope与费用/容量/恢复门。

| 文件 | bytes | 官方预期LFS SHA256（未实测） |
|---|---:|---|
| `subsamples/ur0.01_ir0.01/test/week_26.parquet` | 976229 | `b59dc3594fbbbef1a556bb503ad7eff8f268ca3298e9093ddc18cf01bede3cff` |
| `subsamples/ur0.01_ir0.01/train/week_00.parquet` | 642528 | `c4a8dae41596b357872ee9dc0297c8abb2b43d3bb936823fc8b7127453ff7741` |
| `subsamples/ur0.01_ir0.01/train/week_01.parquet` | 922739 | `f19dd3ea9e992c1f10aea85702afd0b328e10888b8a37105d341e27aeda97467` |
| `subsamples/ur0.01_ir0.01/train/week_02.parquet` | 942380 | `2875cbccc3753b657beeeaccf3e9f08ab13276984cdfa01c046822156cc0cdb5` |
| `subsamples/ur0.01_ir0.01/train/week_03.parquet` | 873211 | `bcc1e2fc1074f79c581560cbb16e0b55de8772046c98df2b47e39811dbfd58a4` |
| `subsamples/ur0.01_ir0.01/train/week_04.parquet` | 937393 | `766d617cbc467b8e538a86dfeae6640c1c6ce664202c9114838865582cc23a4c` |
| `subsamples/ur0.01_ir0.01/train/week_05.parquet` | 632565 | `c00228268b91c94458a19a5c003d84dbdf0c5779690cbf73157ee87d8ab2fb8a` |
| `subsamples/ur0.01_ir0.01/train/week_06.parquet` | 799185 | `67ac9fefb171f700cc26aebea50ac71b00aef9da8aa938297b7985565578c237` |
| `subsamples/ur0.01_ir0.01/train/week_07.parquet` | 983655 | `8507384d295c7b284476775662d62d8efa7ab6a4de537174e8ac137d931e9a9f` |
| `subsamples/ur0.01_ir0.01/train/week_08.parquet` | 1026089 | `2d7f64545ec8cb7b8ebbbe3fcb7f720b47297446031b6a6d74f7a11b700bc26a` |
| `subsamples/ur0.01_ir0.01/train/week_09.parquet` | 994597 | `6bed90de8472d792ff82bed80c5d7227c06c1e76c781b603c2d18fafed62533c` |
| `subsamples/ur0.01_ir0.01/train/week_10.parquet` | 955811 | `8ac871a9561686cb0073389b2f006e1a31246040678ea00e229fb368bf72263c` |
| `subsamples/ur0.01_ir0.01/train/week_11.parquet` | 947285 | `b65b0c1ea3d943bd7d9bb1865f886226159116ede0438d0681df43cdded592ef` |
| `subsamples/ur0.01_ir0.01/train/week_12.parquet` | 976913 | `274f44b9caf07f8c3b5ebb4f910e77d387021bffc6009fd7b4a064907b0e410a` |
| `subsamples/ur0.01_ir0.01/train/week_13.parquet` | 1082104 | `751d066a4b85e70676ff4254065cd5deaee903a98198c4eebab12fcd35632949` |
| `subsamples/ur0.01_ir0.01/train/week_14.parquet` | 1025175 | `47ed124a1ea4ac77391d22ee13ff5d4a00ab6de62b9be4f8ba02af3a2b4597cd` |
| `subsamples/ur0.01_ir0.01/train/week_15.parquet` | 1034010 | `8a5e82d603e53e39b7c6a98af1091a24b806c961fd1f74d5a733749f8a12b75d` |
| `subsamples/ur0.01_ir0.01/train/week_16.parquet` | 1009990 | `48bed166e1d9fe2d2467fd1c528cf933a18e5008fe3d3d6a3935b685aec12608` |
| `subsamples/ur0.01_ir0.01/train/week_17.parquet` | 974139 | `9e63b92aeb3e16b4f8daeefe91bd89a106802e68b086bc74ef0e76a86d8ed415` |
| `subsamples/ur0.01_ir0.01/train/week_18.parquet` | 971015 | `efd0e5ff6ffc02e4ec603496f4d241937434dbdfb670bf6df42bc7670329620b` |
| `subsamples/ur0.01_ir0.01/train/week_19.parquet` | 977046 | `a44e9f4cfc69c1662e563b6f9000dbec3a6674af1c641a43bd9b2af523ce60f7` |
| `subsamples/ur0.01_ir0.01/train/week_20.parquet` | 936557 | `b35b54f7aa35a6ee1a4fa8b5a184a8237836810c52efddaf063d90ae0ef4a6d0` |
| `subsamples/ur0.01_ir0.01/train/week_21.parquet` | 1011466 | `d50be2f1a7772a34c1046c656524885526507878a43b1ceb4c46c47d9709cdff` |
| `subsamples/ur0.01_ir0.01/train/week_22.parquet` | 998624 | `479721322e6d497794d7adf43ba71b01815080680b6c1e63edd9289de1a316a5` |
| `subsamples/ur0.01_ir0.01/train/week_23.parquet` | 1104283 | `8cb680ca0427cdbe30af90b575fb19dae63ecc961e25317583db025e7e86c63c` |
| `subsamples/ur0.01_ir0.01/train/week_24.parquet` | 1032816 | `b103a275faf89cd90b000501db11b12b97817a334bf83bd473a4b21d477592b6` |
| `subsamples/ur0.01_ir0.01/validation/week_25.parquet` | 1072155 | `73ebe28c42cbe9c457d77d25a475e3e2b355f200ef877140a9d3918206ac2570` |
| `metadata/item_embeddings.npz` | 2362338243 | `6a4024ad7623f5d8150e59c5faaaf42fb7d7f48e2beec44c6917e5e57debea7c` |
| `metadata/items_metadata.parquet` | 155747843 | `607ffd09d3fb6627f7eec07ed53583d45c47263a2e6538e4deee66cd8a22e51e` |
| `metadata/users_metadata.parquet` | 72526148 | `51fa9528f71bb0da947311d7baa4727d325285a3e748b54816477b82492aff20` |

## 验收矩阵

| 验收项 | 状态 | 依据 / stop原因 |
|---|---|---|
| SSH指南读取、两次独立连接 | Passed（限定连接） | 两命令exit0；非完整reconnect验收 |
| 当前费率与费用预测 | Blocked | rate_fields为空，费用Unknown |
| 工作目录独立volume属性 | 已读，只限属性 | workspace_is_volume=false；未恢复验证 |
| GPU/driver/VRAM | 未执行 | 费用门暂停 |
| CPU affinity、cgroup quota及实际RAM | 未执行 | 未用宿主数冒充容器额度 |
| 工作盘free空间及quota | 未执行 | 无容量验收结论 |
| torch/FAISS/Python3.11/PyYAML版本与缺包清单 | 未执行 | 未探查；无依据称缺包 |
| 确定性CUDA tensor、CPU对照、峰值VRAM/elapsed | 未执行 | 费用门暂停 |
| 服务器HTTPS clone及固定commit | 未执行 | 费用门暂停，无覆盖 |
| 远程P1 tests、唯一目录smoke/verify | 未执行 | 费用门暂停 |
| VK-LSVD许可及source manifest | 有限审计通过；未下载 | Apache-2.0官方声明及固定revision/inventory已核，未见冲突；不再分发raw，不保证未知权利 |
| 数据下载与checksum | 未执行 | 费用、容量、具体下载及恢复gate未通过 |
| HF吞吐、disk I/O | 未执行 | 未下载模型/数据、未运行fio |
| 持久化/恢复 | 未执行 | volume=false；具体安全方案需确认 |

## 首轮暂停时的建议（历史，当前下一步以文末为准）

1. 请Astra先取得当前控制台计算费率及storage/入站流量计价，确认时限下US$1可控。不要发送连接信息或账单credential。实例若仍运行会持续计费，本轮未替用户停止。
2. 费用门放行后，在剩余时间/预算内依次读取目标目录规则、查GPU、cgroup/affinity/RAM、free disk/quota、现有Python与torch/FAISS。缺包时给准确清单再审批，不能根据image指南假定已安装版本。
3. 现有torch满足时仅做小型确定性CPU/CUDA对照与同步计时；现有P1依赖满足时再固定代码commit执行P1 tests/smoke/verify，产物唯一目录，不删除失败证据。
4. 建议将HF本地模型tokens/s模板替换为固定revision dataset文件传输的bytes/s、elapsed、重试与费用记录，因为P2目标是数据可用性；这只是建议，Astra须决定并修订gate，本轮未偷改模板。Disk I/O建议另批小型有界临时文件测试，未默认4GB fio。
5. P2完整完成还需下载后的checksum/inventory与经批准的恢复实测；在无volume情形下先规划服务器外备份及保留策略，不能执行destroy试探。

## 交付与review

本轮只收口文档，Astra负责fresh-context review。没有独立reviewer验收结论，不称P2 Complete。原main、旧P1 worktree和服务器项目文件未修改。生产源码未改，不重跑与本轮无关的本机代码测试；本轮检查为输入hash、文件inventory、公开边界和diff格式。


## 续执行授权与接口（2026-10-07，取代前文费用门暂停的下一动作）

Eddy明确撤销费率gate和原US$1费用门；费用仍Unknown，当前继续已批准的资源、最小搭建与30文件下载。不训练、不P3、不改变计费资源。恢复危险操作仍单独审批。

接口选择：复用已安装官方huggingface_hub.snapshot_download，固定repo_id/repo_type/revision，allow_patterns精确30路径，token=False，local_dir为服务器受忽略的新目录，max_workers=4。随后标准库流式SHA256/大小核验，输出脱敏inventory，不解析真实数据。没有新增原创生产下载脚本，不产生自编脚本TDD交付；失败保留partial，使用官方library安全重试。

本机一次文档更新命令出现UTF-8解析SyntaxError（exit1，在执行前失败，无文件修改），改为显式/usr/bin/python3并指定workdir后执行；不算生产RED，未改系统配置。


## 续执行实际证据（2026-10-06 UTC；Sydney日期为2026-10-07）

### 资源与隔离环境

- 17:55:28 UTC读取`/workspace/AGENTS.md`（与已读完整base/PyTorch指南一致），再查看资源；不是根据租用页面猜资源。
- GPU query exit0：`NVIDIA RTX A6000, 49140 MiB, 595.91.07`。显存为设备总容量，不声称全部空闲。
- 可见cgroup v2 root：`cpu.max=1024000 100000`，quota为10.24 CPU；affinity64逻辑核，不拿64当可用CPU配额。
- `memory.max=211900432384` bytes，18:00:28 UTC current=6,193,717,248；min(cgroup余量, /proc MemAvailable)上界为205,706,715,136 bytes，满足64GiB检查。宿主MemTotal未用作容器RAM。
- `shutil.disk_usage`项目路径total=536,870,912,000 bytes，下载后的free=533,483,663,360 bytes，满足300GiB门槛。quota工具未提供，未独立查询硬配额；statvfs容量证据不保证隐藏宿主/租户限制。
- 新目录代号`SERVER_PROJECT=/workspace/vkgr-p2-20261006t1755`，创建前确认不存在；没有覆盖已有checkout。
- HTTPS clone命令使用`--depth 1 --branch main --no-checkout`，`git rev-parse HEAD`匹配固定0376d9a完整hash后`git checkout --detach`；已读项目AGENTS。无额外remote配置，clone的默认origin不用于push。
- uv0.12.10选择CPython3.11.16；`uv venv --python 3.11 --managed-python .venv`及`uv sync --locked --no-dev`均exit0，实际PyYAML6.0.2（MIT）。uv.lock未改变。只创建项目隔离环境，没有改系统Python/driver或现有torch环境。
- 现有`/venv/main`实际Python3.12.14、torch2.11.0+cu128、wheel CUDA12.8、HF hub1.30.0、PyYAML6.0.3。该环境只用于既有torch和官方下载工具；不冒充P1环境。
- FAISS原来未安装。单独`cache/faiss-env`使用同一3.11解释器，从PyPI安装精确`faiss-cpu==1.15.1`（MIT）、numpy2.4.6、packaging26.3；实际版本/metadata记录，不向P1环境或全局环境加包。CPU `IndexFlatIP`四个one-hot向量检索精确命中，通过；`get_num_gpus()=0`，GPU FAISS仍不可用，不能称原runbook的GPU可见性门通过。PyPI来源 https://pypi.org/project/faiss-cpu/1.15.1/ 。没有复制第三方实现，也未改项目LICENSE。

### CUDA确定性小型对照

17:56:16.295197 UTC，使用现有torch，exit0。输入为float32 `arange(64).reshape(8,8)`和2×单位矩阵；独立CPU参考同时断言等于输入×2。GPU矩阵乘同步后与CPU逐元素torch.equal为True，sum4032.0。peak allocated=8,521,216 bytes，peak reserved=23,068,672 bytes；包含输入传输/首次调用的elapsed=0.095156894秒。不是训练/吞吐benchmark，未执行任何真实数据计算。

### P1 fresh regression、smoke/verify

17:58:23 UTC开始，完整命令exit0：

```bash
cd "$SERVER_PROJECT"
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-p2-20261006t1755
PYTHONPATH=src .venv/bin/python -B -m vkgr verify --manifest outputs/p1-p2-20261006t1755/run_manifest.json
```

31 tests in1.567s，OK；CLI分别输出P1 synthetic smoke completed / P1 manifest verified。result SHA256=`565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28`。只读synthetic fixture，不做模型评估。

### 固定30文件下载与完整性

官方huggingface_hub下载指南：https://huggingface.co/docs/huggingface_hub/guides/download 。本轮复用现有1.30.0 library，未引入新生产脚本。

实际调用（paths来自上文固定30文件metadata）：

```python
snapshot_download(
    repo_id="deepvk/VK-LSVD", repo_type="dataset",
    revision="084a18386e149a454dd6dbe0123478e75c6b09ce",
    allow_patterns=paths, local_dir=str(target), token=False, max_workers=4,
)
```

目标为`$SERVER_PROJECT/data/raw/vk-lsvd-084a183`，创建前确认不存在；输入metadata只传脱敏公开JSON到`cache/p2-source.json`，没有传本机数据、私钥或其他未列明内容。目标为服务器受忽略目录，没有Mac raw data或模型下载。

- 下载UTC start：2026-10-06T17:59:43.263460+00:00；全部hash核验结束：2026-10-06T17:59:55.428056+00:00；exit0。
- 30/30实际文件大小及逐文件流式SHA256全部匹配上文官方LFS expected值；去除library `.cache`控制文件后的实际文件名集合与allowlist完全一致，没有多取数据文件。
- 实测bytes=2,616,452,194；snapshot_download wall-clock=10.219652068秒；bytes/elapsed=256,021,650.896 bytes/s。这是文件落地有效吞吐，未隔离既有HF缓存/Xet去重，不能当作纯网络带宽或长期吞吐。
- 外层重试0；library内部重试次数未instrument，不伪造“所有重试0”。失败策略为保留partial并使用官方library重试，本次没有失败或自动删除。
- 服务器inventory：`cache/p2-inventory.json`，SHA256=`431a0bc3d74356f1f2b698dfe6594441221a6ba6c9c7d0ae921ae30c3bbd17d6`。该JSON只记录路径/hash/大小/UTC/吞吐，留在服务器；本节完整expected表现已由实际校验30/30确认。
- source metadata JSON SHA256=`f034c2ee23cb6249d7a0b03f2e6037ada15db228d84fc7ff02affe0b89398f51`。
- 未解析parquet/embedding、validation/test schema或任何指标；只是字节/hash校验，不冻结label/split、不进入P3。

### 有界Disk I/O

18:00:28 UTC，唯一新ignored文件`cache/p2-io-20261006t1800.bin`，64MiB，保留不删：

```bash
dd if=/dev/zero of="$SERVER_PROJECT/cache/p2-io-20261006t1800.bin" bs=1M count=64 oflag=direct conv=fsync
dd if="$SERVER_PROJECT/cache/p2-io-20261006t1800.bin" of=/dev/null bs=1M iflag=direct
```

两次exit0。按Python perf_counter含进程启动开销计：write0.036115286秒、read0.020345770秒；有效1,858,184,484.836 bytes/s写、3,298,418,520.663 bytes/s读。文件SHA256=`3b6a07d0d404fab4e23b6d34bc6696a6a312dd92821332385e5af7c01c421351`。这是短小同步direct-I/O probe，不等于持续4GB fio、IOPS/latency全面benchmark；未伪造后两项指标。

### Reconnect与Git边界

18:01:03.405186 UTC新SSH会话，exit0，能进入同一项目目录、核对固定commit与inventory/manifest。30文件逐个`git check-ignore -q`均exit0，`git ls-files -- data/raw cache`为空，真实数据/环境/IO测试文件未tracked。

- P1 manifest SHA256=`41f6918a61857d19c6f2c393d598b7c791eb6bd881eea3a68b688715ff460645`。
- 源码fingerprint=`eecdaf51b3f04da447773d4c959a3d765bc85ac63f9d347faca30c45ccc00f72`。
- overlay传输前server Git status仅P1运行目录untracked；无source/config/lock变化。后续只同步明确列出的治理文件，不能把文档dirty冒充源码变更。
- Reconnect已证明连接/目录/文件保持；长任务断线resume和stop/start/host-loss恢复尚未实测。有限短任务已退出，不留下未管理训练/下载后台服务。

## 下载完成时验收矩阵（历史，已由文末最终矩阵取代）

| 项目 | 当前结果 |
|---|---|
| SSH/reconnect目录与文件保持 | Passed，长任务保护未测试 |
| GPU/driver与小型CUDA运算 | Passed |
| 容器CPU quota/RAM >=64GiB | Passed（可见cgroup与进程证据） |
| 工作盘free >=300GiB | Passed；独立quota查询Unknown |
| Python3.11/PyYAML6.0.2隔离环境 | Passed |
| P1 full tests/smoke/verify | Passed，31 tests |
| FAISS CPU小型检索 | Passed；GPU FAISS0，原GPU可见门未通过 |
| HF固定revision文件吞吐 | Passed，30文件实际落地；cache/dedup未隔离 |
| MVP inventory与大小/SHA256 | Passed，30/30 |
| 有界64MiB direct-I/O | Passed；持续fio/IOPS/latency未覆盖 |
| 完整持久化/恢复 | 未执行，I-023 Open |
| 费用 | Unknown；用户撤销费用gate，非阻塞 |

当前状态为P2实现与下载可复核，等待Astra fresh review；不称P2 Complete。剩余恢复方案、GPU FAISS验收口径、quota与长任务保护限制由Astra审查；不要把资源/数据通过误称训练或模型效果通过。


## Astra独立验收归属（2026-10-07）

以下是主协调Astra在另一个对话独立执行并转达的证据，不冒充Sol直接重跑：

- 2026-10-06T18:09:42.955277+00:00开始，SSH exit0；Astra标准库逐文件重算30文件size/SHA256，全部匹配2,616,452,194 bytes；精确文件集合、每文件git check-ignore及git ls-files data/raw/cache为空通过。固定HEAD与source metadata hash复核通过。
- 服务器项目.venv完整31 unittest in1.618s，OK；Astra新建`outputs/p1-astra-p2-20261007-review`，smoke/verify exit0，result hash与本轮Sol运行一致。该目录为可解释的新untracked运行证据，不删除、不纳入提交。
- Astra只读tmux v2日志，8步、DONE、stderr0及log SHA256一致，exit0。
- 2026-10-06T18:20:23.551171+00:00，Astra在GPU FAISS环境独立运行8×8 float32 identity与CPU→GPU top1；索引与距离同时匹配CPU及独立identity oracle，GPU count1、GpuIndexFlat、FAISS1.11.0，exit0。

## GPU FAISS与tmux后续实测

### GPU FAISS官方二进制来源与失败保留

官方支持渠道：[facebookresearch/faiss INSTALL](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)中的pytorch/nvidia/conda-forge；包构建元数据来自 https://api.anaconda.org/package/pytorch/faiss-gpu 。不是把CPU wheel的get_num_gpus=0改口成GPU通过。

服务器缺conda/mamba工具，依照[micromamba官方手动安装](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)只提取自包含binary到`cache/tools/micromamba`；没有shell init或系统安装。实际2.9.0，官方archive bytes6,988,090，archive SHA256=`8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd`，binary SHA256=`366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3`。

- 第一次FAISS1.15.1/Python3.11/pytorch-cuda12.1 dry-run exit1，solver_problems=[unsupported request]，记录在`cache/faiss-gpu-solve-failed.json`。官方API的该新版构建与所需Python/CUDA组合不匹配；没有安装失败解算方案。
- 按官方main label构建，选择精确FAISS1.11.0 `py3.11_h4818125_0_cuda12.1.1`及libfaiss1.11.0 `h4818125_0_cuda12.1.1`。初始兼容解算70包、约1.38GB，包含不必要的pytorch-cuda metapackage；未安装。
- 精简解算移除该meta，固定Python3.11、numpy1.26.4、cuda-version12.1，48包、725,327,110 download bytes，dry-run exit0；不含torch/pytorch或driver包。实际cudart12.4.127和cublas12.4.5.8来自nvidia，不能因build标CUDA12.1就声称所有运行库精确12.1；本机driver下实际GPU运行已验证。
- 依赖URL域名仅conda.anaconda.org，channel限定pytorch/nvidia/conda-forge；由dry-run结果生成完整48包@EXPLICIT列表，每包带publisher校验值，不在安装时再次随意解算。
- `micromamba --no-rc create -y -p cache/faiss-gpu-env -f cache/faiss-gpu-explicit.txt`，MAMBA_ROOT_PREFIX仅本项目cache/mamba-root；start18:17:46.658937 UTC、end18:18:29.115466 UTC，exit0。
- @EXPLICIT SHA256=`2076a4200311d2efc1cae4b206d272676674900abb0ee682a240fd817a999b9e`。完整实际48包版本/build/channel/hash/license在`cache/faiss-gpu-validation.json`；FAISS/libfaiss为MIT，CUDA/Intel runtime含各自专有许可，不把整套环境称MIT或随bundle再分发。项目LICENSE不变。

关键FAISS artifact：GPU包SHA256=`d1185ef05f317c9f68c9dc1d9d6490cbd23556f81f779d41f0df7dc697d422a6`，libfaiss SHA256=`0924601ba80c14b5946031529975da6128f42fcb527388ed397b086d0a1a3376`。

### GPU实际查询验证

2026-10-06T18:19:12.683066 UTC，exit0：Python3.11.17、FAISS1.11.0、numpy1.26.4、get_num_gpus1，8×8 one-hot float32输入，CPU IndexFlatIP → index_cpu_to_gpu(0)后search top1。GPU/CPU索引与分数相符，独立oracle为每行自身index及分数1；结果全部正确。temp GPU resource memory上限16,777,216 bytes；建索引/查询wall-clock0.441096841秒，非性能benchmark或训练。验证JSON SHA256=`9892144a9c0f113bf146f07cf40b5b7e89b0ff5f6436044e1c37e8c535c321f2`。

P1 .venv和既有torch均未修改；GPU conda环境、CPU wheel环境与P1环境分别隔离。CPU wheel1.15.1仍不支持GPU，不混用版本。

### tmux synthetic SSH断线保护

第一版嵌套inline命令返回tmux启动exit0但没有日志；后续检查FileNotFoundError exit1，不能算Passed。原子进程stderr未保存，不能确定单一原因；引号/换行构造为排查假设，不冒充已证实根因。

第二版使用唯一`cache/p2-tmux-v2.py`（仅验收synthetic临时文件，不进入仓库生产脚本），本机compile及远程py_compile先通过。`tmux -f /dev/null -S <project-cache-socket> new-session -d -s synthetic-probe <python script>`隔离socket/config，stderr输出独立cache文件。启动后检查has-session与日志存在，再结束SSH；不修改平台supervisor服务，也不暴露网络服务。

- 首个会话检查UTC18:14:24.869752：进程alive，1条日志。任务8步，每步间隔2秒；随后SSH退出。
- 新会话18:15:48.589955复核：8条step及DONE，stderr0 bytes，进程正常完成；第一步UTC18:14:23.887332，末步18:14:37.888839，晚于首会话退出命令。
- 日志SHA256=`ab9a5db6437dc8eb3b457e854535d52cba2923e24ab248ec632031354aeae54e`；脚本SHA256=`c3abe6fb52b1ec1a874ed1e7626f9218861864f7671b7e9a3d532fe0f77cf260`。日志/stderr/script/socket保留，不删除旧证据，不留长期训练任务。
- 只证明此16秒synthetic任务不依赖SSH会话存活，不证明任意训练checkpoint/resume、stop/start或主机故障恢复。

### Quota只读边界

findmnt查询exit0：workspace为overlay，visible quota mount flags为空，quota/xfs_quota工具未提供。statvfs total500GiB、free充足是可检查容量证据；不能据无quota标志声称没有隐藏宿主/平台硬配额。独立hard quota仍Unknown，未改mount/cgroup/sysctl。

## 可审批恢复方案 — 冻结46文件小bundle，尚未备份/恢复

仅在服务器准备，未传回Mac、未restore。下列是18:16:55.993552 UTC的冻结文件快照，不是文档最后修改后的live文件hash；bundle已冻结，后续文档更新不改变它。包含Git tracked公开原创文件以及明确列出的公开source/inventory、synthetic marker与P1 synthetic产物；不含.git、raw data、npz/parquet/embedding/checkpoint、任何环境、driver、凭据或连接信息。

- 46 files，uncompressed279,765 bytes；tar.gz86,365 bytes。
- Server bundle：`cache/p2-recovery-20261006t1817.tar.gz`，SHA256=`68c0eedc62a111b397379bc9c814c553b1d2128833b7a1d75fa3738f4e127a5d`。
- Plan：`cache/p2-recovery-plan.json`，SHA256=`ad9cfe247c3f5656d2cc1229ce522a2b905cc78af3107f9975ada55f7a5e4936`。
- 制备18:18:34.765955 UTC，exit0；每成员大小/hash与清单一致，成员是常规文件、相对路径、无..；tar uid/gid0且uname/gname空，mtime0，不携带server owner身份。扫描未见IP或私钥header，文件扩展/目录白名单排除大产物与凭据。
- 原数据保持原位、未删除。这个bundle包含较早治理快照，恢复时以该已审阅快照为准，不谎称包含最终P2最新状态。

| Bundle成员 | bytes | SHA256 |
|---|---:|---|
| `.github/pull_request_template.md` | 1538 | `5514c7993eab015f854f7ba10c3c552473f8c38b627e7bb91622f579028ef46c` |
| `.gitignore` | 1816 | `92f07235398e8db94194caba1e47d7a74bdec3fb50e4718636e38844c02f7bd8` |
| `.python-version` | 5 | `49a506dd32096b010d75205acf3430c9ae6c40351888129499e5a5e487126c93` |
| `AGENTS.md` | 3750 | `0bbfbc84dda33d58f88844008c16fe24ba4da1e48e0fe098e7adaaa2f652fe55` |
| `CLAUDE.md` | 134 | `e71d365e9d1ee9dee07dfde80de9b4de9a461e87da3e75520b94eeeaaca8c486` |
| `README.md` | 5323 | `d0e0aad0b35035494d0572cd1cb361ab12aaa766e9320c69d775e6d6292fd771` |
| `STATE.md` | 4378 | `14ae524aa4a74978cdc0b2769d89149cb3c8cf89e4aa46e7acc1a208da8fa0d5` |
| `artifacts/README.md` | 655 | `e79c0a740c9dd1da0dc97c88ad04f90998429f32b76ca8ac569111684bdf24df` |
| `cache/p2-inventory.json` | 6647 | `431a0bc3d74356f1f2b698dfe6594441221a6ba6c9c7d0ae921ae30c3bbd17d6` |
| `cache/p2-recovery-marker.txt` | 49 | `b1b49b3b269e4b59e2dec14cde925536efbd231c2d70495c845b03d8b1529bef` |
| `cache/p2-source.json` | 7876 | `f034c2ee23cb6249d7a0b03f2e6037ada15db228d84fc7ff02affe0b89398f51` |
| `configs/mvp.yaml` | 807 | `9fd1a23c963e74ba457011ddffdfbd9d48b24e7e6d6ac1bbaf4ed7b079f4ce87` |
| `configs/scale.yaml` | 948 | `042ee38cc81896ab53985cdd18142ae27e2d0fdb803689b8187b6d0c3d1d49c2` |
| `configs/smoke.yaml` | 453 | `f36adf05e53af17c2e18a893f0c766955c0a9912ab3325b6ad7ccc30020d5664` |
| `data/README.md` | 1277 | `1abda429a503f905fdfbe4fa3ab1410edc91e18db472efcea80a6d48eb9e7165` |
| `docs/architecture.md` | 6546 | `e11eeefde7958879056f526f2e3cfca409aa95a13cb24922d9f8695648bef03c` |
| `docs/codex_handoff.md` | 3593 | `63d02e3888418e2c787ff3242cf5983b496be4a5dd593f98d26e7b016c0f6b53` |
| `docs/data_contract.md` | 8027 | `ee7f64d5f3365df4e278e4b1145f4f8d12d1826010009c0f640ead4e01d40e33` |
| `docs/decision_log.md` | 15119 | `83e5a07c7740b405565119baf72846877458f41bb903a2c762fde41c1c0e2124` |
| `docs/experiment_log.md` | 3369 | `d75302128dab0427674e5e8d2346a64c98f84c2f34e3d2999d08b79a53b67fb0` |
| `docs/issue_log.md` | 14669 | `9ec0e102eb300d7291e2bdb010ae849241bb97dba61079b5952f1b9b201c3db6` |
| `docs/p1_validation.md` | 80519 | `ef8f56463513150ae37d5ccd0efc65f6c98e87ac1c26df1d02f0e3cd95593a62` |
| `docs/p2_validation.md` | 25307 | `2411e8a2ba3fa153cb36772843247be67f1220071fadf43d38424481564f6d72` |
| `docs/project_charter.md` | 4882 | `01ed8700758115460a19117283a57ac42322dda2ae7a5d0941b765e4003a93ba` |
| `docs/server_runbook.md` | 7906 | `6f4515956303ec59a10f9f0143f6b1822f74c5568cabdc72acfe59c5d2de5971` |
| `handoff.md` | 17530 | `fc59deebc03477f88b2e5142da7ffdf53ca45bb7b9962e0de0cb0b0eae150ff1` |
| `outputs/README.md` | 766 | `1de3b576486f31343eb1a745c7a430aebcf82d51ff44eefbbd2aacef4814f116` |
| `outputs/p1-p2-20261006t1755/result.json` | 226 | `565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28` |
| `outputs/p1-p2-20261006t1755/run_manifest.json` | 1594 | `41f6918a61857d19c6f2c393d598b7c791eb6bd881eea3a68b688715ff460645` |
| `pyproject.toml` | 179 | `1b5b67d69b79b2c3d8afaf4f1bd8932887cf72dbd58b5c49f5b2bc9aa0da1db6` |
| `scripts/README.md` | 655 | `c2ec0a19249779c40a9811ad4368f8e1d05b7225edaa678557ed1fb8a0de0de6` |
| `src/vkgr/README.md` | 922 | `9373937d0dc4f2d578c595d8a8bb016d3c5344e3f7c34bddcd79c81c857fb8ba` |
| `src/vkgr/__init__.py` | 33 | `323fedc682877b25bcd5250f80b22c8d9b213af4d5caaff2494e855ba9e6f1bc` |
| `src/vkgr/__main__.py` | 1557 | `b3261b3886687607d3cc83ebe722bf69719200e17d1d8bd2195d001f8247ec59` |
| `src/vkgr/config.py` | 2491 | `540a878c5cfc8945bdfcc083609770e32cae92cebf566f980f9235fe0ba24546` |
| `src/vkgr/manifest.py` | 7710 | `3b06a12b3129aec87320eebbcc17a5859814e99fb479d57d10a6a8febe415f17` |
| `src/vkgr/runtime.py` | 2987 | `ede19b05543327ec790f4c59301bd05006c750f89e4c149d75f0193cb4d8f4e8` |
| `tasks.md` | 9112 | `b0e2ae313b30159c8622fced6602c5ea3c98484b226e1fa36a59f78b276f79f9` |
| `tests/README.md` | 1198 | `51f59ce75a12626744b41a4f1ff63aa0dfa515ff47b5d356b2965f251d678902` |
| `tests/fixtures/p1_synthetic.json` | 107 | `f4cf83b4302ca3a669fe63f1710e8d556fa0e97d18241466bf646f50ed832c90` |
| `tests/test_boundaries.py` | 4441 | `1828edec5e7e59668f9b691e800378366cdb46bf79254922edf948020b8cdfe4` |
| `tests/test_cli.py` | 3405 | `c5f83711d2da0bc807979967eed885ef80a42f1effbe615f49b37a5aaf727ca3` |
| `tests/test_config.py` | 3745 | `47a74a0a6bbacceb98e38f16f738999bf6ca7f3c261f3aa51e8bc9f50c31608e` |
| `tests/test_manifest.py` | 7699 | `e5b86cf8d7e6894e2f0462b682daec7ab1238d2e5c2018c50368c83448234839` |
| `tests/test_runtime.py` | 4297 | `ed7fedeea35f697ebc1aaab8464f29fefd3743c6732c213878baa7dac6fc6696` |
| `uv.lock` | 3518 | `1a778881fd23cf8d3b4215e66ac7f118427783767a043d6ae1a0cfa2c7ce1420` |

### 批准后拟执行的明确动作（当前未执行）

1. 使用SSH transport的scp将唯一bundle及公开receipt存到Mac唯一新`/tmp/vkgr-p2-recovery-<UTC>/`目录；先test目录不存在。只传86KB bundle与小JSON，不传raw数据/环境，不外传第三方服务。核对archive SHA256与上表成员hash，保留原件，不删除旧文件。
2. 将同一bundle/receipt经SSH返回服务器唯一新inbox文件名，核对archive hash；服务器原checkout、data和原bundle保留。
3. 在服务器全新`/workspace/vkgr-p2-restore-<UTC>` clone固定GitHub commit0376d9a；先确认目录不存在和git hash。预检archive成员集合正好等于46文件、无symlink/absolute/..，再仅在这个新checkout恢复指定文件；逐文件重算size/hash。
4. 在新checkout建立独立Python3.11/PyYAML6.0.2 lock环境，重新跑31tests、新唯一目录P1 smoke/verify，并对照synthetic result/marker哈希。新增环境为恢复目录专用，不借用旧环境冒充恢复通过。
5. 原始数据的灾后策略为按固定HF revision/30路径重新获取并逐文件SHA256验证；这不是把raw data带回Mac。此轮恢复是否实际重新下载2.44GiB，须Astra向Eddy明确列出并审批；原数据不删、不挪。

具体命令骨架（占位UTC与路径仅在执行时填入，不在本轮运行）：

```bash
# Mac: fresh backup directory, then SSH-transport copy and hash check
mkdir /tmp/vkgr-p2-recovery-<UTC>
scp vast-a6000:/workspace/vkgr-p2-20261006t1755/cache/p2-recovery-20261006t1817.tar.gz /tmp/vkgr-p2-recovery-<UTC>/bundle.tar.gz
shasum -a 256 /tmp/vkgr-p2-recovery-<UTC>/bundle.tar.gz
# Server: fresh restore checkout (only after approval)
git clone --depth 1 --branch main --no-checkout https://github.com/EddyHai57/vk-generative-retrieval.git /workspace/vkgr-p2-restore-<UTC>
cd /workspace/vkgr-p2-restore-<UTC>
test "$(git rev-parse HEAD)" = 0376d9a8119f3500cbf78519675cc7bf18013c1b
git checkout --detach 0376d9a8119f3500cbf78519675cc7bf18013c1b
# Verify archive member whitelist/types and hashes before extraction.
tar --extract --file=<returned-verified-bundle> --directory=. --no-same-owner
# Recheck all 46 member hashes against verified receipt, then rebuild/test.
uv venv --python 3.11 --managed-python .venv
uv sync --locked --no-dev
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-restore-<UTC>
PYTHONPATH=src .venv/bin/python -B -m vkgr verify --manifest outputs/p1-restore-<UTC>/run_manifest.json
```

这是目录级backup roundtrip/恢复与环境重建方案，不模拟实例stop/start、recycle、destroy或host-loss；固定源重下策略也不等于跨主机演练。当前runbook完整恢复gate差异必须由Astra向Eddy呈报，不自行缩小gate后宣布P2完成。实际backup/restore仍待用户确认，无新增收费存储、volume、关机或第三方上传。

## 本轮最终状态（取代此前GPU/长任务未执行口径）

GPU FAISS实测及Astra复核均通过；tmux synthetic断线保护及Astra日志复核通过。数据30/30与P1回归也有Astra独立fresh证据。Quota依然只限可见容量/挂载信息；恢复只准备可审批bundle，未执行备份回Mac/恢复，完整P2未Complete。原main与旧P1 worktree未修改，不commit/push、不训练、不P3。


## 已批准目录恢复实测（2026-10-07 Australia/Sydney）

本节取代此前“尚未备份/恢复”的当前口径；冻结46文件清单与较早文档快照保持原样。Astra转达Eddy“完全批准”前述小bundle Mac backup roundtrip/新目录恢复方案；没有批准stop/recycle/destroy/new volume、训练/P3或重新下载原始数据。

- Mac备份（受本项目/cache/**忽略）：`cache/p2-recovery-20261006t183146/bundle.tar.gz`及`receipt.json`；本机完整路径为`<WORKTREES>/p2-server-acceptance/vk-generative-retrieval/cache/p2-recovery-20261006t183146`。archive86,365 bytes，SHA256=`68c0eedc62a111b397379bc9c814c553b1d2128833b7a1d75fa3738f4e127a5d`；公开receipt SHA256=`4eb77350a253da357c2c8119b9c8c056052bc871e0a2c139f9b0f65c318bbc6b`。
- Server返回inbox：`/workspace/vkgr-p2-recovery-inbox-20261006t183146`；恢复新checkout：`/workspace/vkgr-p2-restore-20261006t183146`。先确认新目录不存在，读取/workspace/AGENTS及项目规则，GitHub固定HEAD=`0376d9a8119f3500cbf78519675cc7bf18013c1b`；只在新checkout恢复批准46文件。
- 三处archive（原server、Mac、returned server）SHA256一致。预检46成员精确白名单、279,765 uncompressed bytes，逐成员size/SHA256、相对路径、常规文件通过；拒绝symlink/hardlink/absolute/..，排除raw/embedding/checkpoint/environment/credential及扫描private key/IP/token/signed URL。手工逐个写入新checkout，不信任tar路径自动解包。
- 46/46恢复后hash匹配，环境/测试结束又复核46/46一致；synthetic marker hash沿用冻结表。实际恢复完成UTC `2026-10-06T18:32:18.933725+00:00`；测试UTC `2026-10-06T18:32:44.792820+00:00`～`2026-10-06T18:32:47.531796+00:00`。
- `uv venv --python 3.11 --managed-python .venv`与`uv sync --locked --no-dev`各exit0，恢复目录独立Python3.11.16/PyYAML6.0.2，未借用旧.venv。uv.lock在包内、hash保持；Python下载cache可能被uv复用，非完全离线重建或无共享cache演练。
- `PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v` fresh **31 tests in1.634s，OK**；新唯一`outputs/p1-restore-fresh`的smoke/verify各exit0，旧恢复`outputs/p1-p2-20261006t1755/run_manifest.json` verify exit0。
- 新synthetic result SHA256=`565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28`，与旧result一致；新manifest SHA256=`92f4e8175d3a1344f841dcbc18e1a1ea2165e90cc534ba434ca54742c48e9f8c`。完整脱敏执行报告留在恢复目录`cache/recovery-tests.json`，SHA256=`3d412256e5cf8953933e84e5de58e01948591a0a9db967c1bc34407fea4fd53a`；6步日志保留`cache/recovery-step-0.log`至`-5.log`，每步命令/exit/hash记录在报告。
- 原数据只做存在/size/ignore检查：30files、2,616,452,194bytes，全部git check-ignore通过且git ls-files data/raw/cache为空；未重复全量hash、未重新下载、未拷贝raw到Mac。固定revision/source/inventory已在包内，支持将来按批准范围重新获取。
- 一次真实失败：首轮恢复远程执行成功后，本机记录函数内exec覆盖expected变量，末尾KeyError('archive_sha256')、exit1；没有损坏恢复文件。只补正元数据，不重复备份/恢复，然后fresh环境/验证全部exit0。不能把首轮整体称exit0；后续可核查报告与hash为成功依据。
- 包内包含依赖锁，**不包含GPU FAISS/torch/driver/Python runtime/venv**；GPU环境重建未演练，原GPU验收证据独立保留。未删除/挪动/覆盖原项目或数据，未改变实例状态，未commit/push。

覆盖结论：目录级server→Mac→server backup roundtrip、46文件恢复、P1环境重建及manifest复验Passed，等待Astrafresh复核。该结果不证明实例stop/start、recycle/destroy、不同host恢复、GPU环境灾后重建或真实数据重获实测；完整P2 gate由Astra/Eddy裁决，当前不自行宣布Complete。可见disk容量通过，隐藏quota仍Unknown；费用仍Unknown且非阻塞。


## Astra独立恢复验收与P2收口（2026-10-07）

本节为最新验收结论，取代前文“等待复核/完整P2未通过”的当前口径，历史记录与冻结包不改写。Astra独立验收：Mac bundle86,365bytes及既有完整SHA256匹配；tar独立遍历46唯一regular members，路径安全、逐成员size/hash匹配receipt，不含raw或环境。UTC2026-10-06T18:34:03.552688+00:00开始，服务器恢复目录46文件独立重算/hash及HEAD0376d9a通过；fresh31tests in1.453s OK，新`outputs/p1-astra-restore-review` smoke/verify及原`outputs/p1-p2-20261006t1755/run_manifest.json` verify均exit0，result SHA256=`565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28`。

这些是Astra另一个对话独立执行并转达的证据，不冒充Sol重跑。其新恢复目录运行产物保留，不删除或提交。资源、环境、下载、GPU FAISS、tmux等此前已有Astra fresh证据。

P2 **验收通过（限已批准范围）**：tasks P2“经批准的平台持久化/恢复验证”按Eddy完全批准的具体目录级方案满足。不追加未经批准的stop/host-loss测试；不伪称全灾备。隐藏quota Unknown，只有可见空间证据；实例stop/start、recycle/destroy、host-loss、GPU环境重建与真实数据重获未测，保留为限制与P8再验风险。停止在commit前和P3范围门；无自动P3/训练/commit/push。

## 最终验收矩阵（当前）

| 验收项 | 结果与范围 |
|---|---|
| 固定GitHub commit、隔离Python3.11/PyYAML环境 | 通过 |
| GPU/CUDA与CPU对照 | 通过 |
| CPU/RAM与可见磁盘容量 | 通过；隐藏quota Unknown |
| P1 tests、smoke、manifest | 执行与Astra独立复核通过 |
| GPU FAISS | 1.11.0实际GPU检索，CPU/oracle及Astra独立复核通过 |
| SSH断线任务保护 | tmux synthetic8步/DONE及独立日志复核通过 |
| MVP下载/精确inventory/大小/SHA256 | 30/30，执行与Astra独立复核通过 |
| Dataset传输与有界direct I/O | 通过；不称持续系统benchmark |
| 已批准目录级Mac往返备份/恢复 | 46/46及P1独立环境重建，Astra复核通过 |
| 实例stop/start、host-loss、GPU环境及raw重获 | 未演练，保留后续风险；不在本次批准恢复范围内 |
| 费用 | Unknown；用户已撤销费用gate |

P2在本次明确批准范围内Complete，未commit/push，停在P3范围确认前。
