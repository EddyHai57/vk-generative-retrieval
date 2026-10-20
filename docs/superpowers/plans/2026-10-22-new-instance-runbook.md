# 新 Vast 实例从零恢复 runbook（今晚只准备，未执行）

授权：decision_log ecfcf8c/feb194e；明天 Eddy 批准具体新实例后执行。旧实例已完成 Stop 前最终只读盘点，本机连接已结束；实际 Stop 尚无回执。**今晚不执行本页任何连接、环境安装、下载、过滤、CUDA 或真实对拍命令。** 不销毁旧实例，旧数据/checkpoint不迁移，保留到新下载核对和首个作业跑通；删除/销毁仍另批。

## 0. 明天开工与容量准入

1. Eddy/Kiro 确认实例：CPU≥16核（建议32）、RAM≥192GB、盘≥1TB、单卡VRAM≥24GB；不选 vGPU 切分卡测成本。付费操作由 Eddy，Codex 不租机。
2. 连接前由 Eddy 提供新实例身份；不复用旧 socket/owner；真实地址/端口/密钥仅私有配置，不进 Git/聊天。确认是**新机器**、所有目标根目录都不存在。原 spent/deadline/名额不重置；新工程run用新 ID，不复投旧实例 ID。
3. 首次只读记录 GPU/driver、cgroup cpu.max/memory.max、disk free/total 与 mount。仅厂商标称不算验收；driver须支持 CUDA12.8。每作业保留 CPU/RAM/VRAM/磁盘≥20%余量；显存以新实例 context-inclusive 峰值×1.15向上256MiB，首次保守声明标 not_measured。

## 1. Git / deploy key / 固定部署

- 公开工作分支与最终 SHA 见夜间汇总报告；先 Git fetch，再新 worktree 固定 SHA；逐文件 SHA 与 Git blob 比对。禁止服务器手改代码、复制 ignored 代码或用未推送脚本。
- 新实例 read-only deploy key：生成新的 ed25519 key，私钥只留新实例权限600，公钥由 Eddy 添加为仓库只读 deploy key；本机登录私钥不拷贝，known_hosts 从可信来源核验。可先 HTTPS clone 公开仓库，不能把可写 token 放 remote URL。
- 科学 anchor 参照源码仍固定 G=b5a4d1c99ba2d8361a2d90c23bac27af5b71769d、P=0ca61bc35ad68bd79b86103b54e470445cbab07f。各自新干净 worktree；用 `configs/anchor-profile/source-pins.json` **全部** source ledger 检查，不只核 HEAD。
- 原 p3c 合同 c89b6466…3274b5f 只能通过已有获批数据通道还原；Git 脱敏版按 §4A 原字节存 OPS。若新机器没有数据通道原字节，STOP 该入口、记待 Kiro/Eddy，不从旧服务器拉取、不从公开 Git 猜造内容，也不以脱敏版替代。此项是实际部署前置。

## 2. 环境：原 lock，不使用 scaffold lock

当前主 checkout 的 uv.lock 是 scaffold，不等于 anchor lock。新增 `configs/new-instance/anchor-uv.lock` 是从本地既有 b5a4d1c 对象逐字节导出的原 lock：
`8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063`。
配套原 pyproject 在同目录；均为新增副本，没有修改冻结文件。不得重新 resolve 或“升级到相近版本”。

在新实例、固定 CODE 根下，ENV_ROOT 必须不存在：

```bash
bash scripts/new_instance_environment.sh --execute-on-new-instance "$CODE" "$ENV_ROOT"
"$ENV_ROOT/.venv/bin/python" -B -m analysis.new_instance_prepare cuda-smoke --lock "$ENV_ROOT/uv.lock" --execute-on-new-instance
```

严格验收 Python3.11.16、NumPy1.26.4、pyarrow19.0.1、PyYAML6.0.2、torch2.11.0+cu128/CUDA12.8，`torch.cuda.is_available()` 与小算子结果。所有环境/代码/lock/SHA写非敏感收据；不符停止，不 CPU fallback。本地夜间测试的3.11.17/macOS只用于 synthetic，不替代此验收。

## 3. 调度服务与只读 status

运维源码独立新干净 worktree，固定原状态 CLI/调度协议 commit `5840e6b686ec9651314a9e28a1a2c0befcc05158`；INFRA-04 的 launcher job-pin 覆盖若具体入口需要，按原工程卡另部署，不把两个树拼成未登记源码。读 `ops/README.md` 与既有 `.agents/tasks/infra-01/design.md`。

```bash
"$PY" -I "$OPS/ops/bin/vkgr-status" --root "$OPS_ROOT" --json
"$PY" -I "$OPS/ops/bin/vkgr-launch" --root "$OPS_ROOT" --run-id new-instance-scheduler-20261022-v1 --kind ops --cwd "$OPS" --cpu-cores 0 --ram-mib 256 --vram-mib 0 --disk-gib 0 --wall-hours 48 --stop-mode signal --no-admission -- "$PY" -I "$OPS/ops/bin/vkgr-sched" serve --root "$OPS_ROOT"
"$PY" -I "$OPS/ops/bin/vkgr-status" --root "$OPS_ROOT" --json
```

以上 scheduler 必须先确认run/queue均无该ID；launcher自身detached，核 supervisor PPID=1/session、非空run.json/owner、资源声明与有限截止；不把 action=written 当 claimed。status 空值/缺失写 UNKNOWN，坏收据不算通过。只读工具也固定SHA，私有 SSH config只在本机，今晚不创建/恢复连接。

每个后续数据/工程入口先走正式 `python -m ...` 的真实只读 preflight、同版 launcher.prepare→metadata→dry链，核run ID在runs与queue各子目录均不存在，再唯一submit、claimed且无rejected；失败按原规则，不复投。夜间synthetic CLI不替代此链。

## 4. 下载 ur0.01（固定29文件）与 week26（固定单文件）

新实例只用独占新目录。为保留现有正式白名单和 G 数据根路径身份，可在**新文件系统**重建原规范根名（先确认不存在）；不能在旧实例执行这些路径。`configs/ur001-preparation/science-inputs-closed.json` 的 A_ROOT 只含train/validation/metadata，不含raw/member/holdout。

`analysis.new_instance_prepare dry` 只输出完整30文件size/SHA清单，不联网；29文件清单及 SHA 保持原样，week26固定 size/LFS oid 来自原裁决。实际下载前 worker 会再次核HF tree 的LFS oid/size，下载后逐文件hash。所有失败部分保留，不覆盖/删除/自动续传到旧文件。

```bash
"$PY" -B -m analysis.new_instance_prepare dry
"$PY" -B -m analysis.new_instance_prepare download-a --root "$A_ROOT" --reference-root "$ISOLATION_REFERENCE_ROOT" --execute-on-new-instance
```

ISOLATION_REFERENCE_ROOT 是预先存在、与A_ROOT分离的路径隔离参照目录（可为空），这里只用于原下载worker防止覆盖/路径嵌套的检查，**不表示已下载子集或完成旧子集覆盖核对**。子集对拍所需参照另下载。

完成必须有原 download-manifest、29文件size/SHA/LFS逐项一致、总量守恒；不重跑原 audit 或旧PF ID，不算反馈/指标。

## 5. week26 重做：严格复用原有界 worker

在独立本地请求文件中填写四个绝对新实例根（配置不放数据根内部）：

```json
{"schema":"ur001-week26-request-v1","roots":{"a_root":"/workspace/vkgr-ur001-data-20261020-v1","raw_root":"/workspace/vkgr-ur001-week26-raw-20261021-v1","membership_root":"/workspace/vkgr-ur001-week26-membership-20261021-v1","holdout_root":"/workspace/vkgr-ur001-week26-holdout-20261021-v1"}}
```

过滤请求额外加 `expected_users_sha256`=`541ca824327f3a714fb078e26d2f0cdebcdafdd8fc1b19c5e0a507ffa2996fd9`，其他键不变。四根不得相同或嵌套。

顺序逐项正式 metadata/dry/唯一新job：

```bash
"$PY" -B -m analysis.new_instance_prepare source-preflight --request "$BASE_REQUEST" --execute-on-new-instance
"$PY" -B -m analysis.new_instance_prepare prepare-users --request "$BASE_REQUEST" --execute-on-new-instance
"$PY" -B -m analysis.new_instance_prepare download-week26 --request "$BASE_REQUEST" --execute-on-new-instance
"$PY" -B -m analysis.new_instance_prepare filter-preflight --request "$FILTER_REQUEST" --execute-on-new-instance
"$PY" -B -m analysis.new_instance_prepare filter --request "$FILTER_REQUEST" --execute-on-new-instance
"$PY" -B -m analysis.new_instance_prepare check-rebuilt --receipt "$FILTER_STDOUT_JSON"
```

raw size=14300986995，SHA/LFS=`90c44ae75084df74b1c0251cd87daa6aa45d66377a77d1022027486c9a8b6993`。规范化用户集合取 A train∪validation 全用户，升序整数compact UTF8 JSON无LF，SHA必须与上文逐字节相等。过滤后四项计数必须与原验收：rows14881823/users81790/items1183510/union12467294 **全部相等**；不相等 STOP 回Kiro，不改代码/预期以追平。过滤parquet及manifest SHA都记录；Parquet writer/runtime若不同可能导致文件SHA不同，不能只凭计数忽略，先按身份问题停止核查。

仅读ID/过滤，不统计反馈，不构造query/history、不推理。raw、membership、holdout分别独立根，原raw永久保留到另批。正式train/eval/verify输入白名单不得含holdout。新实例用同正式science CLI补做holdout误传负例，要求 `HOLDOUT_INPUT_REJECTED`、非零退出、data_opened=false；合法dry只准入，backend仍关闭。fixture负例不是真实前置验收。

## 6. 子集参照与 1a→1b→1c

1. 在新实例重新下载原 `ur0.01_ir0.01` 参照、逐文件身份验收；重建prepare/history/C1/结构SID等参照的manifest DAG。原p3c数据通道、旧源ledger与绝对数据根若无法还原即STOP该入口，**不能搬旧checkpoint/run、放宽hash或偷换全量数据作对拍**。从零重建的SGNS/SID等是否需要额外工程卡/成本，先交Kiro确认，不为跑通擅开科学训练。
2. **1a parity**：用两套同输入原 profile request（原source/config/input/runtime pins严格相同；scratch/output均新），分别启动独立进程。新模块不改旧source；clone step只在内存绑定构建器。request模板沿原 profile schema，不能删源账本/p3c字段。每路original/cached-cold/cached-warm各自独占输出；默认8步，覆盖不足用获批fixture补，不靠改科学batch。

```bash
"$PY" -B -m analysis.tensor_cache_parity dry
"$PY" -B -m analysis.tensor_cache_parity capture --request "$ORIGINAL_REQUEST" --output "$ORIGINAL_TRACE" --mode original --steps 8
"$PY" -B -m analysis.tensor_cache_parity capture --request "$COLD_REQUEST" --output "$COLD_TRACE" --mode cached --cache-state cold --steps 8
"$PY" -B -m analysis.tensor_cache_parity capture --request "$WARM_REQUEST" --output "$WARM_TRACE" --mode cached --cache-state warm --steps 8
"$PY" -B -m analysis.tensor_cache_parity compare --original "$ORIGINAL_TRACE" --cached "$COLD_TRACE"
"$PY" -B -m analysis.tensor_cache_parity compare --original "$ORIGINAL_TRACE" --cached "$WARM_TRACE"
```

先真实CLI/preflight/同launcher链再准入capture。逐步tensor dtype/shape/stride/byte digest、参数、反向/clip后梯度、optimizer、loss、cursor、全部RNG都等；任一差异STOP/ALERT。trace只留新实例私有工程目录，不进Git。冷/热同口径短剖析仍须另出10+100+10工程收据；当前capture不报告吞吐或峰值，不充当剖析完成。
3. **1b G decode/verify**：只用已授权范围内重新建立的anchor best checkpoint与固定512条validation query，只计时不计算/输出召回；先独立工程卡及真实入口准备。今晚未实现/执行该新入口，当前不得直接开跑。
4. **1c P synthetic exact evaluation**：按942a5b0独立工程入口，用12.0M×64 synthetic向量与正式query数量、原blockwise_topk路径计时，不读真实指标；卡/资源准入后执行。今晚未实现/执行该入口。1b、1c证据未齐不编造完整全量ETA。

## 7. 下一阶段门禁与收口

- SGNS算术已达数十小时情景，见experiment_log；提速/口径调整先回Kiro，不先跑全量SGNS。
- 合同草稿的validation/test history差异与同周顺序规则待Kiro；未冻结不生成科学派生物。BHKM只设计冻结，不实现。
- 盘账加入raw/metadata、两份week26、derived/SGNS/SID/trie、外排临时与全部checkpoint；逐路径保留，不删旧产物凑盘。sched_forecast只在真实作业表已冻结后实跑。
- 首批全量P/G科学训练仍等Eddy明确入队确认与卡；test正式读数仍另批。旧实例继续作为备份，不自动destroy。

每项填写：source/config/data/lock SHA、run ID、UTC、正式argv/metadata/dry/claimed/no-rejected、原始stdout verified与MF绑定、资源峰值口径。未执行项一律NOT_RUN；本runbook本身不是已通过的服务器验收。
