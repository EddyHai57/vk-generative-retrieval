# G full training backend：正式 UR001 adapter 实现

状态：**实现与本地 synthetic 验证完成；独立 C review、固定公开源码部署、真实 subset 对拍与真实 full PF 尚未完成。不得入队，不改 passes。** 下方原 preparation 报告保留原文，已由本节替代其当前状态。

## 本次闭包与科学边界

依据 `aa72a86` 第 7 条、`863f53a` full contract、`6f8b56e` 工程授权。只改 backend、其测试和本报告。C 档：新增 producer / own-job / predecessor 绑定与 UR001 context；由其他 reviewer 独立审查。

- 原 A1 `run` / `verify` / config 的 code object 保持不变；model、optimizer、EpochCursor、`train_batch_d`、evaluator、EarlyStop、checkpoint selection、runtime、resource guard 均为原对象。只替换 `_contexts` 与运载 `_binding`，后者明确叫 UR001 并绑定原 full SID producer，没有伪造旧 prepared / history / CONTENT_V1 metadata。
- config 逐字节核原三份登记配置，科学字段逐字段相等，只换 SID version / manifest 数据身份。K229 depth3、原拼 batch、batch1 decode、workers1 不改。GPU CSR 解码是另一项工作，本 adapter 没有假称已接入。
- 正式 CLI `plan / preflight / train / verify --inputs --request --request-sha256`，train / verify 还须显式批准开关。无 submit。role-only legacy dry 仅 admission；role-only PF / train / verify 继续拒绝，不能冒正式 PF。
- 正式 admission 先于 request / producer / payload 开启；query 只读 UR001-validation，全部 history 位置限 week00–24、target 限 train00–24 / validation25，ordinal / strict ordering / OOV 校验；不会构建或读 test query 表。
- UR001 原 manifest + upstream + 6 artifacts + 26 个物理输入、producer 原 stdout / terminal / job 一起绑定；SID producer parent.binding 的 stage2 manifest SHA 须等于同一个 UR001 manifest。
- final SID 使用 `UR001-native-full-SID-trie-v1`、五个闭合 artifact；调用独立 verify receipt 的原 `consume`，**0 次 sid.fit**。不调用会做第三次 replay 的 SID CLI verify。SID producer 实现候选 `7704086d6234103cfd35db9e8d7281ec403c3342`，entry SHA `3e7a1b1aa06d6e8b71269d0b702ad4e45a0ee1c5ec2f3be432f7db2bacc2876b` 硬 pin；完整源码 ledger / commit 再由固定 request 绑定，公开压平后的 commit 由 owner 生成。
- producer 在自己的固定干净源码部署中消费，不从当前 consumer 树任意 import；完整 filesystem ledger 拒绝 ignored shadow / symlink。consume 后回核 source、receipt bytes、codec、canonical catalog digest；用原 `build_codec_d` 重建 codec，并逐项比较保存 trie keys / key_items / bases。
- own-job 完整 argv / inputs / config / git / runtime / resource / numerical env 复用既有 P backend 的原封闭检查 code object，仅替换 G invocation 和 G input closure。PF→train→verify 的每一步独立原 stdout / terminal / receipt 链绑定；close 再核 source/runtime、UR001、SID、own-job 和 predecessor。
- 新缓存用独占 JSONL offsets mmap；训练仍逐 batch 读取原 row，不全物化训练表。原 resource guard 检查当前进程树、真实 CUDA 与挂载余量；进度沿用原 epoch / evaluation progress。

## p3c 数据通道读取位置

G 不调用旧 P3-D prepared / side / CONTENT_V1 loader。`stage_inputs` 从 private OPS `request.p3c` 核固定原字节 SHA，并列入 own-job inputs；原 `check_job` 在 context 前及 close 逐字节 hash。`execute` 仅把这一明确命名的数据身份带到原 positional API，新的 `_contexts` 不将它解析为旧合同或旧 content。G consumer 自身为干净 Git source ledger，**不伪装 docs/p3c_validation.md overlay**；P backend 的旧 native tree overlay 纪律未改。错 SHA 和遗漏 p3c 的反例均拒绝。

## 本地证据

- fresh RED：缺 `read_transport` / authenticated context接口，缺 G invocation，缺注册 producer entry 硬 pin，以及缺 fullB / G 共享 stage2 gate；各自最小 GREEN 后相关回归。
- 最终相关回归：**74 passed / 1 existing skip**；包括本文件 30 tests、storage、producer provenance、原 A1。原 CPU synthetic K229 codec 首 batch tensors / loss / 全部参数更新 / eval logits / RNG 逐位相等；新完整 context fixture 使用真实 codec / JSONL / S0，原 `run` / `verify` code object identity 相等。fixture 对 producer 的 mock 仅用于局部门控，不能证明真实 producer / PF。
- 原真实 launcher 代码的 synthetic 部署 → metadata → module dry / 拒绝链保留；本次增加正式 train批准门与正式 holdout PF 子进程反例。没有真实 launcher 正式 full request PF 的回执，不把 role-only 集成链冒充它。
- **9 / 9 隔离 mutants semantic kill**：config、history、共享 stage2、trie 比较、p3c、producer entry、显式批准、G module argv、先行 admission。每个独立全树 clone、唯一缓存、`python -B`、`PYTHONDONTWRITEBYTECODE=1`、pytest cache disabled。初轮 trie 反例被另一个 stale manifest 身份门挡住，保留该轮日志；仅同步坏 fixture 的合法 SHA 后，完整九项独立重跑全 kill，不改生产门控。
- 默认本机沙箱 import torch 因 shared-memory abort；保留原件，未改 KMP / Torch 参数。原样使用已批准的提权本地测试成功，不访问服务器。数值线程固定原单线程环境。
- 本地原件：`<LOCAL_REVIEW_CACHE>/g-backend-final-regression.log`、`<LOCAL_REVIEW_CACHE>/g-backend-isolated-mutants/summary.json` 与逐项 log；owner 保留真实路径。`git diff --check` 无输出。

## 尚需 owner 执行

独立 C review；最终 public source / 请求 pin；真实 A40 subset 原路径对拍；engineering=False 正式 module full PF 与 holdout negative、真实 launcher metadata / dry；取得实际 final SID 独立 verify 产物后再核全闭包。实现测试不是科学有效结论；没有读取服务器、真实数据、checkpoint、test，没有训练或修改任务 passes。

---

## 原 preparation 报告（原文留存，当前状态以上文为准）

# G full training backend preparation

状态：**preparation only；正式 full SID/trie 协议缺失，formal preflight / train / verify 均关闭**。本报告是实现说明，不是独立 C review 或科学验收。不得把 dry 的 exit 0 当作 PF 成功，不得将任务 `passes` 改为 true。

## 范围与依据

依据 decision_log `aa72a86` 第 5、7 节、UR001 full contract 第 84–85 行及执行者对工程适配边界的确认，只新增 backend、测试和本报告。未改旧入口、科学 config、seed、门禁、epoch / batching、数值实现或已有文件。复核档位 C：未来正式 context / identity 绑定接线必须独立复核。

- `registered_config` 逐字节核对原 A1 三份 seed 配置，只替换数据身份 `sid` 和 `sid_manifest_sha256`。UR001 K229 depth3 使用新身份，不能冒用 subset 身份。
- JSONL 仅复用已审 `p_full_compact_inputs.JsonRows` 的 storage；逐 batch 返回原 row，保留调用顺序和重复索引。偏移表 mmap，不构造全训练 rows，也不复用 P 的科学计算。源文件变化、非法 / 越界索引和 bounded holdout 路径均拒绝。
- S0 adapter 沿用原 `_counts`、原 nearest-rank 定义和精确 `SliceStatistics` 类型；完整 catalog、typed counts / thresholds、query OOV / user 身份机械核对。聚合仍调用原 aggregator。
- diagnostic context 明确为 synthetic / engineering，只服务原函数对拍；它不是正式输入签发路径。
- native facade 首先复核原 native source ledger。原 `run` / `verify` / config / binding bytecode 对象不变；model、optimizer、cursor、train、evaluation、runtime 等原函数对象不变。新 `_contexts` 无条件拒绝 final SID/trie acquisition，formal science 不会执行。
- 正式 module CLI 先做原 raw-role admission；`dry` 只回非空 admission 回执，`verified=false`、`engineering=false`、`data_opened=false`。其余科学命令非零返回 `FINAL_SID_CONTRACT_UNAVAILABLE`。holdout 误指先返回 `HOLDOUT_INPUT_REJECTED`，不触及 SID 或 data loader。

## 验证

使用项目 Python，`python -B`、`PYTHONDONTWRITEBYTECODE=1`、单线程 BLAS / OMP、pytest cache 禁用。

1. fresh RED：新模块不存在时 collection ImportError；补严格 index 反例后 RED，最小修复后 GREEN。
2. 新文件 13 tests；连同复用 storage、producer provenance helper、原 A1 回归，共 **57 passed，1 existing skip**。命令：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <PROJECT_PYTHON> -B -m pytest tests/test_g_full_training_backend.py tests/test_p_full_compact_inputs.py tests/test_producer_dag_binding.py tests/test_sgns_subset_a1.py -q -p no:cacheprovider
```

3. K229 depth3 toy codec + 原正式 seed42 architecture / batch config：lazy vs 原 ListExamples 的首 batch tensors、loss、全部更新参数、eval logits 与 torch RNG **逐位相等**。这是 CPU synthetic 对拍，不是服务器真实 subset 或 full PF。
4. 原 OPS `5840e6b686ec9651314a9e28a1a2c0befcc05158` 的真实 `launcher.prepare`：临时 committed CODE、OPS 保留 public 副本、固定 sha overlay 的数据通道布局，metadata → 同 module dry → 同 module preflight 拒绝。无 mock、无 submit。public / overlay 均为明确 synthetic fixture，**不是正式 p3c 字节核对证据**。
5. **6/6 隔离变异 killed**：batch 反序、config 判定失效、final SID 假接受、threshold 值比较缺失、formal engineering 标志错误、storage holdout 门失效。每个 mutant 使用独立源码新拷贝、`python -B`、禁 bytecode、唯一 cache prefix；未共享缓存。机器可读结果由执行者保存在本地验证目录，路径不上公开 Git。

## 未完成与恢复条件

- 没有已冻结 final UR001 SID/trie producer 的 schema / loader / exact pin 协议。`require_final_sid` 因此无条件拒绝；caller 的 `verified=true`、K229 / depth 字段或旧 subset bundle 都不能放行。
- `producer_dag_binding.check_consumer` / `verify_producer` 只复用现有 helper，未建立 UR001 → final SID/trie → G consumer 的正式科学闭包；不能用其既有 regression 代替此项。consumer 自身 source / runtime / lock、producer 原 manifest + run / terminal / stdout 仍须在 final 协议可用后逐项接线及重读复核。
- 正式 data context 尚未签发：train / validation / catalog / supported / S0 的完整 producer 绑定、真实输入语义与 pinned DAG 仍待实现 / 验证。未读取真实数据、checkpoint 或 test。
- native `_source` / `_runtime` 未替换为 final consumer 闭包，也未生成正式 job；当前 formal `_contexts` 先行关闭，不能暴露旧 subset ledger 为 full 证明。
- 下一步：冻结 final SID/trie producer 协议后，以最小新增逻辑替换拒绝，完成 producer / consumer 独立 exact 闭包和 UR001 context 签发；独立 C review；由主执行者 push 固定 SHA，真实 subset 原函数对拍、工程 false 的只读 full PF 与 holdout negative。禁止部署本 preparation 为正式 full PF。
