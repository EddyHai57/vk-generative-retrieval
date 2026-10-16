密级：公开

# 自生成难负样本 DPO：工程实现与吞吐入口

状态：待一次独立 code review；未部署、未实测 CUDA 吞吐、未冻结 R2、未提交训练。

## 范围与科学源

基于 `gtok01b/g` commit `fdc988443672dd729a81dde6c0e7ca42a448cd13`。A1/T3 config 为 `configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json`，文件 SHA256 `9a938cf3ac722e0ad11fd8136d5a037b0f44516cf53e4df85a8188770dca737e`。该文件逐字段保持原样。仅新增 objective primitives、train-only 输入读取和 microbenchmark CLI；没有 R2 两阶段训练编排、没有修改原入口或任何 gate。

`ranking_r2.py` 提供全 SID 序列 log probability 求和、β=.1 的 DPO + 权重 1 的正样本 CE、独立复制并冻结的 reference、固定 seed0 抽 20,000 条无放回训练样本和 beam 顺序负样本过滤。CE 参数应传与原 SFT 相同 reduction 的正样本 next-SID CE。无合法负样本返回 None，不自动补样或改 beam；正式 R2 数据集缺样策略仍须明确记录后再接训练。

## 历史 like 的来源

负样本排除 target 与完整 `past_likes`，不能排除 observed history 来代替点赞。

- `history_v2.py:79–90`：`seen` 是用户从训练开始到 target 前的未截断 like set；先对 target 核 `past_likes == sorted(seen)`，再追加当前事件。
- `generative_data.py:310–313`：按用户验证完整 like prefix 递推。
- `history_v2.overlay_examples` 只替换 `history/history_positions`，保留 `past_likes`。
- 新 train-only loader 在 overlay 前复核相同的 prefix 规则、全局 target 顺序、catalog 和每份读取文件的 SHA。只读取 P4B manifest、catalog、examples 与 history manifest 的 `train-l50.json`；不用原 `_contexts`，后者会读取 validation。

## Owner 执行入口

独立 review 后，主 owner 按原流程 commit/push、指定 SHA worktree 部署并逐文件校验、准入与真实只读 preflight 后执行；不得在服务器直接编辑。使用既有 runtime venv。CLI 的 path 和 SHA 由 owner 从原已冻结收据填入，本文不写连接信息或本地绝对路径。

```sh
PYTHONPATH=src <RUNTIME_PYTHON> -m vkgr.ranking_r2_microbench \
  --prepared <P4B_PREPARE_MANIFEST> --prepared-sha256 <SHA256> \
  --history <HISTORY_V2_MANIFEST> --history-sha256 <SHA256> \
  --sid <T3_SID_MANIFEST> --sid-sha256 <SHA256> \
  --content <CONTENT_NPY> --content-sha256 <SHA256> \
  --config configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json \
  --queries 128
```

只接受固定 seed42 原 config bytes；`--queries` 只取固定 20,000 抽样序列的前缀，不改变正式采样名额。不足 20,000 训练行 STOP。真实输入要求 CUDA，固定 CPU 1 核、beam10、b1，随机初始化真实 A1 shape，不训练，不加载 checkpoint。不支持 validation/test/checkpoint 参数。只输出一行摘要 JSON 到 stdout；没有任何写文件代码，不保存逐 query 预测或训练行。主 owner 可把 stdout 重定向到新的工程 run receipt，不能覆盖原 run 目录。

收据包含执行 commit、科学源 commit、输入/config SHA、sample indices SHA、seed、model shape、runtime、wall time、吞吐与 20,000 条线性估算。解码 warmup 单列排除，主 wall 测量含负样本过滤。CUDA allocated peak 只反映单模型挖掘，不是 DPO 双模型显存声明。

## 限制与 stop gates

**随机权重结果仅为工程 shape/吞吐预演，不是新 R2 SFT best 的实际挖掘吞吐，不能据此声称正式 mining 已通过。** 权重会影响访问的 prefix、history 长度分布和缓存命中。线性估算不含加载、初始化和 warmup；阈值附近必须把这些额外成本纳入 owner 判断，不自动冻结/入队。估算超过 4 小时回 Kiro；不得自行改样本数或 beam。

正式挖掘只可读新 R2 自己 SFT 的 best；当前代码无此加载能力。DPO 双模型真实 shape 显存、后续 SFT/阶段2训练入口、verify 和科学冻结都未完成。该实现不授权读取旧 CP、改变选 best 或 early stopping 规则。

## 本地验证

既有 runtime venv、CPU、synthetic fixtures，`PYTHONPATH=src <RUNTIME_PYTHON> -m unittest tests.test_ranking_r2`。覆盖 DPO 手算/梯度、序列求和、reference 复制冻结/dropout、seed0 固定抽样、target 与完整 likes 排除、无合法负样本、无 validation/test 文件打开、SHA 篡改拒绝、benchmark 不改模型权重、CLI 参数边界与执行 commit 收据。没有 CUDA、没有真实训练输入验证。

本轮执行证据：新增 tests 10 个、`test_sid_depth_beam` 13 个、`test_generative_g6_unit` 12 个，共 **35 tests GREEN**（`PYTHONPATH=src:tests`）。额外 `test_generative_g6_v2` 的 13 个集成 tests 被基线已有 frozen 文档 hash 漂移阻塞：`artifacts/p3d-retrieval-contract-manifest.json` 声明 `docs/p3c_validation.md` 为 `c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f`，科学源 commit 中实际文件为 `5449378c3e2c00d5af4102da73e93a28839898107c139df110f29d50c321c466`。已用 `git show <SCIENTIFIC_BASE_COMMIT>:docs/p3c_validation.md` 核实该漂移在改动前即存在；未修改文件或重写 pin。该集成回归没有全部通过，不得宣称全 suite GREEN。
