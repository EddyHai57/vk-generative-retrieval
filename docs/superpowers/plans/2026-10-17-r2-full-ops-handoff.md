密级：公开

# 自生成难负样本偏好训练（R2）：owner工程操作入口

## 实现与 source 范围

基于owner更新后的 `974c6751f49a085680f4cbe75862bbf86b7d9f4e`；不改旧A1入口/config/gates，也不改已发布的R2 microbenchmark/primitives/tests。新增 `ranking_r2_config/data/sft/engine/run`、三配置、`scripts/ranking_r2_ops.py`与tests。独立SFT沿原A1/T3科学源 `fdc988443672dd729a81dde6c0e7ca42a448cd13` 的训练数学，只换新schema/source/data-root。完整规则来源见新增R2 frozen-evidence与主仓实验卡。

## Owner 顺序

一次独立code review → push工作分支 → 服务器只读fetch/指定SHA Git worktree → 全部source文件sha256核对 → 资源准入 → `preflight` → `cuda-preview` → `dry-run` → 入队前告知Eddy → 一次性 `submit`。

```sh
<RUNTIME_PYTHON> -B scripts/ranking_r2_ops.py preflight --request <PRIVATE_REQUEST>
<RUNTIME_PYTHON> -B scripts/ranking_r2_ops.py cuda-preview --request <PRIVATE_REQUEST>
<RUNTIME_PYTHON> -B scripts/ranking_r2_ops.py dry-run --request <PRIVATE_REQUEST>
<RUNTIME_PYTHON> -B scripts/ranking_r2_ops.py submit --request <PRIVATE_REQUEST>
```

请求与原始收据留在私有ops目录。请求字段沿已修R1协议：`deployment/source_commit/files/data_root/python/templates/pinned_manifest/queue/runs/ops/scheduler_ops/preflight_receipt/cuda_receipt/deadline_epoch`。`templates`是42/2024/2025原同seed A1/T3 claimed-job JSON；只读模板、config和其`--output/manifest.json`指标metadata，不读原CP。`files`覆盖 `ranking_r2_run.SOURCE_MEMBERS` 与新ops脚本；source/config变化后所有本次PF/CUDA/dry收据必须更新，不能复用974的microbench准入当作完整DPO准入。

- `preflight`：原G3/History/T3真实loader，engineering=False，accepted data root独立；全部原输入pins不变，锁定三seed新config和原control metadata SHA。阶段产物未产生前，只冻结static input binding；完整SFT/mining/reference binding由新run生成并在最终verify重建。
- `cuda-preview`：真实A1/T3 shape，先2个合成SFT step，再独立frozen reference与actor optimizer做2个合成DPO step；H50、effective512、micro256 pairs，包含chosen+rejected双序列；reference byte digest/无梯度/optimizer actor-only必须成立。beam100/b1与自身新工程state的内存reload一致。它不训练真实样本，也不读旧CP。
- 显存：按本PID nvidia-smi/context baseline + allocator max reserved/allocated与逐阶段own process总量，保守取总量峰值；乘1.15向上取整256MiB。proof需含reference/actor参数数目、reference unchanged和actor-only optimizer，与PF的source/data/SID/runtime/model shape及seed42 config SHA匹配。缺本PID/context或篡改字段即STOP，不用外界global free delta。不沿用R1的6656MiB声明。
- `dry-run`：核以上proof，再对三seed调用原 `launcher.prepare`；不改原gate。训练argv只更换新runner/config/output/scratch并加入control metadata路径/SHA，原data pins保持；重写PYTHONPATH到新DEP，固定data root、batched b1、单evaluator，清掉旧run进度env，由launcher给新run身份。
- `submit`：复用已修R1 seal-plan→detached PPID1/session+lock ready→bound release协议。controller零publication；driver唯一发布三intent并接续verify，poll900，sealed绝对deadline≤48h不重置。RPC kill before release零发布；after release driver独立接续；partial failure写ALERT，未pub不retry，已pub仍唯一verify，队列可见后异常要核完整job身份。不kill/restart/resubmit，不自动重跑训练/verify。

总训练输出 `sft/`、`mining/`、`dpo/` 和总manifest。总verify直接输出verified才接受；SFT阶段会先自身verify后进入mining。训练/verify失败保留旧attempt，不适用“工程运输失败一次重验”去重跑科学训练。最终三seed判定仍按原卡，主owner写experiment_log/Kiro裁决。

## 本地验证

```sh
PYTHONPATH=src:tests <RUNTIME_PYTHON> -m unittest \
 tests.test_ranking_r2_full tests.test_ranking_r2_ops \
 tests.test_ranking_r2_review_guards tests.test_ranking_r2_submission \
 tests.test_ranking_r2 tests.test_sid_depth_beam tests.test_generative_g6_unit
```

实际结果：60 tests GREEN。包括真实本地detached进程的RPC kill/第二publish异常/ready与spawn失败、参考冻结与optimizer隔离、两阶段synthetic train→verify、epoch0对照只报告/不作best、固定样本skip、跨seed在CP读取前拒绝、mining/hash篡改、原control metadata不打开CP、CUDA proof/reference/context/config gate。旧G6-v2 fixture受既有P3C文档pin漂移限制，未重写pin；没有声明全suite或真实GPU通过。新完整runner真实PF/CUDA/显存/dry-run尚未执行，由主owner在review后推进。
