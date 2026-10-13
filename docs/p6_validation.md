密级: 公开

# P6 validation

## Scope / frozen implementation

只tasks P6，Eddy过夜授权经Astra批准brief scope，不commit/push/资源生命周期/P7。原生worktree codex/p6-tiger-smoke基点0376d9a8119f3500cbf78519675cc7bf18013c1b；已验收P5最新36文件overlay逐hash记录cache/p6-entry-overlay.json。main/P5原件不改。新增原创src/vkgr/tiger.py、training.py、tiger_run.py、对应3tests、configs/p6.json、必要training optional dependency+真实uv.lock、本validation与状态/事实源小范围文档。生产P3/P4/P5、旧config、冻结合同/evaluator/SID不改。

输入：P3 frozen manifest SHA3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f；P4 fixed run SHA65d90cc9fb4df88824786a794349e3e2a6724ff0a8e08e4f95b573c699bdb801；P5 fixed run SHA669029504ab18df5f6a6109820d3d4a36a243f6ab0d3e8c46fc8ce17b1b6bbe9。消费static pinned artifacts，不放松旧producer，不重跑label/SID。真实training-examples、audit queries与每个artifact重新hash，history positions严格早于target；P4audit与P3全集固定SHA抽样逐字比较，OOV原样保留分母。

配置在任何P6模型指标之前冻结。Encoder-decoder128维/4heads/encoder2+decoder2/ff512/dropout0；AdamW lr.001/weight_decay.01，clip1，seed42，batch64，最多1000step，原首次train started_at起2700秒（resume不重置）；UTC训练deadline2026-10-07T02:16:21+00:00，global2026-10-07T02:46:21+00:00。teacher forcing decoder=BOS+前3target digits，label=4target digits；PAD0/BOS1，4digit disjoint offsets，history最近50item的200token flat序列，learned absolute position。source padding mask传encoder及cross-attention，decoder bool causal mask。无EOS。No mixed precision / no scheduler / no tuning。

Beam20每层跨所有活跃beam global top20，累计full-vocabulary logsoftmax之后再合法next约束；不是按合法词重归一化。固定4digit→P5唯一item，最终累计分数tie按item ID。encoder memory每query只计算一次、beam复用。不要求loss降低或优于baseline。时间bound不足1000明示partial_time_bound，任何实际resume/decode不闭环只benchmark scaffold。

## 环境 / provenance

Mac独立Python3.11.17、torch2.11.0；server新独立P6 .venv Python3.11.16、torch2.11.0+cu128、CUDA runtime12.8。官方uv resolver真实生成34package lock，training为optional extra；Linux显式https://download.pytorch.org/whl/cu128，Mac官方PyPI，base PyYAML6.0.2/numpy1.26.4/pyarrow19.0.1不变。未用3.12 /venv/main，不改driver/globalenv。官方cp311 manylinux wheel存在，fresh CUDA tensor运算True/1.0，不是只凭nvidia-smi。

[官方wheel目录](https://download.pytorch.org/whl/cu128/torch/)；[PyTorch2.11.0 license](https://github.com/pytorch/pytorch/blob/v2.11.0/LICENSE)允许有条件source/binary redistribution，BSD-style。仅安装官方dependency，不复制其模型代码或未知许可TIGER实现。CUDA官方bundled依赖是runtime使用，不把这些wheel/vendor代码重新公开；完整公开release/provenance余项仍受I-007/P13。所有具体installed versions在server install.log和runtime env manifest，uv.lock保留精确transitive hashes。

完整/etc/vast-agents-guide.md已读，仅SSH alias vast-a6000，不打印连接字段。P6 server root /workspace/vkgr-p6-20261006t2050/code-v1；只读GIT_DIR=/workspace/vkgr-p2-20261006t1755/.git、GIT_OPTIONAL_LOCKS=0，worktree指向当前快照，不复制secret/.git。Runtime数据/checkpoint/optimizer/完整SID/features/逐query预测只server ignored。Mac只有原创/synthetic/脱敏摘要。

## 实际 TDD / verification ledger

命令prefix：`OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -B -m unittest`，working directory为P6 worktree。日志完整保留ignored cache。没有把依赖失败冒充feature RED。

| Round | Fresh RED (exit/key output) | GREEN (exit/key output) |
|---|---|---|
| R01 model/mask/beam | `discover -s tests -p test_tiger.py -v`，缺tiger ImportError，exit1 | 同命令3tests，exit0；namespace/teacherforcing/OOV、causal/PAD invariance、widebeam exhaustive |
| R02 training/checkpoint | `discover -s tests -p test_training.py -v`，缺training ImportError，exit1 | 2tests，exit0；CPU不中断5step对2+3真实checkpoint/resume逐参数/RNG/cursorequal、首次start/deadline |
| R03 gates | `discover -s tests -p test_tiger_run.py -v`，缺tiger_run ImportError，exit1 | 3tests，exit0；strict frozen config/hash drift/temporal/OOV/fail publish。config生成用system python缺PYTHONPATH失败，随后的tests成功不代表config成功；随后项目Python独立生成config并读回。 |
| R04 session | 同runner命令4tests，缺train_session，exit1 | 首GREEN因test第三run复用exclusive log目录而FileExists，exit1；纠正synthetic test新目录、green-retry4tests exit0；实际train/session resume与deadline0step安全checkpoint |
| R05 decode | runner5tests，缺evaluate_model，exit1 | 5tests，exit0；真实toy模型4digit→item，OOVmiss/全catalog候选、既有item evaluator/deadline拒绝 |
| R06 CLI/resource | runner7tests，缺run/ResourceMonitor，failure artifact断言FAIL，exit1 | 7tests，exit0；runner失败保留failure.json，无success；sampler异常传播 |
| R07 verify | runner8tests，缺verify_run，exit1 | 8tests，exit0；拒绝伪success，不仅凭status/exit |

独立Sol阶段review发现测试洞而非model逻辑blocker：初beam4items/width20不裁剪、causal只查position0；补81item/width5独立逐层global oracle与每位置future invariant，5modeltests GREEN。Checkpoint等价测试两个路径都可能没训练，且dropout0不耗globalRNG；补至少一个参数确相对initial改变，保存前推进torch/numpy/python RNG、重新构建并扰乱后load比下一值。完整日志cache/review-coverage-green.log/full-v1.log。

独立review与作者自检识别fresh-process verify环境比较在seed_all前会误拒绝deterministic flag；在比较前初始化CUBLAS_WORKSPACE_CONFIG及seed_all，未启动受影响正式run。补fresh subprocess regression，9runner tests exit0；临时独立源码副本分别移除env初始化/global beam裁剪/RNG恢复后，目标tests各exit1 FAIL，日志cache/review-{env,beam,rng}-mutation-red.log，不修改真实源码、不伪装历史TDD。原源码恢复GREEN。

Mac完整source-ready suite：86tests5.102s exit0，既有FAISS skip1；本阶段torch核心tests实际执行，没有skip掩盖torch。后补subprocess总87tests以最新fresh日志为准。

Server首full-v1 86tests4.868s exit1/35errors：snapshot未传只读Git上下文；第二full-v1-git-context 86tests5.468s exit1/21errors：缺既有public synthetic fixture；第三full-v1-git-public-fixtures 87tests10.486s exit1/19errors：缺P1 tests依赖outputs父目录。补三个小public fixtures与outputs/README，旧日志保留；不把失败当成功。这些是快照准备不足，不改旧源码/测试绕过错误。最终fresh gate下节记录。

一次状态/source脚本使用system python处理中文stdin报Non-UTF-8 SyntaxError，原脚本未执行；随后项目.venv Python含coding声明重试明确P6 state/source updated，未掩盖失败。

## 历史runtime启动 / review gate（已由后文闭环证据取代）

Source fingerprint6472ddb1ca6f103a895ba5cdd5a727b1af7717125d3b6029997e5e83c7ba4d32；首42file source archive SHA7b0aa3e74e30b96eb3daa4bcfeda43d7a1f05d8e3d94fc7ca124c70057ca62d7，后补public fixtures/outputs README/新增test不影响production fingerprint。作者真实load_inputs全hash/时序/P4audit/static SID通过，输出15条input evidence、固定两split query数量与vocab；数据数字摘要最终只写experiment_log。该启动时点Source完整独立Sol review通过，Astra另fresh全suite通过，runtime当时尚未完成。本段是历史快照；当前runtime闭环与独立验收以文末证据为准，整体P6仍待最终docs/public裁决。

限制：official-order assumption/hidden timestamp、train-only conservative history、P5未收敛/nonmonotonic objective、固定audit代表性与OOV候选域仍有效。此encoder-decoder/SID是原创small TIGER-style工程smoke，tokenizer不是RQ-VAE，不称忠实TIGER复现。资源250ms采样+边界check不保证瞬时硬余量/hidden quota，VRAM用实际allocator peak与可见全GPU free分开。Decode latency单query tokenize+encoder+beam+item mapping且CUDA同步，不含input load，不与P4不同边界latency直接排名。最终successmanifest最后写，失败原件不覆盖；deadline后的独立decode replay需要新bounded授权，不私改已冻结config延长本轮。

### 正式runtime启动gate

最终server `PYTHONPATH=src ... unittest discover -s tests -v`：87tests7.262s exit0，既有FAISS skip1。真实`load_inputs`全hash/static SID/全部train temporal/OOV/P4 fixed sample逐字验证exit0；没有重算旧producer或读取指标来改config。独立Sol完整source review无额外blocker，Astra fresh86tests exit0；已补fresh subprocess与mutation coverage，允许正式runtime。

专用tmux `-L vkgr-p6-20261006t2050 -f /dev/null`、session `p6-pipeline`。Script server code-v1/cache/p6-pipeline.sh；整个bash `timeout --signal=TERM 3300`，每stage真实独立Python process。先train到100保存并processexit，再resume至1000/原首次timebound，再decode两固定audit，再verify；任一stage非0立即保存failed状态停止。日志cache/{train,resume,decode,verify}-v1.stdout/.stderr/.exit与pipeline-v1.status，exclusive输出outputs/p6-smoke-v1-20261006t2130。目录label不充当实际start UTC，真实start只读checkpoint/stage。此节仅启动证据，不能冒充已完成。


## 正式runtime闭环与待协调验收

Server唯一有效run /workspace/vkgr-p6-20261006t2050/code-v1/outputs/p6-smoke-v1-20261006t2130/run_manifest.json，SHA17a0b10b6216b2b350e9b842117ccac3e1df172509e5395076daa651d6c45f23。Public artifacts/p6-smoke-manifest.json SHA599cedf0168cee963bd4ed5a53d9df7c759c4640a981c1a3969e3279b6c162db，只含role/hash/size/boundary、env/frozen config、actual step/UTC与claim，无private roots、queries、IDs、SIDs、centers或weights。Machine success是runtime通过，不替代Astra阶段验收。

Source保持6472ddb1ca6f103a895ba5cdd5a727b1af7717125d3b6029997e5e83c7ba4d32；config SHAcfd4ff78bf754cad80f5a942b1f6d9c69cd09fed252f79690a67f1968ce1cb2b、真实uv.lock SHA8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063。19个P3/P4/P5受保护overlay src/tests/config/public artifacts逐hash未变。Mac最终fresh全suite87tests5.145s exit0，既有FAISS skip1；server最终87tests7.262s exit0，同skip。不skip核心torchtests。

四stage cache/train-v1.exit、resume-v1.exit、decode-v1.exit、verify-v1.exit均0，stderr空；pipeline-v1.status=completed。独立train process实际step100保存并exit，resume stdout明确from100→1000/statusstep_limit，随后两个固定audit与完整checkpoint/prediction/item replay完成。Verify stdout `P6 source/input/checkpoint/actual-step/query/prediction/item-metric replay verified`；不只凭exit0。全部配置/质量/资源/负结果数字唯一人工摘要experiment_log。

Initial checkpoint checkpoint-100-from-0.pt SHAca896391e719909bf41e90c3d45609217f84daa4d5095f5e61c4feece4842299；final checkpoint checkpoint-1000-from-100.pt SHA17d6f3706b1bfed206f9eefe1242fff398ae24527e461306f93aa76c2c50dc24。它们及optimizer/RNG仅server，从本轮caller-pinned自有文件weights_only读取；不从任意外部pickle加载。作者另用torch/stdlib（未import vkgr）独立核checkpoint SHA、optimizer真实steps、finite logs连续性、绑定/原started_at与resume参数确有变化；这是作者自检，Astrafresh独立script另核，不互相冒充。

Server artifacts包括stage-100.json、stage-final.json、train-from-0.jsonl、train-from-100.jsonl、两个checkpoint、metrics.json、validation-audit.json/test-audit.json、run_manifest.json；全部SHA/size在server/public machine manifest，所有完整预测/query留server，不回Mac。Snapshot旧失败/旧日志不删除、不覆盖；未新建资源/改driver/Git配置/commit/push。Source spec/quality独立Sol通过，完整runtime/docs/public最后gate由Astra裁决；当前Implementation ready / awaiting coordinator review，不自主P7。

### Astra独立runtime验收证据

Astra fresh执行独立Sol只读脚本（SHA3c87483e1c4c2658962bc1b5a623dbd94eb05f887efa979abd17a6947faa98cd）exit0，明确`independent_audit_pass`。未import P6生产runner，独立核全部input/artifact SHA、两checkpoint model/optimizer/cursor/binding、全部实际step日志连续性、resume后参数变化、P4selected queries逐字相等、catalog/SID roundtrip与所有预测合法性，再以手写Recall/NDCG核所有split/slice公式与分母全部一致。结果数字只experiment_log。不能把作者脚本冒充Astra独立核对；此项是Astra实际执行新独立reviewer脚本。整体P6 formal closure仍待最终docs/public review与协调裁决。


## Astra正式P6协调验收

本节取代前述awaiting coordinator review口径，历史段只代表当时状态。Astra正式裁决P6 PASS / Complete。最终独立Sol docs/public review无blocker，先前source/spec/quality通过；Astra fresh真实train/resume/decode/replay与独立checkpoint/log continuity/query/itemmetric/all hashes核对全部通过。既有server/public/source/config/lock/checkpoint SHA均未改变，没有为文档收口重跑训练、SSH或tests。

下一task P7 stretch已在过夜授权范围，由Astra另行dispatch，本worker只P6。完整大产物只server ignored，main/P5不改，未commit/push/资源生命周期；Complete是批准small smoke工程闭环，不升级为TIGER忠实复现、全test、方法优势、online或causal claim。
