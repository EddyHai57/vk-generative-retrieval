# G2 validation — local engineering passed, pending Astra

密级: 公开

G2本地工程交付已可验收，尚未由Astra判定Complete。只使用原创synthetic fixture和CPU；真实训练、test及CUDA gates关闭。当前worktree与版本见STATE.md。批准范围是新增三模块、三tests、fixture、小型manifest和本文件；既有文件仅允许更新四个状态/证据文档。

## 实现与验收含义

- generative_training.py：真实Tiger core的full-vocab四位置CE、FP32 AdamW、effective batch 512及实际tail权重；无放回epoch cursor、外部hash绑定JSONL store；完整model/optimizer/RNG/cursor checkpoint。
- generative_evaluation.py：beam100合法前缀展开、SID与item tie规则、bounded穷举oracle、collision-corrected item指标；OOV保留分母，空组null；不可变ordinal shards及断点续缺。
- generative_experiment.py：sealed factory context、共享train-only统计合同、累积预算ledger、train/resume/evaluate/verify/select CLI；actual重训及逐query decode replay。checkpoint和shard只能在安全写完、核hash后获得receipt。

共享slice输入采用P4E双文件schema，完整train-like event counts包含重复events和single-like用户；nearest-rank positive median/p90及固定user桶，不用target-frequency proxy。此处只有synthetic同schema证据；真实P4E pins须另经Astra验收，不能据此启动G3。

恢复只接受绑定同一输入、source、配置、parent、预算的安全状态。SIGINT/SIGTERM只置停止flag，完成当前step/shard后保存；受控InterruptedError恢复已验证owned shards。未知异常、非finite、非法候选STOP，不自动重试；没有任意SIGKILL或I/O故障恢复承诺。

独立evaluate/verify/select只接受完整run，沿原deadline计费。operation journal在同一run持久保存，首操作之后必须传caller持有的最新external head；缺失、过旧、尾部截断和删除records均拒绝。失败操作也产出failed head，abandoned begin在持有正确外部head时计入实际经过时间。不得自行读取最新head绕过外部锚点。

## Fresh验证与review

最终命令：在本worktree执行，PYTHON为既有p6-tiger-smoke/.venv/bin/python。

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" -B -m unittest discover -s tests -v
git diff --check
```

最终full regression：222 tests / 41.262s，exit0，既有FAISS skip1。日志/private/tmp/g2-final-regression.log。测试包含手算CE、512/256/128与tail17独立更新oracle、513无放回、fresh subprocess恢复完整state与随机draw、beam穷举、OOV分母、缺失统计拒绝、3+3+1 shards、真实SIGTERM及CLI resume、累计预算与外部journal head拒绝。

首次217-test回归因旧P1写managed outputs遭sandbox PermissionError；正常授权执行后通过，未放宽旧测试。CLI初次GREEN的重复__main__ class identity、JSONL writer额外换行等失败及修复保留在cache/g2-execution-ledger.md。

独立只读GPT-6.1 Sol/medium fresh review实际发现metadata seal、OOV逐query replay、evaluation中断、deadline/wall和AdamW moments五项缺口；逐项fresh RED→GREEN后scoped review确认Closed。随后发现journal尾部删除可回滚累计wall，作者fresh RED→GREEN后独立实测有效head前进、旧head拒绝、删尾/删全拒绝、abandoned begin计费，最终Closed。review边界及证据见cache/g2-review-result.md；不把review或exit0代替Astra验收。

## 持久化CLI证据与公开入口

cache/g2-final-acceptance-v1/commands.jsonl保存六个真实subprocess完整argv、cwd、stdout、stderr及exit；receipts.json保存外部pins。顺序为prepare→train(step1安全中断)→resume→evaluate(shard-size3)→verify(--replay，传evaluate head)→select(传verify head)，全部exit0。原有2epoch/60秒/单次恢复上限未扩大，actual训练与decode replay通过，shards为3+3+1。六步精确参数可从commands.jsonl重放到新的exclusive目录，禁止复用既有输出。fresh --help在OMP_NUM_THREADS=1/PYTHONPATH=src下exit0，实际支持prepare/train/resume/evaluate/select/verify，描述real/test gates closed。

公开小型摘要为artifacts/g2-synthetic-manifest.json，包含40-member source、实际环境、core配置、synthetic指标摘要与CPU观察，不含checkpoint、完整SID map或逐query预测。指标数字只在docs/experiment_log.md维护；该fixture不能证明真实效果、CUDA一致性或模型间质量差异。RSS是进程lifetime high-water观察，latency是此CPU小fixture的tokenization+encoder+beam观察。

G2 source身份是G1显式33成员加三模块/三tests/fixture，共40；排除本摘要及可追加validation文档。G0/G1 manifests、旧source、合同、lock、历史capsule及旧validation bytes不改。最终相对开工158-file baseline的范围核验见cache/g2-final-scope.json，不能用含继承未提交overlay的git status判断本任务改动。

已结束本地进程，没有本对话server scheduler/monitor、SSH、GPU、真实训练、commit或push。Astra的fresh验收须检查真实产物和scope。原60秒deadline到期后旧run仍可只读审计，但live replay正确拒绝；如需新的actual replay，应在新exclusive目录执行新的有界synthetic链，不能延长旧deadline。下一task只在G2正式验收及新brief后开始。
