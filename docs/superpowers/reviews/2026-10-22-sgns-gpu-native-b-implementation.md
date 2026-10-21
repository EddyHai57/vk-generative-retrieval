# SGNS GPU 原生实现 B：工程交付与有界基准入口

状态：CPU tiny 工程验证通过；**CUDA 吞吐、B 的 ≤12 h 判据与四门均未实测/未验收**。无 SSH、submit、full fit、validation、test 或完整向量导出。本变更为 C 档，待独立审查后由 owner 执行真实只读 PF 与 CUDA 子集基准。依据 decision_log `4b0a8a1` B 路线及 `6f8b56e` 夜间自主推进；保留此前 `e672373` 全 score parity 提交，不继续其非阻塞工作。

## 文件与实现来源

仅新增：`src/vkgr/sgns_gpu_native.py`、`analysis/sgns_gpu_native_b.py`、两份对应 test、`tests/fixtures/sgns_gpu_native_b_launcher_chain.py` 与本报告。未改原 `sid_coll_sgns.py`、`SGNSConfig` 或旧入口。

独立实现原 SGNS 的 SUM-loss 解析梯度，复用现有 PyTorch tensor gather、index_add、searchsorted 和 torch.compile；没有引入/复制第三方训练实现或新增依赖。CPU 本地 torch 2.5.1 无 CUDA；正式 runtime 仍严格消费原 lock 与 `new_instance_prepare.runtime`，不能以本地版本替代。CUDA torch.compile 使用现有 PyTorch 发行版的 Inductor 工具链；若冻结环境不支持，须如实记录，不能偷偷安装依赖或用 CPU fallback 冒称 GPU。

## source 与 config 语义逐项清单

| 项目 | 原语义与 B 保留方式 |
|---|---|
| dimension / window / negatives / epochs / batch | 64 / 5 / 5 / 5 / 4096；正式构造器只接受原默认 SGNSConfig |
| seed | 42；独立 device Generator，跨 epoch 连续；不承诺原 NumPy RNG 流 |
| init | 沿原 CPU Generator(42)、uniform ±0.5/64；output 全零；一次拷入 GPU |
| positive order | user 升序、center 升序、context 从旧到新；同 item 正对排除，重复事件保留，禁止跨 user |
| assembly / batch | flattened event rows 与 user ordinal 常驻 GPU；按 center chunk 装配固定 offsets；chunk 尾 carry 到下一 chunk，batch 仍4096；epoch尾单独更新，不跨 epoch 合并 |
| event frequency | 所有 observed train events 的出现次数；包含重复与无正对 item；非正对频率或 unique user 频率 |
| negative probability | event frequency^.75 归一化；拒 center 与 positive，5个独立有放回样本 |
| loss / optimizer | 正对 SUM + 每对5个负样本 SUM；SGD，无 momentum/weight decay/clip |
| batch update | 全部 center/positive/negative gather 都在任何写入前；由同一批旧参数计算 dc/dp/dn，再 index_add；重复 index 累加，非逐 pair 原地异步更新 |
| LR | 按本 batch 前的全局 processed / (pairs_per_epoch×5)，.025 线性降到 .0000025；epoch 不重置 |
| precision | 参数/点积/梯度 float32；概率CDF与LR进度算术 float64；无 autocast、混精或 TF32 matmul |
| export / health | 仅 center；原 float64 归一化/无正对 content fallback 与原 health_check；API 返回不隐式写文件 |
| 不增加的机制 | 无 subsampling、distance/watch weighting、early stopping、近似采样或参数试探 |

原 SGNS SHA256 保持 `2cc25d1997c12fe1e262e455e6644980186b5b9d0e00805014dee3148675876d`。

## 条件 CDF 与原拒绝采样分布

令原概率为 p，两个禁止 item 的 row 为 a<b；其质量分别 A=p[a]、B=p[b]，原 CDF 为 F。取 u~Uniform[0,1-A-B)，映回原质量轴：

`x = u + A*I(u >= F[a]-A) + B*I(u >= F[b]-A-B)`，然后 `searchsorted(F,x,right=True)`。

它恰好删除 a/b 两个区间并将其余区间按原长度拼接，因此每个允许 item j 的概率为 `p[j]/(1-p[a]-p[b])`，即原 rejection sampler 接受后的条件分布。每个样本独立有放回。GPU 随机数流与原不同，已获授权；float64 有限精度亦须实测。设备端逐 batch 累积 forbidden-sample/nonfinite 标志，epoch/有界前缀结束统一核查，失败不输出向量或成功回执。

采样频率检验：CPU tiny 固定两种禁止项，250,000 个样本对已知条件分布检验频率，再以原 NumPy rejection sampler 作独立频率参照；含零频CDF间隙、反向禁止索引与仅剩一个允许 item。生产路径每 epoch 在 GPU 记录**逐 item** observed counts，并累加精确条件期望：

`E[count_j] = p[j] * Σ_pairs 5/(1-p[c]-p[pos]) - Σ_pairs containing j 5*p[j]/(1-p[c]-p[pos])`。

每 batch 只 O(batch) scatter，无 O(N) 遍历或 host readback；epoch末 O(N) 计算 expected≥5 item 的 Pearson 分量总和、最大标准化残差、两边总质量，并压成固定64个row-modulo bins公开在聚合回执。**无完整逐item频数导出，无新统计门槛/p-value或科学转正结论。**owner 可对同一真实子集的原采样器按同一条件期望核对频率；不能要求不同 RNG 的次数逐位相同。

## 性能路径与边界

正对装配、采样、gather、dot、sigmoid、梯度与scatter都在同一设备；更新函数可 `torch.compile(fullgraph=True)` 融合。每步无 `.item/.cpu/.numpy/.tolist`；固定shape频数 scatter避免 CUDA bincount 动态shape潜在同步。只有 large center chunk compaction、epoch/前缀边界与汇总需同步/回 CPU。NaN/Inf 设备标志在前缀/epoch结束检查，不声称在首个坏 batch 即时 host 中止。

基准 warmup 是临时前缀，之后恢复原 init/output/progress/seed，再从原第一个 pair 开始计时；计时包含首次与后续 GPU pair 装配、采样、更新及频数审计，warmup不会隐藏首个装配成本或改变计时前缀。临时表最终丢弃，无模型产物。

`NativeSGNS` 支持五 epoch 核心；`fit_sgns` 返回原 health_check 后的 center vectors/audit，**本次未调用正式 fit**。部分 epoch 前缀标记为不可继续/重启，以免无意重复正对。CLI 故意只提供 PF/有界benchmark，不提供 full-fit 或训练提交能力；后续与四门/SID链的产物适配仍需 owner 完成并独立核对。

## 正式 CLI 与绑定

```text
python -B -m analysis.sgns_gpu_native_b dry
python -B -m analysis.sgns_gpu_native_b prepare --stage preflight|benchmark --request <REQUEST> --request-sha256 <SHA> --ops-code <PINNED_OPS>
python -B -m analysis.sgns_gpu_native_b preflight|benchmark --request <REQUEST> --request-sha256 <SHA> --execute-on-new-instance
python -B -m analysis.sgns_gpu_native_b verify --stage preflight|benchmark --request <REQUEST> --request-sha256 <SHA>
```

Request schema `sgns-gpu-native-b-request-v1`，精确字段：`schema, runner_source, bundle, bundle_sha256, lock, ops_root, output, preflight_run_id, benchmark_run_id, warmup_batches, measured_batches, compiled`。warmup限0..32、measurement限1..4096 batch；建议首次8/256、compiled=true，均是基准运载字段，科学 SGNSConfig 不变。

Input DAG：原 subset bundle manifest SHA固定，原 read_bundle逐字节验证 catalog/content/history 与序列，runtime/lock与 producer 严格相同；producer source写入回执，不强迫 consumer commit相同。own source精确全树核对；request、own job/source/argv/config/seed/resources、ordered input role pins在阶段前后重核。真实 readonly PF不更新参数，必须重现原每epoch34,770,216正对；benchmark先接受其 successful terminal/stdout/result/job，再临时更新前缀。verify只读核终态、完整计数、rate/projection算术与5×measured_pairs的负样本质量；不能只认exit0。路径拒test/holdout、symlink、越出独占输出边界；不覆盖旧输出/run。

资源首投保守：CPU2 / RAM16,384MiB / VRAM8,192MiB / disk1GiB / wall2h，`not_measured`；三种 native threads=1，按实际余量准入。入口输出 VRAM 只取 torch counter，明确 `torch_counter_excludes_context`，**不能替代含context资源声明实测**。CPU首次保守资源可按owner当前机器余量调整，但须新固定request/job与review依据，不伪称已有峰值。

机械投影：原 full 总正对18,787,015,620 / measured pair/s /3600；≤12h只表示B吞吐判据，不代表四门通过。投影不含full setup/final health、SID或G链；正式CUDA结果仍 UNKNOWN。B达标与否、回退原CUDA/82h以及最终一次A1验收训练均由owner依冻结裁决推进，不新增名额。

## 验证与 review 包

- Fresh RED：核心7个tests因模块不存在；CLI/基准3个tests缺模块；measurement回执缺校验函数。最小实现后最终 **16 tests GREEN**。
- 原 subset SGNS **13 tests GREEN**；原 SGNS文件SHA未变。
- CPU `torch.compile(fullgraph)` 更新 smoke通过；本地不是正式CUDA runtime，无CUDA编译/吞吐声明。
- 真实 pinned launcher.prepare→metadata→own job→同module dry/非法argv链通过：**9 input roles**，实际nested resource schema，4种 own job（seed/resources/argv/input顺序）伪造拒绝。独立固定commit部署布局；transport synthetic fixture，不等于真实GPU PF。
- **8/8隔离 mutation kill**：正对顺序、same-item过滤、frequency^.75、条件排除、SUM梯度、LR取值点、PF前置、PF只读标志。每项新拷贝，`python -B`、`PYTHONDONTWRITEBYTECODE=1`、唯一cache prefix；由目标AssertionError捕获，非语法或导入失败。
- 日志与单次diff：`cache/sgns-gpu-native-b/local-validation/`（ignored）。复现tests：`PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m unittest discover -s tests -p 'test_sgns_gpu_native*.py'`；launcher fixture按文件路径同形运行。
- 核心 module SHA：`dd2acadd8d38a224410ff21b3d1bc0df3d81b5a7d4ee72d31cedf46125e671c4`；CLI module SHA：`ffaaedecd8987f6dbf32d28e6b68f38a9afedb96507fe2c71c4bbc61abb2536f`。
- 待办：独立C审、公开push卫生、固定SHA部署、真实只读PF与verify、CUDA基准与verify、owner回填原stdout与context-inclusive显存。未做这些前，不能宣称B达标或直接开始full fit。

## 独立 C 审缺口修复（2026-10-22，原实现记录保留）

原独立 reviewer 在固定实现上复现 nested JSON 的 int/float、bool/int 混同、request 的 hash/parse 两读、阶段进度缺失和 own-job env 未闭合。此轮按 C 档处理；以下为执行者自验，独立 reviewer 复验仍待完成。

- 新增递归 JSON 类型与值比较：SGNSConfig、runtime、source、input binding、own job、stdout/result、阶段前后闭包均保留 nested 精确字段集合；terminal exit_code 与 full_pair_count 不接受 bool/int/float 混同。launcher/result 根对象允许既有扩展字段。runtime 先经 JSON 归一化，避免 TorchVersion 字符串子类误拒；cuda_smoke_executed=0 仍拒绝。
- request raw bytes 单读，hash 与 parse 消费同一 bytes；替换读取反例与禁止第二次 digest/read 测试通过。
- stderr JSON 输出 input/build/warmup-compile/measurement/postcheck/execute 阶段。每 64 batch 与末尾报告 batches_enqueued；CUDA synchronize 后才输出 completed_batches，stdout 仍只保留正式回执。
- own job env 精确等于 PYTHONPATH=部署树/src、OMP/OPENBLAS/MKL_NUM_THREADS=1，加 launcher 注入的 VKGR_RUN_ID 与 VKGR_PROGRESS_PATH。prepare 用真实 pinned launcher 的 _job_env/_recorded_env 核对继承环境；额外 recorded whitelist/VKGR 字段拒绝。execute 在读输入前核对全部六值。未登记的 CUBLAS_WORKSPACE_CONFIG 或额外 VKGR 环境须清理后重新 prepare，不能临时修改已绑定 job。
- fresh RED：原 nested/runtime/terminal/单读/progress 反例共 6 failures；full_pair_count 浮点反例另 1 failure；env 缺失、额外、PYTHONPATH 与实际进程 progress/PYTHONPATH 共 5 failures。最小修复后最终 **23 tests GREEN**；原 subset SGNS **13 tests GREEN**。
- 真实 pinned launcher.prepare → metadata → own job → 同 module dry/非法 argv：**9 input roles、7 job mutations、2 inherited environment mutations** 全通过。env 由真实 launcher helper 生成，未 mock；部署布局包含独立 runner/OPS clone。fixture 曾将拒绝理由误查 stderr，已改为 CLI 实际 stdout 并重验通过；不涉及产品行为或科学代码。
- **15/15 隔离 mutation kill**：原 8 项，加 recursive type、nested keys、single-read、stderr progress、full-count type、job env、process env。每项独立新拷贝，python -B、PYTHONDONTWRITEBYTECODE=1、唯一 cache prefix；均由目标 AssertionError 捕获。
- 本轮证据位于 ignored `cache/sgns-gpu-native-b/review-fix-validation/`；真实 launcher 最终回执指针 `cache/sgns-gpu-native-b/launcher-1792609951290261000`。核心 SHA 仍为 `dd2acadd8d38a224410ff21b3d1bc0df3d81b5a7d4ee72d31cedf46125e671c4`；修复后 CLI SHA 为 `d9524bfbd9a6d67fedaf349b120a4005d1a90ba6a6e7998ed3e5150443329b4e`。

数学核心、SGNSConfig、原 SGNS 文件、科学输入与门槛均未修改；未执行真实 GPU PF、CUDA benchmark、正式 fit、SSH 或 submit。B 是否满足 ≤12h 仍 UNKNOWN，四门仍未验收。本修复固定提交后交原独立 reviewer 复验，未取得复验前不宣称 C 审通过。
