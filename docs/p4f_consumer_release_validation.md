# P4F consumer release v3 本地验证

密级: 公开

本轮新增显式 external-SHA consumer release 模式，用于没有 Git 的受控部署。默认调用仍要求真实 Git；不 catch Git 失败、不复制 `.git`、不伪造 HEAD。只修改 retrieval_ann/retrieval_infer_run 与 matching tests；历史 SOURCE/PLAN、producer、adapter、fusion、manifest.source_fingerprint/git_context、lock/config 不变。当前 READY awaiting Astra fresh acceptance；本地验证不等于 Linux/真实 FAISS/ANN 质量验收。

## 接口与阻断顺序

`consumer_identity(*, consumer_release=None)` 接受 `{path, sha256}`。export_vectors、build_index、audit_index、fuse、infer、verify 都新增同名 keyword-only 参数。五个现有 CLI 子命令接受成对 `--consumer-release PATH --consumer-release-sha256 SHA`；audit 仍从 index 子命令进入。只传一项、missing release、SHA/schema/member/父链/runtime 漂移在 load_input 调用或 output 创建前拒绝。未引入环境变量/global identity fallback。

release 必须为 canonical JSON（sort_keys、紧凑 separators、无尾 newline），exact 顶层字段：schema_version=1、kind=p4f_consumer_release、parents、declared_build_git、members、dependency_bundle、runtime。重复 key/nonfinite、未知字段、bool schema、非法 SHA、unsafe relative member、任何路径组件 symlink 均拒绝。外部 SHA 是 Astra 下一 exact gate 的可信入口；一致性 hash 不是独立真实性证明，不能自行把新 SHA 当获批 release。

parents 按原 bytes/SHA 绑定 accepted source-v2、Astra acceptance-v2、原 transport；34 个 source/lock/config/python/tests 成员为原 32 transport 与 v2 七文件的并集。只允许获批两 consumer 与两 tests 不同于父包；其余 30 项逐项不变。package 内任何未列文件（包括 .pyc/.pyw/native sidecar/cache）拒绝；部署需干净 source tree 并使用 `-B`，不会自动删除 cache。旧 producer 的科学 input/checkpoint/history/catalog/query 链仍按原规则核，不被 release 身份替换。

返回独立 identity_kind=verified_deployment_release，含 release SHA、父链 SHA、实际 source_fingerprint/lock/config、dependency bundle SHA/runtime。actual_git_commit/dirty=null，git_status=not_queried_explicit_release，声明的 build Git 仅为 declared_build_git；即使本地有 Git，explicit 模式也不把声明值填 actual。默认 Git 模式返回原 schema。

export→index→audit、route export→fusion→selection→infer→completion/verify 显式模式均要求同一完整 consumer identity。resume 沿用 run/shard binding；另一个仍合法但 SHA 不同的 release 也拒绝。release 产物不能经默认 native-Git 模式混用。ANN infer 仍为原 closed gate，exact inference 行为和科学算法不变。

## 依赖与部署闭包

ignored `cache/p4f-consumer-release-v3/deployment-closure-v3.json` 包含 34 actual members 与三份原 parent 的部署相对路径；source-manifest-v3.json 独立保存新七文件 pins，未追写 v2。依赖 binary 不进 public Git。

`dependency-members-v1.json` 从已验收两 wheel 的原 RECORD SHA/size 推导完整 68 regular members，RECORD 自身小文本单独 hash；未读取执行 native code 或实际解包。runtime release 中 dependency_bundle={root,fixed_root,wheels,members}，两 root 必须 canonical absolute、独立、no symlink；实际 target 所有 regular files 完整 SHA/size/集合绑定，不容额外 cache/native member。external release gate 必须核 members 与该 approved RECORD 表完全一致，而不能只核 wheel 声明。

实际 observer 使用 numpy/torch/pyarrow/yaml/packaging/faiss module version+origin bytes hash、对应 distribution version+METADATA bytes hash、distribution root，以及实际 Python version/interpreter bytes SHA。packaging26.3/faiss-cpu1.15.1 必须来自 approved isolated target，其余 numpy1.26.4/torch2.11.0+cu128/pyarrow19.0.1/PyYAML6.0.2 来自原 fixed root。完整 target native members 有 hash；固定环境记录这些版本、entry source/metadata/interpreter 身份，**不是原 venv 全部文件逐字节闭包**。该范围如需升级，必须另 gate，不能称已证明完整环境不可变。

runtime-template.json 中未取得的 Linux executable/origin/metadata SHA 保持 null，标 proposal_only_not_usable_release；本 task 不构造可执行批准 release，也不 SSH 填这些字段。下一 gate 必须独立取得实际 metadata、组成 release、fresh 审 external SHA 后才能启动真实 CPU synthetic 路径。

## 验证与限制

fresh identity RED、flow RED 与 scoped GREEN 证据只在新 ignored cache；第一次测试 harness 的 missing mock helper、Mac /var symlink、missing import/patch 顺序错误保留，未冒充 clean RED。独立 review 两个 blocking（额外 .pyc 可漏过；公开 tests 依赖 ignored cache/个人路径）均关闭：前者 fresh failing proof→全部 package files gate，后者改为完全自包含 synthetic 父链和 68 tiny mock files；生产 trust pins 未变。

新 tests 覆盖合法无 Git explicit release、默认无 Git 失败、external SHA/strict schema/父链/source/lock/config/symlink/额外文件/runtime/member 漂移、resealed protected source/parent、CLI 半参数/fail-before-input、mock index/audit artifact 身份、tiny checkpoint export/fusion/infer interruption/resume/verify 与 changed release 拒绝。实际 runtime observation 分支用 tiny mock modules/distributions验证 bytes 和 owning metadata/origin；不在 Mac import Linux FAISS。

相关 regression 命令：`PYTHONPATH=src:tests OMP_NUM_THREADS=1 <approved-local-python> -B -m unittest test_retrieval_ann test_retrieval_infer_run test_retrieval_fusion test_p4d_v2_verifier_compat -v`。旧 native Git 路径与历史 producer mutation 拒绝一起保留。完整命令、exit、count、log hashes 与独立 review 限制见 cache review；指标数字仍唯一 experiment_log。

未 SSH/真实数据/GPU/index 质量评测/安装/commit/push；Linux ABI/native FAISS、实际 runtime metadata、source 运输、资源监督与真实 CPU oracle 都仍待独立 gate。独立 reviewer 的 sandbox 禁止 ps，因此其 mock index review 对 resident_bytes 仅 mock；本轮自身 regression 可执行原 RSS observer，但不证明 Linux 资源峰值。静态只读校验不是对敌对并发文件替换的 atomic snapshot 保证；受控不可变运输/启动边界仍须下一 gate。
