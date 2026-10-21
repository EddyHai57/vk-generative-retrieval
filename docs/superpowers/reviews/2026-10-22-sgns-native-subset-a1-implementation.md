# Native B 子集 A1 与四门运载实施报告

## 范围与源码边界

按主执行者授权，本任务仅新增 `analysis/sgns_native_subset_a1.py`、`tests/test_sgns_native_subset_a1.py` 与本报告。无 SSH、push、submit、真实数据读取、训练、validation 查询或 test 读取。工程实现须由独立 C review 后再部署；本报告不裁决科学有效、转正或 anchor。

公共基线为已发布 SID `f36bf15a1844e2f305c8fd2c9d71655e2465ee29`（完整144文件账本）。新增源码差量严格为29个原字节依赖加本3新文件，共32路径；不得修改任何基线文件，不允许额外路径或开放式豁免。请求的 producer_delta 对32条路径逐项钉 SHA，原依赖 SHA 还须等于下表固定值；实际 source.files 必须等于 SID 完整 files 账本并上该固定差量。旧 CLI 包括已经独立复核的 `c573b79` progress Path→mapping 修复，sha256为 `04d1ef8ad82a5bb291b9ed4a3a81cbdde532d0a2b8dc0e63fb1fba941c814010`，不重改旧代码。

## 运输适配

CLI 只有 dry、plan、derive、preflight、train、verify、accept、consume，没有 submit/retry。各阶段新 run_id 为 `sgns-native-subset-a1-<stage>-v1`，不复用旧 A1 job/family。derive 请求 schema 为 `sgns-native-subset-a1-input-request-v1`，其余请求为 `sgns-native-subset-a1-request-v1`；共同字段为 source、lock、sid_root、sid_request、producer_delta、ops_root、ops_code。derive另含 raw_data、ur001、content、output_root；A1另含 data_root、prepared、history、content、input_request、output_root、scratch_root。文件角色均为显式 path/sha256 record。

derive直接复用原 `_raw_payloads` 和 `write_family`：重新构建绑定最终 B SID manifest SHA 的独立family，保持原 raw/DAG/oracle、历史与validation科学输入检查。A1原 `_execute` code/defaults/closure不变；仅在单次独立namespace替换 read_request、identity、own_job、predecessor和core proxy。core proxy仅替新RUN_IDS、engine_for与closure。原engine_for继续执行原bytecode，其namespace只替producer closure、精确source delta与固定输入模块依赖。原run/verify/_contexts/_binding/_load_sid等仍由原native_functions构造，code/defaults/closure不改。沿用仓库现行FunctionType模式，没有module globals monkeypatch。只有明确的原相对/绝对input模块名可获取namespace proxy，不能由请求选择模块或可执行文件。

SID消费在其原部署ROOT绑定原source_identity bytecode，先核完整清洁source_tree和固定entry SHA；新A1树只包含原SID树加精确delta。native B receipt直接使用显式backend B和native_audit_verification，未构造旧numpy original_audit_verification。原family/science schema保留为内层科学契约，新增外层 `sgns-native-subset-a1-stage-v1` 明示 backend B/authority4b0a8a1、request/source/job、原science receipt与其原始字节SHA、SID request。原inner的authority及native source adapter字段属于保留的旧函数输出，不作为新B权威捷径。

每阶段绑定完整原OPS metadata：固定argv/cwd/source/seed、资源与环境、逐文件inputs/bytes、OPS source、overlay、stop_mode。成功只认原terminal中的typed exit0、signal None、bool wall_cap_exceeded=False以及原stdout与receipt字节闭包。preflight/train使用保守not_measured资源；verify/accept按原train含context显存及RSS×1.15向上取整256 MiB。输出与scratch双向排斥SID、native producer、CPU reference及family证据根；不覆盖旧证据。

## 原科学验收与消费接口

原A1 seed42、配置、训练预算、EarlyStop、decode/eval_workers和validation hits≥378不改。predecessor保留原preflight计数/runtime语义、原训练manifest/best/last/epoch artifact绑定。accept重新只读消费原verified结果，复用原 `_read_evaluation`、EarlyStop、validation_hits、collision_gate及原 `check_proof`；重新核每epoch rows/computation、选中epoch、整数hits、native verify已发布replay、health、CPU/CUDA SID gates及整数碰撞×1.10。它不调用fit、run或model verify。聚合证据只包含三阶段固定输出，不将accept自己的receipt加入其proof，避免自引用；新replay scratch与原A1 scratch为分开的目录。

`consume(receipt_path,sha256,scratch)` 只接 `sgns-native-subset-four-gate-receipt-v1` 的成功原job/stdout/terminal，返回 `{receipt_sha256, request_sha256, producer, proof}`；proof严格为原7字段 health、sid_gates、collision、validation_hits、hits_gate、a1_native、evidence。普通SID/A1 verify receipt、status-only或旧backend receipt一律拒绝。消费时再次只读重算，要求proof逐字节语义一致，并在结束复核request/source/runtime、receipt与原terminal/stdout快照；不凭已写receipt直接放行fullfit。

## 验证与限制

最初4项入口测试fresh RED（ImportError），实现最小GREEN后扩充运输/原科学函数复用测试。实际公开多树测试从f36bf15分别创建SID producer和A1 runner；A1只补29原字节依赖+3新文件，另建305 native producer和固定OPS树，执行真实模块导入及原check_native/registered_config → 正式CLI plan → 原launcher.prepare/metadata → 完整job binding → 同形式dry。没有mock缺依赖；科学文件使用运输fixture，不构成真实preflight。dry沿原read_request consume=False模式只允许检查运输结构，实际derive/train/verify仍强制原frozen锁/content pin，并有负例证明dry pin不能进入实际stage。

新测试另覆盖：封闭namespace及异常后原globals身份、原native bytecode、原progress mapping到达模型构造边界而未建模、精确source增量、祖先/子孙输出互斥、完整job资源/环境/argv/inputs、阶段typed cap、未验收receipt先拒绝、总receipt独立重算一致与终态竞态，以及原EarlyStop/整数hits/四门只读重放。synthetic科学fixture只用于测试代码路径，不是项目科学结果。

最终相关回归命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 <project-python> -B -m unittest tests.test_sgns_native_subset_a1 tests.test_sgns_subset_a1 tests.test_sgns_subset_a1_progress_transport tests.test_sgns_subset_a1_inputs tests.test_sgns_subset_acceptance`。运行57项，56通过、1项旧固定树专用测试skip，38.367s；随后not_measured资源标签调整后14项新入口测试全部通过，39.234s；最后两项断言加严定点复验2项通过，1.072s。没有把skip称为通过。

19/19作者隔离变异全部由语义断言kill，没有SyntaxError/ImportError等伪kill。每项独立完整源码新副本、python -B、PYTHONDONTWRITEBYTECODE=1和唯一缓存前缀，已隔离；未变异独立control12项通过（排除需要实际Git多树的launcher和engine lock取原Git字节两项）。覆盖delta成员/旧字节/依赖pin、callback与namespace、原锁/content、输出互斥、outer/inner SHA、stage/consume cap、总receipt guard、独立重算相等及snapshot、原selected epoch、整数hits/native verify、碰撞和原四门检查。初轮delta-members存活已保留原记录，补“同时移除声明与实际路径”断言后完整重验；另将总receipt guard攻击补齐request角色，排除KeyError假kill，再完整重验19项。历史证据不覆盖。

私人证据指针：gitignored `cache/sgns-native-a1-validation/red.log`、`final-regression.log`、`final-entry-tests.log`、`final-tightened-tests.log`、`mutations.json`、`mutations-semantic-final.log`，以及两次较早变异记录。`git diff --check`通过；新增三文件卫生扫描0命中，均小于1 MiB。公开候选仍须对144基线加32路径的整体diff再扫。本地CPU环境不能替代正式入口真实服务器preflight，CUDA峰值与真实科学结果均UNKNOWN。旧legacy integrated专用测试要求其旧固定树而skip；本新增测试独立验证当前实际公开B多树布局。独立C结论由主执行者另行登记。

## C review 修正（保留初稿及失败证据）

独立复核发现 caller replay/consume scratch 未覆盖 CPU/native/OPS 证据树，以及 runtime 的 TorchVersion 字符串子类在 JSON 往返后被严格 type 比较误拒。按 C 档只修本新运输 adapter：replay 在写入前复用 context 的完整 immutable evidence，另排斥自身 OPS/code/data/output/scratch、全部请求 path records 和请求文件，双向拒绝同路径、子目录与父目录；已存在 scratch 仍拒绝。原科学函数与四门规则不改。

identity 的 runtime 仅在 JSON 运输边界做 canonical JSON 往返，原 preflight native_runtime 也在同一边界归一；原科学 stage receipt 直接核原始 bytes == encoded(receipt)。没有改 exact/matches，也没有放松 bool/int/float。新增真实 torch.__version__（TorchVersion）测试及 verified=1、hits=378.0、runtime flag=0 的拒绝例。SID 基线同步为 root 已审修复 f36bf15，原字节入口 SHA 为 c06a18247eab9f21d2ab981f7a657c27f5dab40d9b7058493e398638913dbba9；它属于144文件固定基线，不占32路径增量，不作额外豁免。

fresh RED：scratch 新接口缺失、真实 TorchVersion 仍为子类，见 scratch-red.log/runtime-red.log。最小 GREEN 后完整17 tests通过（34.403s），含实际 f36bf15 SID/A1 多树 launcher→metadata→dry；相关原 producer/SID/A1/四门回归76项，75通过、1项旧固定树专用测试skip（92.683s）。新12/12隔离变异全部 semantic assertion kill，原19/19完整重验也全部 semantic assertion kill。所有变异独立源码拷贝、python -B、PYTHONDONTWRITEBYTECODE=1、唯一缓存前缀，已隔离。最初新变异 receipt 负例因已有输出引发 FileExistsError，未计语义kill；旧证据保留，改为每个负例独立输出后完整重验11项，再补preflight runtime变异完整重验12项。

本轮证据均在 gitignored cache/sgns-native-a1-validation/：scratch-red.log、runtime-red.log、scratch-runtime-targeted.log、c-review-final-17-tests.log、c-review-related-regression.log、c-review-runtime-final.log、c-review-all-mutations.json/log、c-review-scratch-runtime-mutations-12-final.json/log。初次 scratch-green 的局部变量遮蔽失败、变异假kill及中间记录全部保留。没有SSH、部署、submit或读取真实 checkpoint/validation/test；实际科学结果仍UNKNOWN。独立C复审待主执行者登记。

## 固定原字节依赖

| 路径 | sha256 |
|---|---|
| `analysis/sgns_subset_a1.py` | `04d1ef8ad82a5bb291b9ed4a3a81cbdde532d0a2b8dc0e63fb1fba941c814010` |
| `analysis/sgns_subset_a1_inputs.py` | `b2800990b7c21d8549f8baeb25d45f1a4cf479b6d0722f09b1598c57fb2fc9b0` |
| `analysis/sgns_subset_acceptance.py` | `983e341009cdcc55f683b81d5c33a4ef590f5d2cf31a2eeff3534ff33acd2a50` |
| `analysis/sgns_subset_acceptance_a1.py` | `750b00663b71cd209b69b8054b7e79337209b4e8304628bf5b8ee2dff0acdd26` |
| `configs/sid_structure/g-structure-a1-s2024.json` | `7121cb5203d16dd8b909bbf0662a559c0e34a006d5dae1dfcd08b654159bceb7` |
| `configs/sid_structure/g-structure-a1-s2025.json` | `0d75b877119e90a0b7fd6ba05b4b8e1850af0a3306d52bc45867f9fafd22b7a4` |
| `configs/sid_structure/g-structure-a1-s42.json` | `9438c9574e687496e42d3ac2453c649fd7d94adc9e51371e4a14aa96d92c69d5` |
| `src/vkgr/coll_sid_a1_run.py` | `b1fce2524ea36731827c51a39f4494f549d652684b16035e21bafb9109582c78` |
| `src/vkgr/generative_b1_batched.py` | `d9f0ffba92fe57cc3dcd137b7083bdc92459561aa03d2bcacb4f5ebdb015bbb6` |
| `src/vkgr/generative_b1_run.py` | `d161aec55f74339129d556991532ec49ce9e520edc6425c7179398c4b5055af5` |
| `src/vkgr/generative_coll_run.py` | `025d6869915ecaa31c27e940c28e02033c095348b60cbecc7e6af938e50f0df2` |
| `src/vkgr/generative_g6_data.py` | `cf02dd617b8625318d31258015d29b1a5f960cc1c8c688b9f888278ca3620655` |
| `src/vkgr/generative_g6_model.py` | `ee0be45888a5d1c142e7dd4cd56b5913ed3add20c415a0760fedeeb8285b347d` |
| `src/vkgr/generative_g6_run.py` | `9c1c4c2c1f995b3e7f79343138f1f726321d109d882aafa7946a17a0997bddbe` |
| `src/vkgr/generative_pilot.py` | `95b6ad8a8115ada305734f59bbca73c9eca11c9dab75890d2dc6edd715144858` |
| `src/vkgr/generative_real.py` | `79ba4fe5770438cf3a107c7064829f04b6d64de020b7b85e0a40a727c578fa9a` |
| `src/vkgr/generative_sharded.py` | `993ac3879c9c51c8a090656e7eb25cad5185d49ac0726208bfae4d94b02efed1` |
| `src/vkgr/generative_tok_data.py` | `48498849f9c96fe4283e5a2e48e8ce3a1a1663ed47db2c9e981d737ebc73ee4c` |
| `src/vkgr/sgns_subset_a1.py` | `ec4875a625145a8465fb3e0b9ba52505583949a7d2e7975cda5284f6375ccc96` |
| `src/vkgr/sgns_subset_a1_inputs.py` | `ffe1343742a86872877a495d4287c6d68a3edbe896910c89645feee577598e8f` |
| `src/vkgr/sid_coll_export.py` | `0d7541e199dcb5397a37b6cc8eec24b9c94b3925b22100725b3fe2c935ab29fc` |
| `src/vkgr/sid_coll_run.py` | `ffd4ce26ba4c79babba3c542080c95d45fcf9eedb490217f70b5eac4841cfef9` |
| `src/vkgr/sid_coll_v1.py` | `48975e424fb21b00d2e0d1fffc4541c338cf9df5c3711fee9005a0266d33191c` |
| `src/vkgr/sid_rqvae_d.py` | `627f904fe75ec85d7cb9752203f96d374eefc4df85db52e14b2025f26cf4eb47` |
| `src/vkgr/sid_structure_a1_run.py` | `94c1fcc3a7a63520180326b0bf489e954283fa7dfbb6739c925ea579535ab20a` |
| `src/vkgr/sid_structure_run.py` | `60e62069704767aad56801c8c6ba39429c32516fb1edad09aad34918b5874e81` |
| `src/vkgr/sid_structure_v1.py` | `faeb469e9e353c6faef449730a882cf6a58f9e19c9cc30a70ae2bf507fb9b78b` |
| `src/vkgr/sid_tok_audit_d.py` | `b96c140b02649884a36c0b2db13f7e9db02316a292d0e83cc6c1c2296346958a` |
| `src/vkgr/sid_v2d.py` | `a2e1e24bb4ca1e7ac5257823113748841ca5b1bbfcd26bc88b8b39c843457e88` |
