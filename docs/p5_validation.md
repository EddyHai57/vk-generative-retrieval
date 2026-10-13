密级: 公开

# P5 validation

## Scope与输入

只执行tasks P5。Eddy过夜授权/Astra brief允许独立worktree、现有服务器真实fit与独立review；不commit/push、新资源、删除旧产物或进入P6。Worktree codex/p5-sid基点0376d9a8119f3500cbf78519675cc7bf18013c1b，P4验收29文件overlay逐字节复制，ignored入口ledger cache/p5-entry-overlay.json SHA a5cdca61a13b45f37facc77085804f8f336e6942f12b51e72356cfd1ecdd7c35。原main/P2/P3/P4文件不改。

允许新增sid.py/sid_run.py、synthetic tests、configs/p5.json、当前validation、必要治理文档与脱敏aggregate manifest。P3/P4生产源码、config、lock、冻结合同和baseline evaluator原样保留。

固定P3合同SHA3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f；P4 run manifest SHA65d90cc9fb4df88824786a794349e3e2a6724ff0a8e08e4f95b573c699bdb801、source51db4ea2320b0f8aae0a92d2036ba273bd1984966c6db553b640a5f17956176e。只读train catalog与官方64D NPZ；P3 catalog/hash/policy/producer先校验，NPZ hash与P4 input记录一致，再按原P4 join_normalized ID join。P5不读holdout，不改512/split audit set。normalized feature cache完整留服务器，source/raw/hash与原transform绑定，不声称旧P4存过normalized cache hash。

## 预注册算法和gate

Astra在任何真实fit前批准：branching16、semantic depth3、seed42、每node最多10 Lloyd iterations；k=min(16,node size)。Canonical sorted uint32 IDs，原P4 normalized float32 64D。Node init为seeded无放回抽k行，Euclidean arithmetic mean centers（float64，不强行renormalize）。Distance逐维float64累加，不使用GEMM与epsilon bucket。全部item-child边按distance、item ID、child ID排序，贪心填floor/ceil容量；这是一种capacity-constrained greedy balanced hierarchical kmeans，不称全局最优。labels不变才converged，耗尽10iter显式unconverged；objective逐轮记录，允许非单调。

每item完整SID固定3语义digits+1 suffix；叶内item ID升序suffix0..size-1，位置token p0:v/p1:v/p2:v/p3:v互不混淆。小catalog继续k=1子树至固定depth，digits自然为0。静态catalog-only encode，OOV拒绝，不做new item assignment。Compact constraint保存sorted mixed-radix uint64 SID keys和uint32 item permutation，prefix通过整数range+binary search查询；合法完整SID无下一token是terminal，不算deadend。

Gate：fullSID collision0、所有catalog item encode/decode roundtrip和reachable、合法nonterminal prefix deadend0、非法prefix/suffix拒绝、每node child size diff<=1、max leaf occupancy<=ceil(N/4096)、采样visible RAM/disk余量>=20%。Semantic collisions与suffix修复后fullSID分开，记录occupancy histogram、suffix vocabulary、utilization、entropy/Gini、prefix occupancy、codebook/node init/iteration/convergence/objective、map/constraint/hash/size/runtime/RAM。余量仅可见host指标，不保证hidden quota。

## 接口和artifact边界

sid.fit/encode/decode/allowed_next/audit与toy collision_hits；后者显式semantic prefix hit不等于完整item hit，不是新的ranking evaluator。sid_run.load_static接受调用方pin的manifest SHA，验证mapping、codebook graph/capacities、constraint、feature与audit交叉一致；P6消费不会要求当前source等于P5 producer。sid_run.verify_run仍严格validate_producer、重新读官方feature、refit并逐map/codebook比较，不削弱P3原producer guard。

完整mapping.npz、constraint.npz、codebook.json、normalized-features.npy及audit留server-only ignored outputs；Public只允许小aggregate和hash，不能复制完整SID、IDs/centers/query到Mac或Git。无新增dependency或第三方实现复制。

## 真实TDD ledger

所有Mac命令在P5 worktree使用PYTHONPATH=src .venv/bin/python -B -m unittest。缺module/API errors如实记录，不冒称assertion failure；test中smallcatalog/invalid/geometric案例在R02补覆盖时已GREEN，不伪造它们曾RED。

| Round | Fresh RED | GREEN |
|---|---|---|
| R01 core | test_sid.py缺sid module ImportError，FAILED errors1；shell末tail使exit0但log清楚FAILED，非通过 | core balance/identical/shuffle/bijection/prefix/OOV1test0.042s exit0 |
| R02 collision distinction | 4tests只有collision_hits缺API ERROR，exit1；其他边界覆盖即时GREEN | 4tests0.042s exit0 |
| R03 artifacts | 缺sid_run module ImportError，exit1 | 2tests0.023s exit0，pin tamper与constraint item permutation rehash拒绝 |
| R04 runner | 缺load_train_features/run/check_headroom APIs，3ERROR，exit1 | 5tests0.192s exit0，freeze tamper先拒绝、codebook center rehash在完整fit replay拒绝 |

入口full P1-P4 58tests6.775s exit0 skip1。Mac v1 full67tests3.595s exit0 skip1（66实际执行）；Astra另fresh67tests3.610s exit0 skip1。Server v1同源码67tests2.530s exit0 skip1；FAISS源码未改，沿用P4已实际通过的reference，不把skip当FAISS通过。

## 环境和候选v1

Mac python3 -m uv sync --offline --locked成功，独立.venv CPython3.11.17、PyYAML6.0.2/NumPy1.26.4/PyArrow19.0.1。初次uv裸命令PATH缺失exit127，使用已安装module修复，非TDD RED。Server new .venv offline缺NumPy wheel exit1，非代码RED；复用P4 code-v1已精确依赖环境CPython3.11.16，未修改该环境或lock。完整server guide与工作区规范已读，SSH只用alias，未记录connection secret。

v1 source fingerprint a0f808c5081301dcb33b42845c1a75232352a34e0a19f404eac3199a8019b969。37文件原创/synthetic包SHA2500b933ab4b65b65796e5cc9ad1d10e31e277546bc6d014e241388cfa4c4eee，server /workspace/vkgr-p5-20261006t2050/code-v1（root时间只是label，非实际start）。独立tmux socket vkgr-p5-20261006t2050，输出outputs/p5-sid-v1-20261006t2032，stdout/stderr/exit在code-v1/cache。真实run于20:32附近启动，尚不称完成。

独立Sol review在v1运行期间指出candidate blocker：success manifest发布先于最后资源gate，monitor异常不能传播。v1不得验收或交P6；保留候选全部证据，等待完整review后集中TDD修复，新版本/新run重新验收。尚未宣称P5 Complete。

## fix1 candidate

独立Sol完整review只发现I-035/I-036资源阻塞，无其他算法/binding blocker。fresh fix1 RED三assertion FAIL（末资源低仍manifest、monitor OSError未抛、static缺producer/resource通过），exit1；fix1 GREEN12tests0.739s exit0，完整Mac70tests4.119s exit0 skip1。Data writer现在只返回unpublished bundle，run在monitor stop/join、最后sample/boundary和success schema后最后xb发布；static schema必须producer/resource/sample/time/elapsed。采样每250ms，fit每node/iteration只检查已观测状态；不保证瞬时硬限制。所有失败部分数据保留，manifest不存在即不可被static消费。

v1自然完成exit0/stderr0，原manifest SHA cf3e4f04b5da41b46b5f4375d61466e295baaee078ec7e837ab58d31204020fa。该候选因review gate失败不验收、不交P6；旧code/source/log/output不改。

fix1 source fingerprint c61ffbc1651dd0b8bd508f498cdde265de886fd4df45aeede6f7ebc1e2e65729；37文件source-v2包SHA be4b7c38675051821f67abd86125833d22d5e2e35ae1755106c4dbdc2026bcf5。全新server code-v2与outputs/p5-sid-v2-20261006t2040真实重读官方NPZ再fit；不复用v1 cache或重hash掩盖变更。Run ID时间为label，实际start由manifest记载。

独立Sol fix1 scoped re-review通过，无新问题；Astra fresh local70tests4.166s exit0 skip1。Server v2 full70tests3.018s exit0 skip1。一次治理文档补记Python stdin Non-UTF-8 SyntaxError真实失败，后用项目.venv Python和明确UTF-8重试；先前独立cat写入issue/decision已成功，不把shell末git成功当整个补记通过。

## v2真实run和独立array oracle

真实v2 run exit0，stderr0；server manifest SHA669029504ab18df5f6a6109820d3d4a36a243f6ab0d3e8c46fc8ce17b1b6bbe9。作者独立脚本cache/p5-independent-audit.py只import NumPy/stdlib、不import vkgr，重核全部artifact SHA、mapping IDs逐字等于冻结catalog、完整SID uniqueness、mixed-radix sorted keys/items、逐item roundtrip、全部prefix与非法next edge、suffix按item ID升序、node capacities/iteration记录。其aggregate输出在server code-v2/cache/independent-v2-audit.json；数值摘要只读experiment_log。作者自检不冒称Astra fresh-context验收。

| Server-only artifact | SHA256 |
|---|---|
| mapping.npz | 9c0c76600dec91957a87119405018793b7216db28f07597556ef79876e4ebfc7 |
| constraint.npz | aa15fc6be69c7169ab9e13fac73c337b40c08403fff52720e3bc04290f2f242f |
| codebook.json | 179f92b081bb56a6c67daaa102f828f2fdd7aa4a68fb679e071036cc79a58d5e |
| normalized-features.npy | a4f04e06bea02779b592690b977ecf285c1d5d41e66c19c8be8c0a6b0810f1e8 |
| audit.json | 1c4df044f1702fcdc5c0cfb9f17c3429318c7134d43187b92afcbf82c5545b9e |

Canonical little-endian float32 normalized array bytes SHA766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84；与NPY file hash不同，后者包含header。Public artifacts/p5-sid-manifest.json SHA26b9073f6bf5685965f12e528a4b8803710bdb108229da7b94d1247a1f5453cc，只含input role/hash、artifact hash/size/boundary与source/config，不包含paths/IDs/SIDs/centers或features。它不是server manifest字节副本。

完整replay由cache/p5-verify-v2.sh后台执行，命令如下；当前完成状态只读STATE。

```bash
cd /workspace/vkgr-p5-20261006t2050/code-v2
export PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 GIT_DIR=/workspace/vkgr-p2-20261006t1755/.git GIT_WORK_TREE=/workspace/vkgr-p5-20261006t2050/code-v2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/workspace/vkgr-p4-20261006t2011/code-v1/.venv/bin/python -B -m vkgr.sid_run run --manifest /workspace/vkgr-p3-20261006t1855/code-v3/outputs/p3-train-v3-20261006t1920/frozen-contract-manifest.json --raw-root /workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183 --p4-manifest /workspace/vkgr-p4-20261006t2011/code-v2/outputs/p4-baselines-v2-20261006t2023/run_manifest.json --config configs/p5.json --output outputs/p5-sid-v2-20261006t2040
/workspace/vkgr-p4-20261006t2011/code-v1/.venv/bin/python -B -m vkgr.sid_run verify --manifest outputs/p5-sid-v2-20261006t2040/run_manifest.json
```

Run输出exclusive，不能复用既有v2目录重跑。所有历史失败/候选仍原目录保留，无delete/commit/push或资源生命周期操作。

### v2完整producer replay完成

cache/verify-v2.exit=0，stdout为`P5 input/fit/map/codebook/constraint/audit replay verified`，stderr0bytes。它严格核当前P5 producer/source/lock/env/config，重读固定P3/P4 input与官方NPZ/hash，重新生成normalized feature bytes和全部greedy fit/map/codebook，并与stored map/node init/centers/objective/iterations逐canonical比较；static validator另检查完整constraint/audit。不是只读checksum或只凭exit0。原run manifest SHA仍669029...bbe9，source c61ffb...5729不变；verify没有修改原run。

Public manifest JSON/schema、role-only input/hash、private-root/connection/ID/secret marker扫描通过，SHA仍26b907...53cc。七个P5新增public/source/test/config文档路径定向扫描通过、I-035/I-036主issue编号各唯一、git diff --check通过。P3/P4受保护16文件source/config/lock/public frozen input逐hash与入口overlay一致，完整大产物路径ignore检查通过。无commit/push/delete/P6。

当前Implementation ready / awaiting coordinator review。独立Sol source/spec/quality fix1 review与Astra localfresh通过；Astra远程fresh exhaustive/static/hash及最终docs/public review另作为正式P5 gate，不以作者独立脚本替代。

### Astra正式阶段验收

Astra确认P5 Complete。Fresh local `PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`：70tests4.166s exit0、既有FAISS skip1；独立Sol source/spec/quality fix1与最终docs/public manifest review通过，无新阻塞。FAISS源码未变，P4已实际通过reference证据继续适用。

Astra SSH独立NumPy/stdlib、未import P5生产模块核对全部120415catalog IDs、完整SID unique、suffix按ID升序、4369prefix、constraint sorted keys/items、273node sizes、feature bytes与5artifact SHA全部通过；不把作者oracle当独立review。真实cache/verify-v2.exit0、stdout明确input/fit/map/codebook/constraint/audit replay verified、stderr空已被Astra另行读取。

第一次coordinator命令错误省略code-v2路径造成FileNotFound；随后按本validation真实路径纠正并重查通过，无artifact/source/public manifest变化，不把失败首命令省略。原manifest/source/public SHA继续保持本文件既有pin。

Astra正式关闭P5，下一task P6已在过夜授权内另行dispatch，本worker仅更新STATE/handoff/validation并停止，不进入P6。不commit/push，原v1/source/log/outputs不删。聚类非最优、未收敛与objective非单调、250ms观测无瞬时硬保证等限制继续有效，不能将阶段通过升级为模型效果或全test claim。
