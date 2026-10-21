# Native B 子集 SID 运载实施报告

## 范围与固定源码增量

本实现仅新增 `analysis/sgns_native_subset_sid.py`、`tests/test_sgns_native_subset_sid.py` 与本报告，不修改任何已有文件。依据执行者先做后审工程授权，为已选定的 B producer 接回原 48³ SID；不包含 full backend、A1 或四门转正结论。独立 C review 由主执行者另行分派。

实际公共 producer 基线为 `305c415871bd60dc0d7d8a064a7e893830a9c20f`。它已有原 science 模块，但没有原 `analysis/sgns_subset_sid.py`。因此公共候选的精确增量为四路径，替代此前三个新文件的假设：上述三个新文件，加原 CLI `analysis/sgns_subset_sid.py`，其 sha256 必须为 `bf824fceb8b95543dec46c7f717112e252fcba8806cd8fd2a569c3703b3c9dcb`。后者只按原字节补入，不修改科学逻辑。请求明确登记四条增量的实际 sha256，拒绝预存文件变化、开放式新增、缺失、重叠及错误旧 CLI pin。原 science pin 为 `6ddc0169619e4d17a5e6749f06a239d5c4ff7217032f903c967e4efdf1a8da21`。

静态递归 import 闭包中另外两个基线缺失模块仅属于旧下载和旧 inventory/main 路径。本入口只调用 runtime 与 `sgns_gpu_subset.fixed_records` → `a40_subset_prepare.inventory`，不进入旧下载路径。实际305部署测试导入这些调用对象并执行固定 metadata inventory，且确认缺失模块仍不存在；不为无关路径扩充源码增量。

## 最小适配与科学复用

沿用仓库 `src/vkgr/sgns_subset_a1.py` 已存在的 `FunctionType` namespace facade 模式。原 execute 的 code/defaults/closure 保持同对象，只在独立 namespace 替换封闭的 load_request、input_stage、cpu_reference、own_job、successful_stage 五个运输 callback；science proxy 仅替换 load_bundle namespace 的 source-delta 检查依赖。原 module globals 从不修改，包括异常退出。原 fit、同 seed replay、usage gate、48³、seed42、max_iter10 和 collision×1.10 全部继续执行原函数对象。

B input 由固定部署的 native producer consumer 完整审计，显式保留 `sgns_backend=B`；不伪造旧 numpy SGNS audit。原 CPU reference 仅复核原 successful verify、runtime、完整 source ledger、原 SGNS stage 和已拟合 bundle，不重拟合或重放 CPU mapping。严格检查 CPU 原 terminal 的 bool wall_cap_exceeded=False。

新外层 `sgns-native-subset-sid-stage-v1` receipt 明确 backend B 与 authority4b0a8a1，内层原 science receipt 原样保留，闭合 native producer、CPU reference、request、原 runtime、canonical input manifest SHA、实际 job argv/cwd/env/resources/inputs、terminal 和 stdout。每阶段开始及结束复核实际输入。consume 返回 LoadedSID、native_input、verification、job_proof、fit_job_proof、cpu_reference；重查原 gates/collision/replay，不声称已通过 A1 或四门。

## 验证

fresh RED 包括最初缺失入口的 ImportError，以及固定四路径/原 CLI pin 用例；最小 GREEN 后跑相关回归。实际 launcher 测试直接从305分别构建 producer 与 SID 两棵树，SID 树只补四路径并生成干净 commit；OPS 使用固定原 launcher，执行 plan → launcher.prepare/metadata → 实际 run binding → 同形式 CLI dry。覆盖22个输入角色、继承环境拒绝、非法 CLI 非零和合法 dry 非空回执；不 submit。测试中的 bundle/CPU receipt 只是运载 fixture，不替代真实只读 preflight。

本轮相关回归：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 <project-python> -B -m unittest tests.test_sgns_native_subset_sid tests.test_sgns_subset_sid tests.test_sgns_gpu_native_fit`，33 tests PASS，89.573s（含10个新入口测试及实际305双树launcher）。

13/13 作者变异被 kill：旧源码字节、delta成员、旧CLI固定pin、callback封闭集合、原namespace、own job、RSS margin、外层producer、内层绑定、own wallcap、CPU wallcap、CPU runtime/replay、input manifest pin。每项在独立新副本，使用 python -B、PYTHONDONTWRITEBYTECODE=1 与唯一缓存前缀，已隔离；未变异独立 control 9 tests PASS，5.276s（不含需要Git双树的launcher用例）。原先失败与初次证据均保留。

证据指针：gitignored `cache/sgns-native-sid-validation/four-delta-red.log`、`four-delta-regression.log`、`four-delta-mutations.log`、`four-delta-isolated-control.log`、`mutations.json`；`git diff --check` 通过。新增三文件无本机绝对路径、credential或IPv4命中，均小于1 MiB；公共候选仍须对压平整体diff再扫。

## 限制与交付

本机 CPU 环境，没有 SSH、服务器作业、真实数据 preflight、CUDA fit、validation/test 读取或 A1 运行。真实部署仍必须先 push 固定 SHA、逐文件校验并通过正式入口真实只读 preflight；仅本地 PASS 不构成科学验收。原 A1 engine/旧四门 consumer 仍绑定旧 producer，需单独最小运输适配及授权后的唯一 A1；不得直接将此 SID 成功当作四门通过。

私人日志和变异副本位于 gitignored cache 及临时目录，不进入公开 diff。未修改旧 SGNS/native core、SID/loader 算法、门槛或原失败记录。公开前由主执行者针对实际305加四路径的整体 diff 扫描并登记固定 delta；私有实现分支不直接作为发布树。

## 独立 C review 后的运载补证

复核者发现原 CPU SGNS fit loader 未拒绝 terminal wall cap，且 CPU SID fit 前驱同样缺少显式检查。修正只在本新增 adapter：分别要求原CPU SGNS fit、CPU SID fit terminal 的 schema/run_id/status/exit/signal/bool wall_cap_exceeded=False；旧science逐字节保持。另将SID新输出与CPU bundle、SGNS output/run、SID fit/verify output及OPS整棵证据树进行双向祖先/子孙排斥。

两个缺口分别有 fresh RED；cap修正后33相关回归PASS（76.239s），最终新增cap/overlap targeted GREEN通过。全部15项隔离变异kill（包含新增CPU前驱cap、CPU输出重叠）；此前失败证据原样保留。该C档修正交原独立reviewer补审，未替独立C作结论。证据：cache/sgns-native-sid-validation/cpu-*.log。
