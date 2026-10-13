密级: 公开

# P3-D 合同冻结与真实验证 ledger

## 做了什么与授权边界

Eddy批准完整传统召回goal，Astra dispatch本worker仅P3-D；不重复问继续、不自派subagent。按顺序读取输入AGENTS/STATE/handoff/tasks、data_contract/data_quality/p3c_validation/decision_log/architecture和已批准协调brief。使用using-git-worktrees建立隔离目录，verification-before-completion要求fresh证据；文档/config task不编造production RED，也不提前实现消费者。

Native worktree `<WORKTREES>/p3d-retrieval-contract/vk-generative-retrieval`；branch `codex/p3d-retrieval-contract`，HEAD `0376d9a8119f3500cbf78519675cc7bf18013c1b`。Native指定ref最初detached；fresh Git发现后 `git switch -c codex/p3d-retrieval-contract` exit0，stdout `Switched to a new branch ...`。建立前新worktree status为空，没有覆盖已有未提交文件。输入完整P3-C overlay为83个Git-visible文件，逐byte带入；ignored `cache/p3d-entry-overlay.json` SHA256 `48b1ae2d87e1f21e8924a98bae8a6a0eb8e7c1c65f3b3fa3ddf937bc9fda09bf`。不是只从main/GitHub的P1恢复。原input83文件在完成检查仍逐hash一致。

仅修改data_contract（append，旧完整bytes前缀保护）、decision_log（append）、tasks阶段指针、STATE/handoff；新增configs/traditional.yaml、本文件、小型P3-D manifest。不改AGENTS/源码/tests/lock/旧configs/旧artifacts/.gitignore、main/旧worktrees。无commit/push/stage/remote/Git配置、下载、训练、raw/holdout值读取或资源生命周期。

原指定artifacts/p3d-retrieval-contract.json被default-deny ignore排除；向协调报告后Astra批准改名 [p3d-retrieval-contract-manifest.json](../artifacts/p3d-retrieval-contract-manifest.json)，不改.gitignore。此文件可进入Git-visible overlay，不force add、不stage。

## 成果与来源

新协议 [traditional-like-v1](data_contract.md#traditional-like-v1) 是语义唯一事实源，[traditional.yaml](../configs/traditional.yaml) 是机器镜像。Manifest单向绑定合同/config完整bytes、旧合同完整前缀hash、P3/P3-C/P4来源文件、旧producer/config/catalog/fullquery/512audit hashes和当前lock；不hash自身，合同/config只放manifest路径，不放其hash。新consumer尚未实现，source fingerprint显式null，不造旧producer身份。

保留旧p3-like-v1。Like targets/history与官方split不变；主评估full eligible保留OOV，原512审计仅regression；有限core/消融/fusion/ANN预算与test已用边界明确。Static content生成时点Unknown、官方物理顺序非timestamp验证、rank排除、all-strict-past-positive mask/target override、无negative skip门以及共同输入都已登记。没有模型指标写入本文件或新manifest，数字成果不属于实验效果。

Astra本轮裁决具体化：ANN overlap/resource/latency和paired model delta复用旧固定validation512 audit及其hash，不重抽样，不能冒充full delta；模型质量/调参用exact full validation。Best architecture只比较各模型seed42 selected-config checkpoint的validation Recall50→NDCG50→model顺序，代表部署/fusion checkpoint固定seed42，其余seed独立稳定性报告。User-balance抽user/prefix均有放回。旧baseline_run.load_static同时加载两holdout与旧ShuffledCursor跨epoch补尾不可直接复用；新consumer按split读取与独立tail实现，不改历史TIGER。

## 实际验证命令与失败处理

本轮没有生产行为修改，TDD不适用；以下assert失败是声明/验收脚本定位，不能称production RED。没有重跑训练或原数据scan，没有把旧test重新称pristine。

1. Fresh独立stdlib/hashlib/YAML检查 `/tmp/p3d_verify.py` 首轮 exit1：配置OOV贡献为字符串`zero`，验证要求数字0。改机器值为0，与合同一致，refresh config绑定。
2. 第二轮 exit1：作者验收脚本错误要求旧P3 producer lock等于当前overlay uv.lock。停止定位，读取P3/P4/P5/P6 validation：P3历史lock由旧manifest绑定，P6加torch后的当前lock由已验收P6 ledger绑定。Astra确认分别验证；不改任何lock、不追写历史identity、不弱化旧producer replay。新manifest额外绑定P3/P6 validation来源，验收脚本改核两个历史/当前事实源。这是oracle假设修正，不是隐瞒producer冲突。
3. 后续fresh脚本 exit0，stdout：`PASS 266 independent assertions; 16 core configs; 65 canonical RRF tuples = 65; 78 entry files protected; 83 source overlay files unchanged`。strict duplicate-key YAML/JSON parse、独立literal规则/枚举、全部bound refs/hash与旧合同bytes前缀通过。脚本不import项目producer；expected grid与RRF枚举用stdlib独立构建，作者自检不等于独立review。
4. `git diff --check` 每次fresh exit0、stdout空。最终报告/状态落位后再执行脚本与diff/public/Git checks，最终stdout/exit存ignored cache并见下节。

可重跑命令（借用既有Mac CPython3.11/PyYAML6.0.2 env，未新装依赖、未写该环境）：

```bash
<LOCAL_VENV_WORKTREE>/.venv/bin/python -B cache/p3d_verify.py
git diff --check
git branch --show-current
git rev-parse HEAD
git diff --cached --name-only
git check-ignore artifacts/p3d-retrieval-contract-manifest.json
```

最后一个exit1、stdout空表示manifest不被ignore，不能误称命令失败。Full生产tests未重跑：源码/tests/依赖/旧配置逐hash保护，本task没有新生产实现，不凭文档验证声称旧tests本轮实跑。

## 当前限制与下一步

Implementation ready / awaiting coordinator review。独立fresh-context review由Astra另派，worker作者检查不能冒充review，当前未声称Astra通过。后续producer replay需历史source/lock环境，static consumer分别校验既有冻结字节/provenance与新实际身份；hash不认证上游真实性。真实schema/order/content snapshot风险沿旧事实源保留；合同不能证明训练收敛、ANN门通过或系统headroom。

本part先恢复完整overlay，再明确新旧版本与共同规则，最后检查机器绑定/枚举，发现并修正了OOV机器type与历史lock oracle误判。可学习点：loss mask可比模型history更长；catalog是候选域不等于正例支持域；固定旧audit用于回归/ANN不代表full validation；consumer新增source不应伪装旧producer。停止等待协调验收，不自动进入P4B或commit/push。

## 最终fresh证据

最终重跑输出与Git/公开边界证据在本节追加；hash均为实际文件bytes，不含本文自身hash以免自引用。

- Final fresh authored verifier exit0：`PASS 270 independent assertions; 16 core configs; 65 canonical RRF tuples = 65; 78 entry files protected; 83 source overlay files unchanged`。
- Final `git diff --check` exit0/stdout空；branch核验exit0为codex/p3d-retrieval-contract，HEAD核验exit0为0376d9a；staged核验exit0为空。
- Manifest ignore probe exit1/stdout空，即未被ignore；8个本task files均UTF-8/small text且无private-key/token/signed-URL payload，config/manifest无绝对路径、连接信息、raw实体数组。
- 本task review package只比较P3-C父overlay，5个修改diff + 3个新文件全文，不混入历史累计diff：`cache/p3d-review-package.md`。
- `docs/data_contract.md` SHA256 `d3a2528731d8f5b9160039b6236e519c761f618d54a4f34939ef4df5ea70efa4`。
- `configs/traditional.yaml` SHA256 `00b8122cc251465506779e79f3820fb7f5a2b674bd22e2d741dd1b65a54bc8d1`。
- `artifacts/p3d-retrieval-contract-manifest.json` SHA256 `36fc3a451f1e8bd2dc691b15c382015b6762fec8342d318ce9771bad5adfc972`。
- `cache/p3d_verify.py` SHA256 `bfab15a5f10b6254501dc3d34c33776068dfe5928cc0065a970bbb4dcdef7c8c`。


## 正式完成裁决

独立fresh-context reviewer spec PASS / quality PASS，无blocker；Astra fresh独立hashlib核验13个bound_files、83个父overlay源文件不变/78个当前保护文件、旧合同8305bytes前缀、core policy/sampling/seeds及旧producer/catalog绑定，正式验收P3-D Complete。协调核manifest SHA256 `36fc3a451f1e8bd2dc691b15c382015b6762fec8342d318ce9771bad5adfc972`。以上协调验证由Astra回传，本worker不将作者检查冒充协调检查。

本节取代上方awaiting状态，历史验证/失败保留。验收后仅同步STATE/handoff/本validation，contract/config/manifest逐hash仍一致；fresh `git diff --check` exit0/stdout空。下一P4B由协调派发，从当前最终overlay继承，不自动实施；无训练、新holdout指标、源码/旧配置/lock修改、commit/push或资源生命周期。
