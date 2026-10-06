# Handoff

密级: 公开

> 新对话接力入口。当前阶段、阻塞和下一task只读`STATE.md`；路线定义只读`tasks.md`。

## 恢复顺序

1. 读取`AGENTS.md`
2. 读取`STATE.md`
3. 读取本文件的“上一轮证据”和“恢复检查”
4. 按`STATE.md`指定的task读取`tasks.md`及相关`docs/`
5. 先复述scope、验收与stop gate，等Eddy确认后执行

## 上一轮证据

- Eddy手动将thin规则落位为`AGENTS.md`
- Harness已拆分：AGENTS保存稳定规范，STATE保存当前快照，handoff保存接力证据，tasks保存路线
- README改为公开导航；CLAUDE只保留兼容指针
- P2/P3/P4/P8/P9阶段owner冲突已修正
- P1 Smoke固定为synthetic/local CPU/no network
- `.gitignore`采用default-deny，拦截数据、大产物、凭据和本地治理草案
- Scaffold与Harness的fresh-context review最终通过
- Eddy明确批准创建bootstrap commit；本handoff随P0 scaffold进入该commit

## 最近验证证据

- Thin AGENTS内容与预审版本一致
- P0～P13阶段编号和owner对齐
- 3份YAML解析与关键contract检查通过
- 7类敏感future paths的`git check-ignore`检查通过
- Secret、内部字段与废弃口径定向搜索无匹配
- 无生产`.py/.sh/.ipynb`和空文件
- Commit前本地Git为`main`、commit=0、remote=0

这些是上一轮的时间点证据，不是永久当前状态；新对话必须重新读取`STATE.md`并执行必要的只读检查。

## 恢复检查

在执行`STATE.md`的下一task前，按任务需要核对：

```text
AGENTS/STATE/handoff/tasks是否存在
当前branch/commit/remote
git status与staged files
YAML和本地链接
secret/data ignore边界
```

不得仅凭本handoff中的历史输出声称当前仍然通过。

## Housekeeping记录

- 旧`AGENTS.draft.md`未删除
- 文件已被`.gitignore`排除
- 是否删除或归档由Eddy决定

## GitHub说明

GitHub remote不属于当前状态事实，本文件只保留方法：

- P1本地开发不依赖remote
- 创建public remote和push属于外部公开操作，必须单独确认
- 可以由Eddy手动创建，或明确授权Kiro使用`gh`

## 下一次接手重点

以`STATE.md`指定的P1为准。

P1只实现基础代码与可审计运行框架：config schema、run manifest、logging、deterministic seed、synthetic fixture和targeted tests。

不得提前下载真实数据、连接GPU或实现P2～P13。

## 新对话启动提示词

```text
请先读取AGENTS.md、STATE.md和handoff.md。
以STATE指定的下一task为准，再读取tasks.md和docs/codex_handoff.md。
先复述scope、验收与stop gate，等我确认后执行。
不要自动创建remote/push或计费资源，不要跨阶段。
```

## P1启动接力（2026-10-07）

用户已批准P1 scope、独立worktree与首次必要依赖联网准备。当前隔离worktree fresh核对：基点df32a1a16a03d6f40e1a1516d95d513413a49916、clean、remote 0；detached HEAD已切换到同基点的codex/p1-runtime。Git共享metadata默认sandbox写入失败，经授权escalation后成功；没有修改main文件、Git配置或创建重复worktree。Python 3.11.17、uv 0.12.23为主协调对话安装的接力证据，本轮已fresh复核；系统Python只启动uv，不运行项目。

## 最新P1交接 — 待主协调Agent验收（2026-10-07）

本节承接上方P0历史记录；P1 scope、独立worktree与首次必要依赖准备已经用户批准，不重复询问。当前状态唯一源是STATE。

- HEAD与批准基点一致；branch codex/p1-runtime，remote0，staged为空。未commit/push/改Git配置；原main fresh检查clean。
- 串行完成config→runtime→manifest→CLI，未委派worker，未实现P2及之后接口。
- 严格config schema、重复YAML/JSON keys拒绝、人工fixture校验、独立seeded整数计算、exclusive输出目录与文件、stderr logging及exit codes、manifest schema v1与hash/语义校验。
- 当前worktree独立.venv为Python3.11.17/PyYAML6.0.2；首次uv0.12.23联网真实生成uv.lock。只审计本轮PyYAML MIT dependency，未来I-007仍Open。
- fresh full unittest 31项通过；两次persistent smoke和各自verify通过，result字节/hash相同；manifest除时间/elapsed/output路径外字段相同。
- Persistent产物：`outputs/p1-20261007-6ec7-a/`、`outputs/p1-20261007-6ec7-b/`。完整hash/命令及实际manifest检查见`docs/p1_validation.md`，不把模型效果数字写进本文件。
- 8个tracked授权文件修改、17个untracked授权文件；两份小manifest保持untracked可review，两份result按既有规则ignored，.venv ignored，无残留test temp目录/未解释repo文件。无真实数据/模型产物/secret/内部内容。
- RED/GREEN ledger保留所有真实失败，也标注了无效errors、oracle修正、symlink测试误通过与文档尾空格失败；不补造历史。
- 关键复盘见issue I-017～I-021；实现作者已读回自检并补fixture symlink修复，自检不冒充fresh-context review。

下一动作：主协调Astra在本worktree fresh核对所有diff（包括untracked），运行`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`，用全新唯一P1目录smoke/verify并检查manifest，复核hash/fingerprint/Git/secret/大产物边界。当前只称Implementation ready / awaiting coordinator review，不称P1 Complete。

Known limits：hash检查一致性而非真实性；verify要求保留匹配输入/源码，源码fingerprint变化将明确拒绝；不实现并发输入修改/symlink竞态或I/O transaction，失败目录不自动删除也不复用。未测峰值RAM/CPU，无模型质量评估。最终review、commit、merge或P2均未执行。

## 最新协调验收 — P1通过，停在commit前（2026-10-07）

本节取代上方“待主协调Agent验收”的下一动作；上方保留为实现交接历史证据。

- 主协调Agent未参与P1实现，本轮独立读取全部生产源码、测试、tracked diff、untracked文件及验证ledger，完成fresh-context review；在批准P1范围内无阻塞发现。
- Fresh完整unittest：31项通过、exit0。新建outputs/p1-coordinator-20261007-a与-b，两次CLI smoke、verify及cmp均exit0。
- 独立hashlib检查config/fixture/result/source fingerprint，核对manifest必需字段、运行环境、Git基点、repo-relative路径和稳定字段一致；具体hash与命令统一见docs/p1_validation.md的协调验收节。
- 原main仍clean；branch codex/p1-runtime、HEAD仍为bootstrap、remote0、未stage/commit/merge/push。仅协调验收产物与治理文档有新增变化，生产源码未修改。
- 当前8个tracked修改、19个授权untracked；四份result与.venv ignored，无未解释文件。P1 Complete不表示代码已合并到main。
- 历史RED/GREEN按ledger审查，未声称重新执行历史中间版本；当前GREEN由协调Agent重新运行。
- 限制继续有效：只证明synthetic scaffold；hash不认证真实性；不保证并发输入变动或I/O transaction，不包含模型质量、GPU或真实数据结论。

下一动作：等待Eddy明确批准commit或下一步。不要自动进入P2；本worktree与全部未提交证据必须保留。

## P1本地整合授权（2026-10-07）

Eddy已明确批准P1 commit并合并main，当前开始本地整合；未批准本次remote/push或自动租用服务器。新增阶段汇报要求已写入docs/codex_handoff.md。提交前fresh unittest再次31项通过（2.757s，exit0）。四份运行manifest/result保留原worktree本地，hash已在验证文档保存；本次只提交原创源码、配置、lock、synthetic fixture、测试与文档，不将临时运行目录纳入提交。整合完成后再更新当前快照。

## 最新本地整合结果（2026-10-07）

本节取代上方等待commit/merge的下一动作。Eddy已批准本地commit与合并：P1实现commit为9f8f6555f41931019c968477920a69d36699fb55，24个原创代码/配置/测试/文档文件，已fast-forward合并main。随后追加本次状态与验证收口文档提交；最新HEAD以fresh Git为准。

main以uv --offline --locked从本地cache建立自身.venv，fresh完整unittest 31项通过（2.834s，exit0），包含CLI smoke/verify、确定性与边界验证。未借用其他worktree产物。生产源码没有额外修改，source fingerprint保持不变。

四份persistent manifest/result保留原P1 worktree且未纳入提交，hash见docs/p1_validation.md。分支/worktree均保留，未删除任何运行证据。原main的AGENTS.draft.md仍ignored。Remote0，未push，未连接服务器/使用GPU/下载真实数据。

下一步：用户自行创建空GitHub仓库；remote/push需后续明确授权。P2须单独确认scope、预算、SSH访问与acceptance probe。截图中48GB RAM低于runbook的64GB门槛，已建议8vCPU/64GB，尚未把推荐当作实际实例事实。每part按docs/codex_handoff.md新增的过程/成果/问题处理/下一步顺序汇报。
