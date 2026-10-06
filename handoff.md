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
