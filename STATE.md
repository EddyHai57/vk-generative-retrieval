# STATE

密级: 公开

> 当前状态唯一事实源。历史证据见`handoff.md`，路线定义见`tasks.md`。

## 当前阶段

- Phase：P0 — Bootstrap与治理
- Status：Complete
- Bootstrap snapshot：已提交；具体hash以fresh Git检查为准
- Next gate：Codex接手前确认P1 scope
- P1：尚未开始

## 当前仓库

- Branch：`main`
- Remote：0
- GitHub remote：未配置
- Production code：无
- Dependencies：未安装
- Data：未下载
- Server：未连接
- Experiments / metrics：无

## 当前阻塞

- 无P0阻塞

## Housekeeping

- 旧`AGENTS.draft.md`保留在本地并已被`.gitignore`忽略
- 删除或归档旧草案需要Eddy明确决定

## 下一任务

Task：P1 — 基础代码与可审计运行框架

Owner：Codex

开始前：读取`AGENTS.md → STATE.md → handoff.md → tasks.md → docs/codex_handoff.md`，复述P1 scope、验收和stop gate，等待Eddy确认。

## 禁止提前执行

- 不跨越P1执行P2及以后阶段
- 不创建GitHub remote或push，除非Eddy明确批准公开操作
- 不连接Thunder或创建计费实例
- 不下载VK-LSVD
