# 固定运维入口参考笔记

依据：3c4cd42第一部分、a87b5bf第1–3条；本项是草稿，不安装权限，不改变scheduler。

|成熟做法|依据与成本|本轮选择|
|---|---|---|
|just / Makefile|[just](https://github.com/casey/just)（CC0-1.0）提供recipe和`--list`；Make同类、但原用于依赖构建。recipe本身仍可执行shell，不能充当权限隔离。|薄just recipes仅调用无参数入口；不allow裸just或可替换justfile的全局参数。本机未装just，recipe仅静态核对，正式子进程测试走同一底层入口。|
|MCP类型化工具|[MCP tools规范](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-06-18/server/tools.mdx)定义inputSchema；服务端仍须鉴权与校验。需要维护server、传输和部署。|本轮不用；当前需求已有CLI，不新写框架。|
|现有ops/bin与ops/tools|固定5840e6b已有launcher/scheduler/status、deploy_spec和g_deploy；服务器仍运行原版本。|复用status与新worktree部署校验；不重写调度、准入、launcher。|

入口范围为status/resources/queue/receipts/deploy及plan（本地固定计划）。所有CLI只含固定词，ssh别名/环境内部设置；不接受路径、SHA、分支或任意远端命令。deploy仅把已公开SGNS工程版本661a444部署至固定新目录，不运行模型。暂省metadata/dry/submit：当前测量作业已claimed，不能重投；第一批科学请求还未获批准。暂省push：公开扫描和压平发布仍走人工可审查流程，不给动态分支入口。3c4cd42允许按现有工具收敛子命令；后续扩展须重新review并审规则。

服务器端实际拦截点（旧代码路径均指5840e6b；本地只恢复两份原字节工具）：

|边界|代码位置与实情|
|---|---|
|新目录、逐文件SHA|ops/tools/remote/g_deploy.py:13先拒目标存在，:15调用git worktree，后续核target_tree与固定pin；新ops/fixed/server.py调用前再拒绝含悬空symlink的目标及pin目录，并限制为固定计划；pin目录由wrapper排他创建，拒绝预查后出现的目录。工作树和Git管理元数据会新增；不覆盖既有部署。|
|只读deploy key|由服务器Git/SSH配置控制，代码不证明远端key权限。原部署收据只能证明fetch成功；仓库只读权限需owner核查，不能把root SSH通路称最小权限。|
|holdout路径|旧launcher没有通用holdout路径拒绝；新入口两端只允许固定词及固定非holdout对象，任何传入路径（含week26/test/holdout）在调用原工具前非零退出。没有为原SSH/root通路新增系统级隔离。|
|请求pin与准入|ops/vkgr_ops/scheduler.py:26 submit调用launcher.prepare；launcher.py:179核pin完整性、:273 prepare核源码/输入/config。不是所有请求都强制提供pinned manifest；不能声称旧scheduler全面拒绝“未pin”。本轮没有submit子命令。|
|小型收据|新ops/fixed/server.py只读固定run_id的terminal.json/run.json，限制大小并拒symlink；不取原始数据、stdout、逐query预测或test指标。|
|客户端规则|[Codex Rules](https://developers.openai.com/codex/rules/)仅匹配argv前缀；多余参数由入口拒绝，目标存在由服务器拒绝。命令链需check独立验证；不会因为首命令allow就声称整条链allow。|

allow草稿是客户端便利设置。相对入口依赖可信项目cwd；源码被改写、其它既有allow规则、root通路均不受这份草稿独立约束。裁决者审过后由Eddy安装；规则失效退回逐次确认，不改本机配置。调研原文、Git blob标识和check原始输出留cache/ops-entrypoints-20261021/；未复制第三方实现代码。
