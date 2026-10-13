# PG-R01 第一阶段接收报告

密级：公开。更新：2026-10-12T17:58:51.893157+08:00。实际指标唯一入口为G worktree `docs/experiment_log.md` 的PG-R01节、P worktree同文件的已接受P4F节。

## 本阶段结论

G原四配置已接收：终态、真实train/best、独立full replay成功与封存链一致，fresh集合/全成员hash通过；winner按原规则为core03（d128/LR0.0003）。三seed确认尚缺，test保持关闭。P没有真实Transformer训练结果；恢复前必须处理完整publisher闭包、source/producer身份、capacity workload和新增有限授权。

本次是接收与预算工作，不是新训练。完整baseline就绪尚不能成立，因为P4H及G补seed未完成。工程终态、产物接受和效果结论分别表述。

## 接收与可比性

G四配置同catalog/examples/SID/query/prepared SHA，all/in-catalog/OOV分母及best metric与selection一致；新stdlib reader均使用nofollow/FD身份、外部SHA和finite bounds。小JSON先限1MiB/总8MiB；大验证收据不下载，在服务器每份32MiB/总256MiB/60s提取；封存集合逐run限400k节点/2GiB读取/60s，只回小摘要。源码44、helper、实际wait日志、best checkpoint与sealed所有成员通过。没有重跑旧suite、训练、数值replay或读取test。

P accepted id_mean、ItemCF、content、Pop、wRRF作为参照；原冻结P4G仍有效，未重验P4F pipeline。G原query/catalog SHA继承同P4B/P3数据。G内容SID与P ID-only features不同：可作部署路线参考，不能将差异全部归因于架构。指标数字只见各experiment_log，本报告不复制。

## P恢复前五项工作

1. **完整publisher验证。** 两次真实失败均FD权限不可读；当前foreign UID helper source/test SHA与交接一致，但服务器仍code-03旧版。局部UID1001 probe成功不等于完整feed、owner/lock/链、controller、关闭和新verifier通过。不要跳过未知进程或删旧锁文件。
2. **重建source与输入身份。** 最新source与旧producer consumer SHA不同。旧train-inputs-02不能改SHA重签；用冻结raw/prepared重新生成新producer及独立目录。旧prepare实测184.196秒，仅此组件实测，不是全部成本。
3. **注册capacity workload。** 当前sidecar有108286条（like18915＋observed89371），两个arm都消费全部sidecar。与正式2520 query相比约43倍，而且worker/独立verifier反复全catalog exact，不能据此承诺4h完整pilot。建议新version取2520 matched train users，每arm2520条；按like历史长度降序、observed长度降序、seed42用户hash tie选择，以偏保守负载。保留最大50及完整120415 catalog；target-free，不读取holdout、计算质量或改变训练86785例/正式2520验证分母。该改动尚未实现，不是偷偷缩验证。
4. **外部有限授权与准入。** 原2/2attempt及过期deadline保留；新的一个attempt登记为累计第3次，原失败和spent不退款。G完整终态可以成为安全门依据，但current permit仍需启动前fresh核资源/身份/absence/unknown writers。

5. **新增授权的机器表示。** 当前Budget硬编码pilot≤4h、attempt≤2，并要求原queue receipt一致；直接填attempt3或改旧deadline会被拒绝。必须新增明确的extension-authority adapter，绑定原queue SHA/两失败收据/旧spent和新human grant，独立记录新增4h与累计attempt3，旧queue不改。不能通过把attempt改回1或取消上限来绕过。

现有controller只有caller填写stage_seconds/finish_seconds/next_atomic_bytes，不能把写入任意数字当作完整formal成本证明。新增cost receipt必须从实际pilot推出30epoch上限、full validation/fresh-best/独立verify、原子写旧+新峰值及cleanup，再由admission检查；缺完整测量保持STOP。

## 当前现场

旧P目录logical158348551 bytes、allocated158691328 bytes，218节点；P/G原owned sessions在本次只读扫描中均不存在。共同资源读到GPU utilization0、free VRAM约98.8%、disk约47.1%；RAM取host/cgroup较保守者约38.3%。这是核查时点，不是未来或当前permit。实例当前仍运行，原唯一G终态receiver保留，本轮无新增monitor/生命周期操作。

旧heartbeat vk-02实际PAUSED，Sol04读到idle；Eddy明确要求当前对话开始PG-R01，当前对话接收唯一执行，原worktrees/作业/spent不变。实时context样本UNKNOWN，不猜比例。

## 证据与下一步

Main ignored cache/pg-r01-reception-v1：initial-read、metrics-read、metadata、seals、closure-0..3、p-scope的stdout与hash receipts；g-results.json、selection.json、budget-proposal.json、task-reception.json。未将raw/历史/逐query/checkpoint/完整验证收据带回本地或Git。

[下一执行计划与预算](../superpowers/plans/2026-10-12-pg-baseline-next-execution.md)。本报告与selection已经一次fresh接收review APPROVE（Main cache/pg-r01-reception-v1/fresh-review.md）；新试跑授权尚未取得，下一阶段features/HSTU不启动。
