# B A1 有限接续（实现，待独立 C）

依据6f8b56e§5–6及4b0a8a1；这是现行有限驱动的B家族适配，不改科学入口、门禁、seed、配置、名额或资源。仅新增driver/test/本报告，原driver与所有失败记录保留。

设计：owner先提交固定96bfb9e原A1 PF；独立driver PPID1、900秒poll、最多200h总截止。只消费同一request SHA、同一完整source/runtime身份。原PF成功→原train→原verify→原accept，任何failed/rejected/消失/漂移/截止即结构化ALERT，无自动重投。accept通过后结束；不提交full fit，不替代task passes或owner验收。

入队前调用原正式CLI plan；再用真实launcher.prepare重做metadata，并调用原transport_job逐字段比较argv/config/inputs/resources/env/OPS/overlay/pinned_files/stop_mode，才调用现有submit_argv；写入后核queue，pending与claimed分别记。完成阶段必须claimed且原terminal succeeded/exit0/no signal/no wallcap，并调用原successful/predecessor；最终原consume核四门。全部检查在固定consumer cwd/PYTHONPATH子进程，driver独立部署身份/自身字节起止核对。

新证据目录在原source/OPS/data/output等全部输入树之外，exclusive创建，不覆盖。各CLI stdout/stderr逐次独占写入；summary只表达机械终态。原科学run代码与stdout不改。正式driver run有if-main与同形子进程dry/非法命令测试。

fresh RED：缺新增module的5项失败；GREEN：9个driver测试+原P driver17项，共26passed。后补正式launcher合同闭包后再次26passed（1.10s）。测试不读真实数据、不训练，不证明真实PF或四门通过。控制流fixture覆盖完整终态各委托一次、缺初始PF绝不提交、证据路径写入前拒绝、deadline/terminal/source/dry身份反例。独立C及新隔离变异仍待，部署/启动NOT_RUN。原件在私有review cache的native-a1-chain-{red-v1,green-v2..v5}.log；token UNKNOWN。

## 独立C首次BLOCKED后的更正

原候选9d7be8a保留。复核发现accept consume误传request、最终driver身份未闭合、queue消失但run残留未立即ALERT；三项fresh RED明确复现（3failed/9passed）。最小修复为固定output/accept/stage-receipt.json及原字节SHA、各阶段检查后与summary前再核driver/request、残留run丢queue立即ALERT；不改科学调用实现和门槛。新增三项及原P driver回归待下次原始日志核定，独立C重验后才能部署。

## 总截止闭包更正

独立C确认原三项已关闭，但最后accept跨截止仍成功；新增两项fresh RED复现。现所有长CLI/submit子进程均设剩余总预算timeout，长调用后与summary前再核截止；末次sleep仅为截止唤醒，不额外缩短轮询准入。原模型job不受driver timeout信号影响。旧候选及拒绝记录保留，仍须独立C末端重验。
