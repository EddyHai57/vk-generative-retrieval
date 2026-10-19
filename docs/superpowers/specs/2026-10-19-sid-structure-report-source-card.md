# 结构匹配只读报告固定源码卡

状态：FROZEN工程口径，依据原结构FINAL、用户顺序第2项与AGENTS §4A；沿Kiro最新科学口径增补描述指标，不改训练门禁、对照或seed。

仅复用原S0 A/B/C与immutable archive算法。源运行固定为结构三个seed与原共现SID同seed；accepted manifest/stdout物理pin在runner常量及私有request中逐项固定，不读取S2结果。报告只出机械数；前缀按最新裁决只作描述并报平均每码物品数、query加权q和top1 C；不生成跨分叉差值，不判机制或自行改anchor。旧“不生成折算指标”已由最新Kiro裁决替代。

## 固定新增执行成员

报告当前执行账本必须严格等于「每个原训练manifest.binding.source.files ∪ 对应下表固定新增成员」。既有成员字节、缺失、多出未列路径均失败；current source commit单独记录并固定部署，不要求其等于旧训练commit。Python与依赖严格匹配每个原runtime.environment；lock若原ledger未列，取该run已登记原Git commit的uv.lock blob核对，不以当前文件值冒充历史。

|执行成员|SHA256|相对结构训练ledger新增|相对原共现A1 ledger新增|
|---|---|---|---|
|uv.lock|8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063|already pinned|yes|
|analysis/sid_structure_report_read.py|e2cad4ae304ccdd58d3e3645e2c9f36f40d9394fc955a5b0d1448f7b029104e6|yes|yes|
|scripts/sid_structure_report_queue.py|5c4a2170631469decc13c94ff41af6047fc3ad07086a7d67be418bd489f039c5|yes|yes|
|analysis/sid_diag.py|e7db6e37cb00f7c165cfd373da53a4538cfa4e54b84113fbbae3bde6d2523161|yes|yes|
|analysis/sid_diag_metrics.py|ad966940d682aa0f4e576eeacfece8cef92b2f9597fec4a41522b7e535e55dc4|yes|yes|
|analysis/phase2_validation_read.py|548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8|yes|yes|

原结构source b5a4d1c99ba2d8361a2d90c23bac27af5b71769d；原共现A1 source a5da4d6d0746ef41ca1ed93028971360dcb89547。两个原Git uv.lock SHA256均为 8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063。

## 入口与作业合同

固定CPU非训练preflight及report两个唯一ID；正式文件入口analysis/sid_structure_report_read.py，私有request实际SHA写入job/config/argv。report只有自身preflight原stdout preflight_ok与source/request/actual run/claim/terminal精确绑定通过后才发布。任何run/queue/intent存在均不重投；writer出错保留ALERT。

真实preflight读取同一路径的原manifest/原始verify/H2查询与train-slices，逐输入SHA及NPZ/shard路径/祖先检查；不解析预测/映射payload。正式report读取原归档后，复用原行覆盖/排序/完整query/history/catalog/计算hash和重算item指标自校验，只输出聚合。原始status还必须绑定自身run_id/run_manifest_sha256、source_commit及request_sha256。

所有结果、完整SHA/UTC与限制只在experiment_log；公开/部署source commit以对应固定部署回执为准，服务器只跑已push且逐文件校验的Git源码。


## 最新口径增量（正式报告出数前）

FINAL卡2026-10-19附记已先于新诊断读取冻结。runner逐结构/seed计算in-catalog top1/top50 A及query加权q，top1 C=(A-q)/(1-q)，q=1记null；报告跨seed算术均值，无top50 C，无跨结构差值。前缀不进入anchor机械条件。

原fit manifest中已pin的audit.json逐输入读取，实际mapping的逐层used/utilization/prefix_nodes、语义叶数/max/碰撞物品/组数/对数及full SID零碰撞必须与其一致；各热度档分母和OOV零命中另机械检查。三seed头中段Δ之和/总Δ按最新冻结规则检查；合理性检查弱、非机制证据。只有原三条件与合理性/完整性同时为真才输出anchor_rule_components_satisfied=true，anchor_change_authority=Kiro_only。

C档新统计实现，原review仅覆盖旧版本，须新的独立review；CPU只读PF/report唯一ID未使用，正式出数仍在固定SHA部署及真实PF后。
