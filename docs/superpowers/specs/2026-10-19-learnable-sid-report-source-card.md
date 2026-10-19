# S2只读补充诊断源码固定增量卡

FROZEN工程口径：依据S2 FINAL、Kiro两卡主判及Eddy两份只读补充报告授权。只报原S0 A/B/C/D/Dprime；不改变S2已收口主判、anchor、门禁、对照或seed。RQ-VAE逐层全局码本，第三位容量描述以实际两位前缀平均物品数为准，不称第三层空置。

固定六源：g-learn-a1-s42/s2024/s2025及g-coll-a1-c1同seed；manifest/stdout物理pins在runner常量及私有request固定。mapping来自各自已验收SID manifest中的mapping.npz pin，no model/checkpoint/embedding；只输出聚合不输出mapping或逐query预测。原train-only热度阈值与分母固定；原best归档完整query/history/catalog/计算hash及item指标复核保留。

S0 D和Dprime原函数逐字复用；训练源来自原S0 plan登记的prepare-manifest与严格顺序train周00..24，原物理行顺序、重复保留、不跨用户接合。D使用原mapping插入顺序+独立有放回randrange，Dprime升序ID+每pair不重复sample，seed0；两随机方法不同、结果不可直接比较，不为统一而改变已冻结函数。预期原有效相邻pair数由既有S0固定，不据新值修改。

报告只读CPU1、VRAM0、RAM<=16384MiB、wall<=2h。唯一PF sid-learn-report-preflight-v1、正式sid-learn-report-v1；metadata-only formal PF不解析mapping/prediction/parquet payload，只读元数据及stat原25文件。formal report成功后原stdout须绑定自身run_id/run_manifest_sha256/request_sha256/source_commit。PF/request/source/argv/claim/terminal完整绑定后才能发布正式报告；任何existing/intent/rejected不重投。

## 固定执行增量

actual执行账本必须精确等于每个原训练source.files加以下其未包含的固定路径/SHA。既有成员字节、Python依赖及lock必须严格相等；旧control缺lock时从其原训练Git commit取blob核对。public source commit另记，不强求等于训练commit。禁止开放式豁免。

|路径|SHA256|
|---|---|
|uv.lock|8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063|
|analysis/sid_learn_report_read.py|f044a6baeb2c894412053ad8b2c645a21b37b6c44ee03d0f4d13f439af7fbb2a|
|scripts/sid_learn_report_queue.py|a76033242da4a16ec56bf650ec07c76c8a9bb35e3f7fe528377e961ac61a584f|
|analysis/sid_diag_prime.py|e2693679630ac4bcae09b474d923a5cb507b692bd12397efa77988204c57813b|
|analysis/sid_diag.py|e7db6e37cb00f7c165cfd373da53a4538cfa4e54b84113fbbae3bde6d2523161|
|analysis/sid_diag_metrics.py|ad966940d682aa0f4e576eeacfece8cef92b2f9597fec4a41522b7e535e55dc4|
|analysis/phase2_validation_read.py|548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8|
