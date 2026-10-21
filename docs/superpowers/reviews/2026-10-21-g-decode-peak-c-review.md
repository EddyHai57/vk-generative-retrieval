# G synthetic decode peak guard：C档独立复核

范围：analysis/g_decode_timing.py、tests/test_g_decode_timing.py及原throughput接口；不读全历史，不运行服务器。

独立review请求模型gpt-6-luna；reviewer未独立验证runtime模型metadata。结论：无blocking finding。require_gpu_peak拒绝非正/非有限峰值、非正整数采样数及bool；preflight在ready前调用。正式GPU测量与workers1/2/4/8仍待原调度器运行，不以本地测试代替。

执行者验证：fresh RED为缺少helper；补丁后6 tests通过；4个独立新缓存副本变异全部拒绝（python -B，PYTHONDONTWRITEBYTECODE=1）。真实A1小型codec/worker测试独立子进程通过；预加载全局codec替身后同6项仍通过。测试隔离属A档，不改变生产判定。

全库测试：196项，2 failures、12 errors、14 skipped。G项原失败已定位为test_throughput_decode收集时全局替换vkgr.sid_depth，独立子进程复验通过；其余失败未判为本改动引入，也未宣称排除。原始日志及全部失败名称仅本地cache/g-decode-timing-20261021/full-suite.log与full-suite-failure-names.json。不能声称全库GREEN。包含缺少ANCHOR_PROFILE_SCHEDULER_CODE、fixture/runtime配置、socket权限、worker类型/内存探针等问题。

本报告只核工程实现，不含召回或转正结论。生产入口未部署、未入队。
