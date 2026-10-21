# A40 原精确评分成本与资源前置（工程执行卡）

依据：Eddy新实例接手明确授权、decision_log942a5b0阶段1c及11249b6；本卡仅固定已授权工程工作的调用细节，不增加科学实验/训练名额，不修改科学合同、seed集合或判据。

## 固定输入与原路径

- 合成catalog **12,000,000 × 64 float32**，query **26,827 × 64 float32**，连续uint32 item ID；独立合成RNG seed42，主进程按8192行固定块生成。原始数据、holdout、checkpoint都不作输入。
- 原P source `0ca61bc35ad68bd79b86103b54e470445cbab07f` 的 `src/vkgr/baselines.py` 原字节副本SHA `ec92bb0b9858e9da7e192f8cc32da96d5a274ad195413220695b093361b96a0b`。成本入口编译该原文件并调用原 `blockwise_topk`，不改函数/NumPy库。
- query block32、item block4096、float64 `einsum(optimize=False)`，item ID tie规则原样。**top-k=50**，来源同一source的 `retrieval_experiment.evaluate_vectors(ks=(10,20,50))`，不是融合导出所需的top100。
- 独立进程、原Python3.11.16/原lock/NumPy1.26.4等精确版本，CPU1；NumPy/BLAS相关线程环境固定1。无GPU计算，VRAM声明0。

## 前置与作业

1. 已发布工具链固定源码63文件保留，新增本成本入口、原已审OPS/RAM adapter与runtime检查、指定A40 YAML别名；新工作分支先push，再新目录逐文件SHA验收。原数据作业/原scheduler不重启、不重投、不重置截止。
2. 资源probe只读实际cgroup/GPU/磁盘，写独占新manifest；选用 `configs/throughput/a40-m153999.yaml`，与已审a40-20261022.yaml逐字相同。SHA `69f8039793ace8dfc67063ab94013a0ae958ad1da75e2f6e369dbc61582c8cc5`。建议不自动覆盖配置，RAM规划513GB、更紧256GB工程上限、400GB盘上限。
3. `generate` 经metadata/dry和唯一queue ID，在新目录生成三份合成npy与inputs manifest；主进程随机数不进入worker。
4. 同正式 `python -B -m analysis.a40_cost preflight` 先核原生成job的successful terminal、固定source/argv、仅lock/YAML输入闭包、原stdout与inputs.json全字段相等，再读取实际大输入size/hash/shape/dtype；不能只信data_kind自报。35×4103合成小样本覆盖32/4096尾块、零向量ties，原reference与workers1/2/8的预测字节、原score block字节摘要全部相同。预检声明CPU9（8workers+父进程）/RAM16384MiB，完整大输入尚未做worker扩展计时，不将小样本说成真实anchor对拍。
5. `run` 再核生成job原stdout链、inputs、script/YAML、baseline、合成标记、scope与workers及绑定的原preflight原始run/terminal/stdout；不接受孤立布尔receipt。原单线程完整评分一次，有限48h cap，CPU1/RAM16384MiB/VRAM0，不重试失败。输出只有成本/资源/身份与预测整体SHA，不导出逐query预测。每阶段新run ID、无覆写、claimed/no rejected与原stdout绑定另验收。

## 报告与停止条件

评分函数wall与CPU单列；RSS峰值口径为整个本worker，包括加载/校验和库开销，不能冒充评分阶段独占峰值。输入先hash，会暖OS page cache；成本不含合成输入生成、用户编码、模型训练、真实query构建或完整端到端评价。没有质量指标，不据此作anchor/转正或跨机器推断。

输入/源码/lock/真实前置/score bytes任一不符停止，不改预期；运行失败保留全部产物，按原规则回报。driver/scheduler不因其它支线终态停本作业。资源只用本进程树与余量，保留20%余量及60%RSS/80%盘线。

BHKM此轮资源probe仅出累计保留的**情景算术**：N=1.2M/12M、K=48/229、depth3、max_iter10（已发布模块默认值情景）；按源码的整项fit上界及两份fit/replay累计相加，含run头预算。K未科学冻结，不选K、不把上界当实测、不把根单次run量当整项。容量不满足回Claude Code(Kiro角色)，本卡不批准删run、改算法或额外付费。

执行入口固定字节：`analysis/a40_cost.py` SHA256 `a18d2523df8ae5e002d161b416e0a229c994884891b9e65e593a63df93a0a15f`。实际发布commit另记发布/部署收据；脚本SHA同时进入输入manifest、前置及成本输出，旧输入或前置与脚本不符即拒绝。
