# G合成解码计时 Implementation Plan

> **For agentic workers:** 本任务在当前对话inline执行；独立review仅看本diff、指定裁决摘录与改动文件，不读全历史。

**Goal:** 按863f53a §15–17完成随机A1、全规模合成平衡SID/trie、512条H50 query与workers1/2/4/8的成本测量。

**Architecture:** 新工程入口复用原sid/codec/A1/beam和throughput_decode；原源码从固定Git逐字节恢复缺失模块，现有文件不改。只读原输入/源码验证，合成输出新目录，独立OPS job detached，无科学训练或test读取。

**Tech Stack:** 原锁Python/NumPy/PyTorch，原OPS launcher与scheduler；本地synthetic测试不替代同CLI服务器正式preflight。

**Spec:** decision_log 863f53a §15–17及docs/task_list.json的g-decode-timing-1b。

## Global Constraints

N=12012895，K229/depth3，capacity=n//k前n%k加1，512 query、H50、batch1、beam100、4步，workers1/2/4/8；只报计时资源。首次context-inclusive峰值实测后才可并行；资源不足排队。不改原job/source/预算/spent/deadline/名额，不删除旧文件。

## Review Focus

恢复源码字节与配置身份；capacity及suffix；正式argv/dry/preflight；只新目录及producer/job/stdout绑定；计时与结果核对分离；显存含context且只看本进程树；并行init/RAM/VRAM预估与真实峰值；原scheduler不中断。

## Steps and validation

- [x] 新CLI/平衡SID/逐query结果测试fresh RED。
- [ ] 恢复固定原源码依赖闭包（已存在三模块必须逐字节相等）；生成source ledger。
- [ ] 最小generate/preflight/timing/job入口；GREEN与原小型SID/codec/beam对照、真实launcher临时布局完整链。
- [ ] 隔离变异、相关回归；独立Luna C审，仅闭合本板块。
- [ ] commit、公开卫生扫描/压平发布、固定SHA新部署与逐文件SHA核对。
- [ ] 同正式CLI合成全规模输入preflight；唯一新ID dry/入队并确认claimed/no rejected，不重投。
- [ ] 原始stdout/manifest绑定验收；experiment_log唯一成本事实，task_list通过须所有验收完成；更新STATE/handoff/map/schedule区间增量。
