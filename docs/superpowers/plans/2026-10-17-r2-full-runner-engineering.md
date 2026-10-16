密级：公开

# SFT + 自生成难负样本 DPO：完整工程 plan

状态：科学规则与成本 ACK 已齐，工程实现完成待一次独立 review；真实工程门禁通过前禁止真实训练。基于主 owner 更新后的 `974c6751f49a085680f4cbe75862bbf86b7d9f4e`，不覆盖既有 microbenchmark/只读 PF 改动。

## 前置与 stop gates

1. Kiro 已冻结负样本缺失策略：跳过、不补抽，固定20,000集合，报告skip条数与比例。
2. Kiro 已冻结 DPO min_epochs=1/max5/pat2；best只从epoch1–5选，不含SFT best epoch0。epoch0与原A1同seed差异只报告，不作门槛。
3. 主 owner 已 ACK 实际 train-only PF 与独立速度工程job，整批工程估算不超过4h；收据SHA与来源见新增 frozen-evidence。仍不把随机权重工程吞吐称作新 SFT best 实际吞吐。
4. Config显式保存以上冻结值与成本收据SHA；缺失/null/篡改拒绝，无未决字段fallback。科学规则来源是Kiro与主owner ACK，不是synthetic测试值。
5. 只有一次独立 code review；后续真实 PF、CUDA（actor+reference+optimizer 全shape）、per-PID context显存、dry-run 全过才由主 owner 入队。未授权本 agent 部署、push、真实训练或改 main 事实源。

## 实现边界与分步验收

1. 新增 R2 config contract 与三 seed 冻结配置；测试未冻结拒绝、SFT逐字段沿A1/T3、固定20k/seed0/beam10/b1、DPO beta=.1/CE1/LR3e-5/max5/pat2。
2. 新增独立 SFT runner：从原A1/T3源继承训练/selection/best+last/verify；新schema/source绑定，只读自己新run的CP。新 data-root helper 将已验收历史数据通道传入原loader，代码root独立。depth3直接用原4-position evaluator，不patch旧模块或全局函数。
3. 新增 mining / DPO engine：SFT best通过自身verify后，一次固定train索引采样，beam10 b1；仅target与完整past_likes排除；无合法负样本按显式冻结策略执行，绝不暗补样。产物保存在新R2 run内部并hash绑定，完整样本/CP不进Git。
4. DPO actor从本run SFT best初始化；reference是独立snapshot，eval/no-grad且不入optimizer。序列logprob按SID各位置求和；DPO(.1)+正样本原CE(mean)权重1；梯度裁剪/AdamW及microbatch沿原。最多5ep，冻结min_epochs，patience2，按原validation R@50与NDCG tie-break选best。
5. 新总runner与verify：阶段目录sft/mining/dpo，独立schemas并绑定同source/config/data/seed、SFT best、mining hash、reference digest、各epoch evaluation；selection逐epoch重放及最终best重解码；拒绝跨seed/旧schema/旧CP/篡改reference或mining输入。
6. Ops沿已修R1 ready/release协议：seal完整3seed plan及绝对48h deadline → detached PPID1/session+lock ready → release → driver唯一发布；partial failure保留已pub并接唯一verify，未pub不retry；poll900/noLLM。新CUDA proof含actor/reference/optimizer+真实shape、own PID nvidia-smi context与allocator峰，gate重算ceil256(1.15×measured)。

## TDD与验证

每个新增行为先运行fresh RED，再最小GREEN，最后相关回归；使用原runtime venv与synthetic fixtures，不读真实数据或旧CP。覆盖DPO手算、梯度符号、reference独立/冻结/不被更新、固定样本索引、完整past_likes而非observed history、无合法负样本跳过不补样策略、缺freeze拒绝、真实两阶段synthetic执行/verify、source/data root分离、mining/reference/config篡改，以及ready/release的RPC kill/partial pub/spawn failure。原A1入口和既有microbench源码保持不变。

当前 context 计数不可用：UNKNOWN。主 owner 是唯一服务器执行者；本 agent 仅本地新增源码/tests/文档，完成后commit交主 owner review。

## 本地阶段完成证据

新增contract、内部独立SFT、mining/DPO engine、总runner/verify与ops均已实现，60 tests GREEN。原974c675的microbenchmark、primitives与其tests未修改，原A1入口/config/gates未修改。下一步是主owner安排一次独立review，再部署新commit并跑真实PF/CUDA/reference显存/dry-run；本agent未server/push/mainfacts。
