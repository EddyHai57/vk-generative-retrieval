密级: 公开

# P4H SASRec-style parallel protocol v2

本协议是 P4H-01 的冻结接口，不是运行许可或实验结果。权威设计输入为主仓 `.agents/tasks/model-audit/p-sasrec-parallel-plan-v5.md`（SHA `ea3f7566750c4fad13dfba80fd03a5f8fdf7e81c4c68aa8912b09b3e0e732121`）及其 APPROVED review（SHA `3de5ad402b6dd9e033d068567d046071ba4998e888914f5ac49c155c035287a1`）。机器配置见 `configs/p4h-sasrec-v2.json`；校验逻辑见 `src/vkgr/retrieval_sasrec_plan.py`。

## 当前边界

- 当前只完成 local P4H-01 plan/schema freeze；不实现 SASRec 模型或 operation controller。
- P4G 33/33 preregistered runs 与传统 selection 保持冻结，common test 为 `CLOSED`。
- `D-P4H-REMOTE-01`、`D-P4H-REMOTE-02`、`D-P4H-G5-01` 均为 `BLOCKED`。
- 不 SSH、pilot、formal train/resume、validation infer、common test、commit、push 或 publish。
- G4 freeze 不等待 P4H；P4H route 缺失、迟到、失败、not eligible 或未批准时固定 `ORIGINAL_P4D_TT`。

## 固定科学 registry

P4H 只比较 strict-past `like|observed` history 与 `mean|ID-only SASRec-style` encoder。两个 mean control 只读复用；新 formal registry 恰好12 slots：每个 history arm 用 seed42 搜索 `(LR,T)=(0.001,0.07),(0.001,0.15),(0.0003,0.07),(0.0003,0.15)`，再只对 matched `(0.001,0.15)` 跑 seeds2024/2025。失败 slot 为 `failed+spent`，不补位、不换 seed、不增加第13个slot。

Validation selection 固定 all-eligible Recall@50→NDCG@50；epoch tie更早，config tie按枚举，matched route tie取like。Eligibility只用2024和2025分别相对同history mean control的 Recall@50 delta严格大于0；seed42标 `selection_conditioned=true`。Route固定真实item ID top80、item-ID升序tie、seen masking=false；P4H只替换best-core TT，不新增第六路，G5仍五路各80、总400。

## Pilot role-aware schema gate

Pilot manifest先验证stage、完整role集合、每个role的exact schema与lineage；这些通过之前禁止resolve/stat/open/hash role path或创建output/control root。

- `prepared-train-rows`：schema `p4h-prepared-train-rows-v1`，fields精确 `{user_id,history,history_positions,target,target_position}`；target event必须来自冻结train partition，history严格早于target。
- `watch-train`：schema `p4h-watch-train-v1`，fields精确 `{example_id,history,history_positions}`；`example_id`与canonical prepared row双向一一对应。
- `train-prefix-target-free-sidecar`：schema精确 `p4h-train-prefix-target-free-v1`，fields精确 `{query_id,history}`。只有这个exact schema ID自身豁免`target` token扫描；record/nested/value semantics仍禁止target/oov/metric/quality supervision。

任何validation/holdout/test lineage、未知/重复/额外role、schema drift或sidecar forbidden semantics都必须 `STOP_BEFORE_PATH`。Pilot machine fields只能从accepted/opened role与event ledger推导 `validation_target_read=false`、`quality_metrics_computed=false`；合法prepared train target明确存在，不能声称“没有任何target字段”。

## Mutating 与 verifier 双 operation

`pilot`、`formal-train|formal-resume`、`infer-route` 都只能先安全 `CLOSED` 为 `completed+unverified`，再由全新 `pilot-verify|formal-verify|infer-route-verify` operation产生 `completed+verified`。两者必须使用不同manifest、output/control root、controller identity、admission、G receipt、worker PGID、caps与closure。Verifier只读predecessor，不能补产物、写旧root或跑optimizer。

Sealed predecessor schema固定为 `p4h-sealed-predecessor-operation-v1`。三个 verifier 都必须满足：

```text
manifest roles = common roles + one sealed-wrapper + input_hashes roles + artifact_hashes roles
roles(sealed.input_hashes) == declared input_hashes roles
roles(sealed.artifact_hashes) == declared artifact_hashes roles
```

四组两两不交叉；sealed wrapper不进入input/artifact集合，也不参与自身 `complete_artifact_set_sha256`。漏项、额外项、重复或跨source归类必须在nested path前失败。Machine counts固定：pilot-verify 21 roles、formal-verify 27、infer-route-verify 25。

## Commands 与 runtime boundary

唯一11 commands为：`prepare-controller`、`pilot`、`pilot-verify`、`formal-train`、`formal-resume`、`formal-verify`、`select`、`completion-verify`、`summarize`、`infer-route`、`infer-route-verify`。每个command只接受 `--operation-manifest` 与 `--operation-manifest-sha256` 两个业务参数。

`select|completion-verify|summarize` 是local bounded聚合，不创建remote controller/admission/G feed/worker PGID。其余command属于remote family，当前全部BLOCKED。P4H-03A未来只提供无CLI typed science API；P4H-03B未来独占parser、manifest、controller、feed、process与closure，依赖方向只能03B→03A。

## G feed schema与claim boundary

Feed descriptor冻结owner UID、mode0700、root identity、`O_CREAT|O_EXCL` publisher lock birth、publisher PID/start ticks/executable SHA、fresh no-other-writer inventory、G job/start identity与append-only receipt chain。Receipt schema为 `p4h-g-feed-receipt-v2`，状态仅 `OPEN|CLOSED|OWNER_LOST`。

保证范围只覆盖 accidental duplicate publisher、stale publisher、wrong G job、错误自报identity、rollback/fork和可观察unknown writer；SHA只证明内容绑定，不提供同UID恶意publisher的密码学身份。任一 `/proc`、ACL、lock holder、executable或writable-FD inventory无法证明时，remote保持BLOCKED或运行中cooperative stop。

## Protection、test与public boundary

Accepted P4D/P4E/P4F/P4G输入hash只从配置读取；任何drift都STOP，不重签旧artifact。Public只允许原创source/test/config/protocol、role/hash manifest与小摘要；raw/prepared/history/query、checkpoint/optimizer/RNG、vectors/index、逐query candidates、真实server path、PID/PGID/start ticks与完整resource trace只留server/ignored。

Common test保持关闭。Remote validation infer授权不能外推为test授权；未来test必须绑定single common-test freeze且结束后禁止回调history、checkpoint、route、weights、quota或fallback。Observed-history结果只称offline association，不称CTR、线上A/B、因果lift或完整exposure identity。
