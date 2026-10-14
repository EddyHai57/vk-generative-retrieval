#!/usr/bin/env python3
"""FEAT-03 mechanical freeze tool (same shape as the FEAT-02 one).
--write --worktree W --card CARD   needs a clean tree; reads HEAD, config sha256 (git show HEAD:<path>), constants and parameter
                                   counts from the code; appends (first time) or regenerates (later) card section 10 between the
                                   feat03-freeze markers. Never touches a line outside the markers.
--check --worktree W --card CARD   exit 0 only if the card equals the code at HEAD (see check()).
Shas, the commit and the time are never typed by hand: they come from git and from the imported code.
The pre-registered thresholds live in THRESHOLDS below; --check fails if the card block and this table differ.
The card is public: nothing written here may contain local paths, IPs or company names.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
BEGIN = '<!-- feat03-freeze:begin -->'
END = '<!-- feat03-freeze:end -->'
BASE = '94eb0bcbf66b73397a7a5d1332ef9c55a8fa39f2'          # FEAT-02 freeze commit (feat02/p HEAD); feat03/p starts here
CONFIG_DIR = 'configs/feat03'
CONSTANTS_FILE = 'configs/feat03/q3-bucket-constants.json'
VARIANTS = (  # key, feature, config file
    ('q1a', 'f3_pos_only', 'p-feat03-q1a-pos-only-s42.json'),
    ('q1b', 'f3_gap_only', 'p-feat03-q1b-gap-only-s42.json'),
    ('q2', 'f4_recency_pool', 'p-feat03-q2-recency-pool-s42.json'),
    ('q3', 'f3_user_activity', 'p-feat03-q3-user-activity-s42.json'),
)
# Every pre-registered threshold, written before any FEAT-03 result exists (design section 7.4). Editing this table after a result exists
# is a post-hoc change and needs its own decision_log entry.
THRESHOLDS = {
    'effective_delta_r50': 0.002, 'effective_delta_hits': 6,
    'q1_carry_mean_hits': 16, 'q1_carry_min_seeds_with_6': 2, 'q1_harm_mean_hits': -6, 'q1_null_band': [-6, 6],
    'f4_mean_delta_hits_M': 32.67, 'additivity_halfwidth': 8, 'additivity_low': 24.67, 'additivity_high': 40.67,
    'p_pool_max': 0.05, 'k_plus_min': 2,
    'q2_harm_mean_hits': -6, 'q2_shape_ln1p5': 0.405, 'q2_shape_other_side': -0.405, 'q2_logit_move_flag': 0.1,
    'q3_harm_mean_hits': -6, 'q3_rho_low': 0.4, 'q3_rho_high': 0.8, 'q3_d3_low': 13.07, 'q3_d3_high': 26.13, 'q3_e_hits': 6,
}
# Printed forms that must appear in the card text (check() looks for each one).
THRESHOLD_TEXT = ('16', '≥ 6', '−6', '24.67', '40.67', '32.67', '13.07', '26.13', '0.4', '0.8', '0.405', '0.1', '0.05', '0.002', 'k₊ ≥ 2', 'M ± 8')
def fail(message):
    print('FREEZE-ERROR: ' + message, file=sys.stderr)
    raise SystemExit(1)
def git(worktree, *args):
    done = subprocess.run(['git', '-C', str(worktree), *args], capture_output=True, text=True)
    if done.returncode != 0:
        fail('git %s failed: %s' % (' '.join(args), done.stderr.strip()))
    return done.stdout
def clean_tree(worktree):
    if git(worktree, 'status', '--porcelain').strip():
        fail('worktree is not clean (commit or stash first)')
def digest_list(values):
    return hashlib.sha256(json.dumps(list(values), separators=(',', ':')).encode()).hexdigest()
def git_blob(worktree, rel):
    raw = subprocess.run(['git', '-C', str(worktree), 'show', 'HEAD:' + rel], capture_output=True)
    if raw.returncode != 0:
        fail('file not committed at HEAD: ' + rel)
    return raw.stdout
# ------------------------------------------------------------------ facts from git and from the code
def code_facts(worktree):
    sys.path.insert(0, str(Path(worktree) / 'src'))
    import torch
    from vkgr import retrieval_feat02 as f2
    from vkgr import retrieval_feat03 as m
    constants = {'POSITION_EDGES': list(f2.POSITION_EDGES), 'WEEKGAP_EDGES': list(f2.WEEKGAP_EDGES), 'LEN_EDGES': list(m.LEN_EDGES), 'ACT_EDGES': list(m.ACT_EDGES),
                 'RECENCY_PARAMS': dict(m.RECENCY_PARAMS), 'FEATURES3': list(m.FEATURES3), 'feat02_bucket_spec_sha256': f2.bucket_spec_sha256(),
                 'bucket_spec3_sha256': m.bucket_spec3_sha256()}
    constants['table_sha256'] = {name: digest_list(constants[name]) for name in ('POSITION_EDGES', 'WEEKGAP_EDGES', 'LEN_EDGES', 'ACT_EDGES')}
    params = {}
    content = torch.zeros(9, f2.EMBEDDING_DIM)
    for key, feature, _ in VARIANTS + (('f4_control', 'f4_control', None),):
        model = m.make_feat03_model(feature, content, [1, 2, 3, 4], 50, 42)
        report, formula = m.parameter_report3(model), m.expected_param_counts3(feature)
        if report != formula:
            fail('parameter_report3 differs from expected_param_counts3 for ' + feature)
        params[key] = {'params_total': report['total'], 'params_trainable': report['trainable_effective'],
                       'tables': {name: [v['total'], v['trainable']] for name, v in sorted(report['by_table'].items())}}
    return constants, params
def config_facts(worktree):
    configs = {}
    for key, feature, name in VARIANTS:
        rel = '%s/%s' % (CONFIG_DIR, name)
        raw = git_blob(worktree, rel)
        value = json.loads(raw)
        if value.get('feature') != feature or value.get('seed') != 42:
            fail('config feature/seed mismatch: ' + rel)
        configs[key] = {'path': rel, 'feature': feature, 'sha256': hashlib.sha256(raw).hexdigest()}
    return configs
def audit_facts(worktree):
    raw = git_blob(worktree, CONSTANTS_FILE)
    value = json.loads(raw)
    sys.path.insert(0, str(Path(worktree) / 'src'))
    from vkgr import retrieval_feat03 as m
    if value['len']['final_edges'] != list(m.LEN_EDGES) or value['act']['final_edges'] != list(m.ACT_EDGES):
        fail('audit constants file and the code disagree')
    return {'path': CONSTANTS_FILE, 'sha256': hashlib.sha256(raw).hexdigest(), 'table_sha256': value['table_sha256'], 'examples': value['examples'],
            'merge_count': value['merge_count'], 'len_shares': value['len']['final_shares'], 'act_shares': value['act']['final_shares'],
            'n_equals_50_share': value['n_equals_50']['share'], 'history_manifest_sha256': value['inputs']['history_manifest_sha256']}
def gather(worktree):
    constants, params = code_facts(worktree)
    return {'schema': 'feat03-freeze-v1', 'status': 'FROZEN', 'commit': git(worktree, 'rev-parse', 'HEAD').strip(), 'base_commit': BASE,
            'configs': config_facts(worktree), 'constants': constants, 'params': params, 'audit': audit_facts(worktree), 'thresholds': THRESHOLDS}
# ------------------------------------------------------------------ card text
def pct(share):
    return '%.2f%%' % (share * 100)
def bucket_rows(edges, shares, last, names):
    bounds = [0] + list(edges) + [last + 1]
    rows = []
    for i, share in enumerate(shares):
        lo, hi = max(bounds[i], 1), bounds[i + 1] - 1
        rows.append('| %d | %s | %s |' % (i + 1, str(lo) if lo == hi else '%d–%d' % (lo, hi), pct(share)))
    return rows
def render(facts, refreeze_lines):
    c, p, cfg, au = facts['constants'], facts['params'], facts['configs'], facts['audit']
    commit, when = facts['commit'], facts['freeze_time_utc']
    def variant_lines(key, label):
        return ['- 状态：**FROZEN**。冻结时间（UTC）：%s。代码 commit：`%s`。' % (when, commit),
                '- config：`%s`，sha256 `%s`。' % (cfg[key]['path'], cfg[key]['sha256']),
                '- 参数（含 F3 的 T、R 两张表）：合计 %d，可训练 %d；各表（总数 / 可训练）：%s。' % (
                    p[key]['params_total'], p[key]['params_trainable'], '，'.join('%s %d / %d' % (n, v[0], v[1]) for n, v in p[key]['tables'].items()))]
    out = [BEGIN, '## 10. FEAT-03（P 线第三批）冻结记录', '',
           '状态：**FROZEN**。冻结时间（UTC）：%s。代码 commit（`feat03/p` 的 HEAD）：`%s`。基线：`%s`（`feat02/p` 的冻结提交）。' % (when, commit, BASE),
           '本节由冻结脚本从 git 与代码机械生成（sha256、commit、常量、参数数都不是手写）；本节之前的既有内容一字未改，只在末尾追加（卡内编号 10 排在 §11 之后，是追加顺序造成的，不代表时间顺序）。冻结之后，任何新的代码 commit 都要重新冻结。', '',
           '### 10.0 总述', '',
           '- 范围：4 个 seed-42 变体，全部新增文件、不需要新 sidecar；`f4_control`（F4 的逐位等价对照）只在测试里存在，不提交训练。',
           '  - Q1a `f3_pos_only`：在 F3 上只加位置桶（诊断，不计名额）。',
           '  - Q1b `f3_gap_only`：在 F3 上只加周间隔桶（诊断，不计名额）。',
           '  - Q2 `f4_recency_pool`：在 F4 上加按新近性分桶的可学习池化权重（计 1 个名额）。',
           '  - Q3 `f3_user_activity`：在 F3 上加池化之后的用户级活动 embedding，字段为历史长度桶与活跃周数桶（计 1 个名额）。',
           '- 共同条件：P 线、`id_mean`、N1（in-batch + logQ α=1，无额外 uniform）、`init=small`、`HIST_V2_OBS` L50、100 / 5 / 3 epoch、attempt 1。对照：Q1a / Q1b / Q3 对 F3，Q2 对 F4；每个变体只改一个变量。',
           '- 名额：FEAT-01 3 个、FEAT-02 3 个已用完（卡内剩余 0）；Q2、Q3 的 2 个计数名额依 decision_log「2026-10-14 20:5x」第 1 条批准，P 线特征计数累计 3+3+2=8，即 8/8。F7 余下的 age / gender / geo 另需批准。',
           '- 补 seed：Q1a、Q1b、Q2、Q3 在 seed 42 之后一律无条件补跑 seed 2024 / 2025（共 8 个 run，不计名额），不看 seed-42 结果、不由任何人事后裁量；依据 decision_log「2026-10-14 22:5x：FEAT-03 补 seed 规则」第 1 条（批准人是 Kiro 依 Eddy「按你的建议来」自审批）。总训练 run 上限 12。',
           '- 部署前闸门：Q2、Q3 各做一次 server 上的 `synthetic-train --device cuda` + `synthetic-verify --device cuda` 预演（不占名额、不读真实数据）；通过前不提交 Q2、Q3 的真实 run。预演结果不属于冻结项。',
           '- 时点规则沿用 §2：特征只来自 target 之前历史事件自己那一行的字段，不读 target 行的任何字段，不读 test（week 26）。分桶边界只用 train 定义并写死在代码里。',
           '- Q3 的偏离：「最近一次事件距查询时刻的周数」没有放进 Q3（训练里它只能取自 target 行的 week，且与验证的含义不同）；推迟到请求时刻 week 的口径审计通过后另立卡。因此 Q3 的结论只覆盖「长度桶 + 活跃周数桶」合在一起。',
           '- 常量代码位置：`src/vkgr/retrieval_feat03.py` 的 `LEN_EDGES`、`ACT_EDGES`、`RECENCY_PARAMS`、`FEATURES3`，以及 `src/vkgr/retrieval_feat02.py` 的 `POSITION_EDGES`、`WEEKGAP_EDGES`（Q1a / Q1b / Q2 复用，零新常量）；`bucket_spec3_sha256` = `%s`（写进每个 run 的 binding，改任一常量旧 manifest 的 verify 就失败）。' % c['bucket_spec3_sha256'], '',
           '### 10.1 Q1a 位置桶（对照 F3，不计名额）', '', '- 定义：e_k\' = e_k + T + R + P[pb_k]，位置桶沿用 F4 的 `POSITION_EDGES`，不加周间隔项。', *variant_lines('q1a', 'Q1a'), '',
           '### 10.2 Q1b 周间隔桶（对照 F3，不计名额）', '', '- 定义：e_k\' = e_k + T + R + G[gb_k]，周间隔桶沿用 F4 的 `WEEKGAP_EDGES`，不加位置项。', *variant_lines('q1b', 'Q1b'), '',
           '### 10.3 Q2 新近性加权池化（对照 F4，计 1 个名额）', '',
           '- 定义：在 F4 的逐事件向量上，用两组标量 logit（位置桶 7 个、周间隔桶 6 个，初值 0）给池化加权，权重 = exp(logit) 在有效位置上归一；PAD 位置权重恰为 0。logit 初值 0 时与 F4 的均值池化等价。',
           '- 位置边界 %s（桶数 %d），表 sha256 `%s`；周间隔边界 %s（桶数 %d），表 sha256 `%s`。' % (tuple(c['POSITION_EDGES']), len(c['POSITION_EDGES']) + 1, c['table_sha256']['POSITION_EDGES'], tuple(c['WEEKGAP_EDGES']), len(c['WEEKGAP_EDGES']) + 1, c['table_sha256']['WEEKGAP_EDGES']),
           *variant_lines('q2', 'Q2'), '',
           '### 10.4 Q3 用户级活动汇总（对照 F3，再比 F4，计 1 个名额）', '',
           '- 定义：e_k\' = e_k + T + R 之后做均值池化，再加用户级项 Len[length_bucket(n)] + Act[activity_bucket(a)]；n = 窗口内有效历史条数，a = 窗口内不同周数。没有位置项与周间隔项。',
           '- 长度边界 %s（桶数 %d），表 sha256 `%s`；活跃周数边界 %s（桶数 %d），表 sha256 `%s`。边界由 train-only 审计（`%s`，sha256 `%s`，%d 个训练样本，合并步骤 %d 次）确认。' % (
               tuple(c['LEN_EDGES']), len(c['LEN_EDGES']) + 1, c['table_sha256']['LEN_EDGES'], tuple(c['ACT_EDGES']), len(c['ACT_EDGES']) + 1, c['table_sha256']['ACT_EDGES'],
               au['path'], au['sha256'], au['examples'], au['merge_count']),
           '- 长度桶 train 占比（n = 50 的 train 占比 %s）：' % pct(au['n_equals_50_share']), '', '| 桶 | n | train 占比 |', '|---|---|---:|', *bucket_rows(c['LEN_EDGES'], au['len_shares'], 50, None), '',
           '- 活跃周数桶 train 占比：', '', '| 桶 | 不同周数 | train 占比 |', '|---|---|---:|', *bucket_rows(c['ACT_EDGES'], au['act_shares'], 25, None), '',
           '- 审计绑定的 history manifest sha256：`%s`。' % au['history_manifest_sha256'],
           *variant_lines('q3', 'Q3'), '',
           '### 10.5 预注册解读规则（看到任何 FEAT-03 结果之前写死；以下阈值全部抄全，之后不改）', '',
           '阈值清单（与 decision_log 22:5x 的关系：该条只列前六个 16、6、0.4、0.8、0.405、0.1；其余 −6、M ± 8（24.67 与 40.67）、`p_pool` < 0.05、k₊ ≥ 2、13.07 与 26.13 在本节新增，但都早于任何 Q 系列结果，预注册有效）：', '',
           '- 有效：ΔR@50 ≥ 0.002，即 ΔH ≥ 6（H = round(R@50 × 2520)；5 个命中 = 0.001984 不过线）。',
           '- Q1 档位：C（携带）均值 ≥ 16 且至少 2 个 seed 的 ΔH ≥ 6；H（有害）均值 ≤ −6；N（无）−6 < 均值 < 6；P（部分）其余。',
           '- 可加性：M = 32.67（F4 相对 F3 的三 seed 均值增量），S = A + B，边界 M ± 8，即 24.67 与 40.67。',
           '- Q2 / Q3 共用必要条件：k₊ ≥ 2（ΔH > 0 的 seed 数）且 `p_pool` < 0.05（三 seed 逐 query 配对 discordant 计数相加后的 exact 双侧二项检验；三个 seed 评同一批 2520 个 query，计数不独立，`p_pool` 偏乐观，所以是必要条件，不是证据强度，不做多重比较校正）。',
           '- Q2 形状：S_pos 或 S_gap 至少一个 ≥ ln 1.5 ≈ 0.405 且两者都 > −0.405 为偏新近；对称为偏老；其余平 / 混合；至少 2 个 seed 同向才算该方向。轨迹旗标：best epoch 与其前 3 个 epoch 的 logit 最大绝对变化 > 0.1。',
           '- Q3：ρ = D3 ÷ 32.67；ρ = 0.4 ⇔ D3 = 13.07，ρ = 0.8 ⇔ D3 = 26.13；分界值归较大一侧（左闭右开）。',
           '- Q3 对 F4：E = mean H(Q3) − mean H(F4)，|E| ≥ 6 为差异阈值。',
           '- 噪声声明：epoch 间 R@50 波动约 ±0.01（约 25 个命中），大于门槛；seed-42 单 run 只给方向。凡写「携带 / 有效 / 增益量级」的结论，只用三 seed 均值加逐 seed 符号一致性；每个 seed 另报 exact McNemar p（描述性，不做选择依据）。', '',
           '#### Q1：Q1a 与 Q1b（对照 F3）', '',
           '阶段 A（seed 42）只记 a = ΔH(Q1a)、b = ΔH(Q1b)，不下结论。阶段 B 取三 seed 均值 A、B 并定档位。读法按 (档位 Q1a, 档位 Q1b)，覆盖全部 16 个组合（3+3+1+1+8）。任一半项为 H 时，在主结论后追加「该半项单独加入有害（均值 ≤ −6）」。', '',
           '| Q1a | Q1b | 主结论 | 允许的措辞 |', '|---|---|---|---|',
           '| C | N、H 或 P | 长度先验为主 | 「窗口内历史长度先验」 |',
           '| N、H 或 P | C | 活动跨度 / 周分布先验为主 | 「活动时间跨度（周间隔分布）先验」 |',
           '| C | C | 两半各自复现一半以上 | 「用户活跃度（长度 + 跨度）先验」 |',
           '| P | P | 两半各带来部分增益 | 「两半各带来部分增益，合计见可加性读法」 |',
           '| N、H 或 P（(P,P) 除外） | N、H 或 P（(P,P) 除外） | 单项归因失败（8 个组合） | 「单项归因失败：F4 的增益不能由任一半项单独复现；是否来自两项联合或其他原因，本批无法区分」。**不得写「共同贡献」** |', '',
           '可加性读法（互斥三档，追加在每一行之后）：', '',
           '| S | 追加读法 |', '|---|---|',
           '| S < 24.67 | 两半之和不到 F4 的四分之三：缺的部分可能来自交互，或 F4 整体才有的效应（单项实验证明不了联合效应存在，只是没被排除）。用于「单项归因失败」行时，仍不得写「共同贡献」 |',
           '| 24.67 ≤ S ≤ 40.67 | 近似可加 |', '| S > 40.67 | 两半之和明显超过 F4：两项重叠（冗余） |', '',
           'Q1 不支持「近期加权」措辞；只有 Q2 能支持。', '',
           '#### Q2：`f4_recency_pool`（对照 F4）', '',
           '阶段 B 取三 seed 均值 D2，k₊ = ΔH > 0 的 seed 数。自上而下取第一个满足的行：', '',
           '| 行 | 条件 | 措辞 |', '|---|---|---|',
           '| 1 | D2 ≤ −6 | 「Q2 劣于 F4」 |',
           '| 2 | D2 ≥ 6，k₊ ≥ 2，`p_pool` < 0.05，方向偏新近 | 「可学习新近性加权带来增益」 |',
           '| 3 | D2 ≥ 6，k₊ ≥ 2，`p_pool` < 0.05，方向为偏老或平 / 混合 | 「增益来自重新加权，但不是近期偏好」；**不得称新近性** |',
           '| 4 | D2 ≥ 6，且 k₊ < 2 或 `p_pool` ≥ 0.05 | 「均值过线但 seed 间不一致或合并检验不显著，未满足近期加权有效的条件，增益不稳」 |',
           '| 5 | −6 < D2 < 6 | 「在 lr=1e-3 与 best-epoch 预算内未检出近期加权增益」；**不得写成「新近性没有价值」**，并带上轨迹旗标的取值 |', '',
           '- 行 2、3 的措辞都要带一句：Q2 的权重同时乘在 F4 的 P、G 加性项上，增益可能来自对这两项的重新加权，不一定是内容部分的近期加权；形状指标只能部分区分这一点。',
           '- 只有 best epoch 的 logit 经 verify 复核，其余 epoch 的值来自 manifest 自述。',
           '- 作 anchor 的条件（不自动切换）：行 2 成立；三 seed 全部 verify 通过；至少 2 个 seed 的 Q2−F4 exact McNemar p < 0.05（报 Bonferroni ×3 后的值）；对新池化代码做一次独立泄漏 review；通过后另写 decision_log 条目。', '',
           '#### Q3：`f3_user_activity`（对照 F3，再比 F4）', '',
           '阶段 B 取三 seed 均值 D3 = mean ΔH(Q3 − F3)。自上而下取第一个满足的行：', '',
           '| 行 | 条件 | 措辞 |', '|---|---|---|',
           '| 1 | D3 ≤ −6 | 「用户级先验有害」；用 `user_activity_buckets` 对照训练 / 验证里 n = 50 的占比解释分布偏移 |',
           '| 2 | D3 ≥ 13.07，且不满足「k₊ ≥ 2 且 `p_pool` < 0.05」 | 「均值过线但 seed 间不一致或合并检验不显著，不下增益量级结论」 |',
           '| 3 | D3 ≥ 26.13（ρ ≥ 0.8），k₊ ≥ 2，`p_pool` < 0.05 | 「Q3 取得了 F4 增益的 80% 以上（ρ ≥ 0.8）；这说明增益量级相当，**不说明两者捕捉同一信息**」 |',
           '| 4 | 13.07 ≤ D3 < 26.13（0.4 ≤ ρ < 0.8），k₊ ≥ 2，`p_pool` < 0.05 | 「Q3 取得了 F4 增益的 40%–80%（0.4 ≤ ρ < 0.8）；这说明增益量级约为 F4 的一半上下，**不说明它们捕捉了同一部分信息，也不说明剩余部分来自哪里**」 |',
           '| 5 | −6 < D3 < 13.07（ρ < 0.4 且未到「有害」） | 「所选的两个粗粒度统计量（n 桶 + 不同周数）相对 F3 的增益不足 F4 增益的 40%（ρ < 0.4；含 D3 ≤ 0，即无增益或为负）」；**不得据此推断 F4 靠逐事件信息**。若 `diagnostics.q3_scale_ratio` 首个 epoch 的中位数 ≫ 1，必须附带「Q3 起点的用户级项量级约为 F4 等价项的 3 倍，阴性结果有一部分可能来自初始化量级」 |', '',
           '- **措辞纪律：行 3 / 4 只允许说增益量级。「复现」「覆盖」「解释了」「近似可得」等词在 Q3 结论里一律不用**，因为 ρ 只比较增益的大小，不比较增益落在哪些 query 上。',
           '- Q3 相对 F3 是否有效，独立判定：有效 ⇔ D3 ≥ 6 且 k₊ ≥ 2 且 `p_pool` < 0.05；不满足时写「Q3 相对 F3 未检出有效增益」。例如 D3 ∈ [6, 13.07) 且满足 k₊、`p_pool` 时，结论是「Q3 相对 F3 有效，但取得的增益不足 F4 的 40%」。该判定不改变上表任何一行的取值与措辞。',
           '- Q3 与 F4 直接比较：E ≥ 6 →「用户级先验优于 F4」（下一批把它叠在 F4 上，另算名额）；E ≤ −6 →「劣于 F4」；其余 →「与 F4 无可检出差异」。',
           '- 描述性统计（只报告，不设阈值，不进任何措辞）：增益 query 重合度 |G(Q3−F3) ∩ G(F4−F3)| ÷ |G(F4−F3)|，对照基线 |G(Q3−F3)| ÷ (2520 − H(F3))。',
           '- 结论范围：只覆盖「长度桶 + 活跃周数桶」合在一起，不能外推成「所有用户级先验」，也归因不到其中一个。', '',
           '#### 共用约束', '',
           '- 所有结论仅限 P 线、L50、id_mean、N1 基线下的 validation（week 25，n = 2520），不外推到 SASRec / G 线 / 其他窗口。',
           '- 失败 attempt 不退名额，修复后用新 run_id。指标数字只在 experiment_log 维护，本节不放结果。', '',
           '### 10.6 风险与已知限制', '',
           '- **继承风险 I-012（F4 泄漏 review LOW 3）**：5 个变体（含 `f4_control`）都建立在同一个 prefix 构造上，即 `history_v2` 的「严格早于 target」。`docs/issue_log.md` 的 I-012（Week 与文件内顺序能否稳定构造 target-prefix）仍是 Open，F4 泄漏 review 的 LOW 3 记的是同一个文件内行序假设的证据缺口。位置桶（Q1a / Q2）与 n（Q3）对同周内的行序最敏感。FEAT-03 不改变也不关闭这条风险。',
           '- 噪声大于门槛：门槛 ΔH ≥ 6 约 0.9 个标准误（F3−N1 三个 seed 的 ΔH 是 +21 / +6 / +29，三 seed 均值标准误约 6.7，只有 3 个样本的粗估）；零效应时 D ≥ 6 的概率粗估约 19%，所以追加 `p_pool` 与 k₊ 作必要条件。seed-42 单 run 只能给方向。',
           '- Q2 可能因预算不足得假阴性：标量 logit 与其余参数同用 lr=1e-3，到 best epoch 时最多移动 1–2 个单位量级。空结果只写「在 lr=1e-3 与 best-epoch 预算内未检出」。',
           '- Q2 同时用位置与周间隔两套 logit，归因不到其中一个；Q3 同样捆了两张表。',
           '- Q3 只能在 L50 窗口里算：n 在 50 处饱和，真实长度 ≥ 50 的用户落在同一个长度桶里；用户级先验受训练 / 验证分布偏移影响，可能是负结果。',
           '- 本地未验证项（UNKNOWN）：GPU 上 `nn.Parameter` 标量 logit 的 best checkpoint 逐位重导出、`synthetic-verify` 在 CUDA 上能否走通，要等部署前的 CUDA 预演；Q2 的 runner 级 PAD 宽度不变性本地只验到 allclose(1e-6)，不是逐位相等；真实数据路径未在本地验证。',
           '- 回归基线不是全绿：验收口径是「相对 FEAT-02 基线无新增失败」。',
           *refreeze_lines, '',
           '### 10.7 机器可读块', '',
           '```json feat03-freeze', json.dumps(facts, indent=1, sort_keys=True, ensure_ascii=False), '```',
           END]
    return '\n'.join(out)
# ------------------------------------------------------------------ marker handling
def block_of(text):
    """Returns (before, inner, after) around the markers, or None when the card has no section 10 yet."""
    if BEGIN not in text and END not in text:
        return None
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        fail('card markers are not exactly one begin followed by one end')
    head, rest = text.split(BEGIN)
    inner, tail = rest.split(END)
    return head, inner, tail
def machine_block(inner):
    start, stop = inner.find('```json feat03-freeze\n'), inner.rfind('\n```')
    if start < 0 or stop < start or inner.count('```json feat03-freeze') != 1:
        fail('card must contain exactly one ```json feat03-freeze block')
    return json.loads(inner[start + len('```json feat03-freeze\n'):stop])
# ------------------------------------------------------------------ commands
def write(args):
    clean_tree(args.worktree)
    facts = gather(args.worktree)
    card = Path(args.card)
    text = card.read_text(encoding='utf-8')
    before_mtime = card.stat().st_mtime_ns
    parts = block_of(text)
    refreeze, when = [], None
    if parts is not None:
        old = machine_block(parts[1])
        refreeze = [line for line in parts[1].splitlines() if line.startswith('- Re-freeze（')]
        if all(old.get(k) == facts[k] for k in ('commit', 'configs', 'constants', 'params', 'audit', 'thresholds')):
            when = old['freeze_time_utc']                               # nothing changed: keep the original freeze time (idempotent)
    if when is None:
        when = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        if parts is not None:
            refreeze.append('- Re-freeze（%s）：取代 commit `%s`，此前未提交任何训练。' % (when, old['commit']))
    facts['freeze_time_utc'] = when
    section = render(facts, refreeze)
    if parts is None:
        new = text + ('' if text.endswith('\n') else '\n') + section + '\n'
    else:
        new = parts[0] + section + parts[2]
    tmp = card.with_name(card.name + '.tmp')
    if card.stat().st_mtime_ns != before_mtime:
        fail('card changed while writing (another writer?); re-run')
    tmp.write_text(new, encoding='utf-8')
    tmp.replace(card)
    print(json.dumps({'status': 'written', 'commit': facts['commit'], 'freeze_time_utc': when, 'refreeze_lines': len(refreeze)}))
    return 0
def check(args):
    clean_tree(args.worktree)
    head = git(args.worktree, 'rev-parse', 'HEAD').strip()
    text = Path(args.card).read_text(encoding='utf-8')
    parts = block_of(text)
    if parts is None:
        fail('card has no section 10 (run --write first)')
    inner = parts[1]
    block = machine_block(inner)
    errors = []
    if block.get('commit') != head:
        errors.append('card commit %s != HEAD %s' % (block.get('commit'), head))
    if block.get('status') != 'FROZEN' or '状态：**FROZEN**' not in inner:
        errors.append('FROZEN missing')
    configs = config_facts(args.worktree)
    for key, entry in configs.items():
        if block.get('configs', {}).get(key) != entry:
            errors.append('config %s differs between card and git HEAD' % key)
        if entry['sha256'] not in inner or entry['path'] not in inner:
            errors.append('config %s sha256/path missing from the section text' % key)
    constants, params = code_facts(args.worktree)
    if block.get('constants') != constants:
        errors.append('constants differ between card and code')
    if block.get('params') != params:
        errors.append('parameter counts differ between card and code')
    audit = audit_facts(args.worktree)
    if block.get('audit') != audit or audit['sha256'] not in inner:
        errors.append('audit constants differ between card and git HEAD')
    if block.get('thresholds') != THRESHOLDS:
        errors.append('thresholds differ between card and the freeze script')
    for token in THRESHOLD_TEXT:
        if token not in inner:
            errors.append('threshold text missing from the section: ' + token)
    if head not in inner or constants['bucket_spec3_sha256'] not in inner:
        errors.append('commit or bucket_spec3_sha256 missing from the section text')
    if 'I-012' not in inner:
        errors.append('inherited risk I-012 missing')
    try:
        datetime.datetime.strptime(block['freeze_time_utc'], '%Y-%m-%dT%H:%M:%SZ')
    except (KeyError, ValueError):
        errors.append('freeze_time_utc missing or malformed')
    if errors:
        for item in errors:
            print('CHECK-FAIL: ' + item, file=sys.stderr)
        return 1
    print(json.dumps({'status': 'frozen_ok', 'commit': head, 'freeze_time_utc': block['freeze_time_utc'],
                      'config_sha256': {k: v['sha256'] for k, v in configs.items()}}, sort_keys=True))
    return 0
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--worktree', required=True)
    parser.add_argument('--card', required=True)
    args = parser.parse_args(argv)
    return write(args) if args.write else check(args)
if __name__ == '__main__':
    raise SystemExit(main())
