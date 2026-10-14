"""GFEAT-02 freeze: append section 11 (GFEAT-02, G side features on A1) to the FEAT-01 experiment card, or check it.

  python gfeat02_freeze.py --card <card.md> --write    append the machine-generated block (refuses a card that already has it)
  python gfeat02_freeze.py --card <card.md> --check    recompute everything and compare; exit 1 on any difference

Everything in the block is computed, not typed: config sha256 (disk == the frozen commit's blob), source sha256 of the GFEAT-02
and composed modules, parameter counts (built with the real G vocabulary, V=80, confirmed by the GARCH-01 server preflight), the
sha256 of the card text before the block (so the earlier sections stay byte-identical). ``--no-git`` is for the unit test only
(the test harness is not this checkout): it records commit null, which a normal --check rejects.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BEGIN = '<!-- gfeat02-freeze:begin -->'
END = '<!-- gfeat02-freeze:end -->'
FENCE = '```json gfeat02-freeze'
SCHEMA = 'gfeat02-freeze-v1'
BASE_COMMIT = '1ffbe2722e3b658beb5f18a246137f05604db343'  # gfeat02/g base: GARCH-01 5983cf6 + GFEAT-01 e4d1118
VOCAB = 80
A1_RUN = 'g-garch01-a1-lr3e-4-s42'
A1_CONFIG = 'configs/garch01/g-garch01-a1-lr3e-4-s42.json'
CONFIGS = {'none_ctrl': ('configs/gfeat02/g-gfeat02-none-ctrl-s42.json', 'none', False),
           'f2_feedback': ('configs/gfeat02/g-gfeat02-f2-feedback-s42.json', 'f2_feedback', True),
           'f3_watch_buckets': ('configs/gfeat02/g-gfeat02-f3-watch-buckets-s42.json', 'f3_watch_buckets', True)}
SOURCES = tuple(f'src/vkgr/{n}.py' for n in ('generative_arch_feat', 'generative_b1_arch_feat', 'tiger_arch', 'generative_arch', 'generative_b1_arch',
                                             'generative_feat', 'generative_feat_eval', 'generative_b1_feat', 'history_side', 'retrieval_features'))
RULE = {'delta_recall_at_50': 0.002, 'hits': 6, 'queries': 2520}  # 0.002 x 2520 = 5.04 -> at least 6 more hits


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, check=True).stdout


def params():
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from vkgr import generative_arch as ga
    from vkgr import generative_arch_feat as gaf
    from vkgr.tiger_arch import TigerArchModel, expected_param_count
    out = {}
    for name, (path, feature, _) in CONFIGS.items():
        config = json.loads((ROOT / path).read_text()); shape = [config[k] for k in ga.SHAPE_KEYS]
        torch.manual_seed(0)
        if feature == 'none':
            model = TigerArchModel(VOCAB, *shape); side = {'params_side': 0, 'params_side_trainable': 0}
        else:
            model = gaf.TigerArchModelSide(VOCAB, *shape); model.add_side_tables(feature)
            side = {k: v for k, v in gaf._side_params(model, feature).items() if k in ('params_side', 'params_side_trainable')}
        base = expected_param_count(VOCAB, *shape[:-1])
        total = sum(p.numel() for p in model.parameters())
        if total != base + side['params_side']:
            raise ValueError('parameter count disagrees with the GARCH-01 closed form + side tables')
        out[name] = {'params_total': total, 'params_arch': base, **side}
    return out


def compute(commit):
    configs = {}
    for name, (path, feature, counted) in CONFIGS.items():
        data = (ROOT / path).read_bytes()
        if json.loads(data)['feature'] != feature:
            raise ValueError(f'{path}: feature is not {feature}')
        configs[name] = {'path': path, 'feature': feature, 'counted': counted, 'sha256': sha256(data)}
    paths = [c['path'] for c in configs.values()] + list(SOURCES) + [A1_CONFIG]
    if commit is not None:
        for path in paths:
            if git('show', f'{commit}:{path}') != (ROOT / path).read_bytes():
                raise ValueError(f'{path}: working tree differs from the frozen commit')
    return {'schema': SCHEMA, 'status': 'FROZEN', 'task_id': 'GFEAT-02', 'commit': commit, 'base_commit': BASE_COMMIT, 'configs': configs,
            'a1_config_sha256': sha256((ROOT / A1_CONFIG).read_bytes()), 'a1_reference_run': A1_RUN,
            'sources_sha256': {p: sha256((ROOT / p).read_bytes()) for p in SOURCES}, 'params': params(), 'vocab': VOCAB, 'rule': RULE}


def render(block):
    c, p = block['configs'], block['params']
    rows = '\n'.join(f"| {name} | `{v['path']}` | {'是' if v['counted'] else '否'} | `{v['sha256']}` | {p[name]['params_total']:,} | {p[name]['params_side']:,}（可训练 {p[name]['params_side_trainable']:,}） |"
                     for name, v in ((n, c[n]) for n in CONFIGS))
    machine = json.dumps(block, indent=1, sort_keys=True)
    return f"""{BEGIN}
## 11. GFEAT-02（A1 上的 G 特征）冻结记录
状态：**FROZEN**。冻结时间（UTC）：{block['freeze_time_utc']}。代码 commit（`gfeat02/g`）：`{block['commit']}`。基线：`{block['base_commit']}`（GARCH-01 + GFEAT-01 合并）。
本节由 `gfeat02_freeze.py --write` 从代码与 git 机械生成（sha256、commit、参数数都不是手写）；本节之前的既有内容一字未改，只在末尾追加（本节之前全文的 sha256 记在机器块 `prefix_sha256`；本卡没有 §10，编号 11 按任务书）。冻结后任何新的代码 commit 都要重新冻结。
### 11.1 研究问题
- GFEAT-01 在 core03（1+1 层）上验证过的两个 encoder side 特征——F2 反馈位、F3 观看时长分桶 + 观看比例分桶——放到论文规模模型 A1（GARCH-01，4+4 层、6×64 heads、d128、FF1024、dropout 0.1、LR 3e-4）上，是否仍提升 R@50。
### 11.2 对照与唯一改变量
- 入口 `python -m vkgr.generative_b1_arch_feat`；注入方式与 GFEAT-01 相同（同一函数对象）：encoder 输入 = (E[token] + side_vector(item)) × √d + position，side 广播到该 item 的 4 个 SID token，decoder 不变。
- none-ctrl（不计名额）：feature=none，走 GARCH-01 的工厂，模型、初始化、训练、评估与 A1 相同；只差 config 的 schema / task_id / feature / side_version 四个键与 `binding.source`。目的：证明新入口不引入偏差。
- F2（计 1 个名额）：唯一改变量 = 加 F2 反馈位 side embedding（6×d）。
- F3（计 1 个名额）：唯一改变量 = 加 F3 两张分桶 side embedding（8×d + 6×d，PAD 行 0）。
- 其余全部同 A1：`SID_BHKM_B16D3_S42`、`HIST_V2_OBS` L50、`HIST_V2_SIDE_V1`、`init=small`、100 / 5 / 3 epoch、microbatch 256、beam 100、seed 42、attempt 1。side 表在 GARCH-01 初始化之后抽取，并在分叉的 RNG 里抽取，所以 F2 / F3 训练中的 dropout mask 与 none-ctrl 相同。
- 特征定义（`history_side.py` / `retrieval_features.py`）与 GFEAT-01 逐字节相同（sha256 见机器块 `sources_sha256`）；分桶边界只用 train 定义并写死在代码里。
### 11.3 预注册解读规则（看到任何 GFEAT-02 数字之前写死）
1. none-ctrl 复现闸门：none-ctrl 的 `selection`（best_metric = [R@50, N@50]、best epoch、stop epoch）必须与 A1 seed42 run `{block['a1_reference_run']}` 的 manifest 完全相同（A1 数值见 experiment_log 的 GARCH-01 节，本卡不复制）。同时报告逐 epoch 的 `model_sha256` / `computation_sha256` 是否相同。**闸门不过 = 新入口有偏差，F2 / F3 结果作废，不解读。**
2. 有效：R@50(F) − R@50(none-ctrl) ≥ {RULE['delta_recall_at_50']}（2520 个 validation query 上即 ≥ {RULE['hits']} 个命中；0.002 × 2520 = 5.04）。
3. 差值在 (−0.002, +0.002) 记 inconclusive（不等于「没有作用」）；≤ −0.002 记有害。
4. F2、F3 都有效时按 R@50 → N@50 → 枚举序（F2 < F3）选一个；入选者补 seed 2024 / 2025（不计名额）。单 seed 结论只能写「seed42 筛选」。
5. 评估只读 validation（n=2520，OOV 623 计入分母），test（week 26）关闭。
### 11.4 配置、sha256 与参数数（V={block['vocab']}，GARCH-01 服务器预检 PF3 确认）
| 组 | config | 计名额 | sha256 | 参数总数 | side 表参数 |
|---|---|---|---|---|---|
{rows}

A1 config sha256：`{block['a1_config_sha256']}`；三个 config 与它只差上面说的四个键。
### 11.5 机器可读块
{FENCE}
{machine}
```
{END}
"""


def read_block(text):
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        raise ValueError('exactly one GFEAT-02 block required')
    section = text[text.index(BEGIN):text.index(END)]
    if section.count(FENCE) != 1:
        raise ValueError('machine block missing')
    body = section.split(FENCE, 1)[1].split('\n```', 1)[0]
    return json.loads(body)


def write(card, use_git):
    text = card.read_text()
    if BEGIN in text or END in text:
        raise ValueError('the card already has a GFEAT-02 block (re-freeze = restore the backup first)')
    commit = git('rev-parse', 'HEAD').decode().strip() if use_git else None
    if use_git and git('status', '--porcelain', '--', *SOURCES, 'configs/gfeat02', A1_CONFIG, 'gfeat02_freeze.py'):
        raise ValueError('frozen files have uncommitted changes')
    prefix = text if text.endswith('\n') else text + '\n'
    block = {**compute(commit), 'freeze_time_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'prefix_sha256': sha256(prefix.encode())}
    card.write_text(prefix + render(block))
    return block


def check(card, use_git):
    text = card.read_text(); block = read_block(text)
    if text[text.index(END) + len(END):].strip():
        raise ValueError('text after the GFEAT-02 block')
    if use_git and block.get('commit') is None:
        raise ValueError('block without a commit')
    if sha256(text[:text.index(BEGIN)].encode()) != block['prefix_sha256']:
        raise ValueError('the card text before section 11 changed')
    expected = {**compute(block['commit'] if use_git else None), 'freeze_time_utc': block['freeze_time_utc'], 'prefix_sha256': block['prefix_sha256']}
    if expected != block:
        raise ValueError('recomputed freeze values differ: ' + ', '.join(sorted(k for k in set(expected) | set(block) if expected.get(k) != block.get(k))))
    if render(block) != text[text.index(BEGIN):]:
        raise ValueError('rendered section differs from the card')
    return block


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--card', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true'); mode.add_argument('--check', action='store_true')
    parser.add_argument('--no-git', action='store_true', help='unit tests only: skip the commit cross-check, record commit null')
    args = parser.parse_args(argv)
    try:
        block = (write if args.write else check)(Path(args.card), not args.no_git)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f'GFEAT-02 freeze failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({'mode': 'write' if args.write else 'check', 'status': 'ok', 'commit': block['commit'], 'configs': {k: v['sha256'] for k, v in block['configs'].items()}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
