# 固定运维入口可安装版本v2：独立C档复核

日期：2026-10-21。复核者：短期独立Codex subagent，只读审查生产文件；仅本报告及本地synthetic验证证据写入。系统声明可核实为GPT-6 / Codex，精确运行时model ID为UNKNOWN，不声称切换模型或使用Luna。依据：AGENTS §4A、decision_log f0e364e与1c5c15f。

## 结论与边界

最终修复版未发现剩余Blocking、Important或Minor问题；上一轮三项发现均已关闭。此结论仅针对下列源码与安装说明，不等于安装完成或规则可安装。

- Eddy本人安装副本：尚无回执。
- 安装后的owner/mode/ACL/SHA、实际安装路径execpolicy实跑：PENDING。
- 默认执行者沙箱内针对实际安装副本的失败写打开探针：PENDING。
- `ops-fixed-entrypoints-and-rules-draft`仍为`passes=false`；本轮不得安装规则，不授权服务器execute/deploy。

## 三项发现的关闭证据

1. **原Blocking：ACL不能只靠W_OK。已关闭。** `check_acl`固定使用`/bin/ls -lde`并清洁环境；非零退出、空结果、未知ACE格式及任何allow ACE均拒绝，已识别deny ACE保留。`root_chain`逐层检查安装root及所有祖先，`payload`逐项检查运行目录/文件，package检查现存目标祖先。真实macOS合成fixture上的`delete_child`、`delete_child,add_subdirectory`、`delete_child,add_file`、`writeattr`、`writesecurity`、`chown`、`append`、`delete`八组变体全部被拒绝。未创建root所有的真实安装fixture，未将普通用户fixture冒称安装证明。
2. **原Important：0700/0600仍可能继承公开读取ACL。已关闭。** 私有目录新增同样的ACL拒绝。真实继承读取ACL复现已在创建回执前失败，目录仍为空。独立隔离私有目录guard后，新回执guard也拒绝继承allow ACL，且在写任何内容前失败；保留的合成回执大小为0。回执另核owner与精确0600；正常mkstemp路径测试通过。
3. **原Important：安装命令失败后仍覆盖旧安装。已关闭。** 特权复制块使用子shell和`set -eu`，先拒绝已有目标及dangling symlink，再排他mkdir；父目录另行核对。对说明中的实际子shell块，以stub sudo在`/bin/sh`和`/bin/zsh`各复现一次已有目标，两者均非零退出、无任何sudo调用。没有实际执行sudo、安装或覆盖文件。

## 其余范围核对

- 安装入口通过原wrapper的`/usr/bin/python3 -I`执行，仅从相邻固定安装目录载入client；运行时root不取自repo/env/cwd。source checkout的execute在SSH前拒绝。
- root所有、禁止写及ACL检查覆盖运行时信任链；只读manifest的精确文件集合绑定wrapper、client、server和install.json，config核对root及固定私有目录。系统解释器/标准库保持系统信任边界。
- 私有transport目录从OS账号库home加固定后缀得到，不读调用者HOME、不推导repo；新回执只写该目录，目录/文件均检查ACL。
- package只排他新建staging；不sudo、不安装副本、不写配置层。仓库草稿保持安装绝对路径占位符，渲染后恰六条allow，包括deploy execute，无宽泛shell/ssh/git前缀；额外argv由入口拒绝。
- verifier只本地执行plan、故意非法argv及execpolicy check；不会执行合法远端action或安装规则。probe只尝试O_WRONLY|O_NOFOLLOW打开server.py，未使用O_TRUNC、write或升级权限，并比较前后SHA。
- 安装说明要求Eddy从公开固定发布版本核SHA后本人安装，已有副本不得覆盖；后续改动须新版本、新review和重新安装。公开SHA与最终交付报告由执行者发布阶段补齐，未完成前说明明确禁止安装。
- wrapper及server与已接受的`4a58d50`字节相同；两份复用deploy工具与`5840e6b`字节相同。未修改服务器执行边界。

## 验证与证据

复核者实际运行：

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest tests.test_fixed_ops_installable tests.test_fixed_ops
git diff --exit-code 4a58d50 -- ops/bin/vkgr-fixed ops/fixed/server.py
git diff --exit-code 5840e6b -- ops/tools/local/deploy_spec.py ops/tools/remote/g_deploy.py
git diff --check
```

新增10项加原11项，共21项全部通过；两组字节比较及diff卫生检查通过。旧ACL复现全部按逐项状态重验，没有等价变异豁免。独立验证使用新缓存目录内的源码拷贝、`python -B`与`PYTHONDONTWRITEBYTECODE=1`隔离；没有以原路径缓存复用替代隔离证据。

- `cache/ops-installable-20261021/reviewer-final-tests.log`：复核者21项测试原始输出。
- `cache/ops-installable-20261021/reviewer-final-reproductions.log`：八组ACL变体、目录与新文件继承ACL、sh/zsh安装失败关闭路径。
- `cache/ops-installable-20261021/acl-red.log`、`acl-green.log`：实现者fresh RED→GREEN，复核者已核原始失败与通过输出。
- 前次缺口原始复现保留于`reviewer-acl.log`、`reviewer-acl-rights.log`、`reviewer-acl-receipt.log`；均位于上述同一gitignored缓存目录，不公开实际本机路径。

## 被复核字节（SHA256）

| 文件 | SHA256 |
| --- | --- |
| ops/bin/vkgr-fixed | e308b476ff6908152b8d3b638e637f9f242f27596dddaba18fa4f478cc290928 |
| ops/fixed/local.py | 72ad533caa3b7179381ca9f3c2a0ac662afca513c94dacde4af9b72ca1031a34 |
| ops/fixed/server.py | 8a3b01c1f8ad894416b8ff401f9bf60f6592a77c2a7594f3fccc710883c5eb9e |
| ops/fixed/package.py | ba495ee320ff3c9dad44137f6e782346ad94b6c87faf0b2489782142762efd48 |
| ops/fixed/verify_install.py | 35bcaa34a2ad4eb616e287539723865e78e0f95e8513ea803d1b884f363773d4 |
| ops/fixed/sandbox_write_probe.py | ae68ffe233bd4e6b1dfea3e14025656330ea0ac421ce4487a781b121ec3ddcd9 |
| docs/ops/codex-allow.rules.draft | eb527e6e0767c7811657656f633fdef7f758bf1b1d87ee0679306eee2a190eee |
| docs/ops/2026-10-21-fixed-ops-v2-install.md | 32a7bd1fc6d6e898a9c35179ad42abde1ef168cff5dbedb088b10dca543e9b87 |
| tests/test_fixed_ops_installable.py | 8073ad93ed0662195e022373fb1a0bce5bee7654b1ad713a1fbf8217b785c4e4 |
