# 固定运维入口v2：Eddy安装副本，暂不安装规则

依据f0e364e §5与1c5c15f。首次范围全部六条：plan、status/resources/queue/receipts execute、deploy execute。旧版实现与静态回执已由f0e364e核对；本版改变运行时信任边界，须C档重新复核。本页不授权服务器execute/deploy，不安装Codex规则，task_list仍false。

## 布局与边界

安装目录采用新版本名，位于执行者沙箱所有可写root之外。目录及所有祖先必须root所有、不允许执行者写；逐层原生ACL检查拒绝任何allow ACE，未知格式/读取失败也拒绝，保留deny ACE；安装root/bin/fixed均0555。客户端、server.py、install.json、manifest.json均root所有/0444，入口为root所有/0555。只能读取该副本的server.py；只读manifest绑定全部运行时字节。安装位置不来自env/cwd，source checkout的execute会在SSH前拒绝。

transport目录固定为OS账号库的home加`Library/Logs/VKGRFixedOps-v2`，不读环境HOME、不从入口位置推导。Eddy预先创建本人所有/0700目录；目录与新回执均拒绝allow ACL，新回执写入内容前核ACL及0600；mkstemp/0600，stdout/stderr各最多64KiB尾部。实际本机路径只落私有映射和本机生成草稿，不进仓库。

仓库规则只含`@INSTALLED_ENTRY@`占位符，package.py生成六条安装绝对路径草稿。prefix依然会匹配额外参数，入口负责拒绝；规则看不到服务器目标状态。相对cwd依赖已移除。root SSH、deploy key及既有服务器权限不由本版改变。

## 安装前置与逐条步骤（全部由Eddy本人执行）

下列占位符由Eddy在本机填入；不将替换后的本机路径提交Git：

- `<REVIEW_CHECKOUT>`：新公开发布分支的独立检出，放在执行者可写root之外。
- `<INSTALL_PARENT>` / `<INSTALL_ROOT>`：root控制的父目录与全新版本目录，路径不含空格或shell字符；不可位于项目、临时目录、visualizations或用户可写祖先下。
- `<NEW_PACKAGE_DIR>` / `<NEW_CHECK_DIR>`：全新私有staging/检查目录，不覆盖旧文件。
- `<PRIVATE_RECEIPTS_DIR>`：package打印配置中固定的OS用户私有目录。
- `<CODEX_CLI>`：本机Codex CLI绝对路径。

发布分支：`codex/fixed-ops-installable-public-20261021-v2`。固定公开SHA及逐文件SHA看同目录的v2交付报告；报告未填完/独立review未完成时不安装。package仅构建文件，不会sudo、复制到安装位置或写Codex配置。

1. 从公开分支新克隆，核HEAD与报告SHA相同；核报告列出的入口、local.py、server.py、package.py、verify_install.py、sandbox_write_probe.py逐文件SHA。不要从日常可写工作树复制。

```text
git clone --single-branch --branch codex/fixed-ops-installable-public-20261021-v2 https://github.com/EddyHai57/vk-generative-retrieval.git '<REVIEW_CHECKOUT>'
git -C '<REVIEW_CHECKOUT>' rev-parse HEAD
/usr/bin/shasum -a 256 '<REVIEW_CHECKOUT>/ops/bin/vkgr-fixed' '<REVIEW_CHECKOUT>/ops/fixed/local.py' '<REVIEW_CHECKOUT>/ops/fixed/server.py' '<REVIEW_CHECKOUT>/ops/fixed/package.py' '<REVIEW_CHECKOUT>/ops/fixed/verify_install.py' '<REVIEW_CHECKOUT>/ops/fixed/sandbox_write_probe.py'
/usr/bin/python3 -I '<REVIEW_CHECKOUT>/ops/fixed/package.py' --install-root '<INSTALL_ROOT>' --output '<NEW_PACKAGE_DIR>'
/bin/cat '<NEW_PACKAGE_DIR>/install.json'
/bin/cat '<NEW_PACKAGE_DIR>/manifest.json'
```

2. 父目录若不存在才用下面mkdir新建（已有则核root所有、无执行者write access，跳过创建，不修改它）。版本root必须不存在，mkdir本身排他拒绝已有目标。以下仅写这个全新版本，不覆盖旧安装。

```text
# 父目录不存在时，Eddy单独创建；存在时先核owner/mode/ACL，不修改它。
# sudo /bin/mkdir -m 0755 '<INSTALL_PARENT>'
(
set -eu
test ! -e '<INSTALL_ROOT>'
test ! -L '<INSTALL_ROOT>'
sudo /bin/mkdir -m 0755 '<INSTALL_ROOT>'
sudo /bin/mkdir -m 0755 '<INSTALL_ROOT>/bin' '<INSTALL_ROOT>/fixed'
sudo /usr/bin/install -o root -g wheel -m 0555 '<NEW_PACKAGE_DIR>/bin/vkgr-fixed' '<INSTALL_ROOT>/bin/vkgr-fixed'
sudo /usr/bin/install -o root -g wheel -m 0444 '<NEW_PACKAGE_DIR>/fixed/local.py' '<INSTALL_ROOT>/fixed/local.py'
sudo /usr/bin/install -o root -g wheel -m 0444 '<NEW_PACKAGE_DIR>/fixed/server.py' '<INSTALL_ROOT>/fixed/server.py'
sudo /usr/bin/install -o root -g wheel -m 0444 '<NEW_PACKAGE_DIR>/install.json' '<INSTALL_ROOT>/install.json'
sudo /usr/bin/install -o root -g wheel -m 0444 '<NEW_PACKAGE_DIR>/manifest.json' '<INSTALL_ROOT>/manifest.json'
sudo /bin/chmod 0555 '<INSTALL_ROOT>' '<INSTALL_ROOT>/bin' '<INSTALL_ROOT>/fixed'
)
# 私有目录不存在时单独创建；已存在时先核0700/owner/无allow ACL。
/bin/mkdir -m 0700 '<PRIVATE_RECEIPTS_DIR>'
```

私有目录若已有，先核本人所有/0700且无symlink，符合则跳过mkdir，不覆盖或chmod既有目录。安装命令中的sudo只由Eddy本人执行；执行者不得代装副本。

3. 安装后先做只读owner/mode/SHA、plan/非法参数检查及六条execpolicy。输出只存新私有检查目录，rendered.rules.draft不会放入任何配置层。不会执行合法远端action，包括deploy。

```text
/usr/bin/python3 -I '<REVIEW_CHECKOUT>/ops/fixed/verify_install.py' --install-root '<INSTALL_ROOT>' --codex '<CODEX_CLI>' --output '<NEW_CHECK_DIR>'
```

4. 告知Codex副本已安装及安装root。Codex在默认沙箱、无提权重跑检查，并运行以下探针。探针仅尝试用O_WRONLY打开server.py，不带O_TRUNC、不调用write；应EACCES/EPERM，前后SHA相同。若工具层直接拒绝，保留拒绝回执；不提权重试探针。目录不存在或安装校验未通过时只写PENDING，不冒称完成证明。

```text
/usr/bin/python3 -I '<REVIEW_CHECKOUT>/ops/fixed/sandbox_write_probe.py' --install-root '<INSTALL_ROOT>'
```

## 后续与现场证据

安装副本、安装后execpolicy和默认沙箱失败探针是三个独立证据。目前没有Eddy安装回执，后两项必须PENDING；staging/模板静态检查不算安装后实跑。C档报告及v2交付报告列出最终状态。本轮任何阶段都不安装规则，passes=false。之后任何入口/client/server/config改动都用新版本、新review，由Eddy重装并按新审批处理旧规则；执行者不能改已装副本。
