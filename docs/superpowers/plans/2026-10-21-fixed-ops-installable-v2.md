# 固定运维入口可安装版本v2

依据f0e364e §5、1c5c15f；采用receiving-code-review与TDD。全部六条，包括deploy execute；rules不得安装，passes=false。Eddy本人安装副本，执行者不能越俎代庖；真实安装后证据须等副本就位，不以staging冒充。

1. 将客户端运行依赖收束到安装root：bin/vkgr-fixed、fixed/local.py、fixed/server.py、install.json、manifest.json。root-owned只读副本，所有祖先root控制、不允许执行者写；只读manifest钉字节。移除运行时依赖可写Git checkout的校验与server读取。源代码execute拒绝；固定词解析不变。
2. 固定私有transport目录由OS账号库给出的home加固定后缀决定，不读HOME/cwd或推导repo；目录预先由Eddy创建、本人所有/0700，回执新文件0600。仓库rules仅占位符，staging生成安装绝对路径的六规则草稿，不写配置层。
3. 增加不安装的package builder、只读验证/execpolicy工具及无写入的O_WRONLY打开探针；TDD、原11项回归、隔离变异、C档重新独立review。公开压平发布，逐文件SHA与逐命令安装说明交Eddy。安装只允许新版本目录，今后变更要新review及重装/旧规则处理。
4. Eddy安装副本后，执行者只读检查实际owner/mode/sha与全部六条execpolicy（草稿仍不安装）；在默认沙箱中尝试无O_TRUNC/无write的写打开，必须EPERM/EACCES并核原字节不变。若尚无安装，这两份现场证据记PENDING，不能伪造完成。

Review focus：被allow代码是否仍可受repo/env/cwd影响；完整依赖和祖先目录/ACL；私有目录/receipt；六规则包括deploy，no broad prefix；验证工具不能安装规则或触发server execute/deploy；publish字节与安装SHA。
