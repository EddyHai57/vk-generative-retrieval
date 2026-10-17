密级: 公开

# 服务器 /tmp 残留报告汇总（2026-10-15，待 Eddy 手动处理）

仅从既有 report 汇总；本轮未在服务器列目录、未确认仍存在、未删除或覆盖任何路径。不能把下表当作删除批准或通配符执行清单。处理前需 Eddy 逐路径核对 owner/活句柄/内容、备份与可恢复性；没有核过的状态均 UNKNOWN。

| 服务器路径 | 报告来源 | 报告用途 / 可恢复性限制 |
|---|---|---|
| `/tmp/gtok01b-preflight-t2z_9m4e` | `.agents/tasks/gtok01b-server-report.md:76` | preflight scratch，报告称 prepared manifest 副本；当前内容/依赖/可恢复性未核 |
| `/tmp/gtok01b-preflight-y3m5khtx` | 同上 | 同上 |
| `/tmp/gtok01bc-preflight-r9n9yux9` | `.agents/tasks/gtok01bc-report.md:40,66` | preflight scratch；报告描述不替代当前逐路径核对 |
| `/tmp/gtok01bc-preflight-a5__ugin` | 同上 | 同上 |
| `/tmp/gtok02-pfA-<UTC>`，约两次 `20261015T085155Z` / `20261015T085444Z` | `.agents/tasks/gtok02-report.md:67` | **只有模式/时间线索，完整裸目录名 UNKNOWN**；不能据通配符删除，需 Eddy 先只读展开准确路径 |

以上是报告中明确归属服务器的路径。报告中另有本机 `/private/tmp` harness、`/tmp/php02-design.bak`、`/tmp/gtok01bc-drvtest`、seqm1 临时副本等，不能混作服务器目录；本轮没有扫描、处理或批准它们。本汇总不涵盖报告未列出的服务器临时文件，不能称为完整 /tmp 盘点。

## 2026-10-16 全权接收补充：GTOK-02 Preflight B

来源`.agents/tasks/gtok02-report.md:105–109`，时间为目录名中的UTC，不是本轮读取时间；报告称是可重新生成的prepared manifest副本。本轮只读报告，未连接服务器逐路径核实、未删除。以下12个加上原表两个pfA时间线索，是该报告所称14个残留；pfA完整目录名仍UNKNOWN。

| 服务器路径 | 来源 |
|---|---|
| `/tmp/gtok02-pfB-b32d3-20261015T102656Z` | gtok02-report:106，首次失败的preflight |
| `/tmp/gtok02-pfB-b64d3-20261015T102852Z` | 同上 |
| `/tmp/gtok02-pfB-b128d3-20261015T103103Z` | 同上 |
| `/tmp/gtok02-pfB-b512d3-20261015T103317Z` | 同上 |
| `/tmp/gtok02-pfB-b1024d3-20261015T103548Z` | 同上 |
| `/tmp/gtok02-pfB-b256d2-20261015T103821Z` | 同上 |
| `/tmp/gtok02-pfB-b32d3-20261015T104119Z` | gtok02-report:107，第二次preflight |
| `/tmp/gtok02-pfB-b64d3-20261015T104315Z` | 同上 |
| `/tmp/gtok02-pfB-b128d3-20261015T104525Z` | 同上 |
| `/tmp/gtok02-pfB-b512d3-20261015T104740Z` | 同上 |
| `/tmp/gtok02-pfB-b1024d3-20261015T105013Z` | 同上 |
| `/tmp/gtok02-pfB-b256d2-20261015T105244Z` | 同上 |

### 本机与主机未明确项（不混入服务器删除清单）

- seq-report:110：`/tmp/seqm1-base-scratch`、`/tmp/seqm1-e2e`、`/tmp/seqm1-e2e-wz1fw_bj`、`/tmp/seqm0/`。报告未逐项标主机；来自本地开发报告，当前主机/存在性/依赖仍UNKNOWN。
- 明确本机：gtok01bc-report:66的`/tmp/gtok01bc-drvtest/**`；php02-report:12的`/tmp/php02-design.bak`；feat01c-seeds-report:81的`/tmp/feat01c-added.txt`；feat03-report:138的`/tmp/f03d_ref`；gtok01r-server-report:245的`/tmp/gtok01r_poll.sh`、`/tmp/gtok01r-server-report.bak-feat008`。
- 旧本机harness/脚本较多，仅以报告原清单为准：batch1-code-report:76–79、garch01-report:125–126、gtok01-server-report:141–146、infra-01-report:186–187、infra-02-report:110、infra-02b-report:98–99、gtok01r-server-report:59。各条当前存在性、活句柄、可恢复性均未独立核实。
- g6prep-impl-report:147的OpenMP残留归属不确定；不要将同时出现的其它任务文件归给本项目或自动清理。全部仅交Eddy核对，不是删除授权。
