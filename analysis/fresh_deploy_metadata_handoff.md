密级：公开

# Fresh deploy：可选 new-only OPS metadata

原必填字段及 Git fetch/branch/commit/files/clean-worktree 安全检查不变。没有下列可选字段时，旧 request、inventory pin、deploy 返回值和 deployment.json 字段集完全不变。

## 新 schema

- `new_ops_documents`：`{安全basename.json: JSON value}`。只允许单层JSON文件名；不能包含路径，`deployment.json`保留给工具。可同时传 parameters.json、templates.json、approval.json、preflight-request.json及其它安全JSON basename。
- `parameters.json` 特别要求非空的字符串key→绝对路径string字典，不允许相对路径或`..`。路径可指向尚未创建的运行输出；工具不读取这些路径的内容。
- `inputs_file_pin`：`{"path": "<绝对既存JSON文件>", "sha256": "<64位小写hex>"}`。只读、不能是最终symlink，先核原始bytes SHA，再核UTF-8/有限JSON/无重复字段。inventory与deploy均重核，Git操作后、创建OPS前还会重核。不会复制或重写这个输入文件。

新文档字节格式是 `json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')`：Python默认分隔符/ensure_ascii，无额外换行，兼容stack identity的默认sort_keys dump hash。所有新文档使用`xb`，OPS用独占mkdir；deployment.json最后用`x`写入，作为完成标记。

若有 parameters.json，deployment.json增加 `parameters_sha256`，从**实际落盘文件bytes**计算。若有 inputs_file_pin，增加 `inputs_sha256`，取实际校验后的外部输入文件bytes SHA。Stack需两者时须同时传这两个可选字段。`inputs_sha256`只绑定inputs_file_pin.path，不代表任意同名新文档；parameters中相应输入路径应指向该固定文件。

此工具不是整目录事务：IO或Git校验失败可能留下新worktree或部分新OPS，不能把存在目录当成deployed。不会覆盖、回滚或删除旧文件，不会写queue，也不会自动重试；失败后读状态对账并停止。只有全部新文档及最后metadata写入完成才返回deployed。

## 私有请求示例

实际路径/数据保留在私有JSON；下列值均为占位符。现有stdin调用方式不变：先inventory，再用其request_sha256作为deploy的inventory_pin。

```json
{
  "root": "/ABS/scheduler-root",
  "source": "/ABS/fetch-repository",
  "destination": "/ABS/new-code-worktree",
  "ops": "/ABS/new-ops",
  "branch": "codex/example-work-branch",
  "commit": "<40HEX_SHA>",
  "files": {"scripts/example_entry.py": "<64HEX_SHA>"},
  "run_ids": ["example-train"],
  "inputs_file_pin": {"path": "/ABS/accepted-inputs.json", "sha256": "<64HEX_SHA>"},
  "new_ops_documents": {
    "parameters.json": {"root": "/ABS/accepted-data-root", "inputs": "/ABS/accepted-inputs.json"},
    "templates.json": {"42": "/ABS/original-template-s42.json"},
    "approval.json": {"scope": "example-only"},
    "preflight-request.json": {"method": "preflight", "parameters": {"root": "/ABS/accepted-data-root"}}
  }
}
```

验证：既有runtime `python -m unittest analysis.test_fresh_deploy`，**9 tests GREEN**。覆盖新4文档/实际bytes hash、legacy字段集、坏pin/非法JSON/重复字段/NaN、inventory后输入漂移、路径与reserved metadata保护、既存OPS与worktree拒绝。Git行为全部mock；未访问真实服务器metadata或queue。本次仅改工具、对应tests及本说明。
