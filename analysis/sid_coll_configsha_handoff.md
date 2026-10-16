# SID-COLL config SHA metadata handoff

This tool changes only the twelve null `proof.expected["binding.config_sha256"]`
values and `control_dir` in a new plan. Every value comes from the original pinned
stage spec's nested seed/config reference, the frozen scientific Git config blob,
the actual small config file, and the original expected data config pin. It never
opens arrays, SID maps, checkpoints, predictions, or unverified recall scores.
Old source, jobs, data identities, resources, created time, deadline, STOP and
intent bytes remain immutable. The original failed records remain in place.

## Private request (JSON file contents)

All paths below are placeholders. Supply real paths only in private metadata.
`source_files` is the complete mapping shared by the twelve actual frozen specs;
its 296 production entries must come from those specs, not this example.
`ops_files` contains every `vkgr_ops/*.py` in the approved ops library. Its
canonical compact sorted UTF-8 JSON plus LF SHA must equal
`run_ops_source.sha256`; use the original run.json identity for commit and SHA.
Config files need not be included in the scientific Python source mapping.

```json
{
  "schema": "vkgr-sid-coll-configsha-handoff-v1",
  "old_plan": {"path": "<OLD_OPS>/pipeline-plan-capacity-v1.json", "sha256": "<OLD_PLAN_SHA256>"},
  "old_control_dir": "<OLD_OPS>/controller",
  "new_plan": "<OLD_OPS>/pipeline-plan-configsha-recovery-v1.json",
  "new_control_dir": "<OLD_OPS>/controller-configsha-recovery-v1",
  "old_owner": {"path": "<OLD_OPS>/controller/driver-pid-v3.json", "sha256": "<OLD_PID_RECORD_SHA256>"},
  "old_pid": 2931675,
  "created_epoch": 1792161406.75411,
  "deadline_epoch": 1792334206.75411,
  "scientific_commit": "11d67681e0cbde09983f6bf482f55ab8a7191bdb",
  "git_repo": "<SCIENTIFIC_GIT_OBJECT_REPOSITORY>",
  "model_root": "<IMMUTABLE_SCIENTIFIC_DEPLOYMENT>",
  "source_files": {"<EVERY_FROZEN_SPEC_PYTHON_SOURCE_PATH>": "<SHA256>"},
  "python": "<EXACT_CURRENT_APPROVED_SYS_EXECUTABLE>",
  "ops_library": "<APPROVED_OPS_LIBRARY_DIRECTORY>",
  "ops_files": {"vkgr_ops/__init__.py": "<SHA256>", "<EVERY_OTHER_PACKAGE_PY_FILE>": "<SHA256>"},
  "run_ops_source": {"commit": "<ORIGINAL_RUN_OPS_COMMIT>", "sha256": "<ORIGINAL_RUN_OPS_PACKAGE_SHA256>"},
  "approval": {"path": "<PRIVATE_METADATA>/approval.json", "sha256": "<APPROVAL_FILE_SHA256>"},
  "helper_root": "<CLEAN_PUSHED_HELPER_GIT_WORKTREE>",
  "helper_commit": "<EXACT_HELPER_HEAD_COMMIT>",
  "helper_source_sha256": "<GIT_AND_PHYSICAL_ANALYSIS_SID_COLL_CONFIGSHA_HANDOFF_PY_SHA256>"
}
```

The request accepts exactly the listed keys; no caller environment or scientific
patch dictionary. `python` must equal current `sys.executable` exactly. The helper
must have the requested Git HEAD, a clean worktree, and the pinned physical/Git
source SHA. The ops runtime environment is constructed by the tool: approved
ops library + model src/root on PYTHONPATH, one CPU thread per math library,
no bytecode, system default executable PATH. It cannot be replaced by a caller.

The pinned approval JSON must have exactly these five keys, matching Eddy's
retained direct answer. No exact human message time is invented:

```json
{
  "source": "Eddy direct reply in this conversation",
  "question_item_id": "<NONEMPTY_RETAINED_QUESTION_ITEM_ID>",
  "answer": "批准按提案恢复",
  "scope": ["<NONEMPTY_ORIGINAL_SCOPE_TEXT>", "<RETAIN_REMAINING_ORIGINAL_SCOPE_ITEMS>"],
  "not_yet_executed": true
}
```

`scope` must be a nonempty list of nonempty strings; preserve all five actual
approved scope items. `not_yet_executed` describes the captured approval artifact,
not the current execution status. It is not rewritten after prepare/start.

## Operation sequence and transport

Deploy the reviewed/pushed helper source and private documents first. Record the
exact helper commit and file SHA in the complete request, then pin the resulting
request file bytes. No `__file__` lookup is required in a bare bootstrap.
The stdin envelope for read-only inventory or new-only prepare is:

```json
{"action":"inventory","request":{"...":"COMPLETE_REQUEST_ABOVE"}}
```

Use `prepare` instead of `inventory` for the sealing action. After inventory is
accepted, prepare creates the sibling new control directory exclusively, copies
every old own intent guard byte for byte, writes the new plan exclusively and
writes `new_control/handoff.json` last. Old branch STOP and final-summary refs
are SHA sealed there but are never copied into active new STOP filenames.
`new_plan_sha256` is SHA256 of compact sorted JSON plus LF; an offline draft
serialized differently may have a different byte SHA while matching all values.
Partial failure preserves all new files. Do not clean, overwrite, or retry it.

The start envelope additionally identifies the deployed complete private request:

```json
{
  "action":"start",
  "request":{"...":"COMPLETE_REQUEST_ABOVE"},
  "request_file":{"path":"<PRIVATE_METADATA>/request.json","sha256":"<EXACT_REQUEST_FILE_BYTES_SHA256>"}
}
```

The file must contain the request object itself, not the transport envelope.
CLI equivalents run from the actual approved interpreter:

```text
<APPROVED_PYTHON> -B <HELPER_ROOT>/analysis/sid_coll_configsha_handoff.py inventory --request <PRIVATE_REQUEST_JSON> --request-sha <SHA256>
<APPROVED_PYTHON> -B <HELPER_ROOT>/analysis/sid_coll_configsha_handoff.py prepare --request <PRIVATE_REQUEST_JSON> --request-sha <SHA256>
<APPROVED_PYTHON> -B <HELPER_ROOT>/analysis/sid_coll_configsha_handoff.py start --request <PRIVATE_REQUEST_JSON> --request-sha <SHA256>
<APPROVED_PYTHON> -B <HELPER_ROOT>/analysis/sid_coll_configsha_handoff.py ack --request <PRIVATE_REQUEST_JSON> --request-sha <SHA256>
```

`_detach` is internal and requires the prior controller launch intent. Do not
call it directly in production. There is no public tick, retry, release, or
arbitrary owner entry in the transport API.

## Owner protocol and audit artifacts

The global lock is derived as `old_control_dir.parent/configsha-handoff.lock`.
Inventory creates no files and verifies old PID absent plus old driver lock free.
Prepare holds the old and global locks continuously during sealing. Start writes
one exclusive `launch-intent.json` then double forks through the deployed helper.
The owner waits at most ten seconds for the launching controller's global lock
to release. Before ready, it validates the sealed/fresh inventory and holds all
three actual locks: old `driver.lock` opened read-only, global handoff lock, and
exclusive new `driver.lock`. This also blocks restarting the old driver.

The owner writes `driver-pid.json` and `owner-ready.json` with actual PID/PPID,
request SHA, corrected plan SHA and the unchanged deadline. The controller ACK
requires the PID alive, actual Linux `/proc` PPID=1, and `/proc/locks` confirming
all three exclusive flock records belong to that exact PID. A lock held by some
other process is insufficient. It then writes the exclusive `owner-release.json`
with owner PID, ready-byte SHA, request/plan SHA and original deadline. No tick or
publication occurs before this release. If the controller exits beforehand,
the owner waits at most 30 seconds (capped by the original deadline), writes
`launch-aborted.json`, and stops with zero new publications. After release the
owner operates independently of the controller/SSH. An uncertain launch is never
restarted. ACK timeout or transport failure is not permission to call start again.

The owner loop calls the unchanged original `sid_coll_ops.tick` and
`EventWaiter`, with its original own-terminal notification and 900-second
fallback, and original summary writer. There is no monkeypatch. It uses the
same plan load/hash check before each tick and the same terminal/deadline logic
as the original driver. The source driver cannot be called directly while
preholding its new lock, so this metadata wrapper holds the locks and reuses the
actual original functions instead. It writes `summary-*.json` and
`summary-final.json` in the new control directory. Only the original verified
receipt and original hits>=260 logic can trigger original conditional seeds.

Inventory audits all 28 stages, including the 12 train/verify stages. Existing
queue identity must equal the materialized frozen job exactly; a terminal run
requires exactly one claimed entry and matching small run.json, inputs/config,
resources, env, clean scientific commit, ops identity, terminal and receipt.
Manifest inputs are restricted to the frozen stage spec and its deployment JSON.
A successful train42 must differ from the old nine-field expected identity only
at the null config SHA. Verify queued/running/accepted is adopted only by exact
identity. Absent, unused verify may be submitted once by the original publisher.
Uncertain intents, missing run manifests, duplicates, failed/rejected attempts,
unresolved existing dynamic identity, source drift or expired deadline STOP.
No failed train is repaired and no train42 is rerun.

## Local verification evidence and limits

The synthetic tests use frozen controller bytes from scientific commit
`11d67681e0cbde09983f6bf482f55ab8a7191bdb`. The real tick integration adopts
train42 and publishes only the missing verify once. The actual detached-process
fault smoke checks PPID=1, all three real held locks, zero publication before
release despite controller exit, independent publications after release, and
repeat-start STOP; only its temporary synthetic scheduler/prepare are fakes.
Linux `/proc` ACK parsing is additionally tested with exact lock inode/PID data;
the development host does not provide Linux `/proc`, so full production ACK
needs the first server metadata check by the main owner. No server or actual
scientific job was run while implementing this tool.

The original fixed controller's 21 existing synthetic tests passed without any
source modifications. The owner's offline proposal SHA
`791d0c00a906513159cec00af924988c3f596853221790d349e19424c185ad4a`
was independently compared against all twelve frozen Git config blobs and the
old plan: only twelve expected pins and control_dir differ. This offline check
does not replace production spec/receipt/manifest/owner inventory.

Final local regression: 26 new handoff tests passed; the complete tools worktree
synthetic suite passed 67 tests. Separately, 21 tests from the original fixed
controller passed from exact Git bytes in a temporary directory. Fresh RED
was observed for missing implementation, transport/source identity, lock
ownership and sealing, release-before-publication, approval and detach intent.
No scientific implementation file was modified.
