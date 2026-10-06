# P1 validation evidence

密级: 公开

本文件只记录实际执行证据；task=P1，最终由主协调Agent fresh验收。

## 执行合同

仅新增授权package/config/lock/tests/fixture/本文件，定点修改smoke config、README与STATE/handoff/decision/issue；不修改AGENTS、tasks或数据合同。输入为人工synthetic fixture和smoke config。输出为唯一新建outputs/p1-<run-id>目录；测试只使用独立temp目录。停止条件为基点变化、事实源冲突、scope越界或manifest无法追溯。

验收命令：`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`；smoke、verify见README。

## 环境准备

Fresh核对Python 3.11.17、uv 0.12.23；系统Python 3.9.6只启动uv。branch switch第一次exit 128（共享Git metadata权限），获准escalation后exit 0。未把环境失败算作TDD RED。

## TDD ledger

### C01 RED valid config interface

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 1

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... FAIL
FAIL: test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke)
AssertionError: P1 load_config interface is missing
Ran 1 test in 0.000s
FAILED (failures=1)
```

### C01 GREEN valid config

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 0

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
Ran 1 test in 0.285s
OK
```

### C02 RED exact field schema

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 1

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ...
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section=None, mutation='extra')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section=None, mutation='missing')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='data', mutation='extra')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='data', mutation='missing')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='resources', mutation='extra')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='resources', mutation='missing')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='run', mutation='extra')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='run', mutation='missing')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='claims', mutation='extra')
AssertionError: ValueError not raised
FAIL: test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) (section='claims', mutation='missing')
AssertionError: ValueError not raised
Ran 2 tests in 0.015s
FAILED (failures=10)
```

### C02 GREEN exact field schema

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 0

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
Ran 2 tests in 0.014s
OK
```

### C03 RED duplicate keys

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 1

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... FAIL
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
FAIL: test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys)
AssertionError: ValueError not raised
Ran 3 tests in 0.015s
FAILED (failures=1)
```

### C03 GREEN duplicate keys

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 0

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
Ran 3 tests in 0.016s
OK
```

### C04 RED scalar types and boundary values

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 1

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ...
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=(), key='schema_version', bad=True)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=(), key='schema_version', bad=2)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=(), key='task_id', bad='P2')
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('resources',), key='network_required', bad=0)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('resources',), key='network_required', bad=True)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('resources',), key='gpu_required', bad=True)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('resources',), key='execution', bad='cuda')
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('run',), key='seed', bad=True)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('run',), key='seed', bad=-1)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('run',), key='seed', bad=4294967296)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('run',), key='seed', bad='42')
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('run',), key='seed', bad=42.0)
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('run',), key='model', bad='tiger')
AssertionError: ValueError not raised
FAIL: test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) (trail=('claims',), key='allowed', bad='model_quality')
AssertionError: ValueError not raised
Ran 4 tests in 0.034s
FAILED (failures=14)
```

### C04 GREEN scalar types and boundaries

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 0

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
Ran 4 tests in 0.180s
OK
```

### C05 RED approved relative fixture path

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 1

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ...
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path='/tmp/fixture.json')
AssertionError: ValueError not raised
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path='../fixture.json')
AssertionError: ValueError not raised
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path='https://example.test/x')
AssertionError: ValueError not raised
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path='tests/fixtures/../p1_synthetic.json')
AssertionError: ValueError not raised
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path='tests//fixtures/p1_synthetic.json')
AssertionError: ValueError not raised
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path='tests/fixtures/other.json')
AssertionError: ValueError not raised
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path=None)
AssertionError: ValueError not raised
FAIL: test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) (path=42)
AssertionError: ValueError not raised
Ran 5 tests in 0.046s
FAILED (failures=8)
```

### C05 GREEN approved relative fixture path

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 0

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
Ran 5 tests in 0.050s
OK
```

### R01 RED fixture interface

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 1

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... FAIL
FAIL: test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture)
AssertionError: P1 synthetic runtime interface is missing
Ran 1 test in 0.001s
FAILED (failures=1)
```

### R01 GREEN fixture interface

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 0

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
Ran 1 test in 0.000s
OK
```

### R02 RED artificial fixture validation

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 1

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ...
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case=None)
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={})
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={'schema_version': True, 'origin': 'hand_authored_synthetic_not_vk_schema', 'values': [1]})
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={'schema_version': 1, 'origin': 'vk', 'values': [1]})
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={'schema_version': 1, 'origin': 'hand_authored_synthetic_not_vk_schema', 'values': []})
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={'schema_version': 1, 'origin': 'hand_authored_synthetic_not_vk_schema', 'values': [True]})
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={'schema_version': 1, 'origin': 'hand_authored_synthetic_not_vk_schema', 'values': [-1]})
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={'schema_version': 1, 'origin': 'hand_authored_synthetic_not_vk_schema', 'values': [1001]})
AssertionError: ValueError not raised
FAIL: test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) (case={'schema_version': 1, 'origin': 'hand_authored_synthetic_not_vk_schema', 'values': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
AssertionError: ValueError not raised
Ran 2 tests in 0.002s
FAILED (failures=9)
```

### R02 GREEN fixture validation

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 0

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
Ran 2 tests in 0.002s
OK
```

### R03 RED seed participates in actual calculation

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 1

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... FAIL
FAIL: test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation)
AssertionError: False is not true : seeded computation is missing
Ran 3 tests in 0.003s
FAILED (failures=1)
```

R03手算期望修正：3×2+7×1+11×5+19×4=144；先前229是测试编写算术错误，未写生产实现，修正后重新RED。

### R03 RED corrected hand calculated oracle

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 1

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... FAIL
FAIL: test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation)
AssertionError: False is not true : seeded computation is missing
Ran 3 tests in 0.003s
FAILED (failures=1)
```

### R03 GREEN seeded computation

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 0

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
Ran 3 tests in 0.002s
OK
```

### M01 RED manifest field contract

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... FAIL
FAIL: test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source)
AssertionError: versioned manifest interface is missing
Ran 1 test in 0.001s
FAILED (failures=1)
```

### M01 GREEN manifest field contract

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
Ran 1 test in 0.095s
OK
```

### M02 RED manifest write and reread

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... FAIL
FAIL: test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest)
AssertionError: False is not true : manifest writer missing
Ran 2 tests in 0.078s
FAILED (failures=1)
```

### M02 GREEN write and reread

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
Ran 2 tests in 0.103s
OK
```

### M03 RED strict manifest schema

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ...
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='schema_version', bad=True)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='schema_version', bad=2)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='task_id', bad='P2')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='status', bad='failed')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='seed', bad=True)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='seed', bad=-1)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='git_commit', bad='pending')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='git_dirty', bad=1)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='source_fingerprint', bad='unknown')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='config_hash', bad='abc')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='python_version', bad='3.9.6')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='dependency_versions', bad={'PyYAML': '0'})
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='started_at_utc', bad='yesterday')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='started_at_utc', bad='2026-01-01T00:00:00+10:00')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='elapsed_seconds', bad=True)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='elapsed_seconds', bad=-1)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='elapsed_seconds', bad=nan)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (key='claim_boundary', bad='model evaluated')
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (structure=<class 'list'>)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (structure=<class 'dict'>)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (structure=<class 'dict'>)
AssertionError: ValueError not raised
FAIL: test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) (structure=<class 'dict'>)
AssertionError: ValueError not raised
Ran 3 tests in 0.136s
FAILED (failures=22)
```

### M03 GREEN strict manifest schema

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
Ran 3 tests in 0.134s
OK
```

### M04 RED safe repo relative paths

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ...
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='config_path', bad='/tmp/x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='config_path', bad='../x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='config_path', bad='outputs/../x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='config_path', bad='https://example.test/x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='config_path', bad='./configs/smoke.yaml')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='config_path', bad='a//b')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='config_path', bad=None)
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='fixture_path', bad='/tmp/x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='fixture_path', bad='../x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='fixture_path', bad='outputs/../x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='fixture_path', bad='https://example.test/x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='fixture_path', bad='./configs/smoke.yaml')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='fixture_path', bad='a//b')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='fixture_path', bad=None)
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='result_path', bad='/tmp/x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='result_path', bad='../x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='result_path', bad='outputs/../x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='result_path', bad='https://example.test/x')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='result_path', bad='./configs/smoke.yaml')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='result_path', bad='a//b')
AssertionError: ValueError not raised
FAIL: test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) (field='result_path', bad=None)
AssertionError: ValueError not raised
Ran 4 tests in 0.159s
FAILED (failures=21)
```

### M04 GREEN repo relative paths

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
Ran 4 tests in 0.160s
OK
```

### M05 RED hashes and missing files

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ...
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... FAIL
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
FAIL: test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) (key='config_hash')
AssertionError: ValueError not raised
FAIL: test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) (key='fixture_hash')
AssertionError: ValueError not raised
FAIL: test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) (key='result_hash')
AssertionError: ValueError not raised
FAIL: test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) (key='source_fingerprint')
AssertionError: ValueError not raised
FAIL: test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs)
AssertionError: ValueError not raised
Ran 5 tests in 0.185s
FAILED (failures=5)
```

### M05 GREEN hashes and missing files

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
Ran 5 tests in 0.202s
OK
```

### M06 RED semantic input and result consistency

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ...
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... FAIL
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
FAIL: test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) (key='seed')
AssertionError: ValueError not raised
FAIL: test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) (key='effective_config')
AssertionError: ValueError not raised
FAIL: test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) (key='fixture_path')
AssertionError: ValueError not raised
FAIL: test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency)
AssertionError: ValueError not raised
Ran 6 tests in 0.216s
FAILED (failures=4)
```

### M06 GREEN semantic consistency

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
Ran 6 tests in 0.220s
OK
```

### R04 RED single-call smoke orchestration

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 1

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result) ... FAIL
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
FAIL: test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result)
AssertionError: False is not true : single-call smoke orchestration missing
Ran 4 tests in 0.087s
FAILED (failures=1)
```

### R04 GREEN smoke orchestration

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 0

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
Ran 4 tests in 0.164s
OK
```

### R05 RED refuse output overwrite

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 1

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_refuses_existing_output_without_changing_it (test_runtime.RuntimeTests.test_refuses_existing_output_without_changing_it) ... FAIL
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
FAIL: test_refuses_existing_output_without_changing_it (test_runtime.RuntimeTests.test_refuses_existing_output_without_changing_it)
AssertionError: (<class 'ValueError'>, <class 'FileExistsError'>) not raised
Ran 5 tests in 0.178s
FAILED (failures=1)
```

### R05 GREEN refuse output overwrite

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_runtime.py -v`

Exit code: 0

```text
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_refuses_existing_output_without_changing_it (test_runtime.RuntimeTests.test_refuses_existing_output_without_changing_it) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
Ran 5 tests in 0.145s
OK
```

### L01 RED smoke and verify CLI

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_cli.py -v`

Exit code: 1

```text
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... FAIL
FAIL: test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli)
AssertionError: 1 != 0 : /Users/eddy/.codex/worktrees/6ec7/vk-generative-retrieval/.venv/bin/python: No module named vkgr.__main__; 'vkgr' is a package and cannot be directly executed
Ran 1 test in 0.036s
FAILED (failures=1)
```

### L01 GREEN smoke and verify CLI

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_cli.py -v`

Exit code: 0

```text
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
Ran 1 test in 0.453s
OK
```

### L02 RED logging to stderr

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_cli.py -v`

Exit code: 1

```text
test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment) ... FAIL
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
FAIL: test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment)
AssertionError: 'INFO' not found in ''
Ran 2 tests in 0.683s
FAILED (failures=1)
```

### L02 GREEN stderr logging

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_cli.py -v`

Exit code: 0

```text
test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment) ... ok
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
Ran 2 tests in 0.688s
OK
```

### L03 RED safe failure diagnostic and exit code

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_cli.py -v`

Exit code: 1

```text
test_invalid_input_exits_two_with_safe_diagnostic (test_cli.CliTests.test_invalid_input_exits_two_with_safe_diagnostic) ...
test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment) ... ok
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
FAIL: test_invalid_input_exits_two_with_safe_diagnostic (test_cli.CliTests.test_invalid_input_exits_two_with_safe_diagnostic) (command='smoke')
AssertionError: 1 != 2 : Traceback (most recent call last):
FAIL: test_invalid_input_exits_two_with_safe_diagnostic (test_cli.CliTests.test_invalid_input_exits_two_with_safe_diagnostic) (command='verify')
AssertionError: 1 != 2 : Traceback (most recent call last):
Ran 3 tests in 1.106s
FAILED (failures=2)
```

### L03 GREEN safe failure diagnostic

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_cli.py -v`

Exit code: 0

```text
test_invalid_input_exits_two_with_safe_diagnostic (test_cli.CliTests.test_invalid_input_exits_two_with_safe_diagnostic) ... ok
test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment) ... ok
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
Ran 3 tests in 1.104s
OK
```

### B01 RED constrain outputs before side effects

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_boundaries.py -v`

Exit code: 1

```text
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... FAIL
FAIL: test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing)
AssertionError: True is not false
Ran 1 test in 0.108s
FAILED (failures=1)
```

### B01 GREEN constrain output paths

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_boundaries.py -v`

Exit code: 0

```text
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
Ran 1 test in 0.106s
OK
```

### B02 RED Git failure must not leave successful-looking output

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_boundaries.py -v`

Exit code: 1

```text
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... FAIL
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
FAIL: test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created)
AssertionError: True is not false
Ran 2 tests in 0.067s
FAILED (failures=1)
```

### B02 GREEN Git preflight before output

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_boundaries.py -v`

Exit code: 0

```text
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... ok
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
Ran 2 tests in 0.078s
OK
```

### M07 RED bind manifest to its own result

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory) ... FAIL
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
FAIL: test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory)
AssertionError: ValueError not raised
Ran 7 tests in 0.253s
FAILED (failures=1)
```

### M07 GREEN bind result to run directory

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
Ran 7 tests in 0.380s
OK
```

### C06 RED normalize malformed YAML failure

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 1

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) ...
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
ERROR: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='run: [')
ERROR: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='!!python/object:os.system {}')
ERROR: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='? [unhashable]\n: value')
ERROR: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='&cycle {data: *cycle}')
Ran 6 tests in 0.049s
FAILED (errors=4)
```

C06初次运行是errors而非明确assertion failure；修改测试捕获实际异常并断言公共ValueError合同，重新fresh RED，未把语法错误当有效RED。

### C06 RED explicit assertion on malformed input failure contract

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 1

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) ...
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
FAIL: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='run: [')
AssertionError: ParserError('while parsing a flow node', <yaml.error.Mark object at 0x10a742010>, "expected the node content, but found '<stream end>'", <yaml.error.Mark object at 0x10a742010>) is not an instance of <class 'ValueError'> : invalid config must fail with ValueError
FAIL: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='!!python/object:os.system {}')
AssertionError: ConstructorError(None, None, "could not determine a constructor for the tag 'tag:yaml.org,2002:python/object:os.system'", <yaml.error.Mark object at 0x10a743f90>) is not an instance of <class 'ValueError'> : invalid config must fail with ValueError
FAIL: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='? [unhashable]\n: value')
AssertionError: TypeError("unhashable type: 'list'") is not an instance of <class 'ValueError'> : invalid config must fail with ValueError
FAIL: test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) (raw='&cycle {data: *cycle}')
AssertionError: ConstructorError(None, None, 'found unconstructable recursive node', <yaml.error.Mark object at 0x10a70fcd0>) is not an instance of <class 'ValueError'> : invalid config must fail with ValueError
Ran 6 tests in 0.047s
FAILED (failures=4)
```

### C06 GREEN malformed YAML public failure contract

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_config.py -v`

Exit code: 0

```text
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
Ran 6 tests in 0.043s
OK
```

### J01 RED reject ambiguous duplicate JSON keys

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_*.py -v`

Exit code: 1

```text
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... ok
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
test_invalid_input_exits_two_with_safe_diagnostic (test_cli.CliTests.test_invalid_input_exits_two_with_safe_diagnostic) ... ok
test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment) ... ok
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_duplicate_json_keys (test_manifest.ManifestTests.test_rejects_duplicate_json_keys) ... FAIL
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_refuses_existing_output_without_changing_it (test_runtime.RuntimeTests.test_refuses_existing_output_without_changing_it) ... ok
test_rejects_duplicate_fixture_keys (test_runtime.RuntimeTests.test_rejects_duplicate_fixture_keys) ... FAIL
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
FAIL: test_rejects_duplicate_json_keys (test_manifest.ManifestTests.test_rejects_duplicate_json_keys)
AssertionError: ValueError not raised
FAIL: test_rejects_duplicate_fixture_keys (test_runtime.RuntimeTests.test_rejects_duplicate_fixture_keys)
AssertionError: ValueError not raised
Ran 25 tests in 1.602s
FAILED (failures=2)
```

### J01 GREEN unambiguous JSON inputs

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_*.py -v`

Exit code: 0

```text
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... ok
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
test_invalid_input_exits_two_with_safe_diagnostic (test_cli.CliTests.test_invalid_input_exits_two_with_safe_diagnostic) ... ok
test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment) ... ok
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_duplicate_json_keys (test_manifest.ManifestTests.test_rejects_duplicate_json_keys) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_refuses_existing_output_without_changing_it (test_runtime.RuntimeTests.test_refuses_existing_output_without_changing_it) ... ok
test_rejects_duplicate_fixture_keys (test_runtime.RuntimeTests.test_rejects_duplicate_fixture_keys) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
Ran 25 tests in 1.585s
OK
```

### M08 RED recorded commit must exist locally

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 1

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_duplicate_json_keys (test_manifest.ManifestTests.test_rejects_duplicate_json_keys) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory) ... ok
test_rejects_nonexistent_git_commit (test_manifest.ManifestTests.test_rejects_nonexistent_git_commit) ... FAIL
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
FAIL: test_rejects_nonexistent_git_commit (test_manifest.ManifestTests.test_rejects_nonexistent_git_commit)
AssertionError: ValueError not raised
Ran 9 tests in 0.300s
FAILED (failures=1)
```

### M08 GREEN local commit existence

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_manifest.py -v`

Exit code: 0

```text
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_duplicate_json_keys (test_manifest.ManifestTests.test_rejects_duplicate_json_keys) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory) ... ok
test_rejects_nonexistent_git_commit (test_manifest.ManifestTests.test_rejects_nonexistent_git_commit) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
Ran 9 tests in 0.323s
OK
```

新增已有边界的characterization checks：symlink拒绝、socket阻断、仅local Git subprocess、global RNG不变、CLI重复run保留字节、argument exit/main接口。它们预计直接GREEN，无需伪造RED或更改已满足的生产行为。

### P1 characterization and fresh full regression

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_*.py -v`

Exit code: 0

```text
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... ok
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses (test_boundaries.BoundaryTests.test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses) ... ok
test_symlink_input_and_output_are_rejected_without_writing (test_boundaries.BoundaryTests.test_symlink_input_and_output_are_rejected_without_writing) ... ok
test_argument_errors_are_nonzero_and_main_returns_int (test_cli.CliTests.test_argument_errors_are_nonzero_and_main_returns_int) ... INFO P1 synthetic smoke completed
test_cli_refuses_existing_run_and_keeps_original_bytes (test_cli.CliTests.test_cli_refuses_existing_run_and_keeps_original_bytes) ... ok
test_invalid_input_exits_two_with_safe_diagnostic (test_cli.CliTests.test_invalid_input_exits_two_with_safe_diagnostic) ... ok
test_logs_success_to_stderr_without_personal_environment (test_cli.CliTests.test_logs_success_to_stderr_without_personal_environment) ... ok
test_smoke_and_verify_cli (test_cli.CliTests.test_smoke_and_verify_cli) ... ok
test_loads_approved_smoke (test_config.ConfigTests.test_loads_approved_smoke) ... ok
test_malformed_yaml_is_explicit_value_error (test_config.ConfigTests.test_malformed_yaml_is_explicit_value_error) ... ok
test_rejects_duplicate_yaml_keys (test_config.ConfigTests.test_rejects_duplicate_yaml_keys) ... ok
test_rejects_unapproved_fixture_paths (test_config.ConfigTests.test_rejects_unapproved_fixture_paths) ... ok
test_rejects_unknown_or_missing_fields (test_config.ConfigTests.test_rejects_unknown_or_missing_fields) ... ok
test_rejects_wrong_types_and_boundary_values (test_config.ConfigTests.test_rejects_wrong_types_and_boundary_values) ... ok
test_manifest_tracks_versioned_inputs_and_actual_source (test_manifest.ManifestTests.test_manifest_tracks_versioned_inputs_and_actual_source) ... ok
test_rejects_config_seed_and_result_inconsistency (test_manifest.ManifestTests.test_rejects_config_seed_and_result_inconsistency) ... ok
test_rejects_duplicate_json_keys (test_manifest.ManifestTests.test_rejects_duplicate_json_keys) ... ok
test_rejects_hash_tamper_and_missing_inputs (test_manifest.ManifestTests.test_rejects_hash_tamper_and_missing_inputs) ... ok
test_rejects_invalid_manifest_structure_and_types (test_manifest.ManifestTests.test_rejects_invalid_manifest_structure_and_types) ... ok
test_rejects_manifest_result_from_another_run_directory (test_manifest.ManifestTests.test_rejects_manifest_result_from_another_run_directory) ... ok
test_rejects_nonexistent_git_commit (test_manifest.ManifestTests.test_rejects_nonexistent_git_commit) ... ok
test_rejects_unsafe_manifest_paths (test_manifest.ManifestTests.test_rejects_unsafe_manifest_paths) ... ok
test_writes_and_reloads_manifest (test_manifest.ManifestTests.test_writes_and_reloads_manifest) ... ok
test_loads_hand_authored_fixture (test_runtime.RuntimeTests.test_loads_hand_authored_fixture) ... ok
test_refuses_existing_output_without_changing_it (test_runtime.RuntimeTests.test_refuses_existing_output_without_changing_it) ... ok
test_rejects_duplicate_fixture_keys (test_runtime.RuntimeTests.test_rejects_duplicate_fixture_keys) ... ok
test_rejects_non_synthetic_or_invalid_fixture (test_runtime.RuntimeTests.test_rejects_non_synthetic_or_invalid_fixture) ... ok
test_run_smoke_creates_verifiable_reproducible_result (test_runtime.RuntimeTests.test_run_smoke_creates_verifiable_reproducible_result) ... ok
test_seed_drives_reproducible_computation (test_runtime.RuntimeTests.test_seed_drives_reproducible_computation) ... ok
Ran 30 tests in 2.686s
OK
```

### B03 RED approved fixture name must not hide external symlink

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_boundaries.py -v`

Exit code: 0

```text
test_fixture_symlink_is_rejected_before_reading_or_git (test_boundaries.BoundaryTests.test_fixture_symlink_is_rejected_before_reading_or_git) ... ok
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... ok
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses (test_boundaries.BoundaryTests.test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses) ... ok
test_symlink_input_and_output_are_rejected_without_writing (test_boundaries.BoundaryTests.test_symlink_input_and_output_are_rejected_without_writing) ... ok
Ran 5 tests in 0.153s
OK
```

B03第一次意外GREEN：temp路径位于macOS /var别名，safe_path因ROOT未resolve提前拒绝，未触达fixture检查。测试ROOT改为canonical后重新RED；不将误通过视为目标证据。

### B03 RED canonical temp root actually exercises fixture symlink

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_boundaries.py -v`

Exit code: 1

```text
test_fixture_symlink_is_rejected_before_reading_or_git (test_boundaries.BoundaryTests.test_fixture_symlink_is_rejected_before_reading_or_git) ... FAIL
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... ok
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses (test_boundaries.BoundaryTests.test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses) ... ok
test_symlink_input_and_output_are_rejected_without_writing (test_boundaries.BoundaryTests.test_symlink_input_and_output_are_rejected_without_writing) ... ok
FAIL: test_fixture_symlink_is_rejected_before_reading_or_git (test_boundaries.BoundaryTests.test_fixture_symlink_is_rejected_before_reading_or_git)
AssertionError: RuntimeError('unsafe fixture reached Git') is not an instance of <class 'ValueError'> : fixture symlink must be rejected at path boundary
Ran 5 tests in 0.155s
FAILED (failures=1)
```

### B03 GREEN fixture path confinement

`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_boundaries.py -v`

Exit code: 0

```text
test_fixture_symlink_is_rejected_before_reading_or_git (test_boundaries.BoundaryTests.test_fixture_symlink_is_rejected_before_reading_or_git) ... ok
test_git_failure_stops_before_output_is_created (test_boundaries.BoundaryTests.test_git_failure_stops_before_output_is_created) ... ok
test_rejects_output_outside_p1_area_before_writing (test_boundaries.BoundaryTests.test_rejects_output_outside_p1_area_before_writing) ... ok
test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses (test_boundaries.BoundaryTests.test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses) ... ok
test_symlink_input_and_output_are_rejected_without_writing (test_boundaries.BoundaryTests.test_symlink_input_and_output_are_rejected_without_writing) ... ok
Ran 5 tests in 0.137s
OK
```

## 环境准备结果与依赖审计

首次uv默认cache失败exit2；改独立tmp cache后默认sandbox PyPI DNS失败exit2。用户已授权首次必要依赖联网安装，escalation执行uv lock/sync成功exit0：Resolved 2 packages（virtual项目自身+PyYAML），Installed 1 package pyyaml==6.0.2。未手写uv.lock，未再次安装Python/uv。

脱敏安装命令形态（Python路径为用户批准的明确绝对路径，未复制到manifest）：`UV_CACHE_DIR=/private/tmp/vkgr-p1-uv-6ec7 python3 -m uv lock --python <approved-python3.11>`，随后同cache执行`python3 -m uv sync --locked --python <approved-python3.11>`。运行与测试全部用当前`.venv/bin/python -B`，没有调用联网安装器。

本轮fresh importlib.metadata/platform复核：python=3.11.17，PyYAML=6.0.2，License=MIT、OSI MIT classifier。官方源https://pypi.org/project/PyYAML/6.0.2/ 与lock内PyPI wheel/sdist SHA-256一致来源；仅作为依赖调用，未复制第三方代码。

## 实现自检与review gate

作者独立读回config/runtime/manifest/CLI及Git diff；检查19字段、重复keys、bool/int、output先验约束、local Git失败、source fingerprint与dirty区分、无env采集/网络/GPU依赖。发现fixture symlink遗漏后按B03 TDD修复。本自检不冒充fresh-context review；按用户安排，主协调Astra负责最终fresh验收，当前不得标P1 Complete。

已知限制：输入与代码须保留匹配；verify以当前source fingerprint核对，不重建历史源码；hash不认证历史dirty、时间或作者。P1不承诺并发输入修改/symlink竞态或I/O transaction，不测峰值RAM/CPU、不训练、不产出效果指标。socket阻断测试覆盖实际路径，源码检查补充“无网络/GPU调用”证据；不以uv的offline flag作唯一证明。

## Final persistent smoke evidence

Fresh full unittest command `PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`: exit 0, Ran 31 tests, OK.

`PYTHONPATH=src .venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-20261007-6ec7-a` → exit 0; stderr INFO.

`PYTHONPATH=src .venv/bin/python -B -m vkgr verify --manifest outputs/p1-20261007-6ec7-a/run_manifest.json` → exit 0; stderr INFO.

`outputs/p1-20261007-6ec7-a/run_manifest.json`: 1591 bytes, SHA-256 `c56dfadfa390dc297cefedf8f7b8fa1f9168383cb5422788a3e5703cfa848827`.

`outputs/p1-20261007-6ec7-a/result.json`: 226 bytes, SHA-256 `565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28`.

`PYTHONPATH=src .venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-20261007-6ec7-b` → exit 0; stderr INFO.

`PYTHONPATH=src .venv/bin/python -B -m vkgr verify --manifest outputs/p1-20261007-6ec7-b/run_manifest.json` → exit 0; stderr INFO.

`outputs/p1-20261007-6ec7-b/run_manifest.json`: 1590 bytes, SHA-256 `e0ae4c1d91ace3bd3d2e1de3063ba58c35c92e7e099e3d6b59be74af9f8bb3b3`.

`outputs/p1-20261007-6ec7-b/result.json`: 226 bytes, SHA-256 `565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28`.

直接读取manifest确认19字段、schema_version1/P1/success、seed42、local_cpu/no network/no GPU/model none、Python3.11.17/PyYAML6.0.2、git_dirty=true、正确基点commit与纯repo-relative路径。result字节完全相同；除started_at_utc/elapsed_seconds/result_path以外所有字段相同。

config_hash: `f36adf05e53af17c2e18a893f0c766955c0a9912ab3325b6ad7ccc30020d5664`

fixture_hash: `f4cf83b4302ca3a669fe63f1710e8d556fa0e97d18241466bf646f50ed832c90`

result_hash: `565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28`

source_fingerprint: `eecdaf51b3f04da447773d4c959a3d765bc85ac63f9d347faca30c45ccc00f72`

Result为人工整数weighted sum，weights=[2,1,5,4]、weighted_sum=144；不是模型效果指标。两个run manifest可直接复核，无原始fixture内容或env/用户名/hostname/绝对路径。

Inventory首轮one-off检查exit1：validation ledger的11处unittest subtest进度行保留尾空格。逐行定位证实只有日志排版问题；已trim尾空格，不改变命令/exit/失败文本。随后重新执行同样检查；未把这次失败隐藏或记成生产RED。secret-pattern rg无匹配exit1（rg标准无匹配码），不是扫描失败。

## Git / public boundary inspection

One-off inventory/AST/hash/whitespace验证exit0。`git diff --check` exit0。tracked modified与untracked逐个对照授权allowlist，无越界文件；staged为空，HEAD仍为批准基点，branch=codex/p1-runtime，remote=0。

tracked modified: `README.md`, `STATE.md`, `configs/smoke.yaml`, `docs/decision_log.md`, `docs/issue_log.md`, `handoff.md`, `src/vkgr/README.md`, `tests/README.md`.

untracked public candidates: `.python-version`, `docs/p1_validation.md`, `outputs/p1-20261007-6ec7-a/run_manifest.json`, `outputs/p1-20261007-6ec7-b/run_manifest.json`, `pyproject.toml`, `src/vkgr/__init__.py`, `src/vkgr/__main__.py`, `src/vkgr/config.py`, `src/vkgr/manifest.py`, `src/vkgr/runtime.py`, `tests/fixtures/p1_synthetic.json`, `tests/test_boundaries.py`, `tests/test_cli.py`, `tests/test_config.py`, `tests/test_manifest.py`, `tests/test_runtime.py`, `uv.lock`.

当时最大public candidate 73628 bytes（validation ledger）；全部<1MiB，无trailing whitespace。ignored仅当前`.venv/`（regular files合计618981 bytes，环境依赖不是项目大产物）和两份226-byte result.json。无残留test temp目录，无未解释repo文件。tmp辅助cache/evidence helper只在独立临时区域，未复制进Git。

源码AST imports只含标准库和PyYAML；读回代码确认subprocess只调用local Git，不采集env/用户名/hostname，不调用网络/GPU/下载。manifest直接检查无绝对路径与身份字段，config/fixture/result各hash由独立hashlib检查匹配。`git check-ignore`确认.venv及两result被忽略；两小manifest按既有default-deny例外保持untracked可review，未stage/commit。定向secret-pattern rg无匹配。

原main追加fresh只读检查：status --porcelain空输出，branch=main；exit0。STATE/handoff已收口为P1 Implementation ready / awaiting coordinator review；最终review由主协调Astra完成，没有冒充通过。

交接收口后的fresh复核：完整unittest再次Ran31/OK，两个persistent manifest再次INFO verified；最终Git/secret/size/whitespace/staging one-off检查exit0，仍8 tracked modifications+17授权untracked、remote0、HEAD/branch匹配、staged空。STATE/handoff/issue/decision已更新；无源码变化，既有manifest source fingerprint仍通过。

## Coordinator fresh acceptance（2026-10-07）

主协调Agent作为未参与实现的reviewer，独立读取config/runtime/manifest/CLI源码、全部测试、tracked diff及untracked交付文件；核对用户收缩后的P1合同与历史TDD ledger。在批准的P1范围内无阻塞发现。历史RED为执行Agent保存的证据，未重新执行历史中间版本；不将当前GREEN冒充历史RED复现。

Fresh commands（当前worktree根目录执行）：

```bash
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-coordinator-20261007-a
PYTHONPATH=src .venv/bin/python -B -m vkgr verify --manifest outputs/p1-coordinator-20261007-a/run_manifest.json
PYTHONPATH=src .venv/bin/python -B -m vkgr smoke --config configs/smoke.yaml --output-dir outputs/p1-coordinator-20261007-b
PYTHONPATH=src .venv/bin/python -B -m vkgr verify --manifest outputs/p1-coordinator-20261007-b/run_manifest.json
cmp outputs/p1-coordinator-20261007-a/result.json outputs/p1-coordinator-20261007-b/result.json
git diff --check
```

全部exit0。unittest关键输出：Ran 31 tests in 2.952s / OK；两次smoke分别INFO completed，两次verify分别INFO verified；cmp无差异。测试包含socket阻断及只允许local Git subprocess的实际smoke/verify路径，源码只使用标准库和PyYAML，无网络/GPU/下载调用。此次验收未调用安装器或联网获取依赖。

独立one-off脚本（未调用生产hash helper）使用hashlib读取实际文件核验config/fixture/result hash，并按README定义重新计算source fingerprint；核对19字段、Python3.11.17、PyYAML6.0.2、dirty=true、实际Git HEAD、路径为repo-relative、无个人目录信息、两份manifest稳定字段一致。Exit0。

| Artifact | SHA-256 |
|---|---|
| outputs/p1-coordinator-20261007-a/run_manifest.json | a9363eaf538e66121ec36d453aa9021e9b81854639a5b7b84ea8160c6af7668d |
| outputs/p1-coordinator-20261007-b/run_manifest.json | a1f8b3cd253f3bc9bad6de68134b2f3b655daac134bd3f99504ec3edc93ad0d1 |
| 两份result.json（字节一致） | 565b24f176e303ddc2283915500132fcfe070808c85236093789058439115d28 |
| source_fingerprint（独立重算） | eecdaf51b3f04da447773d4c959a3d765bc85ac63f9d347faca30c45ccc00f72 |

Fresh公开边界检查：44个tracked/untracked public candidates均小于1MiB，定向secret模式无匹配；无数据/模型/index等禁止后缀；ignored只含.venv与四份result。所有文件无行末空格；AGENTS/tasks/data_contract/architecture/mvp/scale无diff。源码与source fingerprint未改变。定向扫描不是完整安全审计。

Git：codex/p1-runtime，HEAD=df32a1a16a03d6f40e1a1516d95d513413a49916，remote0。原main clean。协调验收仅新增两份manifest与ignored results、更新STATE/handoff/本证据文档，不修改生产实现；未commit/merge/push，等待Eddy批准。

验收结论：P1满足批准范围，Complete。已知限制保持不变：synthetic scaffold only；哈希仅检查一致性；需保留匹配源码与输入；不保证并发输入变化/I/O transaction；没有模型效果或真实数据结论。P2尚未开始。

## 本地提交前复核（2026-10-07）

用户已批准commit和本地合并。Fresh `PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`：exit0，Ran31 in2.757s，OK。本轮仅追加治理/教学汇报要求，不修改生产代码。运行manifest/results保留原worktree，提交候选明确排除outputs运行目录；文档已保存对应hash与复核方法。
