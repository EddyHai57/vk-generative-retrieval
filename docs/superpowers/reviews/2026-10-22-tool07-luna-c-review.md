# Tool 7 resource probe / centralized config — independent C review

Review date: 2026-10-22
Requested reviewer dispatch: `gpt-6-luna`, high reasoning. The execution runtime did not independently return a verifiable model identity; this report records the dispatch instruction, not an independently attested runtime identity.

## Scope and boundaries

Read-only review of `src/vkgr/throughput_probe.py`, `analysis/throughput_probe.py`, `src/vkgr/throughput_config.py`, the four `configs/throughput/*.yaml` files, `tests/test_throughput_probe.py`, the throughput spec, decision records `11249b6` and `2a795cb`, and the requested `cache/throughput-20261022/07-{red,green}.log` receipts. No server, data, scientific training, or test command was run for this review. The red/green logs were inspected as existing evidence.

## Finding — cgroup ancestor limits can be missed (C blocker for trusting recommendations on nested cgroups)

`src/vkgr/throughput_probe.py:26-36` reads only `/sys/fs/cgroup/cpu.max` and `/sys/fs/cgroup/memory.max`. It does not resolve the current process's cgroup path (for example through `/proc/self/cgroup`) or walk parent limits. On a cgroup v2 host where the process sits in a nested slice and the mounted root reports `max` or a larger quota, the probe can report host/root capacity instead of the effective limit. `recommend()` then uses that record to choose workers and a memory cap (`:10-20`). This can over-recommend parallel workers and memory. The same root-only assumption appears in the runtime memory gate at `src/vkgr/throughput_config.py:86-91`, so a configured guard can miss the effective nested memory ceiling.

Disposition: fix ancestor-aware cgroup resolution and add a small fixture covering a constrained nested cgroup before relying on tool 7 for a real instance. This finding blocks confidence in probe recommendations/guard enforcement on nested cgroups; it does not invalidate the current static A40 YAML, which is explicitly a local plan and is not auto-applied.

## A40 configuration and resource interpretation

The user-provided instance evidence says 64 host CPUs, an effective cgroup quota of 61.2 CPU cores, 513 GB planning RAM, 48 GB VRAM, and a 500 GB instance disk. Keep host CPU count and cgroup quota as separate facts: 64 is the host listing; 61.2 is the effective quota supplied for planning. Do not describe the A40 host as locally measured. The 500 GB disk is the instance disk; the local workstation disk below is unrelated.

The requested A40 proposal is present in `configs/throughput/a40-20261022.yaml`: assembler/BHKM/eval/week workers 8, prefetch 2, memory cap 256,000,000,000 bytes, disk cap 400,000,000,000 bytes, decode workers 1, and `bhkm_external: false`. It is a recommendation/configuration only; `resource_manifest()` returns `recommendation_applied: false` and `science_config_changed: false` (`analysis/throughput_probe.py:48-52`). This is not a scientific configuration change and must not be described as an automatic application.

The configured 256 GB cap is below 60% of the stated 513 GB planning RAM (307.8 GB decimal), and the 400 GB disk cap corresponds to 80% of a 500 GB decimal disk. Runtime hard lines remain the registered 60% RSS and 80% disk limits in `throughput_config.py:84-95`; configured caps can tighten them. These arithmetic checks do not prove instance capacity or an observed peak.

`bhkm_external: false` leaves the original dense BHKM route selected. `bhkm_rows: 16384`, `bhkm_workers: 8`, and `temp_dir` do not by themselves establish external spilling while that flag remains false. If external mode is enabled later, preserve all temporary runs as required; the spec's roughly 41 GiB figure is a per-round estimate and does not cap multi-round/node/replay retention. Full dense peak RSS and full temporary-disk demand remain UNKNOWN pending the authorized new-instance preflight.

## Local resource observation (this review environment)

- OS: macOS 27.0.1 arm64.
- `os.cpu_count()`: 8. This is the local host-visible count; cgroup quota: UNKNOWN (Linux cgroup interface unavailable on Darwin).
- RAM total: UNKNOWN. `sysctl` was denied in this execution environment; `/proc/meminfo` and Linux cgroup files are absent.
- GPU/driver/VRAM: UNKNOWN (`nvidia-smi` unavailable).
- Workspace filesystem: 494,384,795,648 bytes total; 330,137,600,000 bytes free. This is local workspace storage, not the A40 instance disk.
- Context-inclusive GPU peak: UNKNOWN; no GPU measurement was performed.

## Configuration identity

`configs/throughput/a40-20261022.yaml` raw-byte SHA-256: `99c9c8be0600ab62c17ffd51881a872fa7ae5cacbf36adc8e43a706d925f214e`.

Normalized `Engineering.identity()` SHA-256 (canonical sorted compact JSON of the 14 dataclass fields): `6c1fa5d8a51fa8ab1a5d3d101684337d66fed3487292dc2de7098e6d32bb6990`.

These hashes are separate identities: raw YAML includes comments/formatting; normalized identity covers configuration values. Neither hash is a scientific input identity.

## Existing test/CLI evidence

- `cache/throughput-20261022/07-red.log`: initial unittest import failed (`ModuleNotFoundError: No module named 'vkgr.throughput_probe'`) because the source package was not on the import path. This is an invocation/setup failure, not a behavioral test failure.
- `cache/throughput-20261022/07-green.log`: 2 tests passed. It records `test_minimum_recommended_and_no_science_fields` and `test_cli_dry_and_invalid`; the CLI dry command returned success and invalid subcommand returned nonzero through the test assertions.
- The green receipt is the evidence inspected; this review did not rerun tests. The test covers 12-core/64-GB and 24-core/128-GB recommendation cases, absence of science fields, and CLI dry/invalid paths. It does not cover nested cgroup limits, unknown-resource serialization, actual GPU discovery, or live hard-guard behavior.

## Spec and quality notes

The reviewed behavior aligns with `docs/superpowers/specs/2026-10-22-throughput-options.md`: no scientific settings are changed, engineering configuration/hash is separate, GPU parallelism requires a new measured peak, and 60% RSS / 80% disk remain hard limits. The A40 YAML's `decode_workers: 1`, disabled external BHKM, 256 GB memory cap, 400 GB disk cap, and `prefetch: 2` match the requested conservative local plan. Worker count 8 is a planning value, not a measured optimum.

The test receipts provide the requested two tests and CLI dry/invalid evidence, with the first failed import and corrected run kept visible. There is no evidence here of a local probe CLI run against a GPU host, a new-instance preflight, measured peak RSS/VRAM, dense BHKM peak, or accumulated external temporary-disk usage. Those remain UNKNOWN and must stay out of claims about measured capacity or throughput.
