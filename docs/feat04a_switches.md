# FEAT-04a switches (P line, new files only)

Entry point: `python -m vkgr.retrieval_feat04a_run <command>`. Every switch below is explicit; the default behaviour of every older entry point is byte-identical (31 sha256 pins, `tests/test_feat04a_untouched.py`).

| Switch | Values | Meaning |
|---|---|---|
| `feature` (config) | `f4_recency_pool` (A, real) / `f4_hist_author_content` (B, real) / `f4_control` (equivalence fixture, synthetic only) | One variable per run on top of the anchor F4-T010 |
| `temperature` (config) | `0.1` real; `0.1` or `0.15` synthetic | The frozen B1 validator pins 0.15, so it is projected back before the frozen validator runs |
| `learning_rate` (config) | `0.001` (float) real | Pinned in code so a run changes one variable only |
| `seed` (config) | 42 / 2024 / 2025 | Same three seeds as the anchor |
| `--device` (synthetic-*) | `cpu` (default) / `cuda` | `synthetic-verify` must use the device of `synthetic-train` (it is part of the binding) |

Commands: `synthetic-train`, `synthetic-verify`, `train`, `verify`, `preflight`.

`preflight` is read-only and runs on CPU: it loads the pinned real inputs, builds the variant and the same-seed anchor control, and checks shapes, finiteness, A's start equal to the anchor, the untouched item tower, and that flipping `target` / `target_position` / `oov` of every validation query leaves the user vectors bit-equal. It writes nothing and never starts training. The logic is the pure function `preflight_check(data, config, spec)`; the CLI only reads the pinned files and calls it.

Configs: `configs/feat04a/` (A: three seeds; B is added with the B gate). Freeze: `scripts/feat04a_freeze.py --variant A|B` (one gate per variant). Memory: `scripts/feat04a_vram_smoke.py`. Device parity with the P-HP-01 H1 runner: `scripts/feat04a_cuda_parity.py --device cuda`.

Inputs spec: the `p-feat02-inputs-v1` schema; a pin that a variant does not use must be null (A: all four sidecar pins null; B: ctx pins null, attr pins set).
