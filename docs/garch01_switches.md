# GARCH-01 switches: TIGER paper-scale G architecture (new version; default G path unchanged)

This file only explains the code switches. Experiment results and decisions live in the experiment/decision logs of the main repo.

## What is new

| Item | Value |
|---|---|
| Entry point | `python -m vkgr.generative_b1_arch train` / `verify` (the default `generative_b1_run` and the INFRA-02 `generative_b1_batched` entries are byte-identical to before) |
| Config schema | `g-b1-arch-run-v1`, `task_id` `G-ARCH-01`, exact field set, strict types (`bool` is not an `int`), configs in `configs/garch01/` |
| Model | `vkgr.tiger_arch.TigerArchModel`: attention with `inner_dim = heads * head_dim` independent of `d_model` (QKV projects `d_model -> 3 * inner_dim`, the output projects `inner_dim -> d_model`); configurable layers, FF width, dropout; post-LN, ReLU |
| Why a custom module | `nn.MultiheadAttention` requires `d_model % heads == 0` (it raises `AssertionError` for 6 heads at `d_model` 128), so 6 heads x 64 (`inner_dim` 384) cannot be expressed |
| Registered structures | `(d_model, heads, head_dim, enc layers, dec layers, ff, dropout)`: anchor shape `(128, 4, 32, 1, 1, 512, 0.0)` (equivalence/parity only) and paper scale `(128, 6, 64, 4, 4, 1024, 0.1)`; any other combination is rejected before output |
| Registered runs | A1 = paper scale, LR 3e-4; A2 = paper scale, LR 1e-3; both seed 42, `microbatch` 256, everything else as the anchor (`SID_BHKM_B16D3_S42`, `HIST_V2_OBS_L50`, `BUDGET_E100_MIN5_P3`, `INIT_EMB_N002`, beam 100) |
| Binding | the run binds `tiger_arch.py`, `generative_arch.py`, `generative_b1_arch.py` in addition to the INFRA-02 members; `init_report.arch` records the structure and the measured parameter count (verify recomputes it). Default / batched / arch runs do not verify each other |

## Fail-closed (before any output)

- `VKGR_EVAL_WORKERS` > 1: sharded workers rebuild the model with the frozen factory.
- `VKGR_DECODE=batched` with a batch size other than 1 (unset batch size means 32 in the batched module and is rejected here): batch > 1 is not bit-equal on GPU.
- Subcommands other than `train` / `verify` (`beam-diagnostic`), `history_len` other than 50, random-SID control, `beam` other than 100.

## Numerical contract with the stock model

The attention mirrors the operator order of `F.multi_head_attention_forward(need_weights=False)`, and initialisation mirrors `nn.Transformer` (module names, registration order, one template per layer type cloned and then xavier-initialised). At anchor shape this gives, on CPU and verified by the tests:

- identical `state_dict` keys/shapes/order and identical same-seed initialisation (so anchor checkpoints load unchanged);
- train mode: bit-identical logits, loss and every gradient; five frozen training steps give the same model digest after every step;
- eval with the stock fast path disabled: bit-identical logits;
- eval with the stock fast path enabled (what the anchor evaluation really runs): not bit-identical. The fused stock kernel computes the same maths in another fp32 order and cannot express independent head dimensions. The only tolerance is `max |dlogit| <= 1e-5`; measured 1.25e-6 (1+1 layers) and 1.64e-6 (2+2 layers).
- batched beam decode with batch size 1 equals the serial decode bit for bit on CPU, including shapes the stock module cannot build (3 heads x 8 on `d_model` 32, 6 x 64 on 128). GPU equality is checked by the preflight script, not assumed.

One property inherited from `nn.Transformer`: biases and LayerNorm parameters of all layers of one type start identical (the layers are clones of one template); only matrices are re-initialised independently.

## Parameter count

Measured (sum of `numel`, vocabulary 80 = 2 + 16 + 16 + 16 + 30, 200-row position table): paper scale 4,532,816 (transformer body 4,486,656); anchor 509,520. The closed form in `tiger_arch.expected_param_count` is calibrated on the stock model and asserted equal to the measured count.
The paper reports about 13M. The structure here matches the paper's body (4+4 layers, 6 x 64 heads, MLP 1024: about 4.46M weights without biases/norms), so the gap is not a structural difference. It is consistent with a T5-style vocabulary of 32,128 tokens (embedding plus untied head about 8.2M, 4.46M + 8.2M is about 12.7M). That reading is an inference from arithmetic, not verified against the paper's code. This work keeps a vocabulary of 80 and does not pad parameters to reach 13M.

## Deviations from the paper (this is not a TIGER reproduction)

| Item | Paper | Here |
|---|---|---|
| Position encoding | T5 relative position bias | absolute learned table, 200 rows (as the anchor) |
| Normalisation | T5 pre-RMSNorm, no biases | post-LayerNorm with biases (as `nn.Transformer`) |
| Attention logit scaling | T5 does not divide the logits by sqrt(head_dim); the scale is folded into the initialisation | PyTorch scaled dot product attention, logits divided by sqrt(head_dim) |
| Embedding dropout | yes | none (dropout in attention probabilities, both residual branches and inside the FF only) |
| Optimisation | 200k steps, batch 256, LR 0.01 then inverse-sqrt | AdamW, constant LR 3e-4 / 1e-3, effective batch 512 (microbatch 256), `BUDGET_E100_MIN5_P3` |
| Input | hashed user token + history up to 20 items | no user token, history up to 50 items (`HIST_V2_OBS_L50`) |
| Vocabulary / parameters | T5 vocabulary, about 13M | 80 tokens, 4.53M |

## Tests and preflight

- Run in the frozen-base harness with `OMP_NUM_THREADS=1`: `tests/test_generative_arch_model.py` (equivalence, capability), `tests/test_generative_arch.py` (registry, factory, entry, preflight script), `tests/test_generative_arch_mutations.py` (injected defects must turn the same guards red). `tests/arch_checks.py` holds the shared guards.
- `scripts/g_arch_parity.py` (server preflight, one JSON line): PF1 anchor shape vs frozen model (init, K training steps, eval logits fast path off/on, serial decode of the reference best checkpoint), PF2 paper scale serial vs batched-b1 decode on every validation query, PF3 paper-scale training steps per microbatch size (finite loss, seconds per step, peak VRAM, parameter count). No per-query rows, no test split.
