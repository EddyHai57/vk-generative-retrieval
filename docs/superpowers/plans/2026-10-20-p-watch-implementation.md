# P watch-positive implementation plan

Status: train-only scanner candidate ready for independent review; training/verify wiring remains pending.

## Frozen behavior

- Read only train weeks 00–24 for the preflight statistics.
- Reuse the duration bins already fixed in `scripts/p3_label_eda.py`.
- Use Kiro-frozen nearest-rank at 80%; compute and report linear interpolation membership differences for description only.
- Auxiliary events exclude like and dislike, require the event item to be P-supported, and use the same strict-past observed L50 history builder.
- Each epoch draws `min(N_aux, N_like)` auxiliary rows without replacement, mixes them with all like rows, and shuffles. logQ uses expected item frequency `n_like + r*n_aux`, with `r=min(1,N_like/N_aux)`.
- Preserve the strict-past-like mask and existing M1-mean settings.

## Local acceptance

- Pure function tests cover train-week rejection, strict-past history, deterministic sampling, and a hand-calculated logQ example.
- Train-only scanner accepts only explicit train inputs and quantile method; it does not open validation or test paths.
- The formal queue request locks the accepted P inputs, train audit, support and source map; only its fixed CPU stats job ID can be inspected or submitted.
- Training and verify integration remains gated on the frozen quantile decision, independent review, source-identity pinning, and root-owned execution authorization.
