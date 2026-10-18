"""GTOK-01R: repaired RQ-VAE fit path (dead-code reset, optimizer arms, read-only diagnostics).

New module next to sid_rqvae.py; the original fit() is untouched and stays the bit-exact reference.
Every arm of PREREG_R is fixed before any probe is run and none of them is tuned afterwards.
Pure torch + numpy: no file access, no use of any label, query or split data.
"""
import math
import re
import time

import numpy as np
import torch

from vkgr import sid_rqvae as rq
from vkgr import sid_tok_audit as au

_RESET = {'reset_check_every': 100, 'dead_window': 500, 'reset_until': 270000, 'rng_reset_offset': 7919}

PREREG_R = {
    'arms': {
        'p0': {'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': False},
        'l1': {'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': True,
               **_RESET},
        'l2': {'optimizer': 'adagrad', 'lr': 0.1, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': True,
               **_RESET},
        'l3': {'optimizer': 'adagrad', 'lr': 0.025, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': True,
               **_RESET},
        'l4': {'optimizer': 'adam', 'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0,
               'reset': True, **_RESET},
    },
    'arm_order': ('l1', 'l2', 'l3', 'l4'),
    'reset': dict(_RESET),
    'probe_steps': 30000,
    'probe_wall_cap_seconds': 3000,
    'usage_diag_every': 2000,
    'diag_subset_rows': 8192,
    'diag_subset_seed': 42,
    'diag_fine_steps': (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                        2500, 3000, 4000, 5000),
    'diag_every_after': 1000,
}


def make_optimizer(params, arm):
    """Optimizer of one arm. Built from the arm table only, never from keyword overrides."""
    if arm not in PREREG_R['arms']:
        raise ValueError(f'unknown arm {arm!r}')
    cfg = PREREG_R['arms'][arm]
    if cfg['optimizer'] == 'adagrad':
        return torch.optim.Adagrad(params, lr=cfg['lr'], initial_accumulator_value=cfg['initial_accumulator_value'],
                                   eps=cfg['eps'])
    return torch.optim.Adam(params, lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'],
                            weight_decay=cfg['weight_decay'])


def new_last_used(model):
    """One int64 table per codebook on the codebook device; every code counts as used at step 0."""
    return [torch.zeros(c.shape[0], dtype=torch.int64, device=c.device) for c in model.codebooks]


def touch_last_used(last_used, codes, step):
    """Mark the codes selected at this step. Index write on the device: no host sync, repeated indices are fine."""
    for table, index in zip(last_used, codes):
        table[index] = step


def reset_due(step, cfg):
    return step % cfg['reset_check_every'] == 0 and cfg['dead_window'] <= step <= cfg['reset_until']


def _layer_source(residuals, layer):
    """Input residual of codebook `layer` (0-based): residuals is [z, r1, r2, r3]; residuals[3] is never used."""
    return residuals[layer].detach()


def _reset_optimizer_rows(optimizer, param, rows):
    state = optimizer.state[param]
    if 'sum' in state:
        state['sum'][rows] = optimizer.defaults['initial_accumulator_value']
    if 'exp_avg' in state:
        state['exp_avg'][rows] = 0.0
        state['exp_avg_sq'][rows] = 0.0


def reset_dead_codes(model, optimizer, residuals, last_used, step, rng_reset, cfg):
    """Replace codes unused for `dead_window` steps by residual samples of the current batch.

    Returns the reset log entries of this call ({step, layer (0-based), n_reset}); empty when the step is not a
    checkpoint. Must run right after optimizer.step(). rng_reset is advanced only when a code is actually reset.
    """
    log = []
    if not reset_due(step, cfg):
        return log
    for layer, (codebook, table) in enumerate(zip(model.codebooks, last_used)):
        dead = (step - table >= cfg['dead_window']).nonzero().flatten()
        n_dead = int(dead.numel())
        if n_dead == 0:
            continue
        src = _layer_source(residuals, layer)
        assert not src.requires_grad, 'reset source must be detached'
        assert n_dead <= len(src), 'more dead codes than rows in the batch'
        rows = torch.from_numpy(rng_reset.choice(len(src), n_dead, replace=False)).to(src.device)
        with torch.no_grad():
            codebook[dead] = src[rows]
        table[dead] = step
        _reset_optimizer_rows(optimizer, codebook, dead)
        log.append({'step': step, 'layer': layer, 'n_reset': n_dead})
    return log


def diag_due(step):
    fine = PREREG_R['diag_fine_steps']
    return step in fine or (step > max(fine) and step % PREREG_R['diag_every_after'] == 0)


def diag_subset_rows(n):
    """Fixed diagnostic subset: drawn from its own generator, never from a global stream."""
    generator = np.random.RandomState(PREREG_R['diag_subset_seed'])
    return generator.choice(n, min(n, PREREG_R['diag_subset_rows']), replace=False)


def _groups(model):
    return {'encoder': list(model.encoder.parameters()), 'decoder': list(model.decoder.parameters()),
            'codebooks': list(model.codebooks.parameters())}


def group_norms(model):
    """Gradient norm of encoder, decoder and codebooks; reads p.grad only."""
    return {name: float(torch.sqrt(sum((p.grad.detach().double() ** 2).sum() for p in params)))
            for name, params in _groups(model).items()}


def snapshot(model):
    return {name: torch.cat([p.detach().reshape(-1) for p in params]).clone()
            for name, params in _groups(model).items()}


def rel_update(model, before):
    """||theta_after - theta_before|| / ||theta_before|| per parameter group."""
    after = snapshot(model)
    return {name: float((after[name] - before[name]).double().norm() / before[name].double().norm())
            for name in before}


def _relu_dead_fractions(seq, h):
    """Run `seq` on h; per ReLU, the share of units whose output is 0 for every row."""
    fractions = []
    for layer in seq:
        h = layer(h)
        if isinstance(layer, torch.nn.ReLU):
            fractions.append(float((~(h > 0).any(0)).float().mean()))
    return fractions, h


def diagnose(model, x, step, grad_norm=None, rel_update=None):
    """Read-only record of the model state on the diagnostic rows x. Changes no parameter, no gradient."""
    with torch.no_grad():
        enc_dead, z = _relu_dead_fractions(model.encoder, x)
        q = rq.quantize(z, list(model.codebooks))
        dec_dead, xhat = _relu_dead_fractions(model.decoder, q['zhat'])
        terms = rq.loss_terms(x, xhat, q)
        usage, perplexity, nearest = [], [], []
        for layer, codes in enumerate(q['codes']):
            counts = torch.unique(codes, return_counts=True)[1].double()
            p = counts / counts.sum()
            usage.append(int(counts.numel()))
            perplexity.append(float(torch.exp(-(p * p.log()).sum())))
            nearest.append(float((q['residuals'][layer] - q['embeds'][layer]).norm(dim=1).mean()))
        return {'step': step, 'recon': float(terms['recon']), 'codebook': float(sum(terms['codebook'])),
                'commit': float(sum(terms['commit'])), 'z_std_mean': float(z.std(0, unbiased=False).mean()),
                'dead_relu': enc_dead + dec_dead, 'usage_sub': usage, 'perplexity': perplexity,
                'nearest_dist': nearest, 'grad_norm': grad_norm, 'rel_update': rel_update}


def fit_r(content, *, K, arm, seed=rq.PREREG['seed'], max_steps, batch_size=rq.PREREG['batch_size'], device='cpu',
          wall_cap_seconds, clock=time.monotonic, log_every=rq.PREREG['log_every'],
          eval_every=rq.PREREG['eval_every'], check_every=rq.PREREG['check_every'],
          probe_from=rq.PREREG['probe_from'], probe_step=rq.PREREG['probe_step'], project=True, diagnostics=True):
    """Fixed-length training of one arm. A usable state is returned only when status is 'complete'.

    arm='p0' with diagnostics=False reproduces sid_rqvae.fit bit for bit. project=False switches off the
    extrapolation stop at probe_step (the wall cap itself stays). Dead-code resets run on their own cadence
    (reset_check_every of the arm), independent of check_every.
    """
    if arm not in PREREG_R['arms']:
        raise ValueError(f'unknown arm {arm!r}')
    if not isinstance(content, np.ndarray) or content.dtype != np.float32 or content.ndim != 2 \
            or content.shape[1] != rq.PREREG['content_dim'] or not len(content) or not np.isfinite(content).all():
        raise ValueError('content must be finite float32 [N, 64]')
    if type(max_steps) is not int or max_steps < 1 or type(batch_size) is not int or batch_size < 1 \
            or type(seed) is not int or seed < 0:
        raise ValueError('invalid fit arguments')
    if check_every < 1 or probe_from % check_every or probe_step % check_every or not 0 < probe_from < probe_step:
        raise ValueError('probe steps must be multiples of check_every')
    cfg = PREREG_R['arms'][arm]
    n = len(content)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = rq.RQVAE(K).to(device)
    X = torch.from_numpy(content).to(device)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    position = 0
    rq.init_codebooks(model, rq._first_batch(X, order, batch_size), seed, rq.PREREG['kmeans_iters'])
    init_digest = rq.state_digest(model.state_dict())
    optimizer = make_optimizer(model.parameters(), arm)
    last_used = new_last_used(model) if cfg['reset'] else None
    rng_reset = np.random.RandomState(seed + cfg['rng_reset_offset']) if cfg['reset'] else None
    sub = X[torch.from_numpy(diag_subset_rows(n)).to(device)] if diagnostics else None
    bad = torch.zeros((), dtype=torch.bool, device=device)
    running = torch.zeros((), dtype=torch.float64, device=device)
    running_recon = torch.zeros((), dtype=torch.float64, device=device)
    counted = 0
    loss_log, recon_log, usage_log, reset_log, diag_log = [], [], [], [], []
    if diagnostics:
        diag_log.append(diagnose(model, sub, 0))
    prefix_digest = None
    at_probe_from = None
    start = clock()

    def failed(status, step, elapsed):
        return {'status': status, 'step': step, 'elapsed': elapsed, 'K': K, 'seed': seed,
                'init_digest': init_digest, 'loss_log': loss_log, 'usage_log': usage_log,
                'arm': arm, 'recon_log': recon_log, 'reset_log': reset_log, 'diagnostics': diag_log,
                'ms_per_step': elapsed * 1000.0 / step}

    for step in range(1, max_steps + 1):
        if position >= n:
            order = rng.permutation(n)
            position = 0
        rows = torch.from_numpy(order[position:position + batch_size]).to(device)
        position += batch_size
        batch = X[rows]
        out = model(batch)
        terms = rq.loss_terms(batch, out['xhat'], out['q'])
        optimizer.zero_grad(set_to_none=True)
        terms['total'].backward()
        record = diagnostics and diag_due(step)
        before = snapshot(model) if record else None
        optimizer.step()
        if cfg['reset']:
            touch_last_used(last_used, out['q']['codes'], step)
            fired = reset_dead_codes(model, optimizer, out['q']['residuals'], last_used, step, rng_reset, cfg)
            reset_log.extend(fired)
        total = terms['total'].detach()
        bad |= ~torch.isfinite(total)
        running += total.double()
        running_recon += terms['recon'].detach().double()
        counted += 1
        if record:
            diag_log.append(diagnose(model, sub, step, group_norms(model), rel_update(model, before)))
        if step % log_every == 0:
            loss_log.append((step, float(running) / counted))
            recon_log.append((step, float(running_recon) / counted))
            running.zero_()
            running_recon.zero_()
            counted = 0
        if step % eval_every == 0:
            usage_log.append((step, rq._usage(model, X)))
        if step % check_every and step != max_steps:
            continue
        if bool(bad) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            return failed('fail_nonfinite', step, clock() - start)
        elapsed = clock() - start
        if elapsed > wall_cap_seconds:
            return failed('fail_wall', step, elapsed)
        if step == probe_from:
            at_probe_from = elapsed
        if step == probe_step:
            prefix_digest = rq.state_digest(model.state_dict())
            rate = (elapsed - at_probe_from) / (probe_step - probe_from)
            if project and elapsed + rate * (max_steps - probe_step) > wall_cap_seconds:
                return failed('fail_wall_projected', step, elapsed)
    state = {'K': K, 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    result = failed('complete', max_steps, elapsed)
    result.update({'state': state, 'prefix_digest': prefix_digest})
    return result


# ---------------------------------------------------------------------------------------------------------------
# FEAT-002: diagnostic criteria (I1 / I2 / I4), probe and decision schemas, arm selection, preflight, override.
# Pure functions on plain records: no file access, no training. Every constant below is written down before any
# probe has run; none of them is a gate. The gates (4/5 usage, R2 > 0, strictly decreasing residuals, 300,000 steps)
# stay owned by sid_tok_audit and are only reused here.
# ---------------------------------------------------------------------------------------------------------------
PREREG_R_SELECT = {
    'variant_K': {'T1': 256, 'T2': 16},
    # reference values of the old (collapsed) fit, taken from its failure record
    'p0_loss_windows': {'T1': (0.8199, 1.0100, 0.8115), 'T2': (0.8754, 0.8116, 0.8115)},
    'p0_loss_tol': 1e-3,
    'p0_usage_ref': (1, 1, 2),
    'p0_usage_slack': 1,
    'p0_final_usage_max': 3,
    'usage_log_steps': tuple(range(2000, 30001, 2000)),
    'a2_from_step': 20000,
    'a2_to_step': 30000,
    'd_points': (20000, 22000, 24000, 26000, 28000, 30000),
    'e_r2_min': 0.05,
    'i1_z_std_min': 1e-3,
    'i1_dead_frac': 0.99,
    'i1_max_step': 3000,
    'i1_usage_max': 3,
    'i2_step': 30000,
    'i2_rel_tol': 1e-3,
    'i4_norm_range': (0.99, 1.01),
    'old_fit_wall_seconds': {'T1': 4354, 'T2': 4302},
    'preflight_limit_seconds': 6000,
}

PROBE_SCHEMA = 'gtok01r-probe-v1'
DECISION_SCHEMA = 'gtok01r-decision-v1'
VARIANTS = ('T1', 'T2')
ARMS = tuple(PREREG_R['arms'])
STATUSES = ('complete', 'fail_nonfinite', 'fail_wall', 'fail_wall_projected')
DECISIONS = tuple(PREREG_R['arm_order']) + ('STOP', 'ESCALATE', 'INVALID_CONTROL')
CRITERIA = ('a1', 'a2', 'b', 'c', 'd', 'e')
PROBE_KEYS = ('schema', 'variant', 'arm', 'K', 'config_sha256', 'code_commit', 'content_file_sha256',
              'content_array_sha256', 'catalog_sha256', 'status', 'steps', 'elapsed', 'ms_per_step', 'loss_log',
              'recon_log', 'usage_log', 'diagnostics', 'final')
DECISION_KEYS = ('schema', 'config_sha256', 'catalog_sha256', 'content_file_sha256', 'content_array_sha256',
                 'probe_results', 'per_arm_per_K', 'decision', 'escalate_listed_arms', 'override', 'preflight')
SHARED_KEYS = ('config_sha256', 'code_commit', 'content_file_sha256', 'content_array_sha256', 'catalog_sha256')
_HEX64 = re.compile(r'[0-9a-f]{64}')
_HEX40 = re.compile(r'[0-9a-f]{40}')


def _bad(message):
    raise ValueError(message)


def _is_int(v):
    return type(v) is int


def _is_num(v):
    return type(v) in (int, float) and math.isfinite(v)


def _is_hex(v, pattern):
    return isinstance(v, str) and pattern.fullmatch(v) is not None


def _check_log(name, log, complete, grid, steps, usage):
    """Entries are [step, value]; steps strictly increasing and <= steps; a complete probe carries the whole grid."""
    if not isinstance(log, list):
        _bad(f'{name} must be a list')
    for entry in log:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not _is_int(entry[0]):
            _bad(f'{name} entries must be [step, value]')
        value = entry[1]
        ok = (isinstance(value, (list, tuple)) and len(value) == 3 and all(_is_int(u) and u >= 0 for u in value)) \
            if usage else _is_num(value)
        if not ok:
            _bad(f'{name} holds a malformed value')
    got = [entry[0] for entry in log]
    if any(a >= b for a, b in zip(got, got[1:])) or any(g < 1 or g > steps for g in got):
        _bad(f'{name} steps must be increasing and within the run')
    if complete and got != list(grid):
        _bad(f'{name} of a complete probe must cover its whole grid')


def _check_diagnostics(records):
    if not isinstance(records, list):
        _bad('diagnostics must be a list')
    for r in records:
        if not isinstance(r, dict) or not _is_int(r.get('step')) or not _is_num(r.get('z_std_mean')):
            _bad('diagnostic record needs step and z_std_mean')
        dead = r.get('dead_relu')
        if not isinstance(dead, (list, tuple)) or len(dead) != 6 or not all(_is_num(x) for x in dead):
            _bad('diagnostic record needs six dead_relu fractions')


def _check_final(final, K):
    if not isinstance(final, dict) or set(final) != {'canonical_usage', 'r2', 'residual_norms'}:
        _bad('final must hold canonical_usage, r2 and residual_norms')
    used, norms = final['canonical_usage'], final['residual_norms']
    if not isinstance(used, (list, tuple)) or len(used) != 3 or not all(_is_int(u) and 0 <= u <= K for u in used):
        _bad('canonical_usage must be three counts within [0, K]')
    if not _is_num(final['r2']) or not isinstance(norms, (list, tuple)) or len(norms) != 4 \
            or not all(_is_num(x) for x in norms):
        _bad('r2 and four residual norms must be finite numbers')


def validate_probe(p):
    """Strict gtok01r-probe-v1 check: exact field set, lower-case arm, digests, status/final coupling, log grids."""
    sel = PREREG_R_SELECT
    if not isinstance(p, dict) or set(p) != set(PROBE_KEYS):
        _bad('probe-result fields differ from gtok01r-probe-v1')
    if p['schema'] != PROBE_SCHEMA:
        _bad('probe-result schema mismatch')
    if p['variant'] not in VARIANTS or p['arm'] not in ARMS:
        _bad('probe-result variant or arm is not in the pre-registered set (arm names are lower case)')
    if not _is_int(p['K']) or p['K'] != sel['variant_K'][p['variant']]:
        _bad('probe-result K does not match its variant')
    if not all(_is_hex(p[k], _HEX64) for k in SHARED_KEYS if k != 'code_commit') or not _is_hex(p['code_commit'], _HEX40):
        _bad('probe-result digests must be lower-case hex (sha256: 64, commit: 40)')
    if p['status'] not in STATUSES:
        _bad('probe-result status invalid')
    complete = p['status'] == 'complete'
    if not _is_int(p['steps']) or not 1 <= p['steps'] <= PREREG_R['probe_steps'] \
            or (complete and p['steps'] != PREREG_R['probe_steps']):
        _bad('probe-result steps invalid')
    if not _is_num(p['elapsed']) or p['elapsed'] <= 0 or not _is_num(p['ms_per_step']) \
            or not math.isclose(p['ms_per_step'], p['elapsed'] * 1000 / p['steps'], rel_tol=1e-9):
        _bad('probe-result elapsed / ms_per_step invalid')
    log_grid = range(rq.PREREG['log_every'], PREREG_R['probe_steps'] + 1, rq.PREREG['log_every'])
    _check_log('loss_log', p['loss_log'], complete, log_grid, p['steps'], False)
    _check_log('recon_log', p['recon_log'], complete, log_grid, p['steps'], False)
    _check_log('usage_log', p['usage_log'], complete, sel['usage_log_steps'], p['steps'], True)
    _check_diagnostics(p['diagnostics'])
    if complete != (p['final'] is not None):
        _bad('final is null exactly when the status is not complete')
    if complete:
        _check_final(p['final'], p['K'])


def validate_probe_set(probes, config_sha256=None):
    """The 10 probe-results cover {p0,l1..l4} x {T1,T2} exactly once and agree on config, commit and inputs."""
    if not isinstance(probes, (list, tuple)) or len(probes) != len(VARIANTS) * len(ARMS):
        _bad('exactly one probe-result per (variant, arm) is required')
    for p in probes:
        validate_probe(p)
    if sorted((p['variant'], p['arm']) for p in probes) != sorted((v, a) for v in VARIANTS for a in ARMS):
        _bad('probe-results must cover every (variant, arm) exactly once')
    for key in SHARED_KEYS:
        if len({p[key] for p in probes}) != 1:
            _bad(f'probe-results disagree on {key}')
    if config_sha256 is not None and probes[0]['config_sha256'] != config_sha256:
        _bad('probe-results were produced under a different config')


def sst_of(content):
    """Sum over features, mean over rows of the squared distance to the column mean (the recon term's convention)."""
    data = np.asarray(content, dtype=np.float64)
    return float(((data - data.mean(0)) ** 2).sum()) / len(data)


def _a_holds(record):
    sel = PREREG_R_SELECT
    return record['z_std_mean'] < sel['i1_z_std_min'] or max(record['dead_relu']) >= sel['i1_dead_frac']


def classify_i1(probe):
    """I1 three-state. Reads the full-catalogue usage_log of the training model, never the subset's usage_sub."""
    sel = PREREG_R_SELECT
    early = [r for r in probe['diagnostics'] if r['step'] <= sel['i1_max_step']]
    full = {s: used for s, used in probe['usage_log']}
    collapsed = all(s in full and max(full[s]) <= sel['i1_usage_max'] for s in sel['usage_log_steps'])
    a_early = any(_a_holds(r) for r in early)
    first = sel['usage_log_steps'][0]
    if a_early and collapsed:
        return 'SUPPORTED'
    if early and not a_early and first in full and max(full[first]) <= sel['i1_usage_max']:
        return 'REFUTED'
    return 'INCONCLUSIVE'


def classify_i2(probe, sst):
    """I2 binary: the recon term (window (29000, 30000]) equals SST within the relative tolerance."""
    sel = PREREG_R_SELECT
    if not sst > 0:
        _bad('sst must be positive')
    values = [v for s, v in probe['recon_log'] if s == sel['i2_step']]
    if len(values) != 1:
        _bad('recon_log has no entry for the I2 step')
    return 'SUPPORTED' if abs(values[0] - sst) / sst < sel['i2_rel_tol'] else 'REFUTED'


def classify_i4(content, catalog_zero_norm_rows):
    """I4 binary: mean row norm inside the range and the zero-norm row count equal to the catalogue's record."""
    lo, hi = PREREG_R_SELECT['i4_norm_range']
    norms = np.linalg.norm(np.asarray(content, dtype=np.float64), axis=1)
    ok = lo <= float(norms.mean()) <= hi and int((norms == 0).sum()) == catalog_zero_norm_rows
    return 'SUPPORTED' if ok else 'REFUTED'


def layers_ok(used, K):
    """R1 on three layers; the 4/5 threshold lives in sid_tok_audit.usage_ok and is reused, never rewritten."""
    return all(au.usage_ok(u, K) for u in used)


def _usage_at(probe, step):
    for s, used in probe['usage_log']:
        if s == step:
            return list(used)
    _bad(f'usage_log has no entry for step {step}')


def arm_criteria(probe):
    """(a1)(a2)(b)(c)(d)(e) of one probe-result; all null unless the probe completed."""
    sel = PREREG_R_SELECT
    out = {'status': probe['status'], **dict.fromkeys(CRITERIA)}
    if probe['status'] != 'complete':
        return out
    K, final = probe['K'], probe['final']
    used, norms = final['canonical_usage'], final['residual_norms']
    low, high = _usage_at(probe, sel['a2_from_step']), _usage_at(probe, sel['a2_to_step'])
    out['a1'] = layers_ok(used, K)
    out['a2'] = all(high[l] > low[l] and high[l] > sel['p0_final_usage_max']
                    for l in range(3) if not layers_ok([used[l]], K))
    out['b'] = final['r2'] > 0
    out['c'] = all(x > y for x, y in zip(norms, norms[1:]))
    out['d'] = all(layers_ok(_usage_at(probe, s), K) for s in sel['d_points'])
    out['e'] = final['r2'] >= sel['e_r2_min']
    return out


def p0_control(probe):
    """Does the P0 control reproduce the old collapse? Components use the evidence that exists; valid needs complete."""
    sel = PREREG_R_SELECT
    ref = sel['p0_loss_windows'][probe['variant']]
    out = {'complete': probe['status'] == 'complete'}
    losses = [v for _, v in probe['loss_log'][:len(ref)]]
    out['loss'] = len(losses) == len(ref) and all(abs(v - r) <= sel['p0_loss_tol'] for v, r in zip(losses, ref))
    cap = [r + sel['p0_usage_slack'] for r in sel['p0_usage_ref']]
    out['usage'] = all(u <= c for _, used in probe['usage_log'] for u, c in zip(used, cap))
    final = probe['final']
    out['final'] = final is None or all(u <= sel['p0_final_usage_max'] for u in final['canonical_usage'])
    out['valid'] = out['complete'] and out['loss'] and out['usage'] and out['final']
    return out


QUALIFY = ('a1', 'b', 'c', 'd', 'e')
CANDIDATE = ('a2', 'b', 'c')


def _passes(c, keys):
    return all(c[k] is True for k in keys)


def _qualified(per_variant):
    return all(_passes(c, QUALIFY) for c in per_variant.values())


def _candidate(per_variant):
    return all(_passes(c, CANDIDATE) for c in per_variant.values())


def _mechanical(control, per_arm):
    order = PREREG_R['arm_order']
    if not all(c['valid'] for c in control.values()):
        return 'INVALID_CONTROL', []
    for arm in order:
        if _qualified(per_arm[arm]):
            return arm, []
    listed = [arm for arm in order if _candidate(per_arm[arm])]
    return ('ESCALATE', listed) if listed else ('STOP', [])


def preflight(by, arm):
    """Projected formal-fit wall per K from the probe step-rate ratio to P0. Reported only, never a gate."""
    sel = PREREG_R_SELECT
    out = {}
    for v in VARIANTS:
        ratio = by[(v, arm)]['ms_per_step'] / by[(v, 'p0')]['ms_per_step']
        out[v] = {'overhead_ratio': ratio, 'projected_fit_seconds': sel['old_fit_wall_seconds'][v] * ratio}
    out['exceeds'] = any(out[v]['projected_fit_seconds'] > sel['preflight_limit_seconds'] for v in VARIANTS)
    return out


def select_arm(probes, override_arm=None, override_source=None):
    """Pure pre-registered selection over the 10 probe-results.

    Returns decision (l1..l4 | STOP | ESCALATE | INVALID_CONTROL), escalate_listed_arms, per_arm_per_K, override,
    preflight and p0_control. An override is accepted only when a fresh recomputation yields ESCALATE, the arm is
    listed and a source is given; otherwise ValueError.
    """
    validate_probe_set(probes)
    if (override_arm is None) != (override_source is None):
        _bad('override arm and override source must be given together')
    by = {(p['variant'], p['arm']): p for p in probes}
    control = {v: p0_control(by[(v, 'p0')]) for v in VARIANTS}
    per_arm = {arm: {v: arm_criteria(by[(v, arm)]) for v in VARIANTS} for arm in PREREG_R['arm_order']}
    decision, listed = _mechanical(control, per_arm)
    override = None
    if override_arm is not None:
        ok = decision == 'ESCALATE' and override_arm in listed and isinstance(override_source, str) \
            and bool(override_source.strip())
        if not ok:
            _bad('override refused: the recomputed decision is not ESCALATE, or the arm is not listed, or no source')
        override = {'arm': override_arm, 'source': override_source, 'escalate_listed_arms': list(listed)}
        decision = override_arm
    pre = preflight(by, decision) if decision in PREREG_R['arm_order'] else None
    return {'decision': decision, 'escalate_listed_arms': listed, 'per_arm_per_K': per_arm, 'override': override,
            'preflight': pre, 'p0_control': control}


def _check_preflight(pre):
    if not isinstance(pre, dict) or set(pre) != {'T1', 'T2', 'exceeds'} or type(pre['exceeds']) is not bool:
        _bad('preflight must hold T1, T2 and a boolean exceeds')
    for v in VARIANTS:
        entry = pre[v]
        if not isinstance(entry, dict) or set(entry) != {'overhead_ratio', 'projected_fit_seconds'} \
                or not all(_is_num(entry[k]) and entry[k] > 0 for k in entry):
            _bad('preflight entries must hold a positive overhead_ratio and projected_fit_seconds')


def _check_per_arm(per_arm):
    order = PREREG_R['arm_order']
    if not isinstance(per_arm, dict) or set(per_arm) != set(order):
        _bad('per_arm_per_K must cover l1..l4')
    for arm in order:
        if not isinstance(per_arm[arm], dict) or set(per_arm[arm]) != set(VARIANTS):
            _bad('per_arm_per_K entries must cover T1 and T2')
        for v in VARIANTS:
            c = per_arm[arm][v]
            if not isinstance(c, dict) or set(c) != {'status', *CRITERIA} or c['status'] not in STATUSES:
                _bad('criteria record malformed')
            values = [c[k] for k in CRITERIA]
            if c['status'] == 'complete':
                if not all(type(x) is bool for x in values):
                    _bad('criteria of a complete probe must be booleans')
            elif any(x is not None for x in values):
                _bad('criteria of a non-complete probe must be null')


def validate_decision(d):
    """Strict gtok01r-decision-v1 check (fixed field set, probe references, criteria, override and preflight)."""
    order = PREREG_R['arm_order']
    if not isinstance(d, dict) or set(d) != set(DECISION_KEYS):
        _bad('decision fields differ from gtok01r-decision-v1')
    if d['schema'] != DECISION_SCHEMA:
        _bad('decision schema mismatch')
    if not all(_is_hex(d[k], _HEX64) for k in ('config_sha256', 'catalog_sha256', 'content_file_sha256',
                                              'content_array_sha256')):
        _bad('decision digests must be lower-case sha256 hex')
    refs = d['probe_results']
    if not isinstance(refs, list) or len(refs) != len(VARIANTS) * len(ARMS):
        _bad('probe_results must hold exactly 10 references')
    for e in refs:
        if not isinstance(e, dict) or set(e) != {'variant', 'arm', 'sha256', 'code_commit'} \
                or e['variant'] not in VARIANTS or e['arm'] not in ARMS or not _is_hex(e['sha256'], _HEX64) \
                or not _is_hex(e['code_commit'], _HEX40):
            _bad('malformed probe reference')
    if sorted((e['variant'], e['arm']) for e in refs) != sorted((v, a) for v in VARIANTS for a in ARMS):
        _bad('probe_results must cover every (variant, arm) exactly once')
    if len({e['code_commit'] for e in refs}) != 1:
        _bad('probe references disagree on code_commit')
    _check_per_arm(d['per_arm_per_K'])
    decision, listed, override = d['decision'], d['escalate_listed_arms'], d['override']
    if decision not in DECISIONS:
        _bad('unknown decision')
    if not isinstance(listed, list) or listed != [a for a in order if a in listed]:
        _bad('escalate_listed_arms must be a list of l1..l4 in fixed order')
    if override is None:
        expect_listed = decision == 'ESCALATE'
    else:
        if not isinstance(override, dict) or set(override) != {'arm', 'source', 'escalate_listed_arms'} \
                or override['arm'] != decision or decision not in order or decision not in listed \
                or override['escalate_listed_arms'] != listed or not isinstance(override['source'], str) \
                or not override['source'].strip():
            _bad('override is inconsistent with the decision')
        expect_listed = True
    if bool(listed) != expect_listed:
        _bad('escalate_listed_arms must be non-empty exactly for ESCALATE or a released arm')
    if decision in order:
        _check_preflight(d['preflight'])
    elif d['preflight'] is not None:
        _bad('preflight must be null unless an l-arm was decided')


def build_decision(probes, probe_sha256s, *, config_sha256, override_arm=None, override_source=None):
    """The decision.json body for a validated probe set; probe_sha256s[i] is the file sha256 of probes[i]."""
    validate_probe_set(probes, config_sha256=config_sha256)
    if not isinstance(probe_sha256s, (list, tuple)) or len(probe_sha256s) != len(probes) \
            or not all(_is_hex(x, _HEX64) for x in probe_sha256s):
        _bad('one sha256 per probe-result is required')
    result = select_arm(probes, override_arm, override_source)
    refs = sorted(({'variant': p['variant'], 'arm': p['arm'], 'sha256': sha, 'code_commit': p['code_commit']}
                   for p, sha in zip(probes, probe_sha256s)),
                  key=lambda e: (VARIANTS.index(e['variant']), ARMS.index(e['arm'])))
    first = probes[0]
    decision = {'schema': DECISION_SCHEMA, 'config_sha256': config_sha256, 'catalog_sha256': first['catalog_sha256'],
                'content_file_sha256': first['content_file_sha256'],
                'content_array_sha256': first['content_array_sha256'], 'probe_results': refs,
                'per_arm_per_K': result['per_arm_per_K'], 'decision': result['decision'],
                'escalate_listed_arms': result['escalate_listed_arms'], 'override': result['override'],
                'preflight': result['preflight']}
    validate_decision(decision)
    return decision

