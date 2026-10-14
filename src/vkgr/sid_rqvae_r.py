"""GTOK-01R: repaired RQ-VAE fit path (dead-code reset, optimizer arms, read-only diagnostics).

New module next to sid_rqvae.py; the original fit() is untouched and stays the bit-exact reference.
Every arm of PREREG_R is fixed before any probe is run and none of them is tuned afterwards.
Pure torch + numpy: no file access, no use of any label, query or split data.
"""
import time

import numpy as np
import torch

from vkgr import sid_rqvae as rq

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
