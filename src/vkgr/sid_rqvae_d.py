"""G6-PREP: RQ-VAE with a configurable number of residual levels (new file; ``sid_rqvae`` is imported, never edited).

``sid_rqvae.RQVAE`` / ``fit`` / ``export_codes`` / ``reconstruction_report`` are pinned to ``PREREG['levels'] == 3``.
Here the level count is a parameter. With ``levels == 3`` the initial weights, the training trajectory, the exported
codes and the reports are bit-identical to the frozen module (tests pin that); with ``levels == 4`` the extra codebook is
created after the first three, so the encoder, the decoder and the first three codebooks start from the same values.
Every other pre-registered constant (``sid_rqvae.PREREG``) is used unchanged. No file access, no labels, no splits.
"""
import time

import numpy as np
import torch
from torch import nn

from . import sid_rqvae as rq

MIN_LEVELS = 2
MAX_LEVELS = 8


def _levels(levels):
    if type(levels) is not int or not MIN_LEVELS <= levels <= MAX_LEVELS:
        raise ValueError('levels must be an int between 2 and 8')
    return levels


class RQVAED(nn.Module):
    """Encoder 64-512-256-128-32, mirrored decoder, ``levels`` separate (K, 32) codebooks (created after the networks)."""

    def __init__(self, K, levels):
        super().__init__()
        if type(K) is not int or K < 2:
            raise ValueError('K must be an int >= 2')
        levels = _levels(levels)
        dims = (rq.PREREG['content_dim'],) + tuple(rq.PREREG['encoder_hidden']) + (rq.PREREG['latent_dim'],)
        self.encoder = rq._mlp(dims)
        self.decoder = rq._mlp(dims[::-1])
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(K, rq.PREREG['latent_dim'])) for _ in range(levels)])

    def forward(self, x):
        z = self.encoder(x)
        q = rq.quantize(z, list(self.codebooks))
        zst = z + (q['zhat'] - z).detach()  # straight-through
        return {'z': z, 'q': q, 'zst': zst, 'xhat': self.decoder(zst)}


def fit_d(content, *, K, levels, seed=rq.PREREG['seed'], max_steps, batch_size=rq.PREREG['batch_size'], device='cpu',
          wall_cap_seconds, clock=time.monotonic, log_every=rq.PREREG['log_every'], eval_every=rq.PREREG['eval_every'],
          check_every=rq.PREREG['check_every'], probe_from=rq.PREREG['probe_from'], probe_step=rq.PREREG['probe_step']):
    """``sid_rqvae.fit`` with ``levels`` codebooks. A usable state is returned only when status is 'complete'."""
    levels = _levels(levels)
    if not isinstance(content, np.ndarray) or content.dtype != np.float32 or content.ndim != 2 \
            or content.shape[1] != rq.PREREG['content_dim'] or not len(content) or not np.isfinite(content).all():
        raise ValueError('content must be finite float32 [N, 64]')
    if type(max_steps) is not int or max_steps < 1 or type(batch_size) is not int or batch_size < 1 \
            or type(seed) is not int or seed < 0:
        raise ValueError('invalid fit arguments')
    if check_every < 1 or probe_from % check_every or probe_step % check_every or not 0 < probe_from < probe_step:
        raise ValueError('probe steps must be multiples of check_every')
    n = len(content)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = RQVAED(K, levels).to(device)
    X = torch.from_numpy(content).to(device)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    position = 0
    rq.init_codebooks(model, rq._first_batch(X, order, batch_size), seed, rq.PREREG['kmeans_iters'])
    init_digest = rq.state_digest(model.state_dict())
    optimizer = torch.optim.Adagrad(model.parameters(), lr=rq.PREREG['lr'],
                                    initial_accumulator_value=rq.PREREG['initial_accumulator_value'],
                                    eps=rq.PREREG['eps'])
    bad = torch.zeros((), dtype=torch.bool, device=device)
    running = torch.zeros((), dtype=torch.float64, device=device)
    counted = 0
    loss_log, usage_log = [], []
    prefix_digest = None
    at_probe_from = None
    start = clock()

    def failed(status, step, elapsed):
        return {'status': status, 'step': step, 'elapsed': elapsed, 'K': K, 'seed': seed,
                'init_digest': init_digest, 'loss_log': loss_log, 'usage_log': usage_log}

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
        optimizer.step()
        total = terms['total'].detach()
        bad |= ~torch.isfinite(total)
        running += total.double()
        counted += 1
        if step % log_every == 0:
            loss_log.append((step, float(running) / counted))
            running.zero_()
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
            if elapsed + rate * (max_steps - probe_step) > wall_cap_seconds:
                return failed('fail_wall_projected', step, elapsed)
    state = {'K': K, 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    result = failed('complete', max_steps, elapsed)
    result.update({'state': state, 'prefix_digest': prefix_digest})
    return result


def state_levels(state):
    """Number of codebooks recorded in a state; the keys must be exactly codebooks.0 .. codebooks.levels-1."""
    if type(state) is not dict or type(state.get('state_dict')) is not dict:
        raise ValueError('invalid state')
    indices = sorted(int(key.split('.')[1]) for key in state['state_dict'] if key.startswith('codebooks.'))
    if indices != list(range(len(indices))):
        raise ValueError('codebook entries must be numbered 0..levels-1')
    return _levels(len(indices))


def _canonical_chunks_d(state, X):
    """float64, CPU, one thread: yields (rows, quantized, reconstruction) per chunk."""
    if state.get('K') is None or not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != rq.PREREG['content_dim']:
        raise ValueError('invalid export inputs')
    model = RQVAED(state['K'], state_levels(state)).to(rq.EXPORT_DTYPE)
    model.load_state_dict({k: v.to(rq.EXPORT_DTYPE) for k, v in state['state_dict'].items()})
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for i in range(0, len(X), rq.EXPORT_CHUNK):
                rows = torch.from_numpy(np.ascontiguousarray(X[i:i + rq.EXPORT_CHUNK])).to(rq.EXPORT_DTYPE)
                q = rq.quantize(model.encoder(rows), list(model.codebooks))
                yield rows, q, model.decoder(q['zhat'])
    finally:
        torch.set_num_threads(previous)


def export_codes_d(state, X):
    """Canonical codes int64 [N, levels]; ties go to the lowest index."""
    levels = state_levels(state)
    parts = [torch.stack(q['codes'], 1).numpy() for _, q, _ in _canonical_chunks_d(state, X)]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros((0, levels), np.int64)


def codebook_usage_d(codes, K):
    codes = np.asarray(codes)
    if codes.ndim != 2 or not MIN_LEVELS <= codes.shape[1] <= MAX_LEVELS or codes.min() < 0 or codes.max() >= K:
        raise ValueError('codes out of range')
    return [int(len(np.unique(codes[:, d]))) for d in range(codes.shape[1])]


def reconstruction_report_d(state, X):
    """mse = mean row squared error; sst = mean row squared distance to the mean; residual_norms = r0 .. r_levels."""
    levels = state_levels(state)
    total_error, norm_sums, count = 0.0, np.zeros(levels + 1), 0
    for rows, q, xhat in _canonical_chunks_d(state, X):
        total_error += float(((rows - xhat) ** 2).sum())
        norm_sums += np.array([float(r.norm(dim=1).sum()) for r in q['residuals']])
        count += len(rows)
    data = np.asarray(X, dtype=np.float64)
    mse = total_error / count
    sst = float(((data - data.mean(0)) ** 2).sum()) / count
    return {'mse': mse, 'sst': sst, 'r2': 1 - mse / sst, 'residual_norms': (norm_sums / count).tolist()}
