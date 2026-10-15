"""Independent RQ-VAE semantic-ID tokenizer core (arXiv 2305.05065 sec 3.1 / 4.1.3).

Original implementation from the paper text; no third-party code is copied.
Pure torch + numpy: no file access, no use of any label, query or split data.
Every choice the paper leaves open is pre-registered in PREREG and the
design notes of the GTOK-01 plan; none of them is tuned afterwards.
"""
import hashlib
import time

import numpy as np
import torch
from torch import nn

PREREG = {
    'content_dim': 64, 'encoder_hidden': (512, 256, 128), 'latent_dim': 32, 'levels': 3,
    'beta': 0.25, 'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7,
    'batch_size': 1024, 'max_steps': 300000, 'wall_cap_seconds': 7200, 'kmeans_iters': 10, 'seed': 42,
    'usage_gate_num_den': (4, 5),  # a level passes iff used * 5 >= K * 4 (integer test)
    'log_every': 1000, 'eval_every': 10000, 'check_every': 100, 'probe_from': 500, 'probe_step': 2000,
}

EXPORT_DTYPE = torch.float64  # canonical export arithmetic
EXPORT_CHUNK = 8192


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    """Encoder 64-512-256-128-32, mirrored decoder, three separate (K, 32) codebooks."""

    def __init__(self, K):
        super().__init__()
        if type(K) is not int or K < 2:
            raise ValueError('K must be an int >= 2')
        dims = (PREREG['content_dim'],) + tuple(PREREG['encoder_hidden']) + (PREREG['latent_dim'],)
        self.encoder = _mlp(dims)
        self.decoder = _mlp(dims[::-1])
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(K, PREREG['latent_dim'])) for _ in range(PREREG['levels'])])

    def forward(self, x):
        z = self.encoder(x)
        q = quantize(z, list(self.codebooks))
        zst = z + (q['zhat'] - z).detach()  # straight-through
        return {'z': z, 'q': q, 'zst': zst, 'xhat': self.decoder(zst)}


def _nearest(r, codebook):
    """Index of the closest row; exact ties go to the lowest index."""
    distance = (r * r).sum(1, keepdim=True) - 2 * (r @ codebook.T) + (codebook * codebook).sum(1)[None, :]
    return distance.argmin(1)


def quantize(z, codebooks):
    """Residual quantization. The residual chain subtracts the detached codeword."""
    residual = z
    residuals, codes, embeds = [residual], [], []
    for codebook in codebooks:
        index = _nearest(residual, codebook)
        embed = codebook[index]
        codes.append(index)
        embeds.append(embed)
        residual = residual - embed.detach()
        residuals.append(residual)
    return {'codes': codes, 'embeds': embeds, 'residuals': residuals, 'zhat': sum(embeds)}


def loss_terms(x, xhat, q, beta=PREREG['beta']):
    """Sum over features, mean over the batch.

    recon reaches encoder and decoder, codebook terms reach the codebooks only,
    commitment terms reach the encoder only.
    """
    recon = ((x - xhat) ** 2).sum(1).mean()
    codebook, commit = [], []
    for residual, embed in zip(q['residuals'], q['embeds']):
        codebook.append(((residual.detach() - embed) ** 2).sum(1).mean())
        commit.append(beta * ((residual - embed.detach()) ** 2).sum(1).mean())
    return {'recon': recon, 'codebook': codebook, 'commit': commit,
            'total': recon + sum(codebook) + sum(commit)}


def kmeans_lloyd(points, k, seed, iters):
    """Seeded Lloyd iterations; initial centres drawn without replacement; empty clusters keep their centre."""
    n = len(points)
    if k > n:
        raise ValueError('more centres than points')
    start = np.random.RandomState(seed).choice(n, k, replace=False)
    data = points.detach().double()
    centers = data[torch.from_numpy(start)].clone()
    for _ in range(iters):
        labels = _nearest(data, centers)
        sums = torch.zeros_like(centers).index_add_(0, labels, data)
        counts = torch.bincount(labels, minlength=k)
        centers = torch.where((counts > 0)[:, None], sums / counts.clamp(min=1)[:, None].double(), centers)
    return centers.to(points.dtype)


def init_codebooks(model, batch, seed, iters=PREREG['kmeans_iters']):
    """Level by level k-means on the given batch only; level d+1 sees the residual left by level d."""
    with torch.no_grad():
        residual = model.encoder(batch)
        for codebook in model.codebooks:
            codebook.copy_(kmeans_lloyd(residual, codebook.shape[0], seed, iters))
            residual = residual - codebook[_nearest(residual.double(), codebook.double())]


def state_digest(state_dict):
    digest = hashlib.sha256()
    for value in state_dict.values():
        digest.update(value.detach().cpu().contiguous().to(torch.float32).numpy().astype('<f4').tobytes())
    return digest.hexdigest()


def _first_batch(X, order, batch_size):
    """Codebook initialisation sees the first batch of the first epoch and nothing else."""
    return X[torch.from_numpy(order[:batch_size]).to(X.device)]


def _usage(model, X):
    seen = [torch.zeros(c.shape[0], dtype=torch.bool, device=X.device) for c in model.codebooks]
    with torch.no_grad():
        for i in range(0, len(X), EXPORT_CHUNK):
            q = quantize(model.encoder(X[i:i + EXPORT_CHUNK]), list(model.codebooks))
            for mask, index in zip(seen, q['codes']):
                mask[index] = True
    return [int(mask.sum()) for mask in seen]


def fit(content, *, K, seed=PREREG['seed'], max_steps, batch_size=PREREG['batch_size'], device='cpu',
        wall_cap_seconds, clock=time.monotonic, log_every=PREREG['log_every'], eval_every=PREREG['eval_every'],
        check_every=PREREG['check_every'], probe_from=PREREG['probe_from'], probe_step=PREREG['probe_step']):
    """Fixed-length training. A usable state is returned only when status is 'complete'."""
    if not isinstance(content, np.ndarray) or content.dtype != np.float32 or content.ndim != 2 \
            or content.shape[1] != PREREG['content_dim'] or not len(content) or not np.isfinite(content).all():
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
    model = RQVAE(K).to(device)
    X = torch.from_numpy(content).to(device)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    position = 0
    init_codebooks(model, _first_batch(X, order, batch_size), seed, PREREG['kmeans_iters'])
    init_digest = state_digest(model.state_dict())
    optimizer = torch.optim.Adagrad(model.parameters(), lr=PREREG['lr'],
                                    initial_accumulator_value=PREREG['initial_accumulator_value'],
                                    eps=PREREG['eps'])
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
        terms = loss_terms(batch, out['xhat'], out['q'])
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
            usage_log.append((step, _usage(model, X)))
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
            prefix_digest = state_digest(model.state_dict())
            rate = (elapsed - at_probe_from) / (probe_step - probe_from)
            if elapsed + rate * (max_steps - probe_step) > wall_cap_seconds:
                return failed('fail_wall_projected', step, elapsed)
    state = {'K': K, 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    result = failed('complete', max_steps, elapsed)
    result.update({'state': state, 'prefix_digest': prefix_digest})
    return result


def _canonical_chunks(state, X):
    """float64, CPU, one thread: yields (rows, quantized, reconstruction) per chunk."""
    if state.get('K') is None or not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != PREREG['content_dim']:
        raise ValueError('invalid export inputs')
    model = RQVAE(state['K']).to(EXPORT_DTYPE)
    model.load_state_dict({k: v.to(EXPORT_DTYPE) for k, v in state['state_dict'].items()})
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for i in range(0, len(X), EXPORT_CHUNK):
                rows = torch.from_numpy(np.ascontiguousarray(X[i:i + EXPORT_CHUNK])).to(EXPORT_DTYPE)
                q = quantize(model.encoder(rows), list(model.codebooks))
                yield rows, q, model.decoder(q['zhat'])
    finally:
        torch.set_num_threads(previous)


def export_codes(state, X):
    """Canonical codes int64 [N, 3]; ties go to the lowest index."""
    parts = [torch.stack(q['codes'], 1).numpy() for _, q, _ in _canonical_chunks(state, X)]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros((0, PREREG['levels']), np.int64)


def assign_fourth_code(ids, codes):
    """Rank inside the group sharing the first three codes (ascending item id); 0 when unique.

    Returns uint32 [N, 4] aligned with the input rows.
    """
    ids = np.asarray(ids)
    codes = np.asarray(codes)
    if ids.ndim != 1 or codes.shape != (len(ids), 3) or len(np.unique(ids)) != len(ids) or (codes < 0).any():
        raise ValueError('invalid ids/codes')
    order = np.lexsort((ids, codes[:, 2], codes[:, 1], codes[:, 0]))
    sorted_codes = codes[order]
    starts = np.r_[True, np.any(sorted_codes[1:] != sorted_codes[:-1], axis=1)]
    rows = np.arange(len(ids))
    rank = rows - np.maximum.accumulate(np.where(starts, rows, 0))
    out = np.zeros((len(ids), 4), dtype=np.uint32)
    out[order, :3] = sorted_codes
    out[order, 3] = rank
    return out


def codebook_usage(codes, K):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != PREREG['levels'] or codes.min() < 0 or codes.max() >= K:
        raise ValueError('codes out of range')
    return [int(len(np.unique(codes[:, d]))) for d in range(codes.shape[1])]


def usage_gate(used, K):
    numerator, denominator = PREREG['usage_gate_num_den']
    return used * denominator >= K * numerator


def reconstruction_report(state, X):
    """mse = mean row squared error; sst = mean row squared distance to the mean; residual_norms = r0..r3."""
    total_error, norm_sums, count = 0.0, np.zeros(PREREG['levels'] + 1), 0
    for rows, q, xhat in _canonical_chunks(state, X):
        total_error += float(((rows - xhat) ** 2).sum())
        norm_sums += np.array([float(r.norm(dim=1).sum()) for r in q['residuals']])
        count += len(rows)
    data = np.asarray(X, dtype=np.float64)
    mse = total_error / count
    sst = float(((data - data.mean(0)) ** 2).sum()) / count
    return {'mse': mse, 'sst': sst, 'r2': 1 - mse / sst, 'residual_norms': (norm_sums / count).tolist()}
