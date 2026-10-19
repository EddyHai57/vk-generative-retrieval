"""Frozen RQ cost components; pure arithmetic, no job submission or training.

The formal benchmark remains on hold. This module is not yet wired into its
worker or report. Historical acceptance must precede any reported extrapolation.
"""
import ast
import hashlib
import math
import os
import sys
import time
from pathlib import Path

PINS = {
    'src/vkgr/sid_rqvae.py': '52173bea9b80ffeb40e25943e3d7069ec53f9bd34a4509c97278194714065067',
    'src/vkgr/sid_rqvae_r.py': 'c786848ab85c6ce2a593ccdfc626688484ae58d2ee0406c889a5ad48c5f54392',
}

def positive_ints(*values):
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('positive plain integer dimensions and counts required')

def positive_seconds(*values):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('positive finite timing required')

def compose(*, n, steps, usage_interval, k, d, levels,
            core_seconds_per_step, assign_seconds_per_unit):
    positive_ints(n, steps, usage_interval, k, d, levels)
    positive_seconds(core_seconds_per_step, assign_seconds_per_unit)
    passes = steps // usage_interval
    usage = n * k * d * levels * assign_seconds_per_unit
    core = steps * core_seconds_per_step
    return {'kind':'composed_from_components', 'n':n, 'steps':steps,
            'usage_interval':usage_interval, 'k':k, 'd':d, 'levels':levels,
            'n_usage':passes, 'usage_item_forwards':n * passes,
            'core_seconds':core, 'usage_pass_seconds':usage,
            'usage_total_seconds':passes * usage,
            'total_fit_seconds':core + passes * usage,
            'scope':'training_core_and_periodic_usage_excludes_setup_and_export',
            'historical_selfcheck_required':True,
            'usage_to_core_wall_ratio':passes * usage / core,
            'usage_wall_fraction':passes * usage / (core + passes * usage),
            'item_forward_count_limitation':'Item forward counts are not wall time; core includes backward/optimizer, usage only forward assignment. Wall shares use separately measured c_core and c_assign.'}

def core_constant(fitted, *, usage_interval):
    positive_ints(usage_interval)
    steps = fitted.get('step')
    positive_ints(steps)
    positive_seconds(fitted.get('elapsed'))
    if fitted.get('status') != 'complete' or steps >= usage_interval or fitted.get('usage_log') != []:
        raise ValueError('core calibration invalid: incomplete or usage-contaminated')
    return fitted['elapsed'] / steps

def usage_constant(points, *, k, d, levels):
    """Origin-constrained least squares in N, matching N*K*D*L*c_assign.

    R² uses centered total sum of squares; it may be negative. No fit-quality
    threshold is invented here. Residuals and intercept=0 are always reported.
    """
    positive_ints(k, d, levels)
    if len(points) not in (2, 3):
        raise ValueError('two or three independently measured N points required')
    for p in points:
        positive_ints(p['n']); positive_seconds(p['seconds'])
    if len({p['n'] for p in points}) != len(points):
        raise ValueError('distinct N points required')
    slope = sum(p['n'] * p['seconds'] for p in points) / sum(p['n']**2 for p in points)
    mean = sum(p['seconds'] for p in points) / len(points)
    residuals = [{**p, 'predicted_seconds':slope * p['n'],
                  'residual_seconds':p['seconds'] - slope * p['n']} for p in points]
    sse = sum(p['residual_seconds']**2 for p in residuals)
    sst = sum((p['seconds'] - mean)**2 for p in points)
    return {'assign_seconds_per_unit':slope / (k*d*levels),
            'seconds_per_item':slope, 'intercept_seconds':0.,
            'regression':'least_squares_through_origin',
            'r_squared':1 - sse/sst if sst else 'UNKNOWN', 'points':residuals}

def historical_check(predicted, observed):
    positive_seconds(predicted, observed)
    ratio = max(predicted/observed, observed/predicted)
    return {'status':'passed' if ratio <= 2 else 'failed',
            'predicted_seconds':predicted, 'observed_seconds':observed,
            'relative_error':abs(predicted-observed)/observed, 'factor_error':ratio,
            'extrapolation':'permitted_by_cost_selfcheck' if ratio <= 2 else 'UNKNOWN'}

def l2_source(root):
    """Read interval and source locations only after verifying original bytes."""
    source = {}
    for name, pin in PINS.items():
        data = (Path(root)/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise ValueError('original L2 source differs')
        source[name] = data.decode()
    core = source['src/vkgr/sid_rqvae.py']; repair = source['src/vkgr/sid_rqvae_r.py']
    tree = ast.parse(core)
    prereg = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'PREREG' for t in n.targets))
    values = ast.literal_eval(prereg)
    line = next(v.lineno for k,v in zip(prereg.keys, prereg.values) if ast.literal_eval(k) == 'eval_every')
    usage = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_usage')
    call = next(n for n in ast.walk(ast.parse(repair)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == '_usage'
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'rq')
    return {'usage_interval':values['eval_every'], 'interval_line':line,
            'files':dict(PINS), 'usage_function':'vkgr.sid_rqvae._usage',
            'usage_function_line':usage.lineno, 'periodic_call_line':call.lineno,
            'diagnostics':False, 'origin_commit':'80f7eed179981c65094c628744099a181a81d907'}

def usage_memory(*, n, k, d, levels, chunk):
    """Principal float32 CPU arrays from original chunked _usage/quantize.

    Distance expressions may coexist with temporaries. These array sizes are
    neither a summed peak bound nor a substitute for observed own-tree RSS.
    """
    positive_ints(n, k, d, levels, chunk)
    b = min(n, chunk)
    return {'scope':'principal_arrays_not_peak_rss', 'input_float32_bytes':n*64*4,
            'distance_float32_bytes':b*k*4, 'seen_bool_bytes':k*levels,
            'code_indices_int64_bytes':b*levels*8,
            'residual_float32_bytes':b*d*(levels+1)*4,
            'embeds_float32_bytes':b*d*levels*4,
            'encoder_largest_hidden_float32_bytes':b*512*4,
            'chunk_items':b}


def measure_usage_once(features, *, k, d, levels, engineering, root):
    """One direct original usage pass; no fit, reset, checkpoint or export.

    The formal parent worker is not wired yet. Formal use requires the original
    Linux queue run. Only tiny synthetic arrays may run in local engineering.
    """
    positive_ints(k, d, levels)
    if type(engineering) is not bool:
        raise ValueError('explicit engineering mode required')
    source = l2_source(root)
    if (d, levels) != (32, 3):
        raise ValueError('fixed original L2 model is D32 L3; no model alteration')
    n = len(features)
    positive_ints(n)
    if engineering:
        if n > 1024 or k > 16:
            raise ValueError('tiny local usage selftest only')
    elif sys.platform != 'linux' or os.environ.get('VKGR_RUN_ID') != 'sid-scale-cost-v1':
        raise ValueError('formal usage calibration only in original Linux queue job')
    import numpy as np
    import torch
    from vkgr import sid_rqvae as rq, sid_rqvae_r as repair
    if (not isinstance(features, np.ndarray) or features.dtype != np.float32
        or features.ndim != 2 or features.shape[1] != 64
        or not features.flags.c_contiguous or not np.isfinite(features).all()):
        raise ValueError('finite contiguous synthetic float32 [N,64] required')
    expected = (Path(root)/'src/vkgr/sid_rqvae.py').resolve()
    if Path(rq.__file__).resolve() != expected or repair.rq is not rq:
        raise ValueError('usage module differs from original L2 training path')
    if rq.PREREG['eval_every'] != source['usage_interval']:
        raise ValueError('runtime usage interval differs from fixed source')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = rq.RQVAE(k).cpu()
    values = torch.from_numpy(features)
    input_sha = hashlib.sha256(features.astype('<f4',copy=False).tobytes(order='C')).hexdigest()
    start = time.perf_counter()
    used = rq._usage(model, values)
    elapsed = time.perf_counter() - start
    positive_seconds(elapsed)
    return {'n':n, 'k':k, 'd':d, 'levels':levels, 'seconds':elapsed,
            'usage_calls':1, 'diagnostics':False, 'engineering':engineering,
            'device':'cpu', 'threads':torch.get_num_threads(),
            'function':source['usage_function'], 'source':source,
            'input_sha256':input_sha, 'usage_counts':used,
            'memory':usage_memory(n=n,k=k,d=d,levels=levels,chunk=rq.EXPORT_CHUNK),
            'timing_scope':'direct_original_usage_only_excludes_model_and_input_setup',
            'checkpoint_read':False, 'checkpoint_written':False}
