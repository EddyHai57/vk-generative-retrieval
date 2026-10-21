"""Exact external-sort adapter for the canonical BHKM implementation.

The caller supplies the frozen SID module so SIDCatalog and its audit code
remain owned by that module.  This module only replaces assignment's edge
materialization with sorted disk runs and a serial merge.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import heapq
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys

import numpy as np


_EDGE = np.dtype([("distance", "<f8"), ("item_id", "<u4"), ("child", "<u4")], align=False)
_EDGE_BYTES = 16


def _distances(rows, centers):
    """Keep the original float64, dimension-by-dimension accumulation order."""
    distances = np.zeros((len(rows), len(centers)), dtype=np.float64)
    for dimension in range(rows.shape[1]):
        delta = rows[:, dimension, None].astype(np.float64) - centers[None, :, dimension]
        distances += delta * delta
    return distances


def _write_sorted_run(task):
    block, item_ids, centers, path, keep_distances, reservation = task
    distances = _distances(block, centers)
    k = len(centers)
    row = np.repeat(np.arange(len(block)), k)
    child = np.tile(np.arange(k), len(block))
    order = np.lexsort((child, np.repeat(item_ids, k), distances.ravel()))
    run = np.empty(len(order), dtype=_EDGE)
    run["distance"] = distances.ravel()[order]
    run["item_id"] = np.repeat(item_ids, k)[order]
    run["child"] = child[order]
    np.save(path, run, allow_pickle=False)
    os.chmod(path, 0o444)
    return str(path), distances if keep_distances else None, reservation


def _bounded_submit(executor, function, values, max_pending):
    """Submit at most max_pending jobs and yield results in input order."""
    iterator = iter(values)
    pending = deque()
    exhausted = False
    while pending or not exhausted:
        while len(pending) < max_pending and not exhausted:
            try:
                value = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.append(executor.submit(function, value))
        if pending:
            yield pending.popleft().result()


def _minimum_memory_limit(*limits):
    available = [int(value) for value in limits if value is not None and int(value) > 0]
    return min(available) if available else None


def _total_memory_bytes():
    host = None
    if hasattr(os, "sysconf"):
        try:
            host = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    limits = []
    paths = [Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")]
    membership = Path("/proc/self/cgroup")
    if membership.exists():
        for line in membership.read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            relative = relative.lstrip("/")
            if hierarchy == "0" and not controllers:
                paths.append(Path("/sys/fs/cgroup") / relative / "memory.max")
            elif "memory" in controllers.split(","):
                paths.append(Path("/sys/fs/cgroup/memory") / relative / "memory.limit_in_bytes")
    for path in paths:
        try:
            value = path.read_text().strip()
            if value.isdigit():
                limit = int(value)
                # cgroup v1 reports a very large sentinel when no cap is set.
                if 0 < limit < 1 << 60:
                    limits.append(limit)
        except OSError:
            continue
    return _minimum_memory_limit(host, *limits)


def _rss_bytes():
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _guard(directory, *, memory_limit_bytes, disk_limit_bytes, projected_memory=0,
           projected_disk=0, operation_bytes=0):
    total_memory = _total_memory_bytes()
    if total_memory is not None and _rss_bytes() + projected_memory > total_memory * 0.60:
        raise MemoryError("registered 60% memory limit exceeded")
    if memory_limit_bytes and _rss_bytes() + projected_memory > memory_limit_bytes:
        raise MemoryError("configured memory limit exceeded")
    usage = shutil.disk_usage(directory)
    if usage.used + projected_disk > usage.total * 0.80:
        raise OSError("registered 80% disk limit exceeded")
    if disk_limit_bytes and usage.used + projected_disk > disk_limit_bytes:
        raise OSError("configured disk limit exceeded")


def _capacity_assign(runs, ids, k):
    n = len(ids)
    capacity = np.full(k, n // k, dtype=np.int64)
    capacity[:n % k] += 1
    labels = np.full(n, -1, dtype=np.int64)
    remaining = n
    mapped = [np.load(path, mmap_mode="r", allow_pickle=False) for path in runs]
    heap = []
    for run_index, run in enumerate(mapped):
        if len(run):
            record = run[0]
            heapq.heappush(heap, (float(record["distance"]), int(record["item_id"]),
                                  int(record["child"]), run_index, 0))
    while heap and remaining:
        _, item_id, child, run_index, offset = heapq.heappop(heap)
        row = int(np.searchsorted(ids, item_id))
        if labels[row] < 0 and capacity[child] > 0:
            labels[row] = child
            capacity[child] -= 1
            remaining -= 1
        offset += 1
        if offset < len(mapped[run_index]):
            record = mapped[run_index][offset]
            heapq.heappush(heap, (float(record["distance"]), int(record["item_id"]),
                                  int(record["child"]), run_index, offset))
    if remaining:
        raise RuntimeError("external greedy assignment left rows unassigned")
    return labels


def _projection(n, k, block_rows, workers, prefetch):
    # Per edge: distance, run record, row/child/order, repeated item key, and
    # conservative lexsort workspace. Also reserve each serialized block.
    rows = min(n, block_rows)
    edge_bytes = rows * k * 96
    input_bytes = rows * (64 * 4 + 8) + k * 64 * 8
    queue_slots = workers * prefetch if workers > 1 else 1
    memory = queue_slots * (edge_bytes + input_bytes)
    if workers > 1:
        # Include a process baseline per worker; the parent RSS is checked by _guard.
        memory += workers * max(_rss_bytes(), 64 * 1024 * 1024)
    disk = n * k * _EDGE_BYTES + 256 * ((n + block_rows - 1) // block_rows)
    return memory, disk


def _trace_memory_bytes(n, branching, depth, max_iter):
    rounds = max_iter * sum(min(n, branching**level) for level in range(depth))
    rows_by_level = n * depth * max_iter
    retained = 2 * rows_by_level * branching * 8  # assignment + objective distances
    retained += rows_by_level * 8  # labels
    retained += rows_by_level * 8  # previous-label convergence snapshots
    retained += 2 * rows_by_level * 64 * 8  # centers before/after each round
    retained += n * depth * 64 * 32  # final node centers as Python lists of floats
    retained += n * (depth + 1) * 4  # retained SID copy
    retained += rounds * 1024  # dict/list/prefix and numpy object overhead
    transient = n * branching * 8  # concatenation while distance chunks are still held
    return retained + transient


def fit(original_module, ids, features, *, root, workers=1, prefetch=1, block_rows=8192,
        branching=16, depth=3, seed=42, max_iter=10, trace=False,
        disk_limit_bytes=0, memory_limit_bytes=0, on_boundary=None):
    """Build a SID catalog with exact external greedy assignment.

    Returns ``(catalog, trace_or_none)``.  ``root`` must be new; run files are
    immutable and retained for independent verification and same-seed replay.
    """
    ids = np.asarray(ids)
    features = np.asarray(features)
    allocation_root=Path(root).parent
    while not allocation_root.exists()and allocation_root!=allocation_root.parent:allocation_root=allocation_root.parent
    _guard(allocation_root,memory_limit_bytes=memory_limit_bytes,disk_limit_bytes=disk_limit_bytes,
           projected_memory=features.nbytes*3+ids.nbytes*8)
    if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= 2**32) or len(np.unique(ids)) != len(ids)):
        raise ValueError("invalid catalog IDs")
    if features.dtype != np.float32 or features.shape != (len(ids), 64) or not np.isfinite(features).all():
        raise ValueError("invalid normalized features")
    norms = np.linalg.norm(features, axis=1)
    if not np.all((norms == 0) | np.isclose(norms, 1, rtol=1e-5, atol=1e-6)):
        raise ValueError("features must match normalized P4 features")
    if any(type(v) is not int or v < 1 for v in (branching, depth, max_iter, workers, prefetch, block_rows)):
        raise ValueError("invalid tokenizer or external-sort config")
    if branching < 2 or depth > 8 or branching**depth * len(ids) > 2**63:
        raise ValueError("invalid tokenizer config")
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid tokenizer seed")
    if type(trace) is not bool or any(type(v) is not int or v < 0 for v in (disk_limit_bytes, memory_limit_bytes)):
        raise ValueError("invalid trace or resource limit")

    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    ids_order = np.argsort(ids)
    ids = ids[ids_order].astype(np.uint32)
    features = features[ids_order]
    # Conservative whole-fit bound accounts for every depth and iteration.
    max_disk = len(ids) * branching * _EDGE_BYTES * depth * max_iter
    # For each depth, total block count is at most N / block_rows plus the
    # number of nonempty nodes; include one header per run in the bound.
    max_runs = sum((len(ids) + block_rows - 1) // block_rows + min(len(ids), branching**level)
                   for level in range(depth)) * max_iter
    max_disk += 256 * max_runs
    startup_memory, _ = _projection(len(ids), branching, block_rows, workers, prefetch)
    trace_memory = _trace_memory_bytes(len(ids), branching, depth, max_iter) if trace else 0
    _guard(root.parent, memory_limit_bytes=memory_limit_bytes, disk_limit_bytes=disk_limit_bytes,
           projected_memory=features.nbytes * (depth + 2) + startup_memory + trace_memory,
           projected_disk=max_disk)
    root.mkdir(exist_ok=False)

    sids = np.zeros((len(ids), depth + 1), dtype=np.uint32)
    nodes = []
    rng = np.random.default_rng(seed)
    details = {"rounds": [], "distances": [], "nodes": nodes} if trace else None
    next_node = 0
    bytes_written = 0
    trace_committed = 0

    def visit(indices, prefix):
        nonlocal next_node, bytes_written, trace_committed
        if on_boundary is not None:on_boundary()
        level = len(prefix)
        if level == depth:
            sids[indices, -1] = np.arange(len(indices))
            return
        node_id = next_node
        next_node += 1
        k = min(branching, len(indices))
        rows = features[indices]
        init = rng.choice(len(indices), k, replace=False)
        centers = rows[init].astype(np.float64)
        previous = None
        objectives = []
        converged = False
        node_rounds = 0
        for iteration in range(max_iter):
            if on_boundary is not None:on_boundary()
            centers_before = centers.copy() if trace else None
            projected_memory, projected_disk = _projection(len(rows), k, block_rows, workers, prefetch)
            _guard(root, memory_limit_bytes=memory_limit_bytes, disk_limit_bytes=disk_limit_bytes,
                   projected_memory=projected_memory + max(0, trace_memory - trace_committed),
                   projected_disk=projected_disk,
                   operation_bytes=bytes_written)
            node_dir = root / f"node-{node_id:08d}"
            node_dir.mkdir(exist_ok=True)
            submitted_disk = 0

            def jobs():
                nonlocal submitted_disk
                for block_index, start in enumerate(range(0, len(rows), block_rows)):
                    stop = min(start + block_rows, len(rows))
                    path = node_dir / f"iter-{iteration:04d}-block-{block_index:08d}.npy"
                    reservation = (stop - start) * k * _EDGE_BYTES + 256
                    _guard(root, memory_limit_bytes=memory_limit_bytes, disk_limit_bytes=disk_limit_bytes,
                           projected_memory=projected_memory + max(0, trace_memory - trace_committed),
                           projected_disk=projected_disk,
                           operation_bytes=bytes_written + submitted_disk)
                    submitted_disk += reservation
                    yield (rows[start:stop], ids[indices[start:stop]], centers, path, trace, reservation)

            written = []
            if workers == 1:
                completed = (_write_sorted_run(job) for job in jobs())
                for entry in completed:
                    written.append(entry)
                    submitted_disk -= entry[2]
                    bytes_written += Path(entry[0]).stat().st_size
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                    for entry in _bounded_submit(pool, _write_sorted_run, jobs(), workers * prefetch):
                        written.append(entry)
                        submitted_disk -= entry[2]
                        bytes_written += Path(entry[0]).stat().st_size
            runs = [entry[0] for entry in written]
            _guard(root, memory_limit_bytes=memory_limit_bytes, disk_limit_bytes=disk_limit_bytes,
                   projected_memory=projected_memory + max(0, trace_memory - trace_committed),
                   operation_bytes=bytes_written)
            distances = (np.concatenate([entry[1] for entry in written], axis=0) if trace else None)
            del written  # release per-block trace matrices after concatenation
            labels = _capacity_assign(runs, ids[indices], k)
            centers = np.stack([rows[labels == j].mean(axis=0, dtype=np.float64) for j in range(k)])
            _guard(root, memory_limit_bytes=memory_limit_bytes, disk_limit_bytes=disk_limit_bytes,
                   projected_memory=projected_memory + max(0, trace_memory - trace_committed),
                   operation_bytes=bytes_written)
            objective_distances = np.empty((len(rows), k), dtype=np.float64) if trace else None
            selected = np.empty(len(rows), dtype=np.float64)
            for start in range(0, len(rows), block_rows):
                stop = min(start + block_rows, len(rows))
                block_distances = _distances(rows[start:stop], centers)
                selected[start:stop] = block_distances[np.arange(stop - start), labels[start:stop]]
                if trace:
                    objective_distances[start:stop] = block_distances
            objectives.append(float(selected.sum(dtype=np.float64)))
            node_rounds += 1
            round_converged = previous is not None and np.array_equal(labels, previous)
            if trace:
                details["rounds"].append({"prefix": list(prefix), "iteration": iteration,
                                           "centers_before": centers_before, "centers": centers.copy(),
                                           "labels": labels.copy(), "objective": objectives[-1],
                                           "converged": round_converged})
                details["distances"].extend([distances, objective_distances])
                trace_committed += (2 * len(rows) * k * 8 + len(rows) * 8 + 2 * k * 64 * 8 + 1024)
            if round_converged:
                converged = True
                break
            previous = labels.copy()
        sizes = np.bincount(labels, minlength=k).tolist()
        node = {"prefix": list(prefix), "init_item_ids": ids[indices[init]].astype(int).tolist(),
                "centers": centers.tolist(), "sizes": sizes, "iterations": node_rounds,
                "converged": converged, "objectives": objectives}
        nodes.append(node)
        for child in range(k):
            selected_indices = indices[labels == child]
            sids[selected_indices, level] = child
            visit(selected_indices, prefix + (child,))

    visit(np.arange(len(ids)), ())
    catalog = original_module.SIDCatalog(ids, sids, branching, depth, nodes)
    if details is not None:
        details["sids"] = sids.copy()
        details["audit"] = catalog.audit()
    return catalog, details


def fit_configured(original_module,ids,features,config,*,run_id,trace_sink=None,**scientific):
    """Single engineering-config switch; original fit is the disabled default."""
    if not config.bhkm_external:
        if trace_sink is not None:raise ValueError('trace_sink requires external diagnostic mode')
        return original_module.fit(ids,features,**scientific)
    if not config.temp_dir or not isinstance(run_id,str)or not run_id or '/'in run_id or run_id in ('.','..'):
        raise ValueError('fresh run_id and configured absolute temp_dir required')
    root=Path(config.temp_dir)/run_id
    if any(p.is_symlink()for p in (root,*root.parents)):raise ValueError('temporary root may not traverse symlinks')
    catalog,trace=fit(original_module,ids,features,root=root,workers=config.bhkm_workers,prefetch=config.prefetch,
        block_rows=config.bhkm_rows,trace=trace_sink is not None,disk_limit_bytes=config.disk_limit_bytes,memory_limit_bytes=config.memory_limit_bytes,**scientific)
    if trace_sink is not None:trace_sink.update(trace)
    return catalog
