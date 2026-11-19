"""Spawn CPU rankers with shared, read-only catalog and ANN files."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import partial
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace

import numpy as np

from .generator_decode import GeneratorRanker
from .generator_eval import prepare_ranker


_BUNDLE_ARRAYS = (
    "ids", "bucket_keys", "bucket_offsets", "bucket_items", "item_bucket", "answer_counts",
)
_THREAD_ENV = dict.fromkeys((
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
), "1")
_spawn_lock = Lock()
_worker_rank = None


@contextmanager
def _spawn_environment():
    # Some BLAS libraries read their limits at import, before the initializer.
    with _spawn_lock:
        original = {name: os.environ.get(name) for name in _THREAD_ENV}
        os.environ.update(_THREAD_ENV)
        try:
            yield
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _stage_array(array, destination):
    array = np.asarray(array)
    base = array.base
    while isinstance(base, np.ndarray):
        if (isinstance(base, np.memmap) and base.filename is not None
                and array.ctypes.data == base.ctypes.data):
            source = Path(base.filename)
            if source.suffix == ".npy":
                try:
                    original = np.load(source, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError, EOFError):
                    break
                if (array.shape == original.shape and array.dtype == original.dtype
                        and array.strides == original.strides and base.offset == original.offset):
                    return source.resolve()
        base = base.base
    np.save(destination, array, allow_pickle=False)
    return destination


def _open_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _initialize_worker(bundle_size, bundle_paths, ranker_paths, config, routes,
                       content_pool_path, content_index_path):
    global _worker_rank
    os.environ.update(_THREAD_ENV)
    from threadpoolctl import threadpool_limits

    bundle = SimpleNamespace(size=bundle_size, **{
        name: _open_array(path) for name, path in bundle_paths.items()
    })
    vectors, train_count, target_count, ties = map(_open_array, ranker_paths)
    ranker = GeneratorRanker(bundle, vectors, train_count, target_count, config, ties=ties)
    if content_pool_path is not None:
        ranker._content_pool = _open_array(content_pool_path)
    if content_index_path is not None:
        import faiss

        faiss.omp_set_num_threads(1)
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        ranker.content_index = faiss.read_index(str(content_index_path), flags)
    threadpool_limits(limits=1)
    _worker_rank = partial(ranker.rank, routes=routes)


def _rank_entry(entry):
    return None if entry is None else _worker_rank(*entry)


def _portable_entry(entry):
    # rank also accepts seen-item iterators, which cannot cross a spawn queue.
    if entry is None or isinstance(entry[-1], np.ndarray):
        return entry
    return (*entry[:-1], list(entry[-1]))


class RankPool:
    """Rank query batches in input order using a prepared GeneratorRanker.

    Catalog and index inputs, including copy-on-write NPY mappings, are treated
    as immutable for the lifetime of the pool.
    """

    def __init__(self, ranker, routes, workers=4, *, temp_directory=None):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self._ranker = ranker
        self._routes = tuple(routes)
        self._workers = workers
        self._temp_directory = temp_directory
        self._resources = None
        self._executor = None

    def __enter__(self):
        if self._resources is not None:
            raise RuntimeError("RankPool is already open")
        with ExitStack() as startup:
            if self._workers > 1:
                self._executor = self._start(startup)
            self._resources = startup.pop_all()
        return self

    def _start(self, startup):
        ranker, routes = self._ranker, self._routes
        prepare_ranker(ranker, routes)
        directory = startup.enter_context(TemporaryDirectory(
            prefix="generator-rank-", dir=self._temp_directory,
        ))
        root = Path(directory).resolve()
        bundle = ranker.bundle
        bundle_paths = {
            name: _stage_array(getattr(bundle, name), root / f"{name}.npy")
            for name in _BUNDLE_ARRAYS
        }
        ranker_paths = tuple(
            _stage_array(getattr(ranker, name), root / f"{name}.npy")
            for name in ("vectors", "train_count", "target_count", "ties")
        )
        content_pool_path = content_index_path = None
        if "content" in routes:
            content_pool_path = _stage_array(ranker._content_pool, root / "content_pool.npy")
            if ranker.content_index is not None:
                import faiss

                content_index_path = root / "content.faiss"
                faiss.write_index(ranker.content_index, str(content_index_path))
        return startup.enter_context(ProcessPoolExecutor(
            max_workers=self._workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(bundle.size, bundle_paths, ranker_paths, ranker.config, routes,
                      content_pool_path, content_index_path),
        ))

    def rank(self, entries):
        """Return one route result dictionary or None for each input entry."""
        if self._resources is None:
            raise RuntimeError("RankPool must be used as a context manager")
        if self._executor is None:
            return [None if entry is None else self._ranker.rank(*entry, routes=self._routes)
                    for entry in entries]
        jobs = list(map(_portable_entry, entries))
        with _spawn_environment():
            results = self._executor.map(_rank_entry, jobs)
        return list(results)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self._resources.__exit__(exc_type, exc_value, traceback)
        finally:
            self._resources = None
            self._executor = None
