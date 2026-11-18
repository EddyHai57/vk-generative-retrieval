"""Training-only, fixed-query baselines with one shared seen-item filter.

Labels are consumed from the foundation sequences. This module never derives
labels or reads validation targets. Ranking ties use the historical DuckDB hash.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path

import duckdb
import numpy as np

from . import evaluation

PADDING = np.iinfo(np.uint32).max
TIE_SEED = 20261108
CANDIDATE_PREPARATION = 'full_ranking_filtered_before_top1000'


def resolve_dataset(foundation):
    """Use the explicit dataset or its parent's conventional dataset child."""
    root = Path(foundation)
    return root / 'dataset' if (root / 'dataset').is_dir() else root


@dataclass
class BaselineData:
    queries: dict
    items: dict
    history_users: np.ndarray
    history_offsets: np.ndarray
    history_items: np.ndarray

    @classmethod
    def load(cls, foundation):
        arrays = resolve_dataset(foundation) / 'arrays'
        with np.load(arrays / 'queries.npz') as archive:
            queries = {key: archive[key] for key in archive.files}
        with np.load(arrays / 'items.npz') as archive:
            items = {key: archive[key] for key in archive.files}
        with np.load(arrays / 'history.npz') as archive:
            users, offsets = archive['user_id'], archive['offsets']
        return cls(queries, items, users, offsets,
                   np.load(arrays / 'history_items.npy', mmap_mode='r'))

    def seen(self, user_id):
        row = np.searchsorted(self.history_users, user_id)
        if row == len(self.history_users) or self.history_users[row] != user_id:
            return self.history_items[:0]
        return self.history_items[self.history_offsets[row]:self.history_offsets[row + 1]]


def _filtered(data, row, ranked, limit, catalog):
    return evaluation.filter_ranked_items(
        ranked, data.seen(data.queries['user_id'][row]), catalog, limit)


def _empty(data, limit):
    if not 0 <= limit <= 1000:
        raise ValueError('limit must be between zero and 1000')
    return np.full((len(data.queries['user_id']), limit), PADDING, dtype=np.uint32)


def popularity_order(items):
    """Return all metadata rows ranked by count, frozen tie hash, and item ID."""
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as connection:
        connection.register('catalog', {key: items[key] for key in ('item_id', 'train_count')})
        return connection.execute(f'''SELECT item_id,hash(item_id,{TIE_SEED}::UBIGINT) AS tie_key
          FROM catalog ORDER BY train_count DESC,tie_key,item_id''').fetchnumpy()


def global_popularity(data, limit=1000):
    result = _empty(data, limit)
    catalog = evaluation.CatalogMembership(data.items['item_id'])
    ranked = popularity_order(data.items)['item_id']
    for row in range(len(result)):
        keep = _filtered(data, row, ranked, limit, catalog)
        result[row, :len(keep)] = keep
    return result


def last_author_popularity(data, limit=1000):
    result = _empty(data, limit)
    catalog = evaluation.CatalogMembership(data.items['item_id'])
    ranked = popularity_order(data.items)['item_id']
    positions = np.searchsorted(data.items['item_id'], ranked)
    authors = data.items['author_id'][positions]
    order = np.argsort(authors, kind='stable')
    author_ids, offsets = np.unique(authors[order], return_index=True)
    offsets = np.r_[offsets, len(order)]
    for row, author in enumerate(data.queries['last_author_id']):
        found = np.searchsorted(author_ids, author)
        pool = ranked[:0]
        if data.queries['history_count'][row] and found < len(author_ids) and author_ids[found] == author:
            pool = ranked[order[offsets[found]:offsets[found + 1]]]
        keep = _filtered(data, row, pool, limit, catalog)
        result[row, :len(keep)] = keep
    return result


def build_covisitation_graph(sequences, train_counts):
    """Small-catalog reference graph used for independent definition tests."""
    pair_users = defaultdict(set)
    for sequence in sequences:
        items = sequence['item_ids']
        for left, right in zip(items, items[1:]):
            if left != right:
                pair_users[tuple(sorted((int(left), int(right))))].add(int(sequence['user_id']))
    edges = []
    for (left, right), users in sorted(pair_users.items()):
        if len(users) >= 2:
            score = len(users) / np.sqrt(train_counts[left] * train_counts[right])
            edges.extend([(left, right, len(users), score), (right, left, len(users), score)])
    edges.sort()
    return {name: np.asarray([edge[column] for edge in edges], dtype=dtype)
            for column, (name, dtype) in enumerate((('anchor_item_id', np.uint32),
                 ('neighbor_item_id', np.uint32), ('shared_users', np.uint32), ('score', np.float64)))}


def covisitation_neighbors(data, graph, limit=1000):
    result = _empty(data, limit)
    catalog = evaluation.CatalogMembership(data.items['item_id'])
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as connection:
        connection.register('neighbors', graph)
        ranked = connection.execute(f'''SELECT anchor_item_id,neighbor_item_id FROM neighbors
          WHERE shared_users>=2 AND score>0
          ORDER BY anchor_item_id,score DESC,hash(neighbor_item_id,{TIE_SEED}::UBIGINT),neighbor_item_id''').fetchnumpy()
    anchors, offsets = np.unique(ranked['anchor_item_id'], return_index=True)
    offsets = np.r_[offsets, len(ranked['anchor_item_id'])]
    for row, item in enumerate(data.queries['last_item_id']):
        found = np.searchsorted(anchors, item)
        pool = ranked['neighbor_item_id'][:0]
        if data.queries['history_count'][row] and found < len(anchors) and anchors[found] == item:
            pool = ranked['neighbor_item_id'][offsets[found]:offsets[found + 1]]
        keep = _filtered(data, row, pool, limit, catalog)
        result[row, :len(keep)] = keep
    return result


def unit(vectors):
    value = np.array(vectors, dtype=np.float32, order='C', copy=True)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    np.divide(value, norm, out=value, where=norm > 0)
    return value


def sequence_rows(foundation):
    """Stream only derived training sequences, one user at a time."""
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as connection:
        paths = sorted((resolve_dataset(foundation) / 'sequences').glob('user_block=*/users.parquet'))
        for path in paths:
            cursor = connection.execute('SELECT user_id,item_ids,main_positive FROM read_parquet(?)', [str(path)])
            while rows := cursor.fetchmany(256):
                for user_id, item_ids, positive in rows:
                    yield dict(user_id=user_id, item_ids=item_ids, main_positive=positive)


def content_profiles(query_users, sequences, vector_item_ids, vectors, recent=200):
    """Latest ALL exposures first, then main ruler bit, irrespective of place.

    Reverse chronological accumulation matches the original float64 prepare
    implementation before its float32 cast and final unit normalization.
    """
    query_users = np.asarray(query_users)
    profiles = np.zeros((len(query_users), vectors.shape[1]), dtype=np.float64)
    counts = np.zeros(len(query_users), dtype=np.uint16)
    for sequence in sequences:
        row = np.searchsorted(query_users, sequence['user_id'])
        if row == len(query_users) or query_users[row] != sequence['user_id']:
            continue
        items = np.asarray(sequence['item_ids'][-recent:][::-1])
        positive = np.asarray(sequence['main_positive'][-recent:][::-1], dtype=bool)
        selected = items[positive]
        if not len(selected):
            continue
        locations = np.searchsorted(vector_item_ids, selected)
        if np.any(locations >= len(vector_item_ids)) or not np.array_equal(vector_item_ids[locations], selected):
            raise ValueError('history item absent from vector catalog')
        # add.at retains chronological accumulation order, unlike pairwise sum.
        np.add.at(profiles, np.full(len(selected), row), unit(vectors[locations]))
        counts[row] = len(selected)
    return unit(profiles), counts


def index_from_centroids(centroids, vectors, batch_size=131072, nprobe=64):
    """Restore a frozen inner-product IVFFlat quantizer without fitting it."""
    import faiss
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    quantizer = faiss.IndexFlatIP(centroids.shape[1])
    quantizer.add(centroids)
    index = faiss.IndexIVFFlat(quantizer, centroids.shape[1], len(centroids), faiss.METRIC_INNER_PRODUCT)
    index.is_trained = True
    index.nprobe = nprobe
    vector_path = Path(vectors) if isinstance(vectors, (str, Path)) else None
    source = np.load(vector_path, mmap_mode='r') if vector_path else vectors
    total = len(source)
    if vector_path:
        source._mmap.close()
    for begin in range(0, total, batch_size):
        source = np.load(vector_path, mmap_mode='r') if vector_path else vectors
        block = unit(source[begin:begin + batch_size])
        if vector_path:
            source._mmap.close()
        index.add(block)
    if index.ntotal != total:
        raise ValueError('restored index does not contain full vector catalog')
    return index


def retrieve_content(data, profiles, index, catalog_ids, initial_k=4096, max_k=131072, limit=1000, batch_size=128):
    """Search the fixed index with bounded candidate refill and no fallback."""
    result = _empty(data, limit)
    if len(profiles) != len(result) or index.ntotal != len(catalog_ids):
        raise ValueError('profile/query or index/catalog mismatch')
    catalog = evaluation.CatalogMembership(data.items['item_id'])
    active = np.flatnonzero(np.linalg.norm(profiles, axis=1) > 0)
    for begin in range(0, len(active), batch_size):
        pending = active[begin:begin + batch_size]
        k = min(initial_k, len(catalog_ids))
        while len(pending):
            _, positions = index.search(np.ascontiguousarray(profiles[pending]), k)
            refill = []
            for row, locations in zip(pending, positions):
                pool = catalog_ids[locations[locations >= 0]]
                keep = _filtered(data, row, pool, limit, catalog)
                result[row] = PADDING
                result[row, :len(keep)] = keep
                if len(keep) < limit:
                    refill.append(row)
            if k >= min(max_k, len(catalog_ids)):
                break
            pending = np.asarray(refill, dtype=np.int64)
            k = min(k * 2, max_k, len(catalog_ids))
    return result


def source_binding(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def audit_reused_graph(data, foundation, graph_path, sample_size=32):
    """Independently recompute sampled complete anchor neighborhoods.

    The sampled anchors are chosen before inspecting labels or targets. Every
    training sequence contributes adjacency, including cross-week boundaries.
    Marginals count ALL training exposures. Source hashes bind the exact graph
    and new sequence bytes used by this read-only audit.
    """
    sequence_paths = sorted((resolve_dataset(foundation) / 'sequences').glob('user_block=*/users.parquet'))
    if not sequence_paths:
        raise ValueError('no foundation sequences')
    with duckdb.connect(config={'threads': 1, 'memory_limit': '1GB'}) as connection:
        connection.from_parquet([str(path) for path in sequence_paths]).create_view('sequences')
        connection.from_parquet(str(graph_path)).create_view('graph')
        connection.register('catalog', {key: data.items[key] for key in ('item_id', 'train_count')})
        count, distinct, bad = connection.execute('''SELECT count(*),count(DISTINCT (anchor_item_id,neighbor_item_id)),
          count(*) FILTER(WHERE anchor_item_id=neighbor_item_id OR shared_users<2 OR score<=0 OR NOT isfinite(score)) FROM graph''').fetchone()
        if count != distinct or bad:
            raise ValueError('invalid graph edge structure')
        mismatches = connection.execute('''SELECT count(*) FROM graph g
          LEFT JOIN catalog a ON g.anchor_item_id=a.item_id
          LEFT JOIN catalog b ON g.neighbor_item_id=b.item_id
          WHERE a.train_count IS NULL OR b.train_count IS NULL OR a.train_count<=0 OR b.train_count<=0
          OR abs(g.score-g.shared_users/sqrt(a.train_count::DOUBLE*b.train_count::DOUBLE))>1e-12''').fetchone()[0]
        if mismatches:
            raise ValueError('graph scores disagree with new training marginals')
        connection.execute(f'''CREATE TEMP TABLE anchors AS SELECT DISTINCT anchor_item_id AS item_id FROM graph
          ORDER BY hash(anchor_item_id,{TIE_SEED}::UBIGINT),anchor_item_id LIMIT {int(sample_size)}''')
        connection.execute('''CREATE TEMP TABLE direct_pairs AS WITH expanded AS (
          SELECT user_id,unnest(item_ids) AS a,unnest(item_ids[2:]) AS b FROM sequences),
          candidates AS MATERIALIZED (SELECT * FROM expanded WHERE b IS NOT NULL
            AND (a IN(SELECT item_id FROM anchors) OR b IN(SELECT item_id FROM anchors))),
          directed AS (SELECT user_id,a AS anchor_item_id,b AS neighbor_item_id FROM candidates
            UNION ALL SELECT user_id,b,a FROM candidates)
          SELECT anchor_item_id,neighbor_item_id,count(DISTINCT user_id) AS shared_users FROM directed
          WHERE anchor_item_id IN(SELECT item_id FROM anchors) AND anchor_item_id<>neighbor_item_id
          GROUP BY anchor_item_id,neighbor_item_id HAVING count(DISTINCT user_id)>=2''')
        differences = connection.execute('''SELECT count(*) FROM (
          (SELECT anchor_item_id,neighbor_item_id,shared_users FROM direct_pairs
           EXCEPT SELECT anchor_item_id,neighbor_item_id,shared_users FROM graph WHERE anchor_item_id IN(SELECT item_id FROM anchors))
          UNION ALL
          (SELECT anchor_item_id,neighbor_item_id,shared_users FROM graph WHERE anchor_item_id IN(SELECT item_id FROM anchors)
           EXCEPT SELECT anchor_item_id,neighbor_item_id,shared_users FROM direct_pairs))''').fetchone()[0]
        if differences:
            raise ValueError('recomputed adjacency supports disagree with reused graph')
        connection.execute('''CREATE TEMP TABLE endpoints AS SELECT item_id FROM anchors
          UNION SELECT neighbor_item_id FROM direct_pairs''')
        marginal_mismatch = connection.execute('''WITH expanded AS(SELECT unnest(item_ids) AS item_id FROM sequences),
          counts AS(SELECT item_id,count(*) AS n FROM expanded WHERE item_id IN(SELECT item_id FROM endpoints) GROUP BY item_id)
          SELECT count(*) FROM endpoints e LEFT JOIN counts n USING(item_id) LEFT JOIN catalog c USING(item_id)
          WHERE coalesce(n.n,0)<>c.train_count''').fetchone()[0]
        if marginal_mismatch:
            raise ValueError('new sequence marginals disagree with new item counts')
        connection.register('queries', {key: data.queries[key] for key in
            ('user_id', 'history_count', 'last_item_id', 'last_author_id')})
        query_mismatch = connection.execute('''SELECT count(*) FROM queries q LEFT JOIN sequences s USING(user_id)
          WHERE q.history_count<>coalesce(len(s.item_ids),0)
          OR (q.history_count>0 AND (q.last_item_id<>s.item_ids[-1] OR q.last_author_id<>s.authors[-1]))''').fetchone()[0]
        if query_mismatch:
            raise ValueError('fixed queries disagree with train-last sequence anchors')
        training_exposures = connection.execute('SELECT sum(len(item_ids)) FROM sequences').fetchone()[0]
        if training_exposures != int(np.asarray(data.items['train_count']).sum()):
            raise ValueError('new sequence exposure total disagrees with item count total')
        sampled_anchors = connection.execute('SELECT count(*) FROM anchors').fetchone()[0]
        sampled_edges = connection.execute('SELECT count(*) FROM direct_pairs').fetchone()[0]
    return dict(status='verified', graph_edges=count, sampled_anchors=sampled_anchors,
                sampled_edges=sampled_edges, training_exposures=training_exposures,
                query_anchors_checked=len(data.queries['user_id']),
                method='full training adjacency, undirected, distinct user support>=2; sampled complete anchor neighborhoods and endpoint marginals',
                sources=[source_binding(path) for path in [Path(graph_path), *sequence_paths]])
