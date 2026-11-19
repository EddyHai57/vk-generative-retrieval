"""Batched legal-SID beam search and item-level generator retrieval."""

from dataclasses import dataclass

import numpy as np
import torch

from .two_tower_retrieval import top_order


class PrefixTrie:
    """Device-resident round22 CSR, with one row per packed legal prefix."""

    def __init__(self, prefixes, size, device):
        if len(prefixes) != 3 or size < 1 or size ** 3 > np.iinfo(np.int64).max:
            raise ValueError("three SID layers with int64 packed keys required")
        self.size = int(size)
        self.device = torch.empty(0, device=device).device
        self.divisors = torch.tensor(
            [self.size ** 2, self.size, 1], dtype=torch.long, device=self.device,
        )
        self.layers = []
        for keys, offsets, digits in prefixes:
            keys, offsets, digits = map(np.asarray, (keys, offsets, digits))
            counts = np.diff(offsets.astype(np.int64))
            if (not len(keys) or len(offsets) != len(keys) + 1
                    or offsets[0] != 0 or offsets[-1] != len(digits)
                    or np.any(counts <= 0) or np.any(counts > self.size)):
                raise ValueError("invalid prefix CSR")
            tensors = tuple(torch.tensor(x.astype(np.int64), device=self.device)
                            for x in (keys, offsets, digits))
            slots = torch.arange(int(counts.max()), device=self.device)
            self.layers.append((*tensors, slots))

    def allowed(self, states, position):
        keys, offsets, digits, slots = self.layers[position]
        rows = torch.searchsorted(keys, states).clamp_max(len(keys) - 1)
        indices = offsets[rows][..., None] + slots
        valid = ((keys[rows] == states)[..., None]
                 & (indices < offsets[rows + 1][..., None]))
        return digits[indices.clamp_max(len(digits) - 1)], valid


@dataclass(frozen=True)
class BeamLevel:
    keys: torch.Tensor
    logp: torch.Tensor


def _level_widths(widths, depth):
    if (not isinstance(widths, (tuple, list)) or len(widths) != depth
            or any(type(width) is not int or width < 1 for width in widths)):
        raise ValueError(f"beam requires {depth} positive integer widths")
    return tuple(widths)


def _top_beams(keys, scores, width):
    if width < scores.shape[1]:
        # The cutoff keeps selection small, but topk alone can drop the wrong
        # tied keys. Among cutoff ties, select the smallest packed keys.
        rank_scores = scores.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
        top_scores, order = torch.topk(rank_scores, width, dim=1)
        cutoff = top_scores[:, -1:]
        tie_keys = keys.masked_fill(rank_scores != cutoff, torch.iinfo(keys.dtype).max)
        tied = torch.topk(tie_keys, width, dim=1, largest=False).indices
        above = top_scores > cutoff
        tie_rank = torch.arange(width, device=scores.device) - above.sum(dim=1, keepdim=True)
        order = torch.where(above, order, tied.gather(1, tie_rank.clamp_min(0)))
        keys, scores = keys.gather(1, order), scores.gather(1, order)

    # Sort only the retained candidates by score and then by packed key.
    lexical = torch.argsort(keys, dim=1, stable=True)
    scores = scores.gather(1, lexical)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    return BeamLevel(keys.gather(1, lexical.gather(1, order)), scores.gather(1, order))


@torch.no_grad()
def beam_search(trie, score, batch_size, beam_width, depth=3):
    """Return each selected beam level, with -1/-inf padding on the device.

    The model callback scores raw digits [batch, active, position] using all
    tokens of the next layer. Pass a scalar to reuse one width or a sequence
    with one width per searched layer.
    """
    if not 1 <= depth <= 3 or batch_size < 1:
        raise ValueError("invalid beam dimensions")
    if type(beam_width) is int:
        beam_width = (beam_width,) * depth
    widths = _level_widths(beam_width, depth)
    states = torch.zeros((batch_size, 1), dtype=torch.long, device=trie.device)
    totals = torch.zeros_like(states, dtype=torch.float64)
    levels = {}
    for position in range(depth):
        prefix = states.clamp_min(0)[..., None] // trie.divisors[3 - position:] % trie.size
        logits = score(prefix)
        if logits.shape != (*states.shape, trie.size) or logits.device != trie.device:
            raise ValueError("SID score must match beam shape, vocabulary and device")
        logp = torch.log_softmax(logits.float(), dim=-1)
        digits, valid = trie.allowed(states, position)
        expanded = totals[..., None] + logp.gather(-1, digits).to(torch.float64)
        scores = expanded.masked_fill(~valid, -torch.inf).flatten(1)
        keys = (states[..., None] * trie.size + digits).masked_fill(~valid, -1).flatten(1)
        width = min(widths[position], trie.layers[position][2].numel())
        level = _top_beams(keys, scores, width)
        states, totals = level.keys, level.logp
        levels[position + 1] = level
    return levels


@dataclass(frozen=True)
class DecodeConfig:
    beam_widths: tuple[int, int, int] = (1024, 1024, 1024)
    prefix_beam_width: int = 64
    prefix_limit: int = 256
    candidate_budget: int = 2000
    k: int = 1000
    new_item_quota: int | None = None
    ranking: str = "item"
    popularity_weight: float = .02
    ann_nlist: int = 4096
    ann_nprobe: int = 64
    ranking_workers: int = 4

    def __post_init__(self):
        object.__setattr__(self, "beam_widths", _level_widths(self.beam_widths, 3))
        if self.beam_widths[-1] < self.k:
            raise ValueError("final beam width must be at least K")
        if type(self.prefix_beam_width) is not int or self.prefix_beam_width < 1:
            raise ValueError("prefix beam width must be a positive integer")
        if type(self.ranking_workers) is not int or self.ranking_workers < 1:
            raise ValueError("ranking_workers must be a positive integer")
        if (not 1 <= self.k <= min(1000, self.candidate_budget)
                or self.prefix_limit < 1 or self.ann_nlist < 1 or self.ann_nprobe < 1):
            raise ValueError("invalid retrieval limits")
        if self.new_item_quota is not None and not 0 <= self.new_item_quota <= self.k:
            raise ValueError("new item quota must be between zero and final K")
        if self.ranking not in ("item", "beam_product"):
            raise ValueError("unknown item ranking")
        if not np.isfinite(self.popularity_weight) or self.popularity_weight < 0:
            raise ValueError("invalid popularity weight")


@dataclass(frozen=True)
class RouteResult:
    items: np.ndarray
    scores: np.ndarray
    sources: np.ndarray
    candidate_count: int
    new_candidates: int
    old_candidates: int


class GeneratorRanker:
    """Collision-aware ranking with fixed rank-score vectors and whole-history filtering."""

    def __init__(self, bundle, item_vectors, train_count, target_count, config=DecodeConfig(),
                 *, ties=None):
        self.bundle = bundle
        self.ids = bundle.ids
        self.vectors = np.asarray(item_vectors)
        self.train_count = np.asarray(train_count)
        self.target_count = np.asarray(target_count)
        expected = (len(self.ids),)
        if (self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.ids)
                or self.vectors.dtype != np.float16 or self.train_count.shape != expected
                or self.target_count.shape != expected
                or np.shape(bundle.answer_counts) != np.shape(bundle.bucket_keys)):
            raise ValueError("item vectors or training counts do not align with the SID bundle")
        self.config = config
        self.ties = ties
        self.content_index = None
        self._content_pool = None

    def _unseen(self, rows, seen):
        if not len(seen):
            return np.ones(len(rows), dtype=bool)
        ids = self.ids[rows]
        found = np.searchsorted(seen, ids)
        return (found == len(seen)) | (seen[np.minimum(found, len(seen) - 1)] != ids)

    def _item_scores(self, rows, query, logp=0.0):
        scores = self.vectors[rows].astype(np.float32) @ query
        if self.config.ranking == "beam_product":
            return np.exp(logp) * scores
        if self.config.popularity_weight:
            scores += self.config.popularity_weight * np.log1p(
                self.target_count[rows].astype(np.float32)
            )
        return scores

    def _level(self, keys, logp, depth):
        keys, logp = np.asarray(keys, dtype=np.int64), np.asarray(logp, dtype=np.float64)
        if (keys.ndim != 1 or keys.shape != logp.shape or np.any(keys < -1)
                or np.any(keys >= self.bundle.size ** depth)):
            raise ValueError("packed beam shape or SID range differs")
        active = keys >= 0
        keys, logp = keys[active], logp[active]
        if len(np.unique(keys)) != len(keys) or not np.all(np.isfinite(logp)):
            raise ValueError("repeated packed beam key or nonfinite beam score")
        return keys.astype(self.bundle.bucket_keys.dtype), logp

    def _expand_full(self, keys, logp, query, seen):
        keys, logp = self._level(keys, logp, 3)
        bucket_keys = self.bundle.bucket_keys
        buckets = np.searchsorted(bucket_keys, keys)
        if np.any(buckets == len(bucket_keys)) or np.any(bucket_keys[buckets] != keys):
            raise ValueError("beam contains an unknown full SID")
        starts = self.bundle.bucket_offsets[buckets].astype(np.int64)
        ends = self.bundle.bucket_offsets[buckets + 1].astype(np.int64)
        lengths = ends - starts
        positions = (np.repeat(starts - np.cumsum(lengths) + lengths, lengths)
                     + np.arange(lengths.sum()))
        rows = np.searchsorted(self.ids, self.bundle.bucket_items[positions])
        keep = self._unseen(rows, seen)
        rows, buckets = rows[keep], self.bundle.item_bucket[rows[keep]]
        scores = self._item_scores(rows, query, np.repeat(logp, lengths)[keep])
        new_sources = np.where(self.bundle.answer_counts[buckets] == 0, "direct", "hitchhike")
        sources = np.where(self.train_count[rows] == 0, new_sources, "old")
        return rows, scores, sources

    def _expand_prefixes(self, keys, logp, query, seen):
        keys, logp = self._level(keys, logp, 2)
        bucket_keys = self.bundle.bucket_keys
        starts = np.searchsorted(bucket_keys, keys * self.bundle.size)
        ends = np.searchsorted(bucket_keys, (keys + 1) * self.bundle.size)
        offsets = self.bundle.bucket_offsets
        selected_rows, selected_scores = [], []
        for start, end, prefix_score in zip(starts, ends, logp):
            if start == end:
                raise ValueError("beam contains an unknown two-digit SID")
            rows = np.searchsorted(self.ids, self.bundle.bucket_items[offsets[start]:offsets[end]])
            rows = rows[self._unseen(rows, seen)]
            scores = self._item_scores(rows, query, prefix_score)
            rows, scores = top_order(rows, scores, self, self.ties, self.config.prefix_limit)
            # Every item competes for the per-prefix cap. Only local-unseen
            # items join the shared full-SID old-item baseline.
            new = self.train_count[rows] == 0
            selected_rows.append(rows[new])
            selected_scores.append(scores[new])
        if selected_rows:
            return np.concatenate(selected_rows), np.concatenate(selected_scores)
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float32)

    def _build_content_index(self):
        import faiss

        pool = np.flatnonzero(self.train_count == 0)
        if len(pool):
            dim = self.vectors.shape[1]
            nlist = min(self.config.ann_nlist, max(1, len(pool) // 40))
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = 42
            index.cp.niter = 12
            rng = np.random.default_rng(42)
            fit_size = min(len(pool), max(nlist * 40, 200_000))
            sample = np.sort(rng.choice(len(pool), fit_size, replace=False))
            index.train(self.vectors[pool[sample]].astype(np.float32))
            for start in range(0, len(pool), 8192):
                index.add(self.vectors[pool[start:start + 8192]].astype(np.float32))
            index.nprobe = min(self.config.ann_nprobe, nlist)
            self.content_index = index
        self._content_pool = pool

    def _nominate_content(self, query, seen):
        if self._content_pool is None:
            self._build_content_index()
        supply = min(len(self._content_pool), max(4096, self.config.candidate_budget))
        while supply:
            _, positions = self.content_index.search(np.ascontiguousarray(query[None]), supply)
            valid = positions[0] >= 0
            rows = self._content_pool[positions[0, valid]]
            rows = rows[self._unseen(rows, seen)]
            if (len(rows) >= self.config.candidate_budget
                    or not valid.all() or supply == len(self._content_pool)):
                scores = self._item_scores(rows, query)
                return top_order(rows, scores, self, self.ties, self.config.candidate_budget)
            supply = min(len(self._content_pool), supply * 2)
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float32)

    def _result(self, rows, scores, sources):
        # Each input is already disjoint (an item has one full code and prefix).
        order = np.lexsort((self.ids[rows], self.ties[rows], -scores))
        rows, scores, sources = rows[order], scores[order], sources[order]
        new = self.train_count[rows] == 0
        budget = self.config.candidate_budget
        quota = self.config.new_item_quota
        if quota is None:
            candidates = min(budget, len(rows))
            new_candidates = int(new[:candidates].sum())
            final = slice(min(candidates, self.config.k))
        else:
            final = np.zeros(len(rows), dtype=bool)
            new_final = np.flatnonzero(new)[:quota]
            final[new_final] = True
            final[np.flatnonzero(~new)[:self.config.k - len(new_final)]] = True
            # Reserve the chosen final items in the capped candidate pool.
            # Fill remaining candidate slots with the highest ranked others.
            rest = np.flatnonzero(~final)[:budget - int(final.sum())]
            candidates = int(final.sum()) + len(rest)
            new_candidates = len(new_final) + int(new[rest].sum())
        return RouteResult(
            self.ids[rows[final]], scores[final], sources[final],
            candidates, new_candidates, candidates - new_candidates,
        )

    def rank(self, full_keys, full_logp, prefix_keys, prefix_logp, query, seen, *,
             routes=("full_sid",)):
        """Rank one query. ANN is built only when the content route is requested."""
        routes = tuple(routes)
        if self.config.ranking == "beam_product" and routes != ("full_sid",):
            raise ValueError("beam_product requires full_sid as the only route")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.vectors.shape[1],) or not np.all(np.isfinite(query)):
            raise ValueError("invalid query vector")
        if self.ties is None:
            from .two_tower_retrieval import tie_keys
            self.ties = tie_keys(self)
        seen = np.unique(seen if isinstance(seen, np.ndarray) else list(seen))
        full_rows, full_scores, full_sources = self._expand_full(full_keys, full_logp, query, seen)
        old = full_sources == "old"
        result = {}
        for route in routes:
            if route == "full_sid":
                result[route] = self._result(full_rows, full_scores, full_sources)
                continue
            if route in ("direct", "hitchhike"):
                selected = full_sources == route
                new_rows, new_scores = full_rows[selected], full_scores[selected]
            elif route == "prefix":
                new_rows, new_scores = self._expand_prefixes(prefix_keys, prefix_logp, query, seen)
            elif route == "content":
                new_rows, new_scores = self._nominate_content(query, seen)
            else:
                raise ValueError(f"unknown retrieval route: {route}")
            result[route] = self._result(
                np.r_[full_rows[old], new_rows],
                np.r_[full_scores[old], new_scores],
                np.r_[full_sources[old], np.full(len(new_rows), route)],
            )
        return result
