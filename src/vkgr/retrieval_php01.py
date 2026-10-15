"""P-HP-01 P model: the FEAT-02 F4 anchor (``f3_f4_recency``) with a configurable embedding dimension (new file; frozen modules untouched).

All five tables (item id embedding, duration bucket T, ratio bucket R, position bucket P, week-gap bucket G) use the
same width ``dim`` in DIMS. Row counts, bucket edges and the F4 formula are imported from the frozen FEAT-02 module.
At dim=64 the model initialises byte for byte like ``make_feat02_model('f3_f4_recency')`` at the same seed: the
constructor repeats ``TwoTower.__init__`` (id_mean branch) step by step instead of calling it, so no extra n x 64 draw
happens, and the tables are drawn in the FEAT-02 order. Initialisation is INIT_EMB_N002 for every dim (no scaling).
"""
import torch

from .retrieval_features import RATIO_BUCKETS, TIMESPENT_BUCKETS
from .retrieval_feat02 import INERT_ROWS, MAX_WEEK, POSITION_EDGES, WEEKGAP_EDGES, Feat02IDMean, bucket_spec_sha256  # noqa: F401
from .retrieval_hp import embedding_report, init_embeddings

DIMS = (32, 64, 128)
FEATURE = 'f3_f4_recency'
CONTENT_DIM = 64                      # width of the official catalog content vectors (validated only; id_mean does not use them)


def make_table(rows, dim):
    """One embedding table with a zero PAD row 0 (same constructor and default draw as the frozen models)."""
    return torch.nn.Embedding(rows, dim, padding_idx=0)


def side_dim(model):
    """Width of the four F3/F4 side tables: always the model dimension."""
    return model.dim


def init_std(dim):
    """INIT_EMB_N002 for every dimension (not scaled with dim)."""
    return 0.02


def _check_dim(dim):
    if type(dim) is not int or dim not in DIMS:
        raise ValueError('unregistered P-HP-01 dimension')
    return dim


class F4IDMeanD(Feat02IDMean):
    def __init__(self, content, supported_indices, max_len, dim):
        torch.nn.Module.__init__(self)                     # TwoTower.__init__ would draw an extra len(content) x 64 table
        if not isinstance(content, torch.Tensor) or content.dtype != torch.float32 or content.ndim != 2 or content.shape[1] != CONTENT_DIM \
                or len(content) < 2 or not torch.isfinite(content).all() or torch.any(content[0]):
            raise ValueError('invalid content/PAD')
        self.architecture = 'id_mean'
        self.embedding = make_table(len(content), _check_dim(dim))
        supported = torch.zeros(len(content), dtype=torch.bool)
        if any(type(i) is not int or not 1 <= i < len(content) for i in supported_indices):
            raise ValueError('invalid supported ID')
        supported[supported_indices] = True; self.register_buffer('supported', supported)
        object.__setattr__(self, 'feature', FEATURE); object.__setattr__(self, 'max_len', max_len); object.__setattr__(self, 'dim', dim)

    def add_anchor_tables(self):
        """F3 tables T and R at the model width, created in the FEAT-02 order."""
        self.timespent_bucket = make_table(TIMESPENT_BUCKETS + 1, side_dim(self))
        self.ratio_bucket = make_table(RATIO_BUCKETS + 1, side_dim(self))
        return ('timespent_bucket', 'ratio_bucket')

    def add_variant_tables(self):
        """F4 tables P and G at the model width, created after the anchor tables are initialised."""
        self.position_table = make_table(len(POSITION_EDGES) + 2, side_dim(self))
        self.weekgap_table = make_table(len(WEEKGAP_EDGES) + 2, side_dim(self))
        return ('position_table', 'weekgap_table')


def _small_init_d(model, names, std):
    for name in names:
        child = model.get_submodule(name)
        with torch.no_grad():
            child.weight.normal_(0.0, std)
            child.weight[child.padding_idx].zero_()


def _draw_tables(model, std):
    """FEAT-02 draw order: anchor tables T, R first, then the variant tables P, G."""
    _small_init_d(model, model.add_anchor_tables(), std)
    _small_init_d(model, model.add_variant_tables(), std)


def make_php01_model(content, supported_indices, max_len, dim):
    """F4 at width ``dim``. Draw order as ``make_feat02_model('f3_f4_recency')``: item embedding, then T and R, then P and G."""
    if type(max_len) is not int or max_len != 50:
        raise ValueError('P-HP-01 is HIST_V2_OBS_L50 only')
    _check_dim(dim)
    model = F4IDMeanD(content, supported_indices, max_len, dim)
    std = init_std(dim)
    init_embeddings(model, std)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _draw_tables(model, std)
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': std, 'embeddings': embedding_report(model)})
    return model


def _report(rows, dim):
    by_table = {name: {'total': n * dim, 'trainable': (n - INERT_ROWS.get(name, 1)) * dim} for name, n in rows.items()}
    return {'total': sum(v['total'] for v in by_table.values()), 'trainable_effective': sum(v['trainable'] for v in by_table.values()), 'by_table': by_table}


def parameter_report_d(model):
    """FEAT-02 convention (item embedding excluded, PAD rows not trainable), counted at the model width."""
    rows = {name: module.weight.shape[0] for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)) and name != 'embedding'}
    return _report(rows, model.dim)


def expected_param_counts_d(dim):
    """Pure formula for ``parameter_report_d`` (no model needed)."""
    rows = {'timespent_bucket': TIMESPENT_BUCKETS + 1, 'ratio_bucket': RATIO_BUCKETS + 1,
            'position_table': len(POSITION_EDGES) + 2, 'weekgap_table': len(WEEKGAP_EDGES) + 2}
    return _report(rows, _check_dim(dim))
