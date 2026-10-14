"""GFEAT-02: the GFEAT-01 history-side features (F2 feedback, F3 watch buckets) on the GARCH-01 paper-scale model (A1).

Nothing here is reachable from the default G path, the GARCH-01 entry or the GFEAT-01 entry. The injection is GFEAT-01's,
reused as the SAME function objects (not a copy):

  x = (E[token] + side_vector(item of that token)) * sqrt(d_model) + position        (encoder input only)

broadcast onto all four SID tokens of a history item; the feature definitions come from the modules shared byte-for-byte
with the P line (``history_side``, ``retrieval_features``) through ``generative_feat``. The encoder / decoder stacks are
GARCH-01's ``TigerArchModel`` (4+4 layers, 6 heads x 64, FF 1024, dropout 0.1).

  feature=none   the GARCH-01 factory itself (plain ``TigerArchModel``): the control reproduces A1
  f2 / f3        GARCH-01 initialisation, then the side tables N(0, 0.02), PAD rows 0. The tables are drawn inside a forked CPU
                 RNG: their values are the draws that follow the base initialisation (as in GFEAT-01), and the global stream
                 afterwards is the one feature=none leaves, so dropout masks during training are the control's.
"""
import torch

from . import generative_arch as ga
from . import generative_feat as gf
from .generative_hp import embedding_report, init_embeddings
from .tiger_arch import TigerArchModel

SCHEMA = 'g-arch-feat-run-v1'
TASK_ID = 'GFEAT-02'
FEATURES = gf.FEATURES
REGISTERED = ((ga.PAPER_SCALE, .0003),)  # formal runs: A1 only (structure, learning rate)


def default_config(feature='none'):
    return {**ga.default_config(), 'schema': SCHEMA, 'task_id': TASK_ID, 'feature': feature, 'side_version': gf.SIDE_VERSION}


def validate_config(value, engineering=False):
    """A1 config + {feature, side_version}; everything else is GARCH-01's frozen contract (no HPO, seed 42/2024/2025)."""
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered GFEAT-02 config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK_ID or value['side_version'] != gf.SIDE_VERSION or value['feature'] not in FEATURES:
        raise ValueError('unregistered GFEAT-02 schema/task/feature/side version')
    arch = {k: v for k, v in value.items() if k not in ('feature', 'side_version')}
    arch.update(schema=ga.SCHEMA, task_id=ga.TASK_ID)
    ga.validate_config(arch, engineering)
    if not engineering and (tuple(value[k] for k in ga.SHAPE_KEYS), value['learning_rate']) not in REGISTERED:
        raise ValueError('formal GFEAT-02 = A1 (paper-scale structure, LR 3e-4) only')
    return value


class TigerArchModelSide(TigerArchModel):
    """TigerArchModel whose ENCODER input also carries a per-history-item side vector; the decoder path is untouched."""
    feature = None
    add_side_tables = gf.TigerModelSide.add_side_tables
    _valid = gf.TigerModelSide._valid
    side_vector = gf.TigerModelSide.side_vector
    embed_source = gf.TigerModelSide.embed_source

    def encode(self, source, side=None):
        padding = source.eq(0)
        if padding.all(dim=1).any():
            raise ValueError('empty source')
        x = self.embed_source(source, side)
        for layer in self.transformer.encoder.layers:
            x = layer(x, padding)
        return self.transformer.encoder.norm(x), padding

    def forward(self, source, decoder, side=None):
        memory, padding = self.encode(source, side)
        return self.decode(memory, padding, decoder)


def _side_tables(model, feature):
    with torch.random.fork_rng(devices=[]):
        model.add_side_tables(feature)


def _side_params(model, feature):
    tables = [getattr(model, n) for n in ('feedback', 'timespent_bucket', 'ratio_bucket') if hasattr(model, n)]
    total = sum(t.weight.numel() for t in tables)
    trainable = sum(t.weight.numel() - (0 if t.padding_idx is None else t.weight.shape[1]) for t in tables)
    return {'feature': feature, 'params_side': total, 'params_side_trainable': trainable, 'params_total': sum(p.numel() for p in model.parameters())}


def make_model(codec, config, *, device):
    feature = config['feature']
    if feature not in FEATURES:
        raise ValueError('unregistered GFEAT-02 feature')
    if feature == 'none':
        return ga.make_model_arch(codec, config, device=device)
    if config['init'] != 'small' or config['history_len'] != 50:
        raise ValueError('GFEAT-02 side features: init=small, history_len=50 only')
    if any(type(config[k]) is not int for k in ga.SHAPE_KEYS[:-1]) or type(config['dropout']) is not float:
        raise ValueError('typed architecture config required')
    model = TigerArchModelSide(codec.vocab_size, *(config[k] for k in ga.SHAPE_KEYS)).to(device=device, dtype=torch.float32)
    init_embeddings(model)  # the GARCH-01 draw sequence: structure, then embedding and position
    arch = ga.arch_report(model)  # before the side tables: the structure's closed-form count, as in the control
    _side_tables(model, feature)
    model.to(device=device, dtype=torch.float32)
    model.init_report = {'init': 'small', 'std': 0.02, 'embeddings': embedding_report(model), 'arch': arch, 'side': _side_params(model, feature)}
    return model
