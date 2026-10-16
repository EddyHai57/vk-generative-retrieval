"""R1 tau=.8 in training AND beam distributions; the A1 parameterization is unchanged."""
import torch
from . import generative_arch as ga
from . import generative_hp as ghp
from .generative_g6_model import TigerArchModelD, full_vocab_ce_d
from .sid_depth import ValidatedCodecD


def _temperature(value):
    # 1.0 exists solely for synthetic parity tests; formal configs require .8.
    if type(value) is not float or value not in (.8, 1.):
        raise ValueError('R1 fixed temperature .8; 1.0 is synthetic parity only')
    return value


class TemperatureModel(TigerArchModelD):
    """Scale decode logits once. The inherited forward calls this same decode path."""
    def __init__(self, *args, temperature, **kwargs):
        self.temperature = _temperature(temperature)
        super().__init__(*args, **kwargs)

    def decode(self, memory, padding, decoder):
        logits = super().decode(memory, padding, decoder)
        return logits if self.temperature == 1. else logits / self.temperature


def make_model(codec, config, *, device):
    if not isinstance(codec, ValidatedCodecD) or codec.depth != 3 or config['history_len'] != 50 or config['init'] != 'small':
        raise ValueError('R1 requires validated depth3, L50, small initialization')
    tau = _temperature(config['temperature'])
    model = TemperatureModel(codec.vocab_size, *(config[k] for k in ga.SHAPE_KEYS),
                             max_tokens=(codec.depth + 1) * config['history_len'], temperature=tau).to(device=device, dtype=torch.float32)
    report = ghp.init_embeddings(model)
    model.init_report = {'init': 'small', 'std': .02, 'embeddings': report, 'arch': ga.arch_report(model)}
    return model


def temperature_ce(raw_logits, labels, temperature=.8):
    """CE oracle for raw logits. Runner receives already-scaled model output, so uses original CE."""
    tau = _temperature(temperature)
    return full_vocab_ce_d(raw_logits if tau == 1. else raw_logits / tau, labels, labels.shape[1])
