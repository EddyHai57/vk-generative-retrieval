"""M1 mean with E pooled m and supported-only C/D candidates. New branch draws last."""
import torch
from . import retrieval_seqm1 as seq
from . import retrieval_feat02 as f2
from . import retrieval_feat03 as f3
from . import retrieval_feat04b as stats
from . import retrieval_feat04combo as combo
from .retrieval_hp import init_embeddings, embedding_report

class SeqComboStack(seq.SeqM1IDMean):
    def anchor_vector(self, histories, side):
        valid = self._check(histories, side)
        # Unweighted F3/F4 events, never candidate stat vectors.
        e = self.event_vectors(histories, side, valid)
        pos = f2.position_bucket(valid)
        gap = f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid)
        logits = f3.recency_logits(self.recency_logit_pos, self.recency_logit_gap, pos, gap, valid)
        logits = logits + stats.behavior_logits(self.behavior_beta, side[..., 0], valid)
        return f3.weighted_mean(e, f3.pooled_weights(logits, valid))

    def candidate_items(self, indices):
        terms = sum(getattr(self, table)(getattr(self, buffer)[indices])
                    for table, buffer, _ in combo.STAT_LAYOUT[combo.E_FEATURE])
        return self.items(indices) + terms * self.supported[indices].unsqueeze(-1).to(terms.dtype)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name in self.stat_buffers:
            actual, expected = state_dict.get(name), getattr(self, name)
            if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
                raise ValueError('immutable stack stat bucket ids mismatch')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def make_model(content, supported_indices, max_len, stat_buckets):
    if type(max_len) is not int or max_len != 50:
        raise ValueError('stack is M1 mean L50 only')
    # Exact original M1 factory draw sequence, using the new subclass (no extra constructor draws).
    model = SeqComboStack(content, supported_indices, max_len, 'mean')
    std = seq.init_std(seq.DIM)
    init_embeddings(model, std)
    seq._draw_tables(model, std)
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': std, 'embeddings': embedding_report(model)})
    model.add_branch()
    f3.init_recency_logits(model)
    stats.init_behavior_beta(model)
    if type(stat_buckets) is not dict:
        raise ValueError('stack requires four stats buffers')
    names = []
    for _, name, edges in combo.STAT_LAYOUT[combo.E_FEATURE]:
        ids = stat_buckets.get(name)
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.shape != (len(content),) or int(ids[0]) != 0:
            raise ValueError('invalid stack stats buffer: ' + name)
        if int(ids.min()) < 0 or int(ids.max()) > len(getattr(stats, edges)) + 1:
            raise ValueError('stack stats bucket out of range')
        model.register_buffer(name, ids.detach().clone()); names.append(name)
    object.__setattr__(model, 'stat_buffers', tuple(names))
    stats.init_stat_tables(model, 'f4_item_popularity')
    stats.init_stat_tables(model, 'f4_item_quality')
    return model


def parameter_report(model):
    base = seq.parameter_report_seqm1(model)
    extras = {'recency_logits': model.recency_logit_pos.numel() + model.recency_logit_gap.numel(),
              'behavior_beta': model.behavior_beta.numel(),
              'candidate_stats': sum(getattr(model, table).weight.numel() for table, _, _ in combo.STAT_LAYOUT[combo.E_FEATURE])}
    return {**base, **extras, 'total': base['total'] + sum(extras.values())}
