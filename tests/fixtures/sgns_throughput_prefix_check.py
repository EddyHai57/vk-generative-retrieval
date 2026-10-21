"""Fresh process protects the real frozen SGNS from other tests' module stubs."""
import json
from unittest.mock import patch
import numpy as np
import torch
from analysis.sgns_throughput import bounded_fit
from vkgr import sid_coll_sgns as sg

config = sg.SGNSConfig(dim=4, window=2, epochs=1, batch_size=2)
catalog = np.arange(1, 8, dtype=np.uint32)
histories = {1: [1, 2, 3, 4, 5, 6, 7] * 3}
content = np.ones((7, 4), dtype=np.float32)
original_step = torch.optim.SGD.step
original_negative = sg._sample_from_cdf
original_pairs = sg.iter_positive_pairs
original_runtime = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled())


def capture_negative(records):
    def sample(*args, **kwargs):
        value = original_negative(*args, **kwargs)
        if len(records) < 2:
            records.append((value.copy(), json.dumps(args[-1].bit_generator.state, sort_keys=True)))
        return value
    return sample

def capture(records):
    def step(optimizer, *args, **kwargs):
        value = original_step(optimizer, *args, **kwargs)
        if len(records) < 2:
            records.append(([p.detach().clone() for group in optimizer.param_groups for p in group['params']],
                            [p.grad.to_dense().clone() for group in optimizer.param_groups for p in group['params']],
                            torch.get_rng_state().clone(), optimizer.state_dict()['param_groups']))
        return value
    return step

reference = []
reference_negative = []
with patch.object(torch.optim.SGD, 'step', capture(reference)), patch.object(sg, '_sample_from_cdf', capture_negative(reference_negative)), patch.object(sg, 'health_check', return_value={}):
    sg.fit_sgns(catalog, histories, content, engineering=True, config=config)
limited = []
limited_negative = []
with patch.object(torch.optim.SGD, 'step', capture(limited)), patch.object(sg, '_sample_from_cdf', capture_negative(limited_negative)), patch.object(sg, 'health_check', side_effect=AssertionError('health forbidden')):
    value = bounded_fit(catalog, histories, content, config=config, engineering=True, limit_batches=2)
assert value['completed_batches'] == 2 and value['measured_positive_pairs'] == 4
assert 'vectors' not in value and len(limited) == 2
for left, right in zip(reference, limited):
    assert all(torch.equal(a, b) for a, b in zip(left[0], right[0]))
    assert all(torch.equal(a, b) for a, b in zip(left[1], right[1]))
    assert torch.equal(left[2], right[2]) and left[3] == right[3]
assert torch.optim.SGD.step is original_step
assert len(reference_negative) == len(limited_negative) == 2
for left, right in zip(reference_negative, limited_negative):
    assert np.array_equal(left[0], right[0]) and left[1] == right[1]
assert sg.iter_positive_pairs is original_pairs
assert original_runtime == (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                            torch.is_deterministic_algorithms_warn_only_enabled())
with patch.object(sg, '_sample_from_cdf', side_effect=RuntimeError('injected sampling failure')):
    try:
        bounded_fit(catalog, histories, content, config=config, engineering=True, limit_batches=2)
    except RuntimeError as error:
        assert str(error) == 'injected sampling failure'
    else:
        raise AssertionError('injected failure swallowed')
assert sg.iter_positive_pairs is original_pairs and torch.optim.SGD.step is original_step
assert original_runtime == (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                            torch.is_deterministic_algorithms_warn_only_enabled())
print(json.dumps({'status': 'prefix_bitwise_equal', 'steps': 2, 'parameters_gradients_optimizer_torch_rng': True}))
