import importlib
import math

import pytest
import torch
from torch.nn import functional as F

from vkgr import tiger_arch


@pytest.fixture(scope="module")
def api():
    # A fresh RED can distinguish the missing entry point from import failures
    # in dependencies once implementation has started.
    assert importlib.util.find_spec("lsvd.generator_model"), "generator model API is missing"
    return importlib.import_module("lsvd.generator_model")


def model(api, mode="event", *, shared_head=True, decoder_layers=1, dropout=0.0):
    torch.manual_seed(4321)
    config = api.GeneratorConfig(
        history_mode=mode, shared_head=shared_head, size=7, vector_dim=4,
        d_model=8, heads=2, head_dim=5, encoder_layers=1, decoder_layers=decoder_layers,
        ff=16, dropout=dropout,
    )
    return api.GeneratorModel(config).eval()


def history(valid):
    valid = torch.as_tensor(valid, dtype=torch.bool)
    rows, width = valid.shape
    positions = torch.arange(width).expand(rows, -1)
    codes = torch.arange(rows * width * 3).reshape(rows, width, 3) % 7
    return dict(
        codes=codes, cold=torch.arange(rows * width * 4).reshape(rows, width, 4).float() / 100,
        feedback=torch.zeros((rows, width, 9)), ratio=torch.zeros((rows, width)),
        place=torch.ones_like(positions), gap=torch.ones_like(positions),
        position=positions, valid=valid,
    )


def test_encoder_dispatch_keeps_the_frozen_history_lengths(api):
    batch = history([[True] * 200, [True, False] + [False] * 198])
    for mode, tokens_per_event in (("event", 1), ("tiger", 3), ("event_no_feedback", 1)):
        generator = model(api, mode)
        memory, padding = generator.encode(batch)
        assert memory.shape == (2, 200 * tokens_per_event, 8)
        torch.testing.assert_close(padding, ~batch["valid"].repeat_interleave(tokens_per_event, 1))
        assert torch.isfinite(memory).all()
    assert tiger_arch.POSITIONS == 200


@pytest.mark.parametrize("mode", ["event", "tiger", "event_no_feedback"])
def test_padding_cannot_change_valid_context_or_dense_query(api, mode):
    original = history([[True, False, False], [True, True, True]])
    changed = {name: value.clone() for name, value in original.items()}
    for name in ("codes", "place", "gap", "position"):
        changed[name][0, 1:] = -99
    for name in ("cold", "feedback", "ratio"):
        changed[name][0, 1:] = float("nan")

    generator = model(api, mode)
    first_memory, first_padding = generator.encode(original)
    second_memory, second_padding = generator.encode(changed)
    torch.testing.assert_close(first_padding, second_padding)
    torch.testing.assert_close(first_memory[~first_padding], second_memory[~second_padding])
    prefixes = torch.empty((2, 0), dtype=torch.long)
    torch.testing.assert_close(
        generator.decode_next(first_memory, first_padding, prefixes),
        generator.decode_next(second_memory, second_padding, prefixes),
    )
    query = generator.user_vector(second_memory, second_padding)
    assert query.shape == (2, 4)
    torch.testing.assert_close(query.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(query, generator.user_vector(first_memory, first_padding))


@pytest.mark.parametrize("mode", ["event", "tiger", "event_no_feedback"])
def test_extra_padding_does_not_change_the_same_sequence(api, mode):
    generator = model(api, mode)
    short = generator.encode(history([[True, True]]))
    long = generator.encode(history([[True, True, False, False]]))
    torch.testing.assert_close(short[0][~short[1]], long[0][~long[1]])
    torch.testing.assert_close(generator.user_vector(*short), generator.user_vector(*long))
    prefixes = torch.tensor([[0, 2]])
    torch.testing.assert_close(generator.decode_next(*short, prefixes), generator.decode_next(*long, prefixes))


@pytest.mark.parametrize("mode", ["event", "tiger", "event_no_feedback"])
def test_true_empty_train_histories_are_masked_but_finite(api, mode):
    generator = model(api, mode)
    memory, padding = generator.encode(history([[], []]))
    assert padding.all()
    assert torch.isfinite(memory).all()
    assert torch.isfinite(generator.user_vector(memory, padding)).all()
    assert torch.isfinite(generator.sid_loss(history([[False], [True]]), torch.zeros((2, 3), dtype=torch.long)))


@pytest.mark.parametrize("feature", ["feedback", "ratio"])
def test_feedback_ablation_removes_feedback_and_watch_ratio(api, feature):
    original = history([[True, True]])
    changed = {**original, feature: torch.ones_like(original[feature])}
    for mode in ("event", "event_no_feedback", "tiger"):
        generator = model(api, mode)
        memory, _ = generator.encode(original)
        modified_memory, _ = generator.encode(changed)
        if mode == "event":
            assert not torch.allclose(memory, modified_memory)
        else:
            torch.testing.assert_close(memory, modified_memory, rtol=0, atol=0)


@pytest.mark.parametrize("feature", ["cold", "place", "gap", "position"])
def test_content_and_context_survive_the_event_ablation(api, feature):
    original = history([[True, True]])
    changed = {**original, feature: original[feature] + 1}
    for mode in ("event_no_feedback", "tiger"):
        generator = model(api, mode)
        memory, _ = generator.encode(original)
        modified_memory, _ = generator.encode(changed)
        if mode == "event_no_feedback":
            assert not torch.allclose(memory, modified_memory)
        else:
            torch.testing.assert_close(memory, modified_memory, rtol=0, atol=0)


def test_teacher_forcing_matches_autoregressive_prefixes(api):
    generator = model(api)
    batch = history([[True, True], [True, False]])
    memory, padding = generator.encode(batch)
    target_codes = torch.tensor([[0, 2, 4], [0, 2, 3]])
    changed_suffix = torch.tensor([[5, 6, 0], [5, 6, 1]])
    logits = generator.sid_logits(memory, padding, target_codes)
    assert logits.shape == (2, 3, 7)
    for layer in range(3):
        next_logits = generator.decode_next(memory, padding, target_codes[:, :layer])
        assert next_logits.shape == (2, 7)
        torch.testing.assert_close(logits[:, layer], next_logits, atol=2e-6, rtol=1e-5)
    only_last_changes = target_codes.clone()
    only_last_changes[:, -1] = 0
    torch.testing.assert_close(logits, generator.sid_logits(memory, padding, only_last_changes))
    changed_logits = generator.sid_logits(memory, padding, changed_suffix)
    torch.testing.assert_close(logits[:, 0], changed_logits[:, 0])
    assert not torch.allclose(logits[:, 1:], changed_logits[:, 1:])
    with pytest.raises(ValueError, match="prefix"):
        generator.decode_next(memory, padding, target_codes)


@pytest.mark.parametrize("mode", ["event", "tiger", "event_no_feedback"])
@pytest.mark.parametrize("microbatch", [1, 2, 8, 128])
def test_beam_scorer_matches_repeated_memory_logits(api, mode, microbatch):
    generator = model(api, mode, decoder_layers=2, dropout=0.4)
    source = history([[True, False, False], [False, False, False], [True, True, False]])
    memory, padding = generator.encode(source)
    score = generator.make_beam_scorer(memory, padding, hypothesis_batch_size=microbatch)
    for depth, width in enumerate((1, 5, 5)):
        all_codes = torch.arange(3 * width * 2).reshape(3, width, 2) % 7
        prefixes = all_codes[..., :depth]
        expected = generator.decode_next(
            memory.repeat_interleave(width, 0), padding.repeat_interleave(width, 0),
            prefixes.reshape(3 * width, depth),
        ).reshape(3, width, 7)
        actual = score(prefixes)
        assert actual.shape == (3, width, 7)
        assert actual.dtype == expected.dtype and actual.device == memory.device
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_beam_scorer_projects_each_memory_once_per_decoder_layer(api, monkeypatch):
    generator = model(api, decoder_layers=2)
    memory, padding = generator.encode(history([[], []]))
    layers = [layer.multihead_attn for layer in generator.transformer.decoder.layers]
    kv_weights = {layer.in_proj_weight[layer.inner:].data_ptr() for layer in layers}
    projections = []
    linear = F.linear

    def record(input, weight, bias=None):
        if weight.data_ptr() in kv_weights:
            projections.append(tuple(input.shape))
        return linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", record)
    score = generator.make_beam_scorer(memory, padding, hypothesis_batch_size=3)
    assert torch.isfinite(score(torch.empty((2, 1, 0), dtype=torch.long))).all()
    assert torch.isfinite(score(torch.zeros((2, 5, 2), dtype=torch.long))).all()
    assert projections == [tuple(memory.shape)] * len(layers)


def test_beam_scorer_requires_eval(api):
    generator = model(api)
    memory, padding = generator.encode(history([[True]]))
    score = generator.make_beam_scorer(memory, padding)
    generator.train()
    with pytest.raises(ValueError, match="eval"):
        generator.make_beam_scorer(memory, padding)
    with pytest.raises(ValueError, match="eval"):
        score(torch.empty((1, 1, 0), dtype=torch.long))


def test_sid_namespaces_share_layer_local_code_ids_between_source_and_decoder(api):
    generator = model(api, "tiger")
    batch = history([[True, False]])
    batch["codes"].zero_()
    token_inputs = []
    handle = generator.sid_embedding.register_forward_pre_hook(
        lambda _module, values: token_inputs.append(values[0].flatten().tolist())
    )
    memory, padding = generator.encode(batch)
    generator.decode_next(memory, padding, torch.tensor([[0, 0]]))
    handle.remove()
    assert token_inputs == [[2, 9, 16, 0, 0, 0], [1, 2, 9]]


def test_losses_are_fp32_weighted_and_dense_uses_logq_and_positive_mask(api):
    generator = model(api)
    batch = history([[True, True], [True, False], [False, False]])
    target_codes = torch.tensor([[0, 2, 4], [1, 2, 3], [6, 5, 4]])
    item_ids = torch.tensor([10, 11, 12])
    vectors = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 2.0, 0]], requires_grad=True)
    probability = torch.tensor([0.25, 0.50, 0.25])
    weights = torch.tensor([1.0, 0.25, 1.0])
    allowed = torch.tensor([[True, False, True], [True, True, True], [False, False, True]])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        total, sid, dense = generator.loss(batch, target_codes, vectors, item_ids, allowed, probability, weights)
        validation = generator.sid_loss(batch, target_codes, weights)
        memory, padding = generator.encode(batch)
        logits = generator.sid_logits(memory, padding, target_codes).float()
        query = generator.user_vector(memory, padding)
    assert {total.dtype, sid.dtype, dense.dtype, validation.dtype} == {torch.float32}
    sid_expected = -sum(
        weights[row] * F.log_softmax(logits[row], -1)[range(3), target_codes[row]].mean()
        for row in range(3)
    ) / weights.sum()
    score = (F.normalize(query, dim=1) @ F.normalize(vectors, dim=1).T) / 0.1 - probability.log()
    dense_expected = (torch.logaddexp(score[0, 0], score[0, 2]) - score[0, 0]
                      + 0.25 * (torch.logsumexp(score[1], 0) - score[1, 1])) / 1.25
    torch.testing.assert_close(sid, sid_expected)
    torch.testing.assert_close(validation, sid_expected)
    torch.testing.assert_close(dense, dense_expected)
    torch.testing.assert_close(total, sid + dense)
    total.backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert vectors.grad is None


def test_sid_only_uses_no_dense_inputs(api):
    generator = model(api, shared_head=False)
    batch = history([[True], [False]])
    memory, padding = generator.encode(batch)
    assert generator.user_vector(memory, padding) is None
    target_codes = torch.zeros((2, 3), dtype=torch.long)
    total, sid, dense = generator.loss(batch, target_codes, None, None, None, None, None)
    assert dense.item() == 0
    assert math.isfinite(total.item())
    torch.testing.assert_close(total, sid)
    torch.testing.assert_close(sid, generator.sid_loss(batch, target_codes))


def test_unknown_history_mode_is_rejected(api):
    with pytest.raises(ValueError, match="history mode"):
        model(api, "unknown")


def test_implicit_loss_weights_are_fp32_even_under_a_float64_default(api):
    generator = model(api, shared_head=False)
    batch = history([[True], [False]])
    target_codes = torch.zeros((2, 3), dtype=torch.long)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        loss = generator.sid_loss(batch, target_codes)
    finally:
        torch.set_default_dtype(previous)
    assert loss.dtype == torch.float32
