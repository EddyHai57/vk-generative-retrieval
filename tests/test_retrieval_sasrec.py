"""P4H-02 fixed ID-only causal SASRec-style model tests."""
import copy
import unittest

import torch

try:
    from vkgr import retrieval_sasrec as sasrec
except ImportError:
    sasrec = None


class ModulePresenceTests(unittest.TestCase):
    def test_fixed_sasrec_module_exists(self):
        self.assertIsNotNone(sasrec, "P4H-02 model module is missing")


@unittest.skipIf(sasrec is None, "P4H-02 model not implemented")
class FixedArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.content = torch.zeros(9, 64, dtype=torch.float32)
        self.content[1:, 0] = torch.arange(1, 9, dtype=torch.float32)

    def model(self, supported=(1, 2, 3, 4, 5, 6, 7, 8)):
        torch.manual_seed(42)
        return sasrec.SASRecIDTwoTower(self.content, list(supported))

    def test_architecture_is_exact_one_layer_causal_last_valid(self):
        model = self.model()
        self.assertEqual(sasrec.ARCHITECTURE_SPEC, {
            "item_embedding_dim": 64,
            "position_count": 50,
            "layers": 1,
            "heads": 4,
            "feedforward_dim": 256,
            "dropout": 0.1,
            "activation": "relu",
            "bias": True,
            "norm_first": False,
            "layer_norm_eps": 1e-5,
            "batch_first": True,
            "attention": "upper_triangular_causal",
            "pooling": "last_valid_hidden",
            "content_used": False,
        })
        layer = model.transformer
        self.assertEqual(layer.self_attn.embed_dim, 64)
        self.assertEqual(layer.self_attn.num_heads, 4)
        self.assertEqual(layer.linear1.out_features, 256)
        self.assertEqual(layer.dropout.p, 0.1)
        self.assertFalse(layer.norm_first)
        self.assertIsNotNone(layer.linear1.bias)
        self.assertEqual(layer.norm1.eps, 1e-5)
        self.assertFalse(hasattr(model, "user_projection"))

    def test_content_values_never_affect_parameters_or_forward(self):
        first = self.model()
        changed = self.content.clone();changed[1:, :] = torch.arange(64, dtype=torch.float32)
        torch.manual_seed(42);second = sasrec.SASRecIDTwoTower(changed, list(range(1, 9)))
        for left, right in zip(first.state_dict().values(), second.state_dict().values()):
            torch.testing.assert_close(left, right)
        first.eval();second.eval();history = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(first.users(history), second.users(history))
        self.assertFalse(any(name == "content" for name, _ in first.named_buffers()))

    def test_items_zero_unsupported_identity_and_zero_gradient(self):
        model = self.model((1, 2))
        with torch.no_grad():
            model.embedding.weight.fill_(3.0)
            model.embedding.weight[1].fill_(1.0)
        torch.testing.assert_close(model.items(torch.tensor([0, 3, 8])), torch.zeros(3, 64))
        model.items(torch.tensor([1, 3])).sum().backward()
        self.assertGreater(float(model.embedding.weight.grad[1].abs().sum()), 0)
        self.assertEqual(float(model.embedding.weight.grad[3].abs().sum()), 0)

    def test_causal_prefix_pad_compaction_order_and_last_valid(self):
        model = self.model().eval()
        compact = torch.tensor([[1, 2, 3]])
        padded = torch.tensor([[0, 1, 0, 2, 3, 0]])
        torch.testing.assert_close(model.users(compact), model.users(padded), atol=1e-6, rtol=1e-6)
        encoded, valid = model.encode_sequence(compact)
        torch.testing.assert_close(model.users(compact), encoded[:, 2])
        self.assertEqual(valid.tolist(), [[True, True, True]])
        first_a, _ = model.encode_sequence(torch.tensor([[1, 2, 3]]))
        first_b, _ = model.encode_sequence(torch.tensor([[1, 7, 8]]))
        torch.testing.assert_close(first_a[:, 0], first_b[:, 0], atol=1e-6, rtol=1e-6)
        self.assertGreater(float((model.users(compact) - model.users(torch.tensor([[3, 2, 1]]))).abs().max().detach()), 1e-5)

    def test_unsupported_tokens_retain_position_only_and_diagnostics(self):
        model = self.model((1,)).eval()
        with torch.no_grad():
            model.embedding.weight.zero_();model.embedding.weight[1, 0] = 1.0
        left = model.users(torch.tensor([[3, 4]]))
        right = model.users(torch.tensor([[7, 8]]))
        torch.testing.assert_close(left, right)
        self.assertGreater(float((left - model.users(torch.tensor([[1, 4]]))).abs().max().detach()), 1e-5)
        diagnostics = model.history_diagnostics(torch.tensor([[3, 4, 0], [1, 3, 0]]))
        self.assertEqual(diagnostics, {
            "valid_token_count": 4,
            "supported_token_count": 1,
            "unsupported_token_count": 3,
            "unsupported_token_fraction": 0.75,
            "all_zero_identity_history_count": 1,
        })

    def test_length_one_and_fifty_are_valid_and_padding_is_stable(self):
        model = self.model().eval()
        self.assertEqual(tuple(model.users(torch.tensor([[1]])).shape), (1, 64))
        long = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8] * 6 + [1, 2]])
        self.assertEqual(long.shape[1], 50)
        self.assertEqual(tuple(model.users(long).shape), (1, 64))
        model.train();torch.manual_seed(7);first = model.users(torch.tensor([[1, 2, 3]]));torch.manual_seed(7);second = model.users(torch.tensor([[1, 2, 3]]))
        torch.testing.assert_close(first, second)
        model.eval();third=model.users(torch.tensor([[1, 2, 3]]));fourth=model.users(torch.tensor([[1, 2, 3]]));torch.testing.assert_close(third, fourth)

    def test_empty_overlong_unknown_wrong_dtype_and_nonfinite_reject(self):
        model = self.model()
        invalid = (
            torch.zeros(1, 3, dtype=torch.long),
            torch.ones(1, 51, dtype=torch.long),
            torch.tensor([[9]], dtype=torch.long),
            torch.tensor([[1.0]]),
            torch.ones(1, 1, 1, dtype=torch.long),
        )
        for history in invalid:
            with self.assertRaises(ValueError):
                model.users(history)
        with self.assertRaises(ValueError):
            model.items(torch.tensor([9]))
        bad = self.content.clone();bad[2, 3] = float("nan")
        with self.assertRaises(ValueError):
            sasrec.SASRecIDTwoTower(bad, [1])
        with torch.no_grad():
            model.positions.weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model.users(torch.tensor([[1]]))

    def test_checkpoint_cannot_replace_train_support(self):
        model = self.model((1, 2))
        state = {name: value.clone() for name, value in model.state_dict().items()}
        state["supported"][3] = True
        with self.assertRaises(ValueError):
            model.load_state_dict(state)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        model.load_state_dict(original)

    def test_existing_training_step_updates_parameters(self):
        model = self.model((1, 2, 3, 4)).train()
        from vkgr.retrieval_training import SamplerStats, train_step
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        rows = [
            {"history": [1, 2], "target": 3, "past_likes": [1, 2]},
            {"history": [2, 1], "target": 4, "past_likes": [1, 2]},
        ]
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        result = train_step(model, optimizer, rows, {1: 1, 2: 2, 3: 3, 4: 4}, SamplerStats(), 0, 0.15)
        self.assertGreater(result["loss"], 0)
        self.assertTrue(any(not torch.equal(before[name], value.detach()) for name, value in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
