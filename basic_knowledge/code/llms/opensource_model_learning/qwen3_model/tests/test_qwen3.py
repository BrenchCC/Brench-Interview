"""Unit tests for Qwen3 core architecture behavior."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from configuration_qwen3 import Qwen3Config
from modeling_qwen3 import Qwen3ForCausalLM
from utils import RotaryEmbedding


def make_toy_config() -> Qwen3Config:
    """Create a compact configuration shared by Qwen3 unit tests.

    Parameters:
        None.
    """
    return Qwen3Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        max_position_embeddings = 32,
        use_sliding_window = True,
        sliding_window = 3,
        max_window_layers = 2,
        rope_parameters = {"rope_theta": 1000000.0},
    )


class Qwen3Test(unittest.TestCase):
    """Verify Qwen3 configuration, Q/K norms, inference, cache, and loss."""

    def setUp(self) -> None:
        """Create deterministic Qwen3 test inputs and model.

        Parameters:
            None.
        """
        torch.manual_seed(1)
        self.config = make_toy_config()
        self.model = Qwen3ForCausalLM(self.config)
        self.token_ids = torch.tensor([[1, 4, 7, 9, 3], [2, 5, 8, 6, 1]])

    def test_official_default_configuration(self) -> None:
        """Keep public Qwen3 defaults and full-attention default layer types.

        Parameters:
            None.
        """
        config = Qwen3Config()
        self.assertEqual(config.vocab_size, 151936)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 22016)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.head_dim, 128)
        self.assertIsNone(config.sliding_window)
        self.assertTrue(all(layer_type == "full_attention" for layer_type in config.layer_types))

    def test_sliding_configuration_and_kv_fallback(self) -> None:
        """Enable trailing sliding layers and derive missing KV-head count.

        Parameters:
            None.
        """
        config = Qwen3Config(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 3,
            num_attention_heads = 4,
            num_key_value_heads = None,
            head_dim = 8,
            use_sliding_window = True,
            sliding_window = 3,
            max_window_layers = 1,
        )
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.layer_types, ["full_attention", "sliding_attention", "sliding_attention"])

    def test_standard_rope_shapes(self) -> None:
        """Create finite standard RoPE tensors with full head dimension.

        Parameters:
            None.
        """
        rotary = RotaryEmbedding(self.config)
        hidden_states = torch.zeros(1, 5, self.config.hidden_size)
        cos, sin = rotary(hidden_states, torch.arange(5).unsqueeze(0))
        self.assertEqual(cos.shape, (1, 5, self.config.head_dim))
        self.assertTrue(torch.isfinite(cos).all())
        self.assertTrue(torch.isfinite(sin).all())

    def test_forward_causality_and_qk_norm(self) -> None:
        """Check logits, causal isolation, and per-head Q/K normalization parameters.

        Parameters:
            None.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.token_ids, use_cache = False)
            changed_tokens = self.token_ids.clone()
            changed_tokens[:, -1] = 11
            changed_output = self.model(changed_tokens, use_cache = False)

        self.assertEqual(output.logits.shape, (2, 5, self.config.vocab_size))
        self.assertTrue(torch.allclose(output.logits[:, :-1], changed_output.logits[:, :-1], atol = 1e-6))
        attention = self.model.model.layers[0].self_attn
        self.assertEqual(attention.q_norm.weight.shape, (self.config.head_dim,))
        self.assertEqual(attention.k_norm.weight.shape, (self.config.head_dim,))

    def test_kv_cache_matches_full_forward(self) -> None:
        """Match full Qwen3 logits with prefill plus cached continuation.

        Parameters:
            None.
        """
        self.model.eval()
        with torch.no_grad():
            full_output = self.model(self.token_ids, use_cache = False)
            prefix_output = self.model(self.token_ids[:, :3], use_cache = True)
            cached_output = self.model(
                self.token_ids[:, 3:],
                past_key_values = prefix_output.past_key_values,
                use_cache = True,
            )

        self.assertEqual(cached_output.past_key_values.get_seq_length(), self.token_ids.shape[1])
        self.assertTrue(torch.allclose(full_output.logits[:, 3:], cached_output.logits, atol = 1e-5))

    def test_training_loss_and_gradients(self) -> None:
        """Compute causal loss and backpropagate into Q/K norms and MLP.

        Parameters:
            None.
        """
        self.model.train()
        output = self.model(self.token_ids, labels = self.token_ids)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        attention = self.model.model.layers[0].self_attn
        self.assertIsNotNone(attention.q_norm.weight.grad)
        self.assertIsNotNone(self.model.model.layers[0].mlp.gate_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
