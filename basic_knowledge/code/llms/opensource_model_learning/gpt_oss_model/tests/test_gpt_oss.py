"""Unit tests for GPT-OSS core architecture behavior."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from configuration_gpt_oss import GptOssConfig
from modeling_gpt_oss import GptOssForCausalLM
from utils import YaRNRotaryEmbedding, build_attention_mask


def make_toy_config() -> GptOssConfig:
    """Create a compact configuration shared by GPT-OSS unit tests.

    Parameters:
        None.
    """
    return GptOssConfig(
        num_hidden_layers = 2,
        num_local_experts = 4,
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 32,
        head_dim = 8,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        sliding_window = 3,
        max_position_embeddings = 32,
        num_experts_per_tok = 2,
        rope_parameters = {
            "rope_type": "yarn",
            "factor": 2.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 16,
        },
    )


class GptOssTest(unittest.TestCase):
    """Verify the minimal GPT-OSS configuration, inference, cache, and loss."""

    def setUp(self) -> None:
        """Create a deterministic model and token batch for each test.

        Parameters:
            None.
        """
        torch.manual_seed(0)
        self.config = make_toy_config()
        self.model = GptOssForCausalLM(self.config)
        self.token_ids = torch.tensor([[1, 4, 7, 9, 3], [2, 5, 8, 6, 1]])

    def test_official_default_configuration(self) -> None:
        """Keep public GPT-OSS defaults and their derived layer types.

        Parameters:
            None.
        """
        config = GptOssConfig()
        self.assertEqual(config.num_hidden_layers, 36)
        self.assertEqual(config.num_local_experts, 128)
        self.assertEqual(config.hidden_size, 2880)
        self.assertEqual(config.num_attention_heads, 64)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.layer_types[:4], ["sliding_attention", "full_attention"] * 2)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")

    def test_config_derivation_and_alias(self) -> None:
        """Derive head dimension and accept the num_experts compatibility alias.

        Parameters:
            None.
        """
        config = GptOssConfig(
            hidden_size = 32,
            num_attention_heads = 4,
            num_key_value_heads = None,
            head_dim = None,
            num_experts = 5,
            num_experts_per_tok = 2,
        )
        self.assertEqual(config.head_dim, 8)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.num_local_experts, 5)

    def test_yarn_shapes_and_scaling(self) -> None:
        """Produce finite YaRN tensors whose magnitude changes with the factor.

        Parameters:
            None.
        """
        positions = torch.arange(5).unsqueeze(0)
        hidden_states = torch.zeros(1, 5, self.config.hidden_size)
        rotary = YaRNRotaryEmbedding(self.config)
        cos, sin = rotary(hidden_states, positions)
        self.assertEqual(cos.shape, (1, 5, self.config.head_dim // 2))
        self.assertTrue(torch.isfinite(cos).all())
        self.assertTrue(torch.isfinite(sin).all())

        unscaled_config = make_toy_config()
        unscaled_config.rope_parameters["factor"] = 1.0
        unscaled_rotary = YaRNRotaryEmbedding(unscaled_config)
        self.assertNotEqual(rotary.attention_scaling, unscaled_rotary.attention_scaling)

    def test_forward_shape_causality_and_router(self) -> None:
        """Check logits, causal isolation, and normalized Top-k routing weights.

        Parameters:
            None.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.token_ids, use_cache = False, output_router_logits = True)
            changed_tokens = self.token_ids.clone()
            changed_tokens[:, -1] = 11
            changed_output = self.model(changed_tokens, use_cache = False, output_router_logits = True)

        self.assertEqual(output.logits.shape, (2, 5, self.config.vocab_size))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.allclose(output.logits[:, :-1], changed_output.logits[:, :-1], atol = 1e-6))
        self.assertEqual(len(output.router_logits), self.config.num_hidden_layers)

        router = self.model.model.layers[0].mlp.router
        flat_states = self.model.model.embed_tokens(self.token_ids).reshape(-1, self.config.hidden_size)
        _, routing_weights, routing_indices = router(flat_states)
        self.assertTrue(torch.allclose(routing_weights.sum(dim = -1), torch.ones(routing_weights.shape[0])))
        self.assertTrue(((routing_indices >= 0) & (routing_indices < self.config.num_local_experts)).all())

    def test_sliding_window_mask(self) -> None:
        """Restrict a sliding-attention query to its configured visible window.

        Parameters:
            None.
        """
        mask = build_attention_mask(
            attention_mask = None,
            query_length = 1,
            key_length = 5,
            past_length = 4,
            layer_type = "sliding_attention",
            sliding_window = 3,
            device = torch.device("cpu"),
            dtype = torch.float32,
        )
        self.assertLess(mask[0, 0, 0, 0].item(), -1e30)
        self.assertLess(mask[0, 0, 0, 1].item(), -1e30)
        self.assertEqual(mask[0, 0, 0, 2].item(), 0.0)

    def test_padding_mask_blocks_invalid_keys(self) -> None:
        """Add padding exclusions on top of the causal attention mask.

        Parameters:
            None.
        """
        mask = build_attention_mask(
            attention_mask = torch.tensor([[1, 0, 1]]),
            query_length = 3,
            key_length = 3,
            past_length = 0,
            layer_type = "full_attention",
            sliding_window = None,
            device = torch.device("cpu"),
            dtype = torch.float32,
        )
        self.assertLess(mask[0, 0, 2, 1].item(), -1e30)
        self.assertEqual(mask[0, 0, 2, 0].item(), 0.0)
        self.assertEqual(mask[0, 0, 2, 2].item(), 0.0)

    def test_kv_cache_matches_full_forward(self) -> None:
        """Match a full forward pass with prefill plus cached continuation.

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

    def test_training_losses_and_gradients(self) -> None:
        """Compute CE plus MoE loss and backpropagate into router and experts.

        Parameters:
            None.
        """
        self.model.train()
        output = self.model(self.token_ids, labels = self.token_ids)
        self.assertTrue(torch.isfinite(output.cross_entropy_loss))
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(self.model.model.layers[0].mlp.router.weight.grad)
        self.assertIsNotNone(self.model.model.layers[0].mlp.experts.gate_up_proj.grad)


if __name__ == "__main__":
    unittest.main()
