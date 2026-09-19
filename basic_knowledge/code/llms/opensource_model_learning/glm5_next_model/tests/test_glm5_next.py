"""Toy tests for the framework-free GLM5-Next learning port."""

import os
import sys
import unittest

import torch

sys.path.append(os.getcwd())

from basic_knowledge.code.llms.opensource_model_learning.glm5_next_model import (
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    Glm5NextForCausalLM,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
    Glm5NextVisionModel,
)


def make_text_config() -> Glm5NextTextConfig:
    """Create a tiny configuration covering both KDA and DSA paths.

    Parameters:
        None.
    """
    return Glm5NextTextConfig(
        vocab_size = 47,
        hidden_size = 32,
        intermediate_size = 64,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        q_lora_rank = 8,
        kv_lora_rank = 8,
        qk_nope_head_dim = 8,
        qk_rope_head_dim = 0,
        v_head_dim = 8,
        index_n_heads = 2,
        index_head_dim = 4,
        index_topk = 4,
        index_kpool = 2,
        linear_head_dim = 4,
        linear_num_heads = 2,
        linear_conv_kernel_dim = 3,
        hc_mult = 2,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        n_group = 1,
        topk_group = 1,
        layer_types = ["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types = ["dense", "sparse"],
        indexer_types = ["full", "full"],
    )


class Glm5NextTest(unittest.TestCase):
    """Exercise official GLM5 pathways at toy scale."""

    def test_hybrid_forward_backward_and_cache(self) -> None:
        """Verify KDA, DSA, mHC, MoE loss, and cache decoding agree.

        Parameters:
            None.
        """
        torch.manual_seed(0)
        model = Glm5NextForCausalLM(make_text_config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        full = model(input_ids, labels = input_ids, use_cache = True, output_router_logits = True)
        self.assertTrue(torch.isfinite(full.loss))
        self.assertEqual(full.logits.shape, (1, 4, 47))
        self.assertEqual(len(full.router_logits), 1)
        full.loss.backward()
        prefill = model(input_ids[:, :3], use_cache = True)
        decode = model(input_ids[:, 3:], past_key_values = prefill.past_key_values, use_cache = True)
        self.assertTrue(torch.allclose(full.logits[:, -1], decode.logits[:, -1], atol = 1e-5, rtol = 1e-4))

    def test_vision_tower(self) -> None:
        """Verify Conv3D patches, packed visual attention, and merger shapes.

        Parameters:
            None.
        """
        config = Glm5NextVisionConfig(
            depth = 1,
            hidden_size = 16,
            num_heads = 2,
            intermediate_size = 32,
            out_hidden_size = 24,
            projection_intermediate_size = 48,
            patch_size = 4,
            temporal_patch_size = 1,
            spatial_merge_size = 2,
            image_size = 8,
        )
        model = Glm5NextVisionModel(config)
        patches = torch.randn(4, 3 * 4 * 4)
        outputs = model(patches, torch.tensor([[1, 2, 2]]))
        self.assertEqual(outputs.last_hidden_state.shape, (1, 24))
        self.assertEqual(outputs.pooler_output.shape, (1, 24))
        self.assertTrue(torch.isfinite(outputs.pooler_output).all())

    def test_conditional_image_feature_injection(self) -> None:
        """Verify source image-placeholder replacement reaches the text decoder.

        Parameters:
            None.
        """
        text_config = make_text_config()
        text_config.num_hidden_layers = 1
        text_config.layer_types = ["linear_attention"]
        text_config.mlp_layer_types = ["dense"]
        text_config.indexer_types = ["full"]
        vision_config = Glm5NextVisionConfig(
            depth = 1,
            hidden_size = 16,
            num_heads = 2,
            intermediate_size = 32,
            out_hidden_size = 32,
            projection_intermediate_size = 48,
            patch_size = 4,
            temporal_patch_size = 1,
            spatial_merge_size = 2,
        )
        config = Glm5NextConfig(
            text_config = text_config,
            vision_config = vision_config,
            image_token_id = 10,
            video_start_token_id = 11,
            video_end_token_id = 12,
        )
        model = Glm5NextForConditionalGeneration(config)
        outputs = model(
            torch.tensor([[1, 10, 2]]),
            pixel_values = torch.randn(4, 3 * 4 * 4),
            image_grid_thw = torch.tensor([[1, 2, 2]]),
        )
        self.assertEqual(outputs.logits.shape, (1, 3, 47))
        self.assertTrue(torch.isfinite(outputs.logits).all())


if __name__ == "__main__":
    unittest.main()
