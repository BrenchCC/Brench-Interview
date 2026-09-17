"""Run a small forward, cache-decoding, and training demonstration for GPT-OSS."""

import torch

from configuration_gpt_oss import GptOssConfig
from modeling_gpt_oss import GptOssForCausalLM


def make_toy_config() -> GptOssConfig:
    """Build a small GPT-OSS configuration suitable for a laptop CPU.

    Parameters:
        None.
    """
    return GptOssConfig(
        num_hidden_layers = 2,
        num_local_experts = 4,
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 64,
        head_dim = 16,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        sliding_window = 4,
        max_position_embeddings = 64,
        num_experts_per_tok = 2,
        rope_parameters = {
            "rope_type": "yarn",
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 16,
        },
    )


def main() -> None:
    """Run a deterministic toy forward pass, cached decode, and training step.

    Parameters:
        None.
    """
    torch.manual_seed(7)
    config = make_toy_config()
    model = GptOssForCausalLM(config)
    token_ids = torch.tensor([[1, 7, 9, 3, 6, 2]])

    # 前向推理 / Forward inference.
    model.eval()
    with torch.no_grad():
        full_output = model(token_ids, use_cache = False, output_router_logits = True)
        prefix_output = model(token_ids[:, :4], use_cache = True)
        cached_output = model(
            token_ids[:, 4:],
            past_key_values = prefix_output.past_key_values,
            use_cache = True,
        )

    print(f"Full logits shape: {tuple(full_output.logits.shape)}")
    print(f"Router layers: {len(full_output.router_logits)}")
    print(
        "Cache matches full forward:",
        torch.allclose(full_output.logits[:, 4:], cached_output.logits, atol = 1e-5),
    )

    # 单步训练 / One training step.
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    train_output = model(token_ids, labels = token_ids)
    train_output.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Cross entropy: {train_output.cross_entropy_loss.item():.4f}")
    print(f"MoE auxiliary loss: {train_output.aux_loss.item():.4f}")
    print(f"Total loss: {train_output.loss.item():.4f}")


if __name__ == "__main__":
    main()
