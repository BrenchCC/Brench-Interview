"""Run a small forward, cache-decoding, and training demonstration for Qwen3."""

import torch

from configuration_qwen3 import Qwen3Config
from modeling_qwen3 import Qwen3ForCausalLM


def make_toy_config() -> Qwen3Config:
    """Build a small Qwen3 configuration suitable for a laptop CPU.

    Parameters:
        None.
    """
    return Qwen3Config(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        max_position_embeddings = 64,
        use_sliding_window = True,
        sliding_window = 4,
        max_window_layers = 2,
        rope_parameters = {"rope_theta": 1000000.0},
    )


def main() -> None:
    """Run deterministic Qwen3 inference, cached decode, and one training step.

    Parameters:
        None.
    """
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(make_toy_config())
    token_ids = torch.tensor([[1, 7, 9, 3, 6, 2]])

    # 前向与缓存解码 / Full forward and cached decoding.
    model.eval()
    with torch.no_grad():
        full_output = model(token_ids, use_cache = False)
        prefix_output = model(token_ids[:, :4], use_cache = True)
        cached_output = model(
            token_ids[:, 4:],
            past_key_values = prefix_output.past_key_values,
            use_cache = True,
        )
    print(f"Full logits shape: {tuple(full_output.logits.shape)}")
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
    print(f"Causal language-model loss: {train_output.loss.item():.4f}")


if __name__ == "__main__":
    main()
