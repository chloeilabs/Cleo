from __future__ import annotations

import torch

from cleo1.config import ModelConfig, load_config
from cleo1.model import CleoTransformer


def small_config(dropout: float = 0.0) -> ModelConfig:
    return ModelConfig(
        vocab_size=280,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        ffn_size=64,
        dropout=dropout,
        bias=True,
    )


def test_default_parameter_count_matches_design():
    config = load_config("configs/tinystories_m4.toml")
    model = CleoTransformer(config.model)
    assert model.parameter_count() == 7_809_024


def test_forward_backward_shapes_and_finite_loss():
    torch.manual_seed(1)
    model = CleoTransformer(small_config(dropout=0.1))
    inputs = torch.randint(0, 280, (4, 16))
    targets = torch.randint(0, 280, (4, 16))
    logits, loss = model(inputs, targets)
    assert logits.shape == (4, 16, 280)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_causal_mask_prevents_future_tokens_from_changing_past_logits():
    torch.manual_seed(2)
    model = CleoTransformer(small_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 99, 98, 97]])
    first_logits, _ = model(first)
    second_logits, _ = model(second)
    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2], rtol=0, atol=0)


def test_cpu_generation_is_seeded():
    model = CleoTransformer(small_config()).eval()
    prompt = torch.tensor([[278, 1, 2]])
    torch.manual_seed(42)
    first = model.generate(prompt.clone(), eos_id=279, max_new_tokens=10, temperature=0.8, top_k=20)
    torch.manual_seed(42)
    second = model.generate(prompt.clone(), eos_id=279, max_new_tokens=10, temperature=0.8, top_k=20)
    assert torch.equal(first, second)


def test_cached_forward_matches_full_forward():
    torch.manual_seed(3)
    model = CleoTransformer(small_config()).eval()
    tokens = torch.randint(0, 280, (2, 11))
    full_logits, _ = model(tokens)

    first_logits, cache = model.forward_with_cache(tokens[:, :4])
    second_logits, cache = model.forward_with_cache(tokens[:, 4:8], cache)
    third_logits, cache = model.forward_with_cache(tokens[:, 8:], cache)
    cached_logits = torch.cat((first_logits, second_logits, third_logits), dim=1)

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-6)
    assert len(cache) == model.config.n_layer
    key, value = cache[0]
    assert key.shape == value.shape == (2, model.config.n_head, 11, 16)


def test_cached_generation_matches_uncached_across_context_rebuild():
    torch.manual_seed(4)
    model = CleoTransformer(small_config()).eval()
    prompt = torch.randint(0, 278, (1, 14))
    cached = model.generate(
        prompt.clone(),
        eos_id=279,
        max_new_tokens=8,
        temperature=0.8,
        top_k=1,
        use_cache=True,
    )
    uncached = model.generate(
        prompt.clone(),
        eos_id=279,
        max_new_tokens=8,
        temperature=0.8,
        top_k=1,
        use_cache=False,
    )
    assert torch.equal(cached, uncached)
