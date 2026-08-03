from __future__ import annotations

import numpy as np
import pytest
import torch

from cleo1.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    stamp_checkpoint_identity,
)
from cleo1.config import ModelConfig
from cleo1.data import TokenCorpus
from cleo1.model import CleoTransformer


def tiny_config(dropout: float = 0.1) -> ModelConfig:
    return ModelConfig(
        vocab_size=280,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        ffn_size=32,
        dropout=dropout,
        bias=True,
    )


def train_step(model, optimizer, inputs, targets):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    return float(loss.item())


def test_checkpoint_resume_reproduces_exact_next_cpu_step(tmp_path):
    torch.manual_seed(123)
    model = CleoTransformer(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first_inputs = torch.randint(0, 280, (2, 8))
    first_targets = torch.randint(0, 280, (2, 8))
    second_inputs = torch.randint(0, 280, (2, 8))
    second_targets = torch.randint(0, 280, (2, 8))
    train_step(model, optimizer, first_inputs, first_targets)
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state(),
        },
        path,
    )
    train_step(model, optimizer, second_inputs, second_targets)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    resumed = CleoTransformer(tiny_config())
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(path)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint["rng"])
    train_step(resumed, resumed_optimizer, second_inputs, second_targets)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_checkpoint_identity_stamp_is_atomic_and_idempotent(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"format_version": 1, "model_state": {"weight": torch.ones(2)}}, path)
    assert stamp_checkpoint_identity(path) is True
    checkpoint = load_checkpoint(path)
    assert checkpoint["format_version"] == 2
    assert checkpoint["identity"]["company_name"] == "Cleo AI"
    assert checkpoint["identity"]["model_name"] == "Cleo 1"
    assert checkpoint["identity"]["model_id"] == "cleo-1"
    assert stamp_checkpoint_identity(path) is False


def test_tiny_corpus_overfits_by_twenty_percent():
    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=280,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=24,
        ffn_size=48,
        dropout=0.0,
        bias=True,
    )
    model = CleoTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    sequence = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 1, 2, 3, 4, 1]])
    with torch.no_grad():
        _, initial = model(sequence, targets)
    assert initial is not None
    for _ in range(60):
        train_step(model, optimizer, sequence, targets)
    with torch.no_grad():
        _, final = model(sequence, targets)
    assert final is not None
    assert final.item() < initial.item() * 0.8


def test_token_corpus_returns_shifted_batches(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype="<u2").tofile(path)
    corpus = TokenCorpus(path, block_size=8)
    starts = torch.tensor([0, 10])
    inputs, targets = corpus.batch_from_starts(starts, torch.device("cpu"))
    assert inputs.tolist()[0] == list(range(8))
    assert targets.tolist()[0] == list(range(1, 9))
    assert inputs.tolist()[1] == list(range(10, 18))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_smoke_forward_backward():
    torch.manual_seed(11)
    device = torch.device("mps")
    model = CleoTransformer(tiny_config()).to(device)
    inputs = torch.randint(0, 280, (4, 8), device=device)
    targets = torch.randint(0, 280, (4, 8), device=device)
    _, loss = model(inputs, targets)
    assert loss is not None and bool(torch.isfinite(loss).item())
    loss.backward()
    torch.mps.synchronize()
