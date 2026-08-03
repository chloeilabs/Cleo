from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from ..checkpoint import atomic_torch_save, capture_rng_state
from ..engine import seed_everything, select_device
from .compute import estimate_compute
from .config import Cleo11Config, load_cleo11_config
from .data_manifest import mixture_manifest, write_mixture_manifest
from .evaluation import evaluation_contract, write_evaluation_contract
from .model import Cleo11Transformer


def _cosine_lr(step: int, *, warmup: int, max_steps: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return floor
    progress = (step - warmup) / max(max_steps - warmup, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + coefficient * (peak - floor)


def write_training_spec(config: Cleo11Config, artifacts_dir: str | Path) -> dict[str, Any]:
    destination = Path(artifacts_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model = Cleo11Transformer(config.model)
    compute = estimate_compute(config, parameter_count=model.parameter_count())
    mixture = write_mixture_manifest(config, destination / "dataset_manifest.json")
    contract = write_evaluation_contract(config, destination / "evaluation_contract.json")
    payload = {
        "model_id": "cleo-1.1",
        "model_name": "Cleo 1.1",
        "architecture": {
            "family": "decoder-only transformer",
            "features": ["RoPE", "RMSNorm", "SwiGLU", "grouped-query attention", "tied embeddings"],
            "config": asdict(config.model),
            "parameter_count": model.parameter_count(),
            "parameter_breakdown": model.parameter_breakdown(),
        },
        "training": asdict(config.training) | {"derived_max_steps": config.derived_max_steps()},
        "compute": compute.to_dict(),
        "dataset_manifest": mixture,
        "evaluation_contract": contract,
        "predecessor": "cleo-1-general-alpha-01",
    }
    (destination / "training_spec.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_smoke_train(
    config: Cleo11Config,
    *,
    requested_device: str = "cpu",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Tiny end-to-end Cleo 1.1 train/eval/checkpoint smoke test on synthetic tokens."""

    seed_everything(config.training.seed)
    device = select_device(requested_device)
    artifacts = Path(output_dir or config.training.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    write_training_spec(config, artifacts)

    model = Cleo11Transformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay,
    )
    steps = config.training.smoke_steps
    batch_size = config.training.smoke_batch_size
    block_size = config.model.block_size
    vocab = config.model.vocab_size
    losses: list[float] = []
    started = time.perf_counter()

    model.train()
    for step in range(steps):
        learning_rate = _cosine_lr(
            step,
            warmup=min(2, steps),
            max_steps=steps,
            peak=config.training.learning_rate,
            floor=config.training.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        inputs = torch.randint(0, vocab, (batch_size, block_size), device=device)
        targets = torch.randint(0, vocab, (batch_size, block_size), device=device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        eval_inputs = torch.randint(0, vocab, (batch_size, block_size), device=device)
        eval_targets = torch.randint(0, vocab, (batch_size, block_size), device=device)
        _, eval_loss = model(eval_inputs, eval_targets)
        assert eval_loss is not None
        prompt = torch.randint(0, vocab - 2, (1, min(8, block_size)), device=device)
        generated = model.generate(
            prompt,
            eos_id=vocab - 1,
            max_new_tokens=4,
            temperature=1.0,
            top_k=1,
            use_cache=True,
        )

    checkpoint_path = artifacts / "cleo11-smoke.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "model_id": "cleo-1.1",
            "stage": "smoke",
            "step": steps,
            "model_config": asdict(config.model),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "smoke_losses": losses,
            "eval_loss": float(eval_loss.item()),
            "parameter_count": model.parameter_count(),
            "dataset_manifest": mixture_manifest(config),
            "evaluation_contract": evaluation_contract(config),
        },
        checkpoint_path,
    )

    # Reload parity check
    reloaded = Cleo11Transformer(config.model).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model_state"])
    reloaded.eval()
    with torch.no_grad():
        first_logits, _ = model(eval_inputs)
        second_logits, _ = reloaded(eval_inputs)
        torch.testing.assert_close(first_logits, second_logits, rtol=1e-5, atol=1e-6)

    report = {
        "accepted_smoke": True,
        "device": str(device),
        "steps": steps,
        "parameter_count": model.parameter_count(),
        "train_losses": losses,
        "final_train_loss": losses[-1],
        "eval_loss": float(eval_loss.item()),
        "generated_tokens": generated.size(1) - prompt.size(1),
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "architecture_features": ["RoPE", "RMSNorm", "SwiGLU", "grouped-query attention"],
        "note": "Smoke test uses synthetic tokens only; it does not satisfy release gates.",
    }
    (artifacts / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def smoke_from_config_path(
    config_path: str | Path,
    *,
    requested_device: str = "cpu",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    return run_smoke_train(
        load_cleo11_config(config_path),
        requested_device=requested_device,
        output_dir=output_dir,
    )
