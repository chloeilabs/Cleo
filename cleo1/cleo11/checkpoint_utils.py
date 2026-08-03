"""Shared Cleo 1.1 checkpoint load/save helpers for post-pretrain stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint
from ..tokenizer import ByteBPETokenizer
from .config import Cleo11Config, Cleo11ModelConfig
from .identity import model_identity_metadata
from .model import Cleo11Transformer
from .pretrain import build_optimizer, select_cleo11_device


@dataclass
class LoadedCleo11:
    model: Cleo11Transformer
    model_config: Cleo11ModelConfig
    tokenizer: ByteBPETokenizer
    tokenizer_path: Path
    tokenizer_checksum: str
    device: torch.device
    source_checkpoint: dict[str, Any]
    source_path: Path


def resolve_tokenizer_path(
    config: Cleo11Config,
    checkpoint: dict[str, Any],
    tokenizer_path: str | Path | None,
) -> Path:
    if tokenizer_path is not None:
        return Path(tokenizer_path)
    manifest = checkpoint.get("data_manifest") or {}
    nested = manifest.get("tokenizer") if isinstance(manifest, dict) else None
    if isinstance(nested, dict) and nested.get("path"):
        return Path(str(nested["path"]))
    return Path(config.prep.tokenizer_path)


def load_cleo11_for_finetune(
    config: Cleo11Config,
    checkpoint_path: str | Path,
    *,
    requested_device: str = "auto",
    tokenizer_path: str | Path | None = None,
) -> LoadedCleo11:
    source = Path(checkpoint_path)
    checkpoint = load_checkpoint(source, map_location="cpu")
    if checkpoint.get("model_id") not in {None, "cleo-1.1"}:
        raise ValueError(f"unexpected model_id in checkpoint: {checkpoint.get('model_id')}")
    tokenizer_file = resolve_tokenizer_path(config, checkpoint, tokenizer_path)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"tokenizer missing: {tokenizer_file}")
    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    tokenizer_checksum = ByteBPETokenizer.checksum(tokenizer_file)
    expected = checkpoint.get("tokenizer_checksum")
    if expected and expected != tokenizer_checksum:
        raise RuntimeError("checkpoint tokenizer checksum does not match tokenizer file")

    model_config = Cleo11ModelConfig(**checkpoint["model_config"])
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config = Cleo11ModelConfig(
            **{**asdict(model_config), "vocab_size": tokenizer.vocab_size}
        )
    device = select_cleo11_device(requested_device)
    model = Cleo11Transformer(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return LoadedCleo11(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_file,
        tokenizer_checksum=tokenizer_checksum,
        device=device,
        source_checkpoint=checkpoint,
        source_path=source,
    )


def save_cleo11_checkpoint(
    path: str | Path,
    *,
    model: Cleo11Transformer,
    model_config: Cleo11ModelConfig,
    stage: str,
    step: int,
    tokenizer_checksum: str,
    source_checkpoint: dict[str, Any],
    extra: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    destination = Path(path)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_id": "cleo-1.1",
        "stage": stage,
        "step": step,
        "model_config": asdict(model_config),
        "model_state": model.state_dict(),
        "tokenizer_checksum": tokenizer_checksum,
        "parameter_count": model.parameter_count(),
        "identity": model_identity_metadata(),
        "rng_state": capture_rng_state(),
        "source_checkpoint": {
            "path": str(source_checkpoint.get("path", "")),
            "stage": source_checkpoint.get("stage"),
            "step": source_checkpoint.get("step"),
            "best_validation_loss": source_checkpoint.get("best_validation_loss"),
        },
        "data_manifest": source_checkpoint.get("data_manifest"),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, destination)
    return destination


def make_finetune_optimizer(
    model: torch.nn.Module,
    config: Cleo11Config,
    *,
    learning_rate: float,
) -> torch.optim.AdamW:
    optimizer = build_optimizer(model, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return optimizer
