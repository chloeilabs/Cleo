from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import Cleo11Config
from .model import Cleo11Transformer


# Approximate training FLOPs ≈ 6ND from Kaplan/Chinchilla-style accounting.
FLOPS_PER_TOKEN_PER_PARAM = 6.0
CHINCHILLA_TOKENS_PER_PARAM = 20.0


@dataclass(frozen=True)
class ComputeEstimate:
    parameter_count: int
    target_tokens: int
    chinchilla_floor_tokens: int
    tokens_per_parameter: float
    meets_chinchilla_floor: bool
    approximate_training_flops: float
    effective_batch_tokens: int
    optimizer_steps: int
    tokens_seen: int
    gpu_hours_at_mfu: dict[str, float]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_parameter_count(config: Cleo11Config) -> int:
    model = Cleo11Transformer(config.model)
    return model.parameter_count()


def estimate_compute(
    config: Cleo11Config,
    *,
    parameter_count: int | None = None,
    mfu_assumptions: dict[str, float] | None = None,
) -> ComputeEstimate:
    """Estimate training compute and whether the token budget meets the Chinchilla floor."""

    params = parameter_count if parameter_count is not None else estimate_parameter_count(config)
    chinchilla_floor = int(params * config.training.tokens_per_parameter_floor)
    target_tokens = config.training.target_tokens
    steps = config.derived_max_steps()
    tokens_seen = steps * config.training.effective_batch_tokens
    flops = FLOPS_PER_TOKEN_PER_PARAM * params * target_tokens

    # Peak dense FLOPs/s assumptions for rough wall-clock planning only.
    assumptions = mfu_assumptions or {
        "a100_80gb_mfu_0.4": 0.4 * 312e12,
        "h100_80gb_mfu_0.4": 0.4 * 989e12,
        "m4_mps_mfu_0.15": 0.15 * 38e12,
    }
    gpu_hours = {
        name: (flops / peak_flops) / 3600.0 for name, peak_flops in assumptions.items()
    }
    notes = (
        "Training FLOPs use the common 6ND approximation and ignore activation checkpointing overhead.",
        "GPU-hour estimates are planning bounds only; realized MFU depends on batching, kernels, and I/O.",
        "The 2.7B-token floor follows ~20 tokens/parameter from Hoffmann et al. 2022 (Chinchilla).",
        "Modern small models often train far beyond Chinchilla; SmolLM-135M used ~600B tokens.",
        "Use cloud GPUs for the 135M pretrain; keep the M4 for development, evaluation, and inference.",
    )
    return ComputeEstimate(
        parameter_count=params,
        target_tokens=target_tokens,
        chinchilla_floor_tokens=chinchilla_floor,
        tokens_per_parameter=target_tokens / params,
        meets_chinchilla_floor=target_tokens >= chinchilla_floor,
        approximate_training_flops=flops,
        effective_batch_tokens=config.training.effective_batch_tokens,
        optimizer_steps=steps,
        tokens_seen=tokens_seen,
        gpu_hours_at_mfu=gpu_hours,
        notes=notes,
    )
