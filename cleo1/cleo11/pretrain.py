from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint, restore_rng_state
from ..engine import seed_everything
from ..tokenizer import ByteBPETokenizer
from .config import Cleo11Config, load_cleo11_config
from .corpus import load_train_corpus, load_validation_corpus
from .model import Cleo11Transformer


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")


def select_cleo11_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def _cosine_lr(step: int, *, warmup: int, max_steps: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return floor
    progress = (step - warmup) / max(max_steps - warmup, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + coefficient * (peak - floor)


def _distributed_setup() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def _distributed_cleanup(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def build_optimizer(model: torch.nn.Module, config: Cleo11Config) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.epsilon,
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    corpus,
    device: torch.device,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        inputs, targets = corpus.random_batch(batch_size, generator, device)
        _, loss = model(inputs, targets)
        assert loss is not None
        total += float(loss.item())
    if was_training:
        model.train()
    return total / max(batches, 1)


def _unwrap(model: torch.nn.Module) -> Cleo11Transformer:
    return model.module if isinstance(model, DDP) else model  # type: ignore[return-value]


@dataclass
class PretrainResult:
    steps: int
    best_validation_loss: float
    final_validation_loss: float
    checkpoint: str
    elapsed_seconds: float
    stopping_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pretrain_cleo11(
    config: Cleo11Config,
    *,
    requested_device: str = "auto",
    resume_path: str | None = None,
    max_steps_override: int | None = None,
    microbatch_override: int | None = None,
) -> PretrainResult:
    manifest_path = Path(config.prep.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"prepared-data manifest missing: {manifest_path}; run cleo11-prepare first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_checksum = ByteBPETokenizer.checksum(config.prep.tokenizer_path)
    if tokenizer_checksum != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("tokenizer checksum does not match the prepared-data manifest")

    # Align model vocab with the prepared tokenizer when synthetic prep shrunk it.
    model_config = config.model
    prepared_vocab = int(manifest["tokenizer"]["vocab_size"])
    if prepared_vocab != model_config.vocab_size:
        model_config = type(model_config)(**{**asdict(model_config), "vocab_size": prepared_vocab})

    distributed, rank, local_rank, world_size = _distributed_setup()
    try:
        if distributed and torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = select_cleo11_device(requested_device)
            if distributed and device.type != "cpu":
                # gloo CPU DDP fallback
                device = torch.device("cpu")

        seed_everything(config.training.seed + rank)
        train_corpus = load_train_corpus(manifest, model_config.block_size)
        validation_corpus = load_validation_corpus(manifest, model_config.block_size)
        model = Cleo11Transformer(model_config).to(device)
        if distributed:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        optimizer = build_optimizer(model, config)

        max_steps = max_steps_override or config.derived_max_steps()
        microbatch = microbatch_override or config.training.initial_microbatch_size
        tokens_per_micro = microbatch * model_config.block_size
        accumulation = max(1, config.training.effective_batch_tokens // (tokens_per_micro * world_size))

        if _is_main(rank):
            print(
                f"Cleo 1.1 pretrain device={device} parameters={_unwrap(model).parameter_count():,} "
                f"train_tokens={len(train_corpus):,} validation_tokens={len(validation_corpus):,} "
                f"world_size={world_size} microbatch={microbatch} accumulation={accumulation} "
                f"max_steps={max_steps:,}",
                flush=True,
            )

        data_generator = torch.Generator(device="cpu").manual_seed(config.training.seed + 1 + rank)
        step = 0
        best_loss = math.inf
        initial_loss = math.nan
        base_elapsed = 0.0
        artifacts = Path(config.training.artifacts_dir)
        if _is_main(rank):
            artifacts.mkdir(parents=True, exist_ok=True)
        latest_path = artifacts / "latest.pt"
        best_path = artifacts / "best.pt"
        metrics_path = artifacts / "pretrain_metrics.jsonl"

        if resume_path:
            checkpoint = load_checkpoint(resume_path, map_location=device)
            _unwrap(model).load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            step = int(checkpoint["step"])
            best_loss = float(checkpoint["best_validation_loss"])
            initial_loss = float(checkpoint.get("initial_validation_loss", math.nan))
            base_elapsed = float(checkpoint.get("elapsed_training_seconds", 0.0))
            if "rng_state" in checkpoint:
                restore_rng_state(checkpoint["rng_state"], data_generator)
            if _is_main(rank):
                print(f"Resumed {resume_path} at step {step:,}", flush=True)
        elif _is_main(rank) and metrics_path.exists():
            metrics_path.unlink()

        run_started = time.monotonic()

        def elapsed() -> float:
            return base_elapsed + (time.monotonic() - run_started)

        def save(path: Path) -> None:
            if not _is_main(rank):
                return
            atomic_torch_save(
                {
                    "format_version": 1,
                    "model_id": "cleo-1.1",
                    "stage": "pretrain",
                    "step": step,
                    "model_config": asdict(model_config),
                    "training_config": asdict(config.training),
                    "model_state": _unwrap(model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "rng_state": capture_rng_state(data_generator),
                    "tokenizer_checksum": tokenizer_checksum,
                    "data_manifest": manifest,
                    "best_validation_loss": best_loss,
                    "initial_validation_loss": initial_loss,
                    "elapsed_training_seconds": elapsed(),
                    "microbatch_size": microbatch,
                    "gradient_accumulation_steps": accumulation,
                    "world_size": world_size,
                    "parameter_count": _unwrap(model).parameter_count(),
                },
                path,
            )

        if not resume_path:
            initial_loss = evaluate_model(
                model,
                validation_corpus,
                device,
                batches=config.training.eval_batches,
                batch_size=microbatch,
                seed=config.training.seed + 2 + rank,
            )
            if distributed:
                tensor = torch.tensor([initial_loss], device=device if device.type == "cuda" else "cpu")
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                initial_loss = float(tensor.item())
            best_loss = initial_loss
            if _is_main(rank):
                _append_metric(
                    metrics_path,
                    {
                        "event": "validation",
                        "step": 0,
                        "loss": initial_loss,
                        "perplexity": _perplexity(initial_loss),
                        "elapsed_seconds": elapsed(),
                    },
                )
                print(
                    f"step=0 validation_loss={initial_loss:.4f} perplexity={_perplexity(initial_loss):.2f}",
                    flush=True,
                )
                save(best_path)
                save(latest_path)

        stopping_reason = "max_steps"
        final_loss = best_loss
        while step < max_steps:
            if config.training.max_wall_time_seconds and elapsed() >= config.training.max_wall_time_seconds:
                stopping_reason = "wall_time_limit"
                break
            model.train()
            learning_rate = _cosine_lr(
                step,
                warmup=config.training.warmup_steps,
                max_steps=max_steps,
                peak=config.training.learning_rate,
                floor=config.training.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(accumulation):
                inputs, targets = train_corpus.random_batch(microbatch, data_generator, device)
                _, loss = model(inputs, targets)
                assert loss is not None
                if not bool(torch.isfinite(loss).item()):
                    stopping_reason = "non_finite_loss"
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                accumulated_loss += float(loss.detach().item())
                (loss / accumulation).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            if not bool(torch.isfinite(grad_norm).item()):
                stopping_reason = "non_finite_gradient"
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            step += 1

            if step % config.training.eval_interval == 0 or step >= max_steps:
                eval_loss = evaluate_model(
                    model,
                    validation_corpus,
                    device,
                    batches=config.training.eval_batches,
                    batch_size=microbatch,
                    seed=config.training.seed + 1000 + step + rank,
                )
                if distributed:
                    tensor = torch.tensor(
                        [eval_loss],
                        device=device if device.type == "cuda" else "cpu",
                    )
                    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                    eval_loss = float(tensor.item())
                final_loss = eval_loss
                if _is_main(rank):
                    _append_metric(
                        metrics_path,
                        {
                            "event": "validation",
                            "step": step,
                            "loss": eval_loss,
                            "perplexity": _perplexity(eval_loss),
                            "train_loss": accumulated_loss / accumulation,
                            "learning_rate": learning_rate,
                            "grad_norm": float(grad_norm),
                            "elapsed_seconds": elapsed(),
                        },
                    )
                    print(
                        f"step={step:,} train_loss={accumulated_loss / accumulation:.4f} "
                        f"validation_loss={eval_loss:.4f} perplexity={_perplexity(eval_loss):.2f} "
                        f"lr={learning_rate:.2e}",
                        flush=True,
                    )
                    save(latest_path)
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        save(best_path)

        if _is_main(rank):
            report = {
                "accepted_pretrain_run": True,
                "steps": step,
                "best_validation_loss": best_loss,
                "final_validation_loss": final_loss,
                "checkpoint": str(best_path),
                "elapsed_seconds": elapsed(),
                "stopping_reason": stopping_reason,
                "parameter_count": _unwrap(model).parameter_count(),
                "profile": manifest.get("profile"),
                "synthetic": manifest.get("synthetic", False),
            }
            (artifacts / "pretrain_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return PretrainResult(
            steps=step,
            best_validation_loss=best_loss,
            final_validation_loss=final_loss,
            checkpoint=str(best_path),
            elapsed_seconds=elapsed(),
            stopping_reason=stopping_reason,
        )
    finally:
        _distributed_cleanup(distributed)


def pretrain_from_config_path(
    config_path: str | Path,
    *,
    requested_device: str = "auto",
    resume_path: str | None = None,
    max_steps_override: int | None = None,
    microbatch_override: int | None = None,
) -> dict[str, Any]:
    result = pretrain_cleo11(
        load_cleo11_config(config_path),
        requested_device=requested_device,
        resume_path=resume_path,
        max_steps_override=max_steps_override,
        microbatch_override=microbatch_override,
    )
    return result.to_dict()
