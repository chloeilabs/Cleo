from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch

from .checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint
from .config import AppConfig, ModelConfig
from .data import TokenCorpus
from .engine import (
    FIXED_PROMPTS,
    build_optimizer,
    configure_device,
    evaluate_model,
    seed_everything,
    select_device,
)
from .identity import (
    IDENTITY_EVAL_EXAMPLES,
    IDENTITY_TRAIN_EXAMPLES,
    IdentityExample,
    identity_facts_present,
    identity_leakage_present,
    identity_response_matches,
    model_identity_metadata,
)
from .model import CleoTransformer
from .tokenizer import ByteBPETokenizer


IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodedIdentityExample:
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class IdentityGeneration:
    prompt: str
    continuation: str
    passed: bool


@dataclass(frozen=True)
class StoryLeakageGeneration:
    prompt: str
    seed: int
    continuation: str
    identity_leakage: bool


def encode_identity_example(
    tokenizer: ByteBPETokenizer,
    example: IdentityExample,
    *,
    block_size: int,
) -> EncodedIdentityExample:
    prompt_ids = tokenizer.encode(example.prompt.rstrip(), bos=True)
    answer_ids = tokenizer.encode(f"\n\n{example.answer}", eos=True)
    sequence = prompt_ids + answer_ids
    if len(sequence) > block_size + 1:
        raise ValueError(
            f"identity example requires {len(sequence) - 1} positions, block size is {block_size}"
        )
    inputs = sequence[:-1]
    targets = sequence[1:]
    prompt_target_count = max(len(prompt_ids) - 1, 0)
    targets[:prompt_target_count] = [IGNORE_INDEX] * prompt_target_count
    return EncodedIdentityExample(tuple(inputs), tuple(targets))


def identity_batch(
    tokenizer: ByteBPETokenizer,
    examples: Iterable[IdentityExample],
    *,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        encode_identity_example(tokenizer, example, block_size=block_size)
        for example in examples
    ]
    if not rows:
        raise ValueError("at least one identity example is required")
    width = max(len(row.inputs) for row in rows)
    inputs = torch.full((len(rows), width), tokenizer.eos_id, dtype=torch.long)
    targets = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for index, row in enumerate(rows):
        inputs[index, : len(row.inputs)] = torch.tensor(row.inputs, dtype=torch.long)
        targets[index, : len(row.targets)] = torch.tensor(row.targets, dtype=torch.long)
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def evaluate_identity_loss(
    model: CleoTransformer,
    tokenizer: ByteBPETokenizer,
    device: torch.device,
    examples: Iterable[IdentityExample] = IDENTITY_EVAL_EXAMPLES,
) -> float:
    inputs, targets = identity_batch(
        tokenizer,
        examples,
        block_size=model.config.block_size,
        device=device,
    )
    model.eval()
    _, loss = model(inputs, targets)
    assert loss is not None
    return float(loss.item())


@torch.no_grad()
def evaluate_identity_generation(
    model: CleoTransformer,
    tokenizer: ByteBPETokenizer,
    device: torch.device,
    examples: Iterable[IdentityExample] = IDENTITY_EVAL_EXAMPLES,
    *,
    max_new_tokens: int = 72,
) -> list[IdentityGeneration]:
    model.eval()
    results: list[IdentityGeneration] = []
    for example in examples:
        prompt_ids = tokenizer.encode(example.prompt, bos=True)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            tokens,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            min_new_tokens=0,
            use_cache=True,
        )
        full_text = tokenizer.decode(generated[0].tolist())
        continuation = full_text[len(example.prompt) :].strip()
        results.append(
            IdentityGeneration(
                prompt=example.prompt,
                continuation=continuation,
                passed=(
                    identity_facts_present(continuation)
                    and identity_response_matches(continuation)
                ),
            )
        )
    return results


def identity_accuracy(results: Iterable[IdentityGeneration]) -> float:
    rows = list(results)
    if not rows:
        raise ValueError("identity evaluation requires at least one result")
    return sum(row.passed for row in rows) / len(rows)


@torch.no_grad()
def evaluate_story_identity_leakage(
    model: CleoTransformer,
    tokenizer: ByteBPETokenizer,
    device: torch.device,
    *,
    seed: int,
    max_new_tokens: int = 300,
) -> list[StoryLeakageGeneration]:
    model.eval()
    results: list[StoryLeakageGeneration] = []
    for index, prompt in enumerate(FIXED_PROMPTS):
        generation_seed = seed + 100 + index
        seed_everything(generation_seed)
        prompt_ids = tokenizer.encode(prompt, bos=True)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            tokens,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
            min_new_tokens=100,
            use_cache=True,
        )
        full_text = tokenizer.decode(generated[0].tolist())
        continuation = full_text[len(prompt) :].strip()
        results.append(
            StoryLeakageGeneration(
                prompt=prompt,
                seed=generation_seed,
                continuation=continuation,
                identity_leakage=identity_leakage_present(continuation),
            )
        )
    return results


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _examples_checksum() -> str:
    payload = [asdict(example) for example in (*IDENTITY_TRAIN_EXAMPLES, *IDENTITY_EVAL_EXAMPLES)]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _generation_rows(results: Iterable[IdentityGeneration]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def _story_generation_rows(
    results: Iterable[StoryLeakageGeneration],
) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def fine_tune_identity(
    checkpoint_path: str | Path,
    output_path: str | Path,
    tokenizer_path: str | Path,
    *,
    requested_device: str = "auto",
    steps: int = 800,
    learning_rate: float = 2e-5,
    story_weight: float = 4.0,
    identity_batch_size: int = 8,
    story_batch_size: int = 16,
    eval_interval: int = 100,
    validation_batches: int = 50,
    max_story_loss_increase: float = 0.03,
    required_identity_accuracy: float = 1.0,
    seed: int = 1337,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if story_weight < 0:
        raise ValueError("story weight cannot be negative")
    if identity_batch_size < 1 or story_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if eval_interval < 1 or validation_batches < 1:
        raise ValueError("evaluation settings must be positive")
    if max_story_loss_increase < 0:
        raise ValueError("story loss tolerance cannot be negative")
    if not 0 < required_identity_accuracy <= 1:
        raise ValueError("required identity accuracy must be in (0, 1]")

    source = Path(checkpoint_path)
    destination = Path(output_path)
    report_path = destination.parent / "identity_finetune_report.json"
    metrics_path = destination.parent / "identity_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    base_checkpoint = load_checkpoint(source, map_location="cpu")
    config = AppConfig.from_dict(base_checkpoint["app_config"])
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    tokenizer_checksum = ByteBPETokenizer.checksum(tokenizer_path)
    if base_checkpoint["tokenizer_checksum"] != tokenizer_checksum:
        raise RuntimeError("checkpoint tokenizer checksum mismatch")

    device = select_device(requested_device)
    configure_device(device, config)
    seed_everything(seed)
    model = CleoTransformer(ModelConfig(**base_checkpoint["model_config"]))
    model.load_state_dict(base_checkpoint["model_state"])
    model.to(device)
    train_corpus = TokenCorpus(config.data.train_tokens, config.model.block_size)
    validation_corpus = TokenCorpus(config.data.validation_tokens, config.model.block_size)
    story_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    example_generator = random.Random(seed + 2)

    print("Measuring base story retention and held-out identity behavior", flush=True)
    baseline_story_loss = evaluate_model(
        model,
        validation_corpus,
        device,
        batches=validation_batches,
        batch_size=story_batch_size,
        seed=seed + 3,
    )
    baseline_identity_loss = evaluate_identity_loss(model, tokenizer, device)
    baseline_generations = evaluate_identity_generation(model, tokenizer, device)
    baseline_accuracy = identity_accuracy(baseline_generations)
    baseline_story_generations = evaluate_story_identity_leakage(
        model, tokenizer, device, seed=seed
    )
    baseline_leakage_count = sum(row.identity_leakage for row in baseline_story_generations)
    _append_jsonl(
        metrics_path,
        {
            "event": "baseline",
            "step": 0,
            "story_validation_loss": baseline_story_loss,
            "identity_loss": baseline_identity_loss,
            "identity_accuracy": baseline_accuracy,
            "story_identity_leakage_count": baseline_leakage_count,
        },
    )
    print(
        f"baseline story_loss={baseline_story_loss:.4f} "
        f"identity_loss={baseline_identity_loss:.4f} identity_accuracy={baseline_accuracy:.0%}",
        flush=True,
    )

    optimizer = build_optimizer(model, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    started = time.monotonic()
    completed_steps = 0
    accepted = False
    final_story_loss = baseline_story_loss
    final_identity_loss = baseline_identity_loss
    final_generations = baseline_generations
    final_accuracy = baseline_accuracy
    final_story_generations = baseline_story_generations
    final_leakage_count = baseline_leakage_count

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        selected = [
            IDENTITY_TRAIN_EXAMPLES[example_generator.randrange(len(IDENTITY_TRAIN_EXAMPLES))]
            for _ in range(identity_batch_size)
        ]
        identity_inputs, identity_targets = identity_batch(
            tokenizer,
            selected,
            block_size=config.model.block_size,
            device=device,
        )
        _, identity_loss = model(identity_inputs, identity_targets)
        assert identity_loss is not None
        if not bool(torch.isfinite(identity_loss).item()):
            raise FloatingPointError(f"non-finite identity loss at step {step}")

        denominator = 1.0 + story_weight
        (identity_loss / denominator).backward()
        story_inputs, story_targets = train_corpus.random_batch(
            story_batch_size, story_generator, device
        )
        _, story_loss = model(story_inputs, story_targets)
        assert story_loss is not None
        if not bool(torch.isfinite(story_loss).item()):
            raise FloatingPointError(f"non-finite story loss at step {step}")
        if story_weight:
            (story_loss * story_weight / denominator).backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        completed_steps = step

        if step % 25 == 0:
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.monotonic() - started
            print(
                f"identity_step={step}/{steps} identity_loss={identity_loss.item():.4f} "
                f"story_loss={story_loss.item():.4f} grad_norm={gradient_norm.item():.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            _append_jsonl(
                metrics_path,
                {
                    "event": "train",
                    "step": step,
                    "identity_loss": float(identity_loss.item()),
                    "story_loss": float(story_loss.item()),
                    "gradient_norm": float(gradient_norm.item()),
                    "learning_rate": learning_rate,
                    "elapsed_seconds": elapsed,
                },
            )

        if step % eval_interval == 0 or step == steps:
            final_story_loss = evaluate_model(
                model,
                validation_corpus,
                device,
                batches=validation_batches,
                batch_size=story_batch_size,
                seed=seed + 3,
            )
            final_identity_loss = evaluate_identity_loss(model, tokenizer, device)
            final_generations = evaluate_identity_generation(model, tokenizer, device)
            final_accuracy = identity_accuracy(final_generations)
            final_story_generations = evaluate_story_identity_leakage(
                model, tokenizer, device, seed=seed
            )
            final_leakage_count = sum(
                row.identity_leakage for row in final_story_generations
            )
            story_ratio = final_story_loss / baseline_story_loss
            accepted = (
                final_accuracy >= required_identity_accuracy
                and story_ratio <= 1.0 + max_story_loss_increase
                and final_leakage_count == 0
            )
            _append_jsonl(
                metrics_path,
                {
                    "event": "evaluation",
                    "step": step,
                    "story_validation_loss": final_story_loss,
                    "story_loss_ratio": story_ratio,
                    "identity_loss": final_identity_loss,
                    "identity_accuracy": final_accuracy,
                    "story_identity_leakage_count": final_leakage_count,
                    "accepted": accepted,
                },
            )
            print(
                f"identity_eval step={step} story_loss={final_story_loss:.4f} "
                f"retention={story_ratio:.3f} identity_loss={final_identity_loss:.4f} "
                f"identity_accuracy={final_accuracy:.0%} "
                f"identity_leaks={final_leakage_count} accepted={accepted}",
                flush=True,
            )
            if accepted:
                break

    elapsed_seconds = time.monotonic() - started
    story_loss_ratio = final_story_loss / baseline_story_loss
    adaptation = {
        "type": "identity_supervised_fine_tune",
        "accepted": accepted,
        "base_checkpoint": str(source),
        "base_checkpoint_sha256": _sha256_file(source),
        "foundation_training_step": int(base_checkpoint["step"]),
        "completed_steps": completed_steps,
        "learning_rate": learning_rate,
        "story_weight": story_weight,
        "identity_batch_size": identity_batch_size,
        "story_batch_size": story_batch_size,
        "seed": seed,
        "train_examples": len(IDENTITY_TRAIN_EXAMPLES),
        "evaluation_examples": len(IDENTITY_EVAL_EXAMPLES),
        "examples_sha256": _examples_checksum(),
        "baseline_story_validation_loss": baseline_story_loss,
        "final_story_validation_loss": final_story_loss,
        "story_loss_ratio": story_loss_ratio,
        "max_story_loss_increase": max_story_loss_increase,
        "baseline_identity_loss": baseline_identity_loss,
        "final_identity_loss": final_identity_loss,
        "baseline_identity_accuracy": baseline_accuracy,
        "final_identity_accuracy": final_accuracy,
        "required_identity_accuracy": required_identity_accuracy,
        "baseline_story_identity_leakage_count": baseline_leakage_count,
        "final_story_identity_leakage_count": final_leakage_count,
        "elapsed_seconds": elapsed_seconds,
    }
    payload = dict(base_checkpoint)
    payload.update(
        {
            "format_version": 2,
            "identity": model_identity_metadata(),
            "adaptation": adaptation,
            "model_state": _cpu_tree(model.state_dict()),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "rng_state": capture_rng_state(story_generator),
            "step": int(base_checkpoint["step"]) + completed_steps,
            "best_validation_loss": min(
                float(base_checkpoint["best_validation_loss"]), final_story_loss
            ),
            "elapsed_training_seconds": float(
                base_checkpoint.get("elapsed_training_seconds", 0.0)
            )
            + elapsed_seconds,
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    atomic_torch_save(payload, destination)

    report = {
        **adaptation,
        "checkpoint": str(destination),
        "checkpoint_sha256": _sha256_file(destination),
        "identity": model_identity_metadata(),
        "baseline_generations": _generation_rows(baseline_generations),
        "final_generations": _generation_rows(final_generations),
        "baseline_story_generations": _story_generation_rows(baseline_story_generations),
        "final_story_generations": _story_generation_rows(final_story_generations),
        "metrics_path": str(metrics_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return report
