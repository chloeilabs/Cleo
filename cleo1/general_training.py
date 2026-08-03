from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable

import torch

from .checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint
from .config import AppConfig, ModelConfig
from .data import TokenCorpus, sha256_file
from .engine import choose_batch_plan, configure_device, evaluate_model, seed_everything, select_device
from .general_data import (
    IGNORE_INDEX,
    GeneralConfig,
    InstructionExample,
    instruction_batch,
    load_instruction_examples,
    render_instruction_prompt,
)
from .identity import IDENTITY_TRAIN_EXAMPLES, model_identity_metadata
from .identity_tuning import (
    evaluate_identity_generation,
    evaluate_identity_loss,
    identity_accuracy,
    identity_batch,
)
from .model import CleoTransformer
from .tokenizer import ByteBPETokenizer


GENERAL_EVAL_PROMPTS = (
    "Explain why the sky appears blue in two short sentences.",
    "List three practical ways to save energy at home.",
    "What is 7 plus 8? Answer with only the number.",
    "Classify the sentiment as positive or negative: The service was excellent.",
    "Write a short Python function named add that returns the sum of two numbers.",
)


@dataclass(frozen=True)
class GeneralGeneration:
    prompt: str
    seed: int
    response: str


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


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def _runtime_app_config(
    checkpoint: dict[str, Any],
    model_config: ModelConfig,
    general_config: GeneralConfig,
) -> AppConfig:
    value = json.loads(json.dumps(checkpoint["app_config"]))
    value["model"] = asdict(model_config)
    value["training"]["initial_microbatch_size"] = (
        general_config.training.initial_microbatch_size
    )
    value["training"]["effective_batch_tokens"] = (
        general_config.training.effective_batch_tokens
    )
    value["training"]["weight_decay"] = general_config.training.weight_decay
    value["training"]["grad_clip"] = general_config.training.grad_clip
    return AppConfig.from_dict(value)


def expand_context_model(
    checkpoint: dict[str, Any],
    *,
    target_block_size: int,
    seed: int,
) -> CleoTransformer:
    source_config = ModelConfig(**checkpoint["model_config"])
    if target_block_size < source_config.block_size:
        raise ValueError("general context cannot be smaller than the source checkpoint context")
    target_value = asdict(source_config)
    target_value["block_size"] = target_block_size
    target_config = ModelConfig(**target_value)
    seed_everything(seed)
    model = CleoTransformer(target_config)
    if target_block_size == source_config.block_size:
        model.load_state_dict(checkpoint["model_state"])
        return model

    source_state = dict(checkpoint["model_state"])
    source_positions = source_state.pop("position_embedding.weight")
    incompatible = model.load_state_dict(source_state, strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys != ["position_embedding.weight"]:
        raise RuntimeError(
            "checkpoint context expansion found unexpected state keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    with torch.no_grad():
        model.position_embedding.weight[: source_config.block_size].copy_(source_positions)
    return model


def _build_optimizer(
    model: CleoTransformer,
    runtime_config: AppConfig,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(runtime_config.training.beta1, runtime_config.training.beta2),
        eps=runtime_config.training.epsilon,
    )


def _pretrain_learning_rate(step: int, steps: int, config: GeneralConfig) -> float:
    training = config.training
    if step <= training.pretrain_warmup_steps:
        return training.pretrain_learning_rate * step / max(training.pretrain_warmup_steps, 1)
    progress = (step - training.pretrain_warmup_steps) / max(
        steps - training.pretrain_warmup_steps, 1
    )
    coefficient = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return training.pretrain_min_learning_rate + coefficient * (
        training.pretrain_learning_rate - training.pretrain_min_learning_rate
    )


@torch.no_grad()
def evaluate_instruction_loss(
    model: CleoTransformer,
    tokenizer: ByteBPETokenizer,
    examples: Iterable[InstructionExample],
    device: torch.device,
    *,
    batch_size: int,
    limit: int | None = None,
) -> float:
    rows = list(examples)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("instruction evaluation requires at least one example")
    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    for start in range(0, len(rows), batch_size):
        inputs, targets = instruction_batch(
            tokenizer,
            rows[start : start + batch_size],
            block_size=model.config.block_size,
            device=device,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        tokens = int((targets != IGNORE_INDEX).sum().item())
        weighted_loss += float(loss.item()) * tokens
        target_tokens += tokens
    return weighted_loss / target_tokens


@torch.no_grad()
def generate_general_responses(
    model: CleoTransformer,
    tokenizer: ByteBPETokenizer,
    device: torch.device,
    *,
    seed: int,
    prompts: Iterable[str] = GENERAL_EVAL_PROMPTS,
    max_new_tokens: int = 96,
) -> list[GeneralGeneration]:
    model.eval()
    rows: list[GeneralGeneration] = []
    for index, prompt in enumerate(prompts):
        generation_seed = seed + index
        seed_everything(generation_seed)
        rendered = render_instruction_prompt(prompt)
        prompt_ids = tokenizer.encode(rendered, bos=True)
        if len(prompt_ids) > model.config.block_size:
            prompt_ids = prompt_ids[-model.config.block_size :]
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            inputs,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            min_new_tokens=0,
            use_cache=True,
        )
        full_text = tokenizer.decode(generated[0].tolist())
        response = full_text[len(rendered) :].strip()
        rows.append(GeneralGeneration(prompt=prompt, seed=generation_seed, response=response))
    return rows


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def generalize_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    tokenizer_path: str | Path,
    general_config: GeneralConfig,
    *,
    requested_device: str = "auto",
    pretrain_steps: int | None = None,
    instruction_steps: int | None = None,
    max_wall_time_seconds: int | None = None,
    promote_to: str | Path | None = None,
    preserve_base_as: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(checkpoint_path)
    destination = Path(output_path)
    manifest_path = Path(general_config.data.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"general-data manifest is missing: {manifest_path}; run prepare-general first"
        )
    general_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    general_manifest_sha256 = sha256_file(manifest_path)
    checkpoint = load_checkpoint(source, map_location="cpu")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    tokenizer_checksum = ByteBPETokenizer.checksum(tokenizer_path)
    if checkpoint["tokenizer_checksum"] != tokenizer_checksum:
        raise RuntimeError("source checkpoint tokenizer checksum mismatch")
    if general_manifest["tokenizer"]["sha256"] != tokenizer_checksum:
        raise RuntimeError("general-data tokenizer checksum mismatch")

    requested_pretrain_steps = (
        general_config.training.pretrain_steps if pretrain_steps is None else pretrain_steps
    )
    requested_instruction_steps = (
        general_config.training.instruction_steps
        if instruction_steps is None
        else instruction_steps
    )
    wall_limit = (
        general_config.training.max_wall_time_seconds
        if max_wall_time_seconds is None
        else max_wall_time_seconds
    )
    if requested_pretrain_steps < 0 or requested_instruction_steps < 0:
        raise ValueError("generalization steps cannot be negative")
    if requested_pretrain_steps + requested_instruction_steps < 1:
        raise ValueError("at least one generalization step is required")
    if wall_limit < 1:
        raise ValueError("wall-time limit must be positive")

    device = select_device(requested_device)
    base_app_config = AppConfig.from_dict(checkpoint["app_config"])
    configure_device(device, base_app_config)
    seed = general_config.training.seed
    model = expand_context_model(
        checkpoint,
        target_block_size=general_config.training.block_size,
        seed=seed,
    ).to(device)
    runtime_config = _runtime_app_config(checkpoint, model.config, general_config)

    train_corpus = TokenCorpus(
        general_config.data.train_tokens, general_config.training.block_size
    )
    general_train_eval = TokenCorpus(
        general_config.data.train_tokens, base_app_config.model.block_size
    )
    general_validation = TokenCorpus(
        general_config.data.validation_tokens, base_app_config.model.block_size
    )
    general_test = TokenCorpus(
        general_config.data.test_tokens, base_app_config.model.block_size
    )
    general_test_long = TokenCorpus(
        general_config.data.test_tokens, general_config.training.block_size
    )
    story_train = TokenCorpus(
        base_app_config.data.train_tokens, general_config.training.block_size
    )
    story_validation = TokenCorpus(
        base_app_config.data.validation_tokens, base_app_config.model.block_size
    )
    instruction_train_rows = load_instruction_examples(general_config.data.instruction_train)
    instruction_validation_rows = load_instruction_examples(
        general_config.data.instruction_validation
    )
    instruction_test_rows = load_instruction_examples(general_config.data.instruction_test)

    artifacts = destination.parent
    artifacts.mkdir(parents=True, exist_ok=True)
    latest_path = artifacts / "general-latest.pt"
    report_path = artifacts / "generalization_report.json"
    metrics_path = artifacts / "generalization_metrics.jsonl"
    samples_path = artifacts / "general_samples.json"
    if metrics_path.exists():
        metrics_path.unlink()

    evaluation_batch_size = min(general_config.training.retention_batch_size, 16)
    print(
        f"Generalizing on {device.type}: parameters={model.parameter_count():,} "
        f"context={model.config.block_size} general_tokens={len(train_corpus):,} "
        f"instruction_examples={len(instruction_train_rows):,}",
        flush=True,
    )
    print("Measuring source-checkpoint capability baselines", flush=True)
    baseline_general_loss = evaluate_model(
        model,
        general_validation,
        device,
        batches=general_config.training.eval_batches,
        batch_size=evaluation_batch_size,
        seed=seed + 10,
    )
    baseline_instruction_loss = evaluate_instruction_loss(
        model,
        tokenizer,
        instruction_validation_rows,
        device,
        batch_size=evaluation_batch_size,
        limit=general_config.training.instruction_eval_examples,
    )
    baseline_story_loss = evaluate_model(
        model,
        story_validation,
        device,
        batches=general_config.training.eval_batches,
        batch_size=evaluation_batch_size,
        seed=seed + 11,
    )
    baseline_identity_loss = evaluate_identity_loss(model, tokenizer, device)
    baseline_identity_generations = evaluate_identity_generation(model, tokenizer, device)
    baseline_identity_accuracy = identity_accuracy(baseline_identity_generations)
    print(
        f"baseline general_loss={baseline_general_loss:.4f} "
        f"instruction_loss={baseline_instruction_loss:.4f} "
        f"story_loss={baseline_story_loss:.4f} "
        f"identity_accuracy={baseline_identity_accuracy:.0%}",
        flush=True,
    )
    _append_jsonl(
        metrics_path,
        {
            "event": "baseline",
            "general_validation_loss": baseline_general_loss,
            "instruction_validation_loss": baseline_instruction_loss,
            "story_validation_loss": baseline_story_loss,
            "identity_loss": baseline_identity_loss,
            "identity_accuracy": baseline_identity_accuracy,
        },
    )

    microbatch, accumulation_steps = choose_batch_plan(
        model, train_corpus, device, runtime_config
    )
    print(
        f"General batch plan: microbatch={microbatch}, accumulation={accumulation_steps}, "
        f"effective_tokens={microbatch * accumulation_steps * model.config.block_size:,}",
        flush=True,
    )
    general_generator = torch.Generator(device="cpu").manual_seed(seed + 20)
    story_generator = torch.Generator(device="cpu").manual_seed(seed + 21)
    source_selector = random.Random(seed + 22)
    instruction_selector = random.Random(seed + 23)
    started = time.monotonic()
    completed_pretrain_steps = 0
    completed_instruction_steps = 0
    stopping_reason = "completed"
    optimizer: torch.optim.AdamW | None = None
    stage = "pretrain"

    def elapsed() -> float:
        return time.monotonic() - started

    def current_metadata(
        *,
        general_loss: float | None = None,
        instruction_loss: float | None = None,
        story_loss: float | None = None,
        identity_value: float | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "general_continued_pretraining_and_instruction_tuning",
            "accepted": False,
            "stage": stage,
            "base_checkpoint": str(source),
            "base_checkpoint_sha256": sha256_file(source),
            "general_manifest": str(manifest_path),
            "general_manifest_sha256": general_manifest_sha256,
            "requested_pretrain_steps": requested_pretrain_steps,
            "completed_pretrain_steps": completed_pretrain_steps,
            "requested_instruction_steps": requested_instruction_steps,
            "completed_instruction_steps": completed_instruction_steps,
            "context_expansion": {
                "from": base_app_config.model.block_size,
                "to": model.config.block_size,
            },
            "baseline_general_validation_loss": baseline_general_loss,
            "general_validation_loss": general_loss,
            "baseline_instruction_validation_loss": baseline_instruction_loss,
            "instruction_validation_loss": instruction_loss,
            "baseline_story_validation_loss": baseline_story_loss,
            "story_validation_loss": story_loss,
            "baseline_identity_accuracy": baseline_identity_accuracy,
            "identity_accuracy": identity_value,
            "elapsed_seconds": elapsed(),
            "stopping_reason": stopping_reason,
        }

    def save_latest(metadata: dict[str, Any]) -> None:
        assert optimizer is not None
        payload = dict(checkpoint)
        app_config_value = runtime_config.to_dict()
        payload.update(
            {
                "format_version": 3,
                "identity": model_identity_metadata(),
                "model_state": _cpu_tree(model.state_dict()),
                "model_config": asdict(model.config),
                "app_config": app_config_value,
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
                "general_data_manifest": general_manifest,
                "generalization": metadata,
                "rng_state": capture_rng_state(general_generator),
                "step": int(checkpoint.get("step", 0))
                + completed_pretrain_steps
                + completed_instruction_steps,
                "elapsed_training_seconds": float(
                    checkpoint.get("elapsed_training_seconds", 0.0)
                )
                + elapsed(),
                "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        atomic_torch_save(payload, latest_path)

    if requested_pretrain_steps:
        optimizer = _build_optimizer(
            model,
            runtime_config,
            learning_rate=general_config.training.pretrain_learning_rate,
            weight_decay=general_config.training.weight_decay,
        )
        for step in range(1, requested_pretrain_steps + 1):
            if elapsed() >= wall_limit:
                stopping_reason = "wall_time_limit"
                break
            learning_rate = _pretrain_learning_rate(
                step, requested_pretrain_steps, general_config
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            story_microbatches = 0
            for _ in range(accumulation_steps):
                use_story = (
                    source_selector.random()
                    < general_config.training.story_pretrain_probability
                )
                corpus = story_train if use_story else train_corpus
                generator = story_generator if use_story else general_generator
                inputs, targets = corpus.random_batch(microbatch, generator, device)
                _, loss = model(inputs, targets)
                assert loss is not None
                if not bool(torch.isfinite(loss).item()):
                    stopping_reason = "non_finite_pretrain_loss"
                    raise FloatingPointError(f"non-finite pretraining loss at step {step}")
                (loss / accumulation_steps).backward()
                accumulated_loss += float(loss.item()) / accumulation_steps
                story_microbatches += int(use_story)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), general_config.training.grad_clip
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                stopping_reason = "non_finite_gradient"
                raise FloatingPointError(f"non-finite gradient norm at pretrain step {step}")
            optimizer.step()
            completed_pretrain_steps = step
            if step % 25 == 0:
                if device.type == "mps":
                    torch.mps.synchronize()
                print(
                    f"general_pretrain={step}/{requested_pretrain_steps} "
                    f"loss={accumulated_loss:.4f} lr={learning_rate:.2e} "
                    f"grad_norm={gradient_norm.item():.3f} story_microbatches={story_microbatches} "
                    f"elapsed={elapsed():.1f}s",
                    flush=True,
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "pretrain",
                        "step": step,
                        "loss": accumulated_loss,
                        "learning_rate": learning_rate,
                        "gradient_norm": float(gradient_norm.item()),
                        "story_microbatches": story_microbatches,
                        "elapsed_seconds": elapsed(),
                    },
                )
            if step % general_config.training.eval_interval == 0 or step == requested_pretrain_steps:
                validation_loss = evaluate_model(
                    model,
                    general_validation,
                    device,
                    batches=general_config.training.eval_batches,
                    batch_size=evaluation_batch_size,
                    seed=seed + 10,
                )
                retained_story_loss = evaluate_model(
                    model,
                    story_validation,
                    device,
                    batches=general_config.training.eval_batches,
                    batch_size=evaluation_batch_size,
                    seed=seed + 11,
                )
                print(
                    f"general_eval pretrain_step={step} general_loss={validation_loss:.4f} "
                    f"ratio={validation_loss / baseline_general_loss:.3f} "
                    f"story_ratio={retained_story_loss / baseline_story_loss:.3f}",
                    flush=True,
                )
                metadata = current_metadata(
                    general_loss=validation_loss, story_loss=retained_story_loss
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "pretrain_evaluation",
                        "step": step,
                        "general_validation_loss": validation_loss,
                        "general_loss_ratio": validation_loss / baseline_general_loss,
                        "story_validation_loss": retained_story_loss,
                        "story_loss_ratio": retained_story_loss / baseline_story_loss,
                        "elapsed_seconds": elapsed(),
                    },
                )
                save_latest(metadata)

    if stopping_reason == "completed" and requested_instruction_steps:
        stage = "instruction"
        optimizer = _build_optimizer(
            model,
            runtime_config,
            learning_rate=general_config.training.instruction_learning_rate,
            weight_decay=general_config.training.weight_decay,
        )
        identity_selector = random.Random(seed + 24)
        denominator = (
            1.0
            + general_config.training.instruction_general_weight
            + general_config.training.instruction_story_weight
            + general_config.training.instruction_identity_weight
        )
        for step in range(1, requested_instruction_steps + 1):
            if elapsed() >= wall_limit:
                stopping_reason = "wall_time_limit"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            selected = [
                instruction_train_rows[
                    instruction_selector.randrange(len(instruction_train_rows))
                ]
                for _ in range(general_config.training.instruction_batch_size)
            ]
            instruction_inputs, instruction_targets = instruction_batch(
                tokenizer,
                selected,
                block_size=model.config.block_size,
                device=device,
            )
            _, instruction_loss = model(instruction_inputs, instruction_targets)
            assert instruction_loss is not None
            (instruction_loss / denominator).backward()

            general_inputs, general_targets = general_train_eval.random_batch(
                general_config.training.retention_batch_size,
                general_generator,
                device,
            )
            _, general_loss = model(general_inputs, general_targets)
            assert general_loss is not None
            (
                general_loss
                * general_config.training.instruction_general_weight
                / denominator
            ).backward()

            story_inputs, story_targets = story_train.random_batch(
                general_config.training.retention_batch_size,
                story_generator,
                device,
            )
            _, story_loss = model(story_inputs, story_targets)
            assert story_loss is not None
            (
                story_loss
                * general_config.training.instruction_story_weight
                / denominator
            ).backward()

            identity_examples = [
                IDENTITY_TRAIN_EXAMPLES[
                    identity_selector.randrange(len(IDENTITY_TRAIN_EXAMPLES))
                ]
                for _ in range(2)
            ]
            identity_inputs, identity_targets = identity_batch(
                tokenizer,
                identity_examples,
                block_size=model.config.block_size,
                device=device,
            )
            _, identity_loss = model(identity_inputs, identity_targets)
            assert identity_loss is not None
            (
                identity_loss
                * general_config.training.instruction_identity_weight
                / denominator
            ).backward()

            losses = (instruction_loss, general_loss, story_loss, identity_loss)
            if not all(bool(torch.isfinite(loss).item()) for loss in losses):
                stopping_reason = "non_finite_instruction_loss"
                raise FloatingPointError(f"non-finite instruction-stage loss at step {step}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), general_config.training.grad_clip
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                stopping_reason = "non_finite_gradient"
                raise FloatingPointError(f"non-finite gradient norm at instruction step {step}")
            optimizer.step()
            completed_instruction_steps = step

            if step % 25 == 0:
                if device.type == "mps":
                    torch.mps.synchronize()
                print(
                    f"instruction_tune={step}/{requested_instruction_steps} "
                    f"instruction_loss={instruction_loss.item():.4f} "
                    f"general_loss={general_loss.item():.4f} story_loss={story_loss.item():.4f} "
                    f"identity_loss={identity_loss.item():.4f} "
                    f"grad_norm={gradient_norm.item():.3f} elapsed={elapsed():.1f}s",
                    flush=True,
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "instruction",
                        "step": step,
                        "instruction_loss": float(instruction_loss.item()),
                        "general_loss": float(general_loss.item()),
                        "story_loss": float(story_loss.item()),
                        "identity_loss": float(identity_loss.item()),
                        "gradient_norm": float(gradient_norm.item()),
                        "elapsed_seconds": elapsed(),
                    },
                )
            if step % general_config.training.eval_interval == 0 or step == requested_instruction_steps:
                validation_general_loss = evaluate_model(
                    model,
                    general_validation,
                    device,
                    batches=general_config.training.eval_batches,
                    batch_size=evaluation_batch_size,
                    seed=seed + 10,
                )
                validation_instruction_loss = evaluate_instruction_loss(
                    model,
                    tokenizer,
                    instruction_validation_rows,
                    device,
                    batch_size=evaluation_batch_size,
                    limit=general_config.training.instruction_eval_examples,
                )
                retained_story_loss = evaluate_model(
                    model,
                    story_validation,
                    device,
                    batches=general_config.training.eval_batches,
                    batch_size=evaluation_batch_size,
                    seed=seed + 11,
                )
                identity_generations = evaluate_identity_generation(model, tokenizer, device)
                retained_identity_accuracy = identity_accuracy(identity_generations)
                print(
                    f"general_eval instruction_step={step} "
                    f"general_ratio={validation_general_loss / baseline_general_loss:.3f} "
                    f"instruction_ratio={validation_instruction_loss / baseline_instruction_loss:.3f} "
                    f"story_ratio={retained_story_loss / baseline_story_loss:.3f} "
                    f"identity_accuracy={retained_identity_accuracy:.0%}",
                    flush=True,
                )
                metadata = current_metadata(
                    general_loss=validation_general_loss,
                    instruction_loss=validation_instruction_loss,
                    story_loss=retained_story_loss,
                    identity_value=retained_identity_accuracy,
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "instruction_evaluation",
                        "step": step,
                        "general_validation_loss": validation_general_loss,
                        "general_loss_ratio": validation_general_loss
                        / baseline_general_loss,
                        "instruction_validation_loss": validation_instruction_loss,
                        "instruction_loss_ratio": validation_instruction_loss
                        / baseline_instruction_loss,
                        "story_validation_loss": retained_story_loss,
                        "story_loss_ratio": retained_story_loss / baseline_story_loss,
                        "identity_accuracy": retained_identity_accuracy,
                        "elapsed_seconds": elapsed(),
                    },
                )
                save_latest(metadata)

    if optimizer is None:
        raise AssertionError("generalization optimizer was not initialized")
    stage = "final_evaluation"
    print("Running held-out final capability evaluation", flush=True)
    final_general_validation_loss = evaluate_model(
        model,
        general_validation,
        device,
        batches=general_config.training.eval_batches,
        batch_size=evaluation_batch_size,
        seed=seed + 10,
    )
    final_general_test_loss = evaluate_model(
        model,
        general_test,
        device,
        batches=general_config.training.eval_batches,
        batch_size=evaluation_batch_size,
        seed=seed + 12,
    )
    final_general_test_loss_512 = evaluate_model(
        model,
        general_test_long,
        device,
        batches=general_config.training.eval_batches,
        batch_size=evaluation_batch_size,
        seed=seed + 13,
    )
    final_instruction_validation_loss = evaluate_instruction_loss(
        model,
        tokenizer,
        instruction_validation_rows,
        device,
        batch_size=evaluation_batch_size,
        limit=general_config.training.instruction_eval_examples,
    )
    final_instruction_test_loss = evaluate_instruction_loss(
        model,
        tokenizer,
        instruction_test_rows,
        device,
        batch_size=evaluation_batch_size,
        limit=general_config.training.instruction_eval_examples,
    )
    final_story_loss = evaluate_model(
        model,
        story_validation,
        device,
        batches=general_config.training.eval_batches,
        batch_size=evaluation_batch_size,
        seed=seed + 11,
    )
    final_identity_loss = evaluate_identity_loss(model, tokenizer, device)
    final_identity_generations = evaluate_identity_generation(model, tokenizer, device)
    final_identity_accuracy = identity_accuracy(final_identity_generations)
    general_generations = generate_general_responses(
        model, tokenizer, device, seed=seed + 1000
    )

    general_loss_ratio = final_general_validation_loss / baseline_general_loss
    instruction_loss_ratio = final_instruction_validation_loss / baseline_instruction_loss
    story_loss_ratio = final_story_loss / baseline_story_loss
    completed_all_steps = (
        completed_pretrain_steps == requested_pretrain_steps
        and completed_instruction_steps == requested_instruction_steps
    )
    accepted = (
        completed_all_steps
        and general_loss_ratio <= general_config.training.required_general_loss_ratio
        and instruction_loss_ratio
        <= general_config.training.required_instruction_loss_ratio
        and story_loss_ratio <= general_config.training.max_story_loss_ratio
        and final_identity_accuracy >= general_config.training.required_identity_accuracy
    )
    final_metadata = current_metadata(
        general_loss=final_general_validation_loss,
        instruction_loss=final_instruction_validation_loss,
        story_loss=final_story_loss,
        identity_value=final_identity_accuracy,
    )
    final_metadata.update(
        {
            "accepted": accepted,
            "completed_all_steps": completed_all_steps,
            "general_loss_ratio": general_loss_ratio,
            "required_general_loss_ratio": general_config.training.required_general_loss_ratio,
            "instruction_loss_ratio": instruction_loss_ratio,
            "required_instruction_loss_ratio": general_config.training.required_instruction_loss_ratio,
            "story_loss_ratio": story_loss_ratio,
            "max_story_loss_ratio": general_config.training.max_story_loss_ratio,
            "identity_loss": final_identity_loss,
            "required_identity_accuracy": general_config.training.required_identity_accuracy,
            "general_test_loss": final_general_test_loss,
            "general_test_loss_512": final_general_test_loss_512,
            "instruction_test_loss": final_instruction_test_loss,
            "parameter_count": model.parameter_count(),
        }
    )
    payload = dict(checkpoint)
    payload.update(
        {
            "format_version": 3,
            "identity": model_identity_metadata(),
            "model_state": _cpu_tree(model.state_dict()),
            "model_config": asdict(model.config),
            "app_config": runtime_config.to_dict(),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "general_data_manifest": general_manifest,
            "generalization": final_metadata,
            "rng_state": capture_rng_state(general_generator),
            "step": int(checkpoint.get("step", 0))
            + completed_pretrain_steps
            + completed_instruction_steps,
            "elapsed_training_seconds": float(
                checkpoint.get("elapsed_training_seconds", 0.0)
            )
            + elapsed(),
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    atomic_torch_save(payload, destination)
    report = {
        **final_metadata,
        "checkpoint": str(destination),
        "checkpoint_sha256": sha256_file(destination),
        "identity": model_identity_metadata(),
        "general_validation_perplexity": _perplexity(final_general_validation_loss),
        "general_test_perplexity": _perplexity(final_general_test_loss),
        "general_test_perplexity_512": _perplexity(final_general_test_loss_512),
        "baseline_identity_generations": [
            asdict(row) for row in baseline_identity_generations
        ],
        "final_identity_generations": [asdict(row) for row in final_identity_generations],
        "general_generations": [asdict(row) for row in general_generations],
        "metrics_path": str(metrics_path),
    }
    _atomic_json(report_path, report)
    _atomic_json(samples_path, [asdict(row) for row in general_generations])
    if accepted and promote_to is not None:
        if preserve_base_as is not None:
            _copy_atomic(source, Path(preserve_base_as))
        if destination.resolve() != Path(promote_to).resolve():
            _copy_atomic(destination, Path(promote_to))
        report["promoted_to"] = str(promote_to)
        report["preserved_base_as"] = str(preserve_base_as) if preserve_base_as else None
        _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return report
