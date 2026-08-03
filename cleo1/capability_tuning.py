from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import torch

from .capabilities import (
    build_capability_examples,
    capability_accuracy,
    evaluate_capability_generation,
)
from .checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint
from .config import AppConfig, ModelConfig
from .data import TokenCorpus, sha256_file
from .engine import configure_device, evaluate_model, seed_everything, select_device
from .general_data import GeneralConfig, instruction_batch, load_instruction_examples
from .general_training import (
    _append_jsonl,
    _atomic_json,
    _build_optimizer,
    _copy_atomic,
    _cpu_tree,
    evaluate_instruction_loss,
    generate_general_responses,
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


def _examples_checksum() -> str:
    train, evaluation = build_capability_examples()
    payload = [asdict(example) for example in (*train, *evaluation)]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fine_tune_capabilities(
    checkpoint_path: str | Path,
    output_path: str | Path,
    tokenizer_path: str | Path,
    general_config: GeneralConfig,
    *,
    requested_device: str = "auto",
    steps: int = 1200,
    learning_rate: float = 2e-5,
    skill_batch_size: int = 12,
    instruction_batch_size: int = 4,
    retention_batch_size: int = 8,
    identity_batch_size: int = 4,
    instruction_weight: float = 0.5,
    general_weight: float = 0.25,
    story_weight: float = 0.1,
    identity_weight: float = 0.5,
    eval_interval: int = 200,
    evaluation_examples: int = 128,
    required_capability_accuracy: float = 0.6,
    max_general_loss_ratio: float = 1.08,
    max_instruction_loss_ratio: float = 1.08,
    max_story_loss_ratio: float = 1.10,
    seed: int = 1337,
    promote_to: str | Path | None = None,
    preserve_source: str | Path | None = None,
    preserve_as: str | Path | None = None,
) -> dict[str, Any]:
    if steps < 1 or eval_interval < 1:
        raise ValueError("capability steps and evaluation interval must be positive")
    if learning_rate <= 0:
        raise ValueError("capability learning rate must be positive")
    if not 0 < required_capability_accuracy <= 1:
        raise ValueError("required capability accuracy must be in (0, 1]")
    source = Path(checkpoint_path)
    destination = Path(output_path)
    checkpoint = load_checkpoint(source, map_location="cpu")
    if "generalization" not in checkpoint:
        raise ValueError("capability tuning requires a generalization checkpoint")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    tokenizer_checksum = ByteBPETokenizer.checksum(tokenizer_path)
    if checkpoint["tokenizer_checksum"] != tokenizer_checksum:
        raise RuntimeError("capability checkpoint tokenizer checksum mismatch")

    app_config = AppConfig.from_dict(checkpoint["app_config"])
    model = CleoTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    device = select_device(requested_device)
    configure_device(device, app_config)
    seed_everything(seed)
    model.to(device)
    generalization = dict(checkpoint["generalization"])
    comparison_block_size = int(
        dict(generalization.get("context_expansion", {})).get("from", 256)
    )
    general_train = TokenCorpus(general_config.data.train_tokens, comparison_block_size)
    general_validation = TokenCorpus(
        general_config.data.validation_tokens, comparison_block_size
    )
    story_train = TokenCorpus(app_config.data.train_tokens, comparison_block_size)
    story_validation = TokenCorpus(
        app_config.data.validation_tokens, comparison_block_size
    )
    instruction_train_rows = load_instruction_examples(general_config.data.instruction_train)
    instruction_validation_rows = load_instruction_examples(
        general_config.data.instruction_validation
    )
    capability_train_rows, capability_evaluation_rows = build_capability_examples()

    artifacts = destination.parent
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts / "capability_metrics.jsonl"
    report_path = artifacts / "capability_report.json"
    latest_path = artifacts / "capability-latest.pt"
    if metrics_path.exists():
        metrics_path.unlink()
    evaluation_batch_size = min(retention_batch_size, 16)

    print(
        f"Capability tuning on {device.type}: train_skills={len(capability_train_rows):,} "
        f"held_out_skills={len(capability_evaluation_rows):,} context={model.config.block_size}",
        flush=True,
    )
    print("Measuring post-generalization capability baselines", flush=True)
    baseline_general_loss = evaluate_model(
        model,
        general_validation,
        device,
        batches=general_config.training.eval_batches,
        batch_size=evaluation_batch_size,
        seed=seed + 1,
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
        seed=seed + 2,
    )
    baseline_identity_loss = evaluate_identity_loss(model, tokenizer, device)
    baseline_identity_rows = evaluate_identity_generation(model, tokenizer, device)
    baseline_identity_accuracy = identity_accuracy(baseline_identity_rows)
    baseline_capability_loss = evaluate_instruction_loss(
        model,
        tokenizer,
        capability_evaluation_rows,
        device,
        batch_size=evaluation_batch_size,
        limit=evaluation_examples,
    )
    baseline_capability_rows = evaluate_capability_generation(
        model,
        tokenizer,
        capability_evaluation_rows,
        device,
        limit=evaluation_examples,
        seed=seed + 3,
    )
    baseline_capability_accuracy = capability_accuracy(baseline_capability_rows)
    print(
        f"baseline capability_accuracy={baseline_capability_accuracy:.1%} "
        f"capability_loss={baseline_capability_loss:.4f} "
        f"identity_accuracy={baseline_identity_accuracy:.0%} "
        f"general_loss={baseline_general_loss:.4f} "
        f"instruction_loss={baseline_instruction_loss:.4f}",
        flush=True,
    )
    _append_jsonl(
        metrics_path,
        {
            "event": "baseline",
            "capability_accuracy": baseline_capability_accuracy,
            "capability_loss": baseline_capability_loss,
            "identity_accuracy": baseline_identity_accuracy,
            "identity_loss": baseline_identity_loss,
            "general_loss": baseline_general_loss,
            "instruction_loss": baseline_instruction_loss,
            "story_loss": baseline_story_loss,
        },
    )

    optimizer = _build_optimizer(
        model,
        app_config,
        learning_rate=learning_rate,
        weight_decay=general_config.training.weight_decay,
    )
    capability_selector = random.Random(seed + 10)
    instruction_selector = random.Random(seed + 11)
    identity_selector = random.Random(seed + 12)
    general_generator = torch.Generator(device="cpu").manual_seed(seed + 13)
    story_generator = torch.Generator(device="cpu").manual_seed(seed + 14)
    denominator = 1 + instruction_weight + general_weight + story_weight + identity_weight
    started = time.monotonic()
    completed_steps = 0
    accepted = False
    final_general_loss = baseline_general_loss
    final_instruction_loss = baseline_instruction_loss
    final_story_loss = baseline_story_loss
    final_identity_loss = baseline_identity_loss
    final_identity_rows = baseline_identity_rows
    final_identity_accuracy = baseline_identity_accuracy
    final_capability_loss = baseline_capability_loss
    final_capability_rows = baseline_capability_rows
    final_capability_accuracy = baseline_capability_accuracy

    def checkpoint_payload(metadata: dict[str, Any]) -> dict[str, Any]:
        payload = dict(checkpoint)
        payload.update(
            {
                "format_version": 3,
                "identity": model_identity_metadata(),
                "model_state": _cpu_tree(model.state_dict()),
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
                "capability_tuning": metadata,
                "rng_state": capture_rng_state(general_generator),
                "step": int(checkpoint.get("step", 0)) + completed_steps,
                "elapsed_training_seconds": float(
                    checkpoint.get("elapsed_training_seconds", 0.0)
                )
                + (time.monotonic() - started),
                "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        return payload

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        selected_capabilities = [
            capability_train_rows[
                capability_selector.randrange(len(capability_train_rows))
            ]
            for _ in range(skill_batch_size)
        ]
        capability_inputs, capability_targets = instruction_batch(
            tokenizer,
            selected_capabilities,
            block_size=model.config.block_size,
            device=device,
        )
        _, capability_loss = model(capability_inputs, capability_targets)
        assert capability_loss is not None
        (capability_loss / denominator).backward()

        selected_instructions = [
            instruction_train_rows[
                instruction_selector.randrange(len(instruction_train_rows))
            ]
            for _ in range(instruction_batch_size)
        ]
        instruction_inputs, instruction_targets = instruction_batch(
            tokenizer,
            selected_instructions,
            block_size=model.config.block_size,
            device=device,
        )
        _, instruction_loss = model(instruction_inputs, instruction_targets)
        assert instruction_loss is not None
        (instruction_loss * instruction_weight / denominator).backward()

        general_inputs, general_targets = general_train.random_batch(
            retention_batch_size, general_generator, device
        )
        _, general_loss = model(general_inputs, general_targets)
        assert general_loss is not None
        (general_loss * general_weight / denominator).backward()

        story_inputs, story_targets = story_train.random_batch(
            retention_batch_size, story_generator, device
        )
        _, story_loss = model(story_inputs, story_targets)
        assert story_loss is not None
        (story_loss * story_weight / denominator).backward()

        selected_identity = [
            IDENTITY_TRAIN_EXAMPLES[
                identity_selector.randrange(len(IDENTITY_TRAIN_EXAMPLES))
            ]
            for _ in range(identity_batch_size)
        ]
        identity_inputs, identity_targets = identity_batch(
            tokenizer,
            selected_identity,
            block_size=model.config.block_size,
            device=device,
        )
        _, identity_loss = model(identity_inputs, identity_targets)
        assert identity_loss is not None
        (identity_loss * identity_weight / denominator).backward()

        losses = (capability_loss, instruction_loss, general_loss, story_loss, identity_loss)
        if not all(bool(torch.isfinite(loss).item()) for loss in losses):
            raise FloatingPointError(f"non-finite capability-stage loss at step {step}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), general_config.training.grad_clip
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError(f"non-finite capability gradient at step {step}")
        optimizer.step()
        completed_steps = step

        if step % 25 == 0:
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.monotonic() - started
            print(
                f"capability_step={step}/{steps} skill_loss={capability_loss.item():.4f} "
                f"instruction_loss={instruction_loss.item():.4f} "
                f"general_loss={general_loss.item():.4f} story_loss={story_loss.item():.4f} "
                f"identity_loss={identity_loss.item():.4f} "
                f"grad_norm={gradient_norm.item():.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            _append_jsonl(
                metrics_path,
                {
                    "event": "train",
                    "step": step,
                    "capability_loss": float(capability_loss.item()),
                    "instruction_loss": float(instruction_loss.item()),
                    "general_loss": float(general_loss.item()),
                    "story_loss": float(story_loss.item()),
                    "identity_loss": float(identity_loss.item()),
                    "gradient_norm": float(gradient_norm.item()),
                    "elapsed_seconds": elapsed,
                },
            )

        if step % eval_interval == 0 or step == steps:
            final_general_loss = evaluate_model(
                model,
                general_validation,
                device,
                batches=general_config.training.eval_batches,
                batch_size=evaluation_batch_size,
                seed=seed + 1,
            )
            final_instruction_loss = evaluate_instruction_loss(
                model,
                tokenizer,
                instruction_validation_rows,
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
                seed=seed + 2,
            )
            final_identity_loss = evaluate_identity_loss(model, tokenizer, device)
            final_identity_rows = evaluate_identity_generation(model, tokenizer, device)
            final_identity_accuracy = identity_accuracy(final_identity_rows)
            final_capability_loss = evaluate_instruction_loss(
                model,
                tokenizer,
                capability_evaluation_rows,
                device,
                batch_size=evaluation_batch_size,
                limit=evaluation_examples,
            )
            final_capability_rows = evaluate_capability_generation(
                model,
                tokenizer,
                capability_evaluation_rows,
                device,
                limit=evaluation_examples,
                seed=seed + 3,
            )
            final_capability_accuracy = capability_accuracy(final_capability_rows)
            general_ratio = final_general_loss / baseline_general_loss
            instruction_ratio = final_instruction_loss / baseline_instruction_loss
            story_ratio = final_story_loss / baseline_story_loss
            original_general_ratio = final_general_loss / float(
                generalization["baseline_general_validation_loss"]
            )
            original_instruction_ratio = final_instruction_loss / float(
                generalization["baseline_instruction_validation_loss"]
            )
            original_story_ratio = final_story_loss / float(
                generalization["baseline_story_validation_loss"]
            )
            accepted = (
                final_capability_accuracy >= required_capability_accuracy
                and final_identity_accuracy >= 1.0
                and general_ratio <= max_general_loss_ratio
                and instruction_ratio <= max_instruction_loss_ratio
                and story_ratio <= max_story_loss_ratio
                and original_general_ratio
                <= float(generalization["required_general_loss_ratio"])
                and original_instruction_ratio
                <= float(generalization["required_instruction_loss_ratio"])
                and original_story_ratio <= float(generalization["max_story_loss_ratio"])
            )
            metadata = {
                "type": "deterministic_general_capability_curriculum",
                "accepted": accepted,
                "completed_steps": completed_steps,
                "requested_steps": steps,
                "learning_rate": learning_rate,
                "examples_sha256": _examples_checksum(),
                "train_examples": len(capability_train_rows),
                "evaluation_examples": min(evaluation_examples, len(capability_evaluation_rows)),
                "baseline_capability_accuracy": baseline_capability_accuracy,
                "capability_accuracy": final_capability_accuracy,
                "required_capability_accuracy": required_capability_accuracy,
                "baseline_capability_loss": baseline_capability_loss,
                "capability_loss": final_capability_loss,
                "baseline_identity_accuracy": baseline_identity_accuracy,
                "identity_accuracy": final_identity_accuracy,
                "general_loss_ratio": general_ratio,
                "max_general_loss_ratio": max_general_loss_ratio,
                "instruction_loss_ratio": instruction_ratio,
                "max_instruction_loss_ratio": max_instruction_loss_ratio,
                "story_loss_ratio": story_ratio,
                "max_story_loss_ratio": max_story_loss_ratio,
                "original_general_loss_ratio": original_general_ratio,
                "original_instruction_loss_ratio": original_instruction_ratio,
                "original_story_loss_ratio": original_story_ratio,
                "elapsed_seconds": time.monotonic() - started,
            }
            print(
                f"capability_eval step={step} accuracy={final_capability_accuracy:.1%} "
                f"identity={final_identity_accuracy:.0%} general_ratio={general_ratio:.3f} "
                f"instruction_ratio={instruction_ratio:.3f} story_ratio={story_ratio:.3f} "
                f"accepted={accepted}",
                flush=True,
            )
            _append_jsonl(metrics_path, {"event": "evaluation", "step": step, **metadata})
            atomic_torch_save(checkpoint_payload(metadata), latest_path)
            if accepted and step >= 400:
                break

    final_generalization = dict(generalization)
    original_general_ratio = final_general_loss / float(
        generalization["baseline_general_validation_loss"]
    )
    original_instruction_ratio = final_instruction_loss / float(
        generalization["baseline_instruction_validation_loss"]
    )
    original_story_ratio = final_story_loss / float(
        generalization["baseline_story_validation_loss"]
    )
    final_generalization.update(
        {
            "accepted": accepted,
            "post_capability_general_validation_loss": final_general_loss,
            "post_capability_instruction_validation_loss": final_instruction_loss,
            "post_capability_story_validation_loss": final_story_loss,
            "post_capability_identity_accuracy": final_identity_accuracy,
            "general_loss_ratio": original_general_ratio,
            "instruction_loss_ratio": original_instruction_ratio,
            "story_loss_ratio": original_story_ratio,
            "identity_accuracy": final_identity_accuracy,
        }
    )
    capability_metadata = {
        "type": "deterministic_general_capability_curriculum",
        "accepted": accepted,
        "completed_steps": completed_steps,
        "requested_steps": steps,
        "learning_rate": learning_rate,
        "examples_sha256": _examples_checksum(),
        "train_examples": len(capability_train_rows),
        "evaluation_examples": min(evaluation_examples, len(capability_evaluation_rows)),
        "baseline_capability_accuracy": baseline_capability_accuracy,
        "capability_accuracy": final_capability_accuracy,
        "required_capability_accuracy": required_capability_accuracy,
        "baseline_capability_loss": baseline_capability_loss,
        "capability_loss": final_capability_loss,
        "baseline_identity_accuracy": baseline_identity_accuracy,
        "identity_accuracy": final_identity_accuracy,
        "identity_loss": final_identity_loss,
        "general_loss_ratio": final_general_loss / baseline_general_loss,
        "instruction_loss_ratio": final_instruction_loss / baseline_instruction_loss,
        "story_loss_ratio": final_story_loss / baseline_story_loss,
        "original_general_loss_ratio": original_general_ratio,
        "original_instruction_loss_ratio": original_instruction_ratio,
        "original_story_loss_ratio": original_story_ratio,
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = checkpoint_payload(capability_metadata)
    payload["generalization"] = final_generalization
    atomic_torch_save(payload, destination)
    general_generations = generate_general_responses(
        model, tokenizer, device, seed=seed + 1000
    )
    report = {
        **capability_metadata,
        "checkpoint": str(destination),
        "checkpoint_sha256": sha256_file(destination),
        "baseline_identity_generations": [asdict(row) for row in baseline_identity_rows],
        "final_identity_generations": [asdict(row) for row in final_identity_rows],
        "baseline_capability_generations": [
            asdict(row) for row in baseline_capability_rows
        ],
        "final_capability_generations": [asdict(row) for row in final_capability_rows],
        "general_generations": [asdict(row) for row in general_generations],
        "metrics_path": str(metrics_path),
    }
    _atomic_json(report_path, report)
    if accepted and promote_to is not None:
        if preserve_source is not None and preserve_as is not None:
            _copy_atomic(Path(preserve_source), Path(preserve_as))
        if destination.resolve() != Path(promote_to).resolve():
            _copy_atomic(destination, Path(promote_to))
        report["promoted_to"] = str(promote_to)
        report["preserved_story_checkpoint"] = str(preserve_as) if preserve_as else None
        _atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return report
