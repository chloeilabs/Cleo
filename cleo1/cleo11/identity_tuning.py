"""Bounded identity adaptation for Cleo 1.1 after instruction tuning."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch

from ..engine import seed_everything
from ..general_data import render_instruction_prompt
from .checkpoint_utils import (
    load_cleo11_for_finetune,
    make_finetune_optimizer,
    save_cleo11_checkpoint,
)
from .config import Cleo11Config, load_cleo11_config
from .identity import (
    CANONICAL_IDENTITY_RESPONSE,
    IDENTITY_EVAL_EXAMPLES,
    IDENTITY_TRAIN_EXAMPLES,
    IdentityExample,
    identity_response_matches,
    model_identity_metadata,
)


IGNORE_INDEX = -100


def encode_identity_example(
    tokenizer,
    example: IdentityExample,
    *,
    block_size: int,
) -> tuple[list[int], list[int]]:
    prompt = render_instruction_prompt(example.prompt)
    prompt_ids = tokenizer.encode(prompt, bos=True)
    answer_ids = tokenizer.encode(example.answer, eos=True)
    sequence = prompt_ids + answer_ids
    if len(sequence) > block_size + 1:
        # Keep the answer and the end of the prompt for tiny smoke contexts.
        keep = block_size + 1
        sequence = sequence[-keep:]
        prompt_len = max(len(sequence) - len(answer_ids), 0)
    else:
        prompt_len = len(prompt_ids)
    inputs = sequence[:-1]
    targets = list(sequence[1:])
    targets[: max(prompt_len - 1, 0)] = [IGNORE_INDEX] * max(prompt_len - 1, 0)
    return inputs, targets


def identity_batch(
    tokenizer,
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
    width = max(len(inputs) for inputs, _ in rows)
    inputs = torch.full((len(rows), width), tokenizer.eos_id, dtype=torch.long)
    targets = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for index, (row_inputs, row_targets) in enumerate(rows):
        inputs[index, : len(row_inputs)] = torch.tensor(row_inputs, dtype=torch.long)
        targets[index, : len(row_targets)] = torch.tensor(row_targets, dtype=torch.long)
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def evaluate_identity_loss(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    examples: Iterable[IdentityExample] = IDENTITY_EVAL_EXAMPLES,
) -> float:
    rows = list(examples)
    inputs, targets = identity_batch(
        tokenizer,
        rows,
        block_size=model.config.block_size,
        device=device,
    )
    model.eval()
    _, loss = model(inputs, targets)
    assert loss is not None
    return float(loss.item())


@torch.no_grad()
def evaluate_identity_generation(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    examples: Iterable[IdentityExample] = IDENTITY_EVAL_EXAMPLES,
    *,
    max_new_tokens: int = 48,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []
    for example in examples:
        prompt = render_instruction_prompt(example.prompt)
        prompt_ids = tokenizer.encode(prompt, bos=True)
        if len(prompt_ids) >= model.config.block_size:
            prompt_ids = prompt_ids[-(model.config.block_size - 1) :]
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            tokens,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_k=1,
            use_cache=True,
        )
        continuation_ids = generated[0, len(prompt_ids) :].tolist()
        # Drop trailing EOS for comparison.
        if continuation_ids and continuation_ids[-1] == tokenizer.eos_id:
            continuation_ids = continuation_ids[:-1]
        continuation = tokenizer.decode(continuation_ids).strip()
        results.append(
            {
                "prompt": example.prompt,
                "expected": example.answer,
                "response": continuation,
                "passed": identity_response_matches(continuation),
            }
        )
    return results


def identity_accuracy(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for row in results if row["passed"]) / len(results)


def identity_tune_cleo11(
    config: Cleo11Config,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    requested_device: str = "auto",
    steps: int = 100,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    eval_interval: int = 25,
    seed: int = 1337,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("identity tuning steps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    seed_everything(seed)
    loaded = load_cleo11_for_finetune(
        config,
        checkpoint_path,
        requested_device=requested_device,
        tokenizer_path=tokenizer_path,
    )
    train_examples = list(IDENTITY_TRAIN_EXAMPLES)
    eval_examples = list(IDENTITY_EVAL_EXAMPLES)

    artifacts = Path(config.training.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = Path(output_path or artifacts / "identity.pt")
    metrics_path = artifacts / "identity_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    optimizer = make_finetune_optimizer(loaded.model, config, learning_rate=learning_rate)
    started = time.perf_counter()
    initial_loss = evaluate_identity_loss(
        loaded.model,
        loaded.tokenizer,
        loaded.device,
        eval_examples,
    )
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": "validation", "step": 0, "identity_loss": initial_loss},
                sort_keys=True,
            )
            + "\n"
        )

    loaded.model.train()
    final_loss = initial_loss
    for step in range(1, steps + 1):
        batch = random.sample(train_examples, k=min(batch_size, len(train_examples)))
        inputs, targets = identity_batch(
            loaded.tokenizer,
            batch,
            block_size=loaded.model_config.block_size,
            device=loaded.device,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = loaded.model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loaded.model.parameters(), config.training.grad_clip)
        optimizer.step()
        if step % eval_interval == 0 or step == steps:
            final_loss = evaluate_identity_loss(
                loaded.model,
                loaded.tokenizer,
                loaded.device,
                eval_examples,
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "validation",
                            "step": step,
                            "train_loss": float(loss.item()),
                            "identity_loss": final_loss,
                            "learning_rate": learning_rate,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            loaded.model.train()

    generations = evaluate_identity_generation(
        loaded.model,
        loaded.tokenizer,
        loaded.device,
        eval_examples,
    )
    accuracy = identity_accuracy(generations)
    source_meta = dict(loaded.source_checkpoint)
    source_meta["path"] = str(loaded.source_path)
    save_cleo11_checkpoint(
        destination,
        model=loaded.model,
        model_config=loaded.model_config,
        stage="identity_tuning",
        step=steps,
        tokenizer_checksum=loaded.tokenizer_checksum,
        source_checkpoint=source_meta,
        optimizer=optimizer,
        extra={
            "identity": model_identity_metadata(),
            "identity_tuning": {
                "steps": steps,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "canonical_response": CANONICAL_IDENTITY_RESPONSE,
                "initial_identity_loss": initial_loss,
                "final_identity_loss": final_loss,
                "held_out_exact_match": accuracy,
                "held_out_count": len(generations),
            },
            "identity_generations": generations,
        },
    )
    report = {
        "accepted_identity_run": True,
        "checkpoint": str(destination),
        "steps": steps,
        "parameter_count": loaded.model.parameter_count(),
        "initial_identity_loss": initial_loss,
        "final_identity_loss": final_loss,
        "held_out_exact_match": accuracy,
        "held_out_passed": sum(1 for row in generations if row["passed"]),
        "held_out_count": len(generations),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(loaded.device),
        "identity": model_identity_metadata(),
    }
    (artifacts / "identity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def identity_tune_from_config_path(
    config_path: str | Path,
    checkpoint_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return identity_tune_cleo11(
        load_cleo11_config(config_path),
        checkpoint_path,
        **kwargs,
    )
