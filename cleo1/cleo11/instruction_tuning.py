"""Answer-only instruction tuning for Cleo 1.1 after pretraining."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any

import torch

from ..engine import seed_everything
from ..general_data import InstructionExample, encode_instruction_example, instruction_batch
from .checkpoint_utils import (
    load_cleo11_for_finetune,
    make_finetune_optimizer,
    save_cleo11_checkpoint,
)
from .config import Cleo11Config, load_cleo11_config
from .curriculum import CurriculumExample, build_instruction_curriculum, curriculum_checksum


def _to_instruction_examples(rows: list[CurriculumExample]) -> list[InstructionExample]:
    return [
        InstructionExample(
            instruction=row.instruction,
            context=row.context,
            response=row.response,
            category=row.category,
        )
        for row in rows
    ]


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate_instruction_loss(
    model: torch.nn.Module,
    tokenizer,
    examples: list[InstructionExample],
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    if not examples:
        raise ValueError("instruction evaluation requires examples")
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        inputs, targets = instruction_batch(
            tokenizer,
            batch,
            block_size=model.config.block_size,
            device=device,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        total += float(loss.item()) * len(batch)
        count += len(batch)
    return total / max(count, 1)


def instruction_tune_cleo11(
    config: Cleo11Config,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    requested_device: str = "auto",
    steps: int = 200,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    eval_interval: int = 50,
    seed: int = 1337,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("instruction tuning steps must be positive")
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
    train_rows, eval_rows = build_instruction_curriculum()
    train_examples = _to_instruction_examples(train_rows)
    eval_examples = _to_instruction_examples(eval_rows)
    # Filter examples that exceed the current context after encoding.
    usable_train: list[InstructionExample] = []
    for example in train_examples:
        try:
            encode_instruction_example(
                loaded.tokenizer,
                example,
                block_size=loaded.model_config.block_size,
            )
        except ValueError:
            continue
        usable_train.append(example)
    if not usable_train:
        raise RuntimeError("no instruction curriculum examples fit the model block size")

    usable_eval: list[InstructionExample] = []
    for example in eval_examples:
        try:
            encode_instruction_example(
                loaded.tokenizer,
                example,
                block_size=loaded.model_config.block_size,
            )
        except ValueError:
            continue
        usable_eval.append(example)
    if not usable_eval:
        usable_eval = usable_train[: min(8, len(usable_train))]

    artifacts = Path(config.training.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    destination = Path(output_path or artifacts / "instruction.pt")
    metrics_path = artifacts / "instruction_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    optimizer = make_finetune_optimizer(loaded.model, config, learning_rate=learning_rate)
    started = time.perf_counter()
    initial_loss = evaluate_instruction_loss(
        loaded.model,
        loaded.tokenizer,
        usable_eval,
        device=loaded.device,
        batch_size=batch_size,
    )
    _append_metric(
        metrics_path,
        {"event": "validation", "step": 0, "instruction_loss": initial_loss},
    )

    loaded.model.train()
    final_loss = initial_loss
    for step in range(1, steps + 1):
        batch = random.sample(usable_train, k=min(batch_size, len(usable_train)))
        inputs, targets = instruction_batch(
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
            final_loss = evaluate_instruction_loss(
                loaded.model,
                loaded.tokenizer,
                usable_eval,
                device=loaded.device,
                batch_size=batch_size,
            )
            _append_metric(
                metrics_path,
                {
                    "event": "validation",
                    "step": step,
                    "train_loss": float(loss.item()),
                    "instruction_loss": final_loss,
                    "learning_rate": learning_rate,
                },
            )
            loaded.model.train()

    source_meta = dict(loaded.source_checkpoint)
    source_meta["path"] = str(loaded.source_path)
    save_cleo11_checkpoint(
        destination,
        model=loaded.model,
        model_config=loaded.model_config,
        stage="instruction_tuning",
        step=steps,
        tokenizer_checksum=loaded.tokenizer_checksum,
        source_checkpoint=source_meta,
        optimizer=optimizer,
        extra={
            "instruction_tuning": {
                "steps": steps,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "train_examples": len(usable_train),
                "eval_examples": len(usable_eval),
                "curriculum_checksum": curriculum_checksum(train_rows + eval_rows),
                "initial_instruction_loss": initial_loss,
                "final_instruction_loss": final_loss,
            }
        },
    )
    report = {
        "accepted_instruction_run": True,
        "checkpoint": str(destination),
        "steps": steps,
        "parameter_count": loaded.model.parameter_count(),
        "initial_instruction_loss": initial_loss,
        "final_instruction_loss": final_loss,
        "train_examples": len(usable_train),
        "eval_examples": len(usable_eval),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(loaded.device),
        "note": (
            "Synthetic curriculum instruction tuning for pipeline wiring; "
            "not a release-quality instruction corpus."
        ),
    }
    (artifacts / "instruction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def instruction_tune_from_config_path(
    config_path: str | Path,
    checkpoint_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return instruction_tune_cleo11(
        load_cleo11_config(config_path),
        checkpoint_path,
        **kwargs,
    )
