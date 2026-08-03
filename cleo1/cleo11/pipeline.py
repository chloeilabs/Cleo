"""Synthetic end-to-end Cleo 1.1 pipeline for CPU/cloud wiring checks."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from .config import Cleo11Config, load_cleo11_config
from .evaluation import run_capability_evaluation
from .identity_tuning import identity_tune_cleo11
from .instruction_tuning import instruction_tune_cleo11
from .prepare import prepare_cleo11_data
from .pretrain import pretrain_cleo11
from .train import write_training_spec


def _rewrite_paths(config: Cleo11Config, root: Path) -> Cleo11Config:
    processed = root / "processed"
    artifacts = root / "artifacts"
    return replace(
        config,
        training=replace(config.training, artifacts_dir=str(artifacts)),
        prep=replace(
            config.prep,
            output_dir=str(processed),
            tokenizer_path=str(processed / "tokenizer.json"),
            train_glob=str(processed / "train-*.bin"),
            validation_tokens_path=str(processed / "validation.bin"),
            manifest_path=str(processed / "manifest.json"),
        ),
    )


def run_synthetic_pipeline(
    config: Cleo11Config,
    *,
    output_dir: str | Path,
    requested_device: str = "cpu",
    pretrain_steps: int = 3,
    instruction_steps: int = 3,
    identity_steps: int = 3,
    microbatch_size: int = 2,
    examples_per_category: int = 4,
    max_new_tokens: int = 24,
) -> dict[str, Any]:
    """Prepare → pretrain → instruct → identity → evaluate on synthetic data."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = _rewrite_paths(config, root)
    write_training_spec(config, config.training.artifacts_dir)

    prepare_manifest = prepare_cleo11_data(
        config,
        profile_name=config.prep.default_profile,
        synthetic=True,
        force=True,
    )
    pretrain = pretrain_cleo11(
        config,
        requested_device=requested_device,
        max_steps_override=pretrain_steps,
        microbatch_override=microbatch_size,
    )
    instruction = instruction_tune_cleo11(
        config,
        pretrain.checkpoint,
        requested_device=requested_device,
        steps=instruction_steps,
        batch_size=microbatch_size,
        eval_interval=max(1, instruction_steps),
        learning_rate=min(config.training.learning_rate, 5e-4),
    )
    identity = identity_tune_cleo11(
        config,
        instruction["checkpoint"],
        requested_device=requested_device,
        steps=identity_steps,
        batch_size=microbatch_size,
        eval_interval=max(1, identity_steps),
        learning_rate=1e-4,
    )
    evaluation = run_capability_evaluation(
        config,
        identity["checkpoint"],
        requested_device=requested_device,
        examples_per_category=examples_per_category,
        max_new_tokens=max_new_tokens,
    )
    report = {
        "pipeline": "cleo11-synthetic",
        "accepted_pipeline_wiring": True,
        "accepted_for_release": bool(evaluation["accepted"]),
        "device": requested_device,
        "stages": {
            "prepare": {
                "synthetic": prepare_manifest["synthetic"],
                "train_tokens": prepare_manifest["corpora"]["train"]["tokens"],
                "validation_tokens": prepare_manifest["corpora"]["validation"]["tokens"],
                "tokenizer": prepare_manifest["tokenizer"]["path"],
            },
            "pretrain": pretrain.to_dict(),
            "instruction_tuning": instruction,
            "identity_tuning": identity,
            "evaluation": {
                "accepted": evaluation["accepted"],
                "scores": evaluation["scores"],
                "gate_report": evaluation["gate_report"],
                "checkpoint": evaluation["checkpoint"],
            },
        },
        "final_checkpoint": identity["checkpoint"],
        "note": (
            "Synthetic pipeline validates stage wiring and artifact continuity. "
            "It does not produce a releasable Cleo 1.1 checkpoint."
        ),
    }
    path = Path(config.training.artifacts_dir) / "pipeline_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(path)
    return report


def pipeline_from_config_path(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_synthetic_pipeline(
        load_cleo11_config(config_path),
        output_dir=output_dir,
        **kwargs,
    )
