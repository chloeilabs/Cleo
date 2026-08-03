from __future__ import annotations

import json
from pathlib import Path

from cleo1.cleo11.config import load_cleo11_config
from cleo1.cleo11.curriculum import answers_match, build_instruction_curriculum
from cleo1.cleo11.evaluation import run_capability_evaluation
from cleo1.cleo11.identity import CANONICAL_IDENTITY_RESPONSE, MODEL_ID, MODEL_NAME
from cleo1.cleo11.identity_tuning import identity_tune_cleo11
from cleo1.cleo11.instruction_tuning import instruction_tune_cleo11
from cleo1.cleo11.pipeline import run_synthetic_pipeline
from cleo1.cleo11.prepare import prepare_cleo11_data
from cleo1.cleo11.pretrain import pretrain_cleo11


def _smoke_config(tmp_path: Path):
    config = load_cleo11_config("configs/cleo11_smoke.toml")
    prep = config.prep
    output = tmp_path / "processed"
    return type(config)(
        model=config.model,
        training=type(config.training)(
            **{
                **config.training.__dict__,
                "artifacts_dir": str(tmp_path / "artifacts"),
                "eval_interval": 2,
                "eval_batches": 2,
                "warmup_steps": 1,
                "effective_batch_tokens": 256,
                "initial_microbatch_size": 2,
            }
        ),
        data=config.data,
        evaluation=config.evaluation,
        prep=type(prep)(
            **{
                **prep.__dict__,
                "output_dir": str(output),
                "tokenizer_path": str(output / "tokenizer.json"),
                "train_glob": str(output / "train-*.bin"),
                "validation_tokens_path": str(output / "validation.bin"),
                "manifest_path": str(output / "manifest.json"),
                "shard_tokens": 4096,
                "default_profile": "dev",
            }
        ),
    )


def test_cleo11_identity_constants():
    assert MODEL_NAME == "Cleo 1.1"
    assert MODEL_ID == "cleo-1.1"
    assert "cleo-1.1" in CANONICAL_IDENTITY_RESPONSE
    assert "Cleo AI" in CANONICAL_IDENTITY_RESPONSE


def test_curriculum_covers_gate_categories():
    train, evaluation = build_instruction_curriculum()
    categories = {row.category for row in (*train, *evaluation)}
    assert categories >= {
        "reasoning",
        "arithmetic",
        "extraction",
        "knowledge",
        "code",
        "safety",
        "instruction_following",
    }
    assert answers_match("3", "3", category="arithmetic")
    assert answers_match("I can't help with that.", "Sorry, I can't help with that.", category="safety")


def test_instruction_and_identity_tune_after_pretrain(tmp_path):
    config = _smoke_config(tmp_path)
    prepare_cleo11_data(config, profile_name="dev", synthetic=True, force=True)
    pretrain = pretrain_cleo11(
        config,
        requested_device="cpu",
        max_steps_override=2,
        microbatch_override=2,
    )
    instruction = instruction_tune_cleo11(
        config,
        pretrain.checkpoint,
        requested_device="cpu",
        steps=2,
        batch_size=2,
        eval_interval=1,
    )
    assert Path(instruction["checkpoint"]).is_file()
    assert (tmp_path / "artifacts" / "instruction_report.json").is_file()

    identity = identity_tune_cleo11(
        config,
        instruction["checkpoint"],
        requested_device="cpu",
        steps=2,
        batch_size=2,
        eval_interval=1,
    )
    assert Path(identity["checkpoint"]).is_file()
    assert identity["identity"]["model_id"] == "cleo-1.1"

    evaluation = run_capability_evaluation(
        config,
        identity["checkpoint"],
        requested_device="cpu",
        examples_per_category=2,
        max_new_tokens=16,
    )
    # Smoke gates use minimum=0.0, so wiring acceptance should pass.
    assert evaluation["accepted"] is True
    assert set(evaluation["scores"]) >= {
        "reasoning",
        "arithmetic",
        "extraction",
        "knowledge",
        "code",
        "safety",
        "instruction_following",
        "identity",
    }


def test_synthetic_pipeline_end_to_end(tmp_path):
    config = load_cleo11_config("configs/cleo11_smoke.toml")
    report = run_synthetic_pipeline(
        config,
        output_dir=tmp_path / "pipeline",
        requested_device="cpu",
        pretrain_steps=2,
        instruction_steps=2,
        identity_steps=2,
        microbatch_size=2,
        examples_per_category=2,
        max_new_tokens=12,
    )
    assert report["accepted_pipeline_wiring"] is True
    assert Path(report["final_checkpoint"]).is_file()
    pipeline_report = Path(report["report_path"])
    assert pipeline_report.is_file()
    payload = json.loads(pipeline_report.read_text(encoding="utf-8"))
    assert payload["stages"]["prepare"]["synthetic"] is True
    assert payload["stages"]["pretrain"]["steps"] == 2
