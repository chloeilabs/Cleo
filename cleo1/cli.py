from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .config import load_config
from .data import prepare_data
from .engine import (
    configure_device,
    evaluate_checkpoint,
    generate_text,
    select_device,
    train_model,
    write_final_artifacts,
)


DEFAULT_CONFIG = "configs/tinystories_m4.toml"
DEFAULT_GENERAL_CONFIG = "configs/general_m4.toml"
DEFAULT_CLEO11_CONFIG = "configs/cleo11_135m.toml"
DEFAULT_CLEO11_SMOKE_CONFIG = "configs/cleo11_smoke.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cleo-1",
        description="Prepare, train, evaluate, and run the Cleo 1 language model from scratch.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="TOML configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download and tokenize TinyStories")
    prepare.add_argument("--force", action="store_true", help="Rebuild processed data")

    prepare_general = subparsers.add_parser(
        "prepare-general",
        help="Download and encode the pinned general-language and instruction corpora",
    )
    prepare_general.add_argument("--general-config", default=DEFAULT_GENERAL_CONFIG)
    prepare_general.add_argument("--force", action="store_true", help="Rebuild general data")

    train = subparsers.add_parser("train", help="Train the transformer")
    train.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    train.add_argument("--resume", metavar="CHECKPOINT")
    train.add_argument(
        "--skip-final-samples",
        action="store_true",
        help="Do not generate the five fixed CPU samples after training",
    )

    identity_tune = subparsers.add_parser(
        "identity-tune",
        help="Teach an existing checkpoint the canonical Cleo AI identity",
    )
    identity_tune.add_argument("--checkpoint", default="artifacts/best.pt")
    identity_tune.add_argument("--output", default="artifacts/cleo-1.pt")
    identity_tune.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    identity_tune.add_argument("--steps", type=int, default=800)
    identity_tune.add_argument("--learning-rate", type=float, default=2e-5)
    identity_tune.add_argument("--story-weight", type=float, default=4.0)
    identity_tune.add_argument("--identity-batch-size", type=int, default=8)
    identity_tune.add_argument("--story-batch-size", type=int, default=16)
    identity_tune.add_argument("--eval-interval", type=int, default=100)
    identity_tune.add_argument("--validation-batches", type=int, default=50)
    identity_tune.add_argument("--max-story-loss-increase", type=float, default=0.03)
    identity_tune.add_argument("--seed", type=int, default=1337)

    generalize = subparsers.add_parser(
        "generalize",
        help="Continue pretraining and instruction-tune Cleo 1 for broader use",
    )
    generalize.add_argument("--general-config", default=DEFAULT_GENERAL_CONFIG)
    generalize.add_argument("--checkpoint", default="artifacts/cleo-1.pt")
    generalize.add_argument("--output", default="artifacts/cleo-1-general.pt")
    generalize.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    generalize.add_argument("--pretrain-steps", type=int)
    generalize.add_argument("--instruction-steps", type=int)
    generalize.add_argument("--max-wall-time-seconds", type=int)
    generalize.add_argument(
        "--promote",
        action="store_true",
        help="Replace artifacts/cleo-1.pt only if every acceptance gate passes",
    )

    capability_tune = subparsers.add_parser(
        "capability-tune",
        help="Teach and gate deterministic general-purpose skills",
    )
    capability_tune.add_argument("--general-config", default=DEFAULT_GENERAL_CONFIG)
    capability_tune.add_argument("--checkpoint", default="artifacts/cleo-1-general.pt")
    capability_tune.add_argument("--output", default="artifacts/cleo-1-capable.pt")
    capability_tune.add_argument(
        "--device", choices=["auto", "mps", "cpu"], default="auto"
    )
    capability_tune.add_argument("--steps", type=int, default=1200)
    capability_tune.add_argument("--learning-rate", type=float, default=2e-5)
    capability_tune.add_argument("--required-accuracy", type=float, default=0.6)
    capability_tune.add_argument(
        "--promote",
        action="store_true",
        help="Replace artifacts/cleo-1.pt only if every release gate passes",
    )

    identity_repair = subparsers.add_parser(
        "general-identity-repair",
        help="Repair exact identity behavior while retaining general capabilities",
    )
    identity_repair.add_argument("--general-config", default=DEFAULT_GENERAL_CONFIG)
    identity_repair.add_argument("--checkpoint", default="artifacts/cleo-1-general.pt")
    identity_repair.add_argument(
        "--output", default="artifacts/cleo-1-general-repaired.pt"
    )
    identity_repair.add_argument(
        "--device", choices=["auto", "mps", "cpu"], default="auto"
    )
    identity_repair.add_argument("--steps", type=int, default=800)
    identity_repair.add_argument("--learning-rate", type=float, default=1e-5)
    identity_repair.add_argument(
        "--promote",
        action="store_true",
        help="Replace artifacts/cleo-1.pt only if every release gate passes",
    )

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate.add_argument("--checkpoint", default="artifacts/cleo-1.pt")
    evaluate.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    evaluate.add_argument("--batches", type=int)

    generate = subparsers.add_parser("generate", help="Generate a model response")
    generate.add_argument("--checkpoint", default="artifacts/cleo-1.pt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    generate.add_argument("--max-new-tokens", type=int, default=300)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--min-new-tokens", type=int, default=0)
    generate.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable the faster attention cache (useful for comparison)",
    )

    web = subparsers.add_parser("web", help="Launch the local Cleo 1 interface")
    web.add_argument("--checkpoint", default="artifacts/cleo-1.pt")
    web.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    web.add_argument("--host", default="127.0.0.1", help="Local bind address")
    web.add_argument("--port", type=int, default=7860)
    web.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")

    package_release = subparsers.add_parser(
        "package-release",
        help="Freeze Cleo 1 general-language alpha 01 packaging metadata and sidecar files",
    )
    package_release.add_argument("--checkpoint", default="artifacts/cleo-1.pt")
    package_release.add_argument("--tokenizer", default="data/processed/tokenizer.json")

    cleo11_spec = subparsers.add_parser(
        "cleo11-spec",
        help="Write the Cleo 1.1 architecture, dataset, compute, and evaluation contract",
    )
    cleo11_spec.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)
    cleo11_spec.add_argument("--output-dir", default="artifacts/cleo11")

    cleo11_estimate = subparsers.add_parser(
        "cleo11-estimate",
        help="Estimate Cleo 1.1 parameter count and training compute",
    )
    cleo11_estimate.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)

    cleo11_smoke = subparsers.add_parser(
        "cleo11-smoke",
        help="Run a tiny end-to-end Cleo 1.1 training smoke test",
    )
    cleo11_smoke.add_argument("--cleo11-config", default=DEFAULT_CLEO11_SMOKE_CONFIG)
    cleo11_smoke.add_argument("--device", choices=["auto", "mps", "cpu"], default="cpu")
    cleo11_smoke.add_argument("--output-dir", default="artifacts/cleo11/smoke")

    cleo11_prepare = subparsers.add_parser(
        "cleo11-prepare",
        help="Stream-encode the FineWeb-Edu-led Cleo 1.1 pretrain mixture",
    )
    cleo11_prepare.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)
    cleo11_prepare.add_argument("--profile", choices=["dev", "full"])
    cleo11_prepare.add_argument("--force", action="store_true")
    cleo11_prepare.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic documents (no Hugging Face download)",
    )
    cleo11_prepare.add_argument("--vocab-size", type=int)
    cleo11_prepare.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help="Reuse an existing tokenizer.json and only rebuild encoded shards",
    )

    cleo11_train = subparsers.add_parser(
        "cleo11-train",
        help="Pretrain Cleo 1.1 from prepared shards",
    )
    cleo11_train.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)
    cleo11_train.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    cleo11_train.add_argument("--resume")
    cleo11_train.add_argument("--max-steps", type=int)
    cleo11_train.add_argument("--microbatch-size", type=int)

    cleo11_launch = subparsers.add_parser(
        "cleo11-launch",
        help="Emit Docker/torchrun commands for cloud Cleo 1.1 pretrain (dry-run by default)",
    )
    cleo11_launch.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)
    cleo11_launch.add_argument("--profile", choices=["dev", "full"], default="full")
    cleo11_launch.add_argument("--nproc", type=int, default=1)
    cleo11_launch.add_argument("--image", default="cleo11-pretrain:latest")
    cleo11_launch.add_argument("--max-steps", type=int)
    cleo11_launch.add_argument(
        "--no-prepare",
        action="store_true",
        help="Assume shards already exist inside the data mount",
    )
    cleo11_launch.add_argument(
        "--emit-script",
        metavar="PATH",
        help="Write a runnable bash launcher script",
    )

    cleo11_instruction = subparsers.add_parser(
        "cleo11-instruction-tune",
        help="Answer-only instruction-tune a Cleo 1.1 pretrain checkpoint",
    )
    cleo11_instruction.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)
    cleo11_instruction.add_argument("--checkpoint", required=True)
    cleo11_instruction.add_argument("--output")
    cleo11_instruction.add_argument("--tokenizer")
    cleo11_instruction.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    cleo11_instruction.add_argument("--steps", type=int, default=200)
    cleo11_instruction.add_argument("--learning-rate", type=float, default=5e-5)
    cleo11_instruction.add_argument("--batch-size", type=int, default=4)
    cleo11_instruction.add_argument("--eval-interval", type=int, default=50)
    cleo11_instruction.add_argument("--seed", type=int, default=1337)

    cleo11_identity = subparsers.add_parser(
        "cleo11-identity-tune",
        help="Teach a Cleo 1.1 checkpoint the canonical Cleo AI / Cleo 1.1 identity",
    )
    cleo11_identity.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)
    cleo11_identity.add_argument("--checkpoint", required=True)
    cleo11_identity.add_argument("--output")
    cleo11_identity.add_argument("--tokenizer")
    cleo11_identity.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    cleo11_identity.add_argument("--steps", type=int, default=100)
    cleo11_identity.add_argument("--learning-rate", type=float, default=1e-5)
    cleo11_identity.add_argument("--batch-size", type=int, default=4)
    cleo11_identity.add_argument("--eval-interval", type=int, default=25)
    cleo11_identity.add_argument("--seed", type=int, default=1337)

    cleo11_evaluate = subparsers.add_parser(
        "cleo11-evaluate",
        help="Run Cleo 1.1 capability gates against a checkpoint",
    )
    cleo11_evaluate.add_argument("--cleo11-config", default=DEFAULT_CLEO11_CONFIG)
    cleo11_evaluate.add_argument("--checkpoint", required=True)
    cleo11_evaluate.add_argument("--tokenizer")
    cleo11_evaluate.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    cleo11_evaluate.add_argument("--max-new-tokens", type=int, default=48)
    cleo11_evaluate.add_argument("--examples-per-category", type=int, default=16)
    cleo11_evaluate.add_argument("--output")
    cleo11_evaluate.add_argument("--seed", type=int, default=1337)

    cleo11_pipeline = subparsers.add_parser(
        "cleo11-pipeline",
        help="Run the synthetic prepare→pretrain→instruct→identity→evaluate wiring pipeline",
    )
    cleo11_pipeline.add_argument("--cleo11-config", default=DEFAULT_CLEO11_SMOKE_CONFIG)
    cleo11_pipeline.add_argument("--output-dir", default="artifacts/cleo11/pipeline")
    cleo11_pipeline.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu"
    )
    cleo11_pipeline.add_argument("--pretrain-steps", type=int, default=3)
    cleo11_pipeline.add_argument("--instruction-steps", type=int, default=3)
    cleo11_pipeline.add_argument("--identity-steps", type=int, default=3)
    cleo11_pipeline.add_argument("--microbatch-size", type=int, default=2)
    cleo11_pipeline.add_argument("--examples-per-category", type=int, default=4)
    cleo11_pipeline.add_argument("--max-new-tokens", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        manifest = prepare_data(config, force=args.force)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-general":
        from .general_data import load_general_config, prepare_general_data

        general_config = load_general_config(args.general_config)
        manifest = prepare_general_data(
            general_config,
            config.data.tokenizer_path,
            force=args.force,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        result = train_model(config, requested_device=args.device, resume_path=args.resume)
        if not args.skip_final_samples:
            write_final_artifacts(config, result.checkpoint)
        return 0 if result.acceptance_passed else 2
    if args.command == "identity-tune":
        from .identity_tuning import fine_tune_identity

        report = fine_tune_identity(
            args.checkpoint,
            args.output,
            config.data.tokenizer_path,
            requested_device=args.device,
            steps=args.steps,
            learning_rate=args.learning_rate,
            story_weight=args.story_weight,
            identity_batch_size=args.identity_batch_size,
            story_batch_size=args.story_batch_size,
            eval_interval=args.eval_interval,
            validation_batches=args.validation_batches,
            max_story_loss_increase=args.max_story_loss_increase,
            seed=args.seed,
        )
        return 0 if report["accepted"] else 2
    if args.command == "generalize":
        from .general_data import load_general_config
        from .general_training import generalize_model

        general_config = load_general_config(args.general_config)
        report = generalize_model(
            args.checkpoint,
            args.output,
            config.data.tokenizer_path,
            general_config,
            requested_device=args.device,
            pretrain_steps=args.pretrain_steps,
            instruction_steps=args.instruction_steps,
            max_wall_time_seconds=args.max_wall_time_seconds,
            promote_to="artifacts/cleo-1.pt" if args.promote else None,
            preserve_base_as="artifacts/cleo-1-story.pt" if args.promote else None,
        )
        return 0 if report["accepted"] else 2
    if args.command == "capability-tune":
        from .capability_tuning import fine_tune_capabilities
        from .general_data import load_general_config

        general_config = load_general_config(args.general_config)
        report = fine_tune_capabilities(
            args.checkpoint,
            args.output,
            config.data.tokenizer_path,
            general_config,
            requested_device=args.device,
            steps=args.steps,
            learning_rate=args.learning_rate,
            required_capability_accuracy=args.required_accuracy,
            promote_to="artifacts/cleo-1.pt" if args.promote else None,
            preserve_source="artifacts/cleo-1.pt" if args.promote else None,
            preserve_as="artifacts/cleo-1-story.pt" if args.promote else None,
        )
        return 0 if report["accepted"] else 2
    if args.command == "general-identity-repair":
        from .general_data import load_general_config
        from .general_identity_repair import repair_general_identity

        general_config = load_general_config(args.general_config)
        report = repair_general_identity(
            args.checkpoint,
            args.output,
            config.data.tokenizer_path,
            general_config,
            requested_device=args.device,
            steps=args.steps,
            learning_rate=args.learning_rate,
            promote_to="artifacts/cleo-1.pt" if args.promote else None,
            preserve_source="artifacts/cleo-1.pt" if args.promote else None,
            preserve_as="artifacts/cleo-1-story.pt" if args.promote else None,
        )
        return 0 if report["accepted"] else 2
    if args.command == "evaluate":
        result = evaluate_checkpoint(
            args.checkpoint,
            requested_device=args.device,
            batches=args.batches,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "generate":
        device = select_device(args.device)
        text = generate_text(
            args.checkpoint,
            config.data.tokenizer_path,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            min_new_tokens=args.min_new_tokens,
            use_cache=not args.no_kv_cache,
        )
        print(text)
        return 0
    if args.command == "web":
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        device = select_device(args.device)
        configure_device(device, config)
        from .web import launch_web

        launch_web(
            args.checkpoint,
            config.data.tokenizer_path,
            device=device,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        return 0
    if args.command == "package-release":
        from .release import package_alpha_release

        payload = package_alpha_release(
            checkpoint=Path(args.checkpoint),
            tokenizer=Path(args.tokenizer),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "cleo11-spec":
        from .cleo11.config import load_cleo11_config
        from .cleo11.train import write_training_spec

        payload = write_training_spec(load_cleo11_config(args.cleo11_config), args.output_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "cleo11-estimate":
        from .cleo11.compute import estimate_compute
        from .cleo11.config import load_cleo11_config

        estimate = estimate_compute(load_cleo11_config(args.cleo11_config))
        print(json.dumps(estimate.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "cleo11-smoke":
        from .cleo11.train import smoke_from_config_path

        report = smoke_from_config_path(
            args.cleo11_config,
            requested_device=args.device,
            output_dir=args.output_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["accepted_smoke"] else 2
    if args.command == "cleo11-prepare":
        from .cleo11.prepare import prepare_from_config_path

        manifest = prepare_from_config_path(
            args.cleo11_config,
            profile=args.profile,
            force=args.force,
            synthetic=args.synthetic,
            vocab_size=args.vocab_size,
            reuse_tokenizer=args.reuse_tokenizer,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "cleo11-train":
        from .cleo11.pretrain import pretrain_from_config_path

        result = pretrain_from_config_path(
            args.cleo11_config,
            requested_device=args.device,
            resume_path=args.resume,
            max_steps_override=args.max_steps,
            microbatch_override=args.microbatch_size,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "cleo11-launch":
        from .cleo11.launch import emit_launch_plan, write_launch_script

        plan = emit_launch_plan(
            profile=args.profile,
            config=args.cleo11_config,
            nproc=args.nproc,
            prepare=not args.no_prepare,
            image=args.image,
            max_steps=args.max_steps,
        )
        if args.emit_script:
            script_path = write_launch_script(args.emit_script, plan)
            plan["emitted_script"] = str(script_path)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.command == "cleo11-instruction-tune":
        from .cleo11.instruction_tuning import instruction_tune_from_config_path

        report = instruction_tune_from_config_path(
            args.cleo11_config,
            args.checkpoint,
            output_path=args.output,
            tokenizer_path=args.tokenizer,
            requested_device=args.device,
            steps=args.steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["accepted_instruction_run"] else 2
    if args.command == "cleo11-identity-tune":
        from .cleo11.identity_tuning import identity_tune_from_config_path

        report = identity_tune_from_config_path(
            args.cleo11_config,
            args.checkpoint,
            output_path=args.output,
            tokenizer_path=args.tokenizer,
            requested_device=args.device,
            steps=args.steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["accepted_identity_run"] else 2
    if args.command == "cleo11-evaluate":
        from .cleo11.evaluation import evaluate_from_config_path

        report = evaluate_from_config_path(
            args.cleo11_config,
            args.checkpoint,
            tokenizer_path=args.tokenizer,
            requested_device=args.device,
            max_new_tokens=args.max_new_tokens,
            examples_per_category=args.examples_per_category,
            output_path=args.output,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["accepted"] else 2
    if args.command == "cleo11-pipeline":
        from .cleo11.pipeline import pipeline_from_config_path

        report = pipeline_from_config_path(
            args.cleo11_config,
            output_dir=args.output_dir,
            requested_device=args.device,
            pretrain_steps=args.pretrain_steps,
            instruction_steps=args.instruction_steps,
            identity_steps=args.identity_steps,
            microbatch_size=args.microbatch_size,
            examples_per_category=args.examples_per_category,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["accepted_pipeline_wiring"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
