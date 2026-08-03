from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .identity import COMPANY_NAME, MODEL_ID, MODEL_NAME, model_identity_metadata


@dataclass(frozen=True)
class ValidationPoint:
    step: int
    loss: float


@dataclass(frozen=True)
class SampleStory:
    prompt: str
    seed: int
    text: str


@dataclass(frozen=True)
class ModelProfile:
    company_name: str
    name: str
    model_id: str
    release: str
    checkpoint_name: str
    parameter_count: int
    training_step: int
    initial_validation_loss: float
    best_validation_loss: float
    best_validation_perplexity: float
    loss_reduction_percent: float
    elapsed_training_seconds: float
    training_tokens_seen: int
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    ffn_size: int
    dropout: float
    dataset_name: str
    dataset_revision: str
    dataset_license: str
    train_stories: int
    validation_stories: int
    train_tokens: int
    validation_tokens: int
    saved_at_utc: str
    runtime_device: str
    benchmark_device: str
    cached_tokens_per_second: float
    uncached_tokens_per_second: float
    cache_speedup: float
    benchmark_new_tokens: int
    benchmark_outputs_equal: bool
    identity_tuned: bool
    identity_tuning_steps: int
    identity_eval_accuracy: float
    identity_story_loss_ratio: float
    generalized: bool
    foundation_steps: int
    foundation_identity_steps: int
    general_pretrain_steps: int
    instruction_tuning_steps: int
    identity_repair_steps: int
    general_baseline_loss: float
    general_validation_loss: float
    general_validation_perplexity: float
    general_loss_reduction_percent: float
    instruction_baseline_loss: float
    instruction_validation_loss: float
    instruction_loss_reduction_percent: float
    story_retention_ratio: float
    general_dataset_name: str
    general_dataset_revision: str
    general_dataset_license: str
    general_train_documents: int
    general_validation_documents: int
    general_train_tokens: int
    general_validation_tokens: int
    instruction_dataset_name: str
    instruction_dataset_revision: str
    instruction_dataset_license: str
    instruction_train_examples: int
    instruction_validation_examples: int
    instruction_test_examples: int
    validation_curve: tuple[ValidationPoint, ...]
    samples: tuple[SampleStory, ...]
    model_card_path: Path | None
    benchmark_path: Path | None

    @property
    def summary(self) -> str:
        if self.generalized:
            return (
                f"{self.parameter_count:,} parameters · general-language alpha · "
                f"step {self.training_step:,} · general validation loss "
                f"{self.general_validation_loss:.4f} · {self.runtime_device}"
            )
        summary = (
            f"{self.parameter_count:,} parameters · step {self.training_step:,} · "
            f"validation loss {self.best_validation_loss:.4f} · {self.runtime_device}"
        )
        if self.identity_tuned:
            summary += f" · identity tune {self.identity_tuning_steps:,} steps"
        return summary

    @property
    def training_duration(self) -> str:
        total_minutes = round(self.elapsed_training_seconds / 60)
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours}h {minutes:02d}m"

    @classmethod
    def from_runtime(
        cls,
        checkpoint_path: str | Path,
        checkpoint: dict[str, Any],
        *,
        parameter_count: int,
        runtime_device: str,
    ) -> "ModelProfile":
        checkpoint_path = Path(checkpoint_path)
        artifact_dir = checkpoint_path.parent
        report = _read_json(artifact_dir / "training_report.json")
        benchmark_path = artifact_dir / "inference_benchmark.json"
        benchmark = _read_json(benchmark_path)
        manifest = dict(checkpoint["data_manifest"])
        model = dict(checkpoint["model_config"])
        training = dict(checkpoint["app_config"]["training"])
        identity = dict(checkpoint.get("identity", model_identity_metadata()))
        adaptation = dict(checkpoint.get("adaptation", {}))
        generalization = dict(checkpoint.get("generalization", {}))
        identity_repair = dict(checkpoint.get("identity_repair", {}))
        generalized = bool(generalization.get("accepted", False))
        general_benchmark_path = artifact_dir / "general_inference_benchmark.json"
        if generalized and general_benchmark_path.exists():
            benchmark_path = general_benchmark_path
            benchmark = _read_json(benchmark_path)

        foundation_initial_loss = float(
            report.get("initial_validation_loss", checkpoint.get("initial_validation_loss"))
        )
        foundation_best_loss = float(
            report.get("best_validation_loss", checkpoint.get("best_validation_loss"))
        )
        foundation_steps = int(
            adaptation.get("foundation_training_step", report.get("step", 0))
        )

        general_initial_loss = float(
            generalization.get("baseline_general_validation_loss", foundation_initial_loss)
        )
        general_final_loss = float(
            identity_repair.get(
                "general_loss",
                generalization.get("general_validation_loss", foundation_best_loss),
            )
        )
        selected_initial_loss = general_initial_loss if generalized else foundation_initial_loss
        selected_best_loss = general_final_loss if generalized else foundation_best_loss
        training_step = int(
            checkpoint.get("step", 0) if generalized else report.get("step", checkpoint.get("step", 0))
        )
        elapsed = float(
            checkpoint.get("elapsed_training_seconds", 0.0)
            if generalized
            else report.get(
                "elapsed_training_seconds", checkpoint.get("elapsed_training_seconds", 0.0)
            )
        )

        general_pretrain_steps = int(generalization.get("completed_pretrain_steps", 0))
        instruction_steps = int(generalization.get("completed_instruction_steps", 0))
        repair_steps = int(identity_repair.get("completed_steps", 0))
        foundation_identity_steps = int(adaptation.get("completed_steps", 0))
        training_tokens_seen = (
            (foundation_steps + general_pretrain_steps)
            * int(training["effective_batch_tokens"])
        )

        if generalized:
            curve = (
                ValidationPoint(step=0, loss=general_initial_loss),
                ValidationPoint(step=general_pretrain_steps, loss=float(generalization["general_validation_loss"])),
                ValidationPoint(
                    step=general_pretrain_steps + instruction_steps + repair_steps,
                    loss=general_final_loss,
                ),
            )
            samples_path = artifact_dir / "general_samples.json"
        else:
            curve = _read_validation_curve(artifact_dir / "metrics.jsonl")
            if not curve:
                curve = (
                    ValidationPoint(step=0, loss=foundation_initial_loss),
                    ValidationPoint(step=training_step, loss=foundation_best_loss),
                )
            samples_path = artifact_dir / "samples.json"

        sample_rows = _read_json(samples_path, default=[])
        samples = tuple(
            SampleStory(
                prompt=str(row["prompt"]),
                seed=int(row["seed"]),
                text=str(row.get("text", row.get("response", ""))),
            )
            for row in sample_rows
        )

        cached = dict(benchmark.get("cached", {}))
        uncached = dict(benchmark.get("uncached", {}))
        corpora = dict(manifest["corpora"])
        train_corpus = dict(corpora["train"])
        validation_corpus = dict(corpora["validation"])
        general_manifest = dict(checkpoint.get("general_data_manifest", {}))
        general_corpus = dict(general_manifest.get("general_corpus", {}))
        general_corpora = dict(general_corpus.get("corpora", {}))
        general_train = dict(general_corpora.get("train", {}))
        general_validation = dict(general_corpora.get("validation", {}))
        instruction_corpus = dict(general_manifest.get("instruction_corpus", {}))
        instruction_splits = dict(instruction_corpus.get("splits", {}))
        model_card_path = artifact_dir / "MODEL_CARD.md"

        instruction_initial_loss = float(
            generalization.get("baseline_instruction_validation_loss", 0.0)
        )
        instruction_final_loss = float(
            identity_repair.get(
                "instruction_loss",
                generalization.get("instruction_validation_loss", 0.0),
            )
        )
        general_reduction = _loss_reduction(general_initial_loss, general_final_loss)
        instruction_reduction = _loss_reduction(
            instruction_initial_loss, instruction_final_loss
        )

        return cls(
            company_name=str(identity["company_name"]),
            name=str(identity["model_name"]),
            model_id=str(identity["model_id"]),
            release="General-language alpha 01" if generalized else "Research release 01",
            checkpoint_name=checkpoint_path.name,
            parameter_count=parameter_count,
            training_step=training_step,
            initial_validation_loss=selected_initial_loss,
            best_validation_loss=selected_best_loss,
            best_validation_perplexity=math.exp(selected_best_loss),
            loss_reduction_percent=_loss_reduction(
                selected_initial_loss, selected_best_loss
            ),
            elapsed_training_seconds=elapsed,
            training_tokens_seen=training_tokens_seen,
            block_size=int(model["block_size"]),
            vocab_size=int(model["vocab_size"]),
            n_layer=int(model["n_layer"]),
            n_head=int(model["n_head"]),
            n_embd=int(model["n_embd"]),
            ffn_size=int(model["ffn_size"]),
            dropout=float(model["dropout"]),
            dataset_name=str(manifest["dataset"]),
            dataset_revision=str(manifest["revision"]),
            dataset_license=str(manifest["license"]),
            train_stories=int(train_corpus["stories"]),
            validation_stories=int(validation_corpus["stories"]),
            train_tokens=int(train_corpus["tokens"]),
            validation_tokens=int(validation_corpus["tokens"]),
            saved_at_utc=str(checkpoint.get("saved_at_utc", "not recorded")),
            runtime_device=runtime_device.upper(),
            benchmark_device=str(benchmark.get("device", "not measured")),
            cached_tokens_per_second=float(cached.get("tokens_per_second", 0.0)),
            uncached_tokens_per_second=float(uncached.get("tokens_per_second", 0.0)),
            cache_speedup=float(benchmark.get("speedup", 0.0)),
            benchmark_new_tokens=int(benchmark.get("new_tokens", 0)),
            benchmark_outputs_equal=bool(benchmark.get("outputs_equal", False)),
            identity_tuned=bool(
                identity_repair.get("accepted", adaptation.get("accepted", False))
            ),
            identity_tuning_steps=repair_steps or foundation_identity_steps,
            identity_eval_accuracy=float(
                identity_repair.get(
                    "identity_accuracy", adaptation.get("final_identity_accuracy", 0.0)
                )
            ),
            identity_story_loss_ratio=float(
                identity_repair.get(
                    "original_story_loss_ratio", adaptation.get("story_loss_ratio", 1.0)
                )
            ),
            generalized=generalized,
            foundation_steps=foundation_steps,
            foundation_identity_steps=foundation_identity_steps,
            general_pretrain_steps=general_pretrain_steps,
            instruction_tuning_steps=instruction_steps,
            identity_repair_steps=repair_steps,
            general_baseline_loss=general_initial_loss,
            general_validation_loss=general_final_loss,
            general_validation_perplexity=math.exp(general_final_loss),
            general_loss_reduction_percent=general_reduction,
            instruction_baseline_loss=instruction_initial_loss,
            instruction_validation_loss=instruction_final_loss,
            instruction_loss_reduction_percent=instruction_reduction,
            story_retention_ratio=float(
                identity_repair.get(
                    "original_story_loss_ratio",
                    generalization.get("story_loss_ratio", 1.0),
                )
            ),
            general_dataset_name=str(general_corpus.get("dataset", "")),
            general_dataset_revision=str(general_corpus.get("revision", "")),
            general_dataset_license=str(general_corpus.get("license", "")),
            general_train_documents=int(general_train.get("documents", 0)),
            general_validation_documents=int(general_validation.get("documents", 0)),
            general_train_tokens=int(general_train.get("tokens", 0)),
            general_validation_tokens=int(general_validation.get("tokens", 0)),
            instruction_dataset_name=str(instruction_corpus.get("dataset", "")),
            instruction_dataset_revision=str(instruction_corpus.get("revision", "")),
            instruction_dataset_license=str(instruction_corpus.get("license", "")),
            instruction_train_examples=int(
                dict(instruction_splits.get("train", {})).get("examples", 0)
            ),
            instruction_validation_examples=int(
                dict(instruction_splits.get("validation", {})).get("examples", 0)
            ),
            instruction_test_examples=int(
                dict(instruction_splits.get("test", {})).get("examples", 0)
            ),
            validation_curve=curve,
            samples=samples,
            model_card_path=model_card_path if model_card_path.exists() else None,
            benchmark_path=benchmark_path if benchmark_path.exists() else None,
        )

    @classmethod
    def placeholder(cls, runtime_device: str = "CPU") -> "ModelProfile":
        general_initial = 4.184773071606954
        general_final = 2.043870921929677
        instruction_initial = 4.137633930296895
        instruction_final = 1.8441442150149854
        return cls(
            company_name=COMPANY_NAME,
            name=MODEL_NAME,
            model_id=MODEL_ID,
            release="General-language alpha 01",
            checkpoint_name="cleo-1.pt",
            parameter_count=7_890_944,
            training_step=23_000,
            initial_validation_loss=general_initial,
            best_validation_loss=general_final,
            best_validation_perplexity=math.exp(general_final),
            loss_reduction_percent=_loss_reduction(general_initial, general_final),
            elapsed_training_seconds=11_333.3,
            training_tokens_seen=180_224_000,
            block_size=512,
            vocab_size=1_024,
            n_layer=6,
            n_head=5,
            n_embd=320,
            ffn_size=1_280,
            dropout=0.1,
            dataset_name="roneneldan/TinyStories",
            dataset_revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
            dataset_license="CDLA-Sharing-1.0",
            train_stories=529_875,
            validation_stories=21_990,
            train_tokens=234_379_409,
            validation_tokens=9_412_338,
            saved_at_utc="2026-08-03T01:20:15Z",
            runtime_device=runtime_device.upper(),
            benchmark_device="Apple M4 (MPS)",
            cached_tokens_per_second=142.6104,
            uncached_tokens_per_second=120.2861,
            cache_speedup=1.1856,
            benchmark_new_tokens=128,
            benchmark_outputs_equal=True,
            identity_tuned=True,
            identity_tuning_steps=100,
            identity_eval_accuracy=1.0,
            identity_story_loss_ratio=1.1697398997738722,
            generalized=True,
            foundation_steps=20_000,
            foundation_identity_steps=300,
            general_pretrain_steps=2_000,
            instruction_tuning_steps=600,
            identity_repair_steps=100,
            general_baseline_loss=general_initial,
            general_validation_loss=general_final,
            general_validation_perplexity=math.exp(general_final),
            general_loss_reduction_percent=_loss_reduction(general_initial, general_final),
            instruction_baseline_loss=instruction_initial,
            instruction_validation_loss=instruction_final,
            instruction_loss_reduction_percent=_loss_reduction(
                instruction_initial, instruction_final
            ),
            story_retention_ratio=1.1697398997738722,
            general_dataset_name="Salesforce/wikitext",
            general_dataset_revision="b08601e04326c79dfdd32d625aee71d232d685c3",
            general_dataset_license="CC-BY-SA-3.0 and GFDL",
            general_train_documents=1_165_029,
            general_validation_documents=2_461,
            general_train_tokens=335_385_799,
            general_validation_tokens=713_052,
            instruction_dataset_name="databricks/databricks-dolly-15k",
            instruction_dataset_revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
            instruction_dataset_license="CC-BY-SA-3.0",
            instruction_train_examples=13_511,
            instruction_validation_examples=750,
            instruction_test_examples=750,
            validation_curve=(
                ValidationPoint(step=0, loss=general_initial),
                ValidationPoint(step=2_000, loss=2.033014444510142),
                ValidationPoint(step=2_700, loss=general_final),
            ),
            samples=(),
            model_card_path=None,
            benchmark_path=None,
        )


def _loss_reduction(initial: float, final: float) -> float:
    if initial <= 0:
        return 0.0
    return 100.0 * (initial - final) / initial


def _read_json(path: Path, *, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_validation_curve(path: Path) -> tuple[ValidationPoint, ...]:
    if not path.exists():
        return ()
    by_step: dict[int, ValidationPoint] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") not in {"validation", "validation_final"}:
            continue
        point = ValidationPoint(step=int(row["step"]), loss=float(row["loss"]))
        by_step[point.step] = point
    return tuple(by_step[step] for step in sorted(by_step))
