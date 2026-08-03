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
    validation_curve: tuple[ValidationPoint, ...]
    samples: tuple[SampleStory, ...]
    model_card_path: Path | None
    benchmark_path: Path | None

    @property
    def summary(self) -> str:
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
        samples_path = artifact_dir / "samples.json"
        sample_rows = _read_json(samples_path, default=[])
        manifest = dict(checkpoint["data_manifest"])
        model = dict(checkpoint["model_config"])
        training = dict(checkpoint["app_config"]["training"])
        identity = dict(checkpoint.get("identity", model_identity_metadata()))
        adaptation = dict(checkpoint.get("adaptation", {}))

        initial_loss = float(
            report.get("initial_validation_loss", checkpoint.get("initial_validation_loss"))
        )
        best_loss = float(
            report.get("best_validation_loss", checkpoint.get("best_validation_loss"))
        )
        training_step = int(report.get("step", checkpoint.get("step", 0)))
        elapsed = float(
            report.get(
                "elapsed_training_seconds",
                checkpoint.get("elapsed_training_seconds", 0.0),
            )
        )
        reduction = 100.0 * (initial_loss - best_loss) / initial_loss
        curve = _read_validation_curve(artifact_dir / "metrics.jsonl")
        if not curve:
            curve = (
                ValidationPoint(step=0, loss=initial_loss),
                ValidationPoint(step=training_step, loss=best_loss),
            )

        samples = tuple(
            SampleStory(
                prompt=str(row["prompt"]),
                seed=int(row["seed"]),
                text=str(row["text"]),
            )
            for row in sample_rows
        )
        cached = dict(benchmark.get("cached", {}))
        uncached = dict(benchmark.get("uncached", {}))
        corpora = dict(manifest["corpora"])
        train_corpus = dict(corpora["train"])
        validation_corpus = dict(corpora["validation"])
        model_card_path = artifact_dir / "MODEL_CARD.md"

        return cls(
            company_name=str(identity["company_name"]),
            name=str(identity["model_name"]),
            model_id=str(identity["model_id"]),
            release="Research release 01",
            checkpoint_name=checkpoint_path.name,
            parameter_count=parameter_count,
            training_step=training_step,
            initial_validation_loss=initial_loss,
            best_validation_loss=best_loss,
            best_validation_perplexity=float(
                report.get("best_validation_perplexity", math.exp(best_loss))
            ),
            loss_reduction_percent=reduction,
            elapsed_training_seconds=elapsed,
            training_tokens_seen=training_step * int(training["effective_batch_tokens"]),
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
            identity_tuned=bool(adaptation.get("accepted", False)),
            identity_tuning_steps=int(adaptation.get("completed_steps", 0)),
            identity_eval_accuracy=float(adaptation.get("final_identity_accuracy", 0.0)),
            identity_story_loss_ratio=float(adaptation.get("story_loss_ratio", 1.0)),
            validation_curve=curve,
            samples=samples,
            model_card_path=model_card_path if model_card_path.exists() else None,
            benchmark_path=benchmark_path if benchmark_path.exists() else None,
        )

    @classmethod
    def placeholder(cls, runtime_device: str = "CPU") -> "ModelProfile":
        return cls(
            company_name=COMPANY_NAME,
            name=MODEL_NAME,
            model_id=MODEL_ID,
            release="Research release 01",
            checkpoint_name="best.pt",
            parameter_count=7_809_024,
            training_step=20_000,
            initial_validation_loss=6.8710,
            best_validation_loss=1.0661,
            best_validation_perplexity=2.9041,
            loss_reduction_percent=84.48,
            elapsed_training_seconds=9_333.5,
            training_tokens_seen=163_840_000,
            block_size=256,
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
            saved_at_utc="2026-08-02T22:25:00Z",
            runtime_device=runtime_device.upper(),
            benchmark_device="Apple M4 (MPS)",
            cached_tokens_per_second=86.5,
            uncached_tokens_per_second=39.9,
            cache_speedup=2.17,
            benchmark_new_tokens=128,
            benchmark_outputs_equal=True,
            identity_tuned=True,
            identity_tuning_steps=300,
            identity_eval_accuracy=1.0,
            identity_story_loss_ratio=1.0013,
            validation_curve=(
                ValidationPoint(step=0, loss=6.8710),
                ValidationPoint(step=1_000, loss=1.8205),
                ValidationPoint(step=5_000, loss=1.2771),
                ValidationPoint(step=10_000, loss=1.1627),
                ValidationPoint(step=15_000, loss=1.0968),
                ValidationPoint(step=20_000, loss=1.0661),
            ),
            samples=(),
            model_card_path=None,
            benchmark_path=None,
        )


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
