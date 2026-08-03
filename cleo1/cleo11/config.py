from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class Cleo11ModelConfig:
    """Modern decoder-only transformer targeting ~135M parameters."""

    vocab_size: int = 16384
    block_size: int = 2048
    n_layer: int = 30
    n_head: int = 10
    n_kv_head: int = 2
    n_embd: int = 640
    ffn_size: int = 1664
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head for grouped-query attention")
        if self.vocab_size < 258:
            raise ValueError("vocab_size must include 256 bytes plus BOS/EOS")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.ffn_size < 1:
            raise ValueError("ffn_size must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass(frozen=True)
class Cleo11TrainingConfig:
    seed: int = 1337
    target_tokens: int = 2_700_000_000
    tokens_per_parameter_floor: float = 20.0
    initial_microbatch_size: int = 4
    effective_batch_tokens: int = 524_288
    max_steps: int = 0  # 0 means derive from target_tokens / effective_batch_tokens
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    grad_clip: float = 1.0
    eval_interval: int = 1000
    eval_batches: int = 50
    artifacts_dir: str = "artifacts/cleo11"
    smoke_steps: int = 8
    smoke_batch_size: int = 2
    max_wall_time_seconds: int = 0  # 0 means no wall-time limit


@dataclass(frozen=True)
class Cleo11DataSource:
    name: str
    role: str
    weight: float
    notes: str = ""
    hub_id: str = ""
    hub_config: str = ""
    license: str = ""
    text_column: str = "text"
    content_backend: str = "hub_text"  # hub_text | softwareheritage_s3


@dataclass(frozen=True)
class Cleo11PrepProfile:
    name: str
    train_tokens: int
    validation_tokens: int
    tokenizer_sample_bytes: int


@dataclass(frozen=True)
class Cleo11PrepConfig:
    output_dir: str = "data/cleo11/processed"
    tokenizer_path: str = "data/cleo11/processed/tokenizer.json"
    train_glob: str = "data/cleo11/processed/train-*.bin"
    validation_tokens_path: str = "data/cleo11/processed/validation.bin"
    manifest_path: str = "data/cleo11/processed/manifest.json"
    default_profile: str = "full"
    shard_tokens: int = 100_000_000
    seed: int = 1337
    text_column: str = "text"
    profiles: tuple[Cleo11PrepProfile, ...] = field(default_factory=tuple)

    def profile(self, name: str) -> Cleo11PrepProfile:
        for item in self.profiles:
            if item.name == name:
                return item
        raise KeyError(f"unknown prep profile: {name}")


@dataclass(frozen=True)
class Cleo11DataConfig:
    tokenizer_vocab_size: int = 16384
    tokenizer_kind: str = "byte_bpe"
    mixture_name: str = "fineweb-edu-led"
    minimum_pretrain_tokens: int = 2_700_000_000
    sources: tuple[Cleo11DataSource, ...] = field(default_factory=tuple)
    instruction_stage: str = "separate_instruction_tuning"
    identity_stage: str = "separate_identity_tuning"


@dataclass(frozen=True)
class Cleo11EvalGate:
    category: str
    metric: str
    minimum: float
    description: str


@dataclass(frozen=True)
class Cleo11EvaluationConfig:
    contract_version: int = 1
    require_all_gates: bool = True
    gates: tuple[Cleo11EvalGate, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Cleo11Config:
    model: Cleo11ModelConfig
    training: Cleo11TrainingConfig
    data: Cleo11DataConfig
    evaluation: Cleo11EvaluationConfig
    prep: Cleo11PrepConfig = field(default_factory=Cleo11PrepConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def derived_max_steps(self) -> int:
        if self.training.max_steps > 0:
            return self.training.max_steps
        return max(
            1,
            (self.training.target_tokens + self.training.effective_batch_tokens - 1)
            // self.training.effective_batch_tokens,
        )


def _parse_sources(raw: list[dict[str, Any]]) -> tuple[Cleo11DataSource, ...]:
    return tuple(Cleo11DataSource(**item) for item in raw)


def _parse_gates(raw: list[dict[str, Any]]) -> tuple[Cleo11EvalGate, ...]:
    return tuple(Cleo11EvalGate(**item) for item in raw)


def _parse_profiles(raw: list[dict[str, Any]]) -> tuple[Cleo11PrepProfile, ...]:
    return tuple(Cleo11PrepProfile(**item) for item in raw)


def cleo11_config_from_dict(value: dict[str, Any]) -> Cleo11Config:
    data_raw = dict(value["data"])
    sources = _parse_sources(list(data_raw.pop("sources", [])))
    evaluation_raw = dict(value["evaluation"])
    gates = _parse_gates(list(evaluation_raw.pop("gates", [])))
    prep_raw = dict(value.get("prep", {}))
    profiles = _parse_profiles(list(prep_raw.pop("profiles", [])))
    if not profiles:
        profiles = (
            Cleo11PrepProfile(
                name="dev",
                train_tokens=50_000_000,
                validation_tokens=2_000_000,
                tokenizer_sample_bytes=4_194_304,
            ),
            Cleo11PrepProfile(
                name="full",
                train_tokens=2_720_000_000,
                validation_tokens=10_000_000,
                tokenizer_sample_bytes=33_554_432,
            ),
        )
    config = Cleo11Config(
        model=Cleo11ModelConfig(**value["model"]),
        training=Cleo11TrainingConfig(**value["training"]),
        data=Cleo11DataConfig(sources=sources, **data_raw),
        evaluation=Cleo11EvaluationConfig(gates=gates, **evaluation_raw),
        prep=Cleo11PrepConfig(profiles=profiles, **prep_raw),
    )
    config.model.validate()
    return config


def load_cleo11_config(path: str | Path) -> Cleo11Config:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return cleo11_config_from_dict(raw)


def default_135m_config() -> Cleo11Config:
    return load_cleo11_config(Path(__file__).resolve().parents[2] / "configs" / "cleo11_135m.toml")
