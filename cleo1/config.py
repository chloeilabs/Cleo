from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    revision: str
    license: str
    train_url: str
    validation_url: str
    train_source: str
    validation_source: str
    train_source_bytes: int
    validation_source_bytes: int
    tokenizer_path: str
    train_tokens: str
    validation_tokens: str
    manifest_path: str
    tokenizer_sample_bytes: int
    tokenizer_vocab_size: int


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1024
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 5
    n_embd: int = 320
    ffn_size: int = 1280
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size < 258:
            raise ValueError("vocab_size must include 256 bytes plus BOS/EOS")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    initial_microbatch_size: int
    effective_batch_tokens: int
    max_steps: int
    max_wall_time_seconds: int
    eval_interval: int
    eval_batches: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    grad_clip: float
    mps_memory_fraction: float
    artifacts_dir: str


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AppConfig":
        config = cls(
            data=DataConfig(**value["data"]),
            model=ModelConfig(**value["model"]),
            training=TrainingConfig(**value["training"]),
        )
        config.model.validate()
        return config


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return AppConfig.from_dict(raw)
