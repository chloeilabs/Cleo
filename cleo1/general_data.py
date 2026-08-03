from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
import tomllib
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import download_file, sha256_file
from .tokenizer import ByteBPETokenizer


IGNORE_INDEX = -100


@dataclass(frozen=True)
class GeneralDataConfig:
    general_dataset: str
    general_revision: str
    general_license: str
    general_config: str
    train_urls: tuple[str, ...]
    train_sources: tuple[str, ...]
    train_source_bytes: tuple[int, ...]
    train_source_sha256: tuple[str, ...]
    validation_url: str
    validation_source: str
    validation_source_bytes: int
    validation_source_sha256: str
    test_url: str
    test_source: str
    test_source_bytes: int
    test_source_sha256: str
    instruction_dataset: str
    instruction_revision: str
    instruction_license: str
    instruction_url: str
    instruction_source: str
    instruction_source_bytes: int
    instruction_source_sha256: str
    train_tokens: str
    validation_tokens: str
    test_tokens: str
    instruction_train: str
    instruction_validation: str
    instruction_test: str
    manifest_path: str

    def validate(self) -> None:
        train_lengths = {
            len(self.train_urls),
            len(self.train_sources),
            len(self.train_source_bytes),
            len(self.train_source_sha256),
        }
        if train_lengths != {len(self.train_urls)} or not self.train_urls:
            raise ValueError("general training source lists must be non-empty and equal length")
        for value in (*self.train_source_sha256, self.validation_source_sha256,
                      self.test_source_sha256, self.instruction_source_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"invalid SHA-256 value: {value}")


@dataclass(frozen=True)
class GeneralTrainingConfig:
    seed: int
    block_size: int
    initial_microbatch_size: int
    effective_batch_tokens: int
    pretrain_steps: int
    instruction_steps: int
    pretrain_learning_rate: float
    pretrain_min_learning_rate: float
    pretrain_warmup_steps: int
    instruction_learning_rate: float
    weight_decay: float
    grad_clip: float
    story_pretrain_probability: float
    instruction_batch_size: int
    retention_batch_size: int
    instruction_general_weight: float
    instruction_story_weight: float
    instruction_identity_weight: float
    eval_interval: int
    eval_batches: int
    instruction_eval_examples: int
    max_wall_time_seconds: int
    required_general_loss_ratio: float
    required_instruction_loss_ratio: float
    max_story_loss_ratio: float
    required_identity_accuracy: float

    def validate(self) -> None:
        if self.block_size < 256:
            raise ValueError("general block size must be at least 256")
        if self.effective_batch_tokens % self.block_size:
            raise ValueError("effective batch tokens must be divisible by general block size")
        if self.pretrain_steps < 0 or self.instruction_steps < 0:
            raise ValueError("training steps cannot be negative")
        if self.pretrain_steps + self.instruction_steps < 1:
            raise ValueError("at least one generalization step is required")
        if not 0 <= self.story_pretrain_probability <= 1:
            raise ValueError("story pretraining probability must be in [0, 1]")
        if not 0 < self.required_identity_accuracy <= 1:
            raise ValueError("required identity accuracy must be in (0, 1]")
        for value in (
            self.pretrain_learning_rate,
            self.pretrain_min_learning_rate,
            self.instruction_learning_rate,
            self.grad_clip,
        ):
            if value <= 0:
                raise ValueError("learning rates and gradient clipping must be positive")


@dataclass(frozen=True)
class GeneralConfig:
    data: GeneralDataConfig
    training: GeneralTrainingConfig


def load_general_config(path: str | Path) -> GeneralConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    data_raw = dict(raw["data"])
    for key in ("train_urls", "train_sources", "train_source_bytes", "train_source_sha256"):
        data_raw[key] = tuple(data_raw[key])
    config = GeneralConfig(
        data=GeneralDataConfig(**data_raw),
        training=GeneralTrainingConfig(**raw["training"]),
    )
    config.data.validate()
    config.training.validate()
    return config


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    context: str
    response: str
    category: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InstructionExample":
        example = cls(
            instruction=str(value["instruction"]).strip(),
            context=str(value.get("context", "")).strip(),
            response=str(value["response"]).strip(),
            category=str(value.get("category", "uncategorized")).strip() or "uncategorized",
        )
        if not example.instruction or not example.response:
            raise ValueError("instruction rows require non-empty instruction and response fields")
        return example


@dataclass(frozen=True)
class EncodedInstructionExample:
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class EncodedTextCorpus:
    path: str
    documents: int
    tokens: int
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "documents": self.documents,
            "tokens": self.tokens,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "dtype": "uint16-le",
        }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, examples: Iterable[InstructionExample]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    categories: Counter[str] = Counter()
    with temporary.open("wb") as handle:
        for example in examples:
            payload = (
                json.dumps(asdict(example), ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
            count += 1
            categories[example.category] += 1
    temporary.replace(path)
    return {
        "path": str(path),
        "examples": count,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "categories": dict(sorted(categories.items())),
    }


def load_instruction_examples(path: str | Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(InstructionExample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid instruction row at {path}:{line_number}") from error
    if not examples:
        raise ValueError(f"no instruction examples found in {path}")
    return examples


def split_instruction_examples(
    examples: Iterable[InstructionExample],
) -> tuple[list[InstructionExample], list[InstructionExample], list[InstructionExample]]:
    """Create deterministic 90/5/5 category-stratified splits."""
    by_category: dict[str, list[tuple[str, InstructionExample]]] = defaultdict(list)
    for example in examples:
        canonical = json.dumps(asdict(example), ensure_ascii=False, sort_keys=True).encode("utf-8")
        by_category[example.category].append((hashlib.sha256(canonical).hexdigest(), example))
    train: list[InstructionExample] = []
    validation: list[InstructionExample] = []
    test: list[InstructionExample] = []
    for category in sorted(by_category):
        rows = [example for _, example in sorted(by_category[category], key=lambda row: row[0])]
        if len(rows) < 3:
            train.extend(rows)
            continue
        validation_count = max(1, round(len(rows) * 0.05))
        test_count = max(1, round(len(rows) * 0.05))
        train_count = len(rows) - validation_count - test_count
        if train_count < 1:
            train_count, validation_count, test_count = len(rows) - 2, 1, 1
        train.extend(rows[:train_count])
        validation.extend(rows[train_count : train_count + validation_count])
        test.extend(rows[train_count + validation_count :])
    key = lambda example: hashlib.sha256(
        json.dumps(asdict(example), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return sorted(train, key=key), sorted(validation, key=key), sorted(test, key=key)


def render_instruction_prompt(instruction: str, context: str = "") -> str:
    prompt = f"Instruction:\n{instruction.strip()}"
    if context.strip():
        prompt += f"\n\nContext:\n{context.strip()}"
    return prompt + "\n\nResponse:\n"


def encode_instruction_example(
    tokenizer: ByteBPETokenizer,
    example: InstructionExample,
    *,
    block_size: int,
) -> EncodedInstructionExample:
    if block_size < 32:
        raise ValueError("instruction block size must be at least 32")
    header = tokenizer.encode("Instruction:\n")
    instruction = tokenizer.encode(example.instruction)
    context_header = tokenizer.encode("\n\nContext:\n") if example.context else []
    context = tokenizer.encode(example.context) if example.context else []
    response_header = tokenizer.encode("\n\nResponse:\n")
    answer = tokenizer.encode(example.response, eos=True)

    max_sequence = block_size + 1
    max_answer = max(8, min(block_size // 3, len(answer)))
    if len(answer) > max_answer:
        answer = answer[: max_answer - 1] + [tokenizer.eos_id]
    fixed = 1 + len(header) + len(context_header) + len(response_header) + len(answer)
    content_budget = max_sequence - fixed
    if content_budget < 1:
        raise ValueError("instruction formatting leaves no room for prompt content")

    if context:
        minimum_instruction = min(len(instruction), max(16, content_budget // 3))
        instruction_take = min(len(instruction), minimum_instruction)
        context_take = min(len(context), content_budget - instruction_take)
        remaining = content_budget - instruction_take - context_take
        if remaining and instruction_take < len(instruction):
            extra = min(remaining, len(instruction) - instruction_take)
            instruction_take += extra
            remaining -= extra
        if remaining and context_take < len(context):
            context_take += min(remaining, len(context) - context_take)
    else:
        instruction_take = min(len(instruction), content_budget)
        context_take = 0

    prompt_ids = (
        [tokenizer.bos_id]
        + header
        + instruction[:instruction_take]
        + context_header
        + context[:context_take]
        + response_header
    )
    sequence = prompt_ids + answer
    if len(sequence) > max_sequence:
        raise AssertionError("encoded instruction exceeds configured context")
    inputs = sequence[:-1]
    targets = sequence[1:]
    targets[: max(len(prompt_ids) - 1, 0)] = [IGNORE_INDEX] * max(len(prompt_ids) - 1, 0)
    return EncodedInstructionExample(tuple(inputs), tuple(targets))


def instruction_batch(
    tokenizer: ByteBPETokenizer,
    examples: Iterable[InstructionExample],
    *,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        encode_instruction_example(tokenizer, example, block_size=block_size)
        for example in examples
    ]
    if not rows:
        raise ValueError("at least one instruction example is required")
    width = max(len(row.inputs) for row in rows)
    inputs = torch.full((len(rows), width), tokenizer.eos_id, dtype=torch.long)
    targets = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for index, row in enumerate(rows):
        inputs[index, : len(row.inputs)] = torch.tensor(row.inputs, dtype=torch.long)
        targets[index, : len(row.targets)] = torch.tensor(row.targets, dtype=torch.long)
    return inputs.to(device), targets.to(device)


def _iter_text_batches(path: str | Path, batch_size: int = 4096) -> Iterator[list[str]]:
    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema.names:
        raise ValueError(f"{path} does not contain a text column")
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
        yield [value.strip() for value in record_batch.column(0).to_pylist() if value and value.strip()]


def encode_text_corpus(
    sources: Iterable[str | Path],
    destination: str | Path,
    tokenizer: ByteBPETokenizer,
) -> EncodedTextCorpus:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    document_count = 0
    token_count = 0
    buffer: list[int] = []

    def flush(handle: Any) -> None:
        nonlocal token_count
        if not buffer:
            return
        payload = np.asarray(buffer, dtype="<u2").tobytes()
        handle.write(payload)
        digest.update(payload)
        token_count += len(buffer)
        buffer.clear()

    started = time.monotonic()
    with temporary.open("wb") as handle:
        for source in sources:
            for texts in _iter_text_batches(source):
                for value in texts:
                    buffer.extend(tokenizer.encode(value, bos=True, eos=True))
                    document_count += 1
                    if len(buffer) >= 1_000_000:
                        flush(handle)
                if document_count and document_count % 100_000 < len(texts):
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"  encoded {document_count:,} documents "
                        f"({token_count / 1_000_000:.1f}M tokens, "
                        f"{document_count / elapsed:.0f} documents/s)",
                        flush=True,
                    )
        flush(handle)
    temporary.replace(destination)
    return EncodedTextCorpus(
        path=str(destination),
        documents=document_count,
        tokens=token_count,
        bytes=destination.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _verify_source(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"source checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return actual


def prepare_general_data(
    config: GeneralConfig,
    tokenizer_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    data = config.data
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"base tokenizer is missing: {tokenizer_path}; run prepare first")
    outputs = [
        Path(data.train_tokens),
        Path(data.validation_tokens),
        Path(data.test_tokens),
        Path(data.instruction_train),
        Path(data.instruction_validation),
        Path(data.instruction_test),
        Path(data.manifest_path),
    ]
    if not force and all(path.exists() for path in outputs):
        print(f"General data already exists at {data.manifest_path}; use --force to rebuild", flush=True)
        return json.loads(Path(data.manifest_path).read_text(encoding="utf-8"))
    if force:
        for path in outputs:
            if path.exists():
                path.unlink()

    train_paths = [Path(value) for value in data.train_sources]
    for url, path, expected_bytes in zip(
        data.train_urls, data.train_sources, data.train_source_bytes, strict=True
    ):
        download_file(url, Path(path), expected_bytes)
    download_file(data.validation_url, Path(data.validation_source), data.validation_source_bytes)
    download_file(data.test_url, Path(data.test_source), data.test_source_bytes)
    download_file(data.instruction_url, Path(data.instruction_source), data.instruction_source_bytes)

    print("Verifying pinned general-data checksums", flush=True)
    train_hashes = [
        _verify_source(path, expected)
        for path, expected in zip(train_paths, data.train_source_sha256, strict=True)
    ]
    validation_hash = _verify_source(Path(data.validation_source), data.validation_source_sha256)
    test_hash = _verify_source(Path(data.test_source), data.test_source_sha256)
    instruction_hash = _verify_source(
        Path(data.instruction_source), data.instruction_source_sha256
    )

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    print("Encoding WikiText training corpus", flush=True)
    train_corpus = encode_text_corpus(train_paths, data.train_tokens, tokenizer)
    print("Encoding WikiText validation corpus", flush=True)
    validation_corpus = encode_text_corpus(
        [data.validation_source], data.validation_tokens, tokenizer
    )
    print("Encoding WikiText test corpus", flush=True)
    test_corpus = encode_text_corpus([data.test_source], data.test_tokens, tokenizer)

    print("Creating deterministic Dolly instruction splits", flush=True)
    all_instructions = load_instruction_examples(data.instruction_source)
    train_instructions, validation_instructions, test_instructions = split_instruction_examples(
        all_instructions
    )
    instruction_splits = {
        "train": _write_jsonl(Path(data.instruction_train), train_instructions),
        "validation": _write_jsonl(
            Path(data.instruction_validation), validation_instructions
        ),
        "test": _write_jsonl(Path(data.instruction_test), test_instructions),
    }

    manifest: dict[str, Any] = {
        "format_version": 1,
        "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": ByteBPETokenizer.checksum(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
        },
        "general_corpus": {
            "dataset": data.general_dataset,
            "configuration": data.general_config,
            "revision": data.general_revision,
            "license": data.general_license,
            "sources": {
                "train": [
                    {
                        "url": url,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": checksum,
                    }
                    for url, path, checksum in zip(
                        data.train_urls, train_paths, train_hashes, strict=True
                    )
                ],
                "validation": {
                    "url": data.validation_url,
                    "path": data.validation_source,
                    "bytes": Path(data.validation_source).stat().st_size,
                    "sha256": validation_hash,
                },
                "test": {
                    "url": data.test_url,
                    "path": data.test_source,
                    "bytes": Path(data.test_source).stat().st_size,
                    "sha256": test_hash,
                },
            },
            "corpora": {
                "train": train_corpus.to_dict(),
                "validation": validation_corpus.to_dict(),
                "test": test_corpus.to_dict(),
            },
        },
        "instruction_corpus": {
            "dataset": data.instruction_dataset,
            "revision": data.instruction_revision,
            "license": data.instruction_license,
            "source": {
                "url": data.instruction_url,
                "path": data.instruction_source,
                "bytes": Path(data.instruction_source).stat().st_size,
                "sha256": instruction_hash,
            },
            "splits": instruction_splits,
        },
    }
    _atomic_json(Path(data.manifest_path), manifest)
    print(f"Wrote general-data manifest to {data.manifest_path}", flush=True)
    return manifest
