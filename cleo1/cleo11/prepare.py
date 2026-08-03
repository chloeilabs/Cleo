from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterator

import numpy as np

from ..tokenizer import ByteBPETokenizer
from .config import Cleo11Config, Cleo11DataSource, Cleo11PrepProfile, load_cleo11_config
from .data_manifest import DEFAULT_MIXTURE


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(config: Cleo11Config) -> tuple[Cleo11DataSource, ...]:
    return config.data.sources or DEFAULT_MIXTURE


def iter_synthetic_documents(seed: int) -> Iterator[str]:
    """Yield deterministic synthetic documents for offline CI / smoke prep."""

    rng = random.Random(seed)
    topics = (
        "photosynthesis",
        "prime numbers",
        "river deltas",
        "python functions",
        "heat transfer",
        "grammar rules",
        "circuit boards",
        "map projections",
    )
    while True:
        topic = topics[rng.randrange(len(topics))]
        sentences = []
        for _ in range(rng.randint(4, 12)):
            sentences.append(
                f"This educational note explains {topic} with example {rng.randint(1, 999)} "
                f"and a short practice problem for students."
            )
        if topic == "python functions":
            sentences.append(
                "def add(left, right):\n    return left + right\n\n"
                f"assert add({rng.randint(1, 9)}, {rng.randint(1, 9)}) >= 2\n"
            )
        yield "\n".join(sentences) + "\n"


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "cleo11 data prep requires the optional cleo11 extras; "
            "install with `uv sync --group cleo11` or `pip install 'cleo-1[cleo11]'`"
        ) from error
    return load_dataset


def iter_hub_documents(source: Cleo11DataSource, *, text_column: str) -> Iterator[str]:
    load_dataset = _require_datasets()
    if not source.hub_id:
        raise ValueError(f"source {source.name!r} is missing hub_id")
    dataset = load_dataset(
        source.hub_id,
        name=source.hub_config or None,
        split="train",
        streaming=True,
    )
    for row in dataset:
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text


def collect_tokenizer_sample(
    document_iters: list[tuple[str, Iterator[str]]],
    *,
    max_bytes: int,
    seed: int,
) -> bytes:
    rng = random.Random(seed)
    sample = bytearray()
    separator = b"\n\n"
    active = list(document_iters)
    while active and len(sample) < max_bytes:
        index = rng.randrange(len(active))
        name, iterator = active[index]
        try:
            document = next(iterator)
        except StopIteration:
            active.pop(index)
            continue
        encoded = document.encode("utf-8")
        remaining = max_bytes - len(sample)
        sample.extend((encoded + separator)[:remaining])
        _ = name
    if len(sample) < 1024:
        raise RuntimeError("tokenizer sample is too small; check data sources")
    return bytes(sample)


class ShardWriter:
    def __init__(self, directory: Path, *, prefix: str, shard_tokens: int) -> None:
        self.directory = directory
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.tokens_in_shard = 0
        self.total_tokens = 0
        self.paths: list[Path] = []
        self._handle = None
        self._digest = hashlib.sha256()
        self._open_next()

    def _open_next(self) -> None:
        if self._handle is not None:
            self._handle.close()
        path = self.directory / f"{self.prefix}-{self.shard_index:05d}.bin"
        self.paths.append(path)
        self._handle = path.open("wb")
        self.tokens_in_shard = 0
        self.shard_index += 1

    def write_tokens(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        assert self._handle is not None
        offset = 0
        while offset < len(token_ids):
            capacity = self.shard_tokens - self.tokens_in_shard
            if capacity <= 0:
                self._open_next()
                capacity = self.shard_tokens
            chunk = token_ids[offset : offset + capacity]
            payload = np.asarray(chunk, dtype="<u2").tobytes()
            self._handle.write(payload)
            self._digest.update(payload)
            self.tokens_in_shard += len(chunk)
            self.total_tokens += len(chunk)
            offset += len(chunk)

    def close(self) -> dict[str, Any]:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        # Drop an empty trailing shard if somehow created with zero bytes.
        kept: list[Path] = []
        for path in self.paths:
            if path.exists() and path.stat().st_size > 0:
                kept.append(path)
            elif path.exists():
                path.unlink()
        self.paths = kept
        return {
            "paths": [str(path) for path in self.paths],
            "tokens": self.total_tokens,
            "bytes": sum(path.stat().st_size for path in self.paths),
            "sha256": self._digest.hexdigest(),
            "dtype": "uint16-le",
            "shard_count": len(self.paths),
        }


def _weighted_choice(sources: list[Cleo11DataSource], rng: random.Random) -> int:
    total = sum(source.weight for source in sources)
    pick = rng.random() * total
    cumulative = 0.0
    for index, source in enumerate(sources):
        cumulative += source.weight
        if pick <= cumulative:
            return index
    return len(sources) - 1


def encode_mixture(
    *,
    sources: tuple[Cleo11DataSource, ...],
    iterators: dict[str, Iterator[str]],
    tokenizer: ByteBPETokenizer,
    train_tokens: int,
    validation_tokens: int,
    shard_tokens: int,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], int]:
    rng = random.Random(seed + 17)
    source_list = list(sources)
    per_source = {source.name: 0 for source in source_list}
    documents = 0
    validation_path = output_dir / "validation.bin"
    validation_tmp = validation_path.with_suffix(".bin.tmp")
    validation_digest = hashlib.sha256()
    validation_count = 0
    train_writer = ShardWriter(output_dir, prefix="train", shard_tokens=shard_tokens)

    with validation_tmp.open("wb") as validation_handle:
        while train_writer.total_tokens < train_tokens or validation_count < validation_tokens:
            index = _weighted_choice(source_list, rng)
            source = source_list[index]
            iterator = iterators[source.name]
            try:
                document = next(iterator)
            except StopIteration:
                # Restart synthetic / exhausted streams by skipping this source briefly.
                continue
            tokens = tokenizer.encode(document, bos=True, eos=True)
            if not tokens:
                continue
            documents += 1
            # Fill validation first so the holdout is not empty on tiny budgets.
            if validation_count < validation_tokens:
                take = min(len(tokens), validation_tokens - validation_count)
                payload = np.asarray(tokens[:take], dtype="<u2").tobytes()
                validation_handle.write(payload)
                validation_digest.update(payload)
                validation_count += take
                tokens = tokens[take:]
            if tokens and train_writer.total_tokens < train_tokens:
                remaining = train_tokens - train_writer.total_tokens
                chunk = tokens[:remaining]
                train_writer.write_tokens(chunk)
                per_source[source.name] += len(chunk)
            if documents % 1000 == 0:
                print(
                    f"  encoded documents={documents:,} "
                    f"train_tokens={train_writer.total_tokens:,} "
                    f"validation_tokens={validation_count:,}",
                    flush=True,
                )

    validation_tmp.replace(validation_path)
    train_info = train_writer.close()
    validation_info = {
        "path": str(validation_path),
        "tokens": validation_count,
        "bytes": validation_path.stat().st_size,
        "sha256": validation_digest.hexdigest(),
        "dtype": "uint16-le",
    }
    return train_info, validation_info, per_source, documents


def _restartable_hub(source: Cleo11DataSource, *, text_column: str) -> Iterator[str]:
    while True:
        yielded = False
        for document in iter_hub_documents(source, text_column=text_column):
            yielded = True
            yield document
        if not yielded:
            raise RuntimeError(f"source {source.name!r} produced no documents")


def _build_iterators(
    config: Cleo11Config,
    sources: tuple[Cleo11DataSource, ...],
    *,
    synthetic: bool,
) -> dict[str, Iterator[str]]:
    if synthetic:
        return {
            source.name: iter_synthetic_documents(config.prep.seed + index)
            for index, source in enumerate(sources)
        }
    return {
        source.name: _restartable_hub(source, text_column=config.prep.text_column)
        for source in sources
    }


def prepare_cleo11_data(
    config: Cleo11Config,
    *,
    profile_name: str | None = None,
    force: bool = False,
    synthetic: bool = False,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    prep = config.prep
    profile = prep.profile(profile_name or prep.default_profile)
    output_dir = Path(prep.output_dir)
    tokenizer_path = Path(prep.tokenizer_path)
    manifest_path = Path(prep.manifest_path)
    outputs = [tokenizer_path, Path(prep.validation_tokens_path), manifest_path]
    if not force and manifest_path.exists() and tokenizer_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("profile") == profile.name and existing.get("synthetic") == synthetic:
            print(
                f"Prepared Cleo 1.1 data already exists at {manifest_path}; use --force to rebuild",
                flush=True,
            )
            return existing

    if force and output_dir.exists():
        for path in output_dir.glob("train-*.bin"):
            path.unlink()
        for path in outputs:
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _sources(config)
    weight_sum = sum(source.weight for source in sources)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"mixture weights must sum to 1.0, got {weight_sum}")

    print(
        f"Preparing Cleo 1.1 profile={profile.name} "
        f"train_tokens={profile.train_tokens:,} validation_tokens={profile.validation_tokens:,} "
        f"synthetic={synthetic}",
        flush=True,
    )
    sample_iters = _build_iterators(config, sources, synthetic=synthetic)
    sample = collect_tokenizer_sample(
        [(name, iterator) for name, iterator in sample_iters.items()],
        max_bytes=profile.tokenizer_sample_bytes,
        seed=prep.seed,
    )
    resolved_vocab = vocab_size or config.data.tokenizer_vocab_size
    if synthetic and resolved_vocab > 1024 and profile.tokenizer_sample_bytes < 8_000_000:
        # Keep synthetic CI runs tractable with the pure-Python BPE trainer.
        resolved_vocab = min(resolved_vocab, 512)
        print(f"Synthetic prep using vocab_size={resolved_vocab}", flush=True)

    print(f"Training byte-level BPE vocab_size={resolved_vocab} on {len(sample):,} bytes", flush=True)
    tokenizer = ByteBPETokenizer.train(
        sample,
        vocab_size=resolved_vocab,
        metadata={
            "mixture": config.data.mixture_name,
            "profile": profile.name,
            "synthetic": synthetic,
            "sample_bytes": len(sample),
            "sample_sha256": hashlib.sha256(sample).hexdigest(),
            "sources": [asdict(source) for source in sources],
        },
    )
    tokenizer.save(tokenizer_path)

    encode_iters = _build_iterators(config, sources, synthetic=synthetic)
    train_info, validation_info, per_source, documents = encode_mixture(
        sources=sources,
        iterators=encode_iters,
        tokenizer=tokenizer,
        train_tokens=profile.train_tokens,
        validation_tokens=profile.validation_tokens,
        shard_tokens=prep.shard_tokens,
        output_dir=output_dir,
        seed=prep.seed,
    )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "model_id": "cleo-1.1",
        "mixture_name": config.data.mixture_name,
        "profile": profile.name,
        "synthetic": synthetic,
        "seed": prep.seed,
        "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": [asdict(source) for source in sources],
        "source_token_counts": per_source,
        "documents_encoded": documents,
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": ByteBPETokenizer.checksum(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "sample_bytes": len(sample),
            "sample_sha256": hashlib.sha256(sample).hexdigest(),
        },
        "corpora": {
            "train": train_info,
            "validation": validation_info,
        },
        "paths": {
            "output_dir": str(output_dir),
            "tokenizer": str(tokenizer_path),
            "train_glob": str(Path(prep.train_glob)),
            "validation": str(Path(prep.validation_tokens_path)),
            "manifest": str(manifest_path),
        },
    }
    _atomic_json(manifest_path, manifest)
    print(f"Wrote Cleo 1.1 data manifest to {manifest_path}", flush=True)
    return manifest


def prepare_from_config_path(
    config_path: str | Path,
    *,
    profile: str | None = None,
    force: bool = False,
    synthetic: bool = False,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    return prepare_cleo11_data(
        load_cleo11_config(config_path),
        profile_name=profile,
        force=force,
        synthetic=synthetic,
        vocab_size=vocab_size,
    )


def clear_prepared_data(output_dir: str | Path) -> None:
    directory = Path(output_dir)
    if directory.exists():
        shutil.rmtree(directory)
