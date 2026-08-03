from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as pq
import torch

from .config import AppConfig
from .tokenizer import ByteBPETokenizer


CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_file(url: str, destination: Path, expected_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_bytes:
        print(f"Using existing {destination} ({expected_bytes:,} bytes)", flush=True)
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "cleo-1/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as error:
        if error.code == 416 and existing == expected_bytes:
            partial.replace(destination)
            return
        raise
    append = existing > 0 and response.status == 206
    if existing and not append:
        existing = 0
    mode = "ab" if append else "wb"
    downloaded = existing
    next_report = downloaded + 32 * CHUNK_SIZE
    print(f"Downloading {destination.name} from byte {existing:,}", flush=True)
    with response, partial.open(mode) as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                print(f"  {downloaded / (1024 * 1024):.1f} MiB", flush=True)
                next_report += 32 * CHUNK_SIZE
    if downloaded != expected_bytes:
        raise RuntimeError(
            f"downloaded size mismatch for {destination}: expected {expected_bytes}, got {downloaded}"
        )
    partial.replace(destination)


def iter_story_batches(path: str | Path, batch_size: int = 4096) -> Iterator[list[str]]:
    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema.names:
        raise ValueError(f"{path} does not contain a text column")
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
        yield [value for value in record_batch.column(0).to_pylist() if value]


def collect_tokenizer_sample(path: str | Path, max_bytes: int) -> bytes:
    sample = bytearray()
    separator = b"\n\n"
    for stories in iter_story_batches(path):
        for story in stories:
            encoded = story.encode("utf-8")
            remaining = max_bytes - len(sample)
            if remaining <= 0:
                return bytes(sample)
            sample.extend((encoded + separator)[:remaining])
    return bytes(sample)


@dataclass(frozen=True)
class EncodedCorpus:
    path: str
    stories: int
    tokens: int
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "stories": self.stories,
            "tokens": self.tokens,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "dtype": "uint16-le",
        }


def encode_parquet(
    source: Path,
    destination: Path,
    tokenizer: ByteBPETokenizer,
) -> EncodedCorpus:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    story_count = 0
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
        for stories in iter_story_batches(source):
            for story in stories:
                buffer.extend(tokenizer.encode(story, bos=True, eos=True))
                story_count += 1
                if len(buffer) >= 1_000_000:
                    flush(handle)
            if story_count and story_count % 50_000 < len(stories):
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"  encoded {story_count:,} stories ({token_count / 1_000_000:.1f}M tokens, "
                    f"{story_count / elapsed:.0f} stories/s)",
                    flush=True,
                )
        flush(handle)
    temporary.replace(destination)
    return EncodedCorpus(
        path=str(destination),
        stories=story_count,
        tokens=token_count,
        bytes=destination.stat().st_size,
        sha256=digest.hexdigest(),
    )


def prepare_data(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    data = config.data
    train_source = Path(data.train_source)
    validation_source = Path(data.validation_source)
    tokenizer_path = Path(data.tokenizer_path)
    train_tokens = Path(data.train_tokens)
    validation_tokens = Path(data.validation_tokens)
    manifest_path = Path(data.manifest_path)
    outputs = [tokenizer_path, train_tokens, validation_tokens, manifest_path]
    if not force and all(path.exists() for path in outputs):
        print(f"Prepared data already exists at {manifest_path}; use --force to rebuild", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if force:
        for path in outputs:
            if path.exists():
                path.unlink()

    download_file(data.train_url, train_source, data.train_source_bytes)
    download_file(data.validation_url, validation_source, data.validation_source_bytes)
    print("Hashing source files", flush=True)
    train_sha = sha256_file(train_source)
    validation_sha = sha256_file(validation_source)
    print(f"Training tokenizer on {data.tokenizer_sample_bytes:,} source bytes", flush=True)
    sample = collect_tokenizer_sample(train_source, data.tokenizer_sample_bytes)
    sample_sha = hashlib.sha256(sample).hexdigest()
    tokenizer = ByteBPETokenizer.train(
        sample,
        vocab_size=data.tokenizer_vocab_size,
        metadata={
            "dataset": data.dataset_name,
            "revision": data.revision,
            "license": data.license,
            "source_sha256": train_sha,
            "sample_bytes": len(sample),
            "sample_sha256": sample_sha,
            "pretokenization": "alternating UTF-8 whitespace/non-whitespace byte spans",
            "tie_break": "highest pair frequency, then lexicographically smallest token-id pair",
        },
    )
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}", flush=True)
    print("Encoding training corpus", flush=True)
    encoded_train = encode_parquet(train_source, train_tokens, tokenizer)
    print("Encoding validation corpus", flush=True)
    encoded_validation = encode_parquet(validation_source, validation_tokens, tokenizer)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset": data.dataset_name,
        "revision": data.revision,
        "license": data.license,
        "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {
            "train": {
                "url": data.train_url,
                "path": str(train_source),
                "bytes": train_source.stat().st_size,
                "sha256": train_sha,
            },
            "validation": {
                "url": data.validation_url,
                "path": str(validation_source),
                "bytes": validation_source.stat().st_size,
                "sha256": validation_sha,
            },
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": ByteBPETokenizer.checksum(tokenizer_path),
            "vocab_size": tokenizer.vocab_size,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "sample_bytes": len(sample),
            "sample_sha256": sample_sha,
        },
        "corpora": {
            "train": encoded_train.to_dict(),
            "validation": encoded_validation.to_dict(),
        },
    }
    _atomic_json(manifest_path, manifest)
    print(f"Wrote data manifest to {manifest_path}", flush=True)
    return manifest


class TokenCorpus:
    def __init__(self, path: str | Path, block_size: int) -> None:
        self.path = Path(path)
        self.block_size = block_size
        self.tokens = np.memmap(self.path, dtype="<u2", mode="r")
        if len(self.tokens) <= block_size:
            raise ValueError(f"{path} has too few tokens for block size {block_size}")

    def __len__(self) -> int:
        return len(self.tokens)

    def batch_from_starts(self, starts: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = starts.tolist()
        sequences = np.stack(
            [np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64) for start in positions]
        )
        batch = torch.from_numpy(sequences)
        return batch[:, :-1].to(device), batch[:, 1:].to(device)

    def random_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(
            0,
            len(self.tokens) - self.block_size - 1,
            (batch_size,),
            generator=generator,
        )
        return self.batch_from_starts(starts, device)


def copy_stream(source: Any, destination: Any) -> None:
    """Compatibility wrapper retained for straightforward download testing."""
    shutil.copyfileobj(source, destination, length=CHUNK_SIZE)
