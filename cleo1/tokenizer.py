from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


_SPAN_PATTERN = re.compile(rb"\s+|\S+")


def _spans(value: bytes) -> Iterator[bytes]:
    for match in _SPAN_PATTERN.finditer(value):
        yield match.group(0)


def _replace_pair(sequence: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    output: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
            output.append(new_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return tuple(output)


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int]]
    vocab_size: int
    metadata: dict[str, Any]

    FORMAT_VERSION = 1

    def __post_init__(self) -> None:
        expected_merges = self.vocab_size - 258
        if len(self.merges) != expected_merges:
            raise ValueError(f"expected {expected_merges} merges, got {len(self.merges)}")
        self.bos_id = self.vocab_size - 2
        self.eos_id = self.vocab_size - 1
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._token_bytes: list[bytes] = [bytes([value]) for value in range(256)]
        for left, right in self.merges:
            if left >= len(self._token_bytes) or right >= len(self._token_bytes):
                raise ValueError("merge references a token that does not exist yet")
            self._token_bytes.append(self._token_bytes[left] + self._token_bytes[right])

    @classmethod
    def train(
        cls,
        corpus: bytes,
        vocab_size: int = 1024,
        metadata: dict[str, Any] | None = None,
    ) -> "ByteBPETokenizer":
        if vocab_size < 258:
            raise ValueError("vocab_size must be at least 258")
        span_counts: Counter[tuple[int, ...]] = Counter(tuple(span) for span in _spans(corpus))
        merges: list[tuple[int, int]] = []
        merge_count = vocab_size - 258
        report_every = max(1, merge_count // 20)
        for offset in range(merge_count):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence, frequency in span_counts.items():
                if len(sequence) < 2:
                    continue
                for pair in zip(sequence, sequence[1:]):
                    pair_counts[pair] += frequency
            if not pair_counts:
                raise ValueError(f"corpus exhausted after {offset} merges")
            best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            new_id = 256 + offset
            left, right = best_pair
            merged_counts: Counter[tuple[int, ...]] = Counter()
            for sequence, frequency in span_counts.items():
                if left in sequence and right in sequence:
                    merged_counts[_replace_pair(sequence, best_pair, new_id)] += frequency
                else:
                    merged_counts[sequence] += frequency
            span_counts = merged_counts
            merges.append(best_pair)
            if offset == 0 or (offset + 1) % report_every == 0 or offset + 1 == merge_count:
                print(
                    f"  BPE merges {offset + 1:,}/{merge_count:,}",
                    flush=True,
                )
        return cls(merges=merges, vocab_size=vocab_size, metadata=dict(metadata or {}))

    def _encode_span(self, span: bytes) -> list[int]:
        pieces = list(span)
        while len(pieces) > 1:
            candidates = (
                (self._ranks[pair], pair)
                for pair in zip(pieces, pieces[1:])
                if pair in self._ranks
            )
            best = min(candidates, default=None)
            if best is None:
                break
            _, pair = best
            pieces = list(_replace_pair(tuple(pieces), pair, 256 + self._ranks[pair]))
        return pieces

    def encode_bytes(self, value: bytes, *, bos: bool = False, eos: bool = False) -> list[int]:
        output = [self.bos_id] if bos else []
        for span in _spans(value):
            output.extend(self._encode_span(span))
        if eos:
            output.append(self.eos_id)
        return output

    def encode(self, value: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        return self.encode_bytes(value.encode("utf-8"), bos=bos, eos=eos)

    def decode_bytes(self, token_ids: Iterable[int], *, skip_special: bool = True) -> bytes:
        output = bytearray()
        for token_id in token_ids:
            if token_id in (self.bos_id, self.eos_id):
                if skip_special:
                    continue
                raise ValueError("special tokens do not map to source bytes")
            if token_id < 0 or token_id >= len(self._token_bytes):
                raise ValueError(f"invalid token id: {token_id}")
            output.extend(self._token_bytes[token_id])
        return bytes(output)

    def decode(self, token_ids: Iterable[int], *, errors: str = "replace") -> str:
        return self.decode_bytes(token_ids).decode("utf-8", errors=errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "kind": "byte_bpe",
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "merges": [list(pair) for pair in self.merges],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format_version") != cls.FORMAT_VERSION or value.get("kind") != "byte_bpe":
            raise ValueError("unsupported tokenizer format")
        tokenizer = cls(
            merges=[tuple(pair) for pair in value["merges"]],
            vocab_size=int(value["vocab_size"]),
            metadata=dict(value.get("metadata", {})),
        )
        if tokenizer.bos_id != value["bos_id"] or tokenizer.eos_id != value["eos_id"]:
            raise ValueError("special-token ids do not match vocabulary")
        return tokenizer

    @staticmethod
    def checksum(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
