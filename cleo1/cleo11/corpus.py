from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ShardedTokenCorpus:
    """Memmap one or more uint16-le token shards as a single logical corpus."""

    def __init__(self, paths: list[str] | list[Path], block_size: int) -> None:
        if not paths:
            raise ValueError("at least one token shard path is required")
        self.block_size = block_size
        self.paths = [Path(path) for path in paths]
        self.shards = [np.memmap(path, dtype="<u2", mode="r") for path in self.paths]
        self.lengths = [len(shard) for shard in self.shards]
        self.total_tokens = int(sum(self.lengths))
        if self.total_tokens <= block_size + 1:
            raise ValueError(
                f"corpus has too few tokens ({self.total_tokens}) for block size {block_size}"
            )
        cumulative = []
        running = 0
        for length in self.lengths:
            running += length
            cumulative.append(running)
        self.cumulative = cumulative

    def __len__(self) -> int:
        return self.total_tokens

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total_tokens:
            raise IndexError(index)
        for shard_index, end in enumerate(self.cumulative):
            start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
            if index < end:
                return shard_index, index - start
        raise IndexError(index)

    def _slice(self, start: int, length: int) -> np.ndarray:
        values = np.empty(length, dtype=np.int64)
        filled = 0
        cursor = start
        while filled < length:
            shard_index, offset = self._locate(cursor)
            shard = self.shards[shard_index]
            take = min(length - filled, len(shard) - offset)
            values[filled : filled + take] = np.asarray(shard[offset : offset + take], dtype=np.int64)
            filled += take
            cursor += take
        return values

    def batch_from_starts(self, starts: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = np.stack(
            [self._slice(int(start), self.block_size + 1) for start in starts.tolist()]
        )
        batch = torch.from_numpy(sequences)
        return batch[:, :-1].to(device), batch[:, 1:].to(device)

    def random_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep starts away from the final block across the concatenated corpus.
        starts = torch.randint(
            0,
            self.total_tokens - self.block_size - 1,
            (batch_size,),
            generator=generator,
        )
        return self.batch_from_starts(starts, device)


def load_train_corpus(manifest: dict, block_size: int) -> ShardedTokenCorpus:
    paths = manifest["corpora"]["train"]["paths"]
    return ShardedTokenCorpus(paths, block_size)


def load_validation_corpus(manifest: dict, block_size: int) -> ShardedTokenCorpus:
    path = manifest["corpora"]["validation"]["path"]
    return ShardedTokenCorpus([path], block_size)
