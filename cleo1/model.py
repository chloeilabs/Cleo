from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


type KVCache = tuple[torch.Tensor, torch.Tensor]
type PastKeyValues = tuple[KVCache, ...]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch_size, sequence_length, channels = x.shape
        qkv = self.qkv(x).view(batch_size, sequence_length, 3, self.n_head, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape != past_value.shape:
                raise ValueError("cached key and value tensors must have matching shapes")
            if past_key.shape[:2] != (batch_size, self.n_head):
                raise ValueError("cached key/value batch or head dimensions do not match the input")
            past_length = past_key.size(-2)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        total_length = past_length + sequence_length
        if total_length > self.causal_mask.size(-1):
            raise ValueError(
                f"cached sequence length {total_length} exceeds block size "
                f"{self.causal_mask.size(-1)}"
            )
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[
            :, :, past_length : past_length + sequence_length, :total_length
        ]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        output = self.resid_dropout(self.proj(attended))
        if use_cache:
            return output, (key, value)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.expand = nn.Linear(config.n_embd, config.ffn_size, bias=config.bias)
        self.proj = nn.Linear(config.ffn_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.expand(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(self.ln_attention(x))
        assert isinstance(attended, torch.Tensor)
        x = x + attended
        return x + self.mlp(self.ln_mlp(x))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        attended = self.attention(
            self.ln_attention(x),
            past_key_value=past_key_value,
            use_cache=True,
        )
        assert isinstance(attended, tuple)
        attention_output, present_key_value = attended
        x = x + attention_output
        return x + self.mlp(self.ln_mlp(x)), present_key_value


class CleoTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attention.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(f"sequence length {sequence_length} exceeds block size {self.config.block_size}")
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        past_key_values: PastKeyValues | None = None,
    ) -> tuple[torch.Tensor, PastKeyValues]:
        """Run an inference step and return per-layer attention keys and values.

        Learned absolute positions make silently shifting a full cache incorrect. The
        generation loop therefore rebuilds the cache from the current context window
        whenever it reaches ``block_size``.
        """
        batch_size, sequence_length = token_ids.shape
        if sequence_length < 1:
            raise ValueError("at least one token is required")
        if past_key_values is None:
            layer_caches: tuple[KVCache | None, ...] = (None,) * len(self.blocks)
            past_length = 0
        else:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("one key/value cache is required for every transformer block")
            layer_caches = past_key_values
            past_lengths = {cache[0].size(-2) for cache in past_key_values}
            if len(past_lengths) != 1:
                raise ValueError("all transformer block caches must have the same length")
            past_length = past_lengths.pop()
            for key, value in past_key_values:
                if key.shape != value.shape or key.size(0) != batch_size:
                    raise ValueError("cached key/value tensors do not match the input batch")
        total_length = past_length + sequence_length
        if total_length > self.config.block_size:
            raise ValueError(
                f"cached sequence length {total_length} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(past_length, total_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        present_key_values: list[KVCache] = []
        for block, layer_cache in zip(self.blocks, layer_caches, strict=True):
            x, present = block.forward_with_cache(x, layer_cache)
            present_key_values.append(present)
        logits = self.lm_head(self.final_norm(x))
        return logits, tuple(present_key_values)

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        next_logits = logits / temperature
        if top_k:
            k = min(top_k, next_logits.size(-1))
            threshold = torch.topk(next_logits, k).values[:, [-1]]
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
        probabilities = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    def generate_steps(
        self,
        token_ids: torch.Tensor,
        *,
        eos_id: int,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        min_new_tokens: int = 0,
        use_cache: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Yield the full token sequence after each newly sampled token."""
        if token_ids.ndim != 2 or token_ids.size(1) < 1:
            raise ValueError("token_ids must have shape (batch, sequence) with at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        self.eval()
        with torch.no_grad():
            context = token_ids[:, -self.config.block_size :]
            cache: PastKeyValues | None = None
            if use_cache:
                logits, cache = self.forward_with_cache(context)
            else:
                logits, _ = self(context)
            for step in range(max_new_tokens):
                next_token = self._sample_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                )
                token_ids = torch.cat((token_ids, next_token), dim=1)
                yield token_ids
                if step + 1 >= min_new_tokens and torch.all(next_token == eos_id):
                    break
                if use_cache:
                    assert cache is not None
                    cached_length = cache[0][0].size(-2)
                    if cached_length < self.config.block_size:
                        logits, cache = self.forward_with_cache(next_token, cache)
                    else:
                        context = token_ids[:, -self.config.block_size :]
                        logits, cache = self.forward_with_cache(context)
                else:
                    context = token_ids[:, -self.config.block_size :]
                    logits, _ = self(context)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        eos_id: int,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        min_new_tokens: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        generated = token_ids
        for generated in self.generate_steps(
            token_ids,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            min_new_tokens=min_new_tokens,
            use_cache=use_cache,
        ):
            pass
        return generated

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
