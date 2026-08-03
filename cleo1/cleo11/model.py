from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from .config import Cleo11ModelConfig


type KVCache = tuple[torch.Tensor, torch.Tensor]
type PastKeyValues = tuple[KVCache, ...]


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        inverse_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_freq", inverse_freq, persistent=False)
        self._build_cache(max_position)

    def _build_cache(self, max_position: int) -> None:
        positions = torch.arange(max_position, dtype=self.inverse_freq.dtype)
        freqs = torch.outer(positions, self.inverse_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, sequence_length: int, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        end = offset + sequence_length
        if end > self.cos_cached.size(-2):
            raise ValueError(
                f"RoPE position {end} exceeds configured max {self.cos_cached.size(-2)}"
            )
        return (
            self.cos_cached[:, :, offset:end, :],
            self.sin_cached[:, :, offset:end, :],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Cleo11ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_rep = config.n_head // config.n_kv_head
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _repeat_kv(self, values: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return values
        batch, n_kv, seq, dim = values.shape
        values = values[:, :, None, :, :].expand(batch, n_kv, self.n_rep, seq, dim)
        return values.reshape(batch, n_kv * self.n_rep, seq, dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch_size, sequence_length, _ = x.shape
        query = self.q_proj(x).view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, sequence_length, self.n_kv_head, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, sequence_length, self.n_kv_head, self.head_dim).transpose(1, 2)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.size(-2)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value)
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        total_length = past_length + sequence_length
        causal = torch.ones(
            sequence_length,
            total_length,
            dtype=torch.bool,
            device=x.device,
        ).tril(diagonal=past_length)
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        output = self.resid_dropout(self.o_proj(attended))
        if use_cache:
            return output, present
        return output


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: Cleo11ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_size, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: Cleo11ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attention(self.attn_norm(x), cos, sin)
        assert isinstance(attended, torch.Tensor)
        x = x + attended
        return x + self.mlp(self.ffn_norm(x))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        attended = self.attention(
            self.attn_norm(x),
            cos,
            sin,
            past_key_value=past_key_value,
            use_cache=True,
        )
        assert isinstance(attended, tuple)
        attention_output, present = attended
        x = x + attention_output
        return x + self.mlp(self.ffn_norm(x)), present


class Cleo11Transformer(nn.Module):
    def __init__(self, config: Cleo11ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(config.head_dim, config.block_size, config.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block size {self.config.block_size}"
            )
        cos, sin = self.rotary(sequence_length, offset=0)
        x = self.embedding_dropout(self.token_embedding(token_ids))
        for block in self.blocks:
            x = block(x, cos, sin)
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
        total_length = past_length + sequence_length
        if total_length > self.config.block_size:
            raise ValueError(
                f"cached sequence length {total_length} exceeds block size "
                f"{self.config.block_size}"
            )
        cos, sin = self.rotary(sequence_length, offset=past_length)
        x = self.embedding_dropout(self.token_embedding(token_ids))
        present_key_values: list[KVCache] = []
        for block, layer_cache in zip(self.blocks, layer_caches, strict=True):
            x, present = block.forward_with_cache(x, cos, sin, layer_cache)
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
        if token_ids.ndim != 2 or token_ids.size(1) < 1:
            raise ValueError("token_ids must have shape (batch, sequence) with at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
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

    def parameter_breakdown(self) -> dict[str, int]:
        embedding = self.token_embedding.weight.numel()
        tied = int(self.config.tie_embeddings)
        lm_head = 0 if tied else self.lm_head.weight.numel()
        blocks = sum(parameter.numel() for parameter in self.blocks.parameters())
        norms = self.final_norm.weight.numel()
        return {
            "embeddings": embedding,
            "transformer_blocks": blocks,
            "final_norm": norms,
            "lm_head": lm_head,
            "total": self.parameter_count(),
            "tied_embeddings": tied,
        }
