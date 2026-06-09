"""Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE).

Implements GQA where n_kv_heads < n_heads. Keys/Values are computed for fewer heads
and repeated (via expand) to match the query head count for attention.

KV cache support for autoregressive generation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(dim: int, max_seq_len: int, base: float = 10000.0,
                          device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Precompute sin/cos frequencies for RoPE.

    Args:
        dim: head_dim (must be even)
        max_seq_len: maximum sequence length
        base: RoPE base frequency (default 10000.0)
    Returns:
        (max_seq_len, dim/2, 2) tensor of [cos, sin] pairs
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
    # Return stacked [cos, sin] along last dim
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (max_seq_len, dim/2, 2)


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor, pos: int = 0) -> torch.Tensor:
    """Apply rotary embeddings to x.

    Args:
        x: (batch, n_heads, seq_len, head_dim)
        cos_sin: (max_seq_len, head_dim//2, 2) from precompute_rope_freqs
        pos: starting position offset
    Returns:
        Rotated x of same shape
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    seq_len = x.shape[2]

    cos = cos_sin[pos:pos + seq_len, :, 0]  # (seq_len, half)
    sin = cos_sin[pos:pos + seq_len, :, 1]  # (seq_len, half)

    x_float = x.float()
    x1 = x_float[..., :half]      # even indices
    x2 = x_float[..., half:]      # odd indices

    # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    cos = cos.view(1, 1, seq_len, half)
    sin = sin.view(1, 1, seq_len, half)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)


class Attention(nn.Module):
    """Grouped-Query Attention with RoPE, causal mask, and optional KV cache."""

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base
        self.context_len = config.context_len
        self.attn_dropout = config.attention_dropout

        self.n_groups = self.n_heads // self.n_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # Precomputed RoPE frequencies
        self.register_buffer(
            "rope_cos_sin",
            precompute_rope_freqs(self.head_dim, self.context_len, self.rope_base),
            persistent=False,
        )

        # Causal mask (cached)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.context_len, self.context_len, dtype=torch.bool))
            .view(1, 1, self.context_len, self.context_len),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: list = None,
    ) -> torch.Tensor:
        """Forward pass with optional KV cache.

        Args:
            x: (batch, seq_len, d_model)
            start_pos: starting position for RoPE (for incremental decoding)
            kv_cache: optional [k_cache, v_cache] tensors for autoregressive generation.
                      Each is (batch, n_kv_heads, max_seq_len, head_dim)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)

        # Reshape to multi-head format
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos_sin, start_pos)
        k = apply_rope(k, self.rope_cos_sin, start_pos)

        # KV cache update (autoregressive generation)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Store new key/value states into cache
            k_cache[:, :, start_pos:start_pos + seq_len] = k
            v_cache[:, :, start_pos:start_pos + seq_len] = v
            # Use the full cached sequence for attention
            k = k_cache[:, :, :start_pos + seq_len]
            v = v_cache[:, :, :start_pos + seq_len]

        # GQA: expand KV heads to match Q heads
        if self.n_groups > 1:
            # (batch, n_kv_heads, seq_len, head_dim) -> (batch, n_heads, seq_len, head_dim)
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, n_heads, seq_len, kv_seq_len)

        # Apply causal mask
        kv_len = k.shape[2]
        if kv_len <= self.context_len:
            mask = self.causal_mask[:, :, :seq_len, :kv_len]
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        if self.attn_dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout)

        # Weighted sum over values
        out = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)

    def extra_repr(self) -> str:
        return (f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
                f"head_dim={self.head_dim}, groups={self.n_groups}")