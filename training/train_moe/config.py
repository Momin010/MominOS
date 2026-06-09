"""MominoMoE Configuration — single source of truth for all hyperparameters."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class MominoMoEConfig:
    # Architecture
    d_model: int = 1024
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: int = 4          # GQA: 4 KV heads (ratio 4:1)
    head_dim: int = 64
    d_ff: int = 2048             # per-expert FFN hidden dim
    n_experts: int = 8
    top_k: int = 2
    n_shared_experts: int = 1   # DeepSeek-style shared expert
    vocab_size: int = 32000
    context_len: int = 2048

    # Norm & activation
    rms_eps: float = 1e-5
    norm_first: bool = True      # pre-norm (Llama-style)
    activation: str = "swiglu"

    # RoPE
    rope_base: float = 10000.0
    rope_scaling: Optional[dict] = None

    # MoE losses
    moe_aux_loss_coeff: float = 0.01   # load balancing
    moe_z_loss_coeff: float = 0.001    # Z-loss for router stability

    # Embedding
    tie_embeddings: bool = True        # tied input/output embedding

    # Weight init
    init_std: float = 0.02
    router_init_std: float = 0.01     # smaller init for router (DeepSeek practice)

    # Dropout (0 = off)
    dropout: float = 0.0
    attention_dropout: float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads {self.n_heads} must be divisible by n_kv_heads {self.n_kv_heads}"
        assert self.head_dim * self.n_heads == self.d_model, \
            f"head_dim* n_heads ({self.head_dim}*{self.n_heads}) must equal d_model {self.d_model}"
        assert self.top_k <= self.n_experts, \
            f"top_k {self.top_k} must be <= n_experts {self.n_experts}"

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def total_params_estimate(self) -> dict:
        """Rough parameter count estimate (in millions)."""
        emb = self.vocab_size * self.d_model / 1e6
        embed_shared = emb if self.tie_embeddings else 2 * emb
        attn_per_layer = (self.d_model * self.d_model * 2 +      # Q, O
                          self.d_model * self.kv_dim * 2) / 1e6  # K, V
        moe_per_layer = (
            self.n_experts * self.d_model * self.d_ff * 3 +      # experts gate/up/down
            self.n_shared_experts * self.d_model * self.d_ff * 3 +  # shared expert
            self.d_model * self.n_experts                         # router
        ) / 1e6
        layer_total = (attn_per_layer + moe_per_layer) * self.n_layers
        total = embed_shared + layer_total
        active = (attn_per_layer + (self.top_k * self.d_model * self.d_ff * 3 +
                  self.n_shared_experts * self.d_model * self.d_ff * 3) / 1e6) * self.n_layers
        return {
            "total_m": round(total, 1),
            "active_m": round(active, 1),
            "embedding_m": round(emb, 1),
            "q8_0_gb": round(total * 1.0 / 1e3, 2),  # ~1 byte per weight in Q8_0
            "kv_cache_mb": round(
                self.context_len * self.n_kv_heads * self.head_dim * self.n_layers * 2 * 2 / 1e6, 1
            ),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MominoMoEConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})