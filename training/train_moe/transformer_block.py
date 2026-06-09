"""Transformer Block — Pre-norm (Llama-style): RMSNorm → Attention → residual → RMSNorm → MoE → residual.

Each block contains:
1. Pre-RMSNorm + GQA Attention (with RoPE)
2. Residual connection
3. Pre-RMSNorm + MoE layer (8 experts top-2 + shared expert)
4. Residual connection
"""
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import Attention
from .moe_layer import MoELayer


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm, attention, and MoE FFN."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm for attention
        self.input_norm = RMSNorm(config.d_model, eps=config.rms_eps)

        # GQA Attention
        self.attention = Attention(config)

        # Pre-norm for MoE
        self.post_attention_norm = RMSNorm(config.d_model, eps=config.rms_eps)

        # MoE Layer (8 experts top-2 + shared expert)
        self.moe = MoELayer(config)

    def forward(self, x, start_pos=0, kv_cache=None):
        """Forward pass with residual connections (pre-norm).

        Args:
            x: (batch, seq_len, d_model)
            start_pos: starting position for RoPE
            kv_cache: optional KV cache for autoregressive generation
        Returns:
            hidden: (batch, seq_len, d_model)
            aux_losses: dict of auxiliary losses from MoE
        """
        # Pre-norm attention with residual
        residual = x
        x = self.input_norm(x)
        x = self.attention(x, start_pos=start_pos, kv_cache=kv_cache)
        x = residual + x

        # Pre-norm MoE with residual
        residual = x
        x = self.post_attention_norm(x)
        x, aux_losses = self.moe(x)
        x = residual + x

        return x, aux_losses