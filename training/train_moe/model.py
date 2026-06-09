"""MominoMoE — Full MoE Transformer Model for MominOS.

Architecture:
- Tied input/output embeddings (shared weight)
- 20 pre-norm transformer blocks with GQA attention + MoE (8 experts top-2 + shared expert)
- Final RMSNorm
- LM head (weight-tied with embedding)

Total params: ~1.22B
Active params per token: ~430M
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MominoMoEConfig
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class MominoMoE(nn.Module):
    """MominoMoE — Complete MoE language model for kernel fault diagnosis."""

    def __init__(self, config: MominoMoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.context_len = config.context_len
        self.tie_embeddings = config.tie_embeddings

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # Gradient checkpointing
        self._gradient_checkpointing_enabled = False

        # Final norm
        self.norm = RMSNorm(config.d_model, eps=config.rms_eps)

        # LM Head (output projection)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if self.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all parameters with normal distribution."""
        # Default init for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Embedding init
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.config.init_std)

        # Router init (smaller std per DeepSeek practice)
        for name, param in self.named_parameters():
            if "router.weight" in name:
                nn.init.normal_(param, mean=0.0, std=self.config.router_init_std)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory during training."""
        self._gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing_enabled = False

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: list = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            start_pos: starting position for RoPE cache
            kv_caches: list of [k_cache, v_cache] per layer for autoregressive generation
            labels: optional (batch, seq_len) target token indices for loss computation
        Returns:
            dict with keys:
                'logits': (batch, seq_len, vocab_size)
                'loss': optional cross-entropy loss
                'aux_losses': dict of auxiliary losses
                'perplexity': optional perplexity
                'moe_metrics': dict of MoE utilization metrics
        """
        batch, seq_len = input_ids.shape

        # Token embeddings
        h = self.embed(input_ids) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)

        # Pass through transformer blocks
        total_aux_losses = {}
        all_moe_metrics = []

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            if self._gradient_checkpointing_enabled and self.training and start_pos == 0 and kv_cache is None:
                h, aux_losses = torch.utils.checkpoint.checkpoint(
                    layer, h, start_pos, kv_cache, use_reentrant=False
                )
            else:
                h, aux_losses = layer(h, start_pos=start_pos, kv_cache=kv_cache)

            # Accumulate auxiliary losses
            for key, val in aux_losses.items():
                if isinstance(val, torch.Tensor) and val.requires_grad:
                    total_aux_losses[key] = total_aux_losses.get(key, 0.0) + val
                elif isinstance(val, (int, float)):
                    pass  # Metrics tracked separately

            # Track MoE metrics
            all_moe_metrics.append({
                k: v for k, v in aux_losses.items()
                if k in ("min_expert_load", "max_expert_load", "router_prob_entropy")
            })

        # Final norm
        h = self.norm(h)

        # LM head
        logits = self.lm_head(h)  # (batch, seq_len, vocab_size)

        result = {
            "logits": logits,
            "aux_losses": total_aux_losses,
            "moe_metrics": all_moe_metrics,
        }

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add auxiliary losses
            aux_loss_sum = sum(
                v for k, v in total_aux_losses.items()
                if isinstance(v, torch.Tensor) and v.requires_grad
            )
            total_loss = loss + aux_loss_sum
            result["loss"] = total_loss
            result["ce_loss"] = loss
            result["perplexity"] = torch.exp(loss)

        return result

    def configure_optimizers(self, weight_decay: float = 0.1, learning_rate: float = 3e-4):
        """Configure AdamW optimizer with parameter-specific weight decay.

        No weight decay for norms and biases.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def summary(self) -> str:
        """Print a detailed model summary with parameter counts, active params, memory."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        embed_params = sum(p.numel() for name, p in self.named_parameters() if "embed" in name)
        attn_params = sum(p.numel() for name, p in self.named_parameters() if "attention" in name)
        moe_params = sum(p.numel() for name, p in self.named_parameters() if "moe." in name or "moe." in name)
        norm_params = sum(p.numel() for name, p in self.named_parameters() if "norm" in name)

        # Active params per token: attention + top-k experts + shared expert
        n_layers = self.config.n_layers
        top_k = self.config.top_k
        d_model = self.config.d_model
        d_ff = self.config.d_ff
        n_heads = self.config.n_heads
        n_kv_heads = self.config.n_kv_heads
        head_dim = self.config.head_dim

        attn_active = (
            d_model * d_model  # Q
            + d_model * n_kv_heads * head_dim * 2  # K, V
            + d_model * d_model  # O
        ) * n_layers

        moe_active = (
            d_model * d_ff * 3 * top_k  # top-k experts
            + d_model * d_ff * 3  # shared expert
            + d_model * self.config.n_experts  # router
        ) * n_layers

        total_active = attn_active + moe_active

        lines = [
            "=" * 60,
            "MominoMoE-1.2B Model Summary",
            "=" * 60,
            f"  Total parameters:     {total_params:,} ({total_params/1e6:.1f}M)",
            f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)",
            f"  Active per token:     {total_active:,} ({total_active/1e6:.1f}M)",
            "",
            "  Component breakdown:",
            f"    Embedding:  {embed_params:,} ({embed_params/1e6:.1f}M)",
            f"    Attention:  {attn_params:,} ({attn_params/1e6:.1f}M)",
            f"    MoE:        {moe_params:,} ({moe_params/1e6:.1f}M)",
            f"    Norms:      {norm_params:,} ({norm_params/1e6:.1f}M)",
            "",
            f"  Config: {self.config.n_layers} layers, {self.config.n_heads} heads, "
            f"{self.config.n_kv_heads} KV heads",
            f"  MoE: {self.config.n_experts} experts, top-{self.config.top_k} + "
            f"{self.config.n_shared_experts} shared",
            f"  Tied embeddings: {self.config.tie_embeddings}",
            "",
            f"  Memory estimates:",
            f"    fp32:  {total_params * 4 / 1e9:.2f} GB",
            f"    fp16:  {total_params * 2 / 1e9:.2f} GB",
            f"    Q8_0:  {total_params * 1 / 1e9:.2f} GB",
            f"  KV cache (fp16): {self.context_len * n_kv_heads * head_dim * n_layers * 2 * 2 / 1e6:.1f} MB",
            "=" * 60,
        ]
        return "\n".join(lines)

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if not only_trainable or p.requires_grad
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Simple autoregressive generation.

        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: max tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            eos_token_id: stop token
        Returns:
            (batch, prompt_len + generated_len) output tokens
        """
        batch, prompt_len = input_ids.shape
        device = input_ids.device

        # Initialize KV cache (use model weight dtype, NOT input_ids.dtype which is Long)
        model_dtype = next(self.parameters()).dtype
        kv_caches = []
        for layer in self.layers:
            k_cache = torch.zeros(
                batch, self.config.n_kv_heads, self.context_len, self.config.head_dim,
                device=device, dtype=model_dtype
            )
            v_cache = torch.zeros_like(k_cache)
            kv_caches.append([k_cache, v_cache])

        # Prefill
        self.eval()
        output = self(input_ids, start_pos=0, kv_caches=kv_caches)
        next_logits = output["logits"][:, -1, :]  # (batch, vocab_size)

        generated = input_ids.clone()

        for pos in range(prompt_len, prompt_len + max_new_tokens):
            # Sample next token
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_k is not None:
                    # Top-k filtering
                    top_k_vals, _ = torch.topk(probs, top_k, dim=-1)
                    probs[probs < top_k_vals[:, -1:]] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                if top_p is not None:
                    # Nucleus filtering
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum = sorted_probs.cumsum(dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, 1)
                    next_token = sorted_indices.gather(-1, next_token)
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if (next_token == eos_token_id).any():
                break

            # Single-token forward with KV cache
            output = self(next_token, start_pos=pos, kv_caches=kv_caches)
            next_logits = output["logits"][:, -1, :]

        return generated