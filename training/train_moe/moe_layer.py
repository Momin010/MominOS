"""MoE Layer — Router + Top-2 Dispatch + Shared Expert + Load Balancing Loss.

Architecture (DeepSeek-inspired):
- 8 routed experts (top-2 activated per token)
- 1 shared expert (always active, captures common knowledge)
- Router with load balancing loss + Z-loss for stability

Reference: DeepSeek-MoE (Dai et al., 2024), Mixtral (Jiang et al., 2024)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swiglu import SwiGLU


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with top-2 routing + shared expert.

    For each token:
    1. Shared expert processes it (always active)
    2. Router outputs logits → softmax → top-2 selection → renormalize
    3. Selected experts process routed tokens
    4. Combine outputs weighted by router probabilities

    Load balancing loss encourages uniform expert utilization.
    Z-loss stabilizes router logits from growing too large.
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.z_loss_coeff = config.moe_z_loss_coeff
        self.dropout = config.dropout

        # Router: learned linear projection to expert logits
        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)

        # Shared expert: always active, captures common knowledge
        self.shared_expert = SwiGLU(self.d_model, self.d_ff, dropout=config.dropout)

        # 8 routed experts
        self.experts = nn.ModuleList([
            SwiGLU(self.d_model, self.d_ff, dropout=config.dropout)
            for _ in range(self.n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass for MoE layer.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_losses: dict with 'load_balancing_loss', 'z_loss', 'num_expert_switches'
        """
        batch, seq_len, d_model = x.shape
        num_tokens = batch * seq_len
        device = x.device
        dtype = x.dtype

        # 1. Shared expert (always active)
        shared_out = self.shared_expert(x)

        # 2. Router logits and routing probabilities
        # Reshape to tokens for per-token routing
        x_flat = x.view(num_tokens, d_model)  # (num_tokens, d_model)
        router_logits = self.router(x_flat)    # (num_tokens, n_experts)
        router_probs = F.softmax(router_logits.float(), dim=-1)  # fp32 for stability

        # Z-loss: encourage router logits to stay small
        z_loss = self.z_loss_coeff * (torch.logsumexp(router_logits.float(), dim=-1) ** 2).mean()

        # 3. Top-k routing
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # (num_tokens, top_k) each

        # Renormalize top-k probabilities
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # 4. Expert dispatch — process tokens assigned to each expert
        expert_outputs = torch.zeros_like(x_flat)

        for expert_idx in range(self.n_experts):
            # Find tokens that have this expert in their top-k
            mask = (topk_indices == expert_idx)  # (num_tokens, top_k)
            token_mask = mask.any(dim=-1)        # (num_tokens,) bool — routed to this expert
            if not token_mask.any():
                continue

            # Get the assigned tokens
            expert_tokens = x_flat[token_mask]  # (n_assigned, d_model)

            # Run through this expert
            expert_out = self.experts[expert_idx](expert_tokens)  # (n_assigned, d_model)

            # Get the routing probabilities for this expert
            # For each token assigned to this expert, find which top-k position it was
            token_expert_probs = torch.zeros(
                token_mask.sum(), device=device, dtype=topk_probs.dtype
            )
            # Find positions in topk_indices matching this expert
            for k in range(self.top_k):
                pos_mask = (topk_indices[:, k] == expert_idx) & token_mask
                token_expert_probs[pos_mask[token_mask]] = topk_probs[pos_mask, k].to(
                    token_expert_probs.dtype
                )

            # Weight output by router probability
            weighted_out = expert_out * token_expert_probs.unsqueeze(-1).to(dtype)
            # Accumulate into output tensor
            expert_outputs[token_mask] += weighted_out.to(expert_outputs.dtype)

        # 5. Reshape back to (batch, seq_len, d_model)
        routed_out = expert_outputs.view(batch, seq_len, d_model)

        # 6. Load balancing loss
        # f_i: fraction of tokens routed to expert i
        tokens_per_expert = torch.zeros(self.n_experts, device=device, dtype=router_probs.dtype)
        for k in range(self.top_k):
            tokens_per_expert.scatter_add_(
                0, topk_indices[:, k], torch.ones(num_tokens, device=device, dtype=router_probs.dtype)
            )
        f_i = tokens_per_expert / num_tokens  # (n_experts,)

        # P_i: average router probability for expert i
        P_i = router_probs.mean(dim=0)  # (n_experts,)

        # Load balancing loss = n_experts * sum(f_i * P_i)
        load_balancing_loss = self.aux_loss_coeff * self.n_experts * (f_i * P_i).sum()

        # 7. Combine shared + routed outputs
        output = shared_out + routed_out

        aux_losses = {
            "load_balancing_loss": load_balancing_loss,
            "z_loss": z_loss,
            "router_prob_entropy": -(router_probs * torch.log(router_probs.clamp(min=1e-10))).sum(dim=-1).mean(),
            "min_expert_load": f_i.min().item(),
            "max_expert_load": f_i.max().item(),
        }

        return output, aux_losses