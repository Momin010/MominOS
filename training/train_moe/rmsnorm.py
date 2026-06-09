"""RMSNorm — used as the only normalization in MominoMoE (pre-norm, Llama-style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Unlike LayerNorm, RMSNorm does not subtract the mean — only scales by RMS.
    γ * (x / sqrt(mean(x²) + eps))
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim) float tensor
        Returns:
            (..., dim) normalized tensor
        """
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x.float() / rms
        return (x_normed * self.weight).to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"