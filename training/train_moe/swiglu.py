"""SwiGLU Activation FFN — standard for Llama-style transformers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU FFN: SwiGLU(x) = silu(x @ W_gate) * (x @ W_up) @ W_down
    Compared to ReLU FFN, SwiGLU has 3 weight matrices instead of 2,
    but consistently outperforms on perplexity (Shazeer, 2020).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        gate_out = F.silu(self.gate(x))
        up_out = self.up(x)
        hidden = gate_out * up_out
        return self.dropout(self.down(hidden))