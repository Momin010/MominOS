"""Q8_0 Block Quantization — llama.cpp-compatible format.

Quantizes fp32/fp16 tensors to blocks of 32 int8 values + 1 fp16 scale per block.

Format per block (34 bytes):
  [0..31]: int8 quantized values (32 bytes)
  [32..33]: float16 scale (2 bytes)

Usage:
    q8_data, scales = quantize_q8_0(tensor)
    recovered = dequantize_q8_0(q8_data, scales, original_shape)
"""
import math
import struct
from typing import Tuple

import torch


Q8_0_BLOCK_SIZE = 32


def quantize_q8_0(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to Q8_0 block format.

    Args:
        tensor: (..., n) float tensor. Last dimension is quantized in blocks of 32.
                Must be contiguous.
    Returns:
        q8_data: int8 tensor with shape (n_blocks, block_size) — the quantized weights
        scales: fp16 tensor with shape (n_blocks,) — scale per block
    """
    orig_dtype = tensor.dtype
    tensor_f32 = tensor.contiguous().float()

    # Flatten everything except last dim to get blocks
    *leading_dims, last_dim = tensor_f32.shape
    flat = tensor_f32.view(-1, last_dim)  # (rows, last_dim)

    # Pad last dimension to multiple of block_size
    n_blocks_per_row = math.ceil(last_dim / Q8_0_BLOCK_SIZE)
    padded_dim = n_blocks_per_row * Q8_0_BLOCK_SIZE

    if padded_dim > last_dim:
        pad = torch.zeros(flat.shape[0], padded_dim - last_dim, dtype=torch.float32)
        flat_padded = torch.cat([flat, pad], dim=-1)
    else:
        flat_padded = flat

    n_rows = flat_padded.shape[0]
    total_blocks = n_rows * n_blocks_per_row

    # Reshape to blocks of 32
    blocks_flat = flat_padded.view(total_blocks, Q8_0_BLOCK_SIZE)  # (total_blocks, 32)

    # Per-block: find max absolute value, compute scale
    abs_max, _ = blocks_flat.abs().max(dim=-1, keepdim=True)  # (total_blocks, 1)
    abs_max = abs_max.clamp(min=1e-10)
    scale = abs_max / 127.0  # (total_blocks, 1)

    # Quantize: round(w / scale), clamp to [-127, 127]
    q8 = torch.round(blocks_flat / scale).clamp(-127, 127).to(torch.int8)

    # Scale as fp16
    scale_fp16 = scale.squeeze(-1).half()

    return q8, scale_fp16


def dequantize_q8_0(
    q8_data: torch.Tensor,
    scales: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """Dequantize Q8_0 block data back to float32.

    Args:
        q8_data: int8 tensor (total_blocks, block_size) — quantized values
        scales: fp16 tensor (total_blocks,) — scale per block
        original_shape: original tensor shape before quantization
    Returns:
        fp32 tensor with the original shape
    """
    total_blocks, block_size = q8_data.shape
    assert block_size == Q8_0_BLOCK_SIZE

    # Dequantize: w * scale
    scale_f32 = scales.float().unsqueeze(-1)  # (total_blocks, 1)
    recovered_flat = q8_data.float() * scale_f32  # (total_blocks, 32)

    # Reconstruct padded flat and trim padding
    *leading_dims, last_dim = original_shape
    n_blocks_per_row = math.ceil(last_dim / Q8_0_BLOCK_SIZE)

    recovered = recovered_flat.view(-1, n_blocks_per_row * Q8_0_BLOCK_SIZE)
    recovered = recovered[:, :last_dim]  # trim padding

    return recovered.view(*leading_dims, last_dim)


def quantize_linear_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Quantize a linear layer weight matrix to Q8_0.

    Args:
        weight: (out_dim, in_dim) float tensor
    Returns:
        q8_data: int8 quantized data
        scales: fp16 scales
        original_shape: for dequantization
    """
    return quantize_q8_0(weight)


def quantize_embedding_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Quantize an embedding weight matrix to Q8_0.

    Args:
        weight: (vocab_size, d_model) float tensor
    Returns:
        q8_data, scales, original_shape
    """
    return quantize_q8_0(weight)


def quantize_model_to_q8(model: torch.nn.Module) -> dict:
    """Quantize all eligible weights in a model to Q8_0 format.

    Quantizes: all Linear weights, Embedding weights
    Skips: norms (RMSNorm weights), biases, router weights (kept fp32 for stability)

    Returns:
        dict mapping tensor_name -> {
            'q8_data': int8 tensor,
            'scales': fp16 tensor,
            'shape': original shape,
            'dtype': 'q8_0' or 'fp32'
        }
    """
    quantized = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip norms and biases — keep as fp32
        if 'norm' in name or 'bias' in name:
            quantized[name] = {
                'data': param.data.float(),
                'shape': param.shape,
                'dtype': 'fp32',
            }
            continue

        # Skip router — keep as fp32 for inference stability
        if 'router' in name:
            quantized[name] = {
                'data': param.data.float(),
                'shape': param.shape,
                'dtype': 'fp32',
            }
            continue

        # Quantize linear weights and embeddings
        if param.dim() >= 2:
            q8_data, scales = quantize_q8_0(param.data)
            quantized[name] = {
                'q8_data': q8_data,
                'scales': scales,
                'shape': param.shape,
                'dtype': 'q8_0',
            }
        else:
            # Keep 1D params as fp32
            quantized[name] = {
                'data': param.data.float(),
                'shape': param.shape,
                'dtype': 'fp32',
            }

    return quantized


def round_trip_test(tensor: torch.Tensor) -> dict:
    """Test Q8_0 round-trip: quantize -> dequantize and measure error.

    Args:
        tensor: input float tensor
    Returns:
        dict with keys: mse, max_error, mean_error, tensor_shape, n_blocks
    """
    q8_data, scales = quantize_q8_0(tensor)
    recovered = dequantize_q8_0(q8_data, scales, tensor.shape)

    error = (tensor.float() - recovered).abs()
    return {
        'mse': (error ** 2).mean().item(),
        'max_error': error.max().item(),
        'mean_error': error.mean().item(),
        'tensor_shape': list(tensor.shape),
        'n_blocks': q8_data.shape[0],
        'original_dtype': str(tensor.dtype),
    }


def round_trip_linear_test(out_dim: int = 1024, in_dim: int = 1024) -> dict:
    """Run Q8_0 round-trip test on a random linear weight matrix."""
    torch.manual_seed(42)
    w = torch.randn(out_dim, in_dim) * 0.02
    result = round_trip_test(w)
    print(f"Q8_0 Round-Trip Test: {out_dim}x{in_dim} linear weight")
    print(f"  MSE:        {result['mse']:.8f}")
    print(f"  Max error:  {result['max_error']:.6f}")
    print(f"  Mean error: {result['mean_error']:.6f}")
    print(f"  Blocks:     {result['n_blocks']}")
    return result


# Simple CLI for testing
if __name__ == "__main__":
    round_trip_linear_test()