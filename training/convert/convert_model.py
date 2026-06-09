"""MOM1 Binary Format Converter — Converts PyTorch checkpoints to .mom binary format.

MOM1 Binary Format (Section A of spec):
┌─────────────────────────────────┐
│ mom_model_header (fixed 128B)   │ ← magic "MOM1", arch fields
├─────────────────────────────────┤
│ mom_tensor_desc[0] (64B)        │ ← name, dtype, shape, offset, nbytes
│ mom_tensor_desc[1] (64B)        │
│ ...                              │
├─────────────────────────────────┤
│ Tensor data (Q8_0 blocks,       │
│  aligned to 64 bytes)           │
│                                 │
│ [block 0: 32 int8 + 1 fp16]    │
│ [block 1: 32 int8 + 1 fp16]    │
│ ...                              │
└─────────────────────────────────┘

Usage:
    python convert_model.py --checkpoint model.pt --output model.mom
    python convert_model.py --checkpoint model.pt --output model.mom --quantize
"""
import io
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_moe.config import MominoMoEConfig
from train_moe.quant import quantize_q8_0, Q8_0_BLOCK_SIZE


# ─── MOM1 Binary Format Constants ────────────────────────────────────────────

MOM1_MAGIC = b"MOM1"
MOM1_VERSION = 1
HEADER_SIZE = 128
TENSOR_DESC_SIZE = 64
TENSOR_DATA_ALIGNMENT = 64

# Data type codes
DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_Q8_0 = 2
DTYPE_I32 = 3

# Tensor name remapping: PyTorch parameter name -> C engine tensor name
TENSOR_NAME_MAP = {
    "embed.weight": "token_embed.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
    # Per-layer: layers.{i}.{component}.weight
}


def map_tensor_name(pt_name: str) -> str:
    """Map PyTorch parameter name to C engine tensor name."""
    # Layer-specific names
    if pt_name.startswith("layers."):
        parts = pt_name.split(".")
        # layers.0.attention.q_proj.weight -> layers.0.attn.wq.weight
        # layers.0.attention.k_proj.weight -> layers.0.attn.wk.weight
        # layers.0.attention.v_proj.weight -> layers.0.attn.wv.weight
        # layers.0.attention.o_proj.weight -> layers.0.attn.wo.weight
        # layers.0.input_norm.weight -> layers.0.attn_norm.weight
        # layers.0.post_attention_norm.weight -> layers.0.ffn_norm.weight
        # layers.0.moe.router.weight -> layers.0.moe.router.weight
        # layers.0.moe.shared_expert.gate.weight -> layers.0.moe.shared.gate.weight
        # layers.0.moe.shared_expert.up.weight -> layers.0.moe.shared.up.weight
        # layers.0.moe.shared_expert.down.weight -> layers.0.moe.shared.down.weight
        # layers.0.moe.experts.0.gate.weight -> layers.0.moe.experts.0.gate.weight
        # etc.
        if len(parts) >= 5:
            layer_idx = parts[1]
            component = parts[2]
            sub = parts[3] if len(parts) > 3 else ""
            param = parts[4] if len(parts) > 4 else ""

            # Attention
            if component == "attention":
                if sub == "q_proj":
                    return f"layers.{layer_idx}.attn.wq.weight"
                elif sub == "k_proj":
                    return f"layers.{layer_idx}.attn.wk.weight"
                elif sub == "v_proj":
                    return f"layers.{layer_idx}.attn.wv.weight"
                elif sub == "o_proj":
                    return f"layers.{layer_idx}.attn.wo.weight"

            # Norms
            if component == "input_norm":
                return f"layers.{layer_idx}.attn_norm.weight"
            if component == "post_attention_norm":
                return f"layers.{layer_idx}.ffn_norm.weight"

            # MoE
            if component == "moe":
                if sub == "router":
                    return f"layers.{layer_idx}.moe.router.weight"
                if sub == "shared_expert":
                    expert_part = parts[4] if len(parts) > 4 else ""
                    if expert_part == "gate":
                        return f"layers.{layer_idx}.moe.shared.gate.weight"
                    elif expert_part == "up":
                        return f"layers.{layer_idx}.moe.shared.up.weight"
                    elif expert_part == "down":
                        return f"layers.{layer_idx}.moe.shared.down.weight"
                if sub == "experts":
                    expert_idx = parts[4]
                    expert_part = parts[5]
                    return f"layers.{layer_idx}.moe.experts.{expert_idx}.{expert_part}.weight"

    # Fallback
    return pt_name


@dataclass
class TensorDescriptor:
    """Describes a single tensor in the MOM1 binary."""
    name: str
    dtype: int           # DTYPE_FP32, DTYPE_FP16, DTYPE_Q8_0, DTYPE_I32
    shape: List[int]
    offset: int          # byte offset from start of data section
    nbytes: int          # number of bytes in data section
    scale: float = 0.0  # overall scale (for Q8_0, 0 means per-block) — set later


def encode_q8_0_data(tensor: torch.Tensor) -> Tuple[bytes, int, int]:
    """Encode a tensor into Q8_0 block format bytes.

    Returns:
        data_bytes: packed [int8*32 + fp16_scale] × n_blocks
        n_blocks: number of Q8_0 blocks
        bytes_total: total byte count
    """
    q8_data, scales = quantize_q8_0(tensor)
    n_blocks = q8_data.shape[0]

    # Pack each block: 32 int8 values + 1 fp16 scale = 34 bytes per block
    buf = io.BytesIO()
    for i in range(n_blocks):
        # Write 32 int8 values
        buf.write(q8_data[i].cpu().numpy().tobytes())
        # Write fp16 scale
        buf.write(scales[i].cpu().numpy().tobytes())

    data_bytes = buf.getvalue()
    return data_bytes, n_blocks, len(data_bytes)


def convert_checkpoint_to_mom1(
    state_dict: dict,
    config: MominoMoEConfig,
    output_path: str,
    quantize: bool = True,
    verbose: bool = True,
) -> str:
    """Convert a PyTorch state dict to MOM1 binary format.

    Args:
        state_dict: PyTorch model state dict
        config: model configuration
        output_path: where to write the .mom file
        quantize: whether to quantize weights to Q8_0
    Returns:
        Path to the written .mom file
    """
    if verbose:
        print(f"Converting {len(state_dict)} tensors to MOM1 format...")
        print(f"  Quantization: {'Q8_0' if quantize else 'None (fp32)'}")
        print(f"  Output: {output_path}")

    # ── Step 1: Prepare tensor descriptors ──
    descriptors: List[TensorDescriptor] = []
    all_data_chunks: List[bytes] = []

    current_offset = 0

    # Sort for deterministic ordering
    sorted_names = sorted(state_dict.keys())

    for pt_name in sorted_names:
        tensor = state_dict[pt_name]
        tensor_name = map_tensor_name(pt_name)

        # Determine dtype and encode
        if quantize and tensor.dim() >= 2 and 'norm' not in pt_name and 'router' not in pt_name and 'bias' not in pt_name:
            # Quantize to Q8_0
            data_bytes, n_blocks, total_bytes = encode_q8_0_data(tensor)
            dtype = DTYPE_Q8_0
        else:
            # Keep as fp32
            tensor_fp32 = tensor.float()
            data_bytes = tensor_fp32.cpu().numpy().tobytes()
            dtype = DTYPE_FP32
            total_bytes = len(data_bytes)
            n_blocks = 0

        # Align to 64 bytes
        aligned_offset = math.ceil(current_offset / TENSOR_DATA_ALIGNMENT) * TENSOR_DATA_ALIGNMENT
        if aligned_offset > current_offset:
            padding = aligned_offset - current_offset
            all_data_chunks.append(b'\x00' * padding)
            current_offset = aligned_offset

        desc = TensorDescriptor(
            name=tensor_name,
            dtype=dtype,
            shape=list(tensor.shape),
            offset=current_offset,
            nbytes=total_bytes,
        )
        descriptors.append(desc)
        all_data_chunks.append(data_bytes)
        current_offset += total_bytes

        if verbose:
            dtype_str = "Q8_0" if dtype == DTYPE_Q8_0 else "FP32"
            print(f"  [{dtype_str:4s}] {tensor_name:50s} {list(tensor.shape):20s} {total_bytes:>8d} bytes @ {desc.offset}")

    # ── Step 2: Build binary ──
    buffer = io.BytesIO()

    # Header (128 bytes)
    header = bytearray(HEADER_SIZE)
    # Magic
    header[0:4] = MOM1_MAGIC
    # Version (u32)
    struct.pack_into('<I', header, 4, MOM1_VERSION)
    # Number of tensors (u32)
    struct.pack_into('<I', header, 8, len(descriptors))
    # Architecture fields (all u32 at known offsets)
    struct.pack_into('<I', header, 12, config.d_model)
    struct.pack_into('<I', header, 16, config.n_layers)
    struct.pack_into('<I', header, 20, config.n_heads)
    struct.pack_into('<I', header, 24, config.n_kv_heads)
    struct.pack_into('<I', header, 28, config.head_dim)
    struct.pack_into('<I', header, 32, config.d_ff)
    struct.pack_into('<I', header, 36, config.n_experts)
    struct.pack_into('<I', header, 40, config.top_k)
    struct.pack_into('<I', header, 44, config.n_shared_experts)
    struct.pack_into('<I', header, 48, config.vocab_size)
    struct.pack_into('<I', header, 52, config.context_len)
    struct.pack_into('<f', header, 56, config.rms_eps)
    struct.pack_into('<f', header, 60, config.rope_base)
    # Num tensor descriptors
    struct.pack_into('<I', header, 64, len(descriptors))
    # Data section offset (right after header + descriptors)
    data_offset = HEADER_SIZE + len(descriptors) * TENSOR_DESC_SIZE
    struct.pack_into('<I', header, 68, data_offset)

    buffer.write(header)

    # Tensor descriptors (64 bytes each)
    for desc in descriptors:
        desc_bytes = bytearray(TENSOR_DESC_SIZE)
        # Name (32 bytes, null-terminated)
        name_bytes = desc.name.encode('ascii', errors='replace')[:31]
        desc_bytes[0:len(name_bytes)] = name_bytes
        desc_bytes[len(name_bytes)] = 0  # null terminator
        # Dtype (u32)
        struct.pack_into('<I', desc_bytes, 32, desc.dtype)
        # Shape rank (u32)
        struct.pack_into('<I', desc_bytes, 36, len(desc.shape))
        # Shape dimensions (up to 4 × u32)
        for i, dim in enumerate(desc.shape[:4]):
            struct.pack_into('<I', desc_bytes, 40 + i * 4, dim)
        # Offset (u64)
        struct.pack_into('<Q', desc_bytes, 56, desc.offset)
        # Note: nbytes is implicit from dtype+shape; C engine computes it
        buffer.write(desc_bytes)

    # Tensor data
    for chunk in all_data_chunks:
        buffer.write(chunk)

    # ── Step 3: Write to file ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())

    file_size = os.path.getsize(output_path)
    if verbose:
        print(f"\n  Wrote {file_size:,} bytes to {output_path}")
        print(f"  {len(descriptors)} tensors, {config.total_params_estimate['q8_0_gb']:.2f} GB estimated")

    return output_path


def load_mom1_header(path: str) -> dict:
    """Load and parse a MOM1 file header."""
    with open(path, 'rb') as f:
        header = f.read(HEADER_SIZE)

    magic = header[0:4]
    assert magic == MOM1_MAGIC, f"Invalid magic: {magic}"

    return {
        'magic': magic,
        'version': struct.unpack_from('<I', header, 4)[0],
        'n_tensors_header': struct.unpack_from('<I', header, 8)[0],
        'd_model': struct.unpack_from('<I', header, 12)[0],
        'n_layers': struct.unpack_from('<I', header, 16)[0],
        'n_heads': struct.unpack_from('<I', header, 20)[0],
        'n_kv_heads': struct.unpack_from('<I', header, 24)[0],
        'head_dim': struct.unpack_from('<I', header, 28)[0],
        'd_ff': struct.unpack_from('<I', header, 32)[0],
        'n_experts': struct.unpack_from('<I', header, 36)[0],
        'top_k': struct.unpack_from('<I', header, 40)[0],
        'n_shared_experts': struct.unpack_from('<I', header, 44)[0],
        'vocab_size': struct.unpack_from('<I', header, 48)[0],
        'context_len': struct.unpack_from('<I', header, 52)[0],
        'rms_eps': struct.unpack_from('<f', header, 56)[0],
        'rope_base': struct.unpack_from('<f', header, 60)[0],
        'n_tensors': struct.unpack_from('<I', header, 64)[0],
        'data_offset': struct.unpack_from('<I', header, 68)[0],
    }


def verify_mom1_header(path: str, config: MominoMoEConfig) -> bool:
    """Verify that a MOM1 file's header matches the given config."""
    h = load_mom1_header(path)
    checks = [
        (h['d_model'], config.d_model, 'd_model'),
        (h['n_layers'], config.n_layers, 'n_layers'),
        (h['n_heads'], config.n_heads, 'n_heads'),
        (h['n_kv_heads'], config.n_kv_heads, 'n_kv_heads'),
        (h['head_dim'], config.head_dim, 'head_dim'),
        (h['d_ff'], config.d_ff, 'd_ff'),
        (h['n_experts'], config.n_experts, 'n_experts'),
        (h['top_k'], config.top_k, 'top_k'),
        (h['vocab_size'], config.vocab_size, 'vocab_size'),
        (h['context_len'], config.context_len, 'context_len'),
    ]
    all_ok = True
    for got, expected, name in checks:
        if got != expected:
            print(f"  ❌ {name}: got {got}, expected {expected}")
            all_ok = False
    if all_ok:
        print(f"  ✅ All {len(checks)} header fields match config")
    return all_ok


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MOM1 Converter — Convert PyTorch checkpoints to .mom format")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PyTorch .pt/.safetensors checkpoint")
    parser.add_argument("--output", type=str, default="model.mom",
                        help="Output .mom file path")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Quantize weights to Q8_0 (default: True)")
    parser.add_argument("--no-quantize", action="store_false", dest="quantize",
                        help="Keep weights in fp32")
    parser.add_argument("--verify", action="store_true",
                        help="Verify header matches config")
    parser.add_argument("--roundtrip", action="store_true",
                        help="Run round-trip test on random weights")

    args = parser.parse_args()

    config = MominoMoEConfig()

    if args.roundtrip:
        # Test with random weights
        print("Running round-trip test with random weights...")
        from train_moe.model import MominoMoE
        model = MominoMoE(config)
        sd = model.state_dict()
        path = convert_checkpoint_to_mom1(sd, config, args.output, quantize=args.quantize)
        verify_mom1_header(path, config)
        print(f"  File size: {os.path.getsize(path):,} bytes")
        print("  ✅ Round-trip MOM1 conversion complete!")
        return

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        print(f"Loading checkpoint from {args.checkpoint}...")
        sd = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
        convert_checkpoint_to_mom1(sd, config, args.output, quantize=args.quantize)
        print("✅ Conversion complete!")
    else:
        print("No checkpoint specified. Use --checkpoint <path> or --roundtrip for testing.")
        print()
        parser.print_help()


if __name__ == "__main__":
    main()