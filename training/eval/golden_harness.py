#!/usr/bin/env python3
"""Golden Intermediate Validation Harness for MominoMoE.

Runs a fixed input through the model, captures every intermediate tensor
via PyTorch forward hooks, and dumps all tensors to a .npz file.

The .npz file is the "golden reference" for cross-validation with the
C inference engine. The C engine team runs the same input and produces
a directory of named .bin/.npy dumps; compare_logits.py compares them.

Usage:
    python golden_harness.py \
        --checkpoint /path/to/model.pt \
        --output golden_output.npz \
        --seed 42 \
        --batch 1 \
        --seq_len 64
"""

import argparse
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_moe import MominoMoE, MominoMoEConfig


def build_model(config: MominoMoEConfig) -> MominoMoE:
    """Instantiate MominoMoE model with given config."""
    model = MominoMoE(config)
    return model


def load_checkpoint(model: MominoMoE, checkpoint_path: str, device: torch.device):
    """Load state dict into model."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle various checkpoint formats
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    elif "module" in state:
        state = state["module"]
    model.load_state_dict(state, strict=False)
    print(f"[harness] Loaded checkpoint: {checkpoint_path}")


def run_golden_harness(
    model: MominoMoE,
    input_ids: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run forward pass with hooks to capture all intermediate tensors.

    Returns a dict mapping tensor names -> numpy arrays.
    """
    captured = {}

    # ── Hook factory ──────────────────────────────────────────────────────
    def make_tensor_hook(name, layer_idx=None):
        """Create a forward hook that captures a tensor output.

        Handles:
        - Plain tensor outputs (most submodules)
        - Tuple outputs where first element is the main tensor and
          second element is an aux_losses dict (MoELayer, TransformerBlock)
        """
        def hook(module, inputs, output):
            key = f"l{layer_idx}_{name}" if layer_idx is not None else name
            if isinstance(output, tuple):
                # e.g. MoELayer returns (hidden_states, aux_losses_dict)
                hidden, aux = output
                if isinstance(hidden, torch.Tensor):
                    captured[key] = hidden.detach().cpu().numpy()
                # Store auxiliary losses
                if isinstance(aux, dict):
                    for ak, av in aux.items():
                        if isinstance(av, torch.Tensor):
                            arr = av.detach().cpu().numpy()
                            if arr.ndim == 0:
                                arr = np.array([arr.item()])
                            captured[f"{key}_aux_{ak}"] = arr
                        elif isinstance(av, (np.floating, float)):
                            captured[f"{key}_aux_{ak}"] = np.array([av])
                        elif av is not None:
                            captured[f"{key}_aux_{ak}"] = np.array(av)
            elif isinstance(output, torch.Tensor):
                captured[key] = output.detach().cpu().numpy()
        return hook

    def make_input_hook(name, layer_idx=None):
        """Capture the INPUT to a module (for residual paths)."""
        def hook(module, inputs):
            key = f"l{layer_idx}_{name}_input" if layer_idx is not None else f"{name}_input"
            inp = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            if isinstance(inp, torch.Tensor):
                captured[key] = inp.detach().cpu().numpy()
        return hook

    # ── Register hooks ────────────────────────────────────────────────────
    model.eval()

    # Embedding
    model.embed.register_forward_hook(make_tensor_hook("embed"))

    # Final norm and LM head
    model.norm.register_forward_hook(make_tensor_hook("final_norm"))
    model.lm_head.register_forward_hook(make_tensor_hook("logits"))

    # Per transformer block
    for i, layer in enumerate(model.layers):
        # Pre-computation: capture input to block (residual start)
        layer.register_forward_pre_hook(make_input_hook("block_input", i))

        # Pre-attention norm
        layer.input_norm.register_forward_hook(make_tensor_hook("pre_attn_norm_out", i))

        # Attention submodules: Q, K, V projections
        layer.attention.q_proj.register_forward_hook(make_tensor_hook("attn_q", i))
        layer.attention.k_proj.register_forward_hook(make_tensor_hook("attn_k", i))
        layer.attention.v_proj.register_forward_hook(make_tensor_hook("attn_v", i))

        # Attention output (after o_proj)
        layer.attention.register_forward_hook(make_tensor_hook("attn_out", i))

        # Pre-MoE norm
        layer.post_attention_norm.register_forward_hook(make_tensor_hook("pre_moe_norm_out", i))

        # MoE router logits
        layer.moe.router.register_forward_hook(make_tensor_hook("moe_router_logits", i))

        # MoE shared expert output
        layer.moe.shared_expert.register_forward_hook(make_tensor_hook("moe_shared_out", i))

        # MoE individual experts
        for j in range(len(layer.moe.experts)):
            layer.moe.experts[j].register_forward_hook(
                make_tensor_hook(f"moe_expert_{j}_out", i)
            )

        # MoE combined output + aux losses
        layer.moe.register_forward_hook(make_tensor_hook("moe_out", i))

        # Block output (after residual + moe)
        layer.register_forward_hook(make_tensor_hook("block_out", i))

    # ── Run forward pass ──────────────────────────────────────────────────
    with torch.no_grad():
        result = model(input_ids)

    # Also capture final logits from result dict
    captured["final_logits"] = result["logits"].detach().cpu().numpy()

    # Capture auxiliary losses from model output
    if "aux_losses" in result and result["aux_losses"]:
        for k, v in result["aux_losses"].items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy()
                if arr.ndim == 0:
                    arr = np.array([arr.item()])
                captured[f"total_aux_{k}"] = arr

    if "moe_metrics" in result and result["moe_metrics"]:
        captured["moe_metrics"] = np.array(
            [str(m) for m in result["moe_metrics"]], dtype=object
        )

    # Store input tokens for reference
    captured["input_ids"] = input_ids.cpu().numpy()

    return captured


def main():
    parser = argparse.ArgumentParser(
        description="MominoMoE Golden Intermediate Validation Harness"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). If not provided, uses random weights."
    )
    parser.add_argument(
        "--output", type=str, default="golden_output.npz",
        help="Output .npz file path (default: golden_output.npz)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[harness] Device: {device}")

    # Build model
    config = MominoMoEConfig()
    model = build_model(config)
    model.to(device)
    print(f"[harness] Model: MominoMoE-1.2B ({sum(p.numel() for p in model.parameters()):,} params)")

    # Load checkpoint if provided
    if args.checkpoint and os.path.isfile(args.checkpoint):
        load_checkpoint(model, args.checkpoint, device)

    # Generate fixed random input
    input_ids = torch.randint(
        0, config.vocab_size,
        (args.batch, args.seq_len),
        device=device,
    )
    print(f"[harness] Input shape: {input_ids.shape}")

    # Run harness
    print("[harness] Running forward pass with hooks...")
    captured = run_golden_harness(model, input_ids, device)

    # Save to .npz
    out_path = args.output
    np.savez_compressed(out_path, **captured)
    print(f"[harness] Saved {len(captured)} tensors to {out_path}")

    # Summary of captured tensors
    total_bytes = sum(arr.nbytes if isinstance(arr, np.ndarray) else 0 for arr in captured.values())
    print(f"[harness] Total data: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"[harness] Tensor names:")
    for name in sorted(captured.keys()):
        arr = captured[name]
        if isinstance(arr, np.ndarray):
            print(f"    {name}: {arr.shape} [{arr.dtype}]")


if __name__ == "__main__":
    main()