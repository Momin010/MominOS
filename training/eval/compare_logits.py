#!/usr/bin/env python3
"""Compare golden .npz intermediates against C engine dumps.

Loads a golden .npz file (produced by golden_harness.py) and compares every
named tensor against corresponding .npy/.bin files in a C engine dump directory.

Reports:
- MSE per tensor
- Max absolute error per tensor
- First divergence point (index where |a-b| > threshold)

Usage:
    python compare_logits.py \
        --golden golden_output.npz \
        --c_engine_dir /path/to/c_dumps/ \
        --threshold 1e-2 \
        --tolerance 5e-2 \
        --verbose
"""

import argparse
import os
import glob
import sys
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
#  Naming convention mapping
# ---------------------------------------------------------------------------
# The golden harness produces keys like:
#   l0_pre_attn_norm_out, l0_attn_q, l0_attn_k, l0_attn_v,
#   l0_attn_out, l0_pre_moe_norm_out, l0_moe_router_logits,
#   l0_moe_shared_out, l0_moe_expert_0_out, l0_moe_out, l0_block_out,
#   embed, final_norm, logits, final_logits
#
# The C engine is expected to produce .npy files with corresponding names
# or a configurable prefix/suffix.

def load_golden_npz(path: str) -> dict:
    """Load golden .npz into dict of numpy arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data}


def load_c_engine_dumps(directory: str) -> dict:
    """Load all .npy or .bin files from a directory into a name->array dict.

    Filename conventions (attempted in order):
      1. {name}.npy
      2. {name}.bin  (loaded as raw float16 or float32)
      3. tensor_{name}.npy
      4. layer_{idx}_{rest}.npy
    """
    result = {}
    if not os.path.isdir(directory):
        print(f"[compare] WARNING: C engine dump directory not found: {directory}")
        return result

    files = os.listdir(directory)
    for fname in sorted(files):
        path = os.path.join(directory, fname)
        if not os.path.isfile(path):
            continue

        # Strip extension to get base name
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()

        if ext == '.npy':
            arr = np.load(path)
            # Infer key name from stem
            key = stem
            # Normalize: strip prefixes like 'tensor_', 'layer_0_', etc.
            key = re.sub(r'^tensor_', '', key)
            result[key] = arr
        elif ext == '.bin':
            # Try to load as float32, then fall back to float16
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size == 0:
                raw = np.fromfile(path, dtype=np.float16)
            result[stem] = raw

    return result


def resolve_tensor(golden_key: str, c_tensors: dict) -> tuple:
    """Find the matching C engine tensor for a given golden key.

    Tries multiple naming conventions:
    1. Exact match
    2. golden_key without 'l0_' prefix
    3. 'tensor_' + golden_key
    4. Various renames:
       - 'logits' -> 'lm_head' or 'final_logits'
       - 'final_norm' -> 'norm'
       - 'embed' -> 'embedding' or 'token_embed'
       - '{layer}_attn_q' -> '{layer}_attention_q_proj'
       - etc.

    Returns (c_key, c_array) or (None, None).
    """
    # Direct match
    if golden_key in c_tensors:
        return golden_key, c_tensors[golden_key]

    # Alternative names
    alt_names = [
        golden_key.replace('attn_q', 'attn_q_proj'),
        golden_key.replace('attn_k', 'attn_k_proj'),
        golden_key.replace('attn_v', 'attn_v_proj'),
        golden_key.replace('attn_out', 'attn_o_proj'),
        golden_key.replace('moe_router_logits', 'moe_router'),
        golden_key.replace('moe_shared_out', 'moe_shared'),
        golden_key.replace('block_out', 'block'),
        golden_key.replace('moe_out', 'moe'),
        f"tensor_{golden_key}",
    ]

    # Strip layer prefix for layer-agnostic matching
    m = re.match(r'l(\d+)_(.*)', golden_key)
    if m:
        layer_idx = m.group(1)
        rest = m.group(2)
        alt_names.extend([
            rest,
            f"layer{layer_idx}_{rest}",
            f"layer_{layer_idx}_{rest}",
        ])

    for alt in alt_names:
        if alt in c_tensors:
            return alt, c_tensors[alt]

    return None, None


def compare_tensors(
    golden: np.ndarray,
    c_arr: np.ndarray,
    name: str,
    threshold: float = 1e-2,
    tolerance: float = 5e-2,
    verbose: bool = False,
) -> dict:
    """Compare two tensors and return metrics.

    Args:
        golden: golden reference array
        c_arr: C engine output array
        name: tensor name for display
        threshold: per-element error threshold for flagging divergence
        tolerance: allowable fraction of divergent elements
        verbose: print details
    Returns:
        dict with keys: mse, max_abs_error, mean_abs_error,
                        n_divergent, total_elements, first_divergence_index,
                        passed (bool)
    """
    # Handle shape mismatches — try to reshape or broadcast
    if golden.shape != c_arr.shape:
        # Try flattening both
        if golden.size == c_arr.size:
            golden_f = golden.flatten()
            c_f = c_arr.flatten()
            if verbose:
                print(f"  [{name}] Shape mismatch {golden.shape} vs {c_arr.shape}, "
                      f"but same size — comparing flattened")
        else:
            # Try transpose
            if golden.shape == c_arr.T.shape and golden.ndim == 2:
                c_arr = c_arr.T
                golden_f = golden.flatten()
                c_f = c_arr.flatten()
                if verbose:
                    print(f"  [{name}] Shape mismatch, using transpose")
            else:
                return {
                    "name": name, "mse": float('inf'),
                    "max_abs_error": float('inf'),
                    "mean_abs_error": float('inf'),
                    "n_divergent": golden.size,
                    "total_elements": golden.size,
                    "first_divergence_index": 0,
                    "golden_shape": list(golden.shape),
                    "c_shape": list(c_arr.shape),
                    "passed": False,
                    "error": f"Shape mismatch: golden={golden.shape}, c={c_arr.shape}"
                }
    else:
        golden_f = golden.flatten()
        c_f = c_arr.flatten()

    # Compute metrics
    diff = np.abs(golden_f.astype(np.float64) - c_f.astype(np.float64))
    mse = np.mean(diff ** 2)
    max_err = float(diff.max())
    mean_err = float(diff.mean())

    divergent = diff > threshold
    n_div = int(divergent.sum())
    total = len(diff)
    div_ratio = n_div / total if total > 0 else 0.0

    first_div = int(divergent.argmax()) if divergent.any() else -1

    passed = div_ratio <= tolerance or mse < threshold

    result = {
        "name": name,
        "mse": float(mse),
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "n_divergent": n_div,
        "total_elements": total,
        "divergent_ratio": div_ratio,
        "first_divergence_index": first_div,
        "golden_shape": list(golden.shape),
        "c_shape": list(c_arr.shape),
        "passed": passed,
    }

    if verbose:
        status = "PASS" if passed else "FAIL"
        shape_str = f" {result['golden_shape']} ->" if result['golden_shape'] != list(c_arr.shape) else ""
        print(f"  {status:4s} | {name:40s} | MSE={mse:.2e} | MaxErr={max_err:.2e} | "
              f"Div={n_div:>5d}/{total} ({div_ratio*100:.1f}%){shape_str}")
        if not passed and first_div >= 0:
            print(f"        First divergence at index {first_div} "
                  f"(golden={golden_f[first_div]:.4f}, c={c_f[first_div]:.4f})")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare golden .npz against C engine dumps"
    )
    parser.add_argument(
        "--golden", type=str, default="golden_output.npz",
        help="Path to golden .npz file"
    )
    parser.add_argument(
        "--c_engine_dir", type=str, default="",
        help="Directory containing C engine .npy/.bin dumps"
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-2,
        help="Per-element absolute error threshold (default: 1e-2)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=5e-2,
        help="Allowable fraction of divergent elements (default: 5e-2 = 5%%)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-tensor details"
    )
    parser.add_argument(
        "--summary", action="store_true", default=True,
        help="Print summary only"
    )
    args = parser.parse_args()

    # ── Load golden ───────────────────────────────────────────────────────
    if not os.path.isfile(args.golden):
        print(f"[compare] ERROR: Golden file not found: {args.golden}")
        sys.exit(1)

    print(f"[compare] Loading golden: {args.golden}")
    golden = load_golden_npz(args.golden)
    print(f"[compare]   {len(golden)} tensors loaded")

    # ── Load C engine dumps ──────────────────────────────────────────────
    c_tensors = {}
    if args.c_engine_dir:
        print(f"[compare] Loading C engine dumps: {args.c_engine_dir}")
        c_tensors = load_c_engine_dumps(args.c_engine_dir)
        print(f"[compare]   {len(c_tensors)} tensors loaded")

        if not c_tensors:
            print("[compare] WARNING: No C engine tensors loaded. "
                  "Run with --c_engine_dir pointing to valid dump directory.")

    # ── Determine which keys to compare ───────────────────────────────────
    # Skip metadata keys (strings, scalars, input_ids)
    skip_patterns = [
        'input_ids', 'moe_metrics', 'total_aux_',
        '_aux_',  # auxiliary loss keys inside per-layer
    ]

    if c_tensors:
        # Match golden keys to C keys
        compare_keys = []
        for gk in sorted(golden.keys()):
            if any(p in gk for p in skip_patterns):
                continue
            if not isinstance(golden[gk], np.ndarray):
                continue
            c_key, c_arr = resolve_tensor(gk, c_tensors)
            if c_key is not None:
                compare_keys.append((gk, c_key, c_arr))
            else:
                if args.verbose:
                    print(f"[compare] No match for golden key: {gk}")
    else:
        # No C engine data — just show golden summary
        print("\n[compare] === Golden Tensor Summary (no C engine data to compare) ===\n")
        for name in sorted(golden.keys()):
            arr = golden[name]
            if isinstance(arr, np.ndarray) and arr.ndim > 0:
                print(f"  {name:50s} shape={list(arr.shape):20s} dtype={str(arr.dtype):10s} "
                      f"range=[{arr.min():.4f}, {arr.max():.4f}]")
        print(f"\n[compare] {len(golden)} tensors in golden file.")
        return

    # ── Compare each tensor ───────────────────────────────────────────────
    print(f"\n[compare] === Comparing {len(compare_keys)} tensors ===\n")
    print(f"  Threshold: |a-b| > {args.threshold} is divergent")
    print(f"  Tolerance: ≤ {args.tolerance*100:.0f}% divergent allowed\n")

    results = []
    for gk, ck, c_arr in compare_keys:
        result = compare_tensors(
            golden[gk], c_arr,
            name=f"{gk} -> {ck}",
            threshold=args.threshold,
            tolerance=args.tolerance,
            verbose=args.verbose,
        )
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = sum(1 for r in results if not r["passed"])

    print(f"\n[compare] {'='*60}")
    print(f"[compare]   TOTAL: {len(results)} tensors compared")
    print(f"[compare]   PASS:  {passed_count}")
    print(f"[compare]   FAIL:  {failed_count}")

    if failed_count > 0:
        print(f"\n[compare] FAILED TENSORS:")
        for r in results:
            if not r["passed"]:
                error_str = r.get("error", f"MSE={r['mse']:.2e}, MaxErr={r['max_abs_error']:.2e}")
                print(f"  ✗ {r['name']}: {error_str}")

    # Overall score
    mse_list = [r["mse"] for r in results if r["mse"] != float('inf')]
    avg_mse = np.mean(mse_list) if mse_list else 0.0
    max_max_err = max(r["max_abs_error"] for r in results if r["max_abs_error"] != float('inf'))

    print(f"\n[compare]   Average MSE:     {avg_mse:.6e}")
    print(f"[compare]   Global max error: {max_max_err:.6e}")
    print(f"[compare]   {'PASS' if failed_count == 0 else 'FAIL'}")
    print(f"[compare] {'='*60}")

    return int(failed_count > 0)


if __name__ == "__main__":
    sys.exit(main())