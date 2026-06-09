#!/usr/bin/env python3
"""Synthetic Data Generation CLI for MominoMoE Kernel Fault Diagnosis Training.

Generates SFT and DPO training pairs using:
  - Built-in template-based generation (offline, no API needed)
  - Teacher model API via OpenRouter or HuggingFace Inference API

Output: JSONL files with {"prompt": ..., "response": ..., "metadata": {...}} for SFT,
or {"prompt": ..., "chosen": ..., "rejected": ..., "metadata": {...}} for DPO.
"""

import argparse
import json
import random
import time
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.templates import FAULT_TEMPLATES, get_templates_by_fault_type, list_fault_types
from data.prompt_templates import (
    build_dpo_pair_from_templates,
    build_sft_prompt,
    OPENROUTER_MODELS,
    HUGGINGFACE_MODELS,
    build_api_payload,
    build_chat_messages,
    SYSTEM_PROMPT_SFT,
    SYSTEM_PROMPT_DPO,
)
from data.quality_filter import QualityFilter


def generate_sft_from_templates(
    templates: List[Dict],
    rng: random.Random,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Generate SFT samples using built-in template text.
    
    Uses the template_sft → template_dpo_chosen path from templates.py.
    Returns list of {"prompt": str, "response": str, "metadata": dict}.
    """
    samples = []
    for t in templates:
        prompt = t["template_sft"]
        response = t["template_dpo_chosen"]
        samples.append({
            "prompt": prompt,
            "response": response,
            "metadata": {
                "id": t["id"],
                "fault_type": t["fault_type"],
                "difficulty": t["difficulty"],
                "symptom": t["symptom"],
                "source": "template",
            }
        })
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def generate_dpo_from_templates(
    templates: List[Dict],
    rng: random.Random,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Generate DPO pairs using built-in template text."""
    samples = []
    for t in templates:
        pair = build_dpo_pair_from_templates(t)
        samples.append({
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "metadata": pair["metadata"],
        })
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def generate_via_api(
    templates: List[Dict],
    api_type: str,
    model_tier: str,
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    output_mode: str = "sft",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> List[Dict]:
    """Generate samples by calling an external teacher model API.
    
    Args:
        templates: List of template dicts
        api_type: "openrouter" or "huggingface"
        model_tier: "fast", "quality", or "economy"
        api_key: API key (reads from env if None)
        max_samples: Max samples to generate
        output_mode: "sft" or "dpo"
        temperature: Generation temperature
        max_tokens: Max tokens in response
    """
    import requests

    if api_type == "openrouter":
        models = OPENROUTER_MODELS
        base_url = "https://openrouter.ai/api/v1/chat/completions"
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mominos.ai",
        }
    elif api_type == "huggingface":
        models = HUGGINGFACE_MODELS
        base_url = "https://api-inference.huggingface.co/models"
        api_key = api_key or os.environ.get("HF_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        raise ValueError(f"Unknown API type: {api_type}")

    model_id = models.get(model_tier, models["economy"])
    
    if output_mode == "sft":
        system_prompt = SYSTEM_PROMPT_SFT
    else:
        system_prompt = SYSTEM_PROMPT_DPO

    samples = []
    selected_templates = templates[:max_samples] if max_samples else templates

    for t in selected_templates:
        if output_mode == "sft":
            user_prompt = build_sft_prompt(t)
        else:
            from data.prompt_templates import build_dpo_prompt
            user_prompt = build_dpo_prompt(t)

        messages = build_chat_messages(system_prompt, user_prompt)
        payload = build_api_payload(
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        print(f"  [{t['id']}] Calling {api_type}/{model_tier}...", flush=True)
        
        try:
            if api_type == "openrouter":
                resp = requests.post(
                    base_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
            elif api_type == "huggingface":
                api_url = f"{base_url}/{model_id}/v1/chat/completions"
                resp = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
            
            samples.append({
                "prompt": user_prompt,
                "response": content,
                "metadata": {
                    "id": t["id"],
                    "fault_type": t["fault_type"],
                    "difficulty": t["difficulty"],
                    "source": f"{api_type}_{model_tier}",
                    "model": model_id,
                }
            })
            print(f"    ✓ Got {len(content)} chars", flush=True)
            
            # Rate limiting
            time.sleep(1.0)
            
        except Exception as e:
            print(f"    ✗ API error: {e}", flush=True)
            # Fall back to template
            samples.append({
                "prompt": user_prompt,
                "response": t["template_dpo_chosen"] if output_mode == "sft" else t["template_dpo_chosen"],
                "metadata": {
                    "id": t["id"],
                    "fault_type": t["fault_type"],
                    "difficulty": t["difficulty"],
                    "source": "template_fallback",
                    "error": str(e),
                }
            })

    return samples


def write_jsonl(samples: List[Dict], output_path: str):
    """Write samples to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(samples)} samples to {output_path}")


def filter_and_dedup(
    samples: List[Dict],
    quality_filter: QualityFilter,
) -> List[Dict]:
    """Apply quality filter and deduplicate samples."""
    seen_prompts = set()
    filtered = []
    
    for s in samples:
        prompt = s.get("prompt", "")
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        
        quality = quality_filter.evaluate(s)
        if quality.get("pass", False):
            s["quality"] = quality
            filtered.append(s)
        else:
            reasons = quality.get("fail_reasons", [])
            print(f"  Filtered out: {s.get('metadata', {}).get('id', '?')}: {', '.join(reasons)}")
    
    return filtered


def augment_samples(
    samples: List[Dict],
    rng: random.Random,
    n_augmented: int = 0,
) -> List[Dict]:
    """Create augmented variants by rephrasing or adding noise to templates.
    Simple implementation: word substitution variants.
    """
    if n_augmented <= 0:
        return samples
    
    augmented = list(samples)
    for _ in range(min(n_augmented, len(samples) * 2)):
        base = rng.choice(samples)
        prompt = base["prompt"]
        # Simple augmentation: add/remove a sentence
        aug_prompt = prompt + " Consider the system logs and kernel version."
        augmented.append({
            "prompt": aug_prompt,
            "response": base["response"],
            "metadata": {**base["metadata"], "source": base["metadata"].get("source", "template") + "_augmented"},
        })
    
    return augmented


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic kernel fault diagnosis training data"
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo", "both"],
        default="sft",
        help="Output format (default: sft)",
    )
    parser.add_argument(
        "--source",
        choices=["template", "api", "both"],
        default="template",
        help="Generation source (default: template = offline)",
    )
    parser.add_argument(
        "--api-type",
        choices=["openrouter", "huggingface"],
        default=None,
        help="Teacher model API type",
    )
    parser.add_argument(
        "--model-tier",
        choices=["fast", "quality", "economy"],
        default="economy",
        help="Teacher model tier (default: economy)",
    )
    parser.add_argument(
        "--fault-types",
        nargs="+",
        default=None,
        help="Filter by fault type(s). Default: all",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Filter by difficulty (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to generate (default: all matching)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/MominOS/training/data/generated",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="Number of augmented variants to add (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and exit without generating",
    )
    
    args = parser.parse_args()
    rng = random.Random(args.seed)
    
    # Filter templates
    templates = FAULT_TEMPLATES
    if args.fault_types:
        filtered = []
        for ft in args.fault_types:
            filtered.extend(get_templates_by_fault_type(ft))
        templates = filtered
    if args.difficulty != "all":
        templates = [t for t in templates if t["difficulty"] == args.difficulty]
    
    if not templates:
        print("No templates match the filter criteria.")
        return
    
    print(f"Using {len(templates)} templates:")
    for t in templates:
        print(f"  {t['id']}: [{t['fault_type']}] {t['symptom'][:60]}...")
    
    if args.dry_run:
        print(f"\nWould generate {min(len(templates), args.max_samples or len(templates))} SFT samples")
        if args.mode in ("dpo", "both"):
            print(f"Would generate {min(len(templates), args.max_samples or len(templates))} DPO pairs")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    quality_filter = QualityFilter()
    
    # ── Generate ─────────────────────────────────────────────────────────
    if args.source in ("template", "both"):
        if args.mode in ("sft", "both"):
            print("\n=== Generating SFT from templates ===")
            sft_samples = generate_sft_from_templates(templates, rng, args.max_samples)
            sft_samples = filter_and_dedup(sft_samples, quality_filter)
            sft_samples = augment_samples(sft_samples, rng, args.augment)
            write_jsonl(sft_samples, os.path.join(args.output_dir, "sft_template.jsonl"))
        
        if args.mode in ("dpo", "both"):
            print("\n=== Generating DPO from templates ===")
            dpo_samples = generate_dpo_from_templates(templates, rng, args.max_samples)
            dpo_samples = filter_and_dedup(dpo_samples, quality_filter)
            write_jsonl(dpo_samples, os.path.join(args.output_dir, "dpo_template.jsonl"))
    
    if args.source in ("api", "both") and args.api_type:
        if args.mode in ("sft", "both"):
            print(f"\n=== Generating SFT via {args.api_type}/{args.model_tier} ===")
            api_sft = generate_via_api(
                templates, args.api_type, args.model_tier,
                max_samples=args.max_samples,
                output_mode="sft",
            )
            api_sft = filter_and_dedup(api_sft, quality_filter)
            write_jsonl(api_sft, os.path.join(args.output_dir, f"sft_{args.api_type}_{args.model_tier}.jsonl"))
        
        if args.mode in ("dpo", "both"):
            print(f"\n=== Generating DPO via {args.api_type}/{args.model_tier} ===")
            api_dpo = generate_via_api(
                templates, args.api_type, args.model_tier,
                max_samples=args.max_samples,
                output_mode="dpo",
            )
            api_dpo = filter_and_dedup(api_dpo, quality_filter)
            write_jsonl(api_dpo, os.path.join(args.output_dir, f"dpo_{args.api_type}_{args.model_tier}.jsonl"))
    
    print("\nDone!")


if __name__ == "__main__":
    main()