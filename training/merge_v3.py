#!/usr/bin/env python3
"""
Merge the MominoMoE-v3 LoRA adapter into the base Qwen3-0.6B weights
and save the result as a regular HuggingFace model for full fine-tuning.
Output: training/model/v3_merged_hf/
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    default="training/model/qwen3-0.6b")
    parser.add_argument("--adapter", default="Momin-Aldahdouh/MominoMoE-v3")
    parser.add_argument("--output",  default="training/model/v3_merged_hf")
    args = parser.parse_args()

    print(f"Loading base model from {args.base}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"Applying v3 LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    size_mb = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
    ) // 1024 // 1024
    print(f"Done. Merged model saved ({size_mb} MB).")

if __name__ == "__main__":
    main()
