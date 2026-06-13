#!/usr/bin/env python3
"""
MominoMoE-v4 full fine-tuning.
Starts from merged v3 weights (all 596M parameters trainable).
No LoRA — real weight updates throughout the entire model.
"""
import argparse, json, os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

SYSTEM_PROMPT = (
    "You are MominOS, a kernel fault diagnostician and OS assistant embedded in an "
    "x86-64 operating system. You diagnose kernel faults, issue tool calls as clean "
    "JSON, generate shell commands and scripts, write and debug code, and answer "
    "system administration questions. Be concise and direct. Never narrate your "
    "reasoning — just respond."
)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def format_sample(sample, tokenizer):
    """Format prompt/response pair as a Qwen3 ChatML string."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sample["prompt"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="training/model/v3_merged_hf",
                        help="Starting weights (merged v3 or any HF model dir)")
    parser.add_argument("--data",       default="training/data/v4_train.jsonl")
    parser.add_argument("--val-data",   default="training/data/v4_val.jsonl")
    parser.add_argument("--output",     default="training/model/v4_full")
    parser.add_argument("--steps",      type=int,   default=10000)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch",      type=int,   default=2)
    parser.add_argument("--grad-accum", type=int,   default=16)
    parser.add_argument("--max-len",    type=int,   default=512)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing to trade compute for memory.
    # Required for full fine-tuning of even a 0.6B model.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # needed when using gradient checkpointing

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    print("Loading data...")
    train_raw = load_jsonl(args.data)
    val_raw   = load_jsonl(args.val_data)
    print(f"  Train: {len(train_raw):,}   Val: {len(val_raw):,}")

    train_texts = [format_sample(s, tokenizer) for s in train_raw]
    val_texts   = [format_sample(s, tokenizer) for s in val_raw]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds   = Dataset.from_dict({"text": val_texts})

    effective_batch = args.batch * args.grad_accum
    print(f"Effective batch size: {effective_batch}  ({args.batch} x {args.grad_accum} accum)")
    print(f"Steps: {args.steps}  LR: {args.lr}  Max seq len: {args.max_len}")

    config = SFTConfig(
        output_dir=args.output,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        fp16=True,
        bf16=False,
        max_seq_length=args.max_len,
        dataset_text_field="text",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        report_to="none",
        # Full fine-tuning: no peft_config
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\n=== Starting full fine-tuning ===")
    trainer.train()

    print(f"\nSaving final model to {args.output}/final ...")
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print("Done.")

if __name__ == "__main__":
    main()
