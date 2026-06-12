#!/usr/bin/env python3
"""
LoRA fine-tune of Qwen3-0.6B on MominOS kernel fault diagnosis data.
Uses HuggingFace PEFT + TRL SFTTrainer.

Usage:
  python3 lora_train.py [options]
"""
import os, sys, json, argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

SYSTEM_PROMPT = (
    "You are MominOS, a kernel fault diagnostician running embedded in an x86-64 OS. "
    "Given a kernel fault report in harness envelope format, identify the fault type, "
    "root cause, and suggest a specific corrective action. Be concise and precise. /no_think"
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_sample(sample, tokenizer):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sample["prompt"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="/root/train/data/kernel_train.jsonl")
    parser.add_argument("--val-data",   default="/root/train/data/kernel_val.jsonl")
    parser.add_argument("--model",      default="/root/train/model/qwen3-0.6b")
    parser.add_argument("--output",     default="/root/train/checkpoints")
    parser.add_argument("--steps",      type=int,   default=5000)
    parser.add_argument("--batch",      type=int,   default=4)
    parser.add_argument("--grad-accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--seq-len",    type=int,   default=1024)
    parser.add_argument("--lora-r",     type=int,   default=16)
    parser.add_argument("--eval-steps", type=int,   default=200)
    args = parser.parse_args()

    print(f"[lora_train] model={args.model}")
    print(f"[lora_train] data={args.data}")
    print(f"[lora_train] steps={args.steps}  batch={args.batch}  grad_accum={args.grad_accum}  lr={args.lr}")
    print(f"[lora_train] effective batch = {args.batch * args.grad_accum}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ─────────────────────────────────────────────────────────────
    print("Loading model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model: {total_params/1e6:.1f}M params")

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "lora" in n.lower()
    )
    print(f"LoRA params to train: will be ~{args.lora_r * 2 * 28 * 7 * 1024 // 1000}k")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_raw = load_jsonl(args.data)
    val_raw   = load_jsonl(args.val_data)

    train_texts = [format_sample(s, tokenizer) for s in train_raw]
    val_texts   = [format_sample(s, tokenizer) for s in val_raw]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds   = Dataset.from_dict({"text": val_texts})
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Training config ───────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output,
        max_seq_length=args.seq_len,

        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,

        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,

        fp16=True,
        optim="adamw_torch",
        weight_decay=0.01,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.2f}M  "
          f"({100*trainable/total_params:.2f}% of model)\n")

    # ── Train ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TRAINING START")
    print("=" * 60)
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output, "final")
    print(f"\nSaving adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
