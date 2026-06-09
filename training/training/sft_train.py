#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) script for MominoMoE with response-masked loss.

Trains the model on instruction-following data where the loss is computed
only on the response (assistant) tokens, not the prompt tokens.
Supports AMP, checkpointing, and gradient accumulation.

Memory strategy (1.2B params on a 22GB GPU):
  weights fp32 ~4.9GB + grads fp32 ~4.9GB + activations/fp16-cache ~3-4GB
  + AdamW states: fp32 ~9.8GB (OOMs) OR bitsandbytes 8-bit ~2.4GB (fits).
Install bitsandbytes so configure_optimizer() picks AdamW8bit; otherwise use
--batch-size 1 and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to fit.
AMP autocast('cuda') handles fp16 during forward pass for memory-efficient matmuls.
"""

import os
import sys
import math
import argparse
import json
import glob
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
from training.dataset import SFTDataset, sft_collate
from training.train_utils import (
    get_cosine_schedule_with_warmup,
    configure_optimizer,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    MetricsLogger,
    clip_gradients,
    get_device,
    get_scaler,
    is_main_process,
)
from training.byte_tokenizer import ByteTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT MominoMoE")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--val-data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/root/MominOS/checkpoints/sft")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    return parser.parse_args()


_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = ByteTokenizer()
    return _tokenizer


def encode_fn(text: str) -> list:
    return get_tokenizer().encode(text, max_length=2048)


def compute_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    n_tokens = int((shift_labels != -100).sum().item())
    if n_tokens == 0:
        # No supervised (response) tokens in this batch: CrossEntropyLoss over an
        # empty set returns nan. Contribute an exact-zero loss instead, keeping the
        # autograd graph valid so backward is a clean no-op.
        return shift_logits.sum() * 0.0, 0
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss, n_tokens


@torch.no_grad()
def evaluate(model, val_loader, args, device):
    model.eval()
    total_loss, total_tokens, n_batches = 0.0, 0, 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if args.fp16 and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                output = model(input_ids, labels=labels)
                loss, nt = compute_loss(output["logits"], labels)
        else:
            output = model(input_ids, labels=labels)
            loss, nt = compute_loss(output["logits"], labels)
        total_loss += loss.item()
        total_tokens += nt
        n_batches += 1
        if n_batches >= 50:
            break
    avg_loss = total_loss / max(n_batches, 1)
    model.train()
    return {"val_loss": avg_loss, "val_perplexity": math.exp(min(avg_loss, 20))}


def train_loop(model, loader, val_loader, optimizer, scheduler, scaler, logger_util, epoch, args, global_step, device, best_val_loss):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if args.fp16 and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                output = model(input_ids, labels=labels)
                loss, nt = compute_loss(output["logits"], labels)
                loss = loss / args.grad_accum_steps
        else:
            output = model(input_ids, labels=labels)
            loss, nt = compute_loss(output["logits"], labels)
            loss = loss / args.grad_accum_steps

        running_loss += loss.item() * args.grad_accum_steps

        if args.fp16 and torch.cuda.is_available():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % args.grad_accum_steps == 0:
            global_step += 1
            if args.fp16 and torch.cuda.is_available():
                scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model, args.grad_clip)
            if args.fp16 and torch.cuda.is_available():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = running_loss / max(args.grad_accum_steps, 1)  # avg loss per micro-batch

            if is_main_process() and (global_step % args.log_every == 0):
                logger_util.print_metrics(global_step, epoch, args.max_steps, avg_loss, scheduler.get_last_lr()[0] if scheduler else args.lr, extra=f"ppl={math.exp(min(avg_loss, 20)):.1f}")

            running_loss = 0.0

            if is_main_process() and val_loader is not None and (global_step % args.eval_every == 0):
                val_metrics = evaluate(model, val_loader, args, device)
                logger_util.log(val_metrics, global_step, prefix="eval")
                vl, vppl = val_metrics["val_loss"], val_metrics["val_perplexity"]
                print(f"  VALIDATION: loss={vl:.4f}, ppl={vppl:.2f}")
                if vl < best_val_loss:
                    best_val_loss = vl
                    save_checkpoint(os.path.join(args.output_dir, "checkpoint_best.pt"), model, optimizer, scheduler, scaler, epoch=epoch, global_step=global_step, config=vars(args))
                    print(f"  New best checkpoint saved (val_loss={vl:.4f})")

            if is_main_process() and (global_step % args.save_every == 0):
                save_checkpoint(os.path.join(args.output_dir, "checkpoint.pt"), model, optimizer, scheduler, scaler, epoch=epoch, global_step=global_step, config=vars(args))

            if global_step >= args.max_steps:
                break
    return global_step, best_val_loss


def main():
    args = parse_args()
    is_main = is_main_process()
    device = get_device()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # Build model in fp32 (AMP autocast handles fp16 forward pass)
    config = MominoMoEConfig.from_json_file(args.model_config) if args.model_config else MominoMoEConfig()
    model = MominoMoE(config)

    if args.init_from:
        load_checkpoint(args.init_from, model)
        if is_main:
            print(f"Initialized from: {args.init_from}")

    model = model.to(device)
    model.gradient_checkpointing_enable()

    if is_main:
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_total:,} (fp32), AMP autocast for fp16 forward")
        print("Gradient checkpointing enabled")

    # Dataset
    dataset = SFTDataset(args.data_path, encode_fn, max_seq_len=args.max_seq_len, response_only_loss=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sft_collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    val_loader = None
    if args.val_data_path:
        val_dataset = SFTDataset(args.val_data_path, encode_fn, max_seq_len=args.max_seq_len, response_only_loss=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sft_collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        if is_main:
            print(f"Validation: {len(val_dataset)} samples")

    optimizer = configure_optimizer(model, learning_rate=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps, min_lr_ratio=args.min_lr_ratio)
    scaler = get_scaler(enabled=args.fp16)
    logger_util = MetricsLogger(use_wandb=args.wandb and is_main, project="mominos_sft")

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        global_step = state.global_step
        start_epoch = state.epoch
        if is_main:
            print(f"Resumed from step {global_step}")

    # Dry run
    if args.dry_run:
        if is_main:
            batch = next(iter(loader))
            input_ids = batch["input_ids"][:1].to(device)
            labels = batch["labels"][:1].to(device)
            with torch.no_grad():
                out = model(input_ids, labels=labels)
                loss, nt = compute_loss(out["logits"], labels)
            print(f"Dry run: {input_ids.shape} -> logits {out['logits'].shape}, loss={loss:.4f}, resp_tokens={nt}")
        return

    # Save config
    if is_main:
        with open(os.path.join(args.output_dir, "sft_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Train
    for epoch in range(start_epoch, 100):
        global_step, best_val_loss = train_loop(model, loader, val_loader, optimizer, scheduler, scaler, logger_util, epoch, args, global_step, device, best_val_loss)
        if global_step >= args.max_steps:
            break

    # Final save
    if is_main:
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        save_checkpoint(final_path, model, optimizer, scheduler, scaler, epoch=start_epoch, global_step=global_step, config=vars(args))

        best_ckpt = find_latest_checkpoint(args.output_dir) if hasattr(find_latest_checkpoint, '__call__') else None
        pattern = os.path.join(args.output_dir, "checkpoint_best*.pt")
        best_files = sorted(glob.glob(pattern))
        if best_files:
            import shutil
            shutil.copy2(best_files[-1], os.path.join(args.output_dir, "sft_best.pt"))
            print(f"sft_best.pt saved from {best_files[-1]}")
        else:
            torch.save({"model_state_dict": model.state_dict(), "global_step": global_step}, os.path.join(args.output_dir, "sft_best.pt"))
            print(f"sft_best.pt saved (final step {global_step})")

    logger_util.finish()


if __name__ == "__main__":
    main()