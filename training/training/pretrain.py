#!/usr/bin/env python3
"""Pretraining script for MominoMoE with MoE auxiliary loss and DDP support.

Trains the model on next-token prediction with:
  - Cross-entropy loss on all tokens
  - MoE load balancing auxiliary loss (from moe_layer.py)
  - Optional z-loss for router logits
  - Gradient accumulation and gradient clipping
  - DDP (Distributed Data Parallel) support
  - Automatic Mixed Precision (AMP)
  - Checkpointing and wandb logging
"""

import os
import sys
import math
import time
import argparse
import json
import random
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root (training/) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
from training.dataset import PretrainDataset, pretrain_collate
from training.train_utils import (
    TrainingConfig,
    get_cosine_schedule_with_warmup,
    configure_optimizer,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    MetricsLogger,
    clip_gradients,
    get_device,
    get_scaler,
    get_ddp_rank,
    get_ddp_world_size,
    is_main_process,
    reduce_metric,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain MominoMoE")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to tokenized .npy file or directory")
    parser.add_argument("--output-dir", type=str,
                        default="/root/MominOS/training/checkpoints/pretrain",
                        help="Output directory for checkpoints")
    
    # Model
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config JSON (uses default if None)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to pretrained weights to initialize from")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Total training steps")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="Minimum LR as fraction of peak")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Number of warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm")
    
    # MoE
    parser.add_argument("--moe-aux-loss-coef", type=float, default=0.01,
                        help="Load balancing loss coefficient")
    parser.add_argument("--moe-z-loss-coef", type=float, default=0.001,
                        help="Router z-loss coefficient")
    
    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--ddp", action="store_true", default=False,
                        help="Enable Distributed Data Parallel")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable wandb logging")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Validate pipeline and exit")
    
    return parser.parse_args()


def setup_ddp(args):
    """Initialize DDP process group."""
    if not args.ddp:
        return 0, 1
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"env://",
        world_size=world_size,
        rank=local_rank,
    )
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size


def compute_pretrain_loss(
    model_output: dict,
    labels: torch.Tensor,
    moe_aux_coef: float = 0.01,
    moe_z_coef: float = 0.001,
) -> dict:
    """Compute pretraining loss including MoE auxiliary losses.
    
    Args:
        model_output: Output from model.forward() with keys:
            'logits', 'aux_loss', 'moe_metrics'
        labels: Target token IDs (with -100 for ignored positions)
        moe_aux_coef: Weight for load balancing loss
        moe_z_coef: Weight for router z-loss
    
    Returns:
        dict with 'loss' (total), 'ce_loss', 'aux_loss', 'z_loss'
    """
    logits = model_output["logits"]
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Cross-entropy loss (ignores -100 positions)
    ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    ce_loss = ce_loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    
    # MoE auxiliary losses from model
    aux_loss = model_output.get("aux_loss", torch.tensor(0.0, device=logits.device))
    moe_metrics = model_output.get("moe_metrics", {})
    z_loss = moe_metrics.get("z_loss", torch.tensor(0.0, device=logits.device))
    
    # Combined loss
    loss = ce_loss + moe_aux_coef * aux_loss + moe_z_coef * z_loss
    
    return {
        "loss": loss,
        "ce_loss": ce_loss.detach(),
        "aux_loss": aux_loss.detach() * moe_aux_coef,
        "z_loss": z_loss.detach() * moe_z_coef,
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    logger_util: MetricsLogger,
    epoch: int,
    args: argparse.Namespace,
    global_step: int,
    local_rank: int,
    world_size: int,
) -> int:
    """Run one pretraining epoch."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_aux_loss = 0.0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].cuda(local_rank) if args.ddp else batch["input_ids"].to(get_device())
        labels = batch["labels"].cuda(local_rank) if args.ddp else batch["labels"].to(get_device())
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.cuda(local_rank) if args.ddp else attention_mask.to(get_device())
        
        # Forward pass with AMP
        if args.fp16 and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                model_output = model(input_ids, labels=labels)
                losses = compute_pretrain_loss(
                    model_output, labels,
                    moe_aux_coef=args.moe_aux_loss_coef,
                    moe_z_coef=args.moe_z_loss_coef,
                )
                loss = losses["loss"] / args.grad_accum_steps
        else:
            model_output = model(input_ids, labels=labels)
            losses = compute_pretrain_loss(
                model_output, labels,
                moe_aux_coef=args.moe_aux_loss_coef,
                moe_z_coef=args.moe_z_loss_coef,
            )
            loss = losses["loss"] / args.grad_accum_steps
        
        # Backward with AMP
        if args.fp16 and torch.cuda.is_available():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Track losses
        total_loss += losses["loss"].item()
        total_ce_loss += losses["ce_loss"].item()
        total_aux_loss += losses["aux_loss"].item()
        
        # Gradient accumulation step
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            global_step += 1
            
            # Gradient clipping (unscaled for AMP)
            if args.fp16 and torch.cuda.is_available():
                scaler.unscale_(optimizer)
            
            grad_norm = clip_gradients(model, args.grad_clip)
            
            # Optimizer step
            if args.fp16 and torch.cuda.is_available():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if is_main_process() and (global_step % args.log_every == 0 or global_step == 1):
                avg_loss = total_loss / (batch_idx + 1)
                avg_ce = total_ce_loss / (batch_idx + 1)
                avg_aux = total_aux_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                
                perplexity = math.exp(min(avg_ce, 20))
                
                metrics = {
                    "loss": avg_loss,
                    "ce_loss": avg_ce,
                    "aux_loss": avg_aux,
                    "perplexity": perplexity,
                    "grad_norm": grad_norm,
                    "lr": lr,
                }
                
                logger_util.log(metrics, global_step, prefix="pretrain")
                logger_util.print_metrics(
                    global_step, epoch, args.max_steps // len(train_loader) + 1,
                    avg_loss, lr,
                    f" | CE: {avg_ce:.4f} | PPL: {perplexity:.2f} | Grad: {grad_norm:.4f}",
                )
            
            # Save checkpoint
            if is_main_process() and (global_step % args.save_every == 0):
                ckpt_path = os.path.join(args.output_dir, f"checkpoint.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler,
                    epoch=epoch, global_step=global_step,
                    config=vars(args),
                    keep_last_n=args.keep_checkpoints,
                )
            
            # Check if max steps reached
            if global_step >= args.max_steps:
                break
    
    return global_step


def main():
    args = parse_args()
    
    # Setup DDP
    local_rank, world_size = setup_ddp(args)
    is_main = is_main_process()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + local_rank)
    
    # Create output dir
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model config
    if args.model_config:
        config = MominoMoEConfig.from_json_file(args.model_config)
    else:
        config = MominoMoEConfig()  # Default 1.2B
    
    if is_main:
        print(f"Model: {config.model_name}")
        print(f"Params: {config.total_params_estimate():.1f}B")
        print(f"Output dir: {args.output_dir}")
    
    # Build model
    model = MominoMoE(config)
    
    # Load pretrained weights if specified
    if args.init_from:
        state = load_checkpoint(args.init_from, model)
        if is_main:
            print(f"Initialized from: {args.init_from} (step {state.global_step})")
    
    # Move model to device
    device = get_device()
    if args.ddp:
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = model.to(device)
    
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
    
    # Dataset
    dataset = PretrainDataset(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    
    # Sampler for DDP
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True,
        )
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=pretrain_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.ddp,
    )
    
    if is_main:
        print(f"Train loader: {len(train_loader)} batches/epoch, "
              f"{len(dataset)} samples")
    
    # Optimizer
    optimizer = configure_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    total_steps = len(train_loader) // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=min(total_steps, args.max_steps),
        min_lr_ratio=args.min_lr_ratio,
    )
    
    # Scaler (AMP)
    scaler = get_scaler(enabled=args.fp16)
    
    # Logger
    logger_util = MetricsLogger(use_wandb=args.wandb and is_main, project="mominos_pretrain")
    
    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        global_step = state.global_step
        start_epoch = state.epoch
        if is_main:
            print(f"Resumed from step {global_step}, epoch {start_epoch}")
    
    # Dry run: validate pipeline
    if args.dry_run:
        if is_main:
            print("\n=== Dry run: validating pipeline ===")
            batch = next(iter(train_loader))
            print(f"Batch shapes: input_ids={batch['input_ids'].shape}, "
                  f"labels={batch['labels'].shape}")
            
            input_ids = batch["input_ids"][:1].to(device)
            labels = batch["labels"][:1].to(device)
            with torch.no_grad():
                output = model(input_ids, labels=labels)
                losses = compute_pretrain_loss(
                    output, labels,
                    moe_aux_coef=args.moe_aux_loss_coef,
                    moe_z_coef=args.moe_z_loss_coef,
                )
            print(f"Output: logits shape={output['logits'].shape}")
            print(f"Loss: {losses['loss'].item():.4f} (CE: {losses['ce_loss'].item():.4f}, "
                  f"Aux: {losses['aux_loss'].item():.4f}, Z: {losses['z_loss'].item():.6f})")
            print("Pipeline valid. Exiting (--dry-run).")
        return
    
    # Save config
    if is_main:
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"\nStarting pretraining for {args.max_steps} steps...\n")
    
    # Training loop
    for epoch in range(start_epoch, args.max_steps // len(train_loader) + 1):
        if args.ddp:
            train_sampler.set_epoch(epoch)
        
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            logger_util, epoch, args, global_step, local_rank, world_size,
        )
        
        if global_step >= args.max_steps:
            break
    
    # Final save
    if is_main:
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_final.pt")
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler,
            epoch=start_epoch, global_step=global_step,
            config=vars(args),
        )
        print(f"\nTraining complete. Final checkpoint: {ckpt_path}")
    
    # Cleanup
    if args.ddp:
        dist.destroy_process_group()
    
    logger_util.finish()


if __name__ == "__main__":
    main()