#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) training script for MominoMoE.

Trains the model using preference pairs (chosen/rejected) with:
  - DPO loss with reference model (frozen)
  - Support for label smoothing
  - Iterative DPO loop (train -> generate -> retrain)
  - Reward hacking monitor
  - Evaluation callbacks
  - DDP, AMP, checkpointing, wandb
"""

import os
import sys
import math
import json
import random
import argparse
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root (training/) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
from training.dataset import DPODataset, dpo_collate
from training.train_utils import (
    get_cosine_schedule_with_warmup,
    configure_optimizer,
    save_checkpoint,
    load_checkpoint,
    MetricsLogger,
    clip_gradients,
    get_device,
    get_scaler,
    get_ddp_rank,
    get_ddp_world_size,
    is_main_process,
)


# ── DPO Loss ─────────────────────────────────────────────────────────────

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    ipo: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute DPO loss.
    
    Args:
        policy_chosen_logps: Log-probs of chosen tokens under policy (batch,)
        policy_rejected_logps: Log-probs of rejected tokens under policy (batch,)
        ref_chosen_logps: Log-probs of chosen tokens under reference (batch,)
        ref_rejected_logps: Log-probs of rejected tokens under reference (batch,)
        beta: Temperature parameter for DPO
        label_smoothing: Label smoothing coefficient (0 = no smoothing)
        ipo: Use IPO loss instead of DPO
    
    Returns:
        dict with 'loss', 'chosen_rewards', 'rejected_rewards', 'accuracies'
    """
    # Compute log-ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios  # Implicit reward
    
    if ipo:
        # IPO loss: (logits - 1/(2*beta))^2
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    else:
        # DPO loss with optional label smoothing
        if label_smoothing > 0:
            # Smoothed DPO: mix positive and negative labels
            neg_loss = F.logsigmoid(-beta * logits)
            pos_loss = F.logsigmoid(beta * logits)
            losses = -(1.0 - label_smoothing) * pos_loss - label_smoothing * neg_loss
        else:
            # Standard DPO
            losses = -F.logsigmoid(beta * logits)
    
    # Compute rewards
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
        accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
    
    return {
        "loss": losses.mean(),
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
        "accuracies": accuracies.mean(),
        "reward_margin": reward_margin,
        "logits": logits.detach(),
    }


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-token log-probabilities for given labels.
    
    Args:
        model: The model
        input_ids: Token IDs (batch, seq_len)
        labels: Token IDs with -100 for ignored positions (batch, seq_len)
        attention_mask: Optional attention mask
    
    Returns:
        Log-probabilities of the non-ignored labels (batch,)
        Shape: (batch_size,) - one average log-prob per sequence
    """
    output = model(input_ids, labels=labels)
    logits = output["logits"]  # (batch, seq_len, vocab_size)
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Log-softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log-prob of the actual label tokens
    per_token_logps = torch.gather(
        log_probs, dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    
    # Mask ignore positions
    mask = (shift_labels != -100).float()
    per_token_logps = per_token_logps * mask
    
    # Sum and normalize by number of valid tokens
    sum_logps = per_token_logps.sum(dim=-1)
    n_tokens = mask.sum(dim=-1).clamp(min=1)
    
    return sum_logps / n_tokens


# ── Reward Hacking Monitor ───────────────────────────────────────────────

@dataclass
class RewardStats:
    """Statistics for monitoring reward hacking."""
    mean_chosen: float = 0.0
    mean_rejected: float = 0.0
    mean_margin: float = 0.0
    accuracy: float = 0.0
    reward_growth_rate: float = 0.0
    reward_variance: float = 0.0
    n_samples: int = 0
    
    window_chosen: List[float] = field(default_factory=list)
    window_rejected: List[float] = field(default_factory=list)
    window_margins: List[float] = field(default_factory=list)
    
    def update(self, chosen: float, rejected: float, margin: float, acc: float):
        self.window_chosen.append(chosen)
        self.window_rejected.append(rejected)
        self.window_margins.append(margin)
        
        # Keep last 100
        max_window = 100
        if len(self.window_chosen) > max_window:
            self.window_chosen.pop(0)
            self.window_rejected.pop(0)
            self.window_margins.pop(0)
        
        n = len(self.window_chosen)
        self.mean_chosen = sum(self.window_chosen) / n
        self.mean_rejected = sum(self.window_rejected) / n
        self.mean_margin = sum(self.window_margins) / n
        self.accuracy = acc
        self.n_samples = n
        
        # Compute growth rate (slope over window)
        if n >= 10:
            xs = list(range(n))
            ys = self.window_margins
            n_ = float(n)
            sx = sum(xs)
            sy = sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            slope = (n_ * sxy - sx * sy) / (n_ * sxx - sx * sx + 1e-8)
            self.reward_growth_rate = slope
            
            # Variance
            mean_y = sy / n_
            self.reward_variance = sum((y - mean_y) ** 2 for y in ys) / n_
    
    def is_hacking(self, threshold_growth: float = 0.5, threshold_var: float = 10.0) -> bool:
        """Detect if reward hacking is occurring.
        
        Warning signs:
        - Reward margin growing too fast (>threshold_growth per step)
        - Reward variance too high (>threshold_var)
        - Very high accuracy with low loss (overconfident)
        """
        warnings = []
        if self.reward_growth_rate > threshold_growth and self.n_samples >= 10:
            warnings.append(f"rapid_reward_growth({self.reward_growth_rate:.3f})")
        if self.reward_variance > threshold_var and self.n_samples >= 10:
            warnings.append(f"high_reward_variance({self.reward_variance:.3f})")
        if self.accuracy > 0.98 and self.n_samples >= 20:
            warnings.append(f"overconfident({self.accuracy:.3f})")
        return warnings
    
    def summary(self) -> str:
        return (
            f"Chosen: {self.mean_chosen:.4f} | Rejected: {self.mean_rejected:.4f} | "
            f"Margin: {self.mean_margin:.4f} | Acc: {self.accuracy:.3f} | "
            f"Growth: {self.reward_growth_rate:.4f}/step"
        )


# ── Eval Callbacks ───────────────────────────────────────────────────────

class EvalCallback:
    """Evaluation callback for DPO training."""
    
    def __init__(self, eval_func: Optional[Callable] = None, eval_every: int = 100):
        self.eval_func = eval_func
        self.eval_every = eval_every
        self.best_metric = 0.0
        self.current_metric = 0.0
    
    def should_eval(self, step: int) -> bool:
        return step > 0 and step % self.eval_every == 0
    
    def __call__(self, model, step: int) -> Dict[str, float]:
        if self.eval_func is None:
            return {}
        
        metrics = self.eval_func(model)
        self.current_metric = metrics.get("eval_score", 0.0)
        if self.current_metric > self.best_metric:
            self.best_metric = self.current_metric
            metrics["best_score"] = self.best_metric
        return metrics


# ── Iterative DPO Loop ───────────────────────────────────────────────────

def generate_new_pairs(
    model: nn.Module,
    prompts: List[str],
    n_samples: int = 1,
    max_new_tokens: int = 256,
    device: Optional[torch.device] = None,
) -> List[Tuple[str, str, str]]:
    """Generate new preference pairs by sampling from the model.
    
    This is a simplified placeholder. In production, use vLLM or HF generate.
    
    Args:
        model: Policy model to sample from
        prompts: List of prompt strings
        n_samples: Number of samples per prompt
        max_new_tokens: Max tokens per generation
        device: Device
    
    Returns:
        List of (prompt, chosen, rejected) tuples
    """
    # Placeholder: in real usage, this would do actual generation
    # The actual implementation would:
    # 1. Tokenize prompts
    # 2. Generate multiple completions with temperature sampling
    # 3. Score them (using reward model or heuristic)
    # 4. Select chosen (best) and rejected (worst)
    logger.warning("generate_new_pairs is a placeholder. "
                   "In production, use vLLM for efficient generation.")
    return []


# ── Training ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DPO Train MominoMoE")
    
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to JSONL DPO data")
    parser.add_argument("--output-dir", type=str,
                        default="/root/MominOS/training/checkpoints/dpo",
                        help="Output directory")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None,
                        help="SFT checkpoint to start from")
    parser.add_argument("--ref-checkpoint", type=str, default=None,
                        help="Reference model checkpoint (if None, freeze init)")
    
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # DPO parameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature parameter")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for DPO (0 = none)")
    parser.add_argument("--ipo", action="store_true", default=False,
                        help="Use IPO loss instead of DPO")
    
    # Iterative DPO
    parser.add_argument("--iterative", action="store_true", default=False,
                        help="Run iterative DPO (train -> generate -> retrain)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of DPO iterations")
    parser.add_argument("--gen-per-iter", type=int, default=500,
                        help="Pairs to generate per iteration")
    
    # Eval
    parser.add_argument("--eval-every", type=int, default=500)
    
    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    
    return parser.parse_args()


def train_dpo_step(
    model: nn.Module,
    ref_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float,
    label_smoothing: float,
    ipo: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Single DPO training step."""
    # Move to device
    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)
    
    # Policy log-probs
    policy_chosen_logps = compute_log_probs(model, chosen_ids, chosen_labels)
    policy_rejected_logps = compute_log_probs(model, rejected_ids, rejected_labels)
    
    # Reference log-probs (no grad)
    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(ref_model, chosen_ids, chosen_labels)
        ref_rejected_logps = compute_log_probs(ref_model, rejected_ids, rejected_labels)
    
    # DPO loss
    loss_info = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
        ipo=ipo,
    )
    
    return loss_info


def train_dpo_epoch(
    model: nn.Module,
    ref_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    logger_util: MetricsLogger,
    reward_monitor: RewardStats,
    eval_callback: EvalCallback,
    epoch: int,
    args: argparse.Namespace,
    global_step: int,
    device: torch.device,
) -> int:
    """Run one DPO training epoch."""
    model.train()
    ref_model.eval()
    total_loss = 0.0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        # Forward
        if args.fp16 and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                loss_info = train_dpo_step(
                    model, ref_model, batch, args.beta,
                    args.label_smoothing, args.ipo, args, device,
                )
                loss = loss_info["loss"] / args.grad_accum_steps
        else:
            loss_info = train_dpo_step(
                model, ref_model, batch, args.beta,
                args.label_smoothing, args.ipo, args, device,
            )
            loss = loss_info["loss"] / args.grad_accum_steps
        
        # Backward
        if args.fp16 and torch.cuda.is_available():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss_info["loss"].item()
        
        # Tracking
        reward_monitor.update(
            loss_info["chosen_rewards"].item(),
            loss_info["rejected_rewards"].item(),
            loss_info["reward_margin"].item(),
            loss_info["accuracies"].item(),
        )
        
        # Gradient accumulation
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
            
            # Check reward hacking
            hacking_warnings = reward_monitor.is_hacking()
            
            # Logging
            if is_main_process() and (global_step % args.log_every == 0):
                avg_loss = total_loss / (global_step * args.grad_accum_steps)
                lr = scheduler.get_last_lr()[0]
                
                metrics = {
                    "dpo_loss": avg_loss,
                    "chosen_reward": loss_info["chosen_rewards"].item(),
                    "rejected_reward": loss_info["rejected_rewards"].item(),
                    "reward_margin": loss_info["reward_margin"].item(),
                    "accuracy": loss_info["accuracies"].item(),
                    "grad_norm": grad_norm,
                    "lr": lr,
                }
                
                if hacking_warnings:
                    metrics["hacking_warning"] = len(hacking_warnings)
                    extra = f" ⚠️ HACKING: {', '.join(hacking_warnings)}"
                else:
                    extra = ""
                
                logger_util.log(metrics, global_step, prefix="dpo")
                logger_util.print_metrics(
                    global_step, epoch,
                    args.max_steps // len(train_loader) + 1,
                    avg_loss, lr,
                    f" | Acc: {loss_info['accuracies'].item():.3f} | "
                    f"Margin: {loss_info['reward_margin'].item():.4f}{extra}",
                )
            
            # Evaluation
            if is_main_process() and eval_callback.should_eval(global_step):
                eval_metrics = eval_callback(model, global_step)
                if eval_metrics:
                    logger_util.log(eval_metrics, global_step, prefix="eval")
            
            # Checkpoint
            if is_main_process() and (global_step % args.save_every == 0):
                ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler,
                    epoch=epoch, global_step=global_step,
                    config=vars(args),
                    keep_last_n=args.keep_checkpoints,
                )
            
            if global_step >= args.max_steps:
                break
    
    return global_step


def main():
    args = parse_args()
    is_main = is_main_process()
    local_rank = 0
    
    # DDP setup
    if args.ddp:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=local_rank)
        torch.cuda.set_device(local_rank)
    
    torch.manual_seed(args.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + local_rank)
    
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{local_rank}") if args.ddp else get_device()
    
    # Config
    config = MominoMoEConfig.from_json_file(args.model_config) if args.model_config else MominoMoEConfig()
    
    # Policy model
    model = MominoMoE(config)
    if args.init_from:
        load_checkpoint(args.init_from, model)
        if is_main:
            print(f"Policy initialized from: {args.init_from}")
    
    if args.ddp:
        model = model.cuda(local_rank)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    else:
        model = model.to(device)
    
    # Reference model (frozen copy of policy)
    ref_model = MominoMoE(config)
    if args.ref_checkpoint:
        load_checkpoint(args.ref_checkpoint, ref_model)
        if is_main:
            print(f"Reference from checkpoint: {args.ref_checkpoint}")
    else:
        # Reference = copy of initial policy (before training)
        ref_model.load_state_dict(
            model.module.state_dict() if args.ddp else model.state_dict()
        )
        if is_main:
            print("Reference model: copy of initial policy")
    
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    if not args.ddp:
        ref_model = ref_model.to(device)
    
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params:,}")
    
    # Dataset
    dataset = DPODataset(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
    )
    
    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, rank=local_rank,
            num_replicas=torch.distributed.get_world_size(),
        )
    else:
        sampler = None
    
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=dpo_collate, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    optimizer = configure_optimizer(model, learning_rate=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(loader) // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=min(total_steps, args.max_steps),
        min_lr_ratio=args.min_lr_ratio,
    )
    
    scaler = get_scaler(enabled=args.fp16)
    logger_util = MetricsLogger(use_wandb=args.wandb and is_main, project="mominos_dpo")
    reward_monitor = RewardStats()
    eval_callback = EvalCallback(eval_every=args.eval_every)
    
    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        global_step = state.global_step
        start_epoch = state.epoch
        if is_main:
            print(f"Resumed from step {global_step}")
    
    # Dry run
    if args.dry_run:
        if is_main:
            print("\n=== Dry run ===")
            batch = next(iter(loader))
            print(f"Batch: chosen_ids={batch['chosen_input_ids'].shape}, "
                  f"rejected_ids={batch['rejected_input_ids'].shape}")
            loss_info = train_dpo_step(
                model, ref_model, batch, args.beta,
                args.label_smoothing, args.ipo, args, device,
            )
            print(f"Loss: {loss_info['loss']:.4f}, Acc: {loss_info['accuracies']:.3f}, "
                  f"Chosen: {loss_info['chosen_rewards']:.4f}, "
                  f"Rejected: {loss_info['rejected_rewards']:.4f}")
            print("Dry run OK")
        return
    
    # Save config
    if is_main:
        with open(os.path.join(args.output_dir, "dpo_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"\nStarting DPO training for {args.max_steps} steps...")
        print(f"  beta={args.beta}, label_smoothing={args.label_smoothing}, "
              f"ipo={args.ipo}")
    
    # Iterative DPO
    if args.iterative:
        if is_main:
            print(f"\n--- Iterative DPO: {args.iterations} iterations ---\n")
        
        for iteration in range(args.iterations):
            if is_main:
                print(f"\n=== Iteration {iteration + 1}/{args.iterations} ===\n")
            
            # Train
            for epoch in range(start_epoch, args.max_steps // len(loader) + 1):
                if args.ddp:
                    sampler.set_epoch(epoch)
                
                global_step = train_dpo_epoch(
                    model, ref_model, loader, optimizer, scheduler, scaler,
                    logger_util, reward_monitor, eval_callback,
                    epoch, args, global_step, device,
                )
                
                if global_step >= args.max_steps:
                    break
            
            if global_step >= args.max_steps and iteration < args.iterations - 1:
                # Reset step count for next iteration
                global_step = 0
            
            # Generate new pairs (placeholder)
            if is_main and iteration < args.iterations - 1:
                new_pairs = generate_new_pairs(
                    model, prompts=[],
                    n_samples=args.gen_per_iter,
                )
                print(f"  Generated {len(new_pairs)} new pairs (placeholder)")
    else:
        # Standard DPO training
        for epoch in range(start_epoch, args.max_steps // len(loader) + 1):
            if args.ddp:
                sampler.set_epoch(epoch)
            
            global_step = train_dpo_epoch(
                model, ref_model, loader, optimizer, scheduler, scaler,
                logger_util, reward_monitor, eval_callback,
                epoch, args, global_step, device,
            )
            
            if global_step >= args.max_steps:
                break
    
    # Final save
    if is_main:
        ckpt_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                        epoch=start_epoch, global_step=global_step, config=vars(args))
        
        print(f"\nDPO training complete. Final: {ckpt_path}")
        print(f"Final reward stats: {reward_monitor.summary()}")
    
    if args.ddp:
        torch.distributed.destroy_process_group()
    logger_util.finish()


import logging
logger = logging.getLogger("dpo_train")


if __name__ == "__main__":
    main()