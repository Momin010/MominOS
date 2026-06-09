#!/usr/bin/env python3
"""Training utilities for MominoMoE: LR scheduling, checkpointing, logging, AMP.

Provides reusable utilities used by pretrain.py, sft_train.py, and dpo_train.py.
"""

import os
import json
import math
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_utils")


# ── Cosine LR Schedule with Warmup ───────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a cosine decay LR schedule with linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Steps for linear warmup
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of peak (default: 0.0)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """Constant LR after linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


# ── Optimizer Configuration ──────────────────────────────────────────────

def configure_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    fused: bool = True,
) -> torch.optim.AdamW:
    """Configure AdamW optimizer with parameter groups for weight decay.
    
    Applies weight decay to all parameters except biases and layer norms.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "rmsnorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Prefer 8-bit AdamW (bitsandbytes): optimizer states drop ~4x
    # (9.8GB -> ~2.4GB for a 1.2B model), which is what keeps SFT under
    # the 22GB GPU budget. Falls back to fused fp32 AdamW if unavailable.
    optimizer = None
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                optim_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
            )
            logger.info("Using bitsandbytes AdamW8bit (8-bit optimizer states)")
        except ImportError:
            logger.warning("bitsandbytes not installed; falling back to fp32 AdamW")

    if optimizer is None:
        # Try fused AdamW if available (faster on CUDA)
        try:
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                fused=fused and torch.cuda.is_available(),
            )
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
            )

    n_params = len(decay_params) + len(no_decay_params)
    logger.info(
        f"Optimizer: {n_params} param groups "
        f"({len(decay_params)} decay, {len(no_decay_params)} no_decay)"
    )
    return optimizer


# ── Gradient Scaler (AMP) ────────────────────────────────────────────────

def get_scaler(enabled: bool = True) -> GradScaler:
    """Get gradient scaler for mixed precision training."""
    return GradScaler('cuda', enabled=enabled and torch.cuda.is_available())


# ── Checkpointing ────────────────────────────────────────────────────────

@dataclass
class CheckpointState:
    """Container for full training state."""
    epoch: int
    global_step: int
    best_metric: float
    model_state: Dict[str, Any]
    optimizer_state: List[Dict]
    scheduler_state: Dict[str, Any]
    scaler_state: Optional[Dict[str, Any]] = None
    config: Optional[Dict] = None
    rng_state: Optional[Dict] = None


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float = 0.0,
    config: Optional[Dict] = None,
    keep_last_n: int = 5,
):
    """Save training checkpoint with all state.
    
    Args:
        path: Path to save (will append _step{global_step}.pt)
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        scaler: AMP scaler state
        epoch: Current epoch
        global_step: Current global step
        best_metric: Best validation metric
        config: Training config dict
        keep_last_n: Max checkpoints to keep (oldest removed)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # Build checkpoint path with step number
    base, ext = os.path.splitext(path)
    ckpt_path = f"{base}_step{global_step}{ext}"
    
    # Capture RNG state
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    
    state = CheckpointState(
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler else {},
        scaler_state=scaler.state_dict() if scaler else None,
        config=config,
        rng_state=rng_state,
    )
    
    # Save
    torch.save({
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
        "model_state_dict": state.model_state,
        "optimizer_state_dict": state.optimizer_state,
        "scheduler_state_dict": state.scheduler_state,
        "scaler_state_dict": state.scaler_state,
        "config": state.config,
        "rng_state": state.rng_state,
    }, ckpt_path)
    
    logger.info(f"Checkpoint saved: {ckpt_path} (step {global_step})")
    
    # Cleanup: keep only last N checkpoints
    _cleanup_old_checkpoints(path, keep_last_n)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    strict: bool = True,
) -> CheckpointState:
    """Load training checkpoint and restore state.
    
    Returns:
        CheckpointState with restored values
    """
    if not os.path.exists(path):
        logger.warning(f"Checkpoint not found: {path}. Starting from scratch.")
        return CheckpointState(
            epoch=0, global_step=0, best_metric=0.0,
            model_state={}, optimizer_state=[], scheduler_state={},
        )
    
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    # Load model
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Load scaler
    if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # Restore RNG
    if "rng_state" in checkpoint and checkpoint["rng_state"]:
        rng = checkpoint["rng_state"]
        torch.set_rng_state(rng.get("torch", torch.get_rng_state()))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
    
    logger.info(
        f"Checkpoint loaded: {path} "
        f"(epoch {checkpoint.get('epoch', 0)}, step {checkpoint.get('global_step', 0)})"
    )
    
    return CheckpointState(
        epoch=checkpoint.get("epoch", 0),
        global_step=checkpoint.get("global_step", 0),
        best_metric=checkpoint.get("best_metric", 0.0),
        model_state=checkpoint.get("model_state_dict", {}),
        optimizer_state=checkpoint.get("optimizer_state_dict", []),
        scheduler_state=checkpoint.get("scheduler_state_dict", {}),
        scaler_state=checkpoint.get("scaler_state_dict"),
        config=checkpoint.get("config"),
        rng_state=checkpoint.get("rng_state"),
    )


def _cleanup_old_checkpoints(path: str, keep_last_n: int):
    """Remove oldest checkpoints, keeping only the most recent N."""
    base = os.path.dirname(path) or "."
    pattern = os.path.basename(path).replace(".pt", "_step*.pt")
    
    import glob
    checkpoints = sorted(glob.glob(os.path.join(base, pattern)))
    
    while len(checkpoints) > keep_last_n:
        oldest = checkpoints.pop(0)
        os.remove(oldest)
        logger.info(f"Removed old checkpoint: {oldest}")


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint") -> Optional[str]:
    """Find the latest checkpoint file in a directory by step number."""
    import glob
    pattern = os.path.join(ckpt_dir, f"{prefix}_step*.pt")
    files = sorted(glob.glob(pattern))
    if files:
        latest = files[-1]
        logger.info(f"Found latest checkpoint: {latest}")
        return latest
    return None


# ── Logging ──────────────────────────────────────────────────────────────

class MetricsLogger:
    """Logs training metrics to console and optionally wandb."""
    
    def __init__(self, use_wandb: bool = False, project: str = "mominos"):
        self.use_wandb = use_wandb
        self.project = project
        self.step_metrics: Dict[str, float] = {}
        self.wandb_run = None
        
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project)
                logger.info(f"Wandb initialized: {project}")
            except (ImportError, Exception) as e:
                logger.warning(f"Wandb init failed (continuing without): {e}")
                self.use_wandb = False
    
    def log(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log metrics for a given step."""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            key = f"{prefix}/{k}" if prefix else k
            self.step_metrics[key] = v
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)
    
    def log_dict(self, metrics: Dict[str, float], step: int):
        """Log a flat dict of metrics."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log(metrics, step=step)
    
    def print_metrics(self, step: int, epoch: int, total_epochs: int, loss: float, lr: float, extra: str = ""):
        """Print formatted metrics line."""
        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | Step {step} | "
            f"Loss: {loss:.4f} | LR: {lr:.2e}{extra}"
        )
    
    def finish(self):
        """Close wandb run."""
        if self.wandb_run:
            self.wandb_run.finish()


# ── Gradient Clipping ────────────────────────────────────────────────────

def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """Clip gradients and return total gradient norm."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, norm_type=norm_type
    )
    return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm


# ── Device and DDP Helpers ───────────────────────────────────────────────

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_ddp_rank() -> int:
    """Get DDP rank (0 if not in DDP)."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_ddp_world_size() -> int:
    """Get DDP world size (1 if not in DDP)."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main DDP process."""
    return get_ddp_rank() == 0


def reduce_metric(metric: torch.Tensor) -> torch.Tensor:
    """Average a metric across DDP processes."""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
        metric /= torch.distributed.get_world_size()
    return metric


# ── Config Helpers ───────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    
    # Model
    model_name: str = "MominoMoE-1.2B"
    
    # Paths
    output_dir: str = "/root/MominOS/training/checkpoints"
    data_dir: str = "/root/MominOS/training/data"
    resume_from: Optional[str] = None
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 100000
    num_epochs: int = 3
    max_seq_len: int = 2048
    
    # Optimizer
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    
    # MoE
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.001
    
    # DPO
    dpo_beta: float = 0.1
    dpo_label_smoothing: float = 0.0
    
    # System
    grad_clip: float = 1.0
    fp16: bool = True
    ddp: bool = False
    ddp_port: str = "29500"
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    keep_checkpoints: int = 5
    seed: int = 42
    num_workers: int = 2
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})