"""Training infrastructure for MominoMoE.

Provides:
  - dataset.py: PretrainDataset, SFTDataset, DPODataset with proper masking
  - train_utils.py: Cosine LR, checkpointing, logging, AMP, DDP helpers
  - pretrain.py: DDP-compatible pretraining with MoE auxiliary loss
  - sft_train.py: Response-masked SFT training
  - dpo_train.py: DPO training with reference model, label smoothing, iterative loop
"""

from .dataset import PretrainDataset, SFTDataset, DPODataset, pretrain_collate, sft_collate, dpo_collate
from .train_utils import (
    TrainingConfig,
    get_cosine_schedule_with_warmup,
    configure_optimizer,
    save_checkpoint,
    load_checkpoint,
    MetricsLogger,
    clip_gradients,
    get_device,
    get_scaler,
)

__all__ = [
    "PretrainDataset",
    "SFTDataset",
    "DPODataset",
    "pretrain_collate",
    "sft_collate",
    "dpo_collate",
    "TrainingConfig",
    "get_cosine_schedule_with_warmup",
    "configure_optimizer",
    "save_checkpoint",
    "load_checkpoint",
    "MetricsLogger",
    "clip_gradients",
    "get_device",
    "get_scaler",
]